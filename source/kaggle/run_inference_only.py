from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legalqa.budget import count_system_budget  # noqa: E402
from legalqa.answer_style import build_answer_style_profile  # noqa: E402
from legalqa.config import load_config  # noqa: E402
from legalqa.data import load_qa, write_json  # noqa: E402
from legalqa.dense_store import ensure_global_dense_index  # noqa: E402
from legalqa.fingerprint import (  # noqa: E402
    adapter_fingerprint,
    checkpoint_fingerprint,
    corpus_content_fingerprint,
    sha256_file,
)
from legalqa.models import (  # noqa: E402
    CrossEncoderReranker,
    DenseEncoder,
    LegalSemanticEncoder,
    QwenTextGenerator,
    unload_model,
)
from legalqa.pipeline import generate_from_evidence  # noqa: E402
from legalqa.recovery import RecoveryManager, restore_recovery_bundle  # noqa: E402
from legalqa.sqlite_retrieval import retrieve_sqlite_batch, sqlite_bm25_search_fn  # noqa: E402
from legalqa.sqlite_store import build_sqlite_store  # noqa: E402
from legalqa.sqlite_store import SQLiteLegalStore  # noqa: E402
from legalqa.submission import create_submission  # noqa: E402
from legalqa.train_memory import ensure_train_memory_index  # noqa: E402
from run_end_to_end import (  # noqa: E402
    find_corpus,
    find_model,
    find_unique_file,
    require_environment,
    run_command,
    write_resolved_config,
)


CACHE_SCHEMA_VERSION = "inference-v5-graph-gate-train-memory-qlora"
SELECTION_WEIGHTS_SCHEMA = "selector-weights-v2-graph-gate"
RUNTIME_MODEL_ROLES = ("generator", "embedding", "reranker", "verifier")


def find_generator_adapter(root: Path, explicit: str | None = None) -> Path | None:
    value = explicit or os.environ.get("LEGALQA_GENERATOR_ADAPTER_DIR")
    if value:
        candidate = Path(value)
        if not (candidate / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Qwen adapter không có adapter_config.json: {candidate}")
        return candidate
    candidates = sorted({path.parent for path in root.rglob("adapter_config.json")})
    # Completed adapter roots are preferred over Trainer checkpoint children.
    completed = [path for path in candidates if (path / "training_complete.json").is_file()]
    preferred = completed or [
        path for path in candidates if not path.name.startswith("checkpoint-")
    ]
    if not preferred:
        return None
    if len(preferred) > 1:
        raise RuntimeError(
            f"Có nhiều Qwen LoRA adapter: {preferred}. Chỉ attach một bản hoặc dùng "
            "--generator-adapter/LEGALQA_GENERATOR_ADAPTER_DIR."
        )
    return preferred[0]


def smoke_inference_models(config: dict) -> dict:
    """Exercise the exact four runtime wrappers before expensive retrieval/generation."""
    smoke_config = copy.deepcopy(config)
    smoke_config["generation"]["max_new_tokens"] = 2
    smoke_config["generation"]["max_context_chars"] = 1000
    report = {}

    instance = None
    try:
        instance = DenseEncoder(smoke_config)
        vectors = instance.encode(["Kiểm tra văn bản pháp luật."], batch_size=1)
        if vectors.shape[0] != 1:
            raise RuntimeError(f"Embedding smoke trả shape bất thường: {vectors.shape}")
        report["embedding"] = {"shape": list(vectors.shape)}
    finally:
        unload_model(instance, "embedding smoke")

    instance = None
    try:
        instance = CrossEncoderReranker(smoke_config)
        scores = instance.score("Quy định này áp dụng thế nào?", ["Điều 1. Phạm vi áp dụng."])
        if len(scores) != 1:
            raise RuntimeError(f"Reranker smoke trả {len(scores)} score.")
        report["reranker"] = {"scores": len(scores)}
    finally:
        unload_model(instance, "reranker smoke")

    instance = None
    try:
        instance = QwenTextGenerator(smoke_config)
        answer = instance.generate(
            "Bạn là trợ lý pháp luật.",
            "Dựa vào căn cứ: Điều 1. Phạm vi áp dụng. Hãy trả lời ngắn.",
        )
        report["generator"] = {"decoded_chars": len(answer)}
    finally:
        unload_model(instance, "Qwen smoke")

    instance = None
    try:
        instance = LegalSemanticEncoder(smoke_config)
        vectors = instance.encode(["Điều 1. Phạm vi áp dụng.", "Quy định về phạm vi áp dụng."])
        if vectors.shape[0] != 2:
            raise RuntimeError(f"ViLegalBERT smoke trả shape bất thường: {vectors.shape}")
        report["verifier"] = {"shape": list(vectors.shape)}
    finally:
        unload_model(instance, "ViLegalBERT smoke")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UIT DSC 2026 LegalQA: full sequential inference, no fine-tuning"
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional question JSON for selector/dev evaluation; default is public-official.json",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Official train.json, or selector_train.json when evaluating grouped dev",
    )
    parser.add_argument(
        "--generator-adapter",
        default=None,
        help="Optional completed Qwen LoRA adapter directory",
    )
    parser.add_argument(
        "--selection-weights",
        default=None,
        help="Optional YAML produced by evaluate_selector.py",
    )
    parser.add_argument("--working", default="/kaggle/working/legalqa_inference")
    parser.add_argument("--submission-output", default="/kaggle/working/submission.zip")
    parser.add_argument(
        "--recovery-output",
        default="/kaggle/working/legalqa_recovery_latest.zip",
        help="Atomic cross-session checkpoint ZIP written periodically",
    )
    parser.add_argument(
        "--resume-bundle",
        default=None,
        help="Recovery ZIP attached as a Kaggle Dataset from an earlier session",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=18000)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--bm25-top-k", type=int, default=140)
    parser.add_argument("--dense-top-k", type=int, default=70)
    parser.add_argument("--fused-top-k", type=int, default=48)
    parser.add_argument("--rerank-top-k", type=int, default=24)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument(
        "--dense-index-dir",
        default="/kaggle/working/legalqa_dense_index",
        help="Writable/reusable global dense index directory",
    )
    parser.add_argument(
        "--dense-index-input",
        default=None,
        help="Optional completed dense index attached as a read-only Kaggle Dataset",
    )
    parser.add_argument("--force-dense-index", action="store_true")
    parser.add_argument(
        "--train-memory-dir",
        default="/kaggle/working/legalqa_dense_index/train_memory",
    )
    parser.add_argument("--train-memory-input", default=None)
    parser.add_argument("--force-train-memory", action="store_true")
    parser.add_argument(
        "--train-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dense-index-only",
        action="store_true",
        help="Build/checkpoint the global dense index, then exit for a two-session Kaggle run",
    )
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument(
        "--model-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load and execute all four runtime wrappers once before the long run",
    )
    parser.add_argument(
        "--reuse-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-unverified-budget", action="store_true")
    args = parser.parse_args()

    positive = {
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_batch_size": args.reranker_batch_size,
        "verifier_batch_size": args.verifier_batch_size,
        "reranker_max_length": args.reranker_max_length,
        "max_context_chars": args.max_context_chars,
        "max_new_tokens": args.max_new_tokens,
        "bm25_top_k": args.bm25_top_k,
        "dense_top_k": args.dense_top_k,
        "fused_top_k": args.fused_top_k,
        "rerank_top_k": args.rerank_top_k,
        "evidence_top_k": args.evidence_top_k,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Các tham số phải >= 1: {invalid}")
    if not args.device.startswith("cuda"):
        raise ValueError("Notebook inference-only yêu cầu device CUDA, ví dụ cuda:0.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải >= 1.")

    started = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    input_root = Path(args.input_root)
    working = Path(args.working)
    submission = Path(args.submission_output)
    working.mkdir(parents=True, exist_ok=True)
    submission.parent.mkdir(parents=True, exist_ok=True)
    environment = require_environment()

    official_public_path = find_unique_file(input_root, "public-official.json")
    public_path = Path(args.questions_file) if args.questions_file else official_public_path
    if not public_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy --questions-file: {public_path}")
    data_root = official_public_path.parent
    official_train_path = data_root / "train.json"
    train_path = Path(args.train_file) if args.train_file else official_train_path
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy official/selector train file: {train_path}")
    train_records = load_qa(train_path, require_answers=True)
    sibling_corpus = data_root / "selected-contexts"
    corpus_path = sibling_corpus if sibling_corpus.is_dir() else find_corpus(data_root)

    discovered = {role: find_model(input_root, role) for role in RUNTIME_MODEL_ROLES}
    generator_adapter = find_generator_adapter(input_root, args.generator_adapter)
    missing = sorted(role for role in RUNTIME_MODEL_ROLES if discovered[role] is None)
    if missing:
        raise FileNotFoundError(
            f"Inference v9 cần 4 checkpoint local (không dùng BARTpho); đang thiếu: {missing}. "
            "Attach Dataset checkpoint đã được BTC duyệt."
        )

    preflight_path = working / "preflight.json"
    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "preflight_kaggle.py"),
            "--input-root",
            str(input_root),
            "--profile",
            "balanced",
            "--inference-only",
            "--output",
            str(preflight_path),
        ]
    )
    environment["preflight"] = json.loads(preflight_path.read_text(encoding="utf-8"))

    print("[Fingerprint] hashing public questions and official corpus", flush=True)
    input_identity = {
        "questions_sha256": sha256_file(public_path),
        "train_sha256": sha256_file(train_path),
        "corpus_sha256": corpus_content_fingerprint(corpus_path),
    }
    print("[Fingerprint] identifying four local checkpoints", flush=True)
    model_identity = {
        role: checkpoint_fingerprint(path)
        for role, path in discovered.items()
    }
    model_identity["generator_adapter"] = (
        adapter_fingerprint(generator_adapter) if generator_adapter else None
    )

    config = load_config(args.config)
    config["paths"]["warmup"] = None
    config["paths"]["corpus"] = str(corpus_path)
    for role, path in discovered.items():
        config["paths"]["models"][role] = str(path)
    config["paths"]["models"]["bartpho"] = None
    config["paths"]["models"]["generator_adapter"] = (
        str(generator_adapter) if generator_adapter else None
    )
    style_profile = build_answer_style_profile(train_records)
    config["generation"]["answer_style_profile"] = style_profile
    config["retrieval"]["train_memory_enabled"] = bool(args.train_memory)
    if args.selection_weights:
        selection_payload = yaml.safe_load(
            Path(args.selection_weights).read_text(encoding="utf-8")
        )
        if not isinstance(selection_payload, dict):
            raise ValueError("--selection-weights phải chứa YAML Object.")
        if selection_payload.get("schema") != SELECTION_WEIGHTS_SCHEMA:
            raise ValueError(
                "Selector weights thuộc pipeline cũ hoặc thiếu schema. Hãy chạy lại "
                "notebook v9.1 selector calibration; expected "
                f"schema={SELECTION_WEIGHTS_SCHEMA!r}."
            )
        selection_override = selection_payload.get("selection", selection_payload)
        if not isinstance(selection_override, dict):
            raise ValueError("--selection-weights phải chứa YAML Object/selection Object.")
        config["selection"].update(selection_override)
        retrieval_override = selection_payload.get("retrieval")
        if retrieval_override is not None:
            if not isinstance(retrieval_override, dict):
                raise ValueError("retrieval trong tuning YAML phải là Object.")
            config["retrieval"].update(retrieval_override)
    config["runtime"].update(
        device=args.device,
        dtype="float16",
        batch_size=1,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        verifier_batch_size=args.verifier_batch_size,
        reranker_max_length=args.reranker_max_length,
        unload_after_stage=True,
        embedding_identity=model_identity["embedding"],
        corpus_identity=input_identity["corpus_sha256"],
        train_identity=input_identity["train_sha256"],
        train_path=str(train_path),
    )
    config["retrieval"].update(
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k,
        rerank_top_k=args.rerank_top_k,
        evidence_top_k=args.evidence_top_k,
    )
    config["generation"].update(
        qwen_enabled=True,
        bartpho_enabled=False,
        extractive_enabled=True,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        simple_max_new_tokens=min(
            args.max_new_tokens,
            int(config["generation"].get("simple_max_new_tokens", args.max_new_tokens)),
        ),
        complex_max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    runtime_fingerprint = {
        key: value
        for key, value in config["runtime"].items()
        if not key.endswith(("_path", "_dir", "_root", "_output"))
    }
    fingerprint_payload = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "inputs": input_identity,
        "models": model_identity,
        "runtime": runtime_fingerprint,
        "corpus": config["corpus"],
        "retrieval": config["retrieval"],
        "routing": config.get("routing", {}),
        "generation": config["generation"],
        "selection": config["selection"],
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    dense_index_dir = Path(args.dense_index_input or args.dense_index_dir)
    if args.dense_index_input and not (dense_index_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"--dense-index-input không chứa manifest.json: {dense_index_dir}"
        )
    if args.force_dense_index:
        resolved_dense = dense_index_dir.resolve()
        resolved_working = Path("/kaggle/working").resolve()
        try:
            resolved_dense.relative_to(resolved_working)
        except ValueError as exc:
            raise RuntimeError(
                "--force-dense-index chỉ được xóa index dưới /kaggle/working."
            ) from exc
        if resolved_dense.exists():
            shutil.rmtree(resolved_dense)
    nested_memory = dense_index_dir / "train_memory"
    train_memory_dir = Path(
        args.train_memory_input
        or (
            nested_memory
            if (
                args.dense_index_input
                and not args.train_file
                and (nested_memory / "manifest.json").is_file()
            )
            else args.train_memory_dir
        )
    )
    if args.train_memory_input and not (train_memory_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"--train-memory-input không chứa manifest.json: {train_memory_dir}"
        )
    if args.force_train_memory:
        resolved_memory = train_memory_dir.resolve()
        resolved_working = Path("/kaggle/working").resolve()
        try:
            resolved_memory.relative_to(resolved_working)
        except ValueError as exc:
            raise RuntimeError(
                "--force-train-memory chỉ được xóa index dưới /kaggle/working."
            ) from exc
        if resolved_memory.exists():
            shutil.rmtree(resolved_memory)
    config["runtime"]["dense_index_dir"] = str(dense_index_dir)
    config["runtime"]["train_memory_dir"] = str(train_memory_dir)
    config["paths"]["artifacts"] = str(working / "artifacts" / run_fingerprint)
    write_json(working / "answer_style_profile.json", style_profile)
    restored_recovery = None
    if args.resume_bundle:
        restored_recovery = restore_recovery_bundle(
            args.resume_bundle,
            working,
            expected_fingerprint=run_fingerprint,
        )
        write_json(preflight_path, environment["preflight"])
    config["runtime"].update(
        recovery_root=str(working),
        recovery_output=str(Path(args.recovery_output)),
        run_fingerprint=run_fingerprint,
        checkpoint_every=args.checkpoint_every,
    )
    resolved_config = working / "resolved_config.yaml"
    write_resolved_config(config, resolved_config)
    recovery = RecoveryManager.from_config(config)

    try:
        budget_report = count_system_budget(config)
        if not budget_report["strictly_below_4b"]:
            raise RuntimeError(f"Tổng tham số không còn dưới 4B: {budget_report}")
    except Exception as exc:
        budget_report = {"error": repr(exc), "must_verify_before_submission": True}
        if not args.allow_unverified_budget:
            raise RuntimeError("Không xác minh được parameter budget <4B; dừng trước inference.") from exc

    smoke_started = time.time()
    model_smoke_report = smoke_inference_models(config) if args.model_smoke else {"skipped": True}
    smoke_minutes = (time.time() - smoke_started) / 60
    if submission.exists():
        submission.unlink()

    database = working / "official_corpus.sqlite"
    index_started = time.time()
    index_report = build_sqlite_store(config, corpus_path, database, force=args.force_index)
    index_minutes = (time.time() - index_started) / 60

    if args.dense_index_only:
        encoder = None
        memory_report = {"enabled": False}
        try:
            encoder = DenseEncoder(config)
            with SQLiteLegalStore(database, readonly=True) as store:
                dense_report = ensure_global_dense_index(
                    config, store, encoder, config["runtime"]["dense_index_dir"]
                )
                if args.train_memory:
                    memory_report = ensure_train_memory_index(
                        config,
                        encoder,
                        train_path,
                        config["runtime"]["train_memory_dir"],
                    )
        finally:
            unload_model(encoder, "global embedding index-only")
        manifest = {
            "mode": "global_dense_index_only",
            "training_performed": False,
            "run_fingerprint": run_fingerprint,
            "inputs": {
                "questions": str(public_path),
                "official_public": str(official_public_path),
                "train": str(train_path),
                "corpus": str(corpus_path),
                **input_identity,
            },
            "models": {key: str(value) if value else None for key, value in config["paths"]["models"].items()},
            "model_identity": model_identity,
            "index": index_report,
            "global_dense_index": dense_report,
            "train_memory_index": memory_report,
            "answer_style_profile": str((working / "answer_style_profile.json").resolve()),
            "budget": budget_report,
            "timing_minutes": {"total": (time.time() - started) / 60},
        }
        write_json(working / "run_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        print(
            "\nDENSE INDEX HOÀN TẤT. Save Version, tạo Kaggle Dataset từ "
            f"{config['runtime']['dense_index_dir']} (gồm train_memory), rồi attach vào phiên inference.",
            flush=True,
        )
        return

    questions = load_qa(public_path, require_answers=False)
    retrieval_cache = working / "cache" / run_fingerprint / "retrieval.v5.jsonl"
    retrieval_started = time.time()
    evidence = retrieve_sqlite_batch(
        config,
        database,
        questions,
        retrieval_cache,
        reuse_cache=args.reuse_retrieval and not args.force_index,
    )
    retrieval_minutes = (time.time() - retrieval_started) / 60
    if recovery:
        recovery.snapshot("retrieval", len(evidence), len(questions), force=True)

    predictions = working / "predictions.json"
    generation_started = time.time()
    if config["generation"].get("agentic_enabled", False):
        with sqlite_bm25_search_fn(config, database) as agentic_search_fn:
            generation_report = generate_from_evidence(
                config, questions, evidence, predictions, agentic_search_fn
            )
    else:
        generation_report = generate_from_evidence(config, questions, evidence, predictions)
    generation_minutes = (time.time() - generation_started) / 60
    submission_report = create_submission(public_path, predictions, submission)

    manifest = {
        "mode": "full_sequential_inference_only",
        "training_performed": False,
        "run_fingerprint": run_fingerprint,
        "restored_recovery": restored_recovery,
        "environment": environment,
        "inputs": {
            "questions": str(public_path),
            "official_public": str(official_public_path),
            "train": str(train_path),
            "corpus": str(corpus_path),
            **input_identity,
        },
        "models": {key: str(value) if value else None for key, value in config["paths"]["models"].items()},
        "model_identity": model_identity,
        "model_smoke": model_smoke_report,
        "runtime": config["runtime"],
        "retrieval_config": config["retrieval"],
        "generation_config": config["generation"],
        "index": index_report,
        "global_dense_index": getattr(evidence, "index_report", {}),
        "train_memory_index": getattr(evidence, "memory_report", {}),
        "answer_style_profile": str((working / "answer_style_profile.json").resolve()),
        "budget": budget_report,
        "generation": generation_report,
        "submission": submission_report,
        "timing_minutes": {
            "model_smoke": smoke_minutes,
            "index": index_minutes,
            "retrieval": retrieval_minutes,
            "generation_and_selection": generation_minutes,
            "total": (time.time() - started) / 60,
        },
    }
    write_json(working / "run_manifest.json", manifest)
    if recovery:
        recovery.snapshot("complete", len(questions), len(questions), force=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"\nHOÀN TẤT INFERENCE-ONLY: {submission} ({submission.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
