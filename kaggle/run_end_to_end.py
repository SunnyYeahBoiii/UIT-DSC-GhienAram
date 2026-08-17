from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.core.budget import count_system_budget  # noqa: E402
from legalqa.core.config import load_config  # noqa: E402
from legalqa.core.data import deterministic_split, load_qa, qa_as_object, write_json  # noqa: E402
from legalqa.pipeline import generate_from_evidence  # noqa: E402
from legalqa.storage.sqlite_retrieval import retrieve_sqlite_batch, sqlite_bm25_search_fn  # noqa: E402
from legalqa.storage.sqlite_store import build_sqlite_store, prepare_sqlite_weak_pairs  # noqa: E402
from legalqa.core.submission import create_submission  # noqa: E402


MODEL_ALIASES = {
    "generator": ("qwen3.5-2b", "qwen35-2b", "qwen352b"),
    "embedding": ("vietnamese_embedding", "vietnamese-embedding", "vietnameseembedding"),
    "reranker": ("vietnamese_reranker", "vietnamese-reranker", "vietnamesereranker"),
    "bartpho": ("bartpho-syllable", "bartphosyllable"),
    "verifier": ("vilegalbert", "vi-legal-bert"),
}


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_unique_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Cần đúng một {name} dưới {root}; tìm thấy {len(matches)}: {matches[:8]}")
    return matches[0]


def find_corpus(root: Path) -> Path:
    named = sorted(path for path in root.rglob("selected-contexts") if path.is_dir())
    if named:
        return max(named, key=lambda path: sum(1 for _ in path.glob("context_*.json")))
    candidates = []
    for directory in [root, *[path for path in root.iterdir() if path.is_dir()]]:
        count = sum(1 for _ in directory.glob("context_*.json"))
        if count:
            candidates.append((count, directory))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy selected-contexts hoặc thư mục chứa context_*.json.")
    return max(candidates)[1]


def find_model(root: Path, role: str) -> Path | None:
    environment = os.environ.get(f"LEGALQA_{role.upper()}_DIR")
    if environment:
        path = Path(environment)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Checkpoint {role} không có config.json: {path}")
        return path
    aliases = tuple(normalized(alias) for alias in MODEL_ALIASES[role])
    candidates = []
    for config_path in root.rglob("config.json"):
        parent = config_path.parent
        haystack = normalized(str(parent))
        if any(alias in haystack for alias in aliases):
            candidates.append(parent)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), len(str(path)), str(path)))
    exact = [path for path in candidates if normalized(path.name) in aliases]
    preferred = exact or candidates
    if len(preferred) > 1:
        variable = f"LEGALQA_{role.upper()}_DIR"
        raise RuntimeError(
            f"Có nhiều checkpoint phù hợp cho {role}: {preferred}. "
            f"Chỉ attach một bản hoặc đặt biến {variable}."
        )
    return preferred[0]


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def run_command(arguments: list[str]) -> None:
    print("\n$", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env={**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(PROJECT / "src")})


def prepare_train_dev(config: dict[str, Any], train_path: Path) -> dict[str, Any]:
    records = load_qa(train_path, require_answers=True)
    train, dev = deterministic_split(records, float(config["split"]["dev_ratio"]), int(config["seed"]))
    data_dir = Path(config["paths"]["artifacts"]) / "data"
    write_json(data_dir / "train.json", qa_as_object(train))
    write_json(data_dir / "dev.json", qa_as_object(dev))
    write_json(data_dir / "train_all.json", qa_as_object(records))
    return {"all": len(records), "train": len(train), "dev": len(dev), "data_dir": str(data_dir)}


def require_environment() -> dict[str, Any]:
    import torch
    import transformers

    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "auto_multimodal": hasattr(transformers, "AutoModelForMultimodalLM"),
    }
    if not result["cuda"]:
        raise RuntimeError("Kaggle Notebook phải bật GPU Accelerator.")
    if not result["auto_multimodal"]:
        raise RuntimeError(
            "Transformers trong notebook chưa hỗ trợ Qwen3.5. Hãy attach wheelhouse đã chuẩn bị "
            "và cài offline trước khi chạy pipeline."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="UIT DSC 2026 Task 2: Kaggle train -> submission.zip")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--working", default="/kaggle/working/legalqa_run")
    parser.add_argument(
        "--submission-output",
        default=None,
        help="Mặc định là submission.zip cạnh thư mục working; đặt đường dẫn trên persistent volume khi chạy server.",
    )
    parser.add_argument("--profile", choices=("fast", "balanced", "full"), default="balanced")
    parser.add_argument("--train-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-reranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--no-reuse-retrieval", action="store_true")
    parser.add_argument("--allow-unverified-budget", action="store_true")
    parser.add_argument("--qwen-max-samples", type=int, default=None)
    parser.add_argument("--reranker-max-examples", type=int, default=24000)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--reranker-grad-accum", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-checkpoint-steps", type=int, default=250)
    parser.add_argument("--qwen-checkpoint-steps", type=int, default=20)
    parser.add_argument("--checkpoint-limit", type=int, default=2)
    parser.add_argument(
        "--resume-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự tiếp tục checkpoint mới nhất, gồm optimizer/scheduler/RNG.",
    )
    args = parser.parse_args()
    positive_values = {
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_grad_accum": args.reranker_grad_accum,
        "reranker_max_length": args.reranker_max_length,
        "reranker_checkpoint_steps": args.reranker_checkpoint_steps,
        "qwen_checkpoint_steps": args.qwen_checkpoint_steps,
        "checkpoint_limit": args.checkpoint_limit,
    }
    invalid = {key: value for key, value in positive_values.items() if value < 1}
    if invalid:
        raise ValueError(f"Các tham số phải >= 1: {invalid}")

    started = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_root, working = Path(args.input_root), Path(args.working)
    working.mkdir(parents=True, exist_ok=True)
    submission = Path(args.submission_output) if args.submission_output else working.parent / "submission.zip"
    submission.parent.mkdir(parents=True, exist_ok=True)
    environment = require_environment()
    config = load_config(args.config)
    config["paths"]["artifacts"] = str(working / "artifacts")
    public_path = find_unique_file(input_root, "public-official.json")
    # The source package may contain smoke-test train.json artifacts. The
    # official train file and corpus are anchored to the unique public file.
    data_root = public_path.parent
    train_path = data_root / "train.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy train.json cạnh public-official.json: {train_path}")
    sibling_corpus = data_root / "selected-contexts"
    corpus_path = sibling_corpus if sibling_corpus.is_dir() else find_corpus(data_root)
    config["paths"]["warmup"] = str(train_path)
    config["paths"]["corpus"] = str(corpus_path)

    discovered = {role: find_model(input_root, role) for role in MODEL_ALIASES}
    required = {"generator"}
    if args.profile in {"balanced", "full"}:
        required |= {"embedding", "reranker", "verifier"}
    if args.profile == "full":
        required.add("bartpho")
    missing = sorted(role for role in required if discovered[role] is None)
    if missing:
        raise FileNotFoundError(
            f"Thiếu checkpoint local cho profile {args.profile}: {missing}. "
            "Attach Kaggle Dataset chứa snapshot đã được BTC duyệt hoặc đặt LEGALQA_<ROLE>_DIR."
        )

    # Fail before the expensive 8,500-document index if any package, CUDA
    # binary, tokenizer, checkpoint shard, PEFT or Trainer API is incompatible.
    preflight_path = working / "preflight.json"
    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "preflight_kaggle.py"),
            "--input-root",
            str(input_root),
            "--profile",
            args.profile,
            "--output",
            str(preflight_path),
        ]
    )
    environment["preflight"] = json.loads(preflight_path.read_text(encoding="utf-8"))
    for role, path in discovered.items():
        config["paths"]["models"][role] = str(path) if path else None

    if args.profile == "fast":
        config["paths"]["models"].update(embedding=None, reranker=None, verifier=None, bartpho=None)
        config["retrieval"].update(dense_top_k=0, rerank_top_k=0)
        config["selection"]["legal_semantic"] = 0.0
        config["generation"]["bartpho_enabled"] = False
    elif args.profile == "balanced":
        config["paths"]["models"]["bartpho"] = None
        config["generation"]["bartpho_enabled"] = False
    else:
        config["generation"]["bartpho_enabled"] = True

    resolved_config = working / "resolved_config.yaml"
    write_resolved_config(config, resolved_config)
    split_report = prepare_train_dev(config, train_path)
    database = working / "official_corpus.sqlite"
    index_report = build_sqlite_store(config, corpus_path, database, force=args.force_index)
    pairs_path = Path(split_report["data_dir"]) / "retrieval_pairs.jsonl"
    pair_report = prepare_sqlite_weak_pairs(
        Path(split_report["data_dir"]) / "train.json", database, pairs_path, top_k=40, negatives=3
    )

    if args.train_reranker and args.profile in {"balanced", "full"}:
        reranker_output = working / "models" / "Vietnamese_Reranker-official"
        if args.force_train or not (reranker_output / "config.json").exists():
            run_command(
                [
                    sys.executable, str(PROJECT / "scripts" / "train_reranker.py"),
                    "--config", str(resolved_config), "--pairs", str(pairs_path),
                    "--output", str(reranker_output), "--epochs", "1",
                    "--batch-size", str(args.reranker_batch_size),
                    "--grad-accum", str(args.reranker_grad_accum),
                    "--learning-rate", "2e-5", "--max-length", str(args.reranker_max_length),
                    "--max-examples", str(args.reranker_max_examples),
                    "--checkpoint-steps", str(args.reranker_checkpoint_steps),
                    "--save-total-limit", str(args.checkpoint_limit),
                    "--resume-from-checkpoint", "auto" if args.resume_training else "none",
                    "--gradient-checkpointing",
                ]
            )
        config["paths"]["models"]["reranker"] = str(reranker_output)
        write_resolved_config(config, resolved_config)

    if args.train_qwen:
        adapter_output = working / "models" / "qwen-official-lora"
        if args.force_train or not (adapter_output / "adapter_config.json").exists():
            command = [
                sys.executable, str(PROJECT / "scripts" / "train_qwen_lora.py"),
                "--config", str(resolved_config), "--pairs", str(pairs_path),
                "--output", str(adapter_output), "--epochs", "1", "--learning-rate", "1e-4",
                "--batch-size", "1", "--grad-accum", "16", "--max-length", "1536",
                "--max-target-tokens", "850", "--lora-r", "8", "--load-in-4bit",
                "--checkpoint-steps", str(args.qwen_checkpoint_steps),
                "--save-total-limit", str(args.checkpoint_limit),
                "--resume-from-checkpoint", "auto" if args.resume_training else "none",
            ]
            if args.qwen_max_samples:
                command += ["--max-train-samples", str(args.qwen_max_samples)]
            run_command(command)
            # The adapter path is stable across pilot/final runs, while its tensors change.
            # Remove answers generated by an older adapter at the same path.
            qwen_cache = Path(config["paths"]["artifacts"]) / "candidate_cache" / "qwen.jsonl"
            if qwen_cache.exists():
                qwen_cache.unlink()
                print(f"[Cache] removed stale Qwen candidates: {qwen_cache}", flush=True)
        config["paths"]["models"]["generator_adapter"] = str(adapter_output)
        write_resolved_config(config, resolved_config)

    try:
        budget_report = count_system_budget(config)
        if not budget_report["strictly_below_4b"]:
            raise RuntimeError(f"Tổng tham số không còn dưới 4B: {budget_report}")
    except Exception as exc:
        # Keep the failure visible in the manifest; checkpoint metadata differs across Transformers revisions.
        budget_report = {"error": repr(exc), "must_verify_before_submission": True}
        print("CẢNH BÁO COUNT-BUDGET:", budget_report, flush=True)
        if not args.allow_unverified_budget:
            raise RuntimeError(
                "Không thể xác minh tổng tham số <4B; pipeline dừng trước khi tạo bài nộp. "
                "Chỉ dùng --allow-unverified-budget để debug, không dùng cho bài thi."
            ) from exc

    public_questions = load_qa(public_path, require_answers=False)
    # Retrieval output depends on the profile and on a freshly trained reranker.
    retrieval_cache = working / f"public_retrieval.{args.profile}.jsonl"
    evidence = retrieve_sqlite_batch(
        config, database, public_questions, retrieval_cache,
        reuse_cache=not args.no_reuse_retrieval and not args.force_train,
    )
    predictions = working / "predictions.json"
    if config["generation"].get("agentic_enabled", False):
        with sqlite_bm25_search_fn(config, database) as agentic_search_fn:
            generation_report = generate_from_evidence(
                config, public_questions, evidence, predictions, agentic_search_fn
            )
    else:
        generation_report = generate_from_evidence(config, public_questions, evidence, predictions)
    submission_report = create_submission(public_path, predictions, submission)

    manifest = {
        "environment": environment,
        "profile": args.profile,
        "inputs": {"train": str(train_path), "public": str(public_path), "corpus": str(corpus_path)},
        "models": {key: str(value) if value else None for key, value in config["paths"]["models"].items()},
        "split": split_report, "index": index_report, "weak_pairs": pair_report,
        "budget": budget_report, "generation": generation_report, "submission": submission_report,
        "recovery": {
            "working": str(working.resolve()),
            "resume_training": args.resume_training,
            "qwen_checkpoint_steps": args.qwen_checkpoint_steps,
            "reranker_checkpoint_steps": args.reranker_checkpoint_steps,
            "checkpoint_limit": args.checkpoint_limit,
        },
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(working / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"\nHOÀN TẤT: {submission} ({submission.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
