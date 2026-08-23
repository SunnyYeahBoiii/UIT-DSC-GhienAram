from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT / "scripts"))

from legalqa.core.data import deterministic_split, load_qa, qa_as_object, write_json  # noqa: E402
from legalqa.core.config import load_config  # noqa: E402
from legalqa.core.fingerprint import (  # noqa: E402
    checkpoint_fingerprint,
    corpus_content_fingerprint,
    sha256_file,
)
from legalqa.storage.sqlite_store import build_sqlite_store  # noqa: E402
from prepare_sft_evidence import build_sft_evidence  # noqa: E402
from run_end_to_end import (  # noqa: E402
    find_corpus,
    find_model,
    find_unique_file,
    require_environment,
    run_command,
    write_resolved_config,
)


def _find_resume_root(input_root: Path) -> Path | None:
    candidates = sorted(
        {
            path.parent
            for path in input_root.rglob("resume_signature.json")
            if any(child.is_dir() and child.name.startswith("checkpoint-") for child in path.parent.iterdir())
        }
    )
    if len(candidates) > 1:
        raise RuntimeError(f"Có nhiều SFT training state để resume: {candidates}")
    return candidates[0] if candidates else None


def _copy_resume(source: Path, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"[SFT resume] copied persistent state: {source} -> {target}", flush=True)


def _require_official_scorer_deps() -> None:
    import importlib.util

    missing = [name for name in ("nltk", "rouge_score") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Thiếu package chấm điểm chính thức của BTC: {missing}. "
            "select_best_sft_checkpoint.py cần nltk + rouge-score để chọn epoch theo METEOR/ROUGE-L thật; "
            "cài offline từ wheelhouse_scoring/ trước khi train, tránh tốn GPU rồi fail ở bước chọn epoch."
        )
    nltk_data_dir = PROJECT / "wheelhouse_scoring" / "nltk_data"
    if nltk_data_dir.is_dir():
        os.environ["NLTK_DATA"] = str(nltk_data_dir)
    from nltk.translate.meteor_score import meteor_score

    try:
        # nltk.data.find('corpora/wordnet') báo lỗi dù corpus có sẵn, và corpus phải ở dạng đã giải nén
        # (không chỉ .zip — cơ chế tự-giải-nén-lúc-chạy của nltk không đáng tin cậy trên mọi filesystem)
        # nên cách kiểm tra đáng tin cậy nhất là gọi trực tiếp hàm sẽ dùng lúc chấm điểm.
        meteor_score([["kiểm", "tra"]], ["kiểm", "tra"])
    except LookupError as error:
        raise RuntimeError(
            "Thiếu nltk corpus wordnet/omw-1.4 (cần cho meteor_score, môi trường Kaggle không có internet). "
            "Attach wheelhouse_scoring/nltk_data làm Kaggle dataset và trỏ NLTK_DATA vào đó trước khi chạy."
        ) from error


def _package_adapter(adapter_dir: Path, output: Path) -> None:
    include_prefixes = ("adapter_", "tokenizer", "special_tokens", "chat_template")
    include_names = {"README.md", "added_tokens.json"}
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.iterdir()):
            if path.is_file() and (path.name.startswith(include_prefixes) or path.name in include_names):
                archive.write(path, path.name)
    os.replace(temporary, output)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SFT LoRA thuần cho Qwen3.5-2B trên train_sft.json đã preprocess")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--working", default="/kaggle/working/legalqa_sft")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="dùng khi không có dev_sft.json riêng đính kèm")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3500,
        help="ngân sách ký tự evidence riêng cho SFT, forward cho cả train và select_best_sft_checkpoint",
    )
    parser.add_argument("--max-target-tokens", type=int, default=700)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=("transformers", "unsloth"),
        default="transformers",
        help="unsloth cần internet để pip install trước khi chạy notebook; adapter sinh ra vẫn là "
        "PEFT chuẩn nên select_best_sft_checkpoint.py/inference không cần đổi gì.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--train-pool-samples",
        type=int,
        default=None,
        help="pool lớn hơn --max-train-samples để mỗi epoch train random subset khác nhau (SGD-style); "
        "để trống = mỗi epoch dùng đúng cùng --max-train-samples câu như cũ.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--print-limit", type=int, default=5, help="số câu in chi tiết ref/pred khi chọn epoch")
    parser.add_argument("--resume-input", default=None)
    parser.add_argument(
        "--dense-index-input",
        default=None,
        help="Thư mục dense index (global-dense-v1) đính kèm dạng Kaggle dataset; tự dò nếu bỏ trống",
    )
    args = parser.parse_args()
    if not 0.05 <= args.dev_ratio <= 0.4:
        raise ValueError("--dev-ratio phải nằm trong [0.05, 0.4] để dev đủ ổn định cho việc chọn epoch.")

    started = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    input_root = Path(args.input_root)
    working = Path(args.working)
    working.mkdir(parents=True, exist_ok=True)
    environment = require_environment()
    _require_official_scorer_deps()

    train_sft_path = find_unique_file(input_root, "train_sft.json")
    dev_sft_candidates = sorted(path for path in input_root.rglob("dev_sft.json") if path.is_file())
    if len(dev_sft_candidates) > 1:
        raise RuntimeError(f"Có nhiều dev_sft.json: {dev_sft_candidates}")
    generator_path = find_model(input_root, "generator")
    if generator_path is None:
        raise FileNotFoundError("Thiếu approved Qwen3.5-2B checkpoint.")
    embedding_path = find_model(input_root, "embedding")
    reranker_path = find_model(input_root, "reranker")
    corpus_path = find_corpus(input_root)

    dense_index_dir = None
    if args.dense_index_input:
        dense_index_dir = Path(args.dense_index_input)
    else:
        manifest_candidates = []
        for manifest_path in input_root.rglob("manifest.json"):
            if manifest_path.parent.name == "train_memory":
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("schema") == "global-dense-v1" and payload.get("status") == "complete":
                manifest_candidates.append(manifest_path.parent)
        if len(manifest_candidates) > 1:
            raise RuntimeError(f"Chỉ attach tối đa một global dense index: {manifest_candidates}")
        dense_index_dir = manifest_candidates[0] if manifest_candidates else None
    if dense_index_dir is not None and not (dense_index_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"--dense-index-input không chứa manifest.json: {dense_index_dir}")

    config = load_config(args.config)
    config["paths"]["models"]["generator"] = str(generator_path)
    config["paths"]["models"]["generator_adapter"] = None
    config["paths"]["models"]["embedding"] = str(embedding_path) if embedding_path else None
    config["paths"]["models"]["reranker"] = str(reranker_path) if reranker_path else None
    config["paths"]["corpus"] = str(corpus_path)
    if dense_index_dir is not None:
        config["runtime"]["dense_index_dir"] = str(dense_index_dir)
    # ensure_global_dense_index() fingerprint corpus/model theo runtime.corpus_identity/
    # embedding_identity (nội dung, ổn định) nếu có, nếu không sẽ rơi về path tuyệt đối —
    # mà path mount Kaggle input đổi giữa session build index và session train SFT nên
    # luôn báo "không khớp" dù corpus/model giống hệt. Set giống hệt run_inference_only.py.
    if embedding_path is not None:
        config["runtime"]["embedding_identity"] = checkpoint_fingerprint(embedding_path)
    config["runtime"]["corpus_identity"] = corpus_content_fingerprint(corpus_path)
    resolved_config = working / "resolved_config.yaml"
    write_resolved_config(config, resolved_config)

    database = working / "official_corpus.sqlite"
    build_sqlite_store(config, corpus_path, database)

    data_dir = working / "data"
    if dev_sft_candidates:
        dev_split = dev_sft_candidates[0]
        dev_qids = {record.question_id for record in load_qa(dev_split, require_answers=False)}
        train_records = [
            record
            for record in load_qa(train_sft_path, require_answers=True)
            if record.question_id not in dev_qids
        ]
        train_split = data_dir / "train_sft_split.json"
        write_json(train_split, qa_as_object(train_records))
        split_info = {
            "source": "attached dev_sft.json",
            "dev_ratio_requested": None,
            "train": len(train_records),
            "dev": len(dev_qids),
        }
    else:
        records = load_qa(train_sft_path, require_answers=True)
        train_records, dev_records = deterministic_split(records, args.dev_ratio, int(config["seed"]))
        train_split = data_dir / "train_sft_split.json"
        dev_split = data_dir / "dev_sft_split.json"
        write_json(train_split, qa_as_object(train_records))
        write_json(dev_split, qa_as_object(dev_records))
        split_info = {
            "source": "deterministic_split",
            "dev_ratio_requested": args.dev_ratio,
            "train": len(train_records),
            "dev": len(dev_records),
        }

    # Chỉ build evidence cho đúng tập câu hỏi thật sự dùng ở downstream (train đã cắt theo
    # --max-train-samples + dev đã cắt theo --max-eval-samples) — tránh chạy full RAG
    # (dense encode + rerank) trên toàn bộ train_sft.json khi chỉ đang smoke-test vài chục câu.
    train_for_evidence = load_qa(train_split, require_answers=False)
    train_pool_size = args.train_pool_samples or args.max_train_samples
    if train_pool_size:
        train_for_evidence = train_for_evidence[:train_pool_size]
    dev_for_evidence = load_qa(dev_split, require_answers=False)
    if args.max_eval_samples:
        # PHẢI cùng seed + cùng thứ tự sort-rồi-shuffle với select_best_sft_checkpoint.py
        # (xem shuffle ở đó khi cắt --max-eval-samples), nếu không 2 bên chọn ra 2 tập N câu
        # khác nhau -> lúc eval thiếu evidence cho câu được chọn, bị skip oan.
        dev_for_evidence = sorted(dev_for_evidence, key=lambda row: row.question_id)
        random.Random(int(config["seed"])).shuffle(dev_for_evidence)
        dev_for_evidence = dev_for_evidence[: args.max_eval_samples]
    seen_ids = set()
    evidence_records = []
    for record in (*train_for_evidence, *dev_for_evidence):
        if record.question_id in seen_ids:
            continue
        seen_ids.add(record.question_id)
        evidence_records.append(record)

    evidence_path = working / "data" / "sft_evidence.jsonl"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_report = build_sft_evidence(config, database, evidence_records, evidence_path)
    print(f"[SFT evidence] {evidence_report}", flush=True)

    adapter_output = working / "qwen-sft-lora"
    resume_input = Path(args.resume_input) if args.resume_input else _find_resume_root(input_root)
    if resume_input:
        _copy_resume(resume_input, adapter_output)

    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "train_qwen_sft_lora.py"),
            "--config",
            str(resolved_config),
            "--train-file",
            str(train_split),
            "--dev-file",
            str(dev_split),
            "--output",
            str(adapter_output),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--grad-accum",
            str(args.grad_accum),
            "--max-length",
            str(args.max_length),
            "--max-target-tokens",
            str(args.max_target_tokens),
            "--evidence",
            str(evidence_path),
            "--max-context-chars",
            str(args.max_context_chars),
            "--lora-r",
            str(args.lora_r),
            "--load-in-4bit",
            "--backend",
            args.backend,
            "--resume-from-checkpoint",
            "auto",
            *(["--max-train-samples", str(args.max_train_samples)] if args.max_train_samples else []),
            *(["--train-pool-samples", str(args.train_pool_samples)] if args.train_pool_samples else []),
            *(["--max-eval-samples", str(args.max_eval_samples)] if args.max_eval_samples else []),
        ]
    )

    select_command = [
        sys.executable,
        str(PROJECT / "scripts" / "select_best_sft_checkpoint.py"),
        "--config",
        str(resolved_config),
        "--checkpoints-root",
        str(adapter_output),
        "--dev-file",
        str(dev_split),
        "--evidence",
        str(evidence_path),
        "--max-context-chars",
        str(args.max_context_chars),
        "--load-in-4bit",
    ]
    if args.max_eval_samples:
        select_command.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if args.print_limit is not None:
        select_command.extend(["--print-limit", str(args.print_limit)])
    run_command(select_command)

    selection_report = json.loads((adapter_output / "epoch_selection_report.json").read_text(encoding="utf-8"))
    best_adapter = Path(selection_report["best_output"])
    adapter_zip = working / "qwen_sft_lora_adapter.zip"
    _package_adapter(best_adapter, adapter_zip)

    manifest = {
        "mode": "plain_sft_lora_epoch_selection",
        "inputs": {
            "train_sft": str(train_sft_path),
            "train_sft_sha256": sha256_file(train_sft_path),
            "corpus": str(corpus_path),
            "dense_index_dir": str(dense_index_dir) if dense_index_dir else None,
            "evidence": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_report": evidence_report,
            "max_context_chars": args.max_context_chars,
            "max_train_samples": args.max_train_samples,
            "train_pool_samples": args.train_pool_samples,
        },
        "environment": environment,
        "split": split_info,
        "epoch_selection": selection_report,
        "adapter_zip": str(adapter_zip.resolve()),
        "elapsed_hours": (time.time() - started) / 3600,
    }
    write_json(working / "sft_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
