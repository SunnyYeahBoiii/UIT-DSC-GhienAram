from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LegalQA on a rented GPU while keeping every recoverable artifact on persistent storage."
    )
    parser.add_argument("--persistent-root", required=True, help="Mounted persistent volume, ví dụ /workspace/legalqa.")
    parser.add_argument("--input-root", default=None, help="Mặc định <persistent-root>/input.")
    parser.add_argument("--profile", choices=("fast", "balanced", "full"), default="full")
    parser.add_argument("--train-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-reranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--qwen-max-samples", type=int, default=None)
    parser.add_argument("--reranker-max-examples", type=int, default=24000)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--reranker-grad-accum", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--qwen-checkpoint-steps", type=int, default=20)
    parser.add_argument("--reranker-checkpoint-steps", type=int, default=250)
    parser.add_argument("--checkpoint-limit", type=int, default=2)
    args = parser.parse_args()

    persistent_root = Path(args.persistent_root).expanduser().resolve()
    if persistent_root == Path(persistent_root.anchor):
        raise ValueError("Không được dùng filesystem root làm persistent-root.")
    if not persistent_root.is_dir():
        raise FileNotFoundError(f"Persistent volume chưa được mount hoặc không tồn tại: {persistent_root}")
    input_root = Path(args.input_root).expanduser().resolve() if args.input_root else persistent_root / "input"
    if not input_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy input-root: {input_root}")

    usage = shutil.disk_usage(persistent_root)
    if usage.free < 40 * 1024**3:
        raise RuntimeError(
            f"Persistent volume chỉ còn {usage.free / 1024**3:.1f} GiB; cần ít nhất 40 GiB trống trước khi chạy."
        )
    working = persistent_root / "legalqa_run"
    submission = persistent_root / "submission.zip"
    working.mkdir(parents=True, exist_ok=True)
    sentinel = persistent_root / ".legalqa_persistent_volume"
    if not sentinel.exists():
        sentinel.write_text(
            "LegalQA persistent-volume sentinel. Do not delete while a resumable run exists.\n",
            encoding="utf-8",
        )

    command = [
        sys.executable,
        str(PROJECT / "kaggle" / "run_end_to_end.py"),
        "--config",
        str(PROJECT / "configs" / "kaggle.yaml"),
        "--profile",
        args.profile,
        "--input-root",
        str(input_root),
        "--working",
        str(working),
        "--submission-output",
        str(submission),
        "--train-qwen" if args.train_qwen else "--no-train-qwen",
        "--train-reranker" if args.train_reranker else "--no-train-reranker",
        "--resume-training" if args.resume_training else "--no-resume-training",
        "--reranker-max-examples",
        str(args.reranker_max_examples),
        "--reranker-batch-size",
        str(args.reranker_batch_size),
        "--reranker-grad-accum",
        str(args.reranker_grad_accum),
        "--reranker-max-length",
        str(args.reranker_max_length),
        "--reranker-checkpoint-steps",
        str(args.reranker_checkpoint_steps),
        "--qwen-checkpoint-steps",
        str(args.qwen_checkpoint_steps),
        "--checkpoint-limit",
        str(args.checkpoint_limit),
    ]
    if args.qwen_max_samples is not None:
        command += ["--qwen-max-samples", str(args.qwen_max_samples)]
    if args.force_index:
        command.append("--force-index")
    if args.force_train:
        command.append("--force-train")

    launch = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "persistent_root": str(persistent_root),
        "input_root": str(input_root),
        "working": str(working),
        "submission": str(submission),
        "free_gib": round(usage.free / 1024**3, 2),
        "command": command,
    }
    (working / "server_launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(PROJECT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    }
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    print(f"HOÀN TẤT: {submission}", flush=True)


if __name__ == "__main__":
    main()
