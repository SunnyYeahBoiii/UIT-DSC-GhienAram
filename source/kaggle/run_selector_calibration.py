from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _find_split(root: Path) -> Path:
    candidates = sorted(
        {
            path.parent
            for path in root.rglob("selector_dev_questions.json")
            if (path.parent / "selector_train.json").is_file()
            and (path.parent / "selector_dev_references.json").is_file()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một selector split artifact: {candidates}")
    return candidates[0]


def _find_dense(root: Path) -> Path:
    candidates = []
    for manifest_path in root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("schema") == "global-dense-v1" and payload.get("status") == "complete":
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một completed global dense index: {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe grouped-dev inference and candidate-oracle selector tuning"
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--dense-index-input", default=None)
    parser.add_argument("--generator-adapter", default=None)
    parser.add_argument("--working", default="/kaggle/working/legalqa_selector_calibration")
    parser.add_argument("--trials", type=int, default=3000)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    split_dir = Path(args.split_dir) if args.split_dir else _find_split(input_root)
    split_manifest_path = split_dir / "manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Selector split thiếu manifest.json: {split_dir}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    grouped_dev_count = int(split_manifest.get("dev", 0))
    if not 300 <= grouped_dev_count <= 500:
        raise RuntimeError(
            f"Candidate-oracle yêu cầu grouped dev 300–500 câu, nhận {grouped_dev_count}."
        )
    dense_dir = (
        Path(args.dense_index_input)
        if args.dense_index_input
        else _find_dense(input_root)
    )
    working = Path(args.working)
    working.mkdir(parents=True, exist_ok=True)
    predictions_zip = working / "selector_dev_predictions.zip"
    command = [
        sys.executable,
        str(PROJECT / "kaggle" / "run_inference_only.py"),
        "--config",
        str(PROJECT / "configs" / "kaggle_inference_t4.yaml"),
        "--input-root",
        str(input_root),
        "--questions-file",
        str(split_dir / "selector_dev_questions.json"),
        "--train-file",
        str(split_dir / "selector_train.json"),
        "--dense-index-input",
        str(dense_dir),
        "--train-memory-dir",
        str(working / "selector_train_memory"),
        "--working",
        str(working / "inference"),
        "--submission-output",
        str(predictions_zip),
        "--recovery-output",
        str(working / "selector_recovery_latest.zip"),
        "--checkpoint-every",
        "25",
    ]
    if args.generator_adapter:
        command.extend(["--generator-adapter", args.generator_adapter])
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

    inference = working / "inference"
    report_path = working / "selector_report.json"
    weights_path = working / "tuned_selection.yaml"
    evaluate = [
        sys.executable,
        str(PROJECT / "scripts" / "evaluate_selector.py"),
        "--trace",
        str(inference / "predictions.trace.jsonl"),
        "--references",
        str(split_dir / "selector_dev_references.json"),
        "--config",
        str(inference / "resolved_config.yaml"),
        "--output",
        str(report_path),
        "--weights-output",
        str(weights_path),
        "--trials",
        str(args.trials),
    ]
    print(" ".join(evaluate), flush=True)
    subprocess.run(evaluate, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "grouped_dev_count": report["count"],
        "selected": report["selected_from_trace"],
        "candidate_oracle": report["candidate_oracle"],
        "tuned_holdout": report["tuning"]["holdout"],
        "weights": str(weights_path.resolve()),
        "full_report": str(report_path.resolve()),
    }
    (working / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
