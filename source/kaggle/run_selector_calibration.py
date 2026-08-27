from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.data import load_qa, read_json  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_identity(path: Path) -> str:
    """Path-independent identity used to collapse exported adapter aliases."""
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_sha256_file(path / "adapter_model.safetensors").encode("ascii"))
    return digest.hexdigest()


def _adapter_preference(path: Path) -> tuple[int, int, str]:
    canonical = 0 if path.name == "qwen-official-lora-split" else 1
    return canonical, len(path.parts), path.as_posix()


def _selector_split_identity(path: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "selector_train.json",
        "selector_dev_questions.json",
        "selector_dev_references.json",
    ):
        payload = read_json(path / name)
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _find_split(root: Path) -> Path:
    candidates = sorted(
        {
            path.parent
            for path in root.rglob("selector_dev_questions.json")
            if (path.parent / "selector_train.json").is_file()
            and (path.parent / "selector_dev_references.json").is_file()
        }
    )
    identities: dict[str, list[Path]] = {}
    for path in candidates:
        identities.setdefault(_selector_split_identity(path), []).append(path)
    if len(identities) != 1:
        raise RuntimeError(f"Cần đúng một nội dung selector split artifact: {identities}")
    aliases = next(iter(identities.values()))
    selected = min(aliases, key=lambda path: (len(path.parts), path.as_posix()))
    if len(aliases) > 1:
        print(f"[Selector split] aliases cùng SHA-256; chọn {selected}: {aliases}", flush=True)
    return selected


def _find_dense(root: Path) -> Path:
    candidates: dict[str, list[Path]] = {}
    for manifest_path in root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("schema") == "global-dense-v2-structured-headings"
            and payload.get("status") == "complete"
        ):
            identity = str(payload.get("identity", ""))
            if not identity:
                raise RuntimeError(f"Dense manifest hoàn tất nhưng thiếu identity: {manifest_path}")
            candidates.setdefault(identity, []).append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một nội dung completed global dense index: {candidates}")
    aliases = next(iter(candidates.values()))
    selected = min(aliases, key=lambda path: (len(path.parts), path.as_posix()))
    if len(aliases) > 1:
        print(f"[Dense index] aliases cùng identity; chọn {selected}: {aliases}", flush=True)
    return selected


def _validate_selector_split(split_dir: Path, manifest: dict) -> dict[str, int]:
    """Fail before GPU inference if a copied selector artifact is incomplete or mixed."""
    questions = load_qa(split_dir / "selector_dev_questions.json")
    train = load_qa(split_dir / "selector_train.json", require_answers=True)
    references = read_json(split_dir / "selector_dev_references.json")
    if not isinstance(references, dict):
        raise ValueError("selector_dev_references.json phải là JSON Object.")
    reference_ids: set[str] = set()
    for raw_id, item in references.items():
        question_id = str(raw_id)
        if not question_id or not isinstance(item, dict):
            raise ValueError(f"Reference selector không hợp lệ: {raw_id!r}")
        answer = item.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Reference selector {question_id} thiếu answer dạng chuỗi.")
        reference_ids.add(question_id)
    question_ids = {item.question_id for item in questions}
    train_ids = {item.question_id for item in train}
    if question_ids != reference_ids:
        raise RuntimeError(
            "ID selector dev questions/references không khớp: "
            f"missing_refs={sorted(question_ids - reference_ids)[:10]}, "
            f"extra_refs={sorted(reference_ids - question_ids)[:10]}"
        )
    overlap = question_ids & train_ids
    if overlap:
        raise RuntimeError(f"Selector split bị leakage train/dev: {sorted(overlap)[:10]}")
    actual = {"dev": len(question_ids), "train": len(train_ids), "total": len(question_ids | train_ids)}
    for key, value in actual.items():
        if int(manifest.get(key, -1)) != value:
            raise RuntimeError(
                f"Selector manifest {key}={manifest.get(key)!r} nhưng dữ liệu thực tế={value}."
            )
    if int(manifest.get("overlap_ids", 0)) != 0:
        raise RuntimeError(f"Selector manifest khai báo overlap_ids != 0: {manifest!r}")
    if not 300 <= actual["dev"] <= 500:
        raise RuntimeError(
            f"Candidate-oracle yêu cầu grouped dev 300–500 câu, nhận {actual['dev']}."
        )
    return actual


def _validate_split_adapter(path: Path) -> Path:
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
        path / "training_complete.json",
        path / "qlora_scope.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"QLoRA split adapter chưa hoàn tất: {missing}")
    scope = json.loads((path / "qlora_scope.json").read_text(encoding="utf-8"))
    if scope.get("train_scope") != "split":
        raise RuntimeError(
            f"Selector calibration bắt buộc dùng adapter scope='split', nhận {scope!r}: {path}"
        )
    complete = json.loads(
        (path / "training_complete.json").read_text(encoding="utf-8")
    )
    if int(complete.get("global_step", 0)) <= 0:
        raise RuntimeError(f"Adapter split không có global_step hoàn tất: {path}")
    return path


def _find_split_adapter(root: Path, explicit: str | None = None) -> Path:
    if explicit:
        return _validate_split_adapter(Path(explicit))
    candidates = sorted(
        {
            path.parent
            for path in root.rglob("qlora_scope.json")
            if (path.parent / "training_complete.json").is_file()
            and (path.parent / "adapter_config.json").is_file()
        }
    )
    valid = []
    for candidate in candidates:
        try:
            scope = json.loads(
                (candidate / "qlora_scope.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if scope.get("train_scope") == "split":
            valid.append(candidate)
    if not valid:
        raise RuntimeError(
            "Cần đúng một completed QLoRA adapter scope='split' cho calibration; "
            f"tìm thấy: {valid}"
        )
    validated = [_validate_split_adapter(path) for path in valid]
    identities: dict[str, list[Path]] = {}
    for path in validated:
        identities.setdefault(_adapter_identity(path), []).append(path)
    if len(identities) != 1:
        report = {
            identity[:12]: [str(path) for path in paths]
            for identity, paths in identities.items()
        }
        raise RuntimeError(
            "Phát hiện nhiều QLoRA split adapter có trọng số KHÁC nhau; "
            f"hãy chỉ attach một checkpoint: {report}"
        )
    aliases = next(iter(identities.values()))
    selected = min(aliases, key=_adapter_preference)
    if len(aliases) > 1:
        print(
            "[Selector calibration] các thư mục adapter là bản sao cùng SHA-256; "
            f"chọn {selected}; aliases={aliases}",
            flush=True,
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe grouped-dev inference and candidate-oracle selector tuning"
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--dense-index-input", default=None)
    parser.add_argument(
        "--generator-adapter",
        default=None,
        help="Completed QLoRA adapter with qlora_scope train_scope='split'; auto-discovered if omitted",
    )
    parser.add_argument("--working", default="/kaggle/working/legalqa_selector_calibration")
    parser.add_argument(
        "--resume-bundle",
        default=None,
        help="Optional selector_recovery_latest.zip from an earlier identical calibration run",
    )
    parser.add_argument("--trials", type=int, default=3000)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    split_dir = Path(args.split_dir) if args.split_dir else _find_split(input_root)
    split_manifest_path = split_dir / "manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Selector split thiếu manifest.json: {split_dir}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_counts = _validate_selector_split(split_dir, split_manifest)
    grouped_dev_count = split_counts["dev"]
    dense_dir = (
        Path(args.dense_index_input)
        if args.dense_index_input
        else _find_dense(input_root)
    )
    generator_adapter = _find_split_adapter(input_root, args.generator_adapter)
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
        "--generator-adapter",
        str(generator_adapter),
        "--expected-adapter-scope",
        "split",
    ]
    if args.resume_bundle:
        command.extend(["--resume-bundle", args.resume_bundle])
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
        "tuning_accepted_on_holdout": report["tuning"]["accepted_on_holdout"],
        "base_holdout": report["tuning"]["base_holdout"],
        "raw_tuned_holdout": report["tuning"]["tuned_holdout"],
        "weights": str(weights_path.resolve()),
        "full_report": str(report_path.resolve()),
    }
    (working / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
