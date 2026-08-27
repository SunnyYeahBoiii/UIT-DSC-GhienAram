from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legalqa.answer_style import build_answer_style_profile  # noqa: E402
from legalqa.config import load_config  # noqa: E402
from legalqa.data import load_qa, read_json, write_json  # noqa: E402
from legalqa.fingerprint import corpus_content_fingerprint, sha256_file  # noqa: E402
from legalqa.sqlite_store import build_sqlite_store, prepare_sqlite_weak_pairs  # noqa: E402
from run_end_to_end import (  # noqa: E402
    find_corpus,
    find_model,
    find_unique_file,
    require_environment,
    run_command,
    write_resolved_config,
)


def _resolve_training_files(
    data_root: Path, prepared_root: Path
) -> tuple[Path, Path, dict[str, int]]:
    """Read only train_clean.json; derive QLoRA targets in writable working storage."""
    clean_path = data_root / "train_clean.json"
    if not clean_path.is_file():
        raise FileNotFoundError(
            f"Thiếu đúng file {clean_path}. Không dùng thay bằng train.json "
            "hoặc train_clean_qlora.json; hãy attach dataset chứa train_clean.json "
            "cạnh public-official.json."
        )
    quality_path = data_root / "training_quality.json"
    if not quality_path.is_file():
        raise FileNotFoundError(
            f"Thiếu {quality_path}; cần danh sách mẫu loại khỏi fine-tuning, "
            "không tự dùng tập dữ liệu khác."
        )
    raw = read_json(clean_path)
    if not isinstance(raw, dict):
        raise ValueError("train_clean.json phải là JSON Object theo question_id.")
    # Validate all questions/answers and duplicate IDs before preparing any output.
    load_qa(clean_path, require_answers=True)
    quality = read_json(quality_path)
    if not isinstance(quality, dict) or not isinstance(quality.get("qlora_excluded_ids"), list):
        raise ValueError("training_quality.json cần qlora_excluded_ids dạng danh sách.")
    declared = quality["qlora_excluded_ids"]
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in declared):
        raise ValueError("qlora_excluded_ids chỉ được chứa ID chuỗi hoặc số nguyên.")
    excluded_ids = {str(value) for value in declared}
    unknown_ids = sorted(excluded_ids - set(raw))
    if unknown_ids:
        raise RuntimeError(
            "training_quality.json không khớp train_clean.json; "
            f"ID loại không tồn tại: {unknown_ids[:10]}"
        )
    qlora_records = {key: value for key, value in raw.items() if key not in excluded_ids}
    if not qlora_records:
        raise RuntimeError("Tập QLoRA rỗng.")
    qlora_path = prepared_root / "qlora_train.json"
    if qlora_path.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("prepared_root phải nằm ngoài dataset đầu vào để không ghi đè dữ liệu.")
    write_json(qlora_path, qlora_records)
    return clean_path, qlora_path, {
        "full_records": len(raw),
        "qlora_records": len(qlora_records),
        "excluded_records": len(excluded_ids),
    }


def _latest_checkpoint_step(root: Path) -> int | None:
    steps = []
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required_state = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        has_adapter = (child / "adapter_model.safetensors").is_file() or (
            child / "adapter_model.bin"
        ).is_file()
        if has_adapter and all((child / name).is_file() for name in required_state):
            steps.append(step)
    return max(steps) if steps else None


def _find_resume_root(input_root: Path, train_scope: str) -> Path | None:
    candidates = []
    malformed = []
    for signature_path in input_root.rglob("resume_signature.json"):
        root = signature_path.parent
        scope_path = root / "qlora_scope.json"
        if _latest_checkpoint_step(root) is None:
            continue
        if not scope_path.is_file():
            malformed.append(f"{root}: thiếu qlora_scope.json")
            continue
        try:
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            malformed.append(f"{root}: qlora_scope.json lỗi: {error}")
            continue
        if scope.get("train_scope") == train_scope:
            candidates.append(root)
    if malformed:
        raise RuntimeError(
            "Tìm thấy QLoRA checkpoint có metadata scope hỏng; không được âm thầm train lại: "
            f"{malformed}"
        )
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise RuntimeError(f"Có nhiều QLoRA training state: {candidates}")
    return candidates[0] if candidates else None


def _recovery_bundle_scope(bundle: Path) -> str | None:
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            if "recovery_manifest.json" not in names or "qlora_scope.json" not in names:
                return None
            manifest = json.loads(archive.read("recovery_manifest.json"))
            scope = json.loads(archive.read("qlora_scope.json"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(manifest, dict) or not isinstance(scope, dict):
        return None
    if manifest.get("schema") != "legalqa-qlora-recovery-v1":
        return None
    return scope.get("train_scope")


def _find_resume_bundle(input_root: Path, train_scope: str) -> Path | None:
    candidates = [
        path
        for path in input_root.rglob("qlora_recovery_latest.zip")
        if _recovery_bundle_scope(path) == train_scope
    ]
    if not candidates:
        return None
    # Kaggle datasets can expose the same attached artifact through aliases.
    # Collapse byte-identical copies, but never guess between different states.
    by_digest: dict[str, list[Path]] = {}
    for candidate in candidates:
        digest = sha256_file(candidate)
        by_digest.setdefault(digest, []).append(candidate)
    if len(by_digest) > 1:
        raise RuntimeError(f"Có nhiều QLoRA recovery ZIP khác nhau: {candidates}")
    return sorted(candidates)[0]


def _extract_recovery_bundle(bundle: Path, working: Path) -> Path:
    if _recovery_bundle_scope(bundle) is None:
        raise RuntimeError(f"Recovery ZIP không đúng schema hoặc thiếu scope: {bundle}")
    digest = sha256_file(bundle)
    target = working / "_resume_bundles" / digest[:16]
    if (target / "resume_signature.json").is_file() and _latest_checkpoint_step(target) is not None:
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in info.filename
                or (relative.parts and ":" in relative.parts[0])
            ):
                raise RuntimeError(f"Recovery ZIP chứa path không an toàn: {info.filename}")
            destination = target.joinpath(*relative.parts)
            try:
                destination.resolve().relative_to(target.resolve())
            except ValueError as error:
                raise RuntimeError(f"Recovery ZIP chứa path không an toàn: {info.filename}") from error
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not (target / "resume_signature.json").is_file() or _latest_checkpoint_step(target) is None:
        raise RuntimeError(f"Recovery ZIP thiếu state có thể resume: {bundle}")
    print(f"[QLoRA resume] extracted atomic recovery: {bundle} -> {target}", flush=True)
    return target


def _copy_resume(source: Path, target: Path) -> None:
    signature_path = source / "resume_signature.json"
    if not signature_path.is_file():
        raise FileNotFoundError(f"QLoRA resume state thiếu resume_signature.json: {source}")
    try:
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"QLoRA resume_signature.json không đọc được: {source}: {error}") from error
    if not isinstance(signature, dict):
        raise RuntimeError(f"QLoRA resume_signature.json phải là JSON Object: {source}")
    source_step = _latest_checkpoint_step(source)
    if source_step is None:
        raise FileNotFoundError(f"QLoRA resume state không có checkpoint hợp lệ: {source}")
    target_step = _latest_checkpoint_step(target)
    if target_step is not None and target_step >= source_step:
        print(
            f"[QLoRA resume] giữ checkpoint local mới hơn/cùng bước: "
            f"target={target_step}, attached={source_step}",
            flush=True,
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    copied_step = _latest_checkpoint_step(target)
    if copied_step != source_step:
        raise RuntimeError(
            f"Sao chép QLoRA checkpoint không toàn vẹn: expected={source_step}, copied={copied_step}"
        )
    print(
        f"[QLoRA resume] copied persistent state step={source_step}: {source} -> {target}",
        flush=True,
    )


def _package_adapter(adapter_dir: Path, output: Path) -> None:
    include_prefixes = ("adapter_", "tokenizer", "special_tokens", "chat_template")
    include_names = {
        "training_complete.json",
        "resume_signature.json",
        "qlora_scope.json",
        "README.md",
        "added_tokens.json",
    }
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.iterdir()):
            if path.is_file() and (
                path.name.startswith(include_prefixes) or path.name in include_names
            ):
                archive.write(path, path.name)
    os.replace(temporary, output)


def _prepare_training_resume(
    input_root: Path,
    working: Path,
    adapter_output: Path,
    scope_metadata: dict,
    resume_input: Path | None,
    auto_resume: bool,
) -> None:
    """Validate data identity before copying or relabelling any training state."""
    def validate_scope(root: Path) -> None:
        scope_path = root / "qlora_scope.json"
        try:
            previous = json.loads(scope_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Không đọc được QLoRA scope: {scope_path}") from error
        if previous != scope_metadata:
            raise RuntimeError(
                "QLoRA resume state không khớp scope/hash dữ liệu hiện tại. "
                "Checkpoint tạo từ train cũ không được phép resume; hãy train mới "
                "trong working directory mới và tắt AUTO_RESUME_TRAINING. "
                f"expected={scope_metadata}; found={previous}; input={root}"
            )

    # Never overwrite metadata on an old local run, even without attached recovery.
    if adapter_output.exists() and any(adapter_output.iterdir()):
        if not auto_resume and resume_input is None:
            raise RuntimeError(
                f"Train mới cần working directory mới; không ghi đè state: {adapter_output}"
            )
        validate_scope(adapter_output)
    if resume_input is None and auto_resume:
        resume_input = _find_resume_root(input_root, scope_metadata["train_scope"])
        if resume_input is None:
            resume_input = _find_resume_bundle(input_root, scope_metadata["train_scope"])
    if resume_input is not None:
        if not resume_input.exists():
            raise FileNotFoundError(f"RESUME_INPUT không tồn tại: {resume_input}")
        if resume_input.is_file():
            resume_input = _extract_recovery_bundle(resume_input, working)
        validate_scope(resume_input)
        _copy_resume(resume_input, adapter_output)
    write_json(adapter_output / "qlora_scope.json", scope_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official-train evidence-grounded QLoRA for Qwen3.5-2B"
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--working", default="/kaggle/working/legalqa_qlora")
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument(
        "--train-scope",
        choices=("split", "full"),
        default="split",
        help="split for honest 400-question calibration; full for final public adapter",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-target-tokens", type=int, default=700)
    parser.add_argument("--max-evidence-chars", type=int, default=5000)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume-input", default=None, help="Thư mục training state hoặc qlora_recovery_latest.zip")
    parser.add_argument(
        "--no-auto-resume", action="store_true",
        help="Không tự tìm checkpoint; train mới cần working directory mới. RESUME_INPUT rõ ràng vẫn được ưu tiên.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=11.0,
        help="Ngân sách tổng của runner; mặc định dừng an toàn sau 11h để Kaggle kịp lưu Output (0 = tắt).",
    )
    parser.add_argument(
        "--recovery-output",
        default=None,
        help="Đường dẫn ZIP recovery nguyên tử (mặc định cạnh thư mục --working).",
    )
    parser.add_argument("--force-index", action="store_true")
    args = parser.parse_args()
    if not 300 <= args.dev_size <= 500:
        raise ValueError("--dev-size phải nằm trong [300, 500] cho selector experiment.")
    if args.max_runtime_hours < 0 or args.max_runtime_hours >= 12:
        raise ValueError("--max-runtime-hours phải bằng 0 hoặc nằm trong (0, 12).")

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

    public_path = find_unique_file(input_root, "public-official.json")
    data_root = public_path.parent
    train_path, qlora_train_path, training_data_counts = _resolve_training_files(
        data_root, working / "prepared_train"
    )
    print(
        "[QLoRA data] "
        f"input={train_path} ({training_data_counts['full_records']:,}); "
        f"prepared_qlora={qlora_train_path} ({training_data_counts['qlora_records']:,}); "
        f"excluded={training_data_counts['excluded_records']:,}",
        flush=True,
    )
    corpus_path = (
        data_root / "selected-contexts"
        if (data_root / "selected-contexts").is_dir()
        else find_corpus(data_root)
    )
    generator_path = find_model(input_root, "generator")
    if generator_path is None:
        raise FileNotFoundError("Thiếu approved Qwen3.5-2B checkpoint.")

    preflight_path = working / "preflight.json"
    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "preflight_kaggle.py"),
            "--input-root",
            str(input_root),
            "--profile",
            "fast",
            "--output",
            str(preflight_path),
        ]
    )

    split_dir = working / "selector_split"
    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "prepare_selector_dev.py"),
            "--train",
            str(qlora_train_path),
            "--output",
            str(split_dir),
            "--dev-size",
            str(args.dev_size),
            "--min-dev-size",
            "300",
            "--max-dev-size",
            "500",
        ]
    )
    train_split = split_dir / "selector_train.json"
    dev_split = split_dir / "selector_dev.json"
    split_manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    actual_dev_size = int(split_manifest["dev"])
    style_profile = build_answer_style_profile(load_qa(train_path, require_answers=True))
    write_json(working / "answer_style_profile.json", style_profile)

    config = load_config(args.config)
    config["paths"]["warmup"] = str(train_path)
    config["paths"]["corpus"] = str(corpus_path)
    config["paths"]["artifacts"] = str(working / "artifacts")
    config["paths"]["models"]["generator"] = str(generator_path)
    config["paths"]["models"]["generator_adapter"] = None
    config["generation"]["answer_style_profile"] = style_profile
    resolved_config = working / "resolved_config.yaml"
    write_resolved_config(config, resolved_config)

    database = working / "official_corpus.sqlite"
    index_report = build_sqlite_store(
        config, corpus_path, database, force=args.force_index
    )
    pairs_dir = working / "pairs"
    training_file = train_split if args.train_scope == "split" else qlora_train_path
    train_pairs = pairs_dir / (
        "selector_train_pairs.jsonl" if args.train_scope == "split" else "official_full_train_pairs.jsonl"
    )
    dev_pairs = pairs_dir / "selector_dev_pairs.jsonl"
    train_pair_report = prepare_sqlite_weak_pairs(
        training_file, database, train_pairs, top_k=40, negatives=3
    )
    dev_pair_report = (
        prepare_sqlite_weak_pairs(dev_split, database, dev_pairs, top_k=40, negatives=1)
        if args.train_scope == "split"
        else None
    )

    adapter_output = working / (
        "qwen-official-lora-split"
        if args.train_scope == "split"
        else "qwen-official-lora-full"
    )
    scope_metadata = {
        "schema": "legalqa-qlora-scope-v2",
        "train_scope": args.train_scope,
        "training_data_sha256": sha256_file(training_file),
        "qlora_source_sha256": sha256_file(qlora_train_path),
        "full_clean_sha256": sha256_file(train_path),
    }
    _prepare_training_resume(
        input_root, working, adapter_output, scope_metadata,
        Path(args.resume_input) if args.resume_input else None,
        auto_resume=not args.no_auto_resume,
    )
    recovery_output = (
        Path(args.recovery_output)
        if args.recovery_output
        else working.parent / "qlora_recovery_latest.zip"
    )
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "train_qwen_lora.py"),
        "--config",
        str(resolved_config),
        "--train-file",
        str(training_file),
        "--pairs",
        str(train_pairs),
        "--output",
        str(adapter_output),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        "1",
        "--grad-accum",
        str(args.grad_accum),
        "--max-length",
        str(args.max_length),
        "--max-target-tokens",
        str(args.max_target_tokens),
        "--max-evidence-chars",
        str(args.max_evidence_chars),
        "--lora-r",
        str(args.lora_r),
        "--load-in-4bit",
        "--checkpoint-steps",
        str(args.checkpoint_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--resume-from-checkpoint",
        "auto" if not args.no_auto_resume or args.resume_input else "none",
        "--recovery-output",
        str(recovery_output),
    ]
    if args.max_runtime_hours > 0:
        elapsed_hours = (time.time() - started) / 3600
        remaining_hours = args.max_runtime_hours - elapsed_hours
        if remaining_hours <= 0.25:
            raise RuntimeError(
                f"Chỉ còn {remaining_hours:.2f}h trong ngân sách; không đủ an toàn để bắt đầu một optimizer step."
            )
        command.extend(["--max-runtime-hours", str(remaining_hours)])
    if args.train_scope == "split":
        command.extend(
            ["--dev-file", str(dev_split), "--dev-pairs", str(dev_pairs)]
        )
    if args.max_train_samples:
        command.extend(["--max-train-samples", str(args.max_train_samples)])
    run_command(command)

    manifest = {
        "mode": "official_train_evidence_grounded_qlora",
        "inputs": {
            "train": str(training_file),
            "train_sha256": sha256_file(training_file),
            "full_clean_train": str(train_path),
            "full_clean_train_sha256": sha256_file(train_path),
            "qlora_source_train": str(qlora_train_path),
            "qlora_source_train_sha256": sha256_file(qlora_train_path),
            "training_data_counts": training_data_counts,
            "corpus": str(corpus_path),
            "corpus_sha256": corpus_content_fingerprint(corpus_path),
        },
        "environment": environment,
        "dev_size_requested": args.dev_size,
        "dev_size": actual_dev_size,
        "train_scope": args.train_scope,
        "index": index_report,
        "train_pairs": train_pair_report,
        "dev_pairs": dev_pair_report,
        "adapter": str(adapter_output.resolve()),
        "selector_split": str(split_dir.resolve()),
        "elapsed_hours": (time.time() - started) / 3600,
    }
    paused_path = adapter_output / "training_paused.json"
    complete_path = adapter_output / "training_complete.json"
    if paused_path.is_file() and not complete_path.is_file():
        pause_state = json.loads(paused_path.read_text(encoding="utf-8"))
        if not recovery_output.is_file():
            raise RuntimeError(f"Training đã pause nhưng thiếu recovery ZIP: {recovery_output}")
        manifest.update(
            {
                "status": "paused",
                "pause_state": pause_state,
                "recovery_zip": str(recovery_output.resolve()),
                "latest_checkpoint_step": _latest_checkpoint_step(adapter_output),
            }
        )
        write_json(working / "qlora_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        print(
            "[QLoRA] PAUSED SAFELY. Hãy Save Version, đưa qlora_recovery_latest.zip thành Dataset, "
            "rồi attach Dataset đó ở session kế tiếp để auto-resume.",
            flush=True,
        )
        return
    if not complete_path.is_file():
        raise RuntimeError("Trainer kết thúc nhưng không có training_complete.json hoặc training_paused.json.")

    adapter_zip = working / "qwen_qlora_adapter.zip"
    _package_adapter(adapter_output, adapter_zip)
    manifest.update(
        {
            "status": "complete",
            "adapter_zip": str(adapter_zip.resolve()),
        }
    )
    write_json(working / "qlora_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
