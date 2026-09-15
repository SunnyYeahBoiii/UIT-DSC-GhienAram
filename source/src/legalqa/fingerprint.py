from __future__ import annotations

import hashlib
from pathlib import Path


BLOCK_SIZE = 8 * 1024 * 1024

RUNTIME_SOURCE_FILES = (
    "SOURCE_CONTRACT.json",
    "pyproject.toml",
    "configs/kaggle_inference_t4.yaml",
    "kaggle/run_end_to_end.py",
    "kaggle/run_inference_only.py",
    "scripts/preflight_kaggle.py",
    "scripts/verify_v5007_adapter_ablation.py",
)
RUNTIME_SOURCE_DIRECTORIES = ("src", "wheelhouse")
SOURCE_TEXT_SUFFIXES = {".json", ".py", ".toml", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def corpus_content_fingerprint(path: str | Path) -> str:
    """Path-independent content hash for the official JSON corpus."""
    source = Path(path)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f"Corpus không có file JSON: {source}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_runtime_checkpoint_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in {"onnx", ".cache", "__pycache__"} for part in relative.parts):
        return False
    name = path.name.lower()
    if name.startswith("model") and path.suffix.lower() == ".safetensors":
        return True
    if name.startswith("pytorch_model") and path.suffix.lower() == ".bin":
        return True
    return path.suffix.lower() in {".json", ".model", ".txt", ".jinja", ".py"}


def checkpoint_fingerprint(path: str | Path) -> str:
    """Stable identity without re-reading every multi-GB weight tensor.

    Runtime configuration/tokenizer files are content-hashed. Large weight files
    contribute their relative name and exact byte size, which is stable across
    Kaggle mount paths and sufficient to prevent accidental model mix-ups.
    """
    root = Path(path)
    files = sorted(
        (item for item in root.rglob("*") if item.is_file() and _is_runtime_checkpoint_file(root, item)),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files or not (root / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint không đầy đủ: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        if size <= 32 * 1024 * 1024:
            digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def adapter_fingerprint(path: str | Path) -> str:
    """Content identity for a PEFT adapter without requiring base config.json."""
    root = Path(path)
    if not (root / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter không có adapter_config.json: {root}")
    files = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_file()
            and (
                item.name.startswith("adapter_")
                or item.name.startswith("tokenizer")
                or item.suffix.lower() in {".json", ".model", ".jinja"}
            )
        ),
        key=lambda item: item.name,
    )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def normalized_text_sha256(path: str | Path) -> str:
    """Hash text after universal-newline and optional UTF-8 BOM normalization.

    Windows CRLF and Unix LF represent the same YAML artifact. Other content,
    including every selector weight, remains part of the identity.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def runtime_source_fingerprint(path: str | Path) -> str:
    """Hash every source/dependency file that can affect public inference.

    The A/B notebook itself is intentionally excluded so it can embed the
    expected digest without creating a self-referential hash. Runtime code,
    configuration, preflight/guard scripts, and the exact offline wheels are
    all content-hashed.
    """
    root = Path(path)
    files: set[Path] = set()
    missing: list[str] = []
    for relative in RUNTIME_SOURCE_FILES:
        candidate = root / relative
        if candidate.is_file():
            files.add(candidate)
        else:
            missing.append(relative)
    for relative in RUNTIME_SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            missing.append(relative + "/")
            continue
        files.update(
            item
            for item in directory.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and ".ipynb_checkpoints" not in item.parts
            and item.suffix.lower() != ".pyc"
        )
    if missing:
        raise FileNotFoundError(f"Runtime source is incomplete under {root}: {missing}")

    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        if item.suffix.lower() in SOURCE_TEXT_SUFFIXES:
            content = item.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            content_size = len(content)
            content_sha = hashlib.sha256(content).hexdigest()
        else:
            content_size = item.stat().st_size
            content_sha = sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
