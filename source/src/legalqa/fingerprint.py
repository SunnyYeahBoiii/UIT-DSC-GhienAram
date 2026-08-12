from __future__ import annotations

import hashlib
from pathlib import Path


BLOCK_SIZE = 8 * 1024 * 1024


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
