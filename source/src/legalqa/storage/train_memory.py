from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..core.data import load_qa
from ..core.fingerprint import sha256_file
from ..core.models import DenseEncoder


TRAIN_MEMORY_SCHEMA = "train-memory-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _identity(config: dict[str, Any], train_path: Path) -> str:
    payload = {
        "schema": TRAIN_MEMORY_SCHEMA,
        "train": config.get("runtime", {}).get("train_identity") or sha256_file(train_path),
        "embedding": config.get("runtime", {}).get("embedding_identity")
        or config.get("paths", {}).get("models", {}).get("embedding"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_records(path: Path, records) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "question_id": record.question_id,
                        "question": record.question,
                        "answer": record.answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_train_memory_index(
    config: dict[str, Any],
    encoder: DenseEncoder,
    train_path: str | Path,
    index_dir: str | Path,
) -> dict[str, Any]:
    """Build a restartable dense index over official training questions."""
    source = Path(train_path)
    records = load_qa(source, require_answers=True)
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    state_path = target / "state.json"
    vectors_path = target / "vectors.f16"
    records_path = target / "records.jsonl"
    identity = _identity(config, source)
    total = len(records)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = int(manifest.get("count", 0)) * int(manifest.get("dimension", 0)) * 2
        is_read_only = not os.access(target, os.W_OK) or "/kaggle/input" in str(target.resolve())
        if (
            manifest.get("status") == "complete"
            and manifest.get("schema") == TRAIN_MEMORY_SCHEMA
            and (manifest.get("identity") == identity or is_read_only)
            and int(manifest.get("count", 0)) == total
            and vectors_path.is_file()
            and vectors_path.stat().st_size == expected
            and records_path.is_file()
        ):
            return {**manifest, "directory": str(target.resolve()), "reused": True}
        raise RuntimeError(
            f"Train-memory index không khớp official train/model: {target}. "
            "Dùng thư mục mới hoặc --force-train-memory."
        )

    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity or int(state.get("count", -1)) != total:
            raise RuntimeError(f"Train-memory checkpoint thuộc train/model khác: {target}")
    else:
        _write_records(records_path, records)

    processed = int(state.get("processed", 0))
    dimension = int(state.get("dimension", 0))
    if processed and (
        dimension <= 0
        or not vectors_path.is_file()
        or vectors_path.stat().st_size != total * dimension * 2
        or not records_path.is_file()
    ):
        raise RuntimeError(f"Train-memory checkpoint thiếu/sai vector hoặc records: {target}")

    block_size = max(1, int(config["retrieval"].get("train_memory_build_block", 128)))
    checkpoint_every = max(
        block_size, int(config["retrieval"].get("train_memory_checkpoint_every", 512))
    )
    next_checkpoint = ((processed // checkpoint_every) + 1) * checkpoint_every
    vectors = None
    if processed:
        vectors = np.memmap(vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension))
    while processed < total:
        batch = records[processed : processed + block_size]
        encoded = encoder.encode([record.question for record in batch])
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise RuntimeError(f"Train-memory embedding shape bất thường: {encoded.shape}")
        if dimension == 0:
            dimension = int(encoded.shape[1])
            with vectors_path.open("wb") as handle:
                handle.truncate(total * dimension * 2)
            vectors = np.memmap(
                vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension)
            )
        elif int(encoded.shape[1]) != dimension:
            raise RuntimeError(
                f"Train-memory dimension thay đổi: {encoded.shape[1]} != {dimension}"
            )
        if vectors is None:
            vectors = np.memmap(
                vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension)
            )
        end = processed + len(batch)
        vectors[processed:end] = encoded.astype(np.float16, copy=False)
        processed = end
        if processed >= next_checkpoint or processed == total:
            vectors.flush()
            _atomic_json(
                state_path,
                {
                    "schema": TRAIN_MEMORY_SCHEMA,
                    "identity": identity,
                    "count": total,
                    "dimension": dimension,
                    "processed": processed,
                },
            )
            print(f"[Train memory] {processed:,}/{total:,} questions", flush=True)
            next_checkpoint += checkpoint_every

    if vectors is not None:
        vectors.flush()
    manifest = {
        "status": "complete",
        "schema": TRAIN_MEMORY_SCHEMA,
        "identity": identity,
        "count": total,
        "dimension": dimension,
        "dtype": "float16",
        "vectors": vectors_path.name,
        "records": records_path.name,
    }
    _atomic_json(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    return {**manifest, "directory": str(target.resolve()), "reused": False}


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def search_train_memory(
    query_vectors: np.ndarray,
    index_dir: str | Path,
    top_k: int,
    *,
    device: str,
    exclude_question_ids: set[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Exact cosine kNN over official train questions."""
    target = Path(index_dir)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["count"])
    dimension = int(manifest["dimension"])
    if query_vectors.ndim != 2 or int(query_vectors.shape[1]) != dimension:
        raise ValueError(
            f"Query vectors shape={query_vectors.shape}, train-memory dimension={dimension}"
        )
    records = _load_records(target / manifest["records"])
    if len(records) != total:
        raise RuntimeError(f"Train-memory records={len(records)} khác manifest={total}")
    record_ids = {str(record["question_id"]) for record in records}
    excluded = (exclude_question_ids or set()) & record_ids
    requested = min(total, max(1, int(top_k)) + len(excluded))
    vectors = np.memmap(
        target / manifest["vectors"], mode="r", dtype=np.float16, shape=(total, dimension)
    )
    if device.startswith("cuda"):
        import torch

        with torch.inference_mode():
            queries = torch.as_tensor(query_vectors, device=device, dtype=torch.float16)
            passages = torch.as_tensor(np.asarray(vectors), device=device, dtype=torch.float16)
            scores, positions = torch.topk(queries @ passages.T, requested, dim=1)
        score_rows = scores.float().cpu().numpy()
        position_rows = positions.cpu().numpy()
    else:
        scores = np.asarray(query_vectors, dtype=np.float32) @ np.asarray(
            vectors, dtype=np.float32
        ).T
        position_rows = np.argpartition(scores, -requested, axis=1)[:, -requested:]
        score_rows = np.take_along_axis(scores, position_rows, axis=1)
        order = np.argsort(-score_rows, axis=1)
        position_rows = np.take_along_axis(position_rows, order, axis=1)
        score_rows = np.take_along_axis(score_rows, order, axis=1)
    result: list[list[dict[str, Any]]] = []
    for positions, scores in zip(position_rows, score_rows):
        ranked = []
        for position, score in zip(positions, scores):
            record = records[int(position)]
            if str(record["question_id"]) in excluded:
                continue
            ranked.append(
                {
                    "question_id": str(record["question_id"]),
                    "question": str(record["question"]),
                    "answer": str(record["answer"]),
                    "similarity": float(score),
                }
            )
            if len(ranked) >= top_k:
                break
        result.append(ranked)
    return result
