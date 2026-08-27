from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .models import DenseEncoder
from .sqlite_store import SQLiteLegalStore


DENSE_INDEX_SCHEMA = "global-dense-v2-structured-headings"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def _identity(config: dict[str, Any], store: SQLiteLegalStore) -> str:
    manifest = store.manifest()
    payload = {
        "schema": DENSE_INDEX_SCHEMA,
        "corpus": config.get("runtime", {}).get("corpus_identity")
        or manifest.get("corpus_fingerprint"),
        "chunks": store.chunk_count(),
        "embedding": config.get("runtime", {}).get("embedding_identity")
        or config.get("paths", {}).get("models", {}).get("embedding"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def ensure_global_dense_index(
    config: dict[str, Any],
    store: SQLiteLegalStore,
    encoder: DenseEncoder,
    index_dir: str | Path,
) -> dict[str, Any]:
    """Build a restartable float16 embedding matrix for every official chunk."""
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    state_path = target / "state.json"
    vectors_path = target / "vectors.f16"
    rowids_path = target / "rowids.i64"
    identity = _identity(config, store)
    total = store.chunk_count()

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_bytes = int(manifest.get("chunks", 0)) * int(manifest.get("dimension", 0)) * 2
        if (
            manifest.get("status") == "complete"
            and manifest.get("identity") == identity
            and int(manifest.get("chunks", 0)) == total
            and vectors_path.is_file()
            and vectors_path.stat().st_size == expected_bytes
            and rowids_path.is_file()
            and rowids_path.stat().st_size == total * 8
        ):
            return {**manifest, "directory": str(target.resolve()), "reused": True}
        raise RuntimeError(
            f"Global dense index hoàn chỉnh nhưng không khớp corpus/model: {target}. "
            "Chọn thư mục output mới hoặc dùng --force-dense-index để dựng lại."
        )

    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity or int(state.get("chunks", -1)) != total:
            raise RuntimeError(
                f"Global dense index dở dang thuộc corpus/model khác: {target}."
            )

    processed = int(state.get("processed", 0))
    dimension = int(state.get("dimension", 0))
    last_rowid = int(state.get("last_rowid", 0))
    if processed:
        expected_vector_bytes = total * dimension * 2
        if (
            dimension <= 0
            or not vectors_path.is_file()
            or vectors_path.stat().st_size != expected_vector_bytes
            or not rowids_path.is_file()
            or rowids_path.stat().st_size != total * 8
        ):
            raise RuntimeError(
                f"Dense checkpoint state tồn tại nhưng vector files thiếu/sai kích thước: {target}"
            )
    build_block = max(1, int(config["retrieval"].get("dense_build_block", 256)))
    checkpoint_every = max(
        build_block, int(config["retrieval"].get("dense_checkpoint_every", 4096))
    )
    next_checkpoint = ((processed // checkpoint_every) + 1) * checkpoint_every
    vectors = rowids = None
    if processed:
        vectors = np.memmap(vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension))
        rowids = np.memmap(rowids_path, mode="r+", dtype=np.int64, shape=(total,))

    for hits in store.iter_chunks(start_rowid=last_rowid, batch_size=build_block):
        encoded = encoder.encode([hit.chunk.text for hit in hits])
        if encoded.ndim != 2 or encoded.shape[0] != len(hits):
            raise RuntimeError(f"Embedding batch trả shape bất thường: {encoded.shape}")
        if dimension == 0:
            dimension = int(encoded.shape[1])
            with vectors_path.open("wb") as handle:
                handle.truncate(total * dimension * 2)
            with rowids_path.open("wb") as handle:
                handle.truncate(total * 8)
            vectors = np.memmap(vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension))
            rowids = np.memmap(rowids_path, mode="r+", dtype=np.int64, shape=(total,))
        elif int(encoded.shape[1]) != dimension:
            raise RuntimeError(
                f"Embedding dimension thay đổi giữa index: {encoded.shape[1]} != {dimension}"
            )
        if vectors is None or rowids is None:
            vectors = np.memmap(vectors_path, mode="r+", dtype=np.float16, shape=(total, dimension))
            rowids = np.memmap(rowids_path, mode="r+", dtype=np.int64, shape=(total,))
        end = processed + len(hits)
        if end > total:
            raise RuntimeError(f"Dense index ghi vượt số chunk: {end}>{total}")
        vectors[processed:end] = encoded.astype(np.float16, copy=False)
        rowids[processed:end] = [hit.rowid for hit in hits]
        processed = end
        last_rowid = hits[-1].rowid
        if processed >= next_checkpoint or processed == total:
            vectors.flush()
            rowids.flush()
            _atomic_json(
                state_path,
                {
                    "schema": DENSE_INDEX_SCHEMA,
                    "identity": identity,
                    "chunks": total,
                    "dimension": dimension,
                    "processed": processed,
                    "last_rowid": last_rowid,
                },
            )
            print(f"[Global dense index] {processed:,}/{total:,} chunks", flush=True)
            next_checkpoint += checkpoint_every

    if processed != total or dimension <= 0:
        raise RuntimeError(
            f"Global dense index chưa hoàn tất: processed={processed}, total={total}, dim={dimension}"
        )
    if vectors is not None:
        vectors.flush()
    if rowids is not None:
        rowids.flush()
    manifest = {
        "status": "complete",
        "schema": DENSE_INDEX_SCHEMA,
        "identity": identity,
        "chunks": total,
        "dimension": dimension,
        "dtype": "float16",
        "vectors": vectors_path.name,
        "rowids": rowids_path.name,
    }
    _atomic_json(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    return {**manifest, "directory": str(target.resolve()), "reused": False}


def search_global_dense_index(
    query_vectors: np.ndarray,
    index_dir: str | Path,
    top_k: int,
    *,
    device: str,
    block_size: int = 8192,
) -> list[list[tuple[int, float]]]:
    """Exact global cosine search streamed from a float16 memmap."""
    target = Path(index_dir)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    total, dimension = int(manifest["chunks"]), int(manifest["dimension"])
    if query_vectors.ndim != 2 or query_vectors.shape[1] != dimension:
        raise ValueError(
            f"Query vectors shape={query_vectors.shape}, dense dimension={dimension}"
        )
    k = min(max(1, int(top_k)), total)
    vectors = np.memmap(target / manifest["vectors"], mode="r", dtype=np.float16, shape=(total, dimension))
    rowids = np.memmap(target / manifest["rowids"], mode="r", dtype=np.int64, shape=(total,))
    block_size = max(k, int(block_size))

    if device.startswith("cuda"):
        import torch

        queries = torch.as_tensor(query_vectors, device=device, dtype=torch.float16)
        best_scores = torch.full((len(queries), k), -float("inf"), device=device, dtype=torch.float16)
        best_rowids = torch.full((len(queries), k), -1, device=device, dtype=torch.int64)
        with torch.inference_mode():
            for start in range(0, total, block_size):
                end = min(total, start + block_size)
                passages = torch.as_tensor(np.asarray(vectors[start:end]), device=device, dtype=torch.float16)
                scores = queries @ passages.T
                local_k = min(k, end - start)
                local_scores, local_positions = torch.topk(scores, local_k, dim=1)
                block_rowids = torch.as_tensor(
                    np.asarray(rowids[start:end]), device=device, dtype=torch.int64
                )
                local_rowids = block_rowids[local_positions]
                combined_scores = torch.cat((best_scores, local_scores), dim=1)
                combined_rowids = torch.cat((best_rowids, local_rowids), dim=1)
                best_scores, positions = torch.topk(combined_scores, k, dim=1)
                best_rowids = torch.gather(combined_rowids, 1, positions)
                if end % (block_size * 8) == 0 or end == total:
                    print(f"[Global dense search] {end:,}/{total:,} chunks", flush=True)
        score_array = best_scores.float().cpu().numpy()
        rowid_array = best_rowids.cpu().numpy()
    else:
        queries = np.asarray(query_vectors, dtype=np.float32)
        best_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
        best_rowids = np.full((len(queries), k), -1, dtype=np.int64)
        for start in range(0, total, block_size):
            end = min(total, start + block_size)
            scores = queries @ np.asarray(vectors[start:end], dtype=np.float32).T
            local = np.argpartition(scores, -min(k, scores.shape[1]), axis=1)[:, -min(k, scores.shape[1]):]
            local_scores = np.take_along_axis(scores, local, axis=1)
            local_rowids = np.asarray(rowids[start:end])[local]
            joined_scores = np.concatenate((best_scores, local_scores), axis=1)
            joined_rowids = np.concatenate((best_rowids, local_rowids), axis=1)
            keep = np.argpartition(joined_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(joined_scores, keep, axis=1)
            best_rowids = np.take_along_axis(joined_rowids, keep, axis=1)
        order = np.argsort(-best_scores, axis=1)
        score_array = np.take_along_axis(best_scores, order, axis=1)
        rowid_array = np.take_along_axis(best_rowids, order, axis=1)

    return [
        [(int(rowid), float(score)) for rowid, score in zip(rowid_row, score_row) if rowid >= 0]
        for rowid_row, score_row in zip(rowid_array, score_array)
    ]
