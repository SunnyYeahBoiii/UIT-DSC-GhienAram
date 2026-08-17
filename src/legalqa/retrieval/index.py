from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .chunking import chunk_documents
from ..core.data import Chunk, load_chunks, load_corpus, write_chunks
from ..core.models import DenseEncoder, unload_model


class LegalIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        dense: np.ndarray | None = None,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense

    @classmethod
    def load(cls, directory: str | Path) -> "LegalIndex":
        directory = Path(directory)
        chunks = load_chunks(directory / "chunks.jsonl")
        bm25 = BM25Index.load(directory / "bm25.json.gz")
        dense_path = directory / "dense.npy"
        dense = np.load(dense_path, mmap_mode="r") if dense_path.exists() else None
        if len(chunks) != bm25.document_count:
            raise ValueError("Số chunk không khớp BM25 index.")
        if dense is not None and dense.shape[0] != len(chunks):
            raise ValueError("Số vector không khớp số chunk.")
        return cls(chunks, bm25, dense)

    def dense_search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.dense is None or top_k <= 0:
            return []
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = np.asarray(self.dense @ vector, dtype=np.float32)
        limit = min(top_k, scores.shape[0])
        if limit <= 0:
            return []
        indices = np.argpartition(-scores, limit - 1)[:limit]
        ranked = indices[np.argsort(-scores[indices], kind="stable")]
        return [(int(index), float(scores[index])) for index in ranked]


def build_index(config: dict[str, Any], corpus_path: str | Path | None = None, sparse_only: bool = False) -> dict[str, Any]:
    artifacts = Path(config["paths"]["artifacts"])
    index_dir = artifacts / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    if corpus_path is None:
        corpus_path = config["paths"].get("corpus")
    if corpus_path is None:
        fallback = artifacts / "data" / "train_answer_corpus.jsonl"
        if not fallback.exists():
            raise FileNotFoundError(
                "Chưa có corpus. Hãy chạy prepare-warmup để smoke-test hoặc đặt paths.corpus "
                "tới kho văn bản chính thức của BTC."
            )
        corpus_path = fallback
    documents = load_corpus(corpus_path, config)
    chunks = chunk_documents(documents, config)
    if not chunks:
        raise ValueError("Không tạo được chunk nào từ corpus.")
    write_chunks(index_dir / "chunks.jsonl", chunks)
    bm25 = BM25Index().fit(chunk.text for chunk in chunks)
    bm25.save(index_dir / "bm25.json.gz")

    dense_shape = None
    embedding_path = config.get("paths", {}).get("models", {}).get("embedding")
    if not sparse_only and embedding_path and int(config["retrieval"].get("dense_top_k", 0)) > 0:
        encoder = DenseEncoder(config)
        vectors = encoder.encode(chunk.text for chunk in chunks)
        np.save(index_dir / "dense.npy", vectors, allow_pickle=False)
        dense_shape = list(vectors.shape)
        unload_model(encoder)

    manifest = {
        "corpus": str(Path(corpus_path).resolve()),
        "documents": len(documents),
        "chunks": len(chunks),
        "dense_shape": dense_shape,
        "sparse_only": dense_shape is None,
    }
    with (index_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest
