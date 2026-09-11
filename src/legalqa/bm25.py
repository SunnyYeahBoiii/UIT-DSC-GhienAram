"""BM25 Okapi trên Chunk — chỉ số tìm kiếm sparse.

Tham số k1=1.5, b=0.75 theo Okapi, min_df=2 lọc token hiếm (noise).
Không thêm citation boost — EDA đo chỉ 0,1% question chứa citation.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .chunking import Chunk

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """lowercase + tách từ bằng regex — giống CountVectorizer token_pattern."""
    return re.findall(r"\w+", text.lower())


class ChunkIndex:
    """BM25 index trên danh sách Chunk. Build trước khi search."""

    def __init__(self) -> None:
        self._vectorizer: CountVectorizer | None = None
        self._tfidf_matrix: sparse.csr_matrix | None = None
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._dl: np.ndarray | None = None
        self._avgdl: float = 0.0
        self._idf: np.ndarray | None = None
        self._N: int = 0

    def build(self, chunks: list[Chunk]) -> None:
        """Xây BM25 index từ danh sách Chunk.

        CountVectorizer tokenizer giống EDA: lowercase + \\w+ pattern.
        min_df=2 lọc token xuất hiện ở <2 chunk — tránh IDF cực lớn.
        """
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._doc_ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]

        self._vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\w+",
            dtype=np.int32,
            min_df=2,
        )
        X = self._vectorizer.fit_transform(texts)
        self._N = X.shape[0]

        # IDF Okapi: log(1 + (N - df + 0.5) / (df + 0.5))
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self._idf = np.log(1 + (self._N - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Document lengths và average
        self._dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        self._avgdl = float(self._dl.mean())

        # BM25 TF weighting: tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        Xc = X.tocsr().astype(np.float32)
        rows = np.repeat(np.arange(self._N), np.diff(Xc.indptr))
        dl_ratios = self._dl[rows] / self._avgdl
        Xc.data = (Xc.data * (_K1 + 1) / (Xc.data + _K1 * (1 - _B + _B * dl_ratios))).astype(
            np.float32
        )
        # W = TF_bm25 * IDF
        self._tfidf_matrix = Xc.multiply(self._idf).tocsr()

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Tìm top_k Chunk phù hợp nhất với query. Trả về (chunk_id, score)."""
        if self._vectorizer is None or self._tfidf_matrix is None:
            raise RuntimeError("Index chưa build — gọi build() trước")

        q_vec = self._vectorizer.transform([query])
        # BM25 score = query_vec * W^T
        scores = (q_vec @ self._tfidf_matrix.T).toarray().ravel()
        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        """Lưu index xuống disk bằng pickle."""
        data = {
            "vectorizer": self._vectorizer,
            "tfidf_matrix": self._tfidf_matrix,
            "chunk_ids": self._chunk_ids,
            "doc_ids": self._doc_ids,
            "dl": self._dl,
            "avgdl": self._avgdl,
            "idf": self._idf,
            "N": self._N,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> ChunkIndex:
        """Tải index từ disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        idx = cls()
        idx._vectorizer = data["vectorizer"]
        idx._tfidf_matrix = data["tfidf_matrix"]
        idx._chunk_ids = data["chunk_ids"]
        idx._doc_ids = data["doc_ids"]
        idx._dl = data["dl"]
        idx._avgdl = data["avgdl"]
        idx._idf = data["idf"]
        idx._N = data["N"]
        return idx
