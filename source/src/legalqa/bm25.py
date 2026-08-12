from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .text import legal_tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[dict[str, int]] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.document_frequencies: dict[str, int] = {}
        self.lengths: list[int] = []
        self.average_length = 0.0
        self.document_count = 0

    def fit(self, texts: Iterable[str]) -> "BM25Index":
        self.term_frequencies = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for document_index, text in enumerate(texts):
            frequencies = Counter(legal_tokens(text))
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                postings[term].append((document_index, frequency))
        self.postings = dict(postings)
        self.document_frequencies = {term: len(values) for term, values in postings.items()}
        self.document_count = len(self.lengths)
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        return self

    def score(self, query: str) -> list[float]:
        query_frequencies = Counter(legal_tokens(query))
        total = self.document_count
        scores = [0.0] * total
        for term, qtf in query_frequencies.items():
            df = self.document_frequencies.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for index, tf in self.postings.get(term, []):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1e-9))
                scores[index] += qtf * idf * (tf * (self.k1 + 1)) / (tf + norm)
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return [(index, score) for index, score in ranked[:top_k] if score > 0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "k1": self.k1,
            "b": self.b,
            "postings": self.postings,
            "document_frequencies": self.document_frequencies,
            "lengths": self.lengths,
            "average_length": self.average_length,
        }
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") not in {1, 2}:
            raise ValueError("Phiên bản BM25 index không được hỗ trợ.")
        instance = cls(float(payload["k1"]), float(payload["b"]))
        if payload["version"] == 1:
            instance.term_frequencies = payload["term_frequencies"]
            postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for index, frequencies in enumerate(instance.term_frequencies):
                for term, frequency in frequencies.items():
                    postings[term].append((index, frequency))
            instance.postings = dict(postings)
        else:
            instance.postings = {
                term: [(int(index), int(frequency)) for index, frequency in values]
                for term, values in payload["postings"].items()
            }
        instance.document_frequencies = payload["document_frequencies"]
        instance.lengths = payload["lengths"]
        instance.document_count = len(instance.lengths)
        instance.average_length = float(payload["average_length"])
        return instance
