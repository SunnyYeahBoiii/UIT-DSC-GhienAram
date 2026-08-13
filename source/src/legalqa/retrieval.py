from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import Chunk
from .index import LegalIndex
from .models import CrossEncoderReranker, DenseEncoder, sigmoid
from .text import citations, document_numbers, legal_tokens, numbers


@dataclass
class Evidence:
    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    reranker_score: float = 0.0
    stage: str = "initial"
    relation_type: str = ""
    relation_path: list[str] = field(default_factory=list)


def rank_agentic_evidence(
    question: str,
    query: str,
    current_evidence: list[Evidence],
    candidates: list[Evidence],
    config: dict[str, Any],
    *,
    obsolete_documents: set[str] | None = None,
) -> list[Evidence]:
    """Rerank tool-call hits with legal, temporal and relation-aware signals.

    This stage intentionally stays CPU-only: the GPU reranker has already been
    unloaded before Qwen starts.  Raw BM25 is normalized per tool call, then
    combined with query overlap, exact document/article matches and graph
    relation metadata.  Weak tail hits and obsolete documents are gated out.
    """
    if not candidates:
        return []
    cfg = config.get("retrieval", {})
    combined_query = f"{question}\n{query}"
    query_tokens = set(legal_tokens(combined_query))
    query_documents = document_numbers(combined_query)
    query_citations = citations(combined_query)
    seed_documents = {item.chunk.doc_id for item in current_evidence}
    obsolete = obsolete_documents or set()
    maximum_bm25 = max((max(0.0, item.bm25_score) for item in candidates), default=0.0)

    rescored: list[Evidence] = []
    for item in candidates:
        searchable = f"{item.chunk.title}\n{item.chunk.heading}\n{item.chunk.text}"
        item_tokens = set(legal_tokens(searchable))
        overlap = len(query_tokens & item_tokens) / max(1, len(query_tokens))
        exact_document = len(query_documents & document_numbers(searchable))
        exact_article = len(query_citations & citations(searchable))
        bm25_signal = max(0.0, item.bm25_score) / maximum_bm25 if maximum_bm25 else 0.0
        relation_signal = float(bool(item.relation_type))
        referenced_by_seed = float(
            bool(item.relation_path)
            and any(document in seed_documents for document in item.relation_path)
        )
        score = (
            float(cfg.get("agentic_bm25_weight", 0.46)) * bm25_signal
            + float(cfg.get("agentic_overlap_weight", 0.34)) * overlap
            + float(cfg.get("agentic_exact_document_boost", 0.45)) * exact_document
            + float(cfg.get("agentic_exact_article_boost", 0.30)) * exact_article
            + float(cfg.get("agentic_relation_boost", 0.18)) * relation_signal
            + float(cfg.get("agentic_reference_document_boost", 0.16)) * referenced_by_seed
        )
        if item.chunk.doc_id in obsolete:
            score -= float(cfg.get("agentic_obsolete_penalty", 0.45))
        rescored.append(
            Evidence(
                chunk=item.chunk,
                score=score,
                bm25_score=item.bm25_score,
                dense_score=item.dense_score,
                reranker_score=item.reranker_score,
                stage=item.stage,
                relation_type=item.relation_type,
                relation_path=list(item.relation_path),
            )
        )

    best_by_chunk: dict[str, Evidence] = {}
    for item in rescored:
        previous = best_by_chunk.get(item.chunk.chunk_id)
        if previous is None or item.score > previous.score:
            best_by_chunk[item.chunk.chunk_id] = item
    ordered = sorted(best_by_chunk.values(), key=lambda item: (-item.score, item.chunk.chunk_id))
    top_score = max((item.score for item in ordered), default=0.0)
    absolute_floor = float(cfg.get("agentic_min_relevance", 0.16))
    relative_floor = top_score * float(cfg.get("agentic_min_score_ratio", 0.35))
    accepted = [item for item in ordered if item.score >= max(absolute_floor, relative_floor)]
    return accepted[: max(1, int(cfg.get("agentic_return_top_k", 12)))]


def _rrf_add(scores: dict[int, float], ranking: list[tuple[int, float]], weight: float, k: int) -> None:
    for rank, (index, _) in enumerate(ranking, 1):
        scores[index] = scores.get(index, 0.0) + weight / (k + rank)


def retrieval_confidence(evidence: list[Evidence]) -> tuple[float, dict[str, float]]:
    """Estimate whether the first-pass retrieval is safe enough for a simple route."""
    if not evidence:
        return 0.0, {
            "top_score": 0.0, "top_reranker": 0.0, "margin": 0.0,
            "method_agreement": 0.0, "document_concentration": 0.0,
        }
    ordered = sorted(evidence, key=lambda item: (-item.score, item.chunk.chunk_id))
    top = ordered[0]
    margin = max(0.0, top.score - ordered[1].score) if len(ordered) > 1 else max(0.0, top.score)
    agreement = float(top.bm25_score > 0 and top.dense_score > 0)
    top_window = ordered[: min(5, len(ordered))]
    concentration = sum(item.chunk.doc_id == top.chunk.doc_id for item in top_window) / len(top_window)
    reranker = max(0.0, min(1.0, top.reranker_score))
    margin_score = min(1.0, margin / 0.25)
    score_signal = min(1.0, max(0.0, top.score) / 0.75)
    confidence = (
        0.48 * reranker + 0.20 * margin_score + 0.13 * agreement
        + 0.12 * concentration + 0.07 * score_signal
    )
    details = {
        "top_score": float(top.score),
        "top_reranker": reranker,
        "margin": float(margin),
        "method_agreement": agreement,
        "document_concentration": concentration,
    }
    return max(0.0, min(1.0, confidence)), details


def _chunk_position(item: Evidence) -> tuple[str, int, int] | None:
    import re

    match = re.match(r"^(.*?)::s(\d+)::w(\d+)", item.chunk.chunk_id)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _merge_text(left: str, right: str) -> str:
    """Merge overlapping legal chunks without repeating the shared tail."""
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right or right in left:
        return left
    max_overlap = min(600, len(left), len(right))
    for size in range(max_overlap, 39, -1):
        if left[-size:].casefold() == right[:size].casefold():
            return f"{left}{right[size:]}".strip()
    return f"{left}\n{right}".strip()


def deduplicate_and_merge_evidence(
    evidence: list[Evidence],
    limit: int,
    *,
    prefer_document_diversity: bool,
    max_documents: int | None = None,
    diversity_bonus: float = 0.05,
    minimum_score_ratio: float = 0.0,
) -> list[Evidence]:
    """Merge adjacent windows and remove near-duplicate evidence slots."""
    if limit <= 0:
        return []
    ordered = sorted(evidence, key=lambda item: (-item.score, item.chunk.chunk_id))
    merged: list[Evidence] = []
    consumed: set[str] = set()
    for item in ordered:
        if item.chunk.chunk_id in consumed:
            continue
        position = _chunk_position(item)
        group = [item]
        if position:
            for other in ordered:
                other_position = _chunk_position(other)
                if (
                    other.chunk.chunk_id not in consumed
                    and other.chunk.chunk_id != item.chunk.chunk_id
                    and other_position
                    and other_position[0] == position[0]
                    and other_position[1] == position[1]
                    and abs(other_position[2] - position[2]) <= 1
                ):
                    group.append(other)
        group.sort(key=lambda member: (_chunk_position(member) or ("", 0, 0))[2])
        for member in group:
            consumed.add(member.chunk.chunk_id)
        if len(group) == 1:
            candidate = item
        else:
            from .data import Chunk

            text = ""
            for member in group:
                text = _merge_text(text, member.chunk.text)
            source_ids = [member.chunk.chunk_id for member in group]
            candidate = Evidence(
                chunk=Chunk(
                    chunk_id=f"{group[0].chunk.doc_id}::merged::{'+'.join(source_ids)}",
                    doc_id=group[0].chunk.doc_id,
                    title=group[0].chunk.title,
                    heading=group[0].chunk.heading,
                    text=text,
                    metadata={**group[0].chunk.metadata, "_merged_chunk_ids": source_ids},
                ),
                score=max(member.score for member in group),
                bm25_score=max(member.bm25_score for member in group),
                dense_score=max(member.dense_score for member in group),
                reranker_score=max(member.reranker_score for member in group),
                stage="merged",
                relation_type=next((member.relation_type for member in group if member.relation_type), ""),
                relation_path=max((member.relation_path for member in group), key=len, default=[]),
            )
        merged.append(candidate)

    if not prefer_document_diversity:
        return merged[:limit]
    selected: list[Evidence] = []
    top_score = max((item.score for item in merged), default=0.0)
    remaining = [
        item
        for item in merged
        if top_score <= 0.0 or item.score >= top_score * max(0.0, minimum_score_ratio)
    ]
    seen_docs: set[str] = set()
    while remaining and len(selected) < limit:
        allowed = [
            (index, item)
            for index, item in enumerate(remaining)
            if item.chunk.doc_id in seen_docs
            or max_documents is None
            or len(seen_docs) < max_documents
        ]
        if not allowed:
            break
        best_index, item = min(
            allowed,
            key=lambda pair: (
                -(
                    pair[1].score
                    + (diversity_bonus if pair[1].chunk.doc_id not in seen_docs else 0.0)
                ),
                pair[1].chunk.chunk_id,
            ),
        )
        remaining.pop(best_index)
        selected.append(item)
        seen_docs.add(item.chunk.doc_id)
    return selected


class HybridRetriever:
    def __init__(
        self,
        config: dict[str, Any],
        index: LegalIndex,
        dense_encoder: DenseEncoder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.config = config
        self.index = index
        self.dense_encoder = dense_encoder
        self.reranker = reranker

    def retrieve_bm25_only(
        self,
        query: str,
        question: str | None = None,
        current_evidence: list[Evidence] | None = None,
        top_k: int | None = None,
    ) -> list[Evidence]:
        """Tra cứu CPU-only và legal-rerank, không cần model GPU còn sống.

        Dùng cho vòng lặp tool-call của model sinh: model có thể tự phát ra
        truy vấn giữa lúc generate, sau khi dense/reranker đã được unload để
        tiết kiệm VRAM (xem pipeline.run_pipeline).
        """
        cfg = self.config["retrieval"]
        limit = int(top_k if top_k is not None else cfg.get("agentic_bm25_top_k", cfg["bm25_top_k"]))
        results = self.index.bm25.search(query, limit)
        candidates = [
            Evidence(chunk=self.index.chunks[chunk_index], score=score, bm25_score=score, stage="tool_call")
            for chunk_index, score in results
        ]
        return rank_agentic_evidence(
            question or query,
            query,
            current_evidence or [],
            candidates,
            self.config,
        )

    def retrieve(self, question: str) -> list[Evidence]:
        cfg = self.config["retrieval"]
        bm25 = self.index.bm25.search(question, int(cfg["bm25_top_k"]))
        dense: list[tuple[int, float]] = []
        if self.index.dense is not None and self.dense_encoder and int(cfg["dense_top_k"]) > 0:
            vector = self.dense_encoder.encode([question])[0]
            dense = self.index.dense_search(vector, int(cfg["dense_top_k"]))

        fused: dict[int, float] = {}
        rrf_k = int(cfg.get("rrf_k", 60))
        _rrf_add(fused, bm25, float(cfg.get("bm25_weight", 1.0)), rrf_k)
        _rrf_add(fused, dense, float(cfg.get("dense_weight", 1.0)), rrf_k)
        bm25_map, dense_map = dict(bm25), dict(dense)

        query_citations, query_numbers = citations(question), numbers(question)
        for index in list(fused):
            chunk = self.index.chunks[index]
            citation_overlap = len(query_citations & citations(chunk.text))
            number_overlap = len(query_numbers & numbers(chunk.text))
            fused[index] += float(cfg.get("citation_boost", 0.0)) * (citation_overlap + 0.25 * number_overlap)

        candidates = sorted(fused, key=lambda index: (-fused[index], index))[: int(cfg["fused_top_k"])]
        rerank_values: dict[int, float] = {}
        if self.reranker and int(cfg.get("rerank_top_k", 0)) > 0:
            rerank_candidates = candidates[: int(cfg["rerank_top_k"])]
            raw_scores = self.reranker.score(question, [self.index.chunks[index].text for index in rerank_candidates])
            rerank_values = {index: sigmoid(score) for index, score in zip(rerank_candidates, raw_scores)}
            for index, value in rerank_values.items():
                fused[index] += float(cfg.get("reranker_weight", 0.7)) * value

        candidates.sort(key=lambda index: (-fused[index], index))
        limit = int(cfg["evidence_top_k"])
        return [
            Evidence(
                chunk=self.index.chunks[index],
                score=float(fused[index]),
                bm25_score=float(bm25_map.get(index, 0.0)),
                dense_score=float(dense_map.get(index, 0.0)),
                reranker_score=float(rerank_values.get(index, 0.0)),
            )
            for index in candidates[:limit]
        ]
