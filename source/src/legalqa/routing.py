from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .legal_graph import has_complex_cue
from .retrieval import Evidence, retrieval_confidence
from .text import citations, document_numbers, normalize_text


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    reason: str
    features: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_route(
    question: str,
    evidence: list[Evidence],
    relation_count: int,
    config: dict[str, Any],
) -> RouteDecision:
    """Route easy one-document questions away from costly graph expansion."""
    retrieval_score, features = retrieval_confidence(evidence)
    cfg = config.get("routing", {})
    simple_threshold = float(cfg.get("simple_confidence_threshold", 0.72))
    minimum_reranker = float(cfg.get("simple_min_reranker", 0.72))
    top_reranker = float(features.get("top_reranker", 0.0))
    explicit_citation = float(bool(citations(question) or document_numbers(question)))
    complex_cue = float(has_complex_cue(question))
    top_docs = {item.chunk.doc_id for item in evidence[:5]}
    multi_document_competition = float(len(top_docs) >= 3)
    discovered_relations = float(relation_count > 0)
    normalized = normalize_text(question)
    subquestion_signal = float(normalized.count("?") > 1 or normalized.count(";") >= 2)
    features.update(
        explicit_citation=explicit_citation,
        complex_cue=complex_cue,
        multi_document_competition=multi_document_competition,
        discovered_relations=discovered_relations,
        subquestion_signal=subquestion_signal,
    )

    force_complex = complex_cue > 0 or subquestion_signal > 0
    weak_retrieval = retrieval_score < float(cfg.get("complex_below_confidence", 0.46))
    ambiguous = multi_document_competition > 0 and features.get("margin", 0.0) < float(
        cfg.get("complex_max_margin", 0.08)
    )
    relation_trigger = discovered_relations > 0 and (
        explicit_citation > 0 or retrieval_score < simple_threshold
    )
    safe_simple = (
        retrieval_score >= simple_threshold
        and top_reranker >= minimum_reranker
        and not force_complex
        and not ambiguous
        and not relation_trigger
    )
    if safe_simple:
        reason = "high-confidence one-document first pass"
        route = "simple"
    else:
        reasons = []
        if force_complex:
            reasons.append("complex cue")
        if weak_retrieval:
            reasons.append("low retrieval confidence")
        if ambiguous:
            reasons.append("multi-document ambiguity")
        if relation_trigger:
            reasons.append("citation/relation discovered")
        if not reasons:
            reasons.append("simple confidence gate not met")
        reason = ", ".join(reasons)
        route = "complex"
    return RouteDecision(route, retrieval_score, reason, features)
