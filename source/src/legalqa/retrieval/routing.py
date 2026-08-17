from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .legal_graph import has_complex_cue
from .retrieval import Evidence, retrieval_confidence
from ..core.text import citations, document_numbers, normalize_text


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
    simple_threshold = float(cfg.get("simple_confidence_threshold", 0.68))
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

    # Relations are extremely common in the corpus and therefore are not, by
    # themselves, evidence that a question needs multi-document expansion.
    # The v8 public trace routed 60.9% of questions to Complex and 96.1% of its
    # final graph chunks had reranker < 0.1. Complex is now opt-in on a strong
    # linguistic cue, genuinely weak retrieval, or *tight* cross-document tie.
    force_complex = complex_cue > 0 or subquestion_signal > 0
    weak_retrieval = retrieval_score < float(cfg.get("complex_below_confidence", 0.30))
    concentration_ceiling = float(
        cfg.get("complex_max_document_concentration", 0.40)
    )
    ambiguous = (
        multi_document_competition > 0
        and features.get("margin", 0.0) < float(cfg.get("complex_max_margin", 0.02))
        and features.get("document_concentration", 0.0) <= concentration_ceiling
    )
    relation_trigger = bool(
        discovered_relations > 0
        and explicit_citation > 0
        and (ambiguous or force_complex)
    )
    if force_complex or weak_retrieval or ambiguous or relation_trigger:
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
    else:
        route = "simple"
        reason = (
            "high-confidence one-document first pass"
            if retrieval_score >= simple_threshold
            else "bounded one-document route; no strong complex evidence"
        )
    return RouteDecision(route, retrieval_score, reason, features)
