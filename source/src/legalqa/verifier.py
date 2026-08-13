from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .answer_style import generation_plan
from .generation import Candidate
from .retrieval import Evidence, retrieval_confidence
from .text import citations, length_prior, numbers, overlap_ratio, split_sentences, words


def _duplicate_ratio(answer: str) -> float:
    sentences = [" ".join(words(sentence)) for sentence in split_sentences(answer) if words(sentence)]
    if not sentences:
        return 0.0
    counts = Counter(sentences)
    duplicate = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate / len(sentences)


def _fluency_score(answer: str) -> tuple[float, float]:
    raw_sentences = split_sentences(answer)
    tokenized = [words(sentence) for sentence in raw_sentences if words(sentence)]
    if not tokenized:
        return 0.0, 1.0
    fragments = sum(len(tokens) < 7 for tokens in tokenized) / len(tokenized)
    average_length = sum(map(len, tokenized)) / len(tokenized)
    length_score = 1.0 if 9 <= average_length <= 45 else max(0.0, 1 - abs(average_length - 27) / 40)
    terminal = float(answer.rstrip().endswith((".", "!", "?", ";", ":", "”")))
    foreign = float(bool(re.search(r"[\u0370-\u03ff\u0600-\u06ff\u0e00-\u0e7f\u4e00-\u9fff]", answer)))
    boilerplate = float(
        bool(re.search(r"\b(?:trả lại câu hỏi|trả lời pháp luật nước ngoài|căn cứ được cung cấp)\b", answer, re.I))
    )
    score = 0.34 * (1 - fragments) + 0.28 * length_score + 0.22 * terminal + 0.16 * (1 - boilerplate)
    score -= 0.7 * foreign
    return max(0.0, min(1.0, score)), fragments


def score_candidates(
    question: str,
    evidence: list[Evidence],
    candidates: list[Candidate],
    config: dict[str, Any],
    semantic_encoder=None,
    retrieval_diagnostics: dict[str, Any] | None = None,
    candidate_evidence_by_source: dict[str, list[Evidence]] | None = None,
) -> list[Candidate]:
    query_content = words(question, remove_stopwords=True)
    weights = config["selection"]
    generation = config["generation"]
    diagnostics = retrieval_diagnostics or {}
    route_payload = diagnostics.get("route", {})
    if "confidence" in route_payload:
        base_retrieval_score = float(route_payload["confidence"])
    else:
        base_retrieval_score, _ = retrieval_confidence(evidence)
    base_retrieval_score = max(0.0, min(1.0, base_retrieval_score))
    source_priors = weights.get(
        "source_priors",
        {
            "qwen3.5-2b": 0.08,
            "qwen3.5-2b-agentic": 0.08,
            "extractive": -0.04,
            "train-memory": 0.0,
        },
    )

    for candidate in candidates:
        generation_details = dict(candidate.details)
        active_evidence = (candidate_evidence_by_source or {}).get(candidate.source, evidence)
        evidence_text = "\n".join(item.chunk.text for item in active_evidence)
        evidence_content = words(evidence_text, remove_stopwords=True)
        evidence_citations = citations(evidence_text)
        evidence_numbers = numbers(evidence_text)
        retrieval_score = base_retrieval_score
        if candidate.source == "qwen3.5-2b-agentic":
            retrieval_score = max(
                retrieval_score,
                float(generation_details.get("tool_retrieval_confidence", retrieval_score)),
            )
        retrieval_score = max(0.0, min(1.0, retrieval_score))
        evidence_trust = 0.30 + 0.70 * retrieval_score
        answer_content = words(candidate.answer, remove_stopwords=True)
        answer_citations = citations(candidate.answer)
        answer_numbers = numbers(candidate.answer)
        evidence_support = overlap_ratio(answer_content, evidence_content)
        query_coverage = overlap_ratio(query_content, answer_content)
        citation_precision = (
            len(answer_citations & evidence_citations) / len(answer_citations)
            if answer_citations else 0.45
        )
        numeric_precision = (
            len(answer_numbers & evidence_numbers) / len(answer_numbers)
            if answer_numbers else 1.0
        )
        unsupported = 0.5 * (1 - citation_precision) + 0.5 * (1 - numeric_precision)
        word_count = len(words(candidate.answer))
        route = str(generation_details.get("route") or route_payload.get("route", "simple"))
        plan = generation_plan(question, route, config)
        length_score = length_prior(
            word_count,
            int(plan["target_words"]),
            int(plan["minimum_words"]),
            int(plan["maximum_words"]),
        )
        duplicate = _duplicate_ratio(candidate.answer)
        precomputed_semantic = generation_details.get("_precomputed_legal_semantic")
        semantic = (
            float(precomputed_semantic)
            if precomputed_semantic is not None
            else semantic_encoder.similarity(candidate.answer, evidence_text)
            if semantic_encoder
            else 0.0
        )
        fluency, fragment_ratio = _fluency_score(candidate.answer)
        truncated = float(bool(generation_details.get("truncated", False)))
        fallback_prior = (
            source_priors.get("qwen3.5-2b", 0.0)
            if candidate.source == "qwen3.5-2b-agentic"
            else 0.0
        )
        source_prior = float(source_priors.get(candidate.source, fallback_prior))
        source_rejected = candidate.source.startswith("bartpho")
        score = (
            float(weights["evidence_support"]) * evidence_support * evidence_trust
            + float(weights["query_coverage"]) * query_coverage
            + float(weights["legal_citations"]) * citation_precision
            + float(weights["numeric_faithfulness"]) * numeric_precision
            + float(weights["legal_semantic"]) * semantic
            + float(weights["length_prior"]) * length_score
            + float(weights.get("fluency", 0.18)) * fluency
            + source_prior
            - float(weights["unsupported_penalty"]) * unsupported
            - float(weights["duplicate_penalty"]) * duplicate
            - float(weights.get("fragment_penalty", 0.14)) * fragment_ratio
            - float(weights.get("truncation_penalty", 0.18)) * truncated
            - float(weights.get("retrieval_risk_penalty", 0.08))
            * (1 - retrieval_score)
            * (unsupported + float(candidate.source.startswith("qwen")) * 0.15)
        )
        if source_rejected:
            score = -1.0
        candidate.score = score
        candidate.details = {
            **{key: value for key, value in generation_details.items() if not key.startswith("_precomputed_")},
            "evidence_support": evidence_support,
            "evidence_trust": evidence_trust,
            "retrieval_confidence": retrieval_score,
            "query_coverage": query_coverage,
            "citation_precision": citation_precision,
            "numeric_precision": numeric_precision,
            "legal_semantic": semantic,
            "length_prior": length_score,
            "unsupported": unsupported,
            "duplicate": duplicate,
            "fluency": fluency,
            "fragment_ratio": fragment_ratio,
            "source_prior": source_prior,
            "truncation_penalty_applied": truncated,
            "words": float(word_count),
        }
    ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.source))
    qwen_margin = float(weights.get("qwen_preference_margin", 0.04))
    qwen = next(
        (
            candidate for candidate in ranked
            if candidate.source.startswith("qwen")
            and not candidate.details.get("truncated", False)
        ),
        None,
    )
    if qwen is not None and ranked and ranked[0].score - qwen.score <= qwen_margin:
        ranked.remove(qwen)
        ranked.insert(0, qwen)
    return ranked
