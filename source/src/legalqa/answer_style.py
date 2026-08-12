from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from .data import QARecord
from .text import citations, numbers, question_type, split_sentences, words


STYLE_PROFILE_SCHEMA = "answer-style-v1"


def _percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return int(ordered[lower])
    value = ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    return int(round(value))


def _opening(answer: str) -> str:
    lowered = answer.casefold().lstrip()
    for label, pattern in (
        ("theo_quy_dinh", r"^theo\s+(?:quy định|điều|khoản|luật|nghị định|thông tư)"),
        ("can_cu", r"^căn cứ\b"),
        ("doi_voi", r"^đối với\b"),
        ("truong_hop", r"^trường hợp\b"),
        ("direct", r"."),
    ):
        if re.search(pattern, lowered):
            return label
    return "direct"


def _summarize(records: Iterable[QARecord]) -> dict[str, Any]:
    rows = [record for record in records if record.answer]
    word_counts = [len(words(record.answer or "")) for record in rows]
    sentence_counts = [len(split_sentences(record.answer or "")) for record in rows]
    openings = Counter(_opening(record.answer or "") for record in rows)
    count = len(rows)
    return {
        "count": count,
        "words": {
            f"p{int(probability * 100):02d}": _percentile(word_counts, probability)
            for probability in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
        "sentences": {
            f"p{int(probability * 100):02d}": _percentile(sentence_counts, probability)
            for probability in (0.25, 0.50, 0.75, 0.90)
        },
        "citation_rate": (
            sum(bool(citations(record.answer or "")) for record in rows) / count if count else 0.0
        ),
        "numeric_rate": (
            sum(bool(numbers(record.answer or "")) for record in rows) / count if count else 0.0
        ),
        "list_rate": (
            sum(bool(re.search(r"(?:^|\n)\s*(?:[-•]|\d+[.)])\s+", record.answer or "")) for record in rows)
            / count
            if count
            else 0.0
        ),
        "openings": dict(openings.most_common()),
    }


def build_answer_style_profile(records: list[QARecord]) -> dict[str, Any]:
    """Learn only aggregate answer-style statistics from official train data."""
    answered = [record for record in records if record.answer]
    grouped: dict[str, list[QARecord]] = {}
    for record in answered:
        grouped.setdefault(question_type(record.question), []).append(record)
    return {
        "schema": STYLE_PROFILE_SCHEMA,
        "count": len(answered),
        "overall": _summarize(answered),
        "by_question_type": {
            label: _summarize(group) for label, group in sorted(grouped.items())
        },
    }


def generation_plan(
    question: str,
    route: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return reference-informed word and token limits for one question."""
    generation = config["generation"]
    profile = generation.get("answer_style_profile") or {}
    label = question_type(question)
    overall = profile.get("overall", {})
    group = profile.get("by_question_type", {}).get(label, {})
    if int(group.get("count", 0)) < int(generation.get("style_min_group_size", 20)):
        group = overall
    distribution = group.get("words", {})
    if distribution:
        if route == "complex":
            target_words = int(distribution.get("p75", distribution.get("p50", 220)))
            maximum_words = int(distribution.get("p95", distribution.get("p90", 420)))
        else:
            target_words = int(distribution.get("p50", 140))
            maximum_words = int(distribution.get("p90", 280))
        minimum_words = int(distribution.get("p10", 35))
        learned = True
    else:
        target_words = int(
            generation.get(
                "complex_target_answer_words" if route == "complex" else "simple_target_answer_words",
                generation.get("target_answer_words", 180),
            )
        )
        minimum_words = int(generation.get("min_answer_words", 35))
        maximum_words = int(generation.get("max_answer_words", 600))
        learned = False
    floor = int(generation.get("style_floor_words", 35))
    ceiling = int(generation.get("style_ceiling_words", 700))
    minimum_words = max(15, min(minimum_words, target_words))
    target_words = max(floor, min(target_words, ceiling))
    maximum_words = max(target_words + 20, min(maximum_words, ceiling))
    tokens_per_word = float(generation.get("tokens_per_vietnamese_word", 1.65))
    planned_tokens = int(math.ceil(maximum_words * tokens_per_word + 24))
    route_cap = int(
        generation.get(
            "complex_max_new_tokens" if route == "complex" else "simple_max_new_tokens",
            generation["max_new_tokens"],
        )
    )
    token_limit = max(96, min(planned_tokens, route_cap, int(generation["max_new_tokens"])))
    return {
        "question_type": label,
        "route": route,
        "learned_from_train": learned,
        "minimum_words": minimum_words,
        "target_words": target_words,
        "maximum_words": maximum_words,
        "max_new_tokens": token_limit,
        "citation_rate": float(group.get("citation_rate", overall.get("citation_rate", 0.0))),
        "list_rate": float(group.get("list_rate", overall.get("list_rate", 0.0))),
    }
