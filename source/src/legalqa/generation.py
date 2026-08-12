from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .answer_style import generation_plan
from .prompts import SYSTEM_PROMPT, qwen_user_prompt
from .retrieval import Evidence
from .text import citations, legal_tokens, normalize_text, numbers, split_sentences, words


@dataclass
class Candidate:
    source: str
    answer: str
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def clean_answer(text: str) -> str:
    answer = normalize_text(text)
    answer = re.sub(r"^(?:câu trả lời|trả lời)\s*[:：]\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)", "", answer).strip()
    return answer


def compact_generated_answer(
    text: str,
    maximum_words: int,
    *,
    hard_trim: bool = False,
) -> tuple[str, dict[str, int | bool]]:
    """Remove exact sentence loops; optionally stop at a complete-sentence boundary."""
    answer = clean_answer(text)
    original_words = len(words(answer))
    sentences = split_sentences(answer)
    selected: list[str] = []
    seen: set[str] = set()
    selected_words = 0
    duplicate_sentences = 0
    for sentence in sentences:
        fingerprint = " ".join(words(sentence))
        if fingerprint and fingerprint in seen:
            duplicate_sentences += 1
            continue
        sentence_words = len(words(sentence))
        # Never cut a single long legal sentence in the middle. Once at least
        # one complete sentence exists, however, stop before crossing p90/p95.
        if hard_trim and selected and selected_words + sentence_words > maximum_words:
            break
        selected.append(sentence.strip())
        selected_words += sentence_words
        if fingerprint:
            seen.add(fingerprint)
    compacted = " ".join(item for item in selected if item).strip() or answer
    if compacted and not compacted.endswith((".", "!", "?", ";", ":")):
        compacted += "."
    compacted_words = len(words(compacted))
    return compacted, {
        "raw_words": original_words,
        "compacted_words": compacted_words,
        "duplicate_sentences_removed": duplicate_sentences,
        "compaction_applied": compacted != answer,
    }


def _sentence_quality(sentence: str) -> float:
    tokens = words(sentence)
    if len(tokens) < 8:
        return 0.0
    placeholder_ratio = sentence.count(".") / max(1, len(sentence))
    letters = sum(character.isalpha() for character in sentence)
    uppercase = sum(character.isupper() for character in sentence)
    if placeholder_ratio > 0.08 or letters / max(1, len(sentence)) < 0.45:
        return 0.0
    if letters > 20 and uppercase / letters > 0.82:
        return 0.25
    return min(1.0, len(tokens) / 24.0)


def extractive_candidate(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    query_tokens = set(legal_tokens(question))
    ranked: list[tuple[float, int, int, str]] = []
    for evidence_rank, item in enumerate(evidence):
        for sentence_rank, sentence in enumerate(split_sentences(item.chunk.text)):
            sentence_tokens = set(legal_tokens(sentence))
            quality = _sentence_quality(sentence)
            if quality <= 0:
                continue
            overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
            citation_bonus = 0.13 if citations(sentence) else 0.0
            number_bonus = 0.08 if numbers(question) & numbers(sentence) else 0.0
            rank_prior = 0.20 / (evidence_rank + 1)
            length_bonus = min(len(words(sentence)), 45) / 450.0
            ranked.append((overlap + citation_bonus + number_bonus + rank_prior + length_bonus + 0.18 * quality, evidence_rank, sentence_rank, sentence))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    target_words = int(plan["target_words"])
    selected: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    count = 0
    for _, evidence_rank, sentence_rank, sentence in ranked:
        sentence_words = words(sentence)
        fingerprint = " ".join(sentence_words)
        if fingerprint in seen:
            continue
        sentence_set = set(sentence_words)
        if any(
            len(sentence_set & set(words(previous))) / max(1, len(sentence_set)) > 0.82
            for _, _, previous in selected
        ):
            continue
        selected.append((evidence_rank, sentence_rank, sentence))
        seen.add(fingerprint)
        count += len(words(sentence))
        if count >= target_words:
            break
    selected.sort(key=lambda item: (item[0], item[1]))
    body = " ".join(sentence for _, _, sentence in selected).strip()
    if not body:
        body = "Chưa đủ thông tin pháp lý để xác định câu trả lời."
    if body and body[0].islower():
        body = body[0].upper() + body[1:]
    if body.casefold().startswith(("theo ", "căn cứ ", "điều ", "khoản ", "điểm ")):
        answer = body
    else:
        answer = f"Theo quy định pháp luật, {body[0].lower() + body[1:] if len(body) > 1 else body}"
    if not answer.rstrip().endswith((".", "!", "?")):
        answer += "."
    return Candidate(
        "extractive",
        clean_answer(answer),
        details={"route": route, "selected_sentences": len(selected), "style_plan": plan},
    )


def make_qwen_candidate(
    generator,
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    token_limit = int(plan["max_new_tokens"])
    result = generator.generate_with_metadata(
        SYSTEM_PROMPT,
        qwen_user_prompt(question, evidence, config, retrieval_diagnostics),
        max_new_tokens=token_limit,
    )
    answer = clean_answer(result.text)
    if not answer:
        raise RuntimeError("Qwen sinh câu trả lời rỗng.")
    compaction = {
        "raw_words": len(words(answer)),
        "compacted_words": len(words(answer)),
        "duplicate_sentences_removed": 0,
        "compaction_applied": False,
    }
    if bool(config.get("generation", {}).get("compact_generated_answers", True)):
        answer, compaction = compact_generated_answer(
            answer,
            int(plan["maximum_words"]),
            hard_trim=bool(
                config.get("generation", {}).get("hard_trim_generated_answers", False)
            ),
        )
    return Candidate(
        "qwen3.5-2b",
        answer,
        details={
            "route": route,
            "generated_tokens": result.generated_tokens,
            "terminated_by_eos": result.terminated_by_eos,
            "truncated": result.truncated,
            "token_limit": token_limit,
            "style_plan": plan,
            **compaction,
            "fewshot_examples": sum(
                float(item.get("similarity", 0.0))
                >= float(config.get("retrieval", {}).get("train_memory_fewshot_threshold", 0.70))
                for item in (retrieval_diagnostics or {}).get("train_memory", [])
            ),
        },
    )


def make_train_memory_candidate(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate | None:
    diagnostics = retrieval_diagnostics or {}
    neighbors = diagnostics.get("train_memory", [])
    if not neighbors:
        return None
    nearest = neighbors[0]
    similarity = float(nearest.get("similarity", 0.0))
    threshold = float(
        config.get("retrieval", {}).get("train_memory_candidate_threshold", 0.86)
    )
    if similarity < threshold:
        return None
    answer = clean_answer(str(nearest.get("answer", "")))
    if not answer:
        return None
    route = diagnostics.get("route", {}).get("route", "simple")
    return Candidate(
        "train-memory",
        answer,
        details={
            "route": route,
            "memory_similarity": similarity,
            "memory_question_id": str(nearest.get("question_id", "")),
            "memory_question": str(nearest.get("question", "")),
            "style_plan": generation_plan(question, route, config),
        },
    )
