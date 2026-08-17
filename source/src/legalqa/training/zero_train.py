from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.text import citations, document_numbers


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)|\n{2,}")
FINGERPRINT_RE = re.compile(r"[^0-9a-zà-ỹđ]+", re.IGNORECASE)


@dataclass(frozen=True)
class KNNThresholds:
    combined: float = 0.94
    word: float = 0.95
    character: float = 0.91
    margin: float = 0.20


@dataclass(frozen=True)
class KNNMatch:
    train_id: str
    answer: str
    combined_score: float
    word_score: float
    character_score: float
    margin: float


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text)).replace("\ufeff", "").lower()
    return re.sub(r"\s+", " ", value).strip()


def answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _qa_rows(payload: dict[str, Any], require_answer: bool) -> list[tuple[str, str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("Dữ liệu QA phải là JSON Object theo question_id.")
    rows: list[tuple[str, str, str]] = []
    for raw_id, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"QA {raw_id} thiếu question dạng chuỗi.")
        answer = value.get("answer", "")
        if require_answer and not isinstance(answer, str):
            raise ValueError(f"QA {raw_id} thiếu answer dạng chuỗi.")
        rows.append((str(raw_id), value["question"], str(answer)))
    return rows


def _same_explicit_legal_scope(public_question: str, train_question: str) -> bool:
    """Reject a near duplicate when its explicit legal anchors changed."""
    public_documents, train_documents = document_numbers(public_question), document_numbers(train_question)
    if public_documents or train_documents:
        if public_documents != train_documents:
            return False
    public_articles, train_articles = citations(public_question), citations(train_question)
    if public_articles or train_articles:
        if public_articles != train_articles:
            return False
    return True


def high_confidence_train_knn(
    train_payload: dict[str, Any],
    public_payload: dict[str, Any],
    thresholds: KNNThresholds | None = None,
) -> tuple[dict[str, KNNMatch], dict[str, Any]]:
    """Find only lexical near duplicates safe enough to copy an official answer.

    Word n-grams protect content words and legal entities; character n-grams
    tolerate small Vietnamese wording changes.  A large top-1/top-2 margin and
    exact explicit legal anchors keep topic-neighbours from becoming copies.
    """
    cfg = thresholds or KNNThresholds()
    train_rows = _qa_rows(train_payload, require_answer=True)
    public_rows = _qa_rows(public_payload, require_answer=False)
    all_questions = [normalize_question(row[1]) for row in train_rows + public_rows]
    if not train_rows or not public_rows:
        return {}, {"train": len(train_rows), "public": len(public_rows), "accepted": 0}

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    word_matrix = word_vectorizer.fit_transform(all_questions)
    character_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=180_000,
    )
    character_matrix = character_vectorizer.fit_transform(all_questions)
    train_count = len(train_rows)
    word_scores = word_matrix[train_count:] @ word_matrix[:train_count].T
    character_scores = character_matrix[train_count:] @ character_matrix[:train_count].T
    combined_scores = 0.55 * word_scores + 0.45 * character_scores

    accepted: dict[str, KNNMatch] = {}
    distribution = {"ge_090": 0, "ge_092": 0, "ge_094": 0, "ge_096": 0, "ge_098": 0}
    rejected_scope = 0
    for index, (public_id, public_question, _) in enumerate(public_rows):
        dense_combined = np.asarray(combined_scores[index].toarray()).ravel()
        best_index = int(dense_combined.argmax())
        best_combined = float(dense_combined[best_index])
        second = float(np.partition(dense_combined, -2)[-2]) if dense_combined.size > 1 else 0.0
        margin = best_combined - second
        for label, threshold in (("ge_090", .90), ("ge_092", .92), ("ge_094", .94), ("ge_096", .96), ("ge_098", .98)):
            distribution[label] += int(best_combined >= threshold)
        best_word = float(word_scores[index, best_index])
        best_character = float(character_scores[index, best_index])
        train_id, train_question, train_answer = train_rows[best_index]
        score_gate = (
            best_combined >= cfg.combined
            and best_word >= cfg.word
            and best_character >= cfg.character
            and margin >= cfg.margin
        )
        if not score_gate:
            continue
        if not _same_explicit_legal_scope(public_question, train_question):
            rejected_scope += 1
            continue
        accepted[public_id] = KNNMatch(
            train_id=train_id,
            answer=train_answer,
            combined_score=best_combined,
            word_score=best_word,
            character_score=best_character,
            margin=margin,
        )

    report = {
        "train": len(train_rows),
        "public": len(public_rows),
        "accepted": len(accepted),
        "rejected_legal_scope": rejected_scope,
        "thresholds": {
            "combined": cfg.combined,
            "word": cfg.word,
            "character": cfg.character,
            "margin": cfg.margin,
        },
        "nearest_score_distribution": distribution,
    }
    return accepted, report


def _candidate_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = trace.get("candidates", [])
    if not isinstance(candidates, list):
        return {}
    return {
        str(candidate.get("source")): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("answer"), str)
    }


def conservative_reselect(trace: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return a different stored candidate only under very strong safety gates."""
    candidates = _candidate_map(trace)
    selected_source = str(trace.get("selected", ""))
    selected = candidates.get(selected_source)
    if selected is None:
        raise ValueError(f"Trace {trace.get('question_id')} không có candidate đã chọn.")
    selected_answer = selected["answer"]

    if selected_source == "extractive" and "qwen3.5-2b" in candidates:
        qwen = candidates["qwen3.5-2b"]
        qwen_details = qwen.get("details", {})
        selected_details = selected.get("details", {})
        safe = (
            not bool(qwen_details.get("truncated", False))
            and float(qwen.get("score", 0.0)) >= float(selected.get("score", 0.0)) - 0.03
            and float(qwen_details.get("evidence_support", 0.0)) >= 0.90
            and float(qwen_details.get("unsupported", 1.0)) <= 0.12
            and float(qwen_details.get("citation_precision", 0.0)) >= 0.90
            and float(qwen_details.get("numeric_precision", 0.0)) >= 0.90
            and float(qwen_details.get("fluency", 0.0))
            >= float(selected_details.get("fluency", 0.0)) + 0.08
        )
        if safe:
            return qwen["answer"], "qwen3.5-2b", {"changed": True, "reason": "strict_qwen_gate"}

    if selected_source == "qwen3.5-2b" and bool(selected.get("details", {}).get("truncated", False)):
        extractive = candidates.get("extractive")
        if extractive is not None:
            safe = (
                float(extractive.get("score", 0.0)) >= float(selected.get("score", 0.0)) - 0.02
                and float(extractive.get("details", {}).get("evidence_support", 0.0)) >= 0.90
                and float(extractive.get("details", {}).get("fluency", 0.0)) >= 0.75
            )
            if safe:
                return extractive["answer"], "extractive", {"changed": True, "reason": "truncated_qwen_fallback"}

    return selected_answer, selected_source, {"changed": False, "reason": "keep_v8_selection"}


def _sentence_fingerprint(sentence: str) -> str:
    normalized = unicodedata.normalize("NFC", sentence).lower()
    return re.sub(r"\s+", " ", FINGERPRINT_RE.sub(" ", normalized)).strip()


def remove_exact_sentence_duplicates(answer: str) -> tuple[str, int]:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(answer.strip()) if piece.strip()]
    if len(pieces) < 2:
        return answer.strip(), 0
    seen: set[str] = set()
    selected: list[str] = []
    removed = 0
    for piece in pieces:
        fingerprint = _sentence_fingerprint(piece)
        if len(fingerprint.split()) >= 8 and fingerprint in seen:
            removed += 1
            continue
        if fingerprint:
            seen.add(fingerprint)
        selected.append(piece)
    if not removed:
        return answer.strip(), 0
    separator = "\n" if "\n" in answer else " "
    return separator.join(selected).strip(), removed


def capitalize_first_letter(answer: str) -> tuple[str, bool]:
    for index, character in enumerate(answer):
        if character.isalpha():
            upper = character.upper()
            if upper != character:
                return answer[:index] + upper + answer[index + 1 :], True
            break
    return answer, False


def close_truncated_answer(answer: str) -> tuple[str, bool]:
    """Drop only an unfinished tail; never impose a word ceiling."""
    stripped = answer.rstrip()
    if not stripped or stripped[-1] in ".!?;:”\"'":
        return stripped, False
    last_boundary = max(stripped.rfind("."), stripped.rfind("!"), stripped.rfind("?"))
    if last_boundary < 0:
        return stripped, False
    prefix, tail = stripped[: last_boundary + 1].rstrip(), stripped[last_boundary + 1 :].strip()
    if len(tail.split()) < 4 or len(prefix.split()) < 40:
        return stripped, False
    if len(prefix.split()) / max(1, len(stripped.split())) < 0.70:
        return stripped, False
    return prefix, True


def conservative_postprocess(answer: str, truncated: bool = False) -> tuple[str, dict[str, Any]]:
    result = unicodedata.normalize("NFC", str(answer)).replace("\ufeff", "").strip()
    actions: dict[str, Any] = {
        "capitalized": False,
        "duplicate_sentences_removed": 0,
        "truncated_tail_removed": False,
    }
    if truncated:
        result, actions["truncated_tail_removed"] = close_truncated_answer(result)
    result, actions["duplicate_sentences_removed"] = remove_exact_sentence_duplicates(result)
    result, actions["capitalized"] = capitalize_first_letter(result)
    return result, actions
