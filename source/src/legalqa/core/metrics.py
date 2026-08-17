from __future__ import annotations

from collections import defaultdict, deque
from statistics import mean
from typing import Any

from .text import words


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = words(prediction), words(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for pred_token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, 1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def meteor_exact(prediction: str, reference: str) -> float:
    """METEOR probe using exact Vietnamese token matches and standard fragmentation penalty.

    It intentionally uses no WordNet/paraphrase resource, so this is a local diagnostic,
    not a claim to reproduce the organizer's hidden scorer bit-for-bit.
    """
    pred, ref = words(prediction), words(reference)
    if not pred or not ref:
        return 0.0
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(ref):
        positions[token].append(index)
    aligned: list[int] = []
    for token in pred:
        if positions[token]:
            aligned.append(positions[token].popleft())
    matches = len(aligned)
    if matches == 0:
        return 0.0
    precision, recall = matches / len(pred), matches / len(ref)
    fmean = (10 * precision * recall) / (recall + 9 * precision) if precision and recall else 0.0
    chunks = 1 + sum(aligned[index] != aligned[index - 1] + 1 for index in range(1, matches))
    penalty = 0.5 * (chunks / matches) ** 3
    return fmean * (1 - penalty)


def evaluate_predictions(predictions: dict[str, str], references: dict[str, str]) -> dict[str, Any]:
    if set(predictions) != set(references):
        missing = sorted(set(references) - set(predictions))[:10]
        extra = sorted(set(predictions) - set(references))[:10]
        raise ValueError(f"ID không khớp. Thiếu={missing}; thừa={extra}")
    rows = []
    for question_id in references:
        prediction, reference = predictions[question_id], references[question_id]
        rows.append(
            {
                "question_id": question_id,
                "meteor_exact_probe": meteor_exact(prediction, reference),
                "rouge_l_f1": rouge_l_f1(prediction, reference),
                "prediction_words": len(words(prediction)),
                "reference_words": len(words(reference)),
            }
        )
    return {
        "count": len(rows),
        "meteor_exact_probe": mean(row["meteor_exact_probe"] for row in rows) if rows else 0.0,
        "rouge_l_f1": mean(row["rouge_l_f1"] for row in rows) if rows else 0.0,
        "prediction_words_mean": mean(row["prediction_words"] for row in rows) if rows else 0.0,
        "reference_words_mean": mean(row["reference_words"] for row in rows) if rows else 0.0,
        "per_question": rows,
        "note": "METEOR ở đây chỉ khớp token chính xác; leaderboard của BTC là nguồn điểm chính thức.",
    }

