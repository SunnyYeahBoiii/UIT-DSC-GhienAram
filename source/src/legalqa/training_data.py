from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import load_qa, write_jsonl
from .index import LegalIndex
from .text import citations, legal_tokens, numbers, overlap_ratio, words


def _features(text: str) -> tuple[list[str], set[str], set[str]]:
    return legal_tokens(text), citations(text), numbers(text)


def _alignment_score(
    answer_features: tuple[list[str], set[str], set[str]],
    passage_features: tuple[list[str], set[str], set[str]],
) -> float:
    answer_tokens, answer_citations, answer_numbers = answer_features
    passage_tokens, passage_citations, passage_numbers = passage_features
    token_score = overlap_ratio(answer_tokens, passage_tokens)
    citation_score = (
        len(answer_citations & passage_citations) / len(answer_citations)
        if answer_citations else 0.0
    )
    number_score = (
        len(answer_numbers & passage_numbers) / len(answer_numbers)
        if answer_numbers else 0.0
    )
    return 0.65 * token_score + 0.25 * citation_score + 0.10 * number_score


def _alignment_query(answer: str) -> str:
    # Long references contain many common legal words. Limit the BM25 query while
    # retaining first-occurrence content terms, citations and numbers.
    unique_content = list(dict.fromkeys(words(answer, remove_stopwords=True)))[:120]
    return " ".join(unique_content + sorted(citations(answer)) + sorted(numbers(answer)))


def prepare_retrieval_pairs(config: dict[str, Any], train_path: str | Path | None = None) -> dict[str, Any]:
    artifacts = Path(config["paths"]["artifacts"])
    train_path = Path(train_path) if train_path else artifacts / "data" / "train.json"
    records = load_qa(train_path, require_answers=True)
    index = LegalIndex.load(artifacts / "index")
    output_path = artifacts / "data" / "retrieval_pairs.jsonl"
    rows = []
    skipped = 0
    chunk_features = [_features(chunk.text) for chunk in index.chunks]
    for item in records:
        # Answer-to-corpus alignment creates weak labels solely from official data.
        answer_features = _features(item.answer or "")
        answer_hits = index.bm25.search(_alignment_query(item.answer or ""), min(40, len(index.chunks)))
        rescored = [
            (chunk_index, _alignment_score(answer_features, chunk_features[chunk_index]))
            for chunk_index, _ in answer_hits
        ]
        rescored.sort(key=lambda pair: (-pair[1], pair[0]))
        if not rescored or rescored[0][1] <= 0:
            skipped += 1
            continue
        positive_index, alignment = rescored[0]
        question_hits = index.bm25.search(item.question, min(50, len(index.chunks)))
        negatives = []
        for chunk_index, _ in question_hits:
            if chunk_index == positive_index:
                continue
            if _alignment_score(answer_features, chunk_features[chunk_index]) < max(0.15, alignment * 0.55):
                negatives.append(index.chunks[chunk_index].text)
            if len(negatives) >= 3:
                break
        if not negatives:
            negatives = [
                index.chunks[chunk_index].text
                for chunk_index, _ in reversed(rescored[-3:])
                if chunk_index != positive_index
            ][:3]
        rows.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "positive": index.chunks[positive_index].text,
                "positive_chunk_id": index.chunks[positive_index].chunk_id,
                "alignment_score": alignment,
                "negatives": negatives,
            }
        )
    write_jsonl(output_path, rows)
    return {"pairs": len(rows), "skipped": skipped, "output": str(output_path.resolve())}
