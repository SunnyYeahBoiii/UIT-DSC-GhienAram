from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.generation.generation import clean_answer


_TAI_DAY_RE = re.compile(r"Tải\s+[^.\n]{0,120}tại\s+đây\.?", re.IGNORECASE)
_TAI_DAY_LINE_RE = re.compile(r"^[^\S\n]*Tải\s+.*?tại\s+đây\.?\s*$", re.IGNORECASE | re.MULTILINE)
_TAI_DAY_STANDALONE_RE = re.compile(r"^[^\S\n]*Tại\s+đây\.?\s*$", re.IGNORECASE | re.MULTILINE)
_TAI_DAY_END_SENTENCE_RE = re.compile(r"\.\s*Tại\s+đây\.?", re.IGNORECASE)
_TAI_DAY_PAREN_RE = re.compile(r"\(\s*tại\s+đây\s*\)", re.IGNORECASE)
_TAI_DAY_BEFORE_CLAUSE_RE = re.compile(r"Tại\s+đây\s+(?=Căn cứ|Theo|Tại|Quan điểm|Hướng dẫn)", re.IGNORECASE)
_TAI_DAY_NHAT_RE = re.compile(r"nhất\s+Tại\s+đây\.?", re.IGNORECASE)
_TAI_DAY_SEMICOLON_RE = re.compile(r"tại\s+đây\s*;", re.IGNORECASE)
_XEM_TAI_DAY_RE = re.compile(r"xem\s+(?:toàn\s+vn|.*?)?tại\s+đây\.?", re.IGNORECASE)
_IMAGE_RE = re.compile(r"\(?\s*(?:Hình|Ảnh)\s+từ\s+Internet\)?", re.IGNORECASE)
_IMAGE_SOURCE_RE = re.compile(r"\(\s*Nguồn\s*(?:ảnh|hình)\s*:?[\s_]*Internet\s*\)", re.IGNORECASE)


def preprocess_answer(text: str) -> str:
    answer = _TAI_DAY_LINE_RE.sub("", text)
    answer = _TAI_DAY_STANDALONE_RE.sub("", answer)
    answer = _TAI_DAY_END_SENTENCE_RE.sub(".", answer)
    answer = _TAI_DAY_PAREN_RE.sub("", answer)
    answer = _TAI_DAY_BEFORE_CLAUSE_RE.sub("", answer)
    answer = _TAI_DAY_NHAT_RE.sub("", answer)
    answer = _TAI_DAY_SEMICOLON_RE.sub("", answer)
    answer = _TAI_DAY_RE.sub("", answer)
    answer = _XEM_TAI_DAY_RE.sub("", answer)
    answer = _IMAGE_SOURCE_RE.sub("", answer)
    answer = _IMAGE_RE.sub("", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return clean_answer(answer)


def main():
    src = PROJECT / "data" / "official" / "train.json"
    dst = PROJECT / "data" / "official" / "train_sft.json"

    with src.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    seen_questions: dict[str, str] = {}
    kept: dict[str, dict[str, str]] = {}
    stats = {
        "total": len(raw),
        "truncated": 0,
        "too_short": 0,
        "duplicate": 0,
        "kept": 0,
    }

    for qid, item in raw.items():
        question = item["question"].strip()
        answer = preprocess_answer(item["answer"])

        if answer.rstrip().endswith("..."):
            stats["truncated"] += 1
            continue

        if len(answer) < 200:
            stats["too_short"] += 1
            continue

        if question in seen_questions:
            stats["duplicate"] += 1
            old_qid = seen_questions[question]
            if len(answer) > len(kept[old_qid]["answer"]):
                del kept[old_qid]
            else:
                continue

        seen_questions[question] = qid
        kept[qid] = {"question": question, "answer": answer}
        stats["kept"] += 1

    output = {qid: kept[qid] for qid in kept}
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
