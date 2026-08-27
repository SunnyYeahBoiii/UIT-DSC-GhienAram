from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|$)")


def normalized_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    return WORD_RE.findall(text)


def normalized_text(text: str) -> str:
    return " ".join(normalized_words(text))


def records_from_json(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        rows = []
        for question_id, value in payload.items():
            if not isinstance(value, dict):
                continue
            rows.append((str(question_id), value))
        return rows

    if isinstance(payload, list):
        rows = []
        for index, value in enumerate(payload):
            if not isinstance(value, dict):
                continue
            question_id = value.get("question_id", value.get("id", index))
            rows.append((str(question_id), value))
        return rows

    raise ValueError("Dữ liệu đầu vào phải là JSON Object hoặc JSON Array.")


def repeated_question_match(
    question: str,
    answer: str,
    min_question_tokens: int,
    min_question_chars: int,
) -> dict[str, Any] | None:
    question_tokens = normalized_words(question)
    normalized_question = " ".join(question_tokens)
    if (
        len(question_tokens) < min_question_tokens
        or len(normalized_question) < min_question_chars
    ):
        return None

    normalized_answer = normalized_text(answer)
    if normalized_question not in normalized_answer:
        return None

    # Tìm câu gốc trong answer để báo cáo dễ kiểm tra thủ công.
    matched_text = ""
    for sentence in SENTENCE_RE.findall(answer):
        if normalized_question in normalized_text(sentence):
            matched_text = sentence.strip()
            break

    return {
        "match_type": "exact_question_after_normalization",
        "question_tokens": len(question_tokens),
        "matched_text": matched_text or question,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xuất các mẫu có nguyên câu hỏi bị lặp trong đáp án."
    )
    parser.add_argument("--input", required=True, help="Đường dẫn train.json")
    parser.add_argument(
        "--output",
        required=True,
        help="JSON chứa các mẫu nghi ngờ và thông tin match",
    )
    parser.add_argument(
        "--remaining-output",
        default=None,
        help="Tùy chọn: JSON Object chứa các mẫu không bị gắn cờ",
    )
    parser.add_argument("--min-question-tokens", type=int, default=6)
    parser.add_argument("--min-question-chars", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    records = records_from_json(payload)

    suspicious: dict[str, Any] = {}
    remaining: dict[str, Any] = {}
    for question_id, row in records:
        question = row.get("question", row.get("query"))
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            remaining[question_id] = row
            continue

        detection = repeated_question_match(
            question,
            answer,
            min_question_tokens=args.min_question_tokens,
            min_question_chars=args.min_question_chars,
        )
        if detection is None:
            remaining[question_id] = row
            continue

        suspicious[question_id] = {
            **row,
            "_repeat_detection": detection,
        }

    write_json(Path(args.output), suspicious)
    if args.remaining_output:
        write_json(Path(args.remaining_output), remaining)

    print(
        json.dumps(
            {
                "input": str(input_path.resolve()),
                "total": len(records),
                "repeated_question": len(suspicious),
                "remaining": len(remaining),
                "output": str(Path(args.output).resolve()),
                "remaining_output": (
                    str(Path(args.remaining_output).resolve())
                    if args.remaining_output
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
