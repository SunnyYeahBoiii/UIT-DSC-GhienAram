from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
INTERNET_VARIANT = r"(?:internet|inernet|internert|internnet)"
IMAGE_MARKER_RE = re.compile(
    rf"[\[(]?\s*(?:nguồn\s+)?(?:hình\s+ảnh|hình|ảnh)"
    rf"(?:\s+(?:từ|trên))?\s*:?\s*{INTERNET_VARIANT}\s*[\])]?",
    flags=re.IGNORECASE,
)
IMAGE_MARKER_AT_END_RE = re.compile(
    rf"[\[(]?\s*(?:nguồn\s+)?(?:hình\s+ảnh|hình|ảnh)"
    rf"(?:\s+(?:từ|trên))?\s*:?\s*{INTERNET_VARIANT}\s*[\])]?[\s.!?]*$",
    flags=re.IGNORECASE,
)
IMAGE_MARKER_BEFORE_DOWNLOAD_CTA_RE = re.compile(
    rf"[\[(]?\s*(?:nguồn\s+)?(?:hình\s+ảnh|hình|ảnh)"
    rf"(?:\s+(?:từ|trên))?\s*:?\s*{INTERNET_VARIANT}\s*[\])]?[\s.!?]*"
    r"(?:tải\s+trọn\s+bộ|tải\s+về|xem\s+chi\s+tiết)\b[\s\S]*$",
    flags=re.IGNORECASE,
)
BOUNDARY_PUNCTUATION_RE = re.compile(r"^[\s?!.,:;…\-–—\"“”'‘’()\[\]{}]*")
DOWNLOAD_CTA_RE = re.compile(
    r"\b(?:tải\s+(?:trọn\s+bộ|về|tại\s+đây)|download)\b",
    flags=re.IGNORECASE,
)
CTA_LINE_START_RE = re.compile(
    r"^\s*(?:[*\-–—>]\s*)?(?:tải\b|xem\s+(?:chi\s+tiết|toàn\s+văn)\b|"
    r"tham\s+khảo\s+thêm\b)[\s\S]*"
    r"(?:tải\s+về|tại\s+(?:đây|đâu)|download)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)
CTA_ANY_START_RE = re.compile(
    r"^\s*(?:[*\-–—>]\s*)?tải\s+(?:mẫu|biểu\s+mẫu|tờ\s+khai|danh\s+mục|"
    r"phụ\s+lục|văn\s+bản|quy\s+định|giấy|báo\s+cáo)\b.*$",
    flags=re.IGNORECASE,
)
TRAILING_CTA_RE = re.compile(
    r"\s*(?::|\-|–|—)?\s*(?:tải\s+về|tại\s+đây|download)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)
WEB_RELATED_LINE_RE = re.compile(
    r"^\s*(?:(?:xem\s+thêm|xem\s+chi\s+tiết)\s*:?.*|>{2,}\s*đã\s+có\b.*)$",
    flags=re.IGNORECASE,
)
TABLE_MARKER_RE = re.compile(r"\+\+table", flags=re.IGNORECASE)
SERVICE_BOILERPLATE_RE = re.compile(
    r"(?:\n\s*\n|\n)?\s*trên\s+đây\s+là\s+(?:"
    r"(?:một\s+số\s+)?thông\s+tin[^\n]{0,260}?"
    r"|[^\n]{0,500}?(?:chúng\s+tôi\s+cung\s+cấp|gửi\s+tới\s+bạn)[^\n]{0,180}?"
    r")(?:trân\s+trọng[.!]?)?\s*$",
    flags=re.IGNORECASE,
)
INCOMPLETE_TAIL_RE = re.compile(
    r"(?:theo\s+đó\s+)?(?:quy\s+định|cụ\s+thể)?\s*(?:như\s+sau)\s*[:.]?\s*$",
    flags=re.IGNORECASE,
)


def token_value(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def normalize_unicode(text: str) -> tuple[str, dict[str, int]]:
    """Normalize harmless web/conversion characters without changing wording."""
    counts = {
        "nbsp": text.count("\u00a0"),
        "bom": text.count("\ufeff"),
        "soft_hyphen": text.count("\u00ad"),
        "non_nfc": int(not unicodedata.is_normalized("NFC", text)),
    }
    normalized = text.replace("\u00a0", " ")
    normalized = normalized.replace("\ufeff", "").replace("\u00ad", "")
    return unicodedata.normalize("NFC", normalized), counts


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (token_value(match.group()), match.start(), match.end())
        for match in WORD_RE.finditer(text)
    ]


def records_from_json(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [
            (str(question_id), row)
            for question_id, row in payload.items()
            if isinstance(row, dict)
        ]
    if isinstance(payload, list):
        rows = []
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            rows.append((str(row.get("question_id", row.get("id", index))), row))
        return rows
    raise ValueError("Dữ liệu đầu vào phải là JSON Object hoặc JSON Array.")


def find_question_occurrences(question: str, answer: str) -> list[tuple[int, int]]:
    question_tokens = [value for value, _, _ in token_spans(question)]
    answer_tokens = token_spans(answer)
    if not question_tokens or len(question_tokens) > len(answer_tokens):
        return []

    size = len(question_tokens)
    occurrences = []
    for start in range(0, len(answer_tokens) - size + 1):
        if [value for value, _, _ in answer_tokens[start : start + size]] != question_tokens:
            continue
        occurrences.append((answer_tokens[start][1], answer_tokens[start + size - 1][2]))
    return occurrences


def remove_image_markers(text: str) -> tuple[str, int]:
    cleaned, count = IMAGE_MARKER_RE.subn("", text)
    return cleaned, count


def remove_final_block(text: str, *, max_tokens: int = 80) -> tuple[str, str | None]:
    """Remove one final paragraph/line, guarded against deleting long legal prose."""
    prefix = text.rstrip()
    paragraph_boundaries = list(re.finditer(r"\n\s*\n", prefix))
    block_start = (
        paragraph_boundaries[-1].end()
        if paragraph_boundaries
        else prefix.rfind("\n") + 1
    )
    candidate = prefix[block_start:].strip()
    if (
        not candidate
        or len(token_spans(candidate)) > max_tokens
        or not has_words(prefix[:block_start])
    ):
        return prefix, None
    return prefix[:block_start].rstrip(), candidate


def remove_trailing_question_block(text: str) -> tuple[str, str | None]:
    """Remove the final paragraph/line when it is a scraped question block."""
    prefix = text.rstrip()
    if not prefix.endswith("?"):
        return prefix, None

    paragraph_boundaries = list(re.finditer(r"\n\s*\n", prefix))
    block_start = (
        paragraph_boundaries[-1].end()
        if paragraph_boundaries
        else prefix.rfind("\n") + 1
    )
    candidate = prefix[block_start:].strip()
    if not candidate or not candidate.endswith("?") or "?" not in candidate:
        return prefix, None
    return prefix[:block_start].rstrip(), candidate


def remove_trailing_image_junk(text: str) -> tuple[str, int, str | None]:
    """Remove a terminal image marker together with its question/caption block."""
    marker_with_cta = IMAGE_MARKER_BEFORE_DOWNLOAD_CTA_RE.search(text)
    if marker_with_cta is not None:
        prefix = text[: marker_with_cta.start()].rstrip()
        without_block, removed_block = remove_final_block(prefix)
        if removed_block is not None:
            return without_block, 1, removed_block

    marker = IMAGE_MARKER_AT_END_RE.search(text)
    if marker is None:
        return text, 0, None

    prefix = text[: marker.start()].rstrip()
    without_block, removed_block = remove_final_block(prefix)
    return (without_block if removed_block is not None else prefix), 1, removed_block


def remove_download_ctas(text: str) -> tuple[str, int]:
    """Remove web download prompts while preserving surrounding legal steps."""
    cleaned_lines: list[str] = []
    removed = 0
    for line in text.splitlines():
        if CTA_ANY_START_RE.fullmatch(line):
            removed += 1
            continue
        if not DOWNLOAD_CTA_RE.search(line) and not CTA_LINE_START_RE.fullmatch(line):
            cleaned_lines.append(line)
            continue
        if CTA_LINE_START_RE.fullmatch(line):
            removed += 1
            continue
        stripped, count = TRAILING_CTA_RE.subn("", line)
        if count:
            removed += count
            if stripped.strip():
                cleaned_lines.append(stripped.rstrip())
            continue
        stripped, count = DOWNLOAD_CTA_RE.subn("", line)
        removed += count
        if stripped.strip():
            cleaned_lines.append(stripped.rstrip())
    return "\n".join(cleaned_lines), removed


def remove_terminal_web_related_line(text: str) -> tuple[str, str | None]:
    prefix = text.rstrip()
    start = prefix.rfind("\n") + 1
    candidate = prefix[start:].strip()
    if WEB_RELATED_LINE_RE.fullmatch(candidate):
        return prefix[:start].rstrip(), candidate
    return prefix, None


def remove_short_dangling_title(text: str) -> tuple[str, str | None]:
    """Remove short, blank-line-separated image captions left by the scraper."""
    prefix = text.rstrip()
    boundaries = list(re.finditer(r"\n\s*\n", prefix))
    if not boundaries:
        return prefix, None
    start = boundaries[-1].end()
    candidate = prefix[start:].strip()
    remaining = prefix[: boundaries[-1].start()].rstrip()
    tokens = token_spans(candidate)
    if (
        not 1 <= len(tokens) <= 12
        or "\n" in candidate
        or re.search(r"[.!?;:]$", candidate)
        or re.search(r"\d", candidate)
        or len(token_spans(remaining)) < 30
    ):
        return prefix, None
    return remaining, candidate


def remove_consecutive_duplicate_lines(text: str) -> tuple[str, int]:
    result: list[str] = []
    removed = 0
    previous_normalized = ""
    for line in text.splitlines():
        normalized = " ".join(value for value, _, _ in token_spans(line))
        if (
            normalized
            and normalized == previous_normalized
            and len(token_spans(line)) >= 8
        ):
            removed += 1
            continue
        result.append(line)
        if normalized:
            previous_normalized = normalized
    return "\n".join(result), removed


def has_words(text: str) -> bool:
    return WORD_RE.search(text) is not None


def boundary_kind(answer: str, start: int, end: int) -> set[str]:
    kinds: set[str] = set()
    if not has_words(answer[:start]):
        kinds.add("start")

    suffix_without_marker, _ = remove_image_markers(answer[end:])
    if not has_words(suffix_without_marker):
        kinds.add("end")
    return kinds


def extend_right_over_boundary_punctuation(text: str, end: int) -> int:
    match = BOUNDARY_PUNCTUATION_RE.match(text[end:])
    return end + (match.end() if match else 0)


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    result = text
    for start, end in sorted(spans, reverse=True):
        result = result[:start] + result[end:]
    return result


def tidy_answer(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[\s:;,\-–—]+", "", text)
    text = re.sub(r"[\s:;,\-–—]+$", "", text)
    return text.strip()


def classify_and_clean(
    question: str,
    answer: str,
    min_question_tokens: int,
    min_question_chars: int,
) -> tuple[str, str, dict[str, Any]]:
    question_token_count = len(token_spans(question))
    normalized_question_length = len(" ".join(value for value, _, _ in token_spans(question)))
    working_answer, unicode_counts = normalize_unicode(answer)
    marker_count = len(IMAGE_MARKER_RE.findall(working_answer))
    working_answer, end_marker_removed, removed_image_block = remove_trailing_image_junk(
        working_answer
    )
    working_answer, other_markers_removed = remove_image_markers(working_answer)
    working_answer, table_markers_removed = TABLE_MARKER_RE.subn("", working_answer)
    working_answer, download_ctas_removed = remove_download_ctas(working_answer)
    working_answer, web_related_line_removed = remove_terminal_web_related_line(
        working_answer
    )
    working_answer, service_boilerplate_removed = SERVICE_BOILERPLATE_RE.subn(
        "", working_answer
    )
    working_answer, duplicate_lines_removed = remove_consecutive_duplicate_lines(
        working_answer
    )
    working_answer = tidy_answer(working_answer)

    removed_question_block = (
        removed_image_block
        if removed_image_block and removed_image_block.rstrip().endswith("?")
        else None
    )
    removed_image_caption = (
        removed_image_block
        if removed_image_block and not removed_image_block.rstrip().endswith("?")
        else None
    )

    # A final question paragraph is navigation/related-content residue on the
    # official scraped pages. It is safe to remove regardless of whether it
    # exactly repeats the source question.
    removed_question_blocks = [removed_question_block] if removed_question_block else []
    for _ in range(5):
        without_tail, candidate_question_block = remove_trailing_question_block(
            working_answer
        )
        if not candidate_question_block or not has_words(without_tail):
            break
        working_answer = without_tail
        removed_question_blocks.append(candidate_question_block)
    if removed_question_blocks:
        removed_question_block = "\n".join(removed_question_blocks)

    # Removing one or more related questions can expose a standalone
    # "Xem thêm:" navigation label immediately before them.
    working_answer, newly_removed_web_line = remove_terminal_web_related_line(
        working_answer
    )
    if newly_removed_web_line:
        web_related_line_removed = (
            "\n".join(
                value
                for value in (web_related_line_removed, newly_removed_web_line)
                if value
            )
        )

    working_answer, dangling_title_removed = remove_short_dangling_title(
        working_answer
    )
    working_answer = tidy_answer(working_answer)
    removed_markers = end_marker_removed + other_markers_removed
    tail_details = {
        "removed_image_markers": removed_markers,
        "removed_trailing_question_block": removed_question_block,
        "removed_image_caption": removed_image_caption,
        "removed_dangling_title": dangling_title_removed,
        "removed_download_ctas": download_ctas_removed,
        "removed_table_markers": table_markers_removed,
        "removed_duplicate_lines": duplicate_lines_removed,
        "removed_web_related_line": web_related_line_removed,
        "removed_service_boilerplate": service_boilerplate_removed,
        "unicode_normalization": unicode_counts,
    }

    if (
        question_token_count < min_question_tokens
        or normalized_question_length < min_question_chars
    ):
        return "clean", working_answer, {
            "question_occurrences": 0,
            **tail_details,
            "reason": "question_below_detection_threshold",
        }

    occurrences = find_question_occurrences(question, working_answer)
    if not occurrences:
        return "clean", working_answer, {
            "question_occurrences": 0,
            **tail_details,
            "reason": "no_exact_repeat",
        }

    removable: list[tuple[int, int]] = []
    positions: list[list[str]] = []
    ambiguous = False
    for start, end in occurrences:
        kinds = boundary_kind(working_answer, start, end)
        positions.append(sorted(kinds) if kinds else ["middle"])
        if not kinds:
            ambiguous = True
            continue
        removable.append(
            (start, extend_right_over_boundary_punctuation(working_answer, end))
        )

    if ambiguous:
        # User-reviewed rule: rows scraped with an Internet-image marker at the
        # very end are accepted after deleting that marker and the complete
        # trailing question block. An occurrence inside the substantive legal
        # explanation is retained because it may be a legitimate restatement.
        if end_marker_removed:
            cleaned = remove_spans(working_answer, removable)
            cleaned = tidy_answer(cleaned)
            if cleaned and has_words(cleaned):
                return "clean", cleaned, {
                    "question_occurrences": len(occurrences),
                    "positions": positions,
                    "removed_question_occurrences": len(removable),
                    **tail_details,
                    "reason": "end_image_marker_rule_applied",
                }
        # Second user-reviewed rule: among unresolved rows, a final paragraph
        # ending in '?' is scraped related-question text. Remove the whole
        # block and accept the remaining substantive legal answer.
        without_tail_questions, review_question_block = remove_trailing_question_block(
            working_answer
        )
        if review_question_block and without_tail_questions and has_words(without_tail_questions):
            return "clean", tidy_answer(without_tail_questions), {
                "question_occurrences": len(occurrences),
                "positions": positions,
                "removed_question_occurrences": len(removable),
                "removed_image_markers": removed_markers,
                "removed_trailing_question_block": review_question_block,
                "reason": "review_tail_question_block_removed",
            }
        return "review", working_answer, {
            "question_occurrences": len(occurrences),
            "positions": positions,
            "image_markers": marker_count,
            **tail_details,
            "reason": "repeat_in_middle_or_mixed_positions",
        }

    cleaned = remove_spans(working_answer, removable)
    cleaned = tidy_answer(cleaned)
    if not cleaned or not has_words(cleaned):
        return "review", answer, {
            "question_occurrences": len(occurrences),
            "positions": positions,
            "image_markers": marker_count,
            **tail_details,
            "reason": "cleaning_would_leave_empty_answer",
        }

    return "clean", cleaned, {
        "question_occurrences": len(occurrences),
        "positions": positions,
        "removed_question_occurrences": len(removable),
        **tail_details,
        "reason": "boundary_repeat_removed",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Xóa câu hỏi lặp ở đầu/cuối đáp án và marker '(Hình từ Internet)'; "
            "đưa trường hợp lặp ở giữa sang JSON review."
        )
    )
    parser.add_argument("--input", required=True, help="train.json gốc")
    parser.add_argument("--clean-output", required=True, help="train_clean.json")
    parser.add_argument("--review-output", required=True, help="JSON cần kiểm tra tiếp")
    parser.add_argument("--report-output", default=None, help="Báo cáo thống kê tùy chọn")
    parser.add_argument(
        "--quality-output",
        default=None,
        help="JSON chứa ID loại khỏi QLoRA và nhóm câu hỏi trùng",
    )
    parser.add_argument(
        "--qlora-output",
        default=None,
        help="Tập train QLoRA đã loại các answer bị mất bảng/biểu mẫu",
    )
    parser.add_argument(
        "--include-review-in-clean",
        action="store_true",
        help="Gộp nguyên trạng toàn bộ mẫu review còn lại vào train_clean",
    )
    parser.add_argument("--min-question-tokens", type=int, default=6)
    parser.add_argument("--min-question-chars", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    records = records_from_json(payload)

    clean: dict[str, Any] = {}
    review: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    auto_cleaned_ids: list[str] = []
    marker_cleaned_ids: list[str] = []
    trailing_question_cleaned_ids: list[str] = []

    for question_id, row in records:
        question = row.get("question", row.get("query"))
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            review[question_id] = {
                **row,
                "_repeat_review": {"reason": "missing_string_question_or_answer"},
            }
            continue

        status, cleaned_answer, details = classify_and_clean(
            question,
            answer,
            min_question_tokens=args.min_question_tokens,
            min_question_chars=args.min_question_chars,
        )
        audit[question_id] = details
        if status == "review":
            review[question_id] = {
                **row,
                "answer": cleaned_answer,
                "_repeat_review": details,
            }
            continue

        clean[question_id] = {**row, "answer": cleaned_answer}
        if details.get("removed_question_occurrences", 0):
            auto_cleaned_ids.append(question_id)
        if details.get("removed_image_markers", 0):
            marker_cleaned_ids.append(question_id)
        if details.get("removed_trailing_question_block"):
            trailing_question_cleaned_ids.append(question_id)

    forced_review_ids: list[str] = []
    if args.include_review_in_clean:
        forced_review_ids = list(review)
        for question_id, row in review.items():
            clean[question_id] = {
                key: value for key, value in row.items() if key != "_repeat_review"
            }
        review = {}

    if set(clean) & set(review):
        raise AssertionError("ID xuất hiện đồng thời trong clean và review.")
    if len(clean) + len(review) != len(records):
        raise AssertionError("Tổng clean + review không bằng dữ liệu đầu vào.")

    def ids_with_detail(key: str) -> list[str]:
        return [question_id for question_id, details in audit.items() if details.get(key)]

    caption_cleaned_ids = ids_with_detail("removed_image_caption")
    dangling_title_cleaned_ids = ids_with_detail("removed_dangling_title")
    download_cta_cleaned_ids = ids_with_detail("removed_download_ctas")
    table_marker_cleaned_ids = ids_with_detail("removed_table_markers")
    duplicate_line_cleaned_ids = ids_with_detail("removed_duplicate_lines")
    web_related_cleaned_ids = ids_with_detail("removed_web_related_line")
    service_boilerplate_cleaned_ids = ids_with_detail("removed_service_boilerplate")
    unicode_cleaned_ids = [
        question_id
        for question_id, details in audit.items()
        if any(details.get("unicode_normalization", {}).values())
    ]

    quality_rows = {**clean, **review}
    incomplete_reasons: dict[str, list[str]] = {}
    long_answer_ids: list[str] = []
    normalized_questions: dict[str, list[str]] = defaultdict(list)
    for question_id, row in quality_rows.items():
        question = str(row.get("question", row.get("query", "")))
        answer = str(row.get("answer", ""))
        answer_tokens = len(token_spans(answer))
        normalized_question = " ".join(
            value for value, _, _ in token_spans(question)
        )
        normalized_questions[normalized_question].append(question_id)
        if answer_tokens < 60 and INCOMPLETE_TAIL_RE.search(answer):
            incomplete_reasons[question_id] = [
                "answer_lost_table_or_form_content_after_scraping"
            ]
        if answer_tokens > 1200:
            long_answer_ids.append(question_id)

    duplicate_question_groups = [
        ids for ids in normalized_questions.values() if len(ids) > 1
    ]
    quality_report = {
        "schema": "legalqa-training-quality-v1",
        "qlora_excluded_ids": sorted(incomplete_reasons),
        "exclusion_reasons": incomplete_reasons,
        "duplicate_question_groups": duplicate_question_groups,
        "split_rule": "keep_every_duplicate_question_group_in_one_split",
        "long_answer_ids_over_1200_words": long_answer_ids,
        "long_answer_rule": "retain_content_but_use_truncation_aware_training",
    }

    report = {
        "input": str(input_path.resolve()),
        "total": len(records),
        "train_clean": len(clean),
        "needs_review": len(review),
        "auto_cleaned_boundary_repeat": len(auto_cleaned_ids),
        "removed_image_marker": len(marker_cleaned_ids),
        "removed_trailing_question_block": len(trailing_question_cleaned_ids),
        "removed_image_caption": len(caption_cleaned_ids),
        "removed_dangling_title": len(dangling_title_cleaned_ids),
        "removed_download_cta": len(download_cta_cleaned_ids),
        "removed_table_marker": len(table_marker_cleaned_ids),
        "removed_duplicate_line": len(duplicate_line_cleaned_ids),
        "removed_web_related_line": len(web_related_cleaned_ids),
        "removed_service_boilerplate": len(service_boilerplate_cleaned_ids),
        "unicode_normalized": len(unicode_cleaned_ids),
        "forced_review_into_clean": len(forced_review_ids),
        "auto_cleaned_ids": auto_cleaned_ids,
        "marker_cleaned_ids": marker_cleaned_ids,
        "trailing_question_cleaned_ids": trailing_question_cleaned_ids,
        "caption_cleaned_ids": caption_cleaned_ids,
        "dangling_title_cleaned_ids": dangling_title_cleaned_ids,
        "download_cta_cleaned_ids": download_cta_cleaned_ids,
        "table_marker_cleaned_ids": table_marker_cleaned_ids,
        "duplicate_line_cleaned_ids": duplicate_line_cleaned_ids,
        "web_related_cleaned_ids": web_related_cleaned_ids,
        "service_boilerplate_cleaned_ids": service_boilerplate_cleaned_ids,
        "unicode_cleaned_ids": unicode_cleaned_ids,
        "forced_review_ids": forced_review_ids,
        "review_ids": list(review),
        "integrity": {
            "clean_review_overlap": 0,
            "accounted_for": len(clean) + len(review),
        },
    }

    write_json(Path(args.clean_output), clean)
    write_json(Path(args.review_output), review)
    if args.report_output:
        write_json(Path(args.report_output), {**report, "audit": audit})
    if args.quality_output:
        write_json(Path(args.quality_output), quality_report)
    if args.qlora_output:
        qlora_rows = {
            question_id: row
            for question_id, row in clean.items()
            if question_id not in incomplete_reasons
        }
        write_json(Path(args.qlora_output), qlora_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
