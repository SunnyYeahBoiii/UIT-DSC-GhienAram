from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .text import citations, document_numbers, normalize_text, numbers, words


CAPTION_RE = re.compile(r"\s*[\[(]?\s*(?:hình|ảnh)\s+từ\s+internet\s*[\])]?[.!]?", re.I)
SOURCE_SLUG_RE = re.compile(
    r"(?i)(?:(?:https?://[^\s]*/)?\b(?:thong[-_ ]tu|nghi[-_ ]dinh|quyet[-_ ]dinh|"
    r"nghi[-_ ]quyet|luat|cong[-_ ]van)(?:[-_][a-z0-9đ]+){4,}\b)"
)
DANGLING_RE = re.compile(
    r"(?i)\b(?:bao gồm|gồm|như sau|được quy định như sau|cụ thể như sau|"
    r"đối với|của|và|hoặc|tại|theo)\s*[:;,\-–—.]?\s*$"
)

ENTITY_ANCHORS = (
    "học sinh", "sinh viên", "học viên", "người lao động", "người sử dụng lao động",
    "công chức", "viên chức", "người dự tuyển", "giáo viên", "trẻ em", "người bệnh",
    "bác sĩ", "doanh nghiệp", "hợp tác xã", "hộ gia đình", "người nộp thuế",
    "chủ đầu tư", "nhà thầu", "cơ sở giáo dục", "cơ sở khám bệnh", "phương tiện",
)

REQUIREMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("count_or_amount", r"\b(?:bao nhiêu|có mấy|số lượng|mức\s+(?:phạt|vay|cho vay)|tỷ lệ)\b"),
    ("condition", r"\b(?:khi nào|trường hợp nào|điều kiện|chỉ định|khi có)\b"),
    ("actor", r"\b(?:do ai|ai thực hiện|người thực hiện|cơ quan nào|đơn vị nào|thẩm quyền)\b"),
    ("components", r"\b(?:gồm những gì|bao gồm những gì|thành phần|hồ sơ gồm|có những)\b"),
    ("form", r"\b(?:mẫu|biểu mẫu|mẫu đơn|mẫu thông báo|mẫu quyết định)\b"),
    ("definition", r"\b(?:là gì|được hiểu là|khái niệm)\b"),
    ("procedure", r"\b(?:thủ tục|trình tự|cách thức|nộp ở đâu)\b"),
)

ANSWER_REQUIREMENT_PATTERNS = {
    "count_or_amount": r"(?:\d|không\s+(?:hạn chế|giới hạn)|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)",
    "condition": r"\b(?:khi|trường hợp|điều kiện|chỉ định|nếu|được áp dụng)\b",
    "actor": r"\b(?:người thực hiện|cơ quan|đơn vị|bộ|sở|ủy ban|uỷ ban|bác sĩ|kỹ thuật viên|"
             r"người có thẩm quyền|chủ tịch|giám đốc)\b",
    "components": r"(?:\bbao gồm\b|\bgồm\b|(?:^|[;:])\s*(?:a\)|b\)|c\)|\d+[.)]))",
    "form": r"\b(?:mẫu số|phụ lục|ban hành kèm theo|các trường thông tin|nội dung mẫu)\b",
    "definition": r"\b(?:là|được hiểu là)\b",
    "procedure": r"\b(?:bước|nộp|tiếp nhận|giải quyết|trình tự|thủ tục)\b",
    "scope_qualification": r"\b(?:đối với|theo|áp dụng|trong trường hợp|phụ thuộc|thuộc chương trình)\b",
}

QUALIFIER_PATTERNS = {
    "maximum": re.compile(r"(?i)\b(?:không vượt quá|tối đa|nhiều nhất|đến)\b"),
    "minimum": re.compile(r"(?i)\b(?:ít nhất|tối thiểu|không dưới|từ)\b"),
    "exact": re.compile(r"(?i)\b(?:bằng|là|đúng)\b"),
}


def question_requirements(question: str) -> list[str]:
    normalized = normalize_text(question).casefold()
    result = [name for name, pattern in REQUIREMENT_PATTERNS if re.search(pattern, normalized, re.I)]
    if (
        len(words(normalized)) <= 12
        and re.search(r"\b(?:hiện nay|về mức|đơn vị được|bao nhiêu)\b", normalized)
        and not question_anchors(normalized)
    ):
        result.append("scope_qualification")
    return list(dict.fromkeys(result))


def normalized_document_numbers(text: str) -> set[str]:
    result: set[str] = set()
    for value in document_numbers(text):
        number, separator, remainder = value.partition("/")
        result.add(f"{int(number)}/{remainder}" if separator else value)
    return result


def question_anchors(question: str) -> set[str]:
    normalized = normalize_text(question).casefold()
    return {anchor for anchor in ENTITY_ANCHORS if anchor in normalized}


def requirement_coverage(question: str, answer: str) -> tuple[float, list[str]]:
    required = question_requirements(question)
    if not required:
        return 1.0, []
    missing = [
        name for name in required
        if not re.search(ANSWER_REQUIREMENT_PATTERNS[name], answer, re.I | re.M)
    ]
    return (len(required) - len(missing)) / len(required), missing


def scope_coverage(question: str, text: str) -> tuple[float, list[str]]:
    anchors = question_anchors(question)
    if not anchors:
        return 1.0, []
    normalized = normalize_text(text).casefold()
    missing = sorted(anchor for anchor in anchors if anchor not in normalized)
    return (len(anchors) - len(missing)) / len(anchors), missing


def _strip_exact_question_edge(question: str, answer: str) -> tuple[str, int]:
    question_clean = normalize_text(question).strip()
    if not question_clean:
        return answer, 0
    removed = 0
    result = answer.strip()
    for edge in ("start", "end"):
        lowered = result.casefold()
        target = question_clean.casefold()
        if edge == "start" and lowered.startswith(target):
            result = result[len(question_clean):].lstrip(" \n\t:-–—")
            removed += 1
        elif edge == "end" and lowered.endswith(target):
            result = result[: len(result) - len(question_clean)].rstrip(" \n\t:-–—")
            removed += 1
    question_tokens = words(question_clean)
    if len(question_tokens) >= 5:
        token_pattern = r"\W+".join(re.escape(token) for token in question_tokens)
        start_match = re.match(rf"(?is)^\W*{token_pattern}\W*", result)
        if start_match:
            result = result[start_match.end() :].lstrip(" \n\t:-–—")
            removed += 1
        end_match = re.search(rf"(?is)\W*{token_pattern}\W*$", result)
        if end_match:
            result = result[: end_match.start()].rstrip(" \n\t:-–—")
            removed += 1
    return result, removed


def sanitize_answer(question: str, answer: str) -> tuple[str, dict[str, Any]]:
    """Remove only presentation artifacts that are safe to identify exactly."""
    original = normalize_text(answer)
    caption_count = len(CAPTION_RE.findall(original))
    cleaned = CAPTION_RE.sub(" ", original)
    slug_count = len(SOURCE_SLUG_RE.findall(cleaned))
    cleaned = SOURCE_SLUG_RE.sub(" ", cleaned)
    cleaned, edge_question_echoes = _strip_exact_question_edge(question, cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"(?:\s*[—–-]\s*){2,}", " — ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip(" \n\t,;:-–—")
    final = cleaned or original
    return final, {
        "caption_removed": caption_count,
        "source_slug_removed": slug_count,
        "edge_question_echo_removed": edge_question_echoes,
        "sanitization_applied": final != original,
    }


def tandem_repeat(text: str, minimum: int = 8, maximum: int = 40) -> dict[str, int]:
    tokens = words(text)
    best = {"block_tokens": 0, "repetitions": 0, "covered_tokens": 0}
    for block_size in range(min(maximum, len(tokens) // 3), minimum - 1, -1):
        for start in range(0, len(tokens) - block_size * 3 + 1):
            block = tokens[start : start + block_size]
            repeats = 1
            cursor = start + block_size
            while tokens[cursor : cursor + block_size] == block:
                repeats += 1
                cursor += block_size
            covered = block_size * repeats
            if repeats >= 3 and covered > best["covered_tokens"]:
                best = {
                    "block_tokens": block_size,
                    "repetitions": repeats,
                    "covered_tokens": covered,
                }
    return best


def repeated_ngram_ratio(text: str, size: int = 12) -> tuple[float, int]:
    tokens = words(text)
    if len(tokens) < size:
        return 0.0, 0
    grams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    counts = Counter(grams)
    repeated_occurrences = sum(count for count in counts.values() if count >= 3)
    return min(1.0, repeated_occurrences / max(1, len(grams))), max(counts.values(), default=0)


def _number_qualifiers(text: str) -> dict[str, set[str]]:
    normalized = normalize_text(text)
    result: dict[str, set[str]] = {}
    for match in re.finditer(r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|đồng|triệu|tỷ|ngày|tháng|năm|giờ|tuổi))?\b", normalized, re.I):
        number = re.sub(r"\s+", "", match.group(0).casefold())
        prefix = normalized[max(0, match.start() - 48) : match.start()]
        qualifiers = {name for name, pattern in QUALIFIER_PATTERNS.items() if pattern.search(prefix)}
        result.setdefault(number, set()).update(qualifiers)
    return result


def _number_qualifier_occurrences(text: str) -> dict[str, list[set[str]]]:
    normalized = normalize_text(text)
    result: dict[str, list[set[str]]] = {}
    for match in re.finditer(r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|đồng|triệu|tỷ|ngày|tháng|năm|giờ|tuổi))?\b", normalized, re.I):
        number = re.sub(r"\s+", "", match.group(0).casefold())
        prefix = normalized[max(0, match.start() - 48) : match.start()]
        qualifiers = {name for name, pattern in QUALIFIER_PATTERNS.items() if pattern.search(prefix)}
        result.setdefault(number, []).append(qualifiers)
    return result


def relational_numeric_conflicts(answer: str, evidence_text: str) -> list[dict[str, Any]]:
    answer_occurrences = _number_qualifier_occurrences(answer)
    evidence_items = _number_qualifiers(evidence_text)
    conflicts: list[dict[str, Any]] = []
    for number, occurrences in answer_occurrences.items():
        evidence_qualifiers = evidence_items.get(number, set())
        if not evidence_qualifiers:
            continue
        for answer_qualifiers in occurrences:
            loses_maximum = "maximum" in evidence_qualifiers and "maximum" not in answer_qualifiers
            loses_minimum = "minimum" in evidence_qualifiers and "minimum" not in answer_qualifiers
            reverses = (
                ("maximum" in evidence_qualifiers and "minimum" in answer_qualifiers)
                or ("minimum" in evidence_qualifiers and "maximum" in answer_qualifiers)
            )
            if loses_maximum or loses_minimum or reverses:
                conflicts.append(
                    {
                        "number": number,
                        "answer": sorted(answer_qualifiers),
                        "evidence": sorted(evidence_qualifiers),
                    }
                )
                break
    return conflicts


def analyze_answer_quality(
    question: str,
    answer: str,
    evidence_text: str,
    generation_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = generation_details or {}
    tandem = tandem_repeat(answer)
    ngram_ratio, ngram_max = repeated_ngram_ratio(answer)
    degenerate_repeat = bool(
        tandem["repetitions"] >= 3
        or (len(words(answer)) >= 160 and ngram_max >= 6 and ngram_ratio >= 0.28)
    )
    coverage, missing_requirements = requirement_coverage(question, answer)
    answer_scope, missing_answer_anchors = scope_coverage(question, answer)
    evidence_scope, missing_evidence_anchors = scope_coverage(question, evidence_text)
    answer_documents = normalized_document_numbers(answer)
    evidence_documents = normalized_document_numbers(evidence_text)
    unsupported_documents = sorted(answer_documents - evidence_documents)
    answer_citations = citations(answer)
    evidence_citations = citations(evidence_text)
    unsupported_citations = sorted(answer_citations - evidence_citations)
    answer_numbers = numbers(answer)
    evidence_numbers = numbers(evidence_text)
    unsupported_numbers = sorted(answer_numbers - evidence_numbers)
    numeric_conflicts = relational_numeric_conflicts(answer, evidence_text)
    dangling = bool(DANGLING_RE.search(answer))
    truncated = bool(details.get("truncated", False))
    severe_reasons: list[str] = []
    if degenerate_repeat:
        severe_reasons.append("degenerate_repeat")
    if dangling:
        severe_reasons.append("dangling_ending")
    if unsupported_documents:
        severe_reasons.append("unsupported_document_citation")
    if numeric_conflicts:
        severe_reasons.append("numeric_relation_conflict")
    if truncated and missing_requirements:
        severe_reasons.append("truncated_missing_requirement")
    if missing_requirements and len(question_requirements(question)) >= 2:
        severe_reasons.append("multi_part_missing_requirement")
    if answer_scope < 1.0 or evidence_scope < 1.0:
        severe_reasons.append("scope_anchor_missing")
    return {
        "degenerate_repeat": degenerate_repeat,
        "tandem_repeat": tandem,
        "repeated_ngram_ratio": ngram_ratio,
        "repeated_ngram_max_count": ngram_max,
        "dangling_ending": dangling,
        "intent_coverage": coverage,
        "missing_requirements": missing_requirements,
        "answer_scope_coverage": answer_scope,
        "evidence_scope_coverage": evidence_scope,
        "missing_answer_anchors": missing_answer_anchors,
        "missing_evidence_anchors": missing_evidence_anchors,
        "unsupported_document_citations": unsupported_documents,
        "unsupported_article_citations": unsupported_citations,
        "unsupported_numbers": unsupported_numbers,
        "numeric_relation_conflicts": numeric_conflicts,
        "repair_reasons": list(dict.fromkeys(severe_reasons)),
        "repair_recommended": bool(severe_reasons),
    }
