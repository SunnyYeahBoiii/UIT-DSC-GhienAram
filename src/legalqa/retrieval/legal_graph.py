from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..core.text import document_numbers, normalize_text


RELATION_PRIORITY = {
    "REPLACES": 1.00,
    "REPEALS": 0.95,
    "AMENDS": 0.90,
    "GUIDES": 0.72,
    "REFERENCES": 0.45,
}

CURRENT_LAW_RE = re.compile(
    r"\b(?:hiện hành|hiện nay|mới nhất|còn hiệu lực|đang có hiệu lực|"
    r"thay thế|bãi bỏ|sửa đổi|bổ sung)\b",
    re.IGNORECASE,
)
COMPLEX_CUE_RE = re.compile(
    r"\b(?:thay thế|bãi bỏ|hết hiệu lực|sửa đổi|bổ sung|dẫn chiếu|"
    r"văn bản khác|các văn bản|đồng thời|trong từng trường hợp|ngoại lệ)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RelationMention:
    target_alias: str
    relation_type: str
    confidence: float
    evidence: str
    reverse: bool = False


@dataclass(frozen=True)
class RelationEdge:
    source_doc_id: str
    target_doc_id: str
    relation_type: str
    confidence: float
    evidence: str
    direction: str = "outgoing"

    @property
    def neighbor_doc_id(self) -> str:
        return self.target_doc_id if self.direction == "outgoing" else self.source_doc_id


def document_aliases(title: str, text: str) -> set[str]:
    """Extract stable document-number aliases from title and the document lead."""
    title_aliases = document_numbers(title)
    if title_aliases:
        return title_aliases
    # Do not scan the whole legal preamble: it normally cites many other laws.
    # A document's own identifier, when absent from the title, is usually next
    # to the header label "Số" in the first page.
    lead = normalize_text(text[:1200])
    header_matches = re.findall(
        r"(?i:\bsố\s*[:.]?\s*)(\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZÀ-ỸĐ0-9-]{2,40})",
        lead,
    )
    return set().union(*(document_numbers(value) for value in header_matches)) if header_matches else set()


def _relation_type(context: str) -> tuple[str, float, bool]:
    lowered = context.lower()
    if re.search(r"được\s+(?:sửa đổi|bổ sung)\s+bởi", lowered):
        return "AMENDS", 0.96, True
    if re.search(r"được\s+thay thế\s+bởi", lowered):
        return "REPLACES", 0.99, True
    if re.search(r"(?:thay thế|thay cho)", lowered):
        return "REPLACES", 0.98, False
    if re.search(r"(?:bãi bỏ|hủy bỏ)", lowered):
        return "REPEALS", 0.97, False
    if re.search(r"(?:sửa đổi|bổ sung)", lowered):
        return "AMENDS", 0.93, False
    if re.search(r"(?:hướng dẫn(?:\s+thi hành)?|quy định chi tiết)", lowered):
        return "GUIDES", 0.82, False
    return "REFERENCES", 0.58, False


def extract_relation_mentions(text: str, own_aliases: set[str] | None = None) -> list[RelationMention]:
    """Extract typed cross-document mentions using only the official corpus.

    The parser is intentionally conservative: document-number matches are exact,
    relations are inferred only from a short local window, and duplicate edges
    keep the strongest confidence.
    """
    normalized = normalize_text(text)
    searchable = "".join(
        character
        for character in unicodedata.normalize("NFD", normalized.replace("Đ", "D").replace("đ", "d"))
        if unicodedata.category(character) != "Mn"
    )
    own = own_aliases or set()
    mentions: dict[tuple[str, str, bool], RelationMention] = {}
    for alias in document_numbers(normalized):
        if alias in own:
            continue
        variants = (
            re.escape(alias),
            re.escape(alias.replace("/", "-")),
            re.escape(alias.replace("/", " ")),
        )
        for variant in variants:
            for match in re.finditer(variant, searchable, flags=re.IGNORECASE):
                start = max(0, match.start() - 180)
                end = min(len(normalized), match.end() + 180)
                context = normalized[start:end]
                relation_type, confidence, reverse = _relation_type(context)
                key = (alias, relation_type, reverse)
                candidate = RelationMention(alias, relation_type, confidence, context, reverse)
                previous = mentions.get(key)
                if previous is None or candidate.confidence > previous.confidence:
                    mentions[key] = candidate
    return sorted(
        mentions.values(),
        key=lambda item: (-RELATION_PRIORITY[item.relation_type], item.target_alias, item.reverse),
    )


def asks_for_current_law(question: str) -> bool:
    return bool(CURRENT_LAW_RE.search(normalize_text(question)))


def has_complex_cue(question: str) -> bool:
    return bool(COMPLEX_CUE_RE.search(normalize_text(question)))


def temporal_relation_adjustment(edge: RelationEdge, question: str) -> float:
    """Score a graph neighbor with explicit direction and temporal semantics."""
    current = asks_for_current_law(question)
    relation = edge.relation_type
    incoming = edge.direction == "incoming"
    if relation == "REPLACES":
        # A --REPLACES--> B: incoming neighbor A is the newer document.
        if incoming:
            return 0.34 if current else 0.23
        return -0.28 if current else 0.08
    if relation == "REPEALS":
        if incoming:
            return 0.30 if current else 0.17
        return -0.35 if current else 0.04
    if relation == "AMENDS":
        return 0.25 if incoming else 0.18
    if relation == "GUIDES":
        return 0.18
    return 0.10
