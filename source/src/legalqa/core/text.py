from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable


WORD_RE = re.compile(r"[0-9]+(?:[.,:/-][0-9]+)*|[a-zà-ỹđ]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ỸĐ0-9\-•])|\n+")
ARTICLE_RE = re.compile(
    r"\b(?:điều|khoản|điểm|chương|mục)\s+[0-9ivxlcdm]+(?:[a-zđ])?\b",
    re.IGNORECASE,
)
LEGAL_DOC_RE = re.compile(
    r"\b(?:luật|bộ luật|nghị định|thông tư|quyết định|nghị quyết|hiến pháp|"
    r"pháp lệnh|công văn)\b(?:[^.!?\n]{0,100})?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|đồng|triệu|tỷ|ngày|tháng|năm|giờ|tuổi))?\b",
    re.IGNORECASE,
)
DOCUMENT_NUMBER_RE = re.compile(
    r"\b(?P<number>\d{1,4})\s*/\s*(?P<year>\d{4})\s*/\s*"
    r"(?P<suffix>[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ0-9-]{1,40})\b",
    re.IGNORECASE,
)
SLUG_DOCUMENT_NUMBER_RE = re.compile(
    r"(?i:(?:luat|bo[-_ ]luat|nghi[-_ ]dinh|thong[-_ ]tu|quyet[-_ ]dinh|"
    r"nghi[-_ ]quyet|phap[-_ ]lenh|cong[-_ ]van))[-_ ]+"
    r"(?P<number>\d{1,4})[-_/](?P<year>\d{4})[-_/]"
    r"(?P<suffix>[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+){0,4})"
)

STOPWORDS = {
    "và", "là", "của", "có", "được", "cho", "trong", "theo", "với", "về",
    "những", "các", "một", "này", "đó", "khi", "thì", "để", "tại", "do",
    "hay", "hoặc", "từ", "đến", "trên", "dưới", "như", "nào", "gì", "ai",
    "bao", "nhiêu", "phải", "không", "tôi", "bị", "sẽ", "đã", "đang",
}


_SENTENCE_TERMINATORS = (".", "!", "?", ":", ";")


def _dewrap(text: str) -> str:
    """Rejoin hard line-wraps from the scraped corpus.

    The official `selected-contexts` passages are stored with a newline (often
    doubled, e.g. ``\\r\\n\\n``) after nearly every visually wrapped line, not
    only at real sentence/paragraph boundaries (a PDF/HTML scrape artifact
    present across the whole corpus, e.g. "B\u1ed8 LAO \u0110\u1ed8NG -\\n\\n TH\u01af\u01a0NG BINH V\u00c0
    X\u00c3 H\u1ed8I"). A line that doesn't end a sentence is joined onto the next one;
    a line that does is kept as its own line, which also lets downstream
    paragraph/sentence splitters (`_window_text`, `split_sentences`) find real
    boundaries instead of every wrapped line.
    """
    joined: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if joined and not joined[-1].endswith(_SENTENCE_TERMINATORS):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return "\n".join(joined)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _dewrap(text)
    return text.strip()


_IMAGE_CAPTION_RE = re.compile(
    r"\(\s*(?:H\u00ecnh|\u1ea2nh)[^()]{0,60}?(?:Internet|minh\s*h[o\u1ecd]a)[^()]{0,20}\)",
    re.IGNORECASE,
)


def strip_source_noise(text: str, question: str | None = None) -> str:
    """B\u1ecf ch\u00fa th\u00edch \u1ea3nh crawl web (vd "(H\u00ecnh t\u1eeb Internet)") v\u00e0 c\u00e2u h\u1ecfi b\u1ecb
    l\u1eb7p l\u1ea1i trong answer th\u00f4 c\u1ee7a train.json (artifact crawl HTML)."""
    answer = _IMAGE_CAPTION_RE.sub("", text)
    question = (question or "").strip()
    if question:
        answer = answer.replace(question, "")
    return normalize_text(answer)


def words(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = [token.lower() for token in WORD_RE.findall(normalize_text(text))]
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def legal_tokens(text: str) -> list[str]:
    """Vietnamese lexical tokens plus adjacent syllable features for BM25."""
    base = words(text)
    content = [token for token in base if token not in STOPWORDS or token.isdigit()]
    bigrams = [f"{content[i]}_{content[i + 1]}" for i in range(len(content) - 1)]
    citations = [re.sub(r"\s+", "_", item.lower()) for item in ARTICLE_RE.findall(text)]
    return base + bigrams + citations


def split_sentences(text: str) -> list[str]:
    result: list[str] = []
    for piece in SENTENCE_RE.split(normalize_text(text)):
        piece = piece.strip(" \t\n")
        if piece:
            result.append(piece)
    return result


def citations(text: str) -> set[str]:
    found = set()
    for match in ARTICLE_RE.findall(normalize_text(text)):
        found.add(re.sub(r"\s+", " ", match.lower()).strip())
    return found


def legal_document_phrases(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", match.group(0).lower()).strip(" ,.;:")
        for match in LEGAL_DOC_RE.finditer(normalize_text(text))
    }


def numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", item.lower()) for item in NUMBER_RE.findall(text)}


def document_numbers(text: str) -> set[str]:
    """Return normalized legal-document identifiers such as ``12/2012/TT-BTP``.

    The official corpus often stores titles as URL slugs, so both normal legal
    notation and ``Thong-tu-12-2012-TT-BTP`` style identifiers are supported.
    Generic years and article numbers are deliberately excluded.
    """
    normalized = normalize_text(text)
    result: set[str] = set()
    for pattern in (DOCUMENT_NUMBER_RE, SLUG_DOCUMENT_NUMBER_RE):
        for match in pattern.finditer(normalized):
            suffix = match.group("suffix").upper().strip("-_/.,;:()[] ")
            suffix = "".join(
                character
                for character in unicodedata.normalize("NFD", suffix.replace("Đ", "D"))
                if unicodedata.category(character) != "Mn"
            )
            if suffix:
                result.add(f"{match.group('number')}/{match.group('year')}/{suffix}")
    return result


def question_type(question: str) -> str:
    q = normalize_text(question).lower()
    patterns = (
        ("procedure", r"\b(thủ tục|hồ sơ|trình tự|đăng ký|nộp ở đâu)\b"),
        ("penalty", r"\b(xử phạt|mức phạt|phạt bao nhiêu|vi phạm)\b"),
        ("deadline", r"\b(thời hạn|bao lâu|khi nào|thời gian)\b"),
        ("eligibility", r"\b(điều kiện|đối tượng|có được|có quyền|đủ điều kiện)\b"),
        ("authority", r"\b(thẩm quyền|cơ quan nào|ai có quyền)\b"),
        ("definition", r"\b(là gì|được hiểu|khái niệm|thế nào)\b"),
        ("rights", r"\b(quyền|nghĩa vụ|trách nhiệm)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, q):
            return label
    return "general"


def answer_length_bucket(answer: str) -> str:
    count = len(words(answer))
    if count < 180:
        return "short"
    if count < 380:
        return "medium"
    if count < 600:
        return "long"
    return "very_long"


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(words(prediction))
    ref = Counter(words(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def overlap_ratio(source: Iterable[str], target: Iterable[str]) -> float:
    source_set, target_set = set(source), set(target)
    if not source_set:
        return 0.0
    return len(source_set & target_set) / len(source_set)


def length_prior(word_count: int, target: int, minimum: int, maximum: int) -> float:
    if word_count <= 0:
        return 0.0
    if word_count < minimum:
        return word_count / max(1, minimum)
    if word_count > maximum:
        return max(0.0, 1.0 - (word_count - maximum) / max(maximum, 1))
    scale = max(target, 1)
    return math.exp(-abs(word_count - target) / scale)
