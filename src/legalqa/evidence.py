"""Mở rộng Chunk trúng ra ranh giới Điều — small-to-big.

Chunk dùng để tìm (nhỏ, tối ưu recall), Evidence dùng để sinh (đầy đủ
ngữ cảnh cho model). 84,1% Document có Điều, 15,9% không có — cần đường lui.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .corpus import Article, Document


@dataclass(slots=True)
class Evidence:
    doc_id: str
    doc_title: str
    article_number: str | None
    text: str


def expand_to_evidence(
    doc: Document, chunk: Chunk, *, max_words: int = 900
) -> Evidence:
    """Mở rộng Chunk ra ranh giới Article chứa nó.

    Nếu Chunk nằm trong Article: lấy trọn text Article đó.
    Nếu Article dài hơn max_words: cắt quanh Chunk, giữ heading.
    Nếu Document không có Article nào (15,9%): lấy cửa sổ ±1 Chunk.
    """
    # Tìm Article chứa chunk bằng so sánh offset
    containing: Article | None = None
    for art in doc.articles:
        if art.start <= chunk.start and chunk.end <= art.end:
            containing = art
            break

    if containing is not None:
        article_text = doc.passage[containing.start : containing.end]
        word_count = len(article_text.split())

        if word_count <= max_words:
            return Evidence(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                article_number=containing.number,
                text=article_text,
            )

        # Article quá dài — cắt quanh Chunk, giữ heading
        return _window_around_chunk(doc, chunk, containing, max_words)

    # Không có Article nào — đường lui: cửa sổ ±1 chunk
    return _window_around_chunk(doc, chunk, None, max_words)


def _window_around_chunk(
    doc: Document, chunk: Chunk, article: Article | None, max_words: int = 900
) -> Evidence:
    """Cắt cửa sổ quanh chunk, giữ heading của Article (nếu có).

    Heading của Article được tính vào max_words — nếu heading dài thì
    cửa sổ co lại tương ứng.
    """
    passage_words = doc.passage.split()
    if not passage_words:
        return Evidence(
            doc_id=doc.doc_id,
            doc_title=doc.title,
            article_number=article.number if article else None,
            text="",
        )

    heading = ""
    heading_words = 0
    if article is not None:
        heading = article.heading + "\n"
        heading_words = len(article.heading.split())

    chunk_word_start = _char_to_word_offset(doc.passage, chunk.start)
    body_budget = max(1, max_words - heading_words)
    half_window = body_budget // 2

    w_start = max(0, chunk_word_start - half_window)
    w_end = min(len(passage_words), w_start + body_budget)
    # Đảm bảo chunk nằm trong window
    if chunk_word_start < w_start:
        w_start = max(0, chunk_word_start - half_window)
        w_end = min(len(passage_words), w_start + body_budget)

    selected_text = " ".join(passage_words[w_start:w_end])

    return Evidence(
        doc_id=doc.doc_id,
        doc_title=doc.title,
        article_number=article.number if article else None,
        text=heading + selected_text,
    )


def _char_to_word_offset(text: str, char_pos: int) -> int:
    """Đếm số từ trước char_pos — cần cho mapping offset-sang-từ."""
    return len(text[:char_pos].split())
