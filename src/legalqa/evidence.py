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
    # Quan hệ với Evidence đã sinh ra nó ("Sửa đổi Điều 127 của ..."). Rỗng với
    # Evidence gốc từ retrieval. Prompt dán nhãn này để model biết đoạn nào sửa đoạn nào.
    relation: str = ""


def expand_to_evidence(
    doc: Document, chunk: Chunk, *, max_words: int = 900
) -> Evidence:
    """Mở rộng Chunk ra trọn các Điều mà nó chạm vào.

    Chunk 256 từ thường dài hơn một Điều: đo trên 800 Document đầu, chỉ 56% Chunk
    nằm trọn trong một Điều. Nên lấy TẤT CẢ Điều giao với Chunk thay vì đòi chứa trọn —
    cắt nửa chừng một Điều làm mất chính đoạn mà 75,9% reference answer trích nguyên văn.
    """
    touched = _touched_articles(doc, chunk)

    if touched:
        start = touched[0].start
        end = touched[-1].end
        text = doc.passage[start:end]
        number = ", ".join(a.number for a in touched)
        if len(text.split()) <= max_words:
            return Evidence(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                article_number=number,
                text=text,
            )
        # Span quá dài — cắt quanh Chunk, giữ heading của Điều đầu tiên.
        # `number` vẫn là TẤT CẢ Điều chạm, không lọc lại theo cửa sổ: đã đo trên
        # 206.647 Evidence có Điều, 85.205 (41,2%) đi đường này, 26.532 trong đó chạm
        # nhiều Điều — và 0 trường hợp cửa sổ đánh rơi một Điều đã khai. Không phải may:
        # cửa sổ 900 từ dựng quanh TÂM Chunk 256 từ nên luôn chứa trọn Chunk, mà mọi
        # Điều trong `touched` đều giao với Chunk. Đừng thêm bước lọc, nó không bắt gì.
        return _window_around_chunk(doc, chunk, touched[0], max_words, number)

    # Document không có Điều nào (15,9% corpus) — đường lui: cửa sổ quanh Chunk
    return _window_around_chunk(doc, chunk, None, max_words, None)


def _touched_articles(doc: Document, chunk: Chunk) -> list[Article]:
    """Các Điều giao với [chunk.start, chunk.end), theo thứ tự xuất hiện."""
    return [
        a for a in doc.articles if a.start < chunk.end and chunk.start < a.end
    ]


def _window_around_chunk(
    doc: Document,
    chunk: Chunk,
    article: Article | None,
    max_words: int,
    number: str | None,
) -> Evidence:
    """Cắt cửa sổ ký tự quanh Chunk, giữ heading của Điều (nếu có).

    Cắt theo offset ký tự chứ không qua split()/join(): 95,8% Document dùng xuống dòng
    để phân khoản, normalize whitespace là mất cấu trúc model cần để trích đúng khoản.
    """
    if not doc.passage.strip():
        return Evidence(doc.doc_id, doc.title, number, "")

    heading = f"{article.heading}\n" if article is not None else ""
    budget = max(1, max_words - len(heading.split()))

    # Ước lượng khoảng ký tự từ ngân sách từ: dùng mật độ ký tự/từ thật của Passage
    words = doc.passage.split()
    chars_per_word = len(doc.passage) / max(1, len(words))
    half = int(budget * chars_per_word / 2)

    center = (chunk.start + chunk.end) // 2
    start = max(0, center - half)
    end = min(len(doc.passage), start + int(budget * chars_per_word))
    body = doc.passage[start:end]

    return Evidence(
        doc_id=doc.doc_id,
        doc_title=doc.title,
        article_number=number,
        text=heading + body,
    )
