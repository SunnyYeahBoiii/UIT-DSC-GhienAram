"""Cắt Document thành Chunk theo từ — đơn vị tìm kiếm.

Chunk là lát cắt Passage dùng làm đơn vị index BM25. Cắt theo từ
(whitespace) để tránh phá vỡ token, size/stride tính bằng từ.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Document

MIN_CHUNK_WORDS = 30


@dataclass(slots=True)
class Chunk:
    chunk_id: str  # f"{doc_id}:{ordinal}"
    doc_id: str
    text: str
    start: int  # offset ký tự trong Passage
    end: int


def chunk_document(doc: Document, size: int = 256, stride: int = 256) -> list[Chunk]:
    """Cắt Passage thành các Chunk, trả về list rỗng nếu passage trống.

    size/stride tính bằng từ. Bỏ chunk cuối nếu < 30 từ và không phải
    chunk đầu — chunk quá ngắn không đủ signal cho BM25.
    """
    words = doc.passage.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    # Xây word_ends: vị trí ký tự kết thúc của mỗi từ (sau ký tự cuối).
    # Cần cho offset end chính xác — tránhbialospace thừa.
    word_starts: list[int] = []
    word_ends: list[int] = []
    i = 0
    while i < len(doc.passage):
        while i < len(doc.passage) and doc.passage[i].isspace():
            i += 1
        if i >= len(doc.passage):
            break
        word_starts.append(i)
        while i < len(doc.passage) and not doc.passage[i].isspace():
            i += 1
        word_ends.append(i)

    for ordinal, word_idx in enumerate(range(0, len(words), stride)):
        word_slice = words[word_idx : word_idx + size]
        if not word_slice:
            break

        is_last_chunk = word_idx + size >= len(words)
        is_first_chunk = ordinal == 0

        # Bỏ chunk cuối nếu quá ngắn và không phải chunk đầu
        if is_last_chunk and not is_first_chunk and len(word_slice) < MIN_CHUNK_WORDS:
            break

        char_start = word_starts[word_idx]
        last_word_idx = word_idx + len(word_slice) - 1
        char_end = word_ends[last_word_idx]

        chunk_text = " ".join(word_slice)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{ordinal}",
                doc_id=doc.doc_id,
                text=chunk_text,
                start=char_start,
                end=char_end,
            )
        )

    return chunks
