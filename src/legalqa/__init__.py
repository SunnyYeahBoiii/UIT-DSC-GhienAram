"""LegalQA v10 — pipeline hỏi đáp pháp luật tiếng Việt, offline Kaggle T4.

Đường đi của một câu hỏi, mỗi mũi tên là một module:

    corpus -> chunking -> bm25 (+dense) -> rerank -> evidence -> graph -> prompt

`import legalqa` KHÔNG kéo torch vào. Năm module lõi (corpus/chunking/evidence/bm25/
clean) chỉ cần numpy+scipy+sklearn để dựng 287.358 chunk và đo BM25 trên máy dev không
GPU; `rerank`/`dense` là tầng tuỳ chọn và import torch TRONG hàm. Re-export ở đây giữ
được tính chất đó vì nó chỉ nạp module, không gọi hàm nào —
`tests/test_init.py::test_import_duoc_khi_khong_co_torch` chặn đứng việc ai đó dời
`import torch` lên top-level của hai module kia.
"""
from __future__ import annotations

from .bm25 import ChunkIndex
from .chunking import Chunk, chunk_document
from .clean import clean_reference_answer, is_usable_for_sft
from .corpus import (
    Article,
    Document,
    dewrap_passage,
    iter_documents,
    load_document,
    parse_articles,
)
from .dense import BiEncoder, DenseIndex, rrf
from .evidence import Evidence, expand_to_evidence
from .graph import Amendment, CrossRef, DocRef, LegalGraph, expand_with_graph, parse_refs
from .prompt import NHAN_GOC, dung_can_cu, dung_prompt
from .rerank import CrossEncoderReranker, Scored
from .retrieve import Retriever, chon_evidence

__all__ = [
    "NHAN_GOC",
    "Amendment",
    "Article",
    "BiEncoder",
    "Chunk",
    "ChunkIndex",
    "CrossEncoderReranker",
    "CrossRef",
    "DenseIndex",
    "DocRef",
    "Document",
    "Evidence",
    "LegalGraph",
    "Retriever",
    "Scored",
    "chon_evidence",
    "chunk_document",
    "clean_reference_answer",
    "dewrap_passage",
    "dung_can_cu",
    "dung_prompt",
    "expand_to_evidence",
    "expand_with_graph",
    "is_usable_for_sft",
    "iter_documents",
    "load_document",
    "parse_articles",
    "parse_refs",
    "rrf",
]
