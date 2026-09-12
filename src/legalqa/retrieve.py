"""Wiring retrieval end-to-end: câu hỏi -> top-N Evidence để nhồi prompt.

Bốn tầng, mỗi tầng có số đo biện minh trong EDA §15:

    BM25 (deep 1.000)  ─┐
                        ├─ RRF k=60 ─> pool 50 ─> cross-encoder ─> top-5 Evidence
    dense (deep 1.000) ─┘

- **pool 50** (§15.6): lợi ích biên 20→50 là 0,48 điểm/phút, 50→100 chỉ 0,03. Gấp 16 lần.
- **RRF theo hạng** (§15.8): điểm BM25 không chặn trên, cosine trong [-1,1]. Chuẩn hoá
  lại thì tỉ lệ trộn phụ thuộc phân phối điểm của từng câu.
- **cross-encoder** (§15.3): +18,8 điểm R@1, +11,0 R@5 so với BM25 thuần cùng pool.
- **top-5 Evidence chứ không top-1** (§15.7): rerank@1 bỏ lại 19,3 điểm so với trần,
  @5 chỉ còn bỏ lại 4,6. Evidence p50 869 từ × 5 ≈ 6.000 token, vừa context Qwen 3B.

Nhánh dense là TUỲ CHỌN: không có `DenseIndex` thì pool lấy thẳng từ BM25 và mọi thứ
sau đó không đổi. Cần vậy vì §15.8 nâng trần 96,7 -> 97,9 chỉ là +1,2 điểm — đủ đáng
118M params, nhưng không đáng để cả pipeline chết khi thiếu file .npy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .dense import rrf
from .evidence import expand_to_evidence

if TYPE_CHECKING:  # pragma: no cover
    from .bm25 import ChunkIndex
    from .chunking import Chunk
    from .corpus import Document
    from .dense import BiEncoder, DenseIndex
    from .evidence import Evidence
    from .rerank import CrossEncoderReranker


@dataclass(frozen=True)
class Retriever:
    """Giữ index và model, không giữ trạng thái giữa các câu hỏi.

    `docs` và `chunks` nạp sẵn vào RAM: corpus.jsonl 470 MB + chunks.jsonl 474 MB vừa
    30 GB RAM của Kaggle, và bài thi 1.000 câu thì đọc lại file mỗi câu là đắt hơn hẳn.
    """

    index: "ChunkIndex"
    chunks: dict[str, "Chunk"]
    docs: dict[str, "Document"]
    reranker: "CrossEncoderReranker | None" = None
    encoder: "BiEncoder | None" = None
    dense: "DenseIndex | None" = None
    pool: int = 50
    deep: int = 1000
    rrf_k: int = 60

    def pool_chunks(self, question: str) -> list["Chunk"]:
        """Pool ứng viên sau khi trộn hai nhánh. Dài đúng `self.pool`."""
        bm = [cid for cid, _ in self.index.search(question, top_k=self.deep)]
        if self.dense is None or self.encoder is None:
            ids = bm[: self.pool]
        else:
            qv = self.encoder.encode([question], is_query=True)[0]
            de = [cid for cid, _ in self.dense.search(qv, top_k=self.deep)]
            ids = rrf([bm, de], k=self.rrf_k)[: self.pool]
        return [self.chunks[c] for c in ids]

    def rank(self, question: str) -> list["Chunk"]:
        """Pool đã xếp lại bằng cross-encoder. Không có reranker thì giữ thứ tự pool."""
        cands = self.pool_chunks(question)
        if self.reranker is None:
            return cands
        # BM25 score không dùng để hoà điểm (§15.3 đo rerank thuần đã đủ), truyền 0.0
        # cho khớp chữ ký thay vì tính lại.
        return [s.chunk for s in self.reranker.rerank(question, [(c, 0.0) for c in cands])]

    def evidence(self, question: str, top_n: int = 5) -> list["Evidence"]:
        """Top-N Evidence để nhồi prompt, đã bỏ trùng."""
        return chon_evidence(self.rank(question), self.docs, top_n=top_n)


def chon_evidence(
    ranked: list["Chunk"], docs: dict[str, "Document"], *, top_n: int = 5
) -> list["Evidence"]:
    """Chuyển Chunk đã xếp hạng thành top-N Evidence, bỏ trùng.

    Trùng là CHẮC CHẮN xảy ra, không phải trường hợp biên: chunk/Document p50 19, và
    pool 50 thường chứa nhiều chunk của cùng một Document. Hai chunk chạm cùng một Điều
    thì `expand_to_evidence` trả text y hệt nhau — nhồi cả hai là đốt 869 từ ngân sách
    prompt để lấy 0 thông tin.

    TODO(người dùng): chọn khoá bỏ trùng. Ba lựa chọn, đánh đổi thật:

    - theo `doc_id`: 5 văn bản khác nhau, rộng nhất. Nhưng 2.097/6.250 câu (33,5%) có
      gold từ 2+ Document, và câu cần HAI Điều của CÙNG một luật sẽ mất Điều thứ hai.
    - theo `(doc_id, article_number)`: 5 Điều khác nhau, cho phép trùng Document. Giữ
      được cả hai khoản của một luật, nhưng một Document áp đảo thì mất chiều rộng.
    - theo text: chính xác nhất nhưng Evidence chồng lấn một phần vẫn lọt (Điều 5,6 và
      Điều 6,7 là hai text khác nhau mà chung Điều 6).

    Trả về đúng thứ tự hạng, tối đa `top_n` phần tử.
    """
    raise NotImplementedError("xem TODO ở trên")
