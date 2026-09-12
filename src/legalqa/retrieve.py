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
    from .graph import LegalGraph
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

    def evidence(
        self, question: str, *, top_n: int = 5, max_chars: int = 24_000
    ) -> list["Evidence"]:
        """Evidence để nhồi prompt, đã bỏ trùng và đã cắt theo ngân sách ký tự.

        CHƯA có phần mở rộng theo legal graph — dùng `evidence_day_du()` cho đường
        đầy đủ. Giữ hàm này riêng vì `eval_rerank.py` đo recall trên seed, thêm
        Evidence graph vào đó là đo một đại lượng khác.
        """
        return chon_evidence(
            self.rank(question), self.docs, top_n=top_n, max_chars=max_chars
        )

    def evidence_day_du(
        self,
        question: str,
        graph: "LegalGraph",
        *,
        top_n: int = 5,
        max_chars: int = 24_000,
        max_extra: int = 3,
    ) -> list["Evidence"]:
        """Seed + phần mở rộng legal graph, xếp đúng thứ tự prompt cần.

        Đây là caller mà `prompt.py` nói tới: cặp "Điều gốc -> khoản sửa đổi" nằm
        trong THỨ TỰ của list này, không nằm ở trường nào của Evidence. Ai sắp lại
        list này là xoá thông tin không dựng lại được.
        """
        from .graph import expand_with_graph

        out: list["Evidence"] = []
        da_dung = 0
        da_thay: set[tuple[str, str]] = set()
        # top_n đếm SEED, không đếm Evidence graph: graph không phải kết quả tìm kiếm,
        # nó là ngữ cảnh bắt buộc của seed. Đếm chung thì một seed có 3 khoản sửa đổi
        # sẽ ăn hết top_n và đẩy seed hạng 2 ra ngoài vì lý do không liên quan tới hạng.
        for seed in self.evidence(question, top_n=top_n, max_chars=max_chars):
            # Seed + phần mở rộng của nó là MỘT cụm: nhận cả cụm hoặc không nhận.
            # Nhận seed rồi cắt mất khoản sửa đổi của nó là đưa vào prompt một Điều đã
            # hết hiệu lực mà không kèm dấu hiệu nào — tệ hơn là không đưa gì.
            cum = [seed, *expand_with_graph(graph, self.docs, seed, max_extra=max_extra)]
            cum = [
                e
                for e in cum
                if (e.doc_id, e.article_number or e.text[:80]) not in da_thay
            ]
            if not cum:
                continue
            ton = sum(len(e.text) for e in cum)
            # `out` rỗng thì nhận bất chấp ngân sách: prompt không có căn cứ nào thì
            # model chắc chắn bịa.
            if out and da_dung + ton > max_chars:
                break
            for e in cum:
                da_thay.add((e.doc_id, e.article_number or e.text[:80]))
            out.extend(cum)
            da_dung += ton
        return out


def chon_evidence(
    ranked: list["Chunk"],
    docs: dict[str, "Document"],
    *,
    top_n: int = 5,
    max_chars: int = 24_000,
    khoa: str = "dieu",
) -> list["Evidence"]:
    """Chuyển Chunk đã xếp hạng thành Evidence để nhồi prompt, đã bỏ trùng.

    Hai giới hạn cùng lúc, ai chặn trước thì thắng: `top_n` và `max_chars`. Số Evidence
    trả về là KẾT QUẢ, không phải tham số — Evidence p50 869 từ nhưng max 26 Điều, nên
    `top_n=5` có thể ra 5 Evidence ngắn hoặc 2 Evidence dài. Chỉ chặn theo `top_n` thì
    prompt phình không kiểm soát; chỉ chặn theo ký tự thì mất mốc dễ đọc. v9.2 đặt
    `max_context_chars: 24000` (configs/kaggle_inference_t4.yaml) — giữ nguyên số đó.

    LUÔN trả về ít nhất 1 Evidence nếu `ranked` không rỗng, kể cả khi nó vượt ngân sách.
    Prompt không có căn cứ nào thì model chắc chắn bịa; Evidence bị cắt cụt vẫn hơn.

    Bỏ trùng là bắt buộc, không phải trường hợp biên: chunk/Document p50 19, pool 50
    thường chứa nhiều chunk cùng một Document. Hai chunk chạm cùng một Điều thì
    `expand_to_evidence` trả text y hệt — nhồi cả hai là đốt 869 từ để lấy 0 thông tin.

    `khoa` chọn mức bỏ trùng:

    - `"dieu"` (mặc định) = `(doc_id, article_number)`. Cho trùng Document, không cho
      trùng Điều. Chọn mặc định này vì 2.097/6.250 câu (33,5%) có gold từ 2+ Document —
      nhưng phần còn lại có nghĩa là 66,5% câu gold nằm trong MỘT văn bản, và câu cần
      hai Điều của cùng một luật là dạng phổ biến trong luật Việt Nam (Điều dẫn chiếu
      Điều). Khoá theo `doc_id` sẽ mất Điều thứ hai của đúng những câu đó.
    - `"van_ban"` = `doc_id`. Rộng nhất, 5 văn bản khác nhau. Dùng khi reranker dồn cả
      pool vào một Document và cần ép đa dạng.
    - `"text"` = chính text. Chính xác nhất nhưng Evidence chồng lấn MỘT PHẦN vẫn lọt
      (Điều 5,6 và Điều 6,7 là hai text khác nhau mà chung Điều 6).

    28,1% Evidence là fallback cửa sổ, `article_number is None` (§7). Với chúng khoá
    `"dieu"` sẽ gộp mọi fallback của cùng một Document thành một — mà chúng là các cửa
    sổ KHÁC NHAU. Nên `article_number is None` thì tụt về khoá text.
    """
    ev: list["Evidence"] = []
    da_thay: set[object] = set()
    da_dung = 0
    for chunk in ranked:
        if len(ev) >= top_n:
            break
        doc = docs.get(chunk.doc_id)
        if doc is None:
            continue
        e = expand_to_evidence(doc, chunk)
        if khoa == "van_ban":
            k: object = e.doc_id
        elif khoa == "text" or e.article_number is None:
            k = e.text
        else:
            k = (e.doc_id, e.article_number)
        if k in da_thay:
            continue
        # Vượt ngân sách thì DỪNG, không phải bỏ qua rồi thử cái sau: `ranked` xếp theo
        # hạng nên cái sau luôn kém hơn. Nhặt một Evidence ngắn hạng 9 thay cho cái dài
        # hạng 3 là đổi chất lượng lấy độ dài — sai chiều.
        if ev and da_dung + len(e.text) > max_chars:
            break
        da_thay.add(k)
        da_dung += len(e.text)
        ev.append(e)
    return ev
