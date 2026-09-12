"""Bi-encoder cho nhánh dense của hybrid search.

Vì sao chỉ là tầng tuỳ chọn, không thay BM25: EDA §15.2 đo được BM25 với tới gold
99,6% trong 3.000 chunk. Dense không mua thêm khả năng tìm thấy. Việc duy nhất nó có
thể làm là kéo gold từ hạng 51+ vào top-50 — tức nâng TRẦN của pool, thứ mà §15.7 chỉ
ra là chỗ nghẽn còn lại duy nhất sau khi reranker đã bão hoà.

Ràng buộc params là thứ định hình module này. Còn 340M dưới trần 4B sau Qwen 3B
(3,09B) + reranker (0,568B), nên embedding tiếng Việt mạnh nhất đều dùng không được:
`AITeamVN/Vietnamese_Embedding` và `BAAI/bge-m3` cùng là XLM-R large 568M. Chỉ vừa
loại base: `intfloat/multilingual-e5-small` (~118M) hay bi-encoder PhoBERT-base (~135M).

Không có thư viện vector database: 287.358 × 384 chiều fp16 = 0,22 GB, nhét thẳng RAM
rồi matmul. HNSW là xấp xỉ — đổi độ chính xác lấy tốc độ mà ở quy mô này không cần.

torch/transformers import TRONG hàm, cùng lý do với rerank.py: năm module lõi phải chạy
được khi chỉ có numpy/scipy/sklearn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

T = TypeVar("T")


class BiEncoder:
    """Encode text thành vector đã chuẩn hoá L2 (nên dot product = cosine).

    `prefix_*`: dòng e5 được train với "query: " / "passage: " ở đầu. Bỏ prefix làm
    điểm tụt rõ rệt, nên tự nhận theo tên model thay vì bắt người gọi nhớ.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 128,
        # 256, KHÔNG phải 512 dù chunk 256 từ ≈ 380-520 subword nên 512 đọc trọn chunk.
        # §15.8 đo ngược: L512 dense @10 91,4 vs L256 92,2, hybrid @50 97,5 vs 97,9 —
        # mean pooling bị loãng, vài trăm token boilerplate bầu áp đảo vài token định vị.
        # Mặc định 512 là cái bẫy: `BiEncoder(path)` không truyền gì sẽ dựng index ở đúng
        # cấu hình EDA ghi "đừng thử lại". build_dense.py đã ép 256, class thì chưa.
        max_len: int = 256,
        device: str | None = None,
        prefix: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)
        self._model.to(self._device).eval()
        # fp16 chỉ khi compute capability >= 7.0. Kaggle cấp P100 (sm_60) khi T4 hết
        # slot, mà torch bản trên Kaggle chỉ build kernel từ sm_70 -> .half() ném
        # "no kernel image is available". Tụt về fp32 chậm hơn ~2x nhưng vẫn ra số.
        if self._device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 7:
            self._model.half()
        self._bs = batch_size
        self._max_len = max_len
        # prefix: "e5" | "none" | None (suy từ tên). Suy theo tên từng làm hỏng một lượt đo
        # — truro7/vn-law-embedding là e5-base finetune nhưng tên không chứa "e5", nên prefix
        # bị bỏ im lặng và điểm tụt mà không ai biết vì sao. Giữ auto cho tiện, nhưng cho
        # người gọi ép được.
        if prefix is None:
            prefix = "e5" if "e5" in model_path.lower() else "none"
        self._q_pre, self._d_pre = ("query: ", "passage: ") if prefix == "e5" else ("", "")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """(n, dim) float16 đã chuẩn hoá. Mean pooling có mask — CLS của model chưa
        train contrastive trên CLS sẽ cho vector kém hơn hẳn."""
        torch = self._torch
        pre = self._q_pre if is_query else self._d_pre
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._bs):
                batch = [pre + t for t in texts[i : i + self._bs]]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_len,
                    return_tensors="pt",
                ).to(self._device)
                h = self._model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(v.float(), dim=-1)
                out.append(v.half().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float16)


def rrf(rank_lists: list[list[T]], k: int = 60) -> list[T]:
    """Reciprocal Rank Fusion. Cộng 1/(k+hạng) qua các danh sách, trả thứ tự đã trộn.

    Dùng HẠNG chứ không dùng điểm: điểm BM25 (không chặn trên) và cosine (trong [-1,1])
    không cùng thang, chuẩn hoá lại thì phụ thuộc phân phối từng câu. Hạng miễn nhiễm.
    k=60 là mặc định trong bài gốc Cormack 2009 — làm phẳng chênh lệch ở top, tránh để
    một nhánh áp đảo chỉ vì nó xếp hạng 1.
    """
    score: dict[T, float] = {}
    for lst in rank_lists:
        for r, item in enumerate(lst, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda x: -score[x])


class DenseIndex:
    """Ma trận embedding của toàn bộ Chunk, giữ thẳng trong RAM.

    287.358 × 384 fp16 = 0,22 GB. Không dùng FAISS/HNSW: ở quy mô này matmul đặc chạy
    ~0,1s/câu, còn HNSW là **xấp xỉ** — đổi độ chính xác lấy tốc độ mà ta không cần, và
    thêm một wheel phải build offline cho Kaggle.

    `search` nhận vector câu hỏi đã chuẩn hoá, trả (chunk_id, cosine) giảm dần.
    """

    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        assert len(chunk_ids) == matrix.shape[0], (len(chunk_ids), matrix.shape)
        self.chunk_ids = chunk_ids
        # fp32 một lần lúc nạp: matmul fp16 trên CPU chậm hơn fp32 vài lần vì numpy
        # không có kernel fp16, và 0,44 GB vẫn thừa chỗ trên Kaggle 30 GB RAM.
        self.matrix = matrix.astype(np.float32)

    @classmethod
    def load(cls, npy: "Path", ids_txt: "Path") -> "DenseIndex":
        ids = ids_txt.read_text(encoding="utf-8").split("\n")
        return cls([i for i in ids if i], np.load(npy))

    def search(self, qvec: np.ndarray, top_k: int = 1000) -> list[tuple[str, float]]:
        sim = self.matrix @ qvec.astype(np.float32)
        k = min(top_k, sim.shape[0])
        # argpartition thay vì argsort: chỉ cần k phần tử đầu, 287k phần tử thì
        # partition O(n) rẻ hơn sort O(n log n) khoảng 10 lần.
        top = np.argpartition(-sim, k - 1)[:k]
        top = top[np.argsort(-sim[top])]
        return [(self.chunk_ids[j], float(sim[j])) for j in top]
