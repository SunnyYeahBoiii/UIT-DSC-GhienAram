"""Cross-encoder rerank trên top-K của BM25.

Vì sao rerank chứ không phải dense (đo trên 1.000 câu weak gold, corpus đã clean):

| | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| BM25 chunk 256/256 | 54,3% | 81,2% | 88,4% | 96,7% |

Gold KHÔNG với tới trong 3.000 chunk đầu: 4/1.000 = 0,4%. Riêng nhóm answer diễn đạt
lại (gold chỉ suy được qua số hiệu, không chung chữ với câu hỏi): R@10 87,5% — không
sụp so với 88,4% chung. Tức BM25 gần như luôn TÌM THẤY văn bản đúng, chỉ XẾP SAI HẠNG.

Hệ quả kiến trúc: thay BM25 bằng bi-encoder dense chỉ mua được tối đa 0,4 điểm recall
với giá 568M params. Còn 15,5 điểm (R@5 81,2% → R@50 96,7%) nằm sẵn trong top-50, chỉ
cần chấm lại. Cross-encoder cho query và chunk nhìn nhau qua attention nên chấm chính
xác hơn hai vector chấm nhau, và chỉ chạy K lần/câu thay vì encode trước 287.358 chunk.

torch/transformers import TRONG hàm: năm module lõi (corpus/chunking/evidence/bm25/
clean) phải chạy được khi chỉ có numpy+scipy+sklearn, để dựng chunk và đo BM25 trên
máy không GPU. Module này là tầng tuỳ chọn, nạp khi có torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .chunking import Chunk

# Chunk 256 từ tiếng Việt ≈ 380-520 token subword. 512 giữ trọn gần hết chunk; cắt ngắn
# hơn là vứt đuôi chunk, mà đuôi chunk hay chứa đúng khoản cần tìm.
MAX_LEN = int(os.environ.get("RERANK_MAX_LEN", "512"))


@dataclass(frozen=True)
class Scored:
    """Một ứng viên sau khi chấm lại. Giữ cả điểm BM25 gốc để hoà điểm được."""

    chunk: "Chunk"
    bm25: float
    rerank: float


class CrossEncoderReranker:
    """Chấm lại (query, chunk) bằng cross-encoder.

    Mặc định `AITeamVN/Vietnamese_Reranker` (XLM-R large, 567.755.777 params, head
    hồi quy sẵn) — zero-shot, không train. Với Qwen 3B là 3,66B, còn trống 340M dưới
    trần 4B.

    Không buộc cứng vào checkpoint đó: `__init__` nhận đường dẫn bất kỳ, và `score`
    đọc được cả head hồi quy (`num_labels=1`) lẫn head phân loại (`num_labels=2`),
    nên đổi checkpoint không phải sửa code gọi.
    """

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(self._device).eval()
        # fp16 chỉ trên CUDA và chỉ từ sm_70: trên CPU không có kernel fp16, còn trên
        # P100 (sm_60, Kaggle cấp khi T4 hết slot) torch không build kernel -> ném
        # "no kernel image is available" ngay ở .half().
        if self._device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 7:
            self._model.half()
        self._batch_size = batch_size
        self._n_labels = int(self._model.config.num_labels)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Điểm liên quan cho từng text. Càng cao càng liên quan."""
        torch = self._torch
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                enc = self._tok(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",  # cắt chunk, KHÔNG cắt câu hỏi
                    max_length=MAX_LEN,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**enc).logits.float()
                # num_labels=1 là cross-encoder hồi quy (Vietnamese_Reranker); =2 là
                # phân loại nhị phân (head tự train). Hai kiểu checkpoint, một API.
                s = logits[:, 0] if self._n_labels == 1 else logits.softmax(-1)[:, 1]
                out.extend(s.tolist())
        return out

    def rerank(
        self, query: str, candidates: list[tuple["Chunk", float]], top_n: int | None = None
    ) -> list[Scored]:
        """Sắp lại top-K của BM25. `candidates` là (Chunk, điểm BM25) theo thứ tự BM25."""
        if not candidates:
            return []
        scores = self.score(query, [c.text for c, _ in candidates])
        ranked = sorted(
            (Scored(chunk=c, bm25=b, rerank=s) for (c, b), s in zip(candidates, scores)),
            key=lambda x: -x.rerank,
        )
        return ranked[:top_n] if top_n else ranked
