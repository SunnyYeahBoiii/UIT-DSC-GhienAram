"""Test luật giữ/loại mẫu SFT. Chạy không cần GPU, không cần dữ liệu thi."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_sft import MAX_SEQ, giu_mau  # noqa: E402


class _Ev:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id


def test_giu_khi_evidence_chua_gold():
    assert giu_mau([_Ev("a"), _Ev("b")], {"b"}, 1000) == ""


def test_loai_khi_khong_co_weak_gold():
    """750/7.000 câu (10,7%) không dựng được weak gold. `set()` rỗng cũng phải loại,
    không chỉ `None` — `any(... in set())` trả False nên nếu quên nhánh này thì mẫu rơi
    vào "retrieval_truot_gold", làm sai luôn số đo chẩn đoán."""
    assert giu_mau([_Ev("a")], None, 1000) == "khong_co_weak_gold"
    assert giu_mau([_Ev("a")], set(), 1000) == "khong_co_weak_gold"


def test_loai_khi_retrieval_truot():
    assert giu_mau([_Ev("a")], {"z"}, 1000) == "retrieval_truot_gold"


def test_khong_con_loai_theo_seq_len():
    """`giu_mau` KHÔNG được loại theo độ dài nữa (quyết định 2026-09-14: bỏ trần token).

    Lọc hai chỗ là nguồn lỗi thật chứ không phải chuyện thẩm mỹ: `build_sft` loại theo
    tokenizer nó tự nạp, `train_sft` loại lại theo chat template — hai con số khác nhau
    trên cùng một mẫu. Cổng duy nhất bây giờ là `train_sft.py --max-seq`, nơi seq_len
    đo đúng thứ sẽ vào GPU.
    """
    assert giu_mau([_Ev("a")], {"a"}, MAX_SEQ + 1) == ""
    assert giu_mau([_Ev("a")], {"a"}, 10 * MAX_SEQ) == ""
