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


def test_loai_khi_vuot_max_seq_du_da_trung_gold():
    """Trúng gold vẫn phải loại nếu quá dài: truncate cắt từ đuôi, mà đuôi là reference
    answer — giữ lại là dạy model dừng giữa câu."""
    assert giu_mau([_Ev("a")], {"a"}, MAX_SEQ + 1) == "vuot_max_seq"
    assert giu_mau([_Ev("a")], {"a"}, MAX_SEQ) == ""
