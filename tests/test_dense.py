"""Test cho tầng dense. Không nạp model — chỉ phần numpy, chạy được khi không có torch."""

from __future__ import annotations

import numpy as np
import pytest

from legalqa.dense import DenseIndex, rrf


def _idx() -> DenseIndex:
    # Ba vector đơn vị trên ba trục: cosine với trục x là đúng [1, 0, 0], thứ hạng
    # biết trước nên bắt được lỗi đảo dấu trong argpartition.
    return DenseIndex(["a", "b", "c"], np.eye(3, dtype=np.float16))


def test_search_giam_dan() -> None:
    q = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    q /= np.linalg.norm(q)
    got = _idx().search(q, top_k=3)
    assert [cid for cid, _ in got] == ["a", "b", "c"]
    assert got[0][1] > got[1][1] > got[2][1]


def test_search_cat_dung_top_k() -> None:
    # top_k lớn hơn số hàng phải kẹp lại, không ném IndexError: pool 1.000 trên corpus
    # nhỏ (test, hay Document lẻ) là chuyện thường.
    assert len(_idx().search(np.array([1, 0, 0], dtype=np.float32), top_k=99)) == 3


def test_len_lech_bao_ngay() -> None:
    with pytest.raises(AssertionError):
        DenseIndex(["a"], np.eye(3, dtype=np.float16))


def test_rrf_uu_tien_dong_thuan() -> None:
    """Mục được CẢ HAI nhánh xếp cao phải thắng mục một nhánh xếp số 1 mà nhánh kia bỏ xa.

    Đây là toàn bộ lý do dùng RRF, nhưng "xếp cao" phải hiểu đúng: với k=60, chênh giữa
    hạng 1 và hạng 2 chỉ là 1/61 vs 1/62 = 0,03%. Nên hai danh sách đảo ngược nhau trên
    3 mục thì mọi mục BẰNG ĐIỂM, không có ai thắng. Đồng thuận chỉ thắng khi nhánh còn
    lại xếp đối thủ xuống thật sâu — chính là tình huống pool 50 trên 287k chunk.
    """
    a = ["x", "y"]
    b = ["y", *(f"d{i}" for i in range(48)), "x"]  # x tụt hạng 50 ở nhánh hai
    got = rrf([a, b])
    assert got[0] == "y"
    assert got.index("y") < got.index("x")


def test_rrf_giu_muc_chi_co_o_mot_nhanh() -> None:
    # Mục chỉ một nhánh tìm ra vẫn phải có mặt — đó là cơ chế bù chéo nâng trần (§15.8).
    assert set(rrf([["x"], ["y"]])) == {"x", "y"}
