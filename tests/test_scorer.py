"""Test lớp bọc quanh code chấm của BTC.

Mục đích KHÔNG phải kiểm rouge_score/nltk chạy đúng — đó là code của người khác. Mục
đích là khoá lại những hành vi kỳ quặc mà ta đã đo, để không ai "sửa cho đúng" rồi lệch
khỏi leaderboard mà không biết.
"""
from __future__ import annotations

import pytest

from legalqa.scorer import cham, chuan_bi_nltk

REF = {"1": "Căn cứ khoản 1 Điều 5 Nghị định 90/2017/NĐ-CP quy định về xử phạt."}


@pytest.fixture(scope="module", autouse=True)
def _nltk():
    chuan_bi_nltk()


def test_giong_het_thi_bang_mot():
    s = cham(dict(REF), REF)
    assert s["meteor"] > 0.99 and s["rouge"] == pytest.approx(1.0)


def test_bo_dau_lam_sup_diem():
    """Đo trên 300 mẫu thật: bỏ dấu kéo METEOR 1,0 -> 0,119 và ROUGE-L 1,0 -> 0,143,
    tức ngang mức trả lời nhầm sang câu khác (0,117 / 0,278).

    Nguyên nhân: tokenizer của rouge_score xoá ký tự có dấu nên "Điều" -> ['i','u'], còn
    "Dieu" -> ['dieu'] — hai bên hết khớp nhau. Test này khoá ràng buộc "model phải sinh
    đủ dấu"; nếu ai đó thêm bước chuẩn hoá bỏ dấu ở hậu xử lý, nó phải đỏ.


    Chỉ khẳng định trên METEOR. Trên một câu ngắn như fixture, ROUGE-L bỏ dấu vẫn được
    0,316 vì bốn con số (1, 5, 90, 2017) không bị regex xoá nên khớp nguyên vẹn — tỉ
    trọng chữ số trong một câu quá cao. Con số 0,143 là trung bình trên 300 đáp án thật,
    nơi chữ số loãng đi. Đặt ngưỡng ROUGE ở đây là đặt theo một mẫu, không theo phép đo.
    """
    khong_dau = {"1": "Can cu khoan 1 Dieu 5 Nghi dinh 90/2017/ND-CP quy dinh ve xu phat."}
    assert cham(khong_dau, REF)["meteor"] < 0.3


def test_tokenizer_rouge_cat_vun_tieng_viet():
    """Khoá hành vi thật của leaderboard: ROUGE-L đo mảnh chữ cái ASCII, không đo từ.

    BTC có sẵn `pyvi.ViTokenizer` nhưng đã comment lại (third_party/scoring.py dòng
    27-31). Nếu ai đó bật nó lên hoặc đổi tokenizer, test này đỏ — và phải đỏ, vì lúc đó
    số nội bộ không còn so được với bảng xếp hạng.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
    from rouge_score import tokenize

    assert tokenize.tokenize("Căn cứ Điều 5", None) == ["c", "n", "c", "i", "u", "5"]


def test_chi_cham_tren_id_chung():
    """Lúc train ta cố tình chấm trên tập con dev để tiết kiệm GPU, nên thiếu id là
    bình thường. BTC ném exception (scoring.py dòng 44-46); ở đây lấy giao."""
    s = cham({"1": REF["1"], "thua": "x"}, REF)
    assert s["n"] == 1 and s["rouge"] == pytest.approx(1.0)
