"""Test dựng prompt. Fixture có `\n\n` và space thừa theo AGENTS.md — ngắt dòng phân
khoản là thứ prompt PHẢI giữ, fixture toàn space đơn sẽ không bắt được nếu ai đó
`" ".join(text.split())` cho gọn."""
from __future__ import annotations

from legalqa.evidence import Evidence
from legalqa.prompt import NHAN_GOC, dung_can_cu, dung_prompt


def _ev(**kw) -> Evidence:
    d = dict(
        doc_id="d1",
        doc_title="Luật  Lao động số 45/2019/QH14",
        article_number="127",
        text="1. Khoản  một.\n\n2. Khoản hai.",
        relation="",
    )
    d.update(kw)
    return Evidence(**d)


def test_evidence_goc_duoc_dan_nhan_khong_de_trong():
    # relation rỗng vẫn phải có nhãn: nhãn trống làm model không phân biệt được căn cứ
    # chính với đoạn phụ trợ.
    assert NHAN_GOC in dung_can_cu([_ev()])


def test_ca_dieu_goc_va_khoan_sua_doi_deu_co_mat():
    """§15.7: KHÔNG tự ghép bản hợp nhất. Cả hai đoạn phải xuất hiện nguyên văn."""
    goc = _ev(text="Điều 127. Nội  dung GỐC.")
    sua = _ev(
        doc_id="d2",
        doc_title="Luật sửa đổi số 50/2022/QH15",
        article_number="3",
        text="Sửa đổi khoản 1 Điều 127 như  sau: NỘI DUNG MỚI.",
        relation="Sửa đổi Điều 127 của Luật Lao động — khoản 1 Điều 3",
    )
    out = dung_can_cu([goc, sua])
    assert "Nội  dung GỐC." in out
    assert "NỘI DUNG MỚI." in out
    assert "Sửa đổi Điều 127" in out


def test_giu_dung_thu_tu_truyen_vao():
    """graph.py trả Evidence sửa đổi NGAY SAU seed sinh ra nó. Sắp lại là phá cặp."""
    a = _ev(text="AAA")
    b = _ev(text="BBB", relation="Được dẫn chiếu")
    out = dung_can_cu([a, b])
    assert out.index("AAA") < out.index("BBB")
    assert dung_can_cu([b, a]).index("BBB") < dung_can_cu([b, a]).index("AAA")


def test_giu_nguyen_ngat_dong_va_space_thua():
    ev = _ev(text="1. Một.\n\n2. Hai.")
    out = dung_can_cu([ev])
    assert "1. Một.\n\n2. Hai." in out
    assert "Luật  Lao động" in out  # hai space giữa từ, không được nén


def test_khong_co_so_dieu_thi_khong_bia_ra():
    """28,1% Evidence là cửa sổ lui, `article_number is None` (§7). Prompt chỉ nêu tên
    văn bản — in "Điều None" là dạy model bịa số Điều."""
    out = dung_can_cu([_ev(article_number=None)])
    assert "Điều" not in out.split("\n")[1]
    assert "None" not in out


def test_cau_hoi_dung_sau_can_cu():
    """Token cuối trước khi sinh phải là câu hỏi, không phải văn bản luật."""
    out = dung_prompt("Điều kiện là gì?", [_ev()])
    assert out.index("Căn cứ pháp luật:") < out.index("Câu hỏi: Điều kiện là gì?")
    assert out.rstrip().endswith("Trả lời:")


def test_danh_so_can_cu_tu_1():
    out = dung_can_cu([_ev(), _ev(text="khác")])
    assert out.startswith("[1] ")
    assert "\n[2] " in out
