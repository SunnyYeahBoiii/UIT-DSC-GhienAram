"""Test clean — xoá noise, giữ [...], lọc SFT."""
from legalqa.clean import clean_reference_answer, is_usable_for_sft


def test_remove_image_captions():
    """Xoá (Hình từ Internet)."""
    answer = "Nội dung trả lời. (Hình từ Internet)\nKết thúc."
    cleaned = clean_reference_answer(answer)
    assert "(Hình từ Internet)" not in cleaned
    assert "Nội dung trả lời" in cleaned


def test_remove_image_captions_variant():
    """Xoá (Nguồn ảnh: Internet)."""
    answer = "Trả lời. (Nguồn ảnh: Internet) End."
    cleaned = clean_reference_answer(answer)
    assert "(Nguồn ảnh: Internet)" not in cleaned


def test_remove_image_captions_illustration():
    """Xoá (Ảnh minh họa)."""
    answer = "Trả lời. (Ảnh minh họa) End."
    cleaned = clean_reference_answer(answer)
    assert "(Ảnh minh họa)" not in cleaned


def test_remove_download_here():
    """Xoá '(tại đây)'."""
    answer = "Tải tài liệu (tại đây) để tham khảo."
    cleaned = clean_reference_answer(answer)
    assert "tại đây" not in cleaned.lower()


def test_keep_ellipsis_legal_citation():
    """Giữ nguyên [...] — trích luật hợp lệ, không phải noise."""
    answer = "Theo Điều 1 [...] thì nội dung vẫn được giữ nguyên."
    cleaned = clean_reference_answer(answer)
    assert "[...]" in cleaned


def test_remove_standalone_at_day():
    """Xoá 'Tại đây.' đứng riêng dòng."""
    answer = "Nội dung.\nTại đây.\nKết luận."
    cleaned = clean_reference_answer(answer)
    assert "Tại đây" not in cleaned


def test_is_usable_normal():
    """Answer bình thường ≥ 200 ký tự, không ... → True."""
    assert is_usable_for_sft("question", "a" * 200)


def test_is_usable_too_short():
    """Answer < 200 ký tự → False."""
    assert not is_usable_for_sft("question", "ngắn")


def test_is_usable_trailing_ellipsis():
    """Answer kết thúc bằng ... → False."""
    assert not is_usable_for_sft("question", "nội dung " * 30 + "...")


def test_is_usable_empty():
    """Answer rỗng → False."""
    assert not is_usable_for_sft("q", "")


def test_caption_anh_khong_phan_biet_hoa_thuong():
    """Bản trước chỉ bắt "(Hình từ Internet)" nên lọt 135/6.929 dòng (1,9%) qua 4 biến
    thể chữ thường. Fixture dùng đúng 4 biến thể đã đo được, không phải biến thể bịa."""
    for bien_the in ("(Hình từ internet)", "(hình từ Internet)", "(hình từ internet)"):
        out = clean_reference_answer(f"Căn cứ Điều 5.\n{bien_the}\nNội dung tiếp theo.")
        assert "nternet" not in out, bien_the


def test_nbsp_thanh_space_chu_khong_bi_xoa():
    """948 lần U+00A0 trong tập. Xoá hẳn là dính hai từ lại: "kinh doanh" -> "kinhdoanh"."""
    out = clean_reference_answer("Căn cứ Điều 5 về kinh doanh." + "x" * 200)
    assert "kinh doanh" in out
    assert " " not in out


def test_bom_bi_xoa_de_duoi_cau_doc_dung():
    """90 lần U+FEFF nằm SAU dấu chấm cuối, làm `endswith(".")` trả False — mọi phép đo
    hình dạng đuôi đều lệch, và token cuối của target là ký tự vô nghĩa."""
    out = clean_reference_answer("Căn cứ Điều 5. Nội dung." + "x" * 200 + ".﻿")
    assert out.endswith(".")


def test_teaser_cau_hoi_cuoi_bi_cat_nhung_giu_lai_phan_than():
    """14,2% target kết thúc bằng tiêu đề bài kế tiếp. Cắt DÒNG chứ không bỏ mẫu."""
    than = "Căn cứ Điều 5 Nghị định 90/2017/NĐ-CP thì mức phạt là 5.000.000 đồng. " * 5
    out = clean_reference_answer(than + "\nLàng nghề cần đáp ứng điều kiện gì?")
    assert "Làng nghề" not in out
    assert "5.000.000 đồng" in out


def test_teaser_khong_cat_duoc_thi_bi_loai():
    """15/986 mẫu chỉ có một dòng và dòng đó là câu hỏi — không cứu được bằng cắt dòng."""
    a = "Mẫu 08/CK-TNCN bản cam kết tổng thu nhập chưa đến mức chịu thuế như thế nào? " * 4
    assert not is_usable_for_sft("q", clean_reference_answer(a))
