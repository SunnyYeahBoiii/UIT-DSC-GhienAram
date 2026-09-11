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
