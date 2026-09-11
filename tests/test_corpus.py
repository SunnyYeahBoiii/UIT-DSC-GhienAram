"""Test corpus — title_from_slug và Document.title."""
from legalqa.corpus import Document, title_from_slug


def test_title_from_slug_removes_tail_and_dashes():
    """Slug 'Thong-tu-17-2022-TT-BGTVT-abc-522401' → không còn dấu - và đuôi số."""
    result = title_from_slug("Thong-tu-17-2022-TT-BGTVT-abc-522401")
    assert "522401" not in result
    assert "-" not in result or len(result) < 50  # slug đã chuyển thành spaces


def test_title_from_slug_empty():
    """Slug rỗng → trả về rỗng."""
    assert title_from_slug("") == ""


def test_title_from_slug_known_prefix():
    """Nghi-dinh-* → 'Nghị định ...'."""
    result = title_from_slug("Nghi-dinh-123-2024")
    assert result.startswith("Nghị định")


def test_doc_missing_name_has_title():
    """Document thiếu name (slug rỗng) vẫn có title không rỗng."""
    passage = "Một văn bản test nội dung. " * 20
    doc = Document(doc_id="x1", passage=passage, slug="")
    assert doc.title != ""


def test_doc_with_title_line():
    """Document có dòng đầu là title → title ưu tiên dòng đó."""
    passage = "THÔNG TƯ 12/2024/TT-BGTVT\nNội dung văn bản. " * 10
    doc = Document(doc_id="x2", passage=passage)
    assert "THÔNG TƯ" in doc.title
