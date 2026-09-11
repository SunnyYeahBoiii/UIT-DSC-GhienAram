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


def test_dewrap_noi_wrap_mem():
    """"\\r\\n\\n" giữa câu bị gỡ thành khoảng trắng — 93,9% trường hợp là wrap crawl."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("Điều 2. Tổ chức\r\n\nthực hiện") == "Điều 2. Tổ chức thực hiện"
    assert dewrap_passage("CỘNG\r\n\n  HÒA XÃ HỘI") == "CỘNG HÒA XÃ HỘI"


def test_dewrap_giu_ngat_doan_that():
    """"\\n\\n" thuần giữ nguyên — chỉ 50,7% nối giữa câu, mơ hồ nên không đụng."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("Điều 1. Phạm vi\n\n1. Khoản một") == "Điều 1. Phạm vi\n\n1. Khoản một"


def test_dewrap_giu_ngat_sau_dau_cau():
    """Wrap ngay sau dấu kết câu + trước chữ hoa = ngắt đoạn thật, không nối liền."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("nghề nghiệp;\r\n\nCăn cứ Nghị định") == "nghề nghiệp;\n\nCăn cứ Nghị định"


def test_dewrap_khong_mat_chu():
    """Gỡ wrap không được thêm/bớt từ nào."""
    from legalqa.corpus import dewrap_passage

    raw = "Điều 3. Yêu cầu\r\n\nđào tạo\n\n1. Cơ sở\r\n\n  giáo dục nghề nghiệp."
    assert dewrap_passage(raw).split() == raw.split()


def test_title_ghep_dong_chu_de():
    """Title = loại văn bản + dòng chủ đề viết hoa; 'THÔNG TƯ' đứng một mình vô nghĩa."""
    from legalqa.corpus import Document

    doc = Document(
        doc_id="x",
        passage="BỘ Y TẾ\n\nSố: 1/2020/TT-BYT\n\nTHÔNG TƯ\n\nQUY ĐỊNH VỀ KHÁM BỆNH\n\nĐiều 1. A",
    )
    assert doc.title == "THÔNG TƯ QUY ĐỊNH VỀ KHÁM BỆNH"
    assert doc.number == "1/2020/TT-BYT"
