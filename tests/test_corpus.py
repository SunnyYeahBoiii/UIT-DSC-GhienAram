"""Test corpus — title_from_slug và Document.title."""
from legalqa.corpus import Document, parse_articles, title_from_slug


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


def test_dewrap_bo_dong_trong_giu_xuong_dong():
    """Dòng trống bị bỏ (59,3% số dòng), xuống dòng đơn giữ nguyên — nó phân khoản."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("Điều 1. Phạm vi\n\n1. Khoản một") == "Điều 1. Phạm vi\n1. Khoản một"


def test_dewrap_giu_ngat_sau_dau_cau():
    """Wrap sau dấu kết câu + trước chữ hoa = ngắt đoạn thật, xuống dòng chứ không nối."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("nghề nghiệp;\r\n\nCăn cứ Nghị định") == "nghề nghiệp;\nCăn cứ Nghị định"


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


def test_dewrap_gop_so_khoan_dung_mot_minh():
    """Dòng chỉ có "1." rồi "\\n\\n" — số khoản không bao giờ là đoạn riêng, phải nối."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("Điều 32. A\n\n1.\n\nCăn cứ vào") == "Điều 32. A\n1. Căn cứ vào"
    assert dewrap_passage("Mở đầu\n\na)\n\nNội dung") == "Mở đầu\na) Nội dung"


def test_dewrap_khong_gop_khoan_binh_thuong():
    """Khoản đã có nội dung cùng dòng thì giữ nguyên ranh giới đoạn."""
    from legalqa.corpus import dewrap_passage

    assert (
        dewrap_passage("Điều 1. A\n\n1. Khoản một\n\n2. Khoản hai")
        == "Điều 1. A\n1. Khoản một\n2. Khoản hai"
    )


def test_dewrap_bo_dong_chi_chua_dau_cham():
    """Dòng `...` là chỗ thuvienphapluat lược nội dung — 85.602 dòng, phải bỏ."""
    from legalqa.corpus import dewrap_passage

    assert dewrap_passage("Cấp 1: nhẹ.\n\n...\n\n...\n\nCấp 3: nặng.") == (
        "Cấp 1: nhẹ.\nCấp 3: nặng."
    )


def test_dewrap_khong_con_dong_trong_hay_space_thua():
    """Sau clean: không dòng trống, không space đầu/cuối dòng, không \\r."""
    from legalqa.corpus import dewrap_passage

    out = dewrap_passage("  BỘ Y TẾ  \r\n\n  Số: 1\n\n\n\n   THÔNG TƯ   \n\n")
    assert "\r" not in out
    assert "\n\n" not in out
    assert all(line == line.strip() and line for line in out.split("\n"))


def test_dieu_cuoi_khong_nuot_phu_luc():
    """Điều cuối phải dừng trước PHỤ LỤC, không lấy tới hết Passage."""
    body = " ".join(f"nội dung {i}" for i in range(60))
    phu_luc = " ".join(f"biểu mẫu {i}" for i in range(150))
    p = f"Điều 1. Phạm vi\n{body}\nPHỤ LỤC I\n{phu_luc}"
    a = parse_articles(p)[-1]
    assert "PHỤ LỤC" not in p[a.start : a.end]
    assert "nội dung 59" in p[a.start : a.end]


def test_dong_bang_giua_noi_dung_khong_cat_dieu():
    """Bỏ đi < 200 từ thì không phải phụ lục — chỉ là dòng "Bảng 1" giữa nội dung."""
    body = " ".join(f"nội dung {i}" for i in range(60))
    p = f"Điều 1. Phạm vi\n{body}\nBảng 1\nmột hai ba bốn năm\n{body}"
    a = parse_articles(p)[-1]
    assert a.end == len(p)


def test_bo_moc_dieu_gia_do_cau_bi_wrap():
    """"...một số khoản của\nĐiều 10 như sau:" — chữ "Điều" rơi vào đầu dòng do wrap.
    Nhận nó làm mốc thì Điều 1 bị cắt đôi, mất sạch khoản sau, và đẻ ra Điều rác mang
    số của BẢN GỐC. Đo trên corpus: 377 ca kiểu này."""
    p = (
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Kiểm toán nhà nước\n"
        "1. Bổ sung khoản 2a vào sau khoản 2 Điều 3 như sau:\n"
        "“2a. Cơ quan, tổ chức có liên quan.”\n"
        "2. Sửa đổi, bổ sung một số khoản của\n"
        "Điều 10 như sau:\n"
        "a) Sửa đổi khoản 3 như sau:\n"
        "“3. Xem xét, quyết định việc kiểm toán.”\n"
        "Điều 2. Hiệu lực thi hành\n"
        "Luật này có hiệu lực từ 01/7/2020.\n"
    )
    arts = parse_articles(p)
    assert [a.number for a in arts] == ["1", "2"]
    # Điều 1 phải ôm trọn cả khoản 2 — đây là khoản bị mất khi nhận mốc giả
    assert "Xem xét, quyết định việc kiểm toán" in p[arts[0].start : arts[0].end]


def test_giu_dieu_that_thieu_dau_cham():
    """"Điều 34 Trách nhiệm của Ủy ban nhân dân" — 839 Điều thật viết thiếu dấu chấm.
    Luật loại mốc giả không được đụng vào nhóm này: tên Điều viết HOA hoặc ở dòng sau."""
    p = "Điều 34 Trách nhiệm của Ủy ban\nNội dung.\nĐiều 35\n1. Khoản một.\n"
    assert [a.number for a in parse_articles(p)] == ["34", "35"]


def test_bo_moc_dieu_trong_liet_ke():
    """"Điều 71, điểm b khoản 4 Điều 103 Luật Các tổ chức tín dụng" — 244 ca liệt kê
    nhiều Điều trong một câu, không phải mốc mở Điều mới."""
    p = "Điều 5. Bãi bỏ\nBãi bỏ các quy định sau:\nĐiều 71, điểm b khoản 4 Điều 103.\n"
    assert [a.number for a in parse_articles(p)] == ["5"]
