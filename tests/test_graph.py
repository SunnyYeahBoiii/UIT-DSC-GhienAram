from legalqa.corpus import Document, parse_articles
from legalqa.evidence import Evidence
from legalqa.graph import DocRef, LegalGraph, expand_with_graph, parse_refs


def _doc(doc_id: str, slug: str, passage: str) -> Document:
    d = Document(doc_id=doc_id, passage=passage, slug=slug)
    d.articles = parse_articles(passage)
    return d


# ---- trích tham chiếu ------------------------------------------------------


def test_ref_dang_so_hieu():
    (r,) = parse_refs("Xử lý theo Điều 5 Nghị định số 100/2019/NĐ-CP.")
    assert (r.article, r.doc.number) == ("5", "100/2019/NĐ-CP")


def test_ref_so_hieu_quoc_hoi_duoi_la_so():
    """QH14 kết thúc bằng số — regex thiếu `[0-9]*` thì mọi Luật vô hình."""
    (r,) = parse_refs("Theo Điều 7 Luật 45/2019/QH14.")
    assert r.doc.number == "45/2019/QH14"


def test_ref_dang_ten_kem_nam():
    """42,1% reference answer trích theo tên, không có số hiệu."""
    (r,) = parse_refs("Theo Điều 127 Luật Các tổ chức tín dụng 2010 quy định")
    assert (r.article, r.doc.name, r.doc.year) == ("127", "Luật Các tổ chức tín dụng", "2010")


def test_ref_luat_nay_khong_phai_canh_ra_ngoai():
    assert parse_refs("theo quy định tại Điều 130a của Luật này") == []


def test_ref_khong_ghep_qua_dau_cham():
    assert parse_refs("Theo Điều 5 Luật này. Xem Nghị định 100/2019/NĐ-CP.") == []


def test_ref_bo_trung_giu_thu_tu():
    t = "Điều 5 Nghị định 100/2019/NĐ-CP, Điều 3 Luật 45/2019/QH14; Điều 5 Nghị định 100/2019/NĐ-CP"
    assert [r.article for r in parse_refs(t)] == ["5", "3"]


# ---- resolve ---------------------------------------------------------------


def _graph():
    goc = _doc(
        "GOC",
        "Luat-cac-to-chuc-tin-dung-2010-108079",
        "Điều 126. Cấm\nNội dung cấm.\nĐiều 127. Hạn chế cấp tín dụng\n"
        "1. Không được cấp tín dụng cho:\nb) Kế toán trưởng;\n",
    )
    sua = _doc(
        "SUA",
        "Luat-sua-doi-Luat-cac-to-chuc-tin-dung-2017-356283",
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Các tổ chức tín dụng\n"
        "19. Sửa đổi khoản 1 Điều 126 như sau:\n“Nội dung mới của 126.”\n"
        "20. Sửa đổi, bổ sung điểm b khoản 1, bổ sung khoản 5 vào Điều 127 như sau:\n"
        "“b) Kế toán trưởng, Chủ tịch Hội đồng quản trị;\n"
        "5. Tổng mức dư nợ không vượt quá 20% vốn tự có.”\n"
        "21. Bãi bỏ Điều 200.\n",
    )
    g = LegalGraph()
    g.build([goc, sua])
    return g, {"GOC": goc, "SUA": sua}


def test_resolve_theo_ten_va_nam_qua_slug():
    g, _ = _graph()
    assert g.resolve(DocRef(name="Luật Các tổ chức tín dụng", year="2010")) == "GOC"


def test_resolve_ten_khac_thu_tu_tu_van_khop():
    """Answer viết "Luật Các tổ chức tín dụng sửa đổi 2017", Slug viết
    "Luat-sua-doi-Luat-cac-to-chuc-tin-dung-2017" — khác thứ tự, cùng tập từ."""
    g, _ = _graph()
    assert g.resolve(DocRef(name="Luật Các tổ chức tín dụng sửa đổi", year="2017")) == "SUA"


def test_resolve_sai_nam_thi_khong_tra_ve():
    g, _ = _graph()
    assert g.resolve(DocRef(name="Luật Các tổ chức tín dụng", year="1999")) is None


def test_before_year_chan_van_ban_tu_sua_chinh_no():
    g, _ = _graph()
    assert g.resolve(DocRef(name="Luật Các tổ chức tín dụng"), before_year="2017") == "GOC"


# ---- cạnh sửa đổi ----------------------------------------------------------


def test_dung_canh_sua_doi_tu_tieu_de_va_khoan():
    g, _ = _graph()
    (am,) = g.amendments_of("GOC", "127")
    assert (am.src_doc_id, am.src_article, am.src_khoan) == ("SUA", "1", "20")
    assert g.amendments_of("GOC", "126")[0].src_khoan == "19"


def test_khoan_lay_tron_ca_phan_trich_nhieu_dong():
    """Nội dung sửa đổi nằm trong ngoặc kép trải nhiều dòng — cắt theo dòng là mất."""
    g, docs = _graph()
    (am,) = g.amendments_of("GOC", "127")
    t = docs["SUA"].passage[am.start : am.end]
    assert "Chủ tịch Hội đồng quản trị" in t
    assert "Tổng mức dư nợ" in t
    assert "Bãi bỏ Điều 200" not in t  # dừng đúng ở khoản 21


def test_evidence_sua_doi_di_kem_nhan_quan_he():
    g, docs = _graph()
    seed = Evidence("GOC", "Luật Các TCTD", "127", "Điều 127. Hạn chế cấp tín dụng")
    (ev,) = expand_with_graph(g, docs, seed)
    assert ev.doc_id == "SUA"
    assert "Sửa đổi Điều 127" in ev.relation and "khoản 20" in ev.relation
    assert "Chủ tịch Hội đồng quản trị" in ev.text


def test_dieu_khong_bi_sua_thi_khong_sinh_evidence_thua():
    g, docs = _graph()
    seed = Evidence("GOC", "Luật Các TCTD", "999", "Điều 999. Không tồn tại")
    assert expand_with_graph(g, docs, seed) == []


# ---- hop theo cross-reference ---------------------------------------------


def _corpus_hop():
    a = _doc("A", "Nghi-dinh-10-2020-ND-CP-1", "Số: 10/2020/NĐ-CP\nĐiều 1. Phạm vi\nÁp dụng Điều 5 Nghị định 100/2019/NĐ-CP.")
    b = _doc("B", "Nghi-dinh-100-2019-ND-CP-2", "Số: 100/2019/NĐ-CP\nĐiều 5. Mức phạt\nPhạt theo Điều 7 Luật 45/2019/QH14.\nĐiều 6. Hiệu lực\nTừ 2020.")
    c = _doc("C", "Luat-45-2019-QH14-3", "Số: 45/2019/QH14\nĐiều 7. Nguyên tắc\nTuân thủ pháp luật.")
    g = LegalGraph()
    g.build([a, b, c])
    return g, {d.doc_id: d for d in (a, b, c)}


def test_hop_lay_tron_dieu_duoc_goi_ten():
    g, docs = _corpus_hop()
    seed = Evidence("A", "Nghị định A", "1", docs["A"].passage)
    out = expand_with_graph(g, docs, seed, max_hop=1)
    assert [(e.doc_id, e.article_number) for e in out] == [("B", "5")]
    assert "Điều 6" not in out[0].text  # chỉ Điều được gọi tên, không cả văn bản


def test_hai_vong_di_tiep_tu_dieu_vua_lay():
    g, docs = _corpus_hop()
    seed = Evidence("A", "Nghị định A", "1", docs["A"].passage)
    out = expand_with_graph(g, docs, seed, max_hop=2)
    assert [(e.doc_id, e.article_number) for e in out] == [("B", "5"), ("C", "7")]


def test_visited_set_chan_quay_vong():
    a = _doc("A", "Nghi-dinh-10-2020-ND-CP-1", "Số: 10/2020/NĐ-CP\nĐiều 1. Phạm vi\nXem Điều 5 Nghị định 100/2019/NĐ-CP.")
    b = _doc("B", "Nghi-dinh-100-2019-ND-CP-2", "Số: 100/2019/NĐ-CP\nĐiều 5. Mức phạt\nNhư Điều 1 Nghị định 10/2020/NĐ-CP.")
    g = LegalGraph()
    g.build([a, b])
    seed = Evidence("A", "Nghị định A", "1", a.passage[a.articles[0].start :])
    out = expand_with_graph(g, {"A": a, "B": b}, seed, max_hop=2)
    assert [(e.doc_id, e.article_number) for e in out] == [("B", "5")]


def test_max_extra_chan_bung_no():
    g, docs = _corpus_hop()
    seed = Evidence("A", "Nghị định A", "1", docs["A"].passage)
    assert len(expand_with_graph(g, docs, seed, max_hop=2, max_extra=1)) == 1


def test_so_hieu_ngoai_corpus_thi_bo_qua():
    g, docs = _corpus_hop()
    seed = Evidence("A", "x", "1", "Theo Điều 5 Nghị định 999/1999/NĐ-CP.")
    assert expand_with_graph(g, docs, seed) == []


# ---- mục tiêu số nhiều -----------------------------------------------------


def test_tach_muc_tieu_khi_liet_ke_song_song():
    from legalqa.graph import _tach_muc_tieu

    assert _tach_muc_tieu(
        "LUẬT THUẾ GIÁ TRỊ GIA TĂNG, LUẬT THUẾ TIÊU THỤ ĐẶC BIỆT VÀ LUẬT QUẢN LÝ THUẾ"
    ) == ["LUẬT THUẾ GIÁ TRỊ GIA TĂNG", "LUẬT THUẾ TIÊU THỤ ĐẶC BIỆT", "LUẬT QUẢN LÝ THUẾ"]


def test_khong_tach_o_ten_loai_khong_phai_liet_ke():
    """"NGHỊ ĐỊNH ... HƯỚNG DẪN LUẬT CHĂN NUÔI" có MỘT mục tiêu. Cắt ở mọi tên loại
    thì sinh cạnh nghị-định-sửa-luật không có thật — 51 ca hỏng đúng kiểu này."""
    from legalqa.graph import _tach_muc_tieu

    t = "NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH LUẬT CHĂN NUÔI"
    assert _tach_muc_tieu(t) == [t]


def test_dau_phay_trong_ten_khong_phai_cho_tach():
    from legalqa.graph import _tach_muc_tieu

    assert _tach_muc_tieu("LUẬT CÁN BỘ, CÔNG CHỨC VÀ LUẬT VIÊN CHỨC") == [
        "LUẬT CÁN BỘ, CÔNG CHỨC",
        "LUẬT VIÊN CHỨC",
    ]


def test_van_ban_sua_nhieu_luat_sinh_canh_cho_tung_luat():
    goc1 = _doc("L1", "Luat-Dau-tu-cong-2019-1", "Điều 6. Phân loại\nNội dung.")
    goc2 = _doc("L2", "Luat-Nha-o-2014-2", "Điều 9. Công nhận\nNội dung.")
    sua = _doc(
        "S",
        "Luat-sua-doi-Luat-Dau-tu-cong-Luat-Nha-o-2024-3",
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Đầu tư công, Luật Nhà ở\n"
        "1. Sửa đổi Điều 6 như sau:\n“Nội dung mới.”\n",
    )
    g = LegalGraph()
    g.build([goc1, goc2, sua])
    assert g.amendments_of("L1", "6")[0].src_khoan == "1"
    assert g.amendments_of("L2", "6")[0].src_khoan == "1"


def test_truy_van_qua_ngan_khong_khop_bua():
    """Mảnh cắt cụt "Thông tư" là tập con của hàng trăm Slug — phải trả None."""
    g, _ = _corpus_hop()
    assert g.resolve(DocRef(name="Nghị định")) is None
    assert g.resolve(DocRef(name="Nghị định", year="2019")) == "B"


def test_moi_dieu_co_muc_tieu_rieng():
    """"LUẬT SỬA ĐỔI 37 LUẬT LIÊN QUAN ĐẾN QUY HOẠCH": mỗi Điều sửa một luật khác.
    Lấy một mục tiêu chung cho cả văn bản thì 3.308/5.760 cạnh đổ nhầm chỗ."""
    a = _doc("A", "Luat-Giao-thong-duong-bo-2008-1", "Điều 8. Hành vi cấm\nNội dung.")
    b = _doc("B", "Luat-Bao-ve-moi-truong-2014-2", "Điều 8. Điều kiện\nNội dung.")
    sua = _doc(
        "S",
        "Luat-sua-doi-cac-Luat-lien-quan-quy-hoach-2018-3",
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Giao thông đường bộ\n"
        "1. Sửa đổi Điều 8 như sau:\n“Nội dung giao thông.”\n"
        "Điều 2. Sửa đổi, bổ sung một số điều của Luật Bảo vệ môi trường\n"
        "1. Sửa đổi Điều 8 như sau:\n“Nội dung môi trường.”\n",
    )
    g = LegalGraph()
    g.build([a, b, sua])
    assert [x.src_article for x in g.amendments_of("A", "8")] == ["1"]
    assert [x.src_article for x in g.amendments_of("B", "8")] == ["2"]
    assert "giao thông" in sua.passage[g.amendments_of("A", "8")[0].start : g.amendments_of("A", "8")[0].end]


def test_dieu_khong_khai_muc_tieu_thi_dung_muc_tieu_tieu_de():
    g, docs = _graph()
    assert g.amendments_of("GOC", "127")[0].src_doc_id == "SUA"


def test_dieu_khai_muc_tieu_tra_truot_thi_bo_han():
    """Rơi về mục tiêu chung là gán nhầm sang văn bản khác — tệ hơn mất cạnh."""
    a = _doc("A", "Luat-Giao-thong-duong-bo-2008-1", "Điều 8. Hành vi cấm\nNội dung.")
    sua = _doc(
        "S",
        "Luat-sua-doi-cac-Luat-lien-quan-quy-hoach-2018-3",
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Giao thông đường bộ\n"
        "1. Sửa đổi Điều 8 như sau:\n“Nội dung giao thông.”\n"
        "Điều 2. Sửa đổi, bổ sung một số điều của Luật Không Có Trong Corpus\n"
        "1. Sửa đổi Điều 8 như sau:\n“Nội dung khác.”\n",
    )
    g = LegalGraph()
    g.build([a, sua])
    assert [x.src_article for x in g.amendments_of("A", "8")] == ["1"]


def test_heading_cut_cut_thi_dung_muc_tieu_chung():
    """"Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số" — wrap cắt mất số hiệu.
    Chuỗi còn lại thành tập từ {nghi, dinh, so}, đủ 3 từ để qua guard và khớp trúng
    Slug có "xo-so" = Nghị định kinh doanh xổ số. Heading nghèo chữ là KHÔNG KHAI."""
    a = _doc("A", "Nghi-dinh-113-2017-ND-CP-huong-dan-Luat-Hoa-chat-1",
             "Số: 113/2017/NĐ-CP\n\nĐiều 9. Hóa chất\nNội dung.")
    xs = _doc("X", "Nghi-dinh-kinh-doanh-xo-so-2016-2", "Điều 9. Xổ số\nNội dung.")
    sua = _doc("S", "Nghi-dinh-82-2022-ND-CP-sua-doi-Nghi-dinh-113-2017-ND-CP-3",
        "NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA NGHỊ ĐỊNH SỐ "
        "113/2017/NĐ-CP NGÀY 09 THÁNG 10 NĂM 2017 CỦA CHÍNH PHỦ\n\n"
        "Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số\n"
        "1. Sửa đổi Điều 9 như sau:\n“Nội dung mới.”\n")
    g = LegalGraph(); g.build([a, xs, sua])
    assert [x.src_article for x in g.amendments_of("A", "9")] == ["1"]
    assert g.amendments_of("X", "9") == []


def test_khoan_tu_khai_so_hieu_khi_dieu_khong_khai():
    """"Điều 12. Bãi bỏ một số văn bản thuộc lĩnh vực mỹ phẩm" không khai mục tiêu.
    Kế thừa mục tiêu Điều 1 là gán sang hẳn văn bản khác; khoản tự nêu số hiệu thì
    tin khoản."""
    a = _doc("A", "Nghi-dinh-dieu-kien-san-xuat-2016-1", "Điều 5. Điều kiện\nNội dung.")
    b = _doc("B", "Thong-tu-06-2011-TT-BYT-quan-ly-my-pham-2",
             "Số: 06/2011/TT-BYT\n\nĐiều 7. Mỹ phẩm\nNội dung.")
    sua = _doc("S", "Nghi-dinh-sua-doi-quy-dinh-lien-quan-2018-3",
        "Điều 1. Sửa đổi, bổ sung một số điều của Nghị định điều kiện sản xuất\n"
        "1. Sửa đổi Điều 5 như sau:\n“Nội dung mới.”\n"
        "Điều 12. Bãi bỏ một số văn bản, quy định thuộc lĩnh vực mỹ phẩm\n"
        "1. Bãi bỏ Điều 7 của Thông tư số 06/2011/TT-BYT ngày 25 tháng 01 năm 2011.\n")
    g = LegalGraph(); g.build([a, b, sua])
    assert [x.src_article for x in g.amendments_of("B", "7")] == ["12"]
    assert [x.src_article for x in g.amendments_of("A", "5")] == ["1"]


def test_khoan_khai_so_hieu_khong_de_lan_khi_dieu_da_khai():
    """Điều đã khai mục tiêu thì số hiệu trong khoản thường là chú thích lịch sử:
    "(được sửa đổi theo quy định tại khoản 3 Điều 1 của Thông tư số 26/2017/TT-NHNN)".
    Tin nó là gán nhầm ngược lại."""
    a = _doc("A", "Thong-tu-19-2016-TT-NHNN-hoat-dong-the-ngan-hang-1",
             "Số: 19/2016/TT-NHNN\n\nĐiều 10. Thẻ\nNội dung.")
    b = _doc("B", "Thong-tu-26-2017-TT-NHNN-sua-doi-Thong-tu-19-2016-2",
             "Số: 26/2017/TT-NHNN\n\nĐiều 3. Khác\nNội dung.")
    sua = _doc("S", "Thong-tu-28-2019-TT-NHNN-sua-doi-Thong-tu-19-2016-3",
        "Điều 1. Sửa đổi, bổ sung một số điều của Thông tư số 19/2016/TT-NHNN\n"
        "1. Sửa đổi khoản 5 Điều 10 (được sửa đổi theo quy định tại khoản 3 Điều 1 "
        "của Thông tư số 26/2017/TT-NHNN) như sau:\n“Nội dung mới.”\n")
    g = LegalGraph(); g.build([a, b, sua])
    assert [x.src_article for x in g.amendments_of("A", "10")] == ["1"]
    assert g.amendments_of("B", "10") == []


def test_khong_rai_moi_muc_tieu_khi_van_ban_co_nhieu_dieu_sua():
    """Văn bản nhiều mục tiêu mà Điều không khai: rải khoản ra mọi mục tiêu là chắc
    chắn sinh cạnh sai, vì mỗi Điều ứng một mục tiêu riêng."""
    a = _doc("A", "Luat-thue-gia-tri-gia-tang-2008-1", "Điều 8. Thuế suất\nNội dung.")
    b = _doc("B", "Luat-quan-ly-thue-2006-2", "Điều 8. Quản lý\nNội dung.")
    sua = _doc("S", "Luat-sua-doi-cac-luat-ve-thue-2014-3",
        "LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA LUẬT THUẾ GIÁ TRỊ GIA TĂNG "
        "VÀ LUẬT QUẢN LÝ THUẾ\n\n"
        "Điều 1. Nội dung\n1. Sửa đổi Điều 8 như sau:\n“Nội dung A.”\n"
        "Điều 2. Nội dung khác\n1. Sửa đổi Điều 8 như sau:\n“Nội dung B.”\n")
    g = LegalGraph(); g.build([a, b, sua])
    assert g.amendments_of("A", "8") == []
    assert g.amendments_of("B", "8") == []


def test_khoan_dang_dao_ngu():
    """"Điều 4 được sửa đổi, bổ sung như sau:" — động từ đứng SAU tên Điều. Thiếu nhánh
    này thì 12 văn bản sửa đổi không sinh được cạnh nào, tổng 871 khoản bị bỏ."""
    a = _doc("A", "Thong-tu-13-2010-TT-BNV-tuyen-dung-cong-chuc-1",
             "Số: 13/2010/TT-BNV\n\nĐiều 4. Thi tuyển\nNội dung.")
    sua = _doc("S", "Thong-tu-03-2019-TT-BNV-sua-doi-Thong-tu-13-2010-2",
        "Điều 1. Sửa đổi, bổ sung một số điều của Thông tư số 13/2010/TT-BNV\n"
        "1. Điều 4 được sửa đổi, bổ sung như sau:\n“Nội dung mới.”\n")
    g = LegalGraph(); g.build([a, sua])
    assert [x.src_khoan for x in g.amendments_of("A", "4")] == ["1"]


def test_dieu_dich_khai_o_heading_khoan_khong_neu():
    """"Điều 1. Sửa đổi, bổ sung Điều 3 của Nghị định số 65/2022/NĐ-CP như sau:" rồi
    khoản chỉ nói "Bổ sung khoản 1a như sau:" — Điều đích chỉ có ở heading."""
    a = _doc("A", "Nghi-dinh-65-2022-ND-CP-chao-ban-trai-phieu-1",
             "Số: 65/2022/NĐ-CP\n\nĐiều 3. Giải thích từ ngữ\nNội dung.")
    sua = _doc("S", "Nghi-dinh-sua-doi-Nghi-dinh-65-2022-ND-CP-2",
        "Điều 1. Sửa đổi, bổ sung Điều 3 của Nghị định số 65/2022/NĐ-CP như sau:\n"
        "1. Bổ sung khoản 1a như sau:\n“1a. Nội dung.”\n"
        "2. Sửa đổi điểm b khoản 3 như sau:\n“b) Nội dung.”\n")
    g = LegalGraph(); g.build([a, sua])
    assert [x.src_khoan for x in g.amendments_of("A", "3")] == ["1", "2"]


def test_heading_khai_tron_khong_chia_khoan():
    """"Điều 1. Bổ sung khoản 7 vào Điều 3 Nghị định số 56/2011/NĐ-CP như sau:" rồi viết
    thẳng nội dung, không có khoản nào. 49 Điều kiểu này — cả Điều là một cạnh."""
    a = _doc("A", "Nghi-dinh-56-2011-ND-CP-phu-cap-uu-dai-1",
             "Số: 56/2011/NĐ-CP\n\nĐiều 3. Mức phụ cấp\nNội dung.")
    sua = _doc("S", "Nghi-dinh-sua-doi-Nghi-dinh-56-2011-ND-CP-2",
        "NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA NGHỊ ĐỊNH SỐ 56/2011/NĐ-CP\n\n"
        "Điều 1. Bổ sung khoản 7 vào Điều 3 Nghị định số 56/2011/NĐ-CP như sau:\n"
        "“7. Mức phụ cấp 40% áp dụng đối với công chức y tế.”\n"
        "Điều 2. Tổ chức thực hiện\n1. Các cơ sở y tế công lập căn cứ quy định.\n")
    g = LegalGraph(); g.build([a, sua])
    (am,) = g.amendments_of("A", "3")
    assert (am.src_article, am.src_khoan) == ("1", "")
    assert "Mức phụ cấp 40%" in sua.passage[am.start : am.end]
