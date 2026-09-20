from legalqa.sanitize import cat_lap


def test_cum_ngan_lap_trong_dong():
    # 151451: "bảo quản, " ×106 rồi hết max-new.
    t = "Căn cứ Điều 5 quy định như sau:\n1. Kho " + "bảo quản, " * 50 + "bảo"
    assert cat_lap(t) == "Căn cứ Điều 5 quy định như sau:\n1. Kho bảo quản,"


def test_dong_co_so_thu_tu_troi():
    # 80189: số thứ tự tăng nhưng nội dung xoay vòng 2 dòng.
    dong = ["(1) Mẫu A tại Việt Nam", "(2) Mẫu B tại nước ngoài"]
    for i in range(3, 12):
        dong.append(f"({i}) Mẫu C  tại Việt Nam" if i % 2 else f"({i}) Mẫu C tại nước ngoài")
    ra = cat_lap("\n".join(dong))
    assert ra.count("Mẫu C") == 2
    assert ra.startswith("(1) Mẫu A tại Việt Nam\n(2) Mẫu B tại nước ngoài\n(3) Mẫu C  tại Việt Nam")


def test_khoi_nhieu_dong_va_duoi_do_dang():
    # 18645: khối lặp nhiều bản, bản cuối bị cắt giữa chừng.
    khoi = "Căn cứ khoản 6 Điều 10 quy định:\n...\n6. Xâm phạm thân thể.\nNhư vậy, có bị cấm.\n"
    t = "Mở đầu.\n" + khoi * 5 + "Căn cứ khoản 6 Điều"
    assert cat_lap(t) == ("Mở đầu.\n" + khoi).rstrip()


def test_giu_hai_ban_va_so_dieu_khac_nhau():
    # Đáp án BTC nhắc lại tên điều ngay sau "như sau:" — 2 bản thì giữ.
    t = "quy định như sau:\nĐiều 28. Thẩm quyền thanh lý\nĐiều 28. Thẩm quyền thanh lý\n1. Nội dung."
    assert cat_lap(t) == t
    # Số không ở đầu dòng không bị chuẩn hoá thành lặp giả.
    t2 = "Theo Điều 1. Điều 2. Điều 3. Điều 4."
    assert cat_lap(t2) == t2


def test_so_thu_tu_kem_chu():
    # b535-rrft 3055: "2z. / 3a. / 3b. X;" số kèm chữ trôi, nội dung giống hệt.
    dong = ["Căn cứ Điều 30 như sau:"] + [f"{i}{c}. Dán nhãn không đúng quy định;" for i in (2, 3) for c in "xyzabc"]
    ra = cat_lap("\n".join(dong))
    assert ra.count("Dán nhãn") == 1
    # "a)" "b)" khác nội dung thì giữ nguyên.
    t = "như sau:\na) Tên doanh nghiệp;\nb) Mã số thuế;\nc) Địa chỉ trụ sở."
    assert cat_lap(t) == t


def test_va_cau_dan():
    from legalqa.sanitize import va_cau_dan
    p = ("Căn cứ pháp luật:\n[1] Căn cứ chính\nĐiều 43, 78/2015/NĐ-CP\nĐiều 43. Đăng ký thay đổi\n"
         "1. Hồ sơ gồm:\na) Thông báo;\n[2] Liên quan\nĐiều 44, X\nKhác\n\nCâu hỏi: Mẫu thông báo\nTrả lời: ")
    ra = va_cau_dan("Căn cứ theo mẫu thông báo như sau:", p)
    assert ra == "Căn cứ theo mẫu thông báo như sau:\nĐiều 43. Đăng ký thay đổi\n1. Hồ sơ gồm:\na) Thông báo;"
    dai = "Căn cứ Điều 5 như sau: " + "x " * 60
    assert va_cau_dan(dai, p) == dai


def test_bo_so_thu_tu_treo_cuoi():
    # 3055: đơn vị lặp bắt đầu giữa dòng "1a." nên bản giữ lại kết bằng "1b." treo.
    t = "như sau:\n\"1a. Dán nhãn có thời hạn, dán nhãn sai;\n" + "".join(f"1{c}. Dán nhãn sai;\n" for c in "bcdefgh")
    assert cat_lap(t) == "như sau:\n\"1a. Dán nhãn có thời hạn, dán nhãn sai;"
