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
