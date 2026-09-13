"""Làm sạch Reference answer trước khi SFT.

18,5% reference answer chứa caption ảnh crawl — model học sẽ sinh hallucination.
0,9% chứa link reference ("tại đây") — nhiễu biên tập.
Phải giữ nguyên `[...]` — trích luật lược bỏ hợp lệ, không phải noise.
"""
from __future__ import annotations

import re

# Caption ảnh crawl — xoá nếu đứng riêng dòng hoặc trong ngoặc.
# `re.I` là bắt buộc, không phải cho tiện: bản trước phân biệt hoa/thường và lọt 135/6.929
# dòng (1,9%) qua 4 biến thể — "(Hình từ internet)" 73, "(hình từ Internet)" 43,
# "(hình từ internet)" 19, "Hình từ Internet)" 1. Ngoặc mở để tuỳ chọn vì biến thể thứ tư
# mất ngoặc trái lúc crawl.
_IMG_CAPTION_RE = re.compile(
    r"\s*\(?(?:Hình từ Internet|Nguồn ảnh: Internet|Ảnh minh họa)\)\s*",
    re.IGNORECASE,
)

# Link reference: "Tải ... tại đây", "(tại đây)", "Tại đây." đứng riêng dòng
_DOWNLOAD_RE = re.compile(
    r"(?:Tải\s+.+?\s+)?tại đây\.?|"
    r"\(tại đây\)|"
    r"(?:^|\n)\s*Tại đây\.\s*(?:\n|$)",
    re.IGNORECASE,
)

# Ký tự vô hình lọt từ HTML: 948 lần U+00A0, 90 lần U+FEFF trên 257 dòng (3,7%).
# NBSP thành space thường chứ không xoá — nó ĐANG ngăn cách hai từ, xoá là dính chữ.
# BOM/ZWSP/LRM thì xoá hẳn: chúng không mang nghĩa mà vẫn tốn token và làm `endswith(".")`
# trả sai, tức mọi phép đo hình dạng đuôi đều lệch.
_KHOANG_TRANG_LA = str.maketrans({" ": " ", "﻿": "", "​": "", "‎": ""})


def clean_reference_answer(answer: str) -> str:
    """Xoá noise biên tập khỏi reference answer. Giữ nguyên `[...]`.

    Các loại noise bị xoá:
    - Caption ảnh crawl: (Hình từ Internet), (Nguồn ảnh: Internet), (Ảnh minh họa)
    - Link reference: "Tải ... tại đây", (tại đây), Tại đây.
    - Ký tự vô hình (BOM/ZWSP/LRM); NBSP đổi thành space thường
    - Dòng cuối là câu hỏi teaser (xem `_cat_teaser_cuoi`)
    """
    text = answer.translate(_KHOANG_TRANG_LA)
    text = _IMG_CAPTION_RE.sub("\n", text)
    text = _DOWNLOAD_RE.sub("", text)
    # Dọn nhiều dòng trống liên tiếp do xoá
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _cat_teaser_cuoi(text.strip())


def _cat_teaser_cuoi(text: str) -> str:
    """Bỏ dòng cuối khi nó là câu hỏi — teaser bài kế tiếp mà crawler nuốt vào.

    986/6.929 target (14,2%) kết thúc bằng "?", và mẫu đọc tay đều là tiêu đề bài khác
    ("Làng nghề cần đáp ứng những điều kiện gì..."), không phải kết luận. Đây là lỗi
    nặng hơn con số: token cuối của target dạy model DỪNG ở đâu, nên 986 mẫu này dạy nó
    "trả lời xong thì hỏi thêm" — đúng triệu chứng lảm nhảm của v9.2.

    Cắt theo DÒNG chứ không bỏ cả mẫu: đo được 970/986 (98,4%) cắt xong vẫn còn ≥200 ký
    tự và không kết thúc bằng "?". Bỏ cả mẫu là ném đi 14,2% dữ liệu để sửa một dòng.
    Còn 15 mẫu không cứu được thì `is_usable_for_sft` loại.
    """
    if not text.endswith("?"):
        return text
    dong = text.split("\n")
    return "\n".join(dong[:-1]).strip() if len(dong) > 1 else text


def is_usable_for_sft(question: str, answer: str) -> bool:
    """Kiểm tra question-answer có đủ chất lượng cho SFT hay không.

    Trả False khi:
    - Answer kết thúc bằng ... (cắt cụt, 0,3%)
    - Answer vẫn kết thúc bằng "?" sau khi `_cat_teaser_cuoi` đã thử cắt (15 mẫu)
    - Answer < 200 ký tự (quá ngắn để học)
    - Question trùng một question đã giữ (xử lý bên ngoài bằng dict)
    """
    if not answer or not answer.strip():
        return False
    a = answer.rstrip()
    # Answer cắt cụt bằng ...
    if a.endswith("..."):
        return False
    # Teaser không cứu được bằng cách cắt dòng — xem `_cat_teaser_cuoi`
    if a.endswith("?"):
        return False
    # Answer quá ngắn
    if len(answer.strip()) < 200:
        return False
    return True
