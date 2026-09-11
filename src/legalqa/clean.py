"""Làm sạch Reference answer trước khi SFT.

18,5% reference answer chứa caption ảnh crawl — model học sẽ sinh hallucination.
0,9% chứa link reference ("tại đây") —噪声 biên tập.
Phải giữ nguyên `[...]` — trích luật lược bỏ hợp lệ, không phải noise.
"""
from __future__ import annotations

import re

# Caption ảnh crawl — xoá nếu đứng riêng dòng hoặc trong ngoặc
_IMG_CAPTION_RE = re.compile(
    r"\s*\((?:Hình từ Internet|Nguồn ảnh: Internet|Ảnh minh họa)\)\s*"
)

# Link reference: "Tải ... tại đây", "(tại đây)", "Tại đây." đứng riêng dòng
_DOWNLOAD_RE = re.compile(
    r"(?:Tải\s+.+?\s+)?tại đây\.?|"
    r"\(tại đây\)|"
    r"(?:^|\n)\s*Tại đây\.\s*(?:\n|$)",
    re.IGNORECASE,
)

# Question bị lặp nguyên văn trong answer —.detect bằng cách check
# question xuất hiệnverbatim trong answer
def _verbatim_question_pattern(question: str) -> re.Pattern[str]:
    """Tạo regex khớp questionverbatim trong answer."""
    escaped = re.escape(question.strip())
    return re.compile(escaped, re.IGNORECASE)


def clean_reference_answer(answer: str) -> str:
    """Xoá noise biên tập khỏi reference answer. Giữ nguyên `[...]`.

    Các loại noise bị xoá:
    - Caption ảnh crawl: (Hình từ Internet), (Nguồn ảnh: Internet), (Ảnh minh họa)
    - Link reference: "Tải ... tại đây", (tại đây), Tại đây.
    """
    text = _IMG_CAPTION_RE.sub("\n", answer)
    text = _DOWNLOAD_RE.sub("", text)
    # Dọn nhiều dòng trống liên tiếp do xoá
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_usable_for_sft(question: str, answer: str) -> bool:
    """Kiểm tra question-answer có đủ chất lượng cho SFT hay không.

    Trả False khi:
    - Answer kết thúc bằng ... (cắt cụt, 0,3%)
    - Answer < 200 ký tự (quá ngắn để học)
    - Question trùng một question đã giữ (xử lý bên ngoài bằng dict)
    """
    if not answer or not answer.strip():
        return False
    # Answer cắt cụt bằng ...
    if answer.rstrip().endswith("..."):
        return False
    # Answer quá ngắn
    if len(answer.strip()) < 200:
        return False
    return True
