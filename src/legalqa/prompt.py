"""Dựng prompt từ Evidence — đơn vị sinh, không phải đơn vị tìm.

Quyết định chốt ở §15.7: khi một Điều bị sửa đổi, prompt chứa **cả hai** — Điều gốc VÀ
khoản sửa đổi — kèm nhãn quan hệ. KHÔNG tự ghép bản hợp nhất.

Vì sao không tự ghép: ghép đòi biết khoản sửa đổi thay thế chính xác đoạn nào của Điều
gốc. `graph.py` chỉ biết *có* quan hệ sửa đổi (bắt được qua heading "Sửa đổi, bổ sung
Điều X"), không biết ranh giới đoạn bị thay. Tự ghép là bịa ra một văn bản không tồn
tại, và 75,9% reference answer trích NGUYÊN VĂN — bịa một ký tự là mất luôn câu đó.

Đưa cả hai vào thì model tự chọn, và nếu nó chọn sai thì lỗi nằm ở một đoạn có thật.
"""
from __future__ import annotations

import os

from .evidence import Evidence

# Tắt số hiệu văn bản khỏi dòng nhận dạng. Vì sao cần cờ: adapter b318 được train trên
# prompt CHƯA có số hiệu; suy luận bằng prompt có số hiệu là đưa model vào phân phối nó
# chưa từng thấy. Cờ để prompt lúc suy luận khớp đúng checkpoint đang dùng, không phải
# để bật/tắt theo ý thích — đặt LEGALQA_SO_HIEU=0 khi chạy checkpoint đời cũ.
CO_SO_HIEU = os.environ.get("LEGALQA_SO_HIEU", "1") != "0"

# Nhãn cho Evidence gốc từ retrieval (`relation` rỗng). Không để trống: model cần biết
# đoạn này là căn cứ chính, không phải một đoạn phụ trợ bị trôi mất nhãn.
NHAN_GOC = "Căn cứ chính"


def _tieu_de(ev: Evidence) -> str:
    """Một dòng nhận dạng Evidence. Có `article_number` thì nêu, vì 75,9% reference
    answer mở đầu bằng cụm dẫn chiếu "Căn cứ Điều ..." — số Điều phải nằm trong prompt
    ở dạng model copy lại được. 28,1% Evidence là cửa sổ lui không có số Điều (§7);
    những đoạn đó chỉ nêu tên văn bản, đừng bịa số."""
    dieu = f"Điều {ev.article_number}, " if ev.article_number else ""
    # Số hiệu đứng TRƯỚC tiêu đề: tiêu đề là dòng ALL-CAPS crawl từ HTML, không mang số
    # hiệu lẫn năm ban hành, nên ở eval b318 model bịa năm văn bản 49/120 câu. Đo trên
    # cùng 120 câu: số hiệu này xuất hiện nguyên văn trong đáp án 65%.
    so = f"{ev.doc_number} " if (ev.doc_number and CO_SO_HIEU) else ""
    return f"{dieu}{so}{ev.doc_title}"


def dung_can_cu(evidences: list[Evidence]) -> str:
    """Khối căn cứ, giữ ĐÚNG thứ tự truyền vào.

    Thứ tự là trách nhiệm của phía gọi, không phải của hàm này: `graph.py` trả Evidence
    sửa đổi ngay sau seed sinh ra nó, nên duyệt tuần tự đã cho đúng cặp "gốc trước, sửa
    đổi sau". Sắp lại ở đây sẽ phá cặp đó mà không có thông tin nào để dựng lại.

    Text giữ nguyên văn theo offset, kể cả `\\n\\n` — ngắt dòng mang cấu trúc phân khoản
    mà model cần để trích đúng khoản.
    """
    khoi: list[str] = []
    for i, ev in enumerate(evidences, 1):
        nhan = ev.relation or NHAN_GOC
        khoi.append(f"[{i}] {nhan}\n{_tieu_de(ev)}\n{ev.text}")
    return "\n\n".join(khoi)


def dung_prompt(question: str, evidences: list[Evidence]) -> str:
    """Phần user của prompt. Căn cứ đặt TRƯỚC câu hỏi.

    Vì sao trước: câu hỏi đứng cuối là thứ model đọc gần nhất trước khi sinh. Đặt căn cứ
    sau câu hỏi thì token cuối cùng là văn bản luật, và model có xu hướng tiếp tục trích
    thay vì trả lời.
    """
    return f"Căn cứ pháp luật:\n{dung_can_cu(evidences)}\n\nCâu hỏi: {question}\nTrả lời:"
