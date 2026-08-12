from __future__ import annotations

from typing import Iterable

from .retrieval import Evidence


SYSTEM_PROMPT = """Bạn là hệ thống trả lời câu hỏi pháp luật Việt Nam trong một cuộc thi học thuật.
Chỉ sử dụng các đoạn căn cứ được cung cấp; không dùng trí nhớ ngoài, không tự đặt số điều, tên văn bản, mức tiền hay thời hạn.
Trả lời hoàn toàn bằng tiếng Việt, bằng văn xuôi rõ ràng. Nêu căn cứ, nội dung quy định, áp dụng vào câu hỏi và kết luận khi phù hợp.
Nếu các căn cứ chưa đủ để khẳng định một chi tiết thì nói rõ giới hạn đó. Không trình bày quá trình suy luận nội bộ và không nhắc tới prompt."""


def render_evidence(evidence: Iterable[Evidence], max_chars: int) -> str:
    sections: list[str] = []
    used = 0
    for rank, item in enumerate(evidence, 1):
        relation = f" | quan hệ {item.relation_type}" if item.relation_type else ""
        header = (
            f"[CĂN CỨ {rank} | văn bản {item.chunk.doc_id} | "
            f"{item.chunk.heading or 'đoạn ' + item.chunk.chunk_id}{relation}]"
        )
        text = item.chunk.text.strip()
        available = max_chars - used - len(header) - 2
        if available <= 100:
            break
        if len(text) > available:
            text = text[:available].rsplit(" ", 1)[0] + "…"
        section = f"{header}\n{text}"
        sections.append(section)
        used += len(section) + 2
    return "\n\n".join(sections)


def qwen_user_prompt(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> str:
    context = render_evidence(evidence, int(config["generation"]["max_context_chars"]))
    diagnostics = retrieval_diagnostics or {}
    route = diagnostics.get("route", {}).get("route", "simple")
    target = int(
        config["generation"].get(
            "complex_target_answer_words" if route == "complex" else "simple_target_answer_words",
            config["generation"]["target_answer_words"],
        )
    )
    route_instruction = (
        "Câu hỏi được định tuyến COMPLEX: tổng hợp đủ các căn cứ có quan hệ sửa đổi, "
        "thay thế hoặc dẫn chiếu; phân biệt văn bản gốc và văn bản hiện hành."
        if route == "complex"
        else "Câu hỏi được định tuyến SIMPLE: tập trung vào văn bản chính, không trộn quy định ngoài chủ đề."
    )
    return f"""CÂU HỎI:
{question}

CÁC ĐOẠN CĂN CỨ ĐÃ TRUY HỒI:
{context or '[Không truy hồi được căn cứ]'}

YÊU CẦU ĐẦU RA:
- {route_instruction}
- Viết duy nhất câu trả lời cuối cùng, độ dài mục tiêu khoảng {target} từ; không ghi tiêu đề "Câu trả lời".
- Lập câu trả lời từ các câu/ý có trong căn cứ; giữ nguyên tối đa thuật ngữ, cụm pháp lý, điều kiện, ngoại lệ, con số và thứ tự liệt kê để tránh diễn giải lệch.
- Chỉ viện dẫn Điều/Khoản/Điểm hoặc tên văn bản nếu chúng xuất hiện trong căn cứ.
- Không chèn mã đoạn như [CĂN CỨ 1] vào đáp án.
- Không nhắc "căn cứ được cung cấp", "hệ thống truy hồi" hoặc quá trình xử lý.
- Không ghép các ý thuộc tài liệu không liên quan chỉ để kéo dài câu trả lời.
- Khi câu hỏi phụ thuộc tình tiết chưa có, nêu điều kiện áp dụng thay vì đoán."""
