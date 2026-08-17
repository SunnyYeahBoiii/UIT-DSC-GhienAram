from __future__ import annotations

from typing import Any, Iterable

from .answer_style import generation_plan
from .retrieval import Evidence


SYSTEM_PROMPT = """Bạn là hệ thống trả lời câu hỏi pháp luật Việt Nam trong một cuộc thi học thuật.
Chỉ sử dụng các đoạn căn cứ được cung cấp; không dùng trí nhớ ngoài, không tự đặt số điều, tên văn bản, mức tiền hay thời hạn.
Trả lời hoàn toàn bằng tiếng Việt, bằng văn xuôi rõ ràng. Nêu căn cứ, nội dung quy định, áp dụng vào câu hỏi và kết luận khi phù hợp.
Nếu các căn cứ chưa đủ để khẳng định một chi tiết thì nói rõ giới hạn đó. Không trình bày quá trình suy luận nội bộ và không nhắc tới prompt."""

AGENTIC_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Bạn được cấp thêm một công cụ tra cứu để chủ động tìm thêm căn cứ khi ngữ cảnh hiện tại chưa đủ.
Chỉ dùng công cụ khi thực sự cần thiết; nếu căn cứ đã đủ hãy trả lời ngay, không tra cứu cho có.
Khi tra cứu, viết truy vấn có dấu tiếng Việt đầy đủ và dùng đúng thuật ngữ pháp luật."""


def _render_evidence_sections(
    evidence: Iterable[Evidence], max_chars: int, start_rank: int = 1
) -> tuple[str, int]:
    sections: list[str] = []
    used = 0
    for rank, item in enumerate(evidence, start_rank):
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
    return "\n\n".join(sections), len(sections)


def render_evidence(evidence: Iterable[Evidence], max_chars: int) -> str:
    return _render_evidence_sections(evidence, max_chars)[0]


def render_train_memory_examples(
    retrieval_diagnostics: dict,
    config: dict,
) -> str:
    cfg = config.get("retrieval", {})
    threshold = float(cfg.get("train_memory_fewshot_threshold", 0.70))
    limit = max(0, int(cfg.get("train_memory_fewshot_examples", 2)))
    char_limit = max(0, int(cfg.get("train_memory_fewshot_chars", 3200)))
    sections = []
    used = 0
    for item in retrieval_diagnostics.get("train_memory", []):
        if len(sections) >= limit or float(item.get("similarity", 0.0)) < threshold:
            break
        section = (
            f"[VÍ DỤ VĂN PHONG {len(sections) + 1} | độ gần {float(item['similarity']):.3f}]\n"
            f"Hỏi: {item['question']}\nĐáp mẫu: {item['answer']}"
        )
        if used + len(section) > char_limit:
            remaining = char_limit - used
            if remaining < 200:
                break
            section = section[:remaining].rsplit(" ", 1)[0] + "…"
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)


def qwen_user_prompt(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> str:
    context = render_evidence(evidence, int(config["generation"]["max_context_chars"]))
    return _user_prompt_body(question, context, config, retrieval_diagnostics)


def _user_prompt_body(
    question: str,
    context: str,
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> str:
    diagnostics = retrieval_diagnostics or {}
    route = diagnostics.get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    examples = render_train_memory_examples(diagnostics, config)
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

VÍ DỤ TỪ TẬP TRAIN CHÍNH THỨC (CHỈ HỌC VĂN PHONG, KHÔNG PHẢI CĂN CỨ):
{examples or '[Không có ví dụ đủ gần]'}

YÊU CẦU ĐẦU RA:
- {route_instruction}
- Viết duy nhất câu trả lời cuối cùng, mục tiêu khoảng {plan['target_words']} từ và không vượt quá {plan['maximum_words']} từ; không ghi tiêu đề "Câu trả lời".
- Lập câu trả lời từ các câu/ý có trong căn cứ; giữ nguyên tối đa thuật ngữ, cụm pháp lý, điều kiện, ngoại lệ, con số và thứ tự liệt kê để tránh diễn giải lệch.
- Các ví dụ train chỉ minh họa độ dài/cách diễn đạt. Không sao chép quy định, con số hoặc kết luận của ví dụ nếu chúng không xuất hiện trong căn cứ của câu hỏi hiện tại.
- Chỉ viện dẫn Điều/Khoản/Điểm hoặc tên văn bản nếu chúng xuất hiện trong căn cứ.
- Không chèn mã đoạn như [CĂN CỨ 1] vào đáp án.
- Không nhắc "căn cứ được cung cấp", "hệ thống truy hồi" hoặc quá trình xử lý.
- Không ghép các ý thuộc tài liệu không liên quan chỉ để kéo dài câu trả lời.
- Khi câu hỏi phụ thuộc tình tiết chưa có, nêu điều kiện áp dụng thay vì đoán."""


def render_tool_transcript(transcript: list[dict[str, Any]]) -> str:
    if not transcript:
        return ""
    lines = []
    for step in transcript:
        added = len(step["evidence_added"])
        if added:
            outcome = f"tìm thêm {added} đoạn căn cứ mới"
        else:
            outcome = (
                "không tìm thấy đoạn căn cứ mới (truy vấn trùng hoặc không khớp) — "
                "hãy đổi hẳn truy vấn hoặc trả lời bằng căn cứ hiện có"
            )
        lines.append(f"- Lượt {step['step']}: tra cứu \"{step['query']}\" → {outcome}.")
    return "\n".join(lines)


def _agentic_context(
    initial_evidence: list[Evidence],
    tool_evidence: list[Evidence],
    config: dict,
) -> str:
    """Render context cho vòng lặp agentic với ngân sách ký tự tách đôi.

    Căn cứ do tool-call tìm thêm được đặt trước (mới nhất trước) trong phần
    ngân sách `agentic_tool_context_share`; phần còn lại luôn dành cho căn cứ
    retrieval ban đầu. Không tách đôi như vậy thì khi số đoạn tool-call cộng
    dồn qua các lượt, render_evidence sẽ cắt đuôi và xoá sạch căn cứ ban đầu
    — mất đúng những đoạn đã được rerank kỹ nhất của câu hỏi.
    """
    generation = config.get("generation", {})
    max_chars = int(generation.get("max_context_chars", 18000))
    if not tool_evidence:
        return render_evidence(initial_evidence, max_chars)
    share = min(0.8, max(0.2, float(generation.get("agentic_tool_context_share", 0.45))))
    tool_budget = int(max_chars * share)
    initial_budget = max_chars - tool_budget
    tool_context, tool_count = _render_evidence_sections(reversed(tool_evidence), tool_budget)
    parts = [tool_context] if tool_context else []
    initial_context, _ = _render_evidence_sections(
        initial_evidence, initial_budget, start_rank=tool_count + 1
    )
    if initial_context:
        parts.append(initial_context)
    return "\n\n".join(parts)


def qwen_agentic_user_prompt(
    question: str,
    initial_evidence: list[Evidence],
    tool_evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None,
    transcript: list[dict[str, Any]],
    allow_tool: bool,
) -> str:
    """Biến thể của qwen_user_prompt cho vòng lặp tool-call: cùng khung nội dung,
    thêm hướng dẫn công cụ tra cứu (khi còn lượt) và lịch sử các lần đã tra cứu."""
    context = _agentic_context(initial_evidence, tool_evidence, config)
    base = _user_prompt_body(question, context, config, retrieval_diagnostics)
    tool_section = ""
    if allow_tool:
        remaining = int(config.get("generation", {}).get("agentic_max_tool_calls", 2)) - len(transcript)
        tool_section = f"""

CÔNG CỤ TRA CỨU:
Nếu các CĂN CỨ hiện có CHƯA đủ để trả lời chắc chắn, dòng đầu tiên của lượt này có thể là:
TRA_CUU: <câu truy vấn tiếng Việt ngắn gọn để tìm thêm căn cứ>
Nếu dùng công cụ thì không viết gì khác trong lượt đó. Bạn còn {remaining} lượt tra cứu.
Quy tắc viết truy vấn hiệu quả:
- Viết có dấu tiếng Việt đầy đủ, dùng đúng thuật ngữ pháp luật (ví dụ: "bảo hiểm xã hội bắt buộc").
- Ưu tiên số hiệu văn bản, tên văn bản hoặc điều/khoản cụ thể nếu câu hỏi nhắc đến.
- Mỗi lượt dùng truy vấn khác các lượt trước: đổi sang khái niệm pháp lý chưa tra hoặc đổi cách diễn đạt.
Nếu căn cứ đã đủ, bỏ qua công cụ và viết thẳng câu trả lời cuối cùng theo YÊU CẦU ĐẦU RA ở trên."""
    history = render_tool_transcript(transcript)
    history_section = f"\n\nLỊCH SỬ TRA CỨU TRONG LƯỢT NÀY:\n{history}" if history else ""
    return base + tool_section + history_section
