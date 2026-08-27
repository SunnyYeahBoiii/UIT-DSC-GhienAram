from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from .answer_style import generation_plan
from .quality import question_requirements
from .retrieval import Evidence
from .text import document_numbers, normalize_text


SYSTEM_PROMPT = """Bạn là hệ thống trả lời câu hỏi pháp luật Việt Nam trong một cuộc thi học thuật.
Chỉ sử dụng các đoạn căn cứ được cung cấp; không dùng trí nhớ ngoài, không tự đặt số điều, tên văn bản, mức tiền hay thời hạn.
Trả lời hoàn toàn bằng tiếng Việt, bằng văn xuôi rõ ràng. Nêu căn cứ, nội dung quy định, áp dụng vào câu hỏi và kết luận khi phù hợp.
Nếu các căn cứ chưa đủ để khẳng định một chi tiết thì nói rõ giới hạn đó. Không trình bày quá trình suy luận nội bộ và không nhắc tới prompt."""

AGENTIC_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Bạn được cấp thêm một công cụ tra cứu để chủ động tìm thêm căn cứ khi ngữ cảnh hiện tại chưa đủ.
Chỉ dùng công cụ khi thực sự cần thiết; nếu căn cứ đã đủ hãy trả lời ngay, không tra cứu cho có."""


def _clip_legal_text(
    text: str,
    *,
    maximum_chars: int,
    maximum_tokens: int | None,
    token_counter: Callable[[str], int] | None,
) -> str:
    """Cut at a clause/paragraph boundary instead of slicing a legal sentence."""
    text = normalize_text(text)
    if len(text) <= maximum_chars and (
        maximum_tokens is None or token_counter is None or token_counter(text) <= maximum_tokens
    ):
        return text
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?;:])\s+|\n+", text)
        if piece.strip()
    ]
    selected: list[str] = []
    for piece in pieces:
        candidate = " ".join([*selected, piece]).strip()
        if len(candidate) > maximum_chars:
            break
        if maximum_tokens is not None and token_counter is not None:
            if token_counter(candidate) > maximum_tokens:
                break
        selected.append(piece)
    if selected:
        return " ".join(selected)
    # A corpus line can itself be very long. Fall back to whole-word clipping,
    # never the previous arbitrary character cut.
    selected_words: list[str] = []
    for word in text.split():
        candidate = " ".join([*selected_words, word])
        if len(candidate) > maximum_chars:
            break
        if maximum_tokens is not None and token_counter is not None:
            if token_counter(candidate) > maximum_tokens:
                break
        selected_words.append(word)
    return " ".join(selected_words).strip()


def render_evidence_with_manifest(
    evidence: Iterable[Evidence],
    max_chars: int,
    *,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    sections: list[str] = []
    manifest: list[dict[str, Any]] = []
    used = 0
    used_tokens = 0
    for rank, item in enumerate(evidence, 1):
        relation = f" | quan hệ {item.relation_type}" if item.relation_type else ""
        title_numbers = sorted(document_numbers(item.chunk.title))
        canonical = title_numbers or sorted(
            document_numbers(f"{item.chunk.heading}\n{item.chunk.text}")
        )
        document_label = (
            f"số {canonical[0]}"
            if canonical
            else f"mã nội bộ {item.chunk.doc_id}"
        )
        header = (
            f"[CĂN CỨ {rank} | {document_label} | "
            f"{item.chunk.heading or 'đoạn ' + item.chunk.chunk_id}{relation}]"
        )
        text = item.chunk.text.strip()
        available = max_chars - used - len(header) - 2
        if available <= 24:
            break
        available_tokens = None
        if max_tokens is not None and token_counter is not None:
            header_tokens = token_counter(header)
            available_tokens = max_tokens - used_tokens - header_tokens
            if available_tokens <= 32:
                break
        clipped = _clip_legal_text(
            text,
            maximum_chars=available,
            maximum_tokens=available_tokens,
            token_counter=token_counter,
        )
        if not clipped:
            continue
        text = clipped
        section = f"{header}\n{text}"
        sections.append(section)
        used += len(section) + 2
        section_tokens = token_counter(section) if token_counter is not None else 0
        used_tokens += section_tokens
        manifest.append(
            {
                "rank": rank,
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "document_number": canonical[0] if canonical else None,
                "chars": len(text),
                "tokens": section_tokens,
                "clipped": text != item.chunk.text.strip(),
            }
        )
    return "\n\n".join(sections), manifest


def render_evidence(
    evidence: Iterable[Evidence],
    max_chars: int,
    *,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> str:
    return render_evidence_with_manifest(
        evidence,
        max_chars,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )[0]


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
    return qwen_prompt_bundle(question, evidence, config, retrieval_diagnostics)[0]


def qwen_prompt_bundle(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    generation_cfg = config["generation"]
    context, manifest = render_evidence_with_manifest(
        evidence,
        int(generation_cfg["max_context_chars"]),
        max_tokens=int(generation_cfg.get("max_context_tokens", 0)) or None,
        token_counter=token_counter,
    )
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
    requirements = question_requirements(question)
    requirement_line = ", ".join(requirements) if requirements else "trả lời trực tiếp trọng tâm"
    prompt = f"""CÂU HỎI:
{question}

CÁC ĐOẠN CĂN CỨ ĐÃ TRUY HỒI:
{context or '[Không truy hồi được căn cứ]'}

VÍ DỤ TỪ TẬP TRAIN CHÍNH THỨC (CHỈ HỌC VĂN PHONG, KHÔNG PHẢI CĂN CỨ):
{examples or '[Không có ví dụ đủ gần]'}

YÊU CẦU ĐẦU RA:
- {route_instruction}
- Các ý bắt buộc cần kiểm tra trước khi kết luận: {requirement_line}.
- Viết duy nhất câu trả lời cuối cùng, mục tiêu khoảng {plan['target_words']} từ và không vượt quá {plan['maximum_words']} từ; không ghi tiêu đề "Câu trả lời".
- Lập câu trả lời từ các câu/ý có trong căn cứ; giữ nguyên tối đa thuật ngữ, cụm pháp lý, điều kiện, ngoại lệ, con số và thứ tự liệt kê để tránh diễn giải lệch.
- Các ví dụ train chỉ minh họa độ dài/cách diễn đạt. Không sao chép quy định, con số hoặc kết luận của ví dụ nếu chúng không xuất hiện trong căn cứ của câu hỏi hiện tại.
- Chỉ viện dẫn Điều/Khoản/Điểm hoặc tên văn bản nếu chúng xuất hiện trong căn cứ.
- Không chèn mã đoạn như [CĂN CỨ 1] vào đáp án.
- Không nhắc "căn cứ được cung cấp", "hệ thống truy hồi" hoặc quá trình xử lý.
- Không ghép các ý thuộc tài liệu không liên quan chỉ để kéo dài câu trả lời.
- Khi câu hỏi phụ thuộc tình tiết chưa có, nêu điều kiện áp dụng thay vì đoán."""
    return prompt, context, manifest


def qwen_repair_prompt_bundle(
    question: str,
    original_answer: str,
    evidence: list[Evidence],
    issues: list[str],
    config: dict,
    retrieval_diagnostics: dict | None = None,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    _, context, manifest = qwen_prompt_bundle(
        question,
        evidence,
        config,
        retrieval_diagnostics,
        token_counter=token_counter,
    )
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    requirements = question_requirements(question)
    issue_text = ", ".join(issues) if issues else "cần kiểm tra lại tính đầy đủ"
    requirement_text = ", ".join(requirements) if requirements else "trả lời đúng trọng tâm"
    prompt = f"""CÂU HỎI:
{question}

CĂN CỨ ĐƯỢC PHÉP DÙNG:
{context or '[Không truy hồi được căn cứ]'}

ĐÁP ÁN NHÁP CẦN SỬA:
{original_answer}

LỖI ĐÃ PHÁT HIỆN:
{issue_text}

YÊU CẦU SỬA:
- Viết lại một đáp án cuối cùng độc lập; không bình luận về bản nháp hay quá trình sửa.
- Bảo đảm các ý: {requirement_text}.
- Chỉ giữ số văn bản, Điều/Khoản/Điểm, con số và đơn vị xuất hiện nguyên văn trong căn cứ.
- Giữ nguyên các quan hệ giới hạn như "tối đa", "tối thiểu", "không vượt quá".
- Không lặp câu hỏi, không chèn "Hình từ Internet", URL hoặc slug tên tệp.
- Dừng sau khi trả lời đủ; không lặp danh sách. Mục tiêu khoảng {plan['target_words']} từ,
  không vượt quá {plan['maximum_words']} từ.
- Nếu căn cứ chưa đủ cho một ý, nói rõ phạm vi chưa xác định thay vì đoán."""
    return prompt, context, manifest


def render_tool_transcript(transcript: list[dict[str, Any]]) -> str:
    if not transcript:
        return ""
    lines = [
        f"- Lượt {step['step']}: tra cứu \"{step['query']}\" → tìm thêm {len(step['evidence_added'])} đoạn căn cứ mới."
        for step in transcript
    ]
    return "\n".join(lines)


def qwen_agentic_user_prompt(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None,
    transcript: list[dict[str, Any]],
    allow_tool: bool,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> str:
    """Biến thể của qwen_user_prompt cho vòng lặp tool-call: cùng khung nội dung,
    thêm hướng dẫn công cụ tra cứu (khi còn lượt) và lịch sử các lần đã tra cứu."""
    base = qwen_prompt_bundle(
        question,
        evidence,
        config,
        retrieval_diagnostics,
        token_counter=token_counter,
    )[0]
    tool_section = ""
    if allow_tool:
        remaining = int(config.get("generation", {}).get("agentic_max_tool_calls", 2)) - len(transcript)
        tool_section = f"""

CÔNG CỤ TRA CỨU:
Nếu các CĂN CỨ hiện có CHƯA đủ để trả lời chắc chắn, dòng đầu tiên của lượt này có thể là:
TRA_CUU: <câu truy vấn tiếng Việt ngắn gọn để tìm thêm căn cứ>
Nếu dùng công cụ thì không viết gì khác trong lượt đó. Bạn còn {remaining} lượt tra cứu.
Nếu căn cứ đã đủ, bỏ qua công cụ và viết thẳng câu trả lời cuối cùng theo YÊU CẦU ĐẦU RA ở trên."""
    history = render_tool_transcript(transcript)
    history_section = f"\n\nLỊCH SỬ TRA CỨU TRONG LƯỢT NÀY:\n{history}" if history else ""
    return base + tool_section + history_section
