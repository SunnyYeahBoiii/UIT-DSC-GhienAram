from __future__ import annotations

import inspect
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .answer_style import generation_plan
from .prompts import AGENTIC_SYSTEM_PROMPT, SYSTEM_PROMPT, qwen_agentic_user_prompt, qwen_user_prompt
from .retrieval import Evidence, deduplicate_and_merge_evidence, retrieval_confidence
from .text import citations, document_numbers, legal_tokens, normalize_text, numbers, split_sentences, words


@dataclass
class Candidate:
    source: str
    answer: str
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def clean_answer(text: str) -> str:
    answer = normalize_text(text)
    answer = re.sub(r"^(?:câu trả lời|trả lời)\s*[:：]\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)", "", answer).strip()
    return answer


def compact_generated_answer(
    text: str,
    maximum_words: int,
    *,
    hard_trim: bool = False,
) -> tuple[str, dict[str, int | bool]]:
    """Remove exact sentence loops; optionally stop at a complete-sentence boundary."""
    answer = clean_answer(text)
    original_words = len(words(answer))
    sentences = split_sentences(answer)
    selected: list[str] = []
    seen: set[str] = set()
    selected_words = 0
    duplicate_sentences = 0
    for sentence in sentences:
        fingerprint = " ".join(words(sentence))
        if fingerprint and fingerprint in seen:
            duplicate_sentences += 1
            continue
        sentence_words = len(words(sentence))
        # Never cut a single long legal sentence in the middle. Once at least
        # one complete sentence exists, however, stop before crossing p90/p95.
        if hard_trim and selected and selected_words + sentence_words > maximum_words:
            break
        selected.append(sentence.strip())
        selected_words += sentence_words
        if fingerprint:
            seen.add(fingerprint)
    compacted = " ".join(item for item in selected if item).strip() or answer
    if compacted and not compacted.endswith((".", "!", "?", ";", ":")):
        compacted += "."
    compacted_words = len(words(compacted))
    return compacted, {
        "raw_words": original_words,
        "compacted_words": compacted_words,
        "duplicate_sentences_removed": duplicate_sentences,
        "compaction_applied": compacted != answer,
    }


def _sentence_quality(sentence: str) -> float:
    tokens = words(sentence)
    if len(tokens) < 8:
        return 0.0
    placeholder_ratio = sentence.count(".") / max(1, len(sentence))
    letters = sum(character.isalpha() for character in sentence)
    uppercase = sum(character.isupper() for character in sentence)
    if placeholder_ratio > 0.08 or letters / max(1, len(sentence)) < 0.45:
        return 0.0
    if letters > 20 and uppercase / letters > 0.82:
        return 0.25
    return min(1.0, len(tokens) / 24.0)


def extractive_candidate(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    query_tokens = set(legal_tokens(question))
    ranked: list[tuple[float, int, int, str]] = []
    for evidence_rank, item in enumerate(evidence):
        for sentence_rank, sentence in enumerate(split_sentences(item.chunk.text)):
            sentence_tokens = set(legal_tokens(sentence))
            quality = _sentence_quality(sentence)
            if quality <= 0:
                continue
            overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
            citation_bonus = 0.13 if citations(sentence) else 0.0
            number_bonus = 0.08 if numbers(question) & numbers(sentence) else 0.0
            rank_prior = 0.20 / (evidence_rank + 1)
            length_bonus = min(len(words(sentence)), 45) / 450.0
            ranked.append((overlap + citation_bonus + number_bonus + rank_prior + length_bonus + 0.18 * quality, evidence_rank, sentence_rank, sentence))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    target_words = int(plan["target_words"])
    selected: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    count = 0
    for _, evidence_rank, sentence_rank, sentence in ranked:
        sentence_words = words(sentence)
        fingerprint = " ".join(sentence_words)
        if fingerprint in seen:
            continue
        sentence_set = set(sentence_words)
        if any(
            len(sentence_set & set(words(previous))) / max(1, len(sentence_set)) > 0.82
            for _, _, previous in selected
        ):
            continue
        selected.append((evidence_rank, sentence_rank, sentence))
        seen.add(fingerprint)
        count += len(words(sentence))
        if count >= target_words:
            break
    selected.sort(key=lambda item: (item[0], item[1]))
    body = " ".join(sentence for _, _, sentence in selected).strip()
    if not body:
        body = "Chưa đủ thông tin pháp lý để xác định câu trả lời."
    if body and body[0].islower():
        body = body[0].upper() + body[1:]
    if body.casefold().startswith(("theo ", "căn cứ ", "điều ", "khoản ", "điểm ")):
        answer = body
    else:
        answer = f"Theo quy định pháp luật, {body[0].lower() + body[1:] if len(body) > 1 else body}"
    if not answer.rstrip().endswith((".", "!", "?")):
        answer += "."
    return Candidate(
        "extractive",
        clean_answer(answer),
        details={"route": route, "selected_sentences": len(selected), "style_plan": plan},
    )


def make_qwen_candidate(
    generator,
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    token_limit = int(plan["max_new_tokens"])
    result = generator.generate_with_metadata(
        SYSTEM_PROMPT,
        qwen_user_prompt(question, evidence, config, retrieval_diagnostics),
        max_new_tokens=token_limit,
    )
    answer = clean_answer(result.text)
    if not answer:
        raise RuntimeError("Qwen sinh câu trả lời rỗng.")
    compaction = {
        "raw_words": len(words(answer)),
        "compacted_words": len(words(answer)),
        "duplicate_sentences_removed": 0,
        "compaction_applied": False,
    }
    if bool(config.get("generation", {}).get("compact_generated_answers", True)):
        answer, compaction = compact_generated_answer(
            answer,
            int(plan["maximum_words"]),
            hard_trim=bool(
                config.get("generation", {}).get("hard_trim_generated_answers", False)
            ),
        )
    return Candidate(
        "qwen3.5-2b",
        answer,
        details={
            "route": route,
            "generated_tokens": result.generated_tokens,
            "terminated_by_eos": result.terminated_by_eos,
            "truncated": result.truncated,
            "token_limit": token_limit,
            "style_plan": plan,
            **compaction,
            "fewshot_examples": sum(
                float(item.get("similarity", 0.0))
                >= float(config.get("retrieval", {}).get("train_memory_fewshot_threshold", 0.70))
                for item in (retrieval_diagnostics or {}).get("train_memory", [])
            ),
        },
    )


_TOOL_CALL_MARKER = re.compile(
    r"^\s*(?:(?:[-*•>_#]\s*)|(?:\d+[.)]\s*)|[*_]{1,4}\s*){0,3}`{0,3}\s*"
    r"tra[_\s]*cuu\s*[:：]",
    re.IGNORECASE,
)


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", text.replace("Đ", "D").replace("đ", "d")
    )
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn")


def parse_agentic_step(raw_text: str) -> tuple[str | None, str | None]:
    """Phân loại một lượt sinh trong vòng lặp agentic.

    Model được hướng dẫn mở đầu bằng "TRA_CUU: <truy vấn>" khi muốn tra cứu
    thêm; mọi output khác coi là câu trả lời cuối. So khớp marker trên bản đã
    bỏ dấu vì model 2B đôi lúc bỏ dấu tiếng Việt ở nhãn lệnh, còn truy vấn giữ
    nguyên dấu để BM25 tra cứu đúng.
    Trả về (query, None) nếu là tool call, hoặc (None, answer) nếu là câu trả lời.
    """
    lines = raw_text.strip().splitlines()
    # Small models sometimes wrap the command in a Markdown fence/bullet or
    # emit one short preface. Search only the first three non-empty lines so a
    # literal marker mentioned later in a genuine answer is not executed.
    inspected = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("```") and "tra" not in stripped.casefold():
            continue
        inspected += 1
        if _TOOL_CALL_MARKER.match(_strip_diacritics(stripped)):
            query = re.split(r"[:：]", stripped, maxsplit=1)[-1].strip(" `*_#")
            if not query:
                for continuation in lines[index + 1 :]:
                    continuation = continuation.strip()
                    if continuation and not continuation.startswith("```"):
                        query = continuation.strip(" `*_#")
                        break
            return query, None
        if inspected >= 3:
            break
    return None, clean_answer(raw_text)


def _normalized_agent_query(query: str) -> str:
    return " ".join(legal_tokens(_strip_diacritics(normalize_text(query))))


def _call_agentic_search(
    search_fn: Callable[..., list[Evidence]],
    query: str,
    question: str,
    current_evidence: list[Evidence],
) -> list[Evidence]:
    """Call the v9.2 search protocol while keeping one-argument test hooks valid."""
    try:
        signature = inspect.signature(search_fn)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        variadic = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        positional, variadic = [], True
    if variadic or len(positional) >= 3:
        return search_fn(query, question, current_evidence)
    return search_fn(query)


def _select_new_tool_evidence(
    new_evidence: list[Evidence],
    seen_chunk_ids: set[str],
    per_call: int,
    max_per_document: int,
) -> list[Evidence]:
    """Chọn các đoạn căn cứ mới cho một lượt tool-call.

    Danh sách vào đã được rank_agentic_evidence xếp hạng; BM25 thô vẫn hay trả
    nhiều window liền kề của cùng một văn bản ở đầu danh sách, nếu lấy nguyên
    top-k thì cả ngân sách lượt tra cứu chỉ chứa một văn bản. Giới hạn
    `max_per_document` giữ tính đa dạng văn bản cho câu hỏi complex; việc gộp
    text trùng của window liền kề do caller đảm nhiệm.
    """
    selected: list[Evidence] = []
    per_document: dict[str, int] = {}
    for item in new_evidence:
        if len(selected) >= per_call:
            break
        source_ids = {
            str(value)
            for value in item.chunk.metadata.get(
                "_merged_chunk_ids", [item.chunk.chunk_id]
            )
        }
        source_ids.add(item.chunk.chunk_id)
        if source_ids & seen_chunk_ids:
            continue
        used = per_document.get(item.chunk.doc_id, 0)
        if used >= max_per_document:
            continue
        selected.append(item)
        per_document[item.chunk.doc_id] = used + 1
    return selected


def make_qwen_agentic_candidate(
    generator,
    question: str,
    evidence: list[Evidence],
    config: dict,
    search_fn: Callable[..., list[Evidence]] | None,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    """Biến thể agentic của make_qwen_candidate: model có thể tự phát ra
    truy vấn ("TRA_CUU: ...") để lấy thêm căn cứ giữa các lượt sinh, thay vì
    chỉ dùng đúng một lần retrieve cố định trước đó (xem pipeline.run_pipeline).
    """
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    token_limit = int(plan["max_new_tokens"])
    generation_cfg = config.get("generation", {})
    max_tool_calls = max(0, int(generation_cfg.get("agentic_max_tool_calls", 2)))
    evidence_per_call = max(1, int(generation_cfg.get("agentic_evidence_per_call", 3)))
    max_per_document = max(1, int(generation_cfg.get("agentic_max_per_document", 2)))

    initial_evidence = list(evidence)
    seen_chunk_ids: set[str] = set()
    for item in initial_evidence:
        seen_chunk_ids.add(item.chunk.chunk_id)
        seen_chunk_ids.update(
            str(value)
            for value in item.chunk.metadata.get("_merged_chunk_ids", [])
        )
    tool_evidence_pool: list[Evidence] = []
    transcript: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    force_final_answer = False
    answer = ""
    generated_tokens = 0
    truncated = False
    terminated_by_eos = True

    for step in range(max_tool_calls + 1):
        allow_tool = search_fn is not None and step < max_tool_calls and not force_final_answer
        prompt = qwen_agentic_user_prompt(
            question, initial_evidence, tool_evidence_pool, config,
            retrieval_diagnostics, transcript, allow_tool,
        )
        result = generator.generate_with_metadata(
            AGENTIC_SYSTEM_PROMPT if allow_tool else SYSTEM_PROMPT,
            prompt,
            max_new_tokens=token_limit,
        )
        generated_tokens += result.generated_tokens
        # Gán (không OR) giống terminated_by_eos: câu trả lời cuối luôn đến từ
        # lượt vừa chạy, nên "truncated" của final answer không được kế thừa
        # từ các lượt tool-call trước đó — một lượt tool-call sinh dài rồi bị
        # cắt (đọc câu TRA_CUU: ở dòng đầu, phần đuôi bị bỏ) không đại diện
        # cho chất lượng của lượt trả lời thật sự.
        truncated = result.truncated
        terminated_by_eos = result.terminated_by_eos
        query, final_answer = parse_agentic_step(result.text)
        if not allow_tool:
            # Lượt cuối không còn hạn mức tra cứu: nếu model vẫn phát "TRA_CUU:"
            # thì đó là output hỏng (marker không phải câu trả lời thật), không
            # được lấy làm đáp án — phải báo lỗi để pipeline loại candidate này
            # thay vì nộp nguyên văn "TRA_CUU: ..." làm câu trả lời.
            if query is not None:
                raise RuntimeError(
                    "Qwen (agentic) vẫn phát TRA_CUU sau khi hết hạn mức tra cứu."
                )
            answer = final_answer or ""
            break
        if query is None:
            answer = final_answer or ""
            break
        if not query:
            raise RuntimeError("Qwen (agentic) phát TRA_CUU nhưng không kèm truy vấn.")
        query = query[: max(32, int(generation_cfg.get("agentic_max_query_chars", 240)))].strip()
        normalized_query = _normalized_agent_query(query)
        if not normalized_query:
            raise RuntimeError("Qwen (agentic) phát truy vấn không có token hợp lệ.")
        if normalized_query in seen_queries:
            transcript.append(
                {
                    "step": step + 1,
                    "query": query,
                    "evidence_added": [],
                    "repeated_query": True,
                }
            )
            # Do not waste the remaining budget on the same search forever.
            # The next pass is forced to produce an answer without tool access.
            force_final_answer = True
            continue
        seen_queries.add(normalized_query)
        # search_fn v9.2 nhận thêm câu hỏi gốc + căn cứ hiện có (để gieo mối
        # quan hệ graph); truyền newest-first như cũ: tool evidence mới nhất
        # trước, rồi căn cứ retrieval ban đầu.
        new_evidence = _call_agentic_search(
            search_fn, query, question, list(reversed(tool_evidence_pool)) + initial_evidence
        )
        raw_selected = _select_new_tool_evidence(
            new_evidence, seen_chunk_ids, evidence_per_call, max_per_document
        )
        for item in raw_selected:
            seen_chunk_ids.add(item.chunk.chunk_id)
            seen_chunk_ids.update(
                str(value)
                for value in item.chunk.metadata.get("_merged_chunk_ids", [])
            )
        # Gộp các window liền kề của cùng điều/khoản: BM25 hay trả về nhiều
        # window chồng lấn (overlap_chars) của cùng một chỗ trong văn bản;
        # giữ nguyên thì text trùng nhau lãng phí ngân sách ký tự context
        # (prompt tách budget theo agentic_tool_context_share, xem prompts.py).
        added = deduplicate_and_merge_evidence(
            raw_selected, evidence_per_call, prefer_document_diversity=False
        )
        for item in added:
            for source_id in item.chunk.metadata.get("_merged_chunk_ids", [item.chunk.chunk_id]):
                seen_chunk_ids.add(source_id)
        tool_evidence_pool.extend(added)
        transcript.append(
            {
                "step": step + 1,
                "query": query,
                "evidence_added": [item.chunk.chunk_id for item in added],
                "result_count": len(new_evidence),
                "relation_hits": sum(bool(item.relation_type) for item in added),
            }
        )

    if not answer:
        raise RuntimeError("Qwen (agentic) sinh câu trả lời rỗng.")
    compaction = {
        "raw_words": len(words(answer)),
        "compacted_words": len(words(answer)),
        "duplicate_sentences_removed": 0,
        "compaction_applied": False,
    }
    if bool(generation_cfg.get("compact_generated_answers", True)):
        answer, compaction = compact_generated_answer(
            answer,
            int(plan["maximum_words"]),
            hard_trim=bool(generation_cfg.get("hard_trim_generated_answers", False)),
        )
    initial_confidence_value = (retrieval_diagnostics or {}).get("route", {}).get(
        "confidence"
    )
    if initial_confidence_value is None:
        initial_confidence, _ = retrieval_confidence(evidence)
    else:
        initial_confidence = float(initial_confidence_value)
    if tool_evidence_pool:
        tool_text = "\n".join(item.chunk.text for item in tool_evidence_pool)
        query_terms = set(legal_tokens(question))
        evidence_terms = set(legal_tokens(tool_text))
        lexical_coverage = len(query_terms & evidence_terms) / max(1, len(query_terms))
        exact_legal = float(
            bool(citations(question) & citations(tool_text))
            or bool(document_numbers(question) & document_numbers(tool_text))
        )
        quality = max(0.0, min(1.0, max(item.score for item in tool_evidence_pool)))
        tool_confidence = min(
            1.0,
            0.50 * initial_confidence
            + 0.30 * lexical_coverage
            + 0.12 * quality
            + 0.08 * exact_legal,
        )
        agentic_confidence = max(initial_confidence, tool_confidence)
    else:
        agentic_confidence = initial_confidence
    return Candidate(
        "qwen3.5-2b-agentic",
        answer,
        details={
            "route": route,
            "generated_tokens": generated_tokens,
            "terminated_by_eos": terminated_by_eos,
            "truncated": truncated,
            "token_limit": token_limit,
            "style_plan": plan,
            **compaction,
            "tool_calls": transcript,
            "tool_calls_used": len(transcript),
            "evidence_added_total": len(tool_evidence_pool),
            "tool_retrieval_confidence": agentic_confidence,
            # Bản sao serializable để verifier dựng evidence RIÊNG cho candidate
            # agentic. Không bao giờ gộp vào evidence chung của Qwen thường,
            # extractive hoặc train-memory.
            "tool_evidence": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "doc_id": item.chunk.doc_id,
                    "title": item.chunk.title,
                    "heading": item.chunk.heading,
                    "text": item.chunk.text,
                    "score": item.score,
                    "bm25_score": item.bm25_score,
                    "stage": item.stage,
                    "relation_type": item.relation_type,
                    "relation_path": item.relation_path,
                    "merged_chunk_ids": [
                        str(value)
                        for value in item.chunk.metadata.get("_merged_chunk_ids", [])
                    ],
                }
                for item in tool_evidence_pool
            ],
        },
    )


def make_train_memory_candidate(
    question: str,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate | None:
    diagnostics = retrieval_diagnostics or {}
    neighbors = diagnostics.get("train_memory", [])
    if not neighbors:
        return None
    nearest = neighbors[0]
    similarity = float(nearest.get("similarity", 0.0))
    threshold = float(
        config.get("retrieval", {}).get("train_memory_candidate_threshold", 0.86)
    )
    if similarity < threshold:
        return None
    answer = clean_answer(str(nearest.get("answer", "")))
    if not answer:
        return None
    route = diagnostics.get("route", {}).get("route", "simple")
    return Candidate(
        "train-memory",
        answer,
        details={
            "route": route,
            "memory_similarity": similarity,
            "memory_question_id": str(nearest.get("question_id", "")),
            "memory_question": str(nearest.get("question", "")),
            "style_plan": generation_plan(question, route, config),
        },
    )
