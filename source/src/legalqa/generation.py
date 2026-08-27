from __future__ import annotations

import inspect
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .answer_style import generation_plan
from .prompts import (
    AGENTIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    qwen_agentic_user_prompt,
    qwen_prompt_bundle,
    qwen_repair_prompt_bundle,
    render_evidence_with_manifest,
)
from .quality import analyze_answer_quality, requirement_coverage, sanitize_answer
from .retrieval import Evidence, retrieval_confidence
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


def _generator_token_counter(generator) -> Callable[[str], int] | None:
    tokenizer = getattr(getattr(generator, "processor", None), "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


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
    ranked: list[tuple[float, float, int, int, str, str]] = []
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
            intent_coverage, _ = requirement_coverage(question, sentence)
            score = (
                overlap + citation_bonus + number_bonus + rank_prior + length_bonus
                + 0.18 * quality + 0.12 * intent_coverage
            )
            ranked.append(
                (score, overlap, evidence_rank, sentence_rank, item.chunk.doc_id, sentence)
            )
    ranked.sort(key=lambda item: (-item[0], item[2], item[3]))
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    target_words = int(plan["target_words"])
    generation_cfg = config.get("generation", {})
    minimum_overlap = float(generation_cfg.get("extractive_min_sentence_overlap", 0.12))
    max_documents = (
        1 if route == "simple"
        else max(1, int(generation_cfg.get("extractive_complex_max_documents", 3)))
    )
    best_by_document: dict[str, float] = {}
    for score, _, _, _, doc_id, _ in ranked:
        best_by_document[doc_id] = max(score, best_by_document.get(doc_id, float("-inf")))
    allowed_documents = {
        doc_id
        for doc_id, _ in sorted(best_by_document.items(), key=lambda item: (-item[1], item[0]))[
            :max_documents
        ]
    }
    selected: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    count = 0
    for _, overlap, evidence_rank, sentence_rank, doc_id, sentence in ranked:
        if doc_id not in allowed_documents or overlap < minimum_overlap:
            continue
        sentence_words = words(sentence)
        fingerprint = " ".join(sentence_words)
        if fingerprint in seen:
            continue
        sentence_set = set(sentence_words)
        if any(
            len(sentence_set & set(words(previous))) / max(1, len(sentence_set)) > 0.82
            for _, _, _, previous in selected
        ):
            continue
        selected.append((evidence_rank, sentence_rank, doc_id, sentence))
        seen.add(fingerprint)
        count += len(words(sentence))
        if count >= target_words:
            break
    selected.sort(key=lambda item: (item[0], item[1]))
    body = " ".join(sentence for _, _, _, sentence in selected).strip()
    if not body:
        body = "Chưa đủ thông tin pháp lý để xác định câu trả lời."
    if body and body[0].islower():
        body = body[0].upper() + body[1:]
    if body.casefold().startswith(("theo ", "căn cứ ", "điều ", "khoản ", "điểm ", "chưa ")):
        answer = body
    else:
        answer = f"Theo quy định pháp luật, {body[0].lower() + body[1:] if len(body) > 1 else body}"
    if not answer.rstrip().endswith((".", "!", "?")):
        answer += "."
    answer, sanitization = sanitize_answer(question, clean_answer(answer))
    visible_evidence = "\n".join(item.chunk.text for item in evidence)
    candidate_details: dict[str, Any] = {
        "route": route,
        "selected_sentences": len(selected),
        "selected_documents": sorted({doc_id for _, _, doc_id, _ in selected}),
        "style_plan": plan,
        **sanitization,
        "_visible_evidence_text": visible_evidence,
    }
    candidate_details["quality"] = analyze_answer_quality(
        question, answer, visible_evidence, candidate_details
    )
    return Candidate(
        "extractive",
        answer,
        details=candidate_details,
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
    token_counter = _generator_token_counter(generator)
    prompt, visible_evidence, evidence_manifest = qwen_prompt_bundle(
        question,
        evidence,
        config,
        retrieval_diagnostics,
        token_counter=token_counter,
    )
    result = generator.generate_with_metadata(
        SYSTEM_PROMPT,
        prompt,
        max_new_tokens=token_limit,
    )
    raw_answer = clean_answer(result.text)
    answer, sanitization = sanitize_answer(question, raw_answer)
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
    generation_details: dict[str, Any] = {
        "route": route,
        "generated_tokens": result.generated_tokens,
        "terminated_by_eos": result.terminated_by_eos,
        "truncated": result.truncated,
        "generation_stop_reason": result.stop_reason,
        "token_limit": token_limit,
        "style_plan": plan,
        **compaction,
        **sanitization,
        "evidence_manifest": evidence_manifest,
        "visible_evidence_chars": len(visible_evidence),
        "_visible_evidence_text": visible_evidence,
        "fewshot_examples": sum(
            float(item.get("similarity", 0.0))
            >= float(config.get("retrieval", {}).get("train_memory_fewshot_threshold", 0.70))
            for item in (retrieval_diagnostics or {}).get("train_memory", [])
        ),
    }
    if bool(config.get("generation", {}).get("log_raw_generated_answers", True)):
        generation_details["raw_answer"] = raw_answer
    generation_details["quality"] = analyze_answer_quality(
        question,
        answer,
        visible_evidence,
        generation_details,
    )
    return Candidate(
        "qwen3.5-2b",
        answer,
        details=generation_details,
    )


def make_qwen_repair_candidate(
    generator,
    question: str,
    original: Candidate,
    evidence: list[Evidence],
    config: dict,
    retrieval_diagnostics: dict | None = None,
) -> Candidate:
    """Run one bounded, evidence-grounded repair for a severely flawed draft."""
    quality = original.details.get("quality") or analyze_answer_quality(
        question,
        original.answer,
        original.details.get("_visible_evidence_text", "\n".join(item.chunk.text for item in evidence)),
        original.details,
    )
    issues = list(quality.get("repair_reasons", []))
    if not issues:
        raise ValueError("Candidate không có lỗi nghiêm trọng để repair.")
    route = (retrieval_diagnostics or {}).get("route", {}).get("route", "simple")
    plan = generation_plan(question, route, config)
    configured_limit = int(config.get("generation", {}).get("repair_max_new_tokens", 700))
    token_limit = min(configured_limit, int(plan["max_new_tokens"]))
    token_counter = _generator_token_counter(generator)
    prompt, visible_evidence, evidence_manifest = qwen_repair_prompt_bundle(
        question,
        original.answer,
        evidence,
        issues,
        config,
        retrieval_diagnostics,
        token_counter=token_counter,
    )
    result = generator.generate_with_metadata(
        SYSTEM_PROMPT,
        prompt,
        max_new_tokens=token_limit,
    )
    raw_answer = clean_answer(result.text)
    answer, sanitization = sanitize_answer(question, raw_answer)
    if not answer:
        raise RuntimeError("Qwen repair sinh câu trả lời rỗng.")
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
            hard_trim=bool(config.get("generation", {}).get("hard_trim_generated_answers", False)),
        )
    generation_details: dict[str, Any] = {
        "route": route,
        "generated_tokens": result.generated_tokens,
        "terminated_by_eos": result.terminated_by_eos,
        "truncated": result.truncated,
        "generation_stop_reason": result.stop_reason,
        "token_limit": token_limit,
        "style_plan": plan,
        **compaction,
        **sanitization,
        "repair_of": original.source,
        "repair_reasons": issues,
        "evidence_manifest": evidence_manifest,
        "visible_evidence_chars": len(visible_evidence),
        "_visible_evidence_text": visible_evidence,
    }
    if bool(config.get("generation", {}).get("log_raw_generated_answers", True)):
        generation_details["raw_answer"] = raw_answer
    generation_details["quality"] = analyze_answer_quality(
        question,
        answer,
        visible_evidence,
        generation_details,
    )
    return Candidate("qwen3.5-2b-repair", answer, details=generation_details)


_TOOL_CALL_MARKER = re.compile(
    r"^\s*(?:(?:[-*•>]\s*)|(?:\d+[.)]\s*))?`{0,3}\s*"
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
            query = re.split(r"[:：]", stripped, maxsplit=1)[-1].strip(" `")
            if not query:
                for continuation in lines[index + 1 :]:
                    continuation = continuation.strip()
                    if continuation and not continuation.startswith("```"):
                        query = continuation.strip(" `")
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
    token_counter = _generator_token_counter(generator)
    max_tool_calls = max(0, int(generation_cfg.get("agentic_max_tool_calls", 2)))
    evidence_per_call = max(1, int(generation_cfg.get("agentic_evidence_per_call", 3)))

    current_evidence = list(evidence)
    seen_chunk_ids = {item.chunk.chunk_id for item in current_evidence}
    tool_evidence_pool: list[Evidence] = []
    transcript: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    force_final_answer = False
    answer = ""
    generated_tokens = 0
    truncated = False
    terminated_by_eos = True
    stop_reason: str | None = None

    if (
        bool(generation_cfg.get("agentic_force_first_search", False))
        and search_fn is not None
        and max_tool_calls > 0
    ):
        seed_query = question[
            : max(32, int(generation_cfg.get("agentic_max_query_chars", 240)))
        ].strip()
        normalized_seed = _normalized_agent_query(seed_query)
        if normalized_seed:
            seed_results = _call_agentic_search(
                search_fn, seed_query, question, current_evidence
            )
            added = [
                item for item in seed_results
                if item.chunk.chunk_id not in seen_chunk_ids
            ][:evidence_per_call]
            for item in added:
                seen_chunk_ids.add(item.chunk.chunk_id)
            current_evidence = added + current_evidence
            tool_evidence_pool.extend(added)
            seen_queries.add(normalized_seed)
            transcript.append(
                {
                    "step": 1,
                    "query": seed_query,
                    "evidence_added": [item.chunk.chunk_id for item in added],
                    "result_count": len(seed_results),
                    "relation_hits": sum(bool(item.relation_type) for item in added),
                    "forced_by_gate": True,
                }
            )

    for step in range(max_tool_calls + 1):
        allow_tool = (
            search_fn is not None
            and len(seen_queries) < max_tool_calls
            and not force_final_answer
        )
        prompt = qwen_agentic_user_prompt(
            question, current_evidence, config, retrieval_diagnostics, transcript, allow_tool,
            token_counter=token_counter,
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
        stop_reason = result.stop_reason
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
                    "step": len(transcript) + 1,
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
        new_evidence = _call_agentic_search(search_fn, query, question, current_evidence)
        added = [item for item in new_evidence if item.chunk.chunk_id not in seen_chunk_ids][:evidence_per_call]
        for item in added:
            seen_chunk_ids.add(item.chunk.chunk_id)
        # Chèn căn cứ mới lên đầu danh sách: model chỉ tra cứu thêm khi thấy
        # thiếu, nên phần vừa tìm được cần ưu tiên ngân sách ký tự của
        # render_evidence hơn căn cứ ban đầu — nếu nối vào cuối, ngân sách
        # (max_context_chars) thường đã bị các đoạn ban đầu dùng hết và căn cứ
        # mới sẽ bị cắt bỏ hoàn toàn (xem prompts.render_evidence).
        current_evidence = added + current_evidence
        tool_evidence_pool.extend(added)
        transcript.append(
            {
                "step": len(transcript) + 1,
                "query": query,
                "evidence_added": [item.chunk.chunk_id for item in added],
                "result_count": len(new_evidence),
                "relation_hits": sum(bool(item.relation_type) for item in added),
            }
        )

    if not answer:
        raise RuntimeError("Qwen (agentic) sinh câu trả lời rỗng.")
    raw_answer = clean_answer(answer)
    answer, sanitization = sanitize_answer(question, raw_answer)
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
    visible_evidence, evidence_manifest = render_evidence_with_manifest(
        current_evidence,
        int(generation_cfg["max_context_chars"]),
        max_tokens=int(generation_cfg.get("max_context_tokens", 0)) or None,
        token_counter=token_counter,
    )
    candidate_details: dict[str, Any] = {
        "route": route,
        "generated_tokens": generated_tokens,
        "terminated_by_eos": terminated_by_eos,
        "truncated": truncated,
        "generation_stop_reason": stop_reason,
        "token_limit": token_limit,
        "style_plan": plan,
        **compaction,
        **sanitization,
        "tool_calls": transcript,
        "tool_calls_used": len(transcript),
        "evidence_added_total": len(tool_evidence_pool),
        "tool_retrieval_confidence": agentic_confidence,
        "evidence_manifest": evidence_manifest,
        "visible_evidence_chars": len(visible_evidence),
        "_visible_evidence_text": visible_evidence,
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
            }
            for item in tool_evidence_pool
        ],
    }
    if bool(generation_cfg.get("log_raw_generated_answers", True)):
        candidate_details["raw_answer"] = raw_answer
    candidate_details["quality"] = analyze_answer_quality(
        question, answer, visible_evidence, candidate_details
    )
    return Candidate(
        "qwen3.5-2b-agentic",
        answer,
        details=candidate_details,
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
    raw_answer = clean_answer(str(nearest.get("answer", "")))
    answer, sanitization = sanitize_answer(question, raw_answer)
    if not answer:
        return None
    route = diagnostics.get("route", {}).get("route", "simple")
    visible_evidence = "\n".join(item.chunk.text for item in evidence)
    details: dict[str, Any] = {
        "route": route,
        "memory_similarity": similarity,
        "memory_question_id": str(nearest.get("question_id", "")),
        "memory_question": str(nearest.get("question", "")),
        "style_plan": generation_plan(question, route, config),
        **sanitization,
        "_visible_evidence_text": visible_evidence,
    }
    details["quality"] = analyze_answer_quality(
        question, answer, visible_evidence, details
    )
    return Candidate(
        "train-memory",
        answer,
        details=details,
    )
