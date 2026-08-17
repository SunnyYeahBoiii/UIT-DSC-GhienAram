from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .core.data import Chunk, load_qa, write_json
from .generation.generation import (
    Candidate,
    extractive_candidate,
    make_qwen_agentic_candidate,
    make_qwen_candidate,
    make_train_memory_candidate,
)
from .retrieval.index import LegalIndex
from .retrieval.legal_graph import asks_for_current_law, has_complex_cue
from .core.models import (
    CrossEncoderReranker,
    DenseEncoder,
    LegalSemanticEncoder,
    QwenTextGenerator,
    unload_model,
)
from .retrieval.retrieval import Evidence, HybridRetriever, retrieval_confidence
from .core.recovery import RecoveryManager
from .core.text import citations, document_numbers
from .generation.verifier import score_candidates


def _candidate_key(question, evidence: list[Evidence], source: str, cache_namespace: str | None) -> str:
    payload = {
        "question_id": question.question_id,
        "question": question.question,
        "source": source,
        "cache_namespace": cache_namespace,
        "evidence": [item.chunk.chunk_id for item in evidence],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _load_candidate_cache(path: Path, include_metadata: bool = False) -> dict[str, Any]:
    result = {}
    if not path.exists():
        return result
    lines = path.read_text(encoding="utf-8").splitlines()
    valid_lines: list[str] = []
    repaired = False
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cache_key = item["cache_key"]
            answer = item["answer"]
            if not isinstance(cache_key, str) or not isinstance(answer, str):
                raise TypeError("cache_key/answer phải là chuỗi")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if any(remaining.strip() for remaining in lines[line_number:]):
                raise ValueError(f"Candidate cache hỏng ở giữa file {path}:{line_number}") from exc
            repaired = True
            print(
                f"[Candidate cache] bỏ dòng cuối dở dang {path}:{line_number}: {exc}",
                flush=True,
            )
            break
        result[cache_key] = (
            {"answer": answer, "metadata": item.get("metadata", {})}
            if include_metadata
            else answer
        )
        valid_lines.append(json.dumps(item, ensure_ascii=False))
    if repaired:
        temporary = path.with_name(f".{path.name}.repair.tmp")
        temporary.write_text(
            "".join(f"{line}\n" for line in valid_lines),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    return result


def _append_candidate(
    path: Path,
    cache_key: str,
    question_id: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "question_id": question_id,
                    "answer": answer,
                    "metadata": metadata or {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def _fatal_generation_error(error: Exception) -> bool:
    message = str(error).lower()
    fatal_markers = (
        "out of memory",
        "cuda error",
        "cublas",
        "cudnn",
        "device-side assert",
        "illegal memory access",
        "no kernel image",
        "expected all tensors to be on the same device",
        "cannot move a model",
        "can't move a model",
    )
    return any(marker in message for marker in fatal_markers)


def _tool_evidence_from_metadata(tool_evidence: list[dict[str, Any]]) -> list[Evidence]:
    """Rebuild cached tool evidence without mutating the shared seed evidence."""
    result: list[Evidence] = []
    seen: set[str] = set()
    for entry in tool_evidence:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id", "")).strip()
        doc_id = str(entry.get("doc_id", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not chunk_id or not doc_id or not text or chunk_id in seen:
            continue
        result.append(
            Evidence(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    title=str(entry.get("title", "")),
                    heading=str(entry.get("heading", "")),
                ),
                score=float(entry.get("score", 0.0)),
                bm25_score=float(entry.get("bm25_score", 0.0)),
                stage=str(entry.get("stage", "tool_call")),
                relation_type=str(entry.get("relation_type", "")),
                relation_path=[str(value) for value in entry.get("relation_path", [])],
            )
        )
        seen.add(chunk_id)
    return result


def _candidate_evidence(seed_evidence: list[Evidence], candidate: Candidate) -> list[Evidence]:
    """Return exactly the evidence visible to one candidate.

    Only the agentic candidate may see its cached tool results.  This prevents
    ordinary Qwen/extractive/memory candidates from receiving verifier credit
    for evidence that they never saw while being generated.
    """
    if candidate.source != "qwen3.5-2b-agentic":
        return list(seed_evidence)
    tool_items = _tool_evidence_from_metadata(candidate.details.get("tool_evidence", []))
    combined: list[Evidence] = []
    seen: set[str] = set()
    for item in [*tool_items, *seed_evidence]:
        if item.chunk.chunk_id in seen:
            continue
        combined.append(item)
        seen.add(item.chunk.chunk_id)
    return combined


def _should_run_agentic(
    question: str,
    evidence: list[Evidence],
    diagnostics: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Gate expensive Agentic Qwen to questions likely to need more evidence."""
    generation_cfg = config.get("generation", {})
    route_payload = (diagnostics or {}).get("route", {})
    route = str(route_payload.get("route", "simple"))
    confidence = route_payload.get("confidence")
    if confidence is None:
        confidence, _ = retrieval_confidence(evidence)
    confidence = float(confidence)
    enabled_routes = {
        str(value).casefold()
        for value in generation_cfg.get("agentic_routes", ["complex"])
    }
    if route.casefold() in enabled_routes:
        return True, f"route={route}"
    threshold = float(generation_cfg.get("agentic_confidence_below", 0.45))
    if confidence < threshold:
        return True, f"retrieval_confidence={confidence:.3f}<{threshold:.3f}"
    if bool(generation_cfg.get("agentic_on_legal_cues", True)):
        features = route_payload.get("features", {})
        explicit_citation = bool(
            features.get("explicit_citation", 0.0)
            or citations(question)
            or document_numbers(question)
        )
        relation_signal = bool(features.get("discovered_relations", 0.0))
        if has_complex_cue(question) or asks_for_current_law(question):
            return True, "temporal/complex legal cue"
        if explicit_citation and relation_signal:
            return True, "explicit citation with relation signal"
    return False, f"simple/high-confidence={confidence:.3f}"


def _trace_evidence(items: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "doc_id": item.chunk.doc_id,
            "score": item.score,
            "bm25_score": item.bm25_score,
            "dense_score": item.dense_score,
            "reranker_score": item.reranker_score,
            "stage": item.stage,
            "relation_type": item.relation_type,
            "relation_path": item.relation_path,
            "text_preview": item.chunk.text[:500],
        }
        for item in items
    ]


def run_pipeline(
    config: dict[str, Any],
    questions_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    questions = load_qa(questions_path, require_answers=False)
    artifacts = Path(config["paths"]["artifacts"])
    index = LegalIndex.load(artifacts / "index")
    model_paths = config.get("paths", {}).get("models", {})

    dense_encoder = None
    if index.dense is not None and model_paths.get("embedding") and int(config["retrieval"].get("dense_top_k", 0)) > 0:
        dense_encoder = DenseEncoder(config)
    reranker = None
    if model_paths.get("reranker") and int(config["retrieval"].get("rerank_top_k", 0)) > 0:
        reranker = CrossEncoderReranker(config)
    retriever = HybridRetriever(config, index, dense_encoder, reranker)
    evidence_by_id = {item.question_id: retriever.retrieve(item.question) for item in questions}
    if config["runtime"].get("unload_after_stage", True):
        unload_model(dense_encoder)
        unload_model(reranker)
        dense_encoder = reranker = None

    # retriever.retrieve_bm25_only chỉ dùng self.index.bm25 (không phải model GPU),
    # nên vẫn hoạt động an toàn sau khi dense_encoder/reranker đã unload ở trên.
    agentic_search_fn = (
        retriever.retrieve_bm25_only if config["generation"].get("agentic_enabled", False) else None
    )
    return generate_from_evidence(config, questions, evidence_by_id, output_path, agentic_search_fn)


def generate_from_evidence(
    config: dict[str, Any],
    questions,
    evidence_by_id: dict[str, list[Evidence]],
    output_path: str | Path | None = None,
    agentic_search_fn: Callable[..., list[Evidence]] | None = None,
) -> dict[str, Any]:
    """Generate and select answers from retrieval results, independent of index backend."""
    artifacts = Path(config["paths"]["artifacts"])
    model_paths = config.get("paths", {}).get("models", {})
    candidate_cache_dir = artifacts / "candidate_cache"
    recovery = RecoveryManager.from_config(config)
    run_namespace = config.get("runtime", {}).get("run_fingerprint")
    retrieval_diagnostics = getattr(evidence_by_id, "diagnostics", {})

    candidates_by_id: dict[str, list[Candidate]] = {item.question_id: [] for item in questions}
    errors: dict[str, list[str]] = {}
    agentic_eligible_count = 0
    agentic_generated_count = 0
    agentic_failed_count = 0
    if config["generation"].get("extractive_enabled", True):
        for item in questions:
            candidates_by_id[item.question_id].append(
                extractive_candidate(
                    item.question,
                    evidence_by_id[item.question_id],
                    config,
                    retrieval_diagnostics.get(item.question_id),
                )
            )

    if config.get("retrieval", {}).get("train_memory_candidate_enabled", True):
        for item in questions:
            candidate = make_train_memory_candidate(
                item.question,
                evidence_by_id[item.question_id],
                config,
                retrieval_diagnostics.get(item.question_id),
            )
            if candidate is not None:
                candidates_by_id[item.question_id].append(candidate)

    if config["generation"].get("qwen_enabled", True) and model_paths.get("generator"):
        cache_path = candidate_cache_dir / "qwen.jsonl"
        cached = _load_candidate_cache(cache_path, include_metadata=True)
        qwen_namespace = run_namespace or model_paths.get("generator_adapter") or model_paths.get("generator")
        missing = [
            item for item in questions
            if _candidate_key(item, evidence_by_id[item.question_id], "qwen", qwen_namespace) not in cached
        ]
        completed = len(questions) - len(missing)
        generator = None
        consecutive_errors = 0
        try:
            generator = QwenTextGenerator(config) if missing else None
            for item in questions:
                key = _candidate_key(
                    item, evidence_by_id[item.question_id], "qwen",
                    qwen_namespace,
                )
                if key in cached:
                    entry = cached[key]
                    candidates_by_id[item.question_id].append(
                        Candidate("qwen3.5-2b", entry["answer"], details=entry.get("metadata", {}))
                    )
                    continue
                try:
                    candidate = make_qwen_candidate(
                        generator,
                        item.question,
                        evidence_by_id[item.question_id],
                        config,
                        retrieval_diagnostics.get(item.question_id),
                    )
                    candidates_by_id[item.question_id].append(candidate)
                    _append_candidate(
                        cache_path,
                        key,
                        item.question_id,
                        candidate.answer,
                        candidate.details,
                    )
                    cached[key] = {"answer": candidate.answer, "metadata": candidate.details}
                    completed += 1
                    consecutive_errors = 0
                    if recovery:
                        recovery.snapshot("qwen", completed, len(questions))
                    if completed % 25 == 0:
                        print(f"[Qwen] generated {completed:,}/{len(questions):,}", flush=True)
                except RuntimeError as exc:
                    errors.setdefault(item.question_id, []).append(f"Qwen: {exc}")
                    consecutive_errors += 1
                    if _fatal_generation_error(exc) or consecutive_errors >= 3:
                        raise RuntimeError(
                            f"Qwen dừng tại question_id={item.question_id} sau "
                            f"{consecutive_errors} lỗi liên tiếp; cache đã hoàn thành={completed}."
                        ) from exc
        finally:
            if config["runtime"].get("unload_after_stage", True):
                unload_model(generator, "Qwen")
            if recovery:
                recovery.snapshot("qwen", completed, len(questions), force=True)

    if config["generation"].get("agentic_enabled", False) and model_paths.get("generator"):
        if agentic_search_fn is None:
            print(
                "[Qwen-agentic] agentic_enabled=true nhưng caller không truyền agentic_search_fn "
                "(backend truy hồi này chưa hỗ trợ tool-call); bỏ qua ứng viên agentic.",
                flush=True,
            )
        else:
            cache_path = candidate_cache_dir / "qwen_agentic.jsonl"
            cached = _load_candidate_cache(cache_path, include_metadata=True)
            base_namespace = run_namespace or model_paths.get("generator_adapter") or model_paths.get("generator")
            agentic_options = {
                key: value
                for section in (config.get("generation", {}), config.get("retrieval", {}))
                for key, value in section.items()
                if key.startswith("agentic_")
            }
            options_hash = hashlib.sha256(
                json.dumps(agentic_options, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            agentic_namespace = (
                f"{base_namespace}::agentic-v3::{options_hash}"
            )
            eligible: list[tuple[Any, str]] = []
            for item in questions:
                allowed, reason = _should_run_agentic(
                    item.question,
                    evidence_by_id[item.question_id],
                    retrieval_diagnostics.get(item.question_id),
                    config,
                )
                if allowed:
                    eligible.append((item, reason))
            agentic_eligible_count = len(eligible)
            print(
                f"[Qwen-agentic] gate selected {agentic_eligible_count:,}/{len(questions):,} questions",
                flush=True,
            )
            missing = [
                item for item, _ in eligible
                if _candidate_key(item, evidence_by_id[item.question_id], "qwen-agentic", agentic_namespace) not in cached
            ]
            completed = len(eligible) - len(missing)
            generator = None
            consecutive_errors = 0
            try:
                generator = QwenTextGenerator(config) if missing else None
                for item, gate_reason in eligible:
                    key = _candidate_key(
                        item, evidence_by_id[item.question_id], "qwen-agentic",
                        agentic_namespace,
                    )
                    if key in cached:
                        entry = cached[key]
                        metadata = dict(entry.get("metadata", {}))
                        metadata.setdefault("agentic_gate", gate_reason)
                        candidates_by_id[item.question_id].append(
                            Candidate("qwen3.5-2b-agentic", entry["answer"], details=metadata)
                        )
                        agentic_generated_count += 1
                        continue
                    try:
                        candidate = make_qwen_agentic_candidate(
                            generator,
                            item.question,
                            evidence_by_id[item.question_id],
                            config,
                            agentic_search_fn,
                            retrieval_diagnostics.get(item.question_id),
                        )
                        candidate.details["agentic_gate"] = gate_reason
                        candidates_by_id[item.question_id].append(candidate)
                        _append_candidate(
                            cache_path,
                            key,
                            item.question_id,
                            candidate.answer,
                            candidate.details,
                        )
                        cached[key] = {"answer": candidate.answer, "metadata": candidate.details}
                        completed += 1
                        agentic_generated_count += 1
                        consecutive_errors = 0
                        if recovery:
                            recovery.snapshot("qwen_agentic", completed, len(eligible))
                        if completed % 25 == 0:
                            print(f"[Qwen-agentic] processed {completed:,}/{len(eligible):,}", flush=True)
                    except Exception as exc:
                        # Agentic Search is an optional candidate branch. Any
                        # question-local parser/search/model error must not
                        # discard the already valid base/extractive/memory
                        # candidates. BaseException subclasses (interrupts and
                        # process exits) are deliberately not swallowed; fatal
                        # CUDA failures are promoted below for safe resume.
                        errors.setdefault(item.question_id, []).append(f"Qwen-agentic: {exc}")
                        if _fatal_generation_error(exc):
                            raise RuntimeError(
                                f"Qwen-agentic gặp lỗi GPU nghiêm trọng tại question_id={item.question_id}; "
                                f"cache đã hoàn thành={completed}."
                            ) from exc
                        consecutive_errors += 1
                        completed += 1
                        agentic_failed_count += 1
                        if recovery:
                            recovery.snapshot("qwen_agentic", completed, len(eligible))
                        warn_after = max(
                            1,
                            int(
                                config.get("generation", {}).get(
                                    "agentic_warn_after_nonfatal_errors", 3
                                )
                            ),
                        )
                        if consecutive_errors == warn_after:
                            print(
                                f"[Qwen-agentic] {consecutive_errors} lỗi không nghiêm trọng liên tiếp; "
                                "tiếp tục các câu còn lại; mỗi câu lỗi vẫn dùng candidate base/extractive/memory.",
                                flush=True,
                            )
            finally:
                if config["runtime"].get("unload_after_stage", True):
                    unload_model(generator, "Qwen-agentic")
                if recovery:
                    recovery.snapshot("qwen_agentic", completed, len(eligible), force=True)

    semantic_encoder = None
    if model_paths.get("verifier") and float(config["selection"].get("legal_semantic", 0.0)) > 0:
        try:
            semantic_encoder = LegalSemanticEncoder(config)
            semantic_entries = []
            evidence_texts = []
            evidence_index: dict[tuple[str, ...], int] = {}
            candidate_evidence_keys: dict[int, tuple[str, ...]] = {}
            for item in questions:
                for candidate in candidates_by_id[item.question_id]:
                    semantic_entries.append((item.question_id, candidate))
                    active_evidence = _candidate_evidence(
                        evidence_by_id[item.question_id], candidate
                    )
                    evidence_key = tuple(evidence.chunk.chunk_id for evidence in active_evidence)
                    candidate_evidence_keys[id(candidate)] = evidence_key
                    if evidence_key not in evidence_index:
                        evidence_index[evidence_key] = len(evidence_texts)
                        evidence_texts.append(
                            "\n".join(evidence.chunk.text for evidence in active_evidence)
                        )
            if semantic_entries:
                print(
                    f"[ViLegalBERT] batching {len(semantic_entries):,} candidates and "
                    f"{len(evidence_texts):,} evidence sets",
                    flush=True,
                )
                answer_vectors = semantic_encoder.encode([candidate.answer for _, candidate in semantic_entries])
                evidence_vectors = semantic_encoder.encode(evidence_texts)
                for row, (_, candidate) in enumerate(semantic_entries):
                    key = candidate_evidence_keys[id(candidate)]
                    cosine = float(np.dot(answer_vectors[row], evidence_vectors[evidence_index[key]]))
                    candidate.details["_precomputed_legal_semantic"] = max(
                        0.0, min(1.0, (cosine + 1.0) / 2.0)
                    )
        finally:
            unload_model(semantic_encoder, "ViLegalBERT")
            semantic_encoder = None

    predictions: dict[str, dict[str, str]] = {}
    traces: list[dict[str, Any]] = []
    for item in questions:
        candidates = candidates_by_id[item.question_id]
        if not candidates:
            candidates = [Candidate("fallback", "Các căn cứ được cung cấp chưa đủ để xác định câu trả lời.")]
        ranked = score_candidates(
            item.question,
            evidence_by_id[item.question_id],
            candidates,
            config,
            semantic_encoder,
            retrieval_diagnostics.get(item.question_id),
            candidate_evidence_by_source={
                candidate.source: _candidate_evidence(
                    evidence_by_id[item.question_id], candidate
                )
                for candidate in candidates
            },
        )
        winner = ranked[0]
        predictions[item.question_id] = {"answer": winner.answer}
        traces.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "selected": winner.source,
                "evidence": _trace_evidence(evidence_by_id[item.question_id]),
                "agentic_evidence": {
                    candidate.source: _trace_evidence(
                        _candidate_evidence(evidence_by_id[item.question_id], candidate)
                    )
                    for candidate in candidates
                    if candidate.source == "qwen3.5-2b-agentic"
                },
                "retrieval": retrieval_diagnostics.get(item.question_id, {}),
                "candidates": [asdict(candidate) for candidate in ranked],
                "errors": errors.get(item.question_id, []),
            }
        )
    unload_model(semantic_encoder)

    target = Path(output_path) if output_path else artifacts / "predictions.json"
    write_json(target, predictions)
    trace_path = target.with_name(f"{target.stem}.trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    if recovery:
        recovery.snapshot("selection", len(questions), len(questions), force=True)
    selected_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    truncated_qwen = 0
    for trace in traces:
        selected_counts[trace["selected"]] = selected_counts.get(trace["selected"], 0) + 1
        route = trace.get("retrieval", {}).get("route", {}).get("route", "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        truncated_qwen += any(
            candidate["source"].startswith("qwen")
            and bool(candidate.get("details", {}).get("truncated", False))
            for candidate in trace["candidates"]
        )
    return {
        "questions": len(questions),
        "predictions": str(target.resolve()),
        "trace": str(trace_path.resolve()),
        "generation_errors": sum(len(value) for value in errors.values()),
        "agentic": {
            "eligible": agentic_eligible_count,
            "generated_or_cached": agentic_generated_count,
            "failed": agentic_failed_count,
        },
        "selected_sources": selected_counts,
        "routes": route_counts,
        "qwen_truncated": truncated_qwen,
    }
