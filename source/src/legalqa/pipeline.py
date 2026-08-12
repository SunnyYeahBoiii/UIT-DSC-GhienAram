from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_qa, write_json
from .generation import Candidate, extractive_candidate, make_qwen_candidate
from .index import LegalIndex
from .models import (
    CrossEncoderReranker,
    DenseEncoder,
    LegalSemanticEncoder,
    QwenTextGenerator,
    unload_model,
)
from .retrieval import Evidence, HybridRetriever
from .recovery import RecoveryManager
from .verifier import score_candidates


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


def _fatal_generation_error(error: RuntimeError) -> bool:
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

    return generate_from_evidence(config, questions, evidence_by_id, output_path)


def generate_from_evidence(
    config: dict[str, Any],
    questions,
    evidence_by_id: dict[str, list[Evidence]],
    output_path: str | Path | None = None,
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

    semantic_encoder = None
    if model_paths.get("verifier") and float(config["selection"].get("legal_semantic", 0.0)) > 0:
        try:
            semantic_encoder = LegalSemanticEncoder(config)
            semantic_entries = []
            evidence_texts = []
            evidence_index: dict[str, int] = {}
            for item in questions:
                evidence_text = "\n".join(evidence.chunk.text for evidence in evidence_by_id[item.question_id])
                evidence_index[item.question_id] = len(evidence_texts)
                evidence_texts.append(evidence_text)
                for candidate in candidates_by_id[item.question_id]:
                    semantic_entries.append((item.question_id, candidate))
            if semantic_entries:
                print(
                    f"[ViLegalBERT] batching {len(semantic_entries):,} candidates and "
                    f"{len(evidence_texts):,} evidence sets",
                    flush=True,
                )
                answer_vectors = semantic_encoder.encode([candidate.answer for _, candidate in semantic_entries])
                evidence_vectors = semantic_encoder.encode(evidence_texts)
                for row, (question_id, candidate) in enumerate(semantic_entries):
                    cosine = float(np.dot(answer_vectors[row], evidence_vectors[evidence_index[question_id]]))
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
        )
        winner = ranked[0]
        predictions[item.question_id] = {"answer": winner.answer}
        traces.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "selected": winner.source,
                "evidence": _trace_evidence(evidence_by_id[item.question_id]),
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
        "selected_sources": selected_counts,
        "routes": route_counts,
        "qwen_truncated": truncated_qwen,
    }
