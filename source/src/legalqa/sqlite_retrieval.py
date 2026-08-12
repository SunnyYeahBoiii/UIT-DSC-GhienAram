from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import Chunk, QARecord
from .dense_store import ensure_global_dense_index, search_global_dense_index
from .legal_graph import asks_for_current_law, temporal_relation_adjustment
from .models import CrossEncoderReranker, DenseEncoder, sigmoid, unload_model
from .retrieval import Evidence, deduplicate_and_merge_evidence
from .routing import decide_route
from .sqlite_store import SQLiteHit, SQLiteLegalStore
from .text import citations, document_numbers, numbers
from .train_memory import ensure_train_memory_index, search_train_memory


class RetrievalBatch(dict[str, list[Evidence]]):
    """Mapping-compatible retrieval output with per-question diagnostics."""

    def __init__(
        self,
        evidence: dict[str, list[Evidence]] | None = None,
        diagnostics: dict[str, dict[str, Any]] | None = None,
        index_report: dict[str, Any] | None = None,
        memory_report: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(evidence or {})
        self.diagnostics = diagnostics or {}
        self.index_report = index_report or {}
        self.memory_report = memory_report or {}


def _evidence_payload(evidence: Evidence, include_chunk: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": evidence.score,
        "bm25_score": evidence.bm25_score,
        "dense_score": evidence.dense_score,
        "reranker_score": evidence.reranker_score,
        "stage": evidence.stage,
        "relation_type": evidence.relation_type,
        "relation_path": evidence.relation_path,
    }
    if include_chunk:
        payload["chunk"] = asdict(evidence.chunk)
    else:
        payload.update(
            chunk_id=evidence.chunk.chunk_id,
            doc_id=evidence.chunk.doc_id,
            title=evidence.chunk.title,
            heading=evidence.chunk.heading,
        )
    return payload


def _save_cache(path: Path, questions: list[QARecord], batch: RetrievalBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            question_id = question.question_id
            handle.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "evidence": [
                            _evidence_payload(item, include_chunk=True) for item in batch[question_id]
                        ],
                        "diagnostics": batch.diagnostics.get(question_id, {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_retrieval_cache(path: str | Path) -> RetrievalBatch:
    result: dict[str, list[Evidence]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in result:
                raise ValueError(f"Retrieval cache lặp question_id={question_id!r}")
            result[question_id] = [
                Evidence(
                    chunk=Chunk(**item["chunk"]),
                    score=float(item["score"]),
                    bm25_score=float(item.get("bm25_score", 0.0)),
                    dense_score=float(item.get("dense_score", 0.0)),
                    reranker_score=float(item.get("reranker_score", 0.0)),
                    stage=str(item.get("stage", "initial")),
                    relation_type=str(item.get("relation_type", "")),
                    relation_path=[str(value) for value in item.get("relation_path", [])],
                )
                for item in row["evidence"]
            ]
            diagnostics[question_id] = row.get("diagnostics", {})
    return RetrievalBatch(result, diagnostics)


def _citation_bonus(question: str, hit: SQLiteHit, config: dict[str, Any]) -> float:
    cfg = config["retrieval"]
    chunk_text = f"{hit.chunk.title}\n{hit.chunk.heading}\n{hit.chunk.text}"
    query_docs, chunk_docs = document_numbers(question), document_numbers(chunk_text)
    query_articles, chunk_articles = citations(question), citations(chunk_text)
    query_numbers, chunk_numbers = numbers(question), numbers(chunk_text)
    exact_document = len(query_docs & chunk_docs)
    exact_article = len(query_articles & chunk_articles)
    generic_numbers = len(query_numbers & chunk_numbers)
    return (
        float(cfg.get("exact_document_boost", 0.34)) * exact_document
        + float(cfg.get("exact_article_boost", 0.20)) * exact_article
        + float(cfg.get("generic_number_boost", 0.01)) * generic_numbers
    )


def _trace_rank(items: list[Evidence], limit: int) -> list[dict[str, Any]]:
    return [_evidence_payload(item, include_chunk=False) for item in items[:limit]]


def _rerank_batch(
    reranker: CrossEncoderReranker | None,
    questions_by_id: dict[str, str],
    candidates_by_id: dict[str, list[Evidence]],
    weight: float,
) -> None:
    if reranker is None:
        return
    pairs: list[tuple[str, str]] = []
    locations: list[tuple[str, int]] = []
    for question_id, items in candidates_by_id.items():
        for index, evidence in enumerate(items):
            pairs.append((questions_by_id[question_id], evidence.chunk.text))
            locations.append((question_id, index))
    values = reranker.score_pairs(pairs)
    for (question_id, index), raw_value in zip(locations, values):
        evidence = candidates_by_id[question_id][index]
        evidence.reranker_score = sigmoid(raw_value)
        evidence.score += weight * evidence.reranker_score


def _filter_graph_candidates(
    question: str,
    route_features: dict[str, float],
    initial: list[Evidence],
    proposed: list[Evidence],
    retrieval_cfg: dict[str, Any],
    *,
    reranker_available: bool,
) -> tuple[list[Evidence], dict[str, float | None]]:
    """Keep graph chunks only when they remain relevant after relation expansion."""
    floor = float(retrieval_cfg.get("graph_min_reranker", 0.12))
    relative = float(retrieval_cfg.get("graph_relative_reranker", 0.30))
    initial_top = max((item.reranker_score for item in initial[:5]), default=0.0)
    threshold = max(floor, relative * initial_top)
    if asks_for_current_law(question) or route_features.get("complex_cue", 0.0) > 0:
        discount = float(
            retrieval_cfg.get("graph_temporal_threshold_discount", 0.60)
        )
        threshold = max(floor * 0.5, threshold * discount)
    lexical_threshold = float(
        retrieval_cfg.get("graph_min_lexical_without_reranker", 2.0)
    )
    if reranker_available:
        accepted = [item for item in proposed if item.reranker_score >= threshold]
    else:
        accepted = [item for item in proposed if item.bm25_score >= lexical_threshold]
    return accepted, {
        "reranker_threshold": threshold if reranker_available else None,
        "lexical_threshold": lexical_threshold if not reranker_available else None,
    }


def retrieve_sqlite_batch(
    config: dict[str, Any],
    database_path: str | Path,
    questions: list[QARecord],
    cache_path: str | Path | None = None,
    reuse_cache: bool = True,
) -> RetrievalBatch:
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists() and reuse_cache:
        try:
            loaded = load_retrieval_cache(cache)
            expected = {item.question_id for item in questions}
            if set(loaded) == expected and all(loaded.diagnostics.get(key) for key in expected):
                print(f"[Retrieval] reused graph-gated cache: {cache}", flush=True)
                return loaded
            print(
                f"[Retrieval] cache thiếu/thừa ID hoặc diagnostics; sẽ tính lại: "
                f"expected={len(expected):,}, actual={len(loaded):,}",
                flush=True,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"[Retrieval] cache hỏng/dở dang; sẽ tính lại: {exc}", flush=True)

    retrieval_cfg = config["retrieval"]
    lexical_k = int(retrieval_cfg.get("bm25_top_k", 160))
    dense_k = int(retrieval_cfg.get("dense_top_k", 80))
    fused_k = int(retrieval_cfg.get("fused_top_k", 48))
    rerank_k = int(retrieval_cfg.get("rerank_top_k", 30))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))
    trace_fused_k = int(retrieval_cfg.get("trace_fused_top_k", fused_k))
    trace_rerank_k = int(retrieval_cfg.get("trace_rerank_top_k", rerank_k))
    trace_lexical_k = int(retrieval_cfg.get("trace_lexical_top_k", min(40, lexical_k)))
    trace_dense_k = int(retrieval_cfg.get("trace_dense_top_k", min(40, dense_k)))
    questions_by_id = {item.question_id: item.question for item in questions}
    model_paths = config.get("paths", {}).get("models", {})

    lexical_by_id: dict[str, list[SQLiteHit]] = {}
    dense_by_id: dict[str, list[tuple[SQLiteHit, float]]] = {item.question_id: [] for item in questions}
    memory_by_id: dict[str, list[dict[str, Any]]] = {
        item.question_id: [] for item in questions
    }
    dense_index_report: dict[str, Any] = {"enabled": False}
    memory_index_report: dict[str, Any] = {"enabled": False}
    with SQLiteLegalStore(database_path, readonly=True) as store:
        for question_number, question in enumerate(questions, 1):
            lexical_by_id[question.question_id] = store.search_chunks(
                question.question, top_k=lexical_k
            )
            if question_number % 100 == 0:
                print(f"[Global BM25] {question_number:,}/{len(questions):,}", flush=True)

        use_dense = bool(
            retrieval_cfg.get("global_dense_enabled", True)
            and model_paths.get("embedding")
            and dense_k > 0
        )
        train_path = config.get("runtime", {}).get("train_path")
        use_memory = bool(
            retrieval_cfg.get("train_memory_enabled", True)
            and model_paths.get("embedding")
            and train_path
            and int(retrieval_cfg.get("train_memory_top_k", 3)) > 0
        )
        if use_dense or use_memory:
            print("[Global dense] loading embedding model", flush=True)
            encoder = DenseEncoder(config)
            try:
                query_vectors = encoder.encode([item.question for item in questions])
                if use_dense:
                    dense_dir = Path(
                        config.get("runtime", {}).get("dense_index_dir")
                        or Path(database_path).with_name("global_dense_index")
                    )
                    dense_index_report = ensure_global_dense_index(
                        config, store, encoder, dense_dir
                    )
                    dense_results = search_global_dense_index(
                        query_vectors,
                        dense_dir,
                        dense_k,
                        device=str(config["runtime"].get("device", "cuda:0")),
                        block_size=int(retrieval_cfg.get("dense_search_block", 8192)),
                    )
                    for question, ranked in zip(questions, dense_results):
                        rowids = [rowid for rowid, _ in ranked]
                        hit_map = {hit.rowid: hit for hit in store.hits_for_rowids(rowids)}
                        dense_by_id[question.question_id] = [
                            (hit_map[rowid], score)
                            for rowid, score in ranked
                            if rowid in hit_map
                        ]
                if use_memory:
                    memory_dir = Path(
                        config.get("runtime", {}).get("train_memory_dir")
                        or Path(database_path).with_name("global_dense_index") / "train_memory"
                    )
                    memory_index_report = ensure_train_memory_index(
                        config, encoder, train_path, memory_dir
                    )
                    memory_results = search_train_memory(
                        query_vectors,
                        memory_dir,
                        int(retrieval_cfg.get("train_memory_top_k", 3)),
                        device=str(config["runtime"].get("device", "cuda:0")),
                        exclude_question_ids={item.question_id for item in questions},
                    )
                    for question, ranked in zip(questions, memory_results):
                        memory_by_id[question.question_id] = ranked
            finally:
                unload_model(encoder, "global embedding")

        fused_by_id: dict[str, list[Evidence]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for question in questions:
            lexical = lexical_by_id[question.question_id]
            dense = dense_by_id[question.question_id]
            lexical_rank = {hit.rowid: rank for rank, hit in enumerate(lexical, 1)}
            dense_rank = {hit.rowid: rank for rank, (hit, _) in enumerate(dense, 1)}
            lexical_map = {hit.rowid: hit for hit in lexical}
            dense_map = {hit.rowid: (hit, score) for hit, score in dense}
            rowids = set(lexical_map) | set(dense_map)
            items: list[Evidence] = []
            for rowid in rowids:
                hit = lexical_map.get(rowid) or dense_map[rowid][0]
                score = 0.0
                if rowid in lexical_rank:
                    score += float(retrieval_cfg.get("bm25_weight", 1.0)) / (
                        rrf_k + lexical_rank[rowid]
                    )
                if rowid in dense_rank:
                    score += float(retrieval_cfg.get("dense_weight", 1.1)) / (
                        rrf_k + dense_rank[rowid]
                    )
                score += _citation_bonus(question.question, hit, config)
                items.append(
                    Evidence(
                        chunk=hit.chunk,
                        score=score,
                        bm25_score=lexical_map[rowid].lexical_score if rowid in lexical_map else 0.0,
                        dense_score=dense_map[rowid][1] if rowid in dense_map else 0.0,
                        stage="initial",
                    )
                )
            items.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
            fused_by_id[question.question_id] = items[:fused_k]
            diagnostics[question.question_id] = {
                "lexical_top": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "bm25_score": hit.lexical_score,
                    }
                    for hit in lexical[:trace_lexical_k]
                ],
                "dense_top": [
                    {"chunk_id": hit.chunk.chunk_id, "doc_id": hit.chunk.doc_id, "dense_score": score}
                    for hit, score in dense[:trace_dense_k]
                ],
                "fused_top": _trace_rank(items, trace_fused_k),
                "train_memory": memory_by_id[question.question_id],
            }

        reranker = None
        if model_paths.get("reranker") and rerank_k > 0:
            print("[Retrieval reranker] initial cross-encoder pass", flush=True)
            reranker = CrossEncoderReranker(config)
            initial_for_rerank = {
                question_id: items[:rerank_k] for question_id, items in fused_by_id.items()
            }
            _rerank_batch(
                reranker,
                questions_by_id,
                initial_for_rerank,
                float(retrieval_cfg.get("reranker_weight", 0.72)),
            )

        routes = {}
        for question in questions:
            items = sorted(
                fused_by_id[question.question_id],
                key=lambda item: (-item.score, item.chunk.chunk_id),
            )
            fused_by_id[question.question_id] = items
            top_docs = list(dict.fromkeys(item.chunk.doc_id for item in items[:5]))
            routes[question.question_id] = decide_route(
                question.question,
                items,
                store.relation_count(top_docs),
                config,
            )

        # Deterministic, bounded legal-retrieval agent. Unlike v8, every graph
        # hop is reranked and relevance-gated *before* its documents are allowed
        # to become the next frontier.
        extras_by_id: dict[str, list[Evidence]] = {item.question_id: [] for item in questions}
        iteration_logs: dict[str, list[dict[str, Any]]] = {item.question_id: [] for item in questions}
        max_iterations = max(1, min(4, int(config.get("routing", {}).get("max_iterations", 2))))
        max_new_docs = max(1, int(config.get("routing", {}).get("max_new_docs_per_iteration", 3)))
        graph_top_per_doc = max(1, int(config.get("routing", {}).get("graph_chunks_per_document", 2)))
        graph_states: dict[str, dict[str, Any]] = {}
        for question in questions:
            question_id = question.question_id
            initial = fused_by_id[question_id]
            if not initial:
                continue
            if routes[question_id].route == "simple":
                for hit in store.neighbor_chunks(initial[0].chunk.chunk_id, radius=1):
                    if hit.chunk.chunk_id == initial[0].chunk.chunk_id:
                        continue
                    extras_by_id[question_id].append(
                        Evidence(
                            hit.chunk,
                            score=initial[0].score * 0.78,
                            bm25_score=hit.lexical_score,
                            stage="parent_child",
                            relation_type="PARENT_CHILD",
                            relation_path=[initial[0].chunk.chunk_id, hit.chunk.chunk_id],
                        )
                    )
                continue
            frontier = list(dict.fromkeys(item.chunk.doc_id for item in initial[:3]))
            graph_states[question_id] = {
                "frontier": frontier,
                "visited": set(frontier),
                "active": True,
            }

        if reranker is not None:
            simple_extras = {
                question_id: items
                for question_id, items in extras_by_id.items()
                if routes[question_id].route == "simple" and items
            }
            _rerank_batch(
                reranker,
                questions_by_id,
                simple_extras,
                float(retrieval_cfg.get("reranker_weight", 0.72)),
            )

        for iteration in range(1, max_iterations + 1):
            proposals: dict[str, list[Evidence]] = {}
            proposal_meta: dict[str, dict[str, Any]] = {}
            for question in questions:
                question_id = question.question_id
                state = graph_states.get(question_id)
                if not state or not state["active"]:
                    continue
                frontier = list(state["frontier"])
                visited = set(state["visited"])
                edges = store.relation_edges(frontier)
                neighbors: dict[str, tuple[float, Any]] = {}
                for edge in edges:
                    neighbor = edge.neighbor_doc_id
                    if neighbor in visited:
                        continue
                    adjustment = temporal_relation_adjustment(edge, question.question)
                    score = edge.confidence + adjustment
                    if neighbor not in neighbors or score > neighbors[neighbor][0]:
                        neighbors[neighbor] = (score, edge)
                chosen = sorted(
                    neighbors.items(), key=lambda item: (-item[1][0], item[0])
                )[:max_new_docs]
                if not chosen:
                    iteration_logs[question_id].append(
                        {"iteration": iteration, "frontier": frontier, "new_documents": [], "stopped": "no_new_relation"}
                    )
                    state["active"] = False
                    continue
                new_docs = [doc_id for doc_id, _ in chosen]
                hits = store.rank_chunks_in_documents(
                    question.question, new_docs, top_per_document=graph_top_per_doc
                )
                edge_by_doc = {doc_id: value[1] for doc_id, value in chosen}
                relation_score_by_doc = {doc_id: value[0] for doc_id, value in chosen}
                proposed_items: list[Evidence] = []
                for hit in hits:
                    edge = edge_by_doc[hit.chunk.doc_id]
                    relation_score = relation_score_by_doc[hit.chunk.doc_id]
                    proposed_items.append(
                        Evidence(
                            hit.chunk,
                            score=(
                                float(retrieval_cfg.get("graph_base_score", 0.03))
                                + float(retrieval_cfg.get("graph_relation_weight", 0.16)) * relation_score
                                + 0.04 * hit.lexical_score
                            ),
                            bm25_score=hit.lexical_score,
                            stage=f"graph_{iteration}",
                            relation_type=edge.relation_type,
                            relation_path=[
                                edge.source_doc_id,
                                edge.relation_type,
                                edge.target_doc_id,
                            ],
                        )
                    )
                proposals[question_id] = proposed_items
                proposal_meta[question_id] = {
                    "frontier": frontier,
                    "new_docs": new_docs,
                    "edge_by_doc": edge_by_doc,
                }

            if not proposals:
                break
            if reranker is not None:
                print(
                    f"[Retrieval reranker] graph relevance gate iteration {iteration}",
                    flush=True,
                )
                _rerank_batch(
                    reranker,
                    questions_by_id,
                    proposals,
                    float(retrieval_cfg.get("reranker_weight", 0.72)),
                )

            accepted_in_iteration = 0
            for question_id, proposed_items in proposals.items():
                question = questions_by_id[question_id]
                state = graph_states[question_id]
                meta = proposal_meta[question_id]
                route_features = routes[question_id].features
                accepted, gate = _filter_graph_candidates(
                    question,
                    route_features,
                    fused_by_id[question_id],
                    proposed_items,
                    retrieval_cfg,
                    reranker_available=reranker is not None,
                )
                accepted_docs = list(
                    dict.fromkeys(item.chunk.doc_id for item in accepted)
                )
                rejected_docs = [
                    doc_id for doc_id in meta["new_docs"] if doc_id not in accepted_docs
                ]
                extras_by_id[question_id].extend(accepted)
                accepted_in_iteration += len(accepted)
                relations = []
                for doc_id in meta["new_docs"]:
                    edge = meta["edge_by_doc"][doc_id]
                    relations.append(
                        {
                            "source_doc_id": edge.source_doc_id,
                            "target_doc_id": edge.target_doc_id,
                            "type": edge.relation_type,
                            "direction": edge.direction,
                            "confidence": edge.confidence,
                            "accepted": doc_id in accepted_docs,
                        }
                    )
                log = {
                    "iteration": iteration,
                    "frontier": meta["frontier"],
                    "proposed_documents": meta["new_docs"],
                    "new_documents": accepted_docs,
                    "rejected_documents": rejected_docs,
                    "proposed_chunks": len(proposed_items),
                    "accepted_chunks": len(accepted),
                    **gate,
                    "relations": relations,
                }
                state["visited"].update(meta["new_docs"])
                if accepted_docs:
                    state["frontier"] = accepted_docs
                else:
                    state["active"] = False
                    log["stopped"] = "no_relevant_relation"
                iteration_logs[question_id].append(log)
            if accepted_in_iteration == 0:
                break

        if reranker is not None:
            unload_model(reranker, "reranker")

        final: dict[str, list[Evidence]] = {}
        for question in questions:
            question_id = question.question_id
            combined = fused_by_id[question_id] + extras_by_id[question_id]
            obsolete_docs: set[str] = set()
            if asks_for_current_law(question.question):
                obsolete_docs = store.obsolete_documents(
                    list(dict.fromkeys(item.chunk.doc_id for item in combined))
                )
                obsolete_penalty = float(
                    retrieval_cfg.get("obsolete_document_penalty", 0.36)
                )
                for item in combined:
                    if item.chunk.doc_id in obsolete_docs:
                        item.score -= obsolete_penalty
            combined.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
            route = routes[question_id]
            if route.route == "simple" and combined:
                main_doc = combined[0].chunk.doc_id
                pool = [item for item in combined if item.chunk.doc_id == main_doc]
                limit = int(retrieval_cfg.get("simple_evidence_top_k", 4))
                selected = deduplicate_and_merge_evidence(
                    pool, limit, prefer_document_diversity=False
                )
            else:
                limit = int(retrieval_cfg.get("complex_evidence_top_k", 8))
                selected = deduplicate_and_merge_evidence(
                    combined,
                    limit,
                    prefer_document_diversity=True,
                    max_documents=int(retrieval_cfg.get("complex_max_documents", 4)),
                    diversity_bonus=float(
                        retrieval_cfg.get("complex_diversity_bonus", 0.04)
                    ),
                    minimum_score_ratio=float(
                        retrieval_cfg.get("complex_min_score_ratio", 0.42)
                    ),
                )
            if not selected:
                selected = deduplicate_and_merge_evidence(
                    combined,
                    int(retrieval_cfg.get("evidence_top_k", 5)),
                    prefer_document_diversity=False,
                )
            final[question_id] = selected
            diagnostics[question_id].update(
                route=route.as_dict(),
                iterations=iteration_logs[question_id],
                temporal={
                    "current_law_requested": asks_for_current_law(question.question),
                    "obsolete_documents_penalized": sorted(obsolete_docs),
                },
                reranked_top=_trace_rank(combined, trace_rerank_k),
                selected_evidence=_trace_rank(selected, len(selected)),
            )

    batch = RetrievalBatch(
        final, diagnostics, dense_index_report, memory_report=memory_index_report
    )
    if cache:
        _save_cache(cache, questions, batch)
    return batch
