from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from legalqa.bm25 import BM25Index
from legalqa.answer_style import build_answer_style_profile, generation_plan
from legalqa.chunking import chunk_documents
from legalqa.data import Chunk, Document, QARecord, deterministic_split, write_json
from legalqa.dense_store import ensure_global_dense_index, search_global_dense_index
from legalqa.generation import Candidate, make_qwen_agentic_candidate, make_train_memory_candidate, parse_agentic_step
from legalqa.models import GenerationResult
from legalqa.fingerprint import adapter_fingerprint, checkpoint_fingerprint, corpus_content_fingerprint
from legalqa.index import build_index
from legalqa.metrics import meteor_exact, rouge_l_f1
from legalqa.legal_graph import temporal_relation_adjustment
from legalqa.pipeline import _append_candidate, _load_candidate_cache, _merge_tool_evidence, run_pipeline
from legalqa.prompts import qwen_user_prompt
from legalqa.retrieval import Evidence, deduplicate_and_merge_evidence
from legalqa.routing import decide_route
from legalqa.submission import create_submission
from legalqa.sqlite_retrieval import (
    _filter_graph_candidates,
    load_retrieval_cache,
    retrieve_sqlite_batch,
    sqlite_bm25_search_fn,
)
from legalqa.sqlite_store import SQLiteLegalStore, build_sqlite_store, prepare_sqlite_weak_pairs
from legalqa.verifier import score_candidates
from legalqa.text import document_numbers
from legalqa.train_memory import ensure_train_memory_index, search_train_memory


def minimal_config(tmp_path: Path) -> dict:
    return {
        "seed": 7,
        "paths": {
            "warmup": str(tmp_path / "warmup.json"),
            "corpus": str(tmp_path / "corpus.jsonl"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": {"generator": None, "embedding": None, "reranker": None, "bartpho": None, "verifier": None},
        },
        "runtime": {"offline": True, "local_files_only": True, "unload_after_stage": True, "batch_size": 2},
        "corpus": {
            "id_fields": ["document_id", "id"], "title_fields": ["title"], "text_fields": ["text"],
            "max_chunk_chars": 500, "overlap_chars": 50, "min_chunk_chars": 20,
        },
        "retrieval": {
            "bm25_top_k": 10, "dense_top_k": 0, "fused_top_k": 5, "rerank_top_k": 0,
            "evidence_top_k": 2, "rrf_k": 60, "bm25_weight": 1.0, "dense_weight": 0.0,
            "reranker_weight": 0.0, "citation_boost": 0.1,
        },
        "generation": {
            "qwen_enabled": False, "bartpho_enabled": False, "extractive_enabled": True,
            "max_context_chars": 2000, "max_new_tokens": 100, "min_answer_words": 5,
            "target_answer_words": 30, "max_answer_words": 100,
        },
        "selection": {
            "evidence_support": 0.4, "query_coverage": 0.2, "legal_citations": 0.1,
            "numeric_faithfulness": 0.1, "legal_semantic": 0.0, "length_prior": 0.1,
            "unsupported_penalty": 0.2, "duplicate_penalty": 0.1,
        },
        "split": {"dev_ratio": 0.2},
    }


def test_split_has_no_overlap():
    records = [QARecord(str(i), f"Câu hỏi điều kiện số {i}?", "Căn cứ Điều 1, đây là đáp án.") for i in range(30)]
    train, dev = deterministic_split(records, 0.2, 42)
    assert {item.question_id for item in train}.isdisjoint(item.question_id for item in dev)
    assert len(train) + len(dev) == len(records)


def test_legal_chunking_and_bm25(tmp_path):
    config = minimal_config(tmp_path)
    docs = [Document("vb1", "Điều 1. Phạm vi áp dụng\nNgười lao động có quyền nghỉ phép hằng năm.\n\nĐiều 2. Thời hạn\nThời hạn là 10 ngày.", "Luật thử nghiệm")]
    chunks = chunk_documents(docs, config)
    assert chunks and all(chunk.doc_id == "vb1" for chunk in chunks)
    index = BM25Index().fit(chunk.text for chunk in chunks)
    assert index.search("nghỉ phép người lao động", 1)


def test_official_corpus_directory_schema(tmp_path):
    from legalqa.data import load_corpus
    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    config["corpus"]["title_fields"].insert(0, "name")
    directory = tmp_path / "selected-contexts"
    directory.mkdir()
    (directory / "context_740.json").write_text(
        json.dumps({"id": 740, "name": "Quyet-dinh-740", "link": "https://example.invalid", "passage": "Điều 1. Nội dung văn bản."}, ensure_ascii=False),
        encoding="utf-8",
    )
    documents = load_corpus(directory, config)
    assert len(documents) == 1
    assert documents[0].doc_id == "740"
    assert documents[0].title == "Quyet-dinh-740"
    assert documents[0].metadata["source_file"] == "context_740.json"


def test_metrics_identity():
    text = "Theo Điều 1, người lao động có quyền nghỉ phép."
    assert rouge_l_f1(text, text) == 1.0
    assert meteor_exact(text, text) > 0.99


def test_sparse_pipeline_and_submission(tmp_path):
    config = minimal_config(tmp_path)
    corpus = Path(config["paths"]["corpus"])
    corpus.write_text(
        json.dumps({"document_id": "vb1", "title": "Luật A", "text": "Điều 5. Người lao động được nghỉ hằng năm theo quy định."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    write_json(questions, {"147194": {"question": "Người lao động được nghỉ như thế nào?"}})
    manifest = build_index(config, sparse_only=True)
    assert manifest["chunks"] >= 1
    predictions = tmp_path / "predictions.json"
    result = run_pipeline(config, questions, predictions)
    assert result["questions"] == 1
    archive_path = tmp_path / "submission.zip"
    validation = create_submission(questions, predictions, archive_path)
    assert validation["valid"]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["submission.json"]


def test_sqlite_store_and_weak_alignment(tmp_path):
    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    directory = tmp_path / "contexts"
    directory.mkdir()
    (directory / "context_1.json").write_text(
        json.dumps({"id": 1, "name": "Luật lao động", "passage": "Điều 10. Người lao động được nghỉ phép hằng năm."}, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "context_2.json").write_text(
        json.dumps({"id": 2, "name": "Luật giao thông", "passage": "Điều 20. Phương tiện phải bảo đảm an toàn giao thông."}, ensure_ascii=False),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    manifest = build_sqlite_store(config, directory, database)
    assert manifest["documents"] == 2
    with SQLiteLegalStore(database, readonly=True) as store:
        hits = store.search("quyền nghỉ phép người lao động", 2)
    assert hits and hits[0].chunk.doc_id == "1"
    train = tmp_path / "train.json"
    write_json(
        train,
        {"q1": {"question": "Người lao động được nghỉ thế nào?", "answer": "Căn cứ Điều 10, người lao động được nghỉ phép hằng năm."}},
    )
    pairs = tmp_path / "pairs.jsonl"
    report = prepare_sqlite_weak_pairs(train, database, pairs, top_k=2, negatives=1)
    assert report["pairs"] == 1
    assert '"positive_doc_id": "1"' in pairs.read_text(encoding="utf-8")


def test_precomputed_semantic_score_avoids_per_candidate_model_call(tmp_path):
    config = minimal_config(tmp_path)
    config["selection"]["legal_semantic"] = 0.12
    candidate = Candidate(
        "extractive",
        "Theo Điều 10, người lao động được nghỉ phép hằng năm.",
        details={"_precomputed_legal_semantic": 0.83},
    )
    evidence = [
        Evidence(
            Chunk("c1", "vb1", "Luật", "Điều 10", "Điều 10. Người lao động được nghỉ phép hằng năm.", {}),
            score=1.0,
        )
    ]

    class MustNotBeCalled:
        def similarity(self, answer, evidence_text):
            raise AssertionError("ViLegalBERT không được gọi từng candidate khi đã batch trước.")

    ranked = score_candidates("Người lao động được nghỉ thế nào?", evidence, [candidate], config, MustNotBeCalled())
    assert ranked[0].details["legal_semantic"] == 0.83


def test_sqlite_partial_build_is_replaced_automatically(tmp_path):
    import sqlite3

    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    corpus = tmp_path / "contexts"
    corpus.mkdir()
    (corpus / "context_1.json").write_text(
        json.dumps({"id": 1, "passage": "Điều 1. Nội dung hợp lệ."}, ensure_ascii=False),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("CREATE TABLE partial(value TEXT)")
    connection.close()
    report = build_sqlite_store(config, corpus, database)
    assert report["documents"] == 1
    with SQLiteLegalStore(database, readonly=True) as store:
        assert store.manifest()["status"] == "complete"


def test_partial_retrieval_cache_is_recomputed(tmp_path):
    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    corpus = tmp_path / "contexts"
    corpus.mkdir()
    (corpus / "context_1.json").write_text(
        json.dumps({"id": 1, "passage": "Điều 1. Người lao động được nghỉ phép."}, ensure_ascii=False),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    build_sqlite_store(config, corpus, database)
    cache = tmp_path / "retrieval.jsonl"
    cache.write_text('{"question_id":"q1","evidence":', encoding="utf-8")
    questions = [QARecord("q1", "Người lao động được nghỉ phép thế nào?")]
    result = retrieve_sqlite_batch(config, database, questions, cache, reuse_cache=True)
    assert set(result) == {"q1"}
    assert set(load_retrieval_cache(cache)) == {"q1"}


def test_candidate_cache_repairs_truncated_last_line(tmp_path):
    cache = tmp_path / "qwen.jsonl"
    _append_candidate(cache, "key-1", "q1", "Đáp án một")
    with cache.open("a", encoding="utf-8") as handle:
        handle.write('{"cache_key":"broken"')
    assert _load_candidate_cache(cache) == {"key-1": "Đáp án một"}
    _append_candidate(cache, "key-2", "q2", "Đáp án hai")
    assert _load_candidate_cache(cache) == {
        "key-1": "Đáp án một",
        "key-2": "Đáp án hai",
    }


def test_fingerprints_are_path_independent_and_content_sensitive(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"same weights")
    assert checkpoint_fingerprint(first) == checkpoint_fingerprint(second)
    (second / "config.json").write_text('{"model_type":"changed"}', encoding="utf-8")
    assert checkpoint_fingerprint(first) != checkpoint_fingerprint(second)

    corpus_a = tmp_path / "corpus-a"
    corpus_b = tmp_path / "corpus-b"
    corpus_a.mkdir()
    corpus_b.mkdir()
    (corpus_a / "context_1.json").write_text('{"id":1}', encoding="utf-8")
    (corpus_b / "context_1.json").write_text('{"id":1}', encoding="utf-8")
    assert corpus_content_fingerprint(corpus_a) == corpus_content_fingerprint(corpus_b)
    (corpus_b / "context_1.json").write_text('{"id":2}', encoding="utf-8")
    assert corpus_content_fingerprint(corpus_a) != corpus_content_fingerprint(corpus_b)


def test_document_number_parser_supports_official_slug_titles():
    title = "Thong-tu-12-2012-TT-BTP-bieu-mau-to-chuc-hoat-dong-trong-tai"
    assert document_numbers(title) == {"12/2012/TT-BTP"}
    assert document_numbers("Nghị định số 10 / 2020 / NĐ-CP") == {"10/2020/ND-CP"}


def test_sqlite_global_chunk_fts_and_replacement_graph(tmp_path):
    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    config["corpus"]["title_fields"].insert(0, "name")
    corpus = tmp_path / "contexts"
    corpus.mkdir()
    (corpus / "old.json").write_text(
        json.dumps(
            {
                "id": "old",
                "name": "Nghi-dinh-10-2020-ND-CP-quy-dinh-cu",
                "passage": "Điều 1. Người lao động được nghỉ năm ngày.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (corpus / "new.json").write_text(
        json.dumps(
            {
                "id": "new",
                "name": "Nghi-dinh-20-2024-ND-CP-quy-dinh-moi",
                "passage": (
                    "Điều 1. Người lao động được nghỉ mười ngày. "
                    "Nghị định này thay thế Nghị định số 10/2020/NĐ-CP."
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    report = build_sqlite_store(config, corpus, database)
    assert report["relations"] >= 1
    with SQLiteLegalStore(database, readonly=True) as store:
        hits = store.search_chunks("người lao động nghỉ mười ngày", 3)
        assert hits and hits[0].chunk.doc_id == "new"
        edges = store.relation_edges(["old"])
        obsolete = store.obsolete_documents(["old", "new"])
    replacement = next(edge for edge in edges if edge.relation_type == "REPLACES")
    assert replacement.neighbor_doc_id == "new"
    assert replacement.direction == "incoming"
    assert obsolete == {"old"}
    assert temporal_relation_adjustment(replacement, "Quy định hiện hành mới nhất?") > 0


def test_router_and_adjacent_chunk_merge(tmp_path):
    config = minimal_config(tmp_path)
    config["routing"] = {
        "simple_confidence_threshold": 0.6,
        "simple_min_reranker": 0.6,
        "complex_below_confidence": 0.4,
        "complex_max_margin": 0.08,
    }
    first = Evidence(
        Chunk(
            chunk_id="doc::s0001::w000", doc_id="doc", title="Luật", heading="Điều 1",
            text="Điều 1. Nội dung đầu và phần chung.", metadata={},
        ),
        score=0.9,
        bm25_score=2.0,
        dense_score=0.8,
        reranker_score=0.98,
    )
    second = Evidence(
        Chunk(
            chunk_id="doc::s0001::w001", doc_id="doc", title="Luật", heading="Điều 1",
            text="phần chung. Nội dung tiếp theo của điều.", metadata={},
        ),
        score=0.4,
        bm25_score=1.0,
        dense_score=0.5,
        reranker_score=0.2,
    )
    route = decide_route("Quyền này là gì?", [first, second], 0, config)
    assert route.route == "simple"
    complex_route = decide_route(
        "Văn bản nào thay thế và sửa đổi quy định này?", [first, second], 1, config
    )
    assert complex_route.route == "complex"
    merged = deduplicate_and_merge_evidence(
        [first, second], 2, prefer_document_diversity=False
    )
    assert len(merged) == 1
    assert "Nội dung tiếp theo" in merged[0].chunk.text


def test_router_does_not_use_common_relations_as_complex_signal(tmp_path):
    config = minimal_config(tmp_path)
    config["routing"] = {
        "simple_confidence_threshold": 0.68,
        "complex_below_confidence": 0.30,
        "complex_max_margin": 0.02,
        "complex_max_document_concentration": 0.40,
    }
    evidence = [
        Evidence(Chunk("a", "doc-a", "Luật A", "", "Quy định chính.", {}), 0.80, 2.0, 0.8, 0.95),
        Evidence(Chunk("b", "doc-b", "Luật B", "", "Quy định khác.", {}), 0.70, 1.5, 0.7, 0.90),
        Evidence(Chunk("c", "doc-c", "Luật C", "", "Quy định tham khảo.", {}), 0.60, 1.0, 0.6, 0.80),
    ]
    route = decide_route("Quyền này được quy định như thế nào?", evidence, 50, config)
    assert route.route == "simple"
    assert "no strong complex evidence" in route.reason or "high-confidence" in route.reason


def test_graph_gate_rejects_irrelevant_relation_and_stops_noise(tmp_path):
    config = minimal_config(tmp_path)["retrieval"]
    config.update(
        graph_min_reranker=0.12,
        graph_relative_reranker=0.30,
        graph_temporal_threshold_discount=0.60,
        graph_min_lexical_without_reranker=2.0,
    )
    initial = [
        Evidence(Chunk("i", "main", "Luật", "", "Nội dung chính.", {}), 0.9, reranker_score=0.8)
    ]
    relevant = Evidence(
        Chunk("g1", "new", "Văn bản sửa đổi", "", "Nội dung liên quan.", {}),
        0.3,
        bm25_score=1.0,
        reranker_score=0.72,
        stage="graph_1",
    )
    noise = Evidence(
        Chunk("g2", "noise", "Văn bản khác", "", "Nội dung không liên quan.", {}),
        0.3,
        bm25_score=1.0,
        reranker_score=0.05,
        stage="graph_1",
    )
    accepted, gate = _filter_graph_candidates(
        "Quy định này áp dụng thế nào?",
        {"complex_cue": 0.0},
        initial,
        [relevant, noise],
        config,
        reranker_available=True,
    )
    assert accepted == [relevant]
    assert gate["reranker_threshold"] == 0.24


def test_complex_evidence_diversity_never_forces_weak_documents():
    evidence = [
        Evidence(Chunk("a1", "a", "A", "", "Quy định A1.", {}), 1.0),
        Evidence(Chunk("a2", "a", "A", "", "Quy định A2.", {}), 0.9),
        Evidence(Chunk("b1", "b", "B", "", "Quy định B.", {}), 0.15),
        Evidence(Chunk("c1", "c", "C", "", "Quy định C.", {}), 0.10),
    ]
    selected = deduplicate_and_merge_evidence(
        evidence,
        4,
        prefer_document_diversity=True,
        max_documents=2,
        diversity_bonus=0.04,
        minimum_score_ratio=0.42,
    )
    assert [item.chunk.chunk_id for item in selected] == ["a1", "a2"]


def test_selector_hard_rejects_bartpho_and_prefers_fluent_qwen(tmp_path):
    config = minimal_config(tmp_path)
    config["selection"].update(
        fluency=0.22,
        fragment_penalty=0.16,
        truncation_penalty=0.22,
        retrieval_risk_penalty=0.10,
        qwen_preference_margin=0.06,
        source_priors={"qwen3.5-2b": 0.08, "extractive": -0.03},
    )
    evidence = [
        Evidence(
            Chunk("c1", "d1", "Luật", "Điều 1", "Điều 1. Người lao động có quyền nghỉ phép mười ngày.", {}),
            score=0.9,
            bm25_score=1.0,
            dense_score=0.8,
            reranker_score=0.95,
        )
    ]
    candidates = [
        Candidate("bartpho-syllable", "trả lại câu hỏi trả tiền trả lương"),
        Candidate("extractive", "ĐIỀU 1. QUYỀN. NGHỈ PHÉP. MƯỜI NGÀY."),
        Candidate("qwen3.5-2b", "Theo Điều 1, người lao động có quyền nghỉ phép mười ngày."),
    ]
    ranked = score_candidates("Người lao động được nghỉ thế nào?", evidence, candidates, config)
    assert ranked[0].source == "qwen3.5-2b"
    assert next(item for item in ranked if item.source == "bartpho-syllable").score == -1.0


def test_restartable_global_dense_index_and_exact_search(tmp_path):
    import numpy as np

    config = minimal_config(tmp_path)
    config["corpus"]["text_fields"].insert(0, "passage")
    config["retrieval"].update(dense_build_block=1, dense_checkpoint_every=1)
    config["runtime"]["embedding_identity"] = "fake-embedding-v1"
    corpus = tmp_path / "contexts"
    corpus.mkdir()
    (corpus / "a.json").write_text(
        json.dumps({"id": "a", "passage": "Điều 1. Người lao động được nghỉ phép."}, ensure_ascii=False),
        encoding="utf-8",
    )
    (corpus / "b.json").write_text(
        json.dumps({"id": "b", "passage": "Điều 2. Phương tiện phải bảo đảm an toàn."}, ensure_ascii=False),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    build_sqlite_store(config, corpus, database)

    class FakeEncoder:
        def encode(self, texts):
            vectors = []
            for text in texts:
                vector = np.array([1.0, 0.0] if "lao động" in text else [0.0, 1.0], dtype=np.float32)
                vectors.append(vector)
            return np.stack(vectors)

    dense_dir = tmp_path / "dense"
    with SQLiteLegalStore(database, readonly=True) as store:
        report = ensure_global_dense_index(config, store, FakeEncoder(), dense_dir)
        reused = ensure_global_dense_index(config, store, FakeEncoder(), dense_dir)
        expected_rowid = store.search_chunks("người lao động nghỉ phép", 1)[0].rowid
    assert report["chunks"] == 2
    assert reused["reused"] is True
    results = search_global_dense_index(
        np.array([[1.0, 0.0]], dtype=np.float32), dense_dir, 1, device="cpu", block_size=1
    )
    assert results[0][0][0] == expected_rowid


def test_answer_style_profile_drives_dynamic_generation_plan(tmp_path):
    records = [
        QARecord(str(index), "Người lao động có quyền gì?", " ".join(["quyền"] * count))
        for index, count in enumerate((40, 60, 80, 100, 120), 1)
    ]
    profile = build_answer_style_profile(records)
    config = minimal_config(tmp_path)
    config["generation"].update(
        answer_style_profile=profile,
        style_min_group_size=2,
        style_floor_words=20,
        style_ceiling_words=200,
        tokens_per_vietnamese_word=1.5,
        simple_max_new_tokens=300,
        complex_max_new_tokens=400,
    )
    simple = generation_plan("Người lao động có quyền gì?", "simple", config)
    complex_plan = generation_plan("Người lao động có quyền gì?", "complex", config)
    assert simple["learned_from_train"] is True
    assert 60 <= simple["target_words"] <= 100
    assert complex_plan["target_words"] >= simple["target_words"]
    assert complex_plan["max_new_tokens"] >= simple["max_new_tokens"]


def test_restartable_train_memory_and_adapter_fingerprint(tmp_path):
    import numpy as np

    train = tmp_path / "train.json"
    write_json(
        train,
        {
            "a": {"question": "Người lao động được nghỉ phép thế nào?", "answer": "Được nghỉ phép."},
            "b": {"question": "Phương tiện phải bảo đảm gì?", "answer": "Phải bảo đảm an toàn."},
        },
    )
    config = minimal_config(tmp_path)
    config["retrieval"].update(train_memory_build_block=1, train_memory_checkpoint_every=1)
    config["runtime"]["embedding_identity"] = "fake-embedding"

    class FakeEncoder:
        def encode(self, texts):
            return np.stack(
                [
                    np.array([1.0, 0.0] if "lao động" in text else [0.0, 1.0], dtype=np.float32)
                    for text in texts
                ]
            )

    memory_dir = tmp_path / "memory"
    first = ensure_train_memory_index(config, FakeEncoder(), train, memory_dir)
    second = ensure_train_memory_index(config, FakeEncoder(), train, memory_dir)
    neighbors = search_train_memory(
        np.array([[1.0, 0.0]], dtype=np.float32),
        memory_dir,
        1,
        device="cpu",
    )
    assert first["count"] == 2 and second["reused"] is True
    assert neighbors[0][0]["question_id"] == "a"

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    assert adapter_fingerprint(adapter) == adapter_fingerprint(adapter)


def test_train_memory_candidate_and_fewshot_prompt_are_gated(tmp_path):
    config = minimal_config(tmp_path)
    config["retrieval"].update(
        train_memory_candidate_threshold=0.86,
        train_memory_fewshot_threshold=0.70,
        train_memory_fewshot_examples=2,
        train_memory_fewshot_chars=2000,
    )
    config["generation"].update(
        simple_max_new_tokens=200,
        complex_max_new_tokens=300,
        style_floor_words=20,
        style_ceiling_words=200,
        tokens_per_vietnamese_word=1.5,
    )
    evidence = [
        Evidence(Chunk("c", "d", "Điều 1. Người lao động được nghỉ phép.", "Luật", "Điều 1", {}), 0.9)
    ]
    diagnostics = {
        "route": {"route": "simple"},
        "train_memory": [
            {
                "question_id": "t1",
                "question": "Người lao động nghỉ phép thế nào?",
                "answer": "Theo Điều 1, người lao động được nghỉ phép.",
                "similarity": 0.91,
            }
        ],
    }
    candidate = make_train_memory_candidate(
        "Người lao động được nghỉ thế nào?", evidence, config, diagnostics
    )
    assert candidate and candidate.source == "train-memory"
    prompt = qwen_user_prompt(
        "Người lao động được nghỉ thế nào?", evidence, config, diagnostics
    )
    assert "VÍ DỤ VĂN PHONG" in prompt
    diagnostics["train_memory"][0]["similarity"] = 0.60
    assert make_train_memory_candidate("Câu hỏi", evidence, config, diagnostics) is None


def test_generated_answer_compaction_removes_loops_and_stops_on_sentence_boundary():
    from legalqa.generation import compact_generated_answer

    repeated = (
        "Theo Điều 1, người lao động được nghỉ phép mười ngày. "
        "Theo Điều 1, người lao động được nghỉ phép mười ngày. "
        "Ngoài ra, người sử dụng lao động phải thông báo lịch nghỉ cho người lao động. "
        "Quy định bổ sung này có rất nhiều từ và không nên được thêm khi đã vượt giới hạn."
    )
    answer, report = compact_generated_answer(repeated, maximum_words=25, hard_trim=True)
    assert answer.count("Theo Điều 1") == 1
    assert report["duplicate_sentences_removed"] == 1
    assert report["compaction_applied"] is True
    assert answer.endswith(".")

    untrimmed, untrimmed_report = compact_generated_answer(
        repeated, maximum_words=25, hard_trim=False
    )
    assert "Quy định bổ sung" in untrimmed
    assert untrimmed_report["duplicate_sentences_removed"] == 1


def test_parse_agentic_step_detects_tool_call_and_final_answer():
    query, answer = parse_agentic_step("TRA_CUU: mức phạt chậm đóng bảo hiểm xã hội")
    assert answer is None
    assert query == "mức phạt chậm đóng bảo hiểm xã hội"

    # Marker phải được nhận dù model bỏ dấu tiếng Việt ở nhãn lệnh.
    query_no_diacritics, _ = parse_agentic_step("tra cuu: thời hạn khiếu nại")
    assert query_no_diacritics == "thời hạn khiếu nại"

    # Model hay bọc nhãn lệnh trong markdown (**TRA_CUU:**) — vẫn phải nhận.
    query_markdown, _ = parse_agentic_step("**TRA_CUU:** mức phạt chậm đóng")
    assert query_markdown == "mức phạt chậm đóng"

    query_none, answer_only = parse_agentic_step("Theo Điều 1, người lao động được nghỉ phép.")
    assert query_none is None
    assert answer_only == "Theo Điều 1, người lao động được nghỉ phép."

    # Marker phát ra nhưng không kèm truy vấn: vẫn phải nhận diện là "có ý định
    # tool-call" (query rỗng) chứ không được rơi xuống nhánh coi nguyên văn
    # "TRA_CUU:" là câu trả lời cuối cùng.
    empty_query, empty_answer = parse_agentic_step("TRA_CUU:")
    assert empty_query == ""
    assert empty_answer is None


def test_agentic_candidate_issues_tool_call_then_answers(tmp_path):
    config = minimal_config(tmp_path)
    config["generation"].update({"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 3})
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {}), 0.9)]
    extra_chunk = Chunk("c2", "d1", "Điều 2. Mức phạt chậm đóng là 0,03%/ngày.", "Luật", "Điều 2", {})

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls == 1:
                assert "CÔNG CỤ TRA CỨU" in user
                return GenerationResult("TRA_CUU: mức phạt chậm đóng", 12, True, False)
            assert "LỊCH SỬ TRA CỨU" in user
            return GenerationResult("Theo Điều 2, mức phạt chậm đóng là 0,03%/ngày.", 20, True, False)

    search_calls: list[str] = []

    def fake_search(query: str) -> list[Evidence]:
        search_calls.append(query)
        return [Evidence(extra_chunk, 0.8)]

    generator = FakeGenerator()
    candidate = make_qwen_agentic_candidate(generator, "Mức phạt chậm đóng bảo hiểm là bao nhiêu?", seed_evidence, config, fake_search)

    assert generator.calls == 2
    assert search_calls == ["mức phạt chậm đóng"]
    assert candidate.source == "qwen3.5-2b-agentic"
    assert "0,03%" in candidate.answer
    assert candidate.details["tool_calls_used"] == 1
    assert candidate.details["evidence_added_total"] == 1


def test_agentic_candidate_skips_search_fn_on_repeated_query(tmp_path):
    # Model phát lại đúng truy vấn cũ (bí chiến lược sau khi lượt đầu không
    # tìm thấy gì mới): không được gọi search_fn lần hai cho cùng truy vấn,
    # tốn thêm một lượt BM25 vô ích trong khi kết quả chắc chắn giống hệt.
    config = minimal_config(tmp_path)
    config["generation"].update({"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 3})
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls < 3:
                # Không phân biệt hoa/thường và dấu — lặp cùng truy vấn.
                return GenerationResult("TRA_CUU: Mức Phạt Chậm Đóng", 12, True, False)
            return GenerationResult("Không tìm thấy căn cứ phù hợp hơn.", 8, True, False)

    search_calls: list[str] = []

    def fake_search(query: str) -> list[Evidence]:
        search_calls.append(query)
        return []

    generator = FakeGenerator()
    candidate = make_qwen_agentic_candidate(
        generator, "Mức phạt chậm đóng bảo hiểm là bao nhiêu?", seed_evidence, config, fake_search,
    )

    assert generator.calls == 3
    assert len(search_calls) == 1
    assert candidate.details["tool_calls_used"] == 2


def test_agentic_candidate_skips_tool_section_when_search_fn_missing(tmp_path):
    config = minimal_config(tmp_path)
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            assert "CÔNG CỤ TRA CỨU" not in user
            return GenerationResult("Theo Điều 1, người lao động được nghỉ phép.", 10, True, False)

    candidate = make_qwen_agentic_candidate(FakeGenerator(), "Người lao động có được nghỉ phép không?", seed_evidence, config, None)
    assert candidate.details["tool_calls_used"] == 0
    assert candidate.details["evidence_added_total"] == 0


def test_agentic_candidate_raises_when_model_ignores_tool_call_limit(tmp_path):
    # Model vẫn phát "TRA_CUU:" ở lượt cuối cùng (đã hết hạn mức) -> không được
    # nộp nguyên văn marker làm câu trả lời, phải báo lỗi để pipeline loại bỏ.
    config = minimal_config(tmp_path)
    config["generation"]["agentic_max_tool_calls"] = 1
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            return GenerationResult("TRA_CUU: gì đó", 5, True, False)

    with pytest.raises(RuntimeError):
        make_qwen_agentic_candidate(FakeGenerator(), "Câu hỏi?", seed_evidence, config, lambda q: [])


def test_agentic_candidate_raises_on_empty_tool_query(tmp_path):
    config = minimal_config(tmp_path)
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            return GenerationResult("TRA_CUU:", 3, True, False)

    with pytest.raises(RuntimeError):
        make_qwen_agentic_candidate(FakeGenerator(), "Câu hỏi?", seed_evidence, config, lambda q: [])


def test_agentic_candidate_prioritizes_tool_evidence_within_context_budget(tmp_path):
    # render_evidence (prompts.py) cắt bỏ hoàn toàn phần vượt max_context_chars.
    # Nếu căn cứ tra cứu thêm bị nối vào CUỐI danh sách, ngân sách ký tự đã bị
    # căn cứ ban đầu dùng hết và nó sẽ không bao giờ lọt vào prompt của lượt kế
    # tiếp -> tool-call trở thành vô ích một cách âm thầm. Test này khoá lại
    # hành vi: căn cứ mới phải được ưu tiên trước.
    config = minimal_config(tmp_path)
    config["generation"]["max_context_chars"] = 260
    long_seed = Chunk("s1", "d1", "X" * 200, "Luật", "Điều 1", {})
    extra_chunk = Chunk("s2", "d1", "Mức phạt là 0,03% mỗi ngày theo Điều 2.", "Luật", "Điều 2", {})
    seed_evidence = [Evidence(long_seed, 0.9)]

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            self.prompts.append(user)
            if self.calls == 1:
                return GenerationResult("TRA_CUU: mức phạt", 5, True, False)
            return GenerationResult("Theo Điều 2, mức phạt là 0,03% mỗi ngày.", 10, True, False)

    def fake_search(query: str) -> list[Evidence]:
        return [Evidence(extra_chunk, 0.8)]

    generator = FakeGenerator()
    make_qwen_agentic_candidate(generator, "Mức phạt là bao nhiêu?", seed_evidence, config, fake_search)

    second_prompt = generator.prompts[1]
    assert "Mức phạt là 0,03%" in second_prompt


def test_merge_tool_evidence_lets_verifier_see_tool_retrieved_citations(tmp_path):
    # Không hợp nhất: verifier chấm answer agentic dựa trên evidence_by_id
    # seed-only, nên "Điều 2"/"0,03%" (chỉ có trong đoạn tra cứu thêm) bị coi
    # là trích dẫn không có căn cứ -> phạt oan chính vì tool-call đã hoạt động
    # đúng. Sau khi hợp nhất, verifier phải thấy đủ căn cứ.
    config = minimal_config(tmp_path)
    seed_chunk = Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {})
    tool_chunk_text = "Điều 2. Mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    evidence_by_id = {"q1": [Evidence(seed_chunk, 0.9)]}
    question = "Mức phạt chậm đóng bảo hiểm xã hội là bao nhiêu?"
    answer_text = "Theo Điều 2, mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    tool_evidence = [{
        "chunk_id": "c2", "doc_id": "d1", "title": "Luật", "heading": "Điều 2",
        "text": tool_chunk_text, "score": 0.8, "bm25_score": 0.8,
    }]

    before = score_candidates(question, evidence_by_id["q1"], [Candidate("qwen3.5-2b-agentic", answer_text)], config)[0]
    assert before.details["citation_precision"] == 0.0

    _merge_tool_evidence(evidence_by_id, "q1", tool_evidence)
    after = score_candidates(question, evidence_by_id["q1"], [Candidate("qwen3.5-2b-agentic", answer_text)], config)[0]
    assert after.details["citation_precision"] == 1.0

    # Gọi lại lần 2 (mô phỏng cache-hit ở lượt chạy sau) không được thêm trùng.
    _merge_tool_evidence(evidence_by_id, "q1", tool_evidence)
    assert len(evidence_by_id["q1"]) == 2


def test_tool_evidence_only_boosts_agentic_candidate_not_other_sources(tmp_path):
    # tool_evidence_by_id là dict riêng (không phải evidence_by_id dùng chung
    # cho mọi candidate). Nếu extra_evidence_by_source rò rỉ sang candidate
    # khác, qwen thường sẽ ăn ké citation_precision của căn cứ nó chưa từng
    # tra cứu -> chọn nhầm answer chỉ vì đứng cạnh agentic trong cùng câu hỏi.
    config = minimal_config(tmp_path)
    seed_chunk = Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {})
    tool_chunk_text = "Điều 2. Mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    evidence_by_id = {"q1": [Evidence(seed_chunk, 0.9)]}
    tool_evidence_by_id: dict[str, list[Evidence]] = {"q1": []}
    question = "Mức phạt chậm đóng bảo hiểm xã hội là bao nhiêu?"
    answer_text = "Theo Điều 2, mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    tool_evidence = [{
        "chunk_id": "c2", "doc_id": "d1", "title": "Luật", "heading": "Điều 2",
        "text": tool_chunk_text, "score": 0.8, "bm25_score": 0.8,
    }]
    _merge_tool_evidence(tool_evidence_by_id, "q1", tool_evidence)

    candidates = [
        Candidate("qwen3.5-2b", answer_text),
        Candidate("qwen3.5-2b-agentic", answer_text),
    ]
    ranked = score_candidates(
        question,
        evidence_by_id["q1"],
        candidates,
        config,
        extra_evidence_by_source={"qwen3.5-2b-agentic": tool_evidence_by_id["q1"]},
    )
    by_source = {candidate.source: candidate for candidate in ranked}
    assert by_source["qwen3.5-2b-agentic"].details["citation_precision"] == 1.0
    assert by_source["qwen3.5-2b"].details["citation_precision"] == 0.0
    # evidence_by_id (dùng cho semantic/trace) không được đổi bởi merge.
    assert len(evidence_by_id["q1"]) == 1


def test_sqlite_bm25_search_fn_returns_evidence_from_tool_call(tmp_path):
    # Trên Kaggle, retrieval ban đầu dùng SQLite (FTS5) chứ không phải
    # HybridRetriever; tool-call agentic phải tự mở connection readonly riêng
    # vì retrieve_sqlite_batch đã đóng connection của nó ngay sau khi trả kết
    # quả (xem docstring sqlite_bm25_search_fn).
    config = minimal_config(tmp_path)
    corpus = tmp_path / "contexts"
    corpus.mkdir()
    (corpus / "doc.json").write_text(
        json.dumps(
            {
                "id": "d1",
                "title": "Luật",
                "text": "Điều 2. Mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "legal.sqlite"
    build_sqlite_store(config, corpus, database)

    with sqlite_bm25_search_fn(config, database) as search_fn:
        results = search_fn("mức phạt chậm đóng bảo hiểm")
        assert results
        assert results[0].chunk.doc_id == "d1"
        assert results[0].stage == "tool_call"
        assert results[0].bm25_score == results[0].score
