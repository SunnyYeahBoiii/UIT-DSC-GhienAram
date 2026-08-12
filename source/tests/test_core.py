from __future__ import annotations

import json
import zipfile
from pathlib import Path

from legalqa.bm25 import BM25Index
from legalqa.chunking import chunk_documents
from legalqa.data import Chunk, Document, QARecord, deterministic_split, write_json
from legalqa.dense_store import ensure_global_dense_index, search_global_dense_index
from legalqa.generation import Candidate
from legalqa.fingerprint import checkpoint_fingerprint, corpus_content_fingerprint
from legalqa.index import build_index
from legalqa.metrics import meteor_exact, rouge_l_f1
from legalqa.legal_graph import temporal_relation_adjustment
from legalqa.pipeline import _append_candidate, _load_candidate_cache, run_pipeline
from legalqa.retrieval import Evidence, deduplicate_and_merge_evidence
from legalqa.routing import decide_route
from legalqa.submission import create_submission
from legalqa.sqlite_retrieval import load_retrieval_cache, retrieve_sqlite_batch
from legalqa.sqlite_store import SQLiteLegalStore, build_sqlite_store, prepare_sqlite_weak_pairs
from legalqa.verifier import score_candidates
from legalqa.text import document_numbers


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
