from __future__ import annotations

import json
import sqlite3
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
from legalqa.pipeline import (
    InferencePaused,
    _append_candidate,
    _candidate_evidence,
    _load_candidate_cache,
    _should_run_agentic,
    generate_from_evidence,
    run_pipeline,
)
from legalqa.recovery import RecoveryError, restore_recovery_bundle
from legalqa.prompts import qwen_user_prompt
from legalqa.retrieval import Evidence, deduplicate_and_merge_evidence, rank_agentic_evidence
from legalqa.routing import decide_route
from legalqa.selector_schema import SELECTION_WEIGHTS_SCHEMA
from legalqa.submission import create_submission
from legalqa.sqlite_retrieval import (
    _filter_graph_candidates,
    RetrievalBatch,
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


def test_selector_schema_is_shared_by_calibration_and_inference():
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    for name, relative in (
        ("selector_eval_contract", "scripts/evaluate_selector.py"),
        ("selector_inference_contract", "kaggle/run_inference_only.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, project / relative)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert module.SELECTION_WEIGHTS_SCHEMA == SELECTION_WEIGHTS_SCHEMA


def test_source_contract_matches_runtime_and_notebooks():
    project = Path(__file__).resolve().parents[1]
    contract = json.loads((project / "SOURCE_CONTRACT.json").read_text(encoding="utf-8"))
    assert contract == {
        "schema": "legalqa-source-contract-v1",
        "release": "0.9.2",
        "agentic_namespace": "agentic-v3",
        "cache_schema": "inference-v7-agentic-error-isolation",
        "selector_schema": SELECTION_WEIGHTS_SCHEMA,
        "adapter_alias_policy": "sha256-dedupe-v1",
        "source_alias_policy": "sha256-dedupe-v1",
        "qlora_recovery_schema": "legalqa-qlora-recovery-v1",
        "inference_pause_schema": "inference-runtime-pause-v1",
        "offline_wheel_profile": "hf-stack-2026-08-v1",
        "offline_wheel_count": 12,
    }
    assert contract["agentic_namespace"] in (
        project / "src" / "legalqa" / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert contract["cache_schema"] in (
        project / "kaggle" / "run_inference_only.py"
    ).read_text(encoding="utf-8")
    wheels = list((project / "wheelhouse").glob("*.whl"))
    assert len(wheels) == contract["offline_wheel_count"]
    assert all(wheel.stat().st_size >= 10_000 for wheel in wheels)
    for notebook in (
        "legalqa_dsc2026_v9_selector_calibration_t4.ipynb",
        "legalqa_dsc2026_v9_qlora_t4.ipynb",
        "legalqa_dsc2026_inference_only_t4.ipynb",
    ):
        source = (project / "kaggle" / notebook).read_text(encoding="utf-8")
        assert "SOURCE_CONTRACT.json" in source
        assert contract["selector_schema"] in source
        assert contract["source_alias_policy"] in source
        assert contract["offline_wheel_profile"] in source
        assert "def source_identity(path)" in source
        assert "transformers-5.14.1-" in source
        assert "require_kaggle_cuda" in source
        assert "Session Kaggle chưa được gắn GPU" in source
        notebook_payload = json.loads(source)
        for cell in notebook_payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), notebook, "exec")
    inference_notebook = (
        project / "kaggle" / "legalqa_dsc2026_inference_only_t4.ipynb"
    ).read_text(encoding="utf-8")
    assert "--require-generator-adapter" in inference_notebook
    assert "--require-selection-weights" in inference_notebook
    assert "--require-prebuilt-indexes" in inference_notebook
    assert "MAX_RUNTIME_HOURS = 11.5" in inference_notebook
    assert "TRAIN_MEMORY = False" in inference_notebook
    assert "input_root.rglob('recovery_state.json')" in inference_notebook
    assert "if p.is_dir()" in inference_notebook
    assert "--auto-resume" in inference_notebook
    assert "status == 'paused'" in inference_notebook


def test_inference_notebook_ignores_selector_recovery_directories(tmp_path):
    import hashlib
    import zipfile

    project = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (project / "kaggle" / "legalqa_dsc2026_inference_only_t4.ipynb").read_text(
            encoding="utf-8"
        )
    )
    discovery_cell = "".join(notebook["cells"][3]["source"])
    discovery_prefix = discovery_cell.split("print('Global dense index:'", 1)[0]

    public_file = tmp_path / "public-official.json"
    public_file.write_text('{"1":{"question":"q"}}', encoding="utf-8")
    public_sha = hashlib.sha256(public_file.read_bytes()).hexdigest()

    def write_recovery(root: Path, fingerprint: str, questions_sha: str) -> None:
        root.mkdir(parents=True)
        (root / "recovery_state.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "run_fingerprint": fingerprint,
                    "stage": "paused",
                    "completed": 1,
                    "total": 2,
                }
            ),
            encoding="utf-8",
        )
        (root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "mode": "full_sequential_inference_only",
                    "inputs": {"questions_sha256": questions_sha},
                }
            ),
            encoding="utf-8",
        )

    public_recovery = tmp_path / "legalqa-recovery-latest"
    write_recovery(public_recovery, "public-fingerprint", public_sha)
    write_recovery(tmp_path / "selector" / "inference", "selector-fingerprint", "dev-sha")
    write_recovery(
        tmp_path / "selector" / "selector_recovery_latest",
        "selector-fingerprint",
        "dev-sha",
    )

    def sha256_path(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    namespace = {
        "AUTO_RESUME": True,
        "hashlib": hashlib,
        "input_root": tmp_path,
        "json": json,
        "public_files": [public_file],
        "sha256_path": sha256_path,
        "zipfile": zipfile,
    }
    exec(discovery_prefix, namespace)
    assert namespace["recovery_inputs"] == [public_recovery]
    assert namespace["resume_bundle"] == public_recovery


def test_qlora_recovery_bundle_keeps_latest_full_trainer_state(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "train_qwen_lora.py"
    spec = importlib.util.spec_from_file_location("qlora_training_recovery", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output = tmp_path / "adapter"
    output.mkdir()
    (output / "resume_signature.json").write_text('{"epochs": 1}', encoding="utf-8")
    (output / "qlora_scope.json").write_text('{"train_scope": "split"}', encoding="utf-8")
    (output / "training_paused.json").write_text('{"status": "paused"}', encoding="utf-8")
    for step in (20, 40):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": 207}), encoding="utf-8"
        )
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(f"{name}-{step}".encode())
    bundle = tmp_path / "qlora_recovery_latest.zip"
    report = module.create_qlora_recovery_bundle(output, bundle)
    assert report["global_step"] == 40
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("recovery_manifest.json"))
    assert manifest["schema"] == "legalqa-qlora-recovery-v1"
    assert "training_paused.json" in names
    assert "checkpoint-40/optimizer.pt" in names
    assert "checkpoint-40/scheduler.pt" in names
    assert "checkpoint-40/rng_state.pth" in names
    assert not any(name.startswith("checkpoint-20/") for name in names)


def test_qlora_recovery_zip_extracts_and_resumes_without_path_traversal(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    train_spec = importlib.util.spec_from_file_location(
        "qlora_training_recovery_roundtrip", project / "scripts" / "train_qwen_lora.py"
    )
    train_module = importlib.util.module_from_spec(train_spec)
    assert train_spec.loader is not None
    train_spec.loader.exec_module(train_module)
    runner_spec = importlib.util.spec_from_file_location(
        "qlora_runner_recovery_roundtrip", project / "kaggle" / "run_qwen_qlora.py"
    )
    runner_module = importlib.util.module_from_spec(runner_spec)
    assert runner_spec.loader is not None
    runner_spec.loader.exec_module(runner_module)

    source = tmp_path / "source"
    checkpoint = source / "checkpoint-60"
    checkpoint.mkdir(parents=True)
    (source / "resume_signature.json").write_text('{"epochs": 1}', encoding="utf-8")
    (source / "qlora_scope.json").write_text('{"train_scope": "split"}', encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 60}', encoding="utf-8")
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    bundle = tmp_path / "qlora_recovery_latest.zip"
    train_module.create_qlora_recovery_bundle(source, bundle)

    extracted = runner_module._extract_recovery_bundle(bundle, tmp_path / "working")
    assert runner_module._latest_checkpoint_step(extracted) == 60
    target = tmp_path / "adapter"
    runner_module._copy_resume(extracted, target)
    assert (target / "checkpoint-60" / "optimizer.pt").read_bytes() == b"optimizer.pt"

    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("recovery_manifest.json", '{"schema":"legalqa-qlora-recovery-v1"}')
        archive.writestr("qlora_scope.json", '{"train_scope":"split"}')
        archive.writestr("../escaped.txt", "bad")
    with pytest.raises(RuntimeError, match="path không an toàn"):
        runner_module._extract_recovery_bundle(malicious, tmp_path / "malicious-working")
    assert not (tmp_path / "escaped.txt").exists()
    malicious_backslash = tmp_path / "malicious-backslash.zip"
    with zipfile.ZipFile(malicious_backslash, "w") as archive:
        archive.writestr("recovery_manifest.json", '{"schema":"legalqa-qlora-recovery-v1"}')
        archive.writestr("qlora_scope.json", '{"train_scope":"split"}')
        archive.writestr("..\\escaped-backslash.txt", "bad")
    with pytest.raises(RuntimeError, match="path không an toàn"):
        runner_module._extract_recovery_bundle(
            malicious_backslash, tmp_path / "malicious-backslash-working"
        )
    assert not (tmp_path / "escaped-backslash.txt").exists()


def test_selector_calibration_requires_completed_split_adapter(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_selector_calibration.py"
    spec = importlib.util.spec_from_file_location("selector_adapter_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    adapter = tmp_path / "qwen-official-lora-split"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "training_complete.json").write_text(
        json.dumps({"global_step": 207}), encoding="utf-8"
    )
    (adapter / "qlora_scope.json").write_text(
        json.dumps({"train_scope": "split"}), encoding="utf-8"
    )
    assert module._find_split_adapter(tmp_path) == adapter
    alias = tmp_path / "qwen_qlora_adapter"
    alias.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_complete.json",
        "qlora_scope.json",
    ):
        (alias / name).write_bytes((adapter / name).read_bytes())
    # Two exported folders with identical PEFT config/weights are one adapter.
    assert module._find_split_adapter(tmp_path) == adapter
    (alias / "adapter_model.safetensors").write_bytes(b"different-weights")
    with pytest.raises(RuntimeError, match="KHÁC nhau"):
        module._find_split_adapter(tmp_path)
    (alias / "adapter_model.safetensors").write_bytes(
        (adapter / "adapter_model.safetensors").read_bytes()
    )
    (adapter / "qlora_scope.json").write_text(
        json.dumps({"train_scope": "full"}), encoding="utf-8"
    )
    (alias / "qlora_scope.json").write_text(
        json.dumps({"train_scope": "full"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        module._find_split_adapter(tmp_path)


def test_public_inference_collapses_identical_completed_adapter_aliases(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_inference_only.py"
    spec = importlib.util.spec_from_file_location("inference_adapter_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    canonical = tmp_path / "qwen-official-lora-full"
    alias = tmp_path / "qwen_qlora_adapter"
    for directory in (canonical, alias):
        directory.mkdir()
        (directory / "adapter_config.json").write_text(
            json.dumps({"r": 8, "base_model_name_or_path": "Qwen3.5-2B"}),
            encoding="utf-8",
        )
        (directory / "adapter_model.safetensors").write_bytes(b"same-weights")
        (directory / "training_complete.json").write_text(
            json.dumps({"global_step": 219}), encoding="utf-8"
        )
        (directory / "qlora_scope.json").write_text(
            json.dumps({"train_scope": "full"}), encoding="utf-8"
        )
    assert module.find_generator_adapter(tmp_path) == canonical
    (alias / "adapter_model.safetensors").write_bytes(b"different-weights")
    with pytest.raises(RuntimeError, match="KHÁC nhau"):
        module.find_generator_adapter(tmp_path)


def test_public_inference_ignores_unrelated_training_complete_marker(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle" / "run_inference_only.py"
    spec = importlib.util.spec_from_file_location("inference_unrelated_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    unrelated = tmp_path / "other-training-output"
    unrelated.mkdir()
    (unrelated / "training_complete.json").write_text(
        json.dumps({"global_step": 12}), encoding="utf-8"
    )
    assert module.find_generator_adapter(tmp_path) is None


def test_public_inference_rejects_split_adapter_and_calibration_accepts_it(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_inference_only.py"
    spec = importlib.util.spec_from_file_location("inference_adapter_scope", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    adapter = tmp_path / "qwen-official-lora-split"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "training_complete.json").write_text(
        json.dumps({"global_step": 207}), encoding="utf-8"
    )
    (adapter / "qlora_scope.json").write_text(
        json.dumps({"train_scope": "split"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="scope='full'"):
        module.find_generator_adapter(tmp_path)
    assert module.find_generator_adapter(tmp_path, expected_scope="split") == adapter


def test_selector_split_integrity_is_checked_before_inference(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_selector_calibration.py"
    spec = importlib.util.spec_from_file_location("selector_split_integrity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    dev = {str(i): {"question": f"Câu hỏi pháp luật {i}?"} for i in range(300)}
    refs = {str(i): {"answer": f"Câu trả lời {i}."} for i in range(300)}
    train = {"train-1": {"question": "Câu train?", "answer": "Đáp án train."}}
    write_json(tmp_path / "selector_dev_questions.json", dev)
    write_json(tmp_path / "selector_dev_references.json", refs)
    write_json(tmp_path / "selector_train.json", train)
    manifest = {"dev": 300, "train": 1, "total": 301, "overlap_ids": 0}
    assert module._validate_selector_split(tmp_path, manifest) == {
        "dev": 300,
        "train": 1,
        "total": 301,
    }
    alias = tmp_path / "selector_split_alias"
    alias.mkdir()
    write_json(alias / "selector_dev_questions.json", dev)
    write_json(alias / "selector_dev_references.json", refs)
    write_json(alias / "selector_train.json", train)
    assert module._find_split(tmp_path) == tmp_path
    changed_refs = dict(refs)
    changed_refs["0"] = {"answer": "Nội dung khác."}
    write_json(alias / "selector_dev_references.json", changed_refs)
    with pytest.raises(RuntimeError, match="nội dung selector split"):
        module._find_split(tmp_path)
    write_json(alias / "selector_dev_references.json", refs)
    refs.pop("299")
    write_json(tmp_path / "selector_dev_references.json", refs)
    with pytest.raises(RuntimeError, match="không khớp"):
        module._validate_selector_split(tmp_path, manifest)


def test_qlora_resume_copy_prefers_newest_valid_checkpoint(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_qwen_qlora.py"
    spec = importlib.util.spec_from_file_location("qlora_resume_copy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    attached = tmp_path / "attached"
    target = tmp_path / "working"
    checkpoint20 = attached / "checkpoint-20"
    checkpoint20.mkdir(parents=True)
    (checkpoint20 / "trainer_state.json").write_text("{}", encoding="utf-8")
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint20 / name).write_bytes(name.encode())
    (attached / "resume_signature.json").write_text("{}", encoding="utf-8")
    target.mkdir()
    (target / "qlora_scope.json").write_text("{}", encoding="utf-8")
    module._copy_resume(attached, target)
    assert module._latest_checkpoint_step(target) == 20
    checkpoint40 = target / "checkpoint-40"
    checkpoint40.mkdir()
    (checkpoint40 / "trainer_state.json").write_text("{}", encoding="utf-8")
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint40 / name).write_bytes(name.encode())
    marker = checkpoint40 / "keep.txt"
    marker.write_text("newer", encoding="utf-8")
    module._copy_resume(attached, target)
    assert marker.read_text(encoding="utf-8") == "newer"


def test_selector_holdout_guard_follows_primary_metric():
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "scripts/evaluate_selector.py"
    spec = importlib.util.spec_from_file_location("selector_holdout_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    baseline = {"meteor_exact_probe": 0.40, "rouge_l_f1": 0.50}
    assert not module.better_primary_metric(
        {"meteor_exact_probe": 0.39, "rouge_l_f1": 0.90}, baseline
    )
    assert module.better_primary_metric(
        {"meteor_exact_probe": 0.41, "rouge_l_f1": 0.10}, baseline
    )


def test_selector_trace_validation_rejects_duplicate_sources(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "scripts/evaluate_selector.py"
    spec = importlib.util.spec_from_file_location("selector_trace_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "selected": "extractive",
                "candidates": [
                    {"source": "extractive", "answer": "Câu trả lời."},
                    {"source": "extractive", "answer": "Câu trả lời khác."},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lặp candidate source"):
        module.load_trace(trace)


def test_inference_rejects_invalid_or_unsafe_selector_overrides(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle/run_inference_only.py"
    spec = importlib.util.spec_from_file_location("selector_override_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = minimal_config(tmp_path)
    valid = {
        "schema": SELECTION_WEIGHTS_SCHEMA,
        "selection": {"evidence_support": 0.2},
        "retrieval": {"train_memory_candidate_threshold": 0.9},
    }
    module.apply_selection_weights(config, valid)
    assert config["selection"]["evidence_support"] == 0.2
    assert config["retrieval"]["train_memory_candidate_threshold"] == 0.9
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        module.apply_selection_weights(
            config,
            {"schema": SELECTION_WEIGHTS_SCHEMA, "selection": {"typo_weight": 1.0}},
        )
    with pytest.raises(ValueError, match="hữu hạn"):
        module.apply_selection_weights(
            config,
            {"schema": SELECTION_WEIGHTS_SCHEMA, "selection": {"evidence_support": float("nan")}},
        )
    with pytest.raises(ValueError, match="không được ghi đè"):
        module.apply_selection_weights(
            config,
            {
                "schema": SELECTION_WEIGHTS_SCHEMA,
                "selection": {"evidence_support": 0.2},
                "retrieval": {"bm25_top_k": 1},
            },
        )


def test_selector_calibration_exposes_resume_bundle_flag():
    source = (
        Path(__file__).resolve().parents[1] / "kaggle/run_selector_calibration.py"
    ).read_text(encoding="utf-8")
    assert '"--resume-bundle"' in source
    assert 'command.extend(["--resume-bundle", args.resume_bundle])' in source


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

    query_none, answer_only = parse_agentic_step("Theo Điều 1, người lao động được nghỉ phép.")
    assert query_none is None
    assert answer_only == "Theo Điều 1, người lao động được nghỉ phép."

    # Marker phát ra nhưng không kèm truy vấn: vẫn phải nhận diện là "có ý định
    # tool-call" (query rỗng) chứ không được rơi xuống nhánh coi nguyên văn
    # "TRA_CUU:" là câu trả lời cuối cùng.
    empty_query, empty_answer = parse_agentic_step("TRA_CUU:")
    assert empty_query == ""
    assert empty_answer is None

    fenced_query, fenced_answer = parse_agentic_step(
        "```text\n- TRA_CỨU:\nĐiều kiện hưởng trợ cấp\n```"
    )
    assert fenced_query == "Điều kiện hưởng trợ cấp"
    assert fenced_answer is None

    prefaced_query, _ = parse_agentic_step(
        "Cần tìm thêm căn cứ.\nTRA_CUU: văn bản thay thế hiện hành"
    )
    assert prefaced_query == "văn bản thay thế hiện hành"

    # Model hay bọc nhãn lệnh trong markdown đậm (**TRA_CUU:**) — vẫn phải nhận.
    bold_query, _ = parse_agentic_step("**TRA_CUU:** mức phạt chậm đóng")
    assert bold_query == "mức phạt chậm đóng"


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
    config["generation"]["max_context_chars"] = 500
    long_seed = Chunk("s1", "d1", "X" * 600, "Luật", "Điều 1", {})
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
    # Ngân sách context được tách đôi (agentic_tool_context_share): căn cứ
    # ban đầu phải vẫn còn chỗ trong prompt lượt sau, không bị pool tool-call
    # cộng dồn đẩy hết ra ngoài dù tổng đã vượt max_context_chars.
    assert "X" * 40 in second_prompt


def test_agentic_tool_evidence_caps_chunks_per_document(tmp_path):
    # BM25 thô hay trả về nhiều window của cùng một văn bản ở đầu danh sách;
    # nếu lấy nguyên top thì các slot của một lượt tra cứu chỉ chứa 1 văn bản.
    # agentic_max_per_document=2 phải giữ slot cuối cho văn bản khác.
    config = minimal_config(tmp_path)
    config["generation"].update(
        {"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 3, "agentic_max_per_document": 2}
    )
    seed_evidence = [Evidence(Chunk("c0", "d0", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return GenerationResult("TRA_CUU: mức phạt", 5, True, False)
            return GenerationResult("Theo Điều 5, mức phạt là 0,03%.", 10, True, False)

    hits = [
        Evidence(Chunk(f"d1::s0001::w00{i}", "d1", f"Điều 2. Khoản {i}.", "Luật", "Điều 2", {}), 5.0 - i)
        for i in range(3)
    ] + [Evidence(Chunk("d2::s0001::w000", "d2", "Điều 5. Mức phạt.", "Nghị định", "Điều 5", {}), 1.5)]

    candidate = make_qwen_agentic_candidate(
        FakeGenerator(), "Mức phạt là bao nhiêu?", seed_evidence, config, lambda query: hits
    )
    # 2 window đầu của d1 được chọn và gộp thành 1 mục; window thứ 3 của d1 bị
    # cap bỏ khỏi lựa chọn, slot cuối dành cho d2.
    entries = candidate.details["tool_evidence"]
    assert sorted(entry["doc_id"] for entry in entries) == ["d1", "d2"]
    d1_text = next(entry["text"] for entry in entries if entry["doc_id"] == "d1")
    assert "Khoản 0" in d1_text and "Khoản 1" in d1_text
    assert "Khoản 2" not in d1_text


def test_agentic_tool_evidence_merges_adjacent_windows(tmp_path):
    # Hai window liền kề của cùng điều (w000/w001) chứa text chồng lấn
    # (overlap_chars) — phải được gộp thành 1 mục căn cứ để đỡ lãng phí ngân
    # sách ký tự; lượt tra cứu sau trả lại cùng chunk đó phải bị lọc là
    # đã thấy (kể cả chunk gốc nằm trong merge).
    config = minimal_config(tmp_path)
    config["generation"].update(
        {"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 3, "agentic_max_per_document": 2}
    )
    seed_evidence = [Evidence(Chunk("c0", "d0", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]
    left = Chunk("d1::s0001::w000", "d1", "Điều 2. Người lao động nghỉ phép 12 ngày.", "Luật", "Điều 2", {})
    right = Chunk(
        "d1::s0001::w001", "d1",
        "Điều 2. Người lao động nghỉ phép 12 ngày. Khi báo trước 30 ngày.",
        "Luật", "Điều 2", {},
    )

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls < 3:
                return GenerationResult(f"TRA_CUU: truy vấn {self.calls}", 5, True, False)
            return GenerationResult("Theo Điều 2, nghỉ phép 12 ngày.", 10, True, False)

    def fake_search(query: str) -> list[Evidence]:
        return [Evidence(left, 3.0), Evidence(right, 2.9)]

    candidate = make_qwen_agentic_candidate(
        FakeGenerator(), "Người lao động nghỉ mấy ngày?", seed_evidence, config, fake_search
    )
    # Lượt 1: 2 window gộp thành 1 mục; lượt 2: cùng 2 chunk đã đánh dấu thấy
    # (kể cả chunk gốc nằm trong merge) -> không thêm gì nữa.
    assert candidate.details["evidence_added_total"] == 1
    assert len(candidate.details["tool_evidence"]) == 1
    merged_text = candidate.details["tool_evidence"][0]["text"]
    assert "báo trước 30 ngày" in merged_text
    assert merged_text.count("nghỉ phép 12 ngày") == 1


def test_agentic_tool_search_skips_sources_already_merged_into_seed(tmp_path):
    config = minimal_config(tmp_path)
    config["generation"].update(
        {"agentic_max_tool_calls": 1, "agentic_evidence_per_call": 2}
    )
    merged_seed = Evidence(
        Chunk(
            "d1::merged::left+right",
            "d1",
            "Điều 2. Nội dung đã gộp.",
            "Luật",
            "Điều 2",
            {"_merged_chunk_ids": ["left", "right"]},
        ),
        0.9,
    )
    duplicate_left = Evidence(
        Chunk("left", "d1", "Điều 2. Nội dung cũ.", "Luật", "Điều 2", {}),
        3.0,
    )
    genuinely_new = Evidence(
        Chunk("new", "d2", "Điều 5. Căn cứ mới.", "Nghị định", "Điều 5", {}),
        2.0,
    )

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return GenerationResult("TRA_CUU: căn cứ mới", 5, True, False)
            return GenerationResult("Theo Điều 5, đây là căn cứ mới.", 8, True, False)

    candidate = make_qwen_agentic_candidate(
        FakeGenerator(),
        "Căn cứ mới là gì?",
        [merged_seed],
        config,
        lambda query: [duplicate_left, genuinely_new],
    )
    assert [entry["chunk_id"] for entry in candidate.details["tool_evidence"]] == [
        "new"
    ]


def test_agentic_candidate_blocks_repeated_query_and_forces_final_answer(tmp_path):
    config = minimal_config(tmp_path)
    config["generation"].update({"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 1})
    seed_evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            if self.calls <= 2:
                return GenerationResult("TRA_CUU: mức phạt chậm đóng", 5, True, False)
            assert "CÔNG CỤ TRA CỨU" not in user
            return GenerationResult("Theo Điều 1, áp dụng quy định đã nêu.", 10, True, False)

    search_calls: list[str] = []

    def fake_search(query: str) -> list[Evidence]:
        search_calls.append(query)
        return []

    candidate = make_qwen_agentic_candidate(
        FakeGenerator(), "Mức phạt là bao nhiêu?", seed_evidence, config, fake_search
    )
    assert search_calls == ["mức phạt chậm đóng"]
    assert candidate.details["tool_calls_used"] == 2
    assert candidate.details["tool_calls"][1]["repeated_query"] is True


def test_agentic_query_dedup_normalizes_vietnamese_d_stroke(tmp_path):
    config = minimal_config(tmp_path)
    config["generation"].update({"agentic_max_tool_calls": 2, "agentic_evidence_per_call": 1})
    seed = [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, system, user, *, max_new_tokens=None):
            self.calls += 1
            outputs = (
                "TRA_CUU: Điều kiện áp dụng",
                "TRA_CUU: dieu kien ap dung",
                "Theo Điều 1, quy định được áp dụng.",
            )
            return GenerationResult(outputs[self.calls - 1], 6, True, False)

    calls: list[str] = []
    candidate = make_qwen_agentic_candidate(
        FakeGenerator(),
        "Điều kiện áp dụng là gì?",
        seed,
        config,
        lambda query: calls.append(query) or [],
    )
    assert calls == ["Điều kiện áp dụng"]
    assert candidate.details["tool_calls"][1]["repeated_query"] is True


def test_tool_evidence_is_candidate_specific_and_cannot_credit_base_qwen(tmp_path):
    config = minimal_config(tmp_path)
    seed_chunk = Chunk("c1", "d1", "Điều 1. Người lao động có quyền nghỉ phép.", "Luật", "Điều 1", {})
    tool_chunk_text = "Điều 2. Mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    seed_evidence = [Evidence(seed_chunk, 0.9)]
    question = "Mức phạt chậm đóng bảo hiểm xã hội là bao nhiêu?"
    answer_text = "Theo Điều 2, mức phạt chậm đóng bảo hiểm xã hội là 0,03 phần trăm mỗi ngày."
    tool_evidence = [{
        "chunk_id": "c2", "doc_id": "d1", "title": "Luật", "heading": "Điều 2",
        "text": tool_chunk_text, "score": 0.8, "bm25_score": 0.8,
    }]

    base = Candidate("qwen3.5-2b", answer_text)
    agentic = Candidate("qwen3.5-2b-agentic", answer_text, details={"tool_evidence": tool_evidence})
    base_evidence = _candidate_evidence(seed_evidence, base)
    agentic_evidence = _candidate_evidence(seed_evidence, agentic)
    assert len(seed_evidence) == 1
    assert len(base_evidence) == 1
    assert len(agentic_evidence) == 2
    ranked = score_candidates(
        question,
        seed_evidence,
        [base, agentic],
        config,
        candidate_evidence_by_source={
            base.source: base_evidence,
            agentic.source: agentic_evidence,
        },
    )
    scored = {candidate.source: candidate for candidate in ranked}
    assert scored["qwen3.5-2b"].details["citation_precision"] == 0.0
    assert scored["qwen3.5-2b-agentic"].details["citation_precision"] == 1.0


def test_candidate_evidence_deduplicates_merged_tool_sources():
    seed = Evidence(
        Chunk("left", "d1", "Điều 2. Nội dung gốc.", "Luật", "Điều 2", {}),
        0.9,
    )
    agentic = Candidate(
        "qwen3.5-2b-agentic",
        "Theo Điều 2, nội dung được áp dụng.",
        details={
            "tool_evidence": [
                {
                    "chunk_id": "d1::merged::left+right",
                    "doc_id": "d1",
                    "title": "Luật",
                    "heading": "Điều 2",
                    "text": "Điều 2. Nội dung gốc và phần tiếp theo.",
                    "score": 1.0,
                    "merged_chunk_ids": ["left", "right"],
                }
            ]
        },
    )
    active = _candidate_evidence([seed], agentic)
    assert len(active) == 1
    assert active[0].chunk.chunk_id == "d1::merged::left+right"


def test_agentic_prior_falls_back_to_base_qwen_prior(tmp_path):
    config = minimal_config(tmp_path)
    config["selection"]["source_priors"] = {"qwen3.5-2b": 0.08}
    evidence = [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]
    candidates = [
        Candidate("qwen3.5-2b", "Theo Điều 1, nội dung được áp dụng."),
        Candidate("qwen3.5-2b-agentic", "Theo Điều 1, nội dung được áp dụng."),
    ]
    ranked = score_candidates("Nội dung áp dụng thế nào?", evidence, candidates, config)
    scores = {candidate.source: candidate.score for candidate in ranked}
    assert scores["qwen3.5-2b"] == pytest.approx(scores["qwen3.5-2b-agentic"])


def test_agentic_gate_skips_simple_confident_and_keeps_complex(tmp_path):
    config = minimal_config(tmp_path)
    config["generation"].update(
        agentic_routes=["complex"],
        agentic_confidence_below=0.45,
        agentic_on_legal_cues=True,
    )
    evidence = [Evidence(Chunk("c1", "d1", "Quy định.", "Luật", "Điều 1", {}), 0.9)]
    simple = {"route": {"route": "simple", "confidence": 0.82, "features": {}}}
    complex_case = {"route": {"route": "complex", "confidence": 0.72, "features": {}}}
    assert _should_run_agentic("Quy định này là gì?", evidence, simple, config)[0] is False
    assert _should_run_agentic("Văn bản nào thay thế quy định này?", evidence, complex_case, config)[0] is True


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("TRA_CUU không hợp lệ"),
        ValueError("tool query lỗi"),
        sqlite3.OperationalError("FTS search lỗi"),
        TypeError("tool metadata sai kiểu"),
        KeyError("relation_type"),
    ],
)
def test_agentic_nonfatal_errors_do_not_abort_or_skip_remaining_questions(
    tmp_path, monkeypatch, error
):
    config = minimal_config(tmp_path)
    config["paths"]["models"]["generator"] = "fake-model"
    config["generation"].update(
        qwen_enabled=False,
        agentic_enabled=True,
        agentic_routes=["complex"],
        agentic_warn_after_nonfatal_errors=3,
    )
    questions = [QARecord(f"q{i}", f"Văn bản nào thay thế quy định số {i}?", None) for i in range(4)]
    seed = Evidence(Chunk("c1", "d1", "Điều 1. Nội dung pháp luật.", "Luật", "Điều 1", {}), 0.9)
    evidence = RetrievalBatch(
        {item.question_id: [seed] for item in questions},
        {
            item.question_id: {"route": {"route": "complex", "confidence": 0.7, "features": {}}}
            for item in questions
        },
    )

    monkeypatch.setattr("legalqa.pipeline.QwenTextGenerator", lambda config: object())

    def fail_agentic(*args, **kwargs):
        raise error

    monkeypatch.setattr("legalqa.pipeline.make_qwen_agentic_candidate", fail_agentic)
    output = tmp_path / "predictions.json"
    result = generate_from_evidence(
        config,
        questions,
        evidence,
        output,
        agentic_search_fn=lambda query: [],
    )
    assert result["questions"] == 4
    assert result["generation_errors"] == 4
    assert result["agentic"]["failed"] == 4
    assert output.is_file()
    cache = Path(config["paths"]["artifacts"]) / "candidate_cache" / "qwen_agentic.jsonl"
    assert not cache.exists()  # transient/nonfatal failures must remain retryable


def test_agentic_fatal_cuda_error_still_aborts_for_safe_resume(tmp_path, monkeypatch):
    config = minimal_config(tmp_path)
    config["paths"]["models"]["generator"] = "fake-model"
    config["generation"].update(
        qwen_enabled=False,
        agentic_enabled=True,
        agentic_routes=["complex"],
    )
    question = QARecord("q1", "Văn bản nào thay thế quy định này?", None)
    seed = Evidence(
        Chunk("c1", "d1", "Điều 1. Nội dung pháp luật.", "Luật", "Điều 1", {}),
        0.9,
    )
    evidence = RetrievalBatch(
        {question.question_id: [seed]},
        {
            question.question_id: {
                "route": {"route": "complex", "confidence": 0.7, "features": {}}
            }
        },
    )
    monkeypatch.setattr("legalqa.pipeline.QwenTextGenerator", lambda config: object())

    def fail_agentic(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("legalqa.pipeline.make_qwen_agentic_candidate", fail_agentic)
    with pytest.raises(RuntimeError, match="GPU nghiêm trọng"):
        generate_from_evidence(
            config,
            [question],
            evidence,
            tmp_path / "predictions.json",
            agentic_search_fn=lambda query: [],
        )


def test_agentic_legal_rerank_prefers_exact_citation_and_rejects_weak_tail(tmp_path):
    config = minimal_config(tmp_path)
    candidates = [
        Evidence(Chunk("weak", "d1", "Nội dung chung.", "Luật", "", {}), 9.0, bm25_score=9.0),
        Evidence(
            Chunk("exact", "d2", "Điều 7 quy định mức phạt.", "Nghị định 12/2024/NĐ-CP", "Điều 7", {}),
            5.0,
            bm25_score=5.0,
            relation_type="REPLACES",
            relation_path=["d1", "REPLACES", "d2"],
        ),
    ]
    ranked = rank_agentic_evidence(
        "Theo Điều 7 Nghị định 12/2024/NĐ-CP, mức phạt là bao nhiêu?",
        "Điều 7 mức phạt",
        [],
        candidates,
        config,
    )
    assert ranked[0].chunk.chunk_id == "exact"


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
        assert results[0].bm25_score > 0
        assert results[0].score > 0
        assert results[0].score != results[0].bm25_score  # legal-reranked, not raw BM25


def test_inference_runtime_budget_pauses_before_loading_qwen(tmp_path, monkeypatch):
    config = minimal_config(tmp_path)
    config["paths"]["models"]["generator"] = "fake-model"
    config["generation"]["qwen_enabled"] = True
    config["runtime"].update(
        stop_at_unix=1.0,
        recovery_root=str(tmp_path),
        recovery_output=str(tmp_path / "legalqa_recovery_latest.zip"),
        run_fingerprint="pause-test",
        checkpoint_every=50,
    )
    question = QARecord("q1", "Quy định này là gì?", None)
    evidence = {
        "q1": [Evidence(Chunk("c1", "d1", "Điều 1. Nội dung.", "Luật", "Điều 1", {}), 0.9)]
    }

    def must_not_load(_config):
        raise AssertionError("Qwen must not load after the safe deadline")

    monkeypatch.setattr("legalqa.pipeline.QwenTextGenerator", must_not_load)
    with pytest.raises(InferencePaused) as raised:
        generate_from_evidence(config, [question], evidence, tmp_path / "predictions.json")
    assert raised.value.stage == "qwen"
    bundle = tmp_path / "legalqa_recovery_latest.zip"
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        state = json.loads(archive.read("recovery_state.json").decode("utf-8"))
    assert state["stage"] == "qwen"
    assert state["run_fingerprint"] == "pause-test"


def test_inference_recovery_rejects_windows_backslash_traversal(tmp_path):
    bundle = tmp_path / "malicious-recovery.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "recovery_state.json",
            json.dumps({"run_fingerprint": "expected", "stage": "qwen"}),
        )
        archive.writestr("..\\escaped.txt", "bad")
    with pytest.raises(RecoveryError, match="(Unsafe path|không an toàn)"):
        restore_recovery_bundle(bundle, tmp_path / "restore", expected_fingerprint="expected")
    assert not (tmp_path / "escaped.txt").exists()


def test_attached_dense_and_train_memory_are_validated_before_models(tmp_path):
    import importlib.util

    project = Path(__file__).resolve().parents[1]
    path = project / "kaggle" / "run_inference_only.py"
    spec = importlib.util.spec_from_file_location("inference_index_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "vectors.f16").write_bytes(b"0" * 12)  # 2 chunks * 3 dims * fp16
    (dense / "rowids.i64").write_bytes(b"0" * 16)
    (dense / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "global-dense-v1",
                "identity": "dense-test",
                "chunks": 2,
                "dimension": 3,
                "vectors": "vectors.f16",
                "rowids": "rowids.i64",
            }
        ),
        encoding="utf-8",
    )
    assert module.validate_dense_index_input(dense)["chunks"] == 2
    with pytest.raises(RuntimeError, match="another corpus or embedding"):
        module.validate_dense_index_input(
            dense, corpus_identity="official-corpus", embedding_identity="approved-embedding"
        )

    memory = dense / "train_memory"
    memory.mkdir()
    (memory / "vectors.f16").write_bytes(b"0" * 12)
    records = [
        {"question_id": "q1", "question": "Câu một?", "answer": "Đáp án một."},
        {"question_id": "q2", "question": "Câu hai?", "answer": "Đáp án hai."},
    ]
    (memory / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (memory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "train-memory-v1",
                "identity": "memory-test",
                "count": 2,
                "dimension": 3,
                "vectors": "vectors.f16",
                "records": "records.jsonl",
            }
        ),
        encoding="utf-8",
    )
    assert module.validate_train_memory_input(memory)["count"] == 2
    with pytest.raises(RuntimeError, match="another train set or embedding"):
        module.validate_train_memory_input(
            memory, train_identity="official-train", embedding_identity="approved-embedding"
        )
    (dense / "vectors.f16").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="missing/truncated"):
        module.validate_dense_index_input(dense)


def test_sqlite_tool_search_recovers_diacritic_less_query(tmp_path):
    # Model 2B đôi lúc phát truy vấn không dấu ("muc phat cham dong") trong khi
    # FTS5 index giữ nguyên dấu -> term không khớp, lượt tool-call trả về 0
    # kết quả. Tool search phải khôi phục dấu từ vocab FTS và vẫn tìm thấy văn
    # bản chứa "mức phạt chậm đóng bảo hiểm xã hội".
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

    with SQLiteLegalStore(database, readonly=True) as store:
        assert store.search_chunks("muc phat cham dong bao hiem", top_k=10) == []

    with sqlite_bm25_search_fn(config, database) as search_fn:
        results = search_fn("muc phat cham dong bao hiem xa hoi")
        assert results
        assert results[0].chunk.doc_id == "d1"
        assert results[0].stage == "tool_call"
