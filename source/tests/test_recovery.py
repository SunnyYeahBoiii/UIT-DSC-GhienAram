from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest

from legalqa.recovery import (
    RecoveryError,
    create_recovery_bundle,
    recovery_input_state,
    restore_recovery_bundle,
)


PROJECT = Path(__file__).resolve().parents[1]


def load_reranker_script():
    path = PROJECT / "scripts" / "train_reranker.py"
    spec = importlib.util.spec_from_file_location("legalqa_train_reranker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_latest_reranker_checkpoint_ignores_partial_directory(tmp_path):
    module = load_reranker_script()
    complete_5 = tmp_path / "checkpoint-5"
    complete_5.mkdir()
    (complete_5 / "training_state.pt").write_bytes(b"complete")
    partial_20 = tmp_path / "checkpoint-20"
    partial_20.mkdir()
    complete_12 = tmp_path / "checkpoint-12"
    complete_12.mkdir()
    (complete_12 / "training_state.pt").write_bytes(b"complete")
    assert module.latest_checkpoint(tmp_path) == complete_12


def test_v7_runner_exposes_recovery_controls():
    runner = (PROJECT / "kaggle" / "run_end_to_end.py").read_text(encoding="utf-8")
    qwen = (PROJECT / "scripts" / "train_qwen_lora.py").read_text(encoding="utf-8")
    assert "--resume-training" in runner
    assert "--submission-output" in runner
    assert 'save_strategy="steps"' in qwen
    assert "resume_from_checkpoint=resume_checkpoint" in qwen
    assert "save_only_model=False" in qwen


def test_inference_only_runner_never_builds_training_pairs():
    runner = (PROJECT / "kaggle" / "run_inference_only.py").read_text(encoding="utf-8")
    models = (PROJECT / "src" / "legalqa" / "models.py").read_text(encoding="utf-8")
    assert "prepare_sqlite_weak_pairs" not in runner
    assert "train_qwen_lora" not in runner
    assert "train_reranker" not in runner
    assert '"training_performed": False' in runner
    assert "--resume-bundle" in runner
    assert "--auto-resume" in runner
    assert "discover_recovery_inputs" in runner
    assert "--checkpoint-every" in runner
    assert "smoke_inference_models" in runner
    assert "corpus_content_fingerprint" in runner
    assert 'if not self.device.startswith("cuda")' in models


def test_inference_recovery_roundtrip_keeps_only_resumable_state(tmp_path):
    first = tmp_path / "first"
    retrieval = first / "cache" / "abc" / "public_retrieval.inference_only.jsonl"
    candidates = first / "artifacts" / "abc" / "candidate_cache" / "qwen.jsonl"
    unrelated = first / "artifacts" / "other" / "candidate_cache" / "qwen.jsonl"
    sqlite = first / "official_corpus.sqlite"
    retrieval.parent.mkdir(parents=True)
    candidates.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    retrieval.write_text('{"question_id":"1"}\n', encoding="utf-8")
    candidates.write_text('{"cache_key":"k","answer":"a"}\n', encoding="utf-8")
    unrelated.write_text('{"cache_key":"wrong","answer":"wrong"}\n', encoding="utf-8")
    sqlite.write_bytes(b"must not be archived")
    bundle = tmp_path / "legalqa_recovery_latest.zip"

    report = create_recovery_bundle(
        first,
        bundle,
        fingerprint="abc",
        stage="qwen",
        completed=50,
        total=1000,
    )
    assert report["completed"] == 50

    second = tmp_path / "second"
    restored = restore_recovery_bundle(bundle, second, expected_fingerprint="abc")
    assert restored["stage"] == "qwen"
    assert (second / retrieval.relative_to(first)).read_text(encoding="utf-8") == retrieval.read_text(encoding="utf-8")
    assert (second / candidates.relative_to(first)).read_text(encoding="utf-8") == candidates.read_text(encoding="utf-8")
    assert not (second / unrelated.relative_to(first)).exists()
    assert not (second / "official_corpus.sqlite").exists()


def test_inference_recovery_accepts_kaggle_extracted_directory(tmp_path):
    first = tmp_path / "first"
    retrieval = first / "cache" / "abc" / "retrieval.v7.jsonl"
    candidates = first / "artifacts" / "abc" / "candidate_cache" / "qwen_agentic.jsonl"
    retrieval.parent.mkdir(parents=True)
    candidates.parent.mkdir(parents=True)
    retrieval.write_text('{"question_id":"1"}\n', encoding="utf-8")
    candidates.write_text('{"cache_key":"k","answer":"a"}\n', encoding="utf-8")
    bundle = tmp_path / "legalqa_recovery_latest.zip"
    create_recovery_bundle(
        first,
        bundle,
        fingerprint="abc",
        stage="paused",
        completed=211,
        total=372,
    )

    extracted = tmp_path / "legalqa_recovery_latest"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extracted)

    restored_root = tmp_path / "restored"
    restored = restore_recovery_bundle(
        extracted,
        restored_root,
        expected_fingerprint="abc",
    )
    assert restored["stage"] == "paused"
    assert restored["completed"] == 211
    assert restored["source_bundle"] == str(extracted.resolve())
    assert recovery_input_state(extracted)["completed"] == 211
    assert recovery_input_state(bundle)["completed"] == 211
    assert (restored_root / retrieval.relative_to(first)).is_file()
    assert (restored_root / candidates.relative_to(first)).is_file()


def test_inference_recovery_directory_rejects_other_fingerprint(tmp_path):
    source = tmp_path / "legalqa_recovery_latest"
    source.mkdir()
    (source / "recovery_state.json").write_text(
        '{"format_version":1,"run_fingerprint":"old","stage":"paused","completed":1,"total":2}',
        encoding="utf-8",
    )
    with pytest.raises(RecoveryError, match="expected=new"):
        restore_recovery_bundle(source, tmp_path / "restored", expected_fingerprint="new")


def test_inference_recovery_rejects_other_fingerprint(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    bundle = tmp_path / "legalqa_recovery_latest.zip"
    create_recovery_bundle(
        root,
        bundle,
        fingerprint="old",
        stage="retrieval",
        completed=1000,
        total=1000,
    )
    with pytest.raises(RecoveryError, match="expected=new"):
        restore_recovery_bundle(bundle, tmp_path / "restored", expected_fingerprint="new")
