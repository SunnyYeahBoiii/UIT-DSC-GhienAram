from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from legalqa.fingerprint import adapter_fingerprint, corpus_content_fingerprint, sha256_file


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "verify_v5007_adapter_ablation.py"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("adapter_ablation_guard", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_ablation_guard_accepts_a_single_consistent_fixture(tmp_path: Path) -> None:
    guard = _load_guard_module()
    public = tmp_path / "public-official.json"
    train = tmp_path / "train.json"
    corpus = tmp_path / "selected-contexts"
    corpus.mkdir()
    public.write_text('{"1": {"question": "Q"}}', encoding="utf-8")
    train.write_text('{"1": {"question": "Q", "answer": "A"}}', encoding="utf-8")
    (corpus / "context_001.json").write_text('{"id": "1", "text": "Điều 1"}', encoding="utf-8")

    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "vectors.f16").write_bytes(b"\0" * (3 * 2 * 2))
    (dense / "rowids.i64").write_bytes(b"\0" * (3 * 8))
    (dense / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "global-dense-v1",
                "identity": "dense-fixture",
                "chunks": 3,
                "dimension": 2,
                "dtype": "float16",
                "vectors": "vectors.f16",
                "rowids": "rowids.i64",
            }
        ),
        encoding="utf-8",
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/models/Qwen3.5-2B"}), encoding="utf-8"
    )
    (adapter / "qlora_scope.json").write_text('{"train_scope": "full"}', encoding="utf-8")
    (adapter / "training_complete.json").write_text('{"global_step": 7}', encoding="utf-8")

    selector = tmp_path / "tuned_selection.yaml"
    selector.write_text(yaml.safe_dump({"schema": "selector-weights-v3-agentic-evidence", "selection": {"fluency": 0.2}}), encoding="utf-8")
    expected = {
        "questions_sha256": sha256_file(public),
        "train_sha256": sha256_file(train),
        "corpus_sha256": corpus_content_fingerprint(corpus),
        "dense_identity": "dense-fixture",
        "dense_chunks": 3,
        "dense_dimension": 2,
        "adapter_fingerprint": adapter_fingerprint(adapter),
        "adapter_scope": "full",
        "adapter_global_step": 7,
        "selector_sha256": sha256_file(selector),
        "selector_schema": "selector-weights-v3-agentic-evidence",
        "expected_run_fingerprint": "fixture-fingerprint",
    }

    report = guard.validate_ablation_inputs(
        public_file=public,
        train_file=train,
        corpus_dir=corpus,
        dense_index_dir=dense,
        adapter_dir=adapter,
        selector_weights=selector,
        expected=expected,
    )

    assert report["status"] == "ok"
    assert report["inputs"]["adapter"]["fingerprint"] == expected["adapter_fingerprint"]
    assert report["inputs"]["dense"]["chunks"] == 3


def test_ablation_notebook_and_runner_pin_both_identities() -> None:
    runner = (PROJECT / "kaggle" / "run_inference_only.py").read_text(encoding="utf-8")
    notebook = (PROJECT / "kaggle" / "legalqa_dsc2026_v5007_adapter_clean_ablation_t4.ipynb").read_text(encoding="utf-8")

    assert "--expected-generator-adapter-fingerprint" in runner
    assert "--expected-run-fingerprint" in runner
    assert "590a732d4ea601e20b68d27407206265a52fd5753402ae2f9dc3e7084ccd850a" in notebook
    assert "8fc470f863a34338" in notebook
    assert "--no-train-memory" in notebook
