from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from legalqa.fingerprint import (
    RUNTIME_SOURCE_DIRECTORIES,
    RUNTIME_SOURCE_FILES,
    adapter_fingerprint,
    corpus_content_fingerprint,
    normalized_text_sha256,
    runtime_source_fingerprint,
    sha256_file,
)
from legalqa.recovery import create_recovery_bundle


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "verify_v5007_adapter_ablation.py"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("adapter_ablation_guard", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    path = PROJECT / "kaggle" / "run_inference_only.py"
    spec = importlib.util.spec_from_file_location("adapter_ablation_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("bom", [False, True])
def test_selector_identity_normalizes_line_endings_and_bom(
    tmp_path: Path, newline: str, bom: bool
) -> None:
    selector = tmp_path / "tuned_selection.yaml"
    baseline = "schema: selector-weights-v3-agentic-evidence\nselection:\n  fluency: 0.2\n"
    selector.write_bytes(baseline.encode("utf-8"))
    expected = normalized_text_sha256(selector)
    encoded = baseline.replace("\n", newline).encode("utf-8")
    selector.write_bytes((b"\xef\xbb\xbf" if bom else b"") + encoded)
    assert normalized_text_sha256(selector) == expected

    notebook = json.loads(
        (PROJECT / "kaggle" / "legalqa_dsc2026_v5007_adapter_clean_ablation_t4.ipynb").read_text(encoding="utf-8")
    )
    discovery = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "weight_candidates = []" in "".join(cell.get("source", []))
    )
    discovery = discovery[discovery.index("weight_candidates = []"):discovery.index("selection_weights = weight_candidates[0]")]
    namespace = {
        "input_root": tmp_path,
        "normalized_text_sha256": normalized_text_sha256,
        "yaml": yaml,
        "EXPECTED_SELECTOR_SHA256": expected,
        "EXPECTED_SELECTOR_SCHEMA": "selector-weights-v3-agentic-evidence",
    }
    exec(compile(discovery, "<notebook-selector-discovery>", "exec"), namespace)
    assert namespace["weight_candidates"] == [selector]


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
        "selector_sha256": normalized_text_sha256(selector),
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
        source_dir=PROJECT,
        expected_source_fingerprint=runtime_source_fingerprint(PROJECT),
        expected_run_fingerprint="fixture-run",
        expected=expected,
    )

    assert report["status"] == "ok"
    assert report["inputs"]["adapter"]["fingerprint"] == expected["adapter_fingerprint"]
    assert report["inputs"]["dense"]["chunks"] == 3
    assert report["inputs"]["source_fingerprint"] == runtime_source_fingerprint(PROJECT)
    assert report["expected_run_fingerprint"] == "fixture-run"

    # The same selector saved by Windows must pass the complete guard too.
    selector.write_bytes(selector.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    windows_report = guard.validate_ablation_inputs(
        public_file=public,
        train_file=train,
        corpus_dir=corpus,
        dense_index_dir=dense,
        adapter_dir=adapter,
        selector_weights=selector,
        expected=expected,
    )
    assert windows_report["inputs"]["selector_sha256"] == expected["selector_sha256"]

    selector.write_text(yaml.safe_dump({"schema": expected["selector_schema"], "selection": {"fluency": 0.3}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Selector YAML content differs"):
        guard.validate_ablation_inputs(
            public_file=public,
            train_file=train,
            corpus_dir=corpus,
            dense_index_dir=dense,
            adapter_dir=adapter,
            selector_weights=selector,
            expected=expected,
        )


def test_ablation_notebook_and_runner_pin_both_identities() -> None:
    runner = (PROJECT / "kaggle" / "run_inference_only.py").read_text(encoding="utf-8")
    notebook = (PROJECT / "kaggle" / "legalqa_dsc2026_v5007_adapter_clean_ablation_t4.ipynb").read_text(encoding="utf-8")

    assert "--expected-generator-adapter-fingerprint" in runner
    assert "--expected-source-fingerprint" in runner
    assert "--expected-run-fingerprint" in runner
    assert "590a732d4ea601e20b68d27407206265a52fd5753402ae2f9dc3e7084ccd850a" in notebook
    assert "97aaa9cabf0385f9d72abf8359910ece5f3e539cd151a65db7225c7159c7f0a1" in notebook
    assert runtime_source_fingerprint(PROJECT) in notebook
    assert "00345b636cec2c37" in notebook
    assert "--no-train-memory" in notebook


def test_runtime_source_fingerprint_detects_runtime_edits_but_ignores_notebook(tmp_path: Path) -> None:
    for relative in RUNTIME_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))
    for relative in RUNTIME_SOURCE_DIRECTORIES:
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fixture.txt").write_bytes(f"fixture:{relative}\n".encode("utf-8"))

    before = runtime_source_fingerprint(tmp_path)
    explicit_text = tmp_path / RUNTIME_SOURCE_FILES[0]
    explicit_text.write_bytes(explicit_text.read_bytes().replace(b"\n", b"\r\n"))
    assert runtime_source_fingerprint(tmp_path) == before

    notebook = tmp_path / "kaggle" / "legalqa_dsc2026_v5007_adapter_clean_ablation_t4.ipynb"
    notebook.write_text("not part of runtime identity", encoding="utf-8")
    assert runtime_source_fingerprint(tmp_path) == before

    (tmp_path / "src" / "fixture.txt").write_text("runtime changed\n", encoding="utf-8")
    assert runtime_source_fingerprint(tmp_path) != before


def test_recovery_discovery_accepts_renamed_zip_and_deduplicates_extracted_alias(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    renamed_zip = tmp_path / "legalqa recovery_latest (6).zip"
    create_recovery_bundle(
        run_root,
        renamed_zip,
        fingerprint="expected-run",
        stage="paused",
        completed=211,
        total=372,
    )
    extracted = tmp_path / "custom-kaggle-dataset-slug"
    with zipfile.ZipFile(renamed_zip) as archive:
        archive.extractall(extracted)

    found = runner.discover_recovery_inputs(
        tmp_path,
        expected_fingerprint="expected-run",
    )
    assert found == [extracted]
