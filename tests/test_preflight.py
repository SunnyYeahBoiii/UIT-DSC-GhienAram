from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_kaggle.py"
SPEC = importlib.util.spec_from_file_location("legalqa_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_find_model_prefers_snapshot_root_over_nested_onnx(tmp_path: Path) -> None:
    root = tmp_path / "Vietnamese_Embedding"
    onnx = root / "onnx"
    onnx.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (onnx / "config.json").write_text("{}", encoding="utf-8")
    assert preflight.find_model(tmp_path, "embedding") == root


def test_find_model_rejects_two_exact_snapshots(tmp_path: Path) -> None:
    for dataset in ("dataset-a", "dataset-b"):
        model = tmp_path / dataset / "ViLegalBERT"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Có nhiều checkpoint"):
        preflight.find_model(tmp_path, "verifier")


def test_checkpoint_index_requires_every_shard(tmp_path: Path) -> None:
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    with pytest.raises(FileNotFoundError, match="thiếu shards"):
        preflight.checkpoint_weight_files(tmp_path)


def test_checkpoint_falls_back_to_legacy_bin(tmp_path: Path) -> None:
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"legacy")
    assert preflight.checkpoint_weight_files(tmp_path) == [weights]
