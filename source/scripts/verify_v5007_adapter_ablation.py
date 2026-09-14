#!/usr/bin/env python3
"""Fail-fast input guard for the v0.5007 clean-adapter A/B experiment.

This tool deliberately validates inputs only.  It never changes the inference
configuration, candidates, retrieval cache, or model outputs.  Its purpose is
to prove that the public run differs from the 0.5007 control solely in the
full QLoRA adapter identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.fingerprint import (  # noqa: E402
    adapter_fingerprint,
    corpus_content_fingerprint,
    sha256_file,
)


EXPECTED = {
    "questions_sha256": "5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5",
    "train_sha256": "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988",
    "corpus_sha256": "39e641056a2300ea493808b6be74f90c81ef367fb0733e415105d71d9d25a893",
    "dense_identity": "b5c6eadb46ee7aceab44b8643fa8a5beefde5420c02cf2146e0fd30e38eccf23",
    "dense_chunks": 215267,
    "dense_dimension": 1024,
    "adapter_fingerprint": "590a732d4ea601e20b68d27407206265a52fd5753402ae2f9dc3e7084ccd850a",
    "adapter_scope": "full",
    "adapter_global_step": 169,
    "selector_sha256": "85a00e46e988b013741ab94c0b5bcf4777b6909aff4d2b3b60d00664f06ada38",
    "selector_schema": "selector-weights-v3-agentic-evidence",
    "expected_run_fingerprint": "8fc470f863a34338",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _validate_dense_index(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "Dense manifest")
    _require(manifest.get("status") == "complete", "Dense index is not complete")
    _require(manifest.get("schema") == "global-dense-v1", "Dense schema is not global-dense-v1")
    _require(
        manifest.get("identity") == expected["dense_identity"],
        "Dense identity differs from v0.5007",
    )
    _require(
        int(manifest.get("chunks", 0)) == int(expected["dense_chunks"]),
        "Dense chunk count differs from v0.5007 (must be 215267)",
    )
    _require(
        int(manifest.get("dimension", 0)) == int(expected["dense_dimension"]),
        "Dense dimension differs from v0.5007 (must be 1024)",
    )
    _require(manifest.get("dtype") == "float16", "Dense dtype must be float16")

    vectors = root / str(manifest.get("vectors", ""))
    rowids = root / str(manifest.get("rowids", ""))
    _require(vectors.name == str(manifest.get("vectors", "")), "Unsafe dense vector filename")
    _require(rowids.name == str(manifest.get("rowids", "")), "Unsafe dense rowid filename")
    expected_vectors = int(expected["dense_chunks"]) * int(expected["dense_dimension"]) * 2
    expected_rowids = int(expected["dense_chunks"]) * 8
    _require(vectors.is_file() and vectors.stat().st_size == expected_vectors, "Dense vectors.f16 is missing or truncated")
    _require(rowids.is_file() and rowids.stat().st_size == expected_rowids, "Dense rowids.i64 is missing or truncated")
    return manifest


def _validate_adapter(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    required = ("adapter_config.json", "adapter_model.safetensors", "training_complete.json", "qlora_scope.json")
    missing = [name for name in required if not (root / name).is_file()]
    _require(not missing, f"Clean adapter is incomplete: {missing}")
    scope = _read_json(root / "qlora_scope.json", "Adapter scope")
    complete = _read_json(root / "training_complete.json", "Adapter completion")
    adapter_config = _read_json(root / "adapter_config.json", "Adapter config")
    _require(scope.get("train_scope") == expected["adapter_scope"], "Adapter must have train_scope='full'")
    _require(
        int(complete.get("global_step", 0)) == int(expected["adapter_global_step"]),
        "Adapter global_step differs from the selected clean adapter",
    )
    base_model = str(adapter_config.get("base_model_name_or_path", ""))
    _require(base_model.rstrip("/\\").endswith("Qwen3.5-2B"), "Adapter base model is not Qwen3.5-2B")
    identity = adapter_fingerprint(root)
    _require(identity == expected["adapter_fingerprint"], "Adapter fingerprint differs from clean A1")
    return {"fingerprint": identity, "global_step": complete["global_step"], "scope": scope["train_scope"]}


def validate_ablation_inputs(
    *,
    public_file: Path,
    train_file: Path,
    corpus_dir: Path,
    dense_index_dir: Path,
    adapter_dir: Path,
    selector_weights: Path,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every non-model input for a one-variable adapter experiment."""
    expected = dict(EXPECTED if expected is None else expected)
    for label, path in {
        "public file": public_file,
        "train file": train_file,
        "corpus": corpus_dir,
        "dense index": dense_index_dir,
        "adapter": adapter_dir,
        "selector": selector_weights,
    }.items():
        _require(path.exists(), f"Missing {label}: {path}")

    public_sha = sha256_file(public_file)
    train_sha = sha256_file(train_file)
    corpus_sha = corpus_content_fingerprint(corpus_dir)
    _require(public_sha == expected["questions_sha256"], "public-official.json differs from v0.5007")
    _require(train_sha == expected["train_sha256"], "train.json differs from v0.5007; do not use train_clean.json")
    _require(corpus_sha == expected["corpus_sha256"], "selected-contexts differs from v0.5007")

    dense = _validate_dense_index(dense_index_dir, expected)
    adapter = _validate_adapter(adapter_dir, expected)
    selector_sha = sha256_file(selector_weights)
    _require(selector_sha == expected["selector_sha256"], "Selector YAML differs from v0.5007")
    try:
        selector = yaml.safe_load(selector_weights.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Selector YAML is unreadable: {selector_weights}: {exc}") from exc
    _require(isinstance(selector, dict), "Selector YAML must be an object")
    _require(selector.get("schema") == expected["selector_schema"], "Selector schema is not v0.5007 schema v3")

    return {
        "status": "ok",
        "experiment": "v5007-clean-adapter-a1",
        "expected_run_fingerprint": expected["expected_run_fingerprint"],
        "inputs": {
            "questions_sha256": public_sha,
            "train_sha256": train_sha,
            "corpus_sha256": corpus_sha,
            "dense": {
                "identity": dense["identity"],
                "chunks": dense["chunks"],
                "dimension": dense["dimension"],
            },
            "adapter": adapter,
            "selector_sha256": selector_sha,
            "selector_schema": selector["schema"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the v0.5007 clean-adapter A/B inputs")
    parser.add_argument("--public-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--selector-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_ablation_inputs(
        public_file=args.public_file,
        train_file=args.train_file,
        corpus_dir=args.corpus_dir,
        dense_index_dir=args.dense_index_dir,
        adapter_dir=args.adapter_dir,
        selector_weights=args.selector_weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("V5007_ADAPTER_ABLATION_PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
