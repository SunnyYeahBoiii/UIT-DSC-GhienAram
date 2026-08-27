import json
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def test_public_v5007_selector_is_frozen() -> None:
    contract = json.loads(
        (PROJECT / "SOURCE_CONTRACT.json").read_text(encoding="utf-8")
    )
    assert contract["release"] == "0.9.2-v5007"

    payload = yaml.safe_load(
        (PROJECT / "configs" / "selector_v5007.yaml").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "selector-weights-v3-agentic-evidence"
    assert payload["retrieval"]["train_memory_candidate_threshold"] == 0.96
    assert payload["selection"]["source_priors"] == {
        "qwen3.5-2b": 0.10860034396087688,
        "qwen3.5-2b-agentic": -0.015121608522814719,
        "extractive": -0.06376520763202459,
        "train-memory": 0.03759912191754071,
    }
    assert "oracle_policy" not in payload["selection"]


def test_public_v5007_runtime_does_not_enable_regressed_v93_features() -> None:
    config = yaml.safe_load(
        (PROJECT / "configs" / "kaggle_inference_t4.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["retrieval"]["agentic_bm25_top_k"] == 60
    assert config["generation"]["agentic_max_tool_calls"] == 2
    assert config["generation"]["agentic_evidence_per_call"] == 3
    assert config["generation"]["agentic_force_first_search"] is True
    assert "style_target_min_ratio" not in config["generation"]
    assert "oracle_policy" not in config["selection"]
    assert config["generation"]["quality_recipe"] == "guarded-v5007-repair-v1"
    assert config["generation"]["selective_repair_enabled"] is True
    assert config["generation"]["repair_max_questions"] == 220
    assert config["runtime"]["reranker_max_windows"] == 3
    assert config["selection"]["document_citation_penalty"] >= 0.40

    notebook = json.loads(
        (PROJECT / "kaggle" / "legalqa_dsc2026_inference_only_t4.ipynb").read_text(
            encoding="utf-8"
        )
    )
    parameters = "".join(notebook["cells"][1]["source"])
    assert "USE_FROZEN_V5007_SELECTOR = True" in parameters
