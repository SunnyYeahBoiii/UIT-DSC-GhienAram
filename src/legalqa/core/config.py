from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình phải là YAML Object: {config_path}")
    config["_config_path"] = str(config_path)
    config["_project_dir"] = str(config_path.parent.parent)
    _resolve_project_paths(config)
    if config.get("runtime", {}).get("offline", True):
        enforce_offline_mode()
    return config


def _resolve_project_paths(config: dict[str, Any]) -> None:
    project = Path(config["_project_dir"])
    paths = config.setdefault("paths", {})
    for name in ("warmup", "corpus", "artifacts"):
        value = paths.get(name)
        if value and not Path(value).is_absolute():
            paths[name] = str((project / value).resolve())
    models = paths.setdefault("models", {})
    for name, value in list(models.items()):
        if value and not Path(value).is_absolute():
            models[name] = str((project / value).resolve())


def enforce_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_artifacts(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["artifacts"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_local_model(config: dict[str, Any], key: str) -> Path:
    raw = config.get("paths", {}).get("models", {}).get(key)
    if not raw:
        raise FileNotFoundError(f"Chưa cấu hình checkpoint local cho mô hình '{key}'.")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy checkpoint local '{key}': {path}. "
            "Hãy tải checkpoint đã được BTC duyệt trước khi bật chế độ offline."
        )
    return path

