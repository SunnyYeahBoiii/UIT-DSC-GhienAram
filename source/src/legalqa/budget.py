from __future__ import annotations

from pathlib import Path
from typing import Any


ROLE_TO_AUTO_CLASS = {
    "generator": "AutoModelForMultimodalLM",
    "embedding": "AutoModel",
    "reranker": "AutoModelForSequenceClassification",
    "bartpho": "AutoModelForSeq2SeqLM",
    "verifier": "AutoModel",
}


def _count_adapter(path: Path) -> int:
    from safetensors import safe_open

    total = 0
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count = 1
                for dimension in shape:
                    count *= dimension
                total += count
    return total


def _count_safetensors(path: Path) -> int:
    from safetensors import safe_open

    total = 0
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                count = 1
                for dimension in handle.get_slice(key).get_shape():
                    count *= dimension
                total += count
    return total


def _count_pytorch_bin(path: Path) -> int:
    """Count unique serialized tensor storage, excluding tied-weight aliases."""
    import torch

    total = 0
    for file in sorted(path.glob("*.bin")):
        # Ignore optimizer/training metadata such as training_args.bin.
        if file.name not in {"pytorch_model.bin"} and not file.name.startswith("pytorch_model-"):
            continue
        kwargs = {"map_location": "cpu", "weights_only": True}
        try:
            state = torch.load(file, mmap=True, **kwargs)
        except (TypeError, RuntimeError):
            state = torch.load(file, **kwargs)
        storages: dict[tuple[int, int], tuple[int, int]] = {}
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            signature = (storage.data_ptr(), storage.nbytes())
            # A storage is expected to have one dtype; retain the largest view defensively.
            elements = storage.nbytes() // max(1, value.element_size())
            previous = storages.get(signature)
            if previous is None or elements > previous[0]:
                storages[signature] = (elements, value.element_size())
        total += sum(elements for elements, _ in storages.values())
    return total


def count_checkpoint_tensors(path: Path) -> int:
    safetensors = _count_safetensors(path)
    if safetensors:
        return safetensors
    return _count_pytorch_bin(path)


def count_architecture(path: str | Path, role: str, trust_remote_code: bool = True) -> int:
    import transformers
    from accelerate import init_empty_weights

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    class_name = ROLE_TO_AUTO_CLASS[role]
    auto_class = getattr(transformers, class_name, None)
    if auto_class is None:
        raise RuntimeError(
            f"Transformers chưa có {class_name}; hãy cài phiên bản trong requirements.txt."
        )
    model_config = transformers.AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=trust_remote_code
    )
    with init_empty_weights(include_buffers=False):
        model = auto_class.from_config(model_config, trust_remote_code=trust_remote_code)
    # init_empty_weights can prevent constructors from tying shared embeddings.
    # Tie them explicitly, then conservatively include any tensors serialized in
    # the checkpoint but not represented by the selected AutoModel head.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    architecture = sum(parameter.numel() for parameter in model.parameters())
    serialized = count_checkpoint_tensors(model_path)
    return max(architecture, serialized)


def count_system_budget(config: dict[str, Any]) -> dict[str, Any]:
    trust = bool(config["runtime"].get("trust_remote_code", True))
    models = config.get("paths", {}).get("models", {})
    components: dict[str, int] = {}
    for role in ROLE_TO_AUTO_CLASS:
        path = models.get(role)
        if path:
            components[role] = count_architecture(path, role, trust)
    adapter_path = models.get("generator_adapter")
    if adapter_path:
        components["generator_adapter"] = _count_adapter(Path(adapter_path))
    total = sum(components.values())
    return {
        "components": components,
        "total_parameters": total,
        "total_billions": total / 1_000_000_000,
        "strictly_below_4b": total < 4_000_000_000,
        "headroom": 4_000_000_000 - total,
        "counting_method": (
            "max(tied registered architecture, unique serialized checkpoint tensors); "
            "LoRA adapter tensors added separately"
        ),
    }
