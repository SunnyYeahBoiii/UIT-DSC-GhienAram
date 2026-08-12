from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import traceback
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
from typing import Any, Callable

from packaging.specifiers import SpecifierSet
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version


MODEL_ALIASES = {
    "generator": ("qwen3.5-2b", "qwen35-2b", "qwen352b"),
    "embedding": ("vietnamese_embedding", "vietnamese-embedding", "vietnameseembedding"),
    "reranker": ("vietnamese_reranker", "vietnamese-reranker", "vietnamesereranker"),
    "bartpho": ("bartpho-syllable", "bartphosyllable"),
    "verifier": ("vilegalbert", "vi-legal-bert"),
}

EXPECTED_MODEL_TYPES = {
    "generator": "qwen3_5",
    "embedding": "xlm-roberta",
    "reranker": "xlm-roberta",
    "bartpho": "mbart",
    "verifier": "roberta",
}

# The exact Hugging Face layer was exercised together. Kaggle owns the large
# CUDA/numeric packages, so those use bounded ranges plus runtime smoke tests.
VERSION_SPECS = {
    "transformers": "==5.14.1",
    "huggingface-hub": "==1.26.1",
    "tokenizers": "==0.22.2",
    "regex": "==2026.7.19",
    "hf-xet": "==1.6.0",
    "click": "==8.4.2",
    "accelerate": "==1.14.0",
    "safetensors": "==0.8.0",
    "sentence-transformers": "==5.7.0",
    "peft": "==0.18.1",
    "bitsandbytes": "==0.49.2",
    "sentencepiece": "==0.2.1",
    "torch": ">=2.6,<3",
    "torchvision": ">=0.21,<1",
    "numpy": ">=1.26,<3",
    "scipy": ">=1,<2",
    "scikit-learn": ">=1,<2",
    "PyYAML": ">=6,<7",
    "Pillow": ">=9.5,<13",
    "packaging": ">=23,<27",
}

DEPENDENCY_ROOTS = (
    "transformers",
    "huggingface-hub",
    "accelerate",
    "sentence-transformers",
    "peft",
    "bitsandbytes",
)


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_model(root: Path, role: str) -> Path | None:
    environment = os.environ.get(f"LEGALQA_{role.upper()}_DIR")
    if environment:
        path = Path(environment)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint {role} không có config.json: {path}")
        return path
    aliases = tuple(normalized(alias) for alias in MODEL_ALIASES[role])
    candidates = []
    for config_path in root.rglob("config.json"):
        parent = config_path.parent
        if any(alias in normalized(str(parent)) for alias in aliases):
            candidates.append(parent)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), len(str(path)), str(path)))
    exact = [path for path in candidates if normalized(path.name) in aliases]
    preferred = exact or candidates
    if len(preferred) > 1:
        variable = f"LEGALQA_{role.upper()}_DIR"
        raise RuntimeError(
            f"Có nhiều checkpoint phù hợp cho {role}: {preferred}. "
            f"Chỉ attach một bản hoặc đặt biến {variable}."
        )
    return preferred[0]


def package_versions() -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Wheelhouse này dành cho Python 3.12, nhưng kernel là {sys.version.split()[0]}. "
            "Hãy dùng Kaggle image Python 3.12 hoặc tạo lại toàn bộ binary wheels."
        )
    installed: dict[str, str] = {}
    errors = []
    for package, spec in VERSION_SPECS.items():
        try:
            found = version(package)
        except PackageNotFoundError:
            errors.append(f"thiếu {package}{spec}")
            continue
        installed[package] = found
        if Version(found) not in SpecifierSet(spec):
            errors.append(f"{package}=={found}, cần {package}{spec}")
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    for package in DEPENDENCY_ROOTS:
        try:
            package_requirements = requires(package) or []
        except PackageNotFoundError:
            continue
        for raw_requirement in package_requirements:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(marker_environment):
                continue
            try:
                dependency_version = version(requirement.name)
            except PackageNotFoundError:
                errors.append(f"{package} yêu cầu nhưng môi trường thiếu {requirement}")
                continue
            if requirement.specifier and Version(dependency_version) not in requirement.specifier:
                errors.append(
                    f"{package} yêu cầu {requirement}, nhưng có {requirement.name}=={dependency_version}"
                )
    if errors:
        raise RuntimeError("; ".join(errors))
    return installed


def cuda_runtime(require_nf4: bool = True) -> dict[str, Any]:
    import torch
    import torchvision
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle Notebook chưa bật GPU Accelerator.")
    capability = torch.cuda.get_device_capability(0)
    if require_nf4 and capability < (6, 0):
        raise RuntimeError(f"NF4 yêu cầu compute capability >= 6.0, GPU hiện tại là {capability}.")

    # A mismatched torch/torchvision pair commonly imports but fails here.
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]])
    scores = torch.tensor([1.0, 0.5])
    kept = nms(boxes, scores, 0.5)
    if kept.numel() == 0:
        raise RuntimeError("torchvision C++ operators không hoạt động.")

    if require_nf4:
        import bitsandbytes as bnb

        # Exercise the actual NF4 CUDA binary, not just `import bitsandbytes`.
        layer = bnb.nn.Linear4bit(
            16,
            8,
            bias=False,
            compute_dtype=torch.float16,
            quant_type="nf4",
            quant_storage=torch.uint8,
        ).cuda()
        inputs = torch.randn(2, 16, device="cuda", dtype=torch.float16)
        outputs = layer(inputs)
        if outputs.shape != (2, 8) or not torch.isfinite(outputs).all():
            raise RuntimeError("bitsandbytes NF4 CUDA smoke test trả về kết quả không hợp lệ.")
        del layer, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "memory_gib": round(properties.total_memory / 1024**3, 2),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "nf4_kernel": True if require_nf4 else "skipped_inference_only",
        "torchvision_nms": True,
    }


def training_apis() -> dict[str, Any]:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "qwen3_5" not in CONFIG_MAPPING:
        raise RuntimeError("Transformers không đăng ký model_type qwen3_5.")
    if not hasattr(transformers, "AutoModelForMultimodalLM"):
        raise RuntimeError("Transformers thiếu AutoModelForMultimodalLM.")
    if not callable(prepare_model_for_kbit_training):
        raise RuntimeError("PEFT thiếu prepare_model_for_kbit_training.")

    # Detect PEFT/Transformers API drift without loading the 2B checkpoint.
    tiny_config = transformers.BertConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=100,
    )
    tiny = transformers.BertForSequenceClassification(tiny_config)
    tiny = get_peft_model(
        tiny,
        LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"]),
    )
    if not any("lora_" in name for name, _ in tiny.named_parameters()):
        raise RuntimeError("PEFT không gắn được LoRA vào mô hình thử.")
    del tiny
    with tempfile.TemporaryDirectory(prefix="legalqa-preflight-") as output_dir:
        training_arguments = transformers.TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=20,
            save_only_model=False,
            restore_callback_states_from_checkpoint=True,
            report_to="none",
            fp16=False,
            bf16=False,
        )
        if training_arguments.save_only_model:
            raise RuntimeError("Trainer đang cấu hình bỏ optimizer/scheduler khỏi checkpoint.")
    return {
        "qwen3_5_registered": True,
        "auto_multimodal": True,
        "peft_lora": True,
        "trainer_accelerate": True,
        "full_state_checkpoint": True,
    }


def optional_kernels() -> dict[str, Any]:
    available = {
        "fla": find_spec("fla") is not None,
        "causal_conv1d": find_spec("causal_conv1d") is not None,
    }
    return {
        **available,
        "qwen_fast_path": all(available.values()),
        "required": False,
        "fallback": "Transformers PyTorch implementation",
    }


def checkpoint_weight_files(path: Path) -> list[Path]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        if not names:
            raise RuntimeError(f"Safetensors index rỗng: {index_path}")
        missing = [name for name in names if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Checkpoint thiếu shards: {missing}")
        return [path / name for name in names]
    safetensors_files = sorted(path.glob("model*.safetensors"))
    if safetensors_files:
        return safetensors_files
    pytorch_files = sorted(path.glob("pytorch_model*.bin"))
    if pytorch_files:
        return pytorch_files
    raise FileNotFoundError(f"Không tìm thấy model weights trong {path}")


def validate_checkpoint(role: str, path: Path) -> dict[str, Any]:
    import transformers
    from accelerate import init_empty_weights
    from safetensors import safe_open

    config = transformers.AutoConfig.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )
    expected = EXPECTED_MODEL_TYPES[role]
    if config.model_type != expected:
        raise RuntimeError(f"{role}: model_type={config.model_type!r}, cần {expected!r}: {path}")
    weights = checkpoint_weight_files(path)
    for weight in weights:
        if weight.stat().st_size <= 0:
            raise RuntimeError(f"Weight file rỗng: {weight}")
        if weight.suffix == ".safetensors":
            with safe_open(str(weight), framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise RuntimeError(f"Safetensors không có tensor: {weight}")

    kwargs = {"local_files_only": True, "trust_remote_code": True}
    if role == "generator":
        processor = transformers.AutoProcessor.from_pretrained(str(path), **kwargs)
        messages = [
            {"role": "system", "content": "Bạn là trợ lý pháp luật."},
            {"role": "user", "content": "Câu hỏi kiểm tra."},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = processor(text=[prompt], padding=True, return_tensors="pt")
        if "input_ids" not in encoded or encoded["input_ids"].numel() == 0:
            raise RuntimeError("Qwen processor không tạo được input_ids cho văn bản.")
        auto_class = transformers.AutoModelForMultimodalLM
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), **kwargs)
        encoded = tokenizer("Kiểm tra văn bản pháp luật.", return_tensors="pt")
        if "input_ids" not in encoded or encoded["input_ids"].numel() == 0:
            raise RuntimeError(f"{role} tokenizer không tạo được input_ids.")
        auto_class = {
            "embedding": transformers.AutoModel,
            "reranker": transformers.AutoModelForSequenceClassification,
            "bartpho": transformers.AutoModelForSeq2SeqLM,
            "verifier": transformers.AutoModel,
        }[role]

    # Build on meta device to prove that this Transformers version maps each
    # config to a real architecture without allocating checkpoint weights.
    with init_empty_weights():
        empty_model = auto_class.from_config(config, trust_remote_code=True)
    architecture = type(empty_model).__name__
    gradient_checkpointing = bool(getattr(empty_model, "supports_gradient_checkpointing", False))
    if role == "reranker" and not gradient_checkpointing:
        raise RuntimeError(f"{role}: architecture không hỗ trợ gradient checkpointing: {architecture}")
    del empty_model
    gc.collect()
    return {
        "path": str(path.resolve()),
        "model_type": config.model_type,
        "architecture": architecture,
        "weight_files": len(weights),
        "weight_gib": round(sum(item.stat().st_size for item in weights) / 1024**3, 3),
        "tokenizer_or_processor": True,
        "gradient_checkpointing": gradient_checkpointing,
    }


def checkpoint_apis(
    input_root: Path, profile: str, *, inference_only: bool = False
) -> dict[str, Any]:
    roles = tuple(
        role for role in MODEL_ALIASES
        if not (inference_only and role == "bartpho")
    )
    discovered = {role: find_model(input_root, role) for role in roles}
    required = {"generator"}
    if profile in {"balanced", "full"}:
        required |= {"embedding", "reranker", "verifier"}
    if profile == "full" and not inference_only:
        required.add("bartpho")
    missing = sorted(role for role in required if discovered[role] is None)
    if missing:
        raise FileNotFoundError(f"Thiếu checkpoint cho profile {profile}: {missing}")
    # Validate every attached approved checkpoint now, including models that a
    # later profile will use, so version failures are not deferred.
    return {
        role: validate_checkpoint(role, path)
        for role, path in discovered.items()
        if path is not None
    }


def run_check(
    report: dict[str, Any], errors: list[dict[str, str]], name: str, check: Callable[[], Any]
) -> None:
    try:
        report[name] = check()
        print(f"[Preflight] {name}: OK", flush=True)
    except Exception as exc:
        report[name] = {"ok": False, "error": repr(exc)}
        errors.append(
            {
                "check": name,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        print(f"[Preflight] {name}: FAILED — {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast Kaggle compatibility audit")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--profile", choices=("fast", "balanced", "full"), default="fast")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--output", default="/kaggle/working/legalqa_preflight.json")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    report: dict[str, Any] = {
        "ok": False,
        "python": sys.version,
        "profile": args.profile,
        "inference_only": args.inference_only,
        "input_root": str(Path(args.input_root).resolve()),
    }
    errors: list[dict[str, str]] = []
    run_check(report, errors, "versions", package_versions)
    run_check(report, errors, "cuda_runtime", lambda: cuda_runtime(require_nf4=not args.inference_only))
    if args.inference_only:
        report["training_apis"] = {"skipped": True, "reason": "inference-only mode"}
        print("[Preflight] training_apis: SKIPPED (inference-only)", flush=True)
    else:
        run_check(report, errors, "training_apis", training_apis)
    run_check(report, errors, "optional_kernels", optional_kernels)
    run_check(
        report,
        errors,
        "checkpoints",
        lambda: checkpoint_apis(
            Path(args.input_root), args.profile, inference_only=args.inference_only
        ),
    )
    report["errors"] = errors
    report["ok"] = not errors
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Preflight] report: {output}", flush=True)
    if errors:
        raise SystemExit(
            "Preflight thất bại; pipeline chưa tạo index hoặc chạy model. Xem errors trong " + str(output)
        )
    print("[Preflight] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
