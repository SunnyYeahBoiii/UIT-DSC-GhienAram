from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import require_local_model


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    terminated_by_eos: bool
    truncated: bool


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Thiếu PyTorch. Hãy cài requirements.txt.") from exc
    return torch


def resolve_device(config: dict[str, Any]) -> str:
    torch = _torch()
    requested = str(config["runtime"].get("device", "auto"))
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(config: dict[str, Any]):
    torch = _torch()
    name = str(config["runtime"].get("dtype", "bfloat16"))
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"float16", "fp16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def unload_model(instance: Any, stage: str | None = None) -> None:
    if instance is None:
        gc.collect()
        return
    torch = _torch()
    for name in ("model", "tokenizer", "processor"):
        if hasattr(instance, name):
            setattr(instance, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        if stage:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"[VRAM after {stage}] allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB",
                flush=True,
            )


class DenseEncoder:
    def __init__(self, config: dict[str, Any]) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.path = require_local_model(config, "embedding")
        self.device = resolve_device(config)
        self.model = SentenceTransformer(
            str(self.path),
            device=self.device,
            trust_remote_code=bool(config["runtime"].get("trust_remote_code", True)),
            local_files_only=bool(config["runtime"].get("local_files_only", True)),
        )
        torch = _torch()
        if self.device.startswith("cuda") and resolve_dtype(config) == torch.float16:
            self.model.half()

    def encode(self, texts: Iterable[str], batch_size: int | None = None) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            items,
            batch_size=batch_size or int(
                self.config["runtime"].get(
                    "embedding_batch_size", self.config["runtime"].get("batch_size", 4)
                )
            ),
            show_progress_bar=len(items) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, config: dict[str, Any]) -> None:
        import transformers

        torch = _torch()
        self.config = config
        self.path = require_local_model(config, "reranker")
        self.device = resolve_device(config)
        kwargs = {
            "local_files_only": bool(config["runtime"].get("local_files_only", True)),
            "trust_remote_code": bool(config["runtime"].get("trust_remote_code", True)),
        }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), **kwargs)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(self.path), dtype=resolve_dtype(config), **kwargs
        )
        self.model.to(self.device).eval()
        self.torch = torch

    def score(self, question: str, passages: list[str]) -> list[float]:
        return self.score_pairs([(question, passage) for passage in passages])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores: list[float] = []
        batch_size = int(
            self.config["runtime"].get(
                "reranker_batch_size", self.config["runtime"].get("batch_size", 4)
            )
        )
        max_length = int(self.config["runtime"].get("reranker_max_length", 1536))
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.float()
            if logits.ndim == 2 and logits.shape[1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)
            scores.extend(float(value) for value in batch_scores.cpu())
        return scores


class QwenTextGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        import transformers

        torch = _torch()
        self.config = config
        self.path = require_local_model(config, "generator")
        self.device = resolve_device(config)
        kwargs = {
            "local_files_only": bool(config["runtime"].get("local_files_only", True)),
            "trust_remote_code": bool(config["runtime"].get("trust_remote_code", True)),
        }
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.path), **kwargs)
        auto_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if auto_class is None:
            raise RuntimeError(
                "Transformers hiện tại chưa có AutoModelForMultimodalLM. "
                "Hãy cài bản Transformers mới theo requirements.txt."
            )
        adapter_path = config.get("paths", {}).get("models", {}).get("generator_adapter")
        model_kwargs = {**kwargs}
        # LoRA áp được lên base ở precision bất kỳ; base 2B (~4.5GB BF16) không
        # cần quantize khi infer — NF4 chỉ thêm nhiễu, load theo dtype config.
        model_kwargs["dtype"] = resolve_dtype(config)
        if self.device.startswith("cuda"):
            device_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            model_kwargs["device_map"] = {"": device_index}
        self.model = auto_class.from_pretrained(str(self.path), **model_kwargs).eval()
        if not self.device.startswith("cuda"):
            self.model.to(self.device)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Cấu hình generator_adapter yêu cầu cài peft.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False).eval()
        self.torch = torch

    def generate_with_metadata(
        self,
        system: str,
        user: str,
        *,
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
    ) -> GenerationResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            encoded = self.processor.apply_chat_template(
                messages,
                **template_kwargs,
            )
        except TypeError:
            # Compatibility fallback for processors whose template wrapper does
            # not yet expose enable_thinking/return_dict.
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.processor(text=[prompt], padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items() if hasattr(value, "to")}
        cfg = self.config["generation"]
        token_limit = int(max_new_tokens or cfg["max_new_tokens"])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": token_limit,
            "min_new_tokens": max(0, int(min_new_tokens)),
            "do_sample": bool(cfg.get("do_sample", False)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
            "no_repeat_ngram_size": int(cfg.get("no_repeat_ngram_size", 5)),
            "pad_token_id": getattr(self.processor.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.processor.tokenizer, "eos_token_id", None),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs.update(
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.9)),
            )
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        answer_ids = output[0, prompt_length:]
        eos_value = generation_kwargs.get("eos_token_id")
        eos_ids = (
            {int(value) for value in eos_value}
            if isinstance(eos_value, (list, tuple, set))
            else {int(eos_value)} if eos_value is not None else set()
        )
        generated_tokens = int(answer_ids.numel())
        terminated_by_eos = bool(
            generated_tokens and eos_ids and int(answer_ids[-1].item()) in eos_ids
        )
        truncated = generated_tokens >= token_limit and not terminated_by_eos
        text = self.processor.decode(answer_ids, skip_special_tokens=True).strip()
        return GenerationResult(text, generated_tokens, terminated_by_eos, truncated)

    def generate(self, system: str, user: str) -> str:
        return self.generate_with_metadata(system, user).text


class LegalSemanticEncoder:
    """ViLegalBERT encoder used only for local candidate/evidence consistency."""

    def __init__(self, config: dict[str, Any]) -> None:
        import transformers

        torch = _torch()
        self.config = config
        self.path = require_local_model(config, "verifier")
        self.device = resolve_device(config)
        kwargs = {
            "local_files_only": bool(config["runtime"].get("local_files_only", True)),
            "trust_remote_code": bool(config["runtime"].get("trust_remote_code", True)),
        }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path), **kwargs)
        self.model = transformers.AutoModel.from_pretrained(
            str(self.path), dtype=resolve_dtype(config), **kwargs
        ).to(self.device).eval()
        self.torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        batch_size = int(
            self.config["runtime"].get(
                "verifier_batch_size", self.config["runtime"].get("batch_size", 4)
            )
        )
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state.float()
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)

    def similarity(self, answer: str, evidence: str) -> float:
        vectors = self.encode([answer, evidence])
        if len(vectors) != 2:
            return 0.0
        cosine = float(np.dot(vectors[0], vectors[1]))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
