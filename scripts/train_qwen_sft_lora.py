from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legalqa.core.config import load_config, require_local_model  # noqa: E402
from legalqa.core.data import Chunk, load_qa  # noqa: E402
from legalqa.core.fingerprint import sha256_file  # noqa: E402
from legalqa.generation.generation import clean_answer  # noqa: E402
from legalqa.generation.prompts import SYSTEM_PROMPT, qwen_user_prompt  # noqa: E402
from legalqa.retrieval.retrieval import Evidence  # noqa: E402
from train_qwen_lora import CausalCollator  # noqa: E402


def load_evidence_by_id(path_value: str | None) -> dict[str, list[Evidence]]:
    """Đọc sft_evidence.jsonl (sinh bởi prepare_sft_evidence.py) và dựng lại Evidence/Chunk thật."""
    evidence_by_id: dict[str, list[Evidence]] = {}
    if not path_value:
        return evidence_by_id
    with Path(path_value).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            evidence_by_id[str(item["question_id"])] = [
                Evidence(
                    chunk=Chunk(**entry["chunk"]),
                    score=entry["score"],
                    bm25_score=entry["bm25_score"],
                    dense_score=entry["dense_score"],
                    reranker_score=entry["reranker_score"],
                    stage=entry["stage"],
                    relation_type=entry["relation_type"],
                    relation_path=entry["relation_path"],
                )
                for entry in item.get("evidence", [])
            ]
    return evidence_by_id


class RagChatDataset(torch.utils.data.Dataset):
    """SFT dataset dùng đúng qwen_user_prompt() thật (RAG evidence thay vì câu hỏi trần)."""

    def __init__(
        self,
        rows,
        tokenizer,
        max_length,
        evidence_by_id,
        max_context_chars,
        max_target_tokens,
        require_evidence,
        config,
    ):
        self.items = []
        self.skipped_without_evidence = 0
        self.skipped_over_max_length = 0
        sft_config = copy.deepcopy(config)
        sft_config.setdefault("generation", {})["max_context_chars"] = max_context_chars
        for row in rows:
            evidence = evidence_by_id.get(row.question_id, [])
            if require_evidence and not evidence:
                self.skipped_without_evidence += 1
                continue
            user = qwen_user_prompt(row.question, evidence, sft_config, retrieval_diagnostics=None)
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            target_answer = clean_answer(row.answer, question=row.question)
            full_messages = prompt_messages + [{"role": "assistant", "content": target_answer}]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids_untrimmed = tokenizer(full, add_special_tokens=False)["input_ids"]
            answer_ids = full_ids_untrimmed[len(prompt_ids) :][:max_target_tokens]
            if len(prompt_ids) + len(answer_ids) > max_length:
                # max_context_chars đã giới hạn evidence theo ngân sách ký tự cấu hình sẵn (render_evidence
                # tự cắt theo từng đoạn); nếu vẫn vượt max_length thì bỏ câu này thay vì cắt thô giữa prompt
                # (cắt thô có thể xoá mất câu hỏi hoặc phần YÊU CẦU ĐẦU RA).
                self.skipped_over_max_length += 1
                continue
            full_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            if labels and all(label == -100 for label in labels):
                continue
            self.items.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _EpochRandomSubsetSampler(torch.utils.data.Sampler):
    """Rút subset_size mẫu random khác nhau mỗi epoch từ pool lớn hơn (giống mini-batch SGD).

    __iter__ được HF Trainer gọi lại (fresh) mỗi epoch khi lặp `for step, inputs in
    enumerate(train_dataloader)` — tăng self.epoch mỗi lần để seed đổi, tránh lặp y hệt.
    """

    def __init__(self, data_len: int, subset_size: int, seed: int):
        self.data_len = data_len
        self.subset_size = min(subset_size, data_len)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        indices = torch.randperm(self.data_len, generator=generator)[: self.subset_size]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.subset_size


def _make_epoch_resample_trainer(transformers_module, epoch_subset_size: int):
    """Trả về subclass Trainer override _get_train_sampler() để mỗi epoch dùng subset random khác
    nhau thay vì RandomSampler mặc định (vẫn full permutation của cùng 1 tập cố định mọi epoch)."""

    class EpochResampleTrainer(transformers_module.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return _EpochRandomSubsetSampler(len(self.train_dataset), epoch_subset_size, self.args.seed)

    return EpochResampleTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT LoRA thuần cho Qwen3.5-2B (không evidence), lưu checkpoint mỗi epoch "
        "để chọn theo METEOR dev bằng scripts/select_best_sft_checkpoint.py."
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", default=None)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-target-tokens", type=int, default=700)
    parser.add_argument(
        "--evidence", default=None, help="sft_evidence.jsonl sinh bởi scripts/prepare_sft_evidence.py (RAG thật)"
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3500,
        help="ngân sách ký tự evidence RIÊNG cho SFT (nhỏ hơn nhiều so với max_context_chars lúc inference "
        "vì --max-length trên T4 có hạn); đo token thật trước khi đổi số này.",
    )
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bỏ câu hỏi không có evidence thay vì dạy model trả lời từ hư không",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("transformers", "unsloth"),
        default="transformers",
        help="unsloth cần internet lúc chạy (pip install unsloth trước) và nhanh/tiết kiệm VRAM hơn "
        "transformers+peft+bitsandbytes thuần; --load-in-4bit vẫn áp dụng cho backend unsloth.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--train-pool-samples",
        type=int,
        default=None,
        help="pool lớn hơn --max-train-samples để mỗi epoch rút random subset khác nhau (kiểu SGD "
        "mini-batch); mặc định = --max-train-samples, tức mọi epoch dùng đúng cùng bộ mẫu như cũ.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none, hoặc đường dẫn checkpoint-N cụ thể trong --output",
    )
    parser.add_argument("--output", default=str(PROJECT / "models" / "qwen-sft-lora"))
    args = parser.parse_args()
    config = load_config(args.config)

    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.trainer_utils import get_last_checkpoint

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resume_value = str(args.resume_from_checkpoint).strip()
    if resume_value.lower() in {"none", "false", "off"}:
        resume_checkpoint = None
    elif resume_value.lower() == "auto":
        resume_checkpoint = get_last_checkpoint(str(output))
    else:
        resume_path = Path(resume_value)
        if not resume_path.is_dir() or not (resume_path / "trainer_state.json").is_file():
            raise FileNotFoundError(f"Checkpoint SFT không hợp lệ: {resume_path}")
        resume_checkpoint = str(resume_path)
    if resume_checkpoint:
        print(f"[SFT resume] Khôi phục model/optimizer/scheduler/RNG từ {resume_checkpoint}", flush=True)

    model_path = require_local_model(config, "generator")
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if args.backend == "unsloth":
        # unsloth khuyến cáo import trước transformers để patch kernel; ở đây torch/transformers
        # đã bị import từ module-level (train_qwen_lora.CausalCollator) nên patch không trọn vẹn
        # 100%, nhưng FastLanguageModel vẫn áp fused kernel + 4bit lúc load model bên dưới.
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=args.max_length,
            dtype=compute_dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=config["runtime"].get("trust_remote_code", True),
        )
        # checkpoint đa phương thức nên from_pretrained trả về cả AutoProcessor (có
        # .image_processor); __call__ đã bị unsloth patch để hiểu arg đầu là ảnh, nên phải lấy
        # đúng sub-tokenizer text thuần để gọi tokenizer(prompt, ...) như phần dataset bên dưới.
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # unsloth's get_peft_regex() fullmatch tên module theo dạng .*?<tag>.*?<tag>.*?<leaf>,
        # nên target_modules phải là TÊN LEAF (vd "out_proj"), không phải full dotted path
        # (full path thì fullmatch không bao giờ khớp -> "No layers to finetune").
        target_modules = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
                and not any(blocked in name.lower() for blocked in ("visual", "vision", "merger", "lm_head"))
            }
        )
        if not target_modules:
            raise RuntimeError("Không tìm thấy linear layer phía ngôn ngữ để gắn LoRA.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=int(config["seed"]),
        )
        model.print_trainable_parameters()
    else:
        kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
        processor = transformers.AutoProcessor.from_pretrained(str(model_path), **kwargs)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        auto_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if auto_class is None:
            raise RuntimeError("Cần Transformers mới có AutoModelForMultimodalLM.")
        model_load_kwargs = dict(kwargs)
        if args.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("--load-in-4bit yêu cầu CUDA và bitsandbytes.")
            # device_map="auto" chia layer qua tất cả GPU thấy được (vd Kaggle T4x2) để tăng VRAM
            # khả dụng cho backward pass; máy 1 GPU thì "auto" cũng chỉ dồn hết về gpu0 như cũ.
            model_load_kwargs.update(
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
                device_map="auto",
            )
        else:
            model_load_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else None
        model = auto_class.from_pretrained(str(model_path), **model_load_kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        for cache_config in (model.config, getattr(model.config, "text_config", None)):
            if cache_config is not None and hasattr(cache_config, "use_cache"):
                cache_config.use_cache = False
        model.gradient_checkpointing_enable()
        target_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and not any(blocked in name.lower() for blocked in ("visual", "vision", "merger", "lm_head"))
        ]
        if not target_modules:
            raise RuntimeError("Không tìm thấy linear layer phía ngôn ngữ để gắn LoRA.")
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    evidence_by_id = load_evidence_by_id(args.evidence)
    train_file = Path(args.train_file)
    train_rows = load_qa(train_file, True)
    pool_size = args.train_pool_samples or args.max_train_samples
    if pool_size:
        train_rows = train_rows[:pool_size]
    train = RagChatDataset(
        train_rows,
        tokenizer,
        args.max_length,
        evidence_by_id,
        args.max_context_chars,
        args.max_target_tokens,
        args.require_evidence,
        config,
    )
    dev_file = Path(args.dev_file) if args.dev_file else None
    dev_rows = load_qa(dev_file, True) if dev_file and dev_file.is_file() else []
    if args.max_eval_samples:
        dev_rows = dev_rows[: args.max_eval_samples]
    dev = RagChatDataset(
        dev_rows,
        tokenizer,
        args.max_length,
        evidence_by_id,
        args.max_context_chars,
        args.max_target_tokens,
        args.require_evidence,
        config,
    )
    print(
        f"[SFT evidence] train bỏ qua: thiếu evidence={train.skipped_without_evidence}, "
        f"vượt max_length={train.skipped_over_max_length} | "
        f"dev bỏ qua: thiếu evidence={dev.skipped_without_evidence}, "
        f"vượt max_length={dev.skipped_over_max_length}",
        flush=True,
    )
    if not train:
        raise RuntimeError("SFT không có sample hợp lệ; kiểm tra --train-file/--evidence/--require-evidence.")

    # subset random khác nhau mỗi epoch chỉ bật khi pool > max_train_samples; nếu không set
    # --train-pool-samples thì pool_size == max_train_samples, train == đúng subset, không random.
    epoch_subset_size = (
        args.max_train_samples if args.max_train_samples and args.max_train_samples < len(train) else None
    )

    run_signature = {
        "train": len(train),
        "epoch_subset_size": epoch_subset_size,
        "dev": len(dev),
        "train_sha256": sha256_file(train_file),
        "dev_sha256": sha256_file(dev_file) if dev_file and dev_file.is_file() else None,
        "evidence_sha256": sha256_file(args.evidence) if args.evidence else None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "max_target_tokens": args.max_target_tokens,
        "max_context_chars": args.max_context_chars,
        "require_evidence": args.require_evidence,
        "lora_r": args.lora_r,
        "load_in_4bit": args.load_in_4bit,
        "backend": args.backend,
    }
    if resume_checkpoint:
        previous_signature_path = Path(resume_checkpoint).parent / "resume_signature.json"
        if not previous_signature_path.is_file():
            raise RuntimeError(
                f"Checkpoint thiếu resume_signature.json: {resume_checkpoint}. "
                "Dùng --resume-from-checkpoint none để train mới."
            )
        previous_signature = json.loads(previous_signature_path.read_text(encoding="utf-8"))
        if previous_signature != run_signature:
            raise RuntimeError(
                "Cấu hình/dữ liệu SFT khác lần tạo checkpoint. "
                f"checkpoint={previous_signature}; current={run_signature}. "
                "Dùng lại đúng tham số cũ hoặc chọn --resume-from-checkpoint none."
            )
    (output / "resume_signature.json").write_text(
        json.dumps(run_signature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    training_args = transformers.TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch" if len(dev) else "no",
        save_strategy="epoch",
        # giữ checkpoint MỌI epoch (không giới hạn) để chọn theo METEOR dev sau training,
        # vì eval_loss trong Trainer không phản ánh trực tiếp METEOR trên câu sinh ra.
        save_total_limit=None,
        logging_steps=5,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        load_best_model_at_end=False,
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )
    if epoch_subset_size:
        trainer = _make_epoch_resample_trainer(transformers, epoch_subset_size)(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=dev if len(dev) else None,
            data_collator=CausalCollator(tokenizer),
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=dev if len(dev) else None,
            data_collator=CausalCollator(tokenizer),
        )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    epoch_checkpoints = sorted(
        (path for path in output.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    manifest = {
        "output": str(output.resolve()),
        "epoch_checkpoints": [str(path) for path in epoch_checkpoints],
        "run_signature": run_signature,
    }
    (output / "sft_training_complete.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
