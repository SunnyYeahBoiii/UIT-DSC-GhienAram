from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.config import load_config, require_local_model  # noqa: E402
from legalqa.data import load_qa  # noqa: E402
from legalqa.fingerprint import sha256_file  # noqa: E402
from legalqa.generation import clean_answer  # noqa: E402
from legalqa.prompts import SYSTEM_PROMPT  # noqa: E402


class ChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_length,
        evidence_by_id=None,
        max_target_tokens=900,
        max_evidence_chars=5000,
        require_evidence=True,
    ):
        self.items = []
        self.skipped_without_evidence = 0
        evidence_by_id = evidence_by_id or {}
        for row in rows:
            evidence = evidence_by_id.get(row.question_id)
            if require_evidence and not evidence:
                self.skipped_without_evidence += 1
                continue
            if evidence:
                user = (
                    f"CÂU HỎI:\n{row.question}\n\nCĂN CỨ PHÁP LÝ:\n{evidence[:max_evidence_chars]}\n\n"
                    "Viết duy nhất câu trả lời cuối cùng bằng tiếng Việt, chỉ dựa trên căn cứ; "
                    "giữ chính xác điều khoản, điều kiện, ngoại lệ và con số. Không nhắc tới hệ thống truy hồi."
                )
            else:
                user = (
                    "Hãy trả lời câu hỏi pháp luật sau bằng tiếng Việt, giữ chính xác căn cứ, điều kiện và con số.\n"
                    f"Câu hỏi: {row.question}"
                )
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
            prompt_budget = max(1, max_length - len(answer_ids))
            if len(prompt_ids) > prompt_budget:
                # Preserve both system/question at the head and evidence/output
                # instructions at the tail instead of silently dropping the query.
                head = max(1, prompt_budget // 3)
                prompt_ids = prompt_ids[:head] + prompt_ids[-(prompt_budget - head) :]
            full_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            if labels and all(label == -100 for label in labels):
                continue
            self.items.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = max_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "p0.yaml"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-target-tokens", type=int, default=900)
    parser.add_argument("--pairs", default=None, help="retrieval_pairs.jsonl để SFT có căn cứ corpus")
    parser.add_argument("--dev-pairs", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--dev-file", default=None)
    parser.add_argument("--max-evidence-chars", type=int, default=5000)
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none, hoặc đường dẫn checkpoint; Trainer khôi phục model/optimizer/scheduler/RNG.",
    )
    parser.add_argument("--output", default=str(PROJECT / "models" / "qwen-warmup-lora"))
    args = parser.parse_args()
    if args.checkpoint_steps < 1 or args.save_total_limit < 1:
        raise ValueError("checkpoint-steps và save-total-limit phải >= 1.")
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
            raise FileNotFoundError(f"Checkpoint Qwen không hợp lệ: {resume_path}")
        resume_checkpoint = str(resume_path)
    if resume_checkpoint:
        print(f"[Qwen resume] Khôi phục model/optimizer/scheduler/RNG từ {resume_checkpoint}", flush=True)

    model_path = require_local_model(config, "generator")
    kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
    processor = transformers.AutoProcessor.from_pretrained(str(model_path), **kwargs)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    auto_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if auto_class is None:
        raise RuntimeError("Cần Transformers mới có AutoModelForMultimodalLM.")
    model_load_kwargs = dict(kwargs)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit yêu cầu CUDA và bitsandbytes.")
        model_load_kwargs.update(
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map={"": 0},
        )
    else:
        model_load_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else None
    model = auto_class.from_pretrained(str(model_path), **model_load_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Qwen3.5 stores the causal decoder flag on text_config; setting only the
    # outer multimodal config leaves use_cache=True and causes a warning/reset
    # on the first gradient-checkpointed step.
    for cache_config in (model.config, getattr(model.config, "text_config", None)):
        if cache_config is not None and hasattr(cache_config, "use_cache"):
            cache_config.use_cache = False
    model.gradient_checkpointing_enable()
    # Discover text-side linear layers instead of assuming one Qwen architecture.
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

    data_dir = Path(config["paths"]["artifacts"]) / "data"

    def load_evidence(path_value):
        evidence_by_id = {}
        if not path_value:
            return evidence_by_id
        with Path(path_value).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    evidence_items = item.get("positive_evidence") or [item["positive"]]
                    evidence_by_id[str(item["question_id"])] = "\n\n".join(
                        f"[CĂN CỨ {index}]\n{text}" for index, text in enumerate(evidence_items, 1)
                    )
        return evidence_by_id

    evidence_by_id = load_evidence(args.pairs)
    dev_evidence_by_id = load_evidence(args.dev_pairs)
    train_file = Path(args.train_file) if args.train_file else data_dir / "train.json"
    dev_file = Path(args.dev_file) if args.dev_file else data_dir / "dev.json"
    train_rows = load_qa(train_file, True)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    train = ChatDataset(
        train_rows,
        tokenizer,
        args.max_length,
        evidence_by_id,
        args.max_target_tokens,
        args.max_evidence_chars,
        args.require_evidence,
    )
    dev_rows = load_qa(dev_file, True) if dev_file.is_file() else []
    dev = ChatDataset(
        dev_rows,
        tokenizer,
        args.max_length,
        dev_evidence_by_id,
        args.max_target_tokens,
        args.max_evidence_chars,
        args.require_evidence,
    )
    if not train:
        raise RuntimeError(
            "QLoRA không có sample hợp lệ; kiểm tra --train-file/--pairs/--require-evidence."
        )
    run_signature = {
        "train": len(train),
        "dev": len(dev),
        "train_sha256": sha256_file(train_file),
        "dev_sha256": sha256_file(dev_file) if dev_file.is_file() else None,
        "pairs_sha256": sha256_file(args.pairs) if args.pairs else None,
        "dev_pairs_sha256": sha256_file(args.dev_pairs) if args.dev_pairs else None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "max_target_tokens": args.max_target_tokens,
        "lora_r": args.lora_r,
        "load_in_4bit": args.load_in_4bit,
        "max_evidence_chars": args.max_evidence_chars,
        "require_evidence": args.require_evidence,
    }
    if resume_checkpoint:
        previous_signature_path = Path(resume_checkpoint).parent / "resume_signature.json"
        if not previous_signature_path.is_file():
            raise RuntimeError(
                f"Checkpoint thiếu resume_signature.json: {resume_checkpoint}. "
                "Không thể xác nhận dữ liệu/tham số; dùng --resume-from-checkpoint none để train mới."
            )
        previous_signature = json.loads(previous_signature_path.read_text(encoding="utf-8"))
        if previous_signature != run_signature:
            raise RuntimeError(
                "Cấu hình/dữ liệu Qwen khác lần tạo checkpoint. "
                f"checkpoint={previous_signature}; current={run_signature}. "
                "Dùng lại đúng tham số cũ hoặc chọn --resume-from-checkpoint none để train mới."
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
        save_strategy="steps",
        save_steps=args.checkpoint_steps,
        logging_steps=5,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        load_best_model_at_end=False,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev if len(dev) else None,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    (output / "training_complete.json").write_text(
        json.dumps(
            {
                "train": len(train),
                "dev": len(dev),
                "global_step": trainer.state.global_step,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
                "skipped_train_without_evidence": train.skipped_without_evidence,
                "skipped_dev_without_evidence": dev.skipped_without_evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "adapter": str(output.resolve()),
                "train": len(train),
                "dev": len(dev),
                "global_step": trainer.state.global_step,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
