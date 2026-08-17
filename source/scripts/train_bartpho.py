from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.core.config import load_config, require_local_model  # noqa: E402
from legalqa.core.data import load_qa  # noqa: E402


class QADataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_source=512, max_target=1024):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source = max_source
        self.max_target = max_target

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        source = (
            "trả lời câu hỏi pháp luật bằng tiếng Việt; giữ nguyên căn cứ, điều kiện, ngoại lệ và con số.\n"
            f"câu hỏi: {row.question}\ntrả lời:"
        )
        inputs = self.tokenizer(source, truncation=True, max_length=self.max_source)
        labels = self.tokenizer(text_target=row.answer, truncation=True, max_length=self.max_target)
        inputs["labels"] = labels["input_ids"]
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "p0.yaml"))
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--output", default=str(PROJECT / "models" / "bartpho-warmup"))
    args = parser.parse_args()
    config = load_config(args.config)

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    model_path = require_local_model(config, "bartpho")
    kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), **kwargs)
    data_dir = Path(config["paths"]["artifacts"]) / "data"
    train = load_qa(data_dir / "train.json", require_answers=True)
    dev = load_qa(data_dir / "dev.json", require_answers=True)
    train_dataset = QADataset(train, tokenizer)
    dev_dataset = QADataset(dev, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=int(config["seed"]),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"saved": str(Path(args.output).resolve()), "train": len(train), "dev": len(dev)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

