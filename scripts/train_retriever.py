from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.core.config import load_config, require_local_model  # noqa: E402


def read_pairs(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def batches(rows, batch_size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start : start + batch_size]]


def main():
    parser = argparse.ArgumentParser(description="In-batch contrastive tuning using official weak labels")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "p0.yaml"))
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--output", default=str(PROJECT / "models" / "Vietnamese_Embedding-warmup"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()
    config = load_config(args.config)
    pairs_path = args.pairs or str(Path(config["paths"]["artifacts"]) / "data" / "retrieval_pairs.jsonl")
    rows = read_pairs(pairs_path)
    if len(rows) < 2:
        raise ValueError("Cần ít nhất hai retrieval pairs.")

    from transformers import AutoModel, AutoTokenizer

    base_path = require_local_model(config, "embedding")
    kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), **kwargs)
    model = AutoModel.from_pretrained(str(base_path), **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(int(config["seed"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not torch.cuda.is_bf16_supported())
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batches(rows, args.batch_size, rng):
            q = tokenizer(
                [row["question"] for row in batch], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt"
            )
            p = tokenizer(
                [row["positive"] for row in batch], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt"
            )
            q = {key: value.to(device) for key, value in q.items()}
            p = {key: value.to(device) for key, value in p.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
                q_vec = mean_pool(model(**q).last_hidden_state, q["attention_mask"])
                p_vec = mean_pool(model(**p).last_hidden_state, p["attention_mask"])
                logits = q_vec @ p_vec.T / args.temperature
                labels = torch.arange(len(batch), device=device)
                loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"epoch": epoch + 1, "loss": sum(losses) / max(1, len(losses))}))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"saved": str(Path(args.output).resolve()), "pairs": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

