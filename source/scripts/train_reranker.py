from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import torch


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.config import load_config, require_local_model  # noqa: E402


def read_rows(path: str | Path) -> list[tuple[str, str, int]]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append((row["question"], row["positive"], 1))
            examples.extend((row["question"], negative, 0) for negative in row.get("negatives", []))
    return examples


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(output: str | Path) -> Path | None:
    root = Path(output)
    checkpoints = [
        path for path in root.glob("checkpoint-*")
        if path.is_dir() and checkpoint_step(path) >= 0 and (path / "training_state.pt").is_file()
    ]
    return max(checkpoints, key=checkpoint_step) if checkpoints else None


def resolve_resume_checkpoint(value: str | None, output: str | Path) -> Path | None:
    if value is None or value.lower() in {"none", "false", "off"}:
        return None
    checkpoint = latest_checkpoint(output) if value.lower() == "auto" else Path(value)
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint reranker không tồn tại: {checkpoint}")
    return checkpoint


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def prune_checkpoints(output: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in output.glob("checkpoint-*") if path.is_dir() and checkpoint_step(path) >= 0),
        key=checkpoint_step,
    )
    for path in checkpoints[:-max(1, keep)]:
        shutil.rmtree(path)


def save_checkpoint(
    output: Path,
    global_step: int,
    epoch: int,
    next_batch: int,
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    keep: int,
    run_signature: dict,
) -> Path:
    checkpoint = output / f"checkpoint-{global_step}"
    temporary = output / f".checkpoint-{global_step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "next_batch": next_batch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "run_signature": run_signature,
    }
    torch.save(state, temporary / "training_state.pt")
    (temporary / "resume.json").write_text(
        json.dumps(
            {"global_step": global_step, "epoch": epoch, "next_batch": next_batch},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.rename(checkpoint)
    prune_checkpoints(output, keep)
    print(f"[Reranker checkpoint] saved: {checkpoint}", flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the Vietnamese cross-encoder on automatic official-data labels")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "p0.yaml"))
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--output", default=str(PROJECT / "models" / "Vietnamese_Reranker-warmup"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none, hoặc đường dẫn checkpoint; auto tiếp tục checkpoint mới nhất trong output.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size và grad-accum phải >= 1.")
    if args.checkpoint_steps < 1 or args.save_total_limit < 1:
        raise ValueError("checkpoint-steps và save-total-limit phải >= 1.")

    config = load_config(args.config)
    pairs_path = args.pairs or str(Path(config["paths"]["artifacts"]) / "data" / "retrieval_pairs.jsonl")
    rows = read_rows(pairs_path)
    if args.max_examples:
        random.Random(int(config["seed"])).shuffle(rows)
        rows = rows[: args.max_examples]
    if len(rows) < 4:
        raise ValueError("Không đủ positive/negative rows.")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, output)
    base_path = require_local_model(config, "reranker")
    load_path = resume_checkpoint or base_path
    kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(str(load_path), **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(str(load_path), **kwargs)
    if args.gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            raise RuntimeError("Checkpoint reranker không hỗ trợ gradient checkpointing.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batches_per_epoch = math.ceil(len(rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_updates = max(1, args.epochs * updates_per_epoch)
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    pairs_stat = Path(pairs_path).stat()
    run_signature = {
        "examples": len(rows),
        "pairs_size": pairs_stat.st_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    start_epoch = 0
    start_batch = 0
    global_step = 0
    if resume_checkpoint:
        state = torch.load(resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("run_signature") != run_signature:
            raise RuntimeError(
                "Cấu hình/dữ liệu reranker khác lần tạo checkpoint. "
                f"checkpoint={state.get('run_signature')}; current={run_signature}. "
                "Dùng lại đúng tham số cũ hoặc chọn --resume-from-checkpoint none để train mới."
            )
        optimizer.load_state_dict(state["optimizer"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["next_batch"])
        global_step = int(state["global_step"])
        if start_batch >= batches_per_epoch:
            start_epoch += 1
            start_batch = 0
        print(
            f"[Reranker resume] {resume_checkpoint} -> epoch={start_epoch + 1}, "
            f"batch={start_batch}, optimizer_step={global_step}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_rows = list(rows)
        random.Random(int(config["seed"]) + epoch).shuffle(epoch_rows)
        losses = []
        model.train()
        first_batch = start_batch if epoch == start_epoch else 0
        for batch_index in range(first_batch, batches_per_epoch):
            start = batch_index * args.batch_size
            batch = epoch_rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row[0] for row in batch],
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            labels = torch.tensor([row[2] for row in batch], device=device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                logits = model(**encoded).logits
                if logits.shape[-1] == 1:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float())
                else:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            is_update = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == batches_per_epoch
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.checkpoint_steps == 0:
                    save_checkpoint(
                        output,
                        global_step,
                        epoch,
                        batch_index + 1,
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        scaler,
                        args.save_total_limit,
                        run_signature,
                    )
            losses.append(float(loss.detach().cpu()))
        print(
            json.dumps(
                {"epoch": epoch + 1, "loss": sum(losses) / max(1, len(losses)), "optimizer_step": global_step}
            ),
            flush=True,
        )
        start_batch = 0

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "training_complete.json").write_text(
        json.dumps(
            {"epochs": args.epochs, "optimizer_step": global_step, "examples": len(rows)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "saved": str(output.resolve()),
                "examples": len(rows),
                "optimizer_steps": global_step,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
