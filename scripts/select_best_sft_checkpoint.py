from __future__ import annotations

import argparse
import copy
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legalqa.core.config import load_config, require_local_model  # noqa: E402
from legalqa.core.data import load_qa  # noqa: E402
from legalqa.core.metrics import evaluate_predictions_official  # noqa: E402
from legalqa.generation.generation import clean_answer  # noqa: E402
from legalqa.generation.prompts import SYSTEM_PROMPT, qwen_user_prompt  # noqa: E402
from train_qwen_sft_lora import load_evidence_by_id  # noqa: E402


def build_prompt(question: str, question_id: str, evidence_by_id, max_context_chars: int, config) -> str:
    sft_config = copy.deepcopy(config)
    sft_config.setdefault("generation", {})["max_context_chars"] = max_context_chars
    evidence = evidence_by_id.get(question_id, [])
    return qwen_user_prompt(question, evidence, sft_config, retrieval_diagnostics=None)


def generate_predictions(
    model, tokenizer, rows, max_new_tokens: int, label: str, evidence_by_id, max_context_chars: int, config,
    print_limit: int = 5,
) -> dict[str, str]:
    predictions = {}
    with torch.no_grad():
        for i, row in enumerate(rows):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_prompt(row.question, row.question_id, evidence_by_id, max_context_chars, config),
                },
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            predictions[row.question_id] = clean_answer(text)
            if i < print_limit:
                print(
                    f"  [{label}] qid={row.question_id}\n"
                    f"    ref : {row.answer[:300]}\n"
                    f"    pred: {predictions[row.question_id][:300]}",
                    flush=True,
                )
            elif i == print_limit:
                print(f"  [{label}] ... còn lại {len(rows) - print_limit} câu không in chi tiết", flush=True)
    return predictions


def save_predictions(path: Path, predictions: dict[str, str], references: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {qid: {"reference": references[qid], "prediction": predictions[qid]} for qid in predictions},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh câu trả lời trên dev cho từng epoch checkpoint SFT, "
        "chọn checkpoint có METEOR cao nhất theo đúng công thức chấm của BTC (không dùng eval_loss)."
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--checkpoints-root", required=True, help="thư mục --output của train_qwen_sft_lora.py")
    parser.add_argument("--dev-file", required=True)
    parser.add_argument(
        "--train-file",
        default=None,
        help="nếu truyền, sau khi chọn xong checkpoint tốt nhất sẽ sinh thêm output trên train để lưu lại đọc sau",
    )
    parser.add_argument(
        "--evidence", default=None, help="sft_evidence.jsonl (RAG thật) — PHẢI khớp file dùng lúc train_qwen_sft_lora.py"
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3500,
        help="PHẢI khớp --max-context-chars lúc train, không thì METEOR đo sai prompt model chưa từng thấy",
    )
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--max-eval-samples", type=int, default=None, help="giới hạn số câu dev để sinh nhanh hơn")
    parser.add_argument("--print-limit", type=int, default=5, help="số câu in chi tiết ref/pred mỗi epoch (0 = không in)")
    parser.add_argument(
        "--max-train-eval-samples",
        type=int,
        default=None,
        help="giới hạn số câu train để sinh output lưu lại (mặc định toàn bộ --train-file)",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--best-output", default=None, help="mặc định <checkpoints-root>/best")
    parser.add_argument("--report", default=None, help="mặc định <checkpoints-root>/epoch_selection_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    evidence_by_id = load_evidence_by_id(args.evidence)
    root = Path(args.checkpoints_root)
    checkpoints = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"Không tìm thấy checkpoint-* dưới {root}.")

    dev_rows = load_qa(Path(args.dev_file), True)
    if args.max_eval_samples:
        # dev_rows đến từ deterministic_split rồi sorted theo question_id — lấy N câu
        # đầu tiên sẽ không đại diện (thiên theo thứ tự id); shuffle theo seed config
        # trước khi cắt để mẫu smoke-test thực sự ngẫu nhiên nhưng vẫn tái lập được.
        rng = random.Random(int(config["seed"]))
        dev_rows = sorted(dev_rows, key=lambda row: row.question_id)
        rng.shuffle(dev_rows)
        dev_rows = dev_rows[: args.max_eval_samples]
    if not dev_rows:
        raise RuntimeError(f"Dev file rỗng: {args.dev_file}")
    references = {row.question_id: row.answer for row in dev_rows}

    import transformers
    from peft import PeftModel

    model_path = require_local_model(config, "generator")
    kwargs = dict(local_files_only=True, trust_remote_code=config["runtime"].get("trust_remote_code", True))
    processor = transformers.AutoProcessor.from_pretrained(str(model_path), **kwargs)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    auto_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if auto_class is None:
        raise RuntimeError("Cần Transformers mới có AutoModelForMultimodalLM.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_load_kwargs = dict(kwargs)
    if args.load_in_4bit:
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
    base_model = auto_class.from_pretrained(str(model_path), **model_load_kwargs)
    base_model.eval()

    results = []
    peft_model = None
    for checkpoint in checkpoints:
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base_model, str(checkpoint), adapter_name=checkpoint.name)
        else:
            peft_model.load_adapter(str(checkpoint), adapter_name=checkpoint.name)
        peft_model.set_adapter(checkpoint.name)
        peft_model.eval()
        predictions = generate_predictions(
            peft_model,
            tokenizer,
            dev_rows,
            args.max_new_tokens,
            checkpoint.name,
            evidence_by_id,
            args.max_context_chars,
            config,
            print_limit=args.print_limit,
        )
        scores = evaluate_predictions_official(predictions, references)
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        predictions_path = root / f"predictions_dev_{checkpoint.name}.json"
        save_predictions(predictions_path, predictions, references)
        results.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": trainer_state.get("epoch"),
                "meteor": scores["meteor"],
                "rouge_l_f1": scores["rouge_l_f1"],
                "dev_predictions_file": str(predictions_path),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    best = max(results, key=lambda row: row["meteor"])
    best_output = Path(args.best_output) if args.best_output else root / "best"
    if best_output.exists():
        shutil.rmtree(best_output)
    shutil.copytree(best["checkpoint"], best_output)

    if args.train_file:
        train_rows = load_qa(Path(args.train_file), True)
        if args.max_train_eval_samples:
            train_rows = train_rows[: args.max_train_eval_samples]
        train_references = {row.question_id: row.answer for row in train_rows}
        best_checkpoint_name = Path(best["checkpoint"]).name
        peft_model.set_adapter(best_checkpoint_name)
        peft_model.eval()
        train_predictions = generate_predictions(
            peft_model,
            tokenizer,
            train_rows,
            args.max_new_tokens,
            f"train/{best_checkpoint_name}",
            evidence_by_id,
            args.max_context_chars,
            config,
            print_limit=args.print_limit,
        )
        train_predictions_path = root / f"predictions_train_{best_checkpoint_name}.json"
        save_predictions(train_predictions_path, train_predictions, train_references)
        best["train_predictions_file"] = str(train_predictions_path)

    report = {
        "note": scores["note"],
        "eval_samples": len(dev_rows),
        "results": results,
        "best": best,
        "best_output": str(best_output.resolve()),
    }
    report_path = Path(args.report) if args.report else root / "epoch_selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
