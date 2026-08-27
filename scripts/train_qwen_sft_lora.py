from __future__ import annotations

import argparse
import copy
import json
import random
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
from legalqa.core.data import Chunk, load_qa, qa_as_object  # noqa: E402
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


def _token_stats(lengths: list[int]) -> str:
    if not lengths:
        return "không có mẫu"
    ordered = sorted(lengths)

    def pct(part: float) -> int:
        return ordered[min(len(ordered) - 1, int(len(ordered) * part))]

    return f"p50={pct(0.5)} p95={pct(0.95)} max={ordered[-1]}"


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
        self.ids = []
        self.skipped_without_evidence = 0
        self.skipped_over_max_length = 0
        self.prompt_token_lengths: list[int] = []
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
            self.prompt_token_lengths.append(len(prompt_ids))
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
            self.ids.append(row.question_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def reshuffle_split(ids: list[str], dev_ratio: float, seed: int, epoch: int) -> tuple[list[str], list[str]]:
    """Mỗi epoch trộn lại toàn bộ pool train+dev theo seed+epoch để ra một cặp train/dev mới.

    Trả về (train_ids, dev_ids); cùng input + epoch nào ra đúng một kết quả nên resume an toàn.
    """
    rng = random.Random(seed + epoch)
    pool = list(ids)
    rng.shuffle(pool)
    dev_count = min(len(pool) - 1, max(1, round(len(pool) * dev_ratio)))
    return pool[dev_count:], pool[:dev_count]


def _sliced_ce_loss(hidden_states, lm_head, labels):
    """CE chỉ trên vùng có label — nhãn supervised luôn nằm CUỐI sequence theo cách build dataset,
    nên chỉ cần logit tại vị trí t-1 để dự đoán token t. Tránh materialize logits toàn sequence
    (~11GB ở seq 12k × vocab ~150k trên T4) trong khi prompt chiếm phần lớn nhưng toàn -100."""
    supervised = labels != -100
    first = int(supervised.float().argmax(dim=1).min().item())
    start = max(first - 1, 0)
    hidden = hidden_states[:, start : labels.shape[1] - 1, :]
    logits = lm_head(hidden)
    shift_labels = labels[:, start + 1 :]
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _unwrap_peft(model):
    base = model.module if hasattr(model, "module") else model
    return base.get_base_model() if hasattr(base, "get_base_model") else base


class _SftTrainerMixin:
    """compute_loss tự thân: trunk forward lấy hidden states rồi _sliced_ce_loss,
    không đi qua forward mặc định của CausalLM (nơi lm_head chạy trên toàn sequence)."""

    def _resolve_trunk_and_head(self):
        base = _unwrap_peft(self.model)
        trunk = getattr(base, "model", None)
        head = getattr(base, "lm_head", None)
        if trunk is None or not hasattr(head, "weight"):
            raise RuntimeError(
                f"Không resolve được .model/.lm_head trên {type(base).__name__}; "
                f"children: {[name for name, _ in base.named_children()]}"
            )
        self._trunk = trunk
        self._head = head

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = self._trunk(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        loss = _sliced_ce_loss(outputs.last_hidden_state, self._head, labels)
        if num_items_in_batch is not None:
            # chuẩn HF: normalize theo tổng token supervised của cả accumulation step
            loss = loss * (labels != -100).sum() / num_items_in_batch
        return (loss, {"loss": loss.detach()}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # eval chỉ cần eval_loss (không compute_metrics); tránh nhánh logits phức tạp của Trainer
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


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


def _make_sft_trainer(transformers_module, epoch_subset_size: int | None):
    """Trainer SFT dùng slice-loss; nếu pool > subset thì thêm rút random mẫu khác mỗi epoch."""

    class SftTrainer(_SftTrainerMixin, transformers_module.Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # resolve một lần để fail sớm trước khi tốn GPU
            self._resolve_trunk_and_head()

    if not epoch_subset_size:
        return SftTrainer

    class EpochResampleSftTrainer(SftTrainer):
        def _get_train_sampler(self, *args, **kwargs):
            return _EpochRandomSubsetSampler(len(self.train_dataset), epoch_subset_size, self.args.seed)

    return EpochResampleSftTrainer


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
    parser.add_argument("--max-target-tokens", type=int, default=700)
    parser.add_argument(
        "--evidence", default=None, help="sft_evidence.jsonl sinh bởi scripts/prepare_sft_evidence.py (RAG thật)"
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=24000,
        help="ngân sách ký tự evidence cho SFT — mặc định khớp max_context_chars lúc inference "
        "(compute_loss chỉ tính logits vùng answer nên seq dài vẫn vừa T4); đo token thật nếu đổi.",
    )
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bỏ câu hỏi không có evidence thay vì dạy model trả lời từ hư không",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
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
        "--reshuffle-split-per-epoch",
        action="store_true",
        help="mỗi epoch trộn lại pool train+dev và cắt lại theo --reshuffle-dev-ratio; checkpoint "
        "epoch nào thi trên dev của epoch đó (dev_sft_step<N>.json ghi cạnh checkpoint cho bước chọn).",
    )
    parser.add_argument(
        "--reshuffle-dev-ratio",
        type=float,
        default=0.15,
        help="tỉ lệ dev khi cắt lại pool mỗi epoch (chỉ dùng với --reshuffle-split-per-epoch).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none, hoặc đường dẫn checkpoint-N cụ thể trong --output",
    )
    parser.add_argument(
        "--load-adapters",
        default=None,
        help="đường dẫn thư mục chứa adapter_model.safetensors + adapter_config.json "
        "(checkpoint-219 của QLoRA) để load LoRA weights vào model trước khi train SFT.",
    )
    parser.add_argument("--output", default=str(PROJECT / "models" / "qwen-sft-lora"))
    args = parser.parse_args()
    if args.reshuffle_split_per_epoch and args.train_pool_samples:
        raise ValueError(
            "--train-pool-samples không dùng cùng --reshuffle-split-per-epoch: "
            "rotation đã tự đổi dữ liệu mỗi epoch rồi."
        )
    config = load_config(args.config)
    # trần token duy nhất của SFT, suy ra từ ngân sách ký tự (không còn --max-length riêng):
    # evidence ≤ max_context_chars ký tự ≈ tối đa ~1 token/ký tự, cộng khung system/hướng dẫn/câu hỏi
    # và answer. Chỉ là ngưỡng drop sample bất khả thi — collator pad theo độ dài thật nên giá trị
    # dư thừa không tốn VRAM gì.
    max_length = args.max_context_chars + args.max_target_tokens + 2048

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
        model_load_kwargs.update(
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map="auto",
        )
        # kernel fast-path của qwen3_5 (flash-linear-attention) không có sẵn offline; nếu
        # transformers đang rơi về eager cho cả full-attention thì sdpa tiết kiệm rất nhiều VRAM
        model_load_kwargs.setdefault("attn_implementation", "sdpa")
    else:
        model_load_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else None
    print(
        f"[SFT load] device_count={torch.cuda.device_count()} load_in_4bit={args.load_in_4bit} "
        f"kwargs={sorted(model_load_kwargs)}",
        flush=True,
    )
    model = auto_class.from_pretrained(str(model_path), **model_load_kwargs)
    loaded_map = getattr(model, "hf_device_map", None)
    print(f"[SFT load] hf_device_map={loaded_map or 'KHONG_CO'}", flush=True)
    # Không dispatch đa GPU: accelerate cấm train model 4-bit nằm trên nhiều device
    # ("You can't train a model that has been loaded in 8-bit or 4-bit precision...").
    # VRAM đủ nhờ slice-loss + attn_implementation="sdpa" (không materialize ma trận S×S).
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

    if args.load_adapters:
        # Nạp LoRA weights đã train (vd checkpoint QLoRA) vào các layer LoRA mới tạo.
        # set_peft_model_state_dict lo mapping prefix "base_model.model." — không dùng
        # load_state_dict thường vì missing_keys sẽ gồm toàn bộ weight gốc (nhiễu).
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors

        adapter_bin = Path(args.load_adapters) / "adapter_model.safetensors"
        if not adapter_bin.is_file():
            raise FileNotFoundError(f"Không tìm thấy adapter_model.safetensors trong {args.load_adapters}")
        print(f"[SFT load-adapters] Đọc LoRA weights từ {adapter_bin}", flush=True)
        adapter_state = load_safetensors(str(adapter_bin))
        result = set_peft_model_state_dict(model, adapter_state)
        # peft trả missing gồm cả weight gốc không bao giờ nằm trong adapter (vd visual.patch_embed);
        # chỉ thiếu key LoRA mới là lệch cấu trúc thật, unexpected khác rỗng luôn là lỗi
        missing_lora = [key for key in (result.missing_keys or []) if "lora_" in key]
        print(
            f"[SFT load-adapters] file có {len(adapter_state)} key "
            f"({sum('lora_' in key for key in adapter_state)} key LoRA)",
            flush=True,
        )
        if result.unexpected_keys or missing_lora:
            raise RuntimeError(
                "Adapter không khớp LoRA hiện tại (kiểm tra --lora-r/target modules). "
                f"missing_lora={missing_lora[:3]} unexpected={list(result.unexpected_keys)[:3]}"
            )
        print("[SFT load-adapters] Đã nạp weights QLoRA xong", flush=True)

    evidence_by_id = load_evidence_by_id(args.evidence)
    train_file = Path(args.train_file)
    train_rows = load_qa(train_file, True)
    pool_size = args.train_pool_samples or args.max_train_samples
    if not args.reshuffle_split_per_epoch and pool_size:
        train_rows = train_rows[:pool_size]
    dev_file = Path(args.dev_file) if args.dev_file else None
    dev_rows = load_qa(dev_file, True) if dev_file and dev_file.is_file() else []

    if args.reshuffle_split_per_epoch:
        if not dev_rows:
            raise RuntimeError("--reshuffle-split-per-epoch cần --dev-file: pool = train + dev mới trộn được.")
        # Pool token hoá ĐÚNG MỘT LẦN; mỗi epoch chỉ gán lại items/ids theo split mới —
        # re-tokenize cả pool mỗi epoch tốn hàng phút CPU vô ích.
        pool_rows = [*train_rows, *dev_rows]
        row_by_id = {row.question_id: row for row in pool_rows}
        pool = RagChatDataset(
            pool_rows,
            tokenizer,
            max_length,
            evidence_by_id,
            args.max_context_chars,
            args.max_target_tokens,
            args.require_evidence,
            config,
        )
        item_by_id = dict(zip(pool.ids, pool.items))
        seed = int(config["seed"])
        missing_evidence = sum(1 for row in pool_rows if row.question_id not in evidence_by_id)
        if missing_evidence:
            print(
                f"[SFT reshuffle] {missing_evidence}/{len(pool_rows)} câu pool thiếu evidence — "
                "bị bỏ ở MỌI epoch; build evidence toàn pool (run_qwen_sft_lora.py --reshuffle-split-per-epoch) "
                "để tận dụng hết dữ liệu.",
                flush=True,
            )
        train = RagChatDataset(
            [], tokenizer, max_length, {}, args.max_context_chars, args.max_target_tokens, False, config
        )
        dev = RagChatDataset(
            [], tokenizer, max_length, {}, args.max_context_chars, args.max_target_tokens, False, config
        )

        def _assign_split(epoch: int) -> None:
            train_ids, dev_ids = reshuffle_split(pool.ids, args.reshuffle_dev_ratio, seed, epoch)
            if args.max_train_samples:
                train_ids = train_ids[: args.max_train_samples]
            if args.max_eval_samples:
                dev_ids = dev_ids[: args.max_eval_samples]
            train.items = [item_by_id[q] for q in train_ids]
            train.ids = train_ids
            dev.items = [item_by_id[q] for q in dev_ids]
            dev.ids = dev_ids

        _assign_split(0)
        print(
            f"[SFT reshuffle] pool={len(item_by_id)} câu, dev_ratio={args.reshuffle_dev_ratio} | pool bỏ qua: "
            f"thiếu evidence={pool.skipped_without_evidence}, vượt max_length={pool.skipped_over_max_length} | "
            f"epoch 1: train={len(train)} dev={len(dev)} (mỗi epoch cắt lại cặp mới)",
            flush=True,
        )
    else:
        train = RagChatDataset(
            train_rows,
            tokenizer,
            max_length,
            evidence_by_id,
            args.max_context_chars,
            args.max_target_tokens,
            args.require_evidence,
            config,
        )
        if args.max_eval_samples:
            # cùng sort+shuffle(seed) với run_qwen_sft_lora.py khi build evidence và
            # select_best_sft_checkpoint.py khi chấm — nếu lệch tập thì dev bị skip oan vì thiếu evidence
            dev_rows = sorted(dev_rows, key=lambda row: row.question_id)
            random.Random(int(config["seed"])).shuffle(dev_rows)
            dev_rows = dev_rows[: args.max_eval_samples]
        dev = RagChatDataset(
            dev_rows,
            tokenizer,
            max_length,
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
    # token đo bằng tokenizer thật — đối chiếu trực tiếp với trần suy ra từ max_context_chars;
    # khi reshuffle thì shell train rỗng, đo trên pool mới phản ánh đúng độ dài thật
    token_probe = pool if args.reshuffle_split_per_epoch else train
    print(
        f"[SFT prompt tokens] max_length={max_length} (suy từ max_context_chars={args.max_context_chars}) | "
        f"train: {_token_stats(token_probe.prompt_token_lengths)} | "
        f"dev: {_token_stats(dev.prompt_token_lengths)}",
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
        "max_length_derived": max_length,
        "max_target_tokens": args.max_target_tokens,
        "max_context_chars": args.max_context_chars,
        "require_evidence": args.require_evidence,
        "lora_r": args.lora_r,
        "load_in_4bit": args.load_in_4bit,
        "reshuffle_split_per_epoch": bool(args.reshuffle_split_per_epoch),
        "reshuffle_dev_ratio": args.reshuffle_dev_ratio if args.reshuffle_split_per_epoch else None,
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
    trainer_cls = _make_sft_trainer(transformers, epoch_subset_size)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev if len(dev) else None,
        data_collator=CausalCollator(tokenizer),
    )
    if args.reshuffle_split_per_epoch:

        class _ReshuffleCallback(transformers.TrainerCallback):
            """Đổi split ngay đầu mỗi epoch; dump dev của epoch ra dev_sft_step<N>.json đúng lúc eval
            để số step khớp tên checkpoint-N (eval_strategy="epoch" chỉ eval ở cuối epoch)."""

            def __init__(self, out_dir: Path):
                self.out_dir = out_dir
                self.current_dev_rows: list = []

            def on_epoch_begin(self, args, state, control, **kwargs):
                # state.epoch == chỉ số epoch đang bắt đầu (0-based): suy trực tiếp từ state nên
                # resume giữa chừng vẫn tái tạo đúng split của epoch đó, không cần lưu callback state.
                _assign_split(round(state.epoch))
                self.current_dev_rows = [row_by_id[q] for q in dev.ids]

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                payload = qa_as_object(self.current_dev_rows)
                (self.out_dir / f"dev_sft_step{state.global_step}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

        trainer.add_callback(_ReshuffleCallback(output))
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
