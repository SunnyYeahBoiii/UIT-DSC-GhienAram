"""SFT bằng LoRA trên Qwen, chấm bằng đúng code của BTC ở từng mốc bước.

Ba quyết định định hình file này, mỗi cái đều có số đo đứng sau.

**1. Loss chỉ tính trên token đáp án, và logits cũng chỉ tính ở đó.**
69,8% 8-gram của reference answer là nguyên văn từ gold passage, tức đây là bài TRÍCH.
Cho loss chảy qua phần prompt là dạy model sinh lại văn bản luật — thứ nó đã được copy
sẵn trong input. Nhưng lý do thứ hai mới là lý do bắt buộc: vocab của model là 248.070,
nên logits của một mẫu seq 6.144 chiếm 6144*248070*2 = 3,05 GB fp16, và HF upcast fp32
trước cross-entropy nên thành 6,1 GB — nhiều hơn cả 4,5 GB trọng số, T4 16 GB OOM ngay.
Vì ~87% token là prompt và bị mask -100, logits ở đó tính ra rồi vứt đi. Đáp án là hậu
tố LIỀN MẠCH nên cắt hidden state theo lát rồi mới nhân lm_head là đúng: 450*248070*2 =
223 MB, giảm 27 lần. Đây là lý do file này tự tính loss thay vì để model tính.

**2. fp16, không bf16.** T4 là sm_75. `is_bf16_supported()` trả True nhưng đó là đường
emulate, chậm. Và fp16 đã được đo an toàn cho model này (§PIPELINE_V10): max |activation|
62,9 so với trần 65.504, `tensor không hữu hạn 0` ở cả 12 cấu hình. Nếu sau này train
trên GPU thuê bằng bf16 thì PHẢI kiểm `max(abs(adapter)) < 65504` trước khi mang về T4.

**3. Chọn checkpoint theo METEOR, không theo loss, cũng không theo trung bình hai metric.**
Đo dải động bằng chính code BTC trên 300 mẫu:

    baseline                    METEOR   ROUGE-L
    trần: chính reference       1,0000    1,0000
    nửa đầu reference           0,4791    0,6650
    reference xáo từ            0,5080    0,3524
    reference của câu KHÁC      0,1170    0,2780
    hằng số 1 câu               0,0424    0,1420

ROUGE-L có SÀN 0,278 cho câu trả lời hoàn toàn sai, vì tokenizer của nó xoá dấu và các
mảnh 'n'/'c'/'i'/'u' khớp ngẫu nhiên. Nó còn thưởng trả lời CỤT: cắt nửa đầu reference
được ROUGE 0,665 nhưng METEOR chỉ 0,479. Chọn theo trung bình hai metric là thưởng cho
model học nói ngắn lại.

Train loss cũng không dùng làm tiêu chí dừng: ~88% đáp án mở đầu cùng một khuôn
"Căn cứ Điều N...", nên loss tiếp tục giảm đúng lúc model bắt đầu học thuộc khuôn câu
thay vì đọc căn cứ — đúng dạng lỗi của v9.2.

Chạy:
    PYTHONPATH=src python scripts/train_sft.py --model $MODEL \
        --data data/clean/sft.jsonl --dev data/clean/dev.jsonl --ra out/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from legalqa.scorer import cham, chuan_bi_nltk


def nap(p: Path, n: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    return rows[:n] if n else rows


def ma_hoa(row: dict, tok, max_seq: int) -> dict | None:
    """prompt -> ids có chat template, answer -> ids, nhãn mask -100 trên phần prompt.

    Dùng `apply_chat_template` chứ không nối chuỗi tay: token điều khiển của Qwen
    (<|im_start|>...) phải khớp TỪNG TOKEN với lúc inference, và nối tay thì sai một
    khoảng trắng là model thấy một phân bố nó chưa từng train.

    `add_generation_prompt=True` để phần prompt kết thúc đúng chỗ model bắt đầu sinh —
    ranh giới này chính là ranh giới mask.
    """
    p_ids = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        add_generation_prompt=True,
        tokenize=True,
    )
    a_ids = tok(row["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(a_ids) > max_seq:
        return None
    return {
        "input_ids": p_ids + a_ids,
        "n_prompt": len(p_ids),  # số token đầu bị mask; đáp án là hậu tố liền mạch
    }


def gop(batch: list[dict], pad_id: int) -> dict:
    """Pad PHẢI. Lát đáp án tính từ `n_prompt` nên pad trái sẽ dịch mọi chỉ số."""
    import torch

    dai = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), dai), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), dai), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = 1
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "n_prompt": torch.tensor([b["n_prompt"] for b in batch]),
        "n_tong": torch.tensor([len(b["input_ids"]) for b in batch]),
    }


def loss_dap_an(model, lo, batch):
    """Cross-entropy chỉ trên token đáp án, lm_head chỉ chạy trên lát đáp án.

    `lo` là hidden state cuối, [B, T, H]. Với mẫu i, token đáp án nằm ở
    [n_prompt, n_tong). Dự đoán token t dùng hidden ở t-1, nên lát hidden cần lấy là
    [n_prompt-1, n_tong-1) và nhãn là input_ids[n_prompt:n_tong].

    Chạy từng mẫu trong batch thay vì vector hoá: batch trên T4 là 1-2 mẫu, và gộp
    thành tensor chữ nhật sẽ đưa lại đúng phần logits thừa mà cả hàm này sinh ra để
    tránh.
    """
    import torch
    import torch.nn.functional as F

    lm = model.get_output_embeddings()
    # Khởi tạo bằng tensor NỐI VÀO ĐỒ THỊ, không phải float 0.0: nếu batch không có
    # token đáp án nào thì `tong` giữ nguyên giá trị khởi tạo, và một float thì
    # `.backward()` nổ giữa lúc train. `lo.sum()*0` giữ đúng device, dtype và graph.
    tong, dem = lo.sum() * 0.0, 0
    for i in range(lo.size(0)):
        a, b = int(batch["n_prompt"][i]), int(batch["n_tong"][i])
        if b - a < 1:
            continue
        h = lo[i, a - 1 : b - 1, :]
        nhan = batch["input_ids"][i, a:b].to(h.device)
        # float() trước cross_entropy: logsumexp trên 248.070 lớp ở fp16 mất chính xác
        # đủ để gradient nhiễu. Chỉ 450 hàng nên upcast ở đây rẻ.
        tong = tong + F.cross_entropy(lm(h).float(), nhan, reduction="sum")
        dem += b - a
    return tong / max(1, dem), dem


def tim_than(model):
    """Trả về hàm gọi phần Transformer (không có lm_head) và in ra đường đã chọn.

    Không hardcode `model.model.model`: đây là model đa phương thức (có
    `video_preprocessor_config.json`), phần ngôn ngữ có thể nằm dưới `.language_model`
    thay vì ngay `.model`. Dò MỘT LẦN lúc khởi động bằng một chuỗi 4 token: sai đường
    thì chết trong hai giây với thông báo rõ, thay vì chết sau vài giờ train.
    """
    import torch

    goc = model.get_base_model() if hasattr(model, "get_base_model") else model
    ung_vien = [("model", lambda: goc.model)]
    if hasattr(goc, "model") and hasattr(goc.model, "language_model"):
        ung_vien.append(("model.language_model", lambda: goc.model.language_model))
    thu = torch.ones((1, 4), dtype=torch.long, device=next(goc.parameters()).device)
    for ten, lay in ung_vien:
        try:
            than = lay()
            with torch.no_grad():
                o = than(input_ids=thu, attention_mask=torch.ones_like(thu))
            h = o.last_hidden_state
            if h.shape[:2] == (1, 4) and h.shape[2] == goc.config.get_text_config().hidden_size:
                print(f"hidden state lấy qua `{ten}`  H={h.shape[2]}", flush=True)
                return than
        except Exception as e:  # noqa: BLE001 — dò đường, lỗi ở đây là thông tin
            print(f"  `{ten}` không dùng được: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("không tìm được phần Transformer — kiểm lại cấu trúc model")


def sinh_va_cham(model, tok, dev: list[dict], max_new: int, bs: int) -> dict:
    """Sinh câu trả lời trên dev rồi chấm bằng code BTC.

    Greedy, không sampling: đang đo chất lượng checkpoint chứ không đo may rủi. Sampling
    làm hai lần eval cùng checkpoint ra hai số khác nhau, và khi đó không phân biệt được
    "model tốt lên" với "lần này bốc trúng".

    Chấm với reference THÔ của BTC (dev dựng từ warmup.json, không qua
    `clean_reference_answer`). Chấm với target đã sạch là đo một đại lượng dễ hơn
    leaderboard.
    """
    import torch

    model.eval()
    du_doan: dict[str, str] = {}
    for i in range(0, len(dev), bs):
        lo = dev[i : i + bs]
        vb = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for r in lo
        ]
        # Pad TRÁI khi sinh: pad phải thì token cuối của chuỗi ngắn là <pad>, model sinh
        # tiếp từ đó. Ngược với lúc train, và đây là lý do phải đặt lại mỗi lần.
        tok.padding_side = "left"
        enc = tok(vb, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for r, seq in zip(lo, out):
            du_doan[r["id"]] = tok.decode(
                seq[enc["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
    tok.padding_side = "right"
    model.train()

    ref = {r["id"]: r["answer"] for r in dev}
    diem = cham(du_doan, ref)
    # Tách riêng nhóm retrieval trượt gold: tập train đã loại hết nhóm này nên model
    # chưa từng thấy context thiếu căn cứ. Điểm trung bình sẽ che mất chỗ nó bịa.
    truot = [r["id"] for r in dev if r.get("ly_do_loai")]
    if truot:
        diem["meteor_truot"] = cham(
            {k: du_doan[k] for k in truot if k in du_doan}, {k: ref[k] for k in truot}
        )["meteor"]
    diem["do_dai_tb"] = sum(len(v.split()) for v in du_doan.values()) / max(1, len(du_doan))
    return diem


def _LOP_MODEL() -> list:
    """Các lớp Auto* thử theo thứ tự. Import trong hàm vì transformers cũ không có
    `AutoModelForImageTextToText` — thiếu lớp nào thì bỏ qua lớp đó, không vỡ."""
    import transformers

    ten = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel")
    return [getattr(transformers, t) for t in ten if hasattr(transformers, t)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/clean/sft.jsonl")
    ap.add_argument("--dev", default="data/clean/dev.jsonl")
    ap.add_argument("--ra", default="out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=6144)
    ap.add_argument("--eval-steps", type=int, default=150)
    ap.add_argument("--dev-n", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--gen-bs", type=int, default=4)
    ap.add_argument("--kien-nhan", type=int, default=3, help="số lần eval không tiến bộ thì dừng")
    ap.add_argument("--seed", type=int, default=0)
    # Phiên Kaggle chết cứng ở 12 giờ, và kernel chết vì hết giờ thì output KHÔNG được
    # lưu — mất cả `tot_nhat/`. Dừng chủ động trước mốc đó để còn ghi được checkpoint.
    ap.add_argument("--gio-toi-da", type=float, default=10.5)
    # Nối tiếp phiên trước: nạp adapter đã có thay vì khởi tạo B=0. KHÔNG khôi phục
    # optimizer/scheduler — mỗi phiên là một chu kỳ cosine riêng. Đặt `--lr` phiên sau
    # thấp hơn (vd 3e-5) để không hất adapter đã hội tụ ra khỏi cực tiểu.
    ap.add_argument("--tiep", default="", help="thư mục adapter của phiên trước")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    chuan_bi_nltk()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model nền là biến thể Qwen đa phương thức: `architectures` không phải ForCausalLM
    # nên AutoModelForCausalLM có thể từ chối ánh xạ. Thử lần lượt thay vì đoán, và IN RA
    # lớp nào nhận — `tim_than` ở dưới dò đường tới hidden state dựa vào đúng lớp đó.
    model = None
    for lop in _LOP_MODEL():
        try:
            model = lop.from_pretrained(
                a.model, dtype=torch.float16, attn_implementation="sdpa"
            ).cuda()
            print("nạp bằng", lop.__name__, flush=True)
            break
        except (ValueError, KeyError) as e:
            print(f"{lop.__name__} không nhận: {e}", flush=True)
    assert model is not None, "không lớp Auto* nào nạp được model nền"
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if a.tiep:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.tiep, is_trainable=True)
        print("nối tiếp adapter:", a.tiep)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=a.r,
                lora_alpha=a.alpha,
                lora_dropout=a.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                # q/k/v/o + MLP. Chỉ gắn vào attention là cách thường thấy nhưng đây là
                # bài TRÍCH văn bản: phần quyết định copy đúng chữ nào nằm ở MLP hơn.
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()
    than = tim_than(model)

    rows = [m for m in (ma_hoa(r, tok, a.max_seq) for r in nap(Path(a.data))) if m]
    # Xáo trước khi cắt: dev.jsonl giữ nguyên thứ tự warmup.json, lấy 120 dòng đầu là
    # lấy một lát cắt tuỳ hứng — tỉ lệ nhóm trượt gold trong đó không phản ánh cả tập,
    # mà `meteor_truot` đọc đúng nhóm đó. Seed cố định để hai lần chạy cùng một dev.
    dev = nap(Path(a.dev))
    random.Random(a.seed).shuffle(dev)
    dev = dev[: a.dev_n] if a.dev_n else dev
    print(f"train {len(rows):,} mẫu | dev {len(dev):,} mẫu", flush=True)

    tong_buoc = int(len(rows) * a.epochs / (a.bs * a.accum))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0
    )
    # Warmup 3%: LoRA khởi tạo B=0 nên bước đầu gradient rất lớn, vào thẳng lr đỉnh là
    # đẩy adapter đi xa trước khi nó kịp thấy dữ liệu. Cosine về 0 để cuối train adapter
    # ổn định thay vì còn nhảy quanh.
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * tong_buoc), tong_buoc)
    scaler = torch.amp.GradScaler("cuda")

    lich_su: list[dict] = []
    tot_nhat, khong_tien_bo, buoc = -1.0, 0, 0
    t0 = time.time()
    order = list(range(len(rows)))
    dung = False

    for ep in range(math.ceil(a.epochs)):
        random.shuffle(order)
        for i in range(0, len(order), a.bs):
            batch = gop([rows[j] for j in order[i : i + a.bs]], tok.pad_token_id)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                lo = than(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state
            loss, _ = loss_dap_an(model, lo, batch)
            scaler.scale(loss / a.accum).backward()

            if (i // a.bs + 1) % a.accum:
                continue
            scaler.unscale_(opt)
            # Ghi grad norm TRƯỚC khi clip: sau clip nó luôn <= 1,0 nên không còn nói
            # được gì. Norm tăng vọt là dấu hiệu phân kỳ, và đó là thứ cần thấy.
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            ).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            buoc += 1

            if buoc % 10 == 0:
                print(
                    f"  b{buoc}/{tong_buoc} ep{ep} loss {loss.item():.4f} "
                    f"gnorm {gnorm:.2f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time()-t0)/60:.0f}p",
                    flush=True,
                )

            if buoc % a.eval_steps == 0 or buoc == tong_buoc:
                d = sinh_va_cham(model, tok, dev, a.max_new, a.gen_bs)
                d |= {"buoc": buoc, "loss": loss.item(), "gnorm": gnorm}
                lich_su.append(d)
                (ra / "lich_su.json").write_text(
                    json.dumps(lich_su, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                print(f"  == EVAL b{buoc}: {d}", flush=True)
                if d["meteor"] > tot_nhat:
                    tot_nhat, khong_tien_bo = d["meteor"], 0
                    model.save_pretrained(ra / "tot_nhat")
                    print(f"  -> lưu checkpoint tốt nhất (METEOR {tot_nhat:.4f})", flush=True)
                else:
                    khong_tien_bo += 1
                    if khong_tien_bo >= a.kien_nhan:
                        print(f"  dừng sớm: {khong_tien_bo} lần eval không tiến bộ", flush=True)
                        dung = True
                        break
            if buoc >= tong_buoc:
                dung = True
                break
            if (time.time() - t0) / 3600 >= a.gio_toi_da:
                model.save_pretrained(ra / "cuoi")
                print(f"  hết {a.gio_toi_da} giờ ở bước {buoc} — lưu {ra/'cuoi'} để "
                      f"chạy tiếp bằng --tiep", flush=True)
                dung = True
                break
        if dung:
            break

    print(f"\nMETEOR tốt nhất {tot_nhat:.4f} -> {ra/'tot_nhat'}")
    print(f"đường eval: {ra/'lich_su.json'}")


if __name__ == "__main__":
    main()
