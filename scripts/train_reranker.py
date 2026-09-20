"""Fine-tune cross-encoder listwise trên pool hybrid thật do mine_rerank.py dựng.

Loss listwise (1 positive + G-1 negative, softmax trên G điểm) chứ không BCE từng cặp:
lúc infer reranker chỉ phải XẾP HẠNG trong pool, điểm tuyệt đối không dùng. BCE ép điểm
về 0/1 cho từng cặp — thừa ràng buộc, và nhãn weak gold nhiễu thì BCE phạt nặng hơn.

lr thấp + 1 epoch: checkpoint gốc đã ăn ~72% khoảng trống pool (§15.6). Mục tiêu là chỉnh
cho miền luật VN của đề, không học lại từ đầu — lr cao là quên năng lực tổng quát.

Chạy: PYTHONPATH=src python scripts/train_reranker.py --model AITeamVN/Vietnamese_Reranker \
          --data data/clean/rerank_train.jsonl --out runs/rerank_ft
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from legalqa.rerank import _sach  # noqa: E402

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="AITeamVN/Vietnamese_Reranker")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--group", type=int, default=16, help="1 positive + group-1 negative")
    p.add_argument("--bs", type=int, default=4, help="số câu hỏi mỗi bước")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = [r for r in rows if r["neg"]]
    texts: dict[str, str] = {}
    can = {c for r in rows for c in r["pos"] + r["neg"]}
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["chunk_id"] in can:
                # cùng lọc ký tự như lúc infer (rerank.score): ký tự điều khiển lẫn
                # trong passage crawl làm rust tokenizers panic và abort tiến trình
                texts[d["chunk_id"]] = _sach(d["text"])

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).cuda()
    # Checkpointing: 4 câu × 16 cặp × 512 token qua XLM-R large không vừa 32 GB nếu giữ
    # đủ activation; chậm ~30% nhưng không phải hạ group — group nhỏ là ít negative.
    model.gradient_checkpointing_enable()
    model.train()

    n_buoc = math.ceil(len(rows) * a.epochs / (a.bs * a.accum))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * n_buoc))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_buoc - s) / max(1, n_buoc - warm)))
    )
    print(f"{len(rows):,} câu · {n_buoc} bước tối ưu · group {a.group} · lr {a.lr}", flush=True)

    thu_tu: list[dict] = []
    while len(thu_tu) < n_buoc * a.bs * a.accum:
        tron = rows[:]
        random.shuffle(tron)
        thu_tu.extend(tron)

    t0, buoc, loss_tb, dung_tb = time.time(), 0, 0.0, 0.0
    for vi in range(0, n_buoc * a.bs * a.accum, a.bs):
        lo = thu_tu[vi : vi + a.bs]
        q, c = [], []
        for r in lo:
            pos = random.choice(r["pos"])
            k = a.group - 1
            neg = random.sample(r["neg"], k) if len(r["neg"]) >= k else random.choices(r["neg"], k=k)
            for cid in [pos] + neg:  # positive luôn ở cột 0 -> nhãn CE = 0
                q.append(_sach(r["question"]))
                c.append(texts[cid])
        enc = tok(q, c, padding=True, truncation="only_second", max_length=a.max_len,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits[:, 0].float().view(len(lo), a.group)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(lo), dtype=torch.long, device="cuda"))
        (loss / a.accum).backward()
        loss_tb += loss.item() / a.accum
        dung_tb += (logits.argmax(1) == 0).float().mean().item() / a.accum

        if (vi // a.bs + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            buoc += 1
            if buoc % 20 == 0 or buoc == n_buoc:
                print(f"  b{buoc}/{n_buoc} loss {loss_tb:.4f} top1 {dung_tb:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/buoc:.2f}s/bước", flush=True)
            loss_tb = dung_tb = 0.0

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"xong {n_buoc} bước {(time.time()-t0)/60:.0f} phút -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
