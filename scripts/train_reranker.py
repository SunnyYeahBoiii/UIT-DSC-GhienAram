"""Fine-tune ViLegalBERT thành cross-encoder rerank, train trên weak gold.

`ntphuc149/ViLegalBERT` là encoder pretrain MLM domain luật (HF ghi `fill-mask`), KHÔNG
có head relevance — không dùng thẳng làm reranker được. Đổi lại nó chỉ 135M params so
với 568M của `Vietnamese_Reranker`, dư ra 432M cho generator.

Nhãn lấy từ `data/clean/gold.jsonl` (xem `scripts/build_gold.py`): chunk thuộc weak gold
document là dương, còn lại là âm. Nhãn yếu ở mức Document nên có nhiễu — một văn bản gold
dài 4.828 từ trung vị, phần lớn chunk của nó không liên quan tới câu hỏi. Chấp nhận nhiễu
này vì dữ liệu thi không có nhãn nào mịn hơn.

Chạy trên Kaggle T4 (máy dev macOS không có CUDA):
    PYTHONPATH=src python scripts/train_reranker.py --model models/ViLegalBERT
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from legalqa.bm25 import ChunkIndex
from legalqa.chunking import Chunk

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))


def nap_ung_vien(top_k: int, n_cau: int) -> list[tuple[str, list[Chunk], set[str]]]:
    """(câu hỏi, top-K chunk của BM25, tập gold doc) cho từng câu có weak gold."""
    by_id: dict[str, Chunk] = {}
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_id[d["chunk_id"]] = Chunk(
                **{k: d[k] for k in ("chunk_id", "doc_id", "start", "end", "text")}
            )
    idx = ChunkIndex.load(CLEAN_DIR / "bm25.pkl")

    gold = [json.loads(l) for l in (CLEAN_DIR / "gold.jsonl").open(encoding="utf-8")]
    random.seed(0)
    random.shuffle(gold)
    out = []
    for g in gold[:n_cau]:
        cands = [by_id[cid] for cid, _ in idx.search(g["question"], top_k=top_k)]
        out.append((g["question"], cands, set(g["gold_docs"])))
    return out


def chon_negative(
    cands: list[Chunk], gold_docs: set[str], n_neg: int, rng: random.Random
) -> list[Chunk]:
    """Chọn chunk âm để train từ top-K của BM25.

    TODO(người dùng): quyết định chiến lược ở đây.

    Ba đánh đổi thật, có số đo kèm:

    1. **Hard vs dễ.** Âm lấy từ hạng cao (1-10) là thứ reranker PHẢI học phân biệt —
       chính 11,2% trượt@10 hiện tại. Nhưng hạng 1-10 cũng là nơi weak gold hay sai:
       gold chỉ ở mức Document, một chunk "âm" hạng 3 có thể đúng hơn cả chunk gold.
       Âm lấy từ hạng 30-50 sạch nhãn hơn nhưng dạy được ít.

    2. **Bản sao qua sửa đổi.** EDA §3: trong 224 câu khớp nguyên văn, 29 câu có ≥3
       Document cùng chứa đoạn đó — văn bản sửa đổi chép nguyên văn bản gốc. Một chunk
       ở Document khác có thể TRÙNG CHỮ với gold. Dạy model rằng nó sai là dạy sai.
       Lọc được bằng cách bỏ chunk có ≥12 từ liên tiếp trùng chunk gold.

    3. **Tỉ lệ âm/dương.** Mỗi câu ~1-2 chunk dương trong top-K, âm thì hàng chục. Lấy
       hết thì lệch lớp nặng; lấy ít thì phí tín hiệu. `n_neg` 4-8 là khoảng thường dùng.

    Trả về tối đa `n_neg` Chunk KHÔNG thuộc `gold_docs`.
    """
    raise NotImplementedError("chọn chiến lược negative rồi implement")


def dung_cap(
    data: list[tuple[str, list[Chunk], set[str]]], n_neg: int
) -> list[tuple[str, str, int]]:
    """(query, text, nhãn) — nhãn 1 nếu chunk thuộc weak gold document."""
    rng = random.Random(0)
    pairs = []
    for q, cands, gd in data:
        pos = [c for c in cands if c.doc_id in gd]
        if not pos:
            continue  # BM25 trượt hẳn — không có dương để học, bỏ câu
        for c in pos:
            pairs.append((q, c.text, 1))
        for c in chon_negative(cands, gd, n_neg * len(pos), rng):
            pairs.append((q, c.text, 0))
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ViLegalBERT")
    p.add_argument("--out", default="artifacts/vilegal_reranker")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--n-cau", type=int, default=6000)
    p.add_argument("--n-neg", type=int, default=6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=512)
    a = p.parse_args()

    data = nap_ung_vien(a.top_k, a.n_cau)
    pairs = dung_cap(data, a.n_neg)
    n_pos = sum(1 for _, _, y in pairs if y)
    print(f"cặp train {len(pairs):,}  dương {n_pos:,}  âm {len(pairs)-n_pos:,}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()

    def collate(batch):
        qs, ts, ys = zip(*batch)
        enc = tok(
            list(qs), list(ts), padding=True, truncation="only_second",
            max_length=a.max_len, return_tensors="pt",
        )
        enc["labels"] = torch.tensor(ys)
        return enc

    dl = DataLoader(pairs, batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    steps = int(len(dl) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    done = 0
    while done < steps:
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            done += 1
            if done % 100 == 0:
                print(f"  step {done}/{steps}  loss {loss.item():.4f}", flush=True)
            if done >= steps:
                break

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
