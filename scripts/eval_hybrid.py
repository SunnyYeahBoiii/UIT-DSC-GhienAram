"""Đo TRẦN của pool: BM25 thuần so với hybrid BM25+dense.

Chỉ đo một đại lượng — gold có nằm trong top-k Document lấy từ pool hay không. KHÔNG
chạy cross-encoder trong vòng lặp, nhanh hơn ~50 lần.

Vì sao chỉ cần trần: §15.7 đo được reranker đã bão hoà pool (pool 20, k=10 ăn 100%
khoảng trống, R@10 đúng bằng trần). Năng lực xếp hạng hết là chỗ nghẽn, nên đường duy
nhất còn lại của dense là kéo gold từ hạng 51+ vào top-50. Reranker sẽ lo phần sau, với
hiệu suất đã biết: ăn ~72% khoảng trống, tức +1 điểm trần ≈ +0,7 điểm R@5.

Mốc phải vượt KHÔNG phải BM25 pool 50 (trần 96,2%) mà là BM25 pool 100 (trần 97,3%) —
pool 100 cũng nâng trần và tốn 0 params. Ngưỡng chốt trước khi chạy:

    < 96,2%    dense kéo tụt, vứt
    96,2-97,3  không hơn pool 100 vốn miễn phí params, vứt
    >= 98%     +1,3 điểm R@5, đáng 135M params, gắn

Chạy: PYTHONPATH=src python scripts/eval_hybrid.py --dense-model <path> --n-cau 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from legalqa.dense import BiEncoder, rrf

from eval_retrieval import load_index  # cùng thư mục scripts/

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))
_K = (10, 20, 50, 100)


def _tran(chunk_ids: list[str], cid2doc: dict[str, str], gold: set[str], k: int) -> bool:
    """Gold có trong k Document đầu không. Gộp chunk về Document trước khi cắt k —
    cắt k trên chunk sẽ đếm nhầm: 50 chunk hay chỉ ra chưa tới 50 Document."""
    seen: list[str] = []
    for cid in chunk_ids:
        d = cid2doc[cid]
        if d not in seen:
            seen.append(d)
            if len(seen) >= k:
                break
    return bool(set(seen) & gold)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    p.add_argument("--n-cau", type=int, default=1000)
    p.add_argument("--deep", type=int, default=1000, help="độ sâu mỗi nhánh trước khi trộn")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gold", default="gold.jsonl")
    # 256 từ tiếng Việt ≈ 380-520 subword (xem rerank.py). Lượt đo đầu để max_len 256 nên
    # cắt mất ~40% mỗi chunk — mà đuôi chunk hay chứa đúng khoản cần tìm. 512 giữ gần trọn.
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--prefix", default="", choices=["", "e5", "none"], help="rỗng = suy từ tên")
    p.add_argument("--cache", default="", help="tên file .npy; rỗng = suy từ tên model")
    a = p.parse_args()

    gold = [json.loads(l) for l in (CLEAN_DIR / a.gold).open(encoding="utf-8")]
    random.seed(0)  # cùng seed với eval_retrieval/eval_rerank -> cùng tập câu
    if a.n_cau < len(gold):
        gold = random.sample(gold, a.n_cau)

    idx, cid2doc = load_index()
    ids: list[str] = []
    texts: list[str] = []
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["chunk_id"])
            texts.append(d["text"])
    print(f"{len(ids):,} chunk · {len(gold):,} câu · {a.dense_model}", flush=True)

    enc = BiEncoder(
        a.dense_model,
        batch_size=a.batch_size,
        max_len=a.max_len,
        prefix=a.prefix or None,
    )

    # Encode một lần rồi cache: 287k chunk mất 20-40 phút T4, còn đo thì chạy nhiều lần
    # khi dò rrf-k và deep.
    # max_len PHẢI nằm trong tên cache: cùng model, khác max_len là hai ma trận khác nhau.
    # Thiếu nó thì lượt 512 nạp lại đúng file 256 của lượt trước và in ra số y nguyên.
    name = a.cache or f"dense_{a.dense_model.rstrip('/').split('/')[-1]}_L{a.max_len}.npy"
    cache = CLEAN_DIR / name
    if cache.exists():
        M = np.load(cache)
        print(f"nạp lại {cache.name} {M.shape}", flush=True)
    else:
        t = time.time()
        M = enc.encode(texts)
        np.save(cache, M)
        print(f"encode {M.shape} {time.time()-t:.0f}s -> {cache.name}", flush=True)

    Q = enc.encode([g["question"] for g in gold], is_query=True).astype(np.float32)
    Mf = M.astype(np.float32)

    hit = {n: {k: 0 for k in _K} for n in ("BM25", "dense", "hybrid")}
    t0 = time.time()
    for i, g in enumerate(gold, 1):
        gd = set(g["gold_docs"])

        bm = [cid for cid, _ in idx.search(g["question"], top_k=a.deep)]
        # argpartition thay vì argsort: chỉ cần deep phần tử lớn nhất trong 287k.
        sim = Mf @ Q[i - 1]
        top = np.argpartition(-sim, a.deep)[: a.deep]
        de = [ids[j] for j in top[np.argsort(-sim[top])]]

        # Trộn theo chunk_id: BM25 và dense không cùng miền chỉ số.
        hy = rrf([bm, de], k=a.rrf_k)

        for ten, lst in (("BM25", bm), ("dense", de), ("hybrid", hy)):
            for k in _K:
                if _tran(lst, cid2doc, gd, k):
                    hit[ten][k] += 1

        if i % 200 == 0:
            print(f"  {i}/{len(gold)}  {(time.time()-t0)/i:.3f}s/câu", flush=True)

    n = len(gold)
    print(f"\n== TRẦN (gold trong top-k Document) · n={n} · deep {a.deep} · rrf-k {a.rrf_k} ==")
    print(f"{'':<10}" + "".join(f"{'@'+str(k):>9}" for k in _K))
    for ten in ("BM25", "dense", "hybrid"):
        print(f"{ten:<10}" + "".join(f"{100*hit[ten][k]/n:8.1f}%" for k in _K))
    print(f"{'chênh':<10}" + "".join(f"{100*(hit['hybrid'][k]-hit['BM25'][k])/n:+8.1f}%" for k in _K))

    d50 = 100 * (hit["hybrid"][50] - hit["BM25"][50]) / n
    print(f"\npool 50: trần {100*hit['BM25'][50]/n:.1f}% -> {100*hit['hybrid'][50]/n:.1f}%")
    print(f"  reranker ăn ~72% khoảng trống (§15.6) -> ước R@5 {0.72*d50:+.1f} điểm")
    print("  mốc phải vượt: BM25 pool 100 = 97,3% trần, giá 0 params")
    print(f"\n{(time.time()-t0)/n:.3f}s/câu")


if __name__ == "__main__":
    main()
