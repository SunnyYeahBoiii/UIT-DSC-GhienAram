"""Đo BM25 + cross-encoder rerank, đấu thẳng với mốc BM25 thuần.

In cùng định dạng bảng với `scripts/eval_retrieval.py` để so trực tiếp, kèm hai mốc
kẹp trên/dưới mà mọi kết quả rerank phải nằm giữa:

- **sàn** = BM25 tại cùng k. Rerank tệ hơn sàn nghĩa là cross-encoder chấm nhiễu hơn
  BM25, phải bỏ.
- **trần** = BM25 R@K của chính pool đưa vào rerank. Rerank chỉ sắp lại pool đó, không
  thêm ứng viên mới, nên không bao giờ vượt được. Đo được 96,7% ở K=50.

Rerank ăn được bao nhiêu phần của khoảng (sàn, trần) mới là con số quyết định có đáng
568M params hay không — R@5 tuyệt đối nói lên rất ít.

Cũng in giây/câu: bài thi 1.000 câu, mỗi câu K cặp qua XLM-R large. Nếu K=50 mà tốn
quá lâu trên T4 thì phải hạ K, và hạ K là hạ trần — đánh đổi phải nhìn thấy bằng số.

Chạy (cần GPU + transformers, máy dev macOS không chạy được):
    PYTHONPATH=src python scripts/eval_rerank.py --model models/Vietnamese_Reranker
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from legalqa.chunking import Chunk
from legalqa.dense import rrf
from legalqa.rerank import CrossEncoderReranker

from eval_retrieval import load_index  # cùng thư mục scripts/

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))
_K = (1, 3, 5, 10, 20, 50)


def _doc_theo_thu_tu(doc_ids: list[str]) -> list[str]:
    """Gộp chunk về Document, giữ thứ hạng của chunk tốt nhất mỗi Document."""
    seen: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Vietnamese_Reranker")
    p.add_argument("--pool", type=int, default=50, help="số chunk BM25 đưa vào rerank")
    p.add_argument("--n-cau", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gold", default="gold.jsonl", help="gold_warmup.jsonl = tập độc lập")
    # Pool hybrid: §15.8 nâng trần 96,7 -> 97,9. Ước +0,9 điểm R@5 qua hệ số 0,72 của
    # §15.6 — cờ này đo con số THẬT thay vì ước.
    p.add_argument("--dense", default="", help="tên .npy trong CLEAN_DIR; rỗng = pool BM25 thuần")
    p.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    p.add_argument("--deep", type=int, default=1000)
    p.add_argument("--rrf-k", type=int, default=60)
    a = p.parse_args()

    gold = [json.loads(l) for l in (CLEAN_DIR / a.gold).open(encoding="utf-8")]
    random.seed(0)  # cùng seed với eval_retrieval.py -> cùng tập câu, so được
    if a.n_cau < len(gold):
        gold = random.sample(gold, a.n_cau)

    idx, cid2doc = load_index()
    # Chunk đầy đủ để lấy text đưa vào cross-encoder; BM25 chỉ trả chunk_id.
    by_id: dict[str, Chunk] = {}
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_id[d["chunk_id"]] = Chunk(
                **{k: d[k] for k in ("chunk_id", "doc_id", "start", "end", "text")}
            )

    # --dense/--dense-model nhận danh sách phẩy: mỗi nhánh là một bi-encoder riêng,
    # RRF trộn theo HẠNG nên không cần chung số chiều (e5 384 + AITeamVN 1024 trộn được).
    nhanh_dense: list[tuple] = []
    if a.dense:
        from legalqa.dense import BiEncoder, DenseIndex

        for npy, mp in zip(a.dense.split(","), a.dense_model.split(",")):
            stem = npy[:-4] if npy.endswith(".npy") else npy
            nhanh_dense.append((
                DenseIndex.load(CLEAN_DIR / f"{stem}.npy", CLEAN_DIR / f"{stem}.ids"),
                BiEncoder(mp, batch_size=256, max_len=256),
            ))

    rr = CrossEncoderReranker(a.model, batch_size=a.batch_size)
    nhanh = f"BM25+{len(nhanh_dense)} dense (RRF)" if nhanh_dense else "BM25"
    print(f"{len(gold):,} câu · pool {a.pool} chunk từ {nhanh} · {a.model}\n", flush=True)

    hit_bm25: Counter = Counter()
    hit_rr: Counter = Counter()
    tran = 0
    t0 = time.time()
    for i, g in enumerate(gold, 1):
        gd = set(g["gold_docs"])
        if not nhanh_dense:
            res = idx.search(g["question"], top_k=a.pool)
        else:
            xep = [[cid for cid, _ in idx.search(g["question"], top_k=a.deep)]]
            for di, en in nhanh_dense:
                qv = en.encode([g["question"]], is_query=True)[0]
                xep.append([cid for cid, _ in di.search(qv, top_k=a.deep)])
            # Điểm BM25 mất nghĩa sau khi trộn theo hạng; truyền 0.0 vì §15.3 đo rerank
            # thuần, không hoà điểm.
            res = [(cid, 0.0) for cid in rrf(xep, k=a.rrf_k)[: a.pool]]
        cands = [(by_id[cid], s) for cid, s in res]

        base = _doc_theo_thu_tu([c.doc_id for c, _ in cands])
        if set(base) & gd:
            tran += 1
        for k in _K:
            if set(base[:k]) & gd:
                hit_bm25[k] += 1

        ranked = rr.rerank(g["question"], cands)
        after = _doc_theo_thu_tu([s.chunk.doc_id for s in ranked])
        for k in _K:
            if set(after[:k]) & gd:
                hit_rr[k] += 1

        if i % 50 == 0:
            print(f"  {i}/{len(gold)}  {(time.time()-t0)/i:.2f}s/câu", flush=True)

    n = len(gold)
    print(f"\n== pool {a.pool} · n={n} ==")
    print(f"{'':<10}" + "".join(f"{'R@'+str(k):>9}" for k in _K))
    for ten, h in (("BM25", hit_bm25), ("+rerank", hit_rr)):
        print(f"{ten:<10}" + "".join(f"{100*h[k]/n:8.1f}%" for k in _K))
    print(f"{'chênh':<10}" + "".join(f"{100*(hit_rr[k]-hit_bm25[k])/n:+8.1f}%" for k in _K))

    print(f"\ntrần (gold có trong pool) {100*tran/n:.1f}%")
    for k in (1, 5, 10):
        con = tran - hit_bm25[k]
        an = hit_rr[k] - hit_bm25[k]
        pct = f"{100*an/con:.0f}%" if con else "—"
        print(f"  k={k:<3} khoảng trống {con:4d} câu · rerank ăn {an:+4d} = {pct}")
    print(f"\n{(time.time()-t0)/n:.2f}s/câu · 1.000 câu ≈ {(time.time()-t0)/n*1000/60:.0f} phút")


if __name__ == "__main__":
    main()
