"""Dựng dữ liệu fine-tune reranker: pool hybrid 50 thật + nhãn chunk cho từng câu train.

Vì sao fine-tune reranker trước dense: trần pool hybrid@50 = 97,9% (§15.8) nhưng rerank
R@5 = 93,6% — reranker bỏ ~4,3 điểm, retrieval chỉ ~2,1. Negative lấy từ CHÍNH pool
hybrid vì đó đúng phân phối reranker gặp lúc infer; negative ngẫu nhiên quá dễ.

Nhãn: gold ở mức Document, index ở mức Chunk — không phải chunk nào của văn bản gold
cũng trả lời. Positive = chunk chứa một shingle 12 từ của Reference answer (cùng cửa sổ
build_gold.py). Không chunk nào khớp thì lấy chunk văn bản gold có điểm teacher cao
nhất. Negative = chunk ngoài văn bản gold VÀ không chứa shingle answer — chunk khác văn
bản mà chép nguyên văn answer là văn bản sửa đổi/trùng nội dung, ép nó xa là dạy sai.

Tập giữ lại: `random.seed(0); random.sample(gold, 1000)` — đúng tập eval_rerank /
eval_hybrid dùng. Bỏ khỏi train để số đo sau fine-tune không đo trên câu đã học.

Chạy: PYTHONPATH=src:scripts python scripts/mine_rerank.py --model AITeamVN/Vietnamese_Reranker
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
from legalqa.rerank import CrossEncoderReranker

from eval_retrieval import load_index  # cùng thư mục scripts/

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))
TRAIN = Path(os.environ.get("TRAIN_JSON", "data/official/train.json"))
_W = 12


def _shingles(text: str) -> set[str]:
    w = text.split()
    return {" ".join(w[i : i + _W]) for i in range(len(w) - _W + 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="AITeamVN/Vietnamese_Reranker")
    # Hai nhánh dense chạy CÙNG lúc: e5-small 118M (đa ngữ, prefix query/passage) và
    # AITeamVN 568M (chuyên luật VN). RRF trộn hạng nên không cần chung số chiều, và
    # 118M+568M vẫn dưới trần khi pipeline chỉ còn MỘT reranker.
    p.add_argument("--dense", default="dense_multilingual-e5-small_L256.npy,dense_Vietnamese_Embedding_L256.npy")
    p.add_argument("--dense-model", default="intfloat/multilingual-e5-small,AITeamVN/Vietnamese_Embedding")
    p.add_argument("--pool", type=int, default=50)
    p.add_argument("--deep", type=int, default=1000)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--giu-lai", type=int, default=1000)
    p.add_argument("--gioi-han", type=int, default=0, help=">0: chỉ lấy N câu đầu (thử nhanh)")
    p.add_argument("--shard", default="0/1", help="k/n: lấy câu thứ k, k+n, ... — chạy n tiến trình song song")
    p.add_argument("--out", default="rerank_train.jsonl")
    a = p.parse_args()

    gold = [json.loads(l) for l in (CLEAN_DIR / "gold.jsonl").open(encoding="utf-8")]
    random.seed(0)
    giu = {g["qid"] for g in random.sample(gold, a.giu_lai)}
    gold = [g for g in gold if g["qid"] not in giu]
    if a.gioi_han:
        gold = gold[: a.gioi_han]
    # Mine nghẽn CPU (BM25 deep 1000), GPU rảnh ~40% — chia tiến trình SAU khi tách tập giữ
    # lại để tập giữ lại không đổi theo số shard.
    k, n = map(int, a.shard.split("/"))
    gold = gold[k::n]
    answers = json.loads(TRAIN.read_text(encoding="utf-8"))

    idx, cid2doc = load_index()
    ids: list[str] = []
    texts: dict[str, str] = {}
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["chunk_id"])
            texts[d["chunk_id"]] = d["text"]
    # Encode câu hỏi cho từng nhánh rồi THẢ encoder: hai bi-encoder cùng nằm trên GPU
    # với cross-encoder 568M là thừa, mà chỉ cần ma trận Q một lần.
    nhanh: list[tuple[np.ndarray, np.ndarray]] = []
    cau = [g["question"] for g in gold]
    for npy, mp in zip(a.dense.split(","), a.dense_model.split(",")):
        M = np.load(CLEAN_DIR / npy).astype(np.float32)
        enc = BiEncoder(mp, batch_size=256, max_len=256)
        nhanh.append((M, enc.encode(cau, is_query=True).astype(np.float32)))
        del enc
    rr = CrossEncoderReranker(a.model, batch_size=64)
    print(f"train {len(gold):,} câu (giữ lại {len(giu):,}) · pool {a.pool}", flush=True)

    dem = {"ok": 0, "khop_answer": 0, "khong_gold_trong_pool": 0}
    t0 = time.time()
    with (CLEAN_DIR / a.out).open("w", encoding="utf-8") as g_out:
        for i, g in enumerate(gold, 1):
            xep = [[cid for cid, _ in idx.search(g["question"], top_k=a.deep)]]
            for M, Q in nhanh:
                sim = M @ Q[i - 1]
                top = np.argpartition(-sim, a.deep)[: a.deep]
                xep.append([ids[j] for j in top[np.argsort(-sim[top])]])
            pool = rrf(xep, k=a.rrf_k)[: a.pool]

            gd = set(g["gold_docs"])
            sh = _shingles(answers.get(g["qid"], {}).get("answer") or "")
            chua_ans = [bool(sh & _shingles(texts[c])) for c in pool]
            la_gold = [cid2doc[c] in gd for c in pool]
            if not any(la_gold):
                dem["khong_gold_trong_pool"] += 1
                continue
            # tokenizers 0.23 + transformers 5: một số chunk làm rust panic
            # (AddedVocabulary bad split) — bỏ câu đó, không chết cả shard.
            try:
                teacher = rr.score(g["question"], [texts[c] for c in pool])
            except BaseException as e:
                print(f"  skip {g['qid']} {type(e).__name__}: {e}", flush=True)
                continue

            pos = [c for c, lg, ca in zip(pool, la_gold, chua_ans) if lg and ca]
            if pos:
                dem["khop_answer"] += 1
            else:
                pos = [max((c for c, lg in zip(pool, la_gold) if lg), key=lambda c: teacher[pool.index(c)])]
            neg = [c for c, lg, ca in zip(pool, la_gold, chua_ans) if not lg and not ca]
            dem["ok"] += 1
            g_out.write(json.dumps({
                "qid": g["qid"], "question": g["question"], "pos": pos, "neg": neg,
                "teacher": dict(zip(pool, teacher)),
            }, ensure_ascii=False) + "\n")

            if i % 200 == 0:
                print(f"  {i}/{len(gold)}  {(time.time()-t0)/i:.2f}s/câu  {dem}", flush=True)
    print(f"xong {dem} -> {CLEAN_DIR / a.out}")


if __name__ == "__main__":
    main()
