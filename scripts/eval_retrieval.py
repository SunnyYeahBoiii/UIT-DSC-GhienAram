"""Đo retrieval trên weak gold — mốc để mọi thay đổi sau này đấu với nó.

Tách số theo KÊNH gold, không chỉ báo một con số tổng:

- `nguyen_van` — answer trích nguyên văn corpus. Câu hỏi và văn bản chung nhiều chữ,
  BM25 có lợi thế sẵn.
- `so_hieu` — answer diễn đạt lại, chỉ còn trích dẫn số hiệu. Nhóm này mới đo được
  BM25 có thật sự mù khi lệch từ vựng hay không.

Gộp hai nhóm vào một con số sẽ giấu mất chênh lệch giữa chúng, mà chênh lệch đó chính
là thứ quyết định có đáng đổi params lấy dense/rerank hay không.

Cũng in THỨ HẠNG của gold trong pool sâu: trượt@10 vì xếp sai hạng (rerank chữa được)
khác hẳn trượt vì không với tới (rerank bó tay).

Chạy: PYTHONPATH=src python scripts/eval_retrieval.py [n_mau] [pool]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from legalqa.bm25 import ChunkIndex
from legalqa.chunking import Chunk

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))
_K = (1, 3, 5, 10, 20, 50)


def load_index() -> tuple[ChunkIndex, dict[str, str]]:
    """Dựng (hoặc nạp lại) BM25 index. Cache pickle vì build mất ~50s, còn eval chạy
    nhiều lần khi dò tham số."""
    cache = CLEAN_DIR / "bm25.pkl"
    cid2doc = {}
    chunks = []
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid2doc[d["chunk_id"]] = d["doc_id"]
            if not cache.exists():
                chunks.append(Chunk(**{k: d[k] for k in ("chunk_id", "doc_id", "start", "end", "text")}))
    if cache.exists():
        return ChunkIndex.load(cache), cid2doc
    idx = ChunkIndex()
    idx.build(chunks)
    idx.save(cache)
    return idx, cid2doc


def main() -> None:
    n_mau = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    pool = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    gold_file = os.environ.get("GOLD_FILE", "gold.jsonl")
    gold = [json.loads(l) for l in (CLEAN_DIR / gold_file).open(encoding="utf-8")]
    random.seed(0)
    if n_mau < len(gold):
        gold = random.sample(gold, n_mau)

    t = time.time()
    idx, cid2doc = load_index()
    print(f"index sẵn sàng {time.time()-t:.0f}s · {len(gold):,} câu · pool {pool} chunk\n")

    hit: dict[str, Counter] = defaultdict(Counter)
    tong: Counter = Counter()
    bucket: Counter = Counter()

    for g in gold:
        gd = set(g["gold_docs"])
        seen: list[str] = []
        for cid, _ in idx.search(g["question"], top_k=pool):
            d = cid2doc[cid]
            if d not in seen:
                seen.append(d)
        rank = next((i + 1 for i, d in enumerate(seen) if d in gd), None)
        # Câu có gold cả hai kênh được tính vào cả hai: chúng là hai lát cắt của cùng tập,
        # không phải phân hoạch.
        nhom = [*g["kenh"], "TẤT CẢ"]
        if g["kenh"] == ["so_hieu"]:
            nhom.append("chỉ_so_hieu")
        for nh in nhom:
            tong[nh] += 1
            for k in _K:
                if rank is not None and rank <= k:
                    hit[nh][k] += 1
        if rank is None:
            bucket["không với tới"] += 1
        elif rank <= 10:
            bucket["≤10"] += 1
        elif rank <= 50:
            bucket["11-50"] += 1
        else:
            bucket["51+"] += 1

    print("== BM25 chunk 256/256 ==")
    print(f"{'nhóm':<14}{'n':>7}" + "".join(f"{'R@'+str(k):>8}" for k in _K))
    for nh in ("TẤT CẢ", "nguyen_van", "so_hieu", "chỉ_so_hieu"):
        if not tong[nh]:
            continue
        row = "".join(f"{100*hit[nh][k]/tong[nh]:7.1f}%" for k in _K)
        print(f"{nh:<14}{tong[nh]:>7,}{row}")

    n = len(gold)
    print("\n== Thứ hạng gold ==")
    for k in ("≤10", "11-50", "51+", "không với tới"):
        print(f"  {k:<16}{bucket[k]:5,}  {100*bucket[k]/n:5.1f}%")


if __name__ == "__main__":
    main()
