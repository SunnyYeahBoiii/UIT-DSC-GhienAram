"""Encode toàn bộ Chunk thành ma trận dense, ghi ra .npy + danh sách chunk_id.

Tách khỏi eval_hybrid.py vì encode là bước ĐẮT và DÙNG LẠI: 436s trên T4 cho 287.358
chunk ở max_len 256, còn đo thì chạy nhiều lần khi dò pool/rrf-k. Bài thi cũng cần
đúng file này.

max_len 256 là số đã chốt ở §15.8, không phải mặc định tiện tay: 512 làm trần hybrid
@50 TỤT 0,4 điểm vì mean pooling bị loãng, lại tốn gấp đôi thời gian.

Chạy: PYTHONPATH=src python scripts/build_dense.py --model intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from legalqa.dense import BiEncoder

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--max-len", type=int, default=256, help="§15.8: 512 tệ hơn VÀ chậm hơn")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--prefix", default="", choices=["", "e5", "none"])
    p.add_argument("--out", default="dense")
    a = p.parse_args()

    ids: list[str] = []
    texts: list[str] = []
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["chunk_id"])
            texts.append(d["text"])
    print(f"{len(ids):,} chunk · {a.model} · max_len {a.max_len}", flush=True)

    enc = BiEncoder(a.model, batch_size=a.batch_size, max_len=a.max_len, prefix=a.prefix or None)
    t = time.time()
    M = enc.encode(texts)
    npy = CLEAN_DIR / f"{a.out}.npy"
    np.save(npy, M)
    # chunk_id ghi riêng: thứ tự dòng khớp thứ tự hàng ma trận. Ghi vào .npz cùng nhau
    # thì mỗi lần đọc id phải nạp cả 0,22 GB ma trận.
    (CLEAN_DIR / f"{a.out}.ids").write_text("\n".join(ids), encoding="utf-8")
    print(f"encode {M.shape} {time.time()-t:.0f}s -> {npy.name} + {a.out}.ids")


if __name__ == "__main__":
    main()
