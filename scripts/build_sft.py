"""Dựng tập SFT: câu hỏi -> prompt (Evidence thật từ retrieval) -> Reference answer.

Vì sao phải chạy retrieval chứ không dùng thẳng gold: `data/official/train.json` chỉ có
hai trường `question` và `answer` — KHÔNG có context. Context lúc inference do retrieval
sinh ra, có nhiễu, thiếu Điều, sai thứ tự. Train trên gold sạch rồi thả vào context nhiễu
là train-inference mismatch: model chưa từng thấy căn cứ sai nằm cạnh căn cứ đúng.

Nên dùng chính `Retriever` mà inference dùng, cùng tham số. Khác một điểm duy nhất:
mẫu nào retrieval không lấy được document weak-gold thì BỎ, không chèn gold vào (xem
`--gold` bên dưới).

Chạy đo ngân sách (không cần GPU, BM25 thuần):
    QWEN_TOKENIZER=... PYTHONPATH=src python scripts/build_sft.py --do-token --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

from legalqa.bm25 import ChunkIndex
from legalqa.chunking import Chunk
from legalqa.corpus import Article, Document
from legalqa.prompt import dung_prompt
from legalqa.retrieve import Retriever

CLEAN = Path("data/clean")


def nap_docs() -> dict[str, Document]:
    """Dựng lại Document từ corpus.jsonl — passage trong đó ĐÃ qua `dewrap_passage()`
    lúc `clean_data.py` ghi, nên mọi offset Article khớp sẵn. Đọc lại raw JSON ở đây là
    lệch offset toàn bộ."""
    docs: dict[str, Document] = {}
    with (CLEAN / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d = Document(
                doc_id=o["doc_id"], passage=o["passage"], slug=o["slug"], link=o["link"]
            )
            d.articles = [
                Article(*a) if isinstance(a, list) else Article(**a)
                for a in o["articles"]
            ]
            docs[d.doc_id] = d
    return docs


def nap_chunks() -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    with (CLEAN / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            c = Chunk(**o) if isinstance(o, dict) else Chunk(*o)
            chunks[c.chunk_id] = c
    return chunks


def nap_gold() -> dict[str, set[str]]:
    g = {}
    with (CLEAN / "gold.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            g[o["qid"]] = set(o["gold_docs"])
    return g


def do_token(rows: list[dict], r: Retriever, tok, top_n: int, max_chars: int) -> None:
    """Đo độ dài prompt THẬT để chốt `max_seq_len`, trước khi sinh cả tập.

    Không đoán từ số ký tự: tỉ lệ token/ký tự của tiếng Việt luật khác hẳn tiếng Anh, và
    `max_chars=24_000` kế thừa từ v9.2 (configs/kaggle_inference_t4.yaml) được đặt cho
    context Qwen 3B chứ không cho seq len lúc train.
    """
    gold = nap_gold()
    dai = []
    co_gold = trung_gold = 0
    for row in rows:
        evs = r.evidence(row["question"], top_n=top_n, max_chars=max_chars)
        g = gold.get(row["id"])
        if g:
            co_gold += 1
            trung_gold += any(e.doc_id in g for e in evs)
        p = dung_prompt(row["question"], evs)
        dai.append((len(tok.encode(p).ids), len(tok.encode(row["answer"]).ids), len(evs)))
    pr = sorted(x for x, _, _ in dai)
    tong = sorted(x + y for x, y, _ in dai)
    n = len(pr)
    q = lambda xs, k: xs[int(k * (len(xs) - 1))]
    print(f"n={n}  top_n={top_n}  max_chars={max_chars}")
    print(f"  gold trong Evidence: {trung_gold}/{co_gold} ({trung_gold/max(1,co_gold):.1%})")
    print(f"  Evidence/câu : mean {statistics.mean(z for _, _, z in dai):.1f}")
    print(f"  prompt token : p50 {q(pr,.5)} p90 {q(pr,.9)} p95 {q(pr,.95)} max {pr[-1]}")
    print(f"  prompt+answer: p50 {q(tong,.5)} p90 {q(tong,.9)} p95 {q(tong,.95)} max {tong[-1]}")
    for cap in (2048, 3072, 4096, 6144, 8192):
        v = sum(1 for x in tong if x > cap)
        print(f"  vượt {cap:5d}: {v} ({v/n:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-token", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="0 = cả tập")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=24_000)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (CLEAN / "train_sft.jsonl").open(encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]
    r = Retriever(
        index=ChunkIndex.load(CLEAN / "bm25.pkl"),
        chunks=nap_chunks(),
        docs=nap_docs(),
    )
    if a.do_token:
        from tokenizers import Tokenizer

        do_token(rows, r, Tokenizer.from_file(os.environ["QWEN_TOKENIZER"]),
                 a.top_n, a.max_chars)
        return
    raise SystemExit("chưa cài phần sinh tập — chạy --do-token trước để chốt ngân sách")


if __name__ == "__main__":
    main()
