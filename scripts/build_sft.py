"""Dựng tập SFT: câu hỏi -> prompt (Evidence thật từ retrieval) -> Reference answer.

Vì sao phải chạy retrieval chứ không dùng thẳng gold: `data/official/train.json` chỉ có
hai trường `question` và `answer` — KHÔNG có context. Context lúc inference do retrieval
sinh ra, có nhiễu, thiếu Điều, sai thứ tự. Train trên gold sạch rồi thả vào context nhiễu
là train-inference mismatch: model chưa từng thấy căn cứ sai nằm cạnh căn cứ đúng.

Nên dùng chính `Retriever` mà inference dùng, cùng tham số.

Chạy đo ngân sách:
    # máy dev, BM25 thuần (cận dưới — không GPU nên không rerank được)
    QWEN_TOKENIZER=... PYTHONPATH=src python scripts/build_sft.py --do-token --n 300
    # Kaggle, có cross-encoder
    QWEN_TOKENIZER=... PYTHONPATH=src python scripts/build_sft.py --do-token --n 300 \
        --model $MODEL --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from legalqa.bm25 import ChunkIndex
from legalqa.chunking import Chunk
from legalqa.corpus import Article, Document
from legalqa.prompt import dung_prompt
from legalqa.retrieve import Retriever, chon_evidence

CLEAN = Path(os.environ.get("CLEAN_DIR", "data/clean"))

# Các cấu hình quét trong MỘT lần rank. Rerank là phần đắt (pool 50 cặp qua XLM-R large
# mỗi câu); `chon_evidence` chỉ đọc lại list đã xếp hạng nên gần như miễn phí. Tính lại
# `rank()` cho từng cấu hình là nhân thời gian GPU lên đúng số cấu hình mà không đổi
# lấy một con số mới nào.
CAU_HINH = ((5, 24_000), (5, 12_000), (3, 12_000), (5, 8_000), (3, 8_000))


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
    keys = ("chunk_id", "doc_id", "start", "end", "text")
    chunks: dict[str, Chunk] = {}
    with (CLEAN / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            c = Chunk(**{k: o[k] for k in keys})
            chunks[c.chunk_id] = c
    return chunks


def nap_index(chunks: dict[str, Chunk]) -> ChunkIndex:
    """Nạp bm25.pkl, tự build nếu chưa có.

    `build_chunks.py` KHÔNG sinh bm25.pkl — nó do `eval_retrieval.load_index()` dựng
    rồi cache. Trên Kaggle chỉ chạy clean/chunk/gold nên file không tồn tại, và gọi
    thẳng `ChunkIndex.load` là chết sau 26 phút dựng dữ liệu, đúng ở bước cuối cùng.
    Build mất ~50s, rẻ hơn nhiều so với mất cả lượt chạy.
    """
    if (CLEAN / "bm25.pkl").exists():
        return ChunkIndex.load(CLEAN / "bm25.pkl")
    # Ghi cache cạnh CLEAN nếu ghi được, không thì vào cwd: khi CLEAN là output của
    # kernel trước thì nó nằm dưới /kaggle/input, read-only. Build xong mà chết ở
    # `save` là mất trắng 50s và cả lượt chạy.
    cache = CLEAN / "bm25.pkl" if os.access(CLEAN, os.W_OK) else Path("bm25.pkl")
    if cache.exists():
        return ChunkIndex.load(cache)
    print(f"không có bm25.pkl trong {CLEAN} — build (~50s) -> {cache}", flush=True)
    idx = ChunkIndex()
    idx.build(list(chunks.values()))
    idx.save(cache)
    return idx


def nap_gold() -> dict[str, set[str]]:
    g: dict[str, set[str]] = {}
    with (CLEAN / "gold.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            g[o["qid"]] = set(o["gold_docs"])
    return g


def do_token(rows: list[dict], r: Retriever, tok, batch: int) -> None:
    """Đo độ dài prompt THẬT để chốt `max_seq_len`, trước khi sinh cả tập.

    Không đoán từ số ký tự: tỉ lệ token/ký tự của tiếng Việt luật khác hẳn tiếng Anh, và
    `max_chars=24_000` kế thừa từ v9.2 (configs/kaggle_inference_t4.yaml) được đặt cho
    context Qwen 3B lúc suy luận, KHÔNG cho seq len lúc train.

    Cột `gold` là tỉ lệ Evidence trả về chứa ít nhất một Document weak-gold. Đây mới là
    đại lượng quyết định: cắt ngân sách rẻ hay đắt tuỳ nó tụt bao nhiêu, không tuỳ token
    giảm bao nhiêu. Một mẫu SFT mà context không chứa gold là mẫu dạy model bịa.
    """
    gold = nap_gold()
    # dict[(top_n, max_chars)] -> (list token prompt, list token đáp, số Evidence, gold hit)
    kq = {c: ([], [], 0, 0) for c in CAU_HINH}
    ans_tok = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        ranked = r.rank(row["question"])  # phần đắt, dùng chung cho mọi cấu hình
        g = gold.get(row["id"])
        a_tok = len(tok.encode(row["answer"]).ids)
        ans_tok[row["id"]] = a_tok
        for cfg in CAU_HINH:
            top_n, max_chars = cfg
            evs = chon_evidence(ranked, r.docs, top_n=top_n, max_chars=max_chars)
            p_tok = len(tok.encode(dung_prompt(row["question"], evs)).ids)
            pr, an, so_ev, hit = kq[cfg]
            pr.append(p_tok)
            an.append(a_tok)
            kq[cfg] = (pr, an, so_ev + len(evs), hit + bool(g and any(e.doc_id in g for e in evs)))
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f}s/câu", flush=True)

    co_gold = sum(1 for row in rows if gold.get(row["id"]))
    n = len(rows)
    print(f"\nn={n}  câu có weak gold: {co_gold}  ({(time.time()-t0)/n:.2f}s/câu)")
    print("\n| top_n | max_chars | gold | ev/câu | p50 | p90 | p95 | max | >4096 | >6144 | >8192 |")
    print("|------:|----------:|-----:|-------:|----:|----:|----:|----:|------:|------:|------:|")
    for cfg in CAU_HINH:
        pr, an, so_ev, hit = kq[cfg]
        tong = sorted(x + y for x, y in zip(pr, an))
        q = lambda xs, k: xs[int(k * (len(xs) - 1))]
        vuot = [f"{sum(1 for x in tong if x > c)/n:5.1%}" for c in (4096, 6144, 8192)]
        print(f"| {cfg[0]} | {cfg[1]:,} | {hit/max(1,co_gold):.1%} | {so_ev/n:.1f} |"
              f" {q(tong,.5)} | {q(tong,.9)} | {q(tong,.95)} | {tong[-1]} | "
              + " | ".join(vuot) + " |")
    print("\n(p50/p90/p95/max và cột >N là prompt + answer, tức seq len thật lúc train)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-token", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="0 = cả tập")
    ap.add_argument("--model", default="", help="đường dẫn reranker; rỗng = BM25 thuần")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool", type=int, default=50)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (CLEAN / "train_sft.jsonl").open(encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]

    reranker = None
    if a.model:
        from legalqa.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(a.model, batch_size=a.batch_size)
    chunks = nap_chunks()
    r = Retriever(
        index=nap_index(chunks),
        chunks=chunks,
        docs=nap_docs(),
        reranker=reranker,
        pool=a.pool,
    )
    if a.do_token:
        from tokenizers import Tokenizer

        do_token(rows, r, Tokenizer.from_file(os.environ["QWEN_TOKENIZER"]), a.batch_size)
        return
    raise SystemExit("chưa cài phần sinh tập — chạy --do-token trước để chốt ngân sách")


if __name__ == "__main__":
    main()
