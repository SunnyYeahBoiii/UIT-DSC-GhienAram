"""Cắt toàn corpus đã clean thành Chunk, ghi data/clean/chunks.jsonl + thống kê.

Đọc corpus.jsonl (đầu ra của clean_data.py) chứ không đọc lại 8.532 file JSON:
mọi offset phải trỏ vào Passage ĐÃ clean, đọc file thô sẽ lệch.

Chạy: PYTHONPATH=src python scripts/build_chunks.py [size] [stride]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from legalqa.chunking import chunk_document
from legalqa.corpus import Article, Document
from legalqa.evidence import expand_to_evidence

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))


def load_clean_docs() -> list[Document]:
    """Dựng lại Document từ corpus.jsonl — Passage đã clean, Article đã có offset."""
    docs = []
    with (CLEAN_DIR / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(
                Document(
                    doc_id=d["doc_id"],
                    passage=d["passage"],
                    slug=d["slug"],
                    link=d["link"],
                    articles=[Article(n, h, s, e) for n, h, s, e in d["articles"]],
                )
            )
    return docs


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    docs = load_clean_docs()
    stat = Counter()
    lens: list[int] = []
    per_doc: list[int] = []
    ev_words: list[int] = []
    n_art: list[int] = []

    out = CLEAN_DIR / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for doc in docs:
            chunks = chunk_document(doc, size=size, stride=stride)
            per_doc.append(len(chunks))
            if not chunks:
                stat["doc_khong_chunk"] += 1
            for c in chunks:
                # Offset phải slice đúng ra text — bug này từng sai 287.716/287.741 chunk
                assert doc.passage[c.start : c.end] == c.text, c.chunk_id
                lens.append(len(c.text.split()))
                ev = expand_to_evidence(doc, c)
                ev_words.append(len(ev.text.split()))
                if ev.article_number is None:
                    stat["evidence_duong_lui"] += 1
                else:
                    n_art.append(ev.article_number.count(",") + 1)
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "start": c.start,
                            "end": c.end,
                            "text": c.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    n = len(lens)
    q = statistics.quantiles(lens, n=10)
    qe = statistics.quantiles(ev_words, n=10)
    print(f"== CHUNK {size}/{stride} ==")
    print(f"  Document            {len(docs):,}  (khong sinh chunk: {stat['doc_khong_chunk']})")
    print(f"  Chunk               {n:,}")
    print(f"  chunk/Document      p50 {statistics.median(per_doc):.0f} | max {max(per_doc):,}")
    print(f"  tu/chunk            p10 {q[0]:.0f} | p50 {statistics.median(lens):.0f} | p90 {q[-1]:.0f}")
    print(f"  -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("\n== EVIDENCE mo rong tu chunk do ==")
    fb = stat["evidence_duong_lui"]
    print(f"  tu/Evidence         p10 {qe[0]:.0f} | p50 {statistics.median(ev_words):.0f} | p90 {qe[-1]:.0f}")
    print(f"  so Dieu gop         p50 {statistics.median(n_art):.0f} | max {max(n_art)}")
    print(f"  roi duong lui       {fb:,} = {100*fb/n:.1f}%")


if __name__ == "__main__":
    main()
