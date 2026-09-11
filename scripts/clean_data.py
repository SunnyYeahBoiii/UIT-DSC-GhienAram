"""Chạy clean toàn bộ dữ liệu, ghi ra data/clean/ và in báo cáo.

Hai luồng làm sạch độc lập:
  Corpus  — gỡ wrap mềm `\\r\\n\\n` của trình crawl (corpus.dewrap_passage)
  Train   — xoá nhiễu biên tập khỏi Reference answer (clean.clean_reference_answer)

Đầu ra materialize sẵn để bước index/SFT không phải parse lại 8.532 file JSON.
Mọi offset trong corpus.jsonl trỏ vào Passage ĐÃ gỡ wrap.

Chạy: PYTHONPATH=src python scripts/clean_data.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from legalqa.clean import clean_reference_answer, is_usable_for_sft
from legalqa.corpus import iter_documents, load_document

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/official"))
OUT_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))


def clean_corpus() -> dict[str, int]:
    """Ghi corpus.jsonl — mỗi dòng một Document đã gỡ wrap, kèm offset các Điều."""
    corpus_dir = DATA_DIR / "selected-contexts"
    n_file = len(list(corpus_dir.glob("*.json")))

    stat = Counter()
    out = OUT_DIR / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for doc in iter_documents(corpus_dir):
            stat["doc"] += 1
            stat["article"] += len(doc.articles)
            if not doc.articles:
                stat["doc_khong_dieu"] += 1
            if doc.number:
                stat["co_so_hieu"] += 1
            if not doc.slug:
                stat["thieu_slug"] += 1
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "number": doc.number,
                        "slug": doc.slug,
                        "link": doc.link,
                        "passage": doc.passage,
                        "articles": [
                            [a.number, a.heading, a.start, a.end] for a in doc.articles
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    stat["file"] = n_file
    stat["rong"] = n_file - stat["doc"]
    stat["byte_out"] = out.stat().st_size
    return dict(stat)


def _count_wrap() -> dict[str, int]:
    """Đếm wrap mềm còn sót — phải bằng 0 sau khi clean."""
    raw_cr = clean_cr = 0
    corpus_dir = DATA_DIR / "selected-contexts"
    for path in sorted(corpus_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8")).get("passage") or ""
        raw_cr += raw.count("\r")
        doc = load_document(path)
        if doc is not None:
            clean_cr += doc.passage.count("\r")
    return {"raw": raw_cr, "clean": clean_cr}


def clean_train() -> dict[str, int]:
    """Ghi train_sft.jsonl — answer đã làm sạch, bỏ mẫu không dùng được, khử trùng lặp.

    Trùng question thì giữ bản answer DÀI nhất: answer dài thường là bản đầy đủ,
    bản ngắn là trích lược.
    """
    raw = json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    items = raw.items() if isinstance(raw, dict) else ((str(i), r) for i, r in enumerate(raw))

    stat = Counter()
    best: dict[str, tuple[str, str, str]] = {}  # question -> (qid, question, answer)
    for qid, rec in items:
        stat["vao"] += 1
        q = (rec.get("question") or "").strip()
        a0 = rec.get("answer") or ""
        a = clean_reference_answer(a0)
        if a != a0:
            stat["bi_sua"] += 1
        if not is_usable_for_sft(q, a):
            stat["loai"] += 1
            continue
        key = re.sub(r"\s+", " ", q.lower())
        prev = best.get(key)
        if prev is None:
            best[key] = (qid, q, a)
        else:
            stat["trung_question"] += 1
            if len(a) > len(prev[2]):
                best[key] = (qid, q, a)

    out = OUT_DIR / "train_sft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for qid, q, a in best.values():
            f.write(
                json.dumps({"id": qid, "question": q, "answer": a}, ensure_ascii=False)
                + "\n"
            )
    stat["giu"] = len(best)
    return dict(stat)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    c = clean_corpus()
    print("== CORPUS ==")
    print(f"  file                {c['file']:,}")
    print(f"  Document ghi ra     {c['doc']:,}  (bo {c['rong']} file Passage rong)")
    print(f"  Dieu tach duoc      {c['article']:,}")
    print(f"  Document khong Dieu {c['doc_khong_dieu']:,} = {100*c['doc_khong_dieu']/c['doc']:.1f}%")
    print(f"  co so hieu          {c['co_so_hieu']:,} = {100*c['co_so_hieu']/c['doc']:.1f}%")
    print(f"  thieu Slug          {c['thieu_slug']:,} = {100*c['thieu_slug']/c['doc']:.1f}%")
    print(f"  -> {OUT_DIR/'corpus.jsonl'}  ({c['byte_out']/1e6:.1f} MB)")

    w = _count_wrap()
    print(f"\n  wrap mem \\r: {w['raw']:,} -> {w['clean']:,}")

    t = clean_train()
    print("\n== TRAIN ==")
    print(f"  vao                 {t['vao']:,}")
    print(f"  answer bi sua       {t.get('bi_sua', 0):,} = {100*t.get('bi_sua',0)/t['vao']:.1f}%")
    print(f"  loai (ngan/cat cut) {t.get('loai', 0):,}")
    print(f"  trung question      {t.get('trung_question', 0):,}")
    print(f"  giu                 {t['giu']:,}")
    print(f"  -> {OUT_DIR/'train_sft.jsonl'}")


if __name__ == "__main__":
    main()
