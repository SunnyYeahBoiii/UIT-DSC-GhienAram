"""Xuất legal graph ra JSON để xem được bằng mắt.

Không ghi vào repo: dữ liệu cuộc thi không được commit. Đặt đích bằng biến môi trường
GRAPH_OUT, mặc định data/clean/graph.js (đã nằm trong .gitignore).

Ghi ra .js gán `window.GRAPH` chứ không .json: trang xem nạp bằng <script src> nên chạy
được ở mọi nơi, không phụ thuộc fetch cùng origin.

Chạy: PYTHONPATH=src python scripts/dump_graph.py [duong_dan_ra]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from build_chunks import load_clean_docs
from legalqa.graph import LegalGraph


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRAPH_OUT", "data/clean/graph.js"))
    docs = load_clean_docs()
    by_id = {d.doc_id: d for d in docs}
    g = LegalGraph()
    g.build(docs)

    # Chỉ xuất văn bản có mặt trong graph — 8.512 văn bản mà chỉ ~550 có cạnh.
    lien_quan: set[str] = set()
    for (dich, _), lst in g._amend.items():
        lien_quan.add(dich)
        lien_quan.update(a.src_doc_id for a in lst)

    canh = []
    for (dich, dieu), lst in g._amend.items():
        for a in lst:
            src = by_id[a.src_doc_id]
            canh.append(
                {
                    "dich": dich,
                    "dieu": dieu,
                    "src": a.src_doc_id,
                    "src_dieu": a.src_article,
                    "khoan": a.src_khoan,
                    "verb": a.verb[:160],
                    # Text khoản cắt 800 ký tự: p90 là 2.418, cắt để trang còn ~4 MB.
                    # Đọc đầy đủ thì tra thẳng corpus theo offset.
                    "text": src.passage[a.start : a.end][:800],
                    "du": max(0, a.end - a.start - 800),
                }
            )

    vb = {
        i: {
            "title": by_id[i].title,
            "number": by_id[i].number,
            "slug": by_id[i].slug,
            "n_dieu": len(by_id[i].articles),
        }
        for i in sorted(lien_quan)
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"van_ban": vb, "canh": canh}, ensure_ascii=False)
    out.write_text(f"window.GRAPH={payload};", encoding="utf-8")
    print(f"van ban trong graph {len(vb):,} | canh {len(canh):,} -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
