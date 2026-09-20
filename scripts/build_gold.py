"""Dựng weak gold cho retrieval từ train.json — không có nhãn gold nào trong dữ liệu thi.

`train.json` 7.000 câu chỉ có `question` + `answer`. Muốn đo retrieval phải tự suy ra
câu hỏi này đáng lẽ trúng Document nào. Hai kênh độc lập, cố ý không dùng BM25:

**Kênh nguyên văn** — cắt 12 từ liên tiếp giữa Reference answer, tìm nguyên văn trong
text Chunk. EDA §3: 75,9% answer trích nguyên văn corpus.

**Kênh số hiệu** — answer trích dẫn "Nghị định 90/2017/NĐ-CP", tra ra doc_id bằng
`Document.number`. Kênh này bắt đúng nhóm mà kênh nguyên văn bỏ sót: answer diễn đạt
lại, không chung chữ nào với luật.

Vì sao KHÔNG lọc ứng viên bằng BM25 trước cho nhanh: gold suy từ đầu ra của chính
retriever đang đo sẽ tự khẳng định nó — mọi câu BM25 trượt sẽ biến mất khỏi tập đo.
Quét toàn corpus chậm hơn vài phút nhưng số đo mới có nghĩa.

Chạy: PYTHONPATH=src python scripts/build_gold.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from legalqa.graph import SO_HIEU_RE

# `corpus.DOC_NUMBER_RE` chỉ bắt "Số:" viết hoa — trượt 309/8.512 văn bản dạng "Luật số:",
# phần lớn là luật/bộ luật (Bộ luật TTHS 101/2015/QH13). Nới ở đây, KHÔNG sửa corpus.py vì
# Document.number còn nuôi legal graph lúc infer.
_SO_RE = re.compile(r"\b[Ss]ố:\s*([\w./\-]+)")

CLEAN_DIR = Path(os.environ.get("CLEAN_DIR", "data/clean"))
TRAIN = Path(os.environ.get("TRAIN_JSON", "data/official/train.json"))
# Tách tên file ra biến để dựng được gold cho warmup.json — 500 câu có Reference answer,
# KHÔNG nằm trong train. Reranker zero-shot nên train không rò rỉ, nhưng mọi ngưỡng/pool
# chốt trên train đều là chốt trên cùng một tập; warmup là tập độc lập để kiểm lại.
GOLD_OUT = os.environ.get("GOLD_OUT", "gold.jsonl")

# Cửa sổ 12 từ: EDA §3 đã đo bằng đúng con số này. Ngắn hơn thì khớp nhầm đoạn boilerplate
# ("theo quy định của pháp luật"), dài hơn thì vỡ vì một dấu cách lệch.
_W = 12


def _probe(answer: str) -> str | None:
    """12 từ giữa answer. Lấy GIỮA vì đầu answer là câu dẫn ("Căn cứ khoản 3 Điều 17...")
    do người viết soạn, không có trong corpus — khớp vào đó là khớp vào văn phong, không
    phải vào văn bản."""
    w = answer.split()
    if len(w) < 30:
        return None
    mid = len(w) // 2
    return " ".join(w[mid - _W // 2 : mid + _W // 2])


# Probe giữa trượt 750/7.000 câu train (answer <30 từ, hoặc đoạn giữa là lời người soạn).
# Dự phòng: dò MỌI cửa sổ 12 từ, gold = văn bản khớp >= nửa số cửa sổ của văn bản khớp
# nhiều nhất — cứu 704/750; kiểm trên câu vốn có gold: top-1 trùng 93,4%, tập trùng 97%.
_NGUONG = 0.5


def _moi_cua_so(answer: str) -> list[str]:
    w = answer.split()
    return [" ".join(w[i : i + _W]) for i in range(len(w) - _W + 1)]


def main() -> None:
    qa = json.loads(TRAIN.read_text(encoding="utf-8"))

    probes: dict[str, str] = {}
    for qid, v in qa.items():
        if not v.get("answer"):
            continue
        p = _probe(v["answer"])
        if p is not None:
            probes[p] = qid
    print(f"câu hỏi           {len(qa):,}")
    print(f"dựng được probe   {len(probes):,}")

    gold: dict[str, set[str]] = defaultdict(set)
    kenh: dict[str, set[str]] = defaultdict(set)

    # Quét một lượt toàn bộ Chunk, sinh shingle 12 từ rồi tra set probe. Đảo chiều so với
    # "mỗi probe quét mọi chunk": 287k × 245 lần tra set thay vì 287k × 7k lần str.find.
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            w = c["text"].split()
            for i in range(len(w) - _W + 1):
                qid = probes.get(" ".join(w[i : i + _W]))
                if qid is not None:
                    gold[qid].add(c["doc_id"])
                    kenh[qid].add("nguyen_van")
    print(f"có gold nguyên văn {len(gold):,}")

    # Lượt 2 chỉ cho câu probe giữa trượt, chạy TRƯỚC kênh số hiệu: câu chỉ có gold số hiệu
    # (văn bản được trích, chưa chắc là văn bản chứa câu trả lời) được thêm văn bản nguồn.
    du_phong: dict[str, set[str]] = defaultdict(set)
    for qid, v in qa.items():
        if v.get("answer") and qid not in gold:
            for p in _moi_cua_so(v["answer"]):
                du_phong[p].add(qid)
    dem: dict[str, Counter] = defaultdict(Counter)
    with (CLEAN_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            w = c["text"].split()
            for i in range(len(w) - _W + 1):
                for qid in du_phong.get(" ".join(w[i : i + _W]), ()):
                    dem[qid][c["doc_id"]] += 1
    for qid, ct in dem.items():
        top = max(ct.values())
        gold[qid] |= {d for d, n in ct.items() if n >= _NGUONG * top}
        kenh[qid].add("nguyen_van_moi_cua_so")
    print(f"cứu bằng mọi cửa sổ {len(dem):,}")

    # Kênh số hiệu. Document.number đọc từ passage nên phải quét corpus.jsonl.
    by_number: dict[str, str] = {}
    with (CLEAN_DIR / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m = _SO_RE.search(d["passage"][:4000])
            if m:
                # Số hiệu trùng nhau giữa hai văn bản là có thật (khác cơ quan ban hành).
                # Giữ bản đầu: gold yếu chấp nhận nhiễu, nhưng không được ghi đè im lặng.
                by_number.setdefault(m.group(1), d["doc_id"])
    print(f"văn bản có số hiệu {len(by_number):,}")

    for qid, v in qa.items():
        if not v.get("answer"):
            continue
        for so in SO_HIEU_RE.findall(v["answer"]):
            did = by_number.get(so)
            if did:
                gold[qid].add(did)
                kenh[qid].add("so_hieu")

    out = CLEAN_DIR / GOLD_OUT
    with out.open("w", encoding="utf-8") as f:
        for qid, docs in gold.items():
            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "question": qa[qid]["question"],
                        "gold_docs": sorted(docs),
                        "kenh": sorted(kenh[qid]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    c = Counter(tuple(sorted(v)) for v in kenh.values())
    n_doc = Counter(len(v) for v in gold.values())
    print(f"\n== GOLD YẾU  {len(gold):,}/{len(qa):,} = {100*len(gold)/len(qa):.1f}% câu ==")
    for k, v in c.most_common():
        print(f"  {'+'.join(k):<22} {v:6,}")
    print("  doc/câu: " + " | ".join(f"{k}:{v:,}" for k, v in sorted(n_doc.items())[:6]))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
