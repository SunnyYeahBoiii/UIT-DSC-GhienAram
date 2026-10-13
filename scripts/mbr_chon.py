"""Chọn 1 đáp án / câu từ nhiều ứng viên bằng MBR (Minimum Bayes Risk).

Vì sao utility là METEOR từng cặp: METEOR là metric có tín hiệu của BTC (ROUGE-L của họ xoá
dấu, xem scorer.py). Ứng viên i được chấm như hyp so với ứng viên j làm ref — cùng chiều với
cách BTC chấm, vì METEOR không đối xứng. Ứng viên được `cat_lap` trước khi so, giống lúc nộp:
bản thô có vòng lặp sẽ kéo lệch độ tương đồng.

Câu chỉ có 1 ứng viên thì lấy luôn. Câu không có ứng viên nào ghi chuỗi rỗng (BTC raise
nếu thiếu id, xem infer.ghi_submission).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalqa.sanitize import cat_lap  # noqa: E402
from legalqa.scorer import cham, chuan_bi_nltk  # noqa: E402


def doc_ung_vien(files: list[Path]) -> tuple[dict[str, list[str]], dict[str, str]]:
    uv, dap_an = defaultdict(list), {}
    for f in files:
        for l in f.open(encoding="utf-8"):
            x = json.loads(l)
            uv[x["id"]].append(cat_lap(x.get("du_doan") or ""))
            if x.get("dap_an"):
                dap_an[x["id"]] = x["dap_an"]
    return uv, dap_an


def chon(ds: list[str]) -> str:
    from nltk.translate.meteor_score import meteor_score

    ds = [d for d in ds if d.strip()]
    if len(ds) <= 1:
        return ds[0] if ds else ""
    tok = [d.split() for d in ds]
    diem = [sum(meteor_score([tok[j]], tok[i]) for j in range(len(ds)) if j != i) for i in range(len(ds))]
    return ds[max(range(len(ds)), key=diem.__getitem__)]


def mbr(uv: dict[str, list[str]], proc: int = 8) -> dict[str, str]:
    ids = list(uv)
    with Pool(proc, initializer=chuan_bi_nltk) as p:
        return dict(zip(ids, p.map(chon, [uv[k] for k in ids], chunksize=8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="các du_doan*.jsonl ứng viên")
    ap.add_argument("--cau-hoi", type=Path, help="private-official.json: đủ id để ghi submission")
    ap.add_argument("--ra", type=Path, help="thư mục ghi submission.json/.zip")
    ap.add_argument("--proc", type=int, default=8)
    a = ap.parse_args()
    chuan_bi_nltk()
    uv, dap_an = doc_ung_vien(a.files)
    so = defaultdict(int)
    for v in uv.values():
        so[len(v)] += 1
    print(f"{len(uv)} câu, số ứng viên/câu: {dict(sorted(so.items()))}")
    kq = mbr(uv, a.proc)
    if dap_an:
        print("MBR:", cham(kq, dap_an))
    if a.ra:
        ids = list(json.load(open(a.cau_hoi, encoding="utf-8"))) if a.cau_hoi else list(kq)
        sub = {k: {"answer": kq.get(k, "")} for k in ids}
        a.ra.mkdir(parents=True, exist_ok=True)
        js = a.ra / "submission.json"
        js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(a.ra / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(js, "submission.json")
        print(f"-> {js} ({len(sub)} câu, {sum(not v['answer'] for v in sub.values())} rỗng)")


if __name__ == "__main__":
    main()
