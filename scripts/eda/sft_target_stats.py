"""EDA cho tập SFT: đầu vào và ĐẦU RA mà model sẽ bị dạy sinh ra.

§1 và §3 của EDA_V10 đã đo độ dài và tỷ lệ trích nguyên văn. Script này đo thứ khác:
HÌNH DẠNG của Reference answer, vì đó là target của SFT. Nếu 90% target mở đầu bằng
"Căn cứ ..." thì model học một khuôn, và khuôn đó phải khớp với cái prompt.py dựng ra.

Chạy: PYTHONPATH=src python scripts/eda/sft_target_stats.py
"""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

SFT = Path("data/clean/train_sft.jsonl")
GOLD = Path("data/clean/gold.jsonl")


def doc(rows: list[dict]) -> None:
    n = len(rows)
    print(f"## Tập: {n} cặp (sau clean_reference_answer + is_usable_for_sft)\n")

    # --- mở đầu answer ---
    mo_dau = Counter()
    for r in rows:
        a = r["answer"].lstrip()
        m = re.match(r"^(Căn cứ|Theo|Tại|Dựa|Điều|Khoản|Hiện nay|Trước hết)\b", a)
        mo_dau[m.group(1) if m else a.split()[0] if a.split() else "?"] += 1
    print("### Từ mở đầu Reference answer")
    for k, v in mo_dau.most_common(10):
        print(f"| {k} | {v} | {v/n:.1%} |")

    # --- có dẫn Điều/khoản ở câu ĐẦU không ---
    cau_dau_co_dieu = sum(
        1 for r in rows if re.search(r"Điều\s+\d", r["answer"].split("\n")[0])
    )
    print(f"\ncâu đầu tiên chứa 'Điều N': {cau_dau_co_dieu} ({cau_dau_co_dieu/n:.1%})")

    # --- cấu trúc: số dòng, số dòng đánh khoản ---
    dong = [r["answer"].count("\n") + 1 for r in rows]
    khoan = [
        len(re.findall(r"^\s*\d+\.", r["answer"], re.M)) for r in rows
    ]
    print(f"số dòng   : mean {statistics.mean(dong):.1f} · p50 {statistics.median(dong)}"
          f" · p90 {sorted(dong)[int(n*0.9)]}")
    print(f"dòng '\\d.': mean {statistics.mean(khoan):.1f} · p50 {statistics.median(khoan)}"
          f" · có ít nhất 1: {sum(1 for k in khoan if k)/n:.1%}")

    # --- kết thúc ---
    ket = Counter()
    for r in rows:
        a = r["answer"].rstrip()
        ket["dấu chấm" if a.endswith(".") else
            "dấu :" if a.endswith(":") else
            "dấu ;" if a.endswith(";") else "khác"] += 1
    print("\n### Ký tự kết thúc")
    for k, v in ket.most_common():
        print(f"| {k} | {v} | {v/n:.1%} |")

    # --- độ dài từ ---
    w = [len(r["answer"].split()) for r in rows]
    q = [len(r["question"].split()) for r in rows]
    print(f"\nanswer (từ) : mean {statistics.mean(w):.0f} · p50 {statistics.median(w)}"
          f" · p90 {sorted(w)[int(n*0.9)]} · p99 {sorted(w)[int(n*0.99)]} · max {max(w)}")
    print(f"question(từ): mean {statistics.mean(q):.1f} · p90 {sorted(q)[int(n*0.9)]}")

    # --- noise còn sót SAU clean ---
    print("\n### Noise còn sót sau clean_reference_answer")
    # `re.I` ở đây KHÔNG phải cho gọn: `clean._IMG_CAPTION_RE` phân biệt hoa/thường nên
    # nó chỉ bắt đúng "(Hình từ Internet)". Đo được 4 biến thể lọt lưới —
    # "(Hình từ internet)" 73, "(hình từ Internet)" 43, "(hình từ internet)" 19,
    # "Hình từ Internet)" 1 — tức 135/6.929 (1,9%) caption ảnh vẫn nằm trong target.
    mau = {
        "(Hình từ Internet)": r"Hình từ Internet",
        "tại đây": r"tại đây",
        "[...]  (hợp lệ, giữ)": r"\[\.\.\.\]",
        "'Trân trọng'": r"Trân trọng",
        "'Như vậy'": r"Như vậy",
        "URL http": r"https?://",
        "'xem thêm'": r"xem thêm",
    }
    for ten, pat in mau.items():
        c = sum(1 for r in rows if re.search(pat, r["answer"], re.I))
        print(f"| {ten} | {c} | {c/n:.1%} |")


def duoi_hong(rows: list[dict]) -> None:
    """Ba lớp đuôi hỏng do crawl, đo riêng vì `is_usable_for_sft` hiện không bắt lớp nào.

    Vì sao quan trọng: token cuối của target là thứ dạy model DỪNG. Nếu 14,2% target
    kết thúc bằng một câu hỏi (teaser bài kế tiếp mà crawler nuốt vào), model học rằng
    "trả lời xong thì đặt thêm câu hỏi" — đúng dạng lảm nhảm đã thấy ở v9.2.
    """
    print("\n### Đuôi target hỏng (crawl artifact)")
    n = len(rows)
    vo_hinh = "\ufeff\u200b\u00a0\u200e"
    hoi = sum(1 for r in rows if r["answer"].rstrip(vo_hinh + " \n\t").endswith("?"))
    lap_q = sum(
        1 for r in rows
        if r["question"].strip().rstrip("? ")[-50:].lower() in r["answer"].lower()
    )
    inv = sum(1 for r in rows if any(c in r["answer"] for c in vo_hinh))
    print(f"| kết thúc bằng câu hỏi | {hoi} | {hoi/n:.1%} |")
    print(f"| answer chứa nguyên văn 50 ký tự cuối của question | {lap_q} | {lap_q/n:.1%} |")
    print(f"| chứa ký tự vô hình (BOM/ZWSP/NBSP) | {inv} | {inv/n:.1%} |")


def do_dai_token(rows: list[dict]) -> None:
    """Độ dài THẬT theo tokenizer Qwen, không phải đếm từ.

    Cần để chốt `max_seq_len`: tiếng Việt luật cho 1,314 token/từ ở answer và 1,217 ở
    question, nên p95 answer 905 token chứ không phải 576 "từ". Bỏ qua nếu chưa có
    tokenizer — máy dev không cài `transformers`.
    """
    path = os.environ.get("QWEN_TOKENIZER")
    if not path or not Path(path).exists():
        print("\n(bỏ qua độ dài token: đặt QWEN_TOKENIZER=<tokenizer.json> để đo)")
        return
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    print("\n### Độ dài token (Qwen BPE)")
    for ten, key in (("question", "question"), ("answer", "answer")):
        xs = sorted(len(tok.encode(r[key]).ids) for r in rows)
        m = len(xs)
        print(f"| {ten} | mean {statistics.mean(xs):.0f} | p50 {xs[m//2]} |"
              f" p95 {xs[int(m*0.95)]} | p99 {xs[int(m*0.99)]} | max {xs[-1]} |")


def gold_stats(rows: list[dict], gold: dict[str, dict]) -> None:
    print("\n## Gắn được Evidence không?")
    n = len(rows)
    co = sum(1 for r in rows if r["id"] in gold)
    print(f"cặp SFT có weak gold document: {co}/{n} ({co/n:.1%})")
    sd = Counter(len(gold[r["id"]]["gold_docs"]) for r in rows if r["id"] in gold)
    print("số gold doc mỗi câu:", dict(sorted(sd.items())[:8]), "...")
    kenh = Counter(k for r in rows if r["id"] in gold for k in gold[r["id"]]["kenh"])
    print("kênh tìm ra gold:", dict(kenh))


def main() -> None:
    rows = [json.loads(l) for l in SFT.open(encoding="utf-8")]
    gold = {}
    for l in GOLD.open(encoding="utf-8"):
        g = json.loads(l)
        gold[g["qid"]] = g
    doc(rows)
    duoi_hong(rows)
    do_dai_token(rows)
    gold_stats(rows, gold)


if __name__ == "__main__":
    main()
