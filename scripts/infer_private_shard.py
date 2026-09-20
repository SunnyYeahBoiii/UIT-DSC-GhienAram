"""Sinh đáp án cho một khoảng câu của prompts.jsonl (private, không có gold) — không chấm.

Tách khỏi `eval_shard.py`/`sinh_va_cham`: private set không có `answer` nên không cham
được, và không cần — chỉ cần sinh. Ghi APPEND + resumable qua `da_co` (như
`infer.py::pha_prompt`) vì 1.918 câu x 1.024 token/câu vượt xa 1 phiên GPU Kaggle —
kernel bị giết giữa chừng thì chạy lại (cùng `--ra`) chỉ sinh nốt phần còn thiếu.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from legalqa.prompt import tin_nhan
from infer import da_co


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True, help="prompts.jsonl (id, question, prompt)")
    ap.add_argument("--ra", required=True, help="file jsonl đích, append + resumable")
    ap.add_argument("--tu", type=int, default=0)
    ap.add_argument("--den", type=int, default=0, help="0 = tới hết")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gen-bs", type=int, default=2)
    ap.add_argument("--gio-toi-da", type=float, default=8.5, help="giờ, dừng an toàn trước 9-12h Kaggle")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    import transformers.integrations.sdpa_attention as _sdpa

    _sdpa.use_gqa_in_sdpa = lambda *_a, **_k: False

    ra = Path(a.ra)
    ra.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in Path(a.prompts).open(encoding="utf-8")]
    rows = rows[a.tu : a.den or None]
    xong = da_co(ra)
    con = [r for r in rows if r["id"] not in xong]
    print(f"shard {a.tu}:{a.den or len(rows) + a.tu}  ({len(rows)} câu, {len(xong)} đã có, còn {len(con)})", flush=True)

    if not con:
        print("== HẾT VIỆC, không sinh gì", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    from train_sft import _LOP_MODEL

    model = None
    for lop in _LOP_MODEL():
        try:
            model = lop.from_pretrained(
                a.model, dtype=torch.float16, attn_implementation="sdpa"
            ).cuda()
            print("nạp bằng", lop.__name__, flush=True)
            break
        except (ValueError, KeyError) as e:
            print(f"{lop.__name__} không nhận: {e}", flush=True)
    assert model is not None, "không lớp Auto* nào nạp được model nền"
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    print("gắn adapter:", a.adapter, flush=True)

    t0 = time.time()
    han = a.gio_toi_da * 3600
    with ra.open("a", encoding="utf-8") as f:
        for i in range(0, len(con), a.gen_bs):
            lo = con[i : i + a.gen_bs]
            vb = [
                tok.apply_chat_template(
                    tin_nhan(r["prompt"], "goc"),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for r in lo
            ]
            tok.padding_side = "left"
            enc = tok(vb, return_tensors="pt", padding=True, add_special_tokens=False).to(
                model.device
            )
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=a.max_new,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
            for r, seq in zip(lo, out):
                du_doan = tok.decode(
                    seq[enc["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                f.write(json.dumps({
                    "id": r["id"], "question": r.get("question"), "du_doan": du_doan,
                    "doc_ids": r.get("doc_ids"),
                }, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(lo)
            if n // 10 > i // 10 or n >= len(con):
                t = time.time() - t0
                print(f"    sinh {n}/{len(con)} câu, {t:.0f}s, ~{t/n:.1f}s/câu", flush=True)
            if time.time() - t0 > han:
                print("  hết ngân sách giờ — dừng, chạy lại để tiếp", flush=True)
                return

    print("== XONG", ra)


if __name__ == "__main__":
    main()
