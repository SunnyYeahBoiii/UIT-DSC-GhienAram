"""Chấm 1 shard dev100.jsonl trên 1 checkpoint, chạy trên Kaggle T4 (song song nhiều acc).

Tách khỏi eval_moc.py vì eval_moc.py chạy vòng lặp chờ mốc mới trên máy train — ở đây
chỉ cần chấm MỘT checkpoint đã có sẵn, một lần, rồi thoát. Dùng lại nguyên
`sinh_va_cham`/`cham`/`tin_nhan` của train_sft.py để số ra khớp tuyệt đối với eval local
(cùng greedy decode, cùng khuôn prompt, cùng công thức METEOR).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from train_sft import _LOP_MODEL, chuan_bi_nltk, nap, sinh_va_cham


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model nền")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dev", required=True, help="dev100.jsonl (đủ 100 câu, KHÔNG cắt trước)")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--tu", type=int, default=0)
    ap.add_argument("--den", type=int, default=0, help="0 = tới hết")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gen-bs", type=int, default=2)
    ap.add_argument("--kieu-prompt", choices=["goc", "card"], default="goc")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    # Cùng lý do với train_sft.py/infer.py: bs không pad hoặc gen-bs 1 thì mask bị bỏ,
    # enable_gqa bật, T4 rơi về kernel MATH T x T -> OOM.
    import transformers.integrations.sdpa_attention as _sdpa

    _sdpa.use_gqa_in_sdpa = lambda *_a, **_k: False

    chuan_bi_nltk()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)

    dev = nap(Path(a.dev))
    dev = dev[a.tu : a.den or None]
    print(f"shard {a.tu}:{a.den or len(nap(Path(a.dev)))}  ({len(dev)} câu)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

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

    diem = sinh_va_cham(model, tok, dev, a.max_new, a.gen_bs, ra=ra, buoc=a.tu, kieu=a.kieu_prompt)
    print("== ĐIỂM:", diem, flush=True)
    (ra / f"diem_{a.tu}_{a.den}.json").write_text(json.dumps(diem, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
