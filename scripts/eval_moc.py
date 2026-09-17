"""Chấm dev các adapter `b{N}/` mà `train_sft.py --eval-rieng` lưu ra, trên một GPU riêng.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_moc.py --model <base> --dev dev100.jsonl \
        --dev-n 0 --ra <thư mục --ra của train> --max-new 1024 --gen-bs 2

Ghi `lich_su.json` và `tot_nhat/` vào đúng thư mục train, cùng định dạng eval trong vòng,
nên `--tiep <ra>/tot_nhat` của phiên sau đọc lại được như cũ. Thoát khi train đã ghi
`cuoi_buoc.json` và mọi mốc đã chấm.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import random
import shutil
import time
from pathlib import Path

from train_sft import _LOP_MODEL, chuan_bi_nltk, nap, sinh_va_cham


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--dev-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gen-bs", type=int, default=2)
    ap.add_argument("--kieu-prompt", choices=["goc", "card"], default="goc")
    # "k/n": chỉ chấm mốc thứ k (theo bước/--eval-steps) trong n phần. Eval dev100 ~30 phút
    # một mốc mà DDP sinh mốc mới mỗi ~12 phút — một GPU eval tụt lại, cần n tiến trình.
    ap.add_argument("--phan", default="0/1")
    ap.add_argument("--eval-steps", type=int, default=40)
    a = ap.parse_args()
    k, n = map(int, a.phan.split("/"))

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    # Cùng bản vá GQA với train_sft: thiếu nó SDPA rơi về kernel MATH, OOM ở T~10k.
    import transformers.integrations.sdpa_attention as _sdpa
    _sdpa.use_gqa_in_sdpa = lambda *_a, **_k: False

    chuan_bi_nltk()
    ra = Path(a.ra)
    # Xáo đúng như train_sft để số đo so được với các mốc eval trong vòng.
    dev = nap(Path(a.dev))
    random.Random(a.seed).shuffle(dev)
    dev = dev[: a.dev_n] if a.dev_n else dev

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    nen = None
    for lop in _LOP_MODEL():
        try:
            nen = lop.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda()
            break
        except (ValueError, KeyError):
            pass
    assert nen is not None, "không lớp Auto* nào nạp được model nền"

    q = ra / "lich_su.json"
    lich_su = json.loads(q.read_text(encoding="utf-8")) if q.exists() else []
    tot_nhat = max((d["meteor"] for d in lich_su), default=-1.0)
    print(f"dev {len(dev)} mẫu | lịch sử {len(lich_su)} mốc | tốt nhất {tot_nhat:.4f}", flush=True)

    while True:
        da = {d["buoc"] for d in lich_su}
        moi = sorted(
            (json.loads(m.read_text(encoding="utf-8")) | {"dir": m.parent}
             for m in ra.glob("b*/moc.json")),
            key=lambda x: x["buoc"],
        )
        moi = [m for m in moi if m["buoc"] not in da and (m["buoc"] // a.eval_steps) % n == k]
        if not moi:
            if (ra / "cuoi_buoc.json").exists():
                break
            time.sleep(60)
            continue
        m = moi[0]
        model = PeftModel.from_pretrained(nen, m["dir"])
        d = sinh_va_cham(model, tok, dev, a.max_new, a.gen_bs, ra, m["buoc"], a.kieu_prompt)
        nen = model.unload()
        d |= {"buoc": m["buoc"], "loss": m["loss"], "gnorm": m["gnorm"]}
        print(f"  == EVAL b{m['buoc']}: {d}", flush=True)
        # Nhiều tiến trình cùng ghi `lich_su.json` và `tot_nhat/`: khoá, đọc lại bản mới
        # nhất rồi mới gộp — không thì tiến trình này đè mất mốc tiến trình kia vừa ghi.
        with open(ra / ".khoa", "w") as kh:
            fcntl.flock(kh, fcntl.LOCK_EX)
            lich_su = json.loads(q.read_text(encoding="utf-8")) if q.exists() else []
            tot_nhat = max((x["meteor"] for x in lich_su), default=-1.0)
            lich_su.append(d)
            q.write_text(json.dumps(lich_su, ensure_ascii=False, indent=1), encoding="utf-8")
            if d["meteor"] > tot_nhat:
                tot_nhat = d["meteor"]
                shutil.rmtree(ra / "tot_nhat", ignore_errors=True)
                shutil.copytree(m["dir"], ra / "tot_nhat")
                print(f"  -> tot_nhat = b{m['buoc']} (METEOR {tot_nhat:.4f})", flush=True)
    print(f"xong: tốt nhất {tot_nhat:.4f}", flush=True)


if __name__ == "__main__":
    main()
