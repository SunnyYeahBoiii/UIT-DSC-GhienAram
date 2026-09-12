"""Đo xem Qwen có tràn fp16 khi prompt dài hay không.

Vì sao cần script này: v9.2 gặp output lặp từ + chữ Hán khi nhét >1 Evidence, và đã vá
triệu chứng (verifier.py:31 trừ điểm câu có CJK, PIPELINE_V9 xoá câu lặp) mà chưa ai đo
nguyên nhân. Giả thuyết: activation Qwen vượt trần fp16 (65.504) -> inf -> NaN sau
softmax -> logits phẳng -> decode lặp và trôi về token tần suất cao của pretraining
(Qwen pretrain nặng tiếng Trung). T4 là sm_75 nên bf16 KHÔNG chạy được, config buộc fp16.

Mảnh bằng chứng mạnh nhất đã có: repetition_penalty 1.1 và no_repeat_ngram_size 5 đã bật
trong kaggle_inference_t4.yaml mà vẫn lặp. Logits processor không sửa được NaN, nên lặp
là hậu quả của lỗi số chứ không phải lỗi sampling.

Script này ĐO, không sửa. Ba đại lượng, cùng một prompt, ba dtype:
  1. |activation| lớn nhất gặp phải  -> so với 65.504
  2. số tensor có phần tử không hữu hạn (inf/nan)
  3. 40 token đầu sinh ra            -> mắt thường thấy ngay rác

Chạy:
    PYTHONPATH=src python scripts/probe_fp16_overflow.py \
        --model models/Qwen3.5-2B --n-chunk 1 3 5
"""

from __future__ import annotations

import argparse

FP16_MAX = 65504.0


def _prompt(n_chunk: int, chunk_words: int = 869) -> str:
    """Prompt dài dần theo số Evidence. 869 từ là p50 Evidence đo ở §7 EDA_V10 —
    dùng số thật để ngưỡng vỡ tìm được ở đây áp được thẳng vào pipeline."""
    khoan = " ".join(f"quy định tại khoản {i % 9 + 1} Điều {i % 40 + 1}" for i in range(chunk_words // 6))
    cc = "\n\n".join(f"[Căn cứ {i+1}] Luật số {70+i}/2020/QH14\n{khoan}" for i in range(n_chunk))
    return f"Căn cứ pháp luật:\n{cc}\n\nCâu hỏi: Điều kiện được hưởng chế độ này là gì?\nTrả lời:"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3.5-2B")
    p.add_argument("--n-chunk", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--dtype", nargs="+", default=["float16", "float32"])
    p.add_argument("--max-new", type=int, default=40)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"GPU {torch.cuda.get_device_name(0)} · sm {'.'.join(map(str, torch.cuda.get_device_capability()))}")
    # bf16 trên sm_75 chỉ emulate, không phải hỗ trợ thật -> in ra để khỏi ai đề xuất lại.
    print(f"bf16 hỗ trợ: {torch.cuda.is_bf16_supported()}\n")

    tok = AutoTokenizer.from_pretrained(a.model)

    for dt_name in a.dtype:
        dt = getattr(torch, dt_name)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map="cuda")
        model.eval()

        stat = {"max": 0.0, "bad": 0, "layer_dau": ""}

        def hook(mod, inp, out, ten=""):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return
            # .float() trước khi lấy max: abs() trên fp16 đã inf thì max cũng inf,
            # nhưng ghi lại ĐƯỢC vị trí đầu tiên mới là thứ chỉ ra layer nào tràn.
            f = t.detach().float()
            if not torch.isfinite(f).all():
                stat["bad"] += 1
                if not stat["layer_dau"]:
                    stat["layer_dau"] = ten
            else:
                stat["max"] = max(stat["max"], float(f.abs().max()))

        handles = [
            m.register_forward_hook(lambda mo, i, o, t=n: hook(mo, i, o, t))
            for n, m in model.named_modules()
            if n.count(".") <= 3 and n
        ]

        for n_chunk in a.n_chunk:
            stat.update(max=0.0, bad=0, layer_dau="")
            ids = tok(_prompt(n_chunk), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
            txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in txt)
            ty = f"{100*stat['max']/FP16_MAX:.0f}%" if stat["max"] else "—"
            print(f"{dt_name:<9} {n_chunk} chunk · {ids['input_ids'].shape[1]:>5} token")
            print(f"  |act| max {stat['max']:>12.1f}  = {ty} trần fp16 · tensor không hữu hạn {stat['bad']}"
                  + (f" (đầu: {stat['layer_dau']})" if stat["layer_dau"] else ""))
            print(f"  CJK {cjk} ký tự · sinh: {txt[:110]!r}\n", flush=True)

        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
