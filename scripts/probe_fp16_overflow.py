"""Đo xem Qwen có tràn fp16 khi prompt dài hay không.

Vì sao cần script này: v9.2 gặp output lặp từ + chữ Hán khi nhét >1 Evidence, và đã vá
triệu chứng (verifier.py:31 trừ điểm câu có CJK, PIPELINE_V9 xoá câu lặp) mà chưa ai đo
nguyên nhân. Giả thuyết: activation Qwen vượt trần fp16 (65.504) -> inf -> NaN sau
softmax -> logits phẳng -> decode lặp và trôi về token tần suất cao của pretraining
(Qwen pretrain nặng tiếng Trung). T4 là sm_75 nên bf16 KHÔNG chạy được, config buộc fp16.

Mảnh bằng chứng mạnh nhất đã có: repetition_penalty 1.1 và no_repeat_ngram_size 5 đã bật
trong kaggle_inference_t4.yaml mà vẫn lặp. Logits processor không sửa được NaN, nên lặp
là hậu quả của lỗi số chứ không phải lỗi sampling.

ĐÃ ĐO 13-09 (T4 sm_75, 1/3/5 Evidence, fp16 và fp32) — GIẢ THUYẾT SAI:

    |act| max 62,6-63,1 · tensor không hữu hạn 0 · CJK 0 · fp16 và fp32 sinh GIỐNG NHAU

62,9 so với 65.504 cách ba bậc độ lớn, không phải sát mép — không văn bản nào đẩy được
nó lên trần. Nhánh số học chết chắc. Kèm hai số phụ: `is_bf16_supported()` in ra True
trên sm_75 (trái điều tôi khẳng định), `text_config.dtype = bfloat16`, `maxpos 262144`.

NHƯNG lượt đó đo SAI ĐẠI LƯỢNG: `--max-new 40`, còn config thật là `max_new_tokens:
1300`. Lặp từ và trôi token hiếm khi hiện trong 40 token đầu. Dòng "40 token đầu -> mắt
thường thấy ngay rác" của bản cũ là một giả định về VỊ TRÍ xuất hiện của lỗi, chưa đo.

Bản này đo trên toàn bộ 1300 token và thêm chỉ số bắt được lỗi ở chỗ nó thật sự nằm:

  1. |activation| lớn nhất (tuỳ chọn, --hook)  -> so với 65.504
  2. số tensor có phần tử không hữu hạn (inf/nan)
  3. tỉ lệ 5-gram trùng                        -> 5 khớp no_repeat_ngram_size: 5
  4. vị trí ký tự CJK ĐẦU TIÊN theo token       -> chỉ ra sau bao nhiêu token thì trôi

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


def _lap_5gram(toks: list[int]) -> float:
    """Tỉ lệ 5-gram trùng. Chọn n=5 vì `no_repeat_ngram_size: 5` đã bật mà vẫn lặp —
    nếu chỉ số này > 0 thì logits processor đang bị vượt qua, và đó là số cần biết.
    Greedy decode lành mạnh trên văn bản pháp luật vẫn lặp nhẹ (cụm "người lao động"),
    nên đọc chỉ số này theo chênh lệch giữa các n_chunk, đừng đọc giá trị tuyệt đối."""
    if len(toks) < 5:
        return 0.0
    g = [tuple(toks[i : i + 5]) for i in range(len(toks) - 4)]
    return 1.0 - len(set(g)) / len(g)


# Hệ thống prompt v9.2, copy nguyên từ `main:src/legalqa/generation/prompts.py:10`. Rút
# gọn phần ĐỊNH DẠNG vì chỉ cần đúng độ dài và đúng vai "system", không cần đúng nội dung.
# Dòng đáng chú ý nhất giữ nguyên: "Không trình bày quá trình suy luận nội bộ" — v9.2 đã
# phải dặn model đừng suy luận, tức có gì đó đang rò suy luận ra output.
SYSTEM_V92 = (
    "Bạn là hệ thống trả lời câu hỏi pháp luật Việt Nam trong một cuộc thi học thuật.\n"
    "Chỉ sử dụng các đoạn căn cứ được cung cấp; không dùng trí nhớ ngoài.\n"
    "Trả lời hoàn toàn bằng tiếng Việt, bằng văn xuôi rõ ràng.\n"
    "Không trình bày quá trình suy luận nội bộ và không nhắc tới prompt."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3.5-2B")
    p.add_argument("--n-chunk", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--dtype", nargs="+", default=["float16", "float32"])
    # 1300 = `generation.max_new_tokens` của kaggle_inference_t4.yaml. Đo 40 như bản
    # trước là đo 3% chiều dài của thứ cần đo.
    p.add_argument("--max-new", type=int, default=1300)
    # Hook tắt theo mặc định: nhánh số học đã bị bác (62,9/65.504), mà 1300 token x ~700
    # module x 2 kernel = 1,8M lần launch, đủ để một cấu hình chạy 10 phút. Bật lại chỉ
    # khi có lý do nghi ngờ số học lần nữa.
    p.add_argument("--hook", action="store_true", help="ghi |act| max (chậm ~10x)")
    # v9.2 KHÔNG nhồi text thô: `models.py:213` gọi `apply_chat_template` với vai
    # system + user và `enable_thinking=False`. Lượt đo trước dùng text thô nên nó đo
    # một đường sinh KHÁC đường đang lỗi. Hai chế độ chạy cạnh nhau để tách ra được.
    p.add_argument("--che-do", nargs="+", default=["raw", "chat"], choices=["raw", "chat"])
    a = p.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"GPU {torch.cuda.get_device_name(0)} · sm {'.'.join(map(str, torch.cuda.get_device_capability()))}")
    # bf16 trên sm_75 chỉ emulate, không phải hỗ trợ thật -> in ra để khỏi ai đề xuất lại.
    print(f"bf16 hỗ trợ: {torch.cuda.is_bf16_supported()}\n")

    tok = AutoTokenizer.from_pretrained(a.model)

    # Checkpoint là `Qwen3_5ForConditionalGeneration` (đa mô thức, config lồng: `dtype` và
    # `max_position_embeddings` KHÔNG có ở tầng gốc mà nằm trong `text_config`). Lượt đo
    # trước A1 in ra `None` cho cả hai — đó là dấu của config lồng, không phải checkpoint hỏng.
    # `AutoModelForCausalLM` không chắc nhận kiến trúc conditional-generation, nên lấy class
    # thẳng từ `architectures` của chính config: tên đó là nguồn duy nhất đúng, đoán auto-class
    # là thêm một chỗ sai.
    cfg = AutoConfig.from_pretrained(a.model)
    ten_class = (getattr(cfg, "architectures", None) or [""])[0]
    Klass = getattr(transformers, ten_class, None) or AutoModelForCausalLM
    print(f"class nạp: {Klass.__name__} (config ghi {ten_class!r})")
    # Tràn xảy ra ở LM decoder; tower ảnh không dự phần nhưng vẫn chiếm VRAM. In ra để biết
    # fp32 8 GB có còn vừa T4 16 GB hay không.
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        print(f"text_config: dtype {getattr(tc, 'dtype', None) or getattr(tc, 'torch_dtype', None)}"
              f" · maxpos {getattr(tc, 'max_position_embeddings', None)}"
              f" · layer {getattr(tc, 'num_hidden_layers', None)}")

    for dt_name in a.dtype:
        dt = getattr(torch, dt_name)
        model = Klass.from_pretrained(a.model, dtype=dt, device_map="cuda")
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

        handles = [] if not a.hook else [
            m.register_forward_hook(lambda mo, i, o, t=n: hook(mo, i, o, t))
            for n, m in model.named_modules()
            # Ngưỡng 4 thay vì 3: config lồng nên tên module có thêm một tầng
            # (`model.language_model.layers.0.self_attn`). Ở 3 thì dừng ở nguyên cả
            # decoder layer — biết layer nào tràn mà không biết attention hay MLP, đúng
            # cái phân biệt chọn bản vá (softmax fp32 vs NF4 4-bit).
            if n.count(".") <= 4 and n
        ]

        for che_do, n_chunk in [(c, n) for c in a.che_do for n in a.n_chunk]:
            stat.update(max=0.0, bad=0, layer_dau="")
            if che_do == "raw":
                ids = tok(_prompt(n_chunk), return_tensors="pt").to("cuda")
            else:
                # Copy đúng `models.py:206-221`, kể cả nhánh dự phòng. Nhánh đó quan
                # trọng: nếu wrapper template KHÔNG nhận `enable_thinking` thì
                # `except TypeError` gọi lại KHÔNG có cờ đó — tức thinking mode về
                # MẶC ĐỊNH (bật) một cách âm thầm. Qwen3 thinking trace nổi tiếng trôi
                # sang tiếng Trung và dài dòng lặp lại; khớp cả hai triệu chứng.
                msg = [
                    {"role": "system", "content": SYSTEM_V92},
                    {"role": "user", "content": _prompt(n_chunk)},
                ]
                kw = dict(tokenize=True, add_generation_prompt=True,
                          return_dict=True, return_tensors="pt")
                try:
                    ids = tok.apply_chat_template(msg, enable_thinking=False, **kw)
                    nhanh = "enable_thinking=False nhận"
                except TypeError:
                    txt0 = tok.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    ids = tok(txt0, return_tensors="pt")
                    nhanh = "TypeError -> BỎ enable_thinking (thinking mặc định)"
                ids = {k: v.to("cuda") for k, v in dict(ids).items() if hasattr(v, "to")}
                print(f"  [chat] {nhanh}")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
            sinh = out[0][ids["input_ids"].shape[1]:]
            txt = tok.decode(sinh, skip_special_tokens=True)
            cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in txt)
            # Vị trí CJK đầu tiên THEO TOKEN, không theo ký tự: nó trả lời "sinh bao nhiêu
            # token thì trôi", tức có hạ max_new_tokens để né được hay không.
            vt = next(
                (
                    j
                    for j in range(len(sinh))
                    if any(0x4E00 <= ord(c) <= 0x9FFF for c in tok.decode(sinh[j : j + 1]))
                ),
                None,
            ) if cjk else None
            lap = _lap_5gram(sinh.tolist())
            # `skip_special_tokens=True` xoá mốc <think> nếu nó là special token, nên
            # decode lại KHÔNG bỏ special để thấy được. Nếu có thì thinking đang rò.
            tho = tok.decode(sinh, skip_special_tokens=False)
            think = sum(tho.count(m) for m in ("<think>", "</think>", "<|thinking|>"))
            ty = f"{100*stat['max']/FP16_MAX:.1f}%" if stat["max"] else "tắt hook"
            print(f"{dt_name:<9} {che_do:<4} {n_chunk} chunk · prompt {ids['input_ids'].shape[1]:>5} token"
                  f" · sinh {len(sinh)} token")
            print(f"  |act| max {stat['max']:>12.1f}  = {ty} trần fp16 · tensor không hữu hạn {stat['bad']}"
                  + (f" (đầu: {stat['layer_dau']})" if stat["layer_dau"] else ""))
            print(f"  lặp 5-gram {100*lap:5.1f}% · CJK {cjk} ký tự · mốc thinking {think}"
                  + (f", token đầu #{vt}/{len(sinh)}" if vt is not None else ""))
            print(f"  đầu : {txt[:150]!r}")
            print(f"  cuối: {txt[-150:]!r}\n", flush=True)

        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
