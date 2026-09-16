"""Suy luận: câu hỏi -> retrieval -> prompt -> sinh đáp án. Dùng cho cả hai việc:

1. Nộp bài: `--cau-hoi data/official/public-official.json --submission`
2. Đo trên nhóm câu ĐÃ BỊ LOẠI khỏi tập train (`retrieval_truot_gold`,
   `khong_co_weak_gold`). `build_sft.giu_mau()` đã ghi rõ cái giá của việc loại: model
   chưa từng thấy context thiếu gold, nên tỉ lệ bịa thật lúc inference CAO HƠN những gì
   dev set cho thấy. Chỉ có cách chạy trên chính nhóm đó mới biết cao hơn bao nhiêu.

Hai pha tách rời và mỗi pha ghi tăng dần ra jsonl. Vì sao: pha retrieval mất ~2s/câu,
pha sinh ~12s/câu; 1.000 câu là ~4 giờ, dài hơn nửa phiên Kaggle. Kernel chết giữa chừng
mà không có file tăng dần là mất trắng cả lượt GPU — đã mất một lượt vì đúng lỗi này.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "src")

from legalqa.prompt import dung_prompt
from legalqa.retrieve import Retriever, chon_evidence

from build_sft import CLEAN, MAX_CHARS, TOP_N, nap_chunks, nap_dense, nap_docs, nap_index


def nap_cau_hoi(duong: Path) -> list[dict]:
    """Đọc cả hai dạng: dict BTC (`{id: {question, answer}}`) và jsonl của ta.

    `answer` của tập test là None — giữ nguyên chứ không đổi thành "", để chỗ chấm điểm
    phân biệt được "không có đáp án" với "đáp án rỗng".
    """
    if duong.suffix == ".jsonl":
        return [json.loads(l) for l in duong.open(encoding="utf-8")]
    d = json.loads(duong.read_text(encoding="utf-8"))
    return [{"id": k, "question": v["question"], "answer": v.get("answer")}
            for k, v in d.items()]


def da_co(duong: Path) -> set[str]:
    """Id đã ghi ở lượt trước. Dòng cuối có thể vỡ nếu kernel bị giết giữa lúc ghi —
    bỏ dòng vỡ thay vì chết, mất 1 câu rẻ hơn mất cả file."""
    if not duong.exists():
        return set()
    ids = set()
    for line in duong.open(encoding="utf-8"):
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return ids




def pha_prompt(rows: list[dict], r: Retriever, ra: Path, han: float,
               max_chars: int = MAX_CHARS) -> None:
    xong = da_co(ra)
    con = [x for x in rows if x["id"] not in xong]
    print(f"pha 1 retrieval: {len(xong)} đã có, còn {len(con)}", flush=True)
    t0 = time.time()
    with ra.open("a", encoding="utf-8") as f:
        for i, row in enumerate(con):
            evs = chon_evidence(r.rank(row["question"]), r.docs,
                                top_n=TOP_N, max_chars=max_chars)
            f.write(json.dumps({
                "id": row["id"],
                "question": row["question"],
                "prompt": dung_prompt(row["question"], evs),
                "answer": row.get("answer"),
                "nhom": row.get("nhom", ""),
                "doc_ids": [e.doc_id for e in evs],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                t = (time.time() - t0) / (i + 1)
                print(f"  ... {i+1}/{len(con)}  {t:.2f}s/câu  còn ~{t*(len(con)-i-1)/60:.0f} phút",
                      flush=True)
            if time.time() - t0 > han:
                print("  hết ngân sách giờ cho pha 1 — dừng, chạy lại để tiếp", flush=True)
                return


def nap_model(base: str, adapter: str):
    import torch
    import transformers

    model = None
    for ten in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        lop = getattr(transformers, ten, None)
        if lop is None:
            continue
        try:
            model = lop.from_pretrained(base, dtype=torch.float16,
                                        attn_implementation="sdpa").cuda()
            print("nạp bằng", ten, flush=True)
            break
        except (ValueError, KeyError) as e:
            print(f"{ten} không nhận: {e}", flush=True)
    assert model is not None, "không lớp Auto* nào nạp được model nền"
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print("gắn adapter:", adapter, flush=True)
    model.eval()
    # `use_cache` phải bật TƯỜNG MINH: thiếu KV cache thì sinh 1.024 token là 1.024 lần
    # forward lại cả tiền tố ~4.000 token. Không ném lỗi, chỉ đốt sạch phiên GPU.
    model.config.use_cache = True
    return model


def pha_sinh(prompts: Path, ra: Path, base: str, adapter: str,
             max_new: int, bs: int, han: float) -> None:
    import torch
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in prompts.open(encoding="utf-8")]
    xong = da_co(ra)
    con = [x for x in rows if x["id"] not in xong]
    print(f"pha 2 sinh: {len(xong)} đã có, còn {len(con)}", flush=True)
    if not con:
        return
    # Sắp theo độ dài prompt: batch toàn câu dài-ngắn lẫn lộn thì pad tới câu dài nhất,
    # phần pad cũng phải chạy qua model. Gom câu cùng cỡ vào một batch rẻ hơn rõ rệt.
    con.sort(key=lambda x: len(x["prompt"]))
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = nap_model(base, adapter)
    t0 = time.time()
    with ra.open("a", encoding="utf-8") as f:
        for i in range(0, len(con), bs):
            lo = con[i : i + bs]
            vb = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}],
                                          add_generation_prompt=True, tokenize=False)
                  for x in lo]
            # Pad TRÁI khi sinh: pad phải thì token cuối của chuỗi ngắn là <pad> và model
            # sinh tiếp từ đó.
            tok.padding_side = "left"
            enc = tok(vb, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            for x, seq in zip(lo, out):
                f.write(json.dumps({
                    "id": x["id"],
                    "question": x["question"],
                    "du_doan": tok.decode(seq[enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip(),
                    "dap_an": x.get("answer"),
                    "nhom": x.get("nhom", ""),
                    "doc_ids": x.get("doc_ids"),
                }, ensure_ascii=False) + "\n")
            f.flush()
            t = (time.time() - t0) / (i + len(lo))
            print(f"  ... {i+len(lo)}/{len(con)}  {t:.2f}s/câu  "
                  f"còn ~{t*(len(con)-i-len(lo))/60:.0f} phút", flush=True)
            if time.time() - t0 > han:
                print("  hết ngân sách giờ cho pha 2 — dừng, chạy lại để tiếp", flush=True)
                return


def ghi_submission(rows: list[dict], du_doan: Path, ra: Path) -> None:
    """Bài nộp phải có ĐỦ mọi id: `eval_qa` của BTC so `len(ids_preds)` với
    `len(ids_truth)` rồi `raise` — thiếu một câu là hỏng cả bài, không phải mất 1 điểm.
    Câu chưa sinh kịp ghi chuỗi rỗng."""
    co = {json.loads(l)["id"]: json.loads(l)["du_doan"]
          for l in du_doan.open(encoding="utf-8")} if du_doan.exists() else {}
    sub = {x["id"]: {"answer": co.get(x["id"], "")} for x in rows}
    thieu = sum(1 for v in sub.values() if not v["answer"])
    js = ra / "submission.json"
    js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(ra / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(js, "submission.json")
    print(f"-> {js}  ({len(sub)} câu, {thieu} rỗng)")


def cham_diem(du_doan: Path) -> None:
    """Chấm bằng đúng code BTC. Chỉ chấm câu CÓ đáp án — tập test không có."""
    from legalqa.scorer import cham, chuan_bi_nltk

    rows = [json.loads(l) for l in du_doan.open(encoding="utf-8")]
    rows = [x for x in rows if x.get("dap_an")]
    if not rows:
        print("không có đáp án tham chiếu — bỏ qua chấm")
        return
    chuan_bi_nltk()
    d = cham({x["id"]: x["du_doan"] for x in rows},
             {x["id"]: x["dap_an"] for x in rows})
    print(f"\nn={len(rows)}  " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))
    nhom = {}
    for x in rows:
        nhom.setdefault(x.get("nhom", ""), []).append(x)
    if len(nhom) > 1:
        for k, v in sorted(nhom.items()):
            dv = cham({x["id"]: x["du_doan"] for x in v},
                      {x["id"]: x["dap_an"] for x in v})
            print(f"  {k or '(không nhãn)':24s} n={len(v):4d}  "
                  + "  ".join(f"{a}={b:.4f}" for a, b in dv.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cau-hoi", required=True)
    ap.add_argument("--model", required=True, help="model nền")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rerank", default="", help="cross-encoder; rỗng = BM25 thuần")
    # Nhánh dense của hybrid search. §15.8 chốt `intfloat/multilingual-e5-small` ở
    # max_len 256: trần pool@50 96,7% -> 97,9% (+1,2), encode 287.358 chunk mất 436s trên
    # T4. ĐỪNG truyền max_len 512 — §15.8 đo ngược (hybrid@50 tụt 0,4 vì mean pooling
    # loãng) và tốn gấp đôi thời gian.
    ap.add_argument("--dense", default="", help="bi-encoder; rỗng = BM25 thuần (không hybrid)")
    ap.add_argument("--ra", default="out")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gen-bs", type=int, default=4)
    ap.add_argument("--submission", action="store_true")
    ap.add_argument("--bo-qua-pha1", action="store_true", help="prompts.jsonl đã đủ")
    ap.add_argument("--gio-pha1", type=float, default=3.0)
    ap.add_argument("--gio-pha2", type=float, default=7.0)
    # Ngân sách ký tự phải KHỚP checkpoint đang chạy, không phải khớp mặc định hiện tại.
    # `MAX_CHARS` giờ là vô hạn (10**9), nhưng adapter b318 được train trên prompt cắt ở
    # 12.000 — suy luận bằng prompt dài gấp rưỡi là đưa model vào phân phối nó chưa từng
    # thấy, cùng loại lỗi mà `LEGALQA_SO_HIEU` sinh ra. Chạy checkpoint đời cũ thì truyền
    # `--max-chars 12000`.
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    a = ap.parse_args()

    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    rows = nap_cau_hoi(Path(a.cau_hoi))
    if a.n:
        rows = rows[: a.n]
    print(f"{len(rows)} câu  |  CLEAN={CLEAN}  |  số hiệu trong prompt: "
          f"{os.environ.get('LEGALQA_SO_HIEU', '1') != '0'}", flush=True)

    prompts = ra / "prompts.jsonl"
    if not a.bo_qua_pha1:
        reranker = None
        if a.rerank:
            from legalqa.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(a.rerank, batch_size=a.batch_size)
        chunks = nap_chunks()
        encoder = dense_idx = None
        if a.dense:
            encoder, dense_idx = nap_dense(a.dense, chunks)
        r = Retriever(index=nap_index(chunks), chunks=chunks, docs=nap_docs(),
                      reranker=reranker, encoder=encoder, dense=dense_idx, pool=a.pool)
        pha_prompt(rows, r, prompts, a.gio_pha1 * 3600, a.max_chars)
        # Trả VRAM của reranker trước khi nạp Qwen: 0,568B fp16 + 2B fp16 + KV cache của
        # batch 4 × 6.000 token cùng lúc là sát trần 16 GB của T4.
        del r, reranker, encoder, dense_idx
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    du_doan = ra / "du_doan.jsonl"
    pha_sinh(prompts, du_doan, a.model, a.adapter, a.max_new, a.gen_bs, a.gio_pha2 * 3600)
    if a.submission:
        ghi_submission(rows, du_doan, ra)
    cham_diem(du_doan)


if __name__ == "__main__":
    main()
