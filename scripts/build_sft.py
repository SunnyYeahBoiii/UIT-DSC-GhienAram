"""Dựng tập SFT: câu hỏi -> prompt (Evidence thật từ retrieval) -> Reference answer.

Vì sao phải chạy retrieval chứ không dùng thẳng gold: `data/official/train.json` chỉ có
hai trường `question` và `answer` — KHÔNG có context. Context lúc inference do retrieval
sinh ra, có nhiễu, thiếu Điều, sai thứ tự. Train trên gold sạch rồi thả vào context nhiễu
là train-inference mismatch: model chưa từng thấy căn cứ sai nằm cạnh căn cứ đúng.

Nên dùng chính `Retriever` mà inference dùng, cùng tham số.

Chạy đo ngân sách:
    # máy dev, BM25 thuần (cận dưới — không GPU nên không rerank được)
    QWEN_TOKENIZER=... PYTHONPATH=src python scripts/build_sft.py --do-token --n 300
    # Kaggle, có cross-encoder
    QWEN_TOKENIZER=... PYTHONPATH=src python scripts/build_sft.py --do-token --n 300 \
        --model $MODEL --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from legalqa.bm25 import ChunkIndex
from legalqa.chunking import Chunk
from legalqa.corpus import Article, Document
from legalqa.prompt import dung_prompt
from legalqa.retrieve import Retriever, chon_evidence

CLEAN = Path(os.environ.get("CLEAN_DIR", "data/clean"))

# Các cấu hình quét trong MỘT lần rank. Rerank là phần đắt (pool 50 cặp qua XLM-R large
# mỗi câu); `chon_evidence` chỉ đọc lại list đã xếp hạng nên gần như miễn phí. Tính lại
# `rank()` cho từng cấu hình là nhân thời gian GPU lên đúng số cấu hình mà không đổi
# lấy một con số mới nào.
CAU_HINH = ((5, 24_000), (5, 12_000), (3, 12_000), (5, 8_000), (3, 8_000))

# Cấu hình chốt sau lần đo có rerank trên Kaggle (n=300, 268 câu có weak gold):
#
#   | top_n | max_chars | gold  | ev/câu | p50  | max  | >4096 |
#   |     5 |    24.000 | 89,2% |    5,0 | 5007 | 8341 | 83,0% |
#   |     5 |    12.000 | 88,1% |    3,4 | 3378 | 5109 |  9,0% |
#   |     3 |    12.000 | 86,6% |    2,9 | 3014 | 5109 |  4,3% |
#   |     5 |     8.000 | 83,2% |    2,1 | 2249 | 3740 |  0,0% |
#
# Quyết định của người dùng (2026-09-14): BỎ trần ký tự, mỗi câu lấy đủ `top_n=5`
# Evidence. Đọc bảng trên theo đúng cột đang đánh đổi: `max_chars=12.000` cắn trước
# `top_n` (3,4 Evidence/câu chứ không phải 5), nên nó là thứ đang vứt bằng chứng —
# và 21,3% câu train bị loại vì "retrieval_truot_gold" là loại SAU khi đã cắt. Trần ký
# tự vừa cắt bằng chứng vừa bị tính vào tỉ lệ trượt.
#
# `chon_evidence` dừng theo `top_n` HOẶC `max_chars`, cái nào chặn trước thì thắng, nên
# tắt trần bằng một số lớn hơn mọi prompt có thể có thay vì sửa chữ ký hàm. 10**9 ký tự
# > tổng corpus, tức nhánh ký tự không bao giờ cắn — `top_n` là ràng buộc duy nhất.
TOP_N, MAX_CHARS = 5, 10**9

# MAX_SEQ giữ lại làm HẰNG SỐ BÁO CÁO, không còn là bộ lọc: `giu_mau` đã bỏ nhánh
# "vuot_max_seq" theo cùng quyết định trên. Vẫn in phân phối quanh mốc này để biết bao
# nhiêu mẫu đắt hơn ngân sách VRAM cũ. Cổng thật sự bây giờ nằm ở `train_sft.py
# --max-seq`, nơi mẫu quá dài bị bỏ ngay lúc mã hoá — một chỗ thay vì hai.
MAX_SEQ = 6_144


def nap_docs() -> dict[str, Document]:
    """Dựng lại Document từ corpus.jsonl — passage trong đó ĐÃ qua `dewrap_passage()`
    lúc `clean_data.py` ghi, nên mọi offset Article khớp sẵn. Đọc lại raw JSON ở đây là
    lệch offset toàn bộ."""
    docs: dict[str, Document] = {}
    with (CLEAN / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d = Document(
                doc_id=o["doc_id"], passage=o["passage"], slug=o["slug"], link=o["link"]
            )
            d.articles = [
                Article(*a) if isinstance(a, list) else Article(**a)
                for a in o["articles"]
            ]
            docs[d.doc_id] = d
    return docs


def nap_chunks() -> dict[str, Chunk]:
    keys = ("chunk_id", "doc_id", "start", "end", "text")
    chunks: dict[str, Chunk] = {}
    with (CLEAN / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            c = Chunk(**{k: o[k] for k in keys})
            chunks[c.chunk_id] = c
    return chunks


def nap_index(chunks: dict[str, Chunk]) -> ChunkIndex:
    """Nạp bm25.pkl, tự build nếu chưa có.

    `build_chunks.py` KHÔNG sinh bm25.pkl — nó do `eval_retrieval.load_index()` dựng
    rồi cache. Trên Kaggle chỉ chạy clean/chunk/gold nên file không tồn tại, và gọi
    thẳng `ChunkIndex.load` là chết sau 26 phút dựng dữ liệu, đúng ở bước cuối cùng.
    Build mất ~50s, rẻ hơn nhiều so với mất cả lượt chạy.
    """
    if (CLEAN / "bm25.pkl").exists():
        return ChunkIndex.load(CLEAN / "bm25.pkl")
    # Ghi cache cạnh CLEAN nếu ghi được, không thì vào cwd: khi CLEAN là output của
    # kernel trước thì nó nằm dưới /kaggle/input, read-only. Build xong mà chết ở
    # `save` là mất trắng 50s và cả lượt chạy.
    cache = CLEAN / "bm25.pkl" if os.access(CLEAN, os.W_OK) else Path("bm25.pkl")
    if cache.exists():
        return ChunkIndex.load(cache)
    print(f"không có bm25.pkl trong {CLEAN} — build (~50s) -> {cache}", flush=True)
    idx = ChunkIndex()
    idx.build(list(chunks.values()))
    idx.save(cache)
    return idx


def nap_dense(model_path: str, chunks: dict):
    """(BiEncoder, DenseIndex) cho nhánh dense. Encode một lần rồi ghi cache ra CLEAN.

    Vì sao encode TẠI ĐÂY chứ không đòi dataset dựng sẵn: 436s trên T4 cho 287.358 chunk
    (§15.8) — rẻ hơn một vòng publish dataset 0,22 GB rồi verify nó đã materialize. Nhưng
    vẫn ghi cache: kernel bị giết giữa pha 2 rồi chạy lại thì 7 phút đó không phải trả lại.

    THỨ TỰ hàng của ma trận phải khớp THỨ TỰ `chunks.jsonl` — `DenseIndex` chỉ giữ list
    chunk_id song song, không có dict tra ngược. Lặp trên `chunks.values()` giữ đúng thứ
    tự chèn của dict (`nap_chunks` đọc tuần tự file), nên hai bên khớp; sắp lại một bên
    là lệch toàn bộ ánh xạ mà không có assert nào bắt được.
    """
    import numpy as np

    from legalqa.dense import BiEncoder, DenseIndex

    enc = BiEncoder(model_path, batch_size=256, max_len=256)
    npy, ids_txt = CLEAN / "dense.npy", CLEAN / "dense.ids"
    if npy.exists() and ids_txt.exists():
        print(f"nạp dense cache: {npy}", flush=True)
        return enc, DenseIndex.load(npy, ids_txt)
    ids = list(chunks.keys())
    t0 = time.time()
    M = enc.encode([chunks[i].text for i in ids])
    print(f"encode dense {M.shape} {time.time()-t0:.0f}s", flush=True)
    try:
        np.save(npy, M)
        ids_txt.write_text("\n".join(ids), encoding="utf-8")
    except OSError as e:
        # CLEAN có thể là thư mục chỉ-đọc (mount dataset trực tiếp). Mất cache thì chỉ
        # tốn lại 7 phút ở lần chạy sau, không đáng để chết sau khi đã encode xong.
        print(f"không ghi được cache dense ({e}) — bỏ qua", flush=True)
    return enc, DenseIndex(ids, M)


def nap_gold(ten: str = "gold.jsonl") -> dict[str, set[str]]:
    g: dict[str, set[str]] = {}
    with (CLEAN / ten).open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            g[o["qid"]] = set(o["gold_docs"])
    return g


def do_token(rows: list[dict], r: Retriever, tok, batch: int) -> None:
    """Đo độ dài prompt THẬT để chốt `max_seq_len`, trước khi sinh cả tập.

    Không đoán từ số ký tự: tỉ lệ token/ký tự của tiếng Việt luật khác hẳn tiếng Anh, và
    `max_chars=24_000` kế thừa từ v9.2 (configs/kaggle_inference_t4.yaml) được đặt cho
    context Qwen 3B lúc suy luận, KHÔNG cho seq len lúc train.

    Cột `gold` là tỉ lệ Evidence trả về chứa ít nhất một Document weak-gold. Đây mới là
    đại lượng quyết định: cắt ngân sách rẻ hay đắt tuỳ nó tụt bao nhiêu, không tuỳ token
    giảm bao nhiêu. Một mẫu SFT mà context không chứa gold là mẫu dạy model bịa.
    """
    gold = nap_gold()
    # dict[(top_n, max_chars)] -> (list token prompt, list token đáp, số Evidence, gold hit)
    kq = {c: ([], [], 0, 0) for c in CAU_HINH}
    ans_tok = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        ranked = r.rank(row["question"])  # phần đắt, dùng chung cho mọi cấu hình
        g = gold.get(row["id"])
        a_tok = len(tok.encode(row["answer"]).ids)
        ans_tok[row["id"]] = a_tok
        for cfg in CAU_HINH:
            top_n, max_chars = cfg
            evs = chon_evidence(ranked, r.docs, top_n=top_n, max_chars=max_chars)
            p_tok = len(tok.encode(dung_prompt(row["question"], evs)).ids)
            pr, an, so_ev, hit = kq[cfg]
            pr.append(p_tok)
            an.append(a_tok)
            kq[cfg] = (pr, an, so_ev + len(evs), hit + bool(g and any(e.doc_id in g for e in evs)))
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f}s/câu", flush=True)

    co_gold = sum(1 for row in rows if gold.get(row["id"]))
    n = len(rows)
    print(f"\nn={n}  câu có weak gold: {co_gold}  ({(time.time()-t0)/n:.2f}s/câu)")
    print("\n| top_n | max_chars | gold | ev/câu | p50 | p90 | p95 | max | >4096 | >6144 | >8192 |")
    print("|------:|----------:|-----:|-------:|----:|----:|----:|----:|------:|------:|------:|")
    for cfg in CAU_HINH:
        pr, an, so_ev, hit = kq[cfg]
        tong = sorted(x + y for x, y in zip(pr, an))
        q = lambda xs, k: xs[int(k * (len(xs) - 1))]
        vuot = [f"{sum(1 for x in tong if x > c)/n:5.1%}" for c in (4096, 6144, 8192)]
        print(f"| {cfg[0]} | {cfg[1]:,} | {hit/max(1,co_gold):.1%} | {so_ev/n:.1f} |"
              f" {q(tong,.5)} | {q(tong,.9)} | {q(tong,.95)} | {tong[-1]} | "
              + " | ".join(vuot) + " |")
    print("\n(p50/p90/p95/max và cột >N là prompt + answer, tức seq len thật lúc train)")


def giu_mau(evs, gold_docs: set[str] | None, seq_len: int = 0) -> str:
    """Lý do loại, hoặc "" nếu giữ. Tách khỏi vòng lặp để test được không cần GPU.

    Quyết định của người dùng: mẫu mà retrieval trượt gold thì LOẠI hẳn, không chèn gold
    vào cũng không đổi thành mẫu từ chối. Lý do: chèn gold thì lúc train gold LUÔN có
    mặt, model học "đáp án chắc chắn nằm đâu đó trong đây" — đúng cơ chế sinh ra bịa khi
    inference trượt. Còn 21,3% mẫu từ chối là tỉ lệ quá cao, model sẽ từ chối cả khi
    context đủ, mà bài thi chấm theo nội dung nên từ chối là 0 điểm.

    Giá phải trả, ghi ra để đừng quên lúc đọc eval: model chưa từng thấy context thiếu
    gold, nên tỉ lệ bịa thật lúc inference cao hơn những gì dev set (lọc cùng cách) cho
    thấy. Dev set phải giữ nguyên mẫu trượt mới đo trung thực.
    """
    if not gold_docs:
        return "khong_co_weak_gold"
    if not any(e.doc_id in gold_docs for e in evs):
        return "retrieval_truot_gold"
    # Không còn loại theo `seq_len`. Trước đây trần ký tự 12.000 đã cắt bằng chứng RỒI
    # mới đo seq_len, nên mẫu "vuot_max_seq" là mẫu dài kể cả sau khi đã bị cắt — loại
    # nó là loại đúng những câu cần nhiều Điều nhất. Mẫu quá dài bây giờ bị bỏ ở
    # `train_sft.py --max-seq` lúc mã hoá, đúng nơi ràng buộc VRAM thật sự nằm.
    return ""


def sinh_tap(rows: list[dict], r: Retriever, tok, ra: Path, ten_gold: str, giu_het: bool) -> None:
    """Sinh tập SFT thật: câu hỏi -> Evidence từ chính Retriever của inference -> prompt.

    Ghi `prompt` và `answer` TÁCH RIÊNG chứ không nối sẵn: prompt-loss masking cần biết
    ranh giới, và nối rồi tách lại bằng chuỗi là mời lỗi khi answer tình cờ chứa
    "Trả lời:". 69,8% 8-gram của answer là nguyên văn từ gold passage (§EDA) nên đây là
    bài trích — masking phần prompt là bắt buộc, không phải tuỳ chọn.
    """
    gold = nap_gold(ten_gold)
    bo = {"khong_co_weak_gold": 0, "retrieval_truot_gold": 0}
    giu = 0
    t0 = time.time()
    with ra.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            ranked = r.rank(row["question"])
            evs = chon_evidence(ranked, r.docs, top_n=TOP_N, max_chars=MAX_CHARS)
            prompt = dung_prompt(row["question"], evs)
            seq = len(tok.encode(prompt).ids) + len(tok.encode(row["answer"]).ids)
            ly_do = giu_mau(evs, gold.get(row["id"]), seq)
            if ly_do:
                bo[ly_do] += 1
            # `giu_het` cho tập dev: ghi cả mẫu bị loại, kèm `ly_do`. Loại mẫu khó khỏi
            # dev là tự nâng điểm, và mẫu retrieval trượt gold chính là chỗ duy nhất
            # phát hiện được model bịa khi context thiếu.
            if ly_do and not giu_het:
                continue
            giu += not ly_do
            f.write(json.dumps({
                "id": row["id"],
                "question": row["question"],
                "prompt": prompt,
                "answer": row["answer"],
                "seq_len": seq,
                "doc_ids": [e.doc_id for e in evs],
                "ly_do_loai": ly_do,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                con = (len(rows) - i - 1) * (time.time() - t0) / (i + 1)
                print(f"  ... {i+1}/{len(rows)}  giữ {giu}  còn ~{con/60:.0f} phút", flush=True)

    n = len(rows)
    print(f"\nvào {n:,}  giữ {giu:,} = {giu/n:.1%}")
    for k, v in bo.items():
        print(f"  loại {k:24s} {v:,} = {v/n:.1%}")
    print(f"  -> {ra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-token", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="0 = cả tập")
    ap.add_argument("--model", default="", help="đường dẫn reranker; rỗng = BM25 thuần")
    # PHẢI khớp `infer.py --dense`: model học đọc prompt dựng từ đúng nhánh retrieval mà
    # nó sẽ gặp lúc sinh. Train trên prompt BM25 rồi infer bằng hybrid là đổi phân phối
    # evidence dưới chân model — cùng họ lỗi "vượt độ dài từng thấy khi train" đã sinh
    # ra vòng lặp ở lượt 4.
    ap.add_argument("--dense", default="", help="bi-encoder; rỗng = BM25 thuần")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--ra", default="", help="file jsonl đầu ra; mặc định CLEAN/sft.jsonl")
    ap.add_argument("--nguon", default="train_sft.jsonl")
    ap.add_argument("--gold", default="gold.jsonl")
    ap.add_argument("--giu-het", action="store_true",
                    help="ghi cả mẫu bị loại kèm ly_do_loai — dùng cho tập dev")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (CLEAN / a.nguon).open(encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]

    reranker = None
    if a.model:
        from legalqa.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(a.model, batch_size=a.batch_size)
    chunks = nap_chunks()
    encoder = dense_idx = None
    if a.dense:
        encoder, dense_idx = nap_dense(a.dense, chunks)
    r = Retriever(
        index=nap_index(chunks),
        chunks=chunks,
        docs=nap_docs(),
        reranker=reranker,
        encoder=encoder,
        dense=dense_idx,
        pool=a.pool,
    )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.environ["QWEN_TOKENIZER"])
    if a.do_token:
        do_token(rows, r, tok, a.batch_size)
        return
    # Không mặc định ghi vào CLEAN: khi CLEAN là output kernel trước thì nó read-only,
    # và lỗi đó chỉ nổ SAU 3,5 giờ GPU. Cùng lớp lỗi với bm25.pkl ở `nap_index`.
    mac_dinh = CLEAN / "sft.jsonl" if os.access(CLEAN, os.W_OK) else Path("sft.jsonl")
    sinh_tap(rows, r, tok, Path(a.ra) if a.ra else mac_dinh, a.gold, a.giu_het)


if __name__ == "__main__":
    main()
