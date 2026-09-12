# Pipeline v10 — runbook

Tài liệu này nói **thứ tự chạy** và **tham số đã chốt**. Lý do của từng con số nằm ở
`docs/EDA_V10.md` (mục được dẫn ở từng dòng); lý do của từng quyết định kiến trúc nằm ở
docstring đầu mỗi module. Không lặp lại ở đây — số chép ra hai chỗ là số sẽ lệch.

## Đường đi của một câu hỏi

```
                    ┌─ BM25 deep 1.000 ──┐
câu hỏi ────────────┤                    ├─ RRF k=60 ─> pool 50
                    └─ dense deep 1.000 ─┘                 │
                                                           v
                               cross-encoder rerank (zero-shot)
                                                           │
                                  top-5 Chunk ─> expand_to_evidence
                                                           │
                                  + legal graph (≤2 hop, max_extra 3)
                                                           │
                                          dung_prompt ─> Qwen ─> sanitizer
```

Chunk là đơn vị **tìm**, Evidence là đơn vị **sinh** (ADR-0001). Ranh giới đó là chỗ
`expand_to_evidence` nằm, và là lý do pipeline có hai đơn vị chứ không một.

## Tham số đã chốt

| tầng | tham số | giá trị | nguồn |
|---|---|---|---|
| chunking | size / stride | 256 / 256 từ | ADR-0001, §13 |
| bm25 | k1 / b | 1,5 / 0,75 | Okapi mặc định |
| bm25 / dense | deep | 1.000 | §15.8 |
| hybrid | RRF k | 60 | §15.8 — trộn theo **hạng**, không chuẩn hoá điểm |
| rerank | pool | 50 | §15.6 — lợi ích biên 20→50 là 0,48 điểm/phút, 50→100 chỉ 0,03 |
| rerank | max_len | 512 | `RERANK_MAX_LEN` |
| dense | max_len | 256 | §15.8 — 512 thua thật, không phải nhiễu |
| evidence | top_n / max_chars | 5 / 24.000 | §15.7; 24.000 lấy từ v9.2 `kaggle_inference_t4.yaml` |
| graph | max_hop / max_extra | 2 / 3 | ADR-0002 — có visited-set, không loop vô hạn |

`top_n=5` đếm **seed**, không đếm Evidence trả về: graph sinh thêm là ngữ cảnh bắt
buộc, không phải kết quả tìm kiếm. Ngân sách ký tự cắt theo **cụm** (seed + phần graph
của nó), không theo từng Evidence — xem `retrieve.evidence_day_du`.

## Recall (đo 13-09, T4x2, một lượt chạy, n=500)

| pool | | R@1 | **R@5** | trần |
|---|---|---:|---:|---:|
| BM25 | BM25 | 56,4 | 81,0 | 96,2 |
| BM25 | +rerank | 75,2 | **92,0** | |
| hybrid | +rerank | 77,2 | **93,6** | 97,6 |

Dense mua +1,6 R@5 với 118M params, 0,22 GB index, +0,02 s/câu. Chi tiết và vì sao ước
cũ (+0,9) thấp một cách hệ thống: §15.8.

## Ngân sách params — trần cứng 4B

| | params |
|---|---:|
| generator Qwen3.5-2B | 2,274B |
| reranker `AITeamVN/Vietnamese_Reranker` | 0,568B |
| dense `intfloat/multilingual-e5-small` | 0,118B |
| **tổng** | **2,96B** (dư 1,04B) |

## Chạy

Dựng dữ liệu (một lần, không GPU):

```bash
PYTHONPATH=src python scripts/clean_data.py
PYTHONPATH=src python scripts/build_chunks.py     # -> 287.358 chunk
PYTHONPATH=src python scripts/build_gold.py       # weak gold, §15.1
```

Dựng dense index (cần GPU, 421s trên T4):

```bash
PYTHONPATH=src python scripts/build_dense.py --model intfloat/multilingual-e5-small
```

Đo lại recall: xem §15.9. Đẩy lên Kaggle:

```bash
./scripts/push_dataset.sh     # codebase -> Dataset legalqa-v10-src
./scripts/push_kernel.sh      # notebook -> kernel
```

> `kernels push` **RESET accelerator** về default (P100, sm_60 — `.half()` vỡ). Sau mỗi
> lần push phải vào UI: Settings → Accelerator → GPU T4 x2 → Save & Run All. Đã đo, xem
> comment cuối `scripts/push_kernel.sh`.

## Kiểm trước khi báo xong

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests
```

## Còn thiếu

- **System prompt** — `src/legalqa/prompt.py` cố tình chưa có. v9.2 `SYSTEM_PROMPT` chứa
  ba miếng vá triệu chứng ở ba tầng khác nhau (dặn model "không trình bày quá trình suy
  luận nội bộ", regex phạt CJK trong `verifier.py`, xoá câu trùng lặp). Ba miếng vá chỉ
  cùng cần thiết nếu có **một** nguyên nhân thượng nguồn. Đang đo nguyên nhân đó
  (`scripts/probe_fp16_overflow.py`); chép nguyên system prompt sang v10 trước khi biết
  kết quả là chép cả vá lẫn bệnh.
- **Qwen SFT** — chưa dựng.
- **Sanitizer câu trả lời cuối** — chưa dựng.

## Nghi can của bug generator (đang đo)

Triệu chứng: nhồi > 1 chunk thì model lặp từ và sinh ký tự Trung Quốc.

| nghi can | trạng thái |
|---|---|
| fp16 overflow | **loại** — max \|activation\| 62,9 / trần 65.504; fp16 ≡ fp32 byte-identical |
| sampling | **loại** — `do_sample: false`, `repetition_penalty 1,1`, `no_repeat_ngram_size 5` đều đã bật |
| Qwen3 thinking mode | **đang đo** — `main:core/models.py` có `except TypeError` gọi lại `apply_chat_template` **bỏ** `enable_thinking`, tức thinking mặc định BẬT âm thầm |
| LoRA adapter đã SFT | chưa loại — probe chỉ nạp base; `main` có commit `slice-aware loss`, prompt loss không mask sẽ dạy model sinh lại prompt = lặp |
