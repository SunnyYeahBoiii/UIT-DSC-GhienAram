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
  cùng cần thiết nếu có **một** nguyên nhân thượng nguồn. Đã đo (bảng dưới): nguyên
  nhân là base model chưa SFT, không phải thinking mode và không phải CJK drift. Nên
  **đừng** chép sang v10 câu "không trình bày quá trình suy luận nội bộ" (chữa bệnh
  không tồn tại — probe đo `mốc thinking 0` ở cả 6 cấu hình) lẫn regex phạt CJK
  (`CJK 0 ký tự` ở cả 6 cấu hình).
- **Qwen SFT** — chưa dựng.
- **Sanitizer câu trả lời cuối** — chưa dựng.

## Nghi can của bug generator (đã đo — v10-probe-and-eval v5, T4 x2)

Triệu chứng báo cáo: nhồi > 1 chunk thì model lặp từ và sinh ký tự Trung Quốc.
**Không tái lập được trên base model.** Đo 6 cấu hình (fp16/fp32 × raw/chat × 1/3/5
chunk), mỗi cấu hình sinh 1300 token:

| n_chunk | lặp 5-gram raw | lặp 5-gram chat |
|---|---|---|
| 1 | 72,7% | 98,2% |
| 3 | 71,4% | 55,4% |
| 5 | 71,4% | 3,2% |

Lặp **giảm** theo n_chunk, nặng nhất ở 1 chunk — ngược hẳn triệu chứng báo cáo. CJK 0
ký tự ở cả 6. Nên nguyên nhân nằm ở tầng TRÊN base model.

| nghi can | trạng thái |
|---|---|
| fp16 overflow | **loại** — max \|activation\| 62,9 / trần 65.504 (lượt đo trước); lượt này fp32 raw 1/3/5 chunk trả text TRÙNG TỪNG KÝ TỰ với fp16, `tensor không hữu hạn 0` |
| sampling | **loại** — `do_sample: false`, `repetition_penalty 1,1`, `no_repeat_ngram_size 5` đều đã bật |
| Qwen3 thinking mode | **loại** — `enable_thinking=False` được nhận, `mốc thinking 0` ở cả 6 cấu hình |
| nhét nhiều chunk | **loại** — lặp giảm 98,2% → 3,2% khi tăng 1 → 5 chunk |
| sinh CJK | **không tái lập** — 0 ký tự CJK ở cả 6 cấu hình |
| LoRA adapter đã SFT | **nghi can chính còn lại** — probe chỉ nạp base; `main` có commit `slice-aware loss`, prompt loss không mask sẽ dạy model sinh lại prompt = lặp |

Nhưng base model **cũng không dùng được như hiện tại**: 5/6 cấu hình chạy hết trần 1300
token mà không tự dừng, và nội dung là bám cụm rồi đếm số (`"...18 tuổi trở lên;"` tới
số 80; `"245, 246, 247, ..."`). Ca duy nhất tự dừng là chat 3 chunk (161 token) và cũng
là ca duy nhất dẫn được số Điều. Đây là hành vi base chưa SFT, không phải lỗi số học.

**Hệ quả:** Qwen SFT là điều kiện CẦN của v10, không phải bước tối ưu.

Cảnh báo về chỉ số: chat 5 chunk có lặp 5-gram 3,2% — sạch nhất bảng — nhưng output là
`"245, 246, 247, ..."`. Đếm số không sinh 5-gram lặp nào. Đừng dùng riêng lặp 5-gram
làm cổng chất lượng.
