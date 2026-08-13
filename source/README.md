# UIT DSC 2026 Task 2 — LegalQA v9.2

Pipeline offline cho Kaggle T4, chỉ dùng dữ liệu chính thức và các checkpoint mã nguồn mở đã được BTC phê duyệt. Tổng tham số runtime vẫn được runner kiểm tra phải nhỏ hơn 4B. BARTpho đã bị loại.

## Những phần mới trong v9/v9.1

- Thống kê độ dài, số câu, kiểu mở đầu, tỷ lệ viện dẫn/liệt kê theo loại câu hỏi từ `train.json`; dùng p50/p75 làm mục tiêu và p90/p95 làm giới hạn.
- Hậu xử lý Qwen theo ranh giới câu hoàn chỉnh, xóa câu lặp và log số từ trước/sau.
- Exact dense kNN trên câu hỏi official train; dùng tối đa hai đáp án gần nhất làm ví dụ **văn phong**, không coi chúng là căn cứ.
- Candidate `train-memory` chỉ được mở khi cosine vượt ngưỡng cao; ngưỡng này được hiệu chỉnh trên grouped-dev.
- Candidate oracle và selector tuning trên 300–500 câu dev, giữ toàn bộ câu có cùng template ở cùng một phía để giảm leakage.
- QLoRA Qwen3.5-2B bằng official answer và evidence truy hồi từ corpus; checkpoint chứa model/optimizer/scheduler/RNG và tự resume đúng `train_scope`.
- Router v9.1 chỉ mở Complex khi có complex cue, retrieval thực sự yếu hoặc nhiều văn bản gần hòa điểm; quan hệ phổ biến trong corpus không còn tự kích hoạt Complex.
- Mỗi graph hop được rerank và relevance-gate trước khi trở thành frontier của vòng sau; mặc định tối đa hai vòng, sáu evidence và bốn văn bản.
- Document diversity dùng bonus nhỏ thay vì cưỡng ép kéo tài liệu điểm thấp vào evidence.
- Style ceiling tăng lên 700 từ; mặc định chỉ xóa câu lặp, không hard-trim đáp án dài ở 520 từ.
- Agentic Qwen được gate theo route/confidence/tín hiệu pháp lý, thay vì chạy
  cho 100% câu hỏi. Tool-call tối đa hai vòng, chặn query lặp và lỗi không
  nghiêm trọng không còn làm dừng toàn bộ inference.
- Tool evidence được chấm riêng cho candidate Agentic; BM25 tool result được
  legal-rerank bằng citation, quan hệ và hiệu lực văn bản trước khi vào prompt.

## Pipeline

```text
Question
  -> global BM25 + exact global dense
  -> citation boosts + RRF + Vietnamese reranker
  -> Simple/Complex router + legal relation graph
  -> merged legal evidence
  -> official-train question kNN
       -> 0–2 style examples for Qwen
       -> high-threshold memory candidate
  -> evidence-grounded Qwen base/QLoRA + extractive candidate
       -> gated Agentic Qwen for complex/low-confidence questions only
       -> BM25 tool call + citation/temporal/relation legal-rerank
  -> learned answer-length compaction
  -> ViLegalBERT + confidence-aware selector
  -> submission.zip
```

Chi tiết nằm trong [PIPELINE_V9.md](PIPELINE_V9.md). Hướng dẫn Kaggle nằm trong [kaggle/README_INFERENCE_ONLY_T4.md](kaggle/README_INFERENCE_ONLY_T4.md).

## Thứ tự chạy khuyến nghị

1. Import `kaggle/legalqa_dsc2026_inference_only_t4.ipynb`, đặt `DENSE_INDEX_ONLY=True`; lưu `/kaggle/working/legalqa_dense_index` thành Dataset. Thư mục này chứa cả global corpus dense index và official train-memory.
2. Import `kaggle/legalqa_dsc2026_v9_qlora_t4.ipynb`, đặt `TRAIN_SCOPE='split'`; lưu toàn bộ `/kaggle/working/legalqa_qlora` để giữ checkpoint và adapter split.
3. Attach dense index và completed split adapter; chạy `kaggle/legalqa_dsc2026_v9_selector_calibration_t4.ipynb` bằng source v9.2. Lưu `tuned_selection.yaml` cùng báo cáo candidate oracle. Không dùng trace/trọng số calibration từ Agentic cũ vì candidate evidence và prior đã thay đổi.
4. Chạy lại notebook QLoRA với `TRAIN_SCOPE='full'`. Không attach state `split` làm state resume cho bước này; code cũng từ chối scope sai.
5. Attach dense index, completed adapter `full` và đúng một `tuned_selection.yaml`; chạy notebook inference với `DENSE_INDEX_ONLY=False`.
6. Nộp duy nhất `/kaggle/working/submission.zip`.

Cả ba notebook v9.2 chỉ nhận source có `SOURCE_CONTRACT.json` khớp đầy
đủ release, Agentic/cache/selector schema và 12 wheel offline. Vì vậy bản
`legalqa_source` cũ nằm kèm trong Dataset QLoRA split/full sẽ bị loại thay
vì có thể được chọn nhầm theo tên thư mục.

Bạn có thể bỏ bước QLoRA và dùng Qwen base, nhưng candidate-oracle nên được chạy trước khi quyết định selector nào tốt hơn. Chỉ chấp nhận trọng số tuned nếu phần grouped holdout cải thiện, không chỉ phần tune.

## Kiểm thử

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts kaggle
```

## Artifact chính

```text
/kaggle/working/legalqa_dense_index/
/kaggle/working/legalqa_qlora/qwen_qlora_adapter.zip
/kaggle/working/legalqa_selector_calibration/calibration_summary.json
/kaggle/working/legalqa_selector_calibration/selector_report.json
/kaggle/working/legalqa_selector_calibration/tuned_selection.yaml
/kaggle/working/legalqa_recovery_latest.zip
/kaggle/working/submission.zip
```

Không có agent loop vô hạn. Graph retrieval và Agentic tool-call đều tối đa hai vòng mặc định, có visited/query-set và dừng ngay khi không còn evidence phù hợp.
