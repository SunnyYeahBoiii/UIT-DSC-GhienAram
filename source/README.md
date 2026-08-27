# UIT DSC 2026 Task 2 — LegalQA v9.2

Baseline public đang được khóa tại METEOR **0.5007** bằng
`configs/selector_v5007.yaml`. Notebook inference mặc định bỏ qua selector
attach ngoài để tránh vô tình dùng lại calibration v9.3 bị regression; xem
`BASELINE_V5007.md`.

Runtime hiện tại là thử nghiệm **guarded-v5007**: giữ frozen selector làm mốc,
nhưng thêm windowed reranking, kiểm tra citation/con số/ý hỏi và Qwen repair có
giới hạn. Xem `INFERENCE_GUARDED_V5007.md`. Do chunking có thêm tiêu đề kỹ thuật,
dense index baseline cũ phải được dựng lại bằng source này.

Pipeline offline cho Kaggle T4, chỉ dùng dữ liệu chính thức và các checkpoint mã nguồn mở đã được BTC phê duyệt. Tổng tham số runtime vẫn được runner kiểm tra phải nhỏ hơn 4B. BARTpho đã bị loại.

## Những phần mới trong v9/v9.1

- Thống kê độ dài, số câu, kiểu mở đầu, tỷ lệ viện dẫn/liệt kê theo loại câu hỏi từ `train.json`; dùng p50/p75 làm mục tiêu và p90/p95 làm giới hạn.
- Hậu xử lý Qwen theo ranh giới câu hoàn chỉnh, xóa câu lặp và log số từ trước/sau.
- Exact dense kNN trên câu hỏi official train; dùng tối đa hai đáp án gần nhất làm ví dụ **văn phong**, không coi chúng là căn cứ.
- Candidate `train-memory` chỉ được mở khi cosine vượt ngưỡng cao; ngưỡng này được hiệu chỉnh trên grouped-dev.
- Candidate oracle và selector tuning trên 300–500 câu dev, giữ toàn bộ câu có cùng template ở cùng một phía để giảm leakage.
- QLoRA Qwen3.5-2B bằng official answer và evidence truy hồi từ corpus; checkpoint chứa model/optimizer/scheduler/RNG, tự dừng an toàn trước hard limit 12 giờ, tạo recovery ZIP nguyên tử và tự resume đúng `train_scope`.
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
       -> quality audit + bounded Qwen repair for severe defects only
       -> gated Agentic Qwen for complex/low-confidence questions only
       -> BM25 tool call + citation/temporal/relation legal-rerank
  -> learned answer-length compaction
  -> ViLegalBERT + confidence-aware selector
  -> submission.zip
```

Chi tiết nằm trong [PIPELINE_V9.md](PIPELINE_V9.md). Hướng dẫn Kaggle nằm trong [kaggle/README_INFERENCE_ONLY_T4.md](kaggle/README_INFERENCE_ONLY_T4.md).

## Thứ tự chạy khuyến nghị

1. Import `kaggle/legalqa_dsc2026_inference_only_t4.ipynb` từ source hiện tại,
   đặt `DENSE_INDEX_ONLY=True`; lưu `/kaggle/working/legalqa_dense_index` thành
   Dataset. Chỉ dùng manifest schema `global-dense-v2-structured-headings`.
2. Import `kaggle/legalqa_dsc2026_v9_qlora_t4.ipynb`, đặt `TRAIN_SCOPE='split'`. Nếu manifest báo `paused`, Save Version rồi tạo Dataset từ `/kaggle/working/qlora_recovery_latest.zip` và attach vào session sau; nếu báo `complete`, lưu adapter split.
3. Attach dense index và completed split adapter; chạy `kaggle/legalqa_dsc2026_v9_selector_calibration_t4.ipynb` bằng source v9.2. Lưu `tuned_selection.yaml` cùng báo cáo candidate oracle. Không dùng trace/trọng số calibration từ Agentic cũ vì candidate evidence và prior đã thay đổi.
4. Chạy lại notebook QLoRA với `TRAIN_SCOPE='full'`. Không attach state `split` làm state resume cho bước này; code cũng từ chối scope sai.
5. Attach dense index, completed adapter `full` và đúng một `tuned_selection.yaml`; chạy notebook inference với `DENSE_INDEX_ONLY=False`.
6. Nộp duy nhất `/kaggle/working/submission.zip`.

Ở mode public cuối, notebook bật ba guard bắt buộc: adapter QLoRA `full`,
`tuned_selection.yaml` và global dense index phải tồn tại, hoàn chỉnh và đúng
cấu trúc. Train-memory chỉ bắt buộc khi `TRAIN_MEMORY=True`; phải giữ nguyên
giá trị này giữa các session resume. Thiếu một artifact sẽ dừng trước khi tải model, không âm thầm
dùng base Qwen/selector mặc định hoặc dựng lại index trong phiên inference.
`MAX_RUNTIME_HOURS=11.5` khiến inference dừng tại ranh giới câu hỏi, ghi recovery
ZIP và kết thúc notebook thành công trước giới hạn 12 giờ. Khi manifest báo
`paused`, lưu ZIP thành Dataset và attach vào phiên kế tiếp; chỉ nộp khi manifest
báo `complete`.

Cả ba notebook v9.2 chỉ nhận source có `SOURCE_CONTRACT.json` khớp đầy
đủ release, Agentic/cache/selector schema và 12 wheel offline. Vì vậy bản
`legalqa_source` cũ nằm kèm trong Dataset QLoRA split/full sẽ bị loại thay
vì có thể được chọn nhầm theo tên thư mục.

Runner public bắt buộc adapter có `qlora_scope.json` với `train_scope=full`;
runner calibration bắt buộc `train_scope=split`. Các bản sao source, adapter,
dense index, recovery và selector weights chỉ được tự gộp khi định danh nội dung
trùng nhau; nội dung khác nhau vẫn dừng sớm. Wheelhouse cũng được kiểm đúng phiên
bản, không chỉ đúng tên package.

CLI vẫn cho phép thí nghiệm ablation bằng Qwen base, nhưng notebook public cuối
cố ý yêu cầu adapter `full` để tránh nộp nhầm cấu hình yếu hơn. Calibration tự
trả về trọng số baseline nếu METEOR trên grouped holdout không cải thiện, thay
vì xuất một cấu hình tuned bị overfit.

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
