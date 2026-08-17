# Kaggle T4 — LegalQA v9.2

## Inputs bắt buộc

Attach Dataset official chứa `train.json`, `public-official.json`, `selected-contexts/`; Dataset source v9; Dataset bốn base checkpoint đã được BTC duyệt. Internet tắt, GPU bật.

Tên Dataset không bị hard-code. Notebook tìm file/manifest theo nội dung và sẽ dừng nếu phát hiện nhiều source, nhiều dense index, nhiều completed adapter, nhiều tuned weights hoặc nhiều recovery ZIP. Recovery v8/v9.1 cũ không được tái sử dụng vì cache namespace v9.2 đã đổi theo Agentic gate/legal-rerank và candidate-specific evidence.

Agentic Qwen mặc định chỉ chạy cho route Complex, retrieval confidence dưới
`0.45`, hoặc câu có tín hiệu dẫn chiếu/thay thế/hiệu lực. Lỗi tool-call không
nghiêm trọng chỉ loại candidate của đúng câu đó và không tắt Agentic ở các câu
còn lại. Vì selector nay có
candidate và prior Agentic riêng, hãy calibration lại bằng v9.2 trước public
inference; không tái dùng `tuned_selection.yaml` sinh từ Agentic cũ.

Selector calibration ghi `/kaggle/working/legalqa_selector_calibration/selector_recovery_latest.zip`.

Với inference recovery, `AUTO_RESUME=True` nhận cả file
`legalqa_recovery_latest.zip` và thư mục cùng tên do Kaggle tự giải nén khi tạo
Dataset. Không cần thêm cell đóng gói ZIP thủ công; notebook xác thực
`recovery_state.json`, fingerprint và truyền trực tiếp ZIP/thư mục cho runner.
Nếu session hết giờ, lưu ZIP đó thành Kaggle Dataset và attach ở session kế;
notebook sẽ tự truyền đúng một recovery bundle khi `AUTO_RESUME=True`.

## Bốn notebook/run mode

### A. Dựng index

Mở `legalqa_dsc2026_inference_only_t4.ipynb`, đặt:

```python
DENSE_INDEX_ONLY = True
TRAIN_MEMORY = True
```

Lưu `/kaggle/working/legalqa_dense_index` thành Kaggle Dataset. Index có checkpoint nội bộ; chỉ manifest `status=complete` mới được dùng cho inference.

### B. QLoRA split

Mở `legalqa_dsc2026_v9_qlora_t4.ipynb`:

```python
TRAIN_SCOPE = 'split'
DEV_SIZE = 400
MAX_TRAIN_SAMPLES = None
```

Smoke test có thể đặt `MAX_TRAIN_SAMPLES=300`, nhưng adapter đó không phải adapter cuối. Khi hết session, Save Version toàn bộ working output để giữ các thư mục `checkpoint-*`, `optimizer.pt`, scheduler và RNG. Session mới chỉ attach một state có cùng scope và giữ nguyên mọi hyperparameter.

### C. Candidate oracle/selector calibration

Attach completed dense index và completed adapter split, rồi chạy `legalqa_dsc2026_v9_selector_calibration_t4.ipynb`. Kết quả quan trọng:

```text
/kaggle/working/legalqa_selector_calibration/calibration_summary.json
/kaggle/working/legalqa_selector_calibration/selector_report.json
/kaggle/working/legalqa_selector_calibration/tuned_selection.yaml
```

Đọc `tuning.holdout`; không chọn cấu hình chỉ vì tune hoặc oracle cao.

### D. QLoRA full và public inference

Chạy notebook QLoRA với `TRAIN_SCOPE='full'`. Không attach state split làm resume. Sau khi hoàn tất, attach duy nhất adapter full, dense index và `tuned_selection.yaml` vào notebook inference:

```python
DENSE_INDEX_ONLY = False
TRAIN_MEMORY = False  # Dùng False nếu Dataset dense hiện có chỉ global index
AUTO_RESUME = True
```

Notebook tự nhận adapter/weights. Nộp duy nhất `/kaggle/working/submission.zip`.
Mode này bắt buộc adapter `full`, `tuned_selection.yaml` và completed
global dense index. Train-memory chỉ bắt buộc khi `TRAIN_MEMORY=True`; giá trị
này phải giữ nguyên khi resume vì nó thuộc run fingerprint. Thiếu artifact
sẽ dừng trước khi tải model thay vì fallback. Giữ
`MAX_RUNTIME_HOURS=11.5`. Nếu cell cuối in `PAUSED SAFELY`, Save
Version, tạo Dataset từ `legalqa_recovery_latest.zip`, attach vào phiên kế tiếp
và giữ `AUTO_RESUME=True`. Chỉ nộp khi cell cuối in `READY`.

## OOM an toàn

Inference giảm lần lượt:

```python
EMBEDDING_BATCH_SIZE = 1
VERIFIER_BATCH_SIZE = 2
MAX_CONTEXT_CHARS = 12000
RERANKER_MAX_LENGTH = 384
```

QLoRA đã batch 1, 4-bit và gradient checkpointing. Không tăng batch; có thể tăng `GRAD_ACCUM` nếu muốn giữ effective batch. Giữ `CHECKPOINT_STEPS=20` hoặc giảm xuống 10 khi session không ổn định.

## Recovery

- Inference ghi atomic `/kaggle/working/legalqa_recovery_latest.zip` định kỳ.
- Inference tự dừng an toàn ở 11,5 giờ tại ranh giới câu hỏi và ghi manifest
  `status=paused`; deadline không làm đổi run fingerprint nên ZIP resume được.
- QLoRA dùng full Trainer checkpoint trong `/kaggle/working/legalqa_qlora/qwen-official-lora-{scope}/checkpoint-*`.
- `/kaggle/working` không tự tồn tại qua session; phải Save Version/tạo Dataset trước khi session mất.
- Recovery/adapters cũ khác fingerprint hoặc khác scope bị từ chối có chủ đích.

## Kiểm tra submission

Notebook xác nhận ZIP chỉ chứa `submission.json`, đủ số ID, mỗi giá trị là Object có `answer` dạng string và JSON UTF-8. Không nộp recovery ZIP hoặc adapter ZIP.
