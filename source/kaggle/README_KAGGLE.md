# Ghi chú notebook Kaggle v9.2

`legalqa_dsc2026_kaggle.ipynb` và `run_end_to_end.py` là runner P0/legacy được giữ lại để tái lập thí nghiệm cũ. Không dùng chúng cho bài v9.2 cuối.

Luồng v9.2 dùng ba notebook:

1. `legalqa_dsc2026_v9_qlora_t4.ipynb` — QLoRA split/full có checkpoint.
2. `legalqa_dsc2026_v9_selector_calibration_t4.ipynb` — grouped-dev, candidate oracle và selector tuning.
3. `legalqa_dsc2026_inference_only_t4.ipynb` — dựng dense/train-memory index hoặc sinh public submission.

## Fine-tune lại bằng dữ liệu sạch trên baseline 0.5007

- Gói data: `public-official.json`, `selected-contexts/`, **train_clean.json**
  (7.000 mẫu) và `training_quality.json`, cùng một thư mục.
- Runner chỉ đọc **train_clean.json**, không chọn `train.json` hoặc
  `train_clean_qlora.json` kể cả khi có sẵn. Thiếu file đúng tên sẽ báo lỗi rõ ràng.
- Runner áp dụng 47 ID loại trong `training_quality.json` và tự tạo
  `/kaggle/working/legalqa_qlora/prepared_train/qlora_train.json` (6.953 mẫu).
  File gốc không bị sửa. Split/dev được tạo từ tập này; `full` dùng toàn bộ mẫu
  hợp lệ. Không cần attach file `train_clean_qlora.json` riêng.
- `train.json` trong gói ZIP chỉ là alias tương thích inference cũ, không dùng
  cho runner QLoRA này.
- Lần đầu: session mới, `AUTO_RESUME_TRAINING=False`, `RESUME_INPUT=None`.
  Giữ model, hyperparameter và selector của baseline 0.5007 để so sánh tác động
  dữ liệu sạch; không cần đổi corpus hoặc dựng lại dense index nếu cấu hình
  chunking/embedding vẫn giữ nguyên.
- Resume lần train sạch: `AUTO_RESUME_TRAINING=True`, attach recovery mới.
  Runner kiểm tra scope + SHA-256 dữ liệu; checkpoint train cũ bị từ chối.
- Dùng notebook/source mới có marker `qlora_data_schema=train-clean-input-v2`;
  source cũ nhúng trong dataset checkpoint không được chọn cho notebook QLoRA.

Notebook QLoRA mặc định đặt `MAX_RUNTIME_HOURS=11.0`. Khi chạm ngân sách này,
Trainer buộc lưu model/optimizer/scheduler/RNG, tạo nguyên tử
`/kaggle/working/qlora_recovery_latest.zip`, ghi manifest `status=paused` và kết
thúc notebook thành công trước hard limit 12 giờ. Hãy **Save Version**, tạo một
Kaggle Dataset từ ZIP đó rồi attach Dataset vào session sau; runner sẽ tự nhận
đúng `train_scope` và hash dữ liệu, rồi resume từ optimizer step mới nhất
(bật `AUTO_RESUME_TRAINING=True`). Khi manifest có
`status=complete`, dùng `legalqa_qlora/qwen_qlora_adapter.zip` cho bước sau.

Notebook public inference đặt `MAX_RUNTIME_HOURS=11.5`. Nó checkpoint candidate
cache/retrieval, ghi `run_manifest.json` với `status=paused`, tạo atomic
`legalqa_recovery_latest.zip` rồi thoát thành công để Kaggle kịp lưu output.
Phiên tiếp theo attach ZIP này và giữ `AUTO_RESUME=True`. Mode public cũng fail
fast nếu thiếu adapter `full`, tuned selector hoặc dense/train-memory index.

Xem `README_INFERENCE_ONLY_T4.md` và `../PIPELINE_V9.md` để biết thứ tự chạy và artifact phải lưu giữa các Kaggle session.
