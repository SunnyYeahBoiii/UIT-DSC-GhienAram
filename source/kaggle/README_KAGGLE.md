# Ghi chú notebook Kaggle v9.2

`legalqa_dsc2026_kaggle.ipynb` và `run_end_to_end.py` là runner P0/legacy được giữ lại để tái lập thí nghiệm cũ. Không dùng chúng cho bài v9.2 cuối.

Luồng v9.2 dùng ba notebook:

1. `legalqa_dsc2026_v9_qlora_t4.ipynb` — QLoRA split/full có checkpoint.
2. `legalqa_dsc2026_v9_selector_calibration_t4.ipynb` — grouped-dev, candidate oracle và selector tuning.
3. `legalqa_dsc2026_inference_only_t4.ipynb` — dựng dense/train-memory index hoặc sinh public submission.

Notebook QLoRA mặc định đặt `MAX_RUNTIME_HOURS=11.0`. Khi chạm ngân sách này,
Trainer buộc lưu model/optimizer/scheduler/RNG, tạo nguyên tử
`/kaggle/working/qlora_recovery_latest.zip`, ghi manifest `status=paused` và kết
thúc notebook thành công trước hard limit 12 giờ. Hãy **Save Version**, tạo một
Kaggle Dataset từ ZIP đó rồi attach Dataset vào session sau; runner sẽ tự nhận
đúng `train_scope` và resume từ optimizer step mới nhất. Khi manifest có
`status=complete`, dùng `legalqa_qlora/qwen_qlora_adapter.zip` cho bước sau.

Notebook public inference đặt `MAX_RUNTIME_HOURS=11.5`. Nó checkpoint candidate
cache/retrieval, ghi `run_manifest.json` với `status=paused`, tạo atomic
`legalqa_recovery_latest.zip` rồi thoát thành công để Kaggle kịp lưu output.
Phiên tiếp theo attach ZIP này và giữ `AUTO_RESUME=True`. Mode public cũng fail
fast nếu thiếu adapter `full`, tuned selector hoặc dense/train-memory index.

Xem `README_INFERENCE_ONLY_T4.md` và `../PIPELINE_V9.md` để biết thứ tự chạy và artifact phải lưu giữa các Kaggle session.
