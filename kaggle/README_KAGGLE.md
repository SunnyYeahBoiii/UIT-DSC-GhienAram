# Ghi chú notebook Kaggle v9.2

`legalqa_dsc2026_kaggle.ipynb` và `run_end_to_end.py` là runner P0/legacy được giữ lại để tái lập thí nghiệm cũ. Không dùng chúng cho bài v9.2 cuối.

Luồng v9.2 dùng ba notebook:

1. `legalqa_dsc2026_v9_qlora_t4.ipynb` — QLoRA split/full có checkpoint.
2. `legalqa_dsc2026_v9_selector_calibration_t4.ipynb` — grouped-dev, candidate oracle và selector tuning.
3. `legalqa_dsc2026_inference_only_t4.ipynb` — dựng dense/train-memory index hoặc sinh public submission.

Xem `README_INFERENCE_ONLY_T4.md` và `../docs/PIPELINE_V9.md` để biết thứ tự chạy và artifact phải lưu giữa các Kaggle session.
