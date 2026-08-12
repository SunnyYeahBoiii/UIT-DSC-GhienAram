# UIT DSC 2026 Task 2 — LegalQA v2

Source này chạy inference tuần tự trên Kaggle, không fine-tune và chỉ dùng dữ
liệu chính thức. Runtime gồm bốn checkpoint đã được BTC phê duyệt: Qwen3.5-2B,
Vietnamese Embedding, Vietnamese Reranker và ViLegalBERT. BARTpho đã bị loại
khỏi runtime và không được tính vào ngân sách tham số.

## Pipeline hiện tại

```text
Question
  -> global chunk BM25 + exact global dense retrieval
  -> typed citation boosts + RRF + cross-encoder rerank
  -> retrieval-confidence Simple/Complex router
     -> Simple: khóa văn bản chính, lấy Điều/Khoản/chunk lân cận
     -> Complex: legal graph REPLACES/REPEALS/AMENDS/GUIDES/REFERENCES,
                 tối đa 3 vòng mặc định, temporal ranking
  -> merge và deduplicate evidence
  -> evidence-first Qwen + extractive fallback
  -> ViLegalBERT/fluency/confidence-aware selector
  -> submission.zip
```

Chi tiết kiến trúc, trace và hiệu chỉnh selector nằm trong
[`PIPELINE_V2.md`](PIPELINE_V2.md). Hướng dẫn Kaggle nằm trong
[`kaggle/README_INFERENCE_ONLY_T4.md`](kaggle/README_INFERENCE_ONLY_T4.md).

## Chạy Kaggle khuyến nghị

1. Attach Dataset dữ liệu, bốn model và source này; bật GPU, tắt Internet.
2. Import `kaggle/legalqa_dsc2026_inference_only_t4.ipynb`.
3. Session 1 đặt `DENSE_INDEX_ONLY=True`; Save Version và tạo Kaggle Dataset
   từ `/kaggle/working/legalqa_dense_index`.
4. Attach dense-index Dataset, đặt `DENSE_INDEX_ONLY=False` và chạy lại.
5. Nộp duy nhất `/kaggle/working/submission.zip`.

Qwen candidate và retrieval cache được ghi vào
`/kaggle/working/legalqa_recovery_latest.zip`. Muốn chạy tiếp ở session mới,
hãy lưu ZIP này thành Kaggle Dataset rồi attach; notebook tự phát hiện đúng một
recovery bundle và runner kiểm tra fingerprint trước khi dùng.

## Kiểm thử

```bash
python -m pytest -q
python -m compileall -q src scripts kaggle
```

## Hiệu chỉnh selector bằng train chính thức

```bash
python scripts/prepare_selector_dev.py \
  --train /path/to/train.json \
  --output /path/to/selector_split \
  --dev-ratio 0.15
```

Sau khi chạy inference trên `selector_dev_questions.json`, dùng
`scripts/evaluate_selector.py` để báo cáo selected score, từng source,
candidate oracle và tìm trọng số selector. Holdout được tách theo nhóm mẫu câu,
không để câu gần trùng xuất hiện ở cả tune và holdout.

## Đầu ra quan trọng

```text
/kaggle/working/submission.zip
/kaggle/working/legalqa_recovery_latest.zip
/kaggle/working/legalqa_inference/run_manifest.json
/kaggle/working/legalqa_inference/predictions.json
/kaggle/working/legalqa_inference/predictions.trace.jsonl
```

Không dùng agent loop vô hạn. Legal graph có visited-set, dừng khi không còn văn
bản mới và bị chặn cứng ở tối đa bốn vòng.
