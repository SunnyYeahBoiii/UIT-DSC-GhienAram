# Kết quả smoke test warm-up

Ngày chạy: 06/08/2026  
Nguồn: `F:/warmup.json`

## Split chống leakage

- Tổng: 500 câu
- Train: 401 câu
- Dev: 99 câu
- Dev ratio thực tế: 19,8%
- Pseudo-corpus: chỉ 401 đáp án train, tạo thành 420 legal chunks
- Dense model, reranker và generator: tắt trong smoke test

## Baseline BM25 + extractive

| Chỉ số local | Điểm |
|---|---:|
| METEOR exact-token probe | 0,237819 |
| ROUGE-L F1 | 0,178718 |
| Số từ đáp án dự đoán trung bình | 246,36 |
| Số từ tham chiếu trung bình | 338,97 |

Đây là kiểm tra chức năng và mốc ablation, không phải ước lượng đáng tin cậy cho leaderboard: pseudo-corpus là các đáp án train thay vì kho 8.500 văn bản. METEOR probe cũng không nhất thiết trùng implementation của BTC.

## Artifact đã kiểm chứng

- `artifacts/smoke/dev_predictions.json`
- `artifacts/smoke/dev_predictions.trace.jsonl`
- `artifacts/smoke/dev_metrics.json`
- `artifacts/smoke/submission.zip`

`submission.zip` đã qua validator: chứa duy nhất `submission.json`, đủ 99 ID dev, UTF-8 không BOM và mọi `answer` là chuỗi.

