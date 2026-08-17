# V8.1 zero-train

Script `scripts/build_v8_zero_train.py` kết hợp ba lớp không cập nhật trọng số mô hình:

1. tái chọn giữa hai candidate v8 bằng cổng an toàn nghiêm ngặt;
2. ghi đè bằng official train answer chỉ khi word TF-IDF, character TF-IDF, top-1 margin và explicit legal anchors đều đạt ngưỡng;
3. viết hoa chữ đầu, bỏ câu trùng chính xác và đóng phần đuôi Qwen bị truncation tại ranh giới câu.

Mặc định không cắt cứng theo số từ, không đổi số hiệu văn bản và luôn giữ answer v8 nếu không có tín hiệu đủ mạnh.

```powershell
python scripts/build_v8_zero_train.py `
  --baseline "C:\Users\LOQ\Downloads\submission.zip" `
  --trace "C:\Users\LOQ\Downloads\legalqa_recovery_latest (1)\predictions.trace.jsonl" `
  --train "C:\Users\LOQ\Downloads\dsc2026-task2-data\train.json" `
  --public "C:\Users\LOQ\Downloads\dsc2026-task2-data\public-official.json" `
  --output-dir "C:\Users\LOQ\Documents\Codex\2026-08-06\b\outputs\legalqa_v8_1_zero_train"
```

ZIP để nộp là `submission_v8_1_zero_train.zip`. `v8_1_zero_train_report.json` và
`v8_1_zero_train_audit.jsonl` giải thích từng thay đổi nhưng không nằm trong ZIP.
