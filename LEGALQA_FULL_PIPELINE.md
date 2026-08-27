# UIT DSC 2026 — Pipeline đầy đủ LegalQA

## 1. Input

- `train.json`: câu hỏi và đáp án train chính thức.
- `public-official.json`: câu hỏi public test.
- `selected-contexts/`: kho văn bản pháp luật.
- Các model P0 đã được BTC phê duyệt.
- Source code, QLoRA adapter và recovery checkpoint nếu có.

### Các model sử dụng

- `Qwen/Qwen3.5-2B`: sinh câu trả lời, được fine-tune QLoRA bằng train chính thức.
- `AITeamVN/Vietnamese_Embedding`: dense retrieval toàn cục.
- `AITeamVN/Vietnamese_Reranker`: rerank các passage truy hồi.
- `ntphuc149/ViLegalBERT`: kiểm chứng độ phù hợp pháp lý của candidate/evidence.
- BARTpho đã được loại khỏi pipeline runtime.

## 2. Kiểm tra và làm sạch dữ liệu

- Kiểm tra JSON, UTF-8, `question_id` và dữ liệu trùng/thiếu.
- Loại đuôi web, câu hỏi bị lặp và `(Hình/Ảnh từ Internet)` khỏi answer train.
- Khôi phục tên và số hiệu văn bản từ `name`, URL hoặc phần đầu văn bản.
- Tính corpus fingerprint để ngăn dùng nhầm index/recovery.

## 3. Xây knowledge base

- Tách văn bản theo Điều, Khoản, Điểm, quy trình và các mục La Mã.
- Tạo SQLite cùng BM25/FTS toàn cục.
- Xây legal graph cha–con, thay thế, sửa đổi và dẫn chiếu đã được kiểm chứng.
- Tạo global dense index và lưu manifest tương thích với corpus/chunking.

## 4. Chuẩn bị train

- Ghép mỗi câu train với evidence phù hợp trong corpus.
- Tạo retrieval pairs và grouped train/dev.
- Xây train-question kNN memory từ dữ liệu chính thức.
- Thống kê độ dài và văn phong answer sau khi làm sạch.

## 5. Train QLoRA

- Qwen base + câu hỏi + retrieved evidence + answer train chính thức.
- Train QLoRA 4-bit, mặc định 1 epoch.
- Lưu checkpoint, optimizer và scheduler định kỳ.
- Tự phát hiện checkpoint mới nhất và resume đến khi hoàn chỉnh.

## 6. Selector calibration

Sinh các candidate trên grouped-dev:

- Extractive.
- Qwen thường.
- Qwen Agentic.
- Train-memory candidate.
- ViLegalBERT dùng để kiểm chứng/đánh giá độ phù hợp.

Dùng candidate oracle để tạo nhãn chọn candidate, sau đó hiệu chỉnh selector theo METEOR; ROUGE-L là chỉ số phụ. Các feature chính gồm citation support, numeric faithfulness, độ dài, truncation, mức bao phủ Điều/Khoản và evidence document coverage.

## 7. Public inference

```text
Question
  → Global BM25 + Dense retrieval + train kNN
  → RRF fusion
  → Reranker
  → Simple/Complex router
      ├─ Simple: khóa document chính + merge chunk lân cận
      └─ Complex: graph/agentic multi-hop có relevance gate
  → Sinh Qwen thường, Qwen Agentic và Extractive candidate
  → Kiểm chứng citation, con số, evidence và truncation
  → Calibrated selector
  → Post-processing tiếng Việt
  → Final answer
```

Các model được nạp tuần tự rồi giải phóng VRAM:

```text
Embedding → unload
Reranker → unload
Qwen + QLoRA → unload
ViLegalBERT → unload
```

## 8. Recovery và checkpoint inference

- Lưu retrieval cache và candidate cache.
- Lưu `predictions.trace.jsonl` và `recovery_state.json`.
- Đóng gói `legalqa_recovery_latest.zip` định kỳ.
- Tạm dừng an toàn trước giới hạn khoảng 11,5 giờ và resume ở session sau.
- Chỉ resume khi source, config, corpus, dense index, adapter và selector có fingerprint phù hợp.

## 9. Tạo submission

- Tạo `submission.json` với đúng một answer dạng string cho mỗi `question_id`.
- Kiểm tra không thiếu/thừa ID và toàn bộ nội dung dùng UTF-8.
- Tạo `submission.zip` chỉ chứa duy nhất `submission.json`.
- So sánh kết quả mới với mốc submission tốt nhất trước khi nộp.

## Hai nhánh chạy song song

### Nhánh nhanh v9.3

```text
Selector calibration mới
→ Public inference mới
→ Validate submission
→ Nộp như một bản A/B
```

Tạm tắt relation expansion không an toàn của Agentic; không reuse retrieval/recovery v9.2.

### Nhánh sạch v9.4

```text
Clean train
→ Sửa metadata, graph và chunking
→ Rebuild SQLite + dense index
→ Train lại QLoRA nếu cần
→ Selector calibration
→ Public inference cuối cùng
```

Sau khi thay đổi chunking hoặc graph, không được dùng lại SQLite, dense index, retrieval cache hay recovery của v9.3.
