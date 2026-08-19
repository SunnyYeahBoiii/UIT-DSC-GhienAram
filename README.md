# UIT DSC 2026 LegalQA v9.2 — gated agentic retrieval

> Baseline hiện hành là public METEOR **0.5007**. Inference mặc định dùng
> `source/configs/selector_v5007.yaml`; xem `source/BASELINE_V5007.md`.
> Thử nghiệm selector v9.3 gây regression 0.4272 đã được tách sang nhánh
> `codex/v9.3-regression-snapshot` và không còn thuộc runtime mặc định.

Repository này chứa **một** cây source hoàn chỉnh tại `source/`. Đây là bản
v9.2 kế thừa v9.1 graph-gated và bổ sung Agentic Search đã sửa lỗi.

## Nội dung thực tế

- `source/src/`: pipeline retrieval, generation, verifier và selector.
- `source/scripts/`: build index, train, calibration, validation.
- `source/kaggle/`: notebook và runner Kaggle T4.
- `source/wheelhouse/`: 12 wheel cố định để cài hoàn toàn offline trên Kaggle.
- `source/tests/`: kiểm thử hồi quy.

Repository **không chứa** dữ liệu cuộc thi, model Hugging Face, dense index,
checkpoint/adapter QLoRA, candidate cache, recovery ZIP hay submission. Hãy
attach các artifact này thành Kaggle Dataset riêng và tuân thủ điều khoản của
Ban Tổ chức.

## Agentic Search v9.2

- Chỉ chạy cho câu Complex, retrieval confidence thấp hoặc có tín hiệu
  dẫn chiếu/thay thế/hiệu lực; không chạy cho toàn bộ câu hỏi.
- Tối đa hai tool call, chặn truy vấn lặp và buộc sinh đáp án cuối.
- Mỗi tool call dùng BM25 toàn cục rồi legal-rerank bằng số văn bản,
  Điều/Khoản, quan hệ graph và trạng thái văn bản hiện hành.
- Tool evidence chỉ thuộc candidate Agentic. Candidate Qwen thường,
  extractive và train-memory không được hưởng điểm verifier từ evidence đó.
- Lỗi tool-call không nghiêm trọng chỉ loại candidate Agentic của câu đó;
  không cache lỗi và không tắt Agentic cho các câu sau. Sau ba lỗi liên
  tiếp pipeline chỉ in cảnh báo. Lỗi GPU nghiêm trọng vẫn dừng an toàn để resume.
- Selector calibration có prior riêng cho `qwen3.5-2b-agentic`.

## Chạy kiểm thử

```powershell
cd F:\UIT-DSC-GHIENARAM\source
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
python -m compileall -q src scripts kaggle
```

Quy trình Kaggle nằm tại `source/kaggle/README_INFERENCE_ONLY_T4.md`. Sau khi
QLoRA split đã hoàn tất, hãy chạy selector calibration bằng chính source v9.2
này; không dùng lại trace calibration sinh từ Agentic phiên bản cũ.
