# UIT DSC 2026 Task 2 — LegalQA v10

Pipeline offline cho Kaggle T4: hỏi đáp pháp luật tiếng Việt dạng dài. Metric chính
METEOR, phụ ROUGE-L. Tổng tham số runtime < 4B.

Repo **không chứa** dữ liệu cuộc thi, checkpoint, dense index, adapter QLoRA hay
submission — xem `.gitignore`.

v9.2 nằm ở nhánh `main`. v10 xây lại từ số 0 sau khi EDA chỉ ra lỗi nền móng.

## Vì sao xây lại

v9.2 index **cả Document**. Document p50 = 4.828 từ (max 1.242.409), nên chuẩn hoá độ
dài của BM25 pha loãng câu hỏi 19 từ. Đo recall trên 224 câu weak-gold:

| Đơn vị index | R@1 | R@5 | R@10 | R@100 |
|---|---:|---:|---:|---:|
| Document (v9.2) | 22,3% | 42,0% | 50,9% | 76,3% |
| Chunk 256/256 (v10) | **55,4%** | **83,0%** | **89,7%** | **97,8%** |

Bốn tầng phía sau của v9.2 (graph, agentic, memory, selector) đều ngồi sau trần cứng
50,9% đó. Sửa đơn vị index đáng giá hơn mọi tầng cộng lại.

Chi tiết và các số đo khác: `docs/EDA_V10.md`.

## Thiết kế

**Small-to-big.** Chunk 256 từ là đơn vị **tìm**; Evidence — mở rộng ra trọn các Điều mà
Chunk chạm vào — là đơn vị **sinh**. Lý do ở `docs/adr/0001`.

**Giữ legal graph, bỏ phần còn lại.** Agentic tool-call, multi-candidate selector,
train-memory đều bị loại; graph chỉ mở rộng khi Citation gọi tên một văn bản cụ thể,
không gate theo số quan hệ (93,1% Document chứa "căn cứ" nên đếm quan hệ là vô nghĩa).
Lý do ở `docs/adr/0002`.

**Làm sạch dữ liệu trước.** Passage crawl từ HTML mang wrap mềm `\r\n\n` — 100% Document
có, 93,9% nối giữa câu. Không gỡ thì 62,3% tiêu đề Điều bị cắt đôi. Gỡ xong còn 0,2%.

## Chạy

```bash
PYTHONPATH=src python -m pytest -q
```

```python
from pathlib import Path
from legalqa.corpus import iter_documents
from legalqa.chunking import chunk_document
from legalqa.evidence import expand_to_evidence

docs = list(iter_documents(Path("data/official/selected-contexts")))
chunks = [c for d in docs for c in chunk_document(d)]      # 287.741 chunk
ev = expand_to_evidence(docs[0], chunks[0])                 # Evidence trọn Điều
```

EDA tái lập được bằng `scripts/eda/` (7 script, đọc `data/official/`).

## Tài liệu

| File | Nội dung |
|---|---|
| `CONTEXT.md` | Glossary — thuật ngữ miền, không có chi tiết cài đặt |
| `docs/EDA_V10.md` | Toàn bộ số đo |
| `docs/adr/` | Quyết định kiến trúc |
| `docs/V10_BUILD_SPEC.md` | Đặc tả 5 module lõi |
| `AGENTS.md` | Hướng dẫn cho agent AI |
