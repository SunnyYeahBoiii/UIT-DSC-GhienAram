# AGENTS.md

Hướng dẫn cho agent AI làm việc trên repo này — UIT DSC 2026 Task 2 LegalQA **v10**
(Kaggle T4, offline). Nhánh `main` giữ nguyên v9.2 để đối chiếu; v10 xây lại từ số 0.

## Đọc trước khi code

1. `CONTEXT.md` — glossary. Dùng đúng tên khái niệm, không tự đặt tên mới.
2. `docs/EDA_V10.md` — mọi con số biện minh cho thiết kế. **Không đề xuất gì mà EDA chưa đo.**
3. `docs/adr/0001` — Chunk là đơn vị index, Evidence là đơn vị prompt.
4. `docs/adr/0002` — bỏ agentic / selector / train-memory, giữ legal graph.
5. `docs/V10_BUILD_SPEC.md` — đặc tả 5 module lõi.

## Ràng buộc cứng — không được phá

- Repo **không chứa** dữ liệu cuộc thi, checkpoint, dense index, adapter QLoRA,
  submission. `.gitignore` loại `/data/`, `/checksums/`, `artifacts/`, `models/`,
  `submission.zip`.
- `pyproject.toml`: `package-dir = {"" = "src"}`.
- **Không** implement agentic tool-call, multi-candidate selector, train-memory —
  ADR-0002 đã loại, có số đo.
- Không agent loop vô hạn: legal graph tối đa 2 vòng, có visited-set.
- `wheelhouse/` đã xoá khỏi v10. Khi cần cài offline trên Kaggle phải build lại trên
  **Linux x86_64 / Python 3.12** (máy dev macOS ARM không build được), hoặc lấy lại bản
  cũ bằng `git checkout main -- wheelhouse`.

## Ràng buộc về dữ liệu

`Document.passage` là text crawl từ HTML, chứa wrap mềm `\r\n\n`. `load_document()` gọi
`dewrap_passage()` trước `parse_articles()` — **mọi offset** `Article`/`Chunk`/`Evidence`
đều trỏ vào Passage đã gỡ wrap. Đừng đọc `passage` thô từ JSON ở chỗ khác.

Chunk giữ **nguyên văn theo offset**, không `" ".join(words)`: ngắt dòng `\n\n` mang cấu
trúc phân khoản mà model cần để trích đúng khoản.

## Cấu trúc

```
src/legalqa/
├── corpus.py      Document, Article, dewrap_passage, parse_articles, iter_documents
├── chunking.py    Chunk, chunk_document          (đơn vị tìm kiếm)
├── evidence.py    Evidence, expand_to_evidence   (đơn vị sinh, small-to-big)
├── bm25.py        ChunkIndex — BM25 Okapi k1=1.5 b=0.75
└── clean.py       clean_reference_answer, is_usable_for_sft
scripts/eda/       7 script EDA tái lập được — đừng đụng
tests/             pytest
docs/, CONTEXT.md
```

Import nội bộ dùng relative import (`from .corpus import Document`).

## Lệnh hay dùng

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests
```

## Quy ước code

- Comment/docstring/commit message: tiếng Việt, ngắn, giải thích **WHY** không phải WHAT.
  Khi có số đo từ EDA thì dẫn số đó.
- Không thêm abstraction/error-handling cho case không xảy ra. Không refactor ngoài phạm vi.
- Test phải bắt được lỗi thật: fixture `" ".join(f"w{i}" ...)` toàn space đơn từng che
  giấu bug offset sai 287.716/287.741 chunk. Dùng fixture có `\n\n` và space thừa.
- `git mv` khi rename/move file để giữ history.

## Việc đang mở

5 module lõi xong, test xanh. Chưa có: wiring retrieval end-to-end, prompt + Qwen SFT,
legal graph hop theo Citation (dùng `Document.number`), sanitizer câu trả lời cuối.
