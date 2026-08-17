# AGENTS.md

Hướng dẫn cho agent AI làm việc trên repo này — UIT DSC 2026 Task 2 LegalQA v9.2 (Kaggle T4, offline pipeline).

## Ràng buộc cứng — không được phá

- Repo **không chứa** dữ liệu cuộc thi, checkpoint, dense index, adapter QLoRA, submission. `.gitignore` loại `/data/`, `/checksums/`, `artifacts/`, `models/`, `submission.zip`.
- `RELEASE_V9_2.md` phải nằm ngay ở root repo — 3 notebook Kaggle (`kaggle/legalqa_dsc2026_v9_qlora_t4.ipynb`, `kaggle/legalqa_dsc2026_v9_selector_calibration_t4.ipynb`, `kaggle/legalqa_dsc2026_inference_only_t4.ipynb`) check tồn tại file này theo đường dẫn cứng khi copy source lên Kaggle.
- `pyproject.toml`: `package-dir = {"" = "src"}`, entry point `legalqa = "legalqa.cli:main"`. `cli.py`, `pipeline.py`, `__main__.py` ở root package `src/legalqa/` — không di chuyển vào subpackage.
- `wheelhouse/` (12 wheel) phải build trên Linux x86_64 Python 3.12 khớp runtime Kaggle — không tự ý đổi version pin trong `scripts/build_wheelhouse.sh` mà không cập nhật `wheelhouse/README.md`.
- `SOURCE_CONTRACT.json` được 3 notebook v9.2 dùng để verify source khớp release — sửa cấu trúc/API công khai thì phải cập nhật cùng.
- Không agent loop vô hạn: graph retrieval và Agentic tool-call tối đa 2 vòng mặc định, có visited/query-set.

## Cấu trúc

```
src/legalqa/
├── cli.py, pipeline.py, __main__.py, __init__.py   # root, không dời
├── core/        config, data, text, models, budget, fingerprint, recovery, selector_schema, metrics, submission
├── retrieval/   bm25, chunking, index, retrieval, routing, legal_graph
├── generation/  answer_style, generation, prompts, verifier
├── storage/     dense_store, sqlite_store, sqlite_retrieval, train_memory
└── training/    training_data, zero_train
scripts/   build index, train, calibration, validation
kaggle/    notebook + runner Kaggle T4
tests/     pytest
configs/, server/, docs/
```

Import nội bộ dùng relative import (`from .sibling import`, `from ..core.module import`). Subpackage `__init__.py` để rỗng, không re-export.

## Lệnh hay dùng

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts kaggle
PYTHONPATH=src python -c "from legalqa.cli import main"
```

## Quy ước code

- Comment/docstring/commit message: tiếng Việt, ngắn, giải thích WHY không phải WHAT.
- Không thêm abstraction/error-handling cho case không xảy ra. Không refactor ngoài phạm vi task.
- Trước khi sửa xong: chạy pytest + compileall, grep quét sót đường dẫn/import cũ nếu di chuyển file.
- `git mv` khi rename/move file để giữ history.

## Việc đang mở

Xem `TODO.md` — Lớp C (`finalize_answer()` sanitizer) chưa implement, vị trí chèn tại `src/legalqa/pipeline.py:555-556`.
