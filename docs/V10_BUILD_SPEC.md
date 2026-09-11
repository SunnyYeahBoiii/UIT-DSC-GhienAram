# V10 Build Spec — đặc tả để agent implement

Đọc trước khi code: `CONTEXT.md` (glossary — dùng đúng tên khái niệm), `docs/EDA_V10.md`
(mọi con số biện minh cho thiết kế này), `docs/adr/0001`, `docs/adr/0002`.

Branch: `feat/v10-rebuild`. Code cũ đã xoá; `main` giữ nguyên v9.2 để đối chiếu.
Python 3.12. Import nội bộ dùng relative import. Comment/docstring tiếng Việt, giải thích WHY.

## Đã có

- `src/legalqa/corpus.py` — `Document`, `Article`, `parse_articles()`, `title_from_slug()`,
  `iter_documents()`. **Không sửa API này**, chỉ dùng.
- `scripts/eda/` — 7 script EDA tái lập được, đừng đụng.

## Cần implement, theo thứ tự

### 1. `src/legalqa/chunking.py`

```python
@dataclass(slots=True)
class Chunk:
    chunk_id: str      # f"{doc_id}:{ordinal}"
    doc_id: str
    text: str
    start: int         # offset ký tự trong Passage
    end: int

def chunk_document(doc: Document, size: int = 256, stride: int = 256) -> list[Chunk]: ...
```

- Cắt theo **từ** (whitespace), `size`/`stride` tính bằng từ. Mặc định 256/256 — ADR-0001.
- Bỏ chunk cuối nếu < 30 từ **và** không phải chunk đầu của Document.
- `start`/`end` phải là offset ký tự thật trong `doc.passage` (cần cho bước mở rộng Evidence).

Acceptance: trên toàn corpus sinh ra **~287.000 Chunk** (EDA đo 287.741 với 256/256).

### 2. `src/legalqa/evidence.py`

```python
@dataclass(slots=True)
class Evidence:
    doc_id: str
    doc_title: str
    article_number: str | None
    text: str

def expand_to_evidence(doc: Document, chunk: Chunk, *, max_words: int = 900) -> Evidence: ...
```

Đây là small-to-big của ADR-0001: Chunk dùng để **tìm**, Evidence dùng để **sinh**.

- Nếu Chunk nằm trong một `Article` (so offset): lấy trọn text của Article đó.
- Nếu Article đó dài hơn `max_words`: cắt quanh Chunk, giữ nguyên `heading` của Article ở đầu.
- Nếu Document không có Article nào (15,9% corpus): lấy cửa sổ ±1 Chunk quanh Chunk trúng.
- `doc_title` lấy từ `doc.title`. **Không bao giờ** đưa Slug thô vào Evidence — xem CONTEXT.md.

### 3. `src/legalqa/bm25.py`

BM25 Okapi k1=1.5, b=0.75 trên Chunk. Tokenize: `lowercase` + `re.findall(r"\w+", ...)`.
Sparse matrix scipy, `min_df=2`. Tham chiếu cài đặt đúng: `scripts/eda/bm25_recall_chunk.py`.

```python
class ChunkIndex:
    def build(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]: ...  # (chunk_id, score)
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "ChunkIndex": ...
```

**Không** thêm citation boost vào query — EDA đo chỉ 0,1% question chứa citation, đó là code chết.

### 4. `src/legalqa/clean.py`

Làm sạch Reference answer trước khi SFT. EDA đo tỷ lệ từng loại noise trên 7.000 câu:

```python
def clean_reference_answer(answer: str) -> str: ...
def is_usable_for_sft(question: str, answer: str) -> bool: ...
```

Phải xoá:
- `(Hình từ Internet)`, `(Nguồn ảnh: Internet)`, `(Ảnh minh họa)` — **18,5%** reference answer có
- `Tải ... tại đây`, `(tại đây)`, câu `Tại đây.` đứng riêng — 0,9%
- câu hỏi bị lặp nguyên văn trong answer

Phải **giữ**: `[...]` — là trích luật lược bỏ hợp lệ (1,0%), không phải noise.

`is_usable_for_sft` trả `False` khi: answer kết thúc bằng `...` (cắt cụt, 0,3%), answer < 200 ký tự,
hoặc question trùng một question đã giữ (giữ bản answer dài nhất).

Acceptance: chạy trên `data/official/train.json` (7.000 câu) giữ lại **6.900–6.950** câu.

### 5. `tests/` — pytest, chạy bằng `PYTHONPATH=src python -m pytest -q`

Mỗi module một file test. Bắt buộc có:
- `test_chunking.py`: offset `start`/`end` slice đúng ra `chunk.text`; chunk cuối < 30 từ bị bỏ.
- `test_evidence.py`: Chunk giữa một Điều → Evidence chứa trọn Điều đó; Document không có Điều →
  đường lui cửa sổ chạy được, không raise.
- `test_corpus.py`: `title_from_slug("Thong-tu-17-2022-TT-BGTVT-abc-522401")` không còn dấu `-`
  và không còn đuôi số id; Document thiếu `name` vẫn có `title` không rỗng.
- `test_clean.py`: từng loại noise ở mục 4, và `[...]` được giữ nguyên.

## Ràng buộc

- **Không** implement: agentic tool-call, multi-candidate selector, train-memory. ADR-0002 đã loại.
- **Không** sửa `docs/`, `CONTEXT.md`, `scripts/eda/`, `data/`, `wheelhouse/`.
- **Không** thêm dependency ngoài: `numpy`, `scipy`, `scikit-learn`, `pytest`. Không torch ở 5 module này.
- Chạy được `python -m compileall -q src tests` và `PYTHONPATH=src python -m pytest -q` trước khi báo xong.
- Commit từng module một, message tiếng Việt.

## Định nghĩa hoàn thành

`PYTHONPATH=src python -m pytest -q` xanh, và script sau chạy được end-to-end trên corpus thật:

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from legalqa.corpus import iter_documents
from legalqa.chunking import chunk_document
from legalqa.evidence import expand_to_evidence
docs = []
for i, d in enumerate(iter_documents(Path('data/official/selected-contexts'))):
    docs.append(d)
    if i >= 200: break
cs = [c for d in docs for c in chunk_document(d)]
print('chunks', len(cs))
e = expand_to_evidence(docs[0], [c for c in cs if c.doc_id == docs[0].doc_id][0])
print(e.doc_title[:80]); print(e.article_number); print(len(e.text.split()), 'tu')
"
```
