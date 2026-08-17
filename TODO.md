# TODO

## Lớp C — `finalize_answer()` sanitizer cuối trước khi ghi submission

Trạng thái: chưa implement. Lớp A (root fix corpus/chunking) và Lớp B (ràng buộc phủ định
trong `SYSTEM_PROMPT` chống caption ảnh hallucination) đã xong. Lớp C là lưới an toàn cuối
cho phần noise Lớp A/B không chặn hết (model vẫn có xác suất không tuân prompt).

### Vị trí implement

- Hàm mới `finalize_answer(question: str, answer: str, *, truncated: bool = False) -> tuple[str, dict]`
  trong `source/src/legalqa/generation.py`, đặt cạnh `clean_answer()`/`compact_generated_answer()`.
- Gọi tại chokepoint duy nhất mọi answer đi qua trước khi vào `predictions.json`/`submission.json`:
  `source/src/legalqa/pipeline.py:555-556`
  ```python
  winner = ranked[0]
  predictions[item.question_id] = {"answer": winner.answer}
  ```
  Sửa thành gọi `finalize_answer(item.question, winner.answer, truncated=bool(winner.details.get("truncated", False)))`,
  gán ngược `winner.answer` (để trace + predictions + submission đồng nhất), lưu actions vào
  `winner.details["_finalize"]`.

### Noise cần chặn (đã xác nhận qua audit `submission.json` cũ)

1. Marker `[CĂN CỨ N ...]` sót lại nếu model không tuân prompt (`prompts.py` dặn không chèn
   nhưng không có gì chặn cứng).
2. Chú thích ảnh sót lại kiểu `(Hình từ Internet)`, `(Ảnh minh họa)` — Lớp B giảm tần suất
   nhưng không đảm bảo 0%.
3. Dòng/chuỗi phân cách rác kiểu `-------`, `======` sót từ evidence chưa dewrap hết.
4. Câu hỏi bị lặp lại nguyên văn giữa answer (case thật: ID `8581` trong submission cũ —
   "...là 50.000.000 đồng. Mức phạt tiền tối đa đối với hành vi vi phạm hành chính về
   Trồng trọt là bao nhiêu? (Hình từ Internet)."). Strip bằng cách xoá occurrence của
   `question.strip()` (có/không dấu `?` cuối) xuất hiện y hệt trong answer.
5. Câu trùng lặp / câu bị cắt cụt giữa chừng (`truncated`) — có sẵn primitive tái dùng, xem
   dưới.

### Tái dùng, không viết lại

Trước khi viết `finalize_answer`, di chuyển 3 hàm generic từ `source/src/legalqa/zero_train.py`
sang `source/src/legalqa/text.py` (thuần regex/string, không phụ thuộc sklearn — tránh kéo
`sklearn`/`TfidfVectorizer` vào hot path của `generation.py`, hiện `zero_train.py` không được
import bởi pipeline chính):

- `remove_exact_sentence_duplicates`
- `capitalize_first_letter`
- `close_truncated_answer`
- helper `_sentence_fingerprint` + `SENTENCE_SPLIT_RE`/`FINGERPRINT_RE`

`zero_train.py` sau đó import lại từ `.text` để giữ nguyên API cho
`tests/test_zero_train.py` và `scripts/build_v8_zero_train.py`.

### Verification khi làm

- `PYTHONPATH=src python -m pytest -q` (tests/test_core.py) — thêm test case cho từng loại
  noise ở trên + test tổng hợp dùng đúng answer thật đã ghi nhận rác làm fixture.
- Xác nhận `remove_exact_sentence_duplicates`/`capitalize_first_letter`/`close_truncated_answer`
  di chuyển không phá `tests/test_zero_train.py`.
- (Tuỳ chọn) wire thêm vào `scripts/build_v8_zero_train.py` nếu còn dùng đường offline này.
