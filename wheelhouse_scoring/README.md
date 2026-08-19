# Offline scoring dependencies cho `legalqa_dsc2026_v9_sft_lora_t4.ipynb`

Tách riêng khỏi `wheelhouse/` (dùng chung cho 3 notebook v9.2 khác đã bị `SOURCE_CONTRACT.json`
khoá cứng ở đúng 12 wheel) để không phá invariant của các notebook đó.

Nội dung:
- `nltk-3.9.1-py3-none-any.whl`, `rouge_score-0.1.2-py3-none-any.whl`, `joblib-1.4.2-py3-none-any.whl`,
  `absl_py-2.1.0-py3-none-any.whl` — pure-Python (`py3-none-any`), không phụ thuộc nền Linux/macOS nào
  cả, dùng để cài offline `pip install --no-index --no-deps *.whl`.
- `nltk_data/corpora/{wordnet,omw-1.4}/` — corpus WordNet mà `nltk.translate.meteor_score.meteor_score`
  cần để tính METEOR đúng công thức BTC (`Scoring-Program-Task-LegalQA/scoring.py`). Trỏ biến môi trường
  `NLTK_DATA` vào thư mục `nltk_data` này. **Phải giữ dạng đã giải nén**, không chỉ `.zip` — từng gặp
  lỗi trên Kaggle (`FileSystemPathPointer` trỏ vào thư mục rỗng) vì cơ chế tự-giải-nén-lúc-chạy của
  nltk không đáng tin cậy trên mọi filesystem; giải nén sẵn từ lúc build tránh hẳn vấn đề này.

Dùng bởi `scripts/select_best_sft_checkpoint.py`, `scripts/validate_official_score.py`, và
`kaggle/run_qwen_sft_lora.py`. Build lại bằng `scripts/build_scoring_wheelhouse.sh` — vì toàn bộ wheel
đều pure-Python nên chạy trên macOS/Linux đều cho kết quả giống nhau (khác với `wheelhouse/` chính vốn
có wheel biên dịch riêng cho Linux x86_64).

## Phụ thuộc `regex` — cố ý không đóng gói lại

`nltk` import `regex` ngay khi load (`nltk/tokenize/casual.py`). Package này KHÔNG được đóng gói
trong `wheelhouse_scoring/` — nó đã có sẵn trong `wheelhouse/` chính (`regex==2026.7.19`, pin bởi
`transformers`/`tokenizers`). Notebook cài `wheelhouse/` trước rồi mới cài `wheelhouse_scoring/`
(xem `legalqa_dsc2026_v9_sft_lora_t4.ipynb` cell-2) nên `regex` luôn sẵn sàng trước khi `nltk` được
import — không cần trùng lặp một wheel biên dịch riêng cho Linux x86_64 ở đây và có nguy cơ lệch
version với bản đã pin. Hệ quả: `wheelhouse_scoring/` KHÔNG dùng độc lập được (vd. venv trống chỉ
cài mỗi thư mục này) — luôn phải cài kèm sau `wheelhouse/` chính, kể cả khi chạy
`scripts/validate_official_score.py` cục bộ (dùng `pip install -e ".[train]"` như bình thường thay
vì các wheel offline này nếu không phải trên Kaggle).
