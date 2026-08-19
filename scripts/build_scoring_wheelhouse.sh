#!/usr/bin/env bash
# Tái tạo wheelhouse_scoring/ (bị .gitignore loại khỏi diff review lớn nhưng vẫn track trong git
# như wheelhouse/ chính — xem wheelhouse_scoring/README.md). Toàn bộ package ở đây là pure-Python
# (py3-none-any) nên khác với build_wheelhouse.sh, script này chạy được trên macOS/Linux đều ra
# cùng kết quả, không bắt buộc Linux x86_64.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scoring_dir="$script_dir/../wheelhouse_scoring"
mkdir -p "$scoring_dir" "$scoring_dir/nltk_data"

# Version pin khớp với wheelhouse_scoring/README.md — đổi version thì cũng phải cập nhật README.
python3 -m pip download --no-deps --dest "$scoring_dir" \
  "nltk==3.9.1" \
  "joblib==1.4.2" \
  "absl-py==2.1.0"

# rouge-score chỉ phát hành sdist trên PyPI, không có wheel sẵn — build wheel pure-Python tại chỗ.
python3 -m pip download --no-deps --dest "$scoring_dir" "rouge-score==0.1.2"
python3 -m pip wheel --no-deps --wheel-dir "$scoring_dir" "$scoring_dir"/rouge_score-0.1.2.tar.gz
rm -f "$scoring_dir"/rouge_score-0.1.2.tar.gz

# Corpus WordNet cho nltk.translate.meteor_score. PHẢI giữ dạng đã giải nén (không chỉ .zip):
# nltk.data.find() ưu tiên thư mục giải nén, và cơ chế tự-giải-nén-lúc-chạy của nltk không đáng tin
# cậy trên mọi filesystem (từng gặp FileSystemPathPointer trỏ vào thư mục rỗng trên Kaggle dù .zip
# có sẵn) — nên giải nén sẵn ở bước build, không để nltk tự lo lúc runtime.
python3 -c "
import nltk
nltk.download('wordnet', download_dir='$scoring_dir/nltk_data')
nltk.download('omw-1.4', download_dir='$scoring_dir/nltk_data')
"
cd "$scoring_dir/nltk_data/corpora"
unzip -q -o wordnet.zip
unzip -q -o omw-1.4.zip
rm -f wordnet.zip omw-1.4.zip
cd - >/dev/null

echo "wheelhouse_scoring sẵn sàng tại $scoring_dir — kiểm tra lại danh sách .whl khớp README trước khi đóng gói Dataset."
