#!/usr/bin/env bash
# Tái tạo wheelhouse/ (bị .gitignore loại khỏi repo vì là binary) trước khi
# đóng gói source/ thành Kaggle Dataset. Phải chạy trên máy Linux x86_64,
# Python 3.12 (khớp runtime Kaggle) — pip chọn wheel theo chính nền đang chạy,
# nên chạy trên nền khác (macOS/Windows/Python khác) sẽ tải sai platform tag.
set -euo pipefail

if [[ "$(python3 -c 'import platform; print(platform.system())')" != "Linux" ]] \
  || [[ "$(python3 -c 'import platform; print(platform.machine())')" != "x86_64" ]]; then
  echo "Cảnh báo: nên chạy script này trên Linux x86_64 (vd. trong chính Kaggle notebook)" \
       "để tải đúng wheel cho runtime Kaggle." >&2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheelhouse_dir="$script_dir/../wheelhouse"
mkdir -p "$wheelhouse_dir"

# Version pin khớp với wheelhouse/README.md (nâng Transformers lên bản nhận
# diện kiến trúc qwen3_5 của Qwen/Qwen3.5-2B). Đổi version ở đây thì cũng phải
# cập nhật README.
python3 -m pip download --no-deps --dest "$wheelhouse_dir" \
  "accelerate==1.14.0" \
  "bitsandbytes==0.49.2" \
  "click==8.4.2" \
  "hf_xet==1.6.0" \
  "huggingface_hub==1.26.1" \
  "peft==0.18.1" \
  "regex==2026.7.19" \
  "safetensors==0.8.0" \
  "sentence_transformers==5.7.0" \
  "sentencepiece==0.2.1" \
  "tokenizers==0.22.2" \
  "transformers==5.14.1"

echo "Wheelhouse sẵn sàng tại $wheelhouse_dir — kiểm tra lại danh sách .whl khớp README trước khi đóng gói Dataset."
