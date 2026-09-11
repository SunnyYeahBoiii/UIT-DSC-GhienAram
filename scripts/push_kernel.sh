#!/usr/bin/env bash
# Đẩy kaggle/eval_rerank.ipynb lên Kaggle. Upload tay qua UI đã hai lần khiến notebook chạy
# bản cũ trong khi file trên máy đã sửa — triệu chứng là traceback trỏ vào dòng không còn
# tồn tại. Đẩy bằng CLI thì phiên bản luôn khớp commit.
#
# Secret GITHUB_TOKEN và loại GPU (T4 x2) API không đặt được, phải bấm trong UI một lần.
set -euo pipefail

NB="${1:-kaggle/eval_rerank.ipynb}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

cp "$NB" "$DIR/v10-eval-rerank.ipynb"
cat > "$DIR/kernel-metadata.json" <<'JSON'
{
  "id": "sunny070306/v10-eval-rerank",
  "title": "v10 eval rerank",
  "code_file": "v10-eval-rerank.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sunny070306/official"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

kaggle kernels push -p "$DIR"
echo "https://www.kaggle.com/code/sunny070306/v10-eval-rerank"
