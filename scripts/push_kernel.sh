#!/usr/bin/env bash
# Đẩy notebook trong kaggle/ lên Kaggle: $1 = file .ipynb, $2 = slug. Upload tay qua UI đã hai
# lần khiến notebook chạy bản cũ trong khi file trên máy đã sửa — triệu chứng là traceback trỏ
# vào dòng không còn tồn tại. Đẩy bằng CLI thì phiên bản luôn khớp commit.
#
# Codebase vào bằng dataset legalqa-v10-src (scripts/push_dataset.sh), không clone GitHub.
set -euo pipefail

NB="${1:-kaggle/eval_rerank.ipynb}"
SLUG="${2:-v10-eval-rerank}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

cp "$NB" "$DIR/$SLUG.ipynb"
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "sunny070306/$SLUG",
  "title": "$SLUG",
  "code_file": "$SLUG.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sunny070306/official", "sunny070306/legalqa-v10-src", "sunny070306/legalqa-warmup"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

# LOẠI GPU KHÔNG ĐẶT ĐƯỢC TỪ ĐÂY. Đã đo trên kaggle CLI 2.2.4: cờ --accelerator nhận cả
# giá trị rác ("BOGUS") mà không báo lỗi, và trường "accelerator"/"machine_shape" trong
# kernel-metadata.json đọc ngược lại luôn ra "Gpu" — API chỉ biết có/không GPU. Lần chạy
# đầu bốc trúng P100 (sm_60) nên .half() vỡ ở phút 30.
# -> Đặt tay: mở notebook, Settings -> Accelerator -> GPU T4 x2. Setting bám theo kernel,
#    đặt một lần là các version sau giữ nguyên.
kaggle kernels push -p "$DIR"
echo "https://www.kaggle.com/code/sunny070306/$SLUG"
