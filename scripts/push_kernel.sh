#!/usr/bin/env bash
# Đẩy notebook trong kaggle/ lên Kaggle: $1 = file .ipynb, $2 = slug. Upload tay qua UI đã hai
# lần khiến notebook chạy bản cũ trong khi file trên máy đã sửa — triệu chứng là traceback trỏ
# vào dòng không còn tồn tại. Đẩy bằng CLI thì phiên bản luôn khớp commit.
#
# Codebase vào bằng dataset legalqa-v10-src (scripts/push_dataset.sh), không clone GitHub.
set -euo pipefail

NB="${1:-kaggle/eval_rerank.ipynb}"
SLUG="${2:-v10-eval-rerank}"
# $3 = model_sources, phân cách bằng dấu phẩy. Ref lấy ở đâu: metadata của notebook cũ
# đã gắn model đó (`kaggle kernels pull <slug> -m`) — CLI 2.2.4 KHÔNG liệt kê được
# instance của model, `kaggle models get` chỉ trả metadata gốc không có danh sách instance.
# Ref base Qwen: sunny070306/qwen3-5-2b-base/TensorFlow2/default/1 (nhãn framework sai từ
# lúc upload nhưng ref phải giữ đúng chữ, sai một ký tự là 404).
MODELS="${3:-}"

# Kiểm cú pháp mọi cell TRƯỚC khi đẩy. Notebook sinh bằng script từng mất "\n" cuối dòng
# -> mọi dòng nối liền nhau, JSON vẫn hợp lệ nên không ai biết cho tới khi Kaggle ném
# SyntaxError sau 12 phút dựng dữ liệu.
python3 "$(dirname "$0")/check_notebooks.py" "$NB"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

cp "$NB" "$DIR/$SLUG.ipynb"
MS="[]"
if [ -n "$MODELS" ]; then
  MS="[$(echo "$MODELS" | sed 's/[^,]*/"&"/g')]"
fi
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
  "model_sources": $MS
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

# In ra thành một BƯỚC, không phải comment: kernel mới bốc P100 hai lần liên tiếp, nên
# đó là default của Kaggle cho slug chưa từng đặt accelerator, không phải bốc thăm.
# Comment trong file không chặn được ai đang nhìn terminal.
if ! kaggle kernels list -m --page-size 100 2>/dev/null | grep -q "sunny070306/$SLUG"; then
  echo
  echo "  !! SLUG MỚI — accelerator đang là default (P100, sm_60, fp16 KHÔNG chạy)."
  echo "  !! Mở link trên -> Settings -> Accelerator -> GPU T4 x2 TRƯỚC khi chạy."
fi
