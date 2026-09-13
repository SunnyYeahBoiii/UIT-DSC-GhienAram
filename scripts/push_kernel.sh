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
# $4 = machine_shape. Đo lại 2026-09-13: `kaggle kernels pull -m` trên kernel đang chạy
# T4 x2 trả về "machine_shape": "NvidiaTeslaT4", và kernels_push gán
# `request.machine_shape = acc or meta["machine_shape"]` (kaggle_api_extended.py:6330).
# Tức API BIẾT loại GPU — ghi chú cũ ở dưới nói ngược là sai, đã sửa.
SHAPE="${4:-NvidiaTeslaT4}"

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
  "model_sources": $MS,
  "machine_shape": "$SHAPE"
}
JSON

# Loại GPU ĐẶT ĐƯỢC từ đây qua "machine_shape" (xem SHAPE ở trên). Bản trước của file
# này khẳng định ngược lại vì chỉ thử cờ --accelerator với giá trị rác và thấy nó không
# báo lỗi — client không validate, nhưng giá trị VẪN được gửi lên. Kết luận cũ dựa trên
# "không báo lỗi", không dựa trên đọc lại giá trị thật.
kaggle kernels push -p "$DIR"
echo "https://www.kaggle.com/code/sunny070306/$SLUG"

# Vẫn phải xác nhận: đặt được KHÔNG có nghĩa là đặt đúng. Đọc lại từ server sau khi đẩy.
echo
echo "  Kiểm accelerator thật sự nhận (đừng tin push, đọc lại từ server):"
echo "    kaggle kernels pull sunny070306/$SLUG -m -p /tmp/k && grep machine_shape /tmp/k/kernel-metadata.json"
