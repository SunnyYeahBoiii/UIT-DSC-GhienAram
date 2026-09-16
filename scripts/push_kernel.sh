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
# $5 = kernel_sources, phân cách bằng dấu phẩy. Dùng output của kernel trước làm input:
# dựng lại clean/chunk/gold mất 26 phút, và kernel bật GPU thì 26 phút đó đốt quota
# GPU dù không chạm GPU. Ngân sách là 30 GPU-h/tuần nên đây không phải tối ưu vặt.
KERNELS="${5:-}"
# KAGGLE_GPU=false cho kernel CPU thuần. Dựng clean/ mất 26 phút mà không chạm GPU —
# bật GPU thì 26 phút đó vẫn bị trừ vào quota 30 giờ/tuần, đúng thứ đang cạn.
GPU="${KAGGLE_GPU:-true}"

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
KS="[]"
if [ -n "$KERNELS" ]; then
  KS="[$(echo "$KERNELS" | sed 's/[^,]*/"&"/g')]"
fi
# Chủ sở hữu lấy từ ~/.kaggle/kaggle.json chứ không ghi cứng "sunny070306": quota GPU
# 30 giờ/tuần đã cạn nên phải chạy bằng tài khoản khác, mà `id` sai chủ thì Kaggle từ
# chối push với lỗi 403 khó đọc. `dataset_sources` cũng theo chủ mới — dataset là riêng
# từng tài khoản, trừ model nền được share thẳng nên giữ ref của chủ cũ.
# Hỏi chính CLI, đừng đọc kaggle.json: máy này auth bằng `credentials.json` (OAuth,
# `auth_method: ACCESS_TOKEN`) còn kaggle.json chỉ có khoá `key` — đọc `username` ở đó
# ném KeyError. `kaggle config view` in ra username của auth ĐANG có hiệu lực, đúng thứ
# quyết định kernel thuộc về ai.
OWNER=$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')
[ -n "$OWNER" ] || { echo "không đọc được username — chạy: kaggle config view"; exit 1; }
echo "chủ sở hữu kernel: $OWNER"
# Mục có sẵn "/" giữ nguyên chủ: dữ liệu thi vẫn nằm ở `sunny070306/official` (được
# share sang), chỉ dataset do ta đẩy mới thuộc $OWNER. Prefix tất tay thành
# "$OWNER/official" là 404 im lặng — kernel chạy rồi mới chết ở assert selected-contexts.
DS=${KAGGLE_DS:-sunny070306/official,legalqa-v10-src,legalqa-clean}
DSJ="[$(echo "$DS" | tr ',' '\n' | sed "s|^\([^/]*\)$|$OWNER/\1|; s|.*|\"&\"|" | paste -sd, -)]"

cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "$OWNER/$SLUG",
  "title": "$SLUG",
  "code_file": "$SLUG.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": $GPU,
  "enable_internet": true,
  "dataset_sources": $DSJ,
  "competition_sources": [],
  "kernel_sources": $KS,
  "model_sources": $MS,
  "machine_shape": "$SHAPE"
}
JSON

# Loại GPU ĐẶT ĐƯỢC từ đây qua "machine_shape" (xem SHAPE ở trên). Bản trước của file
# này khẳng định ngược lại vì chỉ thử cờ --accelerator với giá trị rác và thấy nó không
# báo lỗi — client không validate, nhưng giá trị VẪN được gửi lên. Kết luận cũ dựa trên
# "không báo lỗi", không dựa trên đọc lại giá trị thật.
# Bắt "successfully pushed" chứ không tin mã thoát: đẩy trong lúc kernel ĐANG CHẠY thì
# Kaggle in URL, thoát 0, mà KHÔNG tạo version mới. 2026-09-14 đã mất một vòng vì thế —
# tưởng v10-build-data chạy bản bỏ trần ký tự, server vẫn giữ bản cũ. Cùng họ với
# `datasets create` in "title already in use" rồi vẫn thoát 0.
OUT=$(kaggle kernels push -p "$DIR" 2>&1)
echo "$OUT"
echo "$OUT" | grep -q "successfully pushed" || {
  echo
  echo "PUSH KHÔNG TẠO VERSION MỚI. Thường vì kernel đang chạy — chờ nó xong rồi đẩy lại."
  echo "    kaggle kernels status $OWNER/$SLUG"
  exit 1
}
echo "https://www.kaggle.com/code/$OWNER/$SLUG"

# Vẫn phải xác nhận: đặt được KHÔNG có nghĩa là đặt đúng. Đọc lại từ server sau khi đẩy.
echo
echo "  Kiểm accelerator thật sự nhận (đừng tin push, đọc lại từ server):"
echo "    kaggle kernels pull $OWNER/$SLUG -m -p /tmp/k && grep machine_shape /tmp/k/kernel-metadata.json"
