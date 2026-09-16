#!/usr/bin/env bash
# Tải output của kernel train và publish adapter thành dataset `legalqa-adapter-phien3`.
#
# Vì sao phải có bước này: `infer_submit.ipynb` tìm adapter bằng
#   glob("**/tot_nhat/adapter_config.json") rồi lọc "phien3" in str(q)
# còn `legalqa-sft-phien3` CHỈ chứa sft.jsonl + dev.jsonl. Không có bước publish thì
# infer chết ở assert "không thấy tot_nhat/ của phiên 3" sau khi đã tải xong reranker.
#
# Vì sao dataset chứ không `kernel_sources`: output kernel KHÔNG chia sẻ được giữa hai
# tài khoản, dataset thì được. Cả pipeline đang chạy chéo hai tài khoản (base model của
# sunny070306, kernel của sunny0703).
set -euo pipefail

KERNEL="${1:-v10-train-sft5}"
# Tên PHẢI chứa "phien3" (infer_submit) hoặc "phien5" (infer_hybrid, adapter train trên
# prompt hybrid) — notebook infer lọc adapter theo đúng chuỗi đó, sai tên là assert chết.
TEN="${2:-legalqa-adapter-phien5}"
case "$TEN" in *phien3*|*phien5*) ;; *) echo "tên dataset phải chứa 'phien3' hoặc 'phien5', notebook infer lọc theo đó"; exit 1;; esac

OWNER=$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')
[ -n "$OWNER" ] || { echo "không đọc được username — chạy: kaggle config view"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# ADAPTER_SRC: thư mục out/ đã tải sẵn trên máy dev. Cần vì output kernel KHÔNG chia sẻ
# giữa hai tài khoản — adapter train bằng `sunny0703` không tải lại được bằng `tunvln`,
# nhưng bản đã nằm trên đĩa thì publish lên tài khoản nào cũng được.
if [ -n "${ADAPTER_SRC:-}" ]; then
  [ -d "$ADAPTER_SRC" ] || { echo "ADAPTER_SRC không tồn tại: $ADAPTER_SRC"; exit 1; }
  mkdir -p "$TMP/raw"
  cp -R "$ADAPTER_SRC" "$TMP/raw/out"
else
  echo "tải output $OWNER/$KERNEL ..."
  kaggle kernels output "$OWNER/$KERNEL" -p "$TMP/raw" --force
fi

# `repo/` là bản sao codebase kernel tự copy ra working — 30 MB không liên quan, và đẩy
# nó lên làm dataset adapter là nhầm lẫn chờ sẵn cho lần sau. Chỉ lấy out/.
SRC="$TMP/raw/out"
[ -d "$SRC" ] || { echo "không thấy out/ trong output kernel — train chưa ghi checkpoint nào"; exit 1; }

DIR="$TMP/ds"
mkdir -p "$DIR"
cp -R "$SRC/." "$DIR/"
# Chọn checkpoint: `tot_nhat` chỉ được ghi khi VƯỢT mốc METEOR cũ. Phiên train lại từ
# đầu chỉ được ~136 bước trong khi phiên 2 cần 318 bước mới chạm đỉnh, nên rất có thể
# `tot_nhat` KHÔNG tồn tại. Khi đó `cuoi` là thứ duy nhất có, và cũng là thứ đúng để
# nối phiên sau. Nhân bản nó thành tot_nhat/ để infer_submit tìm được mà không phải sửa
# notebook — ghi rõ ra đây vì đây là một quyết định, không phải một mẹo.
if [ ! -d "$DIR/tot_nhat" ]; then
  [ -d "$DIR/cuoi" ] || { echo "không có cả tot_nhat/ lẫn cuoi/ — không có gì để publish"; exit 1; }
  echo "CHÚ Ý: không có tot_nhat/ (chưa vượt mốc METEOR cũ). Dùng cuoi/ làm tot_nhat/."
  cp -R "$DIR/cuoi" "$DIR/tot_nhat"
fi
ls "$DIR/tot_nhat"

cat > "$DIR/dataset-metadata.json" <<JSON
{
  "title": "$TEN",
  "id": "$OWNER/$TEN",
  "licenses": [{"name": "unknown"}]
}
JSON

# Hỏi TỒN TẠI trước, đừng dựa vào mã thoát: `datasets create` trên dataset đã có in
# "title already in use" rồi vẫn trả về 0, nên `create || version` không bao giờ chạy tới.
MSG="adapter tu $KERNEL $(date +%Y-%m-%d\ %H:%M)"
if kaggle datasets files "$OWNER/$TEN" >/dev/null 2>&1; then
  kaggle datasets version -p "$DIR" -r zip -m "$MSG"
else
  kaggle datasets create -p "$DIR" -r zip
fi
echo "https://www.kaggle.com/datasets/$OWNER/$TEN"
echo
echo "Bước sau — đẩy infer:"
echo "  KAGGLE_DS=sunny070306/official,legalqa-v10-src,legalqa-clean,$TEN \\"
echo "    bash scripts/push_kernel.sh kaggle/infer_hybrid.ipynb v10-infer-hybrid \\"
echo "    \"sunny070306/qwen3-5-2b-base/TensorFlow2/default/1\" NvidiaTeslaT4"
