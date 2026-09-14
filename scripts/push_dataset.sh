#!/usr/bin/env bash
# Đẩy codebase lên Kaggle Dataset `legalqa-v10-src`.
#
# Vì sao Dataset chứ không clone GitHub: repo private, fine-grained PAT qua Kaggle Secrets
# trả 404 (GitHub giấu repo private khỏi token không có quyền) — hai vòng debug không đổi
# lấy gì. Dataset không cần auth và chạy được cả khi Internet OFF.
#
# git archive: chỉ lấy file đã track, nên data/ và checkpoint không lọt vào — ràng buộc
# repo không chứa dữ liệu thi vẫn giữ nguyên ở đây.
set -euo pipefail

BRANCH="${1:-feat/v10-rebuild}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

git archive "$BRANCH" | tar -x -C "$DIR"
git log --oneline -1 "$BRANCH" > "$DIR/COMMIT.txt"
# Chủ sở hữu hỏi CLI, không ghi cứng: quota GPU 30 giờ/tuần của sunny070306 đã cạn nên
# phải chạy bằng tài khoản khác, mà dataset là riêng từng tài khoản.
OWNER=$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')
[ -n "$OWNER" ] || { echo "không đọc được username — chạy: kaggle config view"; exit 1; }

cat > "$DIR/dataset-metadata.json" <<JSON
{
  "title": "legalqa-v10-src",
  "id": "$OWNER/legalqa-v10-src",
  "licenses": [{"name": "unknown"}]
}
JSON

# `version` hỏng khi dataset chưa tồn tại trên tài khoản mới; thử create trước, im lặng
# bỏ qua nếu đã có rồi mới lên version.
MSG="$(git log --oneline -1 "$BRANCH")"
kaggle datasets create -p "$DIR" -r zip 2>/dev/null \
  || kaggle datasets version -p "$DIR" -r zip -m "$MSG"
echo "https://www.kaggle.com/datasets/$OWNER/legalqa-v10-src"
