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
cat > "$DIR/dataset-metadata.json" <<'JSON'
{
  "title": "legalqa-v10-src",
  "id": "sunny070306/legalqa-v10-src",
  "licenses": [{"name": "unknown"}]
}
JSON

kaggle datasets version -p "$DIR" -r zip -m "$(git log --oneline -1 "$BRANCH")"
echo "https://www.kaggle.com/datasets/sunny070306/legalqa-v10-src"
