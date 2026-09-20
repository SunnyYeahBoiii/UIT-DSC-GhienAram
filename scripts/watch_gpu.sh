#!/usr/bin/env bash
# Theo dõi GPU + tiến độ mine/train reranker trên instance Vast.
# Chạy TRÊN máy GPU:  bash scripts/watch_gpu.sh
# Từ laptop:          ssh -p 35050 root@HOST 'bash /workspace/legalqa/scripts/watch_gpu.sh'
# Ctrl-C để thoát; không đụng process train.

set -u
ROOT="${LEGALQA_ROOT:-/workspace/legalqa}"
LOG="$ROOT/runs"
CLEAN="$ROOT/data/clean"
SEC="${WATCH_SEC:-2}"

if [[ ! -d "$ROOT" ]]; then
  echo "không thấy $ROOT — export LEGALQA_ROOT=... hoặc chạy trên instance GPU" >&2
  exit 1
fi

# tmux/ssh hay nuốt \033[2J; in ngăn cách rõ vẫn đọc được khi không clear được.
ve_man_hinh() {
  printf '\033[H\033[2J' 2>/dev/null || true
}

dong_cuoi() {
  local f="$1" n="${2:-8}"
  if [[ -f "$f" ]]; then
    tail -n "$n" "$f" | sed 's/\r/\n/g' | grep -v '^[[:space:]]*$' | tail -n "$n"
  else
    echo "(chưa có $f)"
  fi
}

while true; do
  ve_man_hinh
  echo "===== $(date -u +%FT%TZ)  refresh ${SEC}s  Ctrl-C thoát ====="
  echo

  echo "--- GPU ---"
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "  %s  util %s%%  mem %s/%s MiB (util %s%%)  %s°C  %s W\n",$1,$2,$4,$5,$3,$6,$7}'
  echo

  echo "--- process ---"
  ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu \
    | grep -E 'mine_rerank|train_reranker|run_rerank_ft' \
    | grep -v grep \
    | awk '{printf "  pid %s  %s  cpu %s%%  mem %s%%  %s\n",$1,$2,$3,$4,substr($0, index($0,$5))}' \
    || echo "  (không có mine/train đang chạy)"
  echo

  echo "--- pipeline ---"
  dong_cuoi "$LOG/pipeline.log" 6
  echo

  echo "--- dữ liệu mine ---"
  wc -l "$CLEAN"/rerank_train*.jsonl 2>/dev/null | awk '{printf "  %8s  %s\n",$1,$2}' || echo "  (chưa có jsonl)"
  echo

  echo "--- mine0 ---"
  dong_cuoi "$LOG/mine0.log" 4
  echo
  echo "--- mine1 ---"
  dong_cuoi "$LOG/mine1.log" 4
  echo
  echo "--- train ---"
  dong_cuoi "$LOG/train.log" 8
  echo

  sleep "$SEC"
done
