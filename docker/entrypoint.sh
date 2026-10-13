#!/usr/bin/env bash
# /repro = gói repro-row1 (candidates/, prompts/, data/, runs/, hf/, models/). /out = thư mục ghi.
set -euo pipefail
R=${REPRO_DIR:-/repro}
O=${OUT_DIR:-/out}
mkdir -p "$O"

# Công thức row1: b80 r0 + b320 r0 + b920 r0-r3 (6 ứng viên/câu, flat MBR).
BO=(b80:0 b320:0 b920:0 b920:1 b920:2 b920:3)

mbr() {  # mbr <thư mục chứa 6 candidate> -> $O/submission.json + submission.zip
  python scripts/mbr_chon.py \
    "$1"/b80_r0.jsonl "$1"/b320_r0.jsonl "$1"/b920_r{0,1,2,3}.jsonl \
    --cau-hoi "$R"/data/private-official.json --ra "$O"
}

gen() {  # gen <ckpt> <rot> -> $O/candidates/<ckpt>_r<rot>.jsonl (append, chạy lại được)
  mkdir -p "$O"/candidates
  python scripts/infer_private_shard.py --model "$R/hf/Qwen3.5-2B" \
    --adapter "$R/runs/sft_v11/$1" --prompts "$R/prompts/prompts_rot$2.jsonl" \
    --ra "$O/candidates/$1_r$2.jsonl" --max-new 1024 --gen-bs 2 --gio-toi-da 1000
}

case "${1:-mbr}" in
  mbr)  # CPU: MBR từ 6 candidate đã đóng gói
    mbr "$R"/candidates ;;
  gen)  # GPU: sinh lại 1 candidate, vd `gen b920 0`
    gen "$2" "$3" ;;
  all)  # GPU: sinh lại cả 6 candidate rồi MBR trên bản vừa sinh
    for x in "${BO[@]}"; do gen "${x%:*}" "${x#*:}"; done
    mbr "$O"/candidates ;;
  *) exec "$@" ;;
esac
