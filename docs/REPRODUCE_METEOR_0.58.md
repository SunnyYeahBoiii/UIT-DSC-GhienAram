# Tái tạo submission METEOR 0,58

Lượt **D1 b135 (vast)**: Qwen3.5-2B + adapter D1 `cuoi/` ở bước 135 (cùng phiên SFT với lượt
0,5742, lấy adapter cuối thay vì `tot_nhat` b80), sinh trên prompts hybrid của lượt 0,5734.
Nộp BTC ngày 2026-09-17. Tag git: `v10-meteor-0.58` → commit `dfa7ce2`.

## Vì sao b135 thắng b80 dù dev thấp hơn

Dev 120 chấm b80 0,6105 > b135 0,5915, nhưng LB ngược lại (0,5742 → 0,58). Sau đó đo được
dev 120 rò rỉ: 77/120 câu nằm nguyên văn trong `sft.jsonl`, nên chọn checkpoint theo dev đó
là chọn theo khả năng thuộc bài. Dev sạch thay thế: `dev100` (0 câu trùng sft, 3-gram
Jaccard < 0,5), backup `~/legalqa-backup/dev100-sach/`.

## Nguồn

| Thành phần | Giá trị |
|---|---|
| Code | `dfa7ce2` (`git archive` lên vast; `scripts/infer.py` không đổi so với `5b9f44d`) |
| Model nền | HF `Qwen/Qwen3.5-2B@15852e8` |
| Adapter | `~/legalqa-backup/sft-D1-qwen35-p5/out/cuoi/` (Kaggle `tunvln/v10-train-d1-qwen35-p5`, `cuoi_buoc` 135), md5 safetensors `16319f01…` |
| Prompts | `~/legalqa-backup/v10-hybrid-p3/prompts.jsonl` md5 `87f08a4b…` |
| Câu hỏi | `public-official.json` md5 `f01beac2…` |
| Phần cứng | vast.ai 4× RTX 5080, torch 2.11 cu128, transformers 5.14.1 |

## Lệnh

```bash
# mỗi shard một GPU, s=0|1
cp prompts.jsonl runs/i-d1b135-s$s/
python scripts/infer.py --cau-hoi public-official.json --model <qwen35> --adapter <cuoi> \
  --ra runs/i-d1b135-s$s --bo-qua-pha1 --tu $((s*500)) --den $((s*500+500)) \
  --max-new 1024 --gen-bs 4 --gio-pha2 100
```

Ghép 2 shard theo thứ tự id của submission 0,5734, `cat_lap` từng câu (cắt 56 câu).
Trung vị 351,5 từ, 134 câu ≥ 700 từ; chỉ 82/1.000 câu trùng nguyên văn b80.

## Bản sao cục bộ

`~/legalqa-backup/v10-d1-b135/`: `du_doan.jsonl` (sha256 f5904c8b…), `submission.json`
(a2792111…), `submission.zip`. Output shard thô: `~/legalqa-backup/d1-b135-vast/s{0,1}.jsonl`.

Batch fp16 (gen-bs 4, GPU khác T4) không tất định tuyệt đối → sinh lại có thể lệch vài phần
nghìn. Muốn đúng 0,58: nộp lại `~/legalqa-backup/v10-d1-b135/submission.zip`.
