# Tái tạo submission METEOR 0,5742

Lượt **D1i b80**: Qwen3.5-2B + adapter D1 (SFT mới trên dữ liệu phiên 5 hybrid) chọn ở bước 80,
sinh trên prompts hybrid của lượt 0,5734. Nộp BTC ngày 2026-09-17.
Tag git: `v10-meteor-0.5742` → commit `5b9f44d`.

## Hai commit tham gia

| Giai đoạn | Commit (COMMIT.txt trong output kernel) | Kernel |
|---|---|---|
| SFT D1 | `07be921` | `tunvln/v10-train-d1-qwen35-p5` (`kaggle/train_sft.ipynb`) |
| Infer 2 shard | `5b9f44d` | `tunvln/v10-infer-d1-b80-s0` [0,500), `tunvln/v10-infer-d1-b80-s1` [500,1000) |

Tag đặt ở commit infer. `git diff 07be921 5b9f44d -- scripts/train_sft.py` chỉ thêm
`--kieu-prompt card` (mặc định giữ khuôn cũ), nên checkout tag vẫn train lại được D1.

## SFT D1

Dữ liệu phiên 5 (`~/legalqa-backup/sft-phien5`): vào 6.584 mẫu, `--max-seq 8704` loại 17 → train
6.567, dev 120. Mẫu > 600 từ nhân 3 (454 mẫu). 135 bước (1/3 epoch) trên T4.
METEOR dev 120: b40 0,5738 · **b80 0,6105** · b120 0,5896 · b135 0,5915 → dùng `tot_nhat` (b80).

Chênh dev → LB: 0,6105 → 0,5742 (−0,036), lớn hơn phiên 3 (−0,017) và C2 (−0,028): b80 được
chọn chính trên dev 120 nên điểm dev lạc quan.

## Infer

Nguồn gắn vào kernel: `sunny070306/official` (1.000 câu), `tunvln/legalqa-v10-src`,
`tunvln/legalqa-test-prompts-hybrid` (prompts lượt 0,5734, bỏ pha retrieval),
`tunvln/legalqa-adapter-d1-b80-phien5`. Model nền tải HF `Qwen/Qwen3.5-2B@15852e8`
(sha256 safetensors trùng model Kaggle dùng lúc train).

```
scripts/infer.py --cau-hoi public-official.json --model <base> --adapter <tot_nhat>
  --ra out --bo-qua-pha1 --tu 0|500 --den 500|1000 --max-new 1024 --gen-bs 2 --gio-pha2 10.5
```

Ghép 2 shard theo thứ tự id của submission 0,5734, `cat_lap` từng câu (cắt 44 câu).

## Bản sao cục bộ

`~/legalqa-backup/v10-d1-b80/` (ngoài repo): `du_doan.jsonl` (sha256 6ac642e0…),
`submission.json` (cf72aac4…), `submission.zip`. Output kernel: `~/legalqa-backup/d1-b80-s{0,1}/`,
adapter + log train: `~/legalqa-backup/sft-D1-qwen35-p5/`.

Batch fp16 trên GPU không tất định tuyệt đối → sinh lại có thể lệch vài phần nghìn. Muốn đúng
0,5742: nộp lại `~/legalqa-backup/v10-d1-b80/submission.zip`.
