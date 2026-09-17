# Tái tạo submission METEOR 0,5734

Lượt **hybrid phiên 3** (lượt B), nộp BTC ngày 2026-09-17. Tag git: `v10-meteor-0.5734` → commit `07be921`.

## Vì sao tag ở 07be921, không phải f3226ec

Ba shard sinh xong lúc 02:53 (17/09), trước commit sanitize `f3226ec` (04:49). Đã kiểm:
`submission.zip` khớp 1.000/1.000 câu với `du_doan.jsonl` thô → bản 0,5734 **chưa** qua
`cat_lap`. Code sinh là `bdf01da` (hybrid retrieval + adapter phiên 3), `07be921` thêm
notebook chia shard `kaggle/infer_hybrid_phien3_shard.ipynb`.

## Kernel Kaggle

| Shard | Kernel | Khoảng câu |
|---|---|---|
| s0 | `tdwuyn/v10-infer-hyb-s0` | [0, 334) |
| s1 | `hngphckhmt2024nguyn/v10-infer-hyb-s1` | [334, 667) |
| s2 | `wiedergeburt/v10-infer-hyb-s2` | [667, 1000) |

Model nền Qwen3.5-2B (= HF `Qwen/Qwen3.5-2B@15852e8`), adapter `tunvln/legalqa-adapter-phien3`
(`tot_nhat/`, không SFT lại), dữ liệu `sunny0703/legalqa-clean`, câu hỏi `sunny070306/official`.

## Bản sao cục bộ

`~/legalqa-backup/v10-hybrid-p3/` (ngoài repo): `du_doan.jsonl`, `prompts.jsonl`,
`submission.{json,zip}`, `SHA256SUMS` (`du_doan.jsonl` f9fa391b…, `submission.json` dc17a15c…).

## Chạy lại

```bash
git checkout v10-meteor-0.5734
d=$(mktemp -d) && git archive v10-meteor-0.5734 | tar -x -C $d && git log -1 --format='%h %s' v10-meteor-0.5734 > $d/COMMIT.txt
# kaggle datasets version -p $d -m "tái tạo 0.5734" cho 3 acc, rồi push notebook shard với --tu/--den như bảng
python scripts/sanitize_submission.py ...   # KHÔNG chạy nếu muốn đúng 0,5734
```

Batch fp16 trên GPU không tất định tuyệt đối → sinh lại có thể lệch vài phần nghìn. Muốn đúng
0,5734: nộp lại `~/legalqa-backup/v10-hybrid-p3/submission.zip`.
