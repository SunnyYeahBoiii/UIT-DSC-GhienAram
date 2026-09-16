# Tái tạo submission METEOR 0,5539 / ROUGE-L 0,5717

Submission **lượt 3** nộp BTC ngày 2026-09-16. Tag git: `v10-meteor-0.5539` → commit `5f7d75d`.

## Vì sao tag ở 5f7d75d là đủ

Lượt 3 NỐI TIẾP lượt trước: 840 câu sinh ở `tunvln/v10-infer-submit` (lượt 1, notebook
`65dc607`), 160 câu còn lại ở `tunvln/v10-infer-submit-3`. `git diff 65dc607 5f7d75d --
src scripts/infer.py scripts/build_sft.py` RỖNG — hai lượt chạy cùng code sinh, chỉ notebook
đổi để nối tiếp. Commit sau (`1d13687`) là lượt 4 (`--max-new 1536`), KHÔNG thuộc bản này.

## Nguồn ngoài git (kernel `tunvln/v10-infer-submit-3`)

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| `tunvln/legalqa-adapter-phien3` | adapter `tot_nhat/` phiên 3 | 87 MB safetensors, cập nhật 2026-09-16 02:16 |
| `sunny0703/legalqa-clean` | chunks, bm25.pkl, corpus | 1,2 GB, cập nhật 2026-09-14 11:56 |
| `tunvln/legalqa-du-doan-phien3` | 840 câu của lượt 1 để nối tiếp | 2,2 MB |
| `tunvln/legalqa-v10-src` | codebase | **đã bị ghi đè** bởi bản sau — đẩy lại từ tag |
| `sunny070306/official` | `public-official.json` 1.000 câu | |
| model `sunny070306/qwen3-5-2b-base/TensorFlow2/default/1` | Qwen3.5-2B nền | |
| `AITeamVN/Vietnamese_Reranker` | reranker, tải HF lúc chạy | Internet ON |

Cấu hình: T4, `LEGALQA_SO_HIEU=1`, top_n=5 không trần ký tự, retrieval **BM25 + rerank**
(chưa có dense), `--batch-size 32 --pool 50 --max-new 1024 --gen-bs 2`, `transformers==5.14.1`
+ `peft` + `nltk`, gỡ `torchao`. Docker:
`gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.

## Bản sao cục bộ

`~/legalqa-backup/v10-meteor-0.5539/` (ngoài repo — repo không chứa dữ liệu/adapter/submission):
`submission_nop_0.5539.{zip,json}` (file đã nộp), `kernel/` (notebook + metadata), `kernel_output/`,
`adapter-phien3/`, `du-doan-phien3/`, `legalqa-clean/`, `SHA256SUMS`.

## Chạy lại

```bash
git checkout v10-meteor-0.5539
# đẩy codebase đúng commit lên dataset src (COMMIT.txt phải = 5f7d75d)
d=$(mktemp -d) && git archive v10-meteor-0.5539 | tar -x -C $d && git log -1 --format='%h %s' v10-meteor-0.5539 > $d/COMMIT.txt
# ... kaggle datasets version -p $d -m "tái tạo 0.5539"
# rồi push kaggle/infer_submit.ipynb với metadata trong backup kernel/kernel-metadata.json
```

Sinh là greedy (`do_sample=False`) nhưng batch sắp theo độ dài prompt và kernel fp16 trên GPU
không tất định tuyệt đối → chạy lại từ đầu có thể lệch vài phần nghìn METEOR.
Muốn ra ĐÚNG 0,5539: nộp lại `submission_nop_0.5539.zip` hoặc nối tiếp từ
`kernel_output/out/du_doan.jsonl` (đủ 1.000 câu → pha 2 không sinh gì).
