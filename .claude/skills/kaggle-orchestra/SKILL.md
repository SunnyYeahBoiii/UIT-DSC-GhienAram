---
name: kaggle-orchestra
description: Chia một việc shard được (build data, infer) thành các khoảng câu liền nhau, chạy song song trên nhiều acc Kaggle, tự đổi acc khi hết quota/timeout, dừng cả lượt khi lỗi code/OOM, rồi merge. Dùng khi người dùng muốn "chia cho nhiều acc", "chạy trên N acc", "build/infer song song trên Kaggle".
---

# Kaggle orchestra

Mọi logic nằm trong `scripts/kaggle_orchestra.py` (tất định, có test `tests/test_kaggle_orchestra.py`).
Skill này chỉ hướng dẫn cách gọi, KHÔNG tự đẩy kernel bằng tay ngoài CLI.

Train KHÔNG shard được — đừng dùng skill cho train.

## Acc

CLI tự quét `~/.kaggle/access_token` và `~/.kaggle/backup_*/access_token`, đổi acc bằng env
`KAGGLE_API_TOKEN=<đường dẫn file>`. **Không bao giờ in/đọc nội dung token.** Share
dataset/model giữa các acc là việc người dùng tự làm — không kiểm tra.

## Job file (`kaggle/jobs/<tên>.json`)

```json
{
  "ten": "build-sft5",                 // [a-z0-9-], thành slug kernel <ten>-s<i>
  "notebook": "kaggle/shard_build_sft.ipynb",
  "tong": 5516,                         // số câu, chia đều [0,tong) cho mọi acc có token
  "output": "sft.jsonl",
  "merge": "jsonl",                     // "jsonl" nối | "submission" -> submission.json + zip
  "gpu": true, "machine_shape": "NvidiaTeslaT4",
  "datasets": ["sunny070306/official", "legalqa-v10-src"],  // không có "/" = dataset của chính acc chạy
  "models": [],
  "params": {"gen_bs": 4},             // đổ vào SHARD của notebook
  "so_shard": 0,                        // 0 = số acc
  "publish": {"acc": "tunvln", "dataset": "legalqa-sft-phien5"}   // tuỳ chọn
}
```

Notebook shard phải có ĐÚNG một cell bắt đầu `# ORCHESTRA-PARAMS`; CLI thay bằng
`SHARD = {job, chi_so, so_shard, tu, den, commit, noi_tiep, output, **params}`.
Hợp đồng notebook/script:
- chỉ xử lý câu `[tu, den)`, ghi `/kaggle/working/out/<output>`, flush từng dòng;
- nếu `noi_tiep` khác null: chép `*.jsonl` từ `/kaggle/input/**/<noi_tiep>*` vào out rồi bỏ qua id đã có;
- chỉ khi xong TRỌN khoảng mới gọi `ghi_moc(out, tu, den)` → `<output>.moc.json`. COMPLETE mà
  không mốc = bị cắt ngân sách → coi như lỗi môi trường, đổi acc chạy tiếp.

Mẫu: `kaggle/shard_smoke.ipynb`, `shard_build_sft.ipynb`, `shard_infer.ipynb`.

## Quy trình

1. `git status` sạch và đã commit — mỗi acc nhận `legalqa-v10-src` đúng commit HEAD.
2. `python scripts/kaggle_orchestra.py plan kaggle/jobs/<job>.json` → cho người dùng xem bảng chia.
3. Người dùng đồng ý → `python scripts/kaggle_orchestra.py run kaggle/jobs/<job>.json --yes`.
4. `watch` chạy nền (Bash `run_in_background`), `--moi 300` cho GPU, `--moi 60` cho smoke CPU:
   `python scripts/kaggle_orchestra.py watch kaggle/jobs/<job>.json --moi 300`.
   Không poll thêm — harness báo khi lệnh thoát.
5. Theo exit code:
   - **0**: mọi shard XONG → `merge` (kiểm phủ kín, mốc, trùng id; publish nếu job khai báo).
   - **2 (HỎNG)**: lỗi code hoặc OOM. CLI đã in link kernel còn chạy → bảo người dùng hủy tay
     (Kaggle không có API hủy). Đọc `orchestra/<job>/s<i>/lan<n>/*.log` tìm traceback, báo
     người dùng. Người dùng sửa code/job (vd hạ `gen_bs`) → commit → `run` lại: shard XONG
     giữ nguyên, shard dở nối tiếp từ output đã có. KHÔNG tự hạ tham số.
   - **3**: còn shard chờ mà không acc nào rảnh/đủ quota ≥1h → báo người dùng, chờ quota refresh
     hoặc thêm acc rồi `watch` lại.
6. `status` in bảng bất cứ lúc nào; state ở `orchestra/<job>/state.json` (gitignored).

## Phân loại lỗi (đã code sẵn, đừng tự phán khác)

| Dấu hiệu | Loại | Hành động |
|---|---|---|
| `OutOfMemoryError` / `CUDA out of memory` | oom | HỎNG cả lượt |
| Traceback khác (cuối log không phải lỗi mạng) | code | HỎNG cả lượt |
| Push báo `not valid`/`not found`/403 | code | HỎNG |
| Push báo quota/limit, không traceback (12h), lỗi mạng, COMPLETE thiếu mốc | môi trường | đổi acc, nối tiếp |

Mỗi shard tối đa `MAX_LAN=6` lần đẩy.
