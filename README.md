# UIT DSC 2026 LegalQA — bundle v8 + v9.1 + dữ liệu

Thư mục này gom các thành phần cần thiết để lưu trữ và chia sẻ nội bộ pipeline
LegalQA:

- `source_v8/`: snapshot v8.0 dùng global retrieval và document graph.
- `source_v9/`: snapshot v9.1 mới nhất, kèm các tiện ích zero-train được bổ sung
  sau snapshot.
- `data/official/`: `train.json`, `public-official.json` và toàn bộ 8.532 JSON
  trong `selected-contexts/`.
- `data/warmup/warmup.json`: 500 câu warm-up.
- `data/documentation/`: tài liệu tổng quan dữ liệu do Ban Tổ chức cung cấp.
- `checksums/`: manifest SHA-256 để kiểm tra file sau khi sao chép hoặc tải về.

Không có checkpoint mô hình, Hugging Face model, adapter QLoRA, dense index,
candidate cache, recovery cache hoặc prediction/submission trong bundle này.

## Cảnh báo về quyền phân phối dữ liệu

Nên tạo **GitHub repository private** và chỉ mời thành viên đội. Dữ liệu cuộc thi
có thể chịu điều khoản hạn chế phát tán của Ban Tổ chức; không chuyển repository
sang public nếu chưa có xác nhận cho phép. Bundle không tự cấp thêm giấy phép cho
source hoặc dữ liệu.

## Dấu vết phiên bản

- v8: `pyproject.toml` phiên bản `0.8.0`, giải nén nguyên trạng từ
  `legalqa_3p88b_kaggle_source_v8_global_graph.zip`.
- v9: `pyproject.toml` phiên bản `0.9.1`. Toàn bộ 80 file chung khớp từng byte
  với `legalqa_3p88b_kaggle_source_v9_1_graph_gate_final.zip`; cây source hiện tại
  có thêm sáu file:
  - `ZERO_TRAIN_V8_1.md`
  - `scripts/build_v8_zero_train.py`
  - `server/README_SERVER.md`
  - `server/run_server.py`
  - `src/legalqa/zero_train.py`
  - `tests/test_zero_train.py`

Checksum archive gốc và thống kê chi tiết nằm trong
[`SOURCE_ORIGINS.md`](SOURCE_ORIGINS.md).

## Đưa lên GitHub

Bundle có working tree khoảng 677 MB. Hai thư mục `wheelhouse/` chứa wheel nhị
phân lớn, vì vậy `.gitattributes` đã cấu hình Git LFS cho `*.whl`, `*.docx`,
`*.zip` và checkpoint tương lai. Cài Git LFS trước khi `git add`:

```powershell
cd "C:\Users\LOQ\Documents\Codex\2026-08-06\b\outputs\legalqa_github_bundle_v8_v9"
git lfs install
git init
git add .
git status
git commit -m "Add LegalQA v8, v9.1 and official competition data"
git branch -M main
git remote add origin <URL_REPOSITORY_PRIVATE>
git push -u origin main
```

Không upload bằng giao diện web của GitHub; hãy dùng Git CLI hoặc GitHub Desktop.
GitHub cảnh báo file Git thường trên 50 MiB và chặn file trên 100 MiB, còn Git
LFS lưu file lớn qua pointer. Xem tài liệu chính thức:

- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

## Kiểm tra tính toàn vẹn

Từ PowerShell, chạy:

```powershell
Get-Content checksums\SHA256SUMS.txt | ForEach-Object {
    $hash, $relative = $_ -split '  ', 2
    $actual = (Get-FileHash -LiteralPath $relative -Algorithm SHA256).Hash
    if ($actual -ne $hash) { throw "Checksum mismatch: $relative" }
}
```

Manifest dùng đường dẫn tương đối và dấu `\`, phù hợp khi chạy từ thư mục gốc
của bundle trên Windows.
