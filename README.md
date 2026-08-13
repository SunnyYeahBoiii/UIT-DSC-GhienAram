# UIT DSC 2026 LegalQA — bundle v9.1 + dữ liệu

Thư mục này gom các thành phần cần thiết để lưu trữ và chia sẻ nội bộ pipeline
LegalQA:

- `source/`: snapshot v9.1 (không còn `source_v8/` riêng — bundle đã hợp nhất
  về một cây source duy nhất), kèm các tiện ích zero-train được bổ sung sau
  snapshot gốc.
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

`pyproject.toml` phiên bản `0.9.1`. Toàn bộ 80 file chung khớp từng byte với
snapshot gốc `legalqa_3p88b_kaggle_source_v9_1_graph_gate_final.zip`; cây
`source/` hiện tại có thêm sáu file so với snapshot đó (snapshot v8.0 riêng đã
không còn được đóng gói cùng bundle này — xem [`SOURCE_ORIGINS.md`](SOURCE_ORIGINS.md)
để biết nguồn gốc lịch sử):

- `ZERO_TRAIN_V8_1.md`
- `scripts/build_v8_zero_train.py`
- `server/README_SERVER.md`
- `server/run_server.py`
- `src/legalqa/zero_train.py`
- `tests/test_zero_train.py`

Checksum archive gốc và thống kê chi tiết nằm trong
[`SOURCE_ORIGINS.md`](SOURCE_ORIGINS.md).

## Đưa lên GitHub

`.gitignore` loại hẳn `source/wheelhouse/` (wheel nhị phân — không nên nằm
trong lịch sử git) và toàn bộ `data/`/`checksums/` (dữ liệu cuộc thi, xem cảnh
báo phân phối ở trên) khỏi repo; hiện repo **không** dùng Git LFS (không có
`.gitattributes`). `git ls-files` ở gốc bundle chỉ nên trả về
`README.md`, `SOURCE_ORIGINS.md`, `.gitignore` và cây `source/`.

Trước khi đóng gói `source/` thành Kaggle Dataset, chạy lại
`source/scripts/build_wheelhouse.sh` để tái tạo `source/wheelhouse/` (xem
[`source/wheelhouse/README.md`](source/wheelhouse/README.md)) — thư mục này
không đi theo git nên mỗi máy đóng gói phải tự tạo lại.

GitHub cảnh báo file Git thường trên 50 MiB và chặn file trên 100 MiB; nếu về
sau cần commit checkpoint hoặc file nhị phân lớn, cân nhắc Git LFS lúc đó thay
vì trước. Xem tài liệu chính thức:

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
