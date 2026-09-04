# v5007 real

Đây là baseline lưu nguyên bốn gói người dùng xác nhận thuộc lần chạy đạt public METEOR `0.5007`.

## Cấu trúc

- `artifacts/original_archives/05_selector_calibration.zip`: selector calibration.
- `artifacts/original_archives/ALL_TRAINING_05.zip`: source/output huấn luyện và QLoRA full.
- `artifacts/original_archives/legalqa_dense_index05.zip`: global dense index.
- `artifacts/original_archives/source_05_inference.zip`: source inference được tải lại.
- `source/`: nội dung được bung nguyên từ `source_05_inference.zip`; đây là điểm bắt đầu cho thay đổi source tiếp theo trên branch này.
- `V5007_REAL_MANIFEST.json`: checksum và kết luận đối chiếu máy đọc được.
- `V5007_PROVENANCE_AUDIT.md`: báo cáo kiểm chứng chi tiết.

## Quy tắc phát triển

1. Không sửa bốn ZIP trong `artifacts/original_archives`; đây là bản chứng cứ bất biến.
2. Mọi thay đổi tiếp theo thực hiện trong `source/` và commit trên branch phát triển tách từ `v5007-real`.
3. Không gọi `source/` là exact final-runtime revision: log thật dùng các guard/cờ CLI không có trong ZIP. Training, dense index và selector đã được xác minh trùng artifact 0.5007.
4. Khi đánh giá cải tiến, chỉ thay một nhóm biến và lưu lại hash của data, adapter, dense index, selector cùng source commit.
