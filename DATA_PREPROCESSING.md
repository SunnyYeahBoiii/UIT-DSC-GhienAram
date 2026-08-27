# Tiền xử lý dữ liệu huấn luyện

Pipeline hiện tại:

1. Kiểm tra schema, ID trùng/rỗng, câu hỏi và đáp án thiếu.
2. Chuẩn hóa NFC; đổi NBSP thành khoảng trắng; xóa BOM và soft-hyphen.
3. Xóa câu hỏi/caption web ở cuối, marker ảnh, CTA tải tài liệu, dòng Xem thêm, boilerplate, title treo, dòng lặp và token ++table.
4. Bảo toàn đủ 7.000 ID trong source/data/train_clean.json. SHA-256: e967d0e3e4e1dc9b8e0495071b6ad8e9cb46362290071b5f706f90d068d3fe7b.
5. Runner QLoRA chỉ đọc train_clean.json, áp dụng 47 ID loại trong training_quality.json (answer mất bảng/biểu mẫu), rồi tạo prepared_train/qlora_train.json gồm 6.953 mẫu trong working. Không tìm thay bằng train.json hoặc train_clean_qlora.json; không sửa file gốc.
6. Chia grouped train/dev; 16 nhóm câu hỏi trùng phải nằm trọn trong cùng một split.
7. Truy hồi corpus, tạo weak-label evidence và prompt Câu hỏi + Căn cứ pháp lý → Đáp án.
8. Giữ 39 answer trên 1.200 từ nhưng dùng huấn luyện có kiểm soát truncation.

Không sử dụng dữ liệu ngoài hoặc data augmentation.
