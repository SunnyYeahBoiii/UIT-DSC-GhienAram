# Kiểm tra checkpoint P0 trước khi chạy Kaggle

Ngày kiểm tra: 2026-08-06  
Nguồn checkpoint local: `F:\legalqa-p0-models`

## Tình trạng file

| Vai trò | Thư mục | Cấu hình | Trọng số | Tokenizer | Kết quả |
|---|---|---:|---:|---:|---|
| Generator | `Qwen3.5-2B` | Có | Safetensors | Có | Đạt |
| Embedding | `Vietnamese_Embedding` | Có | Safetensors | Có | Đạt |
| Reranker | `Vietnamese_Reranker` | Có | Safetensors | Có | Đạt |
| Candidate generator cũ | `bartpho-syllable` | Có | PyTorch BIN | Có | Không dùng trong v9 |
| Verifier | `ViLegalBERT` | Có | Safetensors | Có | Đạt |

Không có checkpoint nào là Git LFS pointer tải dở. Mỗi thư mục có `config.json`, tokenizer và file trọng số có kích thước hợp lý.

## Kiểm tra ngân sách tham số

Đếm trực tiếp số phần tử tensor trong các checkpoint; với BARTpho, ba tên `shared.weight`, `encoder.embed_tokens.weight` và `decoder.embed_tokens.weight` dùng chung một storage nên chỉ tính một lần.

| Thành phần | Số tham số/tensor elements bảo thủ |
|---|---:|
| Qwen3.5-2B (gồm vision tower trong checkpoint) | 2.274.069.824 |
| Vietnamese_Embedding | 567.754.752 |
| Vietnamese_Reranker | 567.755.777 |
| BARTpho-syllable | 395.814.912 — không tải/không dùng trong v9 |
| ViLegalBERT, tính cả tensor có trong checkpoint | 135.063.809 |
| Tổng bốn checkpoint runtime v9 | 3.544.644.162 |
| Qwen LoRA rank 8 dự kiến | 8.778.240 |
| Tổng v9 dự kiến sau LoRA | **3.553.422.402** |
| Khoảng trống tới 4B | **446.577.598** |

Kết quả này là phép kiểm tra bảo thủ ở mức tensor. BARTpho bị loại hoàn toàn khỏi config/runtime v9. Pipeline Kaggle vẫn phải đếm lại bốn kiến trúc đã đăng ký và adapter thực tế; bài nộp bị chặn nếu tổng không còn nhỏ hơn `4.000.000.000`.

## Dung lượng không cần thiết khi upload

Snapshot local có một số định dạng dự phòng không được pipeline PyTorch sử dụng:

- `bartpho-syllable/tf_model.h5`: khoảng 1,51 GiB.
- `Vietnamese_Embedding/onnx/`: khoảng 2,18 GiB.
- Các thư mục `.cache/huggingface`: metadata tải xuống, không cần cho inference offline.

Có thể loại các mục này khỏi **bản staging upload**, nhưng nên giữ nguyên snapshot gốc trên máy. Không được loại `config.json`, tokenizer, custom tokenizer code, `model.safetensors`, `pytorch_model.bin`, `modules.json` hoặc `1_Pooling/config.json`.

## Trạng thái dữ liệu chính thức

Tại thời điểm kiểm tra, ba đường dẫn ban đầu sau không còn tồn tại:

- `C:\Users\LOQ\Downloads\train.json`
- `C:\Users\LOQ\Downloads\public-official.json`
- `C:\Users\LOQ\Downloads\selected-contexts`

Cần tìm lại hoặc khôi phục các file này trước khi tạo Kaggle Dataset dữ liệu. Không thể train/sinh submission chỉ với checkpoint model.
