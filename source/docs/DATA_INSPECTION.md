# Kiểm tra dữ liệu chính thức Task 2

Nguồn được kiểm tra trực tiếp ngày 06/08/2026:

- `C:\Users\LOQ\Downloads\train.json`
- `C:\Users\LOQ\Downloads\public-official.json`
- `C:\Users\LOQ\Downloads\selected-contexts`
- `C:\Users\LOQ\Downloads\DSC2026_Task2_LegalQA_Data_Overview.docx`

## Kết quả cấu trúc

| Thành phần | Kết quả |
|---|---:|
| Train | 7.000 Q/A; keyset duy nhất `answer|question` |
| Public | 1.000 câu; keyset `answer|question`; toàn bộ answer là `null` |
| Corpus files | 8.532 |
| Corpus có passage | 8.512 |
| Corpus passage rỗng | 20 |
| Corpus dung lượng | 488.693.666 byte |
| Schema không có name | 1.125 file (`id|link|passage`) |
| Schema có name | 7.407 file (`id|link|name|passage`) |

Train không chứa context ID/ground-truth document. Vì vậy retrieval supervision phải được tạo tự động bằng answer-to-corpus alignment, không gán nhãn tay.

## Benchmark code trên corpus đầy đủ

- SQLite schema v2: 8.512 documents, 208.649 legal chunks.
- Dung lượng index: khoảng 794,7 MiB.
- Thời gian build trên máy kiểm tra: 393,4 giây; Kaggle có thể khác.
- Weak alignment 50 câu đầu: mean 0,6520; median 0,6594; P10 0,4078; min 0,2946; max 0,9859.
- 6/6 unit/integration tests pass.

Điểm alignment là overlap nội bộ giữa answer và positive evidence, không phải METEOR/leaderboard. Kết quả này đủ tốt để dùng làm weak supervision, nhưng trace vẫn cần được theo dõi vì một đáp án có thể dựa trên nhiều văn bản sửa đổi/bổ sung.

## Độ dài

### Question train (số từ)

- Mean: 19,69
- Median: 19
- P90: 28
- P99: 36
- Max: 60

### Answer train (số từ)

- Mean: 347,38
- Median: 312
- P90: 576
- P99: 1.024
- Max: 2.435

Điều này xác nhận output là long-form reference-style QA. Generator và metric tuning không nên mặc định câu trả lời ngắn 50–100 từ.

## Hash để kiểm tra dữ liệu Kaggle upload

```text
train.json SHA256:
2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988

public-official.json SHA256:
5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5

corpus aggregate SHA256 (sorted filename + per-file SHA256):
c172797243f7d7c8979f55b1d07f04f80ab9de647ce918bce8294d0bf5d89c2e
```

Chạy lại bằng:

```powershell
python kaggle/verify_inputs.py `
  --train "C:\Users\LOQ\Downloads\train.json" `
  --public "C:\Users\LOQ\Downloads\public-official.json" `
  --corpus "C:\Users\LOQ\Downloads\selected-contexts"
```

## Nội dung tài liệu DOCX đã đối chiếu

Tài liệu xác nhận:

- Input là câu hỏi pháp luật tiếng Việt; output là câu trả lời tự nhiên bằng văn xuôi.
- Bộ dữ liệu gồm train, warm-up, public, private và selected contexts.
- Corpus record dùng `id`, `name`, `link`, `passage`; `name` thực tế không xuất hiện ở mọi file.
- METEOR là độ đo chính để xếp hạng; ROUGE-L là độ đo phụ.
- Câu trả lời tham chiếu giữ nhiều đoạn trích điều khoản, danh sách và cả quy định trước đây, nên cần ưu tiên evidence coverage và thứ tự nội dung.

LibreOffice không có trong môi trường hiện tại nên renderer DOCX chuẩn của skill không tạo được ảnh trang. Nội dung và bảng đã được kiểm tra cấu trúc bằng `python-docx`; file nguồn không bị chỉnh sửa.
