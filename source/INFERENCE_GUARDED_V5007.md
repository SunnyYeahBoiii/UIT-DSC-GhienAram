# Inference guarded-v5007

Đây là nhánh nâng cấp có kiểm soát từ submission METEOR `0.5007`. Adapter QLoRA
và file `configs/selector_v5007.yaml` vẫn là mốc so sánh; source mới thêm kiểm
tra căn cứ và một candidate repair có giới hạn.

## Thay đổi runtime

1. Reranker chia chunk dài thành tối đa 3 cửa sổ token, giữ nguyên câu hỏi và
   lấy cửa sổ phù hợp nhất.
2. Chunker nhận diện thêm tiêu đề kỹ thuật như `II. CHỈ ĐỊNH` và
   `1. Người thực hiện`.
3. Context Qwen được đóng gói theo tokenizer thật, cắt ở biên mệnh đề; header
   hiển thị số văn bản từ title thay vì gọi `doc_id` nội bộ là số văn bản.
4. Candidate được kiểm tra: caption/slug/question echo, vòng lặp, câu kết dở,
   các ý hỏi, đúng đối tượng, số văn bản, con số và quan hệ tối đa/tối thiểu.
5. Qwen chỉ repair tối đa 220 câu có lỗi nghiêm trọng, một lượt/câu. Candidate
   repair bị phạt mạnh nếu không giảm đủ quality risk so với Qwen gốc.
6. Khi router mở nhánh Agentic, hệ thống bắt buộc thực hiện ít nhất một lượt
   tìm kiếm CPU thật; truy vấn lặp bị chặn và lỗi tool không làm hỏng cả run.
7. Extractive ưu tiên một văn bản cho Simple, tối đa ba văn bản cho Complex và
   không lấy câu có lexical overlap dưới ngưỡng.
8. Selector calibration đọc thêm các feature mới nhưng public vẫn có thể dùng
   frozen selector v5007; các trọng số guard nằm trong config runtime.

## Tương thích cache/index

- Cache schema: `inference-v8-quality-repair-windowed-rerank`.
- SQLite schema: `5`.
- Dense schema: `global-dense-v2-structured-headings`.

Do chunking thay đổi, **dense index cũ không tương thích** và runner sẽ từ chối
trước khi load model. Chạy notebook inference với `DENSE_INDEX_ONLY=True`, tạo
Dataset từ `legalqa_dense_index`, rồi attach index mới cho public inference.
Recovery/candidate cache v7 không được resume vào run v8.

## Trình tự kiểm chứng

1. Chạy grouped-dev với adapter split, lưu candidate oracle và từng feature.
2. So sánh frozen v5007 và guarded-v5007; kiểm tra riêng nhóm loop, citation,
   numeric relation, multi-part, form và low-retrieval.
3. Chỉ dùng selector mới nếu METEOR probe holdout không giảm; probe local không
   được coi là giống tuyệt đối scorer của BTC.
4. Public run giữ `MAX_RUNTIME_HOURS=11.5`; recovery lưu riêng các stage
   `qwen`, `qwen_repair`, `qwen_agentic` và selection.
