# V8 public trace audit dẫn đến v9.1

Nguồn audit: submission 0,3841 và recovery hoàn chỉnh gồm 1.000 predictions, 1.000 Qwen cache rows, 1.000 retrieval rows và 1.000 trace rows. Mọi ID khớp, không có generation error.

## Điểm giữ lại

- 653 Qwen và 347 extractive answers được chọn.
- Selector loại 319/320 Qwen outputs bị truncate; chỉ một output truncate lọt vào submission.
- 391 câu Simple, trong đó 329 câu chọn Qwen.
- Citation rate của output gần official train.
- Global BM25/dense/reranker chạy ổn định, parameter budget 3,544B.

## Lỗi được sửa

- 609/1.000 câu bị route Complex, cao hơn nhiều so với ước lượng nghiệp vụ khoảng 20%.
- 592 câu chạy đủ ba graph iterations.
- 544 câu nhận graph evidence; 2.177/2.265 final graph chunks có reranker dưới 0,1.
- Complex evidence thường bị ép thành tám document khác nhau.
- Extractive output có median 469 từ và 78 answers bắt đầu bằng chữ thường.
- Official train có median 314 từ và 82,5% answers mở đầu bằng “Theo”/“Căn cứ”, trong khi v8 public median 396 từ và chỉ 12,5% có kiểu mở đầu đó.

## Kiểm tra replay

Áp dụng rule router v9.1 lên các route features đã log của v8:

- Complex dự kiến: 322/1.000 (32,2%).
- Trong 1.098 graph chunks cũ thuộc các câu này, 29 chunk vượt adaptive graph gate (2,6%).
- 29 chunk gồm 17 `REPLACES`, 11 `AMENDS`, 1 `REFERENCES`.

Đây là replay chẩn đoán trên candidate cũ, không phải phép đo METEOR và không thay thế grouped-dev/Codabench.

## Mapping sang code

- `src/legalqa/routing.py`: Complex opt-in, relation count không còn là trigger độc lập.
- `src/legalqa/sqlite_retrieval.py`: rerank/gate từng graph hop, early stop, accepted-only frontier.
- `src/legalqa/retrieval.py`: score-aware diversity, tối đa bốn document.
- `src/legalqa/generation.py`: sửa extractive opening; dedup Qwen nhưng không hard-trim mặc định.
- `configs/kaggle_inference_t4.yaml`: hai graph hops, sáu evidence, ceiling 700.
- `scripts/evaluate_selector.py`: tuning YAML v9.1 có schema riêng; weights cũ bị từ chối.
