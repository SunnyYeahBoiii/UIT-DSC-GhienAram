# LegalQA v9.1 — graph-gated retrieval, train memory và evidence-grounded QLoRA

## 0. Sửa lỗi từ public trace v8

Full trace v8 cho thấy 609/1.000 câu bị route Complex, 592 câu chạy đủ ba vòng và 2.177/2.265 graph chunks cuối có reranker dưới 0,1. V9.1 thay đổi nguyên tắc:

- relation count không tự kích hoạt Complex vì phần lớn văn bản pháp luật đều có quan hệ;
- Complex chỉ được mở bởi complex cue/subquestion, confidence dưới 0,30 hoặc cross-document tie rất sát;
- mỗi vòng graph được cross-encoder rerank trước, threshold bằng `max(0.12, 0.30 × top-initial-reranker)`;
- câu hỏi sửa đổi/thay thế/hiện hành được discount threshold còn 60%, nhưng không bỏ relevance gate;
- chỉ accepted document được làm frontier vòng sau; không accepted thì dừng `no_relevant_relation`;
- tối đa hai vòng, ba document mới mỗi vòng, sáu evidence cuối và tối đa bốn document;
- diversity chỉ cộng bonus 0,04 và loại candidate dưới 42% score của top evidence.

Static replay trên trace v8 dự báo Complex giảm từ 60,9% xuống khoảng 32,2%. Trong các graph chunks cũ thuộc nhóm này, chỉ 2,6% vượt relevance gate mới; đây là phép replay chẩn đoán, không phải điểm Codabench.

## 1. Answer-style profile

`scripts/analyze_train_answers.py` và `src/legalqa/answer_style.py` thống kê official answers theo toàn tập và theo `question_type`:

- p10/p25/p50/p75/p90/p95 số từ;
- phân vị số câu;
- tỷ lệ citation, số, liệt kê;
- kiểu mở đầu thường gặp.

Nếu một nhóm có ít hơn 20 mẫu, hệ thống dùng phân phối toàn tập. Simple lấy p50 làm mục tiêu và p90 làm trần; Complex lấy p75 và p95. Style ceiling là 700 từ, gần p95 official train 691. Qwen output được xóa câu lặp; hard trim mặc định tắt để không làm mất reference dài. Trace ghi `raw_words`, `compacted_words`, `duplicate_sentences_removed`, EOS và truncation.

## 2. Official train-question memory

`src/legalqa/train_memory.py` tạo exact cosine index float16 bằng cùng Vietnamese embedding checkpoint đang dùng cho corpus. Index có fingerprint theo SHA-256 của train và embedding checkpoint, ghi state định kỳ và có thể tiếp tục trong cùng working directory.

Mỗi public query lấy top 4 train questions. Hai cách dùng được tách rõ:

1. **Few-shot style:** tối đa hai hàng có similarity >= 0.70 được chèn vào prompt; prompt ghi rõ đây không phải căn cứ và cấm sao chép chi tiết không có trong evidence hiện tại.
2. **Direct candidate:** chỉ tạo ở similarity >= 0.86 mặc định. Selector calibration có thể nâng ngưỡng; không hạ dưới ngưỡng tạo candidate.

Khi đánh giá grouped-dev, memory chỉ được xây từ `selector_train.json`; ID dev không đi vào memory.

## 3. Grouped-dev và candidate oracle

`scripts/prepare_selector_dev.py --dev-size 400` nhóm câu theo loại câu hỏi và 18 content tokens đầu sau chuẩn hóa số. Một nhóm không bị tách giữa train/dev. Trong 400 câu dev, `scripts/evaluate_selector.py` lại giữ template group nguyên vẹn khi chia 80% tune và 20% untouched holdout.

Báo cáo gồm:

- điểm candidate đã chọn;
- điểm từng source;
- candidate oracle — chọn candidate tốt nhất bằng reference cho từng câu;
- selector sau tuning trên tune;
- kết quả cuối trên untouched holdout;
- sweep ngưỡng train-memory.

`meteor_exact_probe` chỉ là phép đo exact-token cục bộ. Điểm Codabench mới là thước đo chính thức. Oracle là trần của candidate pool, không phải cấu hình có thể dùng trực tiếp khi public không có reference.

## 4. QLoRA bằng official train + retrieved evidence

`kaggle/run_qwen_qlora.py`:

1. tạo SQLite legal corpus;
2. truy hồi weak positive evidence bằng question + citation/numeric terms trong official answer;
3. chọn tối đa ba supporting documents có alignment;
4. train Qwen3.5-2B 4-bit NF4, LoRA r=8 mặc định;
5. chỉ tính loss trên assistant answer;
6. lưu checkpoint model, optimizer, scheduler, callback và RNG mỗi 20 optimizer steps;
7. kiểm tra SHA-256 dữ liệu/pairs và toàn bộ hyperparameter trước khi resume.

Hai scope không được trộn:

- `split`: loại grouped-dev khỏi train, dùng để đo candidate oracle và selector trung thực.
- `full`: train trên toàn bộ official train để tạo adapter cuối cho public inference.

Mỗi state có `qlora_scope.json`; runner từ chối resume nếu scope không khớp. Nếu Kaggle gần hết giờ, Save Version toàn bộ `/kaggle/working/legalqa_qlora`, biến output đó thành Dataset, attach vào session mới và giữ nguyên hyperparameter.

## 5. Inference và cache identity

Inference fingerprint v5 chứa hash question file, train file, corpus, bốn base checkpoint, Qwen adapter và toàn bộ retrieval/generation/selection config. Vì vậy:

- adapter khác không dùng nhầm Qwen cache;
- selector train khác không dùng nhầm train-memory;
- recovery cũ không được dùng âm thầm;
- thay đổi threshold/length plan tạo namespace cache mới.

Runtime tuần tự: embedding → unload → reranker → unload → Qwen → unload → ViLegalBERT → unload. BARTpho không được tải.

## 6. Quyết định dựa trên oracle

So sánh ba khoảng cách:

- **oracle − selected lớn:** candidate pool đã có đáp án tốt; ưu tiên sửa selector/threshold.
- **oracle thấp:** ưu tiên retrieval, evidence construction hoặc QLoRA.
- **QLoRA source tốt hơn base nhưng selected không tăng:** sửa source prior/penalty trên tune và xác nhận ở holdout.

Chỉ mang `tuned_selection.yaml` sang public khi holdout cải thiện. Sau đó train adapter `full` riêng và giữ nguyên retrieval/selector đã chọn.

V9.1 yêu cầu tuning YAML có `schema: selector-weights-v2-graph-gate`. Trọng số v9 cũ bị từ chối vì graph gate đã thay đổi phân phối evidence và candidate score; phải chạy lại grouped calibration.
