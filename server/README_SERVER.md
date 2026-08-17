# Chạy LegalQA trên GPU server có persistent volume

## Bố trí dữ liệu

Mount persistent volume rồi đặt dữ liệu/checkpoint vào cấu trúc sau. Ví dụ volume là `/workspace/legalqa`:

```text
/workspace/legalqa/
  input/
    dsc2026-task2-data/
      train.json
      public-official.json
      selected-contexts/context_*.json
    legalqa-p0-models/
      Qwen3.5-2B/
      Vietnamese_Embedding/
      Vietnamese_Reranker/
      bartpho-syllable/
      ViLegalBERT/
```

Source có thể nằm ngoài volume, nhưng `input`, `legalqa_run`, model đã train, cache và `submission.zip` đều nằm trên volume.

## Lệnh full khuyến nghị cho GPU 40–48 GB

Chạy trong `tmux` hoặc `screen`:

```bash
python server/run_server.py \
  --persistent-root /workspace/legalqa \
  --profile full \
  --train-qwen \
  --train-reranker \
  --resume-training \
  --reranker-batch-size 2 \
  --reranker-grad-accum 8 \
  --qwen-checkpoint-steps 20 \
  --reranker-checkpoint-steps 250
```

Nếu reranker vẫn OOM, chạy lại đúng lệnh nhưng đổi thành:

```text
--reranker-batch-size 1 --reranker-grad-accum 16
```

Không thêm `--force-index` hoặc `--force-train` khi chỉ muốn tiếp tục. Runner sẽ tái sử dụng SQLite, weak labels, checkpoint model và candidate cache hiện có.

## Khôi phục sau khi server bị dừng

Mount lại đúng volume, đặt source vào máy mới rồi chạy lại cùng lệnh. Mặc định `--resume-training` sẽ:

- tìm `legalqa_run/models/Vietnamese_Reranker-official/checkpoint-*` mới nhất;
- khôi phục reranker model, optimizer, scheduler, scaler và RNG;
- tìm `legalqa_run/models/qwen-official-lora/checkpoint-*` mới nhất;
- để Hugging Face Trainer khôi phục Qwen adapter, optimizer, scheduler và RNG;
- dùng lại SQLite, retrieval cache và từng candidate Qwen/BARTpho đã sinh.

Mỗi loại checkpoint được giữ tối đa 2 bản. Qwen lưu mỗi 20 optimizer steps. Reranker full-state có thể chiếm khoảng 6–7 GB cho một checkpoint nên mặc định lưu mỗi 250 optimizer steps để tránh I/O quá lớn; khi bị ngắt, quá trình tiếp tục từ checkpoint hoàn chỉnh gần nhất.

## Kết quả trên volume

```text
/workspace/legalqa/
  submission.zip
  legalqa_run/
    official_corpus.sqlite
    preflight.json
    run_manifest.json
    predictions.json
    predictions.trace.jsonl
    models/
      Vietnamese_Reranker-official/
      qwen-official-lora/
```

Chỉ xóa Pod sau khi đã xác nhận `submission.zip`, adapter, reranker và `run_manifest.json` nằm trên persistent volume và đã tải thêm một bản sao về máy cá nhân.
