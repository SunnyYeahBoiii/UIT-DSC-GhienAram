# Kaggle T4 — full sequential inference-only

Bản này **không huấn luyện hoặc fine-tune bất kỳ mô hình nào**. `train.json` không được đọc, weak labels không được tạo và không có Qwen LoRA adapter. Pipeline v2 sử dụng bốn checkpoint đã được BTC duyệt; BARTpho đã bị loại sau khi artifact public cho thấy output hỏng. Xem `PIPELINE_V2.md` để chạy global dense index theo hai session và hiệu chỉnh selector bằng train chính thức.

```text
public question
  → global chunk SQLite FTS5 + global dense index
  → typed citation boost + RRF
  → unload + CUDA cache cleanup
  → Vietnamese_Reranker (cuda:0, FP16, batch 1, length 512)
  → Simple/Complex router + bounded legal relation graph expansion
  → unload + CUDA cache cleanup
  → evidence-first Qwen3.5-2B base (cuda:0, FP16, deterministic)
  → unload + CUDA cache cleanup
  → ViLegalBERT base (cuda:0, FP16, batch 4)
  → unload + CUDA cache cleanup
  → candidate selector → submission.zip
```

## 1. Kaggle inputs

Attach đúng ba Dataset:

```text
dsc2026-task2-data/
  public-official.json
  selected-contexts/context_*.json

legalqa-p0-models/
  Qwen3.5-2B/
  Vietnamese_Embedding/
  Vietnamese_Reranker/
  ViLegalBERT/

legalqa-inference-source/
  configs/
  kaggle/
  scripts/
  src/
  wheelhouse/
  pyproject.toml
```

`train.json` có thể vẫn nằm trong Dataset chính thức nhưng runner inference-only hoàn toàn không sử dụng nó.

## 2. Notebook

Import `kaggle/legalqa_dsc2026_inference_only_t4.ipynb`, bật GPU và tắt Internet. Cấu hình an toàn mặc định:

```python
DEVICE = 'cuda:0'
EMBEDDING_BATCH_SIZE = 2
RERANKER_BATCH_SIZE = 1
VERIFIER_BATCH_SIZE = 4
RERANKER_MAX_LENGTH = 512
MAX_CONTEXT_CHARS = 18000
MAX_NEW_TOKENS = 900
BM25_TOP_K = 140
DENSE_TOP_K = 70
FUSED_TOP_K = 48
RERANK_TOP_K = 24
MODEL_SMOKE = True
DENSE_INDEX_ONLY = False
CHECKPOINT_EVERY = 50
AUTO_RESUME = True
RECOVERY_OUTPUT = '/kaggle/working/legalqa_recovery_latest.zip'
```

Kaggle có 2 T4 nhưng chế độ ổn định này chủ động dùng một GPU. VRAM không được cộng tự động; tải model trên một GPU rồi giải phóng trước model tiếp theo tránh DDP/FSDP và giảm rủi ro phiên bản. GPU thứ hai không được dùng trong bản này.

`MODEL_SMOKE = True` tải tuần tự và chạy một input rất nhỏ qua bốn wrapper runtime trước khi xây retrieval dài. Việc này phát hiện sớm lỗi tokenizer, kiến trúc, CUDA, dtype và API sinh văn bản.

## 3. Nếu OOM

Không tăng batch. Thử lần lượt:

```python
EMBEDDING_BATCH_SIZE = 1
VERIFIER_BATCH_SIZE = 2
MAX_CONTEXT_CHARS = 9000
MAX_NEW_TOKENS = 450
```

Reranker đã ở batch 1. Nếu vẫn OOM tại reranker, giảm:

```python
RERANKER_MAX_LENGTH = 384
```

Đổi các tham số sẽ tạo fingerprint/cache mới, nên không dùng nhầm candidate của cấu hình trước.

## 4. Checkpoint và chạy tiếp qua session

- SQLite được tái sử dụng nếu corpus fingerprint không đổi.
- Retrieval cache được tái sử dụng khi đủ đúng 1.000 ID.
- Qwen ghi từng candidate cùng token count/EOS/truncation ngay sau khi sinh.
- Cache nằm trong `/kaggle/working/legalqa_inference` và có fingerprint theo model/runtime/retrieval/generation.

- Ngay sau retrieval, sau mỗi `CHECKPOINT_EVERY` câu Qwen và sau mỗi stage, runner ghi nguyên tử:

```text
/kaggle/working/legalqa_recovery_latest.zip
```

ZIP này chứa retrieval cache v3, Qwen candidate cache, cấu hình và trạng thái. Nó không chứa corpus SQLite, global dense matrix hay bốn model. Hãy lưu `/kaggle/working/legalqa_dense_index` thành Dataset riêng sau session `DENSE_INDEX_ONLY=True`.

Để tiếp tục ở session kế tiếp:

1. Sau session hiện tại, lưu `legalqa_recovery_latest.zip` bằng **Save Version / Save Output**, hoặc tải nó xuống rồi tạo một Kaggle Dataset mới.
2. Attach Dataset recovery đó cùng data, model và source vào notebook mới.
3. Giữ `AUTO_RESUME = True`. Notebook tự tìm đúng một file mang tên `legalqa_recovery_latest.zip` và truyền `--resume-bundle`.
4. Runner kiểm tra fingerprint. Nếu model hay tham số inference khác phiên trước, nó dừng thay vì âm thầm dùng nhầm cache.

Fingerprint v3 băm nội dung public/corpus, cấu hình runtime và định danh bốn checkpoint. Recovery ZIP của pipeline v2 cũ không tương thích và sẽ bị từ chối có chủ đích.

Log khôi phục đúng có dạng:

```text
[Recovery] restored ... stage=qwen completed=350/1000
```

Trong cùng một session, nếu một cell bị ngắt, chạy lại mà không bật `FORCE_INDEX`; cache trong working được dùng tiếp. Qua session mới, phải attach recovery ZIP vì `/kaggle/working` không phải persistent input. Không attach hai recovery ZIP cùng lúc.

Lưu ý: code không thể tự ghi vào `/kaggle/input` vì đây là vùng chỉ đọc. Nếu session bị hủy mà bạn chưa Save Output hoặc tải recovery ZIP ra ngoài, Kaggle có thể xóa working disk; không có đoạn Python nào khôi phục được dữ liệu đã bị nền tảng xóa.

Runner còn tự phục hồi SQLite/candidate/retrieval cache bị cắt ở lần ghi cuối. Lỗi CUDA/OOM làm stage dừng ngay và ghi recovery thay vì lặp lỗi trên toàn bộ 1.000 câu.

## 5. Kết quả

```text
/kaggle/working/submission.zip
/kaggle/working/legalqa_recovery_latest.zip
/kaggle/working/legalqa_inference/run_manifest.json
/kaggle/working/legalqa_inference/preflight.json
/kaggle/working/legalqa_inference/predictions.json
/kaggle/working/legalqa_inference/predictions.trace.jsonl
```

Manifest phải có:

```json
{
  "mode": "full_sequential_inference_only",
  "training_performed": false
}
```

Chỉ nộp `submission.zip`.
