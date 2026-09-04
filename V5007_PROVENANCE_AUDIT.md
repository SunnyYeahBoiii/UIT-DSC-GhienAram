# Kiểm chứng provenance bản public METEOR 0.5007

Ngày kiểm chứng: 2026-09-04

## Kết luận

- Chuỗi artifact thực sự tạo ra submission đạt **METEOR 0.5007** đã được xác định bằng SHA-256.
- Nhánh `codex/v5007-baseline` là bản **đóng băng/tái dựng sau khi có điểm**, đủ gần để làm mốc tái lập về mặt hành vi, nhưng **không phải bản sao byte-for-byte của source Kaggle đã chạy**.
- Source hiện hành trong `F:\UIT-DSC-GhienAram\source` **không phải** baseline 0.5007. Nó đã đổi schema chunk, dense index, evidence view, generation, agentic search, repair và selector features.
- `F:\source_05_inference.zip` và `C:\Users\LOQ\Downloads\source_05.zip` giống nhau ở cả **92/92 file**, nhưng không phải đúng revision source đã thực thi lần inference cuối: runner trong ZIP không hiểu một số cờ xuất hiện trong log thật và config mặc định không khớp resolved config của recovery.

## Chuỗi artifact đã xác minh

### Submission và recovery

- Submission: `C:\Users\LOQ\Downloads\submission_05.zip`
- SHA-256 ZIP: `947B7163D3AA29F0797B622675F641B83C2D70CB9BB1927DBB48FD40E4C929DB`
- SHA-256 `submission.json` bên trong: `7B61761296B87E722720FB48894E17A0CCA14CCA7579D8D5110B92557715308E`
- Recovery sinh submission: `C:\Users\LOQ\Downloads\legalqa_recovery_latest (2).zip`
- SHA-256 recovery ZIP: `54C835450DD99B4AD075FA6EC65C9C09D3D8C5D3A17F06A0F82720C1C6297FA5`
- `predictions.json` trong recovery có cùng SHA-256 `7B617...08E` với `submission.json`; do đó đây là recovery chính xác đã tạo submission 0.5007.
- Recovery state: fingerprint `de25e0c7228918c3`, stage `complete`, `1000/1000`.

### Dữ liệu và corpus

- Public questions SHA-256: `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5`
- Official train SHA-256: `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988`
- Corpus SHA-256: `39e641056a2300ea493808b6be74f90c81ef367fb0733e415105d71d9d25a893`
- Bản 0.5007 dùng **official train gốc**, không dùng `train_clean.json` tạo về sau.

### Knowledge base và dense index

- Dense index thật: `C:\Users\LOQ\Downloads\results (6)\legalqa_dense_index`
- Schema: `global-dense-v1`
- Dense identity: `b5c6eadb46ee7aceab44b8643fa8a5beefde5420c02cf2146e0fd30e38eccf23`
- SQLite schema: `4`
- Documents: `8,512`
- Chunks: `215,267`
- Aliases: `12,114`
- Relations: `191,311`
- Vector dimension: `1,024`, dtype `float16`
- `manifest.json`: `6AB8AA06B521EFFA31224F52C49AB94ED171CB73D3ECFDE2D669FE45FAB8E522`
- `rowids.i64`: `839A12619DB41C1332F82A525ACAF5C46E04E467FB183D853EA1A66F82A4FD71`
- `vectors.f16`: `4D15E1EDCBD51DC3EFDC8FA4BD84C87EEB7A54AF8EE49C8821B572D578F901D7`

### QLoRA và model runtime

- Full adapter: `F:\results\legalqa_qlora\qwen-official-lora-full`
- Adapter identity: `3b3907596eeaaf1bbe95aebcd23c9ae4489a2fec45f464a2b25354d844dc2d03`
- Full adapter được train 1 epoch trên 7,000 record từ official train gốc với schema 4 / 215,267 chunks.
- Split adapter dùng cho calibration: `C:\Users\LOQ\Downloads\train_qlora_split\legalqa_qlora\qwen-official-lora-split`
- Split adapter identity: `490f6eaab37113964a89f17ebe87840b886641dfc1def537f35cd0750ab6fe61`
- Runtime models: Qwen3.5-2B, Vietnamese_Embedding, Vietnamese_Reranker và ViLegalBERT.

### Selector calibration

- Calibration thật: `F:\results (7)\legalqa_selector_calibration`
- `tuned_selection.yaml` SHA-256: `85A00E46E988B013741AB94C0B5BCF4777B6909AFF4D2B3B60D00664F06ADA38`
- Calibration fingerprint: `94af37c559fca58c`
- Grouped-dev: 400 câu
- METEOR probe được chọn: `0.5121896`
- Candidate oracle: `0.5973415`
- Holdout trước tune: `0.4980851`
- Holdout sau tune: `0.5446577`
- File `source\configs\selector_v5007.yaml` có toàn bộ trọng số vận hành giống hệt selector thật. Hash khác vì file local thêm comment và lược bỏ metadata báo cáo calibration.

### Cấu hình inference thật

- Source runtime được log là `/kaggle/input/datasets/akiravn/source-running-inference/source`.
- Dense runtime là `/kaggle/input/datasets/phcthin701/legalqa-dense-index/legalqa_dense_index`.
- Selector runtime là `/kaggle/input/datasets/akiravn/legalqa-selector-calibration/legalqa_selector_calibration/tuned_selection.yaml`.
- `max_context_chars=18000`
- `BM25_TOP_K=140`, `DENSE_TOP_K=70`, `FUSED_TOP_K=48`
- `RERANK_TOP_K=24`, `EVIDENCE_TOP_K=5`
- Train memory tắt.
- Kết quả chọn candidate: Qwen thường `959`, Qwen agentic `9`, extractive `32`.
- Router: simple `678`, complex `322`.
- Qwen bị truncation `285` câu.
- Agentic đủ điều kiện/sinh candidate `372/372`, nhưng selector chỉ chọn 9 câu agentic.
- Không có generation error.

## Đối chiếu bốn ZIP được tải lại ngày 2026-09-04

### `F:\legalqa_dense_index05.zip`

- Trùng tuyệt đối với `C:\Users\LOQ\Downloads\results (6)\legalqa_dense_index`.
- Cả ba file `manifest.json`, `rowids.i64` và `vectors.f16` đều trùng kích thước và SHA-256.
- Kết luận: **đúng dense index đã xác định cho baseline 0.5007**.

### `F:\05_selector_calibration.zip`

- Các file chung trùng tuyệt đối với `F:\results (7)\legalqa_selector_calibration`.
- Bản mới chỉ bung `selector_recovery_latest.zip` và `selector_dev_predictions.zip` thành thư mục; sau khi chuẩn hóa cách đóng gói, nội dung bên trong trùng **10/10 file**.
- `tuned_selection.yaml` và toàn bộ báo cáo calibration trùng.
- Kết luận: **đúng selector calibration đã dùng**.

### `F:\ALL_TRAINING_05.zip`

- `legalqa_qlora` trùng với `F:\results\legalqa_qlora`; khác biệt biểu kiến duy nhất là `qwen_qlora_adapter.zip` được bung thành 9 file. Sau khi chuẩn hóa cách đóng gói, 9/9 file trùng SHA-256.
- `qlora_recovery_latest` trùng 14/14 file với recovery ZIP đã tìm trước đó.
- `legalqa_source` trùng toàn bộ 89 source/artifact file với `F:\results\legalqa_source`; thư mục cũ chỉ có thêm 25 file `__pycache__` không ảnh hưởng source.
- Kết luận: **đúng training output/source đã xác định cho baseline 0.5007**.

### `F:\source_05_inference.zip`

- Trùng **92/92 file** với `C:\Users\LOQ\Downloads\source_05.zip` đã kiểm tra trước đây.
- Đây là source refactor package-layout (`core/`, `generation/`, `retrieval/`, `storage/`, `training/`) và thuộc chuỗi phát triển quanh bản 0.5007.
- Tuy nhiên, nó không thể là đúng revision chạy lệnh inference cuối đã sinh recovery `de25e0c7228918c3`, vì:
  - Log thật truyền `--max-runtime-hours 11.5`, `--auto-resume`, `--require-generator-adapter`, `--require-selection-weights` và `--require-prebuilt-indexes`; runner trong ZIP không định nghĩa bất kỳ cờ nào trong số này.
  - Recovery thật chứa `runtime.max_runtime_hours`, `runtime_started_unix`, `stop_at_unix` và khôi phục một recovery có stage `paused`; source ZIP không có `InferencePaused` hoặc runtime-budget checks.
  - Resolved config thật dùng `agentic_bm25_top_k=60`, `agentic_max_tool_calls=2`, `agentic_evidence_per_call=3`. Config trong ZIP dùng lần lượt `80`, `3`, `4`, đồng thời thêm `agentic_max_per_document`, `agentic_tool_context_share` và `no_repeat_ngram_size`; các khóa này không xuất hiện trong resolved config thật.
- Kết luận: **đây là một source bundle của nhánh 0.5, nhưng không phải source revision thực thi cuối cùng theo bằng chứng runtime**.

## So sánh các đối tượng dễ bị gọi nhầm

| Thành phần | Runtime/artifact thật | `codex/v5007-baseline` | `source_05_inference.zip` | Source hiện hành |
|---|---|---|---|---|
| Nguồn gốc | Chạy Kaggle tạo submission đã xác minh | Tái dựng/đóng băng sau khi có điểm | Bundle cùng họ 0.5 nhưng lệch revision runtime | Nhánh phát triển mới |
| SQLite | schema 4, 215,267 chunks | schema 4, 215,267 chunks | cấu hình schema 4 | schema 5, khoảng 659,538 chunks |
| Dense | global-dense-v1 | global-dense-v1 | global-dense-v1 | global-dense-v2 structured headings |
| Context thực chạy | 18,000 ký tự | 18,000 ký tự | notebook đặt 18,000; YAML mặc định 24,000 | 5,000 ký tự + schema alignment |
| Selector | YAML calibration attach ngoài | Trọng số vận hành giống hệt, file local có comment | Có thể attach đúng YAML | Có thể dùng calibration mới hoặc frozen selector tùy notebook |
| QLoRA data | Official train gốc | Trỏ về adapter gốc | Có thể trỏ adapter gốc | Có các adapter train-clean mới |
| Generation | Agentic 2 tool calls, 3 evidence/call | Khớp các thông số này; thêm guard/recovery thấy trong log | Mặc định 3 tool calls, 4 evidence/call và logic mới | Có repair, clipping, repeat stopping và thay đổi agentic |
| Quan hệ với 0.5007 | Chính xác | Mốc tái lập gần runtime nhất hiện có | Không phải exact final runtime revision | Không phải baseline |

## Mức độ chắc chắn

- **Chắc chắn tuyệt đối về output**: submission và recovery khớp SHA-256 từng byte.
- **Chắc chắn cao về model/data/index/config**: các identity và thống kê nằm trong manifest của chính recovery/calibration.
- **Source final vẫn chưa có bản byte-for-byte**: ZIP mới bổ sung thêm bằng chứng source, nhưng chính nó mâu thuẫn với command line và resolved config của lần chạy. `codex/v5007-baseline` hiện gần hành vi runtime hơn vì có đúng các guard/cờ được log, song vẫn là bản tái dựng hậu nghiệm.

## Quy ước từ nay

- Gọi chuỗi artifact ở trên là **`v5007-observed`** hoặc **baseline thực nghiệm 0.5007**.
- Gọi nhánh `codex/v5007-baseline` là **`v5007-reconstructed`**, không gọi là source gốc byte-for-byte.
- Có thể dùng ba artifact tải lại (training, dense, selector) làm bản chuẩn. Với `source_05_inference.zip`, chỉ coi là source tham chiếu cùng chuỗi phát triển, không coi là exact final runtime revision.
- Khi so sánh cải tiến mới, cố định official corpus/train hash, dense identity, adapter identity, selector hash và cấu hình inference; chỉ thay đúng một nhóm biến mỗi lần.
