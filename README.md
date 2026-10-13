# UIT DSC 2026 Task 2 — LegalQA v10

Giấy phép: MIT (`LICENSE`). Docker: `docker/README.md`.

Pipeline offline cho Kaggle T4: hỏi đáp pháp luật tiếng Việt dạng dài. Metric chính
METEOR, phụ ROUGE-L. Tổng tham số runtime < 4B.

Repo **không chứa** dữ liệu cuộc thi, checkpoint, dense index, adapter QLoRA hay
submission — xem `.gitignore`.

v9.2 nằm ở nhánh `main`. v10 xây lại từ số 0 sau khi EDA chỉ ra lỗi nền móng.

## Vì sao xây lại

v9.2 index **cả Document**. Document p50 = 4.828 từ (max 1.242.409), nên chuẩn hoá độ
dài của BM25 pha loãng câu hỏi 19 từ. Đo recall trên 224 câu weak-gold:

| Đơn vị index | R@1 | R@5 | R@10 | R@100 |
|---|---:|---:|---:|---:|
| Document (v9.2) | 22,3% | 42,0% | 50,9% | 76,3% |
| Chunk 256/256 (v10) | **55,4%** | **83,0%** | **89,7%** | **97,8%** |

Bốn tầng phía sau của v9.2 (graph, agentic, memory, selector) đều ngồi sau trần cứng
50,9% đó. Sửa đơn vị index đáng giá hơn mọi tầng cộng lại.

Chi tiết và các số đo khác: `docs/EDA_V10.md`.

## Thiết kế

**Small-to-big.** Chunk 256 từ là đơn vị **tìm**; Evidence — mở rộng ra trọn các Điều mà
Chunk chạm vào — là đơn vị **sinh**. Lý do ở `docs/adr/0001`.

**Giữ legal graph, bỏ phần còn lại.** Agentic tool-call, multi-candidate selector,
train-memory đều bị loại; graph chỉ mở rộng khi Citation gọi tên một văn bản cụ thể,
không gate theo số quan hệ (93,1% Document chứa "căn cứ" nên đếm quan hệ là vô nghĩa).
Lý do ở `docs/adr/0002`.

**Làm sạch dữ liệu trước.** Passage crawl từ HTML mang wrap mềm `\r\n\n` — 100% Document
có, 93,9% nối giữa câu. Không gỡ thì 62,3% tiêu đề Điều bị cắt đôi. Gỡ xong còn 0,2%.

## Chạy

```bash
PYTHONPATH=src python -m pytest -q
```

```python
from pathlib import Path
from legalqa.corpus import iter_documents
from legalqa.chunking import chunk_document
from legalqa.evidence import expand_to_evidence

docs = list(iter_documents(Path("data/official/selected-contexts")))
chunks = [c for d in docs for c in chunk_document(d)]      # 287.741 chunk
ev = expand_to_evidence(docs[0], chunks[0])                 # Evidence trọn Điều
```

EDA tái lập được bằng `scripts/eda/` (7 script, đọc `data/official/`).

## Tái lập kết quả Private Test (bản nộp BTC, LB 0,6520)

Theo yêu cầu BTC (email 28/9/2026, hạn nộp 30/9/2026): mã nguồn phải tái lập được kết
quả Private Test, kèm README chi tiết (môi trường, thư viện, cài đặt, cấu hình, dữ
liệu, cách chạy) và thông tin checkpoint/tài nguyên ngoài. Mục này tóm tắt; chi tiết
đầy đủ nằm ở `README.md` ở gốc gói `repro-row1/` (không commit vào Git vì chứa
checkpoint — xem "Checkpoint và tài nguyên ngoài" bên dưới).

### 1. Môi trường

- Python 3.12, CUDA 12.4 (sinh candidate cần GPU; bước MBR không cần GPU).
- Image tái lập: `Dockerfile` (base `python:3.12-slim`, linux/amd64), chạy offline hoàn
  toàn (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `NLTK_DATA=/app/nltk_data`). Image
  đã build sẵn: `repro-row1/docker/legalqa-row1_amd64.tar.gz`.

### 2. Thư viện

`requirements-lock.txt`: 68 gói ghim version + sha256, gồm `torch==2.6.0` (cu124),
`transformers==5.14.1`, `tokenizers==0.22.2` (0.23.x làm rust tokenizer panic),
`peft==0.21.0` (bản lưu adapter), `nltk`, `rouge-score`, `tiktoken`. Wheel linux x86_64 /
CPython 3.12 đóng gói sẵn ở `wheels/`. Cài offline không cần Docker:

```bash
pip install --no-index --find-links wheels --require-hashes -r requirements-lock.txt
export PYTHONPATH=$PWD/src:$PWD/scripts NLTK_DATA=$PWD/nltk_data HF_HUB_OFFLINE=1
```

### 3. Cài đặt và cấu hình

```bash
docker load < repro-row1/docker/legalqa-row1_amd64.tar.gz                  # dùng image có sẵn
# hoặc build lại, cũng không cần mạng:
docker build --network none --platform linux/amd64 -t legalqa:row1 .
```

Mã nguồn gói này là commit `5b13ea7` (`COMMIT.txt`) — commit đã sinh prompt/candidate của
row1. HEAD sau đó đổi retrieval (RRF k=20, `rank_for_evidence`) nên không dựng lại đúng
prompt row1.

Cấu hình bản nộp tốt nhất (`row1`): flat MBR trên 6 ứng viên = `b80` rotation `r0` +
`b320` rotation `r0` + `b920` rotation `r0`–`r3`. Rotation là hoán vị vòng của thứ tự 5
Evidence trong prompt (`r0` = thứ tự theo điểm rerank, không thêm tham số). Retrieval:
RRF (k=60) giữa BM25 + `multilingual-e5-small` + `Vietnamese_Embedding`, pool 50, rerank
bằng `rerank_v11`, lấy top-5 Evidence.

### 4. Dữ liệu và checkpoint (tài nguyên ngoài)

Repo không chứa dữ liệu cuộc thi, checkpoint, dense index (`.gitignore`) — theo ràng buộc
đề bài (không nộp lại dữ liệu BTC đã cấp). Gói `repro-row1/` mang mọi thứ tái lập cần:

| Tài nguyên | Vai trò | Nguồn |
|---|---|---|
| `Qwen/Qwen3.5-2B` | model nền LoRA | HuggingFace (Apache-2.0), Kaggle mirror `sunny070306/qwen3-5-2b-base` |
| `intfloat/multilingual-e5-small` | dense retrieval | HuggingFace (MIT) |
| `AITeamVN/Vietnamese_Embedding`, `AITeamVN/Vietnamese_Reranker` | dense retrieval, base reranker | HuggingFace (Apache-2.0) |
| `runs/sft_v11/{b80,b320,b920}/adapter_model.safetensors` | LoRA đã train (r=32, α=64, lr 1e-4, `sft_v11.jsonl` 6.500 dòng, 938 bước) | đính kèm trong `repro-row1/`, md5 ghi ở `repro-row1/README.md` |
| `models/rerank_v11/model.safetensors` | reranker fine-tune trên `train.json` LegalQA | đính kèm trong `repro-row1/`, md5 `e11f3764…` (2.271.071.852 byte) |
| Dense cache (`bm25.pkl`, `dense_*.npy/.ids`, `chunks.jsonl`) | retrieval artifact | đính kèm trong `repro-row1/data/clean/` |
| `data/private-official.json` | 1.918 id câu hỏi Private Test | đính kèm |

Tổng tham số runtime **3,527B < 4B** (LoRA không cộng thêm khi hợp nhất). Giấy phép mọi model: Apache-2.0/MIT, không
dùng API ngoài, không dữ liệu ngoài cuộc thi.

### 5. Cách chạy

```bash
R=$PWD/repro-row1; mkdir -p out
# Bước MBR — CPU, tái lập submission từ 6 candidate đã đóng gói:
docker run --rm --network none -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 mbr
# Sinh lại 1 candidate — GPU (nvidia-container-toolkit), ckpt ∈ {b80,b320,b920}, rot ∈ {0..3}:
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 gen b920 0
# Sinh lại cả 6 candidate rồi MBR trên bản vừa sinh — GPU:
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 all
```

Đã xác minh (29/09/2026): `docker build --network none` thành công; `mbr` trong container
`--network none` cho `submission.json` trùng từng byte bản đã nộp (md5 `09885861…`); trong
container nạp offline được tokenizer/config Qwen3.5-2B, 3 adapter, reranker. Chế độ `gen`
chưa chạy trong Docker (máy đóng gói không có GPU CUDA); cùng script, cùng prompt đã sinh
lại 100 câu đầu `b920` r0 trên Kaggle T4, khớp 100/100 từng ký tự.

## Huấn luyện lại từ đầu (reranker + LoRA)

Các bước dưới đây sinh lại `rerank_v11` và 3 adapter `sft_v11/{b80,b320,b920}` từ dữ liệu
BTC cấp (`data/official/`). Lệnh và tham số lấy đúng từ script đã chạy trên máy train
(`pipeline5.sh`, `pipeline6.sh` trong backup), không phải giá trị mặc định của script.

Thư viện lúc train: `transformers==5.14.1`, `tokenizers==0.22.2`. **Không dùng
`tokenizers` 0.23.x**: bản này làm rust tokenizer panic (`AddedVocabulary bad split`) hoặc
segfault giữa lúc mine, và abort cả tiến trình chứ không ném exception bắt được.

### 1. Xử lý dữ liệu trước khi fine-tune reranker

```bash
export PYTHONPATH=src:scripts TOKENIZERS_PARALLELISM=false
python scripts/clean_data.py                     # -> data/clean/corpus.jsonl, train đã sạch
python scripts/build_chunks.py 256 256           # -> data/clean/chunks.jsonl (287.741 chunk)
python scripts/build_gold.py                     # -> data/clean/gold.jsonl (6.955 câu weak gold)
python scripts/build_dense.py --model intfloat/multilingual-e5-small --prefix e5
python scripts/build_dense.py --model AITeamVN/Vietnamese_Embedding
python scripts/mine_rerank.py --model AITeamVN/Vietnamese_Reranker --out rerank_train.jsonl
```

1. **Làm sạch corpus** (`clean_data.py`): gỡ wrap mềm `\r\n\n` của trình crawl HTML
   (100% Document có, 93,9% rơi giữa câu). Mọi offset Chunk/Evidence về sau trỏ vào
   Passage đã gỡ wrap.
2. **Làm sạch Reference answer** (`clean.clean_reference_answer`): xoá caption ảnh, link
   "tải tại đây", ký tự vô hình (BOM/ZWSP/LRM), và cắt dòng teaser cuối là câu hỏi
   (986/6.929 answer kết thúc bằng "?"). Answer sạch được dùng làm nhãn ở bước 5.
3. **Chunk** 256 từ, bước 256 (`build_chunks.py`). Chunk giữ nguyên văn theo offset, kể cả
   `\n\n` phân khoản.
4. **Weak gold** (`build_gold.py`): `train.json` không có nhãn văn bản, nên tự suy ra văn
   bản gold qua hai kênh độc lập. Kênh nguyên văn tìm shingle 12 từ của answer trong
   chunk. Kênh số hiệu tra số văn bản mà answer trích dẫn (vd `90/2017/NĐ-CP`) qua
   `Document.number`. Không lọc ứng viên bằng BM25, để gold không bị lệch theo chính
   retriever cần đo.
5. **Dense cache + BM25**: encode toàn bộ chunk với max_len 256 (512 làm trần hybrid@50
   tụt 0,4 điểm). BM25 Okapi k1=1,5 b=0,75 được cache ở `bm25.pkl`.
6. **Mine dữ liệu rerank** (`mine_rerank.py`):
   - **Tập giữ lại:** `random.seed(0); random.sample(gold, 1000)` được bỏ khỏi train, để
     số đo sau fine-tune không đo trên câu đã học. Còn lại 5.955 câu.
   - **Pool:** lấy đúng phân phối reranker gặp lúc infer. BM25, e5-small và
     Vietnamese_Embedding mỗi nhánh lấy top 1.000, trộn bằng RRF (k=60) rồi cắt pool 50.
   - **Positive:** chunk thuộc văn bản gold **và** chứa một shingle 12 từ của answer. Nếu
     không chunk nào khớp, lấy chunk văn bản gold có điểm teacher (reranker gốc) cao nhất.
   - **Negative:** chunk trong pool **ngoài** văn bản gold **và** không chứa shingle
     answer. Chunk khác văn bản mà chép nguyên văn answer thường là văn bản sửa đổi
     hoặc trùng nội dung, nên ép nó làm negative là dạy sai.
   - **Bỏ câu:** câu không có chunk gold nào trong pool thì bị loại (~2%).
   - **Lọc ký tự:** câu hỏi và chunk qua `rerank._sach` (NFC, xoá ký tự điều khiển C0/C1 và
     zero-width) trước khi tokenize, giống hệt lúc infer.
   - **Kết quả:** `rerank_train.jsonl` **5.818 dòng**. Mỗi dòng gồm `pos`, `neg` và điểm
     `teacher` của cả pool.

### 2. Fine-tune reranker `rerank_v11`

```bash
python scripts/train_reranker.py --model AITeamVN/Vietnamese_Reranker \
  --data data/clean/rerank_train.jsonl --out runs/rerank_v11 \
  --bs 4 --accum 4 --group 16 --lr 1e-5 --epochs 1.0
```

| Tham số | Giá trị | Lý do |
|---|---|---|
| Base | `AITeamVN/Vietnamese_Reranker` (XLM-R large, 568M) | cross-encoder |
| Loss | listwise: softmax CE trên 1 positive + 15 negative (`group 16`) | lúc infer chỉ cần **xếp hạng** trong pool; BCE từng cặp ép điểm về 0/1 là thừa và phạt nặng nhãn weak gold nhiễu |
| Batch | 4 câu × accum 4 = 16 câu/bước (mỗi câu 16 cặp) | 1 epoch = 364 bước |
| lr / lịch | 1e-5, warmup 10% rồi giảm tuyến tính; AdamW wd 0,01; clip 1,0 | lr thấp + 1 epoch: chỉ chỉnh cho miền luật VN, lr cao làm quên năng lực tổng quát |
| Độ dài | max_len 512, `truncation="only_second"` | cắt passage, giữ nguyên câu hỏi |
| Số học | autocast bf16, gradient checkpointing, seed 0 | 4×16×512 token qua XLM-R large không vừa VRAM nếu giữ đủ activation |

Có một lượt chạy khác dùng `--bs 2 --accum 8`, cũng hiệu dụng 16 câu/bước. Log train của
reranker không còn giữ được, nên không xác định được `rerank_v11` sinh từ lượt nào. Bản
đầy đủ có md5 `e11f376466ce993edeba80dcf7acd7aa` (2.271.071.852 byte, F32).

### 3. Dựng dữ liệu SFT bằng reranker đã fine-tune

```bash
export QWEN_TOKENIZER=<snapshot Qwen3.5-2B>/tokenizer.json   # đếm token để lọc mẫu quá dài
python scripts/build_sft.py --model runs/rerank_v11 --dense AITeamVN/Vietnamese_Embedding \
  --batch-size 64 --pool 50 --nguon train_sft.jsonl --gold gold.jsonl --ra data/clean/sft_v11.jsonl
python scripts/build_sft.py --model runs/rerank_v11 --dense AITeamVN/Vietnamese_Embedding \
  --batch-size 64 --pool 50 --nguon dev100.jsonl --gold gold.jsonl --giu-het \
  --ra data/clean/dev100_v11.jsonl
```

Retrieval dựng prompt SFT gồm BM25 + Vietnamese_Embedding + `rerank_v11`, top-5 Evidence.
Khác pipeline infer ở chỗ **không có nhánh e5-small**: `build_sft.py` chỉ nhận một
bi-encoder, và giữ đúng công thức cũ để không đổi hai biến cùng lúc. Vào 6.908 câu, giữ
**6.500** (94,1%). Bị loại 367 câu retrieval trượt văn bản gold và 41 câu không dựng được
weak gold. Dev dựng bằng **cùng** reranker, `--giu-het` giữ cả câu trượt gold để không tự
nâng điểm.

### 4. Fine-tune LoRA `sft_v11`

```bash
python scripts/train_sft.py --model <snapshot Qwen/Qwen3.5-2B @ 15852e8c> \
  --data data/clean/sft_v11.jsonl --dev data/clean/dev100_v11.jsonl --dev-n 0 \
  --ra runs/sft_v11 --eval-rieng --epochs 2.0 --nhan-dai 3 --nguong-dai 600 \
  --lr 1e-4 --r 32 --alpha 64 --bs 2 --accum 8 --eval-steps 40 --max-seq 8704 --gio-toi-da 24
```

| Tham số | Giá trị | Lý do |
|---|---|---|
| Base | `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | adapter gắn với đúng revision này |
| LoRA | r=32, α=64, dropout 0,05, bias none | 21.823.488 tham số train (1,15%) |
| Target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | bài toán là **trích** văn bản, và phần quyết định copy chữ nào nằm nhiều ở MLP |
| Loss | CE **chỉ trên token đáp án**; lm_head chỉ chạy trên lát đáp án, chia khối 512 | 69,8% 8-gram của answer là nguyên văn passage. Logits toàn chuỗi (vocab 248.070) làm OOM T4 |
| Prompt | `apply_chat_template(add_generation_prompt=True)`, pad phải khi train | token điều khiển khớp từng token với lúc infer |
| Oversample | answer > 600 từ nhân 3 bản (518 mẫu) | dạy model cách **kết thúc** đáp án dài, giảm lặp vòng |
| Độ dài | max_seq 8.704 token | 7.536 mẫu vào, giữ 7.507 (loại 29) |
| Batch / lịch | bs 2 × accum 8 = 16 mẫu/bước; 2 epoch = **938 bước**; AdamW lr 1e-4, wd 0, warmup 3% rồi cosine về 0; clip 1,0 | |
| Số học | fp16 + GradScaler, gradient checkpointing, seed 0 | max \|activation\| 32,8 so với trần fp16 65.504 |
| Mốc lưu | mỗi 40 bước + cuối mỗi epoch; chấm dev riêng bằng METEOR (code BTC) | chọn theo METEOR, không theo loss: ROUGE-L có sàn 0,278 và thưởng câu trả lời cụt |

Bản nộp dùng các mốc **b80, b320, b920**, không dùng mốc cuối b938. Phiên train bị OOM
và chạy nối tiếp từ `b160` bằng `--tiep runs/sft_v11/b160 --moc 160`. Lượt nối tiếp dùng
lr bằng giá trị cosine tại bước 160 (9,30e-5) và 1,659 epoch còn lại. Thứ tự mẫu đi tiếp
đúng hoán vị cũ, không xáo lại.

## Tài liệu

| File | Nội dung |
|---|---|
| `CONTEXT.md` | Glossary — thuật ngữ miền, không có chi tiết cài đặt |
| `docs/EDA_V10.md` | Toàn bộ số đo |
| `docs/adr/` | Quyết định kiến trúc |
| `docs/V10_BUILD_SPEC.md` | Đặc tả 5 module lõi |
| `AGENTS.md` | Hướng dẫn cho agent AI |
