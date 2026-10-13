# ghienaram — UIT DSC 2026 Task 2 LegalQA: hướng dẫn tái lập kết quả Private Test

Đội **ghienaram**. Bản nộp tốt nhất: `submission/submission.zip` (23/09/2026), private
LB **0,6520**. Gói này tái lập được kết quả đó **không cần Internet**. Gói có đủ mã nguồn,
thư viện (wheel), Docker image, model base, checkpoint đã train, dữ liệu retrieval, prompt
và câu trả lời ứng viên.

Mục lục (theo yêu cầu BTC, email 28/09/2026):
[1. Môi trường](#1-môi-trường) ·
[2. Thư viện](#2-thư-viện) ·
[3. Cài đặt](#3-cài-đặt) ·
[4. Cấu hình](#4-cấu-hình) ·
[5. Dữ liệu](#5-dữ-liệu) ·
[6. Cách chạy](#6-cách-chạy) ·
[7. Mô hình, checkpoint, tài nguyên ngoài](#7-mô-hình-checkpoint-tài-nguyên-ngoài) ·
[8. Kiểm chứng và giới hạn](#8-kiểm-chứng-và-giới-hạn)

## Cấu trúc gói

```
repro-row1/
├── README.md                          file này
├── MD5SUMS                            md5 mọi file (md5sum -c MD5SUMS)
├── docker/legalqa-row1_amd64.tar.gz   Docker image legalqa:row1 dựng sẵn
├── source/                            mã nguồn (commit 5b13ea7) + Dockerfile + wheels/ + nltk_data/
├── hf/                                4 model base tải từ HuggingFace
├── runs/sft_v11/{b80,b320,b920}/      3 adapter LoRA đã train (checkpoint của đội)
├── models/rerank_v11/                 reranker đã fine-tune (checkpoint của đội)
├── data/                              câu hỏi private + corpus/chunk/BM25/dense đã dựng
├── prompts/prompts_rot{0..3}.jsonl    prompt đã dựng cho 1.918 câu private
├── candidates/                        6 bộ câu trả lời ứng viên (đầu vào MBR)
└── submission/                        bản đã nộp, để đối chiếu
```

Tổng 22 GB.

## 1. Môi trường

| Thành phần | Yêu cầu |
|---|---|
| OS | Linux x86_64 có Docker. Image là `linux/amd64`. |
| Python | 3.12 (có sẵn trong image; `pyproject.toml` yêu cầu `>=3.12,<3.13`) |
| GPU | Chỉ cần khi sinh lại câu trả lời (mục 6B). NVIDIA, CUDA 12.4: driver ≥ 525 và `nvidia-container-toolkit`. Candidate gốc sinh trên **Kaggle T4 16 GB**, fp16. |
| CPU | Đủ cho bước MBR (mục 6A). `mbr_chon.py` mặc định dùng 8 tiến trình. |
| Đĩa | ~30 GB trống (gói 22 GB + image giải nén) |
| Mạng | **Không cần**, kể cả lúc build image |

Biến môi trường offline đặt sẵn trong image: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `NLTK_DATA=/app/nltk_data`,
`PYTHONPATH=/app/src:/app/scripts`.

## 2. Thư viện

`source/requirements-lock.txt` gồm **68 gói**, ghim version và sha256. Wheel tương ứng
(linux x86_64, CPython 3.12) nằm sẵn ở `source/wheels/`. Các gói chính:

| Gói | Version | Ghi chú |
|---|---|---|
| torch | 2.6.0 (CUDA 12.4) | |
| transformers | 5.14.1 | version lúc sinh candidate và lúc train |
| tokenizers | 0.22.2 | **không dùng 0.23.x**: rust tokenizer panic `AddedVocabulary bad split` |
| peft | 0.21.0 | version đã lưu adapter |
| accelerate, safetensors, tiktoken, sentencepiece | theo lock | `tiktoken` cần cho tokenizer reranker |
| numpy, scipy, scikit-learn | theo lock | |
| nltk, rouge-score | 3.10.3, 0.0.4 | chấm METEOR/ROUGE-L. Data wordnet + omw-1.4 ở `source/nltk_data/` |

## 3. Cài đặt

**Cách 1 — Docker, dùng image dựng sẵn (khuyến nghị):**

```bash
docker load < repro-row1/docker/legalqa-row1_amd64.tar.gz     # -> legalqa:row1
```

**Cách 2 — build lại image, vẫn offline:**

```bash
cd repro-row1/source
docker build --network none --platform linux/amd64 -t legalqa:row1 .
```

**Cách 3 — không dùng Docker** (Linux x86_64, Python 3.12):

```bash
cd repro-row1/source
python3.12 -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links wheels --require-hashes -r requirements-lock.txt
export PYTHONPATH=$PWD/src:$PWD/scripts NLTK_DATA=$PWD/nltk_data \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

## 4. Cấu hình

Pipeline của bản nộp (row1):

1. **Retrieval lai:** BM25 Okapi (k1=1,5, b=0,75) + `intfloat/multilingual-e5-small` +
   `AITeamVN/Vietnamese_Embedding`. Mỗi nhánh lấy top-1.000, trộn bằng RRF (k=60), cắt pool
   50 chunk. Chunk dài 256 từ.
2. **Rerank:** `models/rerank_v11` chấm 50 chunk, lấy top-5, rồi mở rộng mỗi chunk thành
   một Evidence (trọn Điều luật chứa nó).
3. **Sinh:** Qwen3.5-2B + adapter LoRA. Greedy (`do_sample=False`), `max_new_tokens=1024`,
   fp16, `gen_bs=2`.
4. **MBR:** 6 ứng viên mỗi câu. Chọn ứng viên có tổng METEOR với 5 ứng viên còn lại lớn
   nhất (`scripts/mbr_chon.py`).

| Adapter | Bước SFT | Rotation dùng |
|---|---:|---|
| `b80`  | 80  | r0 |
| `b320` | 320 | r0 |
| `b920` | 920 | r0, r1, r2, r3 |

Rotation r*k* là thứ tự 5 Evidence trong prompt dịch vòng *k* vị trí; r0 là thứ tự theo điểm
rerank. Tham số dòng lệnh đầy đủ nằm ở `source/docker/entrypoint.sh`. Tham số trên Kaggle
nằm ở `source/kaggle/jobs/infer_priv_<ckpt>_r<rot>.json`.

Mã nguồn là commit **`5b13ea7`** (`source/COMMIT.txt`), chính commit đã sinh prompt và
candidate của row1.

## 5. Dữ liệu

Chỉ dùng dữ liệu BTC cấp, không dùng dữ liệu ngoài.

| File | Nguồn | Nội dung |
|---|---|---|
| `data/private-official.json` | Private Test của BTC | 1.918 câu hỏi `{id: {question, answer: null}}` |
| `data/clean/corpus.jsonl` | `selected-contexts/*.json` của BTC, qua `scripts/clean_data.py` | 8.512 văn bản luật, đã gỡ wrap mềm `\r\n\n` của trình crawl HTML |
| `data/clean/chunks.jsonl` | `scripts/build_chunks.py 256 256` | chunk 256 từ, nguyên văn theo offset |
| `data/clean/bm25.pkl` | `ChunkIndex` (`src/legalqa/bm25.py`) | chỉ mục BM25 |
| `data/clean/dense_*_L256.{npy,ids}` | `scripts/build_dense.py`, max_len 256 | vector chunk của 2 bi-encoder |
| `prompts/prompts_rot{0..3}.jsonl` | notebook `source/kaggle/shard_prompt.ipynb` (retrieval → rerank → `dung_prompt`) | prompt cho 1.918 câu, mỗi dòng `{id, question, prompt}` |
| `candidates/<ckpt>_r<rot>.jsonl` | `scripts/infer_private_shard.py` trên Kaggle T4 | 1.918 câu trả lời mỗi file |

Dữ liệu train (`train.json` của BTC) chỉ dùng để fine-tune. Cách xử lý dữ liệu trước khi
fine-tune nằm ở `source/README.md`, mục "Huấn luyện lại từ đầu".

## 6. Cách chạy

```bash
R=$PWD/repro-row1; mkdir -p out
```

**A. Tái tạo submission từ 6 candidate có sẵn** (CPU, ~2-3 phút):

```bash
docker run --rm --network none -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 mbr
md5sum out/submission.json $R/submission/submission.json      # phải trùng: 09885861…
```

**B. Sinh lại câu trả lời bằng GPU từ prompt + checkpoint, rồi MBR:**

```bash
# cả 6 candidate (b80 r0, b320 r0, b920 r0-r3), rồi MBR -> out/submission.json
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 all

# hoặc từng candidate: ckpt ∈ {b80,b320,b920}, rot ∈ {0,1,2,3} -> out/candidates/b920_r0.jsonl
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 gen b920 0
```

`gen` ghi nối tiếp và bỏ qua id đã có. Dừng giữa chừng rồi chạy lại cùng lệnh thì nó chạy
tiếp. Trên Kaggle, mỗi candidate chia 8 shard chạy song song
(`source/kaggle/infer_private_shard.ipynb`).

**Không dùng Docker** (sau cài đặt cách 3, đứng ở `repro-row1/`):

```bash
python source/scripts/mbr_chon.py candidates/b80_r0.jsonl candidates/b320_r0.jsonl \
  candidates/b920_r{0,1,2,3}.jsonl --cau-hoi data/private-official.json --ra out
python source/scripts/infer_private_shard.py --model hf/Qwen3.5-2B --adapter runs/sft_v11/b920 \
  --prompts prompts/prompts_rot0.jsonl --ra out/candidates/b920_r0.jsonl --max-new 1024 --gen-bs 2
```

**C. Huấn luyện lại reranker + LoRA từ đầu:** lệnh, siêu tham số và cách xử lý dữ liệu nằm
ở `source/README.md`, mục "Huấn luyện lại từ đầu".

## 7. Mô hình, checkpoint, tài nguyên ngoài

**Model base** (tải từ HuggingFace ngày 24/09/2026, nằm ở `hf/`):

| Thư mục | Repo HF | Vai trò | Tham số | Giấy phép |
|---|---|---|---:|---|
| `hf/Qwen3.5-2B` | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` | LLM sinh câu trả lời | ~2B | Apache-2.0 |
| `hf/multilingual-e5-small` | `intfloat/multilingual-e5-small` | bi-encoder retrieval | 118M | MIT |
| `hf/Vietnamese_Embedding` | `AITeamVN/Vietnamese_Embedding` | bi-encoder retrieval | 568M | Apache-2.0 |
| `hf/Vietnamese_Reranker` | `AITeamVN/Vietnamese_Reranker` | base của reranker | 568M | Apache-2.0 |

**Checkpoint do đội train:**

| Đường dẫn | Mô tả | md5 |
|---|---|---|
| `runs/sft_v11/b80/adapter_model.safetensors` | LoRA r=32 α=64 trên Qwen3.5-2B, bước 80 | `244c2f341a5d9188ea3e2280658ddcd7` |
| `runs/sft_v11/b320/adapter_model.safetensors` | như trên, bước 320 | `ac41c7e2bc16272c5668533b4b559045` |
| `runs/sft_v11/b920/adapter_model.safetensors` | như trên, bước 920 | `fcc6a48741e5b3c1503d24e737dc23b7` |
| `models/rerank_v11/model.safetensors` | Vietnamese_Reranker fine-tune listwise trên `train.json` (2.271.071.852 byte) | `e11f376466ce993edeba80dcf7acd7aa` |

Các checkpoint này có bản sao trên HuggingFace Hub (private, cấp quyền khi BTC yêu cầu):
`Phuong0703/VLegalQwen` và `Phuong0703/VLegal-Reranker`.

`adapter_config.json` ghi `base_model_name_or_path=/workspace/hf/hub/...` (đường dẫn trên
máy train). Code nạp base qua tham số `--model hf/Qwen3.5-2B`, không dùng trường này.

**Tổng tham số lúc chạy: 3,527B (< 4B).** Không gọi API ngoài, không dùng dữ liệu ngoài
cuộc thi.

**Checksum khác:**

| File | md5 |
|---|---|
| `prompts/prompts_rot0.jsonl` | `a8e8b17e81be39600682aa13bd0afe57` |
| `prompts/prompts_rot1.jsonl` | `4c1592a177f55b90e7a568a72b6b2e8e` |
| `prompts/prompts_rot2.jsonl` | `a699b2a901900deb4af527c93a28880b` |
| `prompts/prompts_rot3.jsonl` | `4023428066db339e81268af61557389f` |
| `data/clean/bm25.pkl` | `0f83e7d8aa08691dd4f411b6aa43480c` |
| `data/clean/dense_multilingual-e5-small_L256.npy` | `a713db004d3b39a29a2a9c88fa048b7a` |
| `data/clean/dense_Vietnamese_Embedding_L256.npy` | `379a5da6dd430fc8ad76d0b178936be8` |
| `submission/submission.json` | `09885861c0dba13fbfe595b60ee8b899` |
| `docker/legalqa-row1_amd64.tar.gz` | `97f997d87b4c4ce8fb0ed1de29627662` |

## 8. Kiểm chứng và giới hạn

**Đã kiểm (29/09/2026, máy tắt mạng `--network none`):**
- `docker build` offline thành công.
- Chạy `mbr` trong container cho `submission.json` **trùng từng byte** bản đã nộp (md5
  `09885861…`, 1.918/1.918 câu). Kết quả vẫn trùng sau khi xoá image rồi `docker load`
  lại từ file `.tar.gz`.
- Trong container nạp offline được tokenizer/config Qwen3.5-2B, 3 adapter và reranker.
- Sinh lại 100 câu đầu bằng `b920` r0 trên Kaggle T4 (cùng script, cùng prompt): khớp
  100/100 từng ký tự với candidate đã đóng gói.

**Giới hạn:**
- Chế độ `gen` chưa chạy **trong Docker** vì máy đóng gói không có GPU CUDA. Version
  torch/peft trên Kaggle lúc sinh candidate chưa ghi lại. GPU hoặc torch khác có thể làm vài
  câu lệch ký tự khi sinh greedy fp16. Mức A (MBR) không bị ảnh hưởng.
- Prompt r1–r3 được cung cấp sẵn, mã sinh không còn lưu. r0 dựng lại được bằng
  `source/kaggle/shard_prompt.ipynb`.
- Train lại chỉ cho trọng số gần đúng, không trùng bit (GPU không deterministic). Log
  train của reranker không còn.
- Khi offline, `nltk.download` in cảnh báo `Error loading wordnet`. Cảnh báo này vô hại:
  data đã có ở `NLTK_DATA`.
