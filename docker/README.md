# Docker — tái tạo submission row1 (private LB 0,6520), offline

Image chỉ chứa mã + thư viện + nltk data. Model/dữ liệu mount từ gói `repro-row1` vào
`/repro` (xem `repro-row1/README.md`). Output ghi ra `/out`.

## Nạp image (không cần mạng)

```bash
docker load < repro-row1/docker/legalqa-row1_amd64.tar.gz      # tag legalqa:row1, linux/amd64
```

Hoặc build lại từ `source/`, cũng offline: 67 wheel nằm ở `source/wheels/`, cài bằng
`--no-index --require-hashes -r requirements-lock.txt`.

```bash
cd repro-row1/source
docker build --network none --platform linux/amd64 -t legalqa:row1 .
```

## Chạy

```bash
R=$PWD/repro-row1; mkdir -p out
# 1) MBR từ 6 candidate đóng gói sẵn — CPU, ~2-3 phút
docker run --rm --network none -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 mbr

# 2) Sinh lại 1 candidate — GPU (NVIDIA driver >= 525, nvidia-container-toolkit)
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 gen b920 0
#    -> out/candidates/b920_r0.jsonl ; checkpoint ∈ {b80,b320,b920}, rotation ∈ {0,1,2,3}

# 3) Sinh lại cả 6 candidate (b80 r0, b320 r0, b920 r0-r3) rồi MBR trên bản vừa sinh — GPU
docker run --rm --network none --gpus all -v $R:/repro:ro -v $PWD/out:/out legalqa:row1 all
```

Kết quả: `out/submission.json` + `out/submission.zip`. `gen` ghi append và bỏ qua id đã có
(`da_co`), nên dừng giữa chừng rồi chạy lại sẽ làm tiếp.

`nltk.download` trong `legalqa.scorer.chuan_bi_nltk` in cảnh báo `Error loading wordnet`
khi không có mạng. Cảnh báo này vô hại: wordnet/omw-1.4 đã nằm ở `/app/nltk_data`
(`NLTK_DATA`).

## Môi trường trong image

Python 3.12, torch 2.6.0+cu124, transformers 5.14.1, tokenizers 0.22.2 (0.23.x làm rust
tokenizer panic), peft 0.21.0 (bản lưu adapter). Danh sách đủ có hash:
`requirements-lock.txt`.
