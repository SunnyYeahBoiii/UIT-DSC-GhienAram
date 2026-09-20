# Pipeline v10 — runbook

Tài liệu này nói **thứ tự chạy** và **tham số đã chốt**. Lý do của từng con số nằm ở
`docs/EDA_V10.md` (mục được dẫn ở từng dòng); lý do của từng quyết định kiến trúc nằm ở
docstring đầu mỗi module. Không lặp lại ở đây — số chép ra hai chỗ là số sẽ lệch.

## Đường đi của một câu hỏi

```
                    ┌─ BM25 deep 1.000 ──┐
câu hỏi ────────────┤                    ├─ RRF k=60 ─> pool 50
                    └─ dense deep 1.000 ─┘                 │
                                                           v
                               cross-encoder rerank (zero-shot)
                                                           │
                                  top-5 Chunk ─> expand_to_evidence
                                                           │
                                  + legal graph (≤2 hop, max_extra 3)
                                                           │
                                          dung_prompt ─> Qwen ─> sanitizer
```

Chunk là đơn vị **tìm**, Evidence là đơn vị **sinh** (ADR-0001). Ranh giới đó là chỗ
`expand_to_evidence` nằm, và là lý do pipeline có hai đơn vị chứ không một.

## Tham số đã chốt

| tầng | tham số | giá trị | nguồn |
|---|---|---|---|
| chunking | size / stride | 256 / 256 từ | ADR-0001, §13 |
| bm25 | k1 / b | 1,5 / 0,75 | Okapi mặc định |
| bm25 / dense | deep | 1.000 | §15.8 |
| hybrid | RRF k | 60 | §15.8 — trộn theo **hạng**, không chuẩn hoá điểm |
| rerank | pool | 50 | §15.6 — lợi ích biên 20→50 là 0,48 điểm/phút, 50→100 chỉ 0,03 |
| rerank | max_len | 512 | `RERANK_MAX_LEN` |
| dense | max_len | 256 | §15.8 — 512 thua thật, không phải nhiễu |
| evidence | top_n / max_chars | 5 / 24.000 | §15.7; 24.000 lấy từ v9.2 `kaggle_inference_t4.yaml` |
| graph | max_hop / max_extra | 2 / 3 | ADR-0002 — có visited-set, không loop vô hạn |

`top_n=5` đếm **seed**, không đếm Evidence trả về: graph sinh thêm là ngữ cảnh bắt
buộc, không phải kết quả tìm kiếm. Ngân sách ký tự cắt theo **cụm** (seed + phần graph
của nó), không theo từng Evidence — xem `retrieve.evidence_day_du`.

## Recall (đo 13-09, T4x2, một lượt chạy, n=500)

| pool | | R@1 | **R@5** | trần |
|---|---|---:|---:|---:|
| BM25 | BM25 | 56,4 | 81,0 | 96,2 |
| BM25 | +rerank | 75,2 | **92,0** | |
| hybrid | +rerank | 77,2 | **93,6** | 97,6 |

Dense mua +1,6 R@5 với 118M params, 0,22 GB index, +0,02 s/câu. Chi tiết và vì sao ước
cũ (+0,9) thấp một cách hệ thống: §15.8.

## Ngân sách params — trần cứng 4B

Quy tắc đếm, BTC trả lời 20-09: *"Số tham số tính theo số lượng mô hình được sử dụng chứ
không tính theo số lần mô hình đó được sử dụng."* → hệ G15 sinh 15 lượt (3 hệ × 5 prompt
rotation) vẫn chỉ tính generator **một lần**. Không nhân theo lượt. Nếu tính theo lượt thì
15 × 2,274 = 34,1B, thủng trần 8,5 lần và toàn bộ hướng MBR chết — nên đây là câu chốt
quan trọng nhất với thiết kế hiện tại.

Đếm theo **model phân biệt**: hai bản reranker là hai checkpoint khác nhau, cùng được nạp
khi dựng prompt, nên tính cả hai. Adapter LoRA khác nhau cũng tính riêng; adapter dùng lại
cho nhiều lượt chỉ tính một lần.

| model | vai trò | lúc sinh? | params |
|---|---|---|---:|
| generator Qwen3.5-2B | sinh đáp án, cả 15 lượt | có | 2.274.069.824 |
| reranker `AITeamVN/Vietnamese_Reranker` (gốc) | xếp evidence cho lượt **A** | không (offline) | 567.755.777 |
| reranker fine-tuned `rerank_ft_g2` | xếp evidence cho lượt **C**, **E** | không (offline) | 567.755.777 |
| dense `AITeamVN/Vietnamese_Embedding` | nhánh dense của pool hybrid, **cả 3 lượt** | không (offline) | 567.755.777 |
| adapter LoRA `b535` (r=32) | lượt **A** và **C** dùng chung | có | 21.823.488 |
| adapter LoRA `SFT g2` (r=32) | lượt **E** | có | 21.823.488 |
| **tổng G15** | | | **4.020.984.131 = 4,021B — VƯỢT trần 20.984.131** |

Biến thể G20 thêm lượt **X** với adapter xáo riêng (+21.823.488) → **4,043B**, vượt 43M.

Bản nộp tốt nhất hiện tại (`v10-private-G15`, LB 0,6427) nằm **sai phía trần**. Ba đường
gỡ, đo trên dev175 (`scratchpad/do_g10.py`, MBR chay gam=1/p=1/lam=0):

| tập lượt | METEOR | Δ vs G15 | params | trần |
|---|---:|---:|---:|:--|
| ACE (G15 hiện tại) | 0,5996 | — | 4,021B | **VƯỢT** |
| AE | 0,5971 | −0,0025 | 4,021B | **VƯỢT** |
| **CE** (bỏ lượt A) | **0,5925** | **−0,0071** | **3,453B** | OK |
| AC | 0,5793 | −0,0204 | 3,999B | OK (dư 1,0M) |
| A / C / E đứng riêng | 0,5784 / 0,5731 / 0,5751 | | 3,431B | OK |
<!-- Đừng đọc hàng này là "A tốt hơn C": 0,5784 − 0,5731 = 0,0053, mà SE của hiệu trên
     dev175 là ~0,0093 → 0,57 SE, tức nhiễu. Ba lượt đứng riêng NGANG nhau; điều đo được
     là cả ba đều thua hệ gộp. Xếp hạng A>E>C ở đây không có căn cứ thống kê. -->

| oracle15 (trần chọn đúng) | 0,6825 | +0,0829 | | |

- **CE**: 0 GPU, mất 0,0071. `CI95 [−0,0167, +0,0009]`, 30 câu tốt lên / 37 xấu đi /
  108 y nguyên, σ từng câu 0,0598 → SE ở n=1918 là 0,0014, nên −0,0071 là 5 SE: có thật
  trên private, không phải nhiễu dev175.
- **Đổi dense sang e5-small** (118M): trả lại 450M → 3,571B, giữ cả 3 lượt. Phải dựng lại
  prompt và sinh lại 15 rot. Chặn phát hiện 20-09: checkpoint `rerank_ft_g2` chỉ tồn tại
  trên instance Vast đã mất SSH, nên không tái lập C/E chính xác; phải thay bằng
  `rerank_ft_ai` (cùng XLM-R large 568M, có trên Kaggle) và đo lại từ đầu.
- **Lượt A dùng luôn `rerank_ft_g2`**: 3,453B nhưng xoá đúng trục đa dạng làm A có giá trị.

Đọc chính: C gần như dư thừa (bỏ C chỉ mất 0,0025), còn bỏ A mất gấp 3. A là lượt duy nhất
khác E ở **cả hai** trục (adapter + reranker); C chung một trục với mỗi bên. Cùng lý do
lượt X thất bại: X chỉ xáo thứ tự evidence, không khác trục nào, Δ −0,0025.

Ghi chú:
- Adapter đếm từ header `adapter_model.safetensors`: 192 tensor F32, 21.823.488 params,
  `r=32 lora_alpha=64`, target 7 module (`q,k,v,o,gate,up,down`_proj).
- Dense **không** phải e5-small: `dung_pr_private.py` mặc định
  `--dense /workspace/base/Vietnamese_Embedding`, và `pri_g2.sh` không ghi đè cờ đó, nên
  cả ba lượt A/C/E đều chạy AITeamVN 568M. §15.8 của EDA loại AITeamVN vì lý do params —
  quyết định đó không bao giờ tới được đường chạy thật.
- **Ba dòng reranker/dense chỉ chạy offline để dựng prompt**, không nạp lúc sinh. Vẫn khai
  vì trần 4B hiểu là trần *trọng số phải tải*, không phải trần tính toán lúc suy luận —
  cách diễn đạt "số lượng mô hình được sử dụng" của BTC ngả về nghĩa này. Nếu BTC xác nhận
  chỉ tính model lúc sinh thì khai báo còn **2,318B**.
- Reranker gốc 0,568B tồn tại **chỉ để phục vụ lượt A**. Bỏ lượt A sẽ giải phóng trọn
  0,568B, nhưng mất nguồn đa dạng thật của MBR: A/C/E đứng riêng đều thua hệ gộp
  (0,5784 / 0,5731 / 0,5751 → G15 0,5996), tức A đóng góp bằng **khác biệt** chứ không
  bằng chất lượng, và khác biệt đó đến từ đúng chỗ nó dùng reranker khác. Đối chứng: lượt
  X vẫn dùng reranker FT như C/E, chỉ xáo thứ tự evidence — đa dạng giả, đo được Δ −0,0025.

> **Sửa số, 20-09.** Bảng cũ ghi `2,274 + 0,568 + 0,118 = 2,96B (dư 1,04B)` — thiếu
> reranker fine-tuned và toàn bộ adapter, tức khai thiếu **0,61B**. Lỗi phát sinh vì bảng
> được viết ở §15.8 khi pipeline mới có một reranker, rồi không cập nhật khi thêm lượt
> C/E. Cùng loại lỗi với ô "Sửa số, 12-09" trong `EDA_V10.md`: một con số áng hoặc lỗi
> thời đứng im trong tài liệu trong khi hệ thống quanh nó đã đổi.
>
> **Sửa lần hai, 20-09.** Bản 3,571B ở trên cũng sai: nó chép dòng dense e5-small từ
> §15.8 thay vì đọc đường chạy. Đường chạy dùng AITeamVN 568M → 4,021B, tức hệ đang
> **vượt trần**, không phải dư 0,429B. Lỗi nặng hơn hai ô trước một bậc: không phải một
> con số lỗi thời mà một **thành phần kiến trúc sai**.
>
> Quy tắc rút ra: **khai báo params phải sinh từ code của đường chạy thật, không chép từ
> tài liệu.** Cụ thể là đọc default của `argparse` trong script dựng prompt cộng cờ mà
> shell wrapper ghi đè, chứ không đọc bảng trong `docs/`.

## Vấn đề thừa tham số — chẩn đoán và đường ra (20-09)

### Thừa bao nhiêu, và chỗ thừa có hình dáng gì

`4.020.984.131 − 4.000.000.000 = 20.984.131`. Con số này **nhỏ hơn một adapter LoRA**
(21.823.488). Tức pipeline không vượt trần vì phình to ở đâu cả — nó vượt vì **vừa đúng
một adapter**, thành phần nhỏ nhất trong bảng, chiếm 0,54% tổng.

Hệ quả phản trực giác: món rẻ nhất để cắt (một adapter, 21,8M) đủ đưa hệ về dưới trần,
còn món to nhất (reranker 568M, 14% tổng) cắt đi thì dư 547M — thừa gấp 26 lần nhu cầu.
Nhưng **giá tính bằng METEOR đi ngược lại kích thước**:

| cắt gì | trả lại | dưới trần? | mất METEOR | vì sao |
|---|---:|:--|---:|---|
| 1 adapter LoRA | 21,8M | vừa đủ (dư 0,8M) | **mất trọn 1 lượt** | adapter LÀ định danh của lượt; hai lượt cùng adapter + cùng reranker là một lượt trùng |
| lượt A (reranker gốc) | 568M | dư 547M | −0,0071 | A là lượt duy nhất khác E ở **cả hai** trục |
| lượt C | 21,8M | vừa đủ | −0,0025 | C chung một trục với mỗi bên → gần dư thừa |
| dense 568M → e5-small | 450M | dư 429M | chưa đo | rẻ nhất về lý thuyết, nhưng phải sinh lại 15 rotation |

Đọc đúng: **trần 4B không siết dung lượng, nó siết số lượng model phân biệt.** Vì BTC tính
theo model chứ không theo lượt, mọi lượt đều miễn phí *nếu* dùng lại đúng bộ model đã khai.
Nhưng MBR chỉ sinh lời khi các lượt **khác nhau**, mà mỗi trục khác biệt thật lại là một
model mới phải khai. Trần params vì thế là trần trực tiếp lên **số trục đa dạng**, và đó
là ràng buộc chi phối cả thiết kế — không phải ràng buộc bộ nhớ.

### Vì sao không chỉ đơn giản là "cắt rồi nộp lại"

`v10-private-G15` (LB 0,6427) **không tái lập được**: checkpoint `rerank_ft_g2` chỉ tồn
tại trên một instance Vast đã mất SSH. Nên mọi phương án ở bảng trên đều không phải là
"sửa bản nộp cũ" — chúng là **dựng lại nhánh pipeline rồi đo lại từ đầu**, với
`rerank_ft_ai` thay cho `rerank_ft_g2`. Hai hệ quả:

1. Các Δ METEOR ở bảng trên đo trên hệ có `rerank_ft_g2`. Sau khi thay checkpoint, chúng
   là **ước lượng định hướng**, phải đo lại chứ không được chép sang bản nộp mới.
2. Bài học vận hành đã thành rule: `~/.claude/rules/gpu-instance.md` — checkpoint train
   xong đẩy về local ngay, không đợi tới lúc destroy.

### Đường đang thi công: v11 — một reranker, hai dense

| model | params |
|---|---:|
| generator Qwen3.5-2B | 2.274.069.824 |
| reranker (một, chọn qua grid) | 567.755.777 |
| dense `AITeamVN/Vietnamese_Embedding` | 567.755.777 |
| dense `intfloat/multilingual-e5-small` | 117.653.760 |
| adapter LoRA `b535` | 21.823.488 |
| **tổng v11** | **3.549.058.626 — dư 450.941.374** |

Ý tưởng: bỏ **một trong hai reranker** (trục đắt nhất) rồi mua lại đa dạng bằng **nhánh
dense thứ hai** trong RRF — e5-small chỉ 118M, rẻ hơn reranker thứ hai 4,8 lần. Grid
`scratchpad/grid/` đang quét 5 reranker × 11 tổ hợp dense trên 1.000 câu weak gold để
chọn cặp sống sót; chỉ cấu hình qua được gate đó mới tiêu GPU sinh đáp án trên dev175.

Dư 450M là cố ý: đủ chỗ cho một adapter thứ hai (21,8M) nếu MBR nhiều lượt quay lại, mà
vẫn còn biên an toàn cho sai số đếm.

### Quy tắc bắt buộc khi khai báo params

Ba lần sửa số trong tài liệu này (12-09, và hai lần 20-09) đều cùng một cơ chế: **một con
số đứng im trong `docs/` trong khi hệ thống quanh nó đã đổi.** Lần nặng nhất không phải số
lỗi thời mà là **thành phần kiến trúc sai** — bảng ghi dense e5-small 118M trong khi đường
chạy thật dùng AITeamVN 568M, vì bảng chép từ §15.8 EDA thay vì đọc code.

Nên trước mỗi lần nộp:

1. Đếm từ **đường chạy thật**: default của `argparse` trong script dựng prompt, **cộng**
   mọi cờ mà shell wrapper ghi đè. Không chép từ `docs/`, không chép từ EDA.
2. Adapter đếm từ header `adapter_model.safetensors`, không suy từ `r`/`target_modules`.
3. Liệt kê theo **checkpoint phân biệt**, không theo vai trò: hai bản cùng kiến trúc
   (gốc + fine-tuned) là **hai** model phải khai, dù cùng 568M.
4. Cộng cả model **chỉ chạy offline** để dựng prompt. Trần hiểu là trần *trọng số phải
   tải*. (Nếu BTC xác nhận chỉ tính model lúc sinh thì khai báo còn **2,318B** — đáng hỏi,
   vì nó nới trần ra gần gấp đôi.)

## Chạy

Dựng dữ liệu (một lần, không GPU):

```bash
PYTHONPATH=src python scripts/clean_data.py
PYTHONPATH=src python scripts/build_chunks.py     # -> 287.358 chunk
PYTHONPATH=src python scripts/build_gold.py       # weak gold, §15.1
```

Dựng dense index (cần GPU, 421s trên T4):

```bash
PYTHONPATH=src python scripts/build_dense.py --model intfloat/multilingual-e5-small
```

Đo lại recall: xem §15.9. Đẩy lên Kaggle:

```bash
./scripts/push_dataset.sh     # codebase -> Dataset legalqa-v10-src
./scripts/push_kernel.sh      # notebook -> kernel
```

> `kernels push` **RESET accelerator** về default (P100, sm_60 — `.half()` vỡ). Sau mỗi
> lần push phải vào UI: Settings → Accelerator → GPU T4 x2 → Save & Run All. Đã đo, xem
> comment cuối `scripts/push_kernel.sh`.

## Kiểm trước khi báo xong

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests
```

## Còn thiếu

- **System prompt** — `src/legalqa/prompt.py` cố tình chưa có. v9.2 `SYSTEM_PROMPT` chứa
  ba miếng vá triệu chứng ở ba tầng khác nhau (dặn model "không trình bày quá trình suy
  luận nội bộ", regex phạt CJK trong `verifier.py`, xoá câu trùng lặp). Ba miếng vá chỉ
  cùng cần thiết nếu có **một** nguyên nhân thượng nguồn. Đã đo (bảng dưới): nguyên
  nhân là base model chưa SFT, không phải thinking mode và không phải CJK drift. Nên
  **đừng** chép sang v10 câu "không trình bày quá trình suy luận nội bộ" (chữa bệnh
  không tồn tại — probe đo `mốc thinking 0` ở cả 6 cấu hình) lẫn regex phạt CJK
  (`CJK 0 ký tự` ở cả 6 cấu hình).
- **Qwen SFT** — chưa dựng.
- **Sanitizer câu trả lời cuối** — chưa dựng.

## Nghi can của bug generator (đã đo — v10-probe-and-eval v5, T4 x2)

Triệu chứng báo cáo: nhồi > 1 chunk thì model lặp từ và sinh ký tự Trung Quốc.
**Không tái lập được trên base model.** Đo 6 cấu hình (fp16/fp32 × raw/chat × 1/3/5
chunk), mỗi cấu hình sinh 1300 token:

| n_chunk | raw fp16 | raw fp32 | chat fp16 | chat fp32 |
|---|---|---|---|---|
| 1 | 72,7% | 72,7% | 98,2% | 98,2% |
| 3 | 71,4% | 71,4% | 55,4% | 55,4% |
| 5 | 71,4% | 69,6% | 3,2% | 3,2% |

Lặp **giảm** theo n_chunk, nặng nhất ở 1 chunk — ngược hẳn triệu chứng báo cáo. CJK 0
ký tự ở cả 6. Nên nguyên nhân nằm ở tầng TRÊN base model.

| nghi can | trạng thái |
|---|---|
| fp16 overflow | **loại** — max \|activation\| 62,9 / trần 65.504 (lượt đo trước); lượt này 11/12 cấu hình fp32 trả text TRÙNG TỪNG KÝ TỰ với fp16, `tensor không hữu hạn 0` ở cả 12. Chỗ lệch duy nhất là raw 5 chunk (prompt 7.159 token): 71,4% vs 69,6%, đuôi khác — trôi số học tích luỹ ở prompt dài nhất, cả hai đều đã suy biến sẵn nên không đổi kết luận |
| sampling | **loại** — `do_sample: false`, `repetition_penalty 1,1`, `no_repeat_ngram_size 5` đều đã bật |
| Qwen3 thinking mode | **loại** — `enable_thinking=False` được nhận, `mốc thinking 0` ở cả 6 cấu hình |
| nhét nhiều chunk | **loại** — lặp giảm 98,2% → 3,2% khi tăng 1 → 5 chunk |
| sinh CJK | **không tái lập** — 0 ký tự CJK ở cả 6 cấu hình |
| LoRA adapter đã SFT | **nghi can chính còn lại** — probe chỉ nạp base; `main` có commit `slice-aware loss`, prompt loss không mask sẽ dạy model sinh lại prompt = lặp |

Nhưng base model **cũng không dùng được như hiện tại**: 5/6 cấu hình chạy hết trần 1300
token mà không tự dừng, và nội dung là bám cụm rồi đếm số (`"...18 tuổi trở lên;"` tới
số 80; `"245, 246, 247, ..."`). Ca duy nhất tự dừng là chat 3 chunk (161 token) và cũng
là ca duy nhất dẫn được số Điều. Đây là hành vi base chưa SFT, không phải lỗi số học.

**Hệ quả:** Qwen SFT là điều kiện CẦN của v10, không phải bước tối ưu.

Cảnh báo về chỉ số: chat 5 chunk có lặp 5-gram 3,2% — sạch nhất bảng — nhưng output là
`"245, 246, 247, ..."`. Đếm số không sinh 5-gram lặp nào. Đừng dùng riêng lặp 5-gram
làm cổng chất lượng.
