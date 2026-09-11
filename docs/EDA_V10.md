# EDA v10 — knowledge base và trần retrieval

Chạy lại từ đầu trên dữ liệu local ngày 11/09/2026. Script: `scripts/eda/` (xem cuối file).
Mọi con số dưới đây đo được, không trích lại từ tài liệu v8/v9 cũ.

## 1. Bộ câu hỏi

| Bộ | n | Q (từ, mean/p50/p90) | A (từ, mean/p50/p90/max) |
|---|---:|---|---|
| `train.json` | 7.000 | 19,7 / 19 / 28 | 347 / 312 / 577 / 2.435 |
| `public-official.json` | 1.000 | 19,3 / 19 / 28 | **toàn bộ null** |
| `warmup.json` | 500 | 19,1 / 19 / 27 | 343 / 303 / 556 / 1.829 |

Schema cả ba: `{question_id: {question, answer}}`. Không có trường `question_type`, `context_id` hay bất kỳ
nhãn retrieval nào. **BTC không cấp supervision cho retrieval.**

Reference answer: mean 7,5 câu, 11,5 dòng. Là văn bản trích luật có xuống dòng, không phải đoạn văn liền.

### Trùng lặp giữa các bộ

| Cặp | Số câu hỏi trùng nguyên văn |
|---|---:|
| train ∩ warmup | **387** |
| public ∩ warmup | **40** |
| train ∩ public | 1 |

40 câu trong public test có câu hỏi y hệt trong warmup — mà warmup **có** Reference answer. Đây là dữ liệu
BTC phát hợp lệ, không phải leak ngoài luồng.

### Noise trong Reference answer

| Mẫu | Tỷ lệ / 7.000 |
|---|---:|
| `(Hình từ Internet)` | **18,5%** |
| `tại đây` (link reference) | 0,9% |
| `[...]` (trích luật lược bỏ — **hợp lệ, giữ**) | 1,0% |
| kết thúc bằng `...` (cắt cụt) | 0,3% |

18,5% Reference answer chứa caption ảnh crawl. Model học từ đó sẽ sinh ra caption — giải thích trực tiếp
hiện tượng hallucination `(Hình từ Internet)` từng ghi nhận trong submission cũ.

### Bất đối xứng citation giữa câu hỏi và câu trả lời

| Mẫu | Trong ANSWER | Trong QUESTION |
|---|---:|---:|
| `Điều N` | **88,2%** | **0,1%** |
| `khoản N` | 58,4% | 0,1% |
| Nghị định | 31,1% | 0,3% |
| Thông tư | 26,6% | 0,2% |
| Luật … | 26,4% | 0,2% |

Đây là phát hiện có hệ quả trực tiếp: **câu hỏi hầu như không chứa Citation nào**, nhưng câu trả lời gần
như luôn phải viện dẫn Điều/khoản. Mọi cơ chế "citation boost trên query" trong pipeline cũ đang boost trên
một tín hiệu tồn tại ở 0,1% truy vấn.

Câu hỏi mở đầu phổ biến: `Hồ sơ đề…` (82), `Cơ quan nào…` (62), `Trách nhiệm của…` (52), `Quy định về…` (48),
`Trường hợp nào…` (44), `Ai có quyền…` (44). Dạng tra cứu thủ tục, không phải suy luận nhiều bước.

## 2. Corpus

| Chỉ số | Giá trị |
|---|---:|
| File | 8.532 |
| Có passage | 8.512 (20 rỗng) |
| Tổng dung lượng từ | **72.825.207 từ** |
| Độ dài Document (từ) | mean 8.556 · p50 **4.828** · p90 18.419 · max **1.242.409** |
| Document > 2.048 từ | 80,7% |
| Document > 8.192 từ | 29,5% |
| Schema `id\|link\|name\|passage` | 7.407 |
| Schema `id\|link\|passage` (thiếu `name`) | 1.125 |
| Nguồn link | 100% thuvienphapluat.vn |

`name` là **Slug URL không dấu**, ví dụ `Thong-tu-17-2022-TT-BGTVT-sua-doi-...-522401`. Không dùng trực tiếp
được — phải parse ra Document title có dấu.

### Cấu trúc bên trong Passage

| Dấu hiệu | Tỷ lệ |
|---|---:|
| có `Điều N.` | 84,1% |
| có `N.` đầu dòng (khoản) | 95,8% |
| có `Chương …` | 41,3% |
| có `Mục N` | 13,8% |

Số Điều/Document (mẫu 2.000): mean 19,1 · p50 12 · p90 40 · max 838. **313/2.000 Document không có Điều nào**
— chunking theo Điều phải có đường lui.

### Legal relation — phổ biến, không phải tín hiệu hiếm

| Quan hệ | Tỷ lệ Document chứa |
|---|---:|
| căn cứ | 93,1% |
| hướng dẫn | 82,7% |
| bổ sung | 78,7% |
| sửa đổi | 68,7% |
| thay thế | 49,7% |
| hết hiệu lực | 13,8% |
| bãi bỏ | 10,0% |

Xác nhận lại nhận định v9.1 bằng số đo trực tiếp: **relation count không được dùng làm gate** vì gần như mọi
Document đều có. Chỉ dùng graph khi Citation trỏ đích danh một văn bản cụ thể.

## 3. Reference answer có trích nguyên văn corpus không?

Lấy mẫu 295 câu train, cắt một đoạn **12 từ liên tiếp ở giữa** Reference answer, tìm nguyên văn trong corpus:

**224/295 = 75,9% khớp nguyên văn.**

Trong 224 câu đó, số Weak gold document mỗi câu: 1 doc (147 câu), 2 doc (48), ≥3 doc (29). Phần đuôi dài là
văn bản bị sao chép lại qua các bản sửa đổi.

Hệ quả: bài toán này **chủ yếu là trích xuất có tái tổ chức**, không phải sinh tự do. Nếu Evidence đúng, model
chỉ cần trích lại đúng và sắp xếp — điều một model 2B thừa sức làm.

## 4. Trần retrieval — số quyết định kiến trúc

BM25 (k1=1.5, b=0.75), truy vấn = câu hỏi thô, đo trên 224 câu có Weak gold document.

### Doc-level (index cả Document như một đơn vị)

| k | 1 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| doc recall | 22,3% | 36,6% | 42,0% | **50,9%** | 59,8% | 71,0% | 76,3% |

MRR 0,318 · rank trung vị của gold = 10.

### Chunk-level (index theo Chunk, gộp ngược về Document)

| Chunk (từ) / stride | chunks | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| 128 / 128 | 571.237 | **55,4%** | **83,9%** | 91,5% |
| 256 / 256 | 287.741 | **55,4%** | 83,0% | 89,7% |
| 256 / 128 | 564.698 | 53,6% | 83,5% | **92,4%** |
| 384 / 384 | 193.186 | 54,9% | 79,5% | 91,1% |
| 512 / 512 | 145.999 | 53,6% | 77,7% | 86,6% |
| 768 / 768 | 98.680 | 50,4% | 75,4% | 84,8% |

**Chỉ đổi đơn vị index từ Document sang Chunk 256 từ: recall@1 nhảy 22,3% → 55,4% (×2,5), recall@10 nhảy
50,9% → 89,7%.** Không thêm dense, không thêm reranker, không thêm graph.

Nguyên nhân: Document median 4.828 từ, chuẩn hoá độ dài của BM25 làm loãng tín hiệu khớp cục bộ. Câu hỏi chỉ
19 từ và nhắm vào một điều khoản duy nhất trong văn bản hàng nghìn từ.

Chunk 256/256 là điểm cân bằng: recall@1 ngang 128 từ nhưng **chỉ một nửa số chunk** (287k vs 571k) — quan
trọng vì chi phí embed dense trên T4 tỷ lệ thuận với số chunk.

## 5. Kết luận rút ra

1. **Nút thắt là retrieval, không phải generation.** Trần doc-level 50,9%@10 của pipeline cũ đặt trần cứng lên
   mọi thứ phía sau. Selector giỏi đến mấy cũng không chọn được câu trả lời từ evidence sai.
2. **Chunk là thay đổi đòn bẩy cao nhất và rẻ nhất**: không GPU, không train, ×2,5 recall@1.
3. **Chunk để tìm ≠ Evidence để sinh.** Chunk 256 từ tối ưu recall nhưng cắt ngang điều khoản. Phải mở rộng
   Chunk trúng ra ranh giới Điều trước khi đưa vào prompt (small-to-big).
4. **Citation boost trên query là code chết** — 0,1% query có citation.
5. **Graph không được gate bằng relation count** — 68–93% Document có relation.
6. **Phải làm sạch train trước khi SFT**: 18,5% Reference answer có caption ảnh.
7. **40 câu public trùng warmup** có sẵn Reference answer.

## 6. Tái lập

```bash
python scripts/eda/flatten_corpus.py      # corpus -> 1 dòng/Document
python scripts/eda/qa_stats.py            # mục 1
python scripts/eda/corpus_stats.py        # mục 2
python scripts/eda/verbatim_gold.py       # mục 3, sinh weak gold
python scripts/eda/bm25_recall.py         # mục 4
```

## 11. Wrap mềm của trình crawl — `\r\n\n`

Passage là text crawl từ HTML, giữ nguyên ngắt dòng **trình bày** (dòng p50 = 34 ký tự,
đúng bề rộng cột hiển thị) chứ không phải ngắt dòng **ngữ nghĩa**. Đo trên 797 Document:

| Chỉ số | Giá trị |
|---|---:|
| Document chứa `\r` | 797/797 = 100% |
| Số lần `\r\n\n` xuất hiện | 427.420 |
| `\r\n\n` nối **giữa câu** | 93,9% |
| `\n\n` thuần nối giữa câu | 50,7% |

Kết luận: `\r` là dấu vết wrap mềm, gỡ được **xác định**. `\n\n` thuần thì mơ hồ
(một nửa là ngắt đoạn thật, dùng để phân khoản) nên **giữ nguyên**.

`dewrap_passage()` chạy trong `load_document()`, trước `parse_articles()` — mọi offset
Article/Chunk/Evidence đều trỏ vào Passage sau bước này.

### Tác động đo được (600 Document)

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Tiêu đề Điều bị cắt đôi (đo nghiêm: dòng sau bắt đầu chữ thường) | 62,3% | **0,2%** |
| Số dòng | 623.526 | 304.873 (−51,1%) |
| Số từ | 5.519.099 | 5.519.099 (không đổi) |
| Số Điều nhận ra | 10.951 | 12.108 |

26 ca còn sót đa số là trích dẫn `Điều 136 của Bộ luật này...` trong thân văn bản,
không phải heading.

Toàn corpus: Document không có Điều 15,9% → **14,9%**. Số Chunk giữ nguyên 287.741.

### `Document.title` và `Document.number`

`TITLE_LINE_RE` trước đây thiếu `re.MULTILINE` nên chỉ khớp đầu chuỗi — mà Passage
luôn mở đầu bằng tên cơ quan ("BỘ LAO ĐỘNG - THƯƠNG BINH VÀ XÃ HỘI"), nên **100%
Document rơi về đường lui Slug không dấu**. Đây chính là nguồn rò chuỗi kiểu
`Thong-tu-27-2017-TT-BLDTBXH-dao-tao-lien-thong-...` vào prompt.

Sau khi sửa (MULTILINE + ghép dòng chủ đề viết hoa ngay sau loại văn bản), đo trên
800 Document: title lui về Slug **14,2%**.

Thêm `Document.number` (số hiệu `27/2017/TT-BLĐTBXH`) — bắt được **91,4%**. Đây là
khoá để đối chiếu Citation với Document trong legal graph (ADR-0002: chỉ mở rộng graph
khi Citation gọi tên một văn bản cụ thể).

### Dòng chỉ chứa số khoản

Sau khi gỡ `\r`, còn một dạng phân mảnh nữa dùng `\n\n` thuần: dòng chỉ chứa mỗi
dấu đánh số rồi xuống dòng.

```
Điều 32. Trách nhiệm của người bị khiếu nại
\n\n 1. \n\n Căn cứ vào quyết định giải quyết khiếu nại...
```

Đo: **3.606/334.524 = 1,1%** số dòng không rỗng. Một dấu đánh số đứng một mình không
bao giờ là đoạn riêng, nên `LONE_MARKER_RE` nối chắc chắn đúng.

Đánh đổi đã đo: trong 1.500 Document có 11 ca dạng `số khoản` + `Điều N`. 9/11 là trích
dẫn trong khoản sửa đổi (`23.` + `Điều 31 được sửa đổi, bổ sung như sau`) — gộp lại là
đúng, vì để tách chúng tạo **heading giả** cắt vụn Điều thật. 2/11 là heading thật bị
mất. Toàn corpus: số Điều tách được 159.242 → 159.136 (−106). Chấp nhận: bớt heading
giả đáng giá hơn giữ 2 heading thật trên 159 nghìn.

## 12. Chạy clean toàn bộ — `scripts/clean_data.py`

Materialize sẵn để bước index/SFT không parse lại 8.532 file JSON.
Đầu ra vào `data/clean/` (nằm trong `.gitignore`).

```
== CORPUS ==                              == TRAIN ==
  file                8.532                 vào                 7.000
  Document ghi ra     8.512                 answer bị sửa       1.495 = 21,4%
  Điều tách được    159.136                 loại (ngắn/cụt)        54
  Document không Điều 1.271 = 14,9%         trùng question         17
  có số hiệu          7.732 = 90,8%         giữ                 6.929
  thiếu Slug          1.105 = 13,0%
  wrap mềm \r  4.407.012 -> 0
```

`corpus.jsonl` 482,7 MB — mỗi dòng một Document: `doc_id`, `title`, `number`, `slug`,
`link`, `passage` (đã gỡ wrap), `articles` (`[số, heading, start, end]`).

`train_sft.jsonl` 6.929 mẫu — khớp ngưỡng nghiệm thu 6.900–6.950 của V10_BUILD_SPEC.
Trùng question thì giữ bản answer dài nhất (bản ngắn thường là trích lược).

Xác minh trên 179 Document lấy ngẫu nhiên: 0 offset Điều sai, 0 ký tự `\r` còn sót.

### Dòng trống và dòng bị lược

Sau hai vòng trên, đo lại toàn corpus thấy phần rác lớn nhất vẫn còn nguyên:

| Loại | Số lượng | Tỷ lệ |
|---|---:|---:|
| Dòng **trống** | 5.703.199 | **59,3% số dòng** |
| Dòng chỉ chứa `...` | 85.602 | 0,9% |
| Dòng thừa space đầu/cuối | 139.629 | 1,5% |
| Khối lặp nguyên văn nửa Passage | 0 | 0% |

Dòng trống là trình bày HTML thuần, không mang thông tin → bỏ. Dòng `...` là chỗ
thuvienphapluat lược nội dung cho người chưa trả phí → bỏ, để lại thì model học sinh
ra `...` giữa câu trả lời.

**Xuống dòng đơn thì GIỮ.** 95,8% Document dùng đầu dòng để phân khoản; đó là cấu trúc
model cần để trích đúng khoản. Xoá hết `\n` là phá chính thứ khiến small-to-big hoạt động.

Kết quả: 343.156.199 → **336.323.568** ký tự, dòng trống 5,7 triệu → **0**.

### Xuống dòng còn lại KHÔNG phải rác

40,0% số dòng ngắn hơn 25 ký tự, và 607/8.512 Document (7,1%) có hơn 60% số dòng như
vậy. Đó là **ô bảng** HTML, mỗi ô một dòng — ví dụ `TCVN-13268-1-2021` là bảng phân cấp
sâu bệnh, các dòng `Sâu cuốn lá nhỏ` / `Cnaphalocrocis medinalis` / `50 con/m2` là ba ô
khác nhau của cùng một hàng. Nối chúng lại là làm hỏng dữ liệu, không phải làm sạch.

Tương tự, 411/8.512 Document (4,8%) chứa tổng cộng 25.935 khối nội dung đã bị lược mất
từ nguồn. Đó là lỗ hổng của corpus, không sửa được ở tầng clean.

### Slug lấy lại được từ link

EDA ban đầu ghi "1.125/8.532 Document không có Slug" như mất mát không cứu được — sai.
`link` có ở **100% Document** và chứa đúng chuỗi Slug đó (`.../Ten-van-ban-<id>.aspx`).
Đối chiếu 6.954 Document có cả hai: khớp hoàn toàn.

`slug_from_link()` lấp đủ 1.105 Document thiếu trường `name`. Hệ quả: Document title
rỗng (`"Văn bản 100050"`) **11,0% → 0,0%**.

## 13. Chunk + Evidence trên corpus đã clean

`scripts/build_chunks.py`, size 256 / stride 256, chạy trên `data/clean/corpus.jsonl`.

| | |
|---|---:|
| Document | 8.512 (0 doc không sinh chunk) |
| Chunk | 287.358 |
| chunk/Document | p50 19, max 4.850 |
| từ/chunk | p10 256, p50 256, p90 256 |

Chunk giảm 287.741 → 287.358 (−383) so với lần đo trên dữ liệu chưa bỏ dòng trống:
85.602 dòng `...` bị xoá làm hụt token từ. Assert `passage[start:end] == text` xanh
287.358/287.358 — offset trỏ đúng vào Passage đã clean.

### Điều mà Chunk chạm vào

159.137 Điều: p50 **187 từ**, p90 638 từ, max 189.311.
Chỉ 5,6% Điều dài hơn ngân sách 900 từ, nhưng Điều dài chứa nhiều Chunk hơn nên
chúng hút **46,5% tổng Chunk**. 25,4% số Điều dài đó là Điều CUỐI của Document —
Điều cuối nuốt luôn phần phụ lục/biểu mẫu phía sau vì `parse_articles` lấy
`end = start của Điều kế tiếp`, Điều cuối thì `end = len(passage)`.

### Evidence sinh từ Chunk đó

| Tình huống | Chunk | % |
|---|---:|---:|
| Lấy trọn Điều (span ≤ 900 từ) | 119.337 | 41,5% |
| Span > 900 từ → cắt cửa sổ, **vẫn giữ** `article_number` + heading | 133.702 | 46,5% |
| Document không có Điều nào → đường lui | 25.720 | 9,0% |
| Chunk nằm trước Điều đầu tiên → đường lui | 8.599 | 3,0% |

từ/Evidence p10 394, p50 873, p90 929. Số Điều gộp p50 1, max 26.

**Đường lui = 34.319 = 11,9%**, đúng bằng 9,0% + 3,0%. Con số 1,3% đo trước đó là
**artefact lấy mẫu**: lấy 1 chunk ngẫu nhiên mỗi Document thì nghiêng về Document CÓ Điều
và về vị trí giữa văn bản, bỏ sót cả 1.271 Document không Điều lẫn phần đầu mỗi văn bản.
Không phải hồi quy.

### 13.1 Cắt phụ lục khỏi Điều cuối

`parse_articles` cho Điều cuối `end = len(passage)` nên nó nuốt trọn phụ lục/biểu mẫu
phía sau. `_cut_phu_luc` dừng biên tại marker phụ lục đứng trọn một dòng ngắn.

Chọn ngưỡng bằng cách áp quy tắc lên **151.896 Điều GIỮA** — biên của chúng vốn đã đúng
nên mọi lần dính đều là dương tính giả:

| Quy tắc | Dương tính giả trên Điều giữa | Cắt được / 2.282 Điều cuối quá dài |
|---|---:|---:|
| marker ở đầu dòng | 0,81% | 83,7% |
| + giữ ≥ 50 từ, bỏ ≥ 200 từ | **0,52%** | **83,1%** |

Điều cuối quá dài: trung bình 6.920 → 817 từ.

### 13.2 Ảnh hưởng lên Evidence

| Tình huống | Trước | Sau |
|---|---:|---:|
| Lấy trọn Điều | 41,5% | **42,4%** |
| Span > 900 từ → cắt cửa sổ | 46,5% | **29,8%** |
| Document không có Điều | 9,0% | 9,0% |
| Chunk trước Điều đầu tiên | 3,0% | 3,0% |
| Chunk nằm trong phụ lục | *(gán nhãn Điều cuối — SAI)* | **15,9%** |

Số Điều tách được không đổi (159.137), từ/Evidence p50 873 → 869.

Đường lui 11,9% → 27,9%. Tăng là **đúng**: 45.785 chunk nằm trong phụ lục trước đây
mang `article_number` của Điều cuối, một nhãn sai. Chúng vẫn được index, chỉ không còn
được quảng cáo là điều khoản.
