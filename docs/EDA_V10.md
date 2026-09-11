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

## 14. Legal graph — hai loại cạnh

Đo trên 8.512 Document đã clean (`src/legalqa/graph.py`).

### 14.1 Tra văn bản: hai kênh, không một

| Kênh | Khoá | Phủ |
|---|---|---|
| Số hiệu | `Document.number` | 7.582 văn bản vào index |
| Tên + năm | tập từ không dấu của Slug | 100% Document có Slug, 96,4% Slug có năm |

Bắt buộc có kênh tên vì **9,2% Document (780) không có số hiệu**, trong đó có đúng cặp văn bản
mà câu 166937 cần: `LUẬT CÁC TỔ CHỨC TÍN DỤNG` (26667) và bản sửa đổi 2017 (157517). Khoá
(tập từ, năm) là duy nhất cho 98,6% Document (42 khoá đụng / 122 văn bản). Chấm điểm bằng
số từ thừa ít nhất nên `Luật Các tổ chức tín dụng sửa đổi` + 2017 khớp Slug
`Luat-sua-doi-...-2017` dù khác thứ tự từ.

### 14.2 Cạnh Amendment

| | |
|---|---|
| Văn bản sửa đổi nhận diện được | 349 (279 tiêu đề + 70 heading Điều) |
| Trong đó tra được mục tiêu | 305 = 87,4% |
| Văn bản thực sự sinh cạnh | 262 |
| Cặp (văn bản, Điều) bị sửa | 4.345 |
| Cạnh (khoản → Điều) | 5.299 |
| Điều bị >1 khoản sửa | 812 |
| Độ dài khoản sửa | p50 596 ký tự, p90 2.418, max 16.600 |

Buộc mục tiêu mở đầu bằng tên loại văn bản là cần thiết: thiếu ràng buộc này, 448 tiêu đề
khớp nhưng 109 trong đó trỏ vào tên **tổ chức** (`...ĐIỀU LỆ CỦA HỘI VẬT LIỆU XÂY DỰNG`) hoặc
mục tiêu số nhiều (`CÁC NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT...`).

Biên khoản cắt ở **khoản số liền sau**, không phải marker `^\d+\.` bất kỳ: phần sửa đổi trích
nguyên văn bản gốc nên mang hệ đánh số riêng — khoản 20 của 157517 chứa dòng `5. Tổng mức dư
nợ...`, cắt ở marker bất kỳ là mất đúng câu quan trọng nhất.

### 14.3 Vì sao không tự vá bản gốc

Corpus chỉ có **17 văn bản hợp nhất (0,2%)**, không có bản nào cho Luật Các TCTD — không có
đường tắt lấy sẵn text sau sửa đổi. Tự ghép khoản/điểm thì sinh ra text luật không tồn tại và
không thể phát hiện từ đầu ra. Quyết định: đưa **cả hai đoạn kèm nhãn quan hệ**
(`Evidence.relation`), để model tự đối chiếu.

Kiểm ca 166937 đầu-cuối: `resolve("Luật Các tổ chức tín dụng", 2010)` → 26667;
`amendments_of("26667","127")` → `[('157517','1','20')]`; Evidence sinh thêm mang nhãn
`Sửa đổi Điều 127 của LUẬT CÁC TỔ CHỨC TÍN DỤNG — khoản 20 Điều 1` và chứa trọn cả điểm b
lẫn khoản 5 mới.

### 14.4 Mổ 51 ca không tra được mục tiêu

Đo từng ca một, không suy đoán:

| Nguyên nhân | Số ca | Cứu được? |
|---|---:|---|
| Văn bản đích **vắng khỏi corpus** | 31 | Không. Giới hạn dữ liệu. |
| Số hiệu đích chỉ xuất hiện trong Slug văn bản khác | 7 | **Không nên** — xem dưới |
| Tiêu đề liệt kê nhiều mục tiêu | 8 | Có |
| Mục tiêu đơn mà vẫn trượt | 5 | 4 ca do `before_year` |

**7 ca số hiệu: bẫy đã tránh.** Ý tưởng "số hiệu không có trong index thì tìm Slug nào
chứa nó" nghe hợp lý, nhưng đo ra 4/7 ứng viên là **một bản sửa đổi KHÁC** của cùng văn
bản gốc (105589 và 152569 cùng sửa Thông tư 59/2021/TT-BQP, bản gốc thì không có trong
corpus) — vá kiểu đó sinh cạnh sửa-đổi-trỏ-sang-bản-sửa-đổi. Thêm nữa `02/2017/TT-BTC`
là chuỗi con của `102/2017/TT-BTC`. Bỏ.

**8 ca liệt kê: chỉ cắt khi song song.** Cắt ở mọi tên loại văn bản làm số ca đa mục tiêu
nổ từ 8 lên 65, phần lớn hỏng cùng một kiểu: `NGHỊ ĐỊNH ... HƯỚNG DẪN LUẬT CHĂN NUÔI` bị
hiểu là sửa cả Luật Chăn nuôi. Chỉ cắt ở tên loại đứng ngay sau `,` hoặc `và` → 19 ca đa
mục tiêu, và `LUẬT SỬA ĐỔI ... LUẬT ĐẦU TƯ CÔNG, LUẬT ĐẦU TƯ THEO PHƯƠNG THỨC ĐỐI TÁC
CÔNG TƯ, LUẬT ĐẦU TƯ, LUẬT NHÀ Ở, LUẬT ĐẤU THẦU...` ra đúng 6 luật đích.

**`before_year` dùng `>` chứ không `>=`:** năm trong Slug bản sửa đổi đôi khi là năm của
bản GỐC (`Phap-lenh-sua-doi-Phap-lenh-ngoai-hoi-2005`). Cùng năm vẫn an toàn vì mục tiêu
trỏ vào chính nó đã bị loại riêng.

### 14.5 Hai chốt chặn khớp bừa

Đo 83 mục tiêu tra qua kênh tên: số từ thừa p50 = 1, p90 = 5, max 11. Đuôi phân bố toàn
rác cùng một dạng — mảnh cắt cụt còn 2 từ (`Thông tư số`, `Quyết định số`) là tập con của
hàng trăm Slug nên khớp vào văn bản bất kỳ.

- **Truy vấn < 3 từ thì trả None**, trừ khi có năm — năm thu hẹp đủ để 2 từ vẫn xác định được.
- **Bỏ ứng viên thừa > 6 từ**: `LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA` từng khớp vào
  `Nghị định 51/2002 hướng dẫn Luật Báo chí` với 11 từ thừa.

### 14.6 Mục tiêu khai theo Điều, không theo văn bản

Lỗi này chỉ lộ ra khi **nhìn graph bằng mắt**: Luật Giao thông đường bộ đứng thứ tư trong
bảng bị-sửa-nhiều-nhất với 87 Điều, trong đó 223 cạnh đến từ một văn bản duy nhất —
`LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 37 LUẬT CÓ LIÊN QUAN ĐẾN QUY HOẠCH`. Văn bản đó
có 37 Điều, **mỗi Điều sửa một luật khác nhau**; lấy một mục tiêu chung cho cả văn bản
(heading Điều 1 = Luật GTĐB) thì toàn bộ 37 Điều đổ vào một luật.

Đo quy mô: **72 văn bản** khai mục tiêu khác nhau theo từng Điều, phủ 3.308/5.760 cạnh.

> **Đính chính.** Con số "57% cạnh gán nhầm" ở bản trước là sai đại lượng: 3.308 là số
> cạnh **thuộc nhóm có nguy cơ**, không phải số cạnh **sai** — phần lớn trong nhóm vẫn
> trúng vì mục tiêu chung tình cờ trùng mục tiêu của Điều. Dựng cả hai bản rồi so từng
> cạnh: 4.013 cạnh giống nhau, 1.129 bộ ba `(nguồn, Điều nguồn, khoản)` **đổi hẳn văn bản
> đích** = **19,6%** gán nhầm thật, 169 bộ ba bị bỏ vì tra trượt. Sai rải trên 72 văn bản
> nguồn, đỉnh là 158 cạnh (Luật GTĐB), không phải một văn bản gánh hết.

Sửa: mục tiêu lấy từ heading của chính Điều đang đọc; chỉ Điều nào không tự khai mới
dùng mục tiêu chung ở tiêu đề. Và khi Điều có khai mà tra trượt thì **bỏ hẳn**, không rơi
về mục tiêu chung — gán nhầm sang văn bản khác tệ hơn mất cạnh.

Kết quả: cạnh 5.760 → 5.299, cặp (văn bản, Điều) 4.183 → **4.345** (tăng vì cạnh phân tán
đúng chỗ thay vì dồn một chỗ). Luật GTĐB 87 → 43 Điều, 223 → 43 cạnh từ văn bản đó. Văn
bản "37 luật quy hoạch" giờ trỏ vào 24 văn bản khác nhau.

### 14.7 Mốc Điều giả cắt đôi Điều thật

Lỗi nằm ở `parse_articles`, không ở graph, nhưng chảy thẳng vào graph. Câu sửa đổi bị
wrap xuống dòng đẩy chữ "Điều" ra đầu dòng:

```
2. Sửa đổi, bổ sung một số khoản của
Điều 10 như sau:
```

`DIEU_RE` nhận `Điều 10` làm mốc mở Điều mới. Hậu quả kép: Điều 1 **bị cắt cụt** mất sạch
khoản từ đó trở đi, và sinh ra một Article rác mang số Điều **của bản gốc**. Văn bản
102693 (`LUẬT SỬA ĐỔI LUẬT KIỂM TOÁN NHÀ NƯỚC`) đẻ ra 4 Điều rác 10/11/64/69.

Đo trên 159.137 Article, ký tự ngay sau số Điều:

| sau số Điều | số ca | phán quyết |
|---|---:|---|
| `.` / `:` / `-` | 157.591 | Điều thật |
| xuống dòng, hoặc chữ HOA (`Điều 34 Trách nhiệm của...`) | 839 | Điều thật, thiếu dấu chấm |
| chữ **thường** (`Điều 10 như sau:`) | 377 | **mốc giả** — câu bị wrap |
| `,` / `;` (`Điều 71, điểm b khoản 4 Điều 103`) | 244 | **mốc giả** — liệt kê |
| `(` (`Điều 252 (tội chiếm đoạt chất ma túy)`) | 12 | **mốc giả** — liệt kê |

Loại 633 mốc giả: Article 159.137 → **158.504**. Không đụng tới 2.950 Article nằm trong
vùng ngoặc kép chưa đóng (214 Document) — đếm ngoặc cộng dồn lệch một cái là hỏng cả đuôi
văn bản, rủi ro cao hơn lợi ích.

### 14.8 Ba lỗi gán mục tiêu còn lại

**a. Heading bị wrap cắt cụt.** `Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số`
— mất số hiệu. Chuỗi `"Nghị định số"` thành tập từ `{nghi, dinh, so}`, đủ 3 từ để qua
chốt chặn §14.5 và khớp trúng Slug chứa `xo-so` = **Nghị định kinh doanh xổ số**. Hai vá:
`_DUOI_SO_RE` phải cắt được cả khi `"số"` đứng cuối chuỗi, và chuỗi mục tiêu nghèo chữ
được coi là **không khai** (rơi về mục tiêu chung ở tiêu đề) chứ không phải khai sai.

Vá `_DUOI_SO_RE` làm lộ một lỗi có sẵn: `\s+số\s+.*$` cắt ngay ở `"một số điều"`, biến
`"Nghị định quy định chi tiết một số điều của Luật Hóa chất"` thành `"Nghị định quy định
chi tiết một"` — truy vấn ngắn đi thì khớp bừa dễ hơn (§14.5). Ràng lại: phần sau `"số"`
phải mở đầu bằng chữ số.

**b. Điều không khai mục tiêu, khoản thì có.** `Điều 12. Bãi bỏ một số văn bản thuộc lĩnh
vực mỹ phẩm` không nêu văn bản nào, nhưng khoản 2 nêu: `Bãi bỏ một số quy định của Thông
tư số 06/2011/TT-BYT`. Kế thừa mục tiêu của Điều 1 là gán sang hẳn văn bản khác.

Chỉ nhận dạng **số hiệu** đứng sau `"của"`. Dạng tên bắt luôn cả văn bản được nhắc vì lý
do khác (`theo mẫu tại Nghị định X`): đo được 106 ca lệch, so với 19 ca ở dạng số hiệu.
Và chỉ dùng khi Điều **không** khai — Điều có khai thì số hiệu trong khoản thường là chú
thích lịch sử: `khoản 5 Điều 10 (được sửa đổi theo ... của Thông tư số 26/2017/TT-NHNN)`,
tin nó là gán nhầm ngược lại.

**c. Rải một khoản ra mọi mục tiêu.** Văn bản khai nhiều mục tiêu ở tiêu đề mà Điều không
tự khai: cũ gán khoản đó cho *tất cả* mục tiêu. Chỉ đúng khi cả phần sửa đổi gói trong
**một** Điều; nhiều Điều thì mỗi Điều ứng một mục tiêu riêng, rải là chắc chắn sinh cạnh
sai. 254 khoản bị chặn.

Kết quả cuối, sau cả vá `parse_articles` lẫn ba vá trên:

| | trước phiên | sau |
|---|---:|---:|
| Article toàn corpus | 159.137 | **158.504** |
| cạnh Amendment | 5.299 | **5.231** |
| cặp (đích, Điều) | 4.345 | **4.306** |
| văn bản nguồn / đích | — | 267 / 367 |
| đích nhận nhiều nhất | 223 | **139** |

Ca `26667` Điều 127 (Luật Các tổ chức tín dụng, sửa bởi khoản 20 Điều 1 văn bản 157517)
giữ nguyên qua cả bốn vòng vá.

### 14.9 Công cụ xem

`scripts/dump_graph.py` xuất graph ra `window.GRAPH` (~5,4 MB, 495 văn bản có cạnh) cho
một trang xem: tìm văn bản, liệt kê Điều bị sửa, mở ra nguyên văn khoản sửa, bấm sang
văn bản nguồn. Lỗi 14.6 tìm ra từ bảng xếp hạng trên trang này chứ không từ test.
