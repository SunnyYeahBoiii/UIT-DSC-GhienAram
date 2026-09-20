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

### 14.9 Phủ — graph đã chạm hết văn bản chưa?

Hai câu hỏi khác nhau, tách ra mới trả lời được:

**Tra cứu** phủ 100%: 8.512/8.512 Document vào được index theo tên, 7.582 (89,1%) có
thêm số hiệu. Không văn bản nào vô hình với `resolve`.

**Cạnh Amendment** chỉ tồn tại cho văn bản SỬA ĐỔI — 350/8.512 = 4,1% corpus. Bản gốc
không sinh cạnh là đúng cấu trúc, không phải thiếu sót.

| Vòng vá | Văn bản sửa sinh được cạnh | Cạnh | Cặp (đích, Điều) |
|---|---|---|---|
| sau `a78adee` | 249/350 = 71,1% | 5.231 | 4.306 |
| + khoản đảo ngữ (`Điều 4 được sửa đổi…`), 871 khoản, mở khoá 12 văn bản | 261 = 74,6% | 5.918 | 4.816 |
| + Điều đích khai ở heading (6 văn bản) + Điều không chia khoản (49 Điều) | **265 = 75,7%** | **5.971** | **4.851** |

Cuối cùng: 5.971 cạnh, 4.851 cặp, 287 văn bản nguồn, 390 văn bản đích.

85 văn bản sửa đổi còn lại không sinh cạnh, phân loại:

- **~41 — văn bản đích KHÔNG có trong corpus.** Trần cứng, vá kiểu gì cũng không qua
  được. 102693 sửa Luật Kiểm toán nhà nước 81/2015/QH13 và Luật Tố tụng hành chính
  93/2015/QH13, cả hai đều vắng; 105589 sửa Thông tư 59/2021/TT-BQP, cũng vắng.
- **24 — không khoản nào mang động từ sửa đổi.**
- **20 — khoản có nhưng không nêu Điều nào.** Gồm 15586 (nhiều mục tiêu, cố ý chặn
  fan-out ở §14.8c: đoán Điều i ↔ mục tiêu i là giả thiết chưa đo).

### 14.9 Công cụ xem

`scripts/dump_graph.py` xuất graph ra `window.GRAPH` (~5,4 MB, 495 văn bản có cạnh) cho
một trang xem: tìm văn bản, liệt kê Điều bị sửa, mở ra nguyên văn khoản sửa, bấm sang
văn bản nguồn. Lỗi 14.6 tìm ra từ bảng xếp hạng trên trang này chứ không từ test.

## 15. Retrieval — BM25 và cross-encoder rerank

### 15.1 Weak gold: không có nhãn retrieval nào trong dữ liệu thi

`train.json` chỉ có `question` + `answer`. Không có "câu này trúng văn bản nào". Muốn đo
retrieval phải tự suy ra. Hai kênh, cố ý **không** dùng BM25 — gold suy từ đầu ra của
chính retriever đang đo sẽ tự khẳng định nó.

| kênh | cách suy | câu |
|---|---|---:|
| `nguyen_van` | 12 từ giữa answer, tìm nguyên văn trong text Chunk | 5.278 |
| `so_hieu` | regex số hiệu trong answer → tra `Document.number` | — |
| **hợp lại** | | **6.250/7.000 = 89,3%** |

Chi tiết: `nguyen_van+so_hieu` 2.921 · `nguyen_van` 2.357 · `so_hieu` 972.
Số doc/câu: 1→4.153, 2→1.187, 3→395, ≥4→304 (bản gốc + bản sửa đổi chép lại nguyên khoản).

Lấy 12 từ **giữa** answer chứ không phải đầu: đầu answer là câu dẫn người viết soạn
("Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP...") — không có trong corpus, khớp vào
đó là khớp vào văn phong.

Gold ở mức **Document**, không phải Điều, và tính "trúng ít nhất một" → chỉ số **nới tay**.
Dùng để so sánh các phương án với nhau thì tốt; đọc như xác suất retrieval đúng thì sai.

### 15.2 Hai bệnh sau một con số "trượt@10"

Thứ hạng gold trong pool 3.000 chunk (n=1.000):

| ≤10 | 11-50 | 51+ | không với tới |
|---:|---:|---:|---:|
| 86,7% | 9,3%* | 3,6%* | **0,4%** |

BM25 gần như luôn **tìm thấy** văn bản đúng (99,6%), chỉ **xếp sai hạng**. Hai bệnh cần
hai thuốc khác nhau: không với tới cần biểu diễn khác (dense); xếp sai hạng cần chấm lại
(rerank). Đo được 0,4% bệnh một, ~11% bệnh hai.

Hệ quả: thay BM25 bằng bi-encoder mua tối đa 0,4 điểm recall với giá 568M params. Còn
15,2 điểm (R@5 81,0% → R@50 96,2%) nằm sẵn trong top-50, chỉ cần sắp lại. → ADR: rerank,
không dense.

### 15.3 Cross-encoder rerank — số đo

`AITeamVN/Vietnamese_Reranker` (XLM-R large, 567.755.777 params), **zero-shot**, pool 50,
n=500, T4x2:

| | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 56,4% | 74,6% | 81,0% | 88,4% | 94,0% | 96,2% |
| +rerank | 75,2% | 89,4% | 92,0% | 94,2% | 96,0% | 96,2% |
| chênh | +18,8 | +14,8 | **+11,0** | +5,8 | +2,0 | **0,0** |

Trần (gold có trong pool 50) = 96,2%. Rerank ăn khoảng trống (trần − BM25):
k=1 **47%** · k=5 **72%** · k=10 **74%**.

Ba phép kiểm nội tại:

- `chênh R@50 = 0,0` **đúng bằng 0** — rerank sắp lại pool 50, không thêm ứng viên, nên
  tập 50 doc đầu không đổi. Lệch khỏi 0 là code sai.
- `+rerank R@50 = trần` — bám khít.
- rerank@1 (75,2%) > BM25@3 (74,6%): ứng viên số 1 sau chấm lại tốt hơn ba ứng viên đầu
  của BM25 gộp. Với generation đó là khác biệt giữa nhồi 3 Evidence và nhồi 1.

Chi phí 1,58 s/câu → 1.000 câu ≈ 26 phút. Params: Qwen 3B (3,09B) + reranker (0,568B)
= 3,66B, còn trống 340M dưới trần 4B.

### 15.4 Chỗ nghẽn đã dịch từ xếp hạng sang trần

Phân rã 500 câu ở k=5:

| | câu | % |
|---|---:|---:|
| trúng@5 sau rerank | 460 | 92,0% |
| trong pool nhưng rerank chưa vớt | 21 | 4,2% |
| gold không trong pool 50 | 19 | **3,8%** |

Hai nhóm trượt gần bằng nhau. Trước rerank, 100% khoảng trống là lỗi xếp hạng; sau rerank
gần một nửa là lỗi pool. Nên bước tiếp theo là quét pool (nâng trần), không phải đổi
reranker.

### 15.5 Kiểm trên tập độc lập — warmup

`warmup.json` 500 câu có Reference answer, KHÔNG nằm trong train. Gold dựng bằng đúng
quy trình 15.1: 456/500 = 91,2% (`nguyen_van+so_hieu` 215 · `nguyen_van` 179 · `so_hieu` 62).

| n=456 | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 54,2% | 74,3% | 83,1% | 89,5% | 92,5% | 94,3% |
| +rerank | 72,1% | 88,2% | 90,1% | 92,5% | 94,1% | 94,3% |
| chênh | +18,0 | +13,8 | +7,0 | +3,1 | +1,5 | **0,0** |

R@1 +18,0 so với +18,8 trên train — trùng khít. R@5 +7,0 so với +11,0 thấp hơn, nhưng
KHÔNG phải vì reranker yếu đi: khoảng trống nhỏ hơn.

| tập | trần | BM25@5 | khoảng trống | ăn |
|---|---:|---:|---:|---:|
| train | 96,2% | 81,0% | 15,2 điểm | 72% |
| warmup | 94,3% | 83,1% | 11,2 điểm | 63% |

Chuẩn hoá theo khoảng trống thì hai tập cùng cỡ. `chênh R@50 = 0,0` lặp lại đúng trên
tập thứ hai. Kết luận 15.3 không phải đặc tính của train.

### 15.6 Quét pool — chốt pool 50

Pool 20 và 100 chạy n=300 (cùng tập: BM25 R@1/R@3/R@5 giống hệt 56,3/75,0/80,0), pool 50
chạy n=500 nên chỉ 20 <-> 100 so trực tiếp được.

| pool | trần | +rerank R@5 | ăn@5 | s/câu | 1.000 câu |
|---:|---:|---:|---:|---:|---:|
| 20 | 89,3% | 86,3% | 68% | 0,93 | 15 ph |
| 50 * | 96,2% | 92,0% | 72% | 1,62 | 27 ph |
| 100 | 97,3% | 92,7% | 73% | 2,88 | 48 ph |

\* khác tập câu

Lợi ích biên: 20→50 được +5,7 điểm R@5 với +12 phút = 0,48 điểm/phút; 50→100 được +0,7
điểm với +21 phút = 0,03 điểm/phút. Gấp 16 lần. **Chốt pool 50.**

`ăn %` gần như đứng yên qua cả ba pool (68/72/73) — reranker không suy giảm khi pool to
ra, nó bị chặn bởi trần. Mà trần đã bão hoà: 89,3 → 96,2 → 97,3, bước cuối chỉ +1,1.

### 15.7 Reranker đã bão hoà pool — chỗ nghẽn dịch lần nữa

Pool 20, k=10: `khoảng trống 9 câu · rerank ăn +9 = 100%`. Rerank R@10 = 89,3% = **đúng
bằng trần**. Trong pool 20 mọi câu có gold đều vào được top-10.

Năng lực xếp hạng không còn là chỗ nghẽn. Đổi reranker khác, hay train lại nó, không mua
được gì — mọi cải thiện phải đến từ đưa thêm ứng viên đúng vào pool.

Chỗ còn lại nhiều nhất là k=1 (pool 100): rerank@1 78,0% so với trần 97,3% = bỏ lại 19,3
điểm. Ở k=5 chỉ còn bỏ lại 4,6 điểm.

Hệ quả cho prompt: **nhồi top-5 Evidence, không phải top-1.** Evidence p50 869 từ × 5
≈ 4.300 từ ≈ 6.000 token, vừa context Qwen 3B. Lấy top-1 là tự vứt 19 điểm recall.

### 15.8 Hybrid BM25 + dense — chốt e5-small, max_len 256

§15.7 khoá đường cải thiện lại còn một: nâng trần pool. Dense chỉ được đo ở đúng đại
lượng đó — gold có nằm trong top-k Document lấy từ pool hay không — nên `eval_hybrid.py`
KHÔNG chạy cross-encoder trong vòng lặp, nhanh hơn ~50 lần.

Mốc phải vượt không phải BM25 pool 50 mà **BM25 pool 100**: nó cũng nâng trần và tốn 0
params. Ngưỡng chốt trước khi chạy: `< 96,2%` vứt · `96,2-97,3%` vứt · `>= 98%` gắn.

n=1.000, deep 1.000, rrf-k 60. Dòng BM25 trùng khít qua cả bốn lượt chạy (88,4/93,6/
96,7/97,5) — cùng seed, cùng tập câu, nên chênh lệch chỉ đến từ nhánh dense.

| model | params | dense @50 | hybrid @50 | chênh | encode |
|---|---:|---:|---:|---:|---:|
| e5-small L256 | 118M | 96,5% | **97,9%** | +1,2 | 436s |
| e5-small L512 | 118M | 96,9% | 97,5% | +0,8 | 873s |
| bkai bi-encoder L256 | 135M | 94,1% | 97,6% | +0,9 | 1.560s |

**Chốt: e5-small, max_len 256.** 97,9% vượt BM25 pool 100 (97,5%) mà còn rẻ hơn nó 4 phút.
Ngân sách 2,274 + 0,568 + 0,118 = **2,96B, dư 1,04B**.

> **Lỗi thời, 20-09.** Con số 2,96B đúng tại thời điểm §15.8 (một reranker, chưa có
> adapter). Pipeline hiện tại có **hai** reranker và hai adapter LoRA. Bảng đếm đầy đủ:
> `docs/PIPELINE_V10.md` mục "Ngân sách params".
>
> **Sửa lần hai, cùng ngày.** Bản vá đầu của ô này ghi 3,571B (dư 0,429B) vì tin dòng
> dense e5-small ở §15.8 này. Đọc code mới thấy đường chạy thật dùng
> `AITeamVN/Vietnamese_Embedding` 568M ở **cả ba lượt** A/C/E: `dung_pr_private.py` mặc
> định `--dense /workspace/base/Vietnamese_Embedding` và `pri_g2.sh` không ghi đè. Tổng
> thật **4.020.984.131 = 4,021B, VƯỢT trần 4B 20.984.131 params**. Tức kết luận "loại
> AITeamVN" của mục này chưa bao giờ tới được đường chạy — mục viết một chuyện, hệ thống
> làm chuyện khác, suốt nhiều tuần, và cả hai bản nộp Task 2 đều ở sai phía trần.

> **Sửa số, 12-09.** Ba dòng trên trước đây ghi generator là 3,09B ("Qwen 3B") — con số
> tôi áng chứ không đếm. Số đếm thật nằm ở `docs/CHECKPOINT_VALIDATION.md` nhánh `main`:
> `Qwen3.5-2B (gồm vision tower trong checkpoint) = 2.274.069.824`. Chênh 816M, tức chỗ
> trống dưới trần 4B không phải 225M mà **1,04B**. Hệ quả: khe dense không còn chật,
> `truro7/vn-law-embedding` (278M, domain pháp luật) hay `e5-base` (278M) vào được →
> 3,12B. Kết luận loại `AITeamVN/Vietnamese_Embedding` ở §15.8 bên dưới vẫn giữ, nhưng
> **lý do phải đổi**: nó yếu hơn `Vietnamese_Reranker` ở bảng của chính tác giả, chứ
> không phải vượt trần. Đây là ví dụ vì sao "đo trước khi tuyên" áp cả cho số params —
> một số áng làm lệch một quyết định kiến trúc.

#### max_len 512 thua, và dạng của nó nói vì sao

Chunk 256 từ tiếng Việt ≈ 380-520 subword, nên L256 cắt mất ~40% mỗi chunk. Đọc trọn
chunk lẽ ra phải tốt hơn. Đo được ngược:

| k | dense L256 | dense L512 | hybrid L256 | hybrid L512 |
|---|---:|---:|---:|---:|
| @10 | 92,2 | 91,4 (−0,8) | 93,1 | 93,5 (+0,4) |
| @20 | 95,1 | 94,8 (−0,3) | 96,5 | 95,9 (−0,6) |
| @50 | 96,5 | 96,9 (+0,4) | 97,9 | 97,5 (−0,4) |
| @100 | 97,9 | 97,6 (−0,3) | 98,8 | 98,7 (−0,1) |

Dense với tới sâu hơn (@50 +0,4) nhưng đỉnh kém đi (@10 −0,8): **mean pooling bị loãng**.
Bình quân 500 token thì vài token mang thông tin định vị bị vài trăm token boilerplate
("theo quy định của pháp luật") bầu áp đảo — vector thành "trung bình của văn bản hành
chính", nhận đúng LOẠI văn bản nhưng không phân biệt được văn bản NÀO.

Cross-encoder không bị vậy vì nó không nén: query nhìn từng token qua attention, token vô
nghĩa bị gán trọng số thấp thay vì được bầu bình đẳng. Cách chữa đúng là đổi pooling
(CLS đã train contrastive, hoặc weighted-mean theo IDF), nhưng ta không train được model
nên hạ max_len là cách rẻ duy nhất — và nó vừa đúng vừa nhanh gấp đôi. **Đừng thử lại 512.**

#### Ba model bị loại vì kiến trúc, không vì chất lượng

`bkai/vietnamese-bi-encoder`, `dangvantuan/vietnamese-embedding`, `keepitreal/vietnamese-sbert`
là **cùng một checkpoint PhoBERT-base** — params trùng đến từng đơn vị (134.998.272) —
với `max_position_embeddings = 258`. Chúng không bao giờ đọc nổi một chunk 256 từ.

Đó là lý do bkai chỉ được 94,1% dense@50 trong khi e5-small 118M được 96,5%. Trên benchmark
Legal Zalo (passage ngắn, lọt trong 258 vị trí) bkai chỉ kém `Vietnamese_Embedding` 1,6 điểm
A@1; trên chunk của ta nó kém e5-small 9,1 điểm @10. Cùng model, hai kết luận trái nhau,
chỉ vì độ dài đầu vào. **Xem `max_position_embeddings` trước cả benchmark.**

#### Vì sao không dùng embedding 568M

`AITeamVN/Vietnamese_Embedding` (568M, nền bge-m3, 8.192 ctx) cộng reranker 568M cộng Qwen
3,09B = **4,23B, vượt trần 226M**. (Số generator này sai — xem ô "Sửa số, 12-09" ở trên;
với 2,274B thật thì tổng là 3,41B và KHÔNG vượt trần. Kết luận không đổi vì lý do dưới
đây mới là lý do chính.) Đường vào duy nhất là thay chỗ reranker, mà bảng Legal
Zalo trong model card của chính tác giả cho reranker thắng embedding ở mọi k:

| model | A@1 | A@5 | A@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Vietnamese_Reranker | 0,794 | 0,954 | 0,974 | 0,867 |
| Vietnamese_Embedding | 0,727 | 0,931 | 0,957 | 0,818 |

Cùng 568M, cùng nhóm tác giả, cùng benchmark. Đổi là nước lỗ — không cần tốn GPU để biết.

> **Xét lại, 20-09.** Lập luận "đường vào duy nhất là thay chỗ reranker" dựa trên tiền đề
> ngân sách chật, nhưng ô "Sửa số, 12-09" ngay phía trên vừa tuyên dư **1,04B** — thừa chỗ
> cho 568M mà không phải động vào reranker. Tiền đề bị chính đoạn văn trên phủ định, tức
> kết luận đúng nhưng mất chỗ dựa. Nay có chỗ dựa thật, và nó mạnh hơn dự kiến: hệ **đã**
> chạy AITeamVN suốt, tổng 4,021B, **vượt** trần 21M. Nên đây không còn là câu hỏi "có nên
> thêm AITeamVN?" mà là "phải bỏ bớt cái gì để xuống dưới trần?". Bài học lặp lại từ vụ
> ViLegalBERT ở §15.8.1, nặng thêm một bậc: ở đó một model card chưa đo dẫn tới kết luận
> sai; ở đây một kết luận **đúng** nhưng không bao giờ được thực thi, và không có phép
> kiểm nào đối chiếu tài liệu với đường chạy để bắt lỗi đó.
>
> Nhánh còn mở, 20-09: đang đo trên Kaggle (kernel `tdwuyn/prb-e5-goc`, `prb-e5-ft`,
> `prb-ai-ft`) đúng phép head-to-head còn thiếu — cùng corpus 287.358 chunk, cùng 500 câu
> train có gold, cùng reranker, chỉ đổi dense e5-small ↔ AITeamVN. Ba cấu hình để tách ảnh
> hưởng của dense khỏi ảnh hưởng của việc phải đổi reranker (`rerank_ft_g2` đã mất cùng
> instance Vast, thay bằng `rerank_ft_ai`). Đây cũng là bước 1 của phương án hạ params
> xuống 3,571B mà vẫn giữ cả ba lượt MBR.
Cùng lý do loại `mainguyen9/vietlegal-e5` (560M). Ứng viên 278M đúng miền pháp luật
`truro7/vn-law-embedding` chưa đo: 118M đã vượt mốc, chưa cần tiêu thêm 160M params.

#### Số thật (đo 13-09, T4x2, cùng một lượt chạy)

Ước cũ đã trả nợ. `eval_rerank.py --pool 50 --n-cau 500` chạy HAI lần **trong cùng
session** — có `--dense dense.npy` và không — nên nhiễu chung (seed, máy, tập câu) triệt
tiêu. Không lấy số §15.3 cũ ra trừ: lượt đó khác máy.

Trần hybrid ở đây 97,6% còn §15.8 đo 97,9%: khác n (500 vs 1.000), lệch 0,3 điểm nằm trong sai số ±1,3 của n=500. Không phải hai đại lượng khác nhau.

| pool | | R@1 | R@3 | **R@5** | R@10 | R@20 | trần |
|---|---|---|---|---|---|---|---|
| BM25 | BM25 | 56,4 | 74,6 | 81,0 | 88,4 | 94,0 | 96,2 |
| BM25 | +rerank | 75,2 | 89,4 | **92,0** | 94,2 | 96,0 | |
| hybrid | BM25 | 61,2 | 84,0 | 89,2 | 93,8 | 97,0 | 97,6 |
| hybrid | +rerank | 77,2 | 91,4 | **93,6** | 96,0 | 97,2 | |
| | **chênh** | +2,0 | +2,0 | **+1,6** | +1,8 | +1,2 | +1,4 |

**Ước +0,9, thật +1,6.** Hệ số 0,72 ước THẤP, và thấp một cách hệ thống — nó mô hình
thiếu một đường nhân quả. RRF làm hai việc: thêm ứng viên (nâng trần 96,2 -> 97,6) và
**sắp lại pool** (nâng sàn BM25 R@5 81,0 -> 89,2 = +8,2 điểm). Công thức
`Δtrần × 0,72` chỉ tính đường thứ nhất.

Bằng chứng cho cách đọc đó nằm ở tỉ lệ ăn khoảng trống, nó TỤT khi pool tốt hơn:

| pool | khoảng trống k=5 | rerank ăn |
|---|---|---|
| BM25 | 76 câu | +55 = 72% |
| hybrid | 42 câu | +22 = 52% |

52% < 72% không phải reranker kém đi. Câu dễ đã bị pool hybrid dọn trước khi reranker
chạm vào, nên 42 câu còn lại khó hơn 76 câu kia. Hệ quả cho thiết kế: **đừng ước R@5 từ
trần nữa** — hệ số không bất biến theo chất lượng pool.

Giá thực: encode 287.358 chunk **421s** (comment cũ trong `eval_hybrid.py` ghi 20-40
phút — sai gấp 4). Vòng đo 1,56s/câu có dense so với 1,58s/câu không có: dense **không**
tốn thời gian chạy, chỉ tốn 118M params và 0,22 GB index.

### 15.8.1 ViLegalBERT và giới hạn của MLM làm retriever (20-09)

`ntphuc149/ViLegalBERT` = PhoBERT-base-v2 continual-pretrain MLM trên 16GB văn bản pháp
lý, 134.410.000 params. Đúng miền, vừa ngân sách — ứng viên hợp lý trên giấy.

**Dự báo ghi trước khi đo** (docstring `scratchpad/vlb/vlb-do.py`): kém hơn, vì nó là MLM
`fill-mask` chưa từng thấy mục tiêu contrastive, nên embedding bị *anisotropy* — mọi vector
dồn vào một nón hẹp, cosine giữa hai câu bất kỳ đều cao, thứ hạng gần vô nghĩa.

Giao thức bước 0: pool BM25 50 chunk dựng **một lần** dùng chung cho mọi biến thể, 500 câu
train có gold, seed 1. Đo khả năng **xếp lại trong pool đã cho** — không phải dựng pool,
nên đường cơ sở phải đo tại chỗ chứ không lấy từ bảng §15.8. Một model không phân biệt nổi
gold trong 50 ứng viên thì không thể phân biệt nổi trong 287.358; test loại trừ hợp lệ và
rẻ hơn bước full corpus 11 lần.

| biến thể | params | hit@5 | recall@5 | Δhit vs bm25 |
|---|---:|---:|---:|---:|
| bm25 (thứ tự gốc, không làm gì) | 0 | 0,8160 | 0,7166 | — |
| **e5-small** (đang dùng) | 118M | **0,8740** | **0,7715** | **+0,0580** |
| phobert-base-v2 + PyVi | 135M | 0,7760 | 0,6727 | −0,0400 |
| **ViLegalBERT + PyVi** | 134M | **0,7620** | **0,6593** | **−0,0540** |
| phobert-base-v2, không segment | 135M | 0,7000 | 0,5979 | −0,1160 |
| ViLegalBERT, không segment | 134M | 0,6480 | 0,5552 | −0,1680 |

Cả hai MLM **tệ hơn cả thứ tự BM25 thô**: xếp lại pool bằng chúng làm hỏng pool. Cổng
trong script chặn bước full corpus, tiết kiệm 26 phút GPU. Dự báo khớp cả dấu lẫn độ lớn.

Hai con số thứ cấp đáng giá hơn con số chính:

**(1) ViLegalBERT 0,7620 < phobert-base-v2 0,7760.** Continual-pretrain trên 16GB văn bản
pháp lý làm embedding *tệ đi* 0,014 cho mục đích truy hồi. Không nghịch lý: MLM tối ưu dự
đoán token bị mask, và pretrain thêm trên một miền hẹp khiến mọi vector trong miền đó
**càng giống nhau hơn** — tốt cho dự đoán token, chết cho phân biệt thứ hạng.
Domain-adaptation làm nón anisotropy *hẹp lại*, không rộng ra. Suy ra quy tắc sàng lọc rẻ:
**kiểm model có head contrastive chưa (`1_Pooling/` trong repo, hoặc model card nói
sentence-transformers) trước khi tốn GPU**; không có head đó thì miền khớp đến đâu cũng
không thành retriever.

**(2) Cột có segment chênh cột không segment 0,076 (phobert) và 0,114 (ViLegalBERT)** —
lớn hơn toàn bộ khoảng cách e5 vs BM25 (0,058). PyVi word segmentation là điều kiện **cần**
cho mọi model PhoBERT-family. Nếu về sau dùng model PhoBERT-family đã train contrastive
(ví dụ bkai ở §15.8), **bỏ segmentation là sai nghiêm trọng hơn chọn sai model**.

Ghi chú vận hành: model gated AGPL-3.0. Đo không phát sinh nghĩa vụ giấy phép (không phân
phối), nhưng đẩy lên dataset Kaggle **công khai** thì có — dataset tạm phải đặt private.
`max_position_embeddings = 258` → trần 256 token thật, ngang e5-small ở L256 nên không phải
bất lợi riêng.

Script: `scratchpad/vlb/vlb-do.py`, kernel `sunny0703/vlb-do`.

### 15.9 Tái lập

```bash
./scripts/push_dataset.sh                    # codebase -> Kaggle Dataset
./scripts/push_kernel.sh                     # notebook -> Kaggle (T4x2)
# hoặc trên máy có GPU:
PYTHONPATH=src python scripts/build_gold.py
PYTHONPATH=src python scripts/eval_retrieval.py 1000 3000
PYTHONPATH=src python scripts/eval_rerank.py --model models/Vietnamese_Reranker --pool 50
PYTHONPATH=src python scripts/eval_hybrid.py --dense-model intfloat/multilingual-e5-small \
    --n-cau 1000 --max-len 256      # trần hybrid, §15.8
PYTHONPATH=src python scripts/build_dense.py --model intfloat/multilingual-e5-small
PYTHONPATH=src python scripts/eval_rerank.py --model models/Vietnamese_Reranker --pool 50 \
    --dense dense.npy               # R@5 thật trên pool hybrid
```
