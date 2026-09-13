# third_party — code chấm CHÍNH THỨC của BTC

Nguồn: `Scoring-Program-Task-LegalQA.zip` do BTC phát hành. Chép **nguyên văn**, không
sửa một ký tự. Đây là bản chạy trên bảng xếp hạng, nên mọi số METEOR/ROUGE-L nội bộ phải
sinh ra từ đúng code này — tự cài lại là đo một đại lượng khác rồi tưởng là đang đo điểm
thi.

- `scoring.py` — entrypoint của BTC (đọc `/app/input`, ghi `/app/output/scores.json`).
  Giữ lại để đối chiếu, không chạy trực tiếp; `src/legalqa/scorer.py` gọi lại đúng hai
  dòng tính điểm ở `eval_qa`.
- `rouge_score/` — thư viện Google (Apache-2.0), BTC đóng gói kèm. Đã bỏ file test,
  `__pycache__`, `.DS_Store`; phần thư viện giữ nguyên.

## Hai điều phải biết trước khi đọc bất kỳ số nào

**1. ROUGE-L không đo từ, nó đo mảnh chữ cái ASCII.** `rouge_score/tokenize.py` chạy
`re.sub(r"[^a-z0-9]+", " ", text.lower())`, tức xoá sạch ký tự có dấu:

    "Căn cứ khoản 1 Điều 5" -> ['c', 'n', 'c', 'kho', 'n', '1', 'i', 'u', '5']

BTC có sẵn `pyvi.ViTokenizer` nhưng đã comment lại (scoring.py dòng 27-31), nên đây là
hành vi thật của leaderboard. Token 'n'/'c'/'i'/'u' dày đặc nên LCS khớp ngẫu nhiên
nhiều: điểm nền cao, khả năng phân biệt thấp. Đừng tối ưu theo nó.

**2. METEOR mới là metric có tín hiệu.** `meteor_score([ref.split()], pred.split())` —
tách theo khoảng trắng, GIỮ dấu, đúng cấp độ từ. Khớp với README ("chính METEOR, phụ
ROUGE-L").

Hệ quả cho generation: **phải viết đủ dấu**. Bỏ dấu thì "dieu" ra một token nguyên vẹn
còn reference có dấu ra ['i','u'] — hai bên hết khớp, mất điểm cả hai metric.
