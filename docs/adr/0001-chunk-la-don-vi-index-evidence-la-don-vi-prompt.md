# Chunk là đơn vị index, Evidence là đơn vị prompt

EDA v10 đo được: index ở mức Document cho BM25 recall@1 22,3% / recall@10 50,9%; index ở mức Chunk 256 từ
cho 55,4% / 89,7% trên cùng truy vấn và cùng weak gold. Nguyên nhân là Document median 4.828 từ, làm loãng
chuẩn hoá độ dài của BM25 trước một câu hỏi chỉ 19 từ nhắm vào một điều khoản.

Quyết định: index theo **Chunk** 256 từ (stride 256) để tìm, rồi **mở rộng** Chunk trúng ra ranh giới `Điều`
chứa nó để tạo **Evidence** đưa vào prompt. Hai đơn vị này cố ý khác kích thước và không được hợp nhất.

## Considered Options

Chunk 128 từ cho recall@1 ngang bằng (55,4%) nhưng sinh 571k chunk thay vì 287k — gấp đôi chi phí embed dense
trên T4, trong khi ngân sách Kaggle là 30 GPU-h/tuần. Chunk 512/768 từ rẻ hơn nhưng recall@10 tụt xuống
86,6%/84,8%.

## Consequences

84,1% Document có `Điều N.` để mở rộng theo; 15,9% còn lại (và 313/2.000 Document không có Điều nào) cần
đường lui — mở rộng theo cửa sổ từ cố định quanh Chunk trúng.
