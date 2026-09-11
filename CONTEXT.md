# LegalQA — UIT DSC 2026 Task 2

Hệ thống trả lời câu hỏi pháp luật tiếng Việt dạng long-form, chạy offline trên Kaggle T4.
Tài liệu này là **glossary**, không phải spec. Không ghi chi tiết implementation vào đây.

## Language

### Dữ liệu nguồn

**Document**:
Một văn bản quy phạm pháp luật, tương ứng đúng một file trong `selected-contexts/`. Đơn vị định danh bởi `id`.
_Avoid_: context, context file, article, passage file

**Passage**:
Toàn bộ nội dung text của một Document. Là trường thô, không phải đơn vị truy hồi.
_Avoid_: content, body, text

**Slug**:
Trường `name` của Document — chuỗi URL không dấu do thuvienphapluat.vn sinh, ví dụ
`Thong-tu-17-2022-TT-BGTVT-...-522401`. **Không phải** tên văn bản đọc được và không tồn tại ở
1.125/8.532 Document.
_Avoid_: name, document name, title

**Document title**:
Tên văn bản tiếng Việt có dấu, phải được **suy ra** từ Slug hoặc từ dòng đầu Passage. Là thứ duy nhất
được phép hiển thị cho người đọc hoặc đưa vào prompt.
_Avoid_: name, slug

**Reference answer**:
Câu trả lời do BTC cung cấp trong `train.json` / `warmup.json`. Là mục tiêu của METEOR, không phải
chân lý pháp lý — chứa cả noise biên tập.
_Avoid_: ground truth, gold answer, label

### Truy hồi

**Chunk**:
Một lát cắt Passage dùng làm đơn vị **tìm kiếm**. Nhỏ, tối ưu cho recall.
_Avoid_: segment, window, span, block

**Evidence**:
Đoạn văn bản thực sự được đưa vào prompt cho model sinh. Được **mở rộng** từ Chunk trúng ra ranh giới
điều khoản đầy đủ. Chunk và Evidence không cùng kích thước và không được dùng lẫn tên.
_Avoid_: context, passage, retrieved chunk, snippet

**Weak gold document**:
Document được suy ra là nguồn của một Reference answer bằng khớp chuỗi nguyên văn, không phải nhãn do
BTC cấp. BTC **không** cung cấp nhãn retrieval nào.
_Avoid_: gold, ground truth document, label, positive

**Legal relation**:
Quan hệ giữa hai Document: sửa đổi, bổ sung, thay thế, bãi bỏ, hướng dẫn, hết hiệu lực. Xuất hiện ở đa số
Document nên **không** phải tín hiệu hiếm.
_Avoid_: link, reference, citation

**Citation**:
Tham chiếu tới vị trí bên trong một Document: Điều N, khoản N, điểm x. Khác Legal relation — Citation trỏ
vào trong, Legal relation trỏ ra ngoài.
_Avoid_: reference, relation

### Sinh câu trả lời

**Candidate**:
Một câu trả lời ứng viên do một nguồn sinh ra cho một câu hỏi. Nhiều Candidate cạnh tranh nhau khi hệ
thống có nhiều hơn một nguồn sinh.
_Avoid_: prediction, output, hypothesis

**Prediction**:
Câu trả lời cuối cùng được chọn và ghi vào submission. Đúng một Prediction cho mỗi câu hỏi.
_Avoid_: answer, candidate, final answer

**Style profile**:
Thống kê phân phối của Reference answer (số từ, số câu, tỷ lệ liệt kê) dùng làm mục tiêu độ dài khi sinh.
Là ràng buộc hình thức, không phải ràng buộc nội dung.
_Avoid_: length plan, format spec
