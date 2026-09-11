# Bỏ agentic tool-call và multi-candidate selector, giữ legal graph

Pipeline v9.2 có 12.313 LOC với bốn tầng xếp chồng: graph retrieval, agentic tool-call, train-memory
candidate, và selector calibration trên candidate oracle. Không tầng nào trong số này có số đo trong repo
chứng minh vượt baseline 0.5007 — con số duy nhất được ghi lại, đạt được **trước** khi các tầng này tồn tại.
Trong khi đó EDA v10 chỉ ra nút thắt nằm ở đơn vị index (recall@10 50,9%), tức trần cứng nằm phía trước mọi
tầng đó.

Quyết định: viết lại từ đầu, **giữ legal graph**, bỏ agentic tool-call, multi-candidate selector và
train-memory candidate. Ngân sách còn lại là một tuần và ~30 GPU-h Kaggle, tương đương 2–3 vòng thực nghiệm
end-to-end; chi phí hiệu chỉnh threshold cho bốn tầng vượt quá ngân sách đó.

## Consequences

Graph được giữ nhưng **không** gate bằng số lượng relation: EDA đo 68,7% Document chứa "sửa đổi" và 93,1%
chứa "căn cứ", nên relation count không phân biệt được gì. Graph chỉ kích hoạt khi có Citation trỏ đích danh
một văn bản cụ thể.

`main` và `codex/v5007-baseline` giữ nguyên code v9.2 để đối chiếu và rollback.
