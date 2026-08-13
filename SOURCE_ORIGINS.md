# Nguồn phiên bản

Repository hiện tại chứa duy nhất cây `source/` phiên bản `0.9.2`.

## Lịch sử

- Nền v9.1: graph-gated retrieval, train memory và evidence-grounded QLoRA.
- Commit Agentic ban đầu: `9aa0eded09d96542669fb34d42ac57fd42dd77cd`.
- v9.2: gated Agentic Search, candidate-specific evidence, selector schema v3,
  legal-reranked tool retrieval và Kaggle wheelhouse đầy đủ.

## Gói Kaggle v9.2

Gói phát hành được tạo từ nguyên trạng cây `source/` sau khi chạy
unit test, compile, parse notebook/config và kiểm tra ZIP. Tên file, SHA-256
và kích thước được cung cấp cùng artifact bên ngoài repository để
không tạo metadata checksum tự tham chiếu.

Checkpoint mô hình, adapter QLoRA, dense index, dữ liệu chính thức và
submission không nằm trong repository hay gói source này.
