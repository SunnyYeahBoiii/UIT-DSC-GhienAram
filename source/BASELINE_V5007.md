# Baseline public METEOR 0.5007

Đây là mốc an toàn để phát triển sau thất bại của selector v9.3 (0.4272).

- Runtime: source contract `0.9.2-v5007`, selector schema
  `selector-weights-v3-agentic-evidence`.
- Generator: Qwen3.5-2B + completed QLoRA adapter `full`.
- Frozen selector: `configs/selector_v5007.yaml`.
- Public run tham chiếu: fingerprint `de25e0c7228918c3`.
- Phân bố source tham chiếu: Qwen thường 959, Agentic 9, extractive 32.
- Submission tham chiếu: 1.000 câu, trung bình khoảng 276,6 từ/câu.

Notebook inference mặc định dùng frozen selector này và bỏ qua mọi
`tuned_selection.yaml` được attach. Chỉ đặt `USE_FROZEN_V5007_SELECTOR=False`
khi chạy một thí nghiệm selector riêng; không dùng output thí nghiệm làm
baseline nếu chưa vượt mốc trên grouped-dev và qua kiểm tra phân bố source.

Nhánh `codex/v9.3-regression-snapshot` lưu toàn bộ thử nghiệm v9.3 để có thể
đối chiếu, nhưng không dùng cho inference baseline.
