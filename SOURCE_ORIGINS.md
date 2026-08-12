# Nguồn và thống kê bundle

## Source v8

- Phiên bản package: `0.8.0`
- Nguồn cục bộ:
  `outputs/legalqa_3p88b_kaggle_source_v8_global_graph.zip`
- SHA-256 archive:
  `2BF8F4946F52A5D980C4D1BCA51A5AD83AF54CFBDCBCF080B8A627A18BD56E5B`
- Số file: 62
- Dung lượng file sau giải nén: 85.526.199 byte

## Source v9

- Phiên bản package: `0.9.1`
- Snapshot nền:
  `outputs/legalqa_3p88b_kaggle_source_v9_1_graph_gate_final.zip`
- SHA-256 snapshot nền:
  `27BB291A7BABEB3093B92681C19721EACD356A0A022F02CF666B3C1C29A69CDC`
- 80 file chung giữa snapshot và `source_v9/` khớp SHA-256.
- Sáu file source/tiện ích mới được liệt kê trong README gốc của bundle.
- Tổng số file trong `source_v9/`: 86
- Tổng dung lượng: 85.664.366 byte

## Dữ liệu

- `train.json`: 7.000 câu hỏi.
- `public-official.json`: 1.000 câu hỏi.
- `warmup.json`: 500 câu hỏi.
- `selected-contexts/`: 8.532 văn bản JSON.
- Tất cả 8.532 văn bản đã được parse JSON thành công.
- Tổng số file dữ liệu (không tính `data/README.md`): 8.536.
- Tổng dung lượng file dữ liệu (không tính `data/README.md`): 506.293.192 byte.
- SHA-256 archive dữ liệu chính thức gốc `dsc2026-task2-data.zip`:
  `3672609ED74F3BC71A24116CEF38C30709418CAF303231F558B183D3269DF4BB`.
- SHA-256 `warmup.json`:
  `B824E4F18BD9181C021498A28E402B7374D6D559F2FF28CAA4120A9D932F82C5`.
- SHA-256 tài liệu overview:
  `ADD8AE1E232A9F339C92EE6FF8D4FC1085D9C9A5DFC2D1B1042EF2C979F9524E`.

Đường dẫn nguồn cục bộ chỉ dùng để truy vết trên máy đóng gói và không phải là
đường dẫn runtime bắt buộc của source.
