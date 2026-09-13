"""Chấm điểm bằng ĐÚNG code của BTC (`third_party/`), không tự cài lại.

Vì sao không tự cài lại: v9.2 có `src/legalqa/core/metrics.py` tự viết, và chính docstring
của nó thừa nhận "not a claim to reproduce the organizer's hidden scorer bit-for-bit".
Một metric xấp xỉ thì mọi so sánh checkpoint đều là so trên một đại lượng khác với đại
lượng quyết định thắng thua. Rẻ hơn nhiều là gọi thẳng code thật.

Hai dòng tính điểm dưới đây chép nguyên từ `eval_qa` trong `third_party/scoring.py`
(dòng 48 và 50), kể cả những chỗ trông kỳ:

- `use_stemmer=False`, và tokenizer mặc định của `rouge_score` xoá sạch ký tự có dấu.
  ROUGE-L ở giải này đo mảnh chữ cái ASCII chứ không đo từ. Xem `third_party/README.md`.
- METEOR nhận `.split()` thuần khoảng trắng, giữ dấu — đây mới là metric có tín hiệu.
- BTC có `pyvi.ViTokenizer` nhưng đã comment lại. Không được "sửa cho đúng": sửa là lệch
  khỏi leaderboard.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

# `rouge_score` là package BTC đóng gói kèm, nằm ngoài `src/`. Import nội bộ của nó là
# tuyệt đối (`from rouge_score import tokenize`) nên thư mục CHA phải nằm trên sys.path.
_THIRD_PARTY = str(Path(__file__).resolve().parents[2] / "third_party")


def chuan_bi_nltk() -> None:
    """Tải wordnet/omw-1.4 — `meteor_score` cần, và nó ném LookupError nếu thiếu.

    BTC gọi `nltk.download` mỗi lần chấm (scoring.py dòng 10-11). Ở đây tách ra để gọi
    một lần lúc khởi động: gọi trong vòng lặp eval là mỗi bước train lại đụng mạng, và
    trên Kaggle tắt internet thì chết giữa chừng thay vì chết ngay lúc bắt đầu.
    """
    import nltk

    for goi in ("wordnet", "omw-1.4"):
        nltk.download(goi, quiet=True)


def cham(du_doan: dict[str, str], tham_chieu: dict[str, str]) -> dict[str, float]:
    """METEOR + ROUGE-L trung bình trên các id CHUNG của hai dict.

    BTC ném exception khi số lượng lệch (scoring.py dòng 44-46). Ở đây lấy giao thay vì
    ném: lúc train ta cố tình chấm trên một tập con dev để tiết kiệm GPU, lệch số lượng
    là bình thường chứ không phải lỗi. Ràng buộc "đủ id" thuộc về lúc nộp bài, không
    thuộc về vòng lặp huấn luyện.
    """
    if _THIRD_PARTY not in sys.path:
        sys.path.insert(0, _THIRD_PARTY)
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    ids = [k for k in tham_chieu if k in du_doan]
    if not ids:
        return {"meteor": 0.0, "rouge": 0.0, "n": 0}
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return {
        "meteor": mean(
            meteor_score([str(tham_chieu[k]).split()], str(du_doan[k]).split()) for k in ids
        ),
        "rouge": mean(
            rs.score(str(tham_chieu[k]), str(du_doan[k]))["rougeL"].fmeasure for k in ids
        ),
        "n": len(ids),
    }
