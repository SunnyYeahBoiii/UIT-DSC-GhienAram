"""Đường ghi output eval chỉ chạy lần đầu sau 70 bước train (~2,7 giờ T4). Nếu nó ném
KeyError hoặc ghi hụt dòng thì mất cả phiên Kaggle mới biết — quá đắt, nên test ở đây.
"""
import importlib.util
import json
from pathlib import Path

# scripts/ không phải package, nạp thẳng theo đường dẫn.
_spec = importlib.util.spec_from_file_location(
    "train_sft", Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
)
train_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_sft)


def _dev():
    # Khoá đúng như build_sft.py ghi ra dev.jsonl (id, question, prompt, answer,
    # seq_len, doc_ids, ly_do_loai).
    return [
        {"id": "a", "question": "Điều 5 quy định gì?", "prompt": "p", "answer": "Đáp án A",
         "seq_len": 10, "doc_ids": ["d1", "d2"], "ly_do_loai": None},
        {"id": "b", "question": "Mức phạt bao nhiêu?", "prompt": "p", "answer": "Đáp án B",
         "seq_len": 11, "doc_ids": ["d3"], "ly_do_loai": "retrieval_truot_gold"},
    ]


def test_ghi_du_dong_va_du_khoa(tmp_path):
    d = train_sft.ghi_du_doan(_dev(), {"a": "sinh A", "b": "sinh B"}, tmp_path, 70)
    assert d.name == "du_doan_b70.jsonl"
    dong = [json.loads(l) for l in d.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in dong] == ["a", "b"]
    assert dong[0].keys() == {"id", "question", "du_doan", "dap_an", "ly_do_loai", "doc_ids"}
    assert dong[1]["ly_do_loai"] == "retrieval_truot_gold"
    assert dong[1]["doc_ids"] == ["d3"]


def test_bo_qua_id_khong_sinh_duoc(tmp_path):
    # `du_doan` thiếu id nào thì bỏ dòng đó, KHÔNG ném KeyError — sinh có thể hụt mẫu.
    d = train_sft.ghi_du_doan(_dev(), {"a": "sinh A"}, tmp_path, 0)
    dong = [json.loads(l) for l in d.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in dong] == ["a"]


def test_giu_dau_tieng_viet(tmp_path):
    # ensure_ascii=False: file phải đọc được bằng mắt, không phải \uXXXX.
    d = train_sft.ghi_du_doan(_dev(), {"a": "Điều 5 Nghị định", "b": "x"}, tmp_path, 1)
    assert "Điều 5 Nghị định" in d.read_text(encoding="utf-8")
