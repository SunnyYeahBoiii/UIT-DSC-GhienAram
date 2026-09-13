"""`ma_hoa` phải mask đúng ranh giới prompt/answer.

Vì sao có file này: transformers 5.x đổi `apply_chat_template` sang trả BatchEncoding.
`len(BatchEncoding)` là SỐ KHOÁ (2), nên `n_prompt` lặng lẽ thành 2 và prompt-loss mask
hỏng toàn bộ — loss vẫn giảm, không ai thấy gì. Kernel v10-train-sft v6 chỉ lộ ra nhờ
`p_ids + a_ids` tình cờ ném TypeError; đừng phụ thuộc vào may mắn đó.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "train_sft", Path(__file__).resolve().parent.parent / "scripts" / "train_sft.py"
)
train_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_sft)


class TokDict:
    """apply_chat_template trả dict — hành vi transformers 5.x."""

    eos_token_id = 9

    def apply_chat_template(self, msgs, **kw):
        return {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5}

    def __call__(self, text, **kw):
        return {"input_ids": [7, 8]}


def test_n_prompt_la_so_token_khong_phai_so_khoa():
    r = train_sft.ma_hoa({"prompt": "p", "answer": "a"}, TokDict(), 100)
    assert r["n_prompt"] == 5
    assert r["input_ids"] == [1, 2, 3, 4, 5, 7, 8, 9]


def test_bo_hang_vuot_max_seq():
    assert train_sft.ma_hoa({"prompt": "p", "answer": "a"}, TokDict(), 7) is None


def test_ids_khong_phai_int_thi_no():
    """Tokenizer trả kiểu lạ phải chết ngay, không được trôi vào vòng train."""

    class TokLa(TokDict):
        def apply_chat_template(self, msgs, **kw):
            return {"input_ids": ["1", "2"]}

    with pytest.raises(AssertionError):
        train_sft.ma_hoa({"prompt": "p", "answer": "a"}, TokLa(), 100)
