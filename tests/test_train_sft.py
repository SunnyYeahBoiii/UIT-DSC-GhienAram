"""Test phần thuần của train_sft.py — chạy trên CPU, không cần peft/GPU.

Thứ đáng test nhất ở đây là chỉ số lát trong `loss_dap_an`: lệch một vị trí thì model
vẫn train, loss vẫn giảm, và không có gì báo lỗi — nó chỉ học dự đoán sai lệch một
token. Đó là dạng bug tốn cả lượt train mới phát hiện.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_sft import gop, loss_dap_an  # noqa: E402


class _ModelGia:
    """lm_head = identity, nên hidden state đóng vai logits luôn. Cho phép dựng tensor
    mà ta biết trước lớp nào được dự đoán ở vị trí nào."""

    def get_output_embeddings(self):
        return lambda h: h


def test_gop_pad_ben_phai():
    b = gop([{"input_ids": [1, 2, 3], "n_prompt": 1}, {"input_ids": [4, 5], "n_prompt": 1}], 0)
    assert b["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert b["n_tong"].tolist() == [3, 2]


def _dung_batch():
    """ids = [0, 0, 1, 2, 3]; prompt là 2 token đầu, đáp án là [1, 2, 3].

    hidden[t] là logits dùng để dự đoán token t+1. Đặt hidden[1]=lớp 1, hidden[2]=lớp 2,
    hidden[3]=lớp 3 -> căn đúng thì loss ~ 0. Các vị trí còn lại đặt lệch hẳn (lớp 4) để
    lệch một token là loss vọt lên.
    """
    V = 5
    ids = torch.tensor([[0, 0, 1, 2, 3]])
    lo = torch.full((1, 5, V), -20.0)
    for t, lop in ((0, 4), (1, 1), (2, 2), (3, 3), (4, 4)):
        lo[0, t, lop] = 20.0
    return lo, {"input_ids": ids, "n_prompt": torch.tensor([2]), "n_tong": torch.tensor([5])}


def test_loss_can_dung_lat_dap_an():
    lo, batch = _dung_batch()
    loss, dem = loss_dap_an(_ModelGia(), lo, batch)
    assert dem == 3
    assert loss.item() < 1e-6


def test_lech_mot_token_thi_loss_vot_len():
    """Chốt rằng test trên bắt được off-by-one thật, chứ không phải xanh vì mọi lát đều
    cho loss thấp. Dịch hidden đi một vị trí = đúng cái bug đang phòng."""
    lo, batch = _dung_batch()
    loss, _ = loss_dap_an(_ModelGia(), lo.roll(1, dims=1), batch)
    assert loss.item() > 10


def test_bo_qua_mau_khong_co_token_dap_an():
    lo, batch = _dung_batch()
    batch["n_prompt"] = torch.tensor([5])
    loss, dem = loss_dap_an(_ModelGia(), lo, batch)
    assert dem == 0 and loss.item() == 0.0


def test_ce_theo_khoi_bang_ce_mot_lan():
    """Lát đáp án dài hơn KHOI_CE phải cho ĐÚNG cùng loss như tính một phát.

    Chia khối là phép biến đổi bộ nhớ để tránh OOM (v10-train-sft3 v1 chết vì gradient
    logits 2,71 GiB của một mẫu đơn lẻ), không được đổi toán. Lệch chỉ số giữa lát
    hidden và lát nhãn trong vòng khối sẽ không làm gì đổ — chỉ làm loss sai lặng lẽ.
    """
    import torch.nn.functional as F

    from train_sft import KHOI_CE

    V, n_ans = 7, KHOI_CE * 2 + 37  # cố ý KHÔNG chia hết cho KHOI_CE
    torch.manual_seed(0)
    n_prompt = 3
    ids = torch.randint(0, V, (1, n_prompt + n_ans))
    lo = torch.randn(1, n_prompt + n_ans, V)
    batch = {
        "input_ids": ids,
        "n_prompt": torch.tensor([n_prompt]),
        "n_tong": torch.tensor([n_prompt + n_ans]),
    }
    loss, dem = loss_dap_an(_ModelGia(), lo, batch)

    h = lo[0, n_prompt - 1 : n_prompt + n_ans - 1, :]
    mong = F.cross_entropy(h.float(), ids[0, n_prompt:], reduction="sum") / n_ans
    assert dem == n_ans
    assert torch.allclose(loss, mong, atol=1e-5), f"{loss.item()} != {mong.item()}"


def test_gradient_chay_qua_checkpoint():
    """`checkpoint` nuốt gradient nếu nối đồ thị sai, và loss vẫn in ra số đẹp.

    Không có test này thì một lượt train 6 giờ có thể chạy với grad toàn 0 mà chỉ dấu
    hiệu duy nhất là gnorm — thứ chỉ đọc được sau khi đã đốt quota.
    """
    V, n_ans = 7, 600  # > KHOI_CE để đi qua nhiều khối
    torch.manual_seed(0)
    n_prompt = 2
    ids = torch.randint(0, V, (1, n_prompt + n_ans))
    lo = torch.randn(1, n_prompt + n_ans, V, requires_grad=True)
    batch = {
        "input_ids": ids,
        "n_prompt": torch.tensor([n_prompt]),
        "n_tong": torch.tensor([n_prompt + n_ans]),
    }
    loss, _ = loss_dap_an(_ModelGia(), lo, batch)
    loss.backward()
    assert lo.grad is not None
    assert torch.isfinite(lo.grad).all()
    assert lo.grad.abs().sum() > 0
    # Token prompt (trừ vị trí cuối, vốn dự đoán token đáp án đầu) KHÔNG được nhận grad.
    assert lo.grad[0, : n_prompt - 1, :].abs().sum() == 0
