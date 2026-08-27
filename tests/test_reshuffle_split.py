from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Cả hai script đều import torch ở mức module — bỏ qua nhẹ nhàng khi môi trường không có torch.
torch = pytest.importorskip("torch")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from train_qwen_lora import reshuffle_split as qlora_split  # noqa: E402
from train_qwen_sft_lora import reshuffle_split as sft_split  # noqa: E402


def _ids(n: int) -> list[str]:
    return [f"qid-{i:04d}" for i in range(n)]


def test_cung_input_cung_epoch_ra_dung_mot_ket_qua():
    """Deterministic theo seed+epoch — điều kiện cần để resume tái tạo đúng split."""
    ids = _ids(100)
    assert sft_split(ids, 0.15, 42, 3) == sft_split(ids, 0.15, 42, 3)
    assert qlora_split(ids, 0.15, 42, 3) == qlora_split(ids, 0.15, 42, 3)


def test_hai_implement_giong_nhau():
    """QLoRA nhân bản công thức từ SFT (tránh import vòng) — hai bản phải luôn khớp."""
    ids = _ids(200)
    for epoch in range(4):
        assert sft_split(ids, 0.15, 7, epoch) == qlora_split(ids, 0.15, 7, epoch)


def test_moi_epoch_mot_tap_moi_va_phu_ket_toan_bo_pool():
    """Mục tiêu của tính năng: mỗi epoch cắt lại cặp train/dev mới, không mất câu nào."""
    ids = _ids(500)
    seen_devs: list[set[str]] = []
    for epoch in range(5):
        train_ids, dev_ids = sft_split(ids, 0.2, 123, epoch)
        assert set(train_ids).isdisjoint(dev_ids)
        assert sorted(train_ids + dev_ids) == ids
        seen_devs.append(set(dev_ids))
        if epoch:
            assert seen_devs[epoch] != seen_devs[epoch - 1], "epoch liền kề trúng cùng một dev"
    assert len({frozenset(d) for d in seen_devs}) == 5, "5 epoch phải ra 5 tập dev khác nhau"


def test_ti_le_dev_gan_bang_dev_ratio():
    ids = _ids(1000)
    _, dev_ids = sft_split(ids, 0.15, 9, 1)
    assert abs(len(dev_ids) / len(ids) - 0.15) < 0.01


def test_pool_nho_van_con_train():
    train_ids, dev_ids = sft_split(_ids(2), 0.15, 1, 0)
    assert len(train_ids) == 1 and len(dev_ids) == 1
