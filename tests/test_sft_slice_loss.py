from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from train_qwen_sft_lora import _sliced_ce_loss  # noqa: E402


def test_slice_ce_ket_qua_khop_full_cross_entropy():
    """Slice-loss phải ra đúng con số như CE toàn sequence (chỉ khác chỗ bỏ qua vùng -100)."""
    torch.manual_seed(0)
    vocab, hidden_size, seq = 11, 4, 9
    hidden = torch.randn(1, seq, hidden_size)
    head = torch.nn.Linear(hidden_size, vocab)
    labels = torch.full((1, seq), -100)
    # nhãn supervised ở CUỐI sequence — giống cách RagChatDataset build
    labels[0, 6:] = torch.randint(0, vocab, (3,))

    sliced = _sliced_ce_loss(hidden, head, labels)

    full_logits = head(hidden).float()
    expected = torch.nn.functional.cross_entropy(
        full_logits[:, :-1].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(sliced)
    assert torch.allclose(sliced, expected, atol=1e-6)
