"""Test hợp đồng của package: import được ở môi trường tối thiểu.

Không phải test hình thức. `rerank.py`/`dense.py` cố tình import torch TRONG hàm để năm
module lõi chạy được khi chỉ có numpy/scipy/sklearn — nhưng `__init__.py` re-export cả
hai, nên một dòng `import torch` bị dời lên top-level của chúng sẽ làm `import legalqa`
chết ở scripts/eda/ và ở máy dev macOS ARM. Lỗi đó chỉ hiện ra khi thiếu torch, tức
KHÔNG hiện trên Kaggle — đúng loại bug cần test chặn.
"""
from __future__ import annotations

import builtins
import sys

import pytest

CAM = {"torch", "transformers", "sentence_transformers"}


def test_import_duoc_khi_khong_co_torch(monkeypatch):
    that = builtins.__import__

    def chan(name, *a, **kw):
        if name.split(".")[0] in CAM:
            raise ModuleNotFoundError(f"chặn trong test: {name}")
        return that(name, *a, **kw)

    for m in [k for k in sys.modules if k == "legalqa" or k.startswith("legalqa.")]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    for m in [k for k in sys.modules if k.split(".")[0] in CAM]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(builtins, "__import__", chan)

    import legalqa  # noqa: PLC0415

    assert legalqa.Retriever is not None


def test_moi_ten_trong_all_deu_truy_cap_duoc():
    import legalqa  # noqa: PLC0415

    thieu = [t for t in legalqa.__all__ if not hasattr(legalqa, t)]
    assert not thieu


@pytest.mark.parametrize("ten", ["Chunk", "Evidence", "Document"])
def test_kieu_loi_khong_bi_bong_boi_re_export(ten):
    """Re-export phải là CÙNG object, không phải bản copy: `isinstance` giữa
    `legalqa.Evidence` và `legalqa.evidence.Evidence` phải khớp."""
    import legalqa  # noqa: PLC0415
    from legalqa import chunking, corpus, evidence  # noqa: PLC0415

    nguon = {"Chunk": chunking, "Evidence": evidence, "Document": corpus}[ten]
    assert getattr(legalqa, ten) is getattr(nguon, ten)
