"""Cắt vòng lặp trong submission.json đã có, ghi submission mới — không cần infer lại.

    python scripts/sanitize_submission.py <submission.json> <thư mục ra>
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalqa.sanitize import cat_lap  # noqa: E402


def main(vao: Path, ra: Path) -> None:
    sub = json.loads(vao.read_text(encoding="utf-8"))
    doi = 0
    for v in sub.values():
        moi = cat_lap(v["answer"])
        if moi != v["answer"].rstrip():
            doi += 1
        v["answer"] = moi
    ra.mkdir(parents=True, exist_ok=True)
    js = ra / "submission.json"
    js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(ra / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(js, "submission.json")
    print(f"-> {js}  ({len(sub)} câu, {doi} câu bị cắt lặp)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
