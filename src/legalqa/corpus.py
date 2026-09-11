"""Đọc Document và tách cấu trúc điều khoản.

Thuật ngữ theo CONTEXT.md: Document = 1 file trong selected-contexts/,
Passage = nội dung thô, Slug = trường `name` (URL không dấu), Document title =
tên tiếng Việt suy ra từ Slug.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Điều đứng đầu dòng: "Điều 12.", "Điều 12a.", "Điều 12 ." — bắt cả biến thể thiếu dấu chấm
DIEU_RE = re.compile(r"^[ \t]*(Điều\s+(\d+[a-zđ]?))\s*[.\-:]?\s*", re.MULTILINE)

# Slug thuvienphapluat luôn kết thúc bằng "-<id số>"; phần đầu là tên văn bản viết không dấu
SLUG_TAIL_RE = re.compile(r"-\d+$")

# Loại văn bản nhận diện được từ đầu Slug, map sang dạng có dấu
SLUG_PREFIX = {
    "Nghi-dinh": "Nghị định",
    "Thong-tu-lien-tich": "Thông tư liên tịch",
    "Thong-tu": "Thông tư",
    "Quyet-dinh": "Quyết định",
    "Nghi-quyet": "Nghị quyết",
    "Phap-lenh": "Pháp lệnh",
    "Cong-van": "Công văn",
    "Chi-thi": "Chỉ thị",
    "Bo-luat": "Bộ luật",
    "Hien-phap": "Hiến pháp",
    "Luat": "Luật",
}

# Dòng đầu Passage thường là tiêu đề văn bản có dấu — nguồn tốt hơn Slug khi có
TITLE_LINE_RE = re.compile(
    r"^\s*((?:NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|LUẬT|BỘ LUẬT|NGHỊ QUYẾT|PHÁP LỆNH|"
    r"CÔNG VĂN|CHỈ THỊ|HIẾN PHÁP)[^\n]{0,200})",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Article:
    """Một Điều trong Document. `start`/`end` là offset ký tự trong Passage."""

    number: str
    heading: str
    start: int
    end: int


@dataclass(slots=True)
class Document:
    doc_id: str
    passage: str
    slug: str = ""
    link: str = ""
    articles: list[Article] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Document title có dấu. Ưu tiên dòng tiêu đề trong Passage, lùi về Slug."""
        m = TITLE_LINE_RE.search(self.passage[:4000])
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return title_from_slug(self.slug) or f"Văn bản {self.doc_id}"


def title_from_slug(slug: str) -> str:
    """Slug không dấu -> tên đọc được. Không khôi phục được dấu, chỉ bỏ nhiễu URL.

    1.125/8.532 Document không có Slug, và Slug đã mất dấu nên đây chỉ là đường lui:
    đưa Slug thô vào prompt từng gây rò chuỗi kiểu `Thong-tu-17-2022-TT-BGTVT-...-522401`.
    """
    if not slug:
        return ""
    s = SLUG_TAIL_RE.sub("", slug)
    for prefix, viet in SLUG_PREFIX.items():
        if s.startswith(prefix + "-"):
            rest = s[len(prefix) + 1 :].replace("-", " ").strip()
            return f"{viet} {rest}"
    return s.replace("-", " ").strip()


def parse_articles(passage: str) -> list[Article]:
    """Tách Passage thành các Điều. 15,9% Document không có Điều nào -> trả về []."""
    marks = list(DIEU_RE.finditer(passage))
    if not marks:
        return []
    out: list[Article] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(passage)
        nl = passage.find("\n", m.start())
        heading = passage[m.start() : nl if nl != -1 else end]
        out.append(
            Article(
                number=m.group(2),
                heading=re.sub(r"\s+", " ", heading).strip()[:200],
                start=m.start(),
                end=end,
            )
        )
    return out


def load_document(path: Path) -> Document | None:
    """Đọc 1 file Document. Trả None nếu Passage rỗng (20/8.532 file)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    passage = str(raw.get("passage") or "")
    if not passage.strip():
        return None
    doc = Document(
        doc_id=str(raw.get("id") or path.stem),
        passage=passage,
        slug=str(raw.get("name") or ""),
        link=str(raw.get("link") or ""),
    )
    doc.articles = parse_articles(passage)
    return doc


def iter_documents(corpus_dir: Path) -> Iterator[Document]:
    for path in sorted(corpus_dir.glob("*.json")):
        doc = load_document(path)
        if doc is not None:
            yield doc
