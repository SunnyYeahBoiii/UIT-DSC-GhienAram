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

# Crawl HTML để lại wrap mềm dưới dạng "\r\n\n": đo trên 797 Document thì 100%
# Document chứa, 427.420 lần xuất hiện, 93,9% nối giữa câu. Ngược lại "\n\n" thuần chỉ
# 50,7% nối giữa câu — mơ hồ nên giữ nguyên. Chỉ gỡ đúng loại có \r.
WRAP_RE = re.compile(r"[ \t]*\r\n*[ \t]*")
ENDPUNCT = ".:;?!…"

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

# Tiêu đề văn bản nằm trên dòng RIÊNG giữa Passage (sau tên cơ quan, số hiệu, ngày),
# không phải dòng đầu — nên bắt buộc MULTILINE. Thiếu nó thì 100% Document rơi về
# đường lui Slug không dấu.
TITLE_LINE_RE = re.compile(
    r"^[ \t]*((?:BỘ LUẬT|NGHỊ ĐỊNH|THÔNG TƯ LIÊN TỊCH|THÔNG TƯ|QUYẾT ĐỊNH|"
    r"NGHỊ QUYẾT|PHÁP LỆNH|CÔNG VĂN|CHỈ THỊ|HIẾN PHÁP|LUẬT)[^\n]{0,200})$",
    re.MULTILINE,
)

# Số hiệu văn bản — khoá để đối chiếu Citation với Document trong legal graph
DOC_NUMBER_RE = re.compile(r"\bSố:\s*([\w./\-]+)")


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
        head = self.passage[:4000]
        m = TITLE_LINE_RE.search(head)
        if not m:
            return title_from_slug(self.slug) or f"Văn bản {self.doc_id}"

        parts = [re.sub(r"\s+", " ", m.group(1)).strip()]
        # Loại văn bản đứng một mình ("THÔNG TƯ") vô nghĩa với model — ghép dòng chủ đề
        # viết hoa ngay sau ("QUY ĐỊNH ĐÀO TẠO LIÊN THÔNG..."), đó mới là tên thật.
        for line in head[m.end() :].split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.isupper() and len(line) > 10:
                parts.append(re.sub(r"\s+", " ", line))
            break
        return " ".join(parts)[:300]

    @property
    def number(self) -> str:
        """Số hiệu văn bản ("27/2017/TT-BLĐTBXH"), rỗng nếu không có."""
        m = DOC_NUMBER_RE.search(self.passage[:4000])
        return m.group(1).strip(" .,;") if m else ""


def dewrap_passage(passage: str) -> str:
    """Gỡ wrap mềm của trình crawl, giữ ngắt đoạn thật.

    Phải chạy TRƯỚC parse_articles và trước chunking: mọi offset Article/Chunk/Evidence
    đều trỏ vào Passage sau bước này. Nếu không gỡ, 75,7% tiêu đề Điều bị cắt đôi
    ("Điều 2. Tổ chức" | "thực hiện") nên Article.heading và Document.title đều sai.
    """
    if "\r" not in passage:
        return passage

    out: list[str] = []
    pos = 0
    for m in WRAP_RE.finditer(passage):
        out.append(passage[pos : m.start()])
        pos = m.end()
        prev = out[-1][-1] if out[-1] else ""
        nxt = passage[pos] if pos < len(passage) else ""
        # Wrap sau dấu kết câu + trước chữ hoa/số = ngắt đoạn thật bị wrap kèm, giữ lại
        if prev in ENDPUNCT and (nxt.isupper() or nxt.isdigit()):
            out.append("\n\n")
        elif prev:
            out.append(" ")
    out.append(passage[pos:])
    return "".join(out)


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
    passage = dewrap_passage(str(raw.get("passage") or ""))
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
