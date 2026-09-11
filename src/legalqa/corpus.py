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

# Dòng đánh dấu nội dung bị thuvienphapluat lược bớt — 85.602 dòng, rác thuần
_ELIDED = {"...", "....", ".....", "…", "..…", "-", "--", "---"}

# Dòng chỉ chứa mỗi số khoản/điểm ("1.", "a)") rồi xuống dòng — 1,1% số dòng.
# Một dấu đánh số đứng một mình không bao giờ là đoạn riêng, nên nối chắc chắn đúng.
# Trường hợp này dùng "\n\n" thuần nên vòng gỡ \r ở trên không bắt được.
LONE_MARKER_RE = re.compile(
    r"^([ \t]*(?:\d{1,3}[a-zđ]?|[a-zđ])[.)])[ \t]*\n+[ \t]*(?=\S)", re.MULTILINE
)

# Điều đứng đầu dòng: "Điều 12.", "Điều 12a.", "Điều 12 ." — bắt cả biến thể thiếu dấu chấm
DIEU_RE = re.compile(r"^[ \t]*(Điều\s+(\d+[a-zđ]?))\s*[.\-:]?\s*", re.MULTILINE)

# Slug thuvienphapluat luôn kết thúc bằng "-<id số>"; phần đầu là tên văn bản viết không dấu
SLUG_TAIL_RE = re.compile(r"-\d+$")

# URL thuvienphapluat mang đúng Slug đó: ".../Ten-van-ban-<id>.aspx".
# 100% Document có link, và cả 1.105 Document thiếu trường `name` đều lấy lại được
# Slug từ đây — Slug không mất, chỉ nằm chỗ khác.
LINK_SLUG_RE = re.compile(r"/([^/]+?-\d+)\.aspx")

# Loại văn bản nhận diện được từ đầu Slug, map sang dạng có dấu.
# Thứ tự quan trọng: tiền tố dài phải đứng trước tiền tố là tiền tố của nó
# ("Thong-tu-lien-tich" trước "Thong-tu").
SLUG_PREFIX = {
    "Nghi-dinh": "Nghị định",
    "Thong-tu-lien-tich": "Thông tư liên tịch",
    "Thong-tu": "Thông tư",
    "Quyet-dinh": "Quyết định",
    "Nghi-quyet": "Nghị quyết",
    "Phap-lenh": "Pháp lệnh",
    "Cong-van": "Công văn",
    "Cong-dien": "Công điện",
    "Chi-thi": "Chỉ thị",
    "Bo-luat": "Bộ luật",
    "Hien-phap": "Hiến pháp",
    "Van-ban-hop-nhat": "Văn bản hợp nhất",
    "Huong-dan": "Hướng dẫn",
    "Thong-bao": "Thông báo",
    "Ke-hoach": "Kế hoạch",
    "Quy-dinh": "Quy định",
    "Dieu-le": "Điều lệ",
    "Cong-uoc": "Công ước",
    "Luat": "Luật",
    # 256 TCVN + 119 QCVN: tiêu chuẩn/quy chuẩn kỹ thuật, không có "Điều" nào
    "TCVN": "TCVN",
    "QCVN": "QCVN",
}

# Tiêu đề văn bản nằm trên dòng RIÊNG giữa Passage (sau tên cơ quan, số hiệu, ngày),
# không phải dòng đầu — nên bắt buộc MULTILINE. Thiếu nó thì 100% Document rơi về
# đường lui Slug không dấu.
TITLE_LINE_RE = re.compile(
    r"^[ \t]*((?:BỘ LUẬT|NGHỊ ĐỊNH|THÔNG TƯ LIÊN TỊCH|THÔNG TƯ|QUYẾT ĐỊNH|"
    r"NGHỊ QUYẾT|PHÁP LỆNH|CÔNG VĂN|CÔNG ĐIỆN|CHỈ THỊ|HIẾN PHÁP|"
    r"VĂN BẢN HỢP NHẤT|TIÊU CHUẨN QUỐC GIA|QUY CHUẨN KỸ THUẬT QUỐC GIA|"
    r"TCVN|QCVN|LUẬT)[^\n]{0,200})$",
    re.MULTILINE,
)

# Phụ lục / biểu mẫu đứng sau Điều CUỐI. `parse_articles` cho Điều cuối
# `end = len(passage)` nên nó nuốt trọn phần này: 2.282 Điều cuối dài quá 900 từ,
# trung bình 6.920 từ, max 189.311. Marker phải chiếm trọn một dòng ngắn để không
# khớp nhầm giữa câu — đo trên 151.896 Điều GIỮA (biên vốn đã đúng) thì chỉ 0,52%
# dính, tức rủi ro cắt nhầm nội dung thật ở mức đó.
PHU_LUC_RE = re.compile(
    r"^[ \t]*(?:PHỤ LỤC|Phụ lục|PHẦN PHỤ LỤC|MẪU SỐ|Mẫu số|DANH MỤC|Danh mục|"
    r"BIỂU|Biểu số|BẢNG|Bảng|BIỂU MẪU)\b[^\n]{0,70}$",
    re.MULTILINE,
)

# Hai chốt chặn quanh điểm cắt: giữ lại < 50 từ nghĩa là cắt sạch cả Điều, bỏ đi
# < 200 từ nghĩa là chỉ một dòng "Bảng 1" lọt giữa nội dung chứ không phải phụ lục.
# Thiếu hai chốt này tỷ lệ dính trên Điều giữa là 0,81% thay vì 0,52%.
PHU_LUC_MIN_KEEP = 50
PHU_LUC_MIN_DROP = 200

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
    passage = _unwrap_cr(passage)
    passage = _collapse_blank_lines(passage)
    return LONE_MARKER_RE.sub(r"\1 ", passage)


def _unwrap_cr(passage: str) -> str:
    """Vòng 1: gỡ wrap mềm `\\r\\n\\n` của trình crawl."""
    if "\r" in passage:
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
        passage = "".join(out)
    return passage


def _collapse_blank_lines(passage: str) -> str:
    """Vòng 3: bỏ dòng trống, dòng `...` và space thừa ở hai đầu mỗi dòng.

    Đo trên 8.512 Document: 59,3% số dòng là dòng TRỐNG (5,7 triệu dòng) — thuần
    trình bày HTML, không mang thông tin. Ngược lại xuống dòng đơn PHẢI giữ:
    95,8% Document dùng đầu dòng để phân khoản, đó là cấu trúc model cần để trích
    đúng khoản.

    Dòng chỉ chứa `...` (85.602 dòng) là chỗ thuvienphapluat lược nội dung cho người
    chưa trả phí — bỏ đi, để lại thì model học sinh ra `...` giữa câu trả lời.
    """
    keep = [
        line
        for raw in passage.split("\n")
        if (line := raw.strip()) and line not in _ELIDED
    ]
    return "\n".join(keep)


def slug_from_link(link: str) -> str:
    """Lấy Slug từ URL thuvienphapluat. Rỗng nếu URL không đúng dạng."""
    m = LINK_SLUG_RE.search(link)
    return m.group(1) if m else ""


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
        if i + 1 < len(marks):
            end = marks[i + 1].start()
        else:
            end = _cut_phu_luc(passage, m.start())
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
    link = str(raw.get("link") or "")
    doc = Document(
        doc_id=str(raw.get("id") or path.stem),
        passage=passage,
        # 13,0% Document thiếu trường `name`, nhưng link luôn có và mang đúng Slug đó
        slug=str(raw.get("name") or "") or slug_from_link(link),
        link=link,
    )
    doc.articles = parse_articles(passage)
    return doc


def iter_documents(corpus_dir: Path) -> Iterator[Document]:
    for path in sorted(corpus_dir.glob("*.json")):
        doc = load_document(path)
        if doc is not None:
            yield doc


def _cut_phu_luc(passage: str, art_start: int) -> int:
    """Biên cuối của Điều CUỐI: dừng trước phụ lục thay vì nuốt tới hết Passage.

    Phụ lục không thuộc Điều nào — để nó trong Điều cuối thì Evidence bị cắt cửa sổ
    và model đọc biểu mẫu thay vì điều khoản. Cắt được 83,1% số Điều cuối quá dài,
    trung bình 6.920 -> 817 từ.
    """
    tail = passage[art_start:]
    for m in PHU_LUC_RE.finditer(tail):
        if (
            len(tail[: m.start()].split()) >= PHU_LUC_MIN_KEEP
            and len(tail[m.start() :].split()) >= PHU_LUC_MIN_DROP
        ):
            return art_start + m.start()
    return len(passage)
