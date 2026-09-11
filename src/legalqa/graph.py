"""Legal graph: nối Điều với Điều của văn bản khác.

Hai loại cạnh, dựng bằng hai cơ chế khác hẳn nhau:

**Cross-reference** — Điều đang đọc trỏ tới Điều của văn bản khác ("Điều 5 Nghị định
100/2019/NĐ-CP"). Trích thẳng từ text đưa vào prompt, không dựng sẵn.

**Amendment** — văn bản sửa đổi khai trong tiêu đề nó sửa văn bản nào, rồi mỗi khoản
khai nó sửa Điều nào ("20. Sửa đổi, bổ sung điểm b khoản 1 ... vào Điều 127 như sau:").
Chiều này phải dựng sẵn: Điều 127 của bản gốc KHÔNG hề nhắc tới văn bản sửa nó.
Đo trên corpus: 489 văn bản sửa đổi, 3.571 cạnh, 2.883 cặp (văn bản, Điều) bị sửa.

Vì sao phải có tầng này: 10,7% reference answer trích luật kèm quan hệ sửa đổi
("Theo Điều 127 Luật Các tổ chức tín dụng 2010, bổ sung bởi Điều 1 Luật ... 2017").
Corpus chỉ có 17 Văn bản hợp nhất (0,2%) nên không có đường tắt — bản gốc và phần sửa
nằm ở hai Document riêng, phải tự nối.

Graph **không** tự vá hai bản thành một: vá sai khoản/điểm sẽ sinh ra điều luật không
tồn tại mà nhìn output không phát hiện được. Trả về cả hai, dán nhãn quan hệ.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from .corpus import Document
from .evidence import Evidence

# Số hiệu văn bản VN. `[0-9]*` sau khối chữ là bắt buộc: thiếu nó thì "45/2019/QH14"
# và "15/2012/QH13" không khớp — tức MỌI Luật vô hình, đúng loại được trích nhiều nhất.
SO_HIEU = r"\d{1,4}[a-zA-Z]?/\d{4}/[A-ZĐ]{2,}[0-9]*(?:-[A-ZĐ0-9]+)*"
SO_HIEU_RE = re.compile(rf"\b({SO_HIEU})\b")

_LOAI = (
    r"Nghị định|Thông tư liên tịch|Thông tư|Bộ luật|Luật|Quyết định|Nghị quyết|"
    r"Pháp lệnh|Văn bản hợp nhất|Chỉ thị|Công văn|Hiến pháp"
)
_LOAI_HOA = (
    r"NGHỊ ĐỊNH|THÔNG TƯ LIÊN TỊCH|THÔNG TƯ|BỘ LUẬT|LUẬT|QUYẾT ĐỊNH|NGHỊ QUYẾT|"
    r"PHÁP LỆNH|VĂN BẢN HỢP NHẤT|CHỈ THỊ|CÔNG VĂN|HIẾN PHÁP"
)

# "Điều 127 Luật Các tổ chức tín dụng 2010" / "Điều 5 của Nghị định 100/2019/NĐ-CP".
# Đo trên 6.929 reference answer: 49,5% dùng dạng số hiệu, 42,1% dùng dạng TÊN — chỉ
# bắt số hiệu là bỏ gần một nửa. Chặn ở [^.\n;]: qua dấu câu là sang tham chiếu khác.
REF_RE = re.compile(
    rf"Điều\s+(\d{{1,3}}[a-zđ]?)\s+(?:của\s+)?((?:{_LOAI})[^.\n;,]{{0,80}})"
)
NAM_RE = re.compile(r"\b((?:19|20)\d{2})\b")

# Tiêu đề văn bản sửa đổi. Mục tiêu BẮT BUỘC mở đầu bằng tên loại văn bản: thiếu ràng
# buộc này thì "QUYẾT ĐỊNH ... ĐIỀU LỆ CỦA HỘI VẬT LIỆU XÂY DỰNG" bị hiểu là sửa một
# văn bản tên "Hội vật liệu xây dựng" — 109/448 trường hợp hỏng đúng kiểu đó.
TITLE_AMEND_RE = re.compile(
    rf"(?:SỬA ĐỔI|BỔ SUNG|THAY THẾ|BÃI BỎ)[^Đ]*?(?:MỘT SỐ ĐIỀU CỦA|CỦA)\s+"
    rf"((?:{_LOAI_HOA})\b.+)$"
)
HEADING_AMEND_RE = re.compile(
    rf"^Điều\s+\d+[a-zđ]?\.\s*(?:Sửa đổi|Bổ sung|Thay thế|Bãi bỏ)[^.]{{0,60}}?"
    rf"(?:một số điều của|của)\s+((?:{_LOAI})\b[^\n]+)"
)

# Khoản trong văn bản sửa đổi tự khai nó động tới Điều nào của bản gốc.
KHOAN_AMEND_RE = re.compile(
    r"^(\d{1,3})\.\s*((?:Sửa đổi|Bổ sung|Bãi bỏ|Thay thế)[^\n]{0,200})", re.MULTILINE
)
KHOAN_BAT_DAU_RE = re.compile(r"^(\d{1,3})\.\s", re.MULTILINE)
DIEU_TRONG_CAU_RE = re.compile(r"Điều\s+(\d{1,3}[a-zđ]?)")

# Đuôi "số 12/2020/TT-BGTVT ngày 29 tháng 5 năm 2020 của Bộ trưởng..." trong tên mục tiêu
_DUOI_SO_RE = re.compile(r"\s+(?:số|SỐ)\s+.*$")

# Một văn bản sửa nhiều luật cùng lúc: "LUẬT THUẾ GIÁ TRỊ GIA TĂNG, LUẬT THUẾ TIÊU THỤ
# ĐẶC BIỆT VÀ LUẬT QUẢN LÝ THUẾ" = 3 mục tiêu. Chỉ cắt ở tên loại đứng ngay sau "," hoặc
# "và" — tức đang LIỆT KÊ SONG SONG. Cắt ở mọi tên loại thì "NGHỊ ĐỊNH ... HƯỚNG DẪN LUẬT
# CHĂN NUÔI" thành hai mục tiêu và sinh cạnh nghị định-sửa-luật không có thật: thử cách đó
# cho 65 ca đa mục tiêu, phần lớn hỏng đúng kiểu này.
_MOC_LOAI_RE = re.compile(rf"(?<=[,\s])(?:{_LOAI_HOA}|{_LOAI})\b")
_NOI_RE = re.compile(r"(?:,|\bvà|\bVÀ)\s*$")
# Từ nối dính vào đuôi mảnh trước khi cắt — để lại thì tập từ thừa một từ "va".
_DUOI_NOI_RE = re.compile(r"[\s,;]*(?:và|VÀ)?[\s,;]*$")


def _bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()


def _tu(s: str) -> frozenset[str]:
    """Tập từ không dấu, bỏ số. Số bị bỏ vì năm được so riêng."""
    return frozenset(t for t in re.split(r"[^a-z0-9]+", _bo_dau(s)) if t and not t.isdigit())


@dataclass(slots=True, frozen=True)
class DocRef:
    """Tham chiếu tới một văn bản: bằng số hiệu, hoặc bằng tên + năm."""

    number: str = ""
    name: str = ""
    year: str = ""


@dataclass(slots=True, frozen=True)
class CrossRef:
    """Tham chiếu tới Điều `article` của văn bản `doc`."""

    doc: DocRef
    article: str


@dataclass(slots=True, frozen=True)
class Amendment:
    """Một khoản của văn bản sửa đổi, và Điều của bản gốc mà nó động tới."""

    src_doc_id: str
    src_article: str
    src_khoan: str
    start: int  # offset khoản trong Passage của src
    end: int
    verb: str  # "Sửa đổi, bổ sung điểm b khoản 1, bổ sung khoản 5 vào Điều 127"


def parse_refs(text: str) -> list[CrossRef]:
    """Trích Cross-reference từ text, giữ thứ tự xuất hiện, bỏ trùng.

    Quét thẳng trên text chứ không tra cạnh dựng sẵn: chỉ được đi theo tham chiếu nằm
    trong đoạn THỰC SỰ đưa vào prompt, không phải mọi tham chiếu của cả Document.
    """
    seen: set[CrossRef] = set()
    out: list[CrossRef] = []
    for m in REF_RE.finditer(text):
        ref = CrossRef(doc=_doc_ref(m.group(2)), article=m.group(1))
        if not ref.doc.number and not ref.doc.name:
            continue
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _doc_ref(cum: str) -> DocRef:
    """"Nghị định 100/2019/NĐ-CP" -> number; "Luật Các tổ chức tín dụng 2010" -> name+year."""
    m = SO_HIEU_RE.search(cum)
    if m:
        return DocRef(number=m.group(1))
    y = NAM_RE.search(cum)
    # Năm đóng vai dấu chấm hết của tên: "Luật Các tổ chức tín dụng 2010 quy định về..."
    # — REF_RE bắt tới 80 ký tự nên nuốt luôn phần sau. Giữ lại phần sau thì tập từ thừa
    # ra vài từ và resolve trượt sang văn bản khác.
    name = (cum[: y.start()] if y else cum).strip(" ,;")
    # "Luật này", "Nghị định này" trỏ vào chính Document đang đọc — không phải cạnh ra ngoài
    if _bo_dau(name).endswith(" nay"):
        return DocRef()
    return DocRef(name=name, year=y.group(1) if y else "")


class LegalGraph:
    """Tra văn bản theo số hiệu / tên, và tra Điều nào đã bị sửa bởi khoản nào.

    Chỉ giữ index, không giữ Passage: corpus đã clean nặng 470 MB, graph mà ôm Passage
    thì không cùng lúc nạp được với BM25 index trên Kaggle T4. Người gọi tự cấp Document.
    """

    __slots__ = ("_by_number", "_by_tokens", "_amend")

    def __init__(self) -> None:
        self._by_number: dict[str, str] = {}
        self._by_tokens: list[tuple[frozenset[str], str, str]] = []
        self._amend: dict[tuple[str, str], list[Amendment]] = {}

    # ---- dựng ----------------------------------------------------------------

    def build(self, docs: list[Document]) -> None:
        self._build_index(docs)
        self._build_amendments(docs)

    def _build_index(self, docs: list[Document]) -> None:
        """Index hai kênh. Kênh Slug là kênh chính, không phải kênh dự phòng:
        9,2% Document không có số hiệu — trong đó có cả `LUẬT CÁC TỔ CHỨC TÍN DỤNG` —
        nhưng 100% có Slug và 96,4% Slug mang năm ban hành."""
        self._by_number = {}
        self._by_tokens = []
        for d in docs:
            if d.number:
                self._by_number.setdefault(d.number, d.doc_id)
            than = re.sub(r"-\d+$", "", d.slug)
            y = NAM_RE.search(than)
            self._by_tokens.append((_tu(than), y.group(1) if y else "", d.doc_id))

    def _build_amendments(self, docs: list[Document]) -> None:
        self._amend = defaultdict(list)
        for d in docs:
            targets = self._amend_target(d)
            if not targets:
                continue
            for art in d.articles:
                body = d.passage[art.start : art.end]
                for m in KHOAN_AMEND_RE.finditer(body):
                    start = art.start + m.start()
                    end = art.start + _khoan_end(body, m.group(1), m.start())
                    dieu_list = dict.fromkeys(DIEU_TRONG_CAU_RE.findall(m.group(2)))
                    for target in targets:
                      for dieu in dieu_list:
                        self._amend[(target, dieu)].append(
                            Amendment(
                                src_doc_id=d.doc_id,
                                src_article=art.number,
                                src_khoan=m.group(1),
                                start=start,
                                end=end,
                                verb=re.sub(r"\s+", " ", m.group(2)).strip(),
                            )
                        )
        self._amend = dict(self._amend)

    def _amend_target(self, doc: Document) -> list[str]:
        """Các văn bản mà `doc` sửa đổi. Tìm ở tiêu đề trước, rồi heading 3 Điều đầu.

        Trả list vì 8 văn bản sửa nhiều luật trong cùng một tiêu đề."""
        m = TITLE_AMEND_RE.search(doc.title)
        ten = m.group(1) if m else None
        if ten is None:
            for art in doc.articles[:3]:
                hm = HEADING_AMEND_RE.match(art.heading)
                if hm:
                    ten = hm.group(1)
                    break
        if ten is None:
            return []
        by = _nam_slug(doc.slug)
        out = []
        for phan in _tach_muc_tieu(ten):
            t = self.resolve(_doc_ref(phan), before_year=by)
            if t and t != doc.doc_id and t not in out:
                out.append(t)
        return out

    # ---- tra ----------------------------------------------------------------

    def resolve(self, ref: DocRef, *, before_year: str = "") -> str | None:
        """Số hiệu tra thẳng; tên tra bằng tập từ Slug.

        `before_year` chặn theo chiều thời gian: một văn bản chỉ sửa được văn bản có
        TRƯỚC nó. Thiếu ràng buộc này, "Luật Các tổ chức tín dụng" trong tiêu đề bản
        sửa đổi 2017 khớp luôn vào chính nó."""
        if ref.number:
            return self._by_number.get(ref.number)
        if not ref.name:
            return None
        ten = ref.name if SO_HIEU_RE.search(ref.name) else _DUOI_SO_RE.sub("", ref.name)
        q = _tu(ten)
        # Truy vấn < 3 từ ("Thông tư", "Quyết định") là mảnh cắt cụt, không phải tên:
        # tập từ quá nhỏ nên là con của hàng trăm Slug và khớp bừa vào văn bản bất kỳ.
        # Chỉ tha khi có năm — năm thu hẹp đủ để 2 từ vẫn xác định được văn bản.
        if not q or (len(q) < 3 and not ref.year):
            return None
        best: tuple[tuple[int, int], str] | None = None
        for toks, year, doc_id in self._by_tokens:
            if ref.year and year and year != ref.year:
                continue
            # `>` chứ không `>=`: năm trong Slug bản sửa đổi đôi khi LÀ năm bản gốc
            # ("Phap-lenh-sua-doi-Phap-lenh-ngoai-hoi-2005"). Khớp cùng năm vẫn an toàn
            # vì `_amend_target` đã loại văn bản trỏ vào chính nó.
            if before_year and year and year > before_year:
                continue
            if not q <= toks:
                continue
            # Thừa > 6 từ nghĩa là Slug nói về chuyện khác mà tình cờ chứa đủ tập từ
            # ("LUẬT SỬA ĐỔI..." khớp "Nghị định hướng dẫn Luật Báo chí"). p90 chỉ 5 từ.
            if len(toks - q) > 6:
                continue
            # ít từ thừa nhất = khớp chặt nhất; hoà thì lấy bản mới hơn
            key = (len(toks - q), -int(year or 0))
            if best is None or key < best[0]:
                best = (key, doc_id)
        return best[1] if best else None

    def amendments_of(self, doc_id: str, article: str) -> list[Amendment]:
        return self._amend.get((doc_id, article), [])

    def __len__(self) -> int:
        return len(self._by_number)

    @property
    def n_amendments(self) -> int:
        return sum(len(v) for v in self._amend.values())


def _nam_slug(slug: str) -> str:
    m = NAM_RE.search(re.sub(r"-\d+$", "", slug))
    return m.group(1) if m else ""


def _khoan_end(body: str, so: str, start: int) -> int:
    """Khoản kết thúc ở khoản SỐ LIỀN SAU. Không cắt theo dòng: nội dung sửa đổi được
    trích nguyên văn nhiều đoạn trong ngoặc kép, cắt theo dòng là mất phần trích. Cũng
    không cắt ở marker "n." bất kỳ: phần trích dẫn mang số khoản CỦA BẢN GỐC, nên khoản
    20 sửa Điều 127 chứa dòng "5. Tổng mức dư nợ..." và bị cắt mất ngay câu quan trọng."""
    ke = str(int(so) + 1) + "."
    for m in KHOAN_BAT_DAU_RE.finditer(body, start + 1):
        if m.group(0).strip() == ke:
            return m.start()
    return len(body)


def expand_with_graph(
    graph: LegalGraph,
    docs_by_id: dict[str, Document],
    seed: Evidence,
    *,
    max_hop: int = 2,
    max_extra: int = 3,
) -> list[Evidence]:
    """Trả về Evidence phụ cho `seed`: phần sửa đổi Điều đó, rồi các Điều nó trỏ tới.

    Amendment đi trước Cross-reference vì nó nói về CHÍNH Điều đang đọc — một Điều đã
    bị sửa mà đưa nguyên bản gốc vào prompt là đưa luật hết hiệu lực.

    `max_hop=2` theo ADR-0002; visited-set chặn quay vòng (văn bản sửa đổi và văn bản
    bị sửa trỏ lẫn nhau nên chu trình là chuyện thường). `max_extra=3` chặn bùng nổ:
    một Điều có tới 150 tham chiếu.
    """
    visited: set[tuple[str, str]] = {(seed.doc_id, seed.article_number or "")}
    out: list[Evidence] = []

    for am in graph.amendments_of(seed.doc_id, seed.article_number or ""):
        src = docs_by_id.get(am.src_doc_id)
        if src is None:
            continue
        out.append(
            Evidence(
                doc_id=src.doc_id,
                doc_title=src.title,
                article_number=am.src_article,
                text=src.passage[am.start : am.end],
                relation=(
                    f"Sửa đổi Điều {seed.article_number} của {seed.doc_title} — "
                    f"khoản {am.src_khoan} Điều {am.src_article}"
                ),
            )
        )
        visited.add((am.src_doc_id, am.src_article))
        if len(out) >= max_extra:
            return out

    frontier = [seed]
    for _ in range(max_hop):
        if not frontier:
            break
        nxt: list[Evidence] = []
        for ev in frontier:
            for ref in parse_refs(ev.text):
                doc_id = graph.resolve(ref.doc)
                if doc_id is None or (doc_id, ref.article) in visited:
                    continue
                visited.add((doc_id, ref.article))
                hit = _article_evidence(docs_by_id.get(doc_id), ref.article)
                if hit is None:
                    continue
                out.append(hit)
                nxt.append(hit)
                if len(out) >= max_extra:
                    return out
        frontier = nxt
    return out


def _article_evidence(doc: Document | None, number: str) -> Evidence | None:
    """Evidence là trọn Điều được gọi tên. None nếu văn bản đích không có Điều đó —
    5,1% trường hợp resolve được văn bản, thường do bản được trỏ tới đã bị thay thế."""
    if doc is None:
        return None
    for a in doc.articles:
        if a.number == number:
            return Evidence(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                article_number=a.number,
                text=doc.passage[a.start : a.end],
                relation="Được dẫn chiếu",
            )
    return None


def _tach_muc_tieu(ten: str) -> list[str]:
    """Cắt chuỗi mục tiêu thành từng văn bản một. Chỉ cắt khi có >=2 tên loại văn bản —
    chuỗi một mục tiêu giữ nguyên để không đụng vào 290 ca vốn đã đúng."""
    moc = [0] + [m.start() for m in _MOC_LOAI_RE.finditer(ten) if _NOI_RE.search(ten[: m.start()])]
    if len(moc) < 2:
        return [ten]
    moc.append(len(ten))
    phan = [_DUOI_NOI_RE.sub("", ten[a:b]).strip(" ,;") for a, b in zip(moc, moc[1:])]
    # Mảnh một từ ("LUẬT") là rác cắt hụt: tập từ quá nhỏ nên khớp bừa vào văn bản bất kỳ.
    return [x for x in phan if len(x.split()) >= 2] or [ten]
