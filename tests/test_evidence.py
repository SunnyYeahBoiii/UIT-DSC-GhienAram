"""Test mở rộng Chunk -> Evidence theo ranh giới Điều."""
from legalqa.chunking import Chunk, chunk_document
from legalqa.corpus import Document, parse_articles
from legalqa.evidence import expand_to_evidence


def _doc(passage: str, doc_id: str = "d1") -> Document:
    d = Document(doc_id=doc_id, passage=passage)
    d.articles = parse_articles(passage)
    return d


def _art(n: int, words: int) -> str:
    return f"Điều {n}. Tiêu đề {n}\n" + " ".join(f"t{n}w{i}" for i in range(words)) + "\n"


def test_chunk_trong_mot_dieu_lay_tron_dieu():
    doc = _doc(_art(1, 300) + _art(2, 300))
    chunk = [c for c in chunk_document(doc, size=40, stride=40) if c.start > 60][0]
    ev = expand_to_evidence(doc, chunk)
    assert ev.article_number == "1"
    assert ev.text.startswith("Điều 1.")
    assert "Điều 2." not in ev.text


def test_chunk_vat_qua_hai_dieu_gop_ca_hai():
    """44% Chunk vắt ranh giới Điều — phải gộp, không được rơi về fallback."""
    doc = _doc(_art(1, 30) + _art(2, 30))
    # Chunk đủ dài để phủ cả hai Điều
    chunk = chunk_document(doc, size=256, stride=256)[0]
    ev = expand_to_evidence(doc, chunk)
    assert ev.article_number == "1, 2"
    assert "Điều 1." in ev.text and "Điều 2." in ev.text


def test_dieu_qua_dai_bi_cat_nhung_giu_heading():
    doc = _doc(_art(7, 5000))
    chunk = chunk_document(doc, size=256, stride=256)[4]
    ev = expand_to_evidence(doc, chunk, max_words=200)
    assert ev.article_number == "7"
    assert ev.text.startswith("Điều 7.")
    assert len(ev.text.split()) <= 400  # ngân sách + sai số ước lượng ký tự/từ


def test_document_khong_co_dieu_dung_duong_lui():
    """15,9% Document không có Điều nào — không được raise."""
    doc = _doc(" ".join(f"w{i}" for i in range(2000)))
    assert doc.articles == []
    chunk = chunk_document(doc)[3]
    ev = expand_to_evidence(doc, chunk)
    assert ev.article_number is None
    assert ev.text.strip()


def test_passage_rong_khong_raise():
    doc = _doc("   ")
    ev = expand_to_evidence(doc, Chunk("x:0", "x", "", 0, 0))
    assert ev.text == ""


def test_doc_title_khong_bao_gio_la_slug_tho():
    doc = Document(doc_id="d9", passage=_art(1, 50), slug="Thong-tu-17-2022-TT-BGTVT-abc-522401")
    doc.articles = parse_articles(doc.passage)
    ev = expand_to_evidence(doc, chunk_document(doc)[0])
    assert "-" not in ev.doc_title or "Thông tư" in ev.doc_title


def test_heading_chen_vao_cua_so_la_nguyen_van_khong_phai_ban_normalize():
    """`Article.heading` bị `re.sub(r"\\s+", " ")` + cắt 200 ký tự trong corpus.py — nó là
    NHÃN, không phải lát cắt. Đo: 4.301/85.205 Evidence đường cửa sổ có heading khác
    nguyên văn Passage, 10.660 bị cắt cụt. Dán nhãn đó vào `text` là phá ràng buộc giữ
    nguyên văn theo offset, đúng chỗ 75,9% reference answer trích nguyên văn."""
    tieu_de = "Điều 9.  Tiêu đề có  space thừa và " + "x" * 250
    doc = _doc(tieu_de + "\n" + " ".join(f"w{i}" for i in range(5000)) + "\n")
    assert len(doc.articles[0].heading) == 200, "fixture phải chạm ngưỡng cắt"
    # Chunk ở GIỮA thân Điều để cửa sổ không tự trùm heading
    chunk = chunk_document(doc, size=256, stride=256)[9]
    ev = expand_to_evidence(doc, chunk, max_words=200)
    assert ev.text.startswith(tieu_de + "\n"), "heading phải nguyên văn, đủ dài, đủ space"


def test_cua_so_da_trum_heading_thi_khong_dan_lai():
    """20,9% Evidence đường cửa sổ in heading hai lần. Hai dòng "Điều 5." liền nhau dạy
    model rằng có hai Điều."""
    doc = _doc(_art(5, 5000))
    # Chunk NGẮN ở ngay đầu Điều -> tâm cửa sổ gần offset 0 -> cửa sổ tự trùm heading
    chunk = chunk_document(doc, size=40, stride=40)[0]
    ev = expand_to_evidence(doc, chunk, max_words=200)
    assert ev.text.count("Điều 5.") == 1
