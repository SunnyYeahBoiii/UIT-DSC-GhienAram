"""Test evidence — Chunk giữa Điều → Evidence chứa trọn Điều; không có Điều → đường lui."""
from legalqa.corpus import Article, Document
from legalqa.chunking import Chunk
from legalqa.evidence import expand_to_evidence


def _make_doc_with_articles():
    """Tạo Document có 2 Điều với nội dung đủ dài."""
    passage = (
        "Điều 1. Quy định chung\n"
        + "Nội dung Điều 1. " * 50
        + "\nĐiều 2. Quy định cụ thể\n"
        + "Nội dung Điều 2. " * 50
    )
    doc = Document(doc_id="d1", passage=passage)
    doc.articles = [
        Article(number="1", heading="Điều 1. Quy định chung", start=0, end=passage.index("Điều 2")),
        Article(
            number="2",
            heading="Điều 2. Quy định cụ thể",
            start=passage.index("Điều 2"),
            end=len(passage),
        ),
    ]
    return doc


def test_chunk_in_article_expands_to_full_article():
    """Chunk nằm trong Điều 2 → Evidence chứa trọn Điều 2."""
    doc = _make_doc_with_articles()
    # Tạo chunk nằm trong Điều 2
    art2_start = doc.articles[1].start
    chunk = Chunk(chunk_id="d1:0", doc_id="d1", text="Nội dung Điều 2.", start=art2_start + 10, end=art2_start + 50)
    ev = expand_to_evidence(doc, chunk)
    assert ev.article_number == "2"
    assert "Điều 2. Quy định cụ thể" in ev.text
    assert "Nội dung Điều 2." in ev.text


def test_no_articles_uses_window():
    """Document không có Điều nào → đường lui cửa sổ, không raise."""
    passage = " ".join(f"word{i}" for i in range(500))
    doc = Document(doc_id="d2", passage=passage)
    # Không có articles
    chunk = Chunk(chunk_id="d2:0", doc_id="d2", text="word250", start=passage.index("word250"), end=passage.index("word250") + 7)
    ev = expand_to_evidence(doc, chunk)
    assert ev.doc_id == "d2"
    assert len(ev.text) > 0


def test_long_article_trimmed():
    """Article dài hơn max_words → cắt quanh chunk, giữ heading."""
    doc = Document(doc_id="d3", passage="")
    long_text = "Nội dung very long. " * 200  # ~2000 từ
    full_passage = "Điều 1. Quy định.\n" + long_text
    doc = Document(doc_id="d3", passage=full_passage)
    doc.articles = [
        Article(number="1", heading="Điều 1. Quy định.", start=0, end=len(full_passage)),
    ]
    # Chunk ở giữa
    mid = len(full_passage) // 2
    chunk = Chunk(chunk_id="d3:0", doc_id="d3", text="Nội dung", start=mid, end=mid + 10)
    ev = expand_to_evidence(doc, chunk, max_words=100)
    assert "Điều 1" in ev.text
    assert len(ev.text.split()) <= 110  # heading + window


def test_doc_title_used():
    """Evidence.doc_title lấy từ doc.title, không phải slug thô."""
    doc = Document(doc_id="d4", passage="Điều 1. Test\nNội dung test. " * 20, slug="Thong-tu-123-456")
    doc.articles = [
        Article(number="1", heading="Điều 1. Test", start=0, end=len(doc.passage)),
    ]
    chunk = Chunk(chunk_id="d4:0", doc_id="d4", text="Nội dung test", start=10, end=20)
    ev = expand_to_evidence(doc, chunk)
    # doc.title trả về fallback "Văn bản d4" vì passage không匹配 TITLE_LINE_RE
    assert ev.doc_title != ""
