"""Test chunking — offset chính xác và bỏ chunk cuối ngắn."""
from legalqa.corpus import Document
from legalqa.chunking import Chunk, chunk_document, MIN_CHUNK_WORDS


def _make_doc(text: str, doc_id: str = "doc1") -> Document:
    """Tạo Document test với passage cho trước."""
    return Document(doc_id=doc_id, passage=text)


def test_start_end_slice_correct():
    """chunk.text phải bằng passage[start:end]."""
    passage = " ".join(f"w{i}" for i in range(500))
    doc = _make_doc(passage)
    chunks = chunk_document(doc, size=100, stride=100)
    for c in chunks:
        assert passage[c.start : c.end] == c.text, (
            f"Chunk {c.chunk_id}: offset không khớp"
        )


def test_single_chunk():
    """Passage ngắn hơn size → đúng 1 chunk."""
    passage = " ".join(f"w{i}" for i in range(50))
    doc = _make_doc(passage)
    chunks = chunk_document(doc, size=256, stride=256)
    assert len(chunks) == 1
    assert chunks[0].text == passage


def test_empty_passage():
    """Passage rỗng → trả về rỗng."""
    doc = _make_doc("")
    assert chunk_document(doc) == []


def test_last_short_chunk_dropped():
    """Chunk cuối < 30 từ và không phải chunk đầu → bị bỏ."""
    # 270 từ: chunk1=256 từ, chunk2=14 từ (< 30, bị bỏ)
    passage = " ".join(f"w{i}" for i in range(270))
    doc = _make_doc(passage)
    chunks = chunk_document(doc, size=256, stride=256)
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 256


def test_last_chunk_kept_if_first():
    """Chunk cuối < 30 từ nhưng là chunk đầu → giữ (không bỏ)."""
    passage = " ".join(f"w{i}" for i in range(15))
    doc = _make_doc(passage)
    chunks = chunk_document(doc, size=256, stride=256)
    assert len(chunks) == 1


def test_stride_overlap():
    """Stride < size tạo chunk trùng lặp."""
    passage = " ".join(f"w{i}" for i in range(300))
    doc = _make_doc(passage)
    chunks = chunk_document(doc, size=100, stride=50)
    # Có overlap giữa các chunk
    assert len(chunks) > 3
    # Chunk đầu có 100 từ
    assert len(chunks[0].text.split()) == 100


def test_chunk_id_format():
    """chunk_id = doc_id:ordinal."""
    passage = " ".join(f"w{i}" for i in range(500))
    doc = _make_doc(passage, doc_id="abc")
    chunks = chunk_document(doc, size=256, stride=256)
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"abc:{i}"
