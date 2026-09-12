"""Test chọn Evidence: bỏ trùng + ngân sách ký tự.

Fixture dùng `\n\n` và space thừa: fixture toàn space đơn từng che bug offset sai
287.716/287.741 chunk (xem AGENTS.md), nên không dùng `" ".join` trơn.
"""
from legalqa.chunking import Chunk, chunk_document
from legalqa.corpus import Document, parse_articles
from legalqa.retrieve import chon_evidence


def _art(n: int, words: int) -> str:
    """Điều có `\n\n` phân khoản và space thừa — giống Passage thật sau dewrap."""
    than = "\n\n".join(
        f"{k}.  " + "  ".join(f"t{n}k{k}w{i}" for i in range(words // 3))
        for k in (1, 2, 3)
    )
    return f"Điều {n}. Tiêu đề {n}\n{than}\n"


def _doc(passage: str, doc_id: str) -> Document:
    d = Document(doc_id=doc_id, passage=passage)
    d.articles = parse_articles(passage)
    return d


def _chunks(doc: Document) -> list[Chunk]:
    return chunk_document(doc, size=40, stride=40)


def test_hai_chunk_cung_dieu_gop_thanh_mot():
    """Hai chunk cạnh nhau trong cùng Điều -> expand ra text y hệt -> chỉ giữ 1.
    Đây là ca CHẮC CHẮN xảy ra với pool 50, không phải ca biên."""
    doc = _doc(_art(1, 300), "d1")
    cs = _chunks(doc)
    assert len(cs) >= 2, "fixture phải sinh >=2 chunk mới test được bỏ trùng"
    ev = chon_evidence(cs[:2], {"d1": doc}, top_n=5)
    assert len(ev) == 1
    assert ev[0].article_number == "1"


def test_hai_dieu_cung_van_ban_deu_duoc_giu():
    """Lý do chọn khoá `(doc_id, article_number)` làm mặc định: câu cần hai Điều của
    CÙNG một luật phải giữ được cả hai. Khoá `van_ban` sẽ mất Điều thứ hai."""
    doc = _doc(_art(1, 120) + _art(2, 120), "d1")
    cs = _chunks(doc)
    ranked = [cs[0], cs[-1]]
    assert len({c.start for c in ranked}) == 2
    ev = chon_evidence(ranked, {"d1": doc}, top_n=5)
    assert [e.article_number for e in ev] == ["1", "2"]
    ev_vb = chon_evidence(ranked, {"d1": doc}, top_n=5, khoa="van_ban")
    assert len(ev_vb) == 1, "khoá van_ban phải gộp — chính là đánh đổi đã ghi trong docstring"


def test_ngan_sach_cat_nhung_luon_tra_ve_it_nhat_mot():
    """max_chars=1 vẫn phải ra 1 Evidence. Prompt không có căn cứ -> model bịa 100%,
    còn Evidence dài quá thì tokenizer cắt — hai hậu quả không cùng cỡ."""
    doc = _doc(_art(1, 120) + _art(2, 120), "d1")
    cs = _chunks(doc)
    ranked = [cs[0], cs[-1]]
    ev = chon_evidence(ranked, {"d1": doc}, top_n=5, max_chars=1)
    assert len(ev) == 1


def test_ngan_sach_chan_truoc_top_n():
    doc = _doc(_art(1, 120) + _art(2, 120), "d1")
    cs = _chunks(doc)
    ranked = [cs[0], cs[-1]]
    du = sum(len(e.text) for e in chon_evidence(ranked, {"d1": doc}, top_n=5))
    it = chon_evidence(ranked, {"d1": doc}, top_n=5, max_chars=du - 1)
    assert len(it) == 1, "vượt ngân sách phải DỪNG, không nhặt tiếp cái ngắn hơn"


def test_giu_dung_thu_tu_hang():
    """Thứ tự phải là thứ tự reranker trả về. Đảo thứ tự input thì output đảo theo —
    nếu không, mọi số R@k đo trên `ranked` đều không nói gì về prompt thật."""
    d1 = _doc(_art(1, 120), "d1")
    d2 = _doc(_art(1, 120), "d2")
    docs = {"d1": d1, "d2": d2}
    a, b = _chunks(d1)[0], _chunks(d2)[0]
    assert [e.doc_id for e in chon_evidence([a, b], docs, top_n=5)] == ["d1", "d2"]
    assert [e.doc_id for e in chon_evidence([b, a], docs, top_n=5)] == ["d2", "d1"]


def test_doc_id_thieu_thi_bo_qua_khong_vo():
    """chunks.jsonl và corpus.jsonl dựng bằng hai script khác nhau; lệch một doc_id
    không được làm sập cả lượt suy luận 1.000 câu."""
    doc = _doc(_art(1, 120), "d1")
    mo = Chunk(chunk_id="dX:0", doc_id="dX", text="abc", start=0, end=3)
    ev = chon_evidence([mo, _chunks(doc)[0]], {"d1": doc}, top_n=5)
    assert [e.doc_id for e in ev] == ["d1"]


def test_fallback_khong_co_so_dieu_thi_khong_gop_bua():
    """28,1% Evidence là fallback cửa sổ, article_number=None. Khoá `(doc_id, None)`
    sẽ gộp mọi cửa sổ của một Document thành một — mà chúng khác nhau. Phải tụt về text."""
    # PHẢI dài hơn cửa sổ fallback (max_words=900): doc ngắn hơn thì mọi chunk đều mở
    # ra trọn Document nên text trùng thật, và bỏ trùng theo text là đúng. Bản đầu của
    # test này để 400 từ và fail — tiền đề sai, không phải code sai.
    tho = "  ".join(f"w{i}" for i in range(3000))  # không có "Điều" -> không parse ra Article
    doc = _doc(tho, "d1")
    assert not doc.articles
    cs = _chunks(doc)
    ranked = [cs[0], cs[-1]]
    ev = chon_evidence(ranked, {"d1": doc}, top_n=5)
    assert all(e.article_number is None for e in ev)
    assert len(ev) == 2, "hai cửa sổ khác nhau không được gộp"
