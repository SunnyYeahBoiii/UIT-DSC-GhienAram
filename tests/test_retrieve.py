"""Test chọn Evidence: bỏ trùng + ngân sách ký tự.

Fixture dùng `\n\n` và space thừa: fixture toàn space đơn từng che bug offset sai
287.716/287.741 chunk (xem AGENTS.md), nên không dùng `" ".join` trơn.
"""
from legalqa.chunking import Chunk, chunk_document
from legalqa.corpus import Document, parse_articles
from legalqa.evidence import Evidence
from legalqa.graph import LegalGraph
from legalqa.retrieve import Retriever, chon_evidence


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


# ---- wiring end-to-end: seed + legal graph ---------------------------------


class _FakeRetriever:
    """Chỉ cần hai thứ mà `evidence_day_du` dùng: `docs` và `evidence()`. Dựng
    Retriever thật đòi ChunkIndex + BM25 — đo một đại lượng khác, không phải wiring."""

    def __init__(self, docs, seeds):
        self.docs = docs
        self._seeds = seeds

    def evidence(self, question, *, top_n=5, max_chars=24_000):
        return self._seeds[:top_n]

    evidence_day_du = Retriever.evidence_day_du


def _corpus_sua_doi():
    """Cùng fixture với test_graph: Điều 127 của GOC bị khoản 20 của SUA sửa."""
    goc = _doc_slug(
        "GOC",
        "Luat-cac-to-chuc-tin-dung-2010-108079",
        "Điều 126. Cấm\nNội dung cấm.\nĐiều 127. Hạn chế cấp tín dụng\n"
        "1. Không được cấp tín dụng cho:\nb) Kế toán trưởng;\n",
    )
    sua = _doc_slug(
        "SUA",
        "Luat-sua-doi-Luat-cac-to-chuc-tin-dung-2017-356283",
        "Điều 1. Sửa đổi, bổ sung một số điều của Luật Các tổ chức tín dụng\n"
        "20. Sửa đổi, bổ sung điểm b khoản 1 vào Điều 127 như sau:\n"
        "“b) Kế toán trưởng, Chủ tịch Hội đồng quản trị;”\n",
    )
    g = LegalGraph()
    g.build([goc, sua])
    return g, {"GOC": goc, "SUA": sua}


def _doc_slug(doc_id: str, slug: str, passage: str) -> Document:
    d = Document(doc_id=doc_id, passage=passage, slug=slug)
    d.articles = parse_articles(passage)
    return d


def _seed_127(docs):
    a = next(a for a in docs["GOC"].articles if a.number == "127")
    return Evidence(
        doc_id="GOC",
        doc_title=docs["GOC"].title,
        article_number="127",
        text=docs["GOC"].passage[a.start : a.end],
    )


def test_khoan_sua_doi_di_ngay_sau_dieu_goc():
    """Cặp "gốc -> sửa đổi" nằm trong THỨ TỰ list, không ở trường nào của Evidence."""
    g, docs = _corpus_sua_doi()
    r = _FakeRetriever(docs, [_seed_127(docs)])
    ev = r.evidence_day_du("hỏi gì", g)
    assert len(ev) == 2
    assert ev[0].relation == "" and ev[0].article_number == "127"
    assert "Sửa đổi Điều 127" in ev[1].relation


def test_seed_va_phan_mo_rong_la_mot_cum_nhan_ca_hoac_khong_nhan():
    """Nhận Điều gốc rồi cắt mất khoản sửa đổi = đưa luật hết hiệu lực vào prompt mà
    không kèm dấu hiệu nào. Ngân sách phải cắt theo CỤM."""
    g, docs = _corpus_sua_doi()
    seed = _seed_127(docs)
    # Ngân sách đủ cho seed nhưng không đủ cho cả cụm, và đã có sẵn một cụm trước đó.
    khac = Evidence(doc_id="GOC", doc_title="x", article_number="126", text="y" * 50)
    r = _FakeRetriever(docs, [khac, seed])
    ev = r.evidence_day_du("hỏi gì", g, max_chars=60)
    assert [e.article_number for e in ev] == ["126"], "cụm 127 phải bị loại NGUYÊN CỤM"


def test_cum_dau_tien_luon_duoc_nhan_du_vuot_ngan_sach():
    g, docs = _corpus_sua_doi()
    r = _FakeRetriever(docs, [_seed_127(docs)])
    assert len(r.evidence_day_du("hỏi gì", g, max_chars=1)) == 2


def test_top_n_dem_seed_khong_dem_evidence_graph():
    """Một seed có nhiều khoản sửa đổi không được ăn hết top_n và đẩy seed hạng sau ra
    vì lý do không liên quan tới hạng."""
    g, docs = _corpus_sua_doi()
    khac = Evidence(doc_id="GOC", doc_title="x", article_number="126", text="z")
    r = _FakeRetriever(docs, [_seed_127(docs), khac])
    ev = r.evidence_day_du("hỏi gì", g, top_n=2)
    assert [e.article_number for e in ev] == ["127", "1", "126"]


def _doc3() -> Document:
    return _doc(_art(1, 30) + _art(2, 30) + _art(3, 30), "d1")


def _chunk_phu(doc: Document, i: int, j: int) -> Chunk:
    """Chunk giả chạm đúng các Điều từ i tới j (0-based) — điều khiển được `touched`."""
    a, b = doc.articles[i], doc.articles[j]
    return Chunk(f"d1:{i}{j}", "d1", doc.passage[a.start : b.end], a.start, b.end)


def test_evidence_bi_bao_tron_thi_bo():
    """Chunk 256 từ chạm nhiều Điều nên `article_number` là chuỗi "1, 2"; hai Chunk kề
    nhau ra "1, 2" và "2" — hai chuỗi khác nhau, cùng chứa trọn Điều 2. Đo trên 400
    Document đầu: 54,3% cặp Chunk kề chung Điều mà khác chuỗi, 53,6% trong số đó là bao
    trọn. Khoá theo chuỗi thì nhồi cả hai để lặp lại đúng Điều đã có."""
    doc = _doc3()
    ev = chon_evidence([_chunk_phu(doc, 0, 1), _chunk_phu(doc, 1, 1)], {"d1": doc}, top_n=5)
    assert [e.article_number for e in ev] == ["1, 2"]


def test_chong_lan_mot_phan_thi_giu_vi_mang_them_dieu_moi():
    """46,4% còn lại chỉ chồng lấn một phần. Mất một Điều đắt hơn lặp một Điều."""
    doc = _doc3()
    ev = chon_evidence([_chunk_phu(doc, 0, 1), _chunk_phu(doc, 1, 2)], {"d1": doc}, top_n=5)
    assert [e.article_number for e in ev] == ["1, 2", "2, 3"]


def test_bao_tron_van_ap_dung_khi_dieu_den_khong_theo_thu_tu():
    """Bỏ trùng theo TẬP, không theo chuỗi: "2" tới sau "1, 2" phải bị bỏ dù đứng ở đâu."""
    doc = _doc3()
    ranked = [_chunk_phu(doc, 2, 2), _chunk_phu(doc, 0, 1), _chunk_phu(doc, 2, 2)]
    ev = chon_evidence(ranked, {"d1": doc}, top_n=5)
    assert [e.article_number for e in ev] == ["3", "1, 2"]
