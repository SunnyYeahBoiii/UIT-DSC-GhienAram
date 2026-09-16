"""Cắt vòng lặp trong câu trả lời model sinh ra — hậu xử lý, không cần GPU.

Lượt hybrid phiên 3 (1.000 câu test) có 3 dạng lặp, đếm bằng tay trên output:
- lặp cụm ngắn trong một dòng: "bảo quản, bảo quản, ..." ×106 (151451);
- lặp dòng có số thứ tự trôi: "(23) X / (24) Y / (25) X ..." tới "(33)" (80189);
- lặp cả khối nhiều dòng: khối 67 từ ×9 rồi bị max-new cắt giữa bản thứ 10 (18645).
Cả ba đều là "tandem repeat" trên chuỗi từ, nên một thuật toán xử lý chung.
"""
from __future__ import annotations

import re

# Số thứ tự đầu dòng "(23)", "24.", "1.2." bị chuẩn hoá thành một ký hiệu, nếu không thì
# vòng lặp 80189 có số tăng dần và không bản nào giống bản nào. Chỉ áp cho từ ĐẦU DÒNG:
# chuẩn hoá mọi số sẽ biến "Điều 1. Điều 2. Điều 3." thành lặp giả.
_SO_THU_TU = re.compile(r"^[(\[]?\d+(?:\.\d+)*[.)\]]?$")

_MOD = (1 << 61) - 1
_CO_SO = 1_000_003


def _tach_tu(text: str) -> tuple[list[str], list[str]]:
    """Trả (từ gốc kèm khoảng trắng phía sau, khoá so sánh). Giữ nguyên khoảng trắng
    gốc để ghép lại không làm mất `\\n` phân khoản."""
    goc, khoa = [], []
    dau_dong = True
    for m in re.finditer(r"(\S+)(\s*)", text):
        tu, ws = m.group(1), m.group(2)
        goc.append(tu + ws)
        k = tu.lower()
        khoa.append("<N>" if dau_dong and _SO_THU_TU.match(k) else k)
        dau_dong = "\n" in ws
    return goc, khoa


def cat_lap(text: str, *, max_chu_ky: int = 300, min_ban: int = 3) -> str:
    """Giữ bản đầu của mỗi đoạn lặp liền nhau, bỏ các bản sau.

    Coi là lặp khi một đơn vị p từ xuất hiện >= `min_ban` bản liền nhau. Không cắt khi
    chỉ 2 bản: đáp án BTC hay nhắc lại tên điều ngay sau "như sau:", luật "2 bản, >= 8 từ"
    cắt nhầm 78/5.676 đáp án train (1,4%); ngưỡng 3 bản chỉ đụng 4 — và cả 4 là lỗi lặp
    có thật trong đáp án ("xuất bản xuất bản xuất bản").
    Bản dở dang cuối (max-new cắt giữa chừng) bị bỏ nếu nó chạm hết chuỗi.
    """
    goc, khoa = _tach_tu(text)
    n = len(khoa)
    if n < 4:
        return text
    h = [0] * (n + 1)
    luy_thua = [1] * (n + 1)
    for i, k in enumerate(khoa):
        h[i + 1] = (h[i] * _CO_SO + hash(k)) % _MOD
        luy_thua[i + 1] = luy_thua[i] * _CO_SO % _MOD

    def bam(a: int, b: int) -> int:
        return (h[b] - h[a] * luy_thua[b - a]) % _MOD

    # Chu kỳ p chỉ có thể là khoảng cách tới lần xuất hiện sau của CÙNG từ đầu đơn vị —
    # thử mọi p ∈ [1, 300] ở mọi vị trí là ~450k phép so mỗi câu, quá chậm cho 1.000 câu.
    vi_tri: dict[str, list[int]] = {}
    for idx, k in enumerate(khoa):
        vi_tri.setdefault(k, []).append(idx)

    ra: list[str] = []
    i = 0
    while i < n:
        nhay = 0
        tran = min(max_chu_ky, (n - i) // 2)
        ung_vien = [q - i for q in vi_tri[khoa[i]] if i < q <= i + tran]
        for p in ung_vien:
            if khoa[i + p] != khoa[i]:
                continue
            don_vi = bam(i, i + p)
            k = 1
            while i + (k + 1) * p <= n and bam(i + k * p, i + (k + 1) * p) == don_vi:
                k += 1
            if k >= min_ban:
                j = i + k * p
                # Đuôi dở dang: phần còn lại là tiền tố của đơn vị lặp.
                con = n - j
                if 0 < con < p and bam(j, n) == bam(i, i + con):
                    j = n
                ra.extend(goc[i:i + p])
                nhay = j - i
                break
        if nhay:
            i += nhay
        else:
            ra.append(goc[i])
            i += 1
    return "".join(ra).rstrip()
