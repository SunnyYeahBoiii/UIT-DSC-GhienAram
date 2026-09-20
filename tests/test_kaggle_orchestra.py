"""kaggle_orchestra: phần thuần + các kịch bản điều phối trên một Kaggle giả.

Kaggle giả chạy notebook THẬT theo nghĩa hợp đồng: đọc cell ORCHESTRA-PARAMS đã chèn,
ghi `out/<output>` từ tu tới den (chép cả phần nối tiếp từ dataset), ghi mốc bằng
`ghi_moc` — nên test bắt được lỗi chèn params, lỗi khoảng, lỗi gộp khi đổi acc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_orchestra as ko  # noqa: E402


# ------------------------------------------------------------------ phần thuần


def test_chia_deu_du_don_shard_dau_va_phu_kin():
    assert ko.chia_deu(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert ko.chia_deu(7000, 4) == [(0, 1750), (1750, 3500), (3500, 5250), (5250, 7000)]
    # Nhiều acc hơn câu: không sinh shard rỗng.
    assert ko.chia_deu(2, 5) == [(0, 1), (1, 2)]


def test_phan_loai_log():
    oom = "Traceback (most recent call last):\n ... torch.OutOfMemoryError: CUDA out of memory."
    assert ko.phan_loai(oom) == "oom"
    code = "pip ok\nTraceback (most recent call last):\n  File x\nKeyError: 'answer'\n"
    assert ko.phan_loai(code) == "code"
    mang = "Traceback (most recent call last):\nrequests.exceptions.ConnectionError: HF down"
    assert ko.phan_loai(mang) == "moi_truong"
    # Không traceback = kernel bị giết từ ngoài (12 giờ, quota): môi trường.
    assert ko.phan_loai("  ... 400/1000  còn ~90 phút\n") == "moi_truong"
    # Cảnh báo mạng lúc pip ở ĐẦU log không được che lỗi code ở cuối.
    dai = "WARNING ConnectionError retry pip\n" + "x\n" * 4000 + "Traceback (most recent call last):\nKeyError: 1\n"
    assert ko.phan_loai(dai) == "code"


def test_dong_loi_bo_qua_duoi_nbconvert_va_papermill():
    log = ("Traceback (most recent call last):\n"
           "papermill.exceptions.PapermillExecutionError: \n"
           "FileNotFoundError                         Traceback (most recent call last)\n"
           "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/repo/COMMIT.txt'\n"
           "[NbConvertApp] Writing 292517 bytes to __results__.html\n")
    assert ko.dong_loi(log) == "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/repo/COMMIT.txt'"
    assert ko.dong_loi("a\nb\n") == "b"


def test_van_ban_log_doc_dinh_dang_json_cua_kaggle():
    raw = json.dumps([{"stream_name": "stdout", "time": 1, "data": "a\n"},
                      {"stream_name": "stderr", "time": 2, "data": "KeyError: x\n"}])
    assert ko.van_ban_log(raw) == "a\nKeyError: x\n"
    assert ko.van_ban_log("text thô") == "text thô"


def test_phan_loai_push():
    assert ko.phan_loai_push("Kernel version 3 successfully pushed.") == "ok"
    assert ko.phan_loai_push("not valid dataset sources: ['tunvln/legalqa-clean']") == "code"
    assert ko.phan_loai_push("You have exceeded your weekly GPU quota") == "moi_truong"
    assert ko.phan_loai_push("") == "moi_truong"


def _nb(so_cell_params=1):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# x"]}]
    for _ in range(so_cell_params):
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                      "source": ["# ORCHESTRA-PARAMS\n", "SHARD = {}\n"]})
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": ["print(SHARD)\n"]})
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _doc_shard(nb: dict) -> dict:
    ns: dict = {}
    for c in nb["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and s.lstrip().startswith(ko.PARAMS_TAG):
            exec(s, ns)
    return ns["SHARD"]


def test_chen_params_dung_mot_cell():
    nb = ko.chen_params(_nb(), {"tu": 5, "den": 9, "ten": "câu có dấu"})
    assert _doc_shard(nb) == {"tu": 5, "den": 9, "ten": "câu có dấu"}
    with pytest.raises(AssertionError):
        ko.chen_params(_nb(0), {})
    with pytest.raises(AssertionError):
        ko.chen_params(_nb(2), {})


def _ghi(f: Path, ids, them=""):
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", encoding="utf-8") as g:
        for i in ids:
            g.write(json.dumps({"id": i, "du_doan": f"đ{i}"}, ensure_ascii=False) + "\n")
        g.write(them)


def test_doc_ids_bo_dong_vo_va_gop_khong_nhan_doi(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _ghi(a, ["q0", "q1"], them='{"id": "q2", "du_')  # dòng cuối vỡ: kernel bị giết giữa lúc ghi
    assert ko.doc_ids(a) == ["q0", "q1"]
    _ghi(b, ["q0", "q1", "q2", "q3"])
    assert ko.gop_jsonl(a, b) == 2
    assert ko.doc_ids(a) == ["q0", "q1", "q2", "q3"]


def _job(tmp_path, **kw):
    d = dict(ten="t", notebook=str(tmp_path / "nb.ipynb"), tong=6, output="o.jsonl", merge="submission",
             gpu=True, datasets=["legalqa-v10-src", "sunny070306/official"])
    d.update(kw)
    return ko.Job(**d)


def test_kiem_ghep_bat_ho_trung_thieu_moc(tmp_path):
    job = _job(tmp_path)
    shards = [{"i": 0, "tu": 0, "den": 3}, {"i": 1, "tu": 3, "den": 6}]
    f0 = tmp_path / "s0" / "out" / "o.jsonl"
    f1 = tmp_path / "s1" / "out" / "o.jsonl"
    _ghi(f0, ["q0", "q1", "q2"])
    ko.ghi_moc(f0, 0, 3)
    _ghi(f1, ["q2", "q4", "q5"])  # q2 trùng, q3 thiếu
    loi = ko.kiem_ghep(job, shards, tmp_path)
    assert any("chưa có mốc" in x for x in loi)
    ko.ghi_moc(f1, 3, 6)
    loi = ko.kiem_ghep(job, shards, tmp_path)
    assert any("q2" in x and "cả s0 và s1" in x for x in loi)
    # Khoảng hở giữa shard.
    loi = ko.kiem_ghep(job, [{"i": 0, "tu": 0, "den": 3}, {"i": 1, "tu": 4, "den": 6}], tmp_path)
    assert any("khoảng hở" in x for x in loi)


def test_kiem_ghep_jsonl_chap_nhan_it_dong_hon_so_cau(tmp_path):
    """build_sft bỏ mẫu trượt gold: 3 câu ra 2 dòng vẫn là shard XONG."""
    job = _job(tmp_path, merge="jsonl", tong=3)
    f0 = tmp_path / "s0" / "out" / "o.jsonl"
    _ghi(f0, ["q0", "q2"])
    ko.ghi_moc(f0, 0, 3)
    assert ko.kiem_ghep(job, [{"i": 0, "tu": 0, "den": 3}], tmp_path) == []
    dich = ko.ghep(job, [{"i": 0, "tu": 0, "den": 3}], tmp_path, tmp_path / "m")
    assert ko.doc_ids(dich) == ["q0", "q2"]


# ------------------------------------------------------------------ Kaggle giả


class KaggleGia:
    """Mỗi push tạo một 'kernel'. `kich_ban[(acc, i)]` = danh sách hành vi cho các lần shard i
    chạy trên acc đó: "xong" | "oom" | "code" | ("dung_giua", n) | "push_quota" | "push_cau_hinh"."""

    def __init__(self, tmp: Path, quota: dict, kich_ban: dict):
        self.tmp, self.quota, self.kich_ban = tmp, quota, kich_ban
        self.kernels: dict[tuple[str, str], dict] = {}
        self.datasets: dict[tuple[str, str], Path] = {}
        self.pushes: list[dict] = []

    def quota_gpu(self, acc):
        return self.quota[acc.ten]

    def push_kernel(self, acc, thu_muc: Path):
        meta = json.loads((thu_muc / "kernel-metadata.json").read_text())
        shard = _doc_shard(json.loads((thu_muc / meta["code_file"]).read_text()))
        self.pushes.append({"acc": acc.ten, "shard": shard, "meta": meta})
        hv_list = self.kich_ban.get((acc.ten, shard["chi_so"]), ["xong"])
        hv = hv_list.pop(0) if len(hv_list) > 1 else hv_list[0]
        if hv == "push_quota":
            return "400: You have exceeded your weekly GPU quota"
        if hv == "push_cau_hinh":
            return "not valid dataset sources: ['x/y']"
        for d in meta["dataset_sources"]:
            if d.startswith("orch-") or "/orch-" in d:
                assert tuple(d.split("/")) in self.datasets, f"dataset nối tiếp {d} chưa được đẩy"
        self.kernels[(acc.ten, meta["title"])] = {"shard": shard, "hv": hv, "meta": meta, "doc": 0}
        return f"Kernel version 1 successfully pushed. {acc.ten}/{meta['title']}"

    def status(self, acc, slug):
        k = self.kernels[(acc.ten, slug)]
        k["doc"] += 1
        if k["doc"] == 1:
            return "RUNNING"
        return {"xong": "COMPLETE", "oom": "ERROR", "code": "ERROR"}.get(
            k["hv"] if isinstance(k["hv"], str) else "", "ERROR")

    def output(self, acc, slug, dich: Path):
        k = self.kernels[(acc.ten, slug)]
        sh, hv = k["shard"], k["hv"]
        out = dich / "out"
        out.mkdir(parents=True, exist_ok=True)
        f = out / sh["output"]
        if sh["noi_tiep"]:
            src = self.datasets[(acc.ten, sh["noi_tiep"])] / sh["output"]
            f.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        co = set(ko.doc_ids(f))
        can = [f"q{i}" for i in range(sh["tu"], sh["den"]) if f"q{i}" not in co]
        if isinstance(hv, tuple):
            can = can[: hv[1]]
        elif hv != "xong":
            can = can[:1]
        _ghi(f, can)
        log = "chạy...\n"
        if hv == "oom":
            log += "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory\n"
        elif hv == "code":
            log += "Traceback (most recent call last):\nKeyError: 'answer'\n"
        elif hv == "xong":
            ko.ghi_moc(f, sh["tu"], sh["den"])
        (dich / f"{slug}.log").write_text(json.dumps([{"stream_name": "stdout", "time": 0, "data": log}]))

    def day_dataset(self, acc, thu_muc: Path, ten, msg):
        assert not list(thu_muc.glob("*.moc.json")), "mốc cũ lọt vào dataset nối tiếp"
        d = self.tmp / "ds" / acc.ten / ten
        d.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(thu_muc, d)
        self.datasets[(acc.ten, ten)] = d

    def dataset_san_sang(self, acc, ten):
        return True


@pytest.fixture
def moi_truong(tmp_path, monkeypatch):
    monkeypatch.setattr(ko, "GOC", tmp_path / "orchestra")
    monkeypatch.setattr(ko, "GIAY_TIN_STATUS", -1)
    (tmp_path / "nb.ipynb").write_text(json.dumps(_nb()))
    return tmp_path


def _dp(tmp, kg, accs=("a", "b", "c"), **kw):
    job = _job(tmp, **kw)
    return ko.DieuPhoi(job, kg, [ko.Acc(t, f"/tok/{t}") for t in accs], ngu=lambda s: None,
                       commit="abc123 msg", day_src=lambda acc: None, log=lambda *x: None)


def test_chay_suon_se_roi_ghep(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 5, "b": 20, "c": 10}, {})
    dp = _dp(moi_truong, kg)
    dp.run()
    # Chia theo quota giảm dần: s0 vào acc nhiều quota nhất.
    assert [s["acc"] for s in dp.st["shards"]] == ["b", "c", "a"]
    assert all(p["meta"]["dataset_sources"][0] == f"{p['acc']}/legalqa-v10-src" for p in kg.pushes)
    assert all(p["meta"]["dataset_sources"][1] == "sunny070306/official" for p in kg.pushes)
    assert dp.watch() == 0
    sub = json.loads(Path(dp.merge()).parent.joinpath("submission.json").read_text())
    assert sorted(sub) == sorted(f"q{i}" for i in range(6))


def test_het_quota_luc_push_doi_sang_acc_ranh(moi_truong):
    # c có 0 giờ: push thất bại, s2 chờ acc rảnh, không bao giờ bị đẩy lại lên c.
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 0}, {("c", 2): ["push_quota"]})
    dp = _dp(moi_truong, kg)
    dp.run()
    assert dp.st["shards"][2]["trang_thai"] == "CHO_ACC"
    assert dp.watch() == 0
    s2 = [p for p in kg.pushes if p["shard"]["chi_so"] == 2]
    assert [p["acc"] for p in s2] == ["c", "a"]
    assert dp.merge()


def test_dung_giua_chung_noi_tiep_tren_acc_khac(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 9}, {("a", 0): [("dung_giua", 1)]})
    dp = _dp(moi_truong, kg)
    dp.run()
    assert dp.watch() == 0
    s0 = [p for p in kg.pushes if p["shard"]["chi_so"] == 0]
    assert [p["acc"] for p in s0] == ["a", "b"]
    assert s0[0]["shard"]["noi_tiep"] is None
    assert s0[1]["shard"]["noi_tiep"] == "orch-t-s0"
    assert "b/orch-t-s0" in s0[1]["meta"]["dataset_sources"]
    # Phần q0 làm trên a được giữ, không nhân đôi sau khi gộp.
    assert ko.doc_ids(ko.GOC / "t" / "s0" / "out" / "o.jsonl") == ["q0", "q1"]
    assert ko.kiem_ghep(dp.job, dp.st["shards"], ko.GOC / "t") == []


def test_oom_hong_ca_luot_khong_day_them_roi_run_lai_giu_output(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 9},
                   {("b", 1): ["oom", "xong"], ("a", 0): [("dung_giua", 1), "xong"]})
    dp = _dp(moi_truong, kg)
    dp.run()
    assert dp.watch() == 2
    assert dp.st["trang_thai"] == "HONG"
    so_push = len(kg.pushes)
    assert so_push == 3, "sau HỎNG không được đẩy shard dở nào sang acc khác"
    # s0 dừng giữa chừng trong lúc lượt hỏng: không retry, chỉ thu output.
    assert dp.st["shards"][0]["trang_thai"] == "DUNG"

    # Người dùng sửa (hạ gen-bs trong job) rồi run lại: s2 XONG giữ nguyên, s0/s1 nối tiếp.
    dp2 = _dp(moi_truong, kg)
    dp2.run()
    pushes2 = kg.pushes[so_push:]
    assert sorted(p["shard"]["chi_so"] for p in pushes2) == [0, 1]
    assert all(p["shard"]["noi_tiep"] for p in pushes2)
    assert dp2.watch() == 0
    assert dp2.merge()


def test_loi_code_dung_ngay(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 9}, {("a", 0): ["code"]})
    dp = _dp(moi_truong, kg)
    dp.run()
    assert dp.watch() == 2
    assert "KeyError" in dp.st["ly_do_hong"]


def test_push_loi_cau_hinh_dung_truoc_khi_day_shard_sau(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 9}, {("a", 0): ["push_cau_hinh"]})
    dp = _dp(moi_truong, kg)
    dp.run()
    assert dp.st["trang_thai"] == "HONG"
    assert len(kg.pushes) == 1


def test_cau_chi_max_lan(moi_truong, monkeypatch):
    monkeypatch.setattr(ko, "MAX_LAN", 3)
    kb = {(t, 0): [("dung_giua", 0)] for t in "abc"}
    kg = KaggleGia(moi_truong, {"a": 20, "b": 10, "c": 9}, kb)
    dp = _dp(moi_truong, kg, tong=3, so_shard=1)
    dp.run()
    assert dp.watch() == 2
    assert "MAX_LAN" in dp.st["ly_do_hong"]
    assert len(kg.pushes) == 3


def test_khong_acc_nao_du_quota_thi_dung_theo_doi(moi_truong):
    kg = KaggleGia(moi_truong, {"a": 0.5, "b": 0}, {("a", 0): ["push_quota"], ("b", 1): ["push_quota"]})
    dp = _dp(moi_truong, kg, accs=("a", "b"))
    dp.run()
    assert dp.watch() == 3


def test_job_cpu_khong_mang_machine_shape(moi_truong):
    kg = KaggleGia(moi_truong, {}, {})
    dp = _dp(moi_truong, kg, gpu=False)
    dp.run()
    assert all("machine_shape" not in p["meta"] and p["meta"]["enable_gpu"] is False for p in kg.pushes)


def test_nhan_hai_chia_round_robin_moi_acc_hai_kernel():
    """nhan=2: 3 acc → 6 shard, acc lặp a,b,c,a,b,c. GPU thứ 2 cùng acc không đẩy song song."""
    accs = [ko.Acc("a", "t"), ko.Acc("b", "t"), ko.Acc("c", "t")]
    job = ko.Job(ten="t", notebook="n.ipynb", tong=12, output="o.jsonl", gpu=False, nhan=2)
    dp = ko.DieuPhoi(job, kg=None, accs=accs, ngu=lambda *_: None, commit="x",
                     day_src=lambda *_: None, log=lambda *_: None)
    st = dp.lap_ke_hoach()
    assert [s["acc"] for s in st["shards"]] == ["a", "b", "c", "a", "b", "c"]
    assert [(s["tu"], s["den"]) for s in st["shards"]] == ko.chia_deu(12, 6)


def test_job_that_trong_repo_doc_duoc():
    goc = Path(__file__).resolve().parents[1]
    for f in sorted((goc / "kaggle" / "jobs").glob("*.json")):
        job = ko.Job.doc(f)
        nb = json.loads((goc / job.notebook).read_text(encoding="utf-8"))
        ko.chen_params(nb, {"tu": 0})
