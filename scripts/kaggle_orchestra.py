"""Chia một job theo câu ra nhiều tài khoản Kaggle, theo dõi, đổi acc khi lỗi môi trường, rồi ghép.

Vì sao có file này: quota GPU 30 giờ/tuần là của TỪNG tài khoản, còn infer 1.000 câu ~7 giờ
và build data ~2,6 giờ đều chia được theo câu. Trước đây đổi acc bằng tay (ghi đè
~/.kaggle/access_token) — một lần ghi đè giữa lúc lệnh khác đang chạy là lệnh đó đẩy
kernel lên nhầm chủ. Ở đây mỗi lệnh `kaggle` mang biến KAGGLE_API_TOKEN riêng, không
chạm file dùng chung.

Luật đã chốt với người dùng (phiên grill 2026-09-16):
- Chia ĐỀU theo khoảng liền nhau `--tu/--den`, cho MỌI acc có token, không lọc quota.
- Lỗi môi trường (hết quota, chạm trần 12 giờ, mạng, kernel dừng khi chưa xong) -> tự
  đổi acc khác, NỐI TIẾP phần đã làm.
- Lỗi code hoặc OOM -> cả lượt HỎNG: không đẩy gì thêm, in link để người dùng hủy tay
  (Kaggle không có API hủy kernel — `cancel_kernel_session` cần session id mà không
  lệnh nào trả về khi kernel đang chạy). Người dùng sửa rồi gọi `run` lại.
- `run` lại LUÔN giữ output cũ, chỉ chạy phần thiếu.

Hợp đồng với script chạy trên Kaggle (xem `ghi_moc`): script nhận `--tu/--den`, nối tiếp
file đầu ra nếu đã có, và CHỈ khi làm xong cả khoảng mới ghi `<đầu ra>.moc.json`. Không
có mốc = chưa xong, bất kể Kaggle báo COMPLETE — `infer.py` tự dừng khi hết ngân sách giờ
mà vẫn thoát 0.

    python scripts/kaggle_orchestra.py plan  kaggle/jobs/infer_hybrid.json
    python scripts/kaggle_orchestra.py run   kaggle/jobs/infer_hybrid.json --yes
    python scripts/kaggle_orchestra.py watch kaggle/jobs/infer_hybrid.json
    python scripts/kaggle_orchestra.py status|merge kaggle/jobs/infer_hybrid.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

GOC = Path(os.environ.get("ORCHESTRA_DIR", "orchestra"))
PARAMS_TAG = "# ORCHESTRA-PARAMS"
# Trần số lần đẩy cho MỘT shard. Không phải chính sách retry — người dùng chọn tự đổi acc
# không giới hạn theo loại lỗi — mà là cầu chì: ràng buộc repo cấm vòng lặp vô hạn, và
# một lỗi môi trường bị phân loại nhầm sẽ đốt quota của mọi acc lần lượt.
MAX_LAN = 6
# Trạng thái kết thúc đọc được ngay sau khi đẩy có thể là của VERSION TRƯỚC: Kaggle cần
# vài chục giây mới gắn slug với version mới. Tin nó là đánh dấu XONG một shard chưa chạy.
GIAY_TIN_STATUS = 120
# Acc còn ít hơn chừng này giờ GPU thì không nhận shard ĐỔI SANG. Lúc chia ban đầu không
# lọc (người dùng chọn), nhưng chuyển một shard vừa chết vì hết quota sang acc 0 giờ là
# chắc chắn chết lần hai.
GIO_TOI_THIEU = 1.0


# ---------------------------------------------------------------- hợp đồng phía Kaggle


def ghi_moc(ra: Path, tu: int, den: int) -> dict:
    """Ghi mốc HOÀN THÀNH cạnh file đầu ra. Script chỉ gọi khi đã xử lý hết [tu, den).

    `so_dong` đếm dòng HỢP LỆ có `id` chứ không đếm (den - tu): build_sft bỏ mẫu trượt
    gold (21,3% ở phiên 3) nên số dòng ít hơn số câu là bình thường. `merge` so số dòng
    tải về với con số này để bắt file bị cắt khi tải.
    """
    ids = doc_ids(ra)
    moc = {"tu": tu, "den": den, "so_dong": len(ids), "xong": True}
    Path(str(ra) + ".moc.json").write_text(json.dumps(moc), encoding="utf-8")
    return moc


def doc_ids(duong: Path) -> list[str]:
    """Id theo thứ tự dòng. Dòng vỡ (kernel bị giết giữa lúc ghi) bị bỏ — mất một câu
    rẻ hơn mất cả file, và câu đó sẽ được chạy lại khi nối tiếp."""
    if not duong.exists():
        return []
    ids = []
    for line in duong.open(encoding="utf-8"):
        try:
            ids.append(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return ids


def bo_dong_vo(duong: Path) -> None:
    """Viết lại file chỉ giữ dòng JSON hợp lệ. Phải gọi TRƯỚC khi mở "a": dòng cuối vỡ không
    có "\n", dòng append đầu tiên dính vào nó và câu đó mất luôn (test bắt được)."""
    if not duong.exists():
        return
    giu = []
    for line in duong.open(encoding="utf-8"):
        try:
            json.loads(line)["id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        giu.append(line if line.endswith("\n") else line + "\n")
    duong.write_text("".join(giu), encoding="utf-8")


# ---------------------------------------------------------------- phần thuần, test được


@dataclass
class Job:
    ten: str
    notebook: str
    tong: int
    output: str  # tên file trong /kaggle/working/out/ mang mốc
    merge: str = "jsonl"  # "jsonl" | "submission"
    gpu: bool = True
    machine_shape: str = "NvidiaTeslaT4"
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    so_shard: int = 0  # 0 = bằng số acc
    publish: dict | None = None

    @staticmethod
    def doc(duong: Path) -> "Job":
        d = json.loads(Path(duong).read_text(encoding="utf-8"))
        d.pop("_ghi_chu", None)
        job = Job(**d)
        # Slug kernel và tên dataset nối tiếp đều sinh từ `ten`; Kaggle chỉ nhận chữ
        # thường, số, gạch ngang — sai là push 400 sau khi đã đẩy src lên mọi acc.
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", job.ten), f"ten không hợp lệ: {job.ten}"
        assert job.merge in ("jsonl", "submission"), job.merge
        return job


def chia_deu(tong: int, k: int) -> list[tuple[int, int]]:
    """Khoảng liền nhau, phần dư dồn cho các shard đầu: 10 câu / 3 -> 4, 3, 3."""
    assert tong > 0 and k > 0
    k = min(k, tong)
    co, du = divmod(tong, k)
    kq, tu = [], 0
    for i in range(k):
        den = tu + co + (i < du)
        kq.append((tu, den))
        tu = den
    return kq


OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|CalledProcessError")
# Traceback do MẠNG không phải lỗi code: tải reranker/e5 từ HuggingFace hoặc pip chết giữa
# chừng cũng in Traceback, và coi đó là lỗi code là dừng cả lượt vì một cú DNS.
MANG_RE = re.compile(
    r"ConnectionError|NameResolutionError|MaxRetryError|ReadTimeout|ConnectTimeout|"
    r"RemoteDisconnected|LocalEntryNotFoundError|IncompleteRead|Temporary failure in name resolution"
)


def van_ban_log(raw: str) -> str:
    """Log Kaggle là mảng JSON `[{stream_name, time, data}, ...]`; bản cũ là text thô."""
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            return "".join(str(x.get("data", "")) for x in d if isinstance(x, dict))
    except json.JSONDecodeError:
        pass
    return raw


def phan_loai(log: str) -> str:
    """"oom" | "code" | "moi_truong". Xét OOM trước: OOM của torch cũng in Traceback."""
    if OOM_RE.search(log):
        return "oom"
    if TRACEBACK_RE.search(log):
        # Chỉ nhìn đuôi: một cảnh báo mạng lúc pip ở đầu log không được phép che lỗi
        # KeyError thật ở cuối.
        return "moi_truong" if MANG_RE.search(log[-3000:]) else "code"
    return "moi_truong"


# Lỗi CẤU HÌNH lúc push (dataset sai chủ, slug sai, notebook hỏng): đổi acc không chữa
# được, nên xếp cùng nhóm lỗi code. Còn lại (quota, số phiên đồng thời, mạng) là môi trường.
# Mẫu "not valid dataset sources" là lỗi thật đã gặp 2026-09-15.
PUSH_CAU_HINH_RE = re.compile(r"not valid|[Ii]nvalid|not found|404|403|[Pp]ermission")


def phan_loai_push(text: str) -> str:
    if "successfully pushed" in text:
        return "ok"
    if re.search(r"quota|[Ll]imit|concurrent|[Mm]aximum", text):
        return "moi_truong"
    return "code" if PUSH_CAU_HINH_RE.search(text) else "moi_truong"


def chen_params(nb: dict, shard: dict) -> dict:
    """Thay cell `# ORCHESTRA-PARAMS` bằng SHARD của shard này. Đúng MỘT cell: hai cell
    thì cell sau ghi đè cell trước và shard nào cũng chạy cùng một khoảng câu."""
    nb = json.loads(json.dumps(nb))
    vt = [i for i, c in enumerate(nb["cells"])
          if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(PARAMS_TAG)]
    assert len(vt) == 1, f"notebook phải có đúng 1 cell {PARAMS_TAG}, đang có {len(vt)}"
    src = (f"{PARAMS_TAG} — sinh bởi scripts/kaggle_orchestra.py, đừng sửa tay\n"
           f"SHARD = {shard!r}\n")  # repr, không json: null/false không phải Python
    nb["cells"][vt[0]]["source"] = src.splitlines(keepends=True)
    return nb


def gop_jsonl(cu: Path, moi: Path) -> int:
    """Hợp hai file theo id, GIỮ dòng cũ. Kernel nối tiếp đã chép file cũ vào đầu ra của nó,
    nên không khử trùng thì mỗi lần đổi acc là nhân đôi phần đầu."""
    bo_dong_vo(cu)
    co = set(doc_ids(cu))
    them = 0
    with cu.open("a", encoding="utf-8") as f:
        for line in moi.open(encoding="utf-8"):
            try:
                i = json.loads(line)["id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if i not in co:
                co.add(i)
                f.write(line if line.endswith("\n") else line + "\n")
                them += 1
    return them


def kiem_ghep(job: Job, shards: list[dict], thu_muc: Path) -> list[str]:
    """Lỗi chặn ghép; rỗng = ghép được. Thu thập HẾT lỗi thay vì dừng ở lỗi đầu: ghép hỏng
    một lần rồi sửa từng lỗi một là mỗi vòng tốn một lần tải output từ Kaggle."""
    loi = []
    ks = sorted(shards, key=lambda s: s["tu"])
    if not ks or ks[0]["tu"] != 0 or ks[-1]["den"] != job.tong:
        loi.append(f"các shard không phủ [0, {job.tong})")
    for a, b in zip(ks, ks[1:]):
        if a["den"] != b["tu"]:
            loi.append(f"khoảng hở/chồng giữa s{a['i']} [{a['tu']},{a['den']}) và s{b['i']} [{b['tu']},{b['den']})")
    thay: dict[str, int] = {}
    for s in ks:
        f = thu_muc / f"s{s['i']}" / "out" / job.output
        moc_f = Path(str(f) + ".moc.json")
        if not moc_f.exists():
            loi.append(f"s{s['i']}: chưa có mốc hoàn thành ({moc_f.name})")
            continue
        moc = json.loads(moc_f.read_text(encoding="utf-8"))
        if (moc["tu"], moc["den"]) != (s["tu"], s["den"]):
            loi.append(f"s{s['i']}: mốc ghi [{moc['tu']},{moc['den']}) khác khoảng được giao")
        ids = doc_ids(f)
        # Mốc ghi trước, file có thể được nối thêm ở lượt sau (gop_jsonl) — nên >=, không ==.
        if len(ids) < moc["so_dong"]:
            loi.append(f"s{s['i']}: {len(ids)} dòng < {moc['so_dong']} dòng trong mốc — file tải về bị cắt")
        if job.merge == "submission" and len(set(ids)) != s["den"] - s["tu"]:
            loi.append(f"s{s['i']}: {len(set(ids))} câu có đáp án, cần {s['den'] - s['tu']}")
        for i in ids:
            if i in thay and thay[i] != s["i"]:
                loi.append(f"id {i} nằm ở cả s{thay[i]} và s{s['i']}")
            thay[i] = s["i"]
    if job.merge == "submission" and len(thay) != job.tong and not loi:
        loi.append(f"ghép ra {len(thay)} câu, cần {job.tong}")
    return loi


def ghep(job: Job, shards: list[dict], thu_muc: Path, ra: Path) -> Path:
    ra.mkdir(parents=True, exist_ok=True)
    ks = sorted(shards, key=lambda s: s["tu"])
    if job.merge == "jsonl":
        dich = ra / job.output
        with dich.open("w", encoding="utf-8") as g:
            for s in ks:
                seen = set()
                for line in (thu_muc / f"s{s['i']}" / "out" / job.output).open(encoding="utf-8"):
                    try:
                        i = json.loads(line)["id"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    if i not in seen:
                        seen.add(i)
                        g.write(line if line.endswith("\n") else line + "\n")
        return dich
    # submission: định dạng BTC `{id: {"answer": str}}`. Thiếu một id là eval_qa raise.
    sub = {}
    for s in ks:
        for line in (thu_muc / f"s{s['i']}" / "out" / job.output).open(encoding="utf-8"):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            sub.setdefault(x["id"], {"answer": x.get("du_doan") or ""})
    js = ra / "submission.json"
    js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(ra / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(js, "submission.json")
    rong = sum(1 for v in sub.values() if not v["answer"].strip())
    print(f"submission: {len(sub)} câu, {rong} rỗng")
    return ra / "submission.zip"


# ---------------------------------------------------------------- Kaggle CLI


@dataclass
class Acc:
    ten: str
    token: str  # ĐƯỜNG DẪN file token, không bao giờ là giá trị token


def tim_acc(thu_muc: Path = Path.home() / ".kaggle") -> list[Acc]:
    """Mọi `access_token` trong ~/.kaggle và ~/.kaggle/backup_*/. Tên acc hỏi Kaggle chứ
    không đọc tên thư mục: `~/.kaggle/access_token` là acc ĐANG dùng, đổi theo lần login
    gần nhất, trùng với một backup nào đó là chuyện thường."""
    ung_vien = [thu_muc / "access_token"] + sorted(thu_muc.glob("backup_*/access_token"))
    kq, da = [], set()
    for p in ung_vien:
        if not p.exists():
            continue
        ten = KaggleCli.username(str(p))
        if ten and ten not in da:
            da.add(ten)
            kq.append(Acc(ten, str(p)))
    return kq


class KaggleCli:
    """Mỏng quanh `kaggle` CLI. Mọi lời gọi mang KAGGLE_API_TOKEN=<file> của đúng acc."""

    @staticmethod
    def _chay(token: str, args: list[str], timeout: int = 600) -> tuple[int, str]:
        env = {**os.environ, "KAGGLE_API_TOKEN": token}
        # Mạng máy dev chập chờn: `kernels output` từng chết vì NameResolutionError. Ba lần
        # là đủ phân biệt chập chờn với mất mạng thật.
        for lan in range(3):
            try:
                p = subprocess.run(["kaggle", *args], env=env, capture_output=True,
                                   text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                out, rc = "timeout", 124
            else:
                out, rc = p.stdout + p.stderr, p.returncode
            if not MANG_RE.search(out) and rc != 124:
                return rc, out
            time.sleep(10 * (lan + 1))
        return rc, out

    @staticmethod
    def username(token: str) -> str:
        _, out = KaggleCli._chay(token, ["config", "view"], timeout=60)
        m = re.search(r"^- username: (\S+)", out, re.M)
        return m.group(1) if m else ""

    def quota_gpu(self, acc: Acc) -> float:
        _, out = self._chay(acc.token, ["quota", "--csv"], timeout=60)
        m = re.search(r"^GPU,[^,]*,([\d.]+)h", out, re.M)
        return float(m.group(1)) if m else 0.0

    def push_kernel(self, acc: Acc, thu_muc: Path) -> str:
        return self._chay(acc.token, ["kernels", "push", "-p", str(thu_muc)])[1]

    def status(self, acc: Acc, slug: str) -> str:
        """RUNNING | QUEUED | COMPLETE | ERROR | CANCEL... ; rỗng nếu chưa đọc được."""
        _, out = self._chay(acc.token, ["kernels", "status", f"{acc.ten}/{slug}"], timeout=120)
        m = re.search(r"KernelWorkerStatus\.(\w+)", out)
        return m.group(1) if m else ""

    def output(self, acc: Acc, slug: str, dich: Path) -> None:
        dich.mkdir(parents=True, exist_ok=True)
        self._chay(acc.token, ["kernels", "output", f"{acc.ten}/{slug}", "-p", str(dich), "--force"],
                   timeout=1800)

    def day_dataset(self, acc: Acc, thu_muc: Path, ten: str, msg: str) -> None:
        (thu_muc / "dataset-metadata.json").write_text(json.dumps(
            {"title": ten, "id": f"{acc.ten}/{ten}", "licenses": [{"name": "unknown"}]}))
        # Hỏi TỒN TẠI trước: `datasets create` trên dataset đã có in "title already in use"
        # rồi vẫn thoát 0 (push_dataset.sh ghi lại lỗi này).
        rc, _ = self._chay(acc.token, ["datasets", "files", f"{acc.ten}/{ten}"], timeout=120)
        if rc == 0:
            self._chay(acc.token, ["datasets", "version", "-p", str(thu_muc), "-r", "zip", "-m", msg])
        else:
            self._chay(acc.token, ["datasets", "create", "-p", str(thu_muc), "-r", "zip"])

    def dataset_san_sang(self, acc: Acc, ten: str) -> bool:
        _, out = self._chay(acc.token, ["datasets", "status", f"{acc.ten}/{ten}"], timeout=120)
        return "ready" in out

    def tai_dataset(self, acc: Acc, ten: str, dich: Path) -> None:
        self._chay(acc.token, ["datasets", "download", f"{acc.ten}/{ten}", "-p", str(dich),
                               "--unzip", "--force"])


# ---------------------------------------------------------------- điều phối


def thu_muc_job(job: Job) -> Path:
    return GOC / job.ten


def doc_state(job: Job) -> dict | None:
    f = thu_muc_job(job) / "state.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def ghi_state(job: Job, st: dict) -> None:
    d = thu_muc_job(job)
    d.mkdir(parents=True, exist_ok=True)
    tam = d / "state.json.tmp"
    tam.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    # Ghi-rồi-đổi-tên: Ctrl-C giữa lúc ghi không được để lại state.json nửa vời, vì state
    # là thứ DUY NHẤT biết shard nào đang chạy trên acc nào.
    tam.replace(d / "state.json")


def commit_hien_tai() -> str:
    return subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True,
                          text=True).stdout.strip()


class DieuPhoi:
    def __init__(self, job: Job, kg, accs: list[Acc], ngu=time.sleep, commit: str = "",
                 day_src=None, log=print):
        self.job, self.kg, self.ngu, self.log = job, kg, ngu, log
        self.accs = {a.ten: a for a in accs}
        self.commit = commit or commit_hien_tai()
        self.day_src = day_src if day_src is not None else self._day_src_that
        self.st = doc_state(job)

    # -- kế hoạch

    def lap_ke_hoach(self) -> dict:
        ten = list(self.accs)
        if self.job.gpu:
            # Không LỌC (người dùng chọn), chỉ SẮP: shard 0 vào acc nhiều quota nhất, để
            # acc 0 giờ — chắc chắn fail rồi đổi — nằm cuối và chờ acc rảnh sớm nhất.
            q = {t: self.kg.quota_gpu(self.accs[t]) for t in ten}
            ten.sort(key=lambda t: -q[t])
        k = self.job.so_shard or len(ten)
        assert k <= len(ten), f"so_shard={k} > {len(ten)} acc"
        shards = [{"i": i, "tu": tu, "den": den, "acc": ten[i], "slug": f"{self.job.ten}-s{i}",
                   "trang_thai": "CHO", "lan": 0, "day_luc": 0, "noi_tiep": None, "lich_su": []}
                  for i, (tu, den) in enumerate(chia_deu(self.job.tong, k))]
        return {"job": self.job.ten, "trang_thai": "DANG_CHAY", "commit": self.commit,
                "src_da_day": {}, "shards": shards}

    def in_bang(self) -> None:
        self.log(f"job {self.st['job']}  [{self.st['trang_thai']}]  commit {self.st['commit']}")
        self.log(f"{'shard':6} {'khoảng':>13} {'acc':14} {'lần':>3}  trạng thái")
        for s in self.st["shards"]:
            ly = s["lich_su"][-1]["ly_do"] if s["lich_su"] else ""
            self.log(f"s{s['i']:<5} {s['tu']:>6}-{s['den']:<6} {s['acc']:14} {s['lan']:>3}  "
                     f"{s['trang_thai']} {ly}")

    # -- src

    def _day_src_that(self, acc: Acc) -> None:
        """Đẩy `legalqa-v10-src` lên acc rồi TẢI VỀ kiểm COMMIT.txt. Không kiểm là tái diễn
        lỗi 2026-09-14: server giữ bản cũ, kernel chạy 3,5 giờ trên code cũ."""
        with tempfile.TemporaryDirectory() as tam:
            d = Path(tam) / "src"
            d.mkdir()
            ar = subprocess.run(["git", "archive", "HEAD"], capture_output=True, check=True)
            subprocess.run(["tar", "-x", "-C", str(d)], input=ar.stdout, check=True)
            (d / "COMMIT.txt").write_text(self.commit + "\n", encoding="utf-8")
            self.kg.day_dataset(acc, d, "legalqa-v10-src", self.commit)
            for _ in range(20):
                self.ngu(30)
                t = Path(tam) / "kiem"
                shutil.rmtree(t, ignore_errors=True)
                self.kg.tai_dataset(acc, "legalqa-v10-src", t)
                c = t / "COMMIT.txt"
                if c.exists() and c.read_text(encoding="utf-8").strip() == self.commit:
                    return
        raise RuntimeError(f"{acc.ten}: legalqa-v10-src không lên đúng commit {self.commit}")

    def dam_bao_src(self, ten_acc: str) -> None:
        if self.st["src_da_day"].get(ten_acc) == self.commit:
            return
        self.log(f"  đẩy src lên {ten_acc} ...")
        self.day_src(self.accs[ten_acc])
        self.st["src_da_day"][ten_acc] = self.commit
        ghi_state(self.job, self.st)

    # -- đẩy một shard

    def _nguon(self, acc: str, s: dict) -> list[str]:
        ds = [d if "/" in d else f"{acc}/{d}" for d in self.job.datasets]
        if s["noi_tiep"]:
            ds.append(f"{acc}/{s['noi_tiep']}")
        return ds

    def _chuan_bi_noi_tiep(self, s: dict, acc: str) -> None:
        """Đẩy phần output dở của shard thành dataset CỦA acc sắp chạy: output kernel không
        đọc chéo được giữa các tài khoản, dataset thì được."""
        out = thu_muc_job(self.job) / f"s{s['i']}" / "out"
        if not any(doc_ids(f) for f in out.glob("*.jsonl")):
            s["noi_tiep"] = None
            return
        ten = f"orch-{self.job.ten}-s{s['i']}"
        with tempfile.TemporaryDirectory() as tam:
            d = Path(tam) / ten
            shutil.copytree(out, d)
            for m in d.glob("*.moc.json"):
                m.unlink()  # mốc cũ không được theo sang: nó sẽ khiến lượt mới trông như đã xong
            self.kg.day_dataset(self.accs[acc], d, ten, f"noi tiep s{s['i']} lan {s['lan']}")
        for _ in range(40):
            if self.kg.dataset_san_sang(self.accs[acc], ten):
                break
            self.ngu(15)
        s["noi_tiep"] = ten

    def day(self, s: dict, acc: str) -> str:
        """Đẩy shard lên acc. Trả kết quả phan_loai_push."""
        self.dam_bao_src(acc)
        s["acc"] = acc
        self._chuan_bi_noi_tiep(s, acc)
        shard = {"job": self.job.ten, "chi_so": s["i"], "so_shard": len(self.st["shards"]),
                 "tu": s["tu"], "den": s["den"], "commit": self.commit,
                 "noi_tiep": s["noi_tiep"], "output": self.job.output, **self.job.params}
        nb = chen_params(json.loads(Path(self.job.notebook).read_text(encoding="utf-8")), shard)
        with tempfile.TemporaryDirectory() as tam:
            d = Path(tam)
            (d / f"{s['slug']}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
            meta = {
                "id": f"{acc}/{s['slug']}", "title": s["slug"], "code_file": f"{s['slug']}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": self.job.gpu, "enable_internet": True,
                "dataset_sources": self._nguon(acc, s), "competition_sources": [],
                "kernel_sources": [], "model_sources": self.job.models,
            }
            # Chỉ job GPU mang machine_shape: kernels_push gán `request.machine_shape = acc
            # or meta["machine_shape"]`, nên job CPU mà mang "NvidiaTeslaT4" là đốt quota GPU.
            if self.job.gpu:
                meta["machine_shape"] = self.job.machine_shape
            (d / "kernel-metadata.json").write_text(json.dumps(meta))
            text = self.kg.push_kernel(self.accs[acc], d)
        kq = phan_loai_push(text)
        s["lan"] += 1
        if kq == "ok":
            s["trang_thai"], s["day_luc"] = "DANG_CHAY", time.time()
            self.log(f"  s{s['i']} -> {acc}/{s['slug']} (lần {s['lan']})")
        else:
            dong = text.strip().splitlines()[-1][:200] if text.strip() else "(rỗng)"
            s["lich_su"].append({"acc": acc, "ket_qua": f"push_{kq}", "ly_do": dong})
            s["trang_thai"] = "CHO_ACC" if kq == "moi_truong" else "LOI"
            self.log(f"  s{s['i']} push lên {acc} thất bại ({kq}): {dong}")
            if kq == "code":
                self._hong(s, dong)
        ghi_state(self.job, self.st)
        return kq

    # -- vòng theo dõi

    def _ban(self) -> set[str]:
        return {s["acc"] for s in self.st["shards"] if s["trang_thai"] == "DANG_CHAY"}

    def chon_acc(self, s: dict) -> str | None:
        """Acc rảnh (không chạy shard nào của job này), nhiều quota nhất, tránh acc vừa làm
        chết shard nếu còn lựa chọn khác."""
        ban = self._ban()
        ung = [t for t in self.accs if t not in ban]
        if self.job.gpu:
            q = {t: self.kg.quota_gpu(self.accs[t]) for t in ung}
            ung = sorted((t for t in ung if q[t] >= GIO_TOI_THIEU), key=lambda t: -q[t])
        vua_chet = s["lich_su"][-1]["acc"] if s["lich_su"] else None
        khac = [t for t in ung if t != vua_chet]
        return (khac or ung or [None])[0]

    def _tai_output(self, s: dict) -> Path:
        """Tải output lần này rồi GỘP vào `s<i>/out/` theo id. Giữ output cũ là quyết định
        của người dùng: `run` lại không bao giờ xoá phần đã làm."""
        goc = thu_muc_job(self.job) / f"s{s['i']}"
        lan = goc / f"lan{s['lan']}"
        shutil.rmtree(lan, ignore_errors=True)
        self.kg.output(self.accs[s["acc"]], s["slug"], lan)
        out = goc / "out"
        out.mkdir(parents=True, exist_ok=True)
        moi = lan / "out" if (lan / "out").exists() else lan
        for f in moi.glob("*.jsonl"):
            if (out / f.name).exists():
                gop_jsonl(out / f.name, f)
            else:
                shutil.copy(f, out / f.name)
        for m in moi.glob("*.moc.json"):
            shutil.copy(m, out / m.name)
        return lan

    def _log_cua(self, lan: Path) -> str:
        return "".join(van_ban_log(p.read_text(encoding="utf-8", errors="replace"))
                       for p in lan.glob("*.log"))

    def _hong(self, s: dict, ly_do: str) -> None:
        self.st["trang_thai"] = "HONG"
        self.st["ly_do_hong"] = f"s{s['i']} trên {s['acc']}: {ly_do}"
        for x in self.st["shards"]:
            # Shard chờ acc xử lý TRƯỚC shard hỏng trong cùng tick thì đang CHO_ACC; gắn DUNG
            # để bảng không gợi ý là nó sẽ còn được đẩy.
            if x["trang_thai"] == "CHO_ACC":
                x["trang_thai"] = "DUNG"
        con = [x for x in self.st["shards"] if x["trang_thai"] == "DANG_CHAY"]
        self.log("\n!!! LƯỢT HỎNG — không đẩy thêm gì. " + self.st["ly_do_hong"])
        if con:
            self.log("Hủy tay các kernel còn chạy (Kaggle không có API hủy):")
            for x in con:
                self.log(f"  https://www.kaggle.com/code/{x['acc']}/{x['slug']}")
        self.log("Sửa code/job rồi `run` lại — output đã có được giữ, chỉ chạy phần thiếu.")

    def xu_ly(self, s: dict) -> None:
        """Một shard DANG_CHAY: đọc status, và nếu đã dừng thì phân loại."""
        tt = self.kg.status(self.accs[s["acc"]], s["slug"])
        if tt in ("", "RUNNING", "QUEUED", "NEW", "CANCEL_REQUESTED"):
            return
        if time.time() - s["day_luc"] < GIAY_TIN_STATUS:
            return
        lan = self._tai_output(s)
        moc = thu_muc_job(self.job) / f"s{s['i']}" / "out" / (self.job.output + ".moc.json")
        if tt == "COMPLETE" and moc.exists():
            s["trang_thai"] = "XONG"
            s["lich_su"].append({"acc": s["acc"], "ket_qua": "xong", "ly_do": ""})
            self.log(f"  s{s['i']} XONG trên {s['acc']}")
            return
        if tt == "COMPLETE":
            loai, ly = "moi_truong", "COMPLETE nhưng chưa có mốc (hết ngân sách giờ trong script)"
        else:
            log = self._log_cua(lan)
            loai = phan_loai(log)
            dong = [l for l in log.strip().splitlines() if l.strip()]
            ly = f"{tt}: " + (dong[-1][:200] if dong else "log rỗng")
        s["lich_su"].append({"acc": s["acc"], "ket_qua": loai, "ly_do": ly})
        self.log(f"  s{s['i']} dừng trên {s['acc']} — {loai}: {ly}")
        if self.st["trang_thai"] == "HONG":
            # Lượt đã hỏng: kernel này dừng vì người dùng hủy tay. Chỉ thu output về.
            s["trang_thai"] = "DUNG"
        elif loai in ("oom", "code"):
            s["trang_thai"] = "LOI"
            self._hong(s, ly)
        else:
            s["trang_thai"] = "CHO_ACC"

    def tick(self) -> None:
        for s in self.st["shards"]:
            if s["trang_thai"] == "DANG_CHAY":
                self.xu_ly(s)
                ghi_state(self.job, self.st)
        if self.st["trang_thai"] == "HONG":
            return
        for s in self.st["shards"]:
            if s["trang_thai"] != "CHO_ACC":
                continue
            if s["lan"] >= MAX_LAN:
                s["trang_thai"] = "LOI"
                self._hong(s, f"đã đẩy {s['lan']} lần, chạm cầu chì MAX_LAN")
                ghi_state(self.job, self.st)
                return
            acc = self.chon_acc(s)
            if acc is None:
                continue
            if self.day(s, acc) == "code":
                return

    def dang_chay(self) -> bool:
        return any(s["trang_thai"] == "DANG_CHAY" for s in self.st["shards"])

    def watch(self, moi: int = 300) -> int:
        """0 = mọi shard XONG; 2 = lượt hỏng; 3 = còn shard chờ nhưng không acc nào nhận."""
        while True:
            self.tick()
            ss = self.st["shards"]
            if all(s["trang_thai"] == "XONG" for s in ss):
                self.st["trang_thai"] = "XONG"
                ghi_state(self.job, self.st)
                self.log("mọi shard XONG — chạy `merge`")
                return 0
            if not self.dang_chay():
                if self.st["trang_thai"] == "HONG":
                    return 2
                if any(s["trang_thai"] == "CHO_ACC" for s in ss):
                    self.log("còn shard chờ mà không acc nào đủ quota và rảnh — dừng theo dõi")
                    return 3
            self.ngu(moi)

    def run(self, ke_hoach: dict | None = None) -> None:
        if self.st is None or self.st["trang_thai"] == "XONG":
            # Nhận bảng người dùng vừa duyệt thay vì lập lại: quota đổi giữa hai lần hỏi là
            # thứ tự acc đổi, và thứ được đẩy khác thứ được duyệt.
            self.st = ke_hoach or self.lap_ke_hoach()
        else:
            # Chạy lại sau HỎNG: commit mới, giữ output cũ. Shard XONG giữ nguyên; shard dở
            # quay về hàng chờ và nối tiếp từ `s<i>/out/`.
            self.st["trang_thai"] = "DANG_CHAY"
            self.st.pop("ly_do_hong", None)
            self.st["commit"] = self.commit
            assert not self.dang_chay(), "còn shard DANG_CHAY — `watch` cho tới khi dừng hết rồi mới run lại"
            for s in self.st["shards"]:
                if s["trang_thai"] != "XONG":
                    s["trang_thai"], s["lan"] = "CHO", 0
        ghi_state(self.job, self.st)
        for s in self.st["shards"]:
            if s["trang_thai"] != "CHO":
                continue
            # Lượt đầu mỗi shard thử đúng acc được chia, KHÔNG lọc quota (người dùng chọn).
            acc = s["acc"] if s["acc"] not in self._ban() else self.chon_acc(s)
            if acc is None:
                s["trang_thai"] = "CHO_ACC"
                continue
            if self.day(s, acc) == "code":
                return

    def merge(self) -> Path:
        thu_muc = thu_muc_job(self.job)
        loi = kiem_ghep(self.job, self.st["shards"], thu_muc)
        if loi:
            raise SystemExit("KHÔNG ghép được:\n  " + "\n  ".join(loi))
        dich = ghep(self.job, self.st["shards"], thu_muc, thu_muc / "merged")
        self.log(f"-> {dich}")
        if self.job.publish:
            acc = self.accs[self.job.publish["acc"]]
            ten = self.job.publish["dataset"]
            with tempfile.TemporaryDirectory() as tam:
                d = Path(tam) / ten
                d.mkdir()
                shutil.copy(dich, d / dich.name)
                self.kg.day_dataset(acc, d, ten, f"ghep {self.job.ten} {self.commit}")
            self.log(f"publish -> https://www.kaggle.com/datasets/{acc.ten}/{ten}")
        return dich


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lenh", choices=["plan", "run", "watch", "status", "merge"])
    ap.add_argument("job")
    ap.add_argument("--yes", action="store_true", help="run: bỏ bước xác nhận bảng chia")
    ap.add_argument("--moi", type=int, default=300, help="watch: giây giữa hai lần kiểm")
    ap.add_argument("--acc", default="", help="giới hạn acc, phân cách phẩy; rỗng = mọi acc có token")
    a = ap.parse_args()

    job = Job.doc(Path(a.job))
    if a.lenh in ("status",):
        st = doc_state(job)
        if not st:
            sys.exit(f"chưa có state cho {job.ten}")
        DieuPhoi(job, KaggleCli(), [], commit="-").in_bang()
        return
    accs = tim_acc()
    if a.acc:
        chon = set(a.acc.split(","))
        accs = [x for x in accs if x.ten in chon]
    assert accs, "không tìm thấy token nào trong ~/.kaggle"
    kg = KaggleCli()
    dp = DieuPhoi(job, kg, accs)

    if a.lenh == "plan":
        dp.st = dp.lap_ke_hoach()
        for t in dp.accs:
            q = f"{kg.quota_gpu(dp.accs[t]):.2f}h GPU" if job.gpu else "CPU"
            dp.log(f"  acc {t:14} {q}")
        dp.in_bang()
        return
    if a.lenh == "run":
        ke = None
        if dp.st is None or dp.st["trang_thai"] == "XONG":
            cu, dp.st = dp.st, dp.lap_ke_hoach()
            dp.in_bang()
            ke, dp.st = dp.st, cu
            if not a.yes and input("Đẩy theo bảng này? [y/N] ").strip().lower() != "y":
                return
        if subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0:
            dp.log("CHÚ Ý: cây làm việc có thay đổi chưa commit — src đẩy lên là HEAD, không gồm chúng")
        dp.run(ke)
        dp.in_bang()
        return
    if a.lenh == "watch":
        assert dp.st, "chưa `run`"
        sys.exit(dp.watch(a.moi))
    if a.lenh == "merge":
        assert dp.st, "chưa `run`"
        dp.merge()


if __name__ == "__main__":
    main()
