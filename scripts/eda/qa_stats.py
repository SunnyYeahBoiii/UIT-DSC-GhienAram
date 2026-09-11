import json, os, re, statistics as st
from collections import Counter

ROOT=os.environ.get("DATA_DIR","data")

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

train=load(f"{ROOT}/official/train.json")
pub=load(f"{ROOT}/official/public-official.json")
warm=load(f"{ROOT}/warmup/warmup.json")

def shape(name, d):
    print(f"\n### {name}")
    print("type:", type(d).__name__, "| n =", len(d))
    if isinstance(d, dict):
        k=next(iter(d)); print("sample key:", repr(k)[:60])
        print("item keys:", sorted(d[k].keys()) if isinstance(d[k],dict) else type(d[k]))
        items=list(d.values())
    else:
        print("item keys:", sorted(d[0].keys()) if isinstance(d[0],dict) else type(d[0]))
        items=d
    return items

tr=shape("train.json", train)
pb=shape("public-official.json", pub)
wm=shape("warmup.json", warm)

def pct(xs,p): 
    xs=sorted(xs); 
    return xs[min(len(xs)-1,int(len(xs)*p/100))]

def dist(name, vals):
    print(f"{name:<28} n={len(vals):<6} mean={st.mean(vals):7.1f} p10={pct(vals,10):<6} p50={pct(vals,50):<6} p90={pct(vals,90):<6} p99={pct(vals,99):<6} max={max(vals)}")

print("\n## Do dai (so tu)")
for nm,items in [("train",tr),("public",pb),("warmup",wm)]:
    qs=[len(str(i.get("question","")).split()) for i in items]
    dist(f"{nm} question", qs)
    ans=[len(str(i.get("answer") or "").split()) for i in items]
    if max(ans)>0: dist(f"{nm} answer", ans)
    else: print(f"{nm} answer{'':<21} TAT CA RONG/NULL")

print("\n## Truong khac trong item (ngoai question/answer)")
for nm,items in [("train",tr),("public",pb),("warmup",wm)]:
    extra=Counter()
    for i in items:
        for k in i: 
            if k not in ("question","answer"): extra[k]+=1
    print(f"{nm}: {dict(extra) or 'khong co'}")

print("\n## Question type (neu co truong)")
for nm,items in [("train",tr),("public",pb)]:
    for key in ("question_type","type","category"):
        if items and key in items[0]:
            print(nm,key,Counter(i[key] for i in items).most_common())

print("\n## Mo dau cau hoi train (3 tu dau, top 15)")
print(Counter(" ".join(str(i["question"]).split()[:3]) for i in tr).most_common(15))

print("\n## Tin hieu trich dan trong ANSWER train (ti le %)")
pats={
 "Dieu N":r"\bĐiều\s+\d+", "Khoan N":r"\bkhoản\s+\d+", "Diem x":r"\bđiểm\s+[a-zđ]\)",
 "Nghi dinh":r"Nghị định", "Thong tu":r"Thông tư", "Luat":r"\bLuật\s+[A-ZĐ]",
 "Bo luat":r"Bộ luật", "Quyet dinh":r"Quyết định", "so .../20xx":r"số\s+[\d/]+/\d{4}",
 "danh sach a) b)":r"\n\s*[a-đ]\)", "gach dau dong":r"\n\s*[-–•]\s",
 "Hinh tu Internet":r"\(Hình từ Internet\)", "Anh minh hoa":r"Ảnh minh họa",
 "tai day":r"tại đây", "[...]":r"\[\.\.\.\]", "truncated ...":r"\.\.\.\s*$",
}
n=len(tr)
for k,p in pats.items():
    c=sum(1 for i in tr if re.search(p, str(i.get("answer") or "")))
    print(f"{k:<22} {c:>5} / {n}  = {100*c/n:5.1f}%")

print("\n## Tin hieu trich dan trong QUESTION train (ti le %)")
for k,p in list(pats.items())[:9]:
    c=sum(1 for i in tr if re.search(p, str(i["question"])))
    print(f"{k:<22} {c:>5} / {n}  = {100*c/n:5.1f}%")

print("\n## Overlap question train vs public vs warmup")
def norm(s): return re.sub(r"\s+"," ",str(s)).strip().lower()
TS={norm(i["question"]) for i in tr}; PS={norm(i["question"]) for i in pb}; WS={norm(i["question"]) for i in wm}
print("train∩public:",len(TS&PS),"| train∩warmup:",len(TS&WS),"| public∩warmup:",len(PS&WS))
print("train unique q:",len(TS),"/",len(tr))

print("\n## Cau tra loi: so cau, so doan")
sents=[len(re.findall(r"[.!?]\s",str(i["answer"] or ""))) for i in tr]
dist("train answer sentences", sents)
paras=[str(i["answer"] or "").count("\n")+1 for i in tr]
dist("train answer lines", paras)
