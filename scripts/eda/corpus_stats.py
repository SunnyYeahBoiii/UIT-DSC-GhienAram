import json, os, re, os, statistics as st
from collections import Counter

D=os.environ.get("CORPUS_DIR","data/official/selected-contexts")
files=sorted(os.listdir(D))
print("so file:", len(files))

schemas=Counter(); lens=[]; empty=0; names=[]; links=[]
doc_types=Counter(); struct=Counter(); years=Counter()
rel=Counter()
REL={"sua doi":r"sửa đổi","bo sung":r"bổ sung","thay the":r"thay thế","bai bo":r"bãi bỏ",
     "het hieu luc":r"hết hiệu lực","huong dan":r"hướng dẫn","can cu":r"[Cc]ăn cứ"}
head_re=re.compile(r"^\s*(Điều\s+\d+[a-z]?\.)", re.M)
chuong_re=re.compile(r"^\s*(Chương\s+[IVXLC]+)", re.M)
muc_re=re.compile(r"^\s*(Mục\s+\d+)", re.M)
khoan_re=re.compile(r"^\s*(\d{1,2})\.\s", re.M)

for fn in files:
    with open(os.path.join(D,fn), encoding="utf-8") as f:
        d=json.load(f)
    schemas["|".join(sorted(d.keys()))]+=1
    p=str(d.get("passage") or "")
    if not p.strip(): empty+=1; continue
    lens.append(len(p.split()))
    nm=str(d.get("name") or "")
    if nm: names.append(nm)
    links.append(str(d.get("link") or ""))
    for t in ["Nghị định","Thông tư","Bộ luật","Luật","Quyết định","Nghị quyết","Pháp lệnh","Công văn","Chỉ thị","Hiến pháp"]:
        if nm.startswith(t) or re.search(rf"^{t}\b", nm): doc_types[t]+=1; break
    else: doc_types["(khac/khong ten)"]+=1
    y=re.search(r"\b(19|20)\d{2}\b", nm)
    if y: years[y.group(0)]+=1
    struct["co Dieu"]+= bool(head_re.search(p))
    struct["co Chuong"]+= bool(chuong_re.search(p))
    struct["co Muc"]+= bool(muc_re.search(p))
    struct["co Khoan so"]+= bool(khoan_re.search(p))
    for k,pat in REL.items():
        if re.search(pat,p): rel[k]+=1

def pct(xs,p):
    xs=sorted(xs); return xs[min(len(xs)-1,int(len(xs)*p/100))]

print("\n## Schema")
for k,v in schemas.most_common(): print(f"  {k:<30} {v}")
print("passage rong:", empty)

print("\n## Do dai passage (so tu)")
print(f"n={len(lens)} mean={st.mean(lens):.0f} p10={pct(lens,10)} p50={pct(lens,50)} p90={pct(lens,90)} p99={pct(lens,99)} max={max(lens)}")
print("tong tu corpus: {:,}".format(sum(lens)))
for th in [512,1024,2048,4096,8192]:
    print(f"  > {th:>5} tu: {sum(1 for x in lens if x>th):>5} ({100*sum(1 for x in lens if x>th)/len(lens):4.1f}%)")

print("\n## Loai van ban (theo truong name)")
for k,v in doc_types.most_common(): print(f"  {k:<20} {v:>5}")
print("\n## Nam ban hanh (top 12)")
print(dict(years.most_common(12)))

print("\n## Cau truc trong passage")
for k,v in struct.items(): print(f"  {k:<16} {v:>5} / {len(lens)} = {100*v/len(lens):.1f}%")

print("\n## Tin hieu quan he (graph) trong passage")
for k,v in rel.most_common(): print(f"  {k:<16} {v:>5} = {100*v/len(lens):.1f}%")

print("\n## So Dieu moi van ban")
cnt=[]
for fn in files[:2000]:
    with open(os.path.join(D,fn), encoding="utf-8") as f: d=json.load(f)
    p=str(d.get("passage") or "")
    if p.strip(): cnt.append(len(head_re.findall(p)))
print(f"mau 2000 file: mean={st.mean(cnt):.1f} p50={pct(cnt,50)} p90={pct(cnt,90)} max={max(cnt)} | =0 Dieu: {sum(1 for x in cnt if x==0)}")

print("\n## Link domain")
print(Counter(re.sub(r"https?://([^/]+).*",r"\1",l) for l in links).most_common(5))

print("\n## Vi du 3 ten van ban")
for n in names[:3]: print(" -", n[:140])
