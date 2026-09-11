import json, os, re, random
SP=os.environ.get("EDA_DIR","artifacts/eda"); os.makedirs(SP, exist_ok=True)
tr=json.load(open("data/official/train.json",encoding="utf-8"))
random.seed(7)
ids=random.sample(list(tr),300)
pats=[]; meta={}
for qid in ids:
    a=re.sub(r"\s+"," ",str(tr[qid]["answer"] or "")).strip()
    # bo cau mo dau (thuong la dan nhap cua nguoi viet), lay doan giua
    w=a.split()
    if len(w)<60: continue
    seg=" ".join(w[len(w)//2:len(w)//2+12])
    if len(seg)<40: continue
    pats.append(seg); meta[seg]=qid
json.dump(meta, open(f"{SP}/meta.json","w"), ensure_ascii=False)
open(f"{SP}/patterns.txt","w",encoding="utf-8").write("\n".join(pats))
print("patterns:", len(pats))
