import json, os, os, re
D=os.environ.get("CORPUS_DIR","data/official/selected-contexts")
SP=os.environ.get("EDA_DIR","artifacts/eda"); os.makedirs(SP, exist_ok=True)
out=open(f"{SP}/corpus_flat.txt","w",encoding="utf-8")
n=0
for fn in sorted(os.listdir(D)):
    d=json.load(open(os.path.join(D,fn),encoding="utf-8"))
    p=re.sub(r"\s+"," ",str(d.get("passage") or "")).strip()
    if not p: continue
    out.write(f"{d.get('id',fn)}\t{p}\n"); n+=1
out.close()
print("docs flattened:", n, "| size MB:", round(os.path.getsize(f"{SP}/corpus_flat.txt")/1e6,1))
