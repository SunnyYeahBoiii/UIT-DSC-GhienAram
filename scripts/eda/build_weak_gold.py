import json, os, re, subprocess, collections
SP=os.environ.get("EDA_DIR","artifacts/eda"); os.makedirs(SP, exist_ok=True)
meta=json.load(open(f"{SP}/meta.json",encoding="utf-8"))
# map line-number -> doc id
ids=[]
for line in open(f"{SP}/corpus_flat.txt",encoding="utf-8"):
    ids.append(line.split("\t",1)[0])
gold=collections.defaultdict(set)
for raw in open(f"{SP}/hits_raw.txt",encoding="utf-8"):
    ln,_,seg=raw.rstrip("\n").partition(":")
    qid=meta.get(seg)
    if qid: gold[qid].add(ids[int(ln)-1])
json.dump({k:sorted(v) for k,v in gold.items()}, open(f"{SP}/gold.json","w"), ensure_ascii=False)
c=collections.Counter(len(v) for v in gold.values())
print("cau co gold doc:", len(gold))
print("phan bo so gold doc/cau:", dict(sorted(c.items())))
