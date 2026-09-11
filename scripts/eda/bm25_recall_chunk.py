import json, os, re, numpy as np, time, sys
from sklearn.feature_extraction.text import CountVectorizer
SP=os.environ.get("EDA_DIR","artifacts/eda"); os.makedirs(SP, exist_ok=True)
SIZE=int(sys.argv[1]); STRIDE=int(sys.argv[2])
t0=time.time()
chunks=[]; owner=[]
for line in open(f"{SP}/corpus_flat.txt",encoding="utf-8"):
    i,_,p=line.partition("\t"); w=p.split()
    for s in range(0,max(1,len(w)),STRIDE):
        c=w[s:s+SIZE]
        if len(c)<30 and s>0: break
        chunks.append(" ".join(c)); owner.append(i)
        if s+SIZE>=len(w): break
owner=np.array(owner)
print(f"size={SIZE} stride={STRIDE} | chunks={len(chunks):,} | {round(time.time()-t0,1)}s")

vec=CountVectorizer(lowercase=True, token_pattern=r"(?u)\w+", dtype=np.int32, min_df=2)
X=vec.fit_transform(chunks)
k1,b=1.5,0.75
dl=np.asarray(X.sum(axis=1)).ravel().astype(np.float32); avgdl=dl.mean()
df=np.asarray((X>0).sum(axis=0)).ravel(); N=X.shape[0]
idf=np.log(1+(N-df+0.5)/(df+0.5)).astype(np.float32)
Xc=X.tocsr().astype(np.float32)
rows=np.repeat(np.arange(N), np.diff(Xc.indptr))
Xc.data=(Xc.data*(k1+1)/(Xc.data+k1*(1-b+b*dl[rows]/avgdl))).astype(np.float32)
W=Xc.multiply(idf).tocsr()
print("indexed", round(time.time()-t0,1),"s | vocab",X.shape[1])

tr=json.load(open("data/official/train.json",encoding="utf-8"))
gold=json.load(open(f"{SP}/gold.json",encoding="utf-8")); qids=sorted(gold)
Q=vec.transform([tr[q]["question"] for q in qids]); Q.data[:]=1.0
S=(Q @ W.T).toarray()
order=np.argsort(-S,axis=1)[:,:400]
print(f"## chunk BM25, gom ve doc (n={len(qids)})")
for k in [1,3,5,10,20,50,100]:
    hit=0
    for r,q in enumerate(qids):
        seen=[]; 
        for ci in order[r]:
            d=owner[ci]
            if d not in seen: seen.append(d)
            if len(seen)>=k: break
        if set(gold[q]) & set(seen): hit+=1
    print(f"  doc recall@{k:<4} = {100*hit/len(qids):5.1f}%")
print("tong", round(time.time()-t0,1),"s")
