import json, os, re, numpy as np, time
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
SP=os.environ.get("EDA_DIR","artifacts/eda"); os.makedirs(SP, exist_ok=True)
t0=time.time()
ids=[]; docs=[]
for line in open(f"{SP}/corpus_flat.txt",encoding="utf-8"):
    i,_,p=line.partition("\t"); ids.append(i); docs.append(p)
print("load", round(time.time()-t0,1),"s | docs",len(docs))

vec=CountVectorizer(lowercase=True, token_pattern=r"(?u)\w+", dtype=np.int32, min_df=2)
X=vec.fit_transform(docs)                      # doc x term
print("vectorize", round(time.time()-t0,1),"s | vocab", X.shape[1], "| nnz", X.nnz)

# BM25 weighting cua doc
k1,b=1.5,0.75
dl=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
avgdl=dl.mean()
df=np.asarray((X>0).sum(axis=0)).ravel()
N=X.shape[0]
idf=np.log(1+(N-df+0.5)/(df+0.5)).astype(np.float32)

Xc=X.tocsr().astype(np.float32)
rows=np.repeat(np.arange(N), np.diff(Xc.indptr))
denom=Xc.data + k1*(1-b+b*dl[rows]/avgdl)
Xc.data=(Xc.data*(k1+1)/denom).astype(np.float32)
W=Xc.multiply(idf).tocsc().tocsr()
print("bm25 weights", round(time.time()-t0,1),"s")

tr=json.load(open("data/official/train.json",encoding="utf-8"))
gold=json.load(open(f"{SP}/gold.json",encoding="utf-8"))
qids=sorted(gold)
Q=vec.transform([re.sub(r"\s+"," ",tr[q]["question"]) for q in qids])
Q.data[:]=1.0                                   # query term presence
S=(Q @ W.T).toarray()                           # n_q x N
print("score", round(time.time()-t0,1),"s")

id2row={d:i for i,d in enumerate(ids)}
order=np.argsort(-S,axis=1)
print("\n## BM25 doc-level recall (n=%d cau co weak gold)"%len(qids))
for k in [1,3,5,10,20,50,100]:
    top=order[:,:k]
    hit=0
    for r,q in enumerate(qids):
        g={id2row[d] for d in gold[q] if d in id2row}
        if g & set(top[r].tolist()): hit+=1
    print(f"  recall@{k:<4} = {100*hit/len(qids):5.1f}%")
# MRR
rr=[]
for r,q in enumerate(qids):
    g={id2row[d] for d in gold[q] if d in id2row}
    pos=[i for i,d in enumerate(order[r]) if d in g]
    rr.append(1/(pos[0]+1) if pos else 0)
print(f"  MRR        = {np.mean(rr):.4f}  | median rank gold = {int(np.median([1/x if x>0 else 9999 for x in rr]))}")
np.save(f"{SP}/order.npy", order[:,:100]); json.dump(qids, open(f"{SP}/qids.json","w"))
