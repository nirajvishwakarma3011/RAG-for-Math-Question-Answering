# retriever_math.py  (updated)
import os, json, re, numpy as np

def _math_tok(s: str):
    s = s.replace("×","*").replace("÷","/").replace("−","-")
    # keep LaTeX-ish tokens, operators, numbers, words
    return [p.lower() for p in re.findall(
        r"(?:\\[A-Za-z]+|[{}\^\_\(\)\[\]=+\-*/]|[0-9\.]+|[A-Za-z]+)", s
    )]

class BM25Retriever:
    def __init__(self, store_dir: str):
        from rank_bm25 import BM25Okapi
        rows = [json.loads(l) for l in open(os.path.join(store_dir, "bm25.jsonl"), "r", encoding="utf-8")]
        self.doc_ids = [r["doc_id"] for r in rows]
        self.Qs      = [r["Q"] for r in rows]
        self.As      = [r["A"] for r in rows]
        self.urls    = [r["url"] for r in rows]
        self.texts   = [r["text"] for r in rows]  # indexed surface = Q + A
        self.bm25 = BM25Okapi([_math_tok(t) for t in self.texts])

    def search(self, query: str, k: int):
        scores = self.bm25.get_scores(_math_tok(query))
        idx = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i]), self.Qs[i], self.As[i], self.urls[i]) for i in idx]

class DenseRetriever:
    def __init__(self, store_dir: str, encoder="sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(encoder)
        self.index = faiss.read_index(os.path.join(store_dir, "faiss.index"))
        meta = json.load(open(os.path.join(store_dir, "meta.json"), "r", encoding="utf-8"))
        self.doc_ids, self.Qs, self.As, self.urls = meta["doc_ids"], meta["Q"], meta["A"], meta["url"]

        # Dense index was built on (Q + A); we’ll recreate that surface for re-mapping
        self.texts = [f"{q} {a}".strip() for q, a in zip(self.Qs, self.As)]

    def search(self, query: str, k: int):
        qv = self.enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(qv.astype(np.float32), k)
        hits = []
        for d, i in zip(D[0].tolist(), I[0].tolist()):
            hits.append((self.doc_ids[i], float(d), self.Qs[i], self.As[i], self.urls[i]))
        return hits

def load_retriever(store_dir: str, mode="hybrid"):
    bm25 = BM25Retriever(store_dir)
    if mode == "bm25": return bm25
    if mode == "dense": return DenseRetriever(store_dir)

    # hybrid = simple union + rescoring
    dense = None
    try: dense = DenseRetriever(store_dir)
    except Exception: pass

    class Hybrid:
        def search(self, query, k):
            b = bm25.search(query, k*2)
            d = dense.search(query, k*2) if dense else []
            pool = {}
            for did, s, q, a, url in b:
                pool.setdefault(did, [q, a, url, 0.0, 0.0])
                pool[did][3] = max(pool[did][3], s)   # bm25
            for did, s, q, a, url in d:
                pool.setdefault(did, [q, a, url, 0.0, 0.0])
                pool[did][4] = max(pool[did][4], s)   # dense
            merged = []
            for did, (q, a, url, sb, sd) in pool.items():
                score = (sb/100.0) + sd
                merged.append((did, score, q, a, url))
            merged.sort(key=lambda x: -x[1])
            return merged[:k]
    return Hybrid()
