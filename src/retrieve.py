"""Step 5 - Retrieve: vector, BM25 and hybrid (reciprocal-rank-fusion) top-k.

Why hybrid: embeddings capture meaning but blur exact tokens such as 'FC-1.2.3',
'BD 100,000' or 'CET1'. BM25 matches those exact tokens. RRF merges the two ranked
lists without needing to calibrate their (incomparable) scores.
"""
import re

from rank_bm25 import BM25Okapi

from config import QUERY_PREFIX, TOP_K
from index import get_collection, get_model, load_chunks

TOKEN_RE = re.compile(r"[a-z]{2}-[a-z0-9]+(?:\.\d+)+[a-z]{0,2}|[a-z0-9]+(?:\.[0-9]+)?")
RULE_ID_RE = re.compile(r"\b([A-Z]{2}-[A-Z0-9]+(?:\.\d+)+[A-Za-z]{0,2})\b")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.col = get_collection()
        self.model = get_model()
        corpus = [tokenize(f"{c['rule_id']} {c['heading']} {c['text']}") for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    # ---- individual retrievers -------------------------------------------------
    def vector(self, query: str, k: int = TOP_K):
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=k)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            out.append({**self.by_id[cid], "score": 1 - dist, "vector_score": 1 - dist})
        return out

    def keyword(self, query: str, k: int = TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order if scores[i] > 0]

    def hybrid(self, query: str, k: int = TOP_K, pool: int = 30, rrf_k: int = 60):
        vec = self.vector(query, pool)
        kw = self.keyword(query, pool)
        fused, vscore = {}, {c["chunk_id"]: c["vector_score"] for c in vec}
        for lst in (vec, kw):
            for rank, c in enumerate(lst):
                fused[c["chunk_id"]] = fused.get(c["chunk_id"], 0) + 1 / (rrf_k + rank + 1)
        # A rule number typed in the question is an explicit lookup: guarantee it is retrieved.
        for rid in RULE_ID_RE.findall(query):
            for c in self.chunks:
                if c["rule_id"] == rid and c["part"] == 0:
                    fused[c["chunk_id"]] = fused.get(c["chunk_id"], 0) + 1.0
        top = sorted(fused, key=lambda cid: -fused[cid])[:k]
        out = []
        for cid in top:
            c = self.by_id[cid]
            if cid not in vscore:  # need a vector similarity for the weak-retrieval check
                import numpy as np
                q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
                d = self.model.encode([c["embed_text"]], normalize_embeddings=True)[0]
                vscore[cid] = float(np.dot(q, d))
            out.append({**c, "score": fused[cid], "vector_score": vscore[cid]})
        return out

    def search(self, query: str, mode: str = "hybrid", k: int = TOP_K):
        return {"vector": self.vector, "bm25": self.keyword, "hybrid": self.hybrid}[mode](query, k)


if __name__ == "__main__":
    import sys
    r = Retriever()
    q = " ".join(sys.argv[1:]) or "What must a bank's risk assessment consider?"
    for mode in ("vector", "hybrid"):
        print(f"--- {mode}")
        for c in r.search(q, mode):
            print(f"{c['rule_id']:<12} {c['score']:.3f}  {c['heading']}")
