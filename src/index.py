"""Step 4 - Index: embed every chunk locally and store in Chroma.

Model: BAAI/bge-small-en-v1.5 (384-dim, runs on CPU, free, no data leaves the machine).
We embed `embed_text` = context prefix (module | chapter | section | rule id) + rule text,
so a chunk 'knows' where it lives even if the rule text itself is generic.
"""
import json

import chromadb

from config import CHROMA_DIR, CHUNKS_FILE, EMBED_MODEL

COLLECTION = "cbb_rules"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index(force: bool = False):
    chunks = load_chunks()
    col = get_collection()
    if col.count() == len(chunks) and not force:
        print(f"Index already built ({col.count()} chunks).")
        return col
    chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection(COLLECTION)
    col = get_collection()
    model = get_model()
    texts = [c["embed_text"] for c in chunks]
    embs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    meta_keys = ["rule_id", "module", "chapter", "section", "heading", "last_changed", "source_url"]
    for i in range(0, len(chunks), 256):
        sl = slice(i, i + 256)
        col.add(
            ids=[c["chunk_id"] for c in chunks[sl]],
            embeddings=embs[sl].tolist(),
            documents=[c["text"] for c in chunks[sl]],
            metadatas=[{k: c[k] for k in meta_keys} for c in chunks[sl]],
        )
    print(f"Indexed {col.count()} chunks into {CHROMA_DIR}")
    return col


if __name__ == "__main__":
    import sys
    build_index(force="--force" in sys.argv)
