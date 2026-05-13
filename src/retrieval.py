import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_index(corpus, model_name):
    model = SentenceTransformer(model_name)
    texts = [f"{p['title']}. {p['text']}" for p in corpus]
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, model


def search(index, model, corpus, query, k):
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, idx = index.search(q, k)
    return [corpus[i] for i in idx[0]]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import EMBEDDING_MODEL, CORPUS_JSON, SAMPLES_JSON, TOP_K
    from src.data import load_json

    corpus = load_json(CORPUS_JSON)
    samples = load_json(SAMPLES_JSON)
    index, model = build_index(corpus, EMBEDDING_MODEL)

    q = samples[0]
    print(q["question"])
    for h in search(index, model, corpus, q["question"], TOP_K):
        print(" ", h["title"])
