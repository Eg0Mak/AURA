import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(CHUNKS_DIR, "faiss.index"))
    print(f"Индекс faiss сохранён")
    return index

def load_faiss_index():
    path = os.path.join(CHUNKS_DIR, "faiss.index")
    return faiss.read_index(path)

def search_faiss(index, query_vector, top_k=5):
    D, I = index.search(query_vector, top_k)
    return I[0], D[0]
