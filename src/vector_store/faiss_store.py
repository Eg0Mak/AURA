import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
INDEX_FILE = os.path.join(CHUNKS_DIR, "faiss.index")

def build_faiss_index(embeddings):
    """Строим или загружаем FAISS-индекс"""
    if os.path.exists(INDEX_FILE):
        print("Используем кэшированный FAISS-индекс...")
        index = faiss.read_index(INDEX_FILE)
        return index

    print("Создаём новый FAISS-индекс...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"Индекс FAISS сохранён: {INDEX_FILE}")
    return index

def load_faiss_index():
    path = os.path.join(CHUNKS_DIR, "faiss.index")
    return faiss.read_index(path)

def search_faiss(index, query_vector, top_k=5):
    D, I = index.search(query_vector, top_k)
    # tf-idf 
    return I[0], D[0]
