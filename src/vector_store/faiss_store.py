import faiss
import os
from dotenv import load_dotenv

from src.vector_store.hybrid_search import INDEX_FILE, build_faiss_index

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")


def load_faiss_index():
    return faiss.read_index(INDEX_FILE)


def search_faiss(index, query_vector, top_k=5):
    D, I = index.search(query_vector, top_k)
    return I[0], D[0]
