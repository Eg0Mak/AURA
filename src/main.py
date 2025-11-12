# main.py
from src.data_preprocessing.clean_data import clean_csv
from src.chunking.splitter import split_documents_with_semantics
from src.embeddings.embedder import get_embeddings
from src.vector_store.faiss_store import build_faiss_index, load_faiss_index, search_faiss
from src.config.embbeding_model import embed_model
from src.vector_store.rerank import rerank
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 5))

# === 1. Очистка CSV ===
# df = clean_csv("websites_updated.csv")

# === 2. Семантическое разбиение ===
# PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
# CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
# input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
# output_csv = os.path.join(CHUNKS_DIR, "chunks_semantic.csv")
# split_documents_with_semantics(input_csv, output_csv)

# === 3. Эмбеддинги ===
embeddings, chunks = get_embeddings()  # читает output_csv с чанками

# === 4. Индексация FAISS ===
# index = build_faiss_index(embeddings)
index = load_faiss_index()

# === 5. Пример поиска ===
query = "Здравствуйте, когда смогу пользоваться кредитной картой?"

# Получаем эмбеддинг для запроса
query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)

# Поиск в FAISS
I, D = search_faiss(index, query_vec, top_k=TOP_K)
I = np.atleast_1d(I).ravel()  # приводим индексы к одномерному массиву

# Извлекаем найденные чанки
retrieved = [chunks[i] for i in I]

# Reranking
reranked = rerank(query, retrieved, top_n=5)

# Вывод результата
print("\nТоп после RERANKING:\n")
for c in reranked:
    print(f"— {c['url']}\n{c['chunk_text'][:200]}...\n")