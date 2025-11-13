# main.py
from src.data_preprocessing.clean_data import clean_csv
from src.chunking.splitter import split_documents_with_semantics
from src.embeddings.embedder import get_embeddings
# from src.vector_store.faiss_store import build_faiss_index, load_faiss_index, search_faiss
from src.vector_store.hybrid_search import build_or_load_tfidf, search_hybrid, build_faiss_index
from src.config.embbeding_model import embed_model
from src.vector_store.rerank import rerank
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 20))
ALPHA = float(os.getenv("HYBRID_ALPHA"))

# 1. Очистка CSV
# df = clean_csv("websites_updated.csv")

# 2.Семантическое разбиение
# PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
# CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
# input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
# output_csv = os.path.join(CHUNKS_DIR, "chunks_semantic.csv")
# split_documents_with_semantics(input_csv, output_csv)

# # === 3. Эмбеддинги ===
# embeddings, chunks = get_embeddings()  # читает output_csv с чанками

# # === 4. Индексация FAISS ===
# # index = build_faiss_index(embeddings)
# index = load_faiss_index()

# # === 5. Пример поиска ===
# query = "Здравствуйте, когда смогу пользоваться кредитной картой?"

# # Получаем эмбеддинг для запроса
# query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)

# # Поиск в FAISS
# I, D = search_faiss(index, query_vec, top_k=TOP_K)
# I = np.atleast_1d(I).ravel()  # приводим индексы к одномерному массиву

# # Извлекаем найденные чанки
# retrieved = [chunks[i] for i in I]

# # Reranking
# reranked = rerank(query, retrieved, top_n=5)

# # Вывод результата
# print("\nТоп после RERANKING:\n")
# for c in reranked:
#     print(f"— {c['url']}\n{c['chunk_text'][:200]}...\n")


# Загружаем эмбеддинги и чанки 
# embeddings, chunks = get_embeddings()  
# index = build_faiss_index(embeddings)

# # Загружаем CSV с вопросами 
# RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
# input_csv_queries = os.path.join(RAW_DATA_DIR, "questions_clean.csv")
# queries_df = pd.read_csv(input_csv_queries)  # q_id, query

# results = []

# for _, row in queries_df.iterrows():
#     q_id = row["q_id"]
#     query = str(row["query"]).strip()
    
#     # Получаем эмбеддинг запроса
#     query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)
    
#     # FAISS-поиск
#     I, D = search_faiss(index, query_vec, top_k=TOP_K)
#     I = np.atleast_1d(I).ravel()
    
#     # Извлекаем найденные id веб-страниц (или используем chunk['web_id'])
#     web_list = [chunks[i]["web_id"] for i in I]
    
#     # Ререйкинг
#     retrieved = [chunks[i] for i in I]
#     reranked = rerank(query, retrieved, top_n=TOP_K)
#     web_list_reranked = [c["web_id"] for c in reranked]
    
#     results.append({
#         "q_id": q_id,
#         "web_list": str(web_list_reranked)  # сохраняем как строку
#     })

# # Сохраняем результат
# output_csv = "queries_rag_results.csv"
# pd.DataFrame(results).to_csv(output_csv, index=False)
# print(f"Разметка сохранена в: {output_csv}")

# 3. Эмбеддинги
embeddings, chunks = get_embeddings()  
index = build_faiss_index(embeddings)

# Загружаем CSV с вопросами
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
input_csv_queries = os.path.join(RAW_DATA_DIR, "questions_clean.csv")
queries_df = pd.read_csv(input_csv_queries)  # q_id, query

results = []

# 4. Индексация FAISS и гибридный поиск
vectorizer, tfidf_matrix = build_or_load_tfidf(chunks)

for _, row in queries_df.iterrows():
    q_id = row["q_id"]
    query = str(row["query"]).strip()
    query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)

    # Передаем готовые TF-IDF объекты
    hybrid_chunks, scores = search_hybrid(
        index, query_vec, query, chunks,
        top_k=TOP_K, alpha=ALPHA,
        tfidf_vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix
    )

    reranked_chunks = rerank(query, hybrid_chunks, top_n=5)
    web_list_reranked = [chunk.get("web_id", "") or chunk.get("url", "") for chunk in reranked_chunks]

    results.append({
        "q_id": q_id,
        "web_list": str(web_list_reranked)  # сохраняем как строку
    })

output_csv = "queries_rag_results.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Разметка сохранена в: {output_csv}")
