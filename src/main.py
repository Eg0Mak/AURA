# src/main.py
from src.data_preprocessing.clean_data import clean_csv
from src.chunking.splitter import split_documents_with_semantics
from src.chunking.recursive_splitter import split_documents_with_recursive
from src.embeddings.embedder import get_embeddings
# from src.vector_store.faiss_store import build_faiss_index, load_faiss_index, search_faiss
from src.vector_store.hybrid_search import build_or_load_tfidf, search_hybrid, build_faiss_index
from src.config.embbeding_model import embed_model
from src.data_preprocessing.marked_llm import process_queries
from src.vector_store.rerank import rerank
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


load_dotenv()
# TOP_K = int(os.getenv("TOP_K", 20))
ALPHA = float(os.getenv("HYBRID_ALPHA"))

TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5)) # топ после гибридного поиска, который идёт на rerank
FAISS_TOP_N = int(os.getenv("FAISS_TOP_N", 10))
TFIDF_TOP_N = int(os.getenv("TFIDF_TOP_N", 10))

# 1. Очистка CSV
df = clean_csv("websites_updated.csv")

# 2.Семантическое разбиение
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
output_csv = os.path.join(CHUNKS_DIR, "chunks_semantic.csv")
# split_documents_with_semantics(input_csv, output_csv)
split_documents_with_recursive(input_csv, output_csv) # Пробую другой сплиттер

# 3. Эмбеддинги
embeddings, chunks = get_embeddings()  
index = build_faiss_index(embeddings)

# Перефрзаирование запросов
# process_queries(
#     input_filename="questions_clean.csv",
#     output_filename="questions_expanded.csv",
#     mode="paraphrase",
#     n_variants=1
# )

# Загружаем CSV с вопросами
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
input_csv_queries = os.path.join(RAW_DATA_DIR, "questions_clean.csv")
queries_df = pd.read_csv(input_csv_queries)  # q_id, query

# PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
# input_csv_queries_expand = os.path.join("PROCESSED_DATA_DIR", "questions_expanded.csv")
# queries_df = pd.read_csv(input_csv_queries_expand)  # q_id, query

results = []

# 4. Индексация FAISS и гибридный поиск
vectorizer, tfidf_matrix = build_or_load_tfidf(chunks)

for _, row in queries_df.iterrows():
    q_id = row["q_id"]
    query = str(row["query"]).strip()
    query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)

    # Гибридный поиск с топ-N
    hybrid_chunks, scores = search_hybrid(
        index, query_vec, query, chunks,
        top_k=TOP_K_RERANK, alpha=ALPHA,
        tfidf_vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        faiss_top_n=FAISS_TOP_N,
        tfidf_top_n=TFIDF_TOP_N
    )

    # rerank топ-5
    reranked_chunks = rerank(query, hybrid_chunks, top_n=TOP_K_RERANK)
    web_list_reranked = [chunk.get("web_id", "") or chunk.get("url", "") for chunk in reranked_chunks]

    results.append({
        "q_id": q_id,
        "web_list": str(web_list_reranked)  # сохраняем как строку
    })

output_csv = "queries_rag_results.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Разметка сохранена в: {output_csv}")