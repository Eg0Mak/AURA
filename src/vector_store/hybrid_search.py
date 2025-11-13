import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
INDEX_FILE = os.path.join(CHUNKS_DIR, "faiss.index")
TFIDF_CACHE_DIR = os.path.join(CHUNKS_DIR, "tfidf_cache")
os.makedirs(TFIDF_CACHE_DIR, exist_ok=True)

TFIDF_VECTORIZER_FILE = os.path.join(TFIDF_CACHE_DIR, "vectorizer.pkl")
TFIDF_MATRIX_FILE = os.path.join(TFIDF_CACHE_DIR, "tfidf_matrix.npz")


# ========== FAISS ==========
def build_faiss_index(embeddings):
    """Строим или загружаем FAISS-индекс"""
    if os.path.exists(INDEX_FILE):
        print("Используем кэшированный FAISS-индекс...")
        return faiss.read_index(INDEX_FILE)

    print("Создаём новый FAISS-индекс...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)  # Важно!
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"Индекс FAISS сохранён: {INDEX_FILE}")
    return index


# ========== TF-IDF ==========
def build_or_load_tfidf(chunks, max_features=50000):
    """Создаёт или загружает TF-IDF матрицу и векторизатор"""
    if os.path.exists(TFIDF_VECTORIZER_FILE) and os.path.exists(TFIDF_MATRIX_FILE):
        print("Используем кэшированный TF-IDF...")
        with open(TFIDF_VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = np.load(TFIDF_MATRIX_FILE, allow_pickle=True)["arr_0"]
        return vectorizer, tfidf_matrix

    print("Создаём новый TF-IDF-векторизатор...")
    texts = [chunk["chunk_text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    # Сохраняем кэш
    with open(TFIDF_VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    np.savez_compressed(TFIDF_MATRIX_FILE, tfidf_matrix)

    print(f"TF-IDF сохранён в {TFIDF_CACHE_DIR}")
    return vectorizer, tfidf_matrix



def search_hybrid(index, query_vector, query_text, chunks, top_k=5, alpha=0.7,
                          tfidf_vectorizer=None, tfidf_matrix=None):
    # === 1. FAISS-поиск с ограничением ===
    faiss.normalize_L2(query_vector)
    search_k = min(top_k * 5, len(chunks))  # ограничиваем поиск
    D, I = index.search(query_vector, search_k)
    faiss_scores = D[0]
    faiss_indices = I[0]

    # === 2. TF-IDF только для найденных FAISS чанков ===
    if tfidf_vectorizer is None or tfidf_matrix is None:
        tfidf_vectorizer, tfidf_matrix = build_or_load_tfidf(chunks)

    query_vec_tfidf = tfidf_vectorizer.transform([query_text]).toarray()
    
    # Только для чанков, найденных FAISS
    tfidf_scores_limited = cosine_similarity(
        query_vec_tfidf, 
        tfidf_matrix[faiss_indices]
    )[0]

    # === 3. Нормализация только найденных scores ===
    faiss_scores_norm = (faiss_scores - faiss_scores.min()) / (
        faiss_scores.max() - faiss_scores.min() + 1e-9
    )
    
    tfidf_scores_norm = (tfidf_scores_limited - tfidf_scores_limited.min()) / (
        tfidf_scores_limited.max() - tfidf_scores_limited.min() + 1e-9
    )

    # === 4. Комбинированная метрика ===
    combined_scores = alpha * faiss_scores_norm + (1 - alpha) * tfidf_scores_norm

    # === 5. Сортировка ===
    top_local_indices = np.argsort(combined_scores)[::-1][:top_k]
    top_global_indices = faiss_indices[top_local_indices]
    
    top_chunks = [chunks[i] for i in top_global_indices]
    top_scores = combined_scores[top_local_indices]

    return top_chunks, top_scores