# src/vector_store/hybrid_search.py
import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity  # больше не используем для больших матриц
from scipy.sparse import save_npz, load_npz, csr_matrix, issparse

load_dotenv()

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
INDEX_FILE = os.path.join(CHUNKS_DIR, "faiss.index")
TFIDF_CACHE_DIR = os.path.join(CHUNKS_DIR, "tfidf_cache")
os.makedirs(TFIDF_CACHE_DIR, exist_ok=True)

TFIDF_VECTORIZER_FILE = os.path.join(TFIDF_CACHE_DIR, "vectorizer.pkl")
TFIDF_MATRIX_FILE = os.path.join(TFIDF_CACHE_DIR, "tfidf_matrix.npz")  # для save_npz/load_npz
TFIDF_NORMS_FILE = os.path.join(TFIDF_CACHE_DIR, "tfidf_norms.npy")    # l2 нормы строк


# FAISS
def build_faiss_index(embeddings):
    """Строим или загружаем FAISS-индекс"""
    if os.path.exists(INDEX_FILE):
        print("Используем кэшированный FAISS-индекс...")
        return faiss.read_index(INDEX_FILE)

    print("Создаём новый FAISS-индекс...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    # предполагаем, что embeddings уже нормализованы при создании / из кэша
    faiss.normalize_L2(embeddings)  # безопасно даже если уже нормализованы
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"Индекс FAISS сохранён: {INDEX_FILE}")
    return index


# TF-IDF
def build_or_load_tfidf(chunks, max_features=50000):
    """
    Создаёт или загружает TF-IDF матрицу (sparse) и векторизатор.
    Сохраняет также L2-нормы строк TF-IDF в отдельный файл для быстрого подсчёта cosine.
    Возвращает (vectorizer, tfidf_matrix).
    """
    # Если кэши есть — загружаем
    if os.path.exists(TFIDF_VECTORIZER_FILE) and os.path.exists(TFIDF_MATRIX_FILE):
        print("Используем кэшированный TF-IDF...")
        with open(TFIDF_VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(TFIDF_MATRIX_FILE)

        # загружаем нормы, если есть
        if os.path.exists(TFIDF_NORMS_FILE):
            tfidf_norms = np.load(TFIDF_NORMS_FILE)
        else:
            # если норм нет, вычислим и сохраним
            tfidf_norms = _compute_and_save_tfidf_norms(tfidf_matrix)
        # сохраняем нормы в объект vectorizer для удобства (необязательно)
        setattr(vectorizer, "_tfidf_norms", tfidf_norms)
        return vectorizer, tfidf_matrix

    # Иначе — создаём новый vectorizer и матрицу
    print("Создаём новый TF-IDF-векторизатор...")
    texts = [chunk["chunk_text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)  # sparse matrix (csr)

    # Сохраняем в кэш (sparse)
    with open(TFIDF_VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    save_npz(TFIDF_MATRIX_FILE, tfidf_matrix)

    # Вычисляем и сохраняем L2-нормы строк
    tfidf_norms = _compute_and_save_tfidf_norms(tfidf_matrix)
    setattr(vectorizer, "_tfidf_norms", tfidf_norms)

    print(f"TF-IDF сохранён в {TFIDF_CACHE_DIR}")
    return vectorizer, tfidf_matrix


def _compute_and_save_tfidf_norms(tfidf_matrix):
    """
    Вычисляет L2-нормы строк TF-IDF (по документам).
    Поддерживает sparse матрицу.
    Сохраняет в TFIDF_NORMS_FILE.
    """
    if issparse(tfidf_matrix):
        # суммируем по строкам элемент-wise квадрат, затем sqrt
        # efficient: (tfidf_matrix.multiply(tfidf_matrix)).sum(axis=1) -> (n_docs, 1) matrix
        sq_sum = tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)
        norms = np.sqrt(np.asarray(sq_sum).reshape(-1))
    else:
        norms = np.linalg.norm(tfidf_matrix, axis=1)
    np.save(TFIDF_NORMS_FILE, norms)
    return norms


# HYBRID SEARCH
def search_hybrid(index, query_vector, query_text, chunks, top_k=5, alpha=0.6,
                  tfidf_vectorizer=None, tfidf_matrix=None, faiss_top_n=20, tfidf_top_n=20):
    """
    Гибридный поиск:
    - Берём top faiss_top_n из FAISS
    - Берём top tfidf_top_n из TF-IDF (быстро через разрежённое умножение)
    - Объединяем кандидатов, нормализуем скор и возвращаем top_k
    Возвращает (final_chunks, final_scores)
    """
    if tfidf_vectorizer is None or tfidf_matrix is None:
        tfidf_vectorizer, tfidf_matrix = build_or_load_tfidf(chunks)

    # FAISS
    faiss.normalize_L2(query_vector)
    # index.search принимает ndarray (nq x d) и возвращает (D, I)
    D, I = index.search(query_vector, faiss_top_n)
    faiss_scores = D[0]
    faiss_indices = I[0]

    # TF-IDF scoring (fast, using sparse)
    # Получаем вектор запроса в tf-idf-пространстве — это sparse (csr)
    query_vec_tfidf = tfidf_vectorizer.transform([query_text])  # shape (1, n_features), sparse

    # Загружаем или вычисляем L2-нормы TF-IDF строк
    tfidf_norms = getattr(tfidf_vectorizer, "_tfidf_norms", None)
    if tfidf_norms is None:
        # если нет в объекте, попробуем загрузить файл
        if os.path.exists(TFIDF_NORMS_FILE):
            tfidf_norms = np.load(TFIDF_NORMS_FILE)
        else:
            # последний вариант — вычислить от переданной матрицы
            tfidf_norms = _compute_and_save_tfidf_norms(tfidf_matrix)
        setattr(tfidf_vectorizer, "_tfidf_norms", tfidf_norms)

    # Если tfidf_matrix - sparse, делаем умножение разрежённой матрицы на разрежённый вектор:
    # result shape (n_docs, 1)
    if issparse(tfidf_matrix):
        # dot product (fast, memory-friendly)
        dot_prod = tfidf_matrix.dot(query_vec_tfidf.T).toarray().ravel()  # shape (n_docs,)
        # query norm:
        q_sq_sum = query_vec_tfidf.multiply(query_vec_tfidf).sum()
        q_norm = float(np.sqrt(q_sq_sum)) if q_sq_sum != 0 else 0.0
        # cosine similarity:
        with np.errstate(divide='ignore', invalid='ignore'):
            tfidf_scores_all = dot_prod / (tfidf_norms * (q_norm + 1e-12))
            tfidf_scores_all = np.nan_to_num(tfidf_scores_all, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # dense path (rare), fallback to efficient dot
        query_dense = query_vec_tfidf.toarray().ravel()
        dot_prod = tfidf_matrix.dot(query_dense)
        q_norm = np.linalg.norm(query_dense) + 1e-12
        tfidf_scores_all = dot_prod / (tfidf_norms * q_norm)

    # топ-N по TF-IDF
    tfidf_top_indices = np.argsort(tfidf_scores_all)[::-1][:tfidf_top_n]

    # Объединяем FAISS + TF-IDF индексы (уникальные)
    combined_indices = np.unique(np.concatenate([faiss_indices, tfidf_top_indices]))

    # Собираем соответствующие значения FAISS и TF-IDF для combined_indices
    # Для индексов не найденных в faiss_indices ставим 0
    faiss_subset = np.array([
        faiss_scores[np.where(faiss_indices == i)[0][0]] if i in faiss_indices else 0.0
        for i in combined_indices
    ], dtype=float)
    tfidf_subset = np.array([tfidf_scores_all[i] for i in combined_indices], dtype=float)

    # Нормализация (min-max)
    def _minmax(arr):
        mn = arr.min() if arr.size else 0.0
        mx = arr.max() if arr.size else 0.0
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        return (arr - mn) / denom

    faiss_norm = _minmax(faiss_subset)
    tfidf_norm = _minmax(tfidf_subset)

    combined_scores = alpha * faiss_norm + (1 - alpha) * tfidf_norm
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    final_chunks = [chunks[int(i)] for i in combined_indices[top_indices]]
    final_scores = combined_scores[top_indices]

    return final_chunks, final_scores
