# # src/vector_store/rerank.py
# from sentence_transformers import CrossEncoder

# # Загружаем модель для reranking

# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# reranker = CrossEncoder(
#     "DiTy/cross-encoder-russian-msmarco",
#     max_length=512,
#     device="mps"  # вместо "cuda"
# )

# def rerank(query: str, retrieved_chunks: list, top_n: int = 5):
#     """
#     Повторно оценивает топ-результаты из FAISS, возвращает отсортированные по релевантности.
#     """
#     pairs = [(query, chunk["chunk_text"]) for chunk in retrieved_chunks]
#     scores = reranker.predict(pairs, batch_size=64)

#     # Сопоставляем баллы с чанками
#     reranked = sorted(
#         zip(retrieved_chunks, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return [chunk for chunk, _ in reranked[:top_n]]


# src/vector_store/rerank.py
from sentence_transformers import CrossEncoder
from functools import lru_cache
import hashlib

# Загружаем модель для reranking
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = CrossEncoder(
    "DiTy/cross-encoder-russian-msmarco",
    max_length=512,
    device="mps"  # вместо "cuda"
)

def _hash_pair(query: str, chunk_text: str) -> str:
    """
    Создает короткий стабильный хеш для пары query+chunk_text
    """
    key = f"{query}::{chunk_text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


@lru_cache(maxsize=100_000)
def _cached_predict(query: str, chunk_text: str) -> float:
    """
    Кэшируем предсказания CrossEncoder для одной пары.
    Используется внутренняя обертка для совместимости с lru_cache.
    """
    return float(reranker.predict([(query, chunk_text)])[0])


def rerank(query: str, retrieved_chunks: list, top_n: int = 5, batch_size: int = 64):
    """
    Повторно оценивает топ-результаты из FAISS, возвращает отсортированные по релевантности.
    Поддерживает кэширование и батчинг, но логику выхода не меняет.
    """
    if not retrieved_chunks:
        return []

    # Создаем пары для оценки
    pairs = [(query, chunk["chunk_text"]) for chunk in retrieved_chunks]

    scores = []
    uncached_pairs = []
    uncached_indices = []

    # Проверяем кэш
    for i, (q, chunk_text) in enumerate(pairs):
        key_hash = _hash_pair(q, chunk_text)
        # проверим есть ли в кэше через LRU 
        try:
            score = _cached_predict(q, chunk_text)
            scores.append(score)
        except Exception:
            # на случай, если что-то пойдет не так, добавим в очередь на обработку
            uncached_pairs.append((q, chunk_text))
            uncached_indices.append(i)
            scores.append(None)

    # Обрабатываем непосчитанные пары батчами
    if uncached_pairs:
        for start in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[start:start + batch_size]
            batch_scores = reranker.predict(batch)
            for j, score in enumerate(batch_scores):
                idx = uncached_indices[start + j]
                q, chunk_text = pairs[idx]
                _cached_predict(q, chunk_text)  # записываем в кэш
                scores[idx] = float(score)

    # Сортировка результатов
    reranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in reranked[:top_n]]