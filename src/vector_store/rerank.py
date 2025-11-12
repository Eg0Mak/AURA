# src/vector_store/rerank.py
from sentence_transformers import CrossEncoder

# Загружаем модель для reranking
# Можно использовать более точную: cross-encoder/ms-marco-MiniLM-L-6-v2
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, retrieved_chunks: list, top_n: int = 5):
    """
    Повторно оценивает топ-результаты из FAISS, возвращает отсортированные по релевантности.
    """
    pairs = [(query, chunk["chunk_text"]) for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    # Сопоставляем баллы с чанками
    reranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in reranked[:top_n]]
