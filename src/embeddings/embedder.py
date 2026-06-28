# src/embeddings/embedder.py
from src.config.embbeding_model import embed_model
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

def get_embeddings(csv_name="chunks_fixed.csv", batch_size=64, force_rebuild=False):
    print('start getting embeddings....')
    cache_file = os.path.join(CHUNKS_DIR, "embeddings.npy")
    csv_path = os.path.join(CHUNKS_DIR, csv_name)
    df = pd.read_csv(csv_path)

    if os.path.exists(cache_file) and not force_rebuild:
        print("Используем кэшированные эмбеддинги...")
        embeddings = np.load(cache_file)
        if embeddings.shape[0] != len(df):
            print("Кэш эмбеддингов не совпадает с чанками, пересчитываем...")
        else:
            chunks = df.to_dict(orient="records")
            return embeddings, chunks

    if force_rebuild and os.path.exists(cache_file):
        print("Данные изменились, пересчитываем эмбеддинги...")

    if df.empty:
        raise ValueError(f"Файл с чанками пустой: {csv_path}")

    # Загружаем CSV
    if "chunk_text" not in df.columns:
        raise ValueError(f"В {csv_path} нет колонки chunk_text")

    texts = df["chunk_text"].tolist()

    # Считаем эмбеддинги батчами
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_embeddings = embed_model._get_text_embedding(batch_texts)
        embeddings.append(batch_embeddings)
        print(start)
    embeddings = np.vstack(embeddings).astype(np.float32)

    # Нормализация
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(cache_file, embeddings)
    print(f"Эмбеддинги сохранены: {embeddings.shape}")

    chunks = df.to_dict(orient="records")
    return embeddings, chunks


if __name__ == "__main__":
    get_embeddings()

    
