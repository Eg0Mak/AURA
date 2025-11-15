# src/embeddings/embedder.py
from src.config.embbeding_model import embed_model
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

def get_embeddings(csv_name="chunks_semantic.csv", batch_size=64):
    print('start getting embeddings....')
    cache_file = os.path.join(CHUNKS_DIR, "embeddings.npy")
    csv_path = os.path.join(CHUNKS_DIR, csv_name)

    if os.path.exists(cache_file):
        print("Используем кэшированные эмбеддинги...")
        embeddings = np.load(cache_file)
        df = pd.read_csv(csv_path)
        chunks = df.to_dict(orient="records")
        return embeddings, chunks

    # Загружаем CSV
    df = pd.read_csv(csv_path)
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

    
