# src/embeddings/embedder.py
from src.config.embbeding_model import embed_model
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

def get_embeddings(csv_name="chunks_semantic.csv"):
    cache_file = os.path.join(CHUNKS_DIR, "embeddings.npy")
    if os.path.exists(cache_file):
        print("Используем кэшированные эмбеддинги...")
        embeddings = np.load(cache_file)
        df = pd.read_csv(os.path.join(CHUNKS_DIR, csv_name))
        chunks = df.to_dict(orient="records")
        return embeddings, chunks

    # Если кэша нет — считаем заново
    df = pd.read_csv(os.path.join(CHUNKS_DIR, csv_name))
    texts = df["chunk_text"].tolist()
    embeddings = embed_model.get_batch_embeddings(texts)
    embeddings = np.array(embeddings, dtype=np.float32)

    np.save(cache_file, embeddings)
    print(f"Эмбеддинги сохранены: {embeddings.shape}")

    chunks = df.to_dict(orient="records")
    return embeddings, chunks

if __name__ == "__main__":
    get_embeddings()

    
