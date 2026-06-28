# src/embeddings/embedder.py
from src.config.embbeding_model import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, embed_model
import hashlib
import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_embeddings(csv_name="chunks_fixed.csv", batch_size=64, force_rebuild=False):
    print('start getting embeddings....')
    cache_file = os.path.join(CHUNKS_DIR, "embeddings.npy")
    metadata_file = os.path.join(CHUNKS_DIR, "embeddings.meta.json")
    csv_path = os.path.join(CHUNKS_DIR, csv_name)
    df = pd.read_csv(csv_path)
    chunks_hash = _file_sha1(csv_path)
    expected_metadata = {
        "csv_name": csv_name,
        "chunks_hash": chunks_hash,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_device": EMBEDDING_DEVICE,
        "rows": len(df),
    }

    if os.path.exists(cache_file) and not force_rebuild:
        print("Используем кэшированные эмбеддинги...")
        embeddings = np.load(cache_file)
        metadata = _load_json(metadata_file)
        metadata_matches = (
            metadata is not None
            and all(metadata.get(key) == value for key, value in expected_metadata.items())
            and metadata.get("embedding_dim") == embeddings.shape[1]
        )
        if embeddings.shape[0] != len(df) or not metadata_matches:
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
    expected_metadata["embedding_dim"] = embeddings.shape[1]
    _save_json(metadata_file, expected_metadata)
    print(f"Эмбеддинги сохранены: {embeddings.shape}")

    chunks = df.to_dict(orient="records")
    return embeddings, chunks


if __name__ == "__main__":
    get_embeddings()

    
