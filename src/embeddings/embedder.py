from src.config.embbeding_model import embed_model
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

def get_embeddings(model_name="intfloat/multilingual-e5-base"):
    """Вычисляет эмбеддинги для всех чанков"""

    chunks_path = os.path.join(CHUNKS_DIR, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["chunk"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    np.save(os.path.join(CHUNKS_DIR, "embeddings.npy"), embeddings)
    print(f"Эмбеддинги сохранены: {embeddings.shape}")

    return np.array(embeddings), chunks

if __name__ == "__main__":
    get_embeddings()