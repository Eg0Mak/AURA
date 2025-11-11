from src.data_preprocessing.clean_data import clean_csv
from src.chunking.splitter import chunk_texts, save_chunks
from src.embeddings.embedder import get_embeddings
from src.vector_store.faiss_store import build_faiss_index, search_faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 5))

# === 1. Очистка CSV ===
df = clean_csv("documents.csv")

# === 2. Разбиение ===
chunks = chunk_texts(df)
save_chunks(chunks)

# === 3. Эмбеддинги ===
embeddings, chunks = get_embeddings()

# === 4. Индексация ===
index = build_faiss_index(embeddings)

# === 5. Пример поиска ===
model = SentenceTransformer("intfloat/multilingual-e5-base")
query = "Здравствуйте, когда смогу пользоваться кредитной картой?"
query_vec = model.encode([query], normalize_embeddings=True)
I, D = search_faiss(index, query_vec, top_k=TOP_K)

print("\nоп найденных чанков:")
for i in I:
    print(f"\n— {chunks[i]['title']}\n{chunks[i]['chunk'][:200]}...")