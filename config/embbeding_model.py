import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")


embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

print(f"Загружена модель эмбеддингов: {EMBEDDING_MODEL_NAME}")
