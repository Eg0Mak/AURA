import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")


embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

print(f"Загружена модель эмбеддингов: {EMBEDDING_MODEL_NAME}")
