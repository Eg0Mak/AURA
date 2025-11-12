import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")

# Загружаем модель один раз
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class EmbedModelWrapper:
    """Обёртка для SentenceTransformer, чтобы совместимо с LlamaIndex"""
    def __init__(self, model):
        self.model = model

    def get_text_embedding(self, text: str):
        return self.model.encode(text, normalize_embeddings=True)

    def get_batch_embeddings(self, texts: list):
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

embed_model = EmbedModelWrapper(_model)
