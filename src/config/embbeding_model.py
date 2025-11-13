import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from typing import List
from pydantic import ConfigDict

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")

class EmbedModelWrapper(BaseEmbedding):
    """Обёртка для SentenceTransformer для LlamaIndex"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.model = SentenceTransformer(model_name, device='mps')  # Теперь работает

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Метод для батч-эмбеддингов, который используется в embedder.py"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

embed_model = EmbedModelWrapper(EMBEDDING_MODEL_NAME)