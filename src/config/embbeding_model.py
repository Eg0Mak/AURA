#src/config/embbeding_model.py
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


# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from llama_index.core.embeddings import BaseEmbedding
# from typing import List
# from pydantic import ConfigDict

# MODEL_NAME = "sberbank-ai/sbert_large_nlu_ru"

# class EmbedModelWrapper(BaseEmbedding):
#     model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

#     def __init__(self, model_name=MODEL_NAME, device="cpu"):
#         super().__init__(model_name=model_name)
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(device)

#     def _mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     # --- текст ---
#     def _get_text_embedding(self, text: str) -> List[float]:
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return self._mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()[0].tolist()

#     # --- запрос, просто делегируем ---
#     def _get_query_embedding(self, query: str) -> List[float]:
#         return self._get_text_embedding(query)

#     async def _aget_text_embedding(self, text: str) -> List[float]:
#         return self._get_text_embedding(text)

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         return self._get_query_embedding(query)

#     def get_batch_embeddings(self, texts: List[str], batch_size=32) -> List[List[float]]:
#         all_embeddings = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i+batch_size]
#             inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             batch_embeddings = self._mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()
#             all_embeddings.extend(batch_embeddings.tolist())
#         return all_embeddings

# embed_model = EmbedModelWrapper()
