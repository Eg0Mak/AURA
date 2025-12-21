from src.data_preprocessing.clean_data import clean_csv
from src.config.embbeding_model import embed_model
from src.vector_store.rerank import Reranker 
from src.config.query_expander_model import QueryExpander
from src.chunking.simple_splitter import split_documents_fixed
from src.embeddings.embedder import get_embeddings
from src.vector_store.hybrid_search import build_or_load_tfidf, search_hybrid, build_faiss_index
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Пути
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")

input_clean_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
output_chuncked_csv = os.path.join(CHUNKS_DIR, "chunks_fixed.csv")

# Переменные окружения
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5)) # топ после гибридного поиска, который идёт на rerank
FAISS_TOP_N = int(os.getenv("FAISS_TOP_N", 10))
TFIDF_TOP_N = int(os.getenv("TFIDF_TOP_N", 10))
ALPHA = float(os.getenv("HYBRID_ALPHA"))


class RAGPipeline:
    def __init__(self):
        self.embedder = embed_model
        # self.QueryExpander = QueryExpander()
        self.reranker = Reranker()

        self.embeddings = None
        self.chunks = None
        self.index = None
        self.vectorizer = None
        self.tfidf_matrix = None

    
    def load(self):
        # Загрузка источников при инициализации
        self._load_sources()
    
    def _load_sources(self):
        # 1. Очистка данных
        clean_csv('websites_updated.csv')

        # 2. Chunking
        split_documents_fixed(input_clean_csv, 'chunks_fixed.csv')

        # 3. Embeddings
        self.embeddings, self.chunks = get_embeddings()

        # 4. FAISS Index
        self.index = build_faiss_index(self.embeddings)

        # 5. Гипбридный поиск
        self.vectorizer, self.tfidf_matrix = build_or_load_tfidf(self.chunks)



    async def run_pipeline(self, query: str) -> str:
        # 1. Преобразование запроса в вектор
        query_vec = np.array([embed_model.get_text_embedding(query)], dtype=np.float32)

        # 2. Гибридный поиск по векторной базе
        hybrid_chunks, scores = search_hybrid(
            self.index, query_vec, query, self.chunks,
            top_k=TOP_K_RERANK, alpha=ALPHA,
            tfidf_vectorizer=self.vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            faiss_top_n=FAISS_TOP_N,
            tfidf_top_n=TFIDF_TOP_N
        )

        # 3. Rerank top-5
        reranked_chunks = self.reranker.rerank(query, hybrid_chunks, top_n=TOP_K_RERANK) 

        return self._build_promt(query, reranked_chunks)


    def _build_promt(self, query: str, context_chunks: list) -> str:
        context_text = "\n\n".join(
            chunk["chunk_text"] for chunk in context_chunks
            if "chunk_text" in chunk
        )

        promt = f'''
        Ты помощник AURA. Используй только контекст ниже, чтобы ответить на вопрос.
        Если ответа нет в контексте, скажи "Я не знаю".

        Контекст:
        {context_text}

        Вопрос: {query}
        Ответ: 
        '''

        return promt


