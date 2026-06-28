from src.data_preprocessing.clean_data import clean_csv
from src.config.embbeding_model import embed_model
from src.vector_store.rerank import Reranker 
from src.config.query_expander_model import QueryExpander
from src.config.llm import LLMAgent
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


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть числом, сейчас: {raw_value!r}") from exc


ALPHA = _get_float_env("HYBRID_ALPHA", 0.6)
if not 0 <= ALPHA <= 1:
    raise ValueError(f"HYBRID_ALPHA должен быть в диапазоне от 0 до 1, сейчас: {ALPHA}")


class RAGPipeline:
    def __init__(self):
        self.embedder = embed_model
        # self.QueryExpander = QueryExpander()
        self.reranker = Reranker()
        self.llmagent = LLMAgent()

        self.embeddings = None
        self.chunks = None
        self.index = None
        self.vectorizer = None
        self.tfidf_matrix = None

    
    def load(self):
        # Загрузка источников при инициализации
        self._load_sources()
    
    def _load_sources(self):
        sources_changed = self._prepare_sources()

        # 1. Embeddings
        self.embeddings, self.chunks = get_embeddings(force_rebuild=sources_changed)

        # 2. FAISS Index
        self.index = build_faiss_index(self.embeddings, force_rebuild=sources_changed)

        # 3. Гибридный поиск
        self.vectorizer, self.tfidf_matrix = build_or_load_tfidf(
            self.chunks,
            force_rebuild=sources_changed,
        )

    def _prepare_sources(self) -> bool:
        raw_csv = os.path.join(RAW_DATA_DIR, "websites_updated.csv")
        sources_changed = False

        if not os.path.exists(input_clean_csv) or self._is_newer(raw_csv, input_clean_csv):
            clean_csv("websites_updated.csv")
            sources_changed = True
        else:
            print(f"Используем существующий очищенный CSV: {input_clean_csv}")

        if not os.path.exists(output_chuncked_csv) or self._is_newer(input_clean_csv, output_chuncked_csv):
            split_documents_fixed(input_clean_csv, "chunks_fixed.csv")
            sources_changed = True
        else:
            print(f"Используем существующие чанки: {output_chuncked_csv}")

        return sources_changed

    @staticmethod
    def _is_newer(source_path: str, target_path: str) -> bool:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Файл не найден: {source_path}")
        if not os.path.exists(target_path):
            return True
        return os.path.getmtime(source_path) > os.path.getmtime(target_path)



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

        message = self._build_promt(query, reranked_chunks)
        print('start generating...')
        answer = self.llmagent.answer(message=message)

        answer = answer.replace('<|im_end|>', '')


        return answer


    def _build_promt(self, query: str, context_chunks: list) -> str:
        context_text = "\n\n".join(
            chunk["chunk_text"] for chunk in context_chunks
            if "chunk_text" in chunk
        )

        SYSTEM_PROMPT = '''
        Ты — RAG-агент, отвечающий строго на основе CONTEXT.

        ПРАВИЛА:
        1. Используй ТОЛЬКО информацию из CONTEXT.
        2. Не используй внешние знания, обучение модели или догадки.


        ЗАПРЕЩЕНО:
        - объяснять, почему ответа нет
        - ссылаться на контекст формулировками вида
        «в предоставленном контексте»
        - рассуждать или делать предположения
        - давать частичный ответ

        ФОРМАТ ОТВЕТА:
        - кратко
        - по существу
        - без вводных фраз
        - без пояснений
        '''

        user_promt = f'''
        Контекст:
        {context_text}

        Вопрос: {query}
        Ответ: 
        '''

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_promt},
        ]

        return messages


