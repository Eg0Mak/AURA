from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os
from src.pipeline.rag_pipeline import RAGPipeline
import asyncio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# load_dotenv()
# # TOP_K = int(os.getenv("TOP_K", 20))
# ALPHA = float(os.getenv("HYBRID_ALPHA"))

# TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5)) # топ после гибридного поиска, который идёт на rerank
# FAISS_TOP_N = int(os.getenv("FAISS_TOP_N", 10))
# TFIDF_TOP_N = int(os.getenv("TFIDF_TOP_N", 10))


rag = RAGPipeline()
rag.load()

query = 'Что такое кредитная карта?'

result = asyncio.run(rag.run_pipeline(query))
print(result)