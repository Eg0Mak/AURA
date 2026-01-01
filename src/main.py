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


# rag = RAGPipeline()
# rag.load()

# query = 'Условия по кредитной карте?' 

# result = asyncio.run(rag.run_pipeline(query))
# print(result)

import gradio as gr
import asyncio

# Инициализация pipeline
rag_pipeline = RAGPipeline()
rag_pipeline.load()

async def chat_fn(message, state):
    # Добавляем сообщение пользователя
    state.append({"role": "user", "content": message})

    # Генерация ответа
    answer = await rag_pipeline.run_pipeline(message)
    state.append({"role": "assistant", "content": answer})

    # Возвращаем список сообщений напрямую
    return state, state

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()       # теперь принимает dict с role и content
    state = gr.State([])

    txt = gr.Textbox(placeholder="Введите вопрос...")
    send_btn = gr.Button("Отправить")

    # Привязка кнопки и Enter
    send_btn.click(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])

    # Кнопка очистки
    clear_btn = gr.Button("Очистить чат")
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

demo.launch(share=True)
