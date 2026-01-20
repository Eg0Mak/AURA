# AURA - Augmented Universal Retrieval Assistant
## Powerful RAG system for finding relevant information



# Структура проекта

```text
rag_project/
│
├── data/                       # Данные (без изменений)
│   ├── raw/
│   ├── processed/
│   └── chunks/
│
├── src/
│   ├── config/                 # Конфигурации (LLM, DB connection)
│   │
│   ├── core/                   
│   │   ├── preprocessing/      # (бывший data_preprocessing)
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── vector_store/
│   │   └── graph/
│   │
│   ├── pipeline/               # СВЯЗУЮЩАЯ ЛОГИКА
│   │   └── rag_pipeline.py     # Главный класс, объединяющий поиск + генерацию
│   │
│   ├── api/                    # BACKEND (REST API)
│   │   ├── main.py             # Точка входа (FastAPI app)
│   │   ├── routes.py           # Эндпоинты (/chat, /indexing)
│   │   ├── schemas.py          # Pydantic модели (Request/Response)
│   │   └── dependencies.py     # Инициализация моделей (Singleton)
│   │
│   └── interface/              # FRONTEND / BOT
│       ├── streamlit_app.py    # Веб-интерфейс (Streamlit)
│       └── telegram_bot.py     # Чат-бот (Aiogram)
│
├── docker/                     # Dockerfiles
│   ├── backend.Dockerfile
│   └── frontend.Dockerfile
│
├── docker-compose.yml          # Оркестрация контейнеров
├── .env
├── requirements.txt
└── main.py                     # Скрипт для локального запуска/тестов
```