# Структура проекта

```text
rag_project/
│
├── data/                     # исходные документы
│   ├── raw/                  # сырые файлы (txt, pdf, docx)
│   |── processed/            # очищенные тексты
│   └── chunks/               # чанки  
|
├── src/
|   ├── config/                   
│   │   ├── __init__.py
|   |   |── embedding_model.py    # единая загрузка embedding-модели (из .env)
│   │   └── llm_model.py          # (опционально) единая загрузка LLM
|   |
│   ├── data_preprocessing/
│   │   └── clean_data.py     # очистка и нормализация текстов
│   │
│   ├── chunking/
│   │   └── splitter.py       # разбиение текста на чанки
│   │
│   ├── embeddings/
│   │   └── embedder.py       # вычисление эмбеддингов
│   │
│   ├── vector_store/
│   │   └── faiss_store.py    # создание и поиск по FAISS
│   │
│   ├── retrieval/
│   │   └── retriever.py      # объединяет эмбеддер и векторное хранилище
│   │
│   ├── evaluation/
│   │   └── hit_at_k.py       # метрика Hit@K
│   │
│   └── main.py               # pipeline — объединяет все этапы
│
├── .env                      # настройки (пути, ключи, параметры)
├── .gitignore
├── requirements.txt
└── README.md
```