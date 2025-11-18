# AURA - Augmented Universal Retrieval Assistant
## Powerful RAG system for finding relevant information









# Структура проекта

```text
rag_project/
│
├── data/                       
│   ├── raw/                    
│   ├── processed/              
│   └── chunks/                 
│
├── src/
|   ├── config/
│   │   ├── llm.py
│   │   └── embedding_model.py  
|   |
|   |
│   ├── data_preprocessing/
│   │   ├── clean_data.py  
│   │   └── marked_llm.py  
|   |
│   ├── chunking/
│   │   ├── splitter.py                 # semantic splitter
|   |   ├── recursive_splitter.py       # recursive splitter  
│   │   └── simple_splitter.py          # custom splitter
|   |   
│   ├── embeddings/
│   │   └── embedder.py    
│   │
│   ├── vector_store/
|   |   ├── faiss_store.py              # base version of FAISS
|   |   ├── hybrid_search.py            # FAISS + Tf-Idf
│   │   └── rerank.py                   # Reranker
|   |
|   ├── graph/
|   |   ├── build_graph.py
|   |   ├── graph_expander.py
│   │   └── graph_store.py
|   | 
│   │
│   ├── evaluation/
│   │   └── hit_at_k.py    
│   │
│   └── main.py            
│
├── .env                   
├── .gitignore
├── requirements.txt
└── README.md
```