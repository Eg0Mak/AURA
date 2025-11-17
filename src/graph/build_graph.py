# src/graph/build_graph.py
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
OUTPUT_FILE = os.path.join(CHUNKS_DIR, "graph.json")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")


class GraphBuilder:
    def __init__(self, chunks, emb_model_name=EMBEDDING_MODEL_NAME):
        self.chunks = chunks

        # Sentence embeddings
        self.emb_model = SentenceTransformer(emb_model_name)

        # NLI model (entailment / contradiction / neutral)
        self.nli_tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    def _compute_embeddings(self):
        texts = [c["chunk_text"] for c in self.chunks]
        return np.array(self.emb_model.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def _nli_score(self, a, b):
        """Return entailment score: 0..1"""
        inputs = self.nli_tokenizer(a, b, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        # label 2 = entailment
        return float(probs[0][2])

    def build(self, top_k=8, nli_threshold=0.55):
        print("Building Graph-RAG index using FAISS...")

        embeddings = self._compute_embeddings()
        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        graph = {c["web_id"]: [] for c in self.chunks}

        print(f"Searching top-{top_k+1} neighbors for each chunk via FAISS...")
        D, I = index.search(embeddings, top_k + 1)  # top_k+1 чтобы исключить себя позже

        for i, neighbors in enumerate(tqdm(I, total=len(I))):
            node = self.chunks[i]
            node_id = node["web_id"]
            text_i = node["chunk_text"]

            for j_idx, j in enumerate(neighbors):
                if j == i:
                    continue

                candidate = self.chunks[j]
                text_j = candidate["chunk_text"]

                # NLI connection
                nli = self._nli_score(text_i, text_j)

                if nli >= nli_threshold:
                    graph[node_id].append({
                        "web_id": candidate.get("web_id", j),
                        "chunk_text": candidate["chunk_text"],
                        "url": candidate.get("url"),
                        "web_id": candidate.get("web_id"),
                        "nli_score": nli,
                        "similarity": float(D[i][j_idx])
                    })

        # Save to JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        print(f"Graph built and saved to {OUTPUT_FILE}")
        return graph
