# src/graph/graph_store.py
import json
import os

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
output_csv = os.path.join(CHUNKS_DIR, "graph.json")

class GraphStore:
    def __init__(self):
        self.graph = None

    def load(self):
        if self.graph is not None:
            return self.graph

        if not os.path.exists(output_csv):
            raise FileNotFoundError(f"Graph file not found: {output_csv}")
        
        print("Using Graph...")

        with open(output_csv, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        return self.graph


graph_store = GraphStore()
