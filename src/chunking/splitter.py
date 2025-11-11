# splitter.py
from config.embedding_model import embed_model
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()

PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def split_documents_with_semantics(input_csv: str, output_path: str):
    df = pd.read_csv(input_csv)
    print(f"Всего документов: {len(df)}")

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        doc = Document(
            text=text,
            metadata={
                "url": row.get("url", ""),
                "web_id": row.get("web_id", ""),
                "kind": row.get("kind", "")
            }
        )

        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            all_chunks.append({
                "web_id": row.get("web_id", ""),
                "url": row.get("url", ""),
                "kind": row.get("kind", ""),
                "chunk_text": node.get_content().strip()
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(output_path, index=False)

    print(f"Чанки сохранены в: {output_path}")
    print(f"Всего чанков: {len(chunks_df)}")


if __name__ == "__main__":
    input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
    output_csv = os.path.join(CHUNKS_DIR, "chunks_semantic.csv")
    split_documents_with_semantics(input_csv, output_csv)
