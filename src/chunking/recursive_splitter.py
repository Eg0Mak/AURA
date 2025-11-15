# src/chunking/recursive_splitter.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def split_documents_with_recursive(input_csv: str, output_path: str):
    """Рекурсивное разбиение текстов на чанки"""
    print("start_chunking.....")
    df = pd.read_csv(input_csv)
    print(f"Всего документов: {len(df)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, # НОВОЕ
        separators=["\n\n", "\n", ".", "?", "!", ";", " ", ""]
    )

    all_chunks = []

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append({
                "web_id": row.get("web_id", ""),
                "url": row.get("url", ""),
                "kind": row.get("kind", ""),
                "chunk_text": chunk.strip()
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(output_path, index=False)

    print(f"Чанки сохранены в: {output_path}")
    print(f"Всего чанков: {len(chunks_df)}")


if __name__ == "__main__":
    input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
    output_csv = os.path.join(CHUNKS_DIR, "chunks_recursive.csv")
    split_documents_with_recursive(input_csv, output_csv)
