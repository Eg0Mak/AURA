# src/chunking/simple_splitter.py
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))



def fixed_size_split(text: str, chunk_size: int, overlap: int):
    """
    Делит текст на фиксированные чанки без рекурсивной логики.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap  # смещаемся назад на overlap

    return chunks


def split_documents_fixed(input_csv: str, filename: str):
    """Фиксированное разбиение текстов на чанки"""
    print("start_simple_chunking...")
    df = pd.read_csv(input_csv)
    print(f"Всего документов: {len(df)}")

    all_chunks = []

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        chunks = fixed_size_split(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk in chunks:
            all_chunks.append({
                "web_id": row.get("web_id", ""),
                "url": row.get("url", ""),
                "kind": row.get("kind", ""),
                "chunk_text": chunk
            })

    output_chuncked_csv = os.path.join(CHUNKS_DIR, filename)
    os.makedirs(os.path.dirname(output_chuncked_csv), exist_ok=True)
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(output_chuncked_csv, index=False)

    print(f"Чанки сохранены в: {output_chuncked_csv}")
    print(f"Всего чанков: {len(chunks_df)}")


if __name__ == "__main__":
    input_csv = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
    output_csv = os.path.join(CHUNKS_DIR, "chunks_fixed.csv")
    split_documents_fixed(input_csv, output_csv)
