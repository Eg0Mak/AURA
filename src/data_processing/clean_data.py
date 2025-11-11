import pandas as pd
import re
import os
from dotenv import load_dotenv
import os

load_dotenv()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")



def clean_text(text: str, preserve_paragraphs: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    
    # Убираем управляющие символы и невидимые пробелы
    text = text.replace("\xa0", " ")
    text = re.sub(r"[©®™℠]", " ", text)

    # Удаляем мусорные символы, но сохраняем нужные для e-mail, телефонов и знаков
    text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

    # Если не нужно сохранять абзацы
    if not preserve_paragraphs:
        text = re.sub(r"\s+", " ", text)
    else:
        # Сохраняем двойные переносы абзацев
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
    
    # Убираем пробелы перед знаками пунктуации
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Убираем двойные пробелы
    text = re.sub(r" {2,}", " ", text)

    return text.strip()

def clean_csv(input_filename: str):
    """Обработка CSV: очистка текстов, удаление пропусков и дублей"""
    input_path = os.path.join(RAW_DATA_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")

    data = pd.read_csv(input_path)
    data = data.dropna(subset=["text"])

    # Очистка текста и заголовков
    data["text"] = data["text"].apply(clean_text)
    data["title"] = data["title"].fillna("").apply(clean_text)

    # Убираем пустые строки и дубли
    data = data[data["text"].str.strip() != ""].drop_duplicates(subset=["text"])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Cleaned data saved in: {output_path}")
    print(len(data))

    return data