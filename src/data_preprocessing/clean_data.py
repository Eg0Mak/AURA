# import pandas as pd
# import re
# import os
# from dotenv import load_dotenv
# import os

# load_dotenv()

# RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
# PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")



# def clean_text(text: str, preserve_paragraphs: bool = False) -> str:
#     if not isinstance(text, str):
#         return ""
    
#     # Убираем управляющие символы и невидимые пробелы
#     text = text.replace("\xa0", " ")
#     text = re.sub(r"[©®™℠]", " ", text)
#     text = text.replace("₽", "рублей")
#     text = text.replace("$", "долларов") 
#     text = text.replace("€", "евро")
#     text = text.replace("→", "-")
#     text = text.replace("/", ",")

#     # Добавляем эквивалентное описание
#     text = text.replace('дм³', 'дм³ (кубических дециметров)')

#     # Убирает длинные последовательности точек, '_', '-'
#     text = re.sub(r"\.{3,}", " ", text)
#     text = re.sub(r"\_{2,}", " ", text)
#     text = re.sub(r"\-{3,}", " ", text)


#     # Удаляем мусорные символы, но сохраняем нужные для e-mail, телефонов и знаков
#     text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

#     # Если не нужно сохранять абзацы
#     if not preserve_paragraphs:
#         text = re.sub(r"\s+", " ", text)
#     else:
#         # Сохраняем двойные переносы абзацев
#         text = re.sub(r"\n{3,}", "\n\n", text)
#         text = re.sub(r"[ \t]+", " ", text)

#     # Пробелы перед пунктуацией
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r" {2,}", " ", text)

#     # Убираем двойные пробелы
#     text = re.sub(r" {2,}", " ", text)

#     # Приведение к нижнему регистру
#     text = text.lower()

#     return text.strip()

# def clean_csv(input_filename: str):
#     """Обработка CSV: очистка текстов, удаление пропусков и дублей"""
#     input_path = os.path.join(RAW_DATA_DIR, input_filename)
#     output_path = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")

#     data = pd.read_csv(input_path)
#     data = data.dropna(subset=["text"])

#     # Очистка текста и заголовков
#     data["text"] = data["text"].apply(clean_text)
#     data["title"] = data["title"].fillna("").apply(clean_text)

#     # Убираем пустые строки и дубли
#     data = data[data["text"].str.strip() != ""].drop_duplicates(subset=["text"])

#     os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
#     # data.to_csv(output_path, index=False)

#     print(f"Cleaned data saved in: {output_path}")
#     print(len(data))

#     return data

# if __name__ == "__main__":
#     clean_csv()

# src/data_preprocessing/clean_data.py
import os
import re
import pandas as pd
import pymorphy3

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text: str) -> str:
    """Лемматизация текста для русского языка"""
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)

def clean_text(text: str, preserve_paragraphs: bool = False, do_lemmatize: bool = True) -> str:
    """Очистка текста + лемматизация"""
    if not isinstance(text, str):
        return ""

    # Замены спецсимволов
    text = text.replace("\xa0", " ")
    text = text.replace("₽", "рублей")
    text = text.replace("$", "долларов")
    text = text.replace("→", "-")
    text = text.replace('дм³', 'дм³ (кубических дециметров)')

    # Убираем длинные последовательности точек, тире, подчёркиваний
    text = re.sub(r"(\.{3,}|_{2,}|-{3,})", " ", text)
    
    # Мусорные символы
    text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

    # Абзацы
    if preserve_paragraphs:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    # Пробелы перед пунктуацией
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)

    # Нижний регистр
    text = text.lower()

    # Лемматизация
    if do_lemmatize:
        text = lemmatize_text(text)

    return text.strip()


def clean_csv(input_filename: str, preserve_paragraphs: bool = False, do_lemmatize: bool = True):
    """Очистка CSV с текстами, лемматизация и сохранение"""
    input_path = os.path.join(RAW_DATA_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")

    data = pd.read_csv(input_path)
    data = data.dropna(subset=["text"])

    data["text"] = data["text"].apply(lambda x: clean_text(x, preserve_paragraphs, do_lemmatize))
    data["title"] = data.get("title", pd.Series([""]*len(data))).apply(
        lambda x: clean_text(x, preserve_paragraphs, do_lemmatize)
    )

    # Убираем пустые строки и дубли
    data = data[data["text"].str.strip() != ""].drop_duplicates(subset=["text"])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Cleaned data saved in: {output_path}")
    print(f"Всего записей: {len(data)}")

    return data