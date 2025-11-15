# src/data_preprocessing/clean_data.py
import os
import re
import pandas as pd
import pymorphy3
from nltk.corpus import stopwords
# import nltk
# nltk.download("stopwords")
# RUSSIAN_STOPWORDS = set(stopwords.words("russian"))

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# morph = pymorphy3.MorphAnalyzer()

# def lemmatize_text(text: str, remove_stopwords: bool = False) -> str:
#     """Лемматизация текста для русского языка"""
#     words = text.split()
#     lemmas = [morph.parse(word)[0].normal_form for word in words]
#     if remove_stopwords:
#         lemmas = [w for w in lemmas if w not in RUSSIAN_STOPWORDS]
#     return " ".join(lemmas)

# def clean_text(text: str, preserve_paragraphs: bool = False, do_lemmatize: bool = False, remove_stopwords: bool = True) -> str:
#     """Очистка текста + лемматизация"""
#     if not isinstance(text, str):
#         return ""

#     # Замены спецсимволов
#     text = text.replace("\xa0", " ")
#     text = text.replace("₽", "рублей")
#     text = text.replace("$", "долларов")
#     text = text.replace("→", "-")
#     text = text.replace('дм³', 'дм³ (кубических дециметров)')

#     # Убираем длинные последовательности точек, тире, подчёркиваний
#     text = re.sub(r"(\.{3,}|_{2,}|-{3,})", " ", text)
    
#     # Мусорные символы
#     text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

#     # Абзацы
#     if preserve_paragraphs:
#         text = re.sub(r"[ \t]+", " ", text)
#         text = re.sub(r"\n{3,}", "\n\n", text)
#     else:
#         text = re.sub(r"\s+", " ", text)

#     # Пробелы перед пунктуацией
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r" {2,}", " ", text)

#     # Нижний регистр
#     text = text.lower()

#     Лемматизация
#     if do_lemmatize:
#         text = lemmatize_text(text, remove_stopwords=remove_stopwords)

#     return text.strip()

def clean_text(text: str, preserve_paragraphs: bool = False):
    """Очистка текста для RAG без потери смысла"""

    if not isinstance(text, str):
        return ""

    # Замены спецсимволов — оставляем только безопасные
    # text = text.replace("\xa0", " ")  # неразрывные пробелы
    # text = text.replace("→", "->")    # стрелки оставляем эквивалентами
    # text = text.replace("₽", " руб.")  # НЕ удаляем валюты
    # text = text.replace("$", " $ ")    # НЕ удаляем доллары
    # text = text.replace("дм³", "дм³")  # оставляем как есть

    # # Убираем шумовые последовательности: ..... ----- ______
    # text = re.sub(r"(\.{3,}|_{2,}|-{3,})", " ", text)

    # # Убираем мусорные символы, но НЕ трогаем: цифры, буквы, валюты
    # text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

    # # Сохраняем абзацы если нужно
    # if preserve_paragraphs:
    #     text = re.sub(r"[ \t]+", " ", text)
    #     text = re.sub(r"\n{3,}", "\n\n", text)
    # else:
    #     text = re.sub(r"\s+", " ", text)

    # # Пробелы перед пунктуацией
    # text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # text = re.sub(r" {2,}", " ", text)

    # # Приводим к нижнему регистру
    # text = text.lower()

    return text.strip()


def clean_csv(input_filename: str, preserve_paragraphs: bool = False):
    """Очистка CSV с текстами, лемматизация и сохранение"""
    input_path = os.path.join(RAW_DATA_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")

    data = pd.read_csv(input_path)
    data = data.dropna(subset=["text"])

    data["text"] = data["text"].apply(lambda x: clean_text(x, preserve_paragraphs))
    data["title"] = data.get("title", pd.Series([""]*len(data))).apply(
        lambda x: clean_text(x, preserve_paragraphs)
    )

    # Убираем пустые строки и дубли
    data = data[data["text"].str.strip() != ""].drop_duplicates(subset=["text"])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Cleaned data saved in: {output_path}")
    print(f"Всего записей: {len(data)}")

    return data