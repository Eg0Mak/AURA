"""
pipeline.py — единый модуль запуска подготовки данных для RAG-системы.

Этапы:
 1. Загрузка исходных данных (CSV или PDF)
 2. Сохранение текстов в processed/
 3. Очистка текстов (cleaner)
 4. Разбиение на чанки (chunk_utils)
 5. Обработка вопросов-ответов (QA)

Запуск:
    python -m data.pipeline
"""

from pathlib import Path
from utils.config import RAW_DIR, PROCESSED_DIR
from data.pdf_loader import load_pdfs
from data.csv_loader import (
    load_knowledge_csv,
    save_kb_to_processed,
    load_qa_csv,
    save_qa_for_eval,
)
from data.cleaner import clean_processed_texts
from utils.chunk_utils import split_processed_texts


# ==========================
# 🚀 Основной пайплайн
# ==========================

def run_data_pipeline():
    """
    Универсальный пайплайн обработки данных.
    Автоматически определяет, что загружать — CSV или PDF.
    """
    print("\n🔧 Запуск пайплайна обработки данных...")

    kb_csv = RAW_DIR / "knowledge_base.csv"
    qa_csv = RAW_DIR / "questions.csv"

    # 1️⃣ База знаний (CSV или PDF)
    if kb_csv.exists():
        print("📘 Обнаружен CSV-файл базы знаний.")
        kb_df = load_knowledge_csv(kb_csv)
        save_kb_to_processed(kb_df, PROCESSED_DIR)
    else:
        print("📄 CSV не найден — ищем PDF в data/raw/")
        load_pdfs(RAW_DIR, PROCESSED_DIR)

    # 2️⃣ Очистка текстов
    print("\n🧹 Этап очистки текстов...")
    clean_processed_texts(PROCESSED_DIR)

    # 3️⃣ Разбиение на чанки
    print("\n✂️  Этап разбиения на чанки...")
    split_processed_texts()

    # 4️⃣ Обработка вопросов и ответов (если есть)
    if qa_csv.exists():
        print("\n❓ Обнаружен файл с вопросами.")
        qa_df = load_qa_csv(qa_csv)
        if qa_df is not None:
            save_qa_for_eval(qa_df, PROCESSED_DIR)
    else:
        print("\nℹ️ Файл с вопросами не найден — этап QA пропущен.")

    print("\n✅ Полный цикл пайплайна завершён.")
    print(f"📂 Обработанные данные: {PROCESSED_DIR}")
    print("📦 Готово к созданию эмбеддингов и индексации.\n")


# ==========================
# 🧪 Тестовый запуск
# ==========================
if __name__ == "__main__":
    """
    Тест пайплайна. Запускает весь процесс обработки данных.
    """
    try:
        run_data_pipeline()
    except Exception as e:
        print(f"❌ Ошибка во время выполнения пайплайна: {e}")
