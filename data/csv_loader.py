"""
loader_csv.py — модуль для загрузки данных из CSV:
 - базы знаний (knowledge_base.csv)
 - базы вопросов и ответов (questions.csv)

Отвечает только за извлечение и сохранение.
Очистка и разбиение выполняются на следующих шагах пайплайна.
"""

from pathlib import Path
import pandas as pd
import json
import re

from utils.config import RAW_DIR, PROCESSED_DIR


# ==========================
# 🧠 Вспомогательные функции
# ==========================

def _try_read_csv(path: Path) -> pd.DataFrame:
    """Пытается считать CSV с разными кодировками."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1251")


def _normalize_text(s: str) -> str:
    """Убирает невидимые символы, множественные пробелы, обрезает концы."""
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip()
    return s


# ==========================
# 📘 Работа с базой знаний
# ==========================

def load_knowledge_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Загружает базу знаний из CSV и возвращает DataFrame.
    Ожидаемые поля: id, title, content.
    """
    if csv_path is None:
        csv_path = RAW_DIR / "knowledge_base.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Файл базы знаний не найден: {csv_path}")

    df = _try_read_csv(csv_path)

    if "content" not in df.columns:
        raise ValueError("❌ В CSV с базой знаний нет столбца 'content'.")

    print(f"📘 Загружена база знаний: {len(df)} записей.")
    return df


def save_kb_to_processed(kb_df: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> None:
    """
    Сохраняет каждую строку базы знаний в отдельный .txt-файл в processed/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for _, row in kb_df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()

        if not content or len(content) < 50:
            print(f"⚠️ Пропущен документ '{title or doc_id}' — слишком мало контента.")
            continue

        # Формируем безопасное имя файла
        if doc_id and title:
            fname = f"{doc_id}_{title}".replace(" ", "_")
        elif title:
            fname = title.replace(" ", "_")
        else:
            fname = f"doc_{_}"

        safe_name = "".join(ch for ch in fname if ch.isalnum() or ch in "._-")
        out_path = output_dir / f"{safe_name}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        saved_count += 1

    print(f"✅ База знаний сохранена: {saved_count} файлов в {output_dir}")


# ==========================
# ❓ Работа с вопросами и ответами
# ==========================

def load_qa_csv(qa_path: Path | None = None) -> pd.DataFrame | None:
    """
    Загружает CSV с вопросами и ответами, если он существует.
    Ожидаемые поля: question, answer.
    """
    if qa_path is None:
        qa_path = RAW_DIR / "questions.csv"

    if not qa_path.exists():
        print("ℹ️ CSV с вопросами не найден, этап QA пропущен.")
        return None

    df = _try_read_csv(qa_path)

    if "question" not in df.columns:
        raise ValueError("❌ В CSV с вопросами нет колонки 'question'.")

    if "answer" not in df.columns:
        print("⚠️ Вопросы найдены, но колонка 'answer' отсутствует. "
              "Будут сохранены только вопросы.")

    # Мягкая предочистка текста
    for col in df.columns:
        df[col] = df[col].astype(str).apply(_normalize_text)

    print(f"❓ Загружено {len(df)} вопросов-ответов.")
    return df


def save_qa_for_eval(qa_df: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> None:
    """
    Сохраняет вопросы-ответы в единый JSON-файл qa_dataset.json
    для последующего тестирования RAG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = qa_df.to_dict(orient="records")
    out_path = output_dir / "qa_dataset.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Вопросы-ответы сохранены в {out_path}")


if __name__ == "__main__":
    """
    Пример использования для локального теста:
      python -m data.loader_csv

    Проверяет работу загрузки и сохранения CSV:
      - считывает knowledge_base.csv и сохраняет тексты в processed/
      - считывает questions.csv и сохраняет qa_dataset.json
    """

    # Загружаем и сохраняем базу знаний
    try:
        kb_df = load_knowledge_csv()
        print(kb_df.head(1))
        save_kb_to_processed(kb_df)
    except Exception as e:
        print(f"❌ Ошибка при обработке базы знаний: {e}")

    # Загружаем и сохраняем базу вопросов
    try:
        qa_df = load_qa_csv()
        if qa_df is not None:
            print(qa_df.head(1))
            save_qa_for_eval(qa_df)
    except Exception as e:
        print(f"❌ Ошибка при обработке базы вопросов: {e}")

    print("\n✅ Тест loader_csv.py завершён.")
