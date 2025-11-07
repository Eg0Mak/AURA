"""
cleaner.py — модуль очистки текстов после извлечения из PDF.
Используется после pdf_loader.py. Очищает артефакты PDF, переносы строк, пробелы и служебные символы.
"""

import re
from pathlib import Path
from utils.config import PROCESSED_DIR


def clean_text(text: str) -> str:
    """
    Выполняет многоступенчатую очистку текста:
      1. Удаляет артефакты PDF (неразрывные пробелы, мусорные символы)
      2. Нормализует тире, маркеры и кавычки
      3. Убирает переносы строк внутри предложений
      4. Сокращает лишние пробелы и пустые строки
    """
    # === 1. Замена артефактов и неразрывных пробелов ===
    text = text.replace("\xa0", " ")   # неразрывные пробелы
    text = text.replace("\t", " ")
    text = text.replace("•", "-").replace("·", "-")

    # === 2. Нормализация символов и пунктуации ===
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("“", "\"").replace("”", "\"").replace("«", "\"").replace("»", "\"")

    # === 3. Убираем переносы строк внутри абзацев (если нет пустой строки между ними) ===
    # Пример: "в 2025 году\nотдел мехатроники" → "в 2025 году отдел мехатроники"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # === 4. Убираем повторяющиеся пробелы ===
    text = re.sub(r"[ ]{2,}", " ", text)

    # === 5. Удаляем лишние пустые строки ===
    text = re.sub(r"\n{3,}", "\n\n", text)

    # === 6. Убираем случайные символы в начале/конце ===
    return text.strip()


def clean_processed_texts(input_dir: Path = PROCESSED_DIR) -> None:
    """
    Проходит по всем .txt-файлам в папке processed/ и очищает их содержимое.
    Если текст после очистки пустой — сохраняет оригинал с предупреждением.
    """
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ В папке {input_dir} нет текстовых файлов для очистки.")
        return

    print(f"🧹 Найдено {len(txt_files)} файлов для очистки.\n")

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned = clean_text(raw_text)

            if not cleaned.strip():
                print(f"⚠️ {file_path.name}: текст пуст после очистки, оригинал сохранён.")
                continue

            # Перезаписываем очищенный файл
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"✅ Очищен: {file_path.name}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {file_path.name}: {e}")

    print("\n📂 Очистка завершена. Проверяйте папку:", input_dir.resolve())


if __name__ == "__main__":
    clean_processed_texts()
