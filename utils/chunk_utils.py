"""
chunk_utils.py — модуль для разбиения очищенных текстов на перекрывающиеся чанки.
Используется после cleaner.py, перед построением эмбеддингов.
"""

from pathlib import Path
from utils.config import PROCESSED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks_from_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Разбивает текст на перекрывающиеся чанки фиксированной длины (по символам).

    Пример:
        chunk_size=1000, overlap=100
        -> чанк 1: символы [0:1000]
        -> чанк 2: символы [900:1900]
    """
    # Проверка на валидные параметры
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE должен быть > 0")
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        # Добавляем только непустые чанки
        if chunk:
            chunks.append(chunk)

        # Передвигаем окно с учётом перекрытия
        start += chunk_size - overlap

    return chunks


def split_processed_texts(
    input_dir: Path = PROCESSED_DIR,
    output_dir: Path = CHUNKS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> None:
    """
    Делит очищенные тексты из processed/ на чанки и сохраняет их в chunks/.
    Каждый чанк сохраняется в отдельный .txt файл с индексом.

    Пример имени:
        report_robotics_chunk_1.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️ В папке {input_dir} нет файлов для разбиения на чанки.")
        return

    print(f"✂️ Найдено {len(txt_files)} файлов для разбиения.\n")

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = create_chunks_from_text(text, chunk_size, overlap)

            if not chunks:
                print(f"⚠️ {file_path.name}: не удалось создать чанки (пустой текст).")
                continue

            base_name = file_path.stem.replace(" ", "_")

            for i, chunk in enumerate(chunks, start=1):
                chunk_filename = f"{base_name}_chunk_{i}.txt"
                chunk_path = output_dir / chunk_filename

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

            print(f"✅ {file_path.name}: создано {len(chunks)} чанков")

        except Exception as e:
            print(f"❌ Ошибка при обработке {file_path.name}: {e}")

    print("\n📂 Разбиение завершено. Проверьте папку:", output_dir.resolve())


if __name__ == "__main__":
    split_processed_texts()
