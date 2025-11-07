from pathlib import Path
from PyPDF2 import PdfReader
from utils.config import RAW_DIR, PROCESSED_DIR


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Извлекает текст из одного PDF-файла постранично.
    Возвращает объединённый текст.
    """
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                print(f"⚠️ Страница {page_num} в {pdf_path.name} пуста или нераспознана.")
            text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения файла {pdf_path.name}: {e}")


def load_pdfs(input_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> None:
    """
    Извлекает текст из всех PDF-файлов в папке raw/ и сохраняет результаты в processed/.
    Пути берутся из config.py, если не заданы явно.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️ В папке {input_dir} нет PDF-файлов.")
        return

    print(f"🔍 Найдено {len(pdf_files)} PDF-файлов для обработки.\n")

    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)

            if not text.strip():
                print(f"⚠️ В {pdf_file.name} не удалось извлечь текст.")
                continue

            output_filename = pdf_file.stem.replace(" ", "_") + ".txt"
            output_file = output_dir / output_filename

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text.strip())

            print(f"✅ {pdf_file.name} → {output_filename}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_file.name}: {e}")

    print("\n📂 Обработка завершена. Проверяйте папку:", output_dir.resolve())


if __name__ == "__main__":
    load_pdfs()
