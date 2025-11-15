def analyze_overlaps(chunks: list[str]):
    """
    chunks — список текстов чанков (в порядке split)
    """
    overlaps = []
    
    for i in range(len(chunks) - 1):
        a = chunks[i]
        b = chunks[i + 1]

        # Ищем максимальный overlap — конец A совпадает с началом B
        max_overlap = 0
        min_len = min(len(a), len(b))

        for k in range(1, min_len + 1):
            if a[-k:] == b[:k]:
                max_overlap = k

        overlaps.append(max_overlap)

    print(f"Количество чанков: {len(chunks)}")
    print(f"Средний overlap: {sum(overlaps) / len(overlaps):.2f}")
    print(f"Максимальный overlap: {max(overlaps)}")
    print(f"Минимальный overlap: {min(overlaps)}")

    # Показываем первые 10
    print("\nПервые 10 overlap значений:", overlaps[:10])

    return overlaps


import pandas as pd

df = pd.read_csv("data/chunks/chunks_semantic.csv")
chunks = df["chunk_text"].tolist()

analyze_overlaps(chunks)