# # src/graph/graph_expander.py
# def graph_expand(initial_chunks, graph_index, max_depth=1, max_neighbors=5):
#     """
#     Возвращает: initial_chunks + соседние узлы из графа.
#     Формат чанков НЕ меняем, чтобы rerank работал как раньше.
#     """
#     seen = set()
#     expanded = []

#     # 1) добавляем базовые
#     for ch in initial_chunks:
#         cid = ch["web_id"]
#         expanded.append(ch)
#         seen.add(cid)

#     # 2) ищем соседей
#     for ch in initial_chunks:
#         cid = ch["web_id"]
#         neighbors = graph_index.get(str(cid), [])[:max_neighbors]

#         for n in neighbors:
#             nid = n["web_id"]
#             if nid not in seen:
#                 expanded.append(n)
#                 seen.add(nid)

#     return expanded


# src/graph/graph_expander.py
from typing import List, Dict, Any
import math

def build_webid_to_chunks_map(chunks: List[Dict[str, Any]]):
    """
    Возвращает словарь web_id (str) -> list of (chunk_index, chunk_dict)
    """
    mapping = {}
    for idx, c in enumerate(chunks):
        wid = str(c.get("web_id", c.get("webId", "")))
        mapping.setdefault(wid, []).append((idx, c))
    return mapping


def graph_expand(
    initial_chunks: List[Dict],
    graph_index: Dict[str, List[Dict]],
    chunks: List[Dict],
    max_depth: int = 1,
    max_neighbors_per_node: int = 5,
    max_neighbors_total: int = 10,
    min_similarity: float = 0.0,
    min_nli: float = 0.0
) -> List[Dict]:
    """
    Возвращает initial_chunks + соседние оригинальные chunk-объекты из `chunks`.
    - graph_index: загруженный graph.json (ключи — строки web_id -> list of neighbor dicts)
    - chunks: исходный список всех чанков (тот же, что и при индексации)
    - дедупликация идёт по индексу чанка (чтобы не терять разные чанки с тем же web_id)
    """
    webid_map = build_webid_to_chunks_map(chunks)

    seen_chunk_idxs = set()
    expanded = []

    # Add base chunks and mark seen by their index in global chunks (if available)
    for base in initial_chunks:
        # Try to find index of base chunk within global chunks:
        # prefer explicit chunk_id or match by web_id + chunk_text
        found_idx = None
        if "chunk_id" in base:
            # if chunk_id was used as integer index
            try:
                found_idx = int(base["chunk_id"])
                if found_idx < 0 or found_idx >= len(chunks):
                    found_idx = None
            except Exception:
                found_idx = None

        if found_idx is None:
            # fallback: match by web_id and chunk_text (fast heuristic)
            wid = str(base.get("web_id", ""))
            text = base.get("chunk_text", "").strip()
            candidates = webid_map.get(wid, [])
            for idx, c in candidates:
                if c.get("chunk_text","").strip() == text:
                    found_idx = idx
                    break

        if found_idx is None:
            # best-effort: append base as-is (can't dedupe)
            expanded.append(base)
        else:
            if found_idx not in seen_chunk_idxs:
                seen_chunk_idxs.add(found_idx)
                expanded.append(chunks[found_idx])

    # Expand by graph neighbors (1 hop). max_depth currently supported ==1.
    # For each initial chunk, collect neighbor web_ids and add corresponding chunks.
    total_added = 0
    for base in initial_chunks:
        if total_added >= max_neighbors_total:
            break

        wid = str(base.get("web_id", ""))
        neighbors = graph_index.get(wid, [])  # neighbors are list of dicts saved in graph.json

        # sort or filter neighbors by stored similarity/nli if present
        filtered = []
        for n in neighbors[: max_neighbors_per_node]:
            sim = float(n.get("similarity", 0.0) if n.get("similarity") is not None else 0.0)
            nli = float(n.get("nli_score", 0.0) if n.get("nli_score") is not None else 0.0)
            if sim >= min_similarity and nli >= min_nli:
                filtered.append(n)
        # iterate filtered neighbors and add original chunks (may be multiple per web_id)
        for n in filtered:
            if total_added >= max_neighbors_total:
                break
            neighbor_wid = str(n.get("web_id", n.get("webId", "")))
            candidates = webid_map.get(neighbor_wid, [])
            # add candidates up to available, but avoid duplicates by index
            for idx, c in candidates:
                if idx in seen_chunk_idxs:
                    continue
                seen_chunk_idxs.add(idx)
                expanded.append(c)
                total_added += 1
                if total_added >= max_neighbors_total:
                    break

    return expanded

