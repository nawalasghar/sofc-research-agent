"""Tool: search_literature - semantic search over the SOFC paper corpus.

Reuses the artefacts and retrieval settings of the existing literature RAG project
(``index/chunks.faiss`` + ``index/chunks.jsonl``, all-MiniLM-L6-v2, cosine similarity
via inner product on unit vectors). Nothing is re-embedded or rebuilt here; the index
location comes from ``paths.rag_root`` in config.yaml.

The FAISS index and the embedding model are loaded once and cached, so the first call
pays a few seconds of model load and later calls are fast.
"""
from __future__ import annotations

import json
from typing import Any

from .. import config

_CACHE: dict[str, Any] = {}


def _short_title(title: str, limit: int = 90) -> str:
    text = (title or "unknown").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _snippet(text: str, n_words: int) -> str:
    words = text.split()
    if len(words) <= n_words:
        return text
    return " ".join(words[:n_words]) + " ..."


def _load() -> tuple[Any, list[dict], Any]:
    """Load (index, metadata, embedding model) once and cache them."""
    if "index" in _CACHE:
        return _CACHE["index"], _CACHE["meta"], _CACHE["model"]

    index_file, meta_file = config.index_files()
    missing = [str(p) for p in (index_file, meta_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "literature index not found: " + ", ".join(missing)
            + " - check paths.rag_root in config.yaml and build the index in that project"
        )

    import faiss  # deferred: heavy, and only this tool needs it
    from sentence_transformers import SentenceTransformer

    with meta_file.open(encoding="utf-8") as f:
        meta = [json.loads(line) for line in f if line.strip()]
    index = faiss.read_index(str(index_file))
    if index.ntotal != len(meta):
        raise ValueError(
            f"index/metadata mismatch: {index.ntotal} vectors vs {len(meta)} chunks - rebuild the index"
        )
    model = SentenceTransformer(str(config.get("literature.embed_model")))

    _CACHE.update(index=index, meta=meta, model=model)
    return index, meta, model


def search_literature(query: str, k: int | None = None) -> dict:
    """Find the passages in the paper corpus most relevant to ``query``.

    Args:
        query: natural-language question or keyword phrase.
        k: number of passages to return (default from config, capped by ``literature.max_k``).

    Returns:
        A JSON-serialisable dict with ``ok``, a one-line ``summary``, and a ``passages``
        list carrying title, page, citation string and a text snippet for each hit.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "tool": "search_literature", "error": "query must not be empty"}

    default_k = int(config.get("literature.default_k", 5))
    max_k = int(config.get("literature.max_k", 20))
    try:
        k_eff = default_k if k is None else int(k)
    except (TypeError, ValueError):
        return {"ok": False, "tool": "search_literature", "error": f"k must be an integer, got {k!r}"}
    k_eff = max(1, min(k_eff, max_k))

    try:
        index, meta, model = _load()
    except Exception as exc:
        return {"ok": False, "tool": "search_literature", "error": str(exc)}

    vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(vector, k_eff)

    n_words = int(config.get("literature.snippet_words", 60))
    passages: list[dict] = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        row = meta[int(idx)]
        title = _short_title(row.get("title", ""))
        passages.append({
            "rank": rank,
            "score": round(float(score), 4),
            "title": title,
            "page": row.get("page"),
            "citation": f"[{title}, p.{row.get('page')}]",
            "file_name": row.get("file_name"),
            "doi": row.get("doi") or None,
            "year": row.get("year") or None,
            "open_access": row.get("open_access") or None,
            "snippet": _snippet(row.get("text", ""), n_words),
        })

    if not passages:
        summary = f"No passages retrieved for {query!r}."
    else:
        top = passages[0]
        summary = (
            f"{len(passages)} passages for {query!r}; best match {top['citation']} "
            f"(score {top['score']:.3f}), corpus size {index.ntotal} chunks."
        )

    return {
        "ok": True,
        "tool": "search_literature",
        "query": query,
        "k": k_eff,
        "n_passages": len(passages),
        "corpus_chunks": int(index.ntotal),
        "summary": summary,
        "passages": passages,
    }
