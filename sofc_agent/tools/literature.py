"""Tool: search_literature - semantic search over the SOFC paper corpus.

Reuses the artefacts and retrieval settings of the existing literature RAG project
(``index/chunks.faiss`` + ``index/chunks.jsonl``, all-MiniLM-L6-v2, cosine similarity
via inner product on unit vectors). Nothing is re-embedded or rebuilt here; the index
location comes from ``paths.rag_root`` in config.yaml.

The FAISS index and the embedding model are loaded once and cached, so the first call
pays a few seconds of model load and later calls are fast.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any

from .. import config

_CACHE: dict[str, Any] = {}

_STOPWORDS = frozenset(
    "about according also and are does doing done for from has have how into its less more "
    "much not than that the their them then there these they this those was were what when "
    "where which while who why with would".split()
)


def _short_title(title: str, limit: int = 90) -> str:
    # Titles scraped from PDF metadata can carry HTML entities ("300-600&#160;&#176;C").
    text = re.sub(r"\s+", " ", html.unescape(title or "unknown")).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _stem(word: str) -> str:
    """Crude 5-letter stem so 'conduct' matches 'conductivity' and 'conduction'."""
    return word[:5]


def _query_stems(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {_stem(w) for w in words if len(w) > 2 and w not in _STOPWORDS}


def _snippet(text: str, n_words: int, query: str = "") -> str:
    """Return the ``n_words`` window of ``text`` that mentions the query terms most.

    A 500-word chunk often opens with an unrelated paragraph, and a small model reading
    only the first 60 words would miss the sentence that made the chunk match. Ties go
    to the earliest window, so a query with no lexical overlap falls back to the start.
    """
    words = text.split()
    if len(words) <= n_words:
        return text
    stems = _query_stems(query)
    start = 0
    if stems:
        hits = [1 if _stem(re.sub(r"[^a-z0-9]", "", w.lower())) in stems else 0 for w in words]
        window = sum(hits[:n_words])
        best = window
        for i in range(1, len(words) - n_words + 1):
            window += hits[i + n_words - 1] - hits[i - 1]
            if window > best:
                best, start = window, i
    end = start + n_words
    return ("... " if start > 0 else "") + " ".join(words[start:end]) + (" ..." if end < len(words) else "")


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
    if query is not None and not isinstance(query, str):
        return {"ok": False, "tool": "search_literature", "error": f"query must be a string, got {type(query).__name__}"}
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
        return {"ok": False, "tool": "search_literature", "error_code": "index_unavailable", "error": str(exc)}

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
            "snippet": _snippet(row.get("text", ""), n_words, query),
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
