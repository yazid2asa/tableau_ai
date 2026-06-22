"""
rag_knowledge.py — RAG layer v2 for Text-to-Viz Agent.

Two ChromaDB collections (persistent — survive restarts):
  1. golden_examples     — curated Superstore Q/A pairs (seeded by rag_ingestion.py)
  2. generation_examples — high-scoring past generations, auto-stored after judge scoring

Hybrid retrieval: BM25 keyword search + ChromaDB cosine similarity, fused via
Reciprocal Rank Fusion (RRF).  Multilingual embedding model for FR/EN support.

tableau_rules collection removed — the LLM already knows Tableau better than
scraped doc chunks, and the static SYSTEM_PROMPT covers the essentials.
"""

import json
import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Module-level state — initialized by init_rag()
_client = None
_examples_col = None       # generation_examples
_golden_col = None         # golden_examples
_initialized = False

# BM25 in-memory index — rebuilt at startup from golden_examples + generation_examples
_bm25_index = None         # rank_bm25.BM25Okapi instance
_bm25_docs: list[dict] = []  # parallel list: [{question, viz_intent_json, judge_score, id, source}]


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _build_bm25_index() -> None:
    """Rebuild in-memory BM25 index from both example collections."""
    global _bm25_index, _bm25_docs

    _bm25_docs = []
    corpus: list[list[str]] = []

    for col, source_label in ((_golden_col, "golden"), (_examples_col, "generation")):
        if col is None or col.count() == 0:
            continue
        try:
            data = col.get(include=["documents", "metadatas"])
            if data["documents"] and data["metadatas"]:
                for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                    _bm25_docs.append({
                        "id": doc_id,
                        "question": doc,
                        "viz_intent_json": meta.get("viz_intent_json", "{}"),
                        "judge_score": float(meta.get("judge_score", 0)),
                        "datasource_name": meta.get("datasource_name", ""),
                        "source": source_label,
                    })
                    corpus.append(_tokenize(doc))
        except Exception as exc:
            logger.warning("rag: BM25 load from %s failed — %s", source_label, exc)

    if not corpus:
        _bm25_index = None
        return

    try:
        from rank_bm25 import BM25Okapi
        _bm25_index = BM25Okapi(corpus)
        logger.info("rag: BM25 index built — %d documents", len(corpus))
    except ImportError:
        logger.warning("rag: rank_bm25 not installed — BM25 disabled, vector-only retrieval")
        _bm25_index = None
    except Exception as exc:
        logger.warning("rag: BM25 index build failed — %s", exc)
        _bm25_index = None


# ---------------------------------------------------------------------------
# Embedding model change detection
# ---------------------------------------------------------------------------

def reset_and_reseed_if_model_changed(collection, current_model: str) -> bool:
    """Check if the embedding model has changed since the collection was last seeded.

    Stores model name in collection metadata.  If model differs from current_model,
    deletes all documents so the caller can re-seed with new embeddings.

    Returns True if a reset was performed (collection is now empty).
    """
    if collection is None:
        return False

    try:
        col_meta = collection.metadata or {}
        stored_model = col_meta.get("embedding_model")

        if stored_model == current_model:
            return False

        # Model changed (or first run with model tracking)
        if collection.count() > 0 and stored_model is not None:
            logger.info(
                "rag: embedding model changed %s → %s — resetting collection %s",
                stored_model, current_model, collection.name,
            )
            # Delete all documents
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

        # Update metadata with current model (exclude hnsw:space — immutable after creation)
        new_meta = {k: v for k, v in col_meta.items() if not k.startswith("hnsw:")}
        new_meta["embedding_model"] = current_model
        collection.modify(metadata=new_meta)
        return True
    except Exception as exc:
        logger.warning("rag: model change detection failed — %s", exc)
        return False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _get_or_recreate_collection(client, name: str, embedding_fn):
    """Get or create a collection with the given embedding function.

    If an existing collection has a conflicting embedding function (e.g., migrating
    from default to sentence-transformer), delete and recreate it.
    """
    try:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )
    except ValueError as exc:
        if "conflict" in str(exc).lower():
            logger.info("rag: embedding function conflict for %s — recreating collection", name)
            client.delete_collection(name=name)
            return client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_fn,
            )
        raise


def init_rag() -> None:
    """Initialize persistent ChromaDB client and open/create example collections.

    Does NOT insert any documents — seeding is done by rag_ingestion.run_ingestion().
    Graceful no-op if chromadb is unavailable or RAG is disabled.
    """
    global _client, _examples_col, _golden_col, _initialized

    if not settings.rag_enabled:
        logger.info("rag: disabled via RAG_ENABLED=false")
        return

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        logger.warning("rag: chromadb not installed — RAG disabled")
        return

    try:
        settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_db_path))

        # Use configurable multilingual embedding model
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.rag_embedding_model,
        )

        _golden_col = _get_or_recreate_collection(
            _client, "golden_examples", embedding_fn,
        )
        _examples_col = _get_or_recreate_collection(
            _client, "generation_examples", embedding_fn,
        )

        # Check for embedding model change — reset if needed
        model = settings.rag_embedding_model
        reset_and_reseed_if_model_changed(_golden_col, model)
        reset_and_reseed_if_model_changed(_examples_col, model)

        _initialized = True
        logger.info(
            "rag: initialized (persistent, model=%s) — golden_examples=%d, generation_examples=%d",
            model, _golden_col.count(), _examples_col.count(),
        )
    except Exception as exc:
        logger.warning("rag: initialization failed — %s", exc)
        _client = None
        _examples_col = None
        _golden_col = None


def rebuild_bm25() -> None:
    """Public entry point to rebuild BM25 index after seeding."""
    if _initialized:
        _build_bm25_index()


# ---------------------------------------------------------------------------
# Collection accessors — used by rag_ingestion.py
# ---------------------------------------------------------------------------

def get_golden_examples_collection():
    """Return the golden_examples ChromaDB collection, or None if RAG not initialized."""
    return _golden_col


# ---------------------------------------------------------------------------
# Retrieval — hybrid BM25 + vector with RRF fusion
# ---------------------------------------------------------------------------

def retrieve_examples(question: str, top_k: int | None = None, datasource_name: str = "") -> list[dict]:
    """Retrieve similar past generations using hybrid BM25 + vector search.

    1. BM25 keyword search on stored questions → BM25 ranks
    2. ChromaDB cosine similarity search → vector ranks
    3. Reciprocal Rank Fusion: score_rrf = 1/(k+rank_bm25) + 1/(k+rank_vector), k=60
    4. Return top_k results sorted by RRF score descending

    Returns list of dicts: {"question": str, "viz_intent": dict, "judge_score": float}.
    """
    k = top_k or settings.rag_examples_top_k
    rrf_k = 60  # RRF constant

    # Collect candidates from both sources keyed by document id
    candidates: dict[str, dict] = {}

    # --- BM25 search ---
    bm25_ranks: dict[str, int] = {}
    if _bm25_index is not None and _bm25_docs:
        try:
            tokens = _tokenize(question)
            scores = _bm25_index.get_scores(tokens)
            # Rank by BM25 score descending — only include docs with score > 0
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            rank_counter = 0
            for idx in ranked_indices:
                if scores[idx] <= 0:
                    break  # scores are sorted descending, so all remaining are 0
                rank_counter += 1
                if rank_counter > k * 3:
                    break
                doc = _bm25_docs[idx]
                doc_id = doc["id"]
                bm25_ranks[doc_id] = rank_counter  # 1-based rank
                candidates[doc_id] = doc
        except Exception as exc:
            logger.warning("rag: BM25 search failed — %s", exc)

    # --- Vector search (ChromaDB) ---
    vector_ranks: dict[str, int] = {}
    for col, source_label in ((_golden_col, "golden"), (_examples_col, "generation")):
        if col is None or col.count() == 0:
            continue
        try:
            results = col.query(
                query_texts=[question],
                n_results=min(k * 3, col.count()),
                include=["documents", "metadatas", "distances"],
            )
            if results["documents"] and results["metadatas"]:
                for rank_offset, (doc_id, doc, meta) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                )):
                    # Assign global vector rank (golden first, then generation)
                    current_rank = len(vector_ranks) + 1
                    if doc_id not in vector_ranks:
                        vector_ranks[doc_id] = current_rank
                    if doc_id not in candidates:
                        candidates[doc_id] = {
                            "id": doc_id,
                            "question": doc,
                            "viz_intent_json": meta.get("viz_intent_json", "{}"),
                            "judge_score": float(meta.get("judge_score", 0)),
                            "datasource_name": meta.get("datasource_name", ""),
                            "source": source_label,
                        }
        except Exception as exc:
            logger.warning("rag: vector search from %s failed — %s", source_label, exc)

    if not candidates:
        return []

    # --- RRF fusion ---
    max_rank = len(candidates) + 1  # fallback rank for missing entries
    rrf_scores: list[tuple[str, float]] = []
    for doc_id in candidates:
        rank_bm25 = bm25_ranks.get(doc_id, max_rank)
        rank_vector = vector_ranks.get(doc_id, max_rank)
        score = 1.0 / (rrf_k + rank_bm25) + 1.0 / (rrf_k + rank_vector)
        rrf_scores.append((doc_id, score))

    # Boost examples from same datasource
    if datasource_name:
        for i, (doc_id, score) in enumerate(rrf_scores):
            cand = candidates[doc_id]
            stored_ds = cand.get("datasource_name", "")
            if stored_ds and stored_ds.lower() == datasource_name.lower():
                rrf_scores[i] = (doc_id, score * 1.5)  # 50% boost
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # Build final results
    results: list[dict] = []
    for doc_id, _ in rrf_scores[:k]:
        cand = candidates[doc_id]
        try:
            viz_intent = json.loads(cand["viz_intent_json"])
        except (json.JSONDecodeError, ValueError):
            viz_intent = {}
        results.append({
            "question": cand["question"],
            "viz_intent": viz_intent,
            "judge_score": cand["judge_score"],
        })

    return results


# ---------------------------------------------------------------------------
# Storage — auto-store high-scoring generations
# ---------------------------------------------------------------------------

def store_successful_generation(
    question: str,
    viz_intent_dict: dict,
    judge_score: float,
    trace_id: str,
    datasource_name: str = "",
) -> None:
    """Store a high-scoring generation as a few-shot example in generation_examples.

    Only stores if judge_score >= rag_store_threshold. Skips duplicates by trace_id.
    After storing, rebuilds BM25 index to include the new example.
    """
    if not _initialized or _examples_col is None:
        return

    if judge_score < settings.rag_store_threshold:
        return

    try:
        _examples_col.upsert(
            ids=[trace_id],
            documents=[question],
            metadatas=[{
                "viz_intent_json": json.dumps(viz_intent_dict, ensure_ascii=False),
                "judge_score": judge_score,
                "datasource_name": datasource_name,
            }],
        )
        logger.info(
            "rag: stored example trace_id=%s score=%.2f q=%r",
            trace_id, judge_score, question[:60],
        )
        # Rebuild BM25 to include new example
        _build_bm25_index()
    except Exception as exc:
        logger.debug("rag: store skipped — %s", exc)
