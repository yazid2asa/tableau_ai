"""
rag.py — RAG layer for Text-to-Viz Agent.

Three ChromaDB collections (persistent — survive restarts):
  1. tableau_rules     — Tableau knowledge chunks (seeded by rag_ingestion.py on first run)
  2. generation_examples — high-scoring past generations, auto-stored after judge scoring
  3. golden_examples   — manually curated Superstore Q/A pairs (seeded by rag_ingestion.py)

Graceful no-op if chromadb is unavailable or RAG is disabled.
"""

import json
import logging

from config import settings

logger = logging.getLogger(__name__)

# Module-level state — initialized by init_rag()
_client = None
_rules_col = None          # tableau_rules
_examples_col = None       # generation_examples
_golden_col = None         # golden_examples
_initialized = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_rag() -> None:
    """Initialize persistent ChromaDB client and open/create all collections.

    Does NOT insert any documents — seeding is done by rag_ingestion.run_ingestion().
    Graceful no-op if chromadb is unavailable or RAG is disabled.
    """
    global _client, _rules_col, _examples_col, _golden_col, _initialized

    if not settings.rag_enabled:
        logger.info("rag: disabled via RAG_ENABLED=false")
        return

    try:
        import chromadb
    except ImportError:
        logger.warning("rag: chromadb not installed — RAG disabled")
        return

    try:
        settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_db_path))

        _rules_col = _client.get_or_create_collection(
            name="tableau_rules",
            metadata={"hnsw:space": "cosine"},
        )
        _examples_col = _client.get_or_create_collection(
            name="generation_examples",
            metadata={"hnsw:space": "cosine"},
        )
        _golden_col = _client.get_or_create_collection(
            name="golden_examples",
            metadata={"hnsw:space": "cosine"},
        )

        _initialized = True
        logger.info(
            "rag: initialized (persistent) — tableau_rules=%d, generation_examples=%d, golden_examples=%d",
            _rules_col.count(),
            _examples_col.count(),
            _golden_col.count(),
        )
    except Exception as exc:
        logger.warning("rag: initialization failed — %s", exc)
        _client = None
        _rules_col = None
        _examples_col = None
        _golden_col = None


# ---------------------------------------------------------------------------
# Collection accessors — used by rag_ingestion.py
# ---------------------------------------------------------------------------

def get_tableau_rules_collection():
    """Return the tableau_rules ChromaDB collection, or None if RAG not initialized."""
    return _rules_col


def get_golden_examples_collection():
    """Return the golden_examples ChromaDB collection, or None if RAG not initialized."""
    return _golden_col


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_knowledge(question: str, top_k: int | None = None) -> list[str]:
    """Retrieve relevant Tableau knowledge chunks from tableau_rules.

    Returns a list of knowledge text strings, or empty list if RAG is unavailable.
    """
    if not _initialized or _rules_col is None or _rules_col.count() == 0:
        return []

    k = top_k or settings.rag_knowledge_top_k
    try:
        results = _rules_col.query(
            query_texts=[question],
            n_results=min(k, _rules_col.count()),
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as exc:
        logger.warning("rag: knowledge retrieval failed — %s", exc)
        return []


def retrieve_examples(question: str, top_k: int | None = None) -> list[dict]:
    """Retrieve similar past generations as few-shot examples.

    Queries both golden_examples (curated, higher quality) and generation_examples
    (auto-stored from live usage), merges results, and returns top-K.

    Returns a list of dicts: {"question": str, "viz_intent": dict, "judge_score": float}.
    """
    k = top_k or settings.rag_examples_top_k
    combined: list[dict] = []

    for col, label in ((_golden_col, "golden"), (_examples_col, "generation")):
        if col is None or col.count() == 0:
            continue
        try:
            results = col.query(
                query_texts=[question],
                n_results=min(k, col.count()),
                include=["documents", "metadatas", "distances"],
            )
            if results["documents"] and results["metadatas"]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    try:
                        combined.append({
                            "question": doc,
                            "viz_intent": json.loads(meta.get("viz_intent_json", "{}")),
                            "judge_score": float(meta.get("judge_score", 0)),
                            "_distance": float(dist),
                            "_source": label,
                        })
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as exc:
            logger.warning("rag: %s example retrieval failed — %s", label, exc)

    # Sort by cosine distance (lower = more similar) and return top-K
    combined.sort(key=lambda x: x["_distance"])
    result = combined[:k]
    # Strip internal keys before returning
    for item in result:
        item.pop("_distance", None)
        item.pop("_source", None)
    return result


# ---------------------------------------------------------------------------
# Storage — auto-store high-scoring generations
# ---------------------------------------------------------------------------

def store_successful_generation(
    question: str,
    viz_intent_dict: dict,
    judge_score: float,
    trace_id: str,
) -> None:
    """Store a high-scoring generation as a few-shot example in generation_examples.

    Only stores if judge_score >= rag_store_threshold. Skips duplicates by trace_id.
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
            }],
        )
        logger.info(
            "rag: stored example trace_id=%s score=%.2f q=%r",
            trace_id, judge_score, question[:60],
        )
    except Exception as exc:
        logger.debug("rag: store skipped — %s", exc)
