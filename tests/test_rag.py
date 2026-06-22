"""
tests/test_rag.py — Tests for hybrid RAG retrieval (BM25 + vector with RRF).
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers — build a minimal in-process RAG environment
# ---------------------------------------------------------------------------

def _seed_golden_examples(golden_col):
    """Seed a subset of golden examples into the collection for testing."""
    examples = [
        {
            "id": "golden_heatmap",
            "question": "Show a heatmap of sales by category and region",
            "viz_intent": {
                "viz_type": "heatmap", "title": "Sales Heatmap",
                "x_field": "Category", "y_field": "Region",
            },
        },
        {
            "id": "golden_profit_margin",
            "question": "Show profit margin by category",
            "viz_intent": {
                "viz_type": "bar_chart", "title": "Profit Margin by Category",
                "x_field": "Category", "y_field": "Taux de Marge",
                "calculated_fields": [{"name": "Taux de Marge", "formula": "SUM([Profit])/SUM([Sales])"}],
            },
        },
        {
            "id": "golden_bar_sales",
            "question": "Show total sales by category as a bar chart sorted from highest to lowest",
            "viz_intent": {
                "viz_type": "bar_chart", "title": "Sales by Category",
                "x_field": "Category", "y_field": "Sales",
            },
        },
        {
            "id": "golden_treemap",
            "question": "Show the breakdown of sales by sub-category as a treemap",
            "viz_intent": {
                "viz_type": "treemap", "title": "Sales by Sub-Category",
                "x_field": "Sub_Category", "y_field": "Sales",
            },
        },
        {
            "id": "golden_scatter",
            "question": "Is there a correlation between sales and profit? Scatter plot please",
            "viz_intent": {
                "viz_type": "scatter", "title": "Sales vs Profit",
                "x_field": "Sales", "y_field": "Profit",
            },
        },
    ]

    ids = [ex["id"] for ex in examples]
    docs = [ex["question"] for ex in examples]
    metas = [
        {
            "viz_intent_json": json.dumps(ex["viz_intent"], ensure_ascii=False),
            "judge_score": 1.0,
        }
        for ex in examples
    ]
    golden_col.upsert(ids=ids, documents=docs, metadatas=metas)


@pytest.fixture
def rag_env(tmp_path):
    """Set up an isolated ChromaDB + BM25 RAG environment for testing."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    import rag_knowledge

    # Use a temp directory for ChromaDB
    client = chromadb.PersistentClient(path=str(tmp_path / "chromadb"))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
    )

    golden_col = client.get_or_create_collection(
        name="golden_examples",
        metadata={"hnsw:space": "cosine", "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"},
        embedding_function=embedding_fn,
    )
    examples_col = client.get_or_create_collection(
        name="generation_examples",
        metadata={"hnsw:space": "cosine", "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"},
        embedding_function=embedding_fn,
    )

    # Seed golden examples
    _seed_golden_examples(golden_col)

    # Inject into module state
    rag_knowledge._client = client
    rag_knowledge._golden_col = golden_col
    rag_knowledge._examples_col = examples_col
    rag_knowledge._initialized = True

    # Build BM25 index
    rag_knowledge._build_bm25_index()

    yield {
        "client": client,
        "golden_col": golden_col,
        "examples_col": examples_col,
    }

    # Cleanup module state
    rag_knowledge._client = None
    rag_knowledge._golden_col = None
    rag_knowledge._examples_col = None
    rag_knowledge._initialized = False
    rag_knowledge._bm25_index = None
    rag_knowledge._bm25_docs = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExactKeywordMatchViaBM25:
    """BM25 should boost exact keyword matches that semantic search misses."""

    def test_exact_keyword_match_via_bm25(self, rag_env):
        from rag_knowledge import retrieve_examples

        results = retrieve_examples("heatmap", top_k=3)
        assert len(results) > 0
        # The heatmap example must be in first position thanks to BM25 exact match
        assert "heatmap" in results[0]["question"].lower()


class TestFrenchQueryMatchesEnglish:
    """Multilingual embeddings should match French queries to English examples."""

    def test_french_query_matches_english_example(self, rag_env):
        from rag_knowledge import retrieve_examples

        results = retrieve_examples("taux de marge par categorie", top_k=3)
        assert len(results) > 0
        # Should match the profit margin golden example
        questions = [r["question"].lower() for r in results]
        matched = any("profit margin" in q or "taux de marge" in q for q in questions)
        assert matched, f"Expected profit margin match, got: {questions}"


class TestHybridBeatsSemantic:
    """Hybrid RRF should rank BM25 exact match above semantic-only results."""

    def test_hybrid_beats_semantic_alone(self, rag_env):
        from rag_knowledge import retrieve_examples

        # "treemap" is an exact keyword — BM25 should rank it #1 via RRF
        results = retrieve_examples("treemap", top_k=3)
        assert len(results) > 0
        assert "treemap" in results[0]["question"].lower()

        # Verify it outranks semantically similar but keyword-different results
        # (e.g. "breakdown of sales" might be semantically close to bar charts)
        assert results[0]["viz_intent"]["viz_type"] == "treemap"


class TestModelChangeTriggerReseed:
    """Changing the embedding model name should wipe and require re-seeding."""

    def test_model_change_triggers_reseed(self, rag_env):
        from rag_knowledge import reset_and_reseed_if_model_changed

        golden_col = rag_env["golden_col"]

        # Collection has 5 documents and model metadata
        assert golden_col.count() == 5

        # Simulate model change
        was_reset = reset_and_reseed_if_model_changed(golden_col, "some-new-model-v99")

        # Should have cleared the collection
        assert was_reset is True
        assert golden_col.count() == 0

        # Metadata should now reflect the new model
        meta = golden_col.metadata
        assert meta.get("embedding_model") == "some-new-model-v99"

        # Re-seed and verify it works
        _seed_golden_examples(golden_col)
        assert golden_col.count() == 5

        # Same model again should NOT trigger reset
        was_reset_again = reset_and_reseed_if_model_changed(golden_col, "some-new-model-v99")
        assert was_reset_again is False
        assert golden_col.count() == 5
