import re
from pydantic_settings import BaseSettings
from pathlib import Path

# Known Tableau sample datasources to auto-exclude
SAMPLE_DS_PATTERNS = [
    "superstore", "world indicators", "sample -", "sample_",
    "regional", "world bank", "earthquake", "global superstore",
    "coffee chain", "startupvcs",
]


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    model_id: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Groq provider
    groq_api_key: str = ""
    groq_model_id: str = "qwen3.6-27b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Google provider (PRIMARY — measured 2026-07-09: gemini-2.5-flash 20/20
    # on eval_intent vs mistral-large 16/20 / mistral-medium 13/20; the repo
    # rule adopts a new primary only on a measured ≥).
    google_api_key: str = ""
    google_model_id: str = "gemini-2.5-flash"

    # Mistral provider (La Plateforme, OpenAI-compatible endpoint) — the paid,
    # reliable FALLBACK #1 behind Google (kicks in on Gemini free-tier 429s,
    # ahead of the free OpenRouter model). mistral-large-latest measured 16/20
    # (> medium's 13/20) — good enough for burst traffic, not for primary.
    mistral_api_key: str = ""
    mistral_model_id: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"

    # Provider selection: "google" (primary) | "mistral" | "openrouter" | "groq" (legacy).
    # Fallback chains (transient errors / missing key — see llm.py):
    #   google → mistral → openrouter ; mistral → google → openrouter ; groq → openrouter
    llm_provider: str = "google"

    # Judge provider (can differ from main LLM)
    judge_provider: str = "google"
    judge_model_id: str = "gemini-2.5-flash"

    database_url: str = "sqlite+aiosqlite:///./data/texttoviz.db"
    output_dir: Path = Path("output")
    chroma_db_path: Path = Path("data/chromadb")

    # LLM-as-a-Judge
    judge_threshold: float = 0.75
    judge_max_retries: int = 1

    # RAG — ChromaDB knowledge retrieval
    rag_enabled: bool = True
    rag_store_threshold: float = 0.85   # min judge score to store as few-shot example
    rag_knowledge_top_k: int = 3        # how many knowledge chunks to retrieve (legacy, unused)
    rag_examples_top_k: int = 3         # how many few-shot examples to retrieve
    rag_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Tableau Server settings
    tableau_server_url: str = ""          # e.g. "https://tableau.company.com"
    tableau_pat_name: str = ""            # Personal Access Token name
    tableau_pat_secret: str = ""          # Personal Access Token secret
    tableau_site_id: str = ""             # Site content URL (empty string = Default site)
    tableau_default_project_id: str = ""  # Project LUID to publish workbooks into
    tableau_datasource_filter: str = ""   # Optional comma-separated datasource names/LUIDs to limit schema fetch

    # LangFuse observability (self-hosted)
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
