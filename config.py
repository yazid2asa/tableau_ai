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
    groq_model_id: str = "mixtral-8x7b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Google provider
    google_api_key: str = ""
    google_model_id: str = "gemini-3.1-flash-lite"

    # Provider selection: "groq" | "openrouter" | "google"
    llm_provider: str = "google"

    # Judge provider (can differ from main LLM)
    judge_provider: str = "google"
    judge_model_id: str = "gemini-3.1-flash-lite"

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
