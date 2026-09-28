from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MOCK_LLM = os.getenv("MOCK_LLM", "1")

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]
