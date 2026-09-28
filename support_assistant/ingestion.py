from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from support_assistant.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL_NAME,
)


def load_documents() -> List[Dict[str, str]]:
    """Load the eight policy documents from the local corpus."""
    documents = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        document_id = path.stem
        documents.append(
            {
                "id": document_id,
                "text": text,
                "source": path.name,
            }
        )

    if len(documents) != 8:
        raise RuntimeError(
            f"Expected 8 policy documents, found {len(documents)} in {DOCS_DIR}."
        )

    return documents


def chunk_document(text: str) -> List[str]:
    """Use one chunk per policy document because each supplied document is short."""
    return [text]


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_store() -> None:
    """Embed all policy chunks and store them in ChromaDB."""
    documents = load_documents()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = get_collection()

    ids = []
    chunks = []
    metadatas = []

    for document in documents:
        for chunk_index, chunk in enumerate(chunk_document(document["text"])):
            chunk_id = f"{document['id']}_chunk_{chunk_index + 1}"
            ids.append(chunk_id)
            chunks.append(chunk)
            metadatas.append(
                {
                    "document_id": document["id"],
                    "source": document["source"],
                    "chunk_index": chunk_index,
                }
            )

    # Upsert is idempotent. Existing generated chunks are replaced rather than duplicated.
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("=" * 60)
    print("SUPPORT ASSISTANT INGESTION COMPLETED")
    print("=" * 60)
    print(f"Documents loaded : {len(documents)}")
    print(f"Chunks embedded  : {len(chunks)}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"ChromaDB path    : {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()
