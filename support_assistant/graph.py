import json
import os
from typing import List, TypedDict

import chromadb
import requests
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from support_assistant.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    MOCK_LLM,
    POLICY_KEYWORDS,
)
from support_assistant.ingestion import build_vector_store, get_collection
from support_assistant.prompt import build_prompt


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    retrieved_sources: List[str]
    answer: str
    sources: List[str]
    confidence: float
    error: str


_embedding_model = None
_collection = None


def is_mock_mode() -> bool:
    return os.getenv("MOCK_LLM", MOCK_LLM) != "0"


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_policy_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()

        if _collection.count() < 8:
            build_vector_store()
            _collection = get_collection()

    return _collection


def call_real_llm(prompt: str) -> str:
    """Optional Groq-compatible real-LLM path used only when MOCK_LLM=0."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY for the optional real-LLM path."
        )

    model = os.getenv(
        "GROQ_MODEL",
        "llama-3.1-8b-instant",
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def classify_intent(state: SupportState) -> SupportState:
    query = state["query"]
    lowered = query.lower()

    if is_mock_mode():
        intent = (
            "policy_question"
            if any(keyword in lowered for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        prompt = f"""
Classify this user query as exactly one label: policy_question or general_question.
Return JSON: {{"intent": "policy_question"}} or {{"intent": "general_question"}}.
Query: {query}
""".strip()
        raw = call_real_llm(prompt)
        try:
            intent = json.loads(raw)["intent"]
            if intent not in {"policy_question", "general_question"}:
                raise ValueError("Invalid intent")
        except Exception as exc:
            raise RuntimeError(f"Intent classification failed: {exc}") from exc

    return {"intent": intent}


def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]
    model = get_embedding_model()
    collection = get_policy_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved_ids = []
    retrieved_sources = []

    for metadata in metadatas:
        retrieved_ids.append(metadata["document_id"])
        retrieved_sources.append(
            f"{metadata['document_id']}_chunk_{metadata['chunk_index'] + 1}"
        )

    if not documents:
        raise RuntimeError("No policy chunks were retrieved from ChromaDB.")

    # ChromaDB was configured for cosine distance. Higher similarity = lower distance.
    cosine_similarities = [1 - float(distance) for distance in distances]

    print("\nRETRIEVAL RESULTS")
    print("-" * 40)
    for index, (source_id, similarity) in enumerate(
        zip(retrieved_sources, cosine_similarities),
        start=1,
    ):
        print(
            f"Top {index}: {source_id} | cosine similarity = {similarity:.4f}"
        )

    top_chunk = documents[0]
    top_chunk_snippet = top_chunk[:200]

    if is_mock_mode():
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        context_parts = []
        for chunk_id, document in zip(retrieved_sources, documents):
            context_parts.append(
                f"[{chunk_id}] {document}"
            )

        prompt = build_prompt(
            query=query,
            context="\n\n".join(context_parts),
        )

        answer, confidence = generate_validated_real_answer(
            prompt=prompt,
            default_sources=retrieved_sources,
        )

    return {
        "retrieved_chunks": documents,
        "retrieved_ids": retrieved_ids,
        "retrieved_sources": retrieved_sources,
        "answer": answer,
        "sources": retrieved_sources,
        "confidence": confidence,
    }


def direct_answer(state: SupportState) -> SupportState:
    query = state["query"]

    if is_mock_mode():
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        prompt = build_prompt(
            query=query,
            context="No policy retrieval is required because this is a general question.",
        )
        answer, confidence = generate_validated_real_answer(
            prompt=prompt,
            default_sources=[],
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": confidence,
    }


def generate_validated_real_answer(
    prompt: str,
    default_sources: List[str],
):
    """Validate real-LLM JSON and retry up to 2 additional times if needed."""
    corrective_prompt = prompt
    last_error = None

    for attempt in range(3):
        try:
            raw = call_real_llm(corrective_prompt)
            parsed = json.loads(raw)
            response = AnswerResponse.model_validate(parsed)

            sources = response.sources or default_sources
            response = AnswerResponse(
                answer=response.answer,
                sources=sources,
                confidence=response.confidence,
            )

            return response.answer, response.confidence

        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            last_error = str(exc)
            corrective_prompt = (
                prompt
                + "\n\nCORRECTION: Your previous response was invalid. "
                "Return ONLY valid JSON with fields answer, sources, confidence. "
                "confidence must be between 0 and 1."
            )

    return (
        f"[REAL_LLM_VALIDATION_ERROR] Could not validate model output after 3 attempts: {last_error}",
        0.0,
    )


def route_after_classification(state: SupportState) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


support_graph = build_graph()


def run_assistant(query: str) -> AnswerResponse:
    state = support_graph.invoke(
        {
            "query": query,
        }
    )

    return AnswerResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=float(state.get("confidence", 1.0)),
    )
