# Support Assistant

A deterministic, retrieval-grounded Zepto policy assistant using local embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

## Graded baseline

The default mode is `MOCK_LLM=1` (or the variable can be left unset). No LLM provider call is made. Embeddings and ChromaDB retrieval still run for policy questions.

## Files

- `docs/doc_01.txt` ... `docs/doc_08.txt` — the supplied Zepto policy corpus.
- `ingestion.py` — loads the eight documents, creates one short-document chunk per policy, embeds with `all-MiniLM-L6-v2`, and stores vectors in a cosine-distance ChromaDB collection.
- `prompt.py` — structured role/context/task/format/length prompt with a negative constraint and few-shot example.
- `graph.py` — TypedDict LangGraph state, intent routing, retrieval, answer generation, Pydantic validation, and optional real-LLM retry logic.
- `main.py` — FastAPI `POST /ask` endpoint.
- `examples.py` — two demonstration calls.
- `Dockerfile` — local container configuration.

## Architecture

```text
8 policy documents
      |
      v
  ingestion.py
      |
      v
all-MiniLM-L6-v2 embeddings
      |
      v
ChromaDB cosine collection
      |
      v
LangGraph: classify_intent
      |
      +----------------------+
      |                      |
policy_question        general_question
      |                      |
      v                      v
retrieve_and_answer    direct_answer
      |                      |
      +----------+-----------+
                 |
                 v
          Pydantic AnswerResponse
                 |
                 v
             FastAPI /ask
```

Ingestion creates embeddings and stores them in the `zepto_policy_documents` ChromaDB collection. For a policy question, `retrieve_and_answer` performs a real top-3 vector search using cosine distance and then the generation branch depends on `MOCK_LLM`. With mock mode, the answer is deterministic and uses the top retrieved chunk. With `MOCK_LLM=0`, the structured prompt is sent to the optional real LLM path and the JSON is validated with up to two additional corrective retries. General questions use `direct_answer`; mock mode returns the fixed canned response without retrieval.

## Setup

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m support_assistant.ingestion
python -m support_assistant.examples
```

The first embedding run may download `all-MiniLM-L6-v2` from its model registry. No LLM API key is needed for the graded mock mode.

## Run FastAPI

From the repository root:

```powershell
uvicorn support_assistant.main:app --reload --host 127.0.0.1 --port 8000
```

Then POST to `/ask`.

Example policy request:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"query":"How much is the delivery fee for an order below INR 149?"}'
```

Example general request:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"query":"What is the capital of India?"}'
```

## Recorded mock-mode examples

### Example 1 — policy question

Request:

```json
{"query":"How much is the delivery fee for an order below INR 149?"}
```

Recorded mock response shape:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
  "sources": [
    "doc_01_chunk_1",
    "doc_03_chunk_1",
    "doc_08_chunk_1"
  ],
  "confidence": 1.0
}
```

### Example 2 — general question

Request:

```json
{"query":"What is the capital of India?"}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Optional real-LLM mode

The graded baseline does not require this. To enable it, set `MOCK_LLM=0` and provide `GROQ_API_KEY`. The implementation uses an OpenAI-compatible Groq HTTP endpoint. Never commit secrets to the repository.

## Docker

From the repository root:

```powershell
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
```

The container serves `POST /ask` on port `7860`.
