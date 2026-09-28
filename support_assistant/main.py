from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_assistant.graph import AnswerResponse, run_assistant


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "endpoint": "POST /ask",
        "mock_llm": True,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    try:
        return run_assistant(request.query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
