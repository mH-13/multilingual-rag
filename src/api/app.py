# src/api/app.py

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from src.api.admin import router as admin_router

from src.rag.rag_pipeline import RAGPipeline
from fastapi.responses import RedirectResponse



app = FastAPI(
    title="Multilingual RAG API - Polyglot",
    version="0.1",
    description="Retrieve & generate answers over a Bangla/English corpus"
)

# Include admin endpoints
app.include_router(admin_router)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
  

# Initialize the RAG pipeline once
pipeline = RAGPipeline()

# Response model
class ContextItem(BaseModel):
    id: str
    score: float
    text: str

class AnswerResponse(BaseModel):
    answer: str
    contexts: list[ContextItem]

# Maximum allowed contexts per request
MAX_K = 10

@app.get("/ask", response_model=AnswerResponse)
def ask(
    q: str = Query(..., description="User question in Bangla or English"),
    k: int = Query(5, ge=1, le=MAX_K, description="Number of context chunks to retrieve")
):
    """
    Retrieve relevant chunks and generate an answer.
    - q: question text (Bangla or English)
    - k: how many context snippets to fetch (1–10)
    """
    try:
        result = pipeline(q, top_k=k)
        return result
    except Exception as e:
        # Return JSON error instead of plain 500
        raise HTTPException(status_code=500, detail=str(e))
