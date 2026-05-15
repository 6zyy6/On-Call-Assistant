from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .agent import OnCallChatAgent
from .schemas import ChatRequest, DocumentPayload
from .search import DocumentStore
from .semantic import SemanticSearchEngine
from .text import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"
TEMPLATE_DIR = PROJECT_ROOT / "templates"

load_env_file(ENV_FILE)

store = DocumentStore()
semantic_engine = SemanticSearchEngine()
chat_agent = OnCallChatAgent(DATA_DIR, store, semantic_engine)


@lru_cache(maxsize=8)
def load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def load_seed_documents() -> None:
    if not DATA_DIR.exists():
        try:
            semantic_engine.rebuild(store.list_documents())
        except RuntimeError as exc:
            print(f"Semantic index skipped: {exc}")
        return

    for html_file in sorted(DATA_DIR.glob("*.html")):
        store.upsert(html_file.stem, html_file.read_text(encoding="utf-8"))

    try:
        semantic_engine.rebuild(store.list_documents())
    except RuntimeError as exc:
        print(f"Semantic index skipped: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_seed_documents()
    yield


app = FastAPI(
    title="On-Call Assistant Search Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/documents")
def upsert_document(payload: DocumentPayload) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = (DATA_DIR / f"{payload.id}.html").resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in target.parents:
        raise HTTPException(status_code=400, detail="Document id is invalid")
    target.write_text(payload.html, encoding="utf-8")
    document = store.upsert(payload.id, payload.html)
    try:
        semantic_engine.rebuild(store.list_documents())
    except RuntimeError as exc:
        print(f"Semantic index skipped after document update: {exc}")
    return JSONResponse(status_code=201, content={"id": document.id, "title": document.title})


@app.get("/v1/search")
def search_documents(q: str = Query(..., min_length=1)) -> dict:
    return {"query": q, "results": store.search(q)}


@app.get("/v2/search")
def semantic_search_documents(q: str = Query(..., min_length=1)) -> dict:
    try:
        results = semantic_engine.search(q)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": q, "results": results}


@app.post("/v3/chat")
def chat_with_agent(payload: ChatRequest) -> dict:
    try:
        return chat_agent.chat(payload.message, payload.history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v3/chat/stream")
def stream_chat_with_agent(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_agent.stream_chat(payload.message, payload.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1", response_class=HTMLResponse)
def search_page() -> str:
    return load_template("v1.html")


@app.get("/v2", response_class=HTMLResponse)
def semantic_search_page() -> str:
    return load_template("v2.html")


@app.get("/v3", response_class=HTMLResponse)
def chat_page() -> str:
    return load_template("v3.html")


@app.get("/")
def root() -> dict:
    return {"message": "Visit /v1 for keyword search, /v2 for semantic search, and /v3 for chat."}
