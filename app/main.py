"""FastAPI entrypoint for the PAYBACK Lightweight Assistant."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import __version__, llm
from .agent import handle
from .catalog import PARTNERS, load_catalog
from .retrieval import BM25Index
from .schemas import AssistRequest, AssistResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assistant")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog = load_catalog()
    _state["index"] = BM25Index(catalog)
    log.info("Indexed %d products across %d partners", len(catalog), len(PARTNERS))
    yield
    _state.clear()


app = FastAPI(
    title="PAYBACK Lightweight Assistant",
    version=__version__,
    description="Multilingual intent detection + cross-partner product retrieval.",
    lifespan=lifespan,
)


_UI_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return _UI_HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "products": len(_state["index"].products),
        "llm": llm.model_name() if llm.available() else "disabled (rules only)",
    }


@app.get("/partners")
def partners():
    return {key: {"label": meta["label"], "profile": meta["profile"], "items": len(meta["items"])}
            for key, meta in PARTNERS.items()}


@app.post("/assist", response_model=AssistResponse)
def assist(request: AssistRequest) -> AssistResponse:
    try:
        response = handle(request.query, _state["index"], request.max_results)
    except Exception:  # defensive: never leak internals to the client
        log.exception("assist failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail="internal error")
    log.info("intent=%s lang=%s partner=%s hits=%d %.2fms",
             response.intent, response.language, response.partner_filter,
             len(response.products), response.latency_ms)
    return response
