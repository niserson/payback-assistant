"""FastAPI entrypoint for the PAYBACK Lightweight Assistant."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import __version__, llm
from .agent import handle
from .catalog import PARTNERS, load_catalog, taxonomy_summary
from .retrieval import BM25Index
from .schemas import AssistRequest, AssistResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assistant")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog = load_catalog()
    _state["index"] = BM25Index(catalog)
    from .intent_model import get_model
    model = get_model()  # warm the learned classifier (trains on first-ever start)
    log.info("Indexed %d products across %d partners; classifier ready (%d function words)",
             len(catalog), len(PARTNERS), len(model.stopwords))
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
    return {key: {"label": meta["label"], "profile": meta["profile"],
                  "categories": len(meta["taxonomy"]),
                  "items": sum(len(v) for v in meta["taxonomy"].values())}
            for key, meta in PARTNERS.items()}


@app.get("/taxonomy")
def taxonomy():
    """Per-partner category trees with product counts."""
    return taxonomy_summary()


def _static_page(filename: str) -> str:
    path = Path(__file__).parent / "static" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not generated yet")
    return path.read_text(encoding="utf-8")


@app.get("/architecture", response_class=HTMLResponse, include_in_schema=False)
def architecture() -> str:
    return _static_page("architecture.html")


@app.get("/taxonomy-tree", response_class=HTMLResponse, include_in_schema=False)
def taxonomy_tree() -> str:
    from .taxonomy_svg import render_page
    return render_page()


@app.get("/demo-notebook", response_class=HTMLResponse, include_in_schema=False)
def demo_notebook() -> str:
    return _static_page("demo_notebook.html")


@app.get("/coldstart-notebook", response_class=HTMLResponse, include_in_schema=False)
def coldstart_notebook() -> str:
    return _static_page("coldstart_context.html")


@app.get("/evaluation-notebook", response_class=HTMLResponse, include_in_schema=False)
def evaluation_notebook() -> str:
    return _static_page("evaluation_notebook.html")


@app.get("/performance-report", response_class=HTMLResponse, include_in_schema=False)
def performance_report() -> str:
    return _static_page("performance_report.html")


@app.get("/optimization-report", response_class=HTMLResponse, include_in_schema=False)
def optimization_report() -> str:
    return _static_page("optimization_report.html")


@app.post("/assist", response_model=AssistResponse)
def assist(request: AssistRequest) -> AssistResponse:
    try:
        response = handle(request.query, _state["index"], request.max_results,
                          request.user_id, request.llm_mode, request.model)
    except Exception:  # defensive: never leak internals to the client
        log.exception("assist failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail="internal error")
    log.info("intent=%s lang=%s partner=%s hits=%d %.2fms",
             response.intent, response.language, response.partner_filter,
             len(response.products), response.latency_ms)
    return response
