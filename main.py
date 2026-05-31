"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CANTILEVER — Sentiment Intelligence API  (main.py)  v2.0.0               ║
║   FastAPI · PyTorch LSTM · NLTK · Pydantic · OWASP-Hardened                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Security Controls                                                          ║
║    [1] CORS            — strict allow-list: localhost:8000 only             ║
║    [2] Input ceiling   — Pydantic Field(max_length=5000) → HTTP 422        ║
║    [3] XSS mitigation  — regex strips all HTML/script tags pre-tokenise    ║
║    [4] DoS mitigation  — hard payload ceiling prevents memory exhaustion   ║
║    [5] Resource guard  — torch.no_grad() on every inference call           ║
║    [6] Safe load       — map_location=cpu, weights_only=True               ║
║    [7] Security headers— X-Frame-Options, X-Content-Type-Options, etc.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import re
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import nltk
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
)
logger = logging.getLogger("cantilever")


# ══════════════════════════════════════════════════════════════════════════════
# § 1  MODEL ARCHITECTURE  (must exactly mirror the training notebook)
# ══════════════════════════════════════════════════════════════════════════════

class SentimentLSTM(nn.Module):
    """
    2-layer stacked LSTM sentiment classifier trained on IMDB (25k reviews).

    Exact architecture matching sentiment_lstm.pth checkpoint:
        Embedding(10001, 400)
        LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
        Dropout(p=0.5)
        Linear(256 → 1)
        Sigmoid()
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        drop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=drop_prob, batch_first=True,
        )
        self.dropout   = nn.Dropout(drop_prob)
        self.fc        = nn.Linear(hidden_dim, output_dim)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, ...]):
        embeds           = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out         = lstm_out[:, -1, :]   # final time-step only
        out              = self.dropout(lstm_out)
        out              = self.fc(out)
        return self.sigmoid(out), hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero-initialised (h_0, c_0) for a fresh sequence."""
        w = next(self.parameters()).data
        return (
            w.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            w.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )


# ══════════════════════════════════════════════════════════════════════════════
# § 2  APPLICATION STATE  (single instance, shared across all requests)
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    model:      SentimentLSTM | None = None
    vocab:      dict[str, int] | None = None
    stop_words: set[str] | None       = None
    lemmatizer: Any | None            = None
    device:     torch.device          = torch.device("cpu")
    ready:      bool                  = False   # set True after full boot


app_state = AppState()


# ══════════════════════════════════════════════════════════════════════════════
# § 3  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH    = os.path.join(BASE_DIR, "vocab.json")
WEIGHTS_PATH  = os.path.join(BASE_DIR, "sentiment_lstm.pth")

MODEL_NAME    = "sentiment_lstm_v1"
API_VERSION   = "v1"

# Hyper-parameters — must be identical to training
VOCAB_SIZE     = 10_000
VOCAB_SIZE_PAD = VOCAB_SIZE + 1   # index 0 reserved for padding
EMBEDDING_DIM  = 400
HIDDEN_DIM     = 256
OUTPUT_DIM     = 1
N_LAYERS       = 2
SEQ_LENGTH     = 200


# ══════════════════════════════════════════════════════════════════════════════
# § 4  STARTUP LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executed once on startup and once on shutdown.
    Downloads NLTK corpora, loads vocab.json, and maps model weights onto CPU.
    Sets app_state.ready = True only after all artefacts load successfully.
    """
    logger.info("⚙  Bootstrapping CANTILEVER API v2.0.0…")

    # NLTK corpora — downloaded once, cached to disk for subsequent runs
    logger.info("  ↳ Ensuring NLTK corpora (stopwords, wordnet)…")
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem   import WordNetLemmatizer

    app_state.stop_words = set(stopwords.words("english"))
    app_state.lemmatizer = WordNetLemmatizer()
    logger.info("  ✔  NLTK ready")

    # Vocabulary mapping — loaded into RAM once
    logger.info(f"  ↳ Loading vocabulary → {VOCAB_PATH}")
    with open(VOCAB_PATH, "r", encoding="utf-8") as fh:
        app_state.vocab = json.load(fh)
    logger.info(f"  ✔  Vocabulary loaded ({len(app_state.vocab):,} tokens)")

    # Model weights — reconstructed architecture, CPU-mapped, eval mode
    logger.info(f"  ↳ Loading model weights → {WEIGHTS_PATH}")
    model = SentimentLSTM(
        vocab_size    = VOCAB_SIZE_PAD,
        embedding_dim = EMBEDDING_DIM,
        hidden_dim    = HIDDEN_DIM,
        output_dim    = OUTPUT_DIM,
        n_layers      = N_LAYERS,
    )
    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location  = torch.device("cpu"),
        weights_only  = True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    app_state.model = model
    app_state.ready = True
    logger.info(f"  ✔  Model '{MODEL_NAME}' loaded — eval() mode active")
    logger.info("🚀  CANTILEVER API is LIVE — http://127.0.0.1:8000")

    yield   # ← application serves requests here

    logger.info("🛑  Shutting down CANTILEVER API")
    app_state.ready = False


# ══════════════════════════════════════════════════════════════════════════════
# § 5  FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

_DESCRIPTION = """
## CANTILEVER Sentiment Intelligence API

A production-grade REST API that serves a **PyTorch LSTM** sentiment classifier
trained on 50,000 IMDB movie reviews (25k train / 25k test), achieving
**86.61% accuracy** on the blind test set.

### Key Capabilities
- **Neural inference** — 2-layer stacked LSTM with 256-dim hidden state
- **NLP preprocessing** — HTML sanitisation, stopword removal, lemmatisation
- **OWASP-hardened** — CORS lockdown, payload ceiling, XSS neutralisation
- **Async-first** — non-blocking handlers, single-startup model caching

### Security Controls
| Control | Implementation |
|---|---|
| CORS | Strict local-origin allow-list only |
| DoS | 5,000-character input ceiling (HTTP 422 on breach) |
| XSS | Regex strips all HTML/script tags before tokenisation |
| Headers | X-Frame-Options · X-Content-Type-Options · Referrer-Policy |

### Versioning
Current stable: **`/api/v1/`**
"""

app = FastAPI(
    title       = "CANTILEVER Sentiment Intelligence API",
    description = _DESCRIPTION,
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
    contact     = {
        "name":  "CANTILEVER",
        "email": "api@cantilever.local",
    },
    license_info = {
        "name": "MIT",
    },
    openapi_tags = [
        {"name": "inference",  "description": "Sentiment analysis endpoints"},
        {"name": "health",     "description": "Liveness and readiness probes"},
        {"name": "system",     "description": "System metadata"},
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# § 6  MIDDLEWARE STACK
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a. CORS — strict allow-list (OWASP A05) ─────────────────────────────────
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = False,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Content-Type", "X-Request-ID"],
)


# ── 6b. Security Headers + Request Timing + Structured Logging ───────────────

class RequestMiddleware(BaseHTTPMiddleware):
    """
    Single-pass middleware that does three things per request:
      1. Injects a unique X-Request-ID header (generated if not supplied).
      2. Measures total wall-clock request duration.
      3. Writes a structured log line: method, path, status, latency, request_id.
      4. Attaches security response headers to every outgoing response.
    """

    SECURITY_HEADERS = {
        "X-Frame-Options":        "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy":        "strict-origin-when-cross-origin",
        "Cache-Control":          "no-store, no-cache, must-revalidate",
        "X-Powered-By":           "CANTILEVER/2.0",
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or forward a unique request identifier
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]

        # Store on request state so endpoint handlers can read it
        request.state.request_id = request_id

        # Time the full request cycle
        t_start   = time.perf_counter()
        response  = await call_next(request)
        latency   = (time.perf_counter() - t_start) * 1_000   # milliseconds

        # Structured access log
        logger.info(
            "%s %s → %d  |  %.1fms  |  req=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency,
            request_id,
        )

        # Attach security + meta headers to every response
        response.headers["X-Request-ID"]    = request_id
        response.headers["X-Latency-Ms"]    = f"{latency:.1f}"
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        return response


app.add_middleware(RequestMiddleware)


# ══════════════════════════════════════════════════════════════════════════════
# § 7  GLOBAL EXCEPTION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override FastAPI's default 422 response to return a clean, consistent
    JSON error body that never leaks internal field paths or raw input data.
    """
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code = 422,
        content     = {
            "error":      "Validation failed",
            "details":    errors,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for any unhandled exception. Returns a generic 500 body
    without exposing stack traces, internal paths, or model state.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":      "Internal server error",
            "message":    "An unexpected error occurred. Please try again.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 8  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ReviewPayload(BaseModel):
    """
    Request body for POST /api/v1/analyze.

    The 5,000-character ceiling is the primary DoS mitigation control.
    FastAPI returns HTTP 422 automatically on violation — the handler
    function is never invoked for out-of-bounds payloads.
    """

    review: str = Field(
        ...,
        min_length  = 1,
        max_length  = 5_000,
        title       = "Movie Review",
        description = "Raw movie review text. Max 5,000 characters.",
        examples    = ["An absolute masterpiece — the performances were breathtaking."],
    )


class AnalysisResponse(BaseModel):
    """Structured response returned by POST /api/v1/analyze."""

    sentiment:  str   = Field(description="'Positive' or 'Negative'")
    confidence: float = Field(description="Model confidence in [0.5, 1.0]")
    latency_ms: float = Field(description="Inference wall-clock time in milliseconds")
    request_id: str   = Field(description="Unique identifier for this request")
    model:      str   = Field(description="Model identifier string")

    model_config = {"json_schema_extra": {
        "example": {
            "sentiment":  "Positive",
            "confidence": 0.9142,
            "latency_ms": 38.7,
            "request_id": "a3f9c12b4d8e",
            "model":      "sentiment_lstm_v1",
        }
    }}


# ══════════════════════════════════════════════════════════════════════════════
# § 9  PREPROCESSING PIPELINE  (unchanged from v1 — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

_RE_HTML  = re.compile(r"<[^>]+>")     # strip all HTML / script tags (XSS)
_RE_ALPHA = re.compile(r"[^a-zA-Z\s]") # keep only letters + whitespace


def preprocess(raw_text: str) -> torch.Tensor:
    """
    Transform a raw review string into a padded LongTensor ready for inference.

    Pipeline (mirrors training notebook exactly — do not alter):
      1. Strip HTML/script tags        (XSS neutralisation)
      2. Remove non-alpha chars        (normalisation)
      3. Lowercase
      4. Tokenise by whitespace
      5. Remove NLTK English stopwords
      6. Lemmatise each token
      7. Map tokens → vocab indices    (OOV tokens silently skipped)
      8. Pre-pad / truncate → 200 ints (zeros at front, content at end)
      9. Return LongTensor shape (1, 200)
    """
    text   = _RE_HTML.sub(" ", raw_text)
    text   = _RE_ALPHA.sub("", text).lower()
    tokens = text.split()

    tokens = [
        app_state.lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in app_state.stop_words
    ]

    encoded: list[int] = [
        app_state.vocab[tok]
        for tok in tokens
        if tok in app_state.vocab
    ]

    features = np.zeros(SEQ_LENGTH, dtype=np.int64)
    if encoded:
        seq              = np.array(encoded[:SEQ_LENGTH], dtype=np.int64)
        features[-len(seq):] = seq

    return torch.from_numpy(features).unsqueeze(0)   # (1, SEQ_LENGTH)


# ══════════════════════════════════════════════════════════════════════════════
# § 10  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 10a. Health / Readiness probes ───────────────────────────────────────────

@app.get(
    "/health",
    tags               = ["health"],
    summary            = "Liveness probe",
    response_description = "Server is alive",
    include_in_schema  = True,
)
async def health() -> JSONResponse:
    """
    GET /health

    Kubernetes / load-balancer liveness probe.
    Returns 200 as long as the process is running — does not check model state.
    """
    return JSONResponse(content={"status": "ok", "service": "cantilever-api"})


@app.get(
    "/ready",
    tags               = ["health"],
    summary            = "Readiness probe",
    response_description = "All artefacts loaded and ready to serve inference",
    include_in_schema  = True,
)
async def ready() -> JSONResponse:
    """
    GET /ready

    Readiness probe — returns 200 only when vocab, model, and NLTK are
    fully loaded. Returns 503 during startup or after a failed boot.
    """
    if app_state.ready:
        return JSONResponse(content={
            "status":    "ready",
            "model":     MODEL_NAME,
            "vocab_size": len(app_state.vocab) if app_state.vocab else 0,
        })
    return JSONResponse(
        status_code = 503,
        content     = {"status": "not_ready", "message": "Artefacts still loading"},
    )


# ── 10b. System metadata ──────────────────────────────────────────────────────

@app.get(
    "/api/v1/info",
    tags               = ["system"],
    summary            = "API metadata",
    response_description = "Version, model info, and capability summary",
    include_in_schema  = True,
)
async def api_info() -> JSONResponse:
    """GET /api/v1/info — Returns static API and model metadata."""
    return JSONResponse(content={
        "api_version":   API_VERSION,
        "app_version":   "2.0.0",
        "model":         MODEL_NAME,
        "architecture":  "2-layer LSTM (emb=400, hidden=256)",
        "vocab_size":    VOCAB_SIZE,
        "seq_length":    SEQ_LENGTH,
        "test_accuracy": "86.61%",
        "dataset":       "Stanford IMDB (50k reviews)",
    })


# ── 10c. Inference endpoint ───────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze",
    tags                 = ["inference"],
    summary              = "Analyse review sentiment",
    response_description = "Sentiment label, confidence score, latency, and request metadata",
    response_model       = AnalysisResponse,
)
async def analyze(payload: ReviewPayload, request: Request) -> JSONResponse:
    """
    **POST /api/v1/analyze**

    Submit a raw movie review (plain text, ≤ 5,000 chars) and receive
    an instant sentiment verdict from the LSTM model.

    ### Processing Steps
    1. Pydantic validates payload length before this handler runs
    2. XSS sanitisation strips all HTML/script structures
    3. NLTK pipeline: lowercase → stopword removal → lemmatisation
    4. Vocab mapping → 200-token padded LongTensor
    5. `torch.no_grad()` LSTM forward pass
    6. Sigmoid output → label + confidence

    ### Security
    - Input ceiling: 5,000 chars (HTTP 422 on breach)
    - XSS-clean: all HTML stripped before tokenisation
    - No stack traces in any error response
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    # Preprocess: sanitise + encode + pad → (1, 200) LongTensor
    input_tensor = preprocess(payload.review).to(app_state.device)

    # Fresh hidden state for this single-sample inference
    hidden = app_state.model.init_hidden(1, app_state.device)

    # Timed inference — no_grad() disables gradient tracking (memory efficient)
    t0 = time.perf_counter()
    with torch.no_grad():
        output, _ = app_state.model(input_tensor, hidden)
    latency_ms = (time.perf_counter() - t0) * 1_000

    probability: float = output.item()
    sentiment          = "Positive" if probability >= 0.5 else "Negative"
    confidence         = max(probability, 1.0 - probability)

    logger.info(
        "Inference — sentiment=%s  conf=%.4f  latency=%.1fms  req=%s",
        sentiment, confidence, latency_ms, request_id,
    )

    return JSONResponse(content={
        "sentiment":  sentiment,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "request_id": request_id,
        "model":      MODEL_NAME,
    })


# ── 10d. Frontend — serves index.html ────────────────────────────────────────

@app.get(
    "/",
    include_in_schema = False,
)
async def serve_frontend() -> FileResponse:
    """GET / — Serves the zero-build React dashboard (index.html)."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# § 11  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host      = "127.0.0.1",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
