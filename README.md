# AI Legislative Analyser

An end-to-end system that scrapes Indian legislative bills from PRS India, stores them in Postgres, processes PDFs into dense embeddings, and exposes a modern FastAPI-powered web UI with an AI assistant for summaries and Q&A.

The stack includes:

- **FastAPI** backend (REST + static frontend)
- **PostgreSQL + SQLAlchemy + pgvector** for bill/chunk storage and semantic search
- **SentenceTransformer (all-MiniLM-L6-v2)** for embeddings
- **Groq LLM (llama-3.1-8b-instant)** for chunk compression, summaries, and Q&A
- **BeautifulSoup + requests** for scraping PRS India
- **Static HTML/CSS/JS** dashboard UI with a carousel and chatbot

---

## Project structure

- `main.py` – FastAPI app, startup bootstrap, endpoints (`/`, `/dashboard`, `/fetch-bill/{query}`, `/ask`).
- `database.py` – SQLAlchemy engine + session + base model.
- `models.py` – ORM models:
  - `Bill`: title, `pdf_url`, local file path, processed flag, `title_embedding`, `summary`.
  - `Chunk`: `bill_id`, `original_text`, `compressed_text`, `embedding`.
- `AI/`
  - `chunker.py` – Cleans raw PDF text and splits into overlapping chunks.
  - `compressor.py` – Uses Groq to compress original chunks into shorter, query-friendly bullets.
  - `extractor.py` – Extracts text from PDFs (via PyMuPDF) and returns original + compressed chunks.
  - `embedder.py` – Lazily loads SentenceTransformer and returns 384‑dimensional embeddings as Python lists.
  - `summarizer.py` – Uses Groq to turn compressed chunks into a public-facing summary for the dashboard.
  - `qa.py` – Uses Groq to answer questions constrained to a set of context chunks.
- `scraper/`
  - `fetch_links.py` – Scrapes PRS India `/billtrack` listing and individual bill pages to find PDFs.
  - `downloader.py` – Downloads PDFs into `data/pdfs`, with simple caching by filename.
  - `scheduler.py` –
    - `process_new_pdfs(initial=True|False)` seeds/updates the `bills` table with new PDF links and title embeddings.
    - `run_scheduler()` runs in a background loop to periodically ingest new links.
- `data/pdfs/` – Local cache of downloaded bill PDFs.
- `static/`
  - `index.html` – Single-page dashboard UI (carousel + summary cards + chatbot).
  - `styles.css` – Dark, responsive layout and chat styling.
  - `script.js` – Frontend logic for fetching bills, cycling the carousel, populating summary cards, and driving the chat.
- `requirements.txt` – Python dependencies.

---

## Features

- **Automated scraping & ingestion**
  - Scrapes bill pages from PRS India and discovers linked PDFs.
  - Inserts new bills into Postgres with a title and optional title embedding.
  - Downloads PDFs on demand and caches them under `data/pdfs/`.

- **PDF processing & embeddings**
  - Extracts text using PyMuPDF.
  - Chunks text (`chunker.py`) with word overlap and sentence-aware boundaries.
  - Compresses chunks via Groq into dense bullet-point summaries.
  - Computes embeddings for each compressed chunk with `all-MiniLM-L6-v2` stored via pgvector.

- **Summaries & dashboard**
  - Generates a concise, public summary per bill (`summarizer.py`).
  - `/dashboard` returns recently processed bills + summaries for the UI.
  - Frontend shows one bill at a time in a carousel with three cards: **Purpose**, **Key Points**, **Impact**.

- **Q&A chatbot**
  - `/ask` accepts a natural-language question and (optionally) a `pdf_url` to force bill context.
  - Uses lexical title matching and embedding search to pick the best bill when `pdf_url` is not specified.
  - Selects the top‑similar chunks and calls Groq to answer strictly from that context.
  - Frontend chat behaves like a messaging app, using paragraphs and tracking the active bill.
  - Lightweight follow‑up detection ("explain", "clarify", "simplify") can reuse the last answer instead of re-searching.

- **Scheduler & background work**
  - On startup, a background thread seeds the DB with bill links and processes a small set of priority PDFs.
  - A long‑interval scheduler periodically rescans PRS India to ingest new PDFs without blocking the API.

---

## Prerequisites

- **Python**: 3.11+ recommended.
- **PostgreSQL** with the `pgvector` extension enabled.
- **Groq API key** (for summaries, compression, Q&A).
- Internet access (for scraping PRS India and downloading PDFs, and for first‑time model download).

---

## Environment configuration

1. **Clone the repo** (or copy this folder) and open it in your editor.

2. **Create `.env`** in the project root:

   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Configure Postgres**

   The connection string is defined in `database.py`:

   ```python
   DATABASE_URL = "postgresql://postgres:isaac@localhost/bills_db"
   ```

   Adjust user, password, host, and DB name as needed, then:

   ```bash
   createdb bills_db
   -- enable pgvector in that database (psql):
   -- CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **(Optional) HF token**

   If you want faster / authenticated downloads for the embedding model:

   ```bash
   export HF_TOKEN=your_hf_token_here
   ```

---

## Setup & installation

Create and activate a virtual environment (example using Python venv):

```bash
cd "AI Legislative Analyser"
python -m venv venv
source venv/bin/activate      # on Linux/macOS
# or
venv\Scripts\activate         # on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Postgres is running and the `bills_db` database exists with `pgvector` enabled.

---

## Running the app

Start the API + frontend with uvicorn:

```bash
uvicorn main:app --host localhost --port 8000
```

On first run:

- The app loads FastAPI and sets up the database tables.
- A background bootstrap thread:
  - Calls `process_new_pdfs(initial=True)` to seed bill links.
  - Fetches a small set of priority bill PDFs.
  - Downloads and processes those PDFs into chunks, embeddings, and summaries.
- A scheduler loop starts to periodically ingest new links.

Open the UI in your browser:

```text
http://localhost:8000/
```

### Using the dashboard

- The main card shows one bill at a time with:
  - A cleaned title and bill counter.
  - Three summary cards: **Purpose**, **Key Points**, **Impact**.
  - A button to open the original PDF in a new tab.
- Use the left/right arrows to cycle through processed bills.
- Click **View More** on a card to open a modal with the full text for that section.

### Using the chatbot

- The floating **"Ask about this bill"** button opens the chat panel.
- Ask questions like:
  - "Details of Transgender bill"
  - "What is the purpose of this bill?"
- The frontend sends your query (and sometimes the current bill `pdf_url`) to `/ask`.
- The backend:
  - Finds the most relevant bill and chunks.
  - Invokes Groq to generate an answer constrained to those chunks.
- Follow‑ups such as "explain clearly" or "simplify" can reuse the last answer instead of re-running the full retrieval.

---

## Key endpoints

- `GET /` – Serves the dashboard UI (`static/index.html`).
- `GET /dashboard` – Returns processed bills with summaries for the frontend carousel.
- `GET /fetch-bill/{query}` –
  - Tries lexical title matching, then semantic search, then DB substring, then live scraping to find the best bill.
  - Lazily downloads and processes the target bill if needed.
  - Returns `pdf_url`, `local_path`, and `processed` status.
- `GET /ask?query=...&pdf_url=...` –
  - When `pdf_url` is provided, locks context to that bill’s chunks.
  - Otherwise, does lexical + embedding search across processed bills and, if needed, calls `fetch_bill` to process on demand.
  - Returns `{ "answer": ..., "pdf_url": ... }`.

---

## Notes on performance

- **Model loading**: `AI/embedder.py` lazy-loads `sentence_transformers` and the `all-MiniLM-L6-v2` model only on the first embedding call, reducing uvicorn startup delay.
- **Scraping**: Initial startup limits how many bill pages are scraped; the scheduler later performs full scans in the background.
- **Streaming logs**: Expect logs like "Visiting: https://prsindia.org/billtrack/..." as the scraper runs.

---

## Development tips

- If you change DB schema, run a fresh migration or drop/recreate tables as needed.
- To debug scraping, call `get_pdf_links()` directly in a small script or REPL.
- To speed up local testing:
  - Temporarily reduce the number of priority bills processed on startup.
  - Comment out or adjust scheduler intervals in `scraper/scheduler.py`.

---

## License

Add your preferred license information here.
