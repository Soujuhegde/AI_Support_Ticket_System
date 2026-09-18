# 🎫 TicketMind

> **AI-Powered Customer Support Ticket Analytics & Anomaly Detection System**  
> Natural language data querying powered by Groq LLM and automated SLA anomaly detection over customer support ticket datasets, exposed via a **FastAPI REST backend** and a modern **Streamlit dashboard**.

---

## 🚀 Quick Start with Docker (Recommended)

You can spin up both the **FastAPI Backend** and the **Streamlit UI** with a **single command**:

### 1. Clone & Configure
```bash
git clone https://github.com/Soujuhegde/AI_Support_Ticket_System.git
cd AI_Support_Ticket_System

# Create .env from the example file
cp .env.example .env
```
Open `.env` and add your free Groq API key (`https://console.groq.com/keys`):
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 2. Run with Docker Compose
```bash
docker-compose up --build
```

### 3. Access Services
- 🌐 **Interactive Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
- 🔌 **FastAPI REST API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

> **Fail-Fast Design:** If `GROQ_API_KEY` is missing or the CSV is not found, the app immediately halts at startup with a clear, actionable log message rather than breaking mid-request.

---

## 💻 Local Setup (Without Docker)

If you prefer running Python directly on your host machine:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Services (in separate terminal windows)
```bash
# Terminal 1 — Start FastAPI Backend (Port 8000)
python -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Start Streamlit Dashboard (Port 8501)
python -m streamlit run src/ui/app.py
```

### 3. Run Offline Test Suite
```bash
python -m pytest tests/ -v
```
*All 22 unit tests run offline in seconds without requiring any network calls or API keys.*

---

## 📂 Project Structure

```
AI_Support_Ticket_System/
├── docker-compose.yml          # Single-command multi-service deployment
├── Dockerfile                  # Container definition (Python 3.11-slim)
├── requirements.txt            # Python dependencies (FastAPI, Streamlit, Pandas, Groq, etc.)
├── .env.example                # Sample environment variables
├── .gitignore                  # Git ignore rules (protects .env secrets)
├── README.md                   # Complete system documentation
│
├── data/
│   └── support_tickets.csv     # 500 support ticket records
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Central config & fail-fast startup validation
│   │
│   ├── data/                   # Data Layer
│   │   ├── __init__.py
│   │   ├── loader.py           # Ingests CSV & enforces clean Pandas dtypes
│   │   └── store.py            # Shared in-memory data store for API & UI
│   │
│   ├── query/                  # Natural Language Query Engine
│   │   ├── __init__.py
│   │   ├── schema.py           # Strict Pydantic JSON schema for query plans
│   │   ├── prompt.py           # Grounded system prompt with schema rules
│   │   ├── llm.py              # Groq API client with retry & exponential backoff
│   │   ├── parser.py           # JSON validation & code fence stripping
│   │   └── runner.py           # Safe, deterministic execution over DataFrame
│   │
│   ├── anomalies/              # Rule-Based Anomaly Detection
│   │   ├── __init__.py
│   │   └── rules.py            # SLA breaches (percentile thresholds) & stale tickets
│   │
│   ├── api/                    # REST API Layer
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application entrypoint
│   │   └── routes.py           # /query, /anomalies, /health endpoints
│   │
│   └── ui/                     # Web Dashboard Layer
│       ├── __init__.py
│       └── app.py              # Light-theme Streamlit dashboard
│
└── tests/                      # Automated Test Suite (22 unit tests)
    ├── test_loader.py          # Data ingestion & missing value tests
    ├── test_parser.py          # Query schema validation & security tests
    ├── test_runner.py          # Query execution & aggregation logic tests
    └── test_rules.py           # Anomaly percentile & threshold tests
```

---

## ⚙️ Architecture & Design Decisions

```
           ┌───────────────────────┐
           │  support_tickets.csv  │
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   src/data/loader.py  │
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   src/data/store.py   │  (Shared In-Memory Store)
           └───────────┬───────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  src/anomalies/  │       │      src/query/      │
│     rules.py     │       │ (prompt → llm →      │
│  (Rule Engine)   │       │  parser → runner)    │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         └─────────────┬──────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  src/api/routes  │       │    src/ui/app.py     │
│   (FastAPI)      │       │     (Streamlit)      │
└──────────────────┘       └──────────────────────┘
```

### 1. Why the LLM Never Touches Raw Data (Safe Query Planner)
- **Zero Arbitrary Code Execution:** The LLM's only responsibility is converting a user's question into a strictly-typed JSON specification (`QuerySpec`).
- **Pydantic Validation:** The JSON plan is validated by `src/query/schema.py` before execution. Any unrecognized columns or malicious operators are rejected with a clean `422 Unprocessable Entity`.
- **Deterministic Execution:** Only our internal query engine (`src/query/runner.py`) interacts with the Pandas DataFrame.
- **100% Offline Testable:** Every single query plan can be tested deterministically without making external network calls.

### 2. Why Anomaly Detection is Rule-Based
- **Explainability:** With 500 rows, complex black-box machine learning models risk overfitting and are hard to interpret. Threshold-based rules provide immediate, defensible business logic:
  - **Resolution SLA Breaches:** Tickets whose resolution time exceeds the 90th percentile of resolved tickets (>43.18 hrs).
  - **Stale Urgent Tickets:** High/Critical tickets unresolved for longer than 24 hours.
- **Historical Snapshot Calibration:** "Current Time" is calibrated to the dataset's latest timestamp (`2024-03-30`) so historical tickets are evaluated accurately.

---

## 🔌 API Reference

### `GET /health`
Liveness check.
```json
{
  "status": "ok"
}
```

### `POST /query`
Converts natural language questions into structured queries and executes them.
**Request:**
```json
{
  "question": "Which agent has the lowest average customer rating?"
}
```
**Response:**
```json
{
  "question": "Which agent has the lowest average customer rating?",
  "query_spec": {
    "metric": "average",
    "metric_column": "customer_rating",
    "filters": [],
    "group_by": "agent_id",
    "sort": "asc",
    "limit": 1
  },
  "answer": "AGT-08 has the lowest average customer_rating (3.48).",
  "result": [
    {
      "agent_id": "AGT-08",
      "value": 3.48
    }
  ]
}
```

### `GET /anomalies`
Returns detected SLA breaches and stale high-priority tickets.
**Response:**
```json
{
  "reference_time": "2024-03-30 18:06:00",
  "resolution_time_threshold_hrs": 43.18,
  "stale_hours_threshold": 24.0,
  "long_resolution_time": [ ... ],
  "stale_high_priority": [ ... ]
}
```

---

## 🧪 Example Questions You Can Ask

| Question | Generated Query Spec | Answer |
| :--- | :--- | :--- |
| *"How many tickets are currently open?"* | `metric=count, filter(status=Open)` | `111 ticket(s) match your query.` |
| *"Which agent has the lowest average rating?"* | `metric=average(customer_rating), group=agent_id, sort=asc, limit=1` | `AGT-08 has the lowest average customer_rating (3.48).` |
| *"Show me 3 Critical tickets"* | `metric=list, filter(priority=Critical), limit=3` | Found 55 matching ticket(s). Showing the first 3. |
| *"What is the average rating for Technical tickets?"* | `metric=average(customer_rating), filter(category=Technical)` | `The average customer_rating is 3.74.` |

---

## 🛠️ Technology Stack
- **AI / LLM:** Groq API (`openai/gpt-oss-120b` / `llama-3` series)
- **Backend API:** FastAPI + Uvicorn
- **Frontend UI:** Streamlit (Custom Light Theme, Scrollable Table Pane, Status Badges)
- **Data Engine:** Pandas + Pydantic
- **Testing:** Pytest (22 tests)
- **Containerization:** Docker & Docker Compose

