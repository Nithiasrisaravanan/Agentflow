# AgentFlow 

> **LLM-Powered Task Automation Agent** — FastAPI · OpenAI Function Calling · Async Orchestration · Redis Caching

AgentFlow is a production-ready ReAct-style agent that decomposes natural-language tasks, selects tools, chains LLM calls for structured outputs, and caches intermediate results in Redis — all over a clean async FastAPI backend.

---

## Features

| Feature | Details |
|---|---|
| **LLM Agent Loop** | ReAct-style Thought → Tool-call → Observation cycles |
| **OpenAI Function Calling** | Structured tool selection via `tools` API parameter |
| **3 Built-in Tools** | `summarizer`, `sql_executor` (SQLite), `search` |
| **Async Orchestration** | `asyncio` event loop with `asyncio.gather` for concurrent tool calls |
| **Redis Caching** | Per-tool and per-task result caching with configurable TTL |
| **Python Threading** | Semaphore-bounded concurrent tool dispatch |
| **Structured Logging** | `structlog` — JSON in production, colourised dev console |
| **Performance Profiling** | `@profile_async` decorator + `ProfileContext` manager with per-op stats |
| **Unit Tests** | `pytest-asyncio` + `fakeredis` — no real Redis or OpenAI required |
| **Docker-ready** | Multi-stage `Dockerfile` + `docker-compose.yml` with Redis service |

---

##  Project Structure

```
agentflow/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan hooks
│   ├── core/
│   │   ├── config.py            # Pydantic-settings (reads .env)
│   │   ├── logging.py           # structlog setup
│   │   └── cache.py             # Async Redis get/set/delete helpers
│   ├── models/
│   │   └── schemas.py           # All Pydantic I/O models
│   ├── tools/
│   │   ├── definitions.py       # OpenAI function-calling specs
│   │   ├── summarizer.py        # Text summarisation tool
│   │   ├── sql_executor.py      # Safe SQLite SELECT executor + seed data
│   │   ├── search.py            # Web search (mock → real provider hook)
│   │   └── registry.py         # Tool registry + concurrent dispatcher
│   ├── agents/
│   │   └── orchestrator.py      # ReAct agent with Redis result caching
│   ├── api/
│   │   └── routes.py            # FastAPI routes
│   └── utils/
│       └── profiler.py          # Performance profiling utilities
├── tests/
│   ├── test_summarizer.py
│   ├── test_sql_executor.py
│   ├── test_search.py
│   ├── test_registry.py
│   ├── test_orchestrator.py
│   ├── test_api.py
│   └── test_profiler.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── requirements.txt
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone <your-repo-url> agentflow
cd agentflow
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

API available at **http://localhost:8000** · Docs at **http://localhost:8000/docs**

### 3. Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Redis (requires Docker or local Redis install)
docker run -d -p 6379:6379 redis:7-alpine

uvicorn app.main:app --reload
```

---

##  Running Tests

```bash
pip install -r requirements.txt
pytest
```

Tests use `fakeredis` — **no real Redis or OpenAI key needed**.

Coverage report is generated in `htmlcov/index.html`.

---

## API Reference

### `POST /api/v1/tasks`

Submit a natural-language task to the agent.

**Request body**
```json
{
  "task": "Summarise the top 5 sales records and find related industry news",
  "context": { "region": "North" },
  "use_cache": true
}
```

**Response**
```json
{
  "task_id": "a1b2c3d4",
  "status": "completed",
  "result": "The top 5 sales records are dominated by Laptop Pro...",
  "steps": [
    {
      "step": 1,
      "thought": "I need to query the database first...",
      "tool_calls": [
        { "tool": "sql_executor", "input": {"query": "SELECT * FROM sales LIMIT 5"}, "output": {...} }
      ]
    }
  ],
  "total_steps": 2,
  "duration_ms": 840.3,
  "cached": false
}
```

### `GET /api/v1/health`

Returns Redis and OpenAI model status.

### `GET /api/v1/tools`

Lists all available tool names and descriptions.

---

##  Configuration

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `REDIS_TTL_SECONDS` | `3600` | Cache TTL |
| `MAX_AGENT_STEPS` | `10` | Max ReAct iterations |
| `MAX_CONCURRENT_TOOLS` | `4` | Semaphore limit for parallel tool calls |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `APP_ENV` | `development` | Set to `production` for JSON logs |

---

## 🔌 Adding a New Tool

1. Create `app/tools/my_tool.py` with an `async def run_my_tool(**kwargs) -> dict` function.
2. Add the OpenAI function spec to `app/tools/definitions.py`.
3. Register it in `app/tools/registry.py` `_REGISTRY` dict.
4. Write tests in `tests/test_my_tool.py`.

---

## Performance Profiling

```python
from app.utils.profiler import profile_async, get_stats

@profile_async("my_operation")
async def my_operation():
    ...

# Later:
print(get_stats())
# {"my_operation": {"calls": 12, "avg_ms": 45.2, "min_ms": 10.1, "max_ms": 120.5, ...}}
```

---

##  Architecture

```
User Request
     │
     ▼
FastAPI Route (/api/v1/tasks)
     │
     ▼
AgentOrchestrator
  ├─ Check Redis cache (task-level)
  ├─ Loop: LLM call → parse tool calls
  │    ├─ dispatch_tools_concurrently()
  │    │    ├─ summarizer  ─┐
  │    │    ├─ sql_executor ─┤ asyncio.gather + semaphore
  │    │    └─ search      ─┘
  │    └─ cache each tool result in Redis
  └─ Final answer → cache in Redis → return TaskResponse
```

---

## License

MIT
