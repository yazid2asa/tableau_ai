# Text-to-Viz Agent — Architecture Overview

> **Status:** M1–M3 complete · 37 tests passing · M4 (RAG + Vector Store) in design

---

## What is it?

A full-stack AI agent embedded in **Tableau Desktop** as an Extension. Users type a natural-language question; the agent generates a fully executable `.twb` workbook file that opens automatically — no manual steps.

```
"Show me monthly revenue by region as a bar chart"
        ↓
    LLM extracts intent → TWBEditor builds XML → .twb auto-opens in Tableau
```

---

## Architecture at a Glance

```
┌─────────────────┐     ┌───────────────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  ① UI           │────▶│  ② RAG Layer  [NEW]        │────▶│  ③ LLM Pipeline  │────▶│  ④ TWB Generator   │
│  Tableau Ext.   │     │  Metadata API              │     │  OpenRouter LLM  │     │  twilize (XSD)     │
│  Chat input     │     │  Vector Store              │     │  VizIntent JSON  │     │  10 chart types    │
│  Chart Cart     │     │  Semantic Retriever        │     │  Task Router     │     │  Multi-sheet .twb  │
└─────────────────┘     │  Pre-processing            │     └────────┬─────────┘     └────────┬───────────┘
                        └───────────────────────────┘              │                         │
                                                                    ▼                         ▼
                                                          ┌─────────────────────────────────────────────┐
                                                          │  ⑤ Quality & Observability                  │
                                                          │  LLM-as-a-Judge · LangFuse · JSONL · SQLite  │
                                                          └─────────────────────────────────────────────┘
                                                                                               │
                                                                                               ▼
                                                                                    ┌──────────────────────┐
                                                                                    │  ⑥ Output & Daemon   │
                                                                                    │  watchdog → tableau://│
                                                                                    └──────────────────────┘
```

---

## ① User Interface — Tableau Extension

| Component | Detail |
|---|---|
| **Shell** | `.trex` manifest + `index.html` (Vanilla JS) |
| **Chat UI** | Multi-turn conversation with SSE live status |
| **Data picker** | Full path to Excel/CSV, persisted via Extension Settings |
| **Chart Cart** | Accumulates VizIntents, supports Remove per chart |
| **Download** | Generates one multi-sheet `.twb` from the entire cart |
| **Daemon badge** | Green indicator when local daemon is running |

---

## ② RAG Layer ✦ PLANNED (M4)

> **Goal:** Give the LLM precise schema awareness without hallucinating field names.

### 2a — RAG on Metadata API

- Connect to **Tableau Metadata API** (GraphQL) to pull the live datasource schema: field names, types, relationships, calculated fields.
- Embed each field description into a **Vector Store** (e.g., ChromaDB or FAISS).
- On each user question → **semantic search** retrieves the top-K most relevant fields.
- Injected into the LLM prompt as structured context → drastically reduces hallucinated field names.

```
User question: "revenue by region"
      ↓ embed query
Vector Store → [SalesAmount (measure), RegionName (dimension), OrderDate (date)]
      ↓ inject into prompt
LLM → accurate VizIntent with real field names
```

### 2b — Vector Store for Datasource Schemas

- Store **multiple datasource schemas** so the agent can work across data sources in the same session.
- Each datasource is indexed once at load time.
- Query time: sub-100ms cosine similarity lookup.

### 2c — Pre-processing / Missing Data Handling

| Scenario | Action |
|---|---|
| Field referenced but not in datasource | Fuzzy-match closest field, warn user |
| Datasource not connected | Prompt user to select a data file via the picker |
| Measure missing for KPI | Auto-suggest `COUNT(*)` or `SUM` of available numeric field |
| Date field absent for time series | Fall back to categorical bar chart, explain why |

---

## ③ LLM Pipeline

### LLM Call #1 — Intent Extraction

```json
Input:  { "question": "...", "metadata": {...}, "rag_context": [...] }
Output: {
  "viz_type": "bar_chart",
  "title": "Monthly Revenue by Region",
  "x_field": "RegionName",
  "y_field": "SalesAmount",
  "filters": [],
  "colors": ["#4C72B0"]
}
```

**Model:** `minimax/minimax-m2.5:free` via OpenRouter (swappable via `MODEL_ID` env var)

### Task Router

After intent extraction, the router dispatches to:

| Route | Trigger |
|---|---|
| `create_chart` | Standard new visualization |
| `add_to_workbook` | Checkbox "Add to my open workbook" |
| `calc_field` | LLM detects a computed metric |
| `filter_view` | User asks to restrict existing view |

### SSE Streaming

`POST /chat/stream` → `text/event-stream`

```
event: status    → "Analyzing your question…"
event: intent    → VizIntent JSON
event: result    → .twb path + judge score
event: done
```

---

## ④ TWB Generator

Built on **twilize** — XSD-validated `.twb` XML generation.

### Supported Chart Types (10)

| viz_type | Tableau Mark | Notes |
|---|---|---|
| `bar_chart` | Bar | |
| `line_chart` | Line | |
| `pie` | Pie | color=x, size=AGG(y) |
| `scatter` | Scatterplot | |
| `area` | Area | |
| `heatmap` | Heatmap (Square) | |
| `treemap` | Tree Map | |
| `gantt` | Gantt Bar | |
| `kpi` | Text | single aggregated number |
| `combo` | Dual Axis (Bar+Line) | |

### Datasource Wiring

`_apply_data_file(ed, path)` rewires the datasource via **direct lxml XML manipulation**:

- **Excel** → `<connection class="excel-direct" filename="..."/>`
- **CSV** → switches class to `textscan`, updates relation references

### Multi-sheet Workbook

`generate_multi_sheet_twb()` → one `TWBEditor`, all cart worksheets in a single pass → one `.twb`.

---

## ⑤ Quality & Observability

### LLM-as-a-Judge (Call #2)

Every generated `.twb` is scored before delivery:

| Criterion | Weight |
|---|---|
| Field/type cohesion | 30% |
| Viz relevance | 35% |
| XML validity | 25% |
| Completeness | 10% |

- **Threshold:** 0.75 — if below, regenerate (max **2 retries**)
- If still below after retries → deliver with `warning: "Qualité partielle — score: X.XX"`
- `ChatResponse` includes `judge_score` and `judge_feedback`

### 3-Layer Observability

```
Layer 1 — Structured JSONL     logs/llm_traces.jsonl   (100MB rotation, 30 backups)
Layer 2 — LangFuse Cloud       Traces · Spans · Scores (cloud.langfuse.com)
Layer 3 — User Feedback        SQLite feedback table   (trace_id, score, comment)
```

### Monitoring Dashboard

`GET /monitoring` — Jinja2 HTML, auto-refresh 30s

**12 KPIs:** total generations · success rate · avg judge score · avg/p95 latency · viz type distribution · feedback counts · judge score buckets · hourly activity

---

## ⑥ Output & Auto-Open Daemon

```
output/ folder
    ↓  watchdog (on_created / on_modified)
daemon.py
    ↓  3s debounce (timestamp-based)
tableau:// protocol handler
    ↓
Tableau Desktop opens .twb automatically
```

- Health endpoint: `http://localhost:8765/status`
- Handles "Add to workbook" re-open (on_modified event)

---

## Data Flow — End to End

```
User types question
    │
    ▼
Extension sends { question, session_id, metadata, data_file_path, conversation_history }
    │
    ▼  [NEW — M4]
RAG Retriever → top-K fields from Vector Store
    │
    ▼
LLM Call #1 → VizIntent JSON
    │
    ├──[pre-processing]── missing field? → fuzzy match / fallback
    │
    ▼
TWBEditor → XSD-validated .twb XML
    │
    ▼
LLM Call #2 (Judge) → score
    │
    ├──[score < 0.75]── retry (max 2×)
    │
    ▼
.twb saved to output/
    │
    ├──▶ Daemon opens Tableau Desktop
    ├──▶ LangFuse trace flushed
    ├──▶ JSONL line written
    └──▶ GenerationLog row inserted in SQLite
```

---

## Milestones

| Milestone | Status | Key Deliverables |
|---|---|---|
| **M1** Core Infrastructure | ✅ Complete | FastAPI, .trex, OpenRouter, Bar/Line charts, daemon |
| **M2** Conversational Agent | ✅ Complete | 5 new chart types, SSE, multi-turn, session mgmt |
| **M2.5** Datasource & Cart | ✅ Complete | Data file picker, chart cart, multi-sheet download |
| **M3** Quality & Observability | ✅ Complete | LLM Judge, LangFuse, JSONL, monitoring dashboard |
| **M4** RAG + Vector Store | 🔵 Design | Metadata API RAG, Vector Store, pre-processing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.11+) |
| Frontend | HTML5 · Vanilla JS · Tableau Extensions API |
| LLM | OpenRouter — `minimax/minimax-m2.5:free` |
| TWB generation | twilize (XSD-validated XML) |
| RAG (planned) | Tableau Metadata API + ChromaDB/FAISS |
| Observability | LangFuse Cloud |
| Database | SQLite via SQLAlchemy async |
| Daemon | Python watchdog + `tableau://` protocol |
| Tests | Pytest + HTTPX · **37 tests passing** |

---

*Last updated: 2026-04-24*
