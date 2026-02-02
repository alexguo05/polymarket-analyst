# Polymarket Analyst

A fundamental analysis pipeline for finding mispriced prediction markets on Polymarket.

## Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  get_markets.py │───▶│generate_queries │───▶│execute_searches │───▶│analyze_evidence │───▶│compute_prediction│
│                 │    │      .py        │    │      .py        │    │      .py        │    │      .py        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼                      ▼
 candidate_markets      market_queries        search_results        evidence_analysis        predictions
 _filtered.json              .json                 .json                  .json                  .json
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create `.env` in the project root:

```env
OPENAI_KEY=sk-...
PERPLEXITY_KEY=pplx-...
```

### 3. Run the Pipeline

```bash
# Step 1: Fetch and filter markets from Polymarket
python src/get_markets.py

# Step 2: Generate research queries for each condition
python src/generate_queries.py --limit 10

# Step 3: Execute searches via Perplexity
python src/execute_searches.py

# Step 4: Score evidence with GPT
python src/analyze_evidence.py

# Step 5: Compute final predictions
python src/compute_prediction.py
```

---

## Scripts

### `src/get_markets.py`

Fetches markets from Polymarket's Gamma API and filters for markets where fundamental analysis can provide an edge.

**Filters Applied:**
| Filter | Value | Purpose |
|--------|-------|---------|
| Liquidity | $5K - $500K | Enough to trade, not dominated by whales |
| Price Range | 20% - 80% | Uncertainty zone where research matters |
| Time to Resolution | 5 - 60 days | Enough time to research, not too far out |
| Condition Volume | ≥ $10K | Sufficient trading activity per outcome |
| Categories | Politics, Elections, Geopolitics, Culture | Semantic markets |
| Excludes | Sports, Crypto, "Will X say Y" markets | Low-signal or speed-based |

**Usage:**
```bash
python src/get_markets.py              # Fetch and filter
python src/get_markets.py --verbose    # Show filter decisions
```

**Outputs:**
- `data/candidate_markets.json` - All markets passing base filters
- `data/candidate_markets_filtered.json` - Refined subset for analysis

---

### `src/generate_queries.py`

Generates 5 targeted research queries per condition using GPT-5 mini.

**Query Categories:**
- `base_rate` - Historical precedent and statistical patterns
- `procedural` - Rules, timelines, legal requirements
- `sentiment` - Expert opinions, polling, public perception
- `recent_developments` - Breaking news, recent announcements
- `structural_factors` - Systemic advantages/disadvantages

**Usage:**
```bash
python src/generate_queries.py                    # Process all conditions
python src/generate_queries.py --limit 10         # First 10 conditions
python src/generate_queries.py --random --limit 5 # Random sample of 5
python src/generate_queries.py --seed 42          # Reproducible random sample
python src/generate_queries.py --regenerate       # Redo existing conditions
python src/generate_queries.py --replace-output   # Overwrite output file
python src/generate_queries.py --dry-run          # Preview without API calls
```

**Output:** `data/market_queries.json`

---

### `src/execute_searches.py`

Executes research queries using Perplexity's Sonar API.

**Features:**
- Returns source-attributed analysis with citations
- Configurable recency filter (day/week/month/year)
- Incremental saving (progress saved after each query)
- Retry logic for failed queries

**Usage:**
```bash
python src/execute_searches.py                     # Execute all queries
python src/execute_searches.py --limit 5           # Limit to 5 conditions
python src/execute_searches.py --query-limit 3     # Max 3 queries per condition
python src/execute_searches.py --recency month     # Search last month only
python src/execute_searches.py --regenerate        # Re-run all searches
python src/execute_searches.py --rerun-empty       # Retry queries with no citations
```

**Output:** `data/search_results.json`

---

### `src/analyze_evidence.py`

Scores evidence quality using GPT-5 Pro with structured output.

**Scoring Rubrics (0.0 - 1.0):**

| Metric | Description |
|--------|-------------|
| **Reliability** | Source trustworthiness (government > newspaper > blog) |
| **Recency** | How current is the information |
| **Relevance** | How directly it addresses the question |
| **Specificity** | Concrete data vs vague speculation |
| **Direction** | +1 (supports YES), -1 (supports NO), 0 (neutral) |
| **Strength** | Magnitude of probability shift (0.0 - 1.0) |

Also estimates **predictability** (0.0 - 1.0): how forecastable is this market from available information.

**Usage:**
```bash
python src/analyze_evidence.py                # Analyze all conditions
python src/analyze_evidence.py --limit 5      # Limit to 5 conditions
python src/analyze_evidence.py --regenerate   # Re-analyze existing
python src/analyze_evidence.py --dry-run      # Preview without API calls
```

**Output:** `data/evidence_analysis.json`

---

### `src/compute_prediction.py`

Computes final probability estimates and trading signals.

**Algorithm:**
1. Anchor on market price (convert to log-odds)
2. For each evidence item: `adjustment = direction × strength × quality`
3. Scale by predictability and evidence weight
4. Convert back to probability
5. Calculate edge and APY

**Usage:**
```bash
python src/compute_prediction.py                   # Compute all predictions
python src/compute_prediction.py --verbose         # Show evidence breakdown
python src/compute_prediction.py --scale 0.3       # More conservative (default: 0.5)
python src/compute_prediction.py --min-edge 0.05   # Only show ≥5% edge
python src/compute_prediction.py --sort apy        # Sort by APY
python src/compute_prediction.py --sort days       # Sort by soonest resolution
```

**Output:** `data/predictions.json`

**Display:**
```
================================================================================
📊 Will Israel strike Iran by January 31, 2026?
   Event: Israel strikes Iran by January 31, 2026?
================================================================================

   Market Price:      20.5%
   Our Estimate:      26.3%
   Edge:              +5.8% (+28.3%)
   Predictability:    65%
   Days Until End:    8 days
   APY:               892%
   Volume:            $13,055,014
   Liquidity:         $86,389

   Recommendation:    🟢 BUY YES
```

---

### `src/check_orderbook.py`

Debug tool to monitor real-time orderbook updates via WebSocket.

**Usage:**
```bash
python src/check_orderbook.py 22862                    # By event ID
python src/check_orderbook.py "israel-strikes-iran"    # By slug (partial match)
```

---

## Data Flow

Each condition (outcome) flows through the pipeline with these fields:

| Stage | Key Fields Added |
|-------|------------------|
| `get_markets.py` | `condition_id`, `yes_price`, `volume`, `liquidity`, `end_date` |
| `generate_queries.py` | `queries[]` with `query_id`, `category`, `priority` |
| `execute_searches.py` | `query_results[]` with `response`, `citations` |
| `analyze_evidence.py` | `evidence_scores[]`, `predictability` |
| `compute_prediction.py` | `final_probability`, `edge`, `apy`, `direction` |

---

## Configuration

### Global Constants

Edit at the top of each script:

**`get_markets.py`:**
```python
MIN_LIQUIDITY = 5_000
MAX_LIQUIDITY = 500_000
MIN_PRICE = 0.20
MAX_PRICE = 0.80
FILTERED_MIN_DAYS_UNTIL_END = 5
FILTERED_MAX_DAYS_UNTIL_END = 60
MIN_CONDITION_VOLUME = 10_000
```

**`generate_queries.py`:**
```python
MODEL = "gpt-5-mini"
RATE_LIMIT_DELAY = 0.5
```

**`execute_searches.py`:**
```python
PERPLEXITY_MODEL = "sonar"  # or "sonar-pro"
RATE_LIMIT_DELAY = 0.5
```

**`analyze_evidence.py`:**
```python
MODEL = "gpt-5-pro"
RATE_LIMIT_DELAY = 15  # Strict rate limits
```

**`compute_prediction.py`:**
```python
EVIDENCE_SCALE = 0.5  # Higher = more responsive to evidence
```

---

## Requirements

```
requests>=2.31.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
python-dateutil>=2.8.0
openai>=1.0.0
pydantic>=2.0.0
websockets>=12.0
```

---

## Philosophy

This system doesn't try to beat the market on speed. It bets on depth:

- **Read the documents others skip** — legal filings, regulatory timelines, procedural rules
- **Connect the dots others miss** — multi-source synthesis across press releases and expert commentary  
- **Understand the procedures others ignore** — legislative calendars, court schedules, bureaucratic processes

The edge isn't in faster data—it's in better interpretation.
