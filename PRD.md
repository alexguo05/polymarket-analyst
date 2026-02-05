# Product Requirements Document (PRD)
# Polymarket Automated Trading System

**Version:** 1.0  
**Date:** February 5, 2026  
**Author:** Alex Guo  
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals & Objectives](#2-goals--objectives)
3. [User Stories](#3-user-stories)
4. [System Architecture](#4-system-architecture)
5. [Feature Requirements](#5-feature-requirements)
6. [Database Schema](#6-database-schema)
7. [API Specification](#7-api-specification)
8. [Security Requirements](#8-security-requirements)
9. [Infrastructure & Deployment](#9-infrastructure--deployment)
10. [Success Metrics](#10-success-metrics)
11. [Timeline & Milestones](#11-timeline--milestones)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Open Questions](#13-open-questions)

---

## 1. Executive Summary

### 1.1 Problem Statement

Manual prediction market trading is time-intensive and prone to emotional decision-making. Identifying mispriced markets requires systematic research across hundreds of markets, continuous monitoring of open positions, and disciplined execution—tasks that are difficult to perform consistently as a human.

### 1.2 Solution

Build an automated trading system that:
- **Discovers** potentially mispriced markets using configurable filters
- **Researches** markets using AI-powered web search and evidence analysis
- **Predicts** fair probabilities by synthesizing evidence
- **Executes** trades on Polymarket when edge thresholds are met
- **Monitors** open positions and re-evaluates when conditions change
- **Reports** performance through a real-time dashboard

### 1.3 Scope

| In Scope | Out of Scope (v1) |
|----------|-------------------|
| Polymarket integration | Other prediction markets (Kalshi, Manifold) |
| Automated research pipeline | Real-time news monitoring |
| Trade execution | Arbitrage strategies |
| Position monitoring | Social sentiment analysis |
| Performance dashboard | Mobile app |
| Statistical analysis | Multi-user support |

---

## 2. Goals & Objectives

### 2.1 Primary Goals

| Goal | Metric | Target |
|------|--------|--------|
| **Profitability** | ROI on deployed capital | > 15% annualized |
| **Accuracy** | Prediction calibration (Brier score) | < 0.20 |
| **Automation** | Human intervention required | < 1 hour/week |
| **Reliability** | System uptime | > 99% |

### 2.2 Success Criteria

1. System autonomously identifies at least 10 actionable opportunities per week
2. Executes trades within 60 seconds of decision
3. Maintains accurate P&L tracking with < 0.1% accounting error
4. Provides clear audit trail for all trading decisions

---

## 3. User Stories

### 3.1 Core User Stories

#### US-001: Market Discovery
> As a trader, I want the system to automatically scan Polymarket for markets matching my criteria, so I don't have to manually browse hundreds of markets.

**Acceptance Criteria:**
- System fetches all active markets from Polymarket API
- Applies configurable filters (liquidity, volume, time horizon, category)
- Excludes unwanted categories (sports, crypto, say/mention markets)
- Outputs ranked list of candidate markets

#### US-002: Automated Research
> As a trader, I want the system to research candidate markets using web search, so I can make informed decisions without manual research.

**Acceptance Criteria:**
- Generates relevant search queries for each market condition
- Executes searches via Perplexity Sonar API
- Extracts and scores evidence from search results
- Provides reasoning for each evidence assessment

#### US-003: Probability Prediction
> As a trader, I want the system to calculate a fair probability for each market condition, so I can identify mispriced opportunities.

**Acceptance Criteria:**
- Anchors on current market price
- Adjusts based on evidence quality, direction, and strength
- Calculates edge (predicted - market price)
- Calculates expected APY based on time to resolution

#### US-004: Trade Execution
> As a trader, I want the system to automatically place bets when edge exceeds my threshold, so I don't miss opportunities.

**Acceptance Criteria:**
- Connects to Polymarket CLOB via API
- Places limit orders at specified prices
- Respects position size limits and risk parameters
- Logs all trade attempts and outcomes

#### US-005: Position Monitoring
> As a trader, I want to monitor my open positions and be alerted to significant changes, so I can react to new information.

**Acceptance Criteria:**
- Tracks all open positions in database
- Monitors current prices vs entry prices
- Alerts when price moves > X% against position
- Triggers re-analysis when significant news detected

#### US-006: Performance Dashboard
> As a trader, I want a web dashboard showing my portfolio, predictions, and performance metrics, so I can assess system effectiveness.

**Acceptance Criteria:**
- Real-time portfolio value and P&L
- Historical performance charts
- Prediction accuracy metrics
- Trade history with full audit trail

#### US-007: Manual Override
> As a trader, I want the ability to manually approve/reject trades before execution, so I maintain control during the testing phase.

**Acceptance Criteria:**
- "Approval required" mode for all trades
- Dashboard interface to approve/reject pending trades
- Configurable auto-approve thresholds
- Full manual mode available

### 3.2 Administrative Stories

#### US-008: Configuration Management
> As an admin, I want to adjust system parameters without code changes, so I can tune the system based on performance.

**Acceptance Criteria:**
- Web UI for configuration changes
- Parameters: filters, thresholds, position limits, API keys
- Changes take effect without system restart
- Audit log of all configuration changes

#### US-009: System Health Monitoring
> As an admin, I want visibility into system health and errors, so I can quickly diagnose issues.

**Acceptance Criteria:**
- Health check endpoints for all services
- Error logging with stack traces
- Alerting for critical failures (trade execution, API errors)
- Dashboard showing pipeline run history

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GOOGLE CLOUD PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  CLOUD SCHEDULER │                                                       │
│  │  ───────────────  │                                                       │
│  │  • 00:00 UTC      │──────┐                                               │
│  │    Full pipeline  │      │                                               │
│  │  • Every 1 hour   │      │                                               │
│  │    Position check │      │                                               │
│  │  • Every 5 min    │      │                                               │
│  │    Price alerts   │      │                                               │
│  └──────────────────┘      │                                               │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      CLOUD RUN - Backend API                         │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FastAPI Application                                           │  │  │
│  │  │  ├── /api/pipeline     - Trigger & status                      │  │  │
│  │  │  ├── /api/predictions  - View predictions                      │  │  │
│  │  │  ├── /api/positions    - Manage positions                      │  │  │
│  │  │  ├── /api/trades       - Trade history                         │  │  │
│  │  │  ├── /api/config       - System configuration                  │  │  │
│  │  │  └── /api/health       - Health checks                         │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Pipeline Orchestrator                                         │  │  │
│  │  │  ├── MarketScanner      (get_markets.py)                       │  │  │
│  │  │  ├── QueryGenerator     (generate_queries.py)                  │  │  │
│  │  │  ├── SearchExecutor     (execute_searches.py)                  │  │  │
│  │  │  ├── EvidenceAnalyzer   (analyze_evidence.py)                  │  │  │
│  │  │  └── PredictionEngine   (compute_prediction.py)                │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Trade Executor                                                │  │  │
│  │  │  ├── PolymarketClient   - API wrapper                          │  │  │
│  │  │  ├── OrderManager       - Place/cancel orders                  │  │  │
│  │  │  └── RiskManager        - Position limits, validation          │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Position Monitor                                              │  │  │
│  │  │  ├── PriceTracker       - Current prices                       │  │  │
│  │  │  ├── AlertEngine        - Threshold alerts                     │  │  │
│  │  │  └── Revalidator        - Re-run analysis                      │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      CLOUD SQL (PostgreSQL)                          │  │
│  │  Tables: markets, conditions, predictions, positions, trades,        │  │
│  │          pipeline_runs, alerts, config                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  SECRET MANAGER  │  │  CLOUD STORAGE   │  │  CLOUD MONITORING        │  │
│  │  • API keys      │  │  • JSON backups  │  │  • Logs                  │  │
│  │  • Private key   │  │  • Historical    │  │  • Alerts                │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VERCEL - Frontend Dashboard                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Next.js Application                                                 │  │
│  │  ├── /                    - Portfolio overview                       │  │
│  │  ├── /positions           - Open positions                           │  │
│  │  ├── /predictions         - Pipeline outputs                         │  │
│  │  ├── /trades              - Trade history                            │  │
│  │  ├── /analytics           - Performance metrics                      │  │
│  │  └── /settings            - Configuration                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Polymarket  │  │  Perplexity  │  │   OpenAI     │  │   Polygon    │   │
│  │  Gamma API   │  │  Sonar API   │  │   GPT API    │  │   Network    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Descriptions

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Cloud Scheduler** | Trigger periodic jobs | GCP Cloud Scheduler |
| **Backend API** | HTTP endpoints, business logic | FastAPI (Python) |
| **Pipeline Orchestrator** | Run research pipeline | Python scripts |
| **Trade Executor** | Place/manage orders | Python + Polymarket SDK |
| **Position Monitor** | Track positions, alerts | Python + WebSocket |
| **Database** | Persistent storage | PostgreSQL (Cloud SQL) |
| **Frontend** | User interface | Next.js + React |
| **Secret Manager** | Secure credential storage | GCP Secret Manager |

### 4.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DAILY PIPELINE FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. MARKET DISCOVERY                                                        │
│  ┌──────────────┐                                                           │
│  │ Polymarket   │───▶ Fetch all active events                               │
│  │ Gamma API    │     Filter by criteria                                    │
│  └──────────────┘     Output: candidate_markets.json                        │
│         │                                                                   │
│         ▼                                                                   │
│  2. QUERY GENERATION                                                        │
│  ┌──────────────┐                                                           │
│  │ OpenAI GPT   │───▶ Generate search queries per condition                 │
│  │              │     Output: market_queries.json                           │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  3. SEARCH EXECUTION                                                        │
│  ┌──────────────┐                                                           │
│  │ Perplexity   │───▶ Execute searches, collect citations                   │
│  │ Sonar API    │     Output: search_results.json                           │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  4. EVIDENCE ANALYSIS                                                       │
│  ┌──────────────┐                                                           │
│  │ OpenAI GPT   │───▶ Score evidence (reliability, relevance, etc.)         │
│  │              │     Output: evidence_analysis.json                        │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  5. PREDICTION COMPUTATION                                                  │
│  ┌──────────────┐                                                           │
│  │ Local Calc   │───▶ Calculate fair probability, edge, APY                 │
│  │              │     Output: predictions.json                              │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  6. TRADE DECISION                                                          │
│  ┌──────────────┐                                                           │
│  │ Risk Manager │───▶ Filter by edge threshold, position limits             │
│  │              │     Create pending trades                                 │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  7. TRADE EXECUTION (if auto-approved)                                      │
│  ┌──────────────┐                                                           │
│  │ Polymarket   │───▶ Place limit orders                                    │
│  │ CLOB API     │     Record in database                                    │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Feature Requirements

### 5.1 Phase 1: Foundation (MVP)

**Goal:** Backend API with database, running existing pipeline

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F1.1 | Pipeline Orchestrator | P0 | Run full pipeline via single API call |
| F1.2 | Database Models | P0 | Store markets, conditions, predictions |
| F1.3 | REST API | P0 | Endpoints for pipeline, predictions |
| F1.4 | Cloud Run Deployment | P0 | Containerized backend on GCP |
| F1.5 | Scheduled Runs | P1 | Cloud Scheduler triggers daily pipeline |
| F1.6 | Basic Logging | P1 | Structured logs for debugging |

**Deliverables:**
- `api/` directory with FastAPI application
- `src/orchestrator.py` coordinating pipeline scripts
- `infrastructure/` with Dockerfile, cloudbuild.yaml
- Database schema and migrations
- Deployed to Cloud Run

### 5.2 Phase 2: Trade Execution

**Goal:** Automatically place bets on Polymarket

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F2.1 | Polymarket Client | P0 | API wrapper for CLOB operations |
| F2.2 | Order Placement | P0 | Place limit orders |
| F2.3 | Order Monitoring | P0 | Track order status (filled, cancelled) |
| F2.4 | Risk Manager | P0 | Position limits, max bet size |
| F2.5 | Approval Workflow | P1 | Manual approval for trades |
| F2.6 | Paper Trading Mode | P1 | Simulate trades without execution |

**Deliverables:**
- `src/trading/` module
- Wallet integration (private key handling)
- Trade logging to database
- Risk parameter configuration

### 5.3 Phase 3: Position Monitoring

**Goal:** Track open positions, detect changes, re-evaluate

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F3.1 | Position Tracker | P0 | Store entry price, current price, P&L |
| F3.2 | Price Monitoring | P0 | Periodic price updates |
| F3.3 | Alert Engine | P1 | Notify on significant price moves |
| F3.4 | Re-analysis Trigger | P1 | Re-run pipeline for open positions |
| F3.5 | Resolution Tracking | P1 | Detect market resolution, calculate final P&L |
| F3.6 | Exit Strategy | P2 | Automated exit on thesis invalidation |

**Deliverables:**
- `src/monitoring/` module
- Position database table with real-time updates
- Alert configuration (email, webhook)
- Resolution detection logic

### 5.4 Phase 4: Frontend Dashboard

**Goal:** Web interface for monitoring and control

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F4.1 | Portfolio Overview | P0 | Total value, P&L, open positions |
| F4.2 | Predictions View | P0 | Pipeline outputs with filtering |
| F4.3 | Trade History | P0 | All executed trades |
| F4.4 | Performance Analytics | P1 | ROI, accuracy, Brier score charts |
| F4.5 | Manual Trade Approval | P1 | Approve/reject pending trades |
| F4.6 | Configuration UI | P2 | Edit system parameters |
| F4.7 | Real-time Updates | P2 | WebSocket for live data |

**Deliverables:**
- `frontend/` Next.js application
- Deployed to Vercel
- Responsive design (desktop + mobile)
- Authentication (single-user initially)

### 5.5 Phase 5: Advanced Features (Future)

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F5.1 | Order Flow Monitoring | P2 | Detect large trades, follow whales |
| F5.2 | News Integration | P2 | Trigger re-analysis on breaking news |
| F5.3 | Multi-market Arbitrage | P3 | Cross-platform opportunities |
| F5.4 | Backtesting | P2 | Test strategies on historical data |
| F5.5 | Strategy Variants | P3 | Multiple prediction models |

---

## 6. Database Schema

### 6.1 Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     events      │     │   conditions    │     │   predictions   │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │◄────│ event_id (FK)   │◄────│ condition_id(FK)│
│ polymarket_id   │     │ id (PK)         │     │ id (PK)         │
│ title           │     │ condition_id    │     │ market_price    │
│ slug            │     │ question        │     │ predicted_price │
│ end_date        │     │ yes_price       │     │ edge            │
│ volume          │     │ volume          │     │ apy             │
│ created_at      │     │ liquidity       │     │ confidence      │
│ updated_at      │     │ created_at      │     │ predictability  │
└─────────────────┘     │ updated_at      │     │ recommendation  │
                        └─────────────────┘     │ created_at      │
                               │                └─────────────────┘
                               │                        │
                               ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    positions    │     │ evidence_scores │     │     trades      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │     │ id (PK)         │
│ condition_id(FK)│     │ prediction_id   │     │ position_id(FK) │
│ side (YES/NO)   │     │ query_id        │     │ side            │
│ entry_price     │     │ reliability     │     │ price           │
│ current_price   │     │ recency         │     │ amount          │
│ size            │     │ relevance       │     │ status          │
│ entry_date      │     │ specificity     │     │ order_id        │
│ status          │     │ direction       │     │ executed_at     │
│ thesis          │     │ strength        │     │ created_at      │
│ exit_price      │     │ reasoning       │     └─────────────────┘
│ exit_date       │     │ created_at      │
│ pnl             │     └─────────────────┘
│ created_at      │
│ updated_at      │
└─────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  pipeline_runs  │     │     alerts      │     │     config      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │     │ key (PK)        │
│ status          │     │ position_id(FK) │     │ value           │
│ started_at      │     │ type            │     │ description     │
│ completed_at    │     │ message         │     │ updated_at      │
│ markets_scanned │     │ acknowledged    │     │ updated_by      │
│ predictions_made│     │ created_at      │     └─────────────────┘
│ trades_executed │     └─────────────────┘
│ error_message   │
│ created_at      │
└─────────────────┘
```

### 6.2 Table Definitions

```sql
-- Events (Polymarket events)
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    polymarket_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    slug VARCHAR(255),
    description TEXT,
    end_date TIMESTAMP,
    volume DECIMAL(20, 2),
    liquidity DECIMAL(20, 2),
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conditions (Individual outcomes within events)
CREATE TABLE conditions (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES events(id),
    condition_id VARCHAR(255) UNIQUE NOT NULL,
    question TEXT NOT NULL,
    yes_price DECIMAL(10, 6),
    volume DECIMAL(20, 2),
    liquidity DECIMAL(20, 2),
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Predictions (Pipeline outputs)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    condition_id INTEGER REFERENCES conditions(id),
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
    market_price DECIMAL(10, 6) NOT NULL,
    predicted_price DECIMAL(10, 6) NOT NULL,
    edge DECIMAL(10, 6) NOT NULL,
    edge_percent DECIMAL(10, 4),
    apy DECIMAL(10, 4),
    days_until_end INTEGER,
    predictability DECIMAL(10, 6),
    recommendation VARCHAR(20),
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Positions (Open bets)
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    condition_id INTEGER REFERENCES conditions(id),
    side VARCHAR(10) NOT NULL CHECK (side IN ('YES', 'NO')),
    entry_price DECIMAL(10, 6) NOT NULL,
    current_price DECIMAL(10, 6),
    size DECIMAL(20, 6) NOT NULL,
    cost_basis DECIMAL(20, 6) NOT NULL,
    current_value DECIMAL(20, 6),
    unrealized_pnl DECIMAL(20, 6),
    entry_date TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'RESOLVED')),
    thesis TEXT,
    exit_price DECIMAL(10, 6),
    exit_date TIMESTAMP,
    realized_pnl DECIMAL(20, 6),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trades (Execution history)
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id),
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price DECIMAL(10, 6) NOT NULL,
    amount DECIMAL(20, 6) NOT NULL,
    total_cost DECIMAL(20, 6) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')),
    order_id VARCHAR(255),
    tx_hash VARCHAR(255),
    error_message TEXT,
    executed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Evidence Scores (Detailed analysis)
CREATE TABLE evidence_scores (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    query_id VARCHAR(255) NOT NULL,
    query_text TEXT,
    source_url TEXT,
    reliability DECIMAL(10, 6),
    recency DECIMAL(10, 6),
    relevance DECIMAL(10, 6),
    specificity DECIMAL(10, 6),
    direction VARCHAR(20),
    strength DECIMAL(10, 6),
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pipeline Runs (Execution history)
CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'RUNNING' CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    trigger VARCHAR(50),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    markets_scanned INTEGER DEFAULT 0,
    conditions_analyzed INTEGER DEFAULT 0,
    predictions_made INTEGER DEFAULT 0,
    trades_executed INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id),
    type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'INFO' CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL')),
    message TEXT NOT NULL,
    data JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Configuration
CREATE TABLE config (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- Indexes
CREATE INDEX idx_conditions_event_id ON conditions(event_id);
CREATE INDEX idx_predictions_condition_id ON predictions(condition_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_condition_id ON positions(condition_id);
CREATE INDEX idx_trades_position_id ON trades(position_id);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
```

---

## 7. API Specification

### 7.1 Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Pipeline** |||
| POST | `/api/pipeline/run` | Trigger full pipeline |
| GET | `/api/pipeline/status/{run_id}` | Get pipeline run status |
| GET | `/api/pipeline/runs` | List pipeline runs |
| **Predictions** |||
| GET | `/api/predictions` | List predictions (with filters) |
| GET | `/api/predictions/{id}` | Get prediction details |
| GET | `/api/predictions/{id}/evidence` | Get evidence scores |
| **Positions** |||
| GET | `/api/positions` | List positions (open/closed) |
| GET | `/api/positions/{id}` | Get position details |
| POST | `/api/positions/{id}/close` | Close position |
| **Trades** |||
| GET | `/api/trades` | List trades |
| GET | `/api/trades/pending` | List pending trades |
| POST | `/api/trades/{id}/approve` | Approve pending trade |
| POST | `/api/trades/{id}/reject` | Reject pending trade |
| **Config** |||
| GET | `/api/config` | Get all configuration |
| PUT | `/api/config/{key}` | Update configuration |
| **Health** |||
| GET | `/api/health` | Health check |
| GET | `/api/health/db` | Database health |

### 7.2 Request/Response Examples

#### POST /api/pipeline/run

**Request:**
```json
{
  "mode": "full",           // "full" | "positions_only"
  "limit": 50,              // max conditions to analyze
  "dry_run": false,         // don't execute trades
  "force_refresh": false    // ignore cache
}
```

**Response:**
```json
{
  "run_id": 123,
  "status": "RUNNING",
  "started_at": "2026-02-05T00:00:00Z",
  "estimated_completion": "2026-02-05T00:30:00Z"
}
```

#### GET /api/predictions

**Request:**
```
GET /api/predictions?min_edge=0.05&recommendation=BUY_YES&sort=apy&limit=20
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 456,
      "condition_id": "0x1234...",
      "event_title": "Will X happen by Y?",
      "outcome_question": "Yes",
      "market_price": 0.35,
      "predicted_price": 0.52,
      "edge": 0.17,
      "edge_percent": 48.57,
      "apy": 156.2,
      "days_until_end": 45,
      "predictability": 0.72,
      "recommendation": "BUY_YES",
      "created_at": "2026-02-05T00:15:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20
}
```

#### GET /api/positions

**Response:**
```json
{
  "positions": [
    {
      "id": 789,
      "condition_id": "0x5678...",
      "event_title": "Will Z happen?",
      "side": "YES",
      "entry_price": 0.42,
      "current_price": 0.48,
      "size": 100.0,
      "cost_basis": 42.0,
      "current_value": 48.0,
      "unrealized_pnl": 6.0,
      "unrealized_pnl_percent": 14.28,
      "entry_date": "2026-02-01T12:00:00Z",
      "days_held": 4,
      "status": "OPEN",
      "thesis": "Evidence suggests higher probability..."
    }
  ],
  "summary": {
    "total_positions": 5,
    "total_cost_basis": 250.0,
    "total_current_value": 285.0,
    "total_unrealized_pnl": 35.0,
    "total_unrealized_pnl_percent": 14.0
  }
}
```

---

## 8. Security Requirements

### 8.1 Critical Security Items

| Item | Requirement | Implementation |
|------|-------------|----------------|
| **Private Key** | Never exposed in code/logs | GCP Secret Manager |
| **API Keys** | Rotatable, not in code | GCP Secret Manager |
| **Database** | Encrypted at rest | Cloud SQL encryption |
| **API Access** | Authenticated | API key or JWT |
| **Frontend** | Authenticated | Password + 2FA |
| **Network** | HTTPS only | Cloud Run default |

### 8.2 Private Key Handling

```python
# NEVER do this:
PRIVATE_KEY = "0x1234..."  # ❌ In code

# ALWAYS do this:
from google.cloud import secretmanager

def get_private_key():
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/polygon-private-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

### 8.3 Risk Controls

| Control | Description | Default |
|---------|-------------|---------|
| Max position size | Maximum $ per single bet | $100 |
| Max total exposure | Maximum $ across all positions | $1000 |
| Max daily trades | Maximum trades per day | 10 |
| Min edge threshold | Minimum edge to consider | 5% |
| Approval mode | Require manual approval | ON |

---

## 9. Infrastructure & Deployment

### 9.1 GCP Services

| Service | Purpose | Estimated Cost |
|---------|---------|----------------|
| Cloud Run | Backend API | $10-50/month |
| Cloud SQL (PostgreSQL) | Database | $15-30/month |
| Cloud Scheduler | Cron jobs | $0.10/job/month |
| Secret Manager | Credentials | $0.06/secret/month |
| Cloud Storage | Backups | $1-5/month |
| Cloud Monitoring | Logs/alerts | $0-20/month |

**Estimated Total:** $30-120/month (depending on usage)

### 9.2 Deployment Configuration

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**cloudbuild.yaml:**
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/polymarket-analyst', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/polymarket-analyst']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'polymarket-analyst'
      - '--image'
      - 'gcr.io/$PROJECT_ID/polymarket-analyst'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
```

### 9.3 Environment Variables

```bash
# GCP
PROJECT_ID=your-project-id
REGION=us-central1

# Database
DATABASE_URL=postgresql://user:pass@host:5432/polymarket

# API Keys (stored in Secret Manager, referenced here)
OPENAI_API_KEY=sm://polymarket/openai-api-key
PERPLEXITY_API_KEY=sm://polymarket/perplexity-api-key
POLYGON_PRIVATE_KEY=sm://polymarket/polygon-private-key

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
APPROVAL_MODE=true
```

---

## 10. Success Metrics

### 10.1 Key Performance Indicators (KPIs)

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| **ROI** | (Total P&L / Capital Deployed) * 100 | > 15% annualized | Monthly |
| **Win Rate** | Profitable trades / Total trades | > 55% | Weekly |
| **Brier Score** | Mean squared error of predictions | < 0.20 | Monthly |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 | Monthly |
| **Avg Edge Captured** | Actual return / Predicted edge | > 70% | Monthly |
| **System Uptime** | Time operational / Total time | > 99% | Daily |
| **Pipeline Success Rate** | Successful runs / Total runs | > 95% | Daily |

### 10.2 Tracking Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE SUMMARY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Portfolio Value:  $1,234.56  (+$234.56 / +23.5%)              │
│  Open Positions:   8                                            │
│  Unrealized P&L:   +$45.23                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  P&L Over Time                                          │   │
│  │  $300 ┤                                    ╭────────    │   │
│  │  $200 ┤                         ╭─────────╯             │   │
│  │  $100 ┤              ╭─────────╯                        │   │
│  │    $0 ┼──────────────╯                                  │   │
│  │       └──────────────────────────────────────────────   │   │
│  │        Jan        Feb        Mar        Apr             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Win Rate: 62%    Avg Edge: 8.3%    Brier: 0.18    Sharpe: 1.4 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Timeline & Milestones

### 11.1 Project Timeline

```
Week 1-2: Phase 1 - Foundation
├── Day 1-2: Project setup, database schema
├── Day 3-5: FastAPI backend, orchestrator
├── Day 6-8: Cloud Run deployment
├── Day 9-10: Cloud Scheduler integration
└── Day 11-14: Testing, documentation

Week 3-4: Phase 2 - Trade Execution
├── Day 15-17: Polymarket API integration
├── Day 18-20: Order placement & monitoring
├── Day 21-23: Risk manager implementation
├── Day 24-26: Paper trading mode
└── Day 27-28: Testing with small amounts

Week 5-6: Phase 3 - Position Monitoring
├── Day 29-31: Position tracker
├── Day 32-34: Price monitoring & alerts
├── Day 35-37: Re-analysis triggers
├── Day 38-40: Resolution tracking
└── Day 41-42: Integration testing

Week 7-9: Phase 4 - Frontend Dashboard
├── Day 43-47: Core dashboard pages
├── Day 48-52: Analytics & charts
├── Day 53-56: Trade approval UI
├── Day 57-60: Settings & config UI
└── Day 61-63: Testing & polish

Week 10: Launch & Monitoring
├── Day 64-66: Final testing
├── Day 67-68: Documentation
├── Day 69-70: Go live (small scale)
```

### 11.2 Milestones

| Milestone | Date | Deliverables |
|-----------|------|--------------|
| M1: Backend MVP | Week 2 | API + DB + Pipeline on Cloud Run |
| M2: Trade Execution | Week 4 | Can place real bets (manual approval) |
| M3: Position Monitoring | Week 6 | Alerts + re-analysis working |
| M4: Dashboard v1 | Week 9 | Full frontend operational |
| M5: Production Launch | Week 10 | Live trading with small amounts |

---

## 12. Risks & Mitigations

### 12.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Private key compromise** | Low | Critical | Secret Manager, minimal permissions |
| **API rate limits** | Medium | High | Caching, backoff, multiple keys |
| **Prediction model failure** | Medium | High | Monitoring, manual override |
| **Market manipulation** | Low | Medium | Volume/liquidity filters |
| **Regulatory changes** | Low | High | Monitor Polymarket status |
| **Cloud service outage** | Low | Medium | Multi-region, manual fallback |
| **Unexpected losses** | Medium | Medium | Position limits, stop-loss |

### 12.2 Contingency Plans

**If private key is compromised:**
1. Immediately rotate key in Secret Manager
2. Transfer remaining funds to new wallet
3. Audit all trades for unauthorized activity
4. Review access logs

**If prediction accuracy degrades:**
1. Pause automated trading
2. Switch to manual approval mode
3. Analyze recent predictions for patterns
4. Adjust model parameters or retrain

**If Polymarket becomes unavailable:**
1. System pauses automatically
2. Positions remain on blockchain
3. Manual withdrawal via direct contract interaction
4. Consider alternative platforms

---

## 13. Open Questions

### 13.1 Technical

| # | Question | Impact | Decision Needed By |
|---|----------|--------|-------------------|
| 1 | Which Polymarket trading library to use? (py-clob-client vs custom) | Phase 2 | Week 2 |
| 2 | How to handle partial order fills? | Phase 2 | Week 3 |
| 3 | What's the minimum bet size on Polymarket? | Phase 2 | Week 2 |
| 4 | Should we use Cloud Run Jobs vs Cloud Run Services for pipeline? | Phase 1 | Week 1 |

### 13.2 Business

| # | Question | Impact | Decision Needed By |
|---|----------|--------|-------------------|
| 1 | Starting capital allocation? | Phase 2 | Week 3 |
| 2 | Maximum single position size? | Phase 2 | Week 3 |
| 3 | At what loss threshold do we pause trading? | Phase 2 | Week 4 |
| 4 | How long to run paper trading before real money? | Phase 2 | Week 4 |

### 13.3 Product

| # | Question | Impact | Decision Needed By |
|---|----------|--------|-------------------|
| 1 | Should the frontend be mobile-responsive? | Phase 4 | Week 5 |
| 2 | Do we need email/SMS alerts or just dashboard? | Phase 3 | Week 4 |
| 3 | Should we track prediction accuracy by category? | Phase 4 | Week 6 |

---

## Appendix A: Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Backend** | Python 3.11 + FastAPI | Existing codebase, async support |
| **Database** | PostgreSQL | Robust, GCP-native |
| **ORM** | SQLAlchemy | Python standard |
| **Frontend** | Next.js 14 + React | Modern, Vercel integration |
| **UI Components** | shadcn/ui | Clean, customizable |
| **Charts** | Recharts | React-native charts |
| **Deployment** | Cloud Run + Vercel | Serverless, cost-effective |
| **CI/CD** | Cloud Build | GCP-native |

---

## Appendix B: Directory Structure

```
polymarket-analyst/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # App entry point
│   ├── config.py                 # Configuration
│   ├── dependencies.py           # Dependency injection
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── predictions.py
│   │   ├── positions.py
│   │   ├── trades.py
│   │   ├── config.py
│   │   └── health.py
│   ├── models/                   # Pydantic models (API)
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── prediction.py
│   │   ├── position.py
│   │   └── trade.py
│   └── services/                 # Business logic
│       ├── __init__.py
│       ├── pipeline_service.py
│       ├── prediction_service.py
│       └── position_service.py
│
├── src/                          # Existing pipeline scripts
│   ├── get_markets.py
│   ├── generate_queries.py
│   ├── execute_searches.py
│   ├── analyze_evidence.py
│   ├── compute_prediction.py
│   ├── orchestrator.py           # NEW: Runs full pipeline
│   ├── trading/                  # NEW: Trade execution
│   │   ├── __init__.py
│   │   ├── polymarket_client.py
│   │   ├── order_executor.py
│   │   └── risk_manager.py
│   └── monitoring/               # NEW: Position monitoring
│       ├── __init__.py
│       ├── position_tracker.py
│       ├── price_monitor.py
│       └── alert_engine.py
│
├── db/                           # Database
│   ├── __init__.py
│   ├── connection.py
│   ├── models.py                 # SQLAlchemy models
│   └── migrations/
│       └── ...
│
├── frontend/                     # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx
│   │   ├── positions/
│   │   ├── predictions/
│   │   ├── trades/
│   │   ├── analytics/
│   │   └── settings/
│   ├── components/
│   ├── lib/
│   └── package.json
│
├── infrastructure/               # Deployment configs
│   ├── Dockerfile
│   ├── cloudbuild.yaml
│   ├── cloudrun.yaml
│   └── scheduler.yaml
│
├── data/                         # Pipeline outputs (local dev)
│   ├── candidate_markets.json
│   ├── market_queries.json
│   ├── search_results.json
│   ├── evidence_analysis.json
│   └── predictions.json
│
├── tests/
│   ├── api/
│   ├── src/
│   └── integration/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── PRD.md                        # This document
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-05 | Alex Guo | Initial draft |

---

*End of PRD*

