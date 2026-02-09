"""SQLAlchemy ORM models for Polymarket Analyst."""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, 
    ForeignKey, Enum, Index, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# ENUMS
# =============================================================================

class PositionStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


class PositionSide(str, enum.Enum):
    YES = "YES"
    NO = "NO"


class TradeSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(str, enum.Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class PipelineStatus(str, enum.Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AlertSeverity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Recommendation(str, enum.Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


# =============================================================================
# MODELS
# =============================================================================

class Event(Base):
    """Polymarket event (contains multiple conditions)."""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    polymarket_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    slug = Column(String(255))
    description = Column(Text)
    end_date = Column(DateTime)
    volume = Column(Float)
    liquidity = Column(Float)
    tags = Column(JSON)  # Store as JSON array
    
    # Filter scores from scanning
    complexity_score = Column(Float)  # How complex/analyzable is this event
    edge_potential = Column(String(20))  # "high", "medium", "low"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conditions = relationship("Condition", back_populates="event", cascade="all, delete-orphan")


class Condition(Base):
    """Individual outcome within an event (what you bet on)."""
    __tablename__ = "conditions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    condition_id = Column(String(255), unique=True, nullable=False, index=True)
    question = Column(Text, nullable=False)
    yes_token_id = Column(String(255))  # Token ID for YES shares (for trading)
    no_token_id = Column(String(255))   # Token ID for NO shares (for trading)
    yes_price = Column(Float)
    volume = Column(Float)
    liquidity = Column(Float)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    event = relationship("Event", back_populates="conditions")
    predictions = relationship("Prediction", back_populates="condition", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="condition", cascade="all, delete-orphan")


class Prediction(Base):
    """Pipeline prediction output."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, ForeignKey("conditions.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    
    # Prices
    market_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    
    # Edge
    edge = Column(Float, nullable=False)
    edge_percent = Column(Float)
    
    # Time value
    days_until_end = Column(Float)  # Changed to Float for decimals
    apy = Column(Float)
    
    # Analysis
    predictability = Column(Float)
    predictability_reason = Column(Text)  # LLM reasoning for predictability
    recommendation = Column(Enum(Recommendation))
    
    # Computed values
    num_evidence = Column(Integer)  # Count of evidence items
    total_raw_adjustment = Column(Float)  # Sum of all raw adjustments
    scaled_adjustment = Column(Float)  # Final scaled shift applied
    
    # Model info
    analysis_model = Column(String(100))  # e.g., "gpt-5.2"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    condition = relationship("Condition", back_populates="predictions")
    pipeline_run = relationship("PipelineRun", back_populates="predictions")
    evidence_scores = relationship("EvidenceScore", back_populates="prediction", cascade="all, delete-orphan")
    
    # Index for querying recent predictions
    __table_args__ = (
        Index("idx_predictions_created_at", "created_at"),
    )


class Position(Base):
    """Open bet position."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, ForeignKey("conditions.id"), nullable=False)
    
    # Position details
    side = Column(Enum(PositionSide), nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    size = Column(Float, nullable=False)  # Number of shares
    cost_basis = Column(Float, nullable=False)  # Total $ spent
    current_value = Column(Float)
    unrealized_pnl = Column(Float)
    
    # Timing
    entry_date = Column(DateTime, nullable=False)
    status = Column(Enum(PositionStatus), default=PositionStatus.OPEN)
    
    # Exit info (if closed)
    exit_price = Column(Float)
    exit_date = Column(DateTime)
    realized_pnl = Column(Float)
    
    # Reasoning
    thesis = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    condition = relationship("Condition", back_populates="positions")
    trades = relationship("Trade", back_populates="position", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="position", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_positions_status", "status"),
    )


class Trade(Base):
    """Trade execution record."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False)
    
    # Trade details
    side = Column(Enum(TradeSide), nullable=False)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    
    # Status
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    order_id = Column(String(255))
    tx_hash = Column(String(255))
    error_message = Column(Text)
    
    executed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    position = relationship("Position", back_populates="trades")
    
    __table_args__ = (
        Index("idx_trades_status", "status"),
    )


class GeneratedQuery(Base):
    """Generated search queries for a condition."""
    __tablename__ = "generated_queries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, ForeignKey("conditions.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    
    query_id = Column(String(255), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    purpose = Column(Text)  # Why this query was chosen
    category = Column(String(50))  # "base_rate", "procedural", "recent_developments", etc.
    priority = Column(String(20))  # "high", "medium", "low"
    
    # Generation metadata
    model = Column(String(100))  # e.g., "gpt-5-mini"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    condition = relationship("Condition")
    pipeline_run = relationship("PipelineRun")


class SearchResult(Base):
    """Raw search query results from Perplexity."""
    __tablename__ = "search_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, ForeignKey("conditions.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    
    query_id = Column(String(255), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    category = Column(String(50))  # "base_rate", "procedural", etc.
    
    # Response from Perplexity
    response_text = Column(Text)  # The full answer text
    citations = Column(JSON)  # List of citation URLs
    
    # Metadata
    model = Column(String(100))  # e.g., "sonar", "sonar-pro"
    searched_at = Column(DateTime)  # When search was executed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    condition = relationship("Condition")
    pipeline_run = relationship("PipelineRun")
    
    __table_args__ = (
        Index("idx_search_results_query_id", "query_id"),
    )


class EvidenceScore(Base):
    """Evidence scoring from analysis."""
    __tablename__ = "evidence_scores"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    
    query_id = Column(String(255), nullable=False)
    query_text = Column(Text)
    key_finding = Column(Text)
    
    # Quality scores with reasoning
    reliability = Column(Float)
    reliability_reason = Column(Text)
    
    recency = Column(Float)
    recency_reason = Column(Text)
    
    relevance = Column(Float)
    relevance_reason = Column(Text)
    
    specificity = Column(Float)
    specificity_reason = Column(Text)
    
    # Direction
    direction = Column(Integer)  # -1, 0, 1
    direction_reason = Column(Text)
    
    # Computed values (from compute_prediction)
    quality_score = Column(Float)  # Combined quality metric
    strength = Column(Float)  # Derived strength
    raw_adjustment = Column(Float)  # Probability adjustment for this query
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="evidence_scores")


class PipelineRun(Base):
    """Pipeline execution history."""
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    status = Column(Enum(PipelineStatus), default=PipelineStatus.RUNNING)
    trigger = Column(String(50))  # "scheduled", "manual", "api"
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Stats
    markets_scanned = Column(Integer, default=0)
    conditions_analyzed = Column(Integer, default=0)
    predictions_made = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    
    # Parameters used
    parameters = Column(JSON)  # {"evidence_scale": 0.5, "min_prob": 0.01, "max_prob": 0.99}
    
    # Cost tracking
    query_gen_tokens = Column(Integer, default=0)
    query_gen_cost = Column(Float, default=0.0)
    search_requests = Column(Integer, default=0)
    search_cost = Column(Float, default=0.0)
    analysis_input_tokens = Column(Integer, default=0)
    analysis_output_tokens = Column(Integer, default=0)
    analysis_reasoning_tokens = Column(Integer, default=0)
    analysis_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Models used
    query_model = Column(String(100))  # e.g., "gpt-5-mini"
    search_model = Column(String(100))  # e.g., "sonar-pro"
    analysis_model = Column(String(100))  # e.g., "gpt-5.2"
    
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="pipeline_run")
    
    __table_args__ = (
        Index("idx_pipeline_runs_status", "status"),
    )


class PriceHistory(Base):
    """Historical price snapshots for conditions."""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(Integer, ForeignKey("conditions.id"), nullable=False)
    
    yes_price = Column(Float, nullable=False)
    volume = Column(Float)
    liquidity = Column(Float)
    
    # Source of this snapshot
    source = Column(String(50))  # "pipeline", "monitor", "api"
    
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    condition = relationship("Condition")
    
    __table_args__ = (
        Index("idx_price_history_condition_time", "condition_id", "recorded_at"),
    )


class Alert(Base):
    """Alert for position monitoring."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    
    type = Column(String(50), nullable=False)  # "price_move", "thesis_invalid", etc.
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.INFO)
    message = Column(Text, nullable=False)
    data = Column(JSON)
    
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    position = relationship("Position", back_populates="alerts")
    
    __table_args__ = (
        Index("idx_alerts_acknowledged", "acknowledged"),
    )


class Config(Base):
    """Runtime configuration storage."""
    __tablename__ = "config"
    
    key = Column(String(255), primary_key=True)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(255))

