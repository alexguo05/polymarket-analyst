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
    days_until_end = Column(Integer)
    apy = Column(Float)
    
    # Analysis
    predictability = Column(Float)
    recommendation = Column(Enum(Recommendation))
    reasoning = Column(Text)
    
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


class EvidenceScore(Base):
    """Evidence scoring from analysis."""
    __tablename__ = "evidence_scores"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    
    query_id = Column(String(255), nullable=False)
    query_text = Column(Text)
    key_finding = Column(Text)
    
    # Quality scores
    reliability = Column(Float)
    recency = Column(Float)
    relevance = Column(Float)
    specificity = Column(Float)
    
    # Direction (derived strength in compute_prediction)
    direction = Column(Integer)  # -1, 0, 1
    direction_reason = Column(Text)
    
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
    
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="pipeline_run")
    
    __table_args__ = (
        Index("idx_pipeline_runs_status", "status"),
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

