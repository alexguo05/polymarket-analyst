"""
Prediction models - final probability predictions.

These models represent the output of compute_prediction.py (Step 5).
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class EvidenceBreakdown(BaseModel):
    """Breakdown of how a single piece of evidence contributed."""
    query_id: str
    key_finding: str
    quality: float  # Average of reliability, recency, relevance, specificity
    direction: int  # -1, 0, or 1
    strength: float  # Derived from quality (not from LLM)
    raw_adjustment: float  # direction * strength


class Prediction(BaseModel):
    """
    Final probability prediction for a single condition.
    
    This is the output of compute_prediction.py for one condition.
    """
    # Condition identifiers
    condition_id: str
    event_id: str
    event_title: str
    outcome_question: str
    
    # Prices
    market_price: float  # Current market price
    predicted_price: float  # Our estimate
    
    # Edge
    edge: float  # predicted - market
    edge_percent: float  # (edge / market) * 100
    
    # Time value
    end_date: Optional[str] = None
    days_until_end: Optional[float] = None
    apy: Optional[float] = None  # Annualized return
    
    # Analysis metadata
    predictability: float
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    
    # Evidence summary
    num_evidence: int
    total_raw_adjustment: float
    scaled_adjustment: float
    evidence_breakdown: list[EvidenceBreakdown]
    
    # Recommendation
    direction: Literal["YES", "NO", "HOLD"]
    
    # Metadata
    computed_at: str  # ISO timestamp


class CostSummary(BaseModel):
    """Cost breakdown for the pipeline run."""
    query_generation: float = 0.0  # Cost of GPT-5-mini
    search_execution: float = 0.0  # Cost of Perplexity Sonar
    evidence_analysis: float = 0.0  # Cost of GPT-5-pro
    total_usd: float = 0.0
    
    total_requests: int = 0
    total_tokens: int = 0


class PipelineOutput(BaseModel):
    """
    Complete output of the pipeline run.
    
    This wraps all predictions with metadata about the run.
    """
    computed_at: str  # ISO timestamp
    
    # Parameters used
    parameters: dict
    
    # Summary statistics
    summary: dict
    
    # Cost tracking
    cost: Optional[CostSummary] = None
    
    # All predictions
    predictions: list[Prediction]

