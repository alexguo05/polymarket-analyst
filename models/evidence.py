"""
Evidence models - search results and analysis.

These models represent:
- Output of execute_searches.py (Step 3): ConditionSearchResults
- Output of analyze_evidence.py (Step 4): ConditionAnalysis
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# SEARCH RESULTS (from execute_searches.py)
# =============================================================================

class QueryResult(BaseModel):
    """Result of executing a single search query."""
    query_id: str
    query: str
    category: str
    response: str  # Full text response from Perplexity
    citations: list[str]  # URLs found
    searched_at: str  # ISO timestamp


class ConditionSearchResults(BaseModel):
    """
    All search results for a single condition.
    
    This is the output of execute_searches.py for one condition.
    """
    # Condition identifiers (passed through)
    condition_id: str
    event_id: str
    event_title: str
    outcome_question: str
    
    # Market data (passed through)
    yes_price: float
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[str] = None
    
    # Search results
    query_results: list[QueryResult]
    
    # Metadata
    searched_at: str  # ISO timestamp
    
    # Cost tracking (total for all queries in this condition)
    total_requests: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


# =============================================================================
# EVIDENCE ANALYSIS (from analyze_evidence.py)
# =============================================================================

class EvidenceScore(BaseModel):
    """
    GPT's scoring of a single piece of evidence.
    
    Note: 'strength' is NOT included - it's derived from quality metrics
    in compute_prediction.py to avoid LLM probability estimation.
    """
    query_id: str
    key_finding: str  # One sentence summary
    
    # Quality metrics (0.0 - 1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    reliability_reason: str
    
    recency: float = Field(ge=0.0, le=1.0)
    recency_reason: str
    
    relevance: float = Field(ge=0.0, le=1.0)
    relevance_reason: str
    
    specificity: float = Field(ge=0.0, le=1.0)
    specificity_reason: str
    
    # Direction only (simple classification, not magnitude)
    direction: int = Field(ge=-1, le=1)  # -1=NO, 0=neutral, 1=YES
    direction_reason: str


class ConditionAnalysis(BaseModel):
    """
    Complete analysis for a single condition.
    
    This is the output of analyze_evidence.py for one condition.
    """
    # Condition identifiers (passed through)
    condition_id: str
    event_id: str
    event_title: str
    outcome_question: str
    
    # Market data at analysis time
    market_price: float  # yes_price at time of analysis
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[str] = None
    
    # Predictability assessment
    predictability: float = Field(ge=0.0, le=1.0)
    predictability_reason: str
    
    # Evidence scores
    evidence_scores: list[EvidenceScore]
    
    # Metadata
    analyzed_at: str  # ISO timestamp
    model: str  # Which model did the analysis
    
    # Cost tracking
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None  # Hidden "thinking" tokens for reasoning models
    cost_usd: Optional[float] = None

