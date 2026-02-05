"""
Query models - search queries for research.

These models represent the output of generate_queries.py (Step 2).
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A single search query for researching a condition."""
    query_id: str
    query: str  # The actual search query string
    purpose: str  # What evidence this query seeks
    category: Literal[
        "base_rate", 
        "procedural", 
        "sentiment", 
        "recent_developments", 
        "structural_factors",
        "other"
    ]
    priority: Literal["high", "medium", "low"]


class ConditionQueries(BaseModel):
    """
    All search queries generated for a single condition.
    
    This is the output of generate_queries.py for one condition.
    """
    # Condition identifiers (passed through from FilteredCondition)
    condition_id: str
    event_id: str
    event_title: str
    outcome_question: str
    
    # Market data at time of query generation
    yes_price: float
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[str] = None
    
    # Generated queries
    queries: list[SearchQuery]
    
    # Metadata
    generated_at: str  # ISO timestamp
    model: str  # Which model generated these
    
    # Cost tracking
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

