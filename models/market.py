"""
Market models - data structures from Polymarket API.

These models represent the output of get_markets.py (Step 1).
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Outcome(BaseModel):
    """A single outcome within an event (what you can bet on)."""
    question: str
    yes_price: Optional[float] = None  # Can be None for some outcomes
    condition_id: str
    yes_token_id: Optional[str] = None  # Token ID for YES shares (for trading)
    no_token_id: Optional[str] = None   # Token ID for NO shares (for trading)
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[str] = None


class CandidateMarket(BaseModel):
    """
    A market that passed initial filters.
    
    This is the output of get_markets.py scan_markets() function.
    Contains multiple outcomes, each of which can be bet on separately.
    """
    event_id: str
    title: str
    slug: Optional[str] = None
    end_date: Optional[str] = None
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    outcomes: list[Outcome]
    tags: Optional[list[str]] = None
    
    # Filter metadata (why it passed/failed)
    filter_scores: Optional[dict] = None


class FilteredCondition(BaseModel):
    """
    A single condition (outcome) ready for query generation.
    
    This flattens the CandidateMarket structure - one entry per outcome.
    This is what flows into generate_queries.py.
    """
    # Condition identifiers
    condition_id: str
    event_id: str
    
    # Display info
    event_title: str
    outcome_question: str
    
    # Token IDs (for trading)
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    
    # Market data
    yes_price: float
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[str] = None
    
    @classmethod
    def from_market_and_outcome(
        cls, 
        market: CandidateMarket, 
        outcome: Outcome
    ) -> "FilteredCondition":
        """Create a FilteredCondition from a market and one of its outcomes."""
        return cls(
            condition_id=outcome.condition_id,
            event_id=market.event_id,
            event_title=market.title,
            outcome_question=outcome.question,
            yes_token_id=outcome.yes_token_id,
            no_token_id=outcome.no_token_id,
            yes_price=outcome.yes_price,
            volume=outcome.volume,
            liquidity=outcome.liquidity,
            end_date=outcome.end_date or market.end_date,
        )

