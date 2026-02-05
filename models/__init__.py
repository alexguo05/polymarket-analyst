"""
Shared Pydantic models for the Polymarket Analyst pipeline.

These models define the data structures that flow between pipeline steps,
ensuring type safety and validation across the entire system.
"""

from models.market import (
    Outcome,
    CandidateMarket,
    FilteredCondition,
)
from models.query import (
    SearchQuery,
    ConditionQueries,
)
from models.evidence import (
    QueryResult,
    ConditionSearchResults,
    EvidenceScore,
    ConditionAnalysis,
)
from models.prediction import (
    EvidenceBreakdown,
    Prediction,
    PipelineOutput,
)

__all__ = [
    # Market
    "Outcome",
    "CandidateMarket", 
    "FilteredCondition",
    # Query
    "SearchQuery",
    "ConditionQueries",
    # Evidence
    "QueryResult",
    "ConditionSearchResults",
    "EvidenceScore",
    "ConditionAnalysis",
    # Prediction
    "EvidenceBreakdown",
    "Prediction",
    "PipelineOutput",
]

