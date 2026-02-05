"""Predictions endpoints."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import Prediction, Condition, Event, Recommendation
from config.settings import settings

router = APIRouter()


@router.get("")
async def list_predictions(
    min_edge: Optional[float] = Query(None, description="Minimum absolute edge"),
    recommendation: Optional[str] = Query(None, description="Filter by recommendation (BUY_YES, BUY_NO, HOLD)"),
    sort: str = Query("edge", description="Sort by: edge, apy, predictability, market"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List predictions with optional filters.
    
    If database is empty, falls back to reading from JSON file.
    """
    # Try database first
    query = db.query(Prediction).join(Condition).join(Event)
    
    # Apply filters
    if min_edge is not None:
        from sqlalchemy import func
        query = query.filter(func.abs(Prediction.edge) >= min_edge)
    
    if recommendation:
        try:
            rec = Recommendation(recommendation)
            query = query.filter(Prediction.recommendation == rec)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid recommendation: {recommendation}")
    
    # Apply sorting
    if sort == "edge":
        from sqlalchemy import func
        query = query.order_by(func.abs(Prediction.edge).desc())
    elif sort == "apy":
        query = query.order_by(Prediction.apy.desc().nullslast())
    elif sort == "predictability":
        query = query.order_by(Prediction.predictability.desc().nullslast())
    elif sort == "market":
        query = query.order_by(Prediction.market_price)
    
    total = query.count()
    predictions = query.offset(offset).limit(limit).all()
    
    # If database is empty, try JSON file
    if total == 0:
        return _get_predictions_from_json(min_edge, recommendation, sort, limit, offset)
    
    return {
        "predictions": [
            {
                "id": p.id,
                "condition_id": p.condition.condition_id,
                "event_title": p.condition.event.title,
                "outcome_question": p.condition.question,
                "market_price": p.market_price,
                "predicted_price": p.predicted_price,
                "edge": p.edge,
                "edge_percent": p.edge_percent,
                "apy": p.apy,
                "days_until_end": p.days_until_end,
                "predictability": p.predictability,
                "recommendation": p.recommendation.value if p.recommendation else None,
                "created_at": p.created_at.isoformat(),
            }
            for p in predictions
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _get_predictions_from_json(
    min_edge: Optional[float],
    recommendation: Optional[str],
    sort: str,
    limit: int,
    offset: int,
):
    """Fallback to read predictions from JSON file."""
    json_path = settings.data_dir / "predictions.json"
    
    if not json_path.exists():
        return {
            "predictions": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "source": "json",
            "message": "No predictions found. Run the pipeline first.",
        }
    
    with open(json_path) as f:
        data = json.load(f)
    
    predictions = data.get("predictions", [])
    
    # Apply filters
    if min_edge is not None:
        predictions = [p for p in predictions if abs(p.get("edge", 0)) >= min_edge]
    
    if recommendation:
        predictions = [p for p in predictions if p.get("direction") == recommendation.replace("BUY_", "")]
    
    # Apply sorting
    if sort == "edge":
        predictions.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)
    elif sort == "apy":
        predictions.sort(key=lambda x: x.get("apy") or 0, reverse=True)
    elif sort == "predictability":
        predictions.sort(key=lambda x: x.get("predictability") or 0, reverse=True)
    elif sort == "market":
        predictions.sort(key=lambda x: x.get("market_price", 0.5))
    
    total = len(predictions)
    predictions = predictions[offset:offset + limit]
    
    return {
        "predictions": predictions,
        "total": total,
        "limit": limit,
        "offset": offset,
        "source": "json",
    }


@router.get("/{prediction_id}")
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get detailed prediction including evidence scores."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "id": prediction.id,
        "condition_id": prediction.condition.condition_id,
        "event_title": prediction.condition.event.title,
        "outcome_question": prediction.condition.question,
        "market_price": prediction.market_price,
        "predicted_price": prediction.predicted_price,
        "edge": prediction.edge,
        "edge_percent": prediction.edge_percent,
        "apy": prediction.apy,
        "days_until_end": prediction.days_until_end,
        "predictability": prediction.predictability,
        "recommendation": prediction.recommendation.value if prediction.recommendation else None,
        "reasoning": prediction.reasoning,
        "created_at": prediction.created_at.isoformat(),
        "evidence_scores": [
            {
                "query_id": e.query_id,
                "key_finding": e.key_finding,
                "reliability": e.reliability,
                "recency": e.recency,
                "relevance": e.relevance,
                "specificity": e.specificity,
                "direction": e.direction,
                "direction_reason": e.direction_reason,
            }
            for e in prediction.evidence_scores
        ],
    }


@router.get("/{prediction_id}/evidence")
async def get_prediction_evidence(prediction_id: int, db: Session = Depends(get_db)):
    """Get just the evidence scores for a prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "prediction_id": prediction_id,
        "evidence_scores": [
            {
                "query_id": e.query_id,
                "key_finding": e.key_finding,
                "reliability": e.reliability,
                "recency": e.recency,
                "relevance": e.relevance,
                "specificity": e.specificity,
                "direction": e.direction,
                "direction_reason": e.direction_reason,
            }
            for e in prediction.evidence_scores
        ],
    }

