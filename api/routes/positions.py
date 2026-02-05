"""Positions management endpoints."""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import Position, PositionStatus, PositionSide, Condition, Event

router = APIRouter()


class PositionSummary(BaseModel):
    """Summary statistics for positions."""
    total_positions: int
    open_positions: int
    closed_positions: int
    resolved_positions: int
    total_cost_basis: float
    total_current_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float


@router.get("")
async def list_positions(
    status: Optional[str] = Query(None, description="Filter by status (OPEN, CLOSED, RESOLVED)"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all positions with optional status filter."""
    query = db.query(Position).join(Condition).join(Event)
    
    if status:
        try:
            pos_status = PositionStatus(status)
            query = query.filter(Position.status == pos_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    total = query.count()
    positions = query.order_by(Position.entry_date.desc()).offset(offset).limit(limit).all()
    
    # Calculate summary
    all_positions = db.query(Position).all()
    summary = _calculate_summary(all_positions)
    
    return {
        "positions": [
            {
                "id": p.id,
                "condition_id": p.condition.condition_id,
                "event_title": p.condition.event.title,
                "outcome_question": p.condition.question,
                "side": p.side.value,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "size": p.size,
                "cost_basis": p.cost_basis,
                "current_value": p.current_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_percent": (
                    (p.unrealized_pnl / p.cost_basis * 100) 
                    if p.cost_basis and p.unrealized_pnl else 0
                ),
                "entry_date": p.entry_date.isoformat(),
                "status": p.status.value,
                "thesis": p.thesis,
            }
            for p in positions
        ],
        "summary": summary.model_dump(),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _calculate_summary(positions: list[Position]) -> PositionSummary:
    """Calculate summary statistics for positions."""
    open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
    closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
    resolved_positions = [p for p in positions if p.status == PositionStatus.RESOLVED]
    
    return PositionSummary(
        total_positions=len(positions),
        open_positions=len(open_positions),
        closed_positions=len(closed_positions),
        resolved_positions=len(resolved_positions),
        total_cost_basis=sum(p.cost_basis or 0 for p in open_positions),
        total_current_value=sum(p.current_value or 0 for p in open_positions),
        total_unrealized_pnl=sum(p.unrealized_pnl or 0 for p in open_positions),
        total_realized_pnl=sum(p.realized_pnl or 0 for p in closed_positions + resolved_positions),
    )


@router.get("/{position_id}")
async def get_position(position_id: int, db: Session = Depends(get_db)):
    """Get detailed position information including trades."""
    position = db.query(Position).filter(Position.id == position_id).first()
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    return {
        "id": position.id,
        "condition_id": position.condition.condition_id,
        "event_title": position.condition.event.title,
        "outcome_question": position.condition.question,
        "side": position.side.value,
        "entry_price": position.entry_price,
        "current_price": position.current_price,
        "size": position.size,
        "cost_basis": position.cost_basis,
        "current_value": position.current_value,
        "unrealized_pnl": position.unrealized_pnl,
        "entry_date": position.entry_date.isoformat(),
        "status": position.status.value,
        "exit_price": position.exit_price,
        "exit_date": position.exit_date.isoformat() if position.exit_date else None,
        "realized_pnl": position.realized_pnl,
        "thesis": position.thesis,
        "trades": [
            {
                "id": t.id,
                "side": t.side.value,
                "price": t.price,
                "amount": t.amount,
                "total_cost": t.total_cost,
                "status": t.status.value,
                "executed_at": t.executed_at.isoformat() if t.executed_at else None,
            }
            for t in position.trades
        ],
        "alerts": [
            {
                "id": a.id,
                "type": a.type,
                "severity": a.severity.value,
                "message": a.message,
                "acknowledged": a.acknowledged,
                "created_at": a.created_at.isoformat(),
            }
            for a in position.alerts
        ],
    }


@router.post("/{position_id}/close")
async def close_position(position_id: int, db: Session = Depends(get_db)):
    """
    Mark a position as closed.
    
    Note: This doesn't execute a trade - it just updates the status.
    Actual trade execution will be implemented in Phase 2.
    """
    position = db.query(Position).filter(Position.id == position_id).first()
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    if position.status != PositionStatus.OPEN:
        raise HTTPException(
            status_code=400, 
            detail=f"Position is not open (status: {position.status.value})"
        )
    
    position.status = PositionStatus.CLOSED
    position.exit_date = datetime.now(timezone.utc)
    position.exit_price = position.current_price
    
    if position.current_value and position.cost_basis:
        position.realized_pnl = position.current_value - position.cost_basis
    
    db.commit()
    
    return {
        "message": "Position closed",
        "position_id": position_id,
        "realized_pnl": position.realized_pnl,
    }

