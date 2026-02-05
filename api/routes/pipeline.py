"""Pipeline management endpoints."""
import asyncio
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import PipelineRun, PipelineStatus
from config.settings import settings

router = APIRouter()

# Track running pipelines (in production, use Redis or DB)
_running_pipelines: dict[int, asyncio.Task] = {}


class PipelineRunRequest(BaseModel):
    """Request to start a pipeline run."""
    mode: str = "full"  # "full" or "positions_only"
    limit: Optional[int] = None
    dry_run: bool = False


class PipelineRunResponse(BaseModel):
    """Response after starting a pipeline run."""
    run_id: int
    status: str
    started_at: str
    message: str


class PipelineStatusResponse(BaseModel):
    """Status of a pipeline run."""
    run_id: int
    status: str
    started_at: str
    completed_at: Optional[str]
    markets_scanned: int
    conditions_analyzed: int
    predictions_made: int
    trades_executed: int
    error_message: Optional[str]


def _run_pipeline_sync(run_id: int, mode: str, limit: Optional[int], dry_run: bool):
    """
    Synchronously run the pipeline (called in background task).
    
    This function runs in a separate thread/process.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.orchestrator import PipelineOrchestrator
    from db.connection import SessionLocal
    from db.models import PipelineRun, PipelineStatus
    
    db = SessionLocal()
    
    try:
        orchestrator = PipelineOrchestrator(
            limit=limit,
            dry_run=dry_run,
            verbose=True,
        )
        
        result = orchestrator.run_full_pipeline()
        
        # Update database record
        run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        if run:
            run.status = PipelineStatus.COMPLETED if result.success else PipelineStatus.FAILED
            run.completed_at = result.completed_at
            run.markets_scanned = result.markets_scanned
            run.conditions_analyzed = result.conditions_analyzed
            run.predictions_made = result.predictions_made
            run.error_message = result.error_message
            db.commit()
            
    except Exception as e:
        # Update with error
        run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        if run:
            run.status = PipelineStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/run", response_model=PipelineRunResponse)
async def start_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start a new pipeline run.
    
    The pipeline runs in the background and updates the database with progress.
    Use GET /api/pipeline/status/{run_id} to check status.
    """
    # Check if pipeline is already running
    running = db.query(PipelineRun).filter(
        PipelineRun.status == PipelineStatus.RUNNING
    ).first()
    
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline already running (run_id={running.id})"
        )
    
    # Create new run record
    run = PipelineRun(
        status=PipelineStatus.RUNNING,
        trigger="api",
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    # Start pipeline in background
    background_tasks.add_task(
        _run_pipeline_sync,
        run.id,
        request.mode,
        request.limit,
        request.dry_run,
    )
    
    return PipelineRunResponse(
        run_id=run.id,
        status="RUNNING",
        started_at=run.started_at.isoformat(),
        message="Pipeline started. Use GET /api/pipeline/status/{run_id} to check progress.",
    )


@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(run_id: int, db: Session = Depends(get_db)):
    """Get the status of a pipeline run."""
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    return PipelineStatusResponse(
        run_id=run.id,
        status=run.status.value,
        started_at=run.started_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        markets_scanned=run.markets_scanned or 0,
        conditions_analyzed=run.conditions_analyzed or 0,
        predictions_made=run.predictions_made or 0,
        trades_executed=run.trades_executed or 0,
        error_message=run.error_message,
    )


@router.get("/runs")
async def list_pipeline_runs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List recent pipeline runs."""
    runs = db.query(PipelineRun).order_by(
        PipelineRun.started_at.desc()
    ).offset(offset).limit(limit).all()
    
    return {
        "runs": [
            {
                "run_id": r.id,
                "status": r.status.value,
                "trigger": r.trigger,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "markets_scanned": r.markets_scanned or 0,
                "predictions_made": r.predictions_made or 0,
            }
            for r in runs
        ],
        "limit": limit,
        "offset": offset,
    }

