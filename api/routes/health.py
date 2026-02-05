"""Health check endpoints."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.connection import get_db
from config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
    }


@router.get("/health/db")
async def database_health(db: Session = Depends(get_db)):
    """Database connectivity check."""
    try:
        # Execute simple query
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@router.get("/health/config")
async def config_check():
    """Check if required configuration is present."""
    checks = {
        "openai_api_key": bool(settings.openai_api_key),
        "perplexity_api_key": bool(settings.perplexity_api_key),
        "database_url": bool(settings.database_url),
    }
    
    all_ok = all(checks.values())
    
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

