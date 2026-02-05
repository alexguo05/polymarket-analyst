"""
FastAPI application for Polymarket Analyst.

Provides HTTP endpoints for:
- Triggering pipeline runs
- Viewing predictions
- Managing positions
- System health checks
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config.settings import settings
from db.connection import get_db, init_db
from api.routes import pipeline, predictions, positions, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Starting Polymarket Analyst API...")
    init_db()
    print("✅ Database initialized")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Polymarket Analyst API",
    description="Automated prediction market analysis and trading",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [
        "https://your-frontend-domain.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(positions.router, prefix="/api/positions", tags=["Positions"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Polymarket Analyst API",
        "version": "1.0.0",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health",
    }

