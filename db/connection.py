"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from config.settings import settings
from db.models import Base


# Create engine based on environment
if settings.environment == "production":
    # Production: PostgreSQL on Cloud SQL
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
else:
    # Development: SQLite
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False}  # SQLite specific
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    # Ensure data directory exists for SQLite
    if "sqlite" in settings.database_url:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

