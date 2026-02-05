"""
Centralized configuration for Polymarket Analyst.

All configuration values are defined here. In production, sensitive values
come from GCP Secret Manager. Locally, they come from .env file.
"""
import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # ENVIRONMENT
    # ==========================================================================
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # ==========================================================================
    # API KEYS
    # ==========================================================================
    openai_api_key: str = Field(default="", alias="OPENAI_KEY")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_KEY")
    
    # ==========================================================================
    # DATABASE
    # ==========================================================================
    database_url: str = Field(
        default="sqlite:///./data/polymarket.db",
        alias="DATABASE_URL"
    )
    
    # ==========================================================================
    # GCP (Production)
    # ==========================================================================
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-central1", alias="GCP_REGION")
    
    # ==========================================================================
    # MODELS
    # ==========================================================================
    query_model: str = Field(default="gpt-5-mini", alias="QUERY_MODEL")
    analysis_model: str = Field(default="gpt-5.2", alias="ANALYSIS_MODEL")  # "gpt-5.2" or "gpt-5.2-pro"
    search_model: str = Field(default="sonar", alias="SEARCH_MODEL")
    
    # ==========================================================================
    # MODEL PRICING (per 1M tokens)
    # ==========================================================================
    @property
    def model_config_map(self) -> dict:
        """Model-specific configuration including API type and pricing."""
        return {
            "gpt-5-mini": {
                "api": "chat",
                "pricing": {"input": 0.15, "output": 0.60},
            },
            "gpt-5.2": {
                "api": "chat",
                "pricing": {"input": 1.75, "output": 14.0},
            },
            "gpt-5.2-pro": {
                "api": "responses",
                "pricing": {"input": 21.0, "output": 168.0, "reasoning": 168.0},
            },
            "sonar": {
                "api": "perplexity",
                "pricing": {"per_request": 0.005},
            },
            "sonar-pro": {
                "api": "perplexity", 
                "pricing": {"per_request": 0.015},
            },
        }
    
    def get_model_config(self, model: str) -> dict:
        """Get configuration for a specific model."""
        return self.model_config_map.get(model, self.model_config_map["gpt-5.2"])
    
    # ==========================================================================
    # RATE LIMITS (seconds between API calls)
    # ==========================================================================
    query_rate_limit_delay: float = Field(default=0.5, alias="QUERY_RATE_LIMIT_DELAY")
    search_rate_limit_delay: float = Field(default=0.5, alias="SEARCH_RATE_LIMIT_DELAY")
    analysis_rate_limit_delay: float = Field(default=1.0, alias="ANALYSIS_RATE_LIMIT_DELAY")  # 15.0 for gpt-5.2-pro
    
    # ==========================================================================
    # MARKET FILTERS (from get_markets.py)
    # ==========================================================================
    min_liquidity: int = Field(default=5_000, alias="MIN_LIQUIDITY")
    max_liquidity: int = Field(default=500_000, alias="MAX_LIQUIDITY")
    min_price: float = Field(default=0.20, alias="MIN_PRICE")
    max_price: float = Field(default=0.80, alias="MAX_PRICE")
    min_hours_until_end: int = Field(default=24, alias="MIN_HOURS_UNTIL_END")
    max_days_until_end: int = Field(default=365, alias="MAX_DAYS_UNTIL_END")
    min_condition_volume: int = Field(default=10_000, alias="MIN_CONDITION_VOLUME")
    
    # Filtered output bounds
    filtered_min_outcome_price: float = Field(default=0.20, alias="FILTERED_MIN_OUTCOME_PRICE")
    filtered_max_outcome_price: float = Field(default=0.80, alias="FILTERED_MAX_OUTCOME_PRICE")
    filtered_min_days_until_end: int = Field(default=5, alias="FILTERED_MIN_DAYS_UNTIL_END")
    filtered_max_days_until_end: int = Field(default=60, alias="FILTERED_MAX_DAYS_UNTIL_END")
    
    # ==========================================================================
    # PREDICTION PARAMETERS (from compute_prediction.py)
    # ==========================================================================
    evidence_scale: float = Field(default=0.5, alias="EVIDENCE_SCALE")
    min_prob: float = Field(default=0.01, alias="MIN_PROB")
    max_prob: float = Field(default=0.99, alias="MAX_PROB")
    min_edge_threshold: float = Field(default=0.02, alias="MIN_EDGE_THRESHOLD")
    
    # ==========================================================================
    # RISK CONTROLS
    # ==========================================================================
    max_position_size: float = Field(default=100.0, alias="MAX_POSITION_SIZE")
    max_total_exposure: float = Field(default=1000.0, alias="MAX_TOTAL_EXPOSURE")
    max_daily_trades: int = Field(default=10, alias="MAX_DAILY_TRADES")
    approval_mode: bool = Field(default=True, alias="APPROVAL_MODE")
    
    # ==========================================================================
    # PATHS
    # ==========================================================================
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

