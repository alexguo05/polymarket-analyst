"""
Cost tracking for API calls.

Tracks token usage and estimates costs for OpenAI and Perplexity APIs.
"""

from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# PRICING (per 1M tokens, as of Feb 2026 - update as needed)
# =============================================================================

PRICING = {
    # OpenAI models (from pricing page)
    "gpt-5-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.2": {"input": 1.75, "output": 14.00},        # $1.75/1M input, $14/1M output
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},  # $21/1M input, $168/1M output
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    
    # Perplexity models (per request, not per token)
    "sonar": {"per_request": 0.005},                    # $5/1000 requests
    "sonar-pro": {"per_request": 0.015},                # $15/1000 requests
}


@dataclass
class UsageStats:
    """Accumulated usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    estimated_cost: float = 0.0
    
    def add_openai_usage(self, usage: dict, model: str):
        """Add usage from OpenAI API response."""
        input_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        total_tok = usage.get("total_tokens", 0) or (input_tok + output_tok)
        
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.total_tokens += total_tok
        self.requests += 1
        
        # Calculate cost
        pricing = PRICING.get(model, PRICING.get("gpt-5-mini"))
        cost = (input_tok * pricing["input"] + output_tok * pricing["output"]) / 1_000_000
        self.estimated_cost += cost
        
        return cost
    
    def add_perplexity_usage(self, usage: dict, model: str):
        """Add usage from Perplexity API response."""
        # Perplexity may return token counts
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)
        
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.total_tokens += input_tok + output_tok
        self.requests += 1
        
        # Perplexity charges per request
        pricing = PRICING.get(model, PRICING.get("sonar"))
        cost = pricing.get("per_request", 0.005)
        self.estimated_cost += cost
        
        return cost
    
    def format_cost(self, cost: float) -> str:
        """Format cost for display."""
        if cost < 0.01:
            return f"${cost:.4f}"
        return f"${cost:.3f}"
    
    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"Requests: {self.requests} | "
            f"Tokens: {self.total_tokens:,} ({self.input_tokens:,} in, {self.output_tokens:,} out) | "
            f"Cost: {self.format_cost(self.estimated_cost)}"
        )


# Global tracker instances (optional - scripts can create their own)
query_gen_stats = UsageStats()
search_stats = UsageStats()
analysis_stats = UsageStats()


def reset_all():
    """Reset all global trackers."""
    global query_gen_stats, search_stats, analysis_stats
    query_gen_stats = UsageStats()
    search_stats = UsageStats()
    analysis_stats = UsageStats()

