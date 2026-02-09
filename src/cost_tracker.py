"""
Cost tracking for API calls.

Tracks token usage and estimates costs for OpenAI and Perplexity APIs.
Pricing is defined in config/settings.py to maintain a single source of truth.
"""

from dataclasses import dataclass


@dataclass
class UsageStats:
    """Accumulated usage statistics for API calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # Hidden "thinking" tokens for reasoning models (gpt-5.2-pro)
    total_tokens: int = 0
    requests: int = 0
    estimated_cost: float = 0.0
    
    def format_cost(self, cost: float) -> str:
        """Format cost for display."""
        if cost < 0.01:
            return f"${cost:.4f}"
        return f"${cost:.3f}"
    
    def summary(self) -> str:
        """Return a summary string."""
        tokens_str = f"{self.total_tokens:,} ({self.input_tokens:,} in, {self.output_tokens:,} out"
        if self.reasoning_tokens > 0:
            tokens_str += f", {self.reasoning_tokens:,} reasoning"
        tokens_str += ")"
        
        return (
            f"Requests: {self.requests} | "
            f"Tokens: {tokens_str} | "
            f"Cost: {self.format_cost(self.estimated_cost)}"
        )
    
    def __add__(self, other: "UsageStats") -> "UsageStats":
        """Allow adding two UsageStats together."""
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )
