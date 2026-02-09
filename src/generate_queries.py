#!/usr/bin/env python3
"""
Generate search queries for each condition (outcome) in candidate markets using GPT-5 mini.

Functions for pipeline:
    generate_queries(conditions: list[FilteredCondition]) -> list[ConditionQueries]

CLI Usage:
    python generate_queries.py              # Generate queries for filtered conditions
    python generate_queries.py --limit 10   # Limit to 10 conditions
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Literal

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.market import FilteredCondition
from models.query import SearchQuery as SearchQueryModel, ConditionQueries
from src.cost_tracker import UsageStats
from config.settings import settings

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Configuration (from settings.py)
MODEL = settings.query_model
RATE_LIMIT_DELAY = settings.query_rate_limit_delay


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS (for OpenAI API)
# =============================================================================

class SearchQuerySchema(BaseModel):
    """Schema for OpenAI structured output."""
    query: str
    purpose: str
    category: Literal["base_rate", "procedural", "sentiment", "recent_developments", "structural_factors", "other"]
    priority: Literal["high", "medium", "low"]


class QueryResponseSchema(BaseModel):
    """Schema for OpenAI structured output."""
    queries: list[SearchQuerySchema]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a research assistant helping analyze prediction markets. Your task is to generate natural language research questions that will be sent to an AI-powered search tool (Perplexity) to gather evidence.

IMPORTANT: You are generating research QUESTIONS that another AI will interpret and search for. 

DO NOT:
- Use search operators (site:, OR, AND, quotes, etc.)
- Specify particular websites or sources
- Use keyword-stuffed phrases
- Be overly specific about dates or filing numbers

Generate 5 research questions specific to the CONDITION QUESTION provided (not the overall event). 
The condition question is a YES/NO bet (e.g., "Will Ken Paxton win?" is separate from "Will John Cornyn win?").

Consider these types of information when deciding what to ask:

- **Base rates**: Historical frequency of similar events (how often does this type of outcome happen?)
- **Procedural**: Official processes, schedules, deadlines, legal requirements that affect THIS specific outcome
- **Sentiment**: Recent statements from key stakeholders, officials, experts about THIS specific outcome
- **Recent developments**: Breaking news or recent events that could affect THIS outcome's probability
- **Structural factors**: Underlying conditions, trends, demographics that favor or disfavor THIS outcome

Choose the 5 most important questions for determining the probability of THIS SPECIFIC OUTCOME."""


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _generate_queries_for_condition(
    client: OpenAI, 
    condition: FilteredCondition,
) -> tuple[Optional[list[SearchQueryModel]], dict]:
    """
    Generate search queries for a single condition using GPT-5 mini with structured output.
    
    Returns:
        Tuple of (queries list or None, usage dict with input_tokens, output_tokens, cost_usd)
    """
    usage_info = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    
    user_prompt = f"""Event: {condition.event_title}

CONDITION TO ANALYZE: {condition.outcome_question}
Current market price: {condition.yes_price:.1%}

Resolution date: {condition.end_date or 'Unknown'}

Generate research questions to help evaluate the probability of: {condition.outcome_question}"""

    try:
        # Use structured output with Pydantic schema
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=QueryResponseSchema,
            max_completion_tokens=4000
        )
        
        # Extract usage/cost
        if hasattr(response, 'usage') and response.usage:
            input_tok = response.usage.prompt_tokens or 0
            output_tok = response.usage.completion_tokens or 0
            # Get pricing from settings
            pricing = settings.model_config_map.get(MODEL, {}).get("pricing", {"input": 0.15, "output": 0.60})
            cost = (input_tok * pricing["input"] + output_tok * pricing["output"]) / 1_000_000
            usage_info = {
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cost_usd": cost,
            }
        
        # Get parsed response
        parsed = response.choices[0].message.parsed
        
        if parsed is None:
            if response.choices[0].message.refusal:
                print(f"  ⚠️  Model refused: {response.choices[0].message.refusal}")
            else:
                print(f"  ⚠️  Empty parsed response")
            return None, usage_info
        
        # Convert to shared model with query IDs
        condition_short = condition.condition_id[-16:]
        queries = []
        for idx, q in enumerate(parsed.queries):
            queries.append(SearchQueryModel(
                query_id=f"{condition_short}_q{idx}",
                query=q.query,
                purpose=q.purpose,
                category=q.category,
                priority=q.priority,
            ))
        return queries, usage_info
            
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None, usage_info


# =============================================================================
# PUBLIC API (for pipeline use)
# =============================================================================

def generate_queries(
    conditions: list[FilteredCondition],
    client: OpenAI = None,
    verbose: bool = True,
) -> tuple[list[ConditionQueries], UsageStats]:
    """
    Generate search queries for a list of conditions.
    
    This is the main entry point for the pipeline.
    
    Args:
        conditions: List of FilteredCondition to generate queries for
        client: Optional OpenAI client (created if not provided)
        verbose: Print progress
        
    Returns:
        Tuple of (List of ConditionQueries, UsageStats with cost info)
    """
    if client is None:
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY not found in environment")
        client = OpenAI(api_key=api_key)
    
    results = []
    stats = UsageStats()
    
    for i, condition in enumerate(conditions):
        if verbose:
            q_short = condition.outcome_question[:45] + "..." if len(condition.outcome_question) > 45 else condition.outcome_question
            print(f"[{i+1}/{len(conditions)}] {q_short} ({condition.yes_price:.1%})")
        
        queries, usage_info = _generate_queries_for_condition(client, condition)
        
        # Track in aggregate stats
        stats.input_tokens += usage_info["input_tokens"]
        stats.output_tokens += usage_info["output_tokens"]
        stats.total_tokens += usage_info["input_tokens"] + usage_info["output_tokens"]
        stats.estimated_cost += usage_info["cost_usd"]
        stats.requests += 1
        
        if queries:
            result = ConditionQueries(
                condition_id=condition.condition_id,
                event_id=condition.event_id,
                event_title=condition.event_title,
                outcome_question=condition.outcome_question,
                yes_price=condition.yes_price,
                volume=condition.volume,
                liquidity=condition.liquidity,
                end_date=condition.end_date,
                queries=queries,
                generated_at=datetime.now(timezone.utc).isoformat(),
                model=MODEL,
                # Store cost per condition
                input_tokens=usage_info["input_tokens"],
                output_tokens=usage_info["output_tokens"],
                cost_usd=usage_info["cost_usd"],
            )
            results.append(result)
            if verbose:
                print(f"  ✅ Generated {len(queries)} queries [${usage_info['cost_usd']:.4f}]")
        else:
            if verbose:
                print(f"  ❌ Failed to generate queries")
        
        # Rate limiting
        if i < len(conditions) - 1:
            time.sleep(RATE_LIMIT_DELAY)
    
    if verbose:
        print(f"\n   💰 Total: {stats.summary()}")
    
    return results, stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate search queries for market conditions")
    parser.add_argument("--input", type=str, default="data/candidate_markets_filtered.json",
                        help="Input file with filtered conditions")
    parser.add_argument("--output", type=str, default="data/market_queries.json",
                        help="Output file for generated queries")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conditions to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be processed without calling API")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate queries for all conditions")
    parser.add_argument("--random", action="store_true",
                        help="Randomly select conditions (use with --limit)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to JSON files")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load conditions from JSON
    print(f"📂 Loading conditions from {input_path}")
    with open(input_path) as f:
        raw_conditions = json.load(f)
    
    # Convert to FilteredCondition models
    conditions = [FilteredCondition(**c) for c in raw_conditions]
    print(f"📊 Found {len(conditions)} conditions")
    
    # Load existing queries to skip already-processed
    existing_condition_ids = set()
    existing_queries = []
    if output_path.exists() and not args.regenerate:
        with open(output_path) as f:
            existing_queries = json.load(f)
        existing_condition_ids = {q['condition_id'] for q in existing_queries}
        print(f"   Found {len(existing_condition_ids)} existing queries")
    
    # Filter out already-processed conditions
    conditions_to_process = [c for c in conditions if c.condition_id not in existing_condition_ids]
    skipped = len(conditions) - len(conditions_to_process)
    
    # Apply limit (with optional random sampling)
    if args.limit and args.limit < len(conditions_to_process):
        if args.random:
            if args.seed is not None:
                random.seed(args.seed)
                print(f"🎲 Random sampling with seed {args.seed}")
            conditions_to_process = random.sample(conditions_to_process, args.limit)
        else:
            conditions_to_process = conditions_to_process[:args.limit]
    
    print(f"📊 Processing {len(conditions_to_process)} conditions ({skipped} skipped)")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Conditions that would be processed:")
        for i, c in enumerate(conditions_to_process):
            q_short = c.outcome_question[:55] + "..." if len(c.outcome_question) > 55 else c.outcome_question
            print(f"  {i+1}. {q_short} ({c.yes_price:.1%})")
        return None
    
    if not conditions_to_process:
        print("\n✅ All conditions already have queries. Use --regenerate to force.")
        return []
    
    # Generate queries using the public API function
    print(f"\n🚀 Generating queries using {MODEL}...\n")
    new_results, stats = generate_queries(conditions_to_process, verbose=True)
    
    if args.no_save:
        print(f"\n✅ Generated {len(new_results)} condition queries (not saved)")
        return new_results, stats
    
    # Convert to dicts for JSON serialization
    new_results_dicts = []
    for r in new_results:
        d = r.model_dump()
        # Convert SearchQuery models to dicts
        d['queries'] = [q.model_dump() for q in r.queries]
        new_results_dicts.append(d)
    
    # Merge with existing
    if existing_queries and not args.regenerate:
        existing_by_id = {q['condition_id']: q for q in existing_queries}
        for r in new_results_dicts:
            existing_by_id[r['condition_id']] = r
        final_results = list(existing_by_id.values())
    else:
        final_results = new_results_dicts
    
    # Save
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(new_results)} new condition queries")
    print(f"✅ Total {len(final_results)} saved to {output_path}")
    
    return new_results


if __name__ == "__main__":
    main()
