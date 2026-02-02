#!/usr/bin/env python3
"""
Generate search queries for each condition (outcome) in candidate markets using GPT-5 mini.
Reads from candidate_markets_filtered.json and outputs to data/market_queries.json

Each event can have multiple conditions (e.g., "Will Ken Paxton win?" and "Will John Cornyn win?").
We generate queries for each condition separately since each has its own price/probability.
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Configuration
MODEL = "gpt-5-mini"  # GPT-5 mini
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS (Pydantic)
# =============================================================================

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SearchQuery(BaseModel):
    """A single search query for researching a market condition."""
    query: str  # The actual search query string
    purpose: str  # What evidence this query seeks
    category: Literal["base_rate", "procedural", "sentiment", "recent_developments", "structural_factors", "other"]
    priority: Literal["high", "medium", "low"]  # Relevance priority


class QueryResponse(BaseModel):
    """Response containing all search queries for a condition."""
    queries: list[SearchQuery]


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
# OUTPUT DATACLASS
# =============================================================================

@dataclass
class ConditionQueries:
    """Queries for a single condition (outcome) within an event."""
    condition_id: str
    event_id: str
    event_title: str
    outcome_question: str
    yes_price: float
    end_date: Optional[str]
    condition_volume: Optional[float]
    condition_liquidity: Optional[float]
    market_meta: dict
    queries: list[dict]
    generated_at: str
    model: str


def generate_queries_for_condition(client: OpenAI, event: dict, outcome: dict) -> Optional[list[dict]]:
    """Generate search queries for a single condition using GPT-5 mini with structured output."""
    
    # Build context about other outcomes in the same event
    other_outcomes = [o for o in event.get('outcomes', []) if o['condition_id'] != outcome['condition_id']]
    other_outcomes_text = ""
    if other_outcomes:
        other_outcomes_text = "\n\nOther outcomes in this event:\n" + "\n".join([
            f"  - {o['question']} (current: {o['yes_price']:.1%})"
            for o in other_outcomes
        ])
    
    user_prompt = f"""Event: {event['title']}

CONDITION TO ANALYZE: {outcome['question']}
Current market price: {outcome['yes_price']:.1%}

Event Description: {event.get('description', 'No description')[:500]}
{other_outcomes_text}

Resolution date: {event.get('end_date', 'Unknown')}
Days until resolution: {event.get('hours_until_end', 0) / 24:.1f}

Generate research questions to help evaluate the probability of: {outcome['question']}"""

    try:
        # Use structured output with Pydantic schema
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=QueryResponse,
            max_completion_tokens=4000
        )
        
        # Get parsed response
        parsed = response.choices[0].message.parsed
        
        if parsed is None:
            # Check for refusal
            if response.choices[0].message.refusal:
                print(f"  ⚠️  Model refused: {response.choices[0].message.refusal}")
            else:
                print(f"  ⚠️  Empty parsed response")
            return None
        
        # Convert Pydantic models to dicts and add query IDs
        # Use condition_id (shortened) for query_id prefix
        condition_short = outcome['condition_id'][-16:]  # Last 16 chars of condition hash
        queries_with_ids = []
        for idx, q in enumerate(parsed.queries):
            query_dict = q.model_dump()
            query_dict['query_id'] = f"{condition_short}_q{idx}"
            queries_with_ids.append(query_dict)
        return queries_with_ids
            
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate search queries for market conditions using GPT-5 mini")
    parser.add_argument("--input", type=str, default="data/candidate_markets_filtered.json",
                        help="Input file with candidate markets")
    parser.add_argument("--output", type=str, default="data/market_queries.json",
                        help="Output file for generated queries")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conditions to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be processed without calling API")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate queries for all conditions, replacing existing ones")
    parser.add_argument("--replace-output", action="store_true",
                        help="Write only newly generated queries (overwrite output file)")
    parser.add_argument("--random", action="store_true",
                        help="Randomly select conditions (use with --limit for random sample)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (use with --random)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load markets
    print(f"📂 Loading markets from {input_path}")
    with open(input_path) as f:
        markets = json.load(f)
    
    # Flatten to list of (event, outcome) pairs
    all_conditions = []
    for market in markets:
        for outcome in market.get('outcomes', []):
            all_conditions.append((market, outcome))
    
    total_conditions = len(all_conditions)
    print(f"📊 Found {total_conditions} conditions across {len(markets)} events")
    
    # Load existing queries to skip already-processed conditions
    existing_queries = []
    existing_condition_ids = set()
    if args.replace_output:
        print("🧹 Replace-output mode: will overwrite file with newly generated queries")
    elif output_path.exists() and not args.regenerate:
        print(f"📂 Loading existing queries from {output_path}")
        with open(output_path) as f:
            existing_queries = json.load(f)
        existing_condition_ids = {q['condition_id'] for q in existing_queries}
        print(f"   Found {len(existing_condition_ids)} existing condition queries")
    elif args.regenerate:
        print("🔄 Regenerate mode: will replace all existing queries")
    
    # Filter out conditions that already have queries
    conditions_to_process = [
        (e, o) for e, o in all_conditions 
        if o['condition_id'] not in existing_condition_ids
    ]
    
    skipped = total_conditions - len(conditions_to_process)
    
    # Apply limit (with optional random sampling)
    if args.limit and args.limit < len(conditions_to_process):
        if args.random:
            if args.seed is not None:
                random.seed(args.seed)
                print(f"🎲 Random sampling with seed {args.seed}")
            else:
                print(f"🎲 Random sampling (no seed)")
            conditions_to_process = random.sample(conditions_to_process, args.limit)
        else:
            conditions_to_process = conditions_to_process[:args.limit]
    
    print(f"📊 Processing {len(conditions_to_process)} conditions ({skipped} skipped, {total_conditions} total)")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Conditions that would be processed:")
        for i, (event, outcome) in enumerate(conditions_to_process):
            q_short = outcome['question'][:55] + "..." if len(outcome['question']) > 55 else outcome['question']
            print(f"  {i+1}. {q_short} ({outcome['yes_price']:.1%})")
        if skipped > 0:
            print(f"\n⏭️  Skipped {skipped} conditions with existing queries")
        return
    
    # Check if there's anything to process
    if not conditions_to_process:
        print("\n✅ All conditions already have queries. Nothing to do.")
        print(f"   Use --regenerate to force regeneration of all queries.")
        return
    
    # Check for API key (only needed for actual runs)
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        print(f"❌ OPENAI_KEY not found in environment")
        print(f"   Expected .env at: {env_path}")
        print(f"   .env exists: {env_path.exists()}")
        sys.exit(1)
    
    print(f"🔑 API key loaded ({len(api_key)} chars)")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Process each condition
    new_results = []
    errors = []
    
    print(f"\n🚀 Generating queries using {MODEL} (structured output)...\n")
    
    for i, (event, outcome) in enumerate(conditions_to_process):
        q_short = outcome['question'][:45] + "..." if len(outcome['question']) > 45 else outcome['question']
        print(f"[{i+1}/{len(conditions_to_process)}] {q_short} ({outcome['yes_price']:.1%})")
        
        queries = generate_queries_for_condition(client, event, outcome)
        
        if queries:
            result = ConditionQueries(
                condition_id=outcome['condition_id'],
                event_id=event['event_id'],
                event_title=event['title'],
                outcome_question=outcome['question'],
                yes_price=outcome['yes_price'],
                end_date=event.get("end_date"),
                condition_volume=outcome.get("volume"),
                condition_liquidity=outcome.get("liquidity"),
                market_meta={
                    "slug": event.get("slug"),
                    "description": event.get("description"),
                    "hours_until_end": event.get("hours_until_end"),
                    "event_liquidity": event.get("liquidity"),
                    "event_volume": event.get("volume"),
                    "condition_liquidity": outcome.get("liquidity"),
                    "tags": event.get("tags"),
                    "tag_ids": event.get("tag_ids"),
                    "resolution_source": event.get("resolution_source"),
                    "outcome_count": event.get("outcome_count"),
                    "complexity_score": event.get("complexity_score"),
                    "edge_potential": event.get("edge_potential"),
                    "updated_at": event.get("updated_at"),
                },
                queries=queries,
                generated_at=datetime.now(timezone.utc).isoformat(),
                model=MODEL
            )
            new_results.append(asdict(result))
            print(f"  ✅ Generated {len(queries)} queries")
        else:
            errors.append(outcome['condition_id'])
            print(f"  ❌ Failed to generate queries")
        
        # Rate limiting
        if i < len(conditions_to_process) - 1:
            time.sleep(RATE_LIMIT_DELAY)
    
    # Merge with existing queries (new results take precedence)
    if existing_queries and not args.regenerate and not args.replace_output:
        # Build a dict of existing queries by condition_id
        existing_by_id = {q['condition_id']: q for q in existing_queries}
        # Update with new results
        for r in new_results:
            existing_by_id[r['condition_id']] = r
        final_results = list(existing_by_id.values())
    else:
        final_results = new_results
    
    # Save results
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(new_results)} new condition queries")
    print(f"✅ Total {len(final_results)} condition queries saved to {output_path}")
    if errors:
        print(f"❌ Failed: {len(errors)} conditions")
    
    # Summary stats
    new_query_count = sum(len(r['queries']) for r in new_results)
    print(f"📊 New queries generated: {new_query_count}")
    if new_results:
        print(f"📊 Average queries per condition: {new_query_count / len(new_results):.1f}")


if __name__ == "__main__":
    main()
