#!/usr/bin/env python3
"""
Execute search queries using Perplexity's Sonar API.

Functions for pipeline:
    execute_searches(queries: list[ConditionQueries]) -> list[ConditionSearchResults]

CLI Usage:
    python execute_searches.py              # Execute searches for all queries
    python execute_searches.py --limit 10   # Limit to 10 conditions
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.query import ConditionQueries
from models.evidence import QueryResult, ConditionSearchResults
from src.cost_tracker import UsageStats
from config.settings import settings

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CONFIGURATION (from settings.py)
# =============================================================================

PERPLEXITY_MODEL = settings.search_model
RATE_LIMIT_DELAY = settings.search_rate_limit_delay
TARGET_CITATIONS_PER_QUERY = 5

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""You are a research analyst gathering evidence for PREDICTION MARKET analysis.

Your goal is to find information that helps estimate the probability of a specific market outcome. Be EXHAUSTIVE and DETAILED in your analysis.

For each source you find, provide an IN-DEPTH analysis:

=== SOURCE [N] ===

**Source Information:**
- Type: [government/university/wire service/major newspaper/local news/blog/social media/other]
- Publication Date: [YYYY-MM-DD or best estimate]
- Author/Organization: [who wrote or published this]

**Relevance:** [directly addresses / partially addresses / tangentially related]
Explain how this source relates to the specific market question.

**Specificity:** [precise data with methodology / concrete claims / general statements / vague speculation]
Assess the concreteness and verifiability of the claims.

**Key Findings:**
Extract ALL relevant factual claims from this source. Include:
- Specific numbers, percentages, statistics
- Direct quotes from officials, experts, or stakeholders
- Dates, deadlines, timelines mentioned
- Names of key people, organizations, or entities involved
- Any methodology details (sample sizes, margins of error, etc.)

Be thorough - include every relevant detail, not just a summary.

**Context & Background:**
What additional context does this source provide? Historical information, related events, explanations of processes or procedures.

**Impact Analysis:**
How does this specific evidence affect the likelihood of the market outcome? Be analytical:
- Does this increase or decrease probability of YES?
- How significant is this evidence?
- Are there caveats or uncertainties?
- Does this contradict or confirm other sources?

---

GUIDELINES:
- Cite sources using [1], [2], [3], etc.
- Be EXHAUSTIVE - include all relevant details, not summaries
- Quote directly when possible
- Note conflicts between sources
- Acknowledge limitations and gaps in available information
- Do NOT fabricate - only report what you actually find

Analyze up to {TARGET_CITATIONS_PER_QUERY} sources. Prioritize quality and depth over brevity.

End with:
=== OVERALL ASSESSMENT ===
Synthesize all findings. What is the weight of evidence? What key questions remain unanswered?"""


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _search_perplexity(
    api_key: str, 
    query: str, 
    condition_context: str, 
    recency: str = "week"
) -> Optional[dict]:
    """Execute a search query using Perplexity's Sonar API."""
    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    user_message = f"""CONDITION BEING EVALUATED:
{condition_context}

RESEARCH QUESTION:
{query}

Search for relevant information and analyze each source you find using the specified format."""

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "search_recency_filter": recency,
        "return_citations": True,
        "return_related_questions": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        
        # Get citations from various locations in API response
        api_citations = (
            data.get("citations", []) or
            message.get("citations", []) or
            data.get("sources", []) or
            message.get("context", {}).get("citations", [])
        )
        
        return {
            "response": content,
            "citations": api_citations,
            "model": data.get("model", PERPLEXITY_MODEL),
            "usage": data.get("usage", {}),
        }
        
    except requests.exceptions.Timeout:
        print(f"    ⚠️  Timeout after 90s")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"    ❌ HTTP error: {e}")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


# =============================================================================
# PUBLIC API (for pipeline use)
# =============================================================================

def execute_searches(
    condition_queries: list[ConditionQueries],
    api_key: str = None,
    recency: str = "week",
    query_limit: int = None,
    verbose: bool = True,
) -> tuple[list[ConditionSearchResults], UsageStats]:
    """
    Execute search queries for a list of conditions.
    
    This is the main entry point for the pipeline.
    
    Args:
        condition_queries: List of ConditionQueries to execute
        api_key: Optional Perplexity API key (uses env if not provided)
        recency: Recency filter ("day", "week", "month", "year")
        query_limit: Optional limit on queries per condition
        verbose: Print progress
        
    Returns:
        Tuple of (List of ConditionSearchResults, UsageStats with cost info)
    """
    if api_key is None:
        api_key = os.getenv("PERPLEXITY_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_KEY not found in environment")
    
    results = []
    usage_stats = UsageStats()
    stats = {"success": 0, "failed": 0}
    
    for i, cond in enumerate(condition_queries):
        if verbose:
            q_short = cond.outcome_question[:45] + "..." if len(cond.outcome_question) > 45 else cond.outcome_question
            print(f"[{i+1}/{len(condition_queries)}] {q_short} ({cond.yes_price:.1%})")
        
        # Build condition context
        condition_context = f"""Event: {cond.event_title}
Condition: {cond.outcome_question}
Current Market Price: {cond.yes_price:.1%} (implied probability)"""
        
        # Execute queries
        queries = cond.queries
        if query_limit:
            queries = queries[:query_limit]
        
        query_results = []
        condition_requests = 0
        condition_tokens = 0
        condition_cost = 0.0
        
        for j, query in enumerate(queries):
            if verbose:
                print(f"  [{j+1}/{len(queries)}] {query.category}: {query.query[:55]}...")
            
            result = _search_perplexity(api_key, query.query, condition_context, recency)
            
            if result:
                stats['success'] += 1
                citations = result.get('citations', [])
                
                # Track usage/cost - Perplexity charges per request (~$0.005)
                usage = result.get('usage', {})
                req_cost = 0.005  # $5/1000 requests
                req_tokens = usage.get('total_tokens', 0)
                
                condition_requests += 1
                condition_tokens += req_tokens
                condition_cost += req_cost
                
                # Also update aggregate stats
                usage_stats.requests += 1
                usage_stats.total_tokens += req_tokens
                usage_stats.estimated_cost += req_cost
                
                query_results.append(QueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    category=query.category,
                    response=result.get('response', ''),
                    citations=citations,
                    searched_at=datetime.now(timezone.utc).isoformat(),
                ))
                
                if verbose:
                    print(f"    ✅ Got response ({len(result.get('response', ''))} chars, {len(citations)} citations) [${req_cost:.4f}]")
            else:
                stats['failed'] += 1
                # Still create a result with empty response
                query_results.append(QueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    category=query.category,
                    response="",
                    citations=[],
                    searched_at=datetime.now(timezone.utc).isoformat(),
                ))
                if verbose:
                    print(f"    ❌ Failed")
            
            # Rate limiting
            if j < len(queries) - 1:
                time.sleep(RATE_LIMIT_DELAY)
        
        # Create search results for this condition with cost tracking
        results.append(ConditionSearchResults(
            condition_id=cond.condition_id,
            event_id=cond.event_id,
            event_title=cond.event_title,
            outcome_question=cond.outcome_question,
            yes_price=cond.yes_price,
            volume=cond.volume,
            liquidity=cond.liquidity,
            end_date=cond.end_date,
            query_results=query_results,
            searched_at=datetime.now(timezone.utc).isoformat(),
            # Cost tracking for this condition
            total_requests=condition_requests,
            total_tokens=condition_tokens,
            cost_usd=condition_cost,
        ))
        
        if verbose:
            print(f"  💰 Condition cost: ${condition_cost:.4f} ({condition_requests} requests)\n")
    
    if verbose:
        print(f"📊 Queries: {stats['success']} succeeded, {stats['failed']} failed")
        print(f"   💰 {usage_stats.summary()}")
    
    return results, usage_stats


def _save_results(output_path: Path, results: list[dict]):
    """Save results to JSON file."""
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# MAIN (CLI entry point)
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Execute searches using Perplexity API")
    parser.add_argument("--input", type=str, default="data/market_queries.json",
                        help="Input file with condition queries")
    parser.add_argument("--output", type=str, default="data/search_results.json",
                        help="Output file for search results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conditions to process")
    parser.add_argument("--query-limit", type=int, default=None,
                        help="Limit queries per condition")
    parser.add_argument("--recency", type=str, default="week",
                        choices=["day", "week", "month", "year"],
                        help="Recency filter for searches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be searched without calling API")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate all search results")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to JSON files")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load condition queries from JSON
    print(f"📂 Loading queries from {input_path}")
    with open(input_path) as f:
        raw_queries = json.load(f)
    
    # Convert to ConditionQueries models
    from models.query import SearchQuery as SearchQueryModel
    condition_queries = []
    for q in raw_queries:
        # Convert query dicts to SearchQuery models
        queries = [SearchQueryModel(**sq) for sq in q.get('queries', [])]
        condition_queries.append(ConditionQueries(
            condition_id=q['condition_id'],
            event_id=q['event_id'],
            event_title=q['event_title'],
            outcome_question=q['outcome_question'],
            yes_price=q['yes_price'],
            volume=q.get('condition_volume'),
            liquidity=q.get('condition_liquidity'),
            end_date=q.get('end_date'),
            queries=queries,
            generated_at=q.get('generated_at', ''),
            model=q.get('model', 'unknown'),
        ))
    
    print(f"📊 Found {len(condition_queries)} conditions")
    
    # Load existing results to skip already-processed
    existing_condition_ids = set()
    existing_results = []
    if output_path.exists() and not args.regenerate:
        with open(output_path) as f:
            existing_results = json.load(f)
        existing_condition_ids = {r['condition_id'] for r in existing_results}
        print(f"   Found {len(existing_condition_ids)} existing results")
    
    # Filter out already-processed conditions
    conditions_to_process = [c for c in condition_queries if c.condition_id not in existing_condition_ids]
    skipped = len(condition_queries) - len(conditions_to_process)
    
    if args.limit:
        conditions_to_process = conditions_to_process[:args.limit]
    
    print(f"📊 Processing {len(conditions_to_process)} conditions ({skipped} skipped)")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Conditions that would be searched:")
        for c in conditions_to_process:
            q_short = c.outcome_question[:55] + "..." if len(c.outcome_question) > 55 else c.outcome_question
            print(f"  📌 {q_short} ({c.yes_price:.1%})")
            for q in c.queries:
                print(f"     [{q.category}] {q.query[:60]}...")
        return None
    
    if not conditions_to_process:
        print("\n✅ All conditions already have results. Use --regenerate to force.")
        return []
    
    # Execute searches
    print(f"\n🚀 Executing searches using {PERPLEXITY_MODEL}...\n")
    new_results, stats = execute_searches(
        conditions_to_process,
        recency=args.recency,
        query_limit=args.query_limit,
        verbose=True,
    )
    
    if args.no_save:
        print(f"\n✅ Executed searches for {len(new_results)} conditions (not saved)")
        return new_results, stats
    
    # Convert to dicts for JSON
    new_results_dicts = []
    for r in new_results:
        d = r.model_dump()
        # Convert QueryResult models to dicts
        d['query_results'] = [qr.model_dump() for qr in r.query_results]
        new_results_dicts.append(d)
    
    # Merge with existing
    if existing_results and not args.regenerate:
        existing_by_id = {r['condition_id']: r for r in existing_results}
        for r in new_results_dicts:
            existing_by_id[r['condition_id']] = r
        final_results = list(existing_by_id.values())
    else:
        final_results = new_results_dicts
    
    # Save
    _save_results(output_path, final_results)
    
    print(f"\n{'='*60}")
    print(f"✅ Executed searches for {len(new_results)} conditions")
    print(f"✅ Total {len(final_results)} saved to {output_path}")
    
    return new_results


if __name__ == "__main__":
    main()
