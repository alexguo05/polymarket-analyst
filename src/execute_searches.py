#!/usr/bin/env python3
"""
Execute search queries using Perplexity's Sonar API.
Returns structured, evidence-based results with scored citations.

Reads from data/market_queries.json (condition-based structure)
Outputs to data/search_results.json
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

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CONFIGURATION
# =============================================================================

PERPLEXITY_MODEL = "sonar"  # Options: "sonar", "sonar-pro"
RATE_LIMIT_DELAY = 0.5  # seconds between API calls
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


def save_results(output_path: Path, results: list[dict]):
    """Save results to JSON file."""
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def search_perplexity(api_key: str, query: str, condition_context: str, recency: str = "week") -> Optional[dict]:
    """
    Execute a search query using Perplexity's Sonar API.
    
    Args:
        api_key: Perplexity API key
        query: The search query
        condition_context: Context about the specific condition being evaluated
        recency: "day", "week", "month", or "year"
    
    Returns:
        Parsed JSON response or None on error
    """
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
        
        # Get citations - check multiple possible locations in API response
        api_citations = (
            data.get("citations", []) or                    # Top-level citations
            message.get("citations", []) or                 # Inside message object
            data.get("sources", []) or                      # Alternative field name
            message.get("context", {}).get("citations", []) # Nested in context
        )
        
        return {
            "response": content,
            "citations": api_citations,
            "model": data.get("model", PERPLEXITY_MODEL),
            "usage": data.get("usage", {}),
            "raw_keys": list(data.keys())  # Debug: show what keys are in response
        }
        
    except requests.exceptions.Timeout:
        print(f"    ⚠️  Timeout after 90s")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"    ❌ HTTP error: {e}")
        if e.response:
            print(f"    Response: {e.response.text[:500]}")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def main():
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
                        help="Regenerate searches for all queries, replacing existing results")
    parser.add_argument("--rerun-empty", action="store_true",
                        help="Re-run queries that returned no citations")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load condition queries (new structure: one entry per condition)
    print(f"📂 Loading queries from {input_path}")
    with open(input_path) as f:
        condition_queries = json.load(f)
    
    total_conditions = len(condition_queries)
    if args.limit:
        condition_queries = condition_queries[:args.limit]
    
    # Load existing results to skip already-executed queries
    existing_results = []
    existing_query_ids = set()
    empty_citation_ids = set()
    if output_path.exists() and not args.regenerate:
        print(f"📂 Loading existing results from {output_path}")
        with open(output_path) as f:
            existing_results = json.load(f)
        # Extract all query_ids from existing results (now per condition)
        for condition_result in existing_results:
            for qr in condition_result.get('query_results', []):
                if 'query_id' in qr:
                    query_id = qr['query_id']
                    citations = qr.get('citations', [])
                    if not citations or len(citations) == 0:
                        empty_citation_ids.add(query_id)
                    existing_query_ids.add(query_id)
        print(f"   Found {len(existing_query_ids)} existing query results")
        if empty_citation_ids:
            print(f"   Found {len(empty_citation_ids)} queries with no citations")
    elif args.regenerate:
        print("🔄 Regenerate mode: will replace all existing results")
    
    # If --rerun-empty, remove empty citation queries from existing_query_ids so they get rerun
    if args.rerun_empty and empty_citation_ids:
        print(f"🔄 Rerun-empty mode: will re-execute {len(empty_citation_ids)} queries with no citations")
        existing_query_ids -= empty_citation_ids
    
    # Count queries to process (excluding already-done ones)
    total_queries = 0
    skipped_queries = 0
    for cond in condition_queries:
        queries = cond.get('queries', [])
        if args.query_limit:
            queries = queries[:args.query_limit]
        for q in queries:
            query_id = q.get('query_id')
            if query_id and query_id in existing_query_ids:
                skipped_queries += 1
            else:
                total_queries += 1
    
    print(f"📊 Processing {len(condition_queries)} conditions ({total_queries} queries to execute, {skipped_queries} skipped)")
    print(f"🔍 Recency filter: {args.recency}")
    print(f"🎯 Target citations per query: {TARGET_CITATIONS_PER_QUERY}")
    
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Queries that would be executed:\n")
        for cond in condition_queries:
            queries = cond.get('queries', [])
            if args.query_limit:
                queries = queries[:args.query_limit]
            # Filter to only show queries that would actually run
            queries_to_run = [q for q in queries if q.get('query_id') not in existing_query_ids]
            if queries_to_run:
                q_short = cond['outcome_question'][:55] + "..." if len(cond['outcome_question']) > 55 else cond['outcome_question']
                print(f"📌 {q_short} ({cond['yes_price']:.1%})")
                for q in queries_to_run:
                    print(f"   [{q.get('category', '?')}] {q['query'][:70]}...")
                print()
        if skipped_queries > 0:
            print(f"⏭️  Would skip {skipped_queries} queries with existing results")
        return
    
    # Check if there's anything to process
    if total_queries == 0:
        print("\n✅ All queries already have results. Nothing to do.")
        print(f"   Use --regenerate to force re-execution of all queries.")
        return
    
    # Check for API key
    api_key = os.getenv("PERPLEXITY_KEY")
    if not api_key:
        print("❌ PERPLEXITY_KEY not found in .env file")
        print("   Add: PERPLEXITY_KEY=\"your-key-here\"")
        sys.exit(1)
    
    print(f"🔑 API key loaded ({len(api_key)} chars)")
    print(f"🤖 Model: {PERPLEXITY_MODEL}")
    print(f"\n{'='*60}")
    print(f"🚀 Starting searches...\n")
    
    # Build a map of existing results by condition_id for merging
    existing_by_condition = {r['condition_id']: r for r in existing_results}
    
    # Initialize results with existing data (will be updated incrementally)
    results_by_condition = dict(existing_by_condition) if not args.regenerate else {}
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for i, cond in enumerate(condition_queries):
        condition_id = cond['condition_id']
        outcome_question = cond['outcome_question']
        yes_price = cond['yes_price']
        
        q_short = outcome_question[:45] + "..." if len(outcome_question) > 45 else outcome_question
        
        # Build condition context for the search
        condition_context = f"""Event: {cond['event_title']}
Condition: {outcome_question}
Current Market Price: {yes_price:.1%} (implied probability)"""
        
        # Get or create condition result structure
        if condition_id in results_by_condition:
            condition_result = results_by_condition[condition_id]
            if 'query_results' not in condition_result:
                condition_result['query_results'] = []
        else:
            condition_result = {
                "condition_id": condition_id,
                "event_id": cond['event_id'],
                "event_title": cond['event_title'],
                "outcome_question": outcome_question,
                "yes_price": yes_price,
                "end_date": cond.get('end_date'),
                "condition_volume": cond.get('condition_volume'),
                "condition_liquidity": cond.get('condition_liquidity'),
                "searched_at": datetime.now(timezone.utc).isoformat(),
                "recency_filter": args.recency,
                "model": PERPLEXITY_MODEL,
                "query_results": []
            }
            results_by_condition[condition_id] = condition_result
        
        # Build index of existing query positions for in-place replacement
        query_index = {qr.get('query_id'): idx for idx, qr in enumerate(condition_result['query_results'])}
        
        queries = cond.get('queries', [])
        if args.query_limit:
            queries = queries[:args.query_limit]
        
        # Filter to queries that need to be executed
        queries_to_run = []
        for q in queries:
            query_id = q.get('query_id')
            if query_id and query_id in existing_query_ids:
                stats['skipped'] += 1
            else:
                queries_to_run.append(q)
        
        if not queries_to_run:
            continue
        
        print(f"[{i+1}/{len(condition_queries)}] {q_short} ({yes_price:.1%}) - {len(queries_to_run)} queries")
        
        for j, query_obj in enumerate(queries_to_run):
            query = query_obj['query']
            query_id = query_obj.get('query_id', f"{condition_id[-16:]}_q{j}")
            category = query_obj.get('category', 'unknown')
            
            print(f"  [{j+1}/{len(queries_to_run)}] {category}: {query[:55]}...")
            
            result = search_perplexity(api_key, query, condition_context, args.recency)
            
            if result:
                stats['success'] += 1
                citations = result.get('citations', [])
                citation_count = len(citations)
                
                # Debug: show API response structure if no citations found
                if citation_count == 0:
                    print(f"    ⚠️  No citations found. API response keys: {result.get('raw_keys', [])}")
                
                new_result = {
                    "query_id": query_id,
                    "query": query,
                    "category": category,
                    "priority": query_obj.get('priority', 'medium'),
                    "response": result.get('response', ''),
                    "citations": citations
                }
                
                # Replace in-place if exists, otherwise append
                if query_id in query_index:
                    condition_result['query_results'][query_index[query_id]] = new_result
                else:
                    condition_result['query_results'].append(new_result)
                    query_index[query_id] = len(condition_result['query_results']) - 1
                
                print(f"    ✅ Got response ({len(result.get('response', ''))} chars, {citation_count} citations)")
            else:
                stats['failed'] += 1
                new_result = {
                    "query_id": query_id,
                    "query": query,
                    "category": category,
                    "priority": query_obj.get('priority', 'medium'),
                    "evidence": [],
                    "search_quality": {"error": "Search failed"}
                }
                
                # Replace in-place if exists, otherwise append
                if query_id in query_index:
                    condition_result['query_results'][query_index[query_id]] = new_result
                else:
                    condition_result['query_results'].append(new_result)
                    query_index[query_id] = len(condition_result['query_results']) - 1
                
                print(f"    ❌ Failed")
            
            # Update timestamp and save after each query
            condition_result['searched_at'] = datetime.now(timezone.utc).isoformat()
            save_results(output_path, list(results_by_condition.values()))
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
        
        print()
    
    print(f"{'='*60}")
    print(f"✅ Saved {len(results_by_condition)} condition results to {output_path}")
    print(f"📊 Queries: {stats['success']} succeeded, {stats['failed']} failed, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()
