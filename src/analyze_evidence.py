#!/usr/bin/env python3
"""
Analyze search results and score evidence quality.

Reads from data/search_results.json (condition-based structure)
Outputs to data/evidence_analysis.json

GPT scores each evidence item on 4 metrics + direction + strength.
All calculations (edge, consensus, etc.) are done manually later.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "gpt-5-pro"
RATE_LIMIT_DELAY = 15  # gpt-5-pro has strict rate limits

# =============================================================================
# RUBRICS (included in prompt)
# =============================================================================

SCORING_RUBRICS = """
## SCORING RUBRICS

RELIABILITY (source trustworthiness, 0.0-1.0):
- 0.80-1.00: Government primary source, court document, peer-reviewed study
- 0.60-0.79: Major newspaper (NYT, Reuters, AP), official statement, established think tank
- 0.40-0.59: Local news, trade publication, secondary reporting
- 0.20-0.39: Blog, opinion piece, partisan outlet
- 0.00-0.19: Anonymous, unverifiable, known unreliable

**WITHOUT verifiable citations: cap reliability at 0.60 maximum**

RECENCY (freshness, 0.0-1.0):
- 0.80-1.00: Within 7 days of today
- 0.60-0.79: 1-4 weeks ago
- 0.40-0.59: 1-3 months ago
- 0.20-0.39: 3-12 months ago
- 0.00-0.19: Over 1 year old

RELEVANCE (how directly it addresses the question, 0.0-1.0):
- 0.80-1.00: Directly answers the specific market question
- 0.60-0.79: Directly relevant but missing key details
- 0.40-0.59: Partially addresses the question
- 0.20-0.39: Related topic but doesn't address specific question
- 0.00-0.19: Tangentially related at best

SPECIFICITY (concreteness of claims, 0.0-1.0):
- 0.80-1.00: Precise data with methodology (poll with n=, exact figures, dates)
- 0.60-0.79: Concrete claims with context (specific numbers, names, dates)
- 0.40-0.59: Some specifics but incomplete
- 0.20-0.39: General claims without supporting data
- 0.00-0.19: Vague speculation

DIRECTION:
- 1 = Evidence INCREASES probability of YES outcome
- -1 = Evidence DECREASES probability of YES outcome
- 0 = Evidence is neutral / doesn't clearly affect probability

STRENGTH (how much this evidence should shift probability, 0.0-1.0):
- 0.80-1.00: Decisive evidence that fundamentally changes the picture
- 0.60-0.79: Strong evidence with clear implications
- 0.40-0.59: Moderate evidence, notable but not definitive
- 0.20-0.39: Weak evidence, minor factor
- 0.00-0.19: Negligible impact

PREDICTABILITY (how predictable this market is from information, 0.0-1.0):
- 0.80-1.00: Highly predictable (structured process, reliable sources, clear signal)
- 0.60-0.79: Moderately predictable (some structure + meaningful data)
- 0.40-0.59: Mixed predictability (signal exists but noisy or partial)
- 0.20-0.39: Low predictability (high noise, weak signal, hard to verify)
- 0.00-0.19: Essentially random (coin-flip, phrasing games, very low signal)

"""

# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class EvidenceScore(BaseModel):
    """Scores for a single query response."""
    query_id: str
    
    # Key takeaway
    key_finding: str = Field(description="One sentence summary of the main finding")
    
    # 4 quality scores with reasoning
    reliability: float = Field(ge=0.0, le=1.0)
    reliability_reason: str
    
    recency: float = Field(ge=0.0, le=1.0)
    recency_reason: str
    
    relevance: float = Field(ge=0.0, le=1.0)
    relevance_reason: str
    
    specificity: float = Field(ge=0.0, le=1.0)
    specificity_reason: str
    
    # Impact assessment
    direction: int = Field(ge=-1, le=1, description="1=increases, -1=decreases, 0=neutral")
    direction_reason: str
    
    strength: float = Field(ge=0.0, le=1.0, description="How much this should shift probability")
    strength_reason: str


class ConditionAnalysis(BaseModel):
    """Analysis output - predictability estimate plus evidence scores."""
    predictability: float = Field(ge=0.0, le=1.0, description="How predictable this market is from information")
    predictability_reason: str
    evidence_scores: list[EvidenceScore]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_analysis_prompt(condition_data: dict, today_date: str) -> str:
    """Build prompt for GPT to score evidence. Does NOT include market price."""
    
    # Build evidence text from query results
    evidence_sections = []
    for i, qr in enumerate(condition_data.get('query_results', [])):
        query_id = qr.get('query_id', f"q{i}")
        query = qr.get('query', '')
        category = qr.get('category', 'unknown')
        response = qr.get('response', '')
        citations = qr.get('citations', [])
        
        citation_info = f"CITATIONS PROVIDED: {len(citations)}" if citations else "NO CITATIONS (cannot verify sources)"
        
        evidence_sections.append(f"""
### Query {query_id} [{category}]
**Question:** {query}

**{citation_info}**

**Response:**
{response[:8000]}
""")
    
    evidence_text = "\n---\n".join(evidence_sections)
    
    prompt = f"""You are scoring evidence quality for a prediction market analysis.

## CONDITION BEING EVALUATED
**Event:** {condition_data['event_title']}
**Outcome Question:** {condition_data['outcome_question']}

This is a YES/NO question. Score how each piece of evidence affects the probability of YES.

## TODAY'S DATE
{today_date}

{SCORING_RUBRICS}

## EVIDENCE TO SCORE

{evidence_text}

## YOUR TASK

First, estimate how predictable this market is from information.
- predictability (0.0-1.0): overall predictability
- predictability_reason: brief explanation tied to the evidence and market structure

Then, for EACH query above, provide scores:
- query_id: The ID shown in the header (e.g., "1d21392da10a9_q0")
- key_finding: One sentence summary of the main finding
- reliability (0.0-1.0) + reliability_reason
- recency (0.0-1.0) + recency_reason
- relevance (0.0-1.0) + relevance_reason  
- specificity (0.0-1.0) + specificity_reason
- direction: +1 (increases YES probability), -1 (decreases), or 0 (neutral)
- direction_reason: Why this direction?
- strength (0.0-1.0) + strength_reason: How much should this shift probability?

IMPORTANT:
- If NO CITATIONS, cap reliability at 0.60 maximum
- Be specific - cite details from the evidence in your reasoning
- Score each query independently

Return ONLY valid JSON matching the schema (no markdown, no extra text).
"""
    return prompt


# =============================================================================
# API CALL
# =============================================================================

def analyze_condition(client: OpenAI, condition_data: dict, max_retries: int = 3) -> ConditionAnalysis | None:
    """Score evidence for a single condition using GPT with retry logic."""
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = build_analysis_prompt(condition_data, today)
    
    schema = ConditionAnalysis.model_json_schema()
    
    # OpenAI requires additionalProperties: false at all object levels
    def add_additional_properties_false(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" or "properties" in obj:
                obj["additionalProperties"] = False
            for v in obj.values():
                add_additional_properties_false(v)
        elif isinstance(obj, list):
            for item in obj:
                add_additional_properties_false(item)
    
    add_additional_properties_false(schema)
    
    # Also handle $defs if present
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            add_additional_properties_false(def_schema)
    
    # Retry loop with exponential backoff for rate limits
    for attempt in range(max_retries):
        try:
            # Responses API uses 'text' parameter with 'format' for structured output
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are an expert analyst scoring evidence quality. Be rigorous and specific."},
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ConditionAnalysis",
                        "schema": schema,
                        "strict": True
                    }
                }
            )

            # Extract text from response
            raw_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                raw_text = response.output_text.strip()
            elif hasattr(response, 'output') and response.output:
                # Handle list of output items
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                raw_text = content_item.text.strip()
                                break
                    if raw_text:
                        break
            
            if not raw_text:
                print("    ❌ Error: Empty response")
                return None

            return ConditionAnalysis.model_validate_json(raw_text)

        except Exception as e:
            error_str = str(e)
            
            # Print full error details
            print(f"    ❌ Error type: {type(e).__name__}")
            print(f"    ❌ Error message: {e}")
            
            # Try to get more details from the exception
            if hasattr(e, 'response'):
                try:
                    print(f"    ❌ Response status: {e.response.status_code}")
                    print(f"    ❌ Response body: {e.response.text}")
                except:
                    pass
            if hasattr(e, 'body'):
                print(f"    ❌ Error body: {e.body}")
            if hasattr(e, 'code'):
                print(f"    ❌ Error code: {e.code}")
            
            # Check for rate limit error (429)
            if "429" in error_str or "rate_limit" in error_str.lower():
                # Extract wait time if provided, otherwise use exponential backoff
                wait_time = 30 * (2 ** attempt)  # 30, 60, 120 seconds
                if attempt < max_retries - 1:
                    print(f"    ⏳ Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
            
            # Check for connection errors - retry those too
            if "connection" in error_str.lower() or "timeout" in error_str.lower():
                wait_time = 10 * (2 ** attempt)  # 10, 20, 40 seconds
                if attempt < max_retries - 1:
                    print(f"    🔄 Connection error. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
            
            return None
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score evidence for prediction market conditions")
    parser.add_argument("--input", type=str, default="data/search_results.json",
                        help="Input file with search results")
    parser.add_argument("--output", type=str, default="data/evidence_analysis.json",
                        help="Output file for analysis")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conditions to analyze")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt without calling API")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-analyze all conditions, replacing existing results")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load search results (condition-based structure)
    print(f"📂 Loading search results from {input_path}")
    with open(input_path) as f:
        search_results = json.load(f)
    
    if args.limit:
        search_results = search_results[:args.limit]
    
    print(f"📊 Found {len(search_results)} conditions to analyze")
    
    if args.dry_run:
        # Show sample prompt
        if search_results:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            prompt = build_analysis_prompt(search_results[0], today)
            print("\n" + "="*60)
            print("SAMPLE PROMPT:")
            print("="*60)
            print(prompt[:3000] + "\n...[truncated]...")
        return
    
    # Check API key
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        print("❌ OPENAI_KEY not found in .env")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    print(f"🔑 API key loaded")
    print(f"🤖 Model: {MODEL}")
    print(f"\n{'='*60}")
    print("🚀 Starting analysis...\n")
    
    # Load existing results for incremental updates
    existing_results = {}
    if output_path.exists() and not args.regenerate:
        with open(output_path) as f:
            existing = json.load(f)
            existing_results = {r['condition_id']: r for r in existing}
        print(f"📂 Loaded {len(existing_results)} existing analyses")
    elif args.regenerate:
        print("🔄 Regenerate mode: will replace all existing analyses")
    
    # Analyze each condition
    results = []
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for i, cond in enumerate(search_results):
        condition_id = cond['condition_id']
        outcome_question = cond['outcome_question']
        market_price = cond['yes_price']
        q_short = outcome_question[:45] + "..." if len(outcome_question) > 45 else outcome_question
        
        # Skip if already analyzed (unless regenerating)
        if condition_id in existing_results and not args.regenerate:
            results.append(existing_results[condition_id])
            stats['skipped'] += 1
            continue
        
        query_count = len(cond.get('query_results', []))
        if query_count == 0:
            print(f"[{i+1}/{len(search_results)}] ⏭️ {q_short} - no query results")
            stats['skipped'] += 1
            continue
        
        print(f"[{i+1}/{len(search_results)}] {q_short} ({query_count} queries)")
        
        analysis = analyze_condition(client, cond)
        
        if analysis:
            stats['success'] += 1
            
            result = {
                "condition_id": condition_id,
                "event_id": cond['event_id'],
                "event_title": cond['event_title'],
                "outcome_question": outcome_question,
                "market_price": market_price,
                "end_date": cond.get('end_date'),
                "condition_volume": cond.get('condition_volume'),
                "condition_liquidity": cond.get('condition_liquidity'),
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "predictability": analysis.predictability,
                "predictability_reason": analysis.predictability_reason,
                
                # Raw evidence scores from GPT
                "evidence_scores": [
                    {
                        "query_id": e.query_id,
                        "key_finding": e.key_finding,
                        "reliability": e.reliability,
                        "reliability_reason": e.reliability_reason,
                        "recency": e.recency,
                        "recency_reason": e.recency_reason,
                        "relevance": e.relevance,
                        "relevance_reason": e.relevance_reason,
                        "specificity": e.specificity,
                        "specificity_reason": e.specificity_reason,
                        "direction": e.direction,
                        "direction_reason": e.direction_reason,
                        "strength": e.strength,
                        "strength_reason": e.strength_reason
                    }
                    for e in analysis.evidence_scores
                ]
            }
            
            results.append(result)
            
            # Save incrementally
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"    ✅ Scored {len(analysis.evidence_scores)} queries")
        else:
            stats['failed'] += 1
            print(f"    ❌ Analysis failed")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved {len(results)} analyses to {output_path}")
    print(f"📊 Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
