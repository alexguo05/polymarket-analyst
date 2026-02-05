#!/usr/bin/env python3
"""
Analyze search results and score evidence quality.

Functions for pipeline:
    analyze_evidence(search_results: list[ConditionSearchResults]) -> list[ConditionAnalysis]

CLI Usage:
    python analyze_evidence.py              # Analyze all search results
    python analyze_evidence.py --limit 10   # Limit to 10 conditions
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import (
    ConditionSearchResults,
    EvidenceScore as EvidenceScoreModel,
    ConditionAnalysis as ConditionAnalysisModel,
)
from src.cost_tracker import UsageStats
from config.settings import settings

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CONFIGURATION (from settings.py)
# =============================================================================
MODEL = settings.analysis_model  # Set in config/settings.py or ANALYSIS_MODEL env var
MODEL_CONFIG = settings.model_config_map
RATE_LIMIT_DELAY = settings.analysis_rate_limit_delay

# =============================================================================
# RUBRICS
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

PREDICTABILITY (how predictable this market is from information, 0.0-1.0):
- 0.80-1.00: Highly predictable (structured process, reliable sources, clear signal)
- 0.60-0.79: Moderately predictable (some structure + meaningful data)
- 0.40-0.59: Mixed predictability (signal exists but noisy or partial)
- 0.20-0.39: Low predictability (high noise, weak signal, hard to verify)
- 0.00-0.19: Essentially random (coin-flip, phrasing games, very low signal)
"""


# =============================================================================
# PYDANTIC SCHEMAS (for OpenAI structured output)
# =============================================================================

class EvidenceScoreSchema(BaseModel):
    """Schema for OpenAI structured output."""
    query_id: str
    key_finding: str
    reliability: float = Field(ge=0.0, le=1.0)
    reliability_reason: str
    recency: float = Field(ge=0.0, le=1.0)
    recency_reason: str
    relevance: float = Field(ge=0.0, le=1.0)
    relevance_reason: str
    specificity: float = Field(ge=0.0, le=1.0)
    specificity_reason: str
    direction: int = Field(ge=-1, le=1)
    direction_reason: str


class ConditionAnalysisSchema(BaseModel):
    """Schema for OpenAI structured output."""
    predictability: float = Field(ge=0.0, le=1.0)
    predictability_reason: str
    evidence_scores: list[EvidenceScoreSchema]


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _build_analysis_prompt(search_results: ConditionSearchResults, today_date: str) -> str:
    """Build prompt for GPT to score evidence."""
    
    # Build evidence text from query results
    evidence_sections = []
    for i, qr in enumerate(search_results.query_results):
        citation_info = f"CITATIONS PROVIDED: {len(qr.citations)}" if qr.citations else "NO CITATIONS (cannot verify sources)"
        
        evidence_sections.append(f"""
### Query {qr.query_id} [{qr.category}]
**Question:** {qr.query}

**{citation_info}**

**Response:**
{qr.response[:8000]}
""")
    
    evidence_text = "\n---\n".join(evidence_sections)
    
    prompt = f"""You are scoring evidence quality for a prediction market analysis.

## CONDITION BEING EVALUATED
**Event:** {search_results.event_title}
**Outcome Question:** {search_results.outcome_question}

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
- query_id: The ID shown in the header
- key_finding: One sentence summary of the main finding
- reliability (0.0-1.0) + reliability_reason
- recency (0.0-1.0) + recency_reason
- relevance (0.0-1.0) + relevance_reason  
- specificity (0.0-1.0) + specificity_reason
- direction: +1 (increases YES probability), -1 (decreases), or 0 (neutral)
- direction_reason: Why this direction?

IMPORTANT:
- If NO CITATIONS, cap reliability at 0.60 maximum
- Be specific - cite details from the evidence in your reasoning
- Score each query independently

Return ONLY valid JSON matching the schema (no markdown, no extra text).
"""
    return prompt


def _add_additional_properties_false(obj):
    """Add additionalProperties: false to all objects in schema."""
    if isinstance(obj, dict):
        if obj.get("type") == "object" or "properties" in obj:
            obj["additionalProperties"] = False
        for v in obj.values():
            _add_additional_properties_false(v)
    elif isinstance(obj, list):
        for item in obj:
            _add_additional_properties_false(item)


def _call_chat_api(client: OpenAI, prompt: str) -> tuple[ConditionAnalysisSchema | None, dict]:
    """Call Chat Completions API for standard models (gpt-5.2)."""
    config = MODEL_CONFIG.get(MODEL, MODEL_CONFIG["gpt-5.2"])
    pricing = config["pricing"]
    
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert analyst scoring evidence quality. Be rigorous and specific. Return valid JSON matching the schema."},
            {"role": "user", "content": prompt}
        ],
        response_format=ConditionAnalysisSchema,
        max_completion_tokens=8000,
    )
    
    # Extract usage/cost
    usage_info = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    if hasattr(response, 'usage') and response.usage:
        input_tok = response.usage.prompt_tokens or 0
        output_tok = response.usage.completion_tokens or 0
        cost = (input_tok * pricing["input"] + output_tok * pricing["output"]) / 1_000_000
        
        print(f"    📊 Usage: {input_tok:,} input + {output_tok:,} output = {input_tok + output_tok:,} total [${cost:.4f}]")
        
        usage_info = {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "reasoning_tokens": 0,
            "cost_usd": cost,
        }
    
    # Get parsed response
    parsed = response.choices[0].message.parsed
    if parsed is None:
        if response.choices[0].message.refusal:
            print(f"    ⚠️  Model refused: {response.choices[0].message.refusal}")
        else:
            print("    ❌ Error: Empty parsed response")
        return None, usage_info
    
    return parsed, usage_info


def _call_responses_api(client: OpenAI, prompt: str, schema: dict) -> tuple[ConditionAnalysisSchema | None, dict]:
    """Call Responses API for reasoning models (gpt-5.2-pro)."""
    config = MODEL_CONFIG.get(MODEL, MODEL_CONFIG["gpt-5.2-pro"])
    pricing = config["pricing"]
    
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
    
    # Extract usage/cost - reasoning models have reasoning tokens!
    usage_info = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        input_tok = getattr(usage, 'input_tokens', 0) or 0
        output_tok = getattr(usage, 'output_tokens', 0) or 0
        
        # Reasoning tokens (hidden thinking) - these are billed!
        reasoning_tok = 0
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            reasoning_tok = getattr(usage.output_tokens_details, 'reasoning_tokens', 0) or 0
        
        cost = (input_tok * pricing["input"] + output_tok * pricing["output"] + reasoning_tok * pricing.get("reasoning", pricing["output"])) / 1_000_000
        
        print(f"    📊 Usage: {input_tok:,} input + {output_tok:,} output + {reasoning_tok:,} reasoning = {input_tok + output_tok + reasoning_tok:,} total [${cost:.4f}]")
        
        usage_info = {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "reasoning_tokens": reasoning_tok,
            "cost_usd": cost,
        }
    
    # Extract text from Responses API
    raw_text = ""
    if hasattr(response, 'output_text') and response.output_text:
        raw_text = response.output_text.strip()
    elif hasattr(response, 'output') and response.output:
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
        return None, usage_info
    
    return ConditionAnalysisSchema.model_validate_json(raw_text), usage_info


def _analyze_condition(
    client: OpenAI, 
    search_results: ConditionSearchResults,
    max_retries: int = 3
) -> tuple[ConditionAnalysisSchema | None, dict]:
    """
    Score evidence for a single condition using GPT.
    Automatically selects API based on MODEL setting.
    
    Returns:
        Tuple of (analysis schema or None, usage dict with input_tokens, output_tokens, cost_usd)
    """
    usage_info = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = _build_analysis_prompt(search_results, today)
    
    # Get model config
    config = MODEL_CONFIG.get(MODEL, MODEL_CONFIG["gpt-5.2"])
    api_type = config["api"]
    
    # Debug: Show which API is being used (first call only)
    if not hasattr(_analyze_condition, '_logged'):
        print(f"    🔧 Using MODEL={MODEL}, API={api_type}")
        _analyze_condition._logged = True
    
    # Build JSON schema for Responses API (only needed for pro)
    schema = None
    if api_type == "responses":
        schema = ConditionAnalysisSchema.model_json_schema()
        _add_additional_properties_false(schema)
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                _add_additional_properties_false(def_schema)
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            # Call appropriate API based on model
            if api_type == "responses":
                return _call_responses_api(client, prompt, schema)
            else:
                return _call_chat_api(client, prompt)

        except Exception as e:
            error_str = str(e)
            print(f"    ❌ Error type: {type(e).__name__}")
            print(f"    ❌ Error message: {e}")
            
            if hasattr(e, 'response'):
                try:
                    print(f"    ❌ Response status: {e.response.status_code}")
                except:
                    pass
            
            # Rate limit - exponential backoff
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = 30 * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"    ⏳ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # Connection error - shorter backoff
            if "connection" in error_str.lower() or "timeout" in error_str.lower():
                wait_time = 10 * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"    🔄 Connection error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            return None, usage_info
    
    return None, usage_info


# =============================================================================
# PUBLIC API (for pipeline use)
# =============================================================================

def analyze_evidence(
    search_results: list[ConditionSearchResults],
    client: OpenAI = None,
    verbose: bool = True,
) -> tuple[list[ConditionAnalysisModel], UsageStats]:
    """
    Analyze search results and score evidence quality.
    
    This is the main entry point for the pipeline.
    
    Args:
        search_results: List of ConditionSearchResults to analyze
        client: Optional OpenAI client (created if not provided)
        verbose: Print progress
        
    Returns:
        Tuple of (List of ConditionAnalysis models, UsageStats with cost info)
    """
    if client is None:
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY not found in environment")
        client = OpenAI(api_key=api_key)
    
    results = []
    usage_stats = UsageStats()
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for i, sr in enumerate(search_results):
        if verbose:
            q_short = sr.outcome_question[:45] + "..." if len(sr.outcome_question) > 45 else sr.outcome_question
            query_count = len(sr.query_results)
            
            if query_count == 0:
                print(f"[{i+1}/{len(search_results)}] ⏭️ {q_short} - no query results")
                stats['skipped'] += 1
                continue
            
            print(f"[{i+1}/{len(search_results)}] {q_short} ({query_count} queries)")
        
        analysis, usage_info = _analyze_condition(client, sr)
        
        # Track in aggregate stats (including reasoning tokens!)
        usage_stats.input_tokens += usage_info["input_tokens"]
        usage_stats.output_tokens += usage_info["output_tokens"]
        reasoning_tok = usage_info.get("reasoning_tokens", 0)
        usage_stats.total_tokens += usage_info["input_tokens"] + usage_info["output_tokens"] + reasoning_tok
        usage_stats.estimated_cost += usage_info["cost_usd"]
        usage_stats.requests += 1
        
        if analysis:
            stats['success'] += 1
            
            # Convert schema to shared model
            evidence_scores = [
                EvidenceScoreModel(
                    query_id=e.query_id,
                    key_finding=e.key_finding,
                    reliability=e.reliability,
                    reliability_reason=e.reliability_reason,
                    recency=e.recency,
                    recency_reason=e.recency_reason,
                    relevance=e.relevance,
                    relevance_reason=e.relevance_reason,
                    specificity=e.specificity,
                    specificity_reason=e.specificity_reason,
                    direction=e.direction,
                    direction_reason=e.direction_reason,
                )
                for e in analysis.evidence_scores
            ]
            
            result = ConditionAnalysisModel(
                condition_id=sr.condition_id,
                event_id=sr.event_id,
                event_title=sr.event_title,
                outcome_question=sr.outcome_question,
                market_price=sr.yes_price,
                volume=sr.volume,
                liquidity=sr.liquidity,
                end_date=sr.end_date,
                predictability=analysis.predictability,
                predictability_reason=analysis.predictability_reason,
                evidence_scores=evidence_scores,
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                model=MODEL,
                # Cost tracking for this condition
                input_tokens=usage_info["input_tokens"],
                output_tokens=usage_info["output_tokens"],
                reasoning_tokens=usage_info.get("reasoning_tokens", 0),
                cost_usd=usage_info["cost_usd"],
            )
            results.append(result)
            
            if verbose:
                print(f"    ✅ Scored {len(evidence_scores)} queries [${usage_info['cost_usd']:.4f}]")
        else:
            stats['failed'] += 1
            if verbose:
                print(f"    ❌ Analysis failed")
        
        # Rate limiting
        if i < len(search_results) - 1:
            time.sleep(RATE_LIMIT_DELAY)
    
    if verbose:
        print(f"\n📊 Analysis: {stats['success']} succeeded, {stats['failed']} failed, {stats['skipped']} skipped")
        print(f"   💰 Total: {usage_stats.summary()}")
    
    return results, usage_stats


# =============================================================================
# MAIN (CLI entry point)
# =============================================================================

def main():
    """CLI entry point."""
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
                        help="Re-analyze all conditions")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to JSON files")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load search results from JSON
    print(f"📂 Loading search results from {input_path}")
    with open(input_path) as f:
        raw_results = json.load(f)
    
    # Convert to ConditionSearchResults models
    from models.evidence import QueryResult
    search_results = []
    for r in raw_results:
        query_results = [QueryResult(**qr) for qr in r.get('query_results', [])]
        search_results.append(ConditionSearchResults(
            condition_id=r['condition_id'],
            event_id=r['event_id'],
            event_title=r['event_title'],
            outcome_question=r['outcome_question'],
            yes_price=r['yes_price'],
            volume=r.get('condition_volume'),
            liquidity=r.get('condition_liquidity'),
            end_date=r.get('end_date'),
            query_results=query_results,
            searched_at=r.get('searched_at', ''),
        ))
    
    print(f"📊 Found {len(search_results)} conditions")
    
    # Load existing results to skip already-processed
    existing_condition_ids = set()
    existing_results = []
    if output_path.exists() and not args.regenerate:
        with open(output_path) as f:
            existing_results = json.load(f)
        existing_condition_ids = {r['condition_id'] for r in existing_results}
        print(f"   Found {len(existing_condition_ids)} existing analyses")
    
    # Filter out already-processed conditions
    results_to_process = [sr for sr in search_results if sr.condition_id not in existing_condition_ids]
    skipped = len(search_results) - len(results_to_process)
    
    if args.limit:
        results_to_process = results_to_process[:args.limit]
    
    print(f"📊 Processing {len(results_to_process)} conditions ({skipped} skipped)")
    
    if args.dry_run:
        if results_to_process:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            prompt = _build_analysis_prompt(results_to_process[0], today)
            print("\n" + "="*60)
            print("SAMPLE PROMPT:")
            print("="*60)
            print(prompt[:3000] + "\n...[truncated]...")
        return None
    
    if not results_to_process:
        print("\n✅ All conditions already analyzed. Use --regenerate to force.")
        return []
    
    # Analyze evidence
    print(f"\n🚀 Analyzing evidence using {MODEL}...\n")
    new_results, stats = analyze_evidence(results_to_process, verbose=True)
    
    if args.no_save:
        print(f"\n✅ Analyzed {len(new_results)} conditions (not saved)")
        return new_results, stats
    
    # Convert to dicts for JSON
    new_results_dicts = []
    for r in new_results:
        d = r.model_dump()
        # Convert EvidenceScore models to dicts
        d['evidence_scores'] = [e.model_dump() for e in r.evidence_scores]
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
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Analyzed {len(new_results)} conditions")
    print(f"✅ Total {len(final_results)} saved to {output_path}")
    
    return new_results


if __name__ == "__main__":
    main()
