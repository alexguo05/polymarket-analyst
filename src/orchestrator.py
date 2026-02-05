#!/usr/bin/env python3
"""
Pipeline Orchestrator - Coordinates the full prediction market analysis pipeline.

Calls pipeline functions directly (no subprocess) with typed data flowing between steps.

Usage:
    python orchestrator.py              # Run full pipeline
    python orchestrator.py --limit 5    # Limit to 5 conditions
    python orchestrator.py --dry-run    # Show what would be processed
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline functions
from src.get_markets import scan_markets, get_filtered_conditions
from src.generate_queries import generate_queries
from src.execute_searches import execute_searches
from src.analyze_evidence import analyze_evidence
from src.compute_prediction import compute_predictions

# Import models
from models.market import CandidateMarket, FilteredCondition
from models.query import ConditionQueries
from models.evidence import ConditionSearchResults, ConditionAnalysis
from models.prediction import Prediction, PipelineOutput, CostSummary


def _save_json(data: list, filepath: Path, verbose: bool = True):
    """Save data to JSON file with feedback."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    if verbose:
        print(f"   💾 Saved to {filepath.name}")


def run_pipeline(
    limit: Optional[int] = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
    save_intermediates: bool = True,
) -> Optional[list[Prediction]]:
    """
    Run the full prediction market analysis pipeline.
    
    Args:
        limit: Maximum number of conditions to process
        dry_run: If True, just show what would be processed without API calls
        skip_existing: Skip conditions that already have results
        verbose: Print progress
        save_intermediates: Save intermediate JSON files
        
    Returns:
        List of Prediction models, or None if dry_run
    """
    start_time = datetime.now(timezone.utc)
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print(f"🚀 POLYMARKET ANALYST PIPELINE")
    print(f"   Started: {start_time.isoformat()}")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Scan Markets
    # =========================================================================
    print(f"\n{'─'*70}")
    print("📡 STEP 1: Scanning markets from Polymarket API...")
    print(f"{'─'*70}")
    
    markets = scan_markets(verbose=verbose)
    print(f"   Found {len(markets)} candidate markets")
    
    conditions = get_filtered_conditions(markets, verbose=verbose)
    print(f"   Found {len(conditions)} tradeable conditions")
    
    if save_intermediates:
        _save_json([m.model_dump() for m in markets], data_dir / "candidate_markets.json", verbose)
        _save_json([c.model_dump() for c in conditions], data_dir / "candidate_markets_filtered.json", verbose)
    
    if not conditions:
        print("❌ No conditions found. Exiting.")
        return None
    
    # Apply limit
    if limit and limit < len(conditions):
        conditions = conditions[:limit]
        print(f"   Limited to {len(conditions)} conditions")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would process {len(conditions)} conditions:")
        for i, c in enumerate(conditions[:10]):
            q_short = c.outcome_question[:55] + "..." if len(c.outcome_question) > 55 else c.outcome_question
            print(f"   {i+1}. {q_short} ({c.yes_price:.1%})")
        if len(conditions) > 10:
            print(f"   ... and {len(conditions) - 10} more")
        return None
    
    # Track total costs
    total_cost = 0.0
    
    # =========================================================================
    # STEP 2: Generate Queries
    # =========================================================================
    print(f"\n{'─'*70}")
    print("🔍 STEP 2: Generating search queries with GPT...")
    print(f"{'─'*70}")
    
    condition_queries, query_stats = generate_queries(conditions, verbose=verbose)
    total_cost += query_stats.estimated_cost
    print(f"   Generated queries for {len(condition_queries)} conditions")
    
    if save_intermediates:
        queries_dicts = [cq.model_dump() for cq in condition_queries]
        _save_json(queries_dicts, data_dir / "market_queries.json", verbose)
    
    if not condition_queries:
        print("❌ No queries generated. Exiting.")
        return None
    
    # =========================================================================
    # STEP 3: Execute Searches
    # =========================================================================
    print(f"\n{'─'*70}")
    print("🌐 STEP 3: Executing searches with Perplexity...")
    print(f"{'─'*70}")
    
    search_results, search_stats = execute_searches(condition_queries, verbose=verbose)
    total_cost += search_stats.estimated_cost
    print(f"   Searched {len(search_results)} conditions")
    
    if save_intermediates:
        results_dicts = [sr.model_dump() for sr in search_results]
        _save_json(results_dicts, data_dir / "search_results.json", verbose)
    
    if not search_results:
        print("❌ No search results. Exiting.")
        return None
    
    # =========================================================================
    # STEP 4: Analyze Evidence
    # =========================================================================
    print(f"\n{'─'*70}")
    print("🧠 STEP 4: Analyzing evidence with GPT...")
    print(f"{'─'*70}")
    
    analyses, analysis_stats = analyze_evidence(search_results, verbose=verbose)
    total_cost += analysis_stats.estimated_cost
    print(f"   Analyzed {len(analyses)} conditions")
    
    if save_intermediates:
        analyses_dicts = [a.model_dump() for a in analyses]
        _save_json(analyses_dicts, data_dir / "evidence_analysis.json", verbose)
    
    if not analyses:
        print("❌ No analyses completed. Exiting.")
        return None
    
    # =========================================================================
    # STEP 5: Compute Predictions
    # =========================================================================
    print(f"\n{'─'*70}")
    print("📊 STEP 5: Computing final predictions...")
    print(f"{'─'*70}")
    
    predictions = compute_predictions(analyses, verbose=verbose)
    print(f"   Computed {len(predictions)} predictions")
    
    # Save final output with cost summary
    if save_intermediates:
        cost_summary = CostSummary(
            query_generation=query_stats.estimated_cost,
            search_execution=search_stats.estimated_cost,
            evidence_analysis=analysis_stats.estimated_cost,
            total_usd=total_cost,
            total_requests=query_stats.requests + search_stats.requests + analysis_stats.requests,
            total_tokens=query_stats.total_tokens + search_stats.total_tokens + analysis_stats.total_tokens,
        )
        
        output = PipelineOutput(
            computed_at=datetime.now(timezone.utc).isoformat(),
            parameters={'evidence_scale': 0.5},
            summary={
                'total': len(predictions),
                'yes_signals': sum(1 for p in predictions if p.direction == 'YES'),
                'no_signals': sum(1 for p in predictions if p.direction == 'NO'),
                'hold_signals': sum(1 for p in predictions if p.direction == 'HOLD'),
            },
            cost=cost_summary,
            predictions=predictions,
        )
        _save_json(output.model_dump(), data_dir / "predictions.json", verbose)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    yes_count = sum(1 for p in predictions if p.direction == 'YES')
    no_count = sum(1 for p in predictions if p.direction == 'NO')
    hold_count = sum(1 for p in predictions if p.direction == 'HOLD')
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"   Duration:         {duration:.1f} seconds")
    print(f"   Conditions:       {len(predictions)}")
    print(f"   BUY YES signals:  {yes_count}")
    print(f"   BUY NO signals:   {no_count}")
    print(f"   HOLD signals:     {hold_count}")
    print(f"   💰 Total cost:     ${total_cost:.4f}")
    print(f"\n   Results saved to: {data_dir}/predictions.json")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Polymarket prediction pipeline"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of conditions to process"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without API calls"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save intermediate JSON files"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()
    
    predictions = run_pipeline(
        limit=args.limit,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        save_intermediates=not args.no_save,
    )
    
    if predictions:
        # Show top predictions
        predictions.sort(key=lambda p: abs(p.edge), reverse=True)
        print(f"\n🔥 TOP SIGNALS BY EDGE:")
        for i, p in enumerate(predictions[:5]):
            direction_symbol = "🟢" if p.direction == "YES" else ("🔴" if p.direction == "NO" else "⚪")
            q_short = p.outcome_question[:50] + "..." if len(p.outcome_question) > 50 else p.outcome_question
            print(f"   {i+1}. {direction_symbol} {q_short}")
            print(f"      Market: {p.market_price:.1%} → Predicted: {p.predicted_price:.1%} (edge: {p.edge:+.1%})")


if __name__ == "__main__":
    main()
