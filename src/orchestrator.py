#!/usr/bin/env python3
"""
Pipeline Orchestrator - Coordinates the full prediction market analysis pipeline.

Calls pipeline functions directly (no subprocess) with typed data flowing between steps.
Saves results to both JSON files and SQLite database.

Usage:
    python orchestrator.py              # Run full pipeline
    python orchestrator.py --limit 5    # Limit to 5 conditions
    python orchestrator.py --dry-run    # Show what would be processed
    python orchestrator.py --no-db      # Skip database saving
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

# Import database
from db.connection import SessionLocal
from db.models import (
    Event as EventDB,
    Condition as ConditionDB,
    Prediction as PredictionDB,
    EvidenceScore as EvidenceScoreDB,
    SearchResult as SearchResultDB,
    GeneratedQuery as GeneratedQueryDB,
    PriceHistory as PriceHistoryDB,
    PipelineRun as PipelineRunDB,
    PipelineStatus,
    Recommendation,
)

# Import config for model names
from config.settings import settings


def _save_json(data: list, filepath: Path, verbose: bool = True):
    """Save data to JSON file with feedback."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    if verbose:
        print(f"   💾 Saved to {filepath.name}")


def _save_to_database(
    markets: list[CandidateMarket],
    conditions: list[FilteredCondition],
    condition_queries: list[ConditionQueries],
    search_results: list[ConditionSearchResults],
    analyses: list,  # ConditionAnalysis models
    predictions: list[Prediction],
    pipeline_run_id: int = None,
    verbose: bool = True,
) -> dict:
    """
    Save pipeline results to SQLite database.
    
    Returns:
        Dict with counts of records created
    """
    db = SessionLocal()
    stats = {
        "events": 0, "conditions": 0, "generated_queries": 0,
        "search_results": 0, "predictions": 0, "evidence_scores": 0, "price_history": 0
    }
    
    try:
        # Build lookups
        analysis_map = {a.condition_id: a for a in analyses}
        prediction_map = {p.condition_id: p for p in predictions}
        query_map = {cq.condition_id: cq for cq in condition_queries}
        search_map = {sr.condition_id: sr for sr in search_results}
        
        # Build market lookup for filter_scores
        market_map = {}
        for m in markets:
            market_map[m.event_id] = m
        
        # Group conditions by event
        events_map = {}
        for cond in conditions:
            if cond.event_id not in events_map:
                events_map[cond.event_id] = {
                    "title": cond.event_title,
                    "conditions": []
                }
            events_map[cond.event_id]["conditions"].append(cond)
        
        # Create/update events and conditions
        condition_db_map = {}  # condition_id -> DB Condition.id
        
        for event_id, event_data in events_map.items():
            event_db = db.query(EventDB).filter(
                EventDB.polymarket_id == event_id
            ).first()
            
            # Get filter scores from market if available
            market = market_map.get(event_id)
            filter_scores = getattr(market, 'filter_scores', None) if market else None
            
            if not event_db:
                event_db = EventDB(
                    polymarket_id=event_id,
                    title=event_data["title"],
                    slug=getattr(market, 'slug', None) if market else None,
                    volume=getattr(market, 'volume', None) if market else None,
                    liquidity=getattr(market, 'liquidity', None) if market else None,
                    tags=getattr(market, 'tags', None) if market else None,
                    complexity_score=filter_scores.get('complexity_score') if filter_scores else None,
                    edge_potential=filter_scores.get('edge_potential') if filter_scores else None,
                )
                db.add(event_db)
                db.flush()
                stats["events"] += 1
            else:
                # Update filter scores if available
                if filter_scores:
                    event_db.complexity_score = filter_scores.get('complexity_score')
                    event_db.edge_potential = filter_scores.get('edge_potential')
                event_db.updated_at = datetime.utcnow()
            
            # Create/update conditions
            for cond in event_data["conditions"]:
                cond_db = db.query(ConditionDB).filter(
                    ConditionDB.condition_id == cond.condition_id
                ).first()
                
                end_date = None
                if cond.end_date:
                    try:
                        end_date = datetime.fromisoformat(cond.end_date.replace('Z', '+00:00'))
                    except:
                        pass
                
                if not cond_db:
                    cond_db = ConditionDB(
                        event_id=event_db.id,
                        condition_id=cond.condition_id,
                        question=cond.outcome_question,
                        yes_token_id=cond.yes_token_id,
                        no_token_id=cond.no_token_id,
                        yes_price=cond.yes_price,
                        volume=cond.volume,
                        liquidity=cond.liquidity,
                        end_date=end_date,
                    )
                    db.add(cond_db)
                    db.flush()
                    stats["conditions"] += 1
                else:
                    cond_db.yes_token_id = cond.yes_token_id
                    cond_db.no_token_id = cond.no_token_id
                    cond_db.yes_price = cond.yes_price
                    cond_db.volume = cond.volume
                    cond_db.liquidity = cond.liquidity
                    cond_db.updated_at = datetime.utcnow()
                
                condition_db_map[cond.condition_id] = cond_db.id
                
                # Save price history snapshot
                price_hist = PriceHistoryDB(
                    condition_id=cond_db.id,
                    yes_price=cond.yes_price,
                    volume=cond.volume,
                    liquidity=cond.liquidity,
                    source="pipeline",
                )
                db.add(price_hist)
                stats["price_history"] += 1
        
        # Save generated queries
        for cq in condition_queries:
            cond_db_id = condition_db_map.get(cq.condition_id)
            if not cond_db_id:
                continue
            
            for q in cq.queries:
                gq_db = GeneratedQueryDB(
                    condition_id=cond_db_id,
                    pipeline_run_id=pipeline_run_id,
                    query_id=q.query_id,
                    query_text=q.query,
                    purpose=q.purpose,
                    category=q.category,
                    priority=q.priority,
                    model=cq.model,
                )
                db.add(gq_db)
                stats["generated_queries"] += 1
        
        # Save search results
        for sr in search_results:
            cond_db_id = condition_db_map.get(sr.condition_id)
            if not cond_db_id:
                continue
            
            for result in sr.query_results:
                searched_at = None
                if hasattr(result, 'searched_at') and result.searched_at:
                    try:
                        searched_at = datetime.fromisoformat(result.searched_at.replace('Z', '+00:00'))
                    except:
                        pass
                
                sr_db = SearchResultDB(
                    condition_id=cond_db_id,
                    pipeline_run_id=pipeline_run_id,
                    query_id=result.query_id,
                    query_text=result.query,
                    category=getattr(result, 'category', None),
                    response_text=result.response,
                    citations=result.citations,
                    model=settings.search_model,
                    searched_at=searched_at,
                )
                db.add(sr_db)
                stats["search_results"] += 1
        
        # Save predictions with all computed values
        for pred in predictions:
            cond_db_id = condition_db_map.get(pred.condition_id)
            if not cond_db_id:
                continue
            
            # Map direction string to recommendation enum
            rec_map = {
                "YES": Recommendation.BUY_YES,
                "NO": Recommendation.BUY_NO,
                "HOLD": Recommendation.HOLD,
            }
            rec_enum = rec_map.get(pred.direction, Recommendation.HOLD)
            
            # Get analysis for predictability_reason
            analysis = analysis_map.get(pred.condition_id)
            
            pred_db = PredictionDB(
                condition_id=cond_db_id,
                pipeline_run_id=pipeline_run_id,
                market_price=pred.market_price,
                predicted_price=pred.predicted_price,
                edge=pred.edge,
                edge_percent=pred.edge_percent,
                days_until_end=pred.days_until_end,
                apy=pred.apy,
                predictability=pred.predictability,
                predictability_reason=getattr(analysis, 'predictability_reason', None) if analysis else None,
                recommendation=rec_enum,
                num_evidence=pred.num_evidence,
                total_raw_adjustment=pred.total_raw_adjustment,
                scaled_adjustment=pred.scaled_adjustment,
                analysis_model=getattr(analysis, 'model', None) if analysis else None,
            )
            db.add(pred_db)
            db.flush()
            stats["predictions"] += 1
            
            # Save evidence scores with computed values
            if analysis and hasattr(analysis, 'evidence_scores'):
                # Build breakdown map for computed values
                breakdown_map = {}
                if hasattr(pred, 'evidence_breakdown'):
                    for eb in pred.evidence_breakdown:
                        breakdown_map[eb.query_id] = eb
                
                for ev in analysis.evidence_scores:
                    breakdown = breakdown_map.get(ev.query_id)
                    
                    ev_db = EvidenceScoreDB(
                        prediction_id=pred_db.id,
                        query_id=ev.query_id,
                        key_finding=ev.key_finding,
                        # Quality scores with reasons
                        reliability=ev.reliability,
                        reliability_reason=ev.reliability_reason,
                        recency=ev.recency,
                        recency_reason=ev.recency_reason,
                        relevance=ev.relevance,
                        relevance_reason=ev.relevance_reason,
                        specificity=ev.specificity,
                        specificity_reason=ev.specificity_reason,
                        # Direction
                        direction=ev.direction,
                        direction_reason=ev.direction_reason,
                        # Computed values from prediction
                        quality_score=breakdown.quality if breakdown else None,
                        strength=breakdown.strength if breakdown else None,
                        raw_adjustment=breakdown.raw_adjustment if breakdown else None,
                    )
                    db.add(ev_db)
                    stats["evidence_scores"] += 1
        
        db.commit()
        
        if verbose:
            print(f"   🗄️  Saved to database:")
            print(f"      • {stats['events']} events, {stats['conditions']} conditions")
            print(f"      • {stats['generated_queries']} queries, {stats['search_results']} search results")
            print(f"      • {stats['predictions']} predictions, {stats['evidence_scores']} evidence scores")
            print(f"      • {stats['price_history']} price snapshots")
        
        return stats
        
    except Exception as e:
        db.rollback()
        if verbose:
            print(f"   ❌ Database error: {e}")
        raise
    finally:
        db.close()


def _create_pipeline_run(
    trigger: str = "cli",
    parameters: dict = None,
) -> int:
    """Create a new pipeline run record and return its ID."""
    db = SessionLocal()
    try:
        run = PipelineRunDB(
            status=PipelineStatus.RUNNING,
            trigger=trigger,
            started_at=datetime.utcnow(),
            parameters=parameters or {},
            # Model names from settings
            query_model=settings.query_model,
            search_model=settings.search_model,
            analysis_model=settings.analysis_model,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run.id
    finally:
        db.close()


def _update_pipeline_run(
    run_id: int,
    status: PipelineStatus,
    markets_scanned: int = 0,
    conditions_analyzed: int = 0,
    predictions_made: int = 0,
    query_stats = None,
    search_stats = None,
    analysis_stats = None,
    error_message: str = None,
):
    """Update pipeline run record with final stats and costs."""
    db = SessionLocal()
    try:
        run = db.query(PipelineRunDB).filter(PipelineRunDB.id == run_id).first()
        if run:
            run.status = status
            run.completed_at = datetime.utcnow()
            run.markets_scanned = markets_scanned
            run.conditions_analyzed = conditions_analyzed
            run.predictions_made = predictions_made
            run.error_message = error_message
            
            # Cost tracking
            total_cost = 0.0
            
            if query_stats:
                run.query_gen_tokens = query_stats.total_tokens
                run.query_gen_cost = query_stats.estimated_cost
                total_cost += query_stats.estimated_cost
            
            if search_stats:
                run.search_requests = search_stats.requests
                run.search_cost = search_stats.estimated_cost
                total_cost += search_stats.estimated_cost
            
            if analysis_stats:
                run.analysis_input_tokens = analysis_stats.input_tokens
                run.analysis_output_tokens = analysis_stats.output_tokens
                run.analysis_reasoning_tokens = analysis_stats.reasoning_tokens
                run.analysis_cost = analysis_stats.estimated_cost
                total_cost += analysis_stats.estimated_cost
            
            run.total_cost = total_cost
            
            db.commit()
    finally:
        db.close()


def run_pipeline(
    limit: Optional[int] = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
    save_intermediates: bool = True,
    save_to_db: bool = True,
) -> Optional[list[Prediction]]:
    """
    Run the full prediction market analysis pipeline.
    
    Args:
        limit: Maximum number of conditions to process
        dry_run: If True, just show what would be processed without API calls
        skip_existing: Skip conditions that already have results
        verbose: Print progress
        save_intermediates: Save intermediate JSON files
        save_to_db: Save results to SQLite database
        
    Returns:
        List of Prediction models, or None if dry_run
    """
    start_time = datetime.now(timezone.utc)
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Pipeline parameters
    pipeline_params = {
        'evidence_scale': 0.5,
        'min_prob': 0.01,
        'max_prob': 0.99,
    }
    
    # Create pipeline run record in DB
    pipeline_run_id = None
    if save_to_db and not dry_run:
        try:
            pipeline_run_id = _create_pipeline_run(
                trigger="cli",
                parameters=pipeline_params,
            )
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not create pipeline run record: {e}")
    
    print("=" * 70)
    print(f"🚀 POLYMARKET ANALYST PIPELINE")
    print(f"   Started: {start_time.isoformat()}")
    if pipeline_run_id:
        print(f"   Run ID: {pipeline_run_id}")
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
            parameters=pipeline_params,
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
    # STEP 6: Save to Database
    # =========================================================================
    if save_to_db and pipeline_run_id:
        print(f"\n{'─'*70}")
        print("🗄️  STEP 6: Saving to database...")
        print(f"{'─'*70}")
        
        try:
            _save_to_database(
                markets=markets,
                conditions=conditions,
                condition_queries=condition_queries,
                search_results=search_results,
                analyses=analyses,
                predictions=predictions,
                pipeline_run_id=pipeline_run_id,
                verbose=verbose,
            )
            _update_pipeline_run(
                run_id=pipeline_run_id,
                status=PipelineStatus.COMPLETED,
                markets_scanned=len(markets),
                conditions_analyzed=len(conditions),
                predictions_made=len(predictions),
                query_stats=query_stats,
                search_stats=search_stats,
                analysis_stats=analysis_stats,
            )
        except Exception as e:
            _update_pipeline_run(
                run_id=pipeline_run_id,
                status=PipelineStatus.FAILED,
                error_message=str(e),
            )
            if verbose:
                print(f"   ❌ Failed to save to database: {e}")
    
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
    print(f"\n   Results saved to:")
    print(f"      📄 {data_dir}/predictions.json")
    if save_to_db and pipeline_run_id:
        print(f"      🗄️  data/polymarket.db (run_id={pipeline_run_id})")
    
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
        "--no-db", action="store_true",
        help="Don't save to SQLite database"
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
        save_to_db=not args.no_db,
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
