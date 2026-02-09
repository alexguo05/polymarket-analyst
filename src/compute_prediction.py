#!/usr/bin/env python3
"""
Compute final probability predictions from evidence analysis.

Functions for pipeline:
    compute_predictions(analyses: list[ConditionAnalysis]) -> list[Prediction]

CLI Usage:
    python compute_prediction.py              # Compute predictions for all analyses
    python compute_prediction.py --verbose    # Show detailed evidence breakdown
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from dateutil import parser as date_parser
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import ConditionAnalysis, EvidenceScore
from models.prediction import Prediction, EvidenceBreakdown, PipelineOutput
from config.settings import settings

# =============================================================================
# CONFIGURATION (can be overridden by settings)
# =============================================================================

EVIDENCE_SCALE = settings.evidence_scale
MIN_PROB = settings.min_prob
MAX_PROB = settings.max_prob
MIN_EDGE_THRESHOLD = settings.min_edge_threshold


# =============================================================================
# MATH UTILITIES
# =============================================================================

def prob_to_logodds(p: float) -> float:
    """Convert probability to log-odds."""
    p = max(MIN_PROB, min(MAX_PROB, p))
    return math.log(p / (1 - p))


def logodds_to_prob(lo: float) -> float:
    """Convert log-odds back to probability."""
    p = 1 / (1 + math.exp(-lo))
    return max(MIN_PROB, min(MAX_PROB, p))


def parse_end_date(end_date_str: str) -> datetime | None:
    """Parse ISO date string to datetime."""
    if not end_date_str:
        return None
    try:
        return date_parser.isoparse(end_date_str)
    except (ValueError, TypeError):
        return None


def calculate_days_until_end(end_date_str: str) -> float | None:
    """Calculate days until the market ends."""
    end_date = parse_end_date(end_date_str)
    if not end_date:
        return None
    now = datetime.now(timezone.utc)
    delta = end_date - now
    return max(0, delta.total_seconds() / 86400)


def calculate_apy(edge: float, days_until_end: float, market_price: float) -> float | None:
    """Calculate annualized return (APY) for a bet."""
    if days_until_end is None or days_until_end <= 0:
        return None
    
    if edge > 0:
        return_per_dollar = edge / market_price if market_price > 0 else 0
    else:
        no_price = 1 - market_price
        return_per_dollar = abs(edge) / no_price if no_price > 0 else 0
    
    periods_per_year = 365 / days_until_end
    apy = (1 + return_per_dollar) ** periods_per_year - 1
    
    return min(apy, 100.0)  # Cap at 10,000% APY


# =============================================================================
# EVIDENCE COMPUTATION
# =============================================================================

def _compute_evidence_adjustment(evidence: EvidenceScore) -> EvidenceBreakdown:
    """Compute the adjustment from a single piece of evidence."""
    quality = (evidence.reliability + evidence.recency + evidence.relevance + evidence.specificity) / 4
    strength = quality  # Derived from quality, not from LLM
    raw_adjustment = evidence.direction * strength
    
    key_finding = evidence.key_finding[:80] + '...' if len(evidence.key_finding) > 80 else evidence.key_finding
    
    return EvidenceBreakdown(
        query_id=evidence.query_id,
        key_finding=key_finding,
        quality=quality,
        direction=evidence.direction,
        strength=strength,
        raw_adjustment=raw_adjustment,
    )


def _compute_single_prediction(
    analysis: ConditionAnalysis,
    evidence_scale: float = EVIDENCE_SCALE,
) -> Prediction:
    """Compute final probability prediction for a single condition."""
    market_price = analysis.market_price
    predictability = analysis.predictability
    
    # Compute adjustments for each piece of evidence
    evidence_breakdown = []
    total_raw_adjustment = 0.0
    
    for ev in analysis.evidence_scores:
        detail = _compute_evidence_adjustment(ev)
        evidence_breakdown.append(detail)
        total_raw_adjustment += detail.raw_adjustment
    
    # Scale adjustment by predictability
    scaled_adjustment = total_raw_adjustment * predictability * evidence_scale
    
    # Apply in log-odds space
    market_logodds = prob_to_logodds(market_price)
    final_logodds = market_logodds + scaled_adjustment
    final_prob = logodds_to_prob(final_logodds)
    
    # Compute edge
    edge = final_prob - market_price
    edge_pct = (edge / market_price) * 100 if market_price > 0 else 0
    
    # Calculate time and APY
    days_until_end = calculate_days_until_end(analysis.end_date)
    apy = calculate_apy(edge, days_until_end, market_price) if days_until_end else None
    
    # Determine direction based on edge threshold from settings
    if edge > MIN_EDGE_THRESHOLD:
        direction = "YES"
    elif edge < -MIN_EDGE_THRESHOLD:
        direction = "NO"
    else:
        direction = "HOLD"
    
    return Prediction(
        condition_id=analysis.condition_id,
        event_id=analysis.event_id,
        event_title=analysis.event_title,
        outcome_question=analysis.outcome_question,
        market_price=market_price,
        predicted_price=round(final_prob, 4),
        edge=round(edge, 4),
        edge_percent=round(edge_pct, 2),
        end_date=analysis.end_date,
        days_until_end=round(days_until_end, 1) if days_until_end else None,
        apy=round(apy, 2) if apy is not None else None,
        predictability=analysis.predictability,
        volume=analysis.volume,
        liquidity=analysis.liquidity,
        num_evidence=len(analysis.evidence_scores),
        total_raw_adjustment=total_raw_adjustment,
        scaled_adjustment=scaled_adjustment,
        evidence_breakdown=evidence_breakdown,
        direction=direction,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# PUBLIC API (for pipeline use)
# =============================================================================

def compute_predictions(
    analyses: list[ConditionAnalysis],
    evidence_scale: float = EVIDENCE_SCALE,
    verbose: bool = False,
) -> list[Prediction]:
    """
    Compute final probability predictions from evidence analyses.
    
    This is the main entry point for the pipeline.
    
    Args:
        analyses: List of ConditionAnalysis to compute predictions for
        evidence_scale: How much evidence moves log-odds (default: 0.5)
        verbose: Print progress
        
    Returns:
        List of Prediction models with final probabilities
    """
    predictions = []
    
    for i, analysis in enumerate(analyses):
        if verbose:
            q_short = analysis.outcome_question[:45] + "..." if len(analysis.outcome_question) > 45 else analysis.outcome_question
            print(f"[{i+1}/{len(analyses)}] {q_short}")
        
        pred = _compute_single_prediction(analysis, evidence_scale)
        predictions.append(pred)
        
        if verbose:
            print(f"    Market: {pred.market_price:.1%} → Predicted: {pred.predicted_price:.1%} ({pred.direction})")
    
    return predictions


def display_prediction(pred: Prediction, verbose: bool = False):
    """Display a prediction result."""
    print(f"\n{'='*80}")
    print(f"📊 {pred.outcome_question}")
    print(f"   Event: {pred.event_title}")
    print(f"{'='*80}")
    
    # Direction indicator
    if pred.direction == 'YES':
        indicator = '🟢 BUY YES'
    elif pred.direction == 'NO':
        indicator = '🔴 BUY NO'
    else:
        indicator = '⚪ HOLD'
    
    print(f"\n   Market Price:      {pred.market_price:.1%}")
    print(f"   Our Estimate:      {pred.predicted_price:.1%}")
    print(f"   Edge:              {pred.edge:+.1%} ({pred.edge_percent:+.1f}%)")
    print(f"   Predictability:    {pred.predictability:.0%}")
    
    if pred.days_until_end is not None:
        print(f"   Days Until End:    {pred.days_until_end:.0f} days")
    if pred.apy is not None:
        apy_str = f"{pred.apy:.0%}" if pred.apy < 10 else f"{pred.apy:.1f}x"
        print(f"   APY:               {apy_str}")
    if pred.volume is not None:
        print(f"   Volume:            ${pred.volume:,.0f}")
    if pred.liquidity is not None:
        print(f"   Liquidity:         ${pred.liquidity:,.0f}")
    
    print(f"\n   Recommendation:    {indicator}")
    
    if verbose:
        print(f"\n   Evidence Breakdown ({pred.num_evidence} items):")
        print(f"   {'─'*70}")
        
        for ev in pred.evidence_breakdown:
            dir_symbol = '+' if ev.direction > 0 else ('-' if ev.direction < 0 else '○')
            print(f"   {dir_symbol} [{ev.query_id}] Quality:{ev.quality:.2f} → {ev.raw_adjustment:+.3f}")
            print(f"     {ev.key_finding}")
        
        print(f"\n   Total raw adjustment: {pred.total_raw_adjustment:+.3f}")
        print(f"   Scaled adjustment:    {pred.scaled_adjustment:+.3f}")


# =============================================================================
# MAIN (CLI entry point)
# =============================================================================

def main():
    global EVIDENCE_SCALE
    
    parser = argparse.ArgumentParser(description='Compute probability predictions from evidence analysis')
    parser.add_argument('--input', default='data/evidence_analysis.json',
                        help='Input evidence analysis file')
    parser.add_argument('--output', default='data/predictions.json',
                        help='Output predictions file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed evidence breakdown')
    parser.add_argument('--scale', type=float, default=EVIDENCE_SCALE,
                        help=f'Evidence scale factor (default: {EVIDENCE_SCALE})')
    parser.add_argument('--min-edge', type=float, default=0.0,
                        help='Only show predictions with abs(edge) >= this value')
    parser.add_argument('--sort', choices=['edge', 'predictability', 'market', 'apy', 'days'],
                        default='edge', help='Sort results by this field')
    parser.add_argument('--no-save', action='store_true',
                        help="Don't save to JSON files")
    
    args = parser.parse_args()
    EVIDENCE_SCALE = args.scale
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load evidence analysis from JSON
    print(f"📖 Loading evidence analysis from {input_path}")
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return None
    
    with open(input_path, 'r') as f:
        evidence_data = json.load(f)
    
    # Convert to ConditionAnalysis models
    analyses = []
    for d in evidence_data:
        evidence_scores = [EvidenceScore(**e) for e in d.get('evidence_scores', [])]
        analyses.append(ConditionAnalysis(
            condition_id=d['condition_id'],
            event_id=d['event_id'],
            event_title=d['event_title'],
            outcome_question=d['outcome_question'],
            market_price=d['market_price'],
            volume=d.get('volume'),  # Fix: was 'condition_volume'
            liquidity=d.get('liquidity'),  # Fix: was 'condition_liquidity'
            end_date=d.get('end_date'),
            predictability=d.get('predictability', 0.5),
            predictability_reason=d.get('predictability_reason', ''),
            evidence_scores=evidence_scores,
            analyzed_at=d.get('analyzed_at', ''),
            model=d.get('model', 'unknown'),
        ))
    
    print(f"   Found {len(analyses)} conditions to process")
    
    # Compute predictions
    predictions = compute_predictions(analyses, EVIDENCE_SCALE, verbose=False)
    
    # Filter by minimum edge
    if args.min_edge > 0:
        predictions = [p for p in predictions if abs(p.edge) >= args.min_edge]
        print(f"   Filtered to {len(predictions)} with edge >= {args.min_edge:.1%}")
    
    # Sort results
    if args.sort == 'edge':
        predictions.sort(key=lambda x: abs(x.edge), reverse=True)
    elif args.sort == 'predictability':
        predictions.sort(key=lambda x: x.predictability, reverse=True)
    elif args.sort == 'market':
        predictions.sort(key=lambda x: x.market_price)
    elif args.sort == 'apy':
        predictions.sort(key=lambda x: x.apy or 0, reverse=True)
    elif args.sort == 'days':
        predictions.sort(key=lambda x: x.days_until_end or 9999)
    
    # Display results
    print(f"\n{'#'*80}")
    print(f"# PREDICTION RESULTS (sorted by {args.sort})")
    print(f"{'#'*80}")
    
    for pred in predictions:
        display_prediction(pred, verbose=args.verbose)
    
    # Summary
    yes_count = sum(1 for p in predictions if p.direction == 'YES')
    no_count = sum(1 for p in predictions if p.direction == 'NO')
    hold_count = sum(1 for p in predictions if p.direction == 'HOLD')
    avg_edge = sum(abs(p.edge) for p in predictions) / len(predictions) if predictions else 0
    
    print(f"\n{'='*80}")
    print(f"📈 SUMMARY")
    print(f"{'='*80}")
    print(f"   Total conditions:  {len(predictions)}")
    print(f"   BUY YES signals:   {yes_count}")
    print(f"   BUY NO signals:    {no_count}")
    print(f"   HOLD signals:      {hold_count}")
    print(f"   Average abs edge:  {avg_edge:.1%}")
    
    if args.no_save:
        print(f"\n✅ Computed {len(predictions)} predictions (not saved)")
        return predictions
    
    # Create pipeline output
    output = PipelineOutput(
        computed_at=datetime.now(timezone.utc).isoformat(),
        parameters={
            'evidence_scale': EVIDENCE_SCALE,
            'min_prob': MIN_PROB,
            'max_prob': MAX_PROB
        },
        summary={
            'total': len(predictions),
            'yes_signals': yes_count,
            'no_signals': no_count,
            'hold_signals': hold_count,
            'avg_abs_edge': round(avg_edge, 4)
        },
        predictions=predictions,
    )
    
    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output.model_dump(), f, indent=2, default=str)
    
    print(f"\n💾 Saved predictions to {output_path}")
    
    return predictions


if __name__ == '__main__':
    main()
