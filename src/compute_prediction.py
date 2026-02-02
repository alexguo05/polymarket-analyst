#!/usr/bin/env python3
"""
Compute final probability predictions from evidence analysis.

Uses the market price as anchor and adjusts based on evidence quality and direction.
Works in log-odds space for proper probability handling.
"""

import argparse
import json
import math
from datetime import datetime, timezone, timedelta
from dateutil import parser as date_parser
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# How much each unit of evidence adjustment moves the log-odds
# Higher = more responsive to evidence, Lower = more anchored to market
EVIDENCE_SCALE = 0.5

# Minimum and maximum final probabilities (avoid 0 or 1)
MIN_PROB = 0.01
MAX_PROB = 0.99


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
    """
    Calculate annualized return (APY) for a bet.
    
    For YES bets: profit = (1 - buy_price) if win, loss = buy_price if lose
    For NO bets: profit = buy_price if win (market goes to 0), loss = (1 - buy_price) if lose
    
    Expected profit per dollar = edge (absolute)
    APY = (1 + expected_return_per_period) ^ (365/days) - 1
    """
    if days_until_end is None or days_until_end <= 0:
        return None
    
    # Edge as fraction of bet
    # For a YES bet at price p with true probability (p + edge):
    # Expected return = (p + edge) * (1 - p) - (1 - p - edge) * p = edge
    # Return per dollar invested = edge / p (for YES) or edge / (1-p) (for NO)
    
    if edge > 0:
        # Buying YES at market_price
        return_per_dollar = edge / market_price if market_price > 0 else 0
    else:
        # Buying NO at (1 - market_price)
        no_price = 1 - market_price
        return_per_dollar = abs(edge) / no_price if no_price > 0 else 0
    
    # Annualize
    periods_per_year = 365 / days_until_end
    apy = (1 + return_per_dollar) ** periods_per_year - 1
    
    # Cap at reasonable values
    return min(apy, 100.0)  # Cap at 10,000% APY


# =============================================================================
# EVIDENCE COMPUTATION
# =============================================================================

def compute_evidence_adjustment(evidence: dict) -> dict:
    """
    Compute the adjustment from a single piece of evidence.
    
    Returns dict with:
    - quality: average of reliability, recency, relevance, specificity
    - direction: -1, 0, or +1
    - strength: 0-1
    - raw_adjustment: direction * strength * quality
    """
    reliability = evidence.get('reliability', 0.5)
    recency = evidence.get('recency', 0.5)
    relevance = evidence.get('relevance', 0.5)
    specificity = evidence.get('specificity', 0.5)
    
    quality = (reliability + recency + relevance + specificity) / 4
    direction = evidence.get('direction', 0)
    strength = evidence.get('strength', 0)
    
    # Raw adjustment: direction * strength * quality
    raw_adjustment = direction * strength * quality
    
    return {
        'query_id': evidence.get('query_id', '?'),
        'key_finding': evidence.get('key_finding', '')[:80] + '...' if len(evidence.get('key_finding', '')) > 80 else evidence.get('key_finding', ''),
        'quality': quality,
        'direction': direction,
        'strength': strength,
        'raw_adjustment': raw_adjustment
    }


def compute_prediction(condition: dict) -> dict:
    """
    Compute final probability prediction for a condition.
    
    Algorithm:
    1. Start with market_price as anchor (convert to log-odds)
    2. For each evidence item, compute adjustment = direction * strength * quality
    3. Sum adjustments, scale by predictability and EVIDENCE_SCALE
    4. Apply to log-odds, convert back to probability
    """
    market_price = condition.get('market_price', 0.5)
    predictability = condition.get('predictability', 0.5)
    evidence_scores = condition.get('evidence_scores', [])
    end_date = condition.get('end_date')
    condition_volume = condition.get('condition_volume')
    condition_liquidity = condition.get('condition_liquidity')
    
    # Compute adjustments for each piece of evidence
    evidence_details = []
    total_raw_adjustment = 0.0
    
    for ev in evidence_scores:
        detail = compute_evidence_adjustment(ev)
        evidence_details.append(detail)
        total_raw_adjustment += detail['raw_adjustment']
    
    # Scale adjustment by predictability
    # Low predictability = stay closer to market price
    # High predictability = trust our evidence more
    scaled_adjustment = total_raw_adjustment * predictability * EVIDENCE_SCALE
    
    # Convert market price to log-odds
    market_logodds = prob_to_logodds(market_price)
    
    # Apply adjustment in log-odds space
    final_logodds = market_logodds + scaled_adjustment
    
    # Convert back to probability
    final_prob = logodds_to_prob(final_logodds)
    
    # Compute edge (difference from market)
    edge = final_prob - market_price
    edge_pct = (edge / market_price) * 100 if market_price > 0 else 0
    
    # Calculate time and APY
    days_until_end = calculate_days_until_end(end_date)
    apy = calculate_apy(edge, days_until_end, market_price) if days_until_end else None
    
    return {
        'condition_id': condition.get('condition_id'),
        'event_id': condition.get('event_id'),
        'event_title': condition.get('event_title'),
        'outcome_question': condition.get('outcome_question'),
        'market_price': market_price,
        'end_date': end_date,
        'days_until_end': round(days_until_end, 1) if days_until_end else None,
        'condition_volume': condition_volume,
        'condition_liquidity': condition_liquidity,
        'predictability': predictability,
        'num_evidence': len(evidence_scores),
        'total_raw_adjustment': total_raw_adjustment,
        'scaled_adjustment': scaled_adjustment,
        'final_probability': round(final_prob, 4),
        'edge': round(edge, 4),
        'edge_percent': round(edge_pct, 2),
        'apy': round(apy, 2) if apy is not None else None,
        'direction': 'YES' if edge > 0.02 else ('NO' if edge < -0.02 else 'HOLD'),
        'evidence_breakdown': evidence_details
    }


# =============================================================================
# DISPLAY
# =============================================================================

def display_prediction(pred: dict, verbose: bool = False):
    """Display a prediction result."""
    print(f"\n{'='*80}")
    print(f"📊 {pred['outcome_question']}")
    print(f"   Event: {pred['event_title']}")
    print(f"{'='*80}")
    
    market = pred['market_price']
    final = pred['final_probability']
    edge = pred['edge']
    edge_pct = pred['edge_percent']
    direction = pred['direction']
    days = pred.get('days_until_end')
    apy = pred.get('apy')
    volume = pred.get('condition_volume')
    liquidity = pred.get('condition_liquidity')
    
    # Direction indicator
    if direction == 'YES':
        indicator = '🟢 BUY YES'
    elif direction == 'NO':
        indicator = '🔴 BUY NO'
    else:
        indicator = '⚪ HOLD'
    
    print(f"\n   Market Price:      {market:.1%}")
    print(f"   Our Estimate:      {final:.1%}")
    print(f"   Edge:              {edge:+.1%} ({edge_pct:+.1f}%)")
    print(f"   Predictability:    {pred['predictability']:.0%}")
    
    # Time and APY
    if days is not None:
        print(f"   Days Until End:    {days:.0f} days")
    if apy is not None:
        apy_str = f"{apy:.0%}" if apy < 10 else f"{apy:.1f}x"
        print(f"   APY:               {apy_str}")
    if volume is not None:
        print(f"   Volume:            ${volume:,.0f}")
    if liquidity is not None:
        print(f"   Liquidity:         ${liquidity:,.0f}")
    
    print(f"\n   Recommendation:    {indicator}")
    
    if verbose:
        print(f"\n   Evidence Breakdown ({pred['num_evidence']} items):")
        print(f"   {'─'*70}")
        
        for ev in pred['evidence_breakdown']:
            dir_symbol = '+' if ev['direction'] > 0 else ('-' if ev['direction'] < 0 else '○')
            adj = ev['raw_adjustment']
            print(f"   {dir_symbol} [{ev['query_id']}] Q:{ev['quality']:.2f} S:{ev['strength']:.2f} → {adj:+.3f}")
            print(f"     {ev['key_finding']}")
        
        print(f"\n   Total raw adjustment: {pred['total_raw_adjustment']:+.3f}")
        print(f"   Scaled adjustment:    {pred['scaled_adjustment']:+.3f}")


# =============================================================================
# MAIN
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
    
    args = parser.parse_args()
    
    # Update global scale if specified
    EVIDENCE_SCALE = args.scale
    
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    
    # Load evidence analysis
    print(f"📖 Loading evidence analysis from {input_path}")
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    with open(input_path, 'r') as f:
        evidence_data = json.load(f)
    
    print(f"   Found {len(evidence_data)} conditions to process")
    
    # Compute predictions
    predictions = []
    for condition in evidence_data:
        pred = compute_prediction(condition)
        predictions.append(pred)
    
    # Filter by minimum edge if specified
    if args.min_edge > 0:
        predictions = [p for p in predictions if abs(p['edge']) >= args.min_edge]
        print(f"   Filtered to {len(predictions)} with edge >= {args.min_edge:.1%}")
    
    # Sort results
    if args.sort == 'edge':
        predictions.sort(key=lambda x: abs(x['edge']), reverse=True)
    elif args.sort == 'predictability':
        predictions.sort(key=lambda x: x['predictability'], reverse=True)
    elif args.sort == 'market':
        predictions.sort(key=lambda x: x['market_price'])
    elif args.sort == 'apy':
        predictions.sort(key=lambda x: x.get('apy') or 0, reverse=True)
    elif args.sort == 'days':
        predictions.sort(key=lambda x: x.get('days_until_end') or 9999)
    
    # Display results
    print(f"\n{'#'*80}")
    print(f"# PREDICTION RESULTS (sorted by {args.sort})")
    print(f"{'#'*80}")
    
    for pred in predictions:
        display_prediction(pred, verbose=args.verbose)
    
    # Summary statistics
    yes_count = sum(1 for p in predictions if p['direction'] == 'YES')
    no_count = sum(1 for p in predictions if p['direction'] == 'NO')
    hold_count = sum(1 for p in predictions if p['direction'] == 'HOLD')
    avg_edge = sum(abs(p['edge']) for p in predictions) / len(predictions) if predictions else 0
    
    print(f"\n{'='*80}")
    print(f"📈 SUMMARY")
    print(f"{'='*80}")
    print(f"   Total conditions:  {len(predictions)}")
    print(f"   BUY YES signals:   {yes_count}")
    print(f"   BUY NO signals:    {no_count}")
    print(f"   HOLD signals:      {hold_count}")
    print(f"   Average abs edge:  {avg_edge:.1%}")
    
    # Save results
    output_data = {
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'evidence_scale': EVIDENCE_SCALE,
            'min_prob': MIN_PROB,
            'max_prob': MAX_PROB
        },
        'summary': {
            'total': len(predictions),
            'yes_signals': yes_count,
            'no_signals': no_count,
            'hold_signals': hold_count,
            'avg_abs_edge': round(avg_edge, 4)
        },
        'predictions': predictions
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved predictions to {output_path}")


if __name__ == '__main__':
    main()

