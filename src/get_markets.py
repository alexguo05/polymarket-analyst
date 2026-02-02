#!/usr/bin/env python3
"""
Polymarket Market Fetcher & Scanner

Fetches markets from Polymarket's Gamma API and filters for markets where
fundamental analysis can provide an edge.

Usage:
    python get_markets.py              # Fetch and filter markets
    python get_markets.py --verbose    # Show detailed filter decisions
"""

import argparse
import json
import re
import requests
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# API
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
API_PAGE_LIMIT = 500

# Liquidity bounds (in USD)
MIN_LIQUIDITY = 5_000
MAX_LIQUIDITY = 500_000

# Price bounds (uncertainty zone)
MIN_PRICE = 0.20
MAX_PRICE = 0.80

# Time bounds
MIN_HOURS_UNTIL_END = 24
MAX_DAYS_UNTIL_END = 365

# Filtered output configuration
FILTERED_MIN_OUTCOME_PRICE = 0.20
FILTERED_MAX_OUTCOME_PRICE = 0.80
FILTERED_MIN_DAYS_UNTIL_END = 5
FILTERED_MAX_DAYS_UNTIL_END = 60

# Condition-level volume filter (filter individual outcomes, not just events)
MIN_CONDITION_VOLUME = 10_000  # Minimum volume per condition in USD

# Tag IDs to INCLUDE (semantic markets where research helps)
INCLUDE_TAG_IDS = {
    2,       # Politics
    144,     # Elections
    100265,  # Geopolitics
    596,     # Culture
    120,     # Finance (non-crypto)
    126,     # Trump
    101970,  # World
}

# Tag IDs to EXCLUDE (markets where speed/data beats research)
EXCLUDE_TAG_IDS = {
    1,       # Sports
    21,      # Crypto
    1312,    # Crypto Prices
    100639,  # Games
    101757,  # Recurring (short-term)
    102127,  # Up or Down (crypto)
    102892,  # 5M timeframe
    102467,  # 15M timeframe
    102175,  # 1H timeframe
    102169,  # Hide From New
    100350,  # Soccer
    28,      # Basketball
    100396,  # NCAA
    102114,  # NCAA Basketball
    100088,  # Hockey
    899,     # NHL
    64,      # Esports
    102780,  # CWBB
    235,     # Bitcoin
    39,      # Ethereum
    101267,  # XRP
    818,     # Solana
    101312,  # Ripple
}

# Slug patterns to exclude
EXCLUDE_SLUG_PATTERNS = [
    r"bitcoin", r"ethereum", r"solana", r"crypto",
    r"\bnfl\b", r"\bnba\b", r"\bmlb\b", r"\bnhl\b", r"\bncaa\b",
    r"premier-league", r"champions-league", r"la-liga", r"serie-a", r"bundesliga",
    r"price-of-", r"-price-", r"5m-candle", r"15m-candle", r"1h-candle",
    # "Will X say/mention/utter/keyword" type markets (low-signal, phrasing games)
    # NOTE: These patterns are checked against BOTH slug and title.
    r"\bwhat will\b.*\bsay\b",
    r"\bwill\b.*\bsay\b",
    r"\bsay\b.*\bduring\b",
    r"\bmention\b",
    r"\butter\b",
    r"\buse the word\b",
    r"\buse the phrase\b",
    r"\bsay the word\b",
    r"\bsay the phrase\b",
]

EXCLUDE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_SLUG_PATTERNS]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CandidateMarket:
    """A market identified as suitable for fundamental analysis."""
    event_id: str
    title: str
    slug: str
    description: str
    current_price: float
    price_range: tuple[float, float]
    liquidity: float
    volume: float
    end_date: Optional[str]
    hours_until_end: Optional[float]
    updated_at: Optional[str]
    tags: list[str]
    tag_ids: list[int]
    resolution_source: Optional[str]
    is_multi_outcome: bool
    outcome_count: int
    outcomes: list[dict]
    complexity_score: float
    edge_potential: str
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['price_range'] = list(d['price_range'])
        return d


# =============================================================================
# API FETCHING
# =============================================================================

def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def fetch_all_events() -> list[dict]:
    """Fetch all active events from Polymarket's Gamma API."""
    events = []
    offset = 0
    page = 0
    
    print(f"🌐 Fetching events from Polymarket Gamma API...")
    
    while True:
        params = {
            "limit": API_PAGE_LIMIT,
            "offset": offset,
            "active": "true",
            "closed": "false",
        }
        
        try:
            resp = requests.get(f"{GAMMA_API_BASE}/events", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  ❌ Error fetching page {page}: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"  ❌ Error parsing response: {e}")
            break
        
        if not data:
            break
        
        # Parse events
        for item in data:
            outcomes = []
            for market in item.get("markets", []):
                # Parse token IDs
                token_ids = []
                if market.get("clobTokenIds"):
                    try:
                        token_ids = json.loads(market["clobTokenIds"])
                    except json.JSONDecodeError:
                        pass
                
                # Parse prices
                prices = []
                if market.get("outcomePrices"):
                    try:
                        prices = [float(p) for p in json.loads(market["outcomePrices"])]
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                outcomes.append({
                    "condition_id": market.get("conditionId", ""),
                    "question": market.get("question", ""),
                    "slug": market.get("slug", ""),
                    "yes_token_id": token_ids[0] if len(token_ids) >= 1 else None,
                    "no_token_id": token_ids[1] if len(token_ids) >= 2 else None,
                    "yes_price": prices[0] if len(prices) >= 1 else None,
                    "no_price": prices[1] if len(prices) >= 2 else None,
                    "active": market.get("active", False),
                    "closed": market.get("closed", False),
                    "volume": safe_float(market.get("volume")),
                    "liquidity": safe_float(market.get("liquidity")),
                    "end_date": market.get("endDate"),
                })
            
            events.append({
                "event_id": str(item.get("id", "")),
                "title": item.get("title", ""),
                "slug": item.get("slug", ""),
                "description": item.get("description", ""),
                "enable_neg_risk": item.get("enableNegRisk", False),
                "active": item.get("active", False),
                "closed": item.get("closed", False),
                "tags": item.get("tags", []),
                "volume": safe_float(item.get("volume")),
                "liquidity": safe_float(item.get("liquidity")),
                "liquidity_clob": safe_float(item.get("liquidityClob")),
                "end_date": item.get("endDate"),
                "updated_at": item.get("updatedAt"),
                "resolution_source": item.get("resolutionSource"),
                "outcome_count": len(outcomes),
                "outcomes": outcomes,
            })
        
        print(f"  Page {page + 1}: {len(data)} events (total: {len(events)})")
        
        if len(data) < API_PAGE_LIMIT:
            break
        
        page += 1
        offset += API_PAGE_LIMIT
        time.sleep(0.05)
    
    print(f"✅ Fetched {len(events)} events")
    return events


# =============================================================================
# FILTERING
# =============================================================================

def get_tag_ids(event: dict) -> set[int]:
    tag_ids = set()
    for tag in event.get("tags", []):
        if isinstance(tag, dict) and tag.get("id"):
            try:
                tag_ids.add(int(tag["id"]))
            except (ValueError, TypeError):
                pass
    return tag_ids


def get_tag_labels(event: dict) -> list[str]:
    return [tag["label"] for tag in event.get("tags", []) if isinstance(tag, dict) and tag.get("label")]


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        date_str = date_str.replace("Z", "+00:00")
        if "." in date_str:
            parts = date_str.split(".")
            if len(parts) == 2:
                frac, tz = parts[1].split("+") if "+" in parts[1] else (parts[1].rstrip("Z"), "00:00")
                date_str = f"{parts[0]}.{frac[:6]}+{tz}"
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


def filter_event(event: dict, verbose: bool = False) -> tuple[bool, str]:
    """Apply all filters to an event. Returns (passes, reason)."""
    title = event.get("title", "")[:60]
    
    # Tag filter
    tag_ids = get_tag_ids(event)
    if tag_ids & EXCLUDE_TAG_IDS:
        return False, "excluded_tag"
    if not (tag_ids & INCLUDE_TAG_IDS):
        return False, "no_semantic_tag"
    
    # Slug filter
    slug = event.get("slug", "").lower()
    title_lower = event.get("title", "").lower()
    for pattern in EXCLUDE_PATTERNS_COMPILED:
        if pattern.search(slug) or pattern.search(title_lower):
            return False, f"excluded_pattern:{pattern.pattern}"
    
    # Status filter
    if event.get("closed") or not event.get("active"):
        return False, "closed_or_inactive"
    
    # Time filter
    end_date = parse_date(event.get("end_date"))
    now = datetime.now(timezone.utc)
    if end_date:
        hours_until_end = (end_date - now).total_seconds() / 3600
        if hours_until_end < MIN_HOURS_UNTIL_END:
            return False, f"ends_too_soon:{hours_until_end:.0f}h"
        if hours_until_end / 24 > MAX_DAYS_UNTIL_END:
            return False, f"ends_too_far:{hours_until_end/24:.0f}d"
    
    # Price filter
    outcomes = event.get("outcomes", [])
    yes_prices = [o.get("yes_price") for o in outcomes if o.get("yes_price") is not None and o.get("yes_price") > 0]
    if not yes_prices:
        return False, "no_price_data"
    
    if len(outcomes) > 1:
        in_zone = any(MIN_PRICE <= p <= MAX_PRICE for p in yes_prices)
        if not in_zone:
            return False, "no_outcomes_in_zone"
    else:
        price = yes_prices[0]
        if price < MIN_PRICE or price > MAX_PRICE:
            return False, f"price_out_of_zone:{price:.2f}"
    
    # Liquidity filter
    liquidity = event.get("liquidity_clob") or event.get("liquidity") or 0
    if liquidity < MIN_LIQUIDITY:
        return False, f"liquidity_too_low:{liquidity:.0f}"
    if liquidity > MAX_LIQUIDITY:
        return False, f"liquidity_too_high:{liquidity:.0f}"
    
    return True, "passed"


def calculate_complexity_score(event: dict) -> float:
    score = 0.0
    tag_ids = get_tag_ids(event)
    
    score += len(tag_ids & INCLUDE_TAG_IDS) * 0.2
    if 2 in tag_ids or 144 in tag_ids:
        score += 0.3
    if 100265 in tag_ids:
        score += 0.25
    
    res_source = (event.get("resolution_source") or "").lower()
    if any(x in res_source for x in [".gov", "official", "government"]):
        score += 0.3
    elif any(x in res_source for x in ["reuters", "ap", "associated press"]):
        score += 0.2
    
    if event.get("outcome_count", 1) > 2:
        score += min(0.3, event.get("outcome_count", 1) * 0.05)
    
    if len(event.get("title", "")) > 50:
        score += 0.1
    
    return min(1.0, score)


def categorize_edge_potential(event: dict, complexity_score: float) -> str:
    liquidity = event.get("liquidity_clob") or event.get("liquidity") or 0
    tag_ids = get_tag_ids(event)
    
    if complexity_score >= 0.5 and 10_000 <= liquidity <= 200_000:
        if 2 in tag_ids or 144 in tag_ids or 100265 in tag_ids:
            return "high"
    
    if complexity_score >= 0.3:
        return "medium"
    
    return "low"


def filter_events(events: list[dict], verbose: bool = False) -> list[CandidateMarket]:
    """Filter events and return candidate markets."""
    candidates = []
    stats = {"total": len(events), "passed": 0, "failed": 0}
    
    print(f"\n🔍 Filtering {len(events)} events...")
    
    for event in events:
        passes, reason = filter_event(event, verbose)
        
        if not passes:
            stats["failed"] += 1
            if verbose:
                print(f"  ❌ {event.get('title', '')[:50]}... | {reason}")
            continue
        
        stats["passed"] += 1
        
        # Build candidate
        outcomes = event.get("outcomes", [])
        yes_prices = [o.get("yes_price") for o in outcomes if o.get("yes_price")]
        price = max(yes_prices) if yes_prices else 0
        price_range = (min(yes_prices), max(yes_prices)) if yes_prices else (0, 0)
        
        end_date = event.get("end_date")
        hours_until_end = None
        if end_date:
            end_dt = parse_date(end_date)
            if end_dt:
                hours_until_end = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        
        complexity = calculate_complexity_score(event)
        
        outcomes_simplified = [{
            "question": o.get("question", "")[:100],
            "yes_price": o.get("yes_price"),
            "condition_id": o.get("condition_id"),
            "volume": o.get("volume"),
            "liquidity": o.get("liquidity"),
        } for o in outcomes]
        
        candidate = CandidateMarket(
            event_id=event.get("event_id", ""),
            title=event.get("title", ""),
            slug=event.get("slug", ""),
            description=(event.get("description") or "")[:500],
            current_price=price,
            price_range=price_range,
            liquidity=event.get("liquidity_clob") or event.get("liquidity") or 0,
            volume=event.get("volume") or 0,
            end_date=end_date,
            hours_until_end=hours_until_end,
            updated_at=event.get("updated_at"),
            tags=get_tag_labels(event),
            tag_ids=list(get_tag_ids(event)),
            resolution_source=event.get("resolution_source"),
            is_multi_outcome=event.get("enable_neg_risk", False),
            outcome_count=event.get("outcome_count", 1),
            outcomes=outcomes_simplified,
            complexity_score=complexity,
            edge_potential=categorize_edge_potential(event, complexity),
        )
        candidates.append(candidate)
    
    # Sort by edge potential
    priority = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda c: (priority.get(c.edge_potential, 3), -c.complexity_score))
    
    print(f"✅ {stats['passed']} candidates passed filters ({stats['failed']} filtered out)")
    
    return candidates


def is_placeholder_outcome(outcome: dict) -> bool:
    """Check if an outcome is a placeholder (not a real option)."""
    question = outcome.get('question', '').lower()
    price = outcome.get('yes_price')
    
    # Exactly 50% is usually a placeholder
    if price is not None and abs(price - 0.5) < 0.001:
        return True
    
    # Placeholder name patterns: "Person A", "Person B", "Player X", etc.
    import re
    placeholder_patterns = [
        r'\bperson\s+[a-z]\b',      # Person A, Person B, etc.
        r'\bplayer\s+[a-z]\b',      # Player A, Player B, etc.
        r'\bteam\s+[a-z]\b',        # Team A, Team B, etc.
        r'\bcandidate\s+[a-z]\b',   # Candidate A, Candidate B, etc.
        r'\boption\s+[a-z]\b',      # Option A, Option B, etc.
        r'\bchoice\s+[a-z]\b',      # Choice A, Choice B, etc.
        r'\bother\b.*\bwin',        # "Other" to win
        r'^other$',                 # Just "Other"
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, question):
            return True
    
    return False


def create_filtered_output(candidates: list[CandidateMarket]) -> list[dict]:
    """Create tighter filtered output for query generation."""
    filtered = []
    min_hours = FILTERED_MIN_DAYS_UNTIL_END * 24
    max_hours = FILTERED_MAX_DAYS_UNTIL_END * 24
    placeholder_count = 0
    low_volume_count = 0
    
    for c in candidates:
        if c.hours_until_end is None:
            continue
        if c.hours_until_end < min_hours or c.hours_until_end > max_hours:
            continue
        
        c_dict = c.to_dict()
        
        # Filter outcomes to uncertainty zone only, excluding placeholders and low volume
        filtered_outcomes = []
        for o in c_dict['outcomes']:
            price = o.get('yes_price')
            if price is None:
                continue
            if not (FILTERED_MIN_OUTCOME_PRICE <= price <= FILTERED_MAX_OUTCOME_PRICE):
                continue
            if is_placeholder_outcome(o):
                placeholder_count += 1
                continue
            # Filter by volume (use condition volume if available, else event volume)
            # Note: Gamma API often returns None for condition-level volume
            condition_volume = o.get('volume')
            event_volume = c_dict.get('volume') or 0
            effective_volume = condition_volume if condition_volume is not None else event_volume
            if effective_volume < MIN_CONDITION_VOLUME:
                low_volume_count += 1
                continue
            # Store effective volume for downstream use
            o['volume'] = effective_volume
            filtered_outcomes.append(o)
        
        if filtered_outcomes:
            c_dict['outcomes'] = filtered_outcomes
            c_dict['outcome_count'] = len(filtered_outcomes)
            filtered.append(c_dict)
    
    if placeholder_count > 0:
        print(f"   ⏭️  Filtered out {placeholder_count} placeholder outcomes (50% price or generic names)")
    if low_volume_count > 0:
        print(f"   ⏭️  Filtered out {low_volume_count} low-volume outcomes (<${MIN_CONDITION_VOLUME:,} volume)")
    
    return filtered


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch and filter Polymarket markets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed filter decisions")
    args = parser.parse_args()
    
    print("=" * 70)
    print("POLYMARKET MARKET FETCHER & SCANNER")
    print("=" * 70)
    
    # Fetch all events from API
    events = fetch_all_events()
    
    # Filter events
    candidates = filter_events(events, verbose=args.verbose)
    
    # Print summary
    high = [c for c in candidates if c.edge_potential == "high"]
    medium = [c for c in candidates if c.edge_potential == "medium"]
    
    print(f"\n📊 RESULTS:")
    print(f"  High potential: {len(high)}")
    print(f"  Medium potential: {len(medium)}")
    print(f"  Low potential: {len(candidates) - len(high) - len(medium)}")
    
    if high:
        print(f"\n🔥 TOP HIGH-POTENTIAL MARKETS:")
        for i, c in enumerate(high[:5], 1):
            days = c.hours_until_end / 24 if c.hours_until_end else 0
            print(f"  {i}. {c.title[:60]}...")
            print(f"     ${c.liquidity:,.0f} liquidity | {days:.0f} days | {c.outcome_count} outcomes")
    
    # Save outputs
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save all candidates
    output_path = data_dir / "candidate_markets.json"
    with open(output_path, "w") as f:
        json.dump([c.to_dict() for c in candidates], f, indent=2)
    print(f"\n✅ Saved {len(candidates)} candidates to {output_path}")
    
    # Save filtered candidates
    filtered = create_filtered_output(candidates)
    filtered_path = data_dir / "candidate_markets_filtered.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"✅ Saved {len(filtered)} filtered candidates to {filtered_path}")
    print(f"   (filtered to {FILTERED_MIN_DAYS_UNTIL_END}-{FILTERED_MAX_DAYS_UNTIL_END} days, {FILTERED_MIN_OUTCOME_PRICE:.0%}-{FILTERED_MAX_OUTCOME_PRICE:.0%} prices)")


if __name__ == "__main__":
    main()
