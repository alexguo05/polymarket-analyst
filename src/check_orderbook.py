#!/usr/bin/env python3
"""
Debug tool to monitor all WebSocket updates for a single market.

Usage:
    python check_orderbook.py <event_id_or_slug>
    python check_orderbook.py <event_id_or_slug> --scroll  # Old scrolling mode
    
Examples:
    python check_orderbook.py 22862
    python check_orderbook.py "super-bowl-champion-2026"
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import websockets

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def load_market(events_file: Path, identifier: str) -> dict | None:
    """Load a market by event_id or slug."""
    with open(events_file) as f:
        events = json.load(f)
    
    for event in events:
        if str(event.get("event_id")) == identifier:
            return event
        if event.get("slug") == identifier:
            return event
        # Partial slug match
        if identifier.lower() in event.get("slug", "").lower():
            return event
    
    return None


def get_tokens_from_market(market: dict) -> list[tuple[str, str, str]]:
    """
    Extract all tokens from a market.
    Returns list of (token_id, token_type, outcome_name)
    """
    tokens = []
    for outcome in market.get("outcomes", []):
        question = outcome.get("question", "")[:40]
        yes_token = outcome.get("yes_token_id")
        no_token = outcome.get("no_token_id")
        
        if yes_token:
            tokens.append((yes_token, "YES", question))
        if no_token:
            tokens.append((no_token, "NO", question))
    
    return tokens


class LiveDisplay:
    """Live-updating terminal display."""
    
    def __init__(self, market: dict, tokens: list[tuple[str, str, str]]):
        self.market = market
        self.token_info = {t[0]: (t[1], t[2]) for t in tokens}
        
        # State for each outcome
        self.outcomes = {}
        for outcome in market.get("outcomes", []):
            condition_id = outcome.get("condition_id", "")
            self.outcomes[condition_id] = {
                "question": outcome.get("question", "")[:50],
                "yes_bid": None,
                "yes_ask": None,
                "no_bid": None,
                "no_ask": None,
                "yes_token": outcome.get("yes_token_id"),
                "no_token": outcome.get("no_token_id"),
            }
        
        # Map tokens to outcomes
        self.token_to_outcome = {}
        for outcome in market.get("outcomes", []):
            cid = outcome.get("condition_id", "")
            self.token_to_outcome[outcome.get("yes_token_id")] = (cid, "yes")
            self.token_to_outcome[outcome.get("no_token_id")] = (cid, "no")
        
        self.message_count = 0
        self.update_count = 0
        self.last_event = ""
        self.last_event_time = ""
        self.last_arb_alert = None  # Track last arb to avoid spam
        self.arb_history = []  # Log of arbs detected
        
    def clear_screen(self):
        """Clear terminal and move cursor to top."""
        print("\033[2J\033[H", end="")
    
    def update_book(self, asset_id: str, bids: list, asks: list):
        """Update orderbook state."""
        if asset_id not in self.token_to_outcome:
            return
        
        cid, side = self.token_to_outcome[asset_id]
        outcome = self.outcomes.get(cid)
        if not outcome:
            return
        
        best_bid = max(bids, key=lambda x: float(x.get("price", 0))) if bids else None
        best_ask = min(asks, key=lambda x: float(x.get("price", 999))) if asks else None
        
        if side == "yes":
            outcome["yes_bid"] = best_bid
            outcome["yes_ask"] = best_ask
        else:
            outcome["no_bid"] = best_bid
            outcome["no_ask"] = best_ask
        
        self.update_count += 1
    
    def render(self):
        """Render the current state."""
        self.clear_screen()
        
        now = datetime.now().strftime("%H:%M:%S")
        
        # Header
        print(f"{'='*70}")
        print(f"📊 {self.market.get('title', '')[:60]}")
        print(f"   Event ID: {self.market.get('event_id')} | Slug: {self.market.get('slug', '')[:30]}")
        print(f"{'='*70}")
        print(f"⏰ {now} | Messages: {self.message_count} | Updates: {self.update_count}")
        print(f"Last: {self.last_event_time} {self.last_event}")
        print(f"{'='*70}")
        print()
        
        # Calculate sums for asks (buy) and bids (mint & sell)
        sum_yes_ask = 0
        sum_no_ask = 0
        sum_yes_bid = 0
        sum_no_bid = 0
        valid_yes_ask = 0
        valid_no_ask = 0
        valid_yes_bid = 0
        valid_no_bid = 0
        
        # Outcomes table
        print(f"{'Outcome':<35} {'YES Bid':>9} {'YES Ask':>9} {'NO Bid':>9} {'NO Ask':>9}")
        print(f"{'-'*35} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
        
        for cid, o in self.outcomes.items():
            q = o["question"][:34]
            
            yes_bid = o["yes_bid"]
            yes_ask = o["yes_ask"]
            no_bid = o["no_bid"]
            no_ask = o["no_ask"]
            
            yes_bid_str = f"${float(yes_bid['price']):.3f}" if yes_bid else "  --"
            yes_ask_str = f"${float(yes_ask['price']):.3f}" if yes_ask else "  --"
            no_bid_str = f"${float(no_bid['price']):.3f}" if no_bid else "  --"
            no_ask_str = f"${float(no_ask['price']):.3f}" if no_ask else "  --"
            
            if yes_ask:
                sum_yes_ask += float(yes_ask['price'])
                valid_yes_ask += 1
            if no_ask:
                sum_no_ask += float(no_ask['price'])
                valid_no_ask += 1
            if yes_bid:
                sum_yes_bid += float(yes_bid['price'])
                valid_yes_bid += 1
            if no_bid:
                sum_no_bid += float(no_bid['price'])
                valid_no_bid += 1
            
            print(f"{q:<35} {yes_bid_str:>9} {yes_ask_str:>9} {no_bid_str:>9} {no_ask_str:>9}")
        
        print(f"{'-'*35} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
        
        # Totals
        n = len(self.outcomes)
        yes_bid_total = f"${sum_yes_bid:.3f}" if valid_yes_bid > 0 else "  --"
        yes_ask_total = f"${sum_yes_ask:.3f}" if valid_yes_ask > 0 else "  --"
        no_bid_total = f"${sum_no_bid:.3f}" if valid_no_bid > 0 else "  --"
        no_ask_total = f"${sum_no_ask:.3f}" if valid_no_ask > 0 else "  --"
        
        print(f"{'TOTAL':<35} {yes_bid_total:>9} {yes_ask_total:>9} {no_bid_total:>9} {no_ask_total:>9}")
        print()
        
        # Check for arbs
        arb_detected = None
        
        # LONG ARB: Buy YES ask + NO ask < $1.00
        if valid_yes_ask == n and valid_no_ask == n:
            ask_sum = sum_yes_ask + sum_no_ask
            if ask_sum < 1.0:
                profit = (1.0 - ask_sum) * 100
                arb_detected = ("LONG", ask_sum, profit)
                print(f"🚨🚨🚨 LONG ARB DETECTED 🚨🚨🚨")
                print(f"    Buy YES asks + NO asks = ${ask_sum:.4f}")
                print(f"    PROFIT: ${1.0 - ask_sum:.4f} ({profit:.2f}%)")
            else:
                print(f"   Asks: YES+NO = ${ask_sum:.4f} (need < $1.00, diff: +${ask_sum - 1:.4f})")
        
        # SHORT ARB (mint & sell): YES bid + NO bid > $1.00
        if valid_yes_bid == n and valid_no_bid == n:
            bid_sum = sum_yes_bid + sum_no_bid
            if bid_sum > 1.0:
                profit = (bid_sum - 1.0) * 100
                arb_detected = ("SHORT", bid_sum, profit)
                print(f"🚨🚨🚨 SHORT ARB DETECTED 🚨🚨🚨")
                print(f"    Mint for $1.00, sell to bids = ${bid_sum:.4f}")
                print(f"    PROFIT: ${bid_sum - 1.0:.4f} ({profit:.2f}%)")
            else:
                print(f"   Bids: YES+NO = ${bid_sum:.4f} (need > $1.00, diff: -${1.0 - bid_sum:.4f})")
        
        # Alert if new arb detected
        if arb_detected and arb_detected != self.last_arb_alert:
            self.last_arb_alert = arb_detected
            self.arb_history.append((now, arb_detected))
            # Terminal bell
            print("\a")  # Beep!
        elif not arb_detected:
            self.last_arb_alert = None
        
        # Show arb history
        if self.arb_history:
            print()
            print(f"--- Arb History ({len(self.arb_history)} found) ---")
            for ts, (arb_type, total, profit) in self.arb_history[-5:]:
                print(f"  [{ts}] {arb_type}: ${total:.4f} → {profit:.2f}%")
        
        print()
        print("Press Ctrl+C to exit")


async def monitor_market_live(market: dict):
    """Connect to WebSocket and monitor with live display."""
    tokens = get_tokens_from_market(market)
    token_ids = [t[0] for t in tokens]
    
    display = LiveDisplay(market, tokens)
    display.render()
    
    async with websockets.connect(WS_URL, ping_interval=30) as ws:
        subscribe_msg = {"assets_ids": token_ids, "type": "market"}
        await ws.send(json.dumps(subscribe_msg))
        
        async for message in ws:
            display.message_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            try:
                data = json.loads(message)
                
                if isinstance(data, list):
                    for book in data:
                        if isinstance(book, dict) and "asset_id" in book:
                            display.update_book(
                                book.get("asset_id", ""),
                                book.get("bids", []),
                                book.get("asks", [])
                            )
                            display.last_event = "📚 Snapshot"
                            display.last_event_time = timestamp
                
                elif isinstance(data, dict):
                    event_type = data.get("event_type", "")
                    
                    if event_type == "book":
                        display.update_book(
                            data.get("asset_id", ""),
                            data.get("bids", []),
                            data.get("asks", [])
                        )
                        display.last_event = "📖 Book update"
                        display.last_event_time = timestamp
                    
                    elif event_type == "last_trade_price":
                        asset_id = data.get("asset_id", "")
                        price = data.get("price", "?")
                        info = display.token_info.get(asset_id, ("?", "?"))
                        display.last_event = f"🔄 Trade {info[0]} @ ${price}"
                        display.last_event_time = timestamp
                
                # Render every update
                display.render()
                
            except Exception as e:
                display.last_event = f"❌ Error: {e}"
                display.last_event_time = timestamp


async def monitor_market_scroll(market: dict):
    """Connect to WebSocket and monitor with scrolling output."""
    tokens = get_tokens_from_market(market)
    token_ids = [t[0] for t in tokens]
    token_info = {t[0]: (t[1], t[2]) for t in tokens}
    
    print(f"\n{'='*70}")
    print(f"Market: {market.get('title')}")
    print(f"Event ID: {market.get('event_id')}")
    print(f"Tokens to monitor: {len(token_ids)}")
    print(f"{'='*70}\n")
    
    async with websockets.connect(WS_URL, ping_interval=30) as ws:
        subscribe_msg = {"assets_ids": token_ids, "type": "market"}
        await ws.send(json.dumps(subscribe_msg))
        print(f"✅ Subscribed\n")
        
        async for message in ws:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            try:
                data = json.loads(message)
                
                if isinstance(data, list):
                    for book in data:
                        if isinstance(book, dict) and "asset_id" in book:
                            asset_id = book.get("asset_id", "")
                            info = token_info.get(asset_id, ("?", "Unknown"))
                            bids = book.get("bids", [])
                            asks = book.get("asks", [])
                            best_ask = min(asks, key=lambda x: float(x.get("price", 999))) if asks else None
                            
                            print(f"[{timestamp}] 📚 {info[0]} {info[1][:30]}...")
                            if best_ask:
                                print(f"    ASK: ${float(best_ask['price']):.4f} x {float(best_ask['size']):.2f}")
                
                elif isinstance(data, dict):
                    event_type = data.get("event_type", "")
                    
                    if event_type == "book":
                        asset_id = data.get("asset_id", "")
                        info = token_info.get(asset_id, ("?", "Unknown"))
                        asks = data.get("asks", [])
                        best_ask = min(asks, key=lambda x: float(x.get("price", 999))) if asks else None
                        
                        print(f"[{timestamp}] 📖 {info[0]} {info[1][:30]}...")
                        if best_ask:
                            print(f"    ASK: ${float(best_ask['price']):.4f}")
                    
                    elif event_type == "last_trade_price":
                        asset_id = data.get("asset_id", "")
                        info = token_info.get(asset_id, ("?", "Unknown"))
                        price = data.get("price", "?")
                        print(f"[{timestamp}] 🔄 TRADE {info[0]} @ ${price}")
                
            except Exception as e:
                print(f"[{timestamp}] ❌ {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor WebSocket updates for a single Polymarket market"
    )
    parser.add_argument(
        "identifier",
        help="Event ID or slug (or partial slug) of the market to monitor"
    )
    parser.add_argument(
        "--scroll", action="store_true",
        help="Use scrolling output instead of live dashboard"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    events_file = project_dir / "data" / "candidate_markets.json"
    
    if not events_file.exists():
        print(f"Events file not found: {events_file}")
        print(f"Run 'python src/get_markets.py' first to generate market data.")
        sys.exit(1)
    
    market = load_market(events_file, args.identifier)
    
    if not market:
        print(f"Market not found: {args.identifier}")
        sys.exit(1)
    
    try:
        if args.scroll:
            asyncio.run(monitor_market_scroll(market))
        else:
            asyncio.run(monitor_market_live(market))
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
