#!/usr/bin/env python3
"""
Market monitor — runs nightly via cron.

Checks all open positions against current Gamma prices.
Flags any position where price has moved significantly against entry.
Writes a JSON summary to ~/.openclaw/bot_monitor.json.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

POLYCLAW_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(POLYCLAW_DIR))

from dotenv import load_dotenv
load_dotenv(POLYCLAW_DIR / ".env")

from lib.gamma_client import GammaClient
from lib.position_storage import PositionStorage

MONITOR_FILE    = Path.home() / ".openclaw" / "bot_monitor.json"
ALERT_THRESHOLD = 0.15   # flag if price moved >15% against entry
DANGER_THRESHOLD = 0.30  # danger flag if moved >30% against entry


async def main():
    storage = PositionStorage()
    gamma   = GammaClient()

    open_positions = storage.get_open()
    if not open_positions:
        print("No open positions to monitor.")
        result = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "open_count": 0,
            "alerts": [],
            "positions": [],
        }
        MONITOR_FILE.write_text(json.dumps(result, indent=2))
        return

    alerts    = []
    summaries = []

    for pos in open_positions:
        try:
            market = await gamma.get_market(pos["market_id"])
            entry_price = float(pos.get("entry_price", 0.5))
            side        = pos.get("position", "YES")

            current_price = market.yes_price if side == "YES" else market.no_price
            change        = current_price - entry_price
            pct_change    = change / entry_price if entry_price > 0 else 0.0

            flag = ""
            if pct_change < -DANGER_THRESHOLD:
                flag = "DANGER"
                alerts.append(
                    f"[DANGER] {side} '{pos['question'][:45]}' "
                    f"{entry_price:.2f}→{current_price:.2f} ({pct_change:+.0%})"
                )
            elif pct_change < -ALERT_THRESHOLD:
                flag = "WARN"
                alerts.append(
                    f"[WARN]   {side} '{pos['question'][:45]}' "
                    f"{entry_price:.2f}→{current_price:.2f} ({pct_change:+.0%})"
                )
            elif pct_change > ALERT_THRESHOLD:
                flag = "PROFIT"

            summaries.append({
                "position_id":   pos["position_id"],
                "question":      pos["question"][:60],
                "side":          side,
                "entry_price":   entry_price,
                "current_price": round(current_price, 3),
                "pct_change":    round(pct_change, 3),
                "flag":          flag,
                "resolved":      market.resolved,
                "outcome":       market.outcome,
                "strategy":      pos.get("strategy", "?"),
            })

        except Exception as e:
            print(f"Error monitoring {pos.get('market_id', '?')}: {e}")

    result = {
        "generated":  datetime.now(timezone.utc).isoformat(),
        "open_count": len(open_positions),
        "alerts":     alerts,
        "positions":  summaries,
    }

    MONITOR_FILE.write_text(json.dumps(result, indent=2))

    print(f"Monitor: {len(open_positions)} positions checked, {len(alerts)} alerts")
    for alert in alerts:
        print(f"  {alert}")


if __name__ == "__main__":
    asyncio.run(main())
