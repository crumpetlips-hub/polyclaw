"""
Self-calibrating parameter engine for the Tedium Bot.

After each position resolves, the actual outcome is recorded and compared
against the model's prediction. Over time this adjusts:

  - min_edge per strategy (tighten if we're losing, loosen if we're winning)
  - kelly_pct per strategy (reduce exposure when predictions are poor)
  - longshot_factor (recalculate from actual longshot win rates)

Calibration only kicks in once MIN_SAMPLE resolved trades exist per strategy.
All adjustments are bounded and use slow EMA updates to prevent oscillation.

Files written:
  ~/.openclaw/bot_performance.json  — raw resolved trade outcomes
  ~/.openclaw/bot_calibration.json  — current calibrated parameters
"""

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

PERF_FILE  = Path.home() / ".openclaw" / "bot_performance.json"
CALIB_FILE = Path.home() / ".openclaw" / "bot_calibration.json"

# Minimum resolved trades per strategy before calibration adjusts anything
MIN_SAMPLE = 15

# EMA learning rate — slow, stable updates
LEARNING_RATE = 0.20

# Parameter bounds
BOUNDS = {
    "min_edge":        (0.01, 0.10),
    "kelly_pct":       (0.02, 0.15),
    "longshot_factor": (0.20, 0.65),
    "min_annualised":  (0.20, 2.00),
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeOutcome:
    """One resolved trade recorded for calibration."""
    position_id: str
    market_id: str
    strategy: str             # bond | longshot | edge | multi
    category: str             # crypto | sports | politics | other
    predicted_edge: float
    predicted_win_prob: float
    entry_price: float
    entry_amount: float
    actual_win: bool
    actual_pnl: float         # USDC profit/loss
    resolved_at: str          # ISO timestamp


@dataclass
class CalibrationParams:
    """Live calibrated parameters. Defaults match bot startup values."""

    # Bond
    bond_min_annualised: float = 0.50
    bond_kelly_pct: float      = 0.10

    # Longshot
    longshot_min_edge: float  = 0.02
    longshot_factor: float    = 0.40
    longshot_kelly_pct: float = 0.10

    # Core edge model
    edge_min_edge: float  = 0.02
    edge_kelly_pct: float = 0.10

    # Multi-outcome
    multi_min_edge: float  = 0.02
    multi_kelly_pct: float = 0.10

    # Metadata
    total_resolved: int = 0
    last_calibrated: str = ""


# ── Category detection ────────────────────────────────────────────────────────

_CRYPTO_TERMS   = {"bitcoin","btc","eth","ethereum","crypto","sol","solana",
                   "doge","polygon","matic","xrp","bnb","usdc","defi","nft",
                   "blockchain","coinbase","binance","price","$"}
_SPORTS_TERMS   = {"nfl","nba","nhl","mlb","football","basketball","soccer",
                   "tennis","golf","match","game","win","beat","league","cup",
                   "championship","playoff","series","lol","dota","esport",
                   "cs2","valorant","pistons","lakers","celtics","nicks",
                   "warriors","heat","bulls","hawks","wizards","gen.g","t1"}
_POLITICS_TERMS = {"election","president","congress","senate","vote","poll",
                   "trump","biden","harris","democrat","republican","party",
                   "minister","chancellor","parliament","governor","mayor"}


def category_of(question: str) -> str:
    q = question.lower()
    words = set(q.replace("?", "").replace(",", "").split())
    if words & _CRYPTO_TERMS or any(t in q for t in ("$", "price", "crypto")):
        return "crypto"
    if words & _SPORTS_TERMS or any(t in q for t in ("vs.", "vs ")):
        return "sports"
    if words & _POLITICS_TERMS:
        return "politics"
    return "other"


# ── Performance store ─────────────────────────────────────────────────────────

class PerformanceStore:
    """Read/write resolved trade outcomes."""

    def __init__(self, path: Path = PERF_FILE):
        self.path = path

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def append(self, outcome: TradeOutcome) -> None:
        records = self.load()
        records.append(asdict(outcome))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(self.path)

    def by_strategy(self, strategy: str) -> list[dict]:
        return [r for r in self.load() if r.get("strategy") == strategy]

    def count(self) -> int:
        return len(self.load())


# ── Calibration engine ────────────────────────────────────────────────────────

class CalibrationEngine:
    """
    Reads performance history and computes updated calibration parameters.

    Algorithm per strategy:
      1. Compute actual_win_rate and predicted_win_rate from resolved trades
      2. Ratio R = actual / predicted
         - R < 0.80  → model is over-confident → raise min_edge 25%, cut kelly 15%
         - R < 0.90  → slight over-confidence  → raise min_edge 10%, cut kelly 8%
         - R > 1.20  → model is under-confident → lower min_edge 15%
         - R > 1.10  → slight under-confidence  → lower min_edge 8%
         - 0.90-1.10 → well-calibrated          → no change
      3. All changes applied as EMA (slow, stable)
      4. Bounded to BOUNDS ranges

    For LONGSHOT specifically:
      Recompute the empirical "factor" from actual win rates vs market prices.
      EMA-update LONGSHOT_FACTOR toward the empirical value.
    """

    def __init__(self, store: Optional[PerformanceStore] = None):
        self.store = store or PerformanceStore()

    def load_params(self) -> CalibrationParams:
        if not CALIB_FILE.exists():
            return CalibrationParams()
        try:
            data = json.loads(CALIB_FILE.read_text())
            return CalibrationParams(**{k: v for k, v in data.items()
                                        if k in CalibrationParams.__dataclass_fields__})
        except Exception:
            return CalibrationParams()

    def save_params(self, params: CalibrationParams) -> None:
        tmp = CALIB_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(params), indent=2))
        tmp.replace(CALIB_FILE)

    def calibrate(self) -> CalibrationParams:
        """Run calibration over all resolved trades. Returns updated params."""
        from datetime import datetime, timezone
        params = self.load_params()
        records = self.store.load()
        params.total_resolved = len(records)
        params.last_calibrated = datetime.now(timezone.utc).isoformat()

        for strategy in ("bond", "longshot", "edge", "multi"):
            subset = [r for r in records if r.get("strategy") == strategy]
            if len(subset) < MIN_SAMPLE:
                continue

            wins = [r for r in subset if r.get("actual_win")]
            actual_wr = len(wins) / len(subset)
            predicted_wr = (sum(r.get("predicted_win_prob", 0.5) for r in subset)
                            / len(subset))

            R = actual_wr / predicted_wr if predicted_wr > 0 else 1.0

            # Determine adjustment magnitude
            if R < 0.80:
                edge_adj, kelly_adj = 1.25, 0.85
            elif R < 0.90:
                edge_adj, kelly_adj = 1.10, 0.92
            elif R > 1.20:
                edge_adj, kelly_adj = 0.85, 1.0
            elif R > 1.10:
                edge_adj, kelly_adj = 0.92, 1.0
            else:
                edge_adj, kelly_adj = 1.0, 1.0

            # EMA update the relevant params
            if strategy == "bond":
                params.bond_min_annualised = _ema_update(
                    params.bond_min_annualised,
                    params.bond_min_annualised * edge_adj,
                    "min_annualised",
                )
                params.bond_kelly_pct = _ema_update(
                    params.bond_kelly_pct,
                    params.bond_kelly_pct * kelly_adj,
                    "kelly_pct",
                )

            elif strategy == "longshot":
                params.longshot_min_edge = _ema_update(
                    params.longshot_min_edge,
                    params.longshot_min_edge * edge_adj,
                    "min_edge",
                )
                params.longshot_kelly_pct = _ema_update(
                    params.longshot_kelly_pct,
                    params.longshot_kelly_pct * kelly_adj,
                    "kelly_pct",
                )
                # Recalibrate LONGSHOT_FACTOR from empirical win rate
                if len(subset) >= MIN_SAMPLE:
                    avg_yes_price = sum(r.get("entry_price", 0.92)
                                        if r.get("position") == "NO"
                                        else r.get("entry_price", 0.08)
                                        for r in subset) / len(subset)
                    yes_subset = [r for r in subset
                                  if r.get("entry_price", 1.0) < 0.20]
                    if yes_subset:
                        empirical_factor = (
                            sum(1 for r in yes_subset if r.get("actual_win"))
                            / len(yes_subset)
                            / (sum(r.get("entry_price", 0.08) for r in yes_subset)
                               / len(yes_subset))
                            if sum(r.get("entry_price", 0.08) for r in yes_subset) > 0
                            else params.longshot_factor
                        )
                        params.longshot_factor = _ema_update(
                            params.longshot_factor,
                            empirical_factor,
                            "longshot_factor",
                        )

            elif strategy == "edge":
                params.edge_min_edge = _ema_update(
                    params.edge_min_edge,
                    params.edge_min_edge * edge_adj,
                    "min_edge",
                )
                params.edge_kelly_pct = _ema_update(
                    params.edge_kelly_pct,
                    params.edge_kelly_pct * kelly_adj,
                    "kelly_pct",
                )

            elif strategy == "multi":
                params.multi_min_edge = _ema_update(
                    params.multi_min_edge,
                    params.multi_min_edge * edge_adj,
                    "min_edge",
                )
                params.multi_kelly_pct = _ema_update(
                    params.multi_kelly_pct,
                    params.multi_kelly_pct * kelly_adj,
                    "kelly_pct",
                )

        self.save_params(params)
        return params


def _ema_update(current: float, target: float, param_type: str) -> float:
    """EMA update with bounds enforcement."""
    lo, hi = BOUNDS.get(param_type, (0.0, 1.0))
    updated = current * (1 - LEARNING_RATE) + target * LEARNING_RATE
    return max(lo, min(hi, updated))


# ── Outcome recording helper ──────────────────────────────────────────────────

def record_resolved_position(position: dict, market) -> Optional[TradeOutcome]:
    """
    Build a TradeOutcome from a resolved position and its market data.
    Returns None if outcome can't be determined.
    """
    from datetime import datetime, timezone

    outcome_str = (market.outcome or "").lower()
    if not outcome_str:
        return None

    pos_side = position.get("position", "YES")
    if pos_side == "YES":
        actual_win = outcome_str.startswith("y")
    else:
        actual_win = outcome_str.startswith("n")

    entry_price  = float(position.get("entry_price", 0.5))
    entry_amount = float(position.get("entry_amount", 0))

    if actual_win:
        # Won: received 1 token worth $1, paid entry_price per token
        tokens = entry_amount / entry_price if entry_price > 0 else 0
        actual_pnl = tokens - entry_amount
    else:
        actual_pnl = -entry_amount

    return TradeOutcome(
        position_id=position.get("position_id", ""),
        market_id=position.get("market_id", ""),
        strategy=position.get("strategy") or "edge",
        category=category_of(position.get("question", "")),
        predicted_edge=float(position.get("predicted_edge") or 0.02),
        predicted_win_prob=float(position.get("predicted_win_prob") or 0.5),
        entry_price=entry_price,
        entry_amount=entry_amount,
        actual_win=actual_win,
        actual_pnl=round(actual_pnl, 4),
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )
