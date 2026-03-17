"""
Self-calibrating parameter engine for the Tedium Bot.

After each position resolves, the actual outcome is recorded and compared
against the model's prediction. Over time this adjusts:

  - min_edge per strategy (tighten if we're losing, loosen if we're winning)
  - kelly_pct per strategy (reduce exposure when predictions are poor)
  - longshot_factor (recalculate from actual longshot win rates)
  - category_kelly_* (reduce sizing in categories where we consistently lose)
  - maturity_edge_boost (raise the bar as the bot accumulates trade history)

Calibration kicks in once MIN_SAMPLE resolved trades exist per strategy.
Kill switch fires if a strategy's win rate drops below KILL_WIN_RATE after
MIN_KILL_SAMPLE trades — effectively disabling it until performance recovers.

Files written:
  ~/.openclaw/bot_performance.json  — raw resolved trade outcomes
  ~/.openclaw/bot_calibration.json  — current calibrated parameters
  ~/.openclaw/bot_self_report.txt   — human-readable performance summary
"""

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

PERF_FILE    = Path.home() / ".openclaw" / "bot_performance.json"
CALIB_FILE   = Path.home() / ".openclaw" / "bot_calibration.json"
REPORT_FILE  = Path.home() / ".openclaw" / "bot_self_report.txt"

# Minimum resolved trades per strategy before calibration adjusts anything
MIN_SAMPLE = 8

# Kill switch: if a strategy's win rate drops below this after MIN_KILL_SAMPLE
# trades, raise its min_edge to maximum (effectively disabling it)
MIN_KILL_SAMPLE = 10
KILL_WIN_RATE   = 0.35

# EMA learning rate — slow, stable updates
LEARNING_RATE = 0.20

# Parameter bounds
BOUNDS = {
    "min_edge":        (0.01, 0.20),   # upper raised to allow kill switch
    "kelly_pct":       (0.02, 0.15),
    "longshot_factor": (0.20, 0.65),
    "min_annualised":  (0.20, 2.00),
    "category_kelly":  (0.20, 1.50),
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
    closing_price: float = 0.0  # market price at resolution — used for CLV
    clv: float = 0.0            # Closing Line Value: entry_price - closing_price
                                # Positive = we got in before market moved our way (edge)
                                # Negative = we got in late (noise trading)


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

    # Per-category Kelly multipliers (1.0 = no change, <1.0 = reduce, >1.0 = increase)
    category_kelly_crypto:   float = 1.0
    category_kelly_sports:   float = 1.0
    category_kelly_politics: float = 1.0
    category_kelly_other:    float = 1.0

    # Maturity boost: added to all min_edge values as the bot accumulates history
    # 0.00 = new bot, 0.01 = 50+ trades, 0.02 = 100+ trades
    maturity_edge_boost: float = 0.00

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

    Per-strategy algorithm:
      1. Compute actual_win_rate and predicted_win_rate
      2. Ratio R = actual / predicted
         - R < 0.80  → over-confident   → raise min_edge 25%, cut kelly 15%
         - R < 0.90  → slight over      → raise min_edge 10%, cut kelly 8%
         - R > 1.20  → under-confident  → lower min_edge 15%
         - R > 1.10  → slight under     → lower min_edge 8%
         - 0.90-1.10 → well-calibrated  → no change
      3. Kill switch: win_rate < KILL_WIN_RATE after MIN_KILL_SAMPLE trades
         → set min_edge to maximum (effectively disables strategy)
      4. All normal changes via EMA; bounded to BOUNDS

    Per-category:
      Track win rate per category (crypto/sports/politics/other) and adjust
      the category_kelly multiplier — reduces sizing in losing categories.

    Maturity boost:
      Raises the min_edge floor as the bot gains experience, making it
      increasingly selective over time.
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

        # ── Per-strategy calibration ──────────────────────────────────────────
        for strategy in ("bond", "longshot", "edge", "multi"):
            subset = [r for r in records if r.get("strategy") == strategy]
            if len(subset) < MIN_SAMPLE:
                continue

            wins = [r for r in subset if r.get("actual_win")]
            actual_wr   = len(wins) / len(subset)
            predicted_wr = (sum(r.get("predicted_win_prob", 0.5) for r in subset)
                            / len(subset))
            R = actual_wr / predicted_wr if predicted_wr > 0 else 1.0

            # Kill switch: consistently underperforming strategy gets disabled
            if len(subset) >= MIN_KILL_SAMPLE and actual_wr < KILL_WIN_RATE:
                _kill_strategy(params, strategy)
                continue  # skip normal EMA update

            # Determine EMA adjustment magnitude from win rate ratio
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
                yes_subset = [r for r in subset if r.get("entry_price", 1.0) < 0.20]
                if yes_subset:
                    avg_price = sum(r.get("entry_price", 0.08) for r in yes_subset) / len(yes_subset)
                    empirical_factor = (
                        sum(1 for r in yes_subset if r.get("actual_win")) / len(yes_subset) / avg_price
                        if avg_price > 0 else params.longshot_factor
                    )
                    params.longshot_factor = _ema_update(
                        params.longshot_factor, empirical_factor, "longshot_factor"
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

        # ── Per-category Kelly multipliers ────────────────────────────────────
        for cat in ("crypto", "sports", "politics", "other"):
            cat_records = [r for r in records if r.get("category") == cat]
            if len(cat_records) < MIN_SAMPLE:
                continue
            cat_wins = sum(1 for r in cat_records if r.get("actual_win"))
            cat_wr   = cat_wins / len(cat_records)
            pred_wr  = (sum(r.get("predicted_win_prob", 0.5) for r in cat_records)
                        / len(cat_records))
            # R > 1 = winning more than expected → allow more sizing
            # R < 1 = losing more than expected → reduce sizing
            R = cat_wr / pred_wr if pred_wr > 0 else 1.0
            attr = f"category_kelly_{cat}"
            current = getattr(params, attr, 1.0)
            target  = max(0.20, min(1.50, R))
            setattr(params, attr, _ema_update(current, target, "category_kelly"))

        # ── Maturity edge boost ───────────────────────────────────────────────
        # As the bot accumulates history it gets pickier about edge quality
        if params.total_resolved >= 100:
            params.maturity_edge_boost = 0.02
        elif params.total_resolved >= 50:
            params.maturity_edge_boost = 0.01
        else:
            params.maturity_edge_boost = 0.00

        self.save_params(params)
        self._write_self_report(params, records)
        return params

    def _write_self_report(self, params: CalibrationParams, records: list[dict]) -> None:
        """Write a human-readable performance summary the bot reads at startup."""
        # Closing Line Value summary
        clv_records = [r for r in records if r.get("clv", 0) != 0]
        avg_clv = sum(r.get("clv", 0) for r in clv_records) / len(clv_records) if clv_records else 0
        clv_verdict = "GENUINE EDGE ✓" if avg_clv > 0 else ("NO EDGE ✗" if clv_records else "no data yet")

        lines = [
            "=== Bobbot Self-Report ===",
            f"Generated: {params.last_calibrated}",
            f"Total resolved: {params.total_resolved}",
            f"Maturity level: {params.maturity_edge_boost:.0%} edge boost",
            f"Avg CLV: {avg_clv:+.4f} — {clv_verdict}",
            "",
            "--- Strategy Performance ---",
        ]
        for strategy in ("bond", "longshot", "edge", "multi"):
            subset = [r for r in records if r.get("strategy") == strategy]
            if not subset:
                lines.append(f"  {strategy:10}: no data yet")
                continue
            wins = sum(1 for r in subset if r.get("actual_win"))
            wr   = wins / len(subset)
            pnl  = sum(r.get("actual_pnl", 0) for r in subset)
            status = " [KILLED]" if _is_killed(params, strategy) else ""
            lines.append(
                f"  {strategy:10}: {len(subset):3} trades  "
                f"{wr:.0%} win rate  ${pnl:+.2f} PnL{status}"
            )

        lines += ["", "--- Category Performance ---"]
        for cat in ("crypto", "sports", "politics", "other"):
            subset = [r for r in records if r.get("category") == cat]
            if not subset:
                continue
            wins = sum(1 for r in subset if r.get("actual_win"))
            wr   = wins / len(subset)
            pnl  = sum(r.get("actual_pnl", 0) for r in subset)
            factor = getattr(params, f"category_kelly_{cat}", 1.0)
            lines.append(
                f"  {cat:10}: {len(subset):3} trades  "
                f"{wr:.0%} win rate  ${pnl:+.2f} PnL  kelly_factor={factor:.2f}"
            )

        lines += [
            "",
            "--- Active Parameters ---",
            f"  bond_min_annualised : {params.bond_min_annualised:.0%}",
            f"  longshot_factor     : {params.longshot_factor:.2f}",
            f"  edge_min_edge       : {params.edge_min_edge:.2%} "
              f"(+{params.maturity_edge_boost:.2%} maturity = "
              f"{params.edge_min_edge + params.maturity_edge_boost:.2%} effective)",
            f"  multi_min_edge      : {params.multi_min_edge:.2%}",
        ]

        REPORT_FILE.write_text("\n".join(lines) + "\n")


# ── Module-level helpers ───────────────────────────────────────────────────────

def _kill_strategy(params: CalibrationParams, strategy: str) -> None:
    """Set a strategy's min_edge to maximum, effectively disabling it."""
    max_edge = BOUNDS["min_edge"][1]
    max_annualised = BOUNDS["min_annualised"][1]
    if strategy == "bond":
        params.bond_min_annualised = max_annualised
    elif strategy == "longshot":
        params.longshot_min_edge = max_edge
    elif strategy == "edge":
        params.edge_min_edge = max_edge
    elif strategy == "multi":
        params.multi_min_edge = max_edge


def _is_killed(params: CalibrationParams, strategy: str) -> bool:
    """Check if a strategy has been disabled by the kill switch."""
    if strategy == "bond":
        return params.bond_min_annualised >= BOUNDS["min_annualised"][1] * 0.95
    elif strategy == "longshot":
        return params.longshot_min_edge >= BOUNDS["min_edge"][1] * 0.95
    elif strategy == "edge":
        return params.edge_min_edge >= BOUNDS["min_edge"][1] * 0.95
    elif strategy == "multi":
        return params.multi_min_edge >= BOUNDS["min_edge"][1] * 0.95
    return False


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

    CLV (Closing Line Value) measures whether our entry beat the closing price.
    Positive CLV = we got in before the market moved our way = genuine edge.
    Negative CLV = we entered late = noise trading.
    """
    from datetime import datetime, timezone

    outcome_str = (market.outcome or "").lower()
    if not outcome_str:
        return None

    pos_side = position.get("position", "YES")
    if pos_side == "YES":
        actual_win   = outcome_str.startswith("y")
        closing_price = market.yes_price  # final market price before resolution
        entry_price  = float(position.get("entry_price", 0.5))
        # CLV: positive if we entered below the closing price (market moved our way)
        clv = closing_price - entry_price
    else:
        actual_win    = outcome_str.startswith("n")
        closing_price = market.no_price
        entry_price   = float(position.get("entry_price", 0.5))
        clv = closing_price - entry_price

    entry_amount = float(position.get("entry_amount", 0))

    if actual_win:
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
        closing_price=round(closing_price, 4),
        clv=round(clv, 4),
    )
