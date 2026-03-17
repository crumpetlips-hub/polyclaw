#!/usr/bin/env python3
"""
Project Tedium Bot - Autonomous Polymarket scanner/trader.

Six-model pipeline:
  1. Bayesian    - probability from order book imbalance + correlated markets
  2. Edge        - EV filter (q - p - c) and pure arb check
  3. Spread      - z-score dislocation across related markets in an event
  4. Stoikov     - passive (GTC) vs aggressive (FOK) execution decision
  5. Kelly       - quarter-Kelly position sizing
  6. Monte Carlo - stress test: skip if P(profitable) < 65%

Usage:
  uv run python scripts/bot.py            # paper mode (default)
  uv run python scripts/bot.py --live     # real trades
  uv run python scripts/bot.py --once     # one scan, then exit
"""

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

POLYCLAW_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(POLYCLAW_DIR))

from dotenv import load_dotenv
load_dotenv(POLYCLAW_DIR / ".env")

from lib.gamma_client import GammaClient, Market
from lib.clob_client import ClobClientWrapper
from lib.wallet_manager import WalletManager
from lib.position_storage import PositionStorage, PositionEntry
from lib.ws_client import PolymarketWSClient
from lib.models import (
    BayesianModel, EdgeModel, SpreadModel,
    StoikovModel, KellyModel, MonteCarloModel,
    BondScanner, LongshotFader, MultiOutcomeScanner,
    TAKER_FEE, SLIPPAGE,
)
from scripts.trade import TradeExecutor
from lib.calibrator import (
    CalibrationEngine, PerformanceStore, CalibrationParams,
    record_resolved_position, category_of, REPORT_FILE,
)


# ── Config ────────────────────────────────────────────────────────────────────

SCAN_INTERVAL      = 60     # seconds between scans
SCAN_LIMIT         = 100    # markets per scan
MIN_LIQUIDITY      = 1000   # skip markets below this USDC liquidity
MIN_EDGE           = 0.02   # 2% minimum edge to trigger
MAX_POSITION_PCT   = 0.05   # max 5% of balance per trade
MAX_DAILY_TRADES   = 10     # daily trade cap
MAX_OPEN_POSITIONS = 5      # don't open more simultaneously
MAX_TRADES_PER_SCAN = 2     # max new trades per scan cycle
MIN_BALANCE        = 5.0    # stop trading if balance drops below this
ALERT_COOLDOWN_H   = 24     # hours before re-alerting same market
LOSS_WINDOW_H      = 12     # rolling window for circuit breaker
MAX_LOSSES         = 3      # losses within window that trip the breaker
CATEGORY_CAP       = 0.30   # max 30% of balance in any single category
TAKE_PROFIT_THRESH = 0.07   # take profit when position gains 7¢ (70% of typical bond return)
MEAN_REVERT_THRESH = 0.05   # sports overreaction threshold (5¢ move in 60s)

# Words that suggest ambiguous resolution criteria — skip these markets
_AMBIGUOUS_TERMS = {
    "substantially", "significantly", "at least", "approximately",
    "around", "roughly", "major", "notable", "considerable",
}

LOG_FILE = Path.home() / ".openclaw" / "logs" / "bot.log"


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tedium-bot")


# ── Bot ───────────────────────────────────────────────────────────────────────

class TediumBot:

    def __init__(self, live: bool = False):
        self.live    = live
        self.running = False

        self.gamma  = GammaClient()
        self.wallet = WalletManager()

        # Calibration — load persisted params first, then build models from them
        self._calib_engine = CalibrationEngine()
        self._perf_store   = PerformanceStore()
        self.calib         = self._calib_engine.load_params()

        self.bayesian     = BayesianModel()
        self.spread       = SpreadModel(z_threshold=2.0)
        self.stoikov      = StoikovModel()
        self.monte_carlo  = MonteCarloModel(scenarios=1000)
        self._init_models()  # builds edge/kelly/bond/longshot/multi from calib params

        # CLOB client cached — auth happens once, not per market
        self._clob: Optional[ClobClientWrapper] = None

        self.daily_trades  = 0
        self.day_start     = datetime.now(timezone.utc).date()
        self.storage       = PositionStorage()
        self._scan_count   = 0  # used to schedule resolution checks

        # Persistent cooldown — survives restarts
        self._cooldown_file = Path.home() / ".openclaw" / "bot_cooldowns.json"
        self.last_alerted: dict[str, datetime] = self._load_cooldowns()

        # Circuit breaker — pause if MAX_LOSSES losses in LOSS_WINDOW_H hours
        self._loss_file = Path.home() / ".openclaw" / "bot_losses.json"
        self._loss_timestamps: list[datetime] = self._load_losses()
        self._breaker_tripped: bool = False
        self._check_breaker()  # restore state from previous run

        # WebSocket real-time order book (started in run())
        self._ws = PolymarketWSClient()

        # Price cache for overreaction/mean-reversion detection
        # market_id → (last_yes_price, timestamp)
        self._price_cache: dict[str, tuple[float, datetime]] = {}

        mode = "LIVE 🔴" if live else "PAPER 📋"
        log.info(
            f"TediumBot starting — {mode} "
            f"(calibrated on {self.calib.total_resolved} resolved trades)"
        )

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_models(self) -> None:
        """Build/rebuild models using current calibration params."""
        c = self.calib
        boost = c.maturity_edge_boost  # raises the bar as bot matures
        self.edge_model   = EdgeModel(min_edge=c.edge_min_edge + boost)
        self.kelly        = KellyModel(kelly_fraction=0.25)
        self.bond_scanner = BondScanner(
            min_price=0.90, max_price=0.97, min_days=2.0, max_days=14,
            min_annualised=c.bond_min_annualised,
        )
        self.ls_fader = LongshotFader(
            max_yes_price=0.10, min_edge=c.longshot_min_edge + boost,
        )
        self.ls_fader.LONGSHOT_FACTOR = c.longshot_factor
        self.multi_scanner = MultiOutcomeScanner(min_edge=c.multi_min_edge + boost)

        if REPORT_FILE.exists():
            log.info(f"[CALIB] Self-report:\n{REPORT_FILE.read_text()}")

    def _apply_calibration(self, new_params: CalibrationParams) -> None:
        """Apply updated calibration params and rebuild models."""
        old = self.calib
        self.calib = new_params
        self._init_models()
        log.info(
            f"[CALIB] Updated — bond_kelly={new_params.bond_kelly_pct:.1%} "
            f"ls_factor={new_params.longshot_factor:.2f} "
            f"edge_min={new_params.edge_min_edge:.2%} "
            f"(was {old.edge_min_edge:.2%}) "
            f"total_resolved={new_params.total_resolved}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_daily(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_start:
            self.daily_trades = 0
            self.day_start = today

    def _balance(self) -> float:
        try:
            return self.wallet.get_balances().usdc_e
        except Exception as e:
            log.error(f"Balance check failed: {e}")
            return 0.0

    def _get_clob(self) -> Optional[ClobClientWrapper]:
        """Return cached CLOB client, initialising once."""
        if self._clob is None and self.wallet.is_unlocked:
            try:
                self._clob = ClobClientWrapper(
                    self.wallet.get_unlocked_key(), self.wallet.address
                )
                # Trigger auth now so it's cached for the whole scan
                _ = self._clob.client
            except Exception as e:
                log.warning(f"CLOB init failed: {e}")
        return self._clob

    def _order_book(self, token_id: str) -> Optional[dict]:
        # Try WebSocket cache first (<50ms latency, no REST cost)
        if token_id and self._ws.connected:
            cached = self._ws.get_order_book(token_id)
            if cached:
                return cached
        # Fall back to REST
        clob = self._get_clob()
        if not clob or not token_id:
            return None
        try:
            return clob.get_order_book(token_id)
        except Exception:
            return None

    def _load_cooldowns(self) -> dict[str, datetime]:
        try:
            if self._cooldown_file.exists():
                raw = json.loads(self._cooldown_file.read_text())
                return {k: datetime.fromisoformat(v) for k, v in raw.items()}
        except Exception:
            pass
        return {}

    def _save_cooldowns(self) -> None:
        try:
            raw = {k: v.isoformat() for k, v in self.last_alerted.items()}
            self._cooldown_file.write_text(json.dumps(raw))
        except Exception as e:
            log.warning(f"Failed to save cooldowns: {e}")

    def _category_kelly_factor(self, question: str) -> float:
        """Return per-category Kelly multiplier from calibration (default 1.0)."""
        cat = category_of(question)
        return getattr(self.calib, f"category_kelly_{cat}", 1.0)

    def _already_alerted(self, key: str) -> bool:
        last = self.last_alerted.get(key)
        if last is None:
            return False
        hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return hours < ALERT_COOLDOWN_H

    def _mark_alerted(self, key: str) -> None:
        self.last_alerted[key] = datetime.now(timezone.utc)
        self._save_cooldowns()

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def _load_losses(self) -> list[datetime]:
        try:
            if self._loss_file.exists():
                raw = json.loads(self._loss_file.read_text())
                return [datetime.fromisoformat(ts) for ts in raw]
        except Exception:
            pass
        return []

    def _save_losses(self) -> None:
        try:
            self._loss_file.write_text(
                json.dumps([ts.isoformat() for ts in self._loss_timestamps])
            )
        except Exception as e:
            log.warning(f"Failed to save losses: {e}")

    def _record_loss(self) -> None:
        """Record a resolved loss and recheck the breaker."""
        self._loss_timestamps.append(datetime.now(timezone.utc))
        self._save_losses()
        self._check_breaker()

    def _check_breaker(self) -> bool:
        """Prune expired losses and update breaker state. Returns True if tripped."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=LOSS_WINDOW_H)
        self._loss_timestamps = [ts for ts in self._loss_timestamps if ts > cutoff]
        self._save_losses()
        was_tripped = self._breaker_tripped
        self._breaker_tripped = len(self._loss_timestamps) >= MAX_LOSSES
        if self._breaker_tripped and not was_tripped:
            log.warning(
                f"[BREAKER] {MAX_LOSSES} losses in {LOSS_WINDOW_H}h — trading PAUSED. "
                f"Will auto-reset when oldest loss expires. Scanning continues."
            )
        elif not self._breaker_tripped and was_tripped:
            log.info("[BREAKER] Reset — trading resumed.")
        return self._breaker_tripped

    # ── Analysis pipeline ─────────────────────────────────────────────────────

    def _prefilter(self, market: Market, correlated_signal: Optional[float]) -> bool:
        """Quick check using Gamma prices only — skip obvious non-starters."""
        if market.closed or market.resolved or not market.active:
            return False
        if market.liquidity < MIN_LIQUIDITY:
            return False
        # Resolution ambiguity filter — oracle manipulation risk
        q_lower = market.question.lower()
        if any(t in q_lower for t in _AMBIGUOUS_TERMS):
            return False
        # Pure arb candidate: YES + NO don't sum to 1
        fees = 0.025
        if 1.0 - (market.yes_price + market.no_price) - fees >= MIN_EDGE:
            return True
        # Correlated signal suggests mispricing
        if correlated_signal is not None:
            if abs(correlated_signal - market.yes_price) >= MIN_EDGE:
                return True
        return False

    def _analyze(
        self,
        market: Market,
        balance: float,
        correlated_signal: Optional[float] = None,
    ) -> Optional[dict]:
        """Run all 6 models on a market. Returns opportunity dict or None."""
        if not self._prefilter(market, correlated_signal):
            return None

        # Subscribe this token to WebSocket for future real-time updates
        if market.yes_token_id:
            asyncio.ensure_future(self._ws.subscribe([market.yes_token_id]))

        # Compute price change from cache for overreaction correction
        now = datetime.now(timezone.utc)
        price_change_24h = 0.0
        if market.id in self._price_cache:
            cached_price, cached_time = self._price_cache[market.id]
            age_h = (now - cached_time).total_seconds() / 3600
            if age_h > 0:
                price_change_24h = market.yes_price - cached_price
        self._price_cache[market.id] = (market.yes_price, now)

        # Model 1: Bayesian — order book fetch only for pre-filtered candidates
        ob = self._order_book(market.yes_token_id)
        model_prob = self.bayesian.estimate(
            market_price=market.yes_price,
            order_book=ob,
            correlated_signal=correlated_signal,
            price_change_24h=price_change_24h,
        )

        # Model 2: Edge
        edge = self.edge_model.calculate(
            market_id=market.id,
            question=market.question,
            p_yes=market.yes_price,
            p_no=market.no_price,
            model_prob_yes=model_prob,
        )
        if edge is None:
            return None

        # Model 4: Stoikov
        days = self.stoikov.days_to_resolution(market.end_date)
        aggressive = self.stoikov.should_hit_aggressively(edge=edge.edge, days_to_resolution=days)

        # Model 5: Kelly — scale by category factor (reduces sizing in losing categories)
        cat_factor = self._category_kelly_factor(market.question)
        kelly = self.kelly.size(
            bankroll=balance,
            win_prob=edge.model_prob,
            market_price=edge.market_price,
            max_position_pct=MAX_POSITION_PCT * cat_factor,
        )
        if kelly.size_usd < 1.0:
            return None

        # Model 6: Monte Carlo
        mc = self.monte_carlo.stress_test(
            win_prob=edge.model_prob,
            market_price=edge.market_price,
            position_size=kelly.size_usd,
        )
        if not mc.passed:
            log.debug(f"MC fail: {market.question[:40]} ({mc.profitable_pct:.0%})")
            return None

        # Agreement filter: require at least 2 independent signals to align
        agreement = 0
        if model_prob > edge.market_price + 0.03:        # Bayesian is confident
            agreement += 1
        if edge.edge > MIN_EDGE * 1.5:                   # Edge is strong, not marginal
            agreement += 1
        if correlated_signal is not None:                 # Spread model confirms
            agreement += 1
        if aggressive:                                    # Stoikov says act urgently
            agreement += 1
        if mc.profitable_pct > 0.75:                     # Monte Carlo strongly passes
            agreement += 1
        if agreement < 2:
            log.debug(f"Low agreement ({agreement}/5): {market.question[:40]}")
            return None

        # Domain score: boost priority for categories where we're winning
        cat_factor = self._category_kelly_factor(market.question)
        domain_score = edge.edge * cat_factor  # higher in categories we're good at

        return {
            "market": market,
            "edge": edge,
            "kelly": kelly,
            "mc": mc,
            "aggressive": aggressive,
            "days": days,
            "domain_score": domain_score,
        }

    def _bond_opp(self, market: Market, balance: float) -> Optional[dict]:
        """Run bond scanner on a market. Returns unified opp dict or None."""
        if market.closed or market.resolved or not market.active:
            return None
        days = self.stoikov.days_to_resolution(market.end_date)
        result = self.bond_scanner.scan(
            market_id=market.id,
            question=market.question,
            yes_price=market.yes_price,
            no_price=market.no_price,
            days=days,
            balance=balance,
            max_position_pct=MAX_POSITION_PCT,
        )
        if result is None:
            return None
        from lib.models import EdgeResult, KellyResult
        edge = EdgeResult(
            market_id=market.id,
            question=market.question,
            position=result.position,
            market_price=result.price,
            model_prob=0.99,
            edge=result.gross_return,
            pure_arb=0.0,
            fees=TAKER_FEE + SLIPPAGE,
        )
        kelly = KellyResult(full_kelly=1.0, quarter_kelly=MAX_POSITION_PCT,
                            size_usd=result.size_usd, max_loss_usd=result.size_usd)
        return {
            "market": market, "edge": edge, "kelly": kelly,
            "mc": None, "aggressive": True, "days": days,
            "strategy": "bond",
            "annualised": result.annualised_return,
        }

    def _longshot_opp(self, market: Market, balance: float) -> Optional[dict]:
        """Run longshot fader on a market. Returns unified opp dict or None."""
        if market.closed or market.resolved or not market.active:
            return None
        days = self.stoikov.days_to_resolution(market.end_date)
        result = self.ls_fader.scan(
            market_id=market.id,
            question=market.question,
            yes_price=market.yes_price,
            no_price=market.no_price,
            balance=balance,
            max_position_pct=MAX_POSITION_PCT,
            days=days,
        )
        if result is None:
            return None
        from lib.models import EdgeResult
        edge = EdgeResult(
            market_id=market.id,
            question=market.question,
            position="NO",
            market_price=result.no_price,
            model_prob=1.0 - result.true_prob_yes,
            edge=result.edge,
            pure_arb=0.0,
            fees=TAKER_FEE + SLIPPAGE,
        )
        kelly = self.kelly.size(
            bankroll=balance,
            win_prob=edge.model_prob,
            market_price=edge.market_price,
            max_position_pct=MAX_POSITION_PCT,
        )
        if kelly.size_usd < 1.0:
            return None
        return {
            "market": market, "edge": edge, "kelly": kelly,
            "mc": None, "aggressive": False, "days": days,
            "strategy": "longshot",
            "yes_price": result.yes_price,
            "domain_score": result.edge * self._category_kelly_factor(market.question),
        }

    def _mean_reversion_opp(self, market: Market, balance: float) -> Optional[dict]:
        """
        Sports overreaction fader (strategy 5).

        If a sports market's price moved >5¢ since last scan, the move is likely
        an overreaction. Fade it: bet against the direction of movement.
        Academic backing: ScienceDirect 2019 (73% win rate documented).
        """
        if market.closed or market.resolved or not market.active:
            return None
        cat = category_of(market.question)
        if cat != "sports":
            return None
        if market.id not in self._price_cache:
            return None

        cached_price, cached_time = self._price_cache[market.id]
        age_s = (datetime.now(timezone.utc) - cached_time).total_seconds()
        if age_s > 180:  # only flag moves within last 3 minutes
            return None

        price_move = market.yes_price - cached_price
        if abs(price_move) < MEAN_REVERT_THRESH:
            return None

        # Fade the move: if YES jumped, buy NO; if YES dropped, buy YES
        if price_move > 0:
            position = "NO"
            entry_price = market.no_price
        else:
            position = "YES"
            entry_price = market.yes_price

        # Edge: the overreaction is expected to revert ~70% of the way
        expected_revert = abs(price_move) * 0.60
        fees = TAKER_FEE + SLIPPAGE
        edge_val = expected_revert - fees
        if edge_val < MIN_EDGE:
            return None

        from lib.models import EdgeResult, KellyResult
        edge = EdgeResult(
            market_id=market.id,
            question=market.question,
            position=position,
            market_price=entry_price,
            model_prob=min(0.75, 0.50 + abs(price_move)),
            edge=edge_val,
            pure_arb=0.0,
            fees=fees,
        )
        kelly = self.kelly.size(
            bankroll=balance,
            win_prob=edge.model_prob,
            market_price=edge.market_price,
            max_position_pct=MAX_POSITION_PCT * 0.6,  # smaller size for mean-reversion
        )
        if kelly.size_usd < 1.0:
            return None
        return {
            "market": market, "edge": edge, "kelly": kelly,
            "mc": None, "aggressive": True,
            "days": self.stoikov.days_to_resolution(market.end_date),
            "strategy": "mean_reversion",
            "price_move": price_move,
            "domain_score": edge_val,
        }

    async def _check_take_profit(self) -> int:
        """
        Check open positions for take-profit opportunities.
        If current price is >= entry + TAKE_PROFIT_THRESH, post a limit sell.
        Returns count of positions closed.
        """
        clob = self._get_clob()
        if not clob:
            return 0

        closed = 0
        for pos in self.storage.get_open():
            try:
                token_id    = pos.get("token_id", "")
                entry_price = float(pos.get("entry_price", 0.5))
                entry_amount = float(pos.get("entry_amount", 0))
                position_id = pos.get("position_id", "")
                side        = pos.get("position", "YES")

                if not token_id:
                    continue

                # Get current price
                market = await self.gamma.get_market(pos["market_id"])
                current = market.yes_price if side == "YES" else market.no_price

                if current - entry_price >= TAKE_PROFIT_THRESH:
                    tokens = entry_amount / entry_price if entry_price > 0 else 0
                    if self.live:
                        order_id, err = clob.sell_gtc(token_id, tokens, current)
                        if order_id:
                            self.storage.update_status(position_id, "closed")
                            self.storage.update_notes(
                                position_id,
                                f"take_profit @ {current:.2f} (entry {entry_price:.2f})"
                            )
                            log.info(
                                f"[TAKE PROFIT] {side} '{pos['question'][:45]}' "
                                f"{entry_price:.2f}→{current:.2f} "
                                f"(+{current - entry_price:.2f}) order={order_id}"
                            )
                            closed += 1
                        else:
                            log.debug(f"Take-profit sell failed: {err}")
                    else:
                        log.info(
                            f"[PAPER TAKE PROFIT] {side} '{pos['question'][:45]}' "
                            f"{entry_price:.2f}→{current:.2f}"
                        )
            except Exception as e:
                log.debug(f"Take-profit check error: {e}")

        return closed

    async def _scan(self, balance: float) -> tuple[list[dict], list]:
        """Fetch markets and run all scanners.

        Returns:
            (single_market_opportunities, multi_outcome_results)
        """
        opportunities: dict[str, dict] = {}  # market_id -> best single-market opp
        multi_opps: list = []                # MultiOutcomeResult list

        # Model 3: Spread + multi-outcome scanner — both use event groups
        correlated: dict[str, float] = {}
        try:
            events = await self.gamma.get_events(limit=50)
            for event in events:
                # Spread dislocation signals
                for a, b, z, cheap in self.spread.scan_event(event.markets):
                    log.info(f"Spread z={z:.2f}: '{event.title[:40]}'")
                    correlated[cheap.id] = cheap.yes_price + abs(z) * 0.01

                # Multi-outcome overround arb
                multi = self.multi_scanner.scan(
                    event_title=event.title,
                    markets=event.markets,
                    balance=balance,
                    max_position_pct=MAX_POSITION_PCT,
                )
                if multi:
                    key = f"multi_{event.id}"
                    if not self._already_alerted(key):
                        multi_opps.append((key, multi))
        except Exception as e:
            log.warning(f"Event scan error: {e}")

        # Individual market analysis + specialist scanners
        try:
            markets = await self.gamma.get_trending_markets(limit=SCAN_LIMIT)
            log.info(f"Scanning {len(markets)} markets (balance: ${balance:.2f})")

            for market in markets:
                # Core 6-model pipeline
                signal = correlated.get(market.id)
                opp = self._analyze(market, balance, correlated_signal=signal)

                # Bond scanner (near-certain, resolves soon)
                bond = self._bond_opp(market, balance)

                # Longshot fader (YES < 10¢, buy NO)
                ls = self._longshot_opp(market, balance)

                # Mean-reversion sports overreaction fader
                mr = self._mean_reversion_opp(market, balance)

                # Keep best opportunity per market (highest edge wins)
                for candidate in (opp, bond, ls, mr):
                    if candidate is None:
                        continue
                    mid = market.id
                    if mid not in opportunities or candidate["edge"].edge > opportunities[mid]["edge"].edge:
                        opportunities[mid] = candidate

        except Exception as e:
            log.error(f"Market scan error: {e}")

        return list(opportunities.values()), multi_opps

    # ── Trade execution ───────────────────────────────────────────────────────

    async def _execute(self, opp: dict) -> bool:
        m, e, k = opp["market"], opp["edge"], opp["kelly"]
        log.info(f"Executing: {e.position} on '{m.question[:50]}' ${k.size_usd:.2f}")
        try:
            executor = TradeExecutor(self.wallet)
            result = await executor.buy_position(
                market_id=m.id,
                position=e.position,
                amount=k.size_usd,
            )
            if result.success:
                log.info(f"Trade executed: {e.position} ${k.size_usd:.2f} TX:{result.split_tx[:20]}")
                self.storage.add(PositionEntry(
                    position_id=str(uuid.uuid4()),
                    market_id=m.id,
                    question=m.question,
                    position=e.position,
                    token_id=result.wanted_token_id,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    entry_amount=k.size_usd,
                    entry_price=e.market_price,
                    split_tx=result.split_tx,
                    clob_order_id=result.clob_order_id,
                    clob_filled=result.clob_filled,
                    strategy=opp.get("strategy", "edge"),
                    predicted_edge=e.edge,
                    predicted_win_prob=e.model_prob,
                ))
                return True
            else:
                log.error(f"Trade failed: {result.error}")
                return False
        except Exception as ex:
            log.error(f"Trade exception: {ex}")
            return False

    async def _execute_multi(self, result) -> int:
        """Execute all legs of a multi-outcome arb. Returns number of legs filled."""
        filled = 0
        for leg in result.legs:
            log.info(f"  [MULTI leg] YES '{leg.question[:50]}' ${leg.size_usd:.2f}")
            try:
                executor = TradeExecutor(self.wallet)
                trade = await executor.buy_position(
                    market_id=leg.market_id,
                    position="YES",
                    amount=leg.size_usd,
                )
                if trade.success:
                    log.info(f"  Leg filled TX:{trade.split_tx[:20]}")
                    filled += 1
                    self.daily_trades += 1
                    self.storage.add(PositionEntry(
                        position_id=str(uuid.uuid4()),
                        market_id=leg.market_id,
                        question=leg.question,
                        position="YES",
                        token_id=trade.wanted_token_id,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        entry_amount=leg.size_usd,
                        entry_price=leg.yes_price,
                        split_tx=trade.split_tx,
                        clob_order_id=trade.clob_order_id,
                        clob_filled=trade.clob_filled,
                        strategy="multi",
                        predicted_edge=result.edge_pct,
                        predicted_win_prob=0.99,
                    ))
                else:
                    log.error(f"  Leg failed: {trade.error}")
            except Exception as ex:
                log.error(f"  Leg exception: {ex}")
        return filled

    def _alert(self, opp: dict) -> None:
        e, k = opp["edge"], opp["kelly"]
        strategy = opp.get("strategy", "edge")
        exec_type = "FOK" if opp["aggressive"] else "GTC"

        if strategy == "bond":
            log.info(
                f"[BOND] {e.position} '{opp['market'].question[:50]}' "
                f"price={e.market_price:.2f} return={opp['annualised']:.0%}pa "
                f"days={opp['days']:.0f} size=${k.size_usd:.2f}"
            )
        elif strategy == "longshot":
            log.info(
                f"[LONGSHOT] NO '{opp['market'].question[:50]}' "
                f"YES={opp['yes_price']:.2f} edge={e.edge:.2%} size=${k.size_usd:.2f}"
            )
        else:
            log.info(
                f"[EDGE] {e.position} '{opp['market'].question[:50]}' "
                f"edge={e.edge:.2%} size=${k.size_usd:.2f} {exec_type}"
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run_once(self) -> None:
        self._reset_daily()
        self._scan_count += 1

        # Refresh breaker state (auto-resets as old losses expire)
        self._check_breaker()

        # Every 10 scans (~10 min): check for resolved positions + recalibrate
        if self._scan_count % 10 == 0:
            newly_resolved = await self._check_resolved_positions()
            if newly_resolved > 0:
                log.info(f"[CALIB] {newly_resolved} new resolved trades — recalibrating...")
                new_params = self._calib_engine.calibrate()
                self._apply_calibration(new_params)

        # Check open positions for take-profit opportunities
        await self._check_take_profit()

        balance = self._balance()
        opportunities, multi_opps = await self._scan(balance)

        total_found = len(opportunities) + len(multi_opps)
        if not total_found:
            log.info("No opportunities this scan.")
            return

        log.info(f"Found {len(opportunities)} single-market + {len(multi_opps)} multi-outcome opportunities")

        if self._breaker_tripped:
            losses_str = ", ".join(
                ts.strftime("%H:%M") for ts in sorted(self._loss_timestamps)
            )
            log.warning(
                f"[BREAKER] Trading paused — {len(self._loss_timestamps)} losses in last {LOSS_WINDOW_H}h "
                f"(at {losses_str} UTC). Scanning only."
            )
            return

        if balance < MIN_BALANCE:
            log.warning(f"[FLOOR] Balance ${balance:.2f} below minimum ${MIN_BALANCE:.2f} — trading paused.")
            return

        storage = PositionStorage()

        # ── Multi-outcome arbs first (highest certainty) ──────────────────────
        scan_trades = 0
        for key, multi in multi_opps:
            if scan_trades >= MAX_TRADES_PER_SCAN:
                log.info(f"Per-scan trade limit reached, deferring multi-arb")
                break
            if self.daily_trades + len(multi.legs) > MAX_DAILY_TRADES:
                log.warning("Daily trade limit would be exceeded by multi-arb, skipping")
                break

            log.info(
                f"[MULTI-ARB] '{multi.event_title[:50]}' "
                f"{len(multi.legs)} legs | sum={multi.total_yes:.3f} | "
                f"edge={multi.edge_pct:.2%} | payout=${multi.guaranteed_payout:.2f}"
            )
            self._mark_alerted(key)

            if self.live:
                filled = await self._execute_multi(multi)
                log.info(f"Multi-arb: {filled}/{len(multi.legs)} legs filled")
                scan_trades += filled
            else:
                log.info(f"[PAPER] Multi-arb would execute {len(multi.legs)} legs, exposure=${multi.total_exposure:.2f}")
                scan_trades += len(multi.legs)

        # ── Single-market opportunities ───────────────────────────────────────
        # Sort by domain_score (edge × category_kelly_factor) — prioritise categories we win in
        opportunities.sort(key=lambda o: o.get("domain_score", o["edge"].edge), reverse=True)

        # Build event fingerprints from existing open positions to avoid
        # trading multiple legs of the same underlying event
        open_fingerprints: set[str] = {
            _event_fingerprint(p["question"])
            for p in storage.get_open()
        }
        scan_trades = 0  # trades placed this scan cycle

        for opp in opportunities:
            market_key = f"{opp['market'].id}_{opp['edge'].position}"

            if self._already_alerted(market_key):
                continue

            if self.daily_trades >= MAX_DAILY_TRADES:
                log.warning("Daily trade limit hit")
                break

            if scan_trades >= MAX_TRADES_PER_SCAN:
                log.info(f"Per-scan trade limit ({MAX_TRADES_PER_SCAN}) reached")
                break

            if len(storage.get_open()) >= MAX_OPEN_POSITIONS:
                log.info("Max open positions reached")
                break

            # Same-event deduplication
            fp = _event_fingerprint(opp["market"].question)
            if fp in open_fingerprints:
                log.debug(f"Skipping duplicate event: '{opp['market'].question[:50]}'")
                continue

            # Category concentration cap — max 30% of balance in any one category
            cat = category_of(opp["market"].question)
            cat_exposure = sum(
                float(p.get("entry_amount", 0))
                for p in storage.get_open()
                if category_of(p.get("question", "")) == cat
            )
            if cat_exposure + opp["kelly"].size_usd > balance * CATEGORY_CAP:
                log.info(
                    f"[CAP] Skipping '{opp['market'].question[:45]}' — "
                    f"{cat} exposure ${cat_exposure:.2f} already at cap"
                )
                continue

            self._alert(opp)
            self._mark_alerted(market_key)
            open_fingerprints.add(fp)

            if self.live:
                success = await self._execute(opp)
                if success:
                    self.daily_trades += 1
                    scan_trades += 1
            else:
                log.info(
                    f"[PAPER] {opp['edge'].position} '{opp['market'].question[:50]}' "
                    f"edge={opp['edge'].edge:.2%} size=${opp['kelly'].size_usd:.2f}"
                )
                scan_trades += 1

    async def run(self) -> None:
        self.running = True

        # Start WebSocket for real-time order book updates
        await self._ws.start()

        # Subscribe existing open positions immediately
        open_token_ids = [
            p["token_id"] for p in self.storage.get_open()
            if p.get("token_id")
        ]
        if open_token_ids:
            await self._ws.subscribe(open_token_ids)
            log.info(f"[WS] Subscribed to {len(open_token_ids)} open position tokens")

        while self.running:
            try:
                await self.run_once()
            except Exception as e:
                log.error(f"Loop error: {e}")
            if self.running:
                log.info(f"Next scan in {SCAN_INTERVAL}s")
                await asyncio.sleep(SCAN_INTERVAL)

    async def _check_resolved_positions(self) -> int:
        """
        Check open positions against Gamma API. For any that have resolved,
        record the outcome and update position status. Returns count resolved.
        """
        open_positions = self.storage.get_open()
        if not open_positions:
            return 0

        newly_resolved = 0
        for pos in open_positions:
            try:
                market = await self.gamma.get_market(pos["market_id"])
                if not market.resolved or not market.outcome:
                    continue

                outcome = record_resolved_position(pos, market)
                if outcome:
                    self._perf_store.append(outcome)
                    self.storage.update_status(pos["position_id"], "resolved")
                    result = "WON" if outcome.actual_win else "LOST"
                    log.info(
                        f"[RESOLVED] {result} '{pos['question'][:50]}' "
                        f"pnl=${outcome.actual_pnl:+.2f} "
                        f"strategy={outcome.strategy}"
                    )
                    newly_resolved += 1
                    if not outcome.actual_win:
                        self._record_loss()
            except Exception as e:
                log.debug(f"Resolution check failed for {pos['market_id']}: {e}")

        return newly_resolved

    def stop(self) -> None:
        log.info("Stopping...")
        self.running = False
        asyncio.ensure_future(self._ws.stop())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _event_fingerprint(question: str) -> str:
    """First 4 words of a question, lowercased — used to detect same-event markets."""
    words = question.lower().split()
    return " ".join(words[:4])


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Project Tedium Bot")
    parser.add_argument("--live", action="store_true", help="Enable real trading")
    parser.add_argument("--once", action="store_true", help="One scan cycle then exit")
    args = parser.parse_args()

    bot = TediumBot(live=args.live)

    def _stop(sig, frame):
        bot.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if args.once:
        await bot.run_once()
    else:
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
