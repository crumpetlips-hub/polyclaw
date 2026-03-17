"""Quantitative models for Polymarket edge detection.

Core pipeline (6 models):
1. BayesianModel  - probability estimation from order book + correlated markets
2. EdgeModel      - EV filter (q - p - c) and pure arb check
3. SpreadModel    - z-score dislocation between related markets in an event
4. StoikovModel   - passive (GTC) vs aggressive (FOK) execution decision
5. KellyModel     - quarter-Kelly position sizing
6. MonteCarloModel - stress test: pass if P(profitable) >= threshold

Specialist scanners:
7. BondScanner    - near-certain contracts (90-97¢) resolving soon = high annualised return
8. LongshotFader  - YES < 10¢ markets are systematically overpriced (longshot bias);
                    buy NO for edge
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


TAKER_FEE = 0.02   # Polymarket FOK taker fee
MAKER_FEE = 0.00   # Polymarket GTC maker fee
SLIPPAGE  = 0.005  # Conservative slippage estimate


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EdgeResult:
    market_id: str
    question: str
    position: str        # "YES" or "NO"
    market_price: float  # Current market price for chosen position
    model_prob: float    # Model's probability estimate
    edge: float          # model_prob - market_price - fees
    pure_arb: float      # 1 - (p_yes + p_no) - fees  (> 0 = free money)
    fees: float


@dataclass
class KellyResult:
    full_kelly: float
    quarter_kelly: float
    size_usd: float
    max_loss_usd: float


@dataclass
class MonteCarloResult:
    scenarios: int
    profitable_pct: float
    expected_value: float
    worst_case: float
    p5_outcome: float
    passed: bool


# ── Model 1: Bayesian ─────────────────────────────────────────────────────────

class BayesianModel:
    """
    P(H|D) = P(D|H) * P(H) / P(D)

    Updates the market price (prior) with:
    - Order book imbalance (bid vs ask depth at top 5 levels)
    - Correlated market signal (from spread model, when a related market reprices)
    """

    def __init__(self, imbalance_weight: float = 0.3, correlation_weight: float = 0.4):
        self.imbalance_weight = imbalance_weight
        self.correlation_weight = correlation_weight

    def estimate(
        self,
        market_price: float,
        order_book: Optional[dict] = None,
        correlated_signal: Optional[float] = None,
    ) -> float:
        """Return posterior probability estimate in [0.01, 0.99]."""
        posterior = market_price

        if order_book:
            imbalance = self._imbalance(order_book)
            if imbalance > 0:
                posterior += self.imbalance_weight * imbalance * (1 - posterior)
            else:
                posterior += self.imbalance_weight * imbalance * posterior

        if correlated_signal is not None:
            posterior = (
                (1 - self.correlation_weight) * posterior
                + self.correlation_weight * correlated_signal
            )

        return max(0.01, min(0.99, posterior))

    def _imbalance(self, order_book: dict) -> float:
        """Order book imbalance in [-1, 1]. Positive = buy pressure."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            bid_size = sum(float(b.get("size", 0)) for b in bids[:5])
            ask_size = sum(float(a.get("size", 0)) for a in asks[:5])
            total = bid_size + ask_size
            return (bid_size - ask_size) / total if total > 0 else 0.0
        except Exception:
            return 0.0


# ── Model 2: Edge ─────────────────────────────────────────────────────────────

class EdgeModel:
    """
    EV_net = q - p - c
    Pure arb edge = 1 - (p_yes + p_no) - c
    """

    def __init__(self, min_edge: float = 0.02):
        self.min_edge = min_edge

    def calculate(
        self,
        market_id: str,
        question: str,
        p_yes: float,
        p_no: float,
        model_prob_yes: float,
        taker: bool = True,
    ) -> Optional[EdgeResult]:
        """Return EdgeResult if edge >= min_edge, else None."""
        fees = (TAKER_FEE if taker else MAKER_FEE) + SLIPPAGE
        pure_arb = 1.0 - (p_yes + p_no) - fees

        ev_yes = model_prob_yes - p_yes - fees
        ev_no = (1.0 - model_prob_yes) - p_no - fees

        best_position = "YES" if ev_yes >= ev_no else "NO"
        best_ev = ev_yes if best_position == "YES" else ev_no
        best_price = p_yes if best_position == "YES" else p_no
        best_prob = model_prob_yes if best_position == "YES" else (1.0 - model_prob_yes)

        if best_ev >= self.min_edge:
            return EdgeResult(
                market_id=market_id,
                question=question,
                position=best_position,
                market_price=best_price,
                model_prob=best_prob,
                edge=best_ev,
                pure_arb=pure_arb,
                fees=fees,
            )

        # Pure arb even when model has no conviction
        if pure_arb >= self.min_edge:
            cheap_pos = "YES" if p_yes <= p_no else "NO"
            cheap_price = p_yes if cheap_pos == "YES" else p_no
            return EdgeResult(
                market_id=market_id,
                question=question,
                position=cheap_pos,
                market_price=cheap_price,
                model_prob=0.5,
                edge=pure_arb,
                pure_arb=pure_arb,
                fees=fees,
            )

        return None


# ── Model 3: Spread ───────────────────────────────────────────────────────────

class SpreadModel:
    """
    z = (S - mu_S) / sigma_S

    Tracks rolling spread history between market pairs in the same event.
    Returns z-score when |z| exceeds threshold (dislocation).
    """

    def __init__(self, z_threshold: float = 2.0, history_size: int = 20):
        self.z_threshold = z_threshold
        self.history_size = history_size
        self._history: dict[str, list[float]] = {}

    def update(self, key: str, spread: float) -> Optional[float]:
        """Update history for key. Returns z-score if dislocated, else None."""
        history = self._history.setdefault(key, [])
        history.append(spread)
        if len(history) > self.history_size:
            history.pop(0)

        if len(history) < 5:
            return None

        mu = sum(history) / len(history)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in history) / len(history))

        if sigma < 0.001:
            return None

        z = (spread - mu) / sigma
        return z if abs(z) > self.z_threshold else None

    def scan_event(self, markets: list) -> list[tuple]:
        """
        Scan all pairs in an event for dislocations.
        Returns list of (market_a, market_b, z_score, cheap_market).
        """
        dislocations = []
        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                a, b = markets[i], markets[j]
                z = self.update(f"{a.id}_vs_{b.id}", a.yes_price - b.yes_price)
                if z is not None:
                    cheap = b if z > 0 else a
                    dislocations.append((a, b, z, cheap))
        return dislocations


# ── Model 4: Stoikov ──────────────────────────────────────────────────────────

class StoikovModel:
    """
    r = s - q * gamma * sigma^2 * (T - t)

    Decides passive (GTC limit) vs aggressive (FOK market) execution.
    """

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    def should_hit_aggressively(
        self,
        edge: float,
        days_to_resolution: float,
        bid_ask_spread: float = 0.02,
    ) -> bool:
        """True = use FOK (aggressive). False = use GTC (passive)."""
        if edge > 0.05:
            return True
        if days_to_resolution < 3:
            return True
        return False

    def days_to_resolution(self, end_date: str) -> float:
        """Days until market resolution from ISO date string."""
        try:
            from datetime import datetime, timezone
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            delta = end - datetime.now(timezone.utc)
            return max(0.0, delta.total_seconds() / 86400)
        except Exception:
            return 30.0


# ── Model 5: Kelly ────────────────────────────────────────────────────────────

class KellyModel:
    """
    f* = (b * p - q) / b
    b = (1 - price) / price  (net odds for prediction market)

    Always uses quarter-Kelly. Hard-capped at max_position_pct.
    """

    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction

    def size(
        self,
        bankroll: float,
        win_prob: float,
        market_price: float,
        max_position_pct: float = 0.10,
    ) -> KellyResult:
        if not (0 < market_price < 1):
            return KellyResult(0.0, 0.0, 0.0, 0.0)

        b = (1 - market_price) / market_price
        full_kelly = max(0.0, (b * win_prob - (1 - win_prob)) / b)
        quarter_kelly = full_kelly * self.kelly_fraction
        capped = min(quarter_kelly, max_position_pct)

        size_usd = round(bankroll * capped, 2)
        return KellyResult(
            full_kelly=full_kelly,
            quarter_kelly=quarter_kelly,
            size_usd=size_usd,
            max_loss_usd=size_usd,
        )


# ── Model 6: Monte Carlo ──────────────────────────────────────────────────────

class MonteCarloModel:
    """
    W(t+1) = W(t) * (1 + r(t))

    Simulates random fill rates, slippage, and execution delays.
    Trade passes if P(profitable) >= pass_threshold.
    """

    def __init__(self, scenarios: int = 1000, pass_threshold: float = 0.65):
        self.scenarios = scenarios
        self.pass_threshold = pass_threshold

    def stress_test(
        self,
        win_prob: float,
        market_price: float,
        position_size: float,
        base_fee: float = 0.025,
    ) -> MonteCarloResult:
        outcomes = []

        for _ in range(self.scenarios):
            fill_rate   = random.uniform(0.5, 1.0)
            slippage    = random.uniform(0.0, 0.02)
            edge_decay  = random.uniform(0.0, 0.01)
            actual_cost = market_price + base_fee + slippage + edge_decay
            actual_cost = min(actual_cost, 0.99)

            if random.random() < win_prob:
                payout = (1.0 - actual_cost) * fill_rate * position_size
            else:
                payout = -actual_cost * fill_rate * position_size

            outcomes.append(payout)

        outcomes.sort()
        n = len(outcomes)
        profitable_pct = sum(1 for o in outcomes if o > 0) / n

        return MonteCarloResult(
            scenarios=n,
            profitable_pct=profitable_pct,
            expected_value=sum(outcomes) / n,
            worst_case=outcomes[0],
            p5_outcome=outcomes[int(n * 0.05)],
            passed=profitable_pct >= self.pass_threshold,
        )


# ── Scanner 7: Bond ───────────────────────────────────────────────────────────

@dataclass
class BondResult:
    market_id: str
    question: str
    position: str        # "YES" or "NO" — whichever is near-certain
    price: float         # e.g. 0.95
    days: float          # days until resolution
    gross_return: float  # e.g. 0.025 (2.5¢ net on 97.5¢ cost)
    annualised_return: float  # e.g. 3.12 (312% annualised)
    size_usd: float


class BondScanner:
    """
    Finds near-certain contracts resolving soon.

    A contract priced at 95¢ resolving in 3 days returns 5.2% in 72 hours —
    312% annualised. No model uncertainty required: the edge is purely
    time-value on a high-confidence outcome.

    Criteria:
    - Price in [min_price, max_price]  (default 90–97¢)
    - Resolves within max_days         (default 14 days)
    - Annualised return >= min_annualised (default 50%)
    """

    def __init__(
        self,
        min_price: float = 0.90,
        max_price: float = 0.97,
        max_days: float = 14.0,
        min_annualised: float = 0.50,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.max_days = max_days
        self.min_annualised = min_annualised

    def scan(
        self,
        market_id: str,
        question: str,
        yes_price: float,
        no_price: float,
        days: float,
        balance: float,
        max_position_pct: float = 0.10,
    ) -> Optional[BondResult]:
        """Return BondResult if a near-certain opportunity exists, else None."""
        if days <= 0 or days > self.max_days:
            return None

        for position, price in (("YES", yes_price), ("NO", no_price)):
            if not (self.min_price <= price <= self.max_price):
                continue

            gross_return = 1.0 - price - (TAKER_FEE + SLIPPAGE)
            if gross_return <= 0:
                continue

            annualised = (gross_return / price) * (365.0 / days)
            if annualised < self.min_annualised:
                continue

            size_usd = round(balance * max_position_pct, 2)
            return BondResult(
                market_id=market_id,
                question=question,
                position=position,
                price=price,
                days=days,
                gross_return=gross_return,
                annualised_return=annualised,
                size_usd=size_usd,
            )

        return None


# ── Scanner 8: Longshot Fader ─────────────────────────────────────────────────

@dataclass
class LongshotResult:
    market_id: str
    question: str
    yes_price: float      # the overpriced longshot
    no_price: float       # what we buy
    true_prob_yes: float  # model's estimate (discounted)
    edge: float           # true_prob_no - no_price - fees
    size_usd: float


class LongshotFader:
    """
    Exploits the longshot bias on Polymarket.

    Empirical research on 95M+ transactions shows contracts priced below 10¢
    lose ~60% of invested capital on average. The market systematically
    overprices low-probability events (consistent with prospect theory).

    Adjustment: true_prob_yes = market_price * LONGSHOT_FACTOR (0.40)
    — i.e. the true win rate is ~40% of the implied probability.

    This creates edge on the NO side:
      edge = (1 - true_prob_yes) - no_price - fees

    Example: YES at 8¢ → true_prob_yes = 3.2% → NO edge ≈ 2.3%
    """

    LONGSHOT_FACTOR = 0.40  # true win rate as fraction of market-implied rate

    def __init__(self, max_yes_price: float = 0.10, min_edge: float = 0.02):
        self.max_yes_price = max_yes_price
        self.min_edge = min_edge

    def scan(
        self,
        market_id: str,
        question: str,
        yes_price: float,
        no_price: float,
        balance: float,
        max_position_pct: float = 0.10,
    ) -> Optional[LongshotResult]:
        """Return LongshotResult if NO has edge from longshot bias, else None."""
        if yes_price > self.max_yes_price or yes_price <= 0:
            return None
        if no_price <= 0:
            return None

        true_prob_yes = yes_price * self.LONGSHOT_FACTOR
        true_prob_no  = 1.0 - true_prob_yes
        fees = TAKER_FEE + SLIPPAGE
        edge = true_prob_no - no_price - fees

        if edge < self.min_edge:
            return None

        size_usd = round(balance * max_position_pct, 2)
        return LongshotResult(
            market_id=market_id,
            question=question,
            yes_price=yes_price,
            no_price=no_price,
            true_prob_yes=true_prob_yes,
            edge=edge,
            size_usd=size_usd,
        )


# ── Scanner 9: Multi-Outcome Overround ────────────────────────────────────────

@dataclass
class MultiOutcomeLeg:
    market_id: str
    question: str
    yes_token_id: str
    condition_id: str
    yes_price: float
    size_usd: float   # proportional share of total capital


@dataclass
class MultiOutcomeResult:
    event_title: str
    legs: list           # list[MultiOutcomeLeg]
    total_yes: float     # sum of YES prices (< 1.00 = underround)
    total_exposure: float
    guaranteed_payout: float  # total_exposure / total_yes
    edge_pct: float      # net profit / total_exposure after fees


class MultiOutcomeScanner:
    """
    Finds events where YES prices across all outcomes sum to less than $1.00.

    Since exactly one outcome must resolve YES, buying YES in every market
    in the correct proportions guarantees a fixed payout regardless of result.

    Proportional sizing: invest (price_i / sum) * capital in market i.
    This gives equal token count across all legs, so payout is identical
    no matter which outcome wins.

    Example:
      3-way "who wins" market: [0.40, 0.35, 0.20] → sum = 0.95
      Invest $9.50 total ($4.00, $3.50, $2.00 per leg)
      Guaranteed payout = $10.00 → gross profit 5.3% before fees

    Safety bounds:
      min_sum = 0.85  — below this, likely missing outcomes (not exhaustive)
      max_sum = 0.985 — above this, fees consume the edge
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        min_sum: float = 0.85,
        max_sum: float = 0.985,
        min_leg_liquidity: float = 500.0,
    ):
        self.min_edge = min_edge
        self.min_sum = min_sum
        self.max_sum = max_sum
        self.min_leg_liquidity = min_leg_liquidity

    def scan(
        self,
        event_title: str,
        markets: list,
        balance: float,
        max_position_pct: float = 0.10,
    ) -> Optional[MultiOutcomeResult]:
        """Return MultiOutcomeResult if a guaranteed-profit arb exists, else None."""
        active = [
            m for m in markets
            if m.active and not m.closed and not m.resolved
            and m.yes_token_id
            and m.liquidity >= self.min_leg_liquidity
        ]
        if len(active) < 2:
            return None

        yes_prices = [m.yes_price for m in active]
        total_yes = sum(yes_prices)

        if not (self.min_sum <= total_yes <= self.max_sum):
            return None

        n = len(active)
        total_capital = balance * max_position_pct

        # Guaranteed payout: total_capital / total_yes
        payout = total_capital / total_yes

        # Fees applied on each leg's payout share
        total_fees = n * (TAKER_FEE + SLIPPAGE) * (payout / n)
        net_profit = payout - total_capital - total_fees
        edge_pct = net_profit / total_capital

        if edge_pct < self.min_edge:
            return None

        # Proportional per-leg sizes
        legs = []
        for m, price in zip(active, yes_prices):
            size = round(total_capital * price / total_yes, 2)
            if size < 1.0:
                return None  # leg too small to execute
            legs.append(MultiOutcomeLeg(
                market_id=m.id,
                question=m.question,
                yes_token_id=m.yes_token_id,
                condition_id=m.condition_id,
                yes_price=price,
                size_usd=size,
            ))

        return MultiOutcomeResult(
            event_title=event_title,
            legs=legs,
            total_yes=total_yes,
            total_exposure=total_capital,
            guaranteed_payout=round(payout, 4),
            edge_pct=edge_pct,
        )
