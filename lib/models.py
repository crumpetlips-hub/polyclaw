"""Six quantitative models for Polymarket edge detection.

1. BayesianModel  - probability estimation from order book + correlated markets
2. EdgeModel      - EV filter (q - p - c) and pure arb check
3. SpreadModel    - z-score dislocation between related markets in an event
4. StoikovModel   - passive (GTC) vs aggressive (FOK) execution decision
5. KellyModel     - quarter-Kelly position sizing
6. MonteCarloModel - stress test: pass if P(profitable) >= threshold
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
