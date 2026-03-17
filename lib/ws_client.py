"""
WebSocket order book client for Polymarket CLOB.

Maintains a real-time local cache of order books by token ID.
Falls back silently to REST polling if WebSocket is unavailable.
"""

import asyncio
import json
import logging
from typing import Optional

log = logging.getLogger("ws-client")

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class OrderBookCache:
    """In-memory cache of latest order books keyed by token_id."""

    def __init__(self):
        self._books: dict[str, dict] = {}

    def update(self, token_id: str, book: dict) -> None:
        self._books[token_id] = book

    def get(self, token_id: str) -> Optional[dict]:
        return self._books.get(token_id)

    def has(self, token_id: str) -> bool:
        return token_id in self._books

    def size(self) -> int:
        return len(self._books)


class PolymarketWSClient:
    """
    Persistent WebSocket connection to the Polymarket CLOB.

    Caches order books for subscribed token IDs. Falls back to REST
    transparently if the WebSocket is not yet connected or disconnected.
    """

    def __init__(self):
        self.cache       = OrderBookCache()
        self._subscribed: set[str] = set()
        self._ws         = None
        self._running    = False
        self._task: Optional[asyncio.Task] = None
        self.connected   = False

    async def start(self) -> None:
        """Start the background WebSocket listener task."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("[WS] Client started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def subscribe(self, token_ids: list[str]) -> None:
        """Subscribe to real-time updates for the given token IDs."""
        new_ids = [t for t in token_ids if t and t not in self._subscribed]
        if not new_ids:
            return
        self._subscribed.update(new_ids)
        if self._ws and self.connected:
            try:
                await self._ws.send(json.dumps({
                    "assets_ids": new_ids,
                    "type": "market",
                }))
                log.debug(f"[WS] Subscribed to {len(new_ids)} tokens")
            except Exception as e:
                log.debug(f"[WS] Subscribe failed: {e}")

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Return cached order book or None if not yet received."""
        return self.cache.get(token_id)

    async def _run(self) -> None:
        """Main WebSocket loop with exponential backoff reconnection."""
        try:
            import websockets
        except ImportError:
            log.warning("[WS] websockets not installed — using REST fallback only")
            return

        backoff = 2
        while self._running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    backoff = 2
                    log.info(f"[WS] Connected — resubscribing to {len(self._subscribed)} tokens")

                    # Re-subscribe on reconnect
                    if self._subscribed:
                        await ws.send(json.dumps({
                            "assets_ids": list(self._subscribed),
                            "type": "market",
                        }))

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msgs = json.loads(raw)
                            if isinstance(msgs, list):
                                for m in msgs:
                                    self._process(m)
                            elif isinstance(msgs, dict):
                                self._process(msgs)
                        except Exception:
                            pass

            except Exception as e:
                self.connected = False
                self._ws = None
                if self._running:
                    log.debug(f"[WS] Disconnected ({e}) — retry in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)

        self.connected = False

    def _process(self, data: dict) -> None:
        """Parse a WebSocket message and update the order book cache."""
        token_id   = data.get("asset_id", "")
        event_type = data.get("event_type", "")

        if not token_id:
            return

        has_book = "asks" in data or "bids" in data
        if event_type in ("book", "price_change") or has_book:
            book = {
                "bids": [
                    {"price": float(b["price"]), "size": float(b["size"])}
                    for b in data.get("bids", [])
                ],
                "asks": [
                    {"price": float(a["price"]), "size": float(a["size"])}
                    for a in data.get("asks", [])
                ],
                "timestamp": data.get("timestamp", ""),
            }
            self.cache.update(token_id, book)
