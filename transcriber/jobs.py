"""FIFO job queue: how many transcriptions one worker runs at once, and who waits in line."""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import HTTPException


class JobQueue:
    """FIFO admission control for one worker process.

    Up to `capacity` jobs run at once; up to `max_waiting` more wait in arrival order. When a job
    finishes, its slot is handed straight to the oldest waiter, so a newcomer can never jump the
    line. A full line, or a wait longer than `wait_timeout`, gets 503 + Retry-After. A waiter whose
    client disconnects leaves the line without taking a slot.
    Only the event loop thread touches this object, so no lock is needed.
    """

    def __init__(self, capacity: int, max_waiting: int, wait_timeout: float):
        self.capacity, self.max_waiting, self.wait_timeout = capacity, max_waiting, wait_timeout
        self.running = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    def stats(self) -> dict[str, int]:
        return {"running": self.running, "waiting": len(self._waiters),
                "capacity": self.capacity, "max_waiting": self.max_waiting}

    @asynccontextmanager
    async def slot(self, is_disconnected: Callable[[], Awaitable[bool]] | None = None) -> AsyncIterator[float]:
        """`async with queue.slot() as waited:` - waited = seconds spent in line."""
        waited = await self._acquire(is_disconnected)
        try:
            yield waited
        finally:
            self._release()

    async def _acquire(self, is_disconnected: Callable[[], Awaitable[bool]] | None) -> float:
        if self.running < self.capacity and not self._waiters:
            self.running += 1
            return 0.0
        if len(self._waiters) >= self.max_waiting:
            raise HTTPException(503, f"Server busy: {self.running} running, {len(self._waiters)} waiting; "
                                     "retry later", headers={"Retry-After": "30"})
        loop = asyncio.get_running_loop()
        started = loop.time()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append(fut)
        try:
            while True:
                remaining = started + self.wait_timeout - loop.time()
                if remaining <= 0:
                    raise HTTPException(503, f"Waited {self.wait_timeout:.0f}s in line; retry later",
                                        headers={"Retry-After": "30"})
                # asyncio.wait, not wait_for: wait_for (3.11) swallows a cancel that arrives just as the
                # slot is handed over, and the request would then run anyway. wait never cancels fut.
                done, _ = await asyncio.wait({fut}, timeout=min(1.0, remaining))
                if done:
                    return round(loop.time() - started, 3)      # slot handed over by _release()
                if is_disconnected is not None and await is_disconnected():
                    # 499 = client closed request (nginx convention); nobody receives it, it is for the log
                    raise HTTPException(499, "Client disconnected while waiting in line")
        except BaseException:
            if fut.done() and not fut.cancelled():
                self._release()          # the slot reached us just as we gave up: pass it on
            else:
                fut.cancel()
                self._waiters.remove(fut)
            raise

    def _release(self) -> None:
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)     # slot goes straight to the oldest waiter; running stays the same
                return
        self.running -= 1
