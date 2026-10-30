"""Share one endpoint request budget between concurrent processes.

The campaign can run several endpoint commands at the same time. Each command
is its own process, so an in-process semaphore does not bound the total load.
This module stores the budget in shared files: a pool of slot files for the
in-flight requests and one pace file for the request starts.

The kernel drops a file lock when its process dies. A crashed command therefore
releases its slots without a cleanup step.
"""

import asyncio
import fcntl
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic

SLOT_POLL_SECONDS = 0.01


class RequestBudget:
    """Limit in-flight requests and request starts across processes.

    Args:
        directory: Shared folder for the slot and pace files.
        slots: Maximum number of requests in flight across all processes.
        requests_per_second: Maximum number of request starts across all
            processes.

    Raises:
        ValueError: A bound is not positive.
    """

    def __init__(self, directory: Path, slots: int, requests_per_second: float) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.directory = directory
        self.slots = slots
        self._interval = 1.0 / requests_per_second

    def _slot_path(self, index: int) -> Path:
        """Return the path of one slot file."""
        return self.directory / f"slot-{index}.lock"

    def _pace_path(self) -> Path:
        """Return the path of the shared start-time file."""
        return self.directory / "pace"

    def _try_acquire(self) -> int | None:
        """Lock one free slot and return its descriptor, or None when all are busy."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for index in range(self.slots):
            descriptor = os.open(self._slot_path(index), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            return descriptor
        return None

    async def acquire(self) -> int:
        """Wait for a free slot and return its locked descriptor."""
        while True:
            descriptor = self._try_acquire()
            if descriptor is not None:
                return descriptor
            await asyncio.sleep(SLOT_POLL_SECONDS)

    def release(self, descriptor: int) -> None:
        """Free a slot. Closing the descriptor drops its lock."""
        os.close(descriptor)

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Hold one request slot for the duration of the block."""
        descriptor = await self.acquire()
        try:
            yield
        finally:
            self.release(descriptor)

    async def wait_turn(self) -> None:
        """Wait until this process may start one request.

        The pace file holds the next allowed start time. Monotonic time is
        system wide, so every process compares the same values.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._pace_path(), os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                text = handle.read().strip()
                try:
                    next_start = float(text) if text else 0.0
                except ValueError:
                    next_start = 0.0
                now = monotonic()
                start = max(now, next_start)
                handle.seek(0)
                handle.truncate()
                handle.write(f"{start + self._interval:.6f}\n")
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)
