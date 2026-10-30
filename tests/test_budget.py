"""Test the request budget shared by concurrent processes."""

import asyncio
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest

from experiments.runner.budget import RequestBudget

HOLD_SLOT = """
import asyncio
import sys
from pathlib import Path

from experiments.runner.budget import RequestBudget

async def main():
    folder = Path(sys.argv[1])
    budget = RequestBudget(folder, 1, 1000.0)
    descriptor = await budget.acquire()
    (folder / "held").write_text("1")
    await asyncio.sleep(30)
    budget.release(descriptor)

asyncio.run(main())
"""


def test_a_second_process_cannot_take_a_held_slot(tmp_path: Path) -> None:
    """Block this process while another process holds the only slot."""

    async def scenario() -> None:
        """Try one acquisition while the other process holds the slot."""
        budget = RequestBudget(tmp_path, 1, 1000.0)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(budget.acquire(), timeout=0.2)

    child = subprocess.Popen(
        [sys.executable, "-c", HOLD_SLOT, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        deadline = monotonic() + 10.0
        while not (tmp_path / "held").exists():
            assert monotonic() < deadline, "the child process did not take the slot"
            assert child.poll() is None, "the child process exited early"
            sleep(0.01)
        asyncio.run(scenario())
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_a_released_slot_can_be_taken_again(tmp_path: Path) -> None:
    """Block a second budget on the only slot until the first releases it."""

    async def scenario() -> None:
        """Acquire, verify the block, then release."""
        first = RequestBudget(tmp_path, 1, 1000.0)
        second = RequestBudget(tmp_path, 1, 1000.0)
        descriptor = await first.acquire()
        blocked = asyncio.create_task(second.acquire())
        await asyncio.sleep(0.05)
        assert not blocked.done()
        first.release(descriptor)
        free = await asyncio.wait_for(blocked, timeout=2.0)
        second.release(free)

    asyncio.run(scenario())


def test_the_pool_holds_no_more_than_its_slots(tmp_path: Path) -> None:
    """Start a third request only after one of two slots is free."""

    async def scenario() -> None:
        """Fill the pool and release it."""
        budget = RequestBudget(tmp_path, 2, 1000.0)
        first = await budget.acquire()
        second = await budget.acquire()
        blocked = asyncio.create_task(budget.acquire())
        await asyncio.sleep(0.05)
        assert not blocked.done()
        budget.release(first)
        third = await asyncio.wait_for(blocked, timeout=2.0)
        budget.release(second)
        budget.release(third)

    asyncio.run(scenario())


def test_start_times_are_spaced_across_processes(tmp_path: Path) -> None:
    """Space two starts by the shared interval."""

    async def scenario() -> float:
        """Start two requests through separate budgets and return the gap."""
        first = RequestBudget(tmp_path, 1, 20.0)
        second = RequestBudget(tmp_path, 1, 20.0)
        starts: list[float] = []

        async def turn(budget: RequestBudget) -> None:
            """Wait for a turn and record the start time."""
            await budget.wait_turn()
            starts.append(monotonic())

        await asyncio.gather(turn(first), turn(second))
        return starts[1] - starts[0]

    assert asyncio.run(scenario()) >= 0.04


@pytest.mark.parametrize(("slots", "rate"), [(0, 10.0), (1, 0.0)])
def test_a_non_positive_bound_is_rejected(tmp_path: Path, slots: int, rate: float) -> None:
    """Reject a budget that cannot limit anything."""
    with pytest.raises(ValueError):
        RequestBudget(tmp_path, slots, rate)
