"""Measure local model time and process memory without implicit downloads."""

import os
import platform
import resource
import sys
from importlib import import_module, util
from time import perf_counter
from typing import Any

from ..datasets import DatasetSpec
from .local import predict_baseline


def hardware_info() -> dict[str, Any]:
    """Read CPU and physical memory information."""
    memory = None
    if hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": memory}


def profile_model(
    model: str, spec: DatasetSpec, train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]], seed: int = 42, device: str = "cpu",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one complete baseline call.

    Args:
        model: Local baseline name.
        spec: Dataset contract.
        train_rows: Full selected context.
        test_rows: Full ordered test set.
        seed: Model seed.
        device: CPU or CUDA device.
        options: Baseline checkpoint and explicit download settings.

    Returns:
        Model output with hardware, wall time, and peak memory measurements.
        Process peak RSS includes previous allocations in this process.
    """
    torch = None
    if device.startswith("cuda") and util.find_spec("torch") is not None:
        torch = import_module("torch")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    result = predict_baseline(model, spec, train_rows, test_rows, seed, device, options)
    if torch is not None:
        torch.cuda.synchronize(device)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["profile"] = {
        **hardware_info(), "wall_seconds": perf_counter() - started,
        "process_peak_rss_bytes": peak * (1 if sys.platform == "darwin" else 1024),
        "rss_measurement": "process_lifetime_high_water_mark",
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if torch is not None else None,
    }
    return result
