from __future__ import annotations

import os
from collections.abc import Mapping


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def available_cpu_count() -> int:
    """Return the CPUs available to this process, respecting VM/container affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def resolve_cpu_workers(requested: int | None = None) -> int:
    if requested is None:
        configured = os.getenv("HAWK_DERM_CPU_WORKERS")
        requested = int(configured) if configured else available_cpu_count()
    if requested < 1:
        raise ValueError("CPU worker count must be at least one")
    return min(int(requested), available_cpu_count())


def cpu_environment(workers: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a child-process environment with one explicit CPU budget."""
    workers = resolve_cpu_workers(workers)
    environment = dict(base or os.environ)
    environment["HAWK_DERM_CPU_WORKERS"] = str(workers)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = str(workers)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environment


def configure_process(workers: int | None = None) -> int:
    """Apply the CPU budget to native libraries already loaded in this process."""
    workers = resolve_cpu_workers(workers)
    os.environ["HAWK_DERM_CPU_WORKERS"] = str(workers)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(workers)

    try:
        from threadpoolctl import threadpool_limits

        # Keep the returned controller alive for the lifetime of the process.
        global _THREADPOOL_LIMITER
        _THREADPOOL_LIMITER = threadpool_limits(limits=workers)
    except ImportError:
        pass

    try:
        import torch

        torch.set_num_threads(workers)
        try:
            torch.set_num_interop_threads(min(4, workers))
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel work starts.
            pass
    except ImportError:
        pass
    return workers


_THREADPOOL_LIMITER = None
