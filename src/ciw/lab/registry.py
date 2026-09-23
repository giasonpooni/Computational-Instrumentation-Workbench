"""Queue definition and task implementations for the experimentalist loop.

The queue definition (``queue.json``) lists every task in its declared order.
Section modules register implementations with :func:`task`. A task without an
implementation, or whose requirements are unavailable, still produces a report:
its state is derived at run time and never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, resources
import json
from typing import Callable

# Section modules are imported in queue order. A missing module leaves its
# tasks deferred rather than failing the whole queue.
SECTION_MODULES = (
    "geodesic_jacobi", "flat_torus_topology", "surfaces_discrete", "observation", "sensor_fusion",
    "exchange_provenance", "lyapunov", "energy_gpu", "manufacturing", "implementation_targets",
    "research_portfolio",
)


@dataclass(frozen=True)
class Implementation:
    task_id: str
    run: Callable
    changed_files: tuple = ()
    regression_tests: tuple = ()
    requires: tuple = ()
    uses: tuple = field(default_factory=tuple)


_REGISTRY: dict[str, Implementation] = {}


def task(task_id: str, *, changed_files=(), regression_tests=(), requires=(), plan=None):
    """Register one queue task.

    ``requires`` lists hard requirements (``module:<name>``, ``provider:<role>``,
    ``tool:<name>`` or ``hardware:<name>``); when one is unavailable the task is
    reported as blocked using the static ``plan`` fields instead of running.
    """
    def decorate(function):
        if plan is not None:
            function.plan = dict(plan)
        if task_id in _REGISTRY:
            raise ValueError(f"Duplicate lab task implementation: {task_id}")
        _REGISTRY[task_id] = Implementation(task_id, function, tuple(changed_files),
                                            tuple(regression_tests), tuple(requires))
        return function
    return decorate


def load_queue() -> dict:
    text = resources.files("ciw.lab").joinpath("queue.json").read_text(encoding="utf-8")
    queue = json.loads(text)
    ids = [item["id"] for item in queue["tasks"]]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("Lab queue task identities must be unique and ordered")
    return queue


def load_implementations() -> tuple[dict, dict]:
    """Import section modules; return implementations and per-module import errors."""
    errors = {}
    for name in SECTION_MODULES:
        try:
            import_module(f"ciw.lab.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"ciw.lab.{name}":
                errors[name] = "section module not implemented"
            else:
                errors[name] = f"missing dependency: {exc.name}"
    queue_ids = {item["id"] for item in load_queue()["tasks"]}
    stray = set(_REGISTRY) - queue_ids
    if stray:
        raise ValueError(f"Implementations for unknown lab tasks: {sorted(stray)}")
    return dict(_REGISTRY), errors
