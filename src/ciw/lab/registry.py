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
import os
from pathlib import Path
import re
from typing import Callable

# Section modules are imported in queue order; a section may span several
# modules sharing its prefix. A missing or failing module leaves its tasks
# deferred with the recorded reason rather than failing the whole queue.
SECTION_MODULES = (
    "geodesic_jacobi", "geodesic_jacobi_limits", "flat_torus_topology", "surfaces_discrete",
    "surfaces_discrete_mesh", "observation", "sensor_fusion", "exchange_provenance",
    "exchange_provenance_bundles", "lyapunov", "energy_gpu", "manufacturing", "implementation_targets",
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


# Queue extensions append tasks after the packaged definition without editing
# it. They come from configure() or from os.pathsep-separated environment
# variables, so clean-room and subprocess runs see the same queue.
EXTENSION_SCHEMA = "ciw.lab-queue-extension.v1"
_CONFIGURED = {"extensions": (), "modules": ()}


def configure(extensions=(), modules=()) -> None:
    """Select queue extension files and extra implementation modules for this process."""
    _CONFIGURED["extensions"] = tuple(str(Path(path)) for path in extensions)
    _CONFIGURED["modules"] = tuple(modules)


def _configured(kind, variable):
    values = list(_CONFIGURED[kind])
    values += [item for item in os.environ.get(variable, "").split(os.pathsep) if item]
    return list(dict.fromkeys(values))


def _extension(path, base_ids, taken_keys, section_number):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != EXTENSION_SCHEMA or set(data) != {"schema", "section", "tasks"}:
        raise ValueError(f"Require {EXTENSION_SCHEMA} with section and tasks: {path}")
    section = data["section"]
    if (not isinstance(section, dict) or set(section) != {"key", "name"}
            or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(section["key"])) or not str(section["name"]).strip()):
        raise ValueError(f"Extension section needs a kebab-case key and a name: {path}")
    if section["key"] in taken_keys:
        raise ValueError(f"Extension section key already in the queue: {section['key']}")
    tasks = []
    for item in data["tasks"]:
        if not isinstance(item, dict) or set(item) != {"id", "title"} or not re.fullmatch(r"T[0-9]{3}", str(item["id"])):
            raise ValueError(f"Extension tasks need exactly an id T### and a title: {path}")
        if int(item["id"][1:]) <= max(int(i[1:]) for i in base_ids) or item["id"] in base_ids:
            raise ValueError(f"Extension task {item['id']} must follow every existing queue task")
        if not isinstance(item["title"], str) or not item["title"].strip():
            raise ValueError(f"Extension task {item['id']} needs a title")
        tasks.append({"id": item["id"], "number": int(item["id"][1:]), "section": section_number,
                      "section_key": section["key"], "title": item["title"]})
        base_ids = base_ids | {item["id"]}
    if not tasks:
        raise ValueError(f"Extension declares no tasks: {path}")
    return {"section": section_number, "key": section["key"], "name": section["name"], "extension": str(path)}, tasks


def load_queue() -> dict:
    """The packaged 168-task definition followed by any configured extensions."""
    text = resources.files("ciw.lab").joinpath("queue.json").read_text(encoding="utf-8")
    queue = json.loads(text)
    for path in _configured("extensions", "CIW_LAB_EXTENSIONS"):
        section, tasks = _extension(path, {t["id"] for t in queue["tasks"]}, {s["key"] for s in queue["sections"]},
                                    len(queue["sections"]) + 1)
        queue["sections"].append(section)
        queue["tasks"].extend(tasks)
    ids = [item["id"] for item in queue["tasks"]]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("Lab queue task identities must be unique and ordered")
    return queue


def load_implementations() -> tuple[dict, dict]:
    """Import section and configured modules; return implementations and per-module import errors."""
    errors = {}
    names = [f"ciw.lab.{name}" for name in SECTION_MODULES] + _configured("modules", "CIW_LAB_MODULES")
    for qualified in names:
        name = qualified.removeprefix("ciw.lab.")
        try:
            import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name == qualified:
                errors[name] = "section module not implemented"
            else:
                errors[name] = f"missing dependency: {exc.name}"
        except Exception as exc:  # retained as the deferral reason
            errors[name] = f"section module failed to import: {type(exc).__name__}: {exc}"
    queue_ids = {item["id"] for item in load_queue()["tasks"]}
    stray = set(_REGISTRY) - queue_ids
    if stray:
        raise ValueError(f"Implementations for unknown lab tasks: {sorted(stray)}")
    return dict(_REGISTRY), errors


def section_implementations(section_key: str) -> dict:
    """Import only one section's modules and return its registered implementations."""
    queue = load_queue()
    ids = {t["id"] for t in queue["tasks"] if t["section_key"] == section_key}
    if not ids:
        raise ValueError(f"Unknown lab section: {section_key}")
    prefix = section_key.replace("-", "_")
    for name in SECTION_MODULES:
        if name.startswith(prefix):
            import_module(f"ciw.lab.{name}")
    return {task_id: implementation for task_id, implementation in _REGISTRY.items() if task_id in ids}


def module_implementations(module: str) -> dict:
    """Import one lab module (e.g. ``exchange_provenance``) and return only the tasks it registers."""
    name = module if module.startswith("ciw.lab.") else f"ciw.lab.{module}"
    import_module(name)
    return {task_id: implementation for task_id, implementation in _REGISTRY.items()
            if getattr(implementation.run, "__module__", None) == name}
