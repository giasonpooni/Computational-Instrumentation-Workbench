"""Optional comparison with the pinned Flat-Torus-Geodesic-Reference (FTR) provider.

FTR declares Python >= 3.12, so it runs in a separate interpreter bound as the
``ftr-python`` provider, with its checkout bound as ``ftr``. The checkout must
be clean and at the revision and tree pinned in
``ciw.geodesic_reference.PINS['flat-torus-reference']``; anything else is
refused before execution. The provider returns lattice folds, loop lengths
and traced closed geodesics; agreement with this module is independent
implementation agreement, not verification by another party.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

FTR_REPOSITORY = "giasonpooni/Flat-Torus-Geodesic-Reference"
FTR_IMPLEMENTATION = "Flat-Torus-Geodesic-Reference"
ENTRY_POINTS = ("flat_torus.fold.fold_to_fundamental_domain", "flat_torus.lengths.loop_length",
                "flat_torus.trajectories.trace_closed_geodesic", "flat_torus.lattice.normalized_lattice")

# Runs in the provider interpreter; the checkout's src directory is put first on sys.path.
_BOOTSTRAP = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy
from flat_torus.fold import fold_to_fundamental_domain
from flat_torus.lattice import ShapeParameter, normalized_lattice
from flat_torus.lengths import Winding, loop_length
from flat_torus.trajectories import trace_closed_geodesic
request = json.loads(sys.stdin.read())
folds = []
for case in request["folds"]:
    shape = ShapeParameter(*case["tau"])
    result = fold_to_fundamental_domain(shape, Winding(*case["winding"]))
    folds.append({"reduced": [result.reduced.x, result.reduced.y], "matrix": list(result.matrix.as_tuple()),
                  "word": [list(pair) for pair in result.word.as_pairs()],
                  "reduced_winding": list(result.reduced_winding.as_tuple()), "length_pair": list(result.length_pair())})
lengths = [loop_length(complex(*case["tau"]), case["m"], case["n"]) for case in request["lengths"]]
traces = []
for case in request["traces"]:
    lattice = normalized_lattice(complex(*case["tau"]))
    start = lattice.from_cover_coordinates(*case["start_cover"])
    trajectory = trace_closed_geodesic(lattice, case["m"], case["n"], start=start, samples=case["samples"])
    traces.append({"crossings": len(trajectory.crossings), "closed": bool(trajectory.closed),
                   "length": float(trajectory.length)})
areas = [normalized_lattice(complex(*tau)).area() for tau in request["areas"]]
print(json.dumps({"python": sys.version.split()[0], "numpy": numpy.__version__, "folds": folds,
                  "lengths": lengths, "traces": traces, "areas": areas}, sort_keys=True, allow_nan=False))
'''


class ProviderRefusal(ValueError):
    """The bound provider does not match its pin or did not execute."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def ftr_pin() -> dict:
    from ..geodesic_reference import PINS

    pin = PINS["flat-torus-reference"]
    return {"revision": pin["revision"], "source_tree": pin["source_tree"], "source_root": pin["source_root"]}


def verify_checkout(checkout) -> dict:
    """Refuse an unreadable, dirty or differently pinned checkout."""
    from .runner import git_identity

    pin = ftr_pin()
    try:
        identity = git_identity(Path(checkout))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProviderRefusal("FTR_CHECKOUT_UNREADABLE", f"Provider checkout is not a readable git repository: {exc}")
    if identity["revision"] != pin["revision"]:
        raise ProviderRefusal("FTR_REVISION_MISMATCH",
                              f"Provider revision {identity['revision']} differs from pin {pin['revision']}")
    if identity["source_tree"] != pin["source_tree"]:
        raise ProviderRefusal("FTR_TREE_MISMATCH", "Provider source tree differs from the pinned tree")
    if identity["dirty"]:
        raise ProviderRefusal("FTR_CHECKOUT_DIRTY", "Provider checkout has uncommitted tracked changes")
    return {"repository": FTR_REPOSITORY, "revision": identity["revision"], "source_tree": identity["source_tree"],
            "dirty": False}


def run_ftr(checkout, interpreter, request: dict, timeout: float = 120.0) -> dict:
    """Execute the pinned provider on one request; returns its data and runtime identity."""
    identity = verify_checkout(checkout)
    interpreter = Path(interpreter)
    if not interpreter.exists():
        raise ProviderRefusal("FTR_INTERPRETER_MISSING", f"Provider interpreter {interpreter} does not exist")
    root = Path(checkout) / ftr_pin()["source_root"]
    try:
        result = subprocess.run([str(interpreter), "-I", "-c", _BOOTSTRAP, str(root)], input=json.dumps(request),
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderRefusal("FTR_EXECUTION_FAILED", f"Provider subprocess did not complete: {exc}")
    if result.returncode != 0:
        raise ProviderRefusal("FTR_EXECUTION_FAILED", result.stderr.strip()[-400:] or "provider exited nonzero")
    data = json.loads(result.stdout)
    after = verify_checkout(checkout)
    if after != identity:
        raise ProviderRefusal("FTR_CHANGED_DURING_EXECUTION", "Provider checkout changed during execution")
    runtime = dict(identity, interpreter=str(interpreter), python=data["python"], numpy=data["numpy"],
                   entry_points=list(ENTRY_POINTS), executed=True)
    return {"data": data, "identity": runtime}


def checker_identity(identity: dict) -> dict:
    return {"implementation": f"{FTR_IMPLEMENTATION}@{identity['revision'][:12]}", "revision": identity["revision"],
            "source_tree": identity["source_tree"]}


def provider_basis(identity: dict) -> dict:
    return {"repository": identity["repository"], "revision": identity["revision"],
            "source_tree": identity["source_tree"], "executed": True}
