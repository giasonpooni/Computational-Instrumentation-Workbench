"""Inputs, guards and retained fixtures for the exchange bundle tasks T091-T100.

Scope: the exact bytes of six small CIW example sources (embedded so the section
also runs from an installed wheel, where ``examples/`` is absent), a reader for
further ``examples/`` sources when the repository is reachable, a protocol
client, an execution guard that refuses the CIW execution entry points (session
analyses, recording operations, the workbench, every workbench workflow's
session/step/adapter entry points, provider adapters and subprocesses) while a
retained workspace is reopened, a synthetic numerical-heat bundle whose values
were never computed by a provider, a lab-written canonical JSON encoder (checked
byte for byte against ``json.dumps`` by T096 and the tests), the malformed exchange
fixture catalogue, the digest manifest of the golden workspaces under
``tests/fixtures/lab``, the platform fingerprint they were written on (with the
OpenBLAS kernel NumPy runs, from ``blas_probe``) and a comparison of the
energy-analysis recomputation to rounding tolerance where CIW's reopen
compares it bit for bit.

Non-claims: the embedded energy log is a synthetic fixture (its ``origin`` field
says so). A golden digest establishes byte identity of a retained file, not the
correctness, authorship or physical validity of its content, and a retained
runtime identity is metadata, not proof of which engine computed the values. The
fabricated heat bundle exists to measure what reopen validation does not check;
it is never presented as a provider result. An equal platform fingerprint does
not guarantee equal rounding; it names what is known to change it.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
import inspect
import json
from pathlib import Path
import struct
import tempfile
import zlib

from .blas_probe import openblas_core

EXAMPLE_SHA256 = {
    "energy-accuracy/baseline.json": "3ff4cb68be07e36d9796fb83402b0f72a46a3f1edff0d9e70addc9ab1695ef81",
    "declared-workloads/numerical-heat.json": "23029d09c154019bb0d9d3ce0b62650cd9fab3af2888ed64fa2073558e130533",
    "proved-heat/source.json": "b27dd0b64b23771e1b74ebb3937c68dcb83098b984a940abc1486e4cf69b8ad8",
    "geodesic-reference/flat-torus.json": "0d69dd3cc8c079b705b9cb5e90b46a21472f252753fdf2ffae5b4c21573e4da2",
    "geodesic-reference/curved-path.json": "4735fa731b4d3e86d357c444f3508c0df367dcef8684bc9152116cac251da7f5",
    "geometry-research/translation-flow.json": "d152bcb2c711952732a80ba725736d60adde713603c9bef9fc6e331769ee5bb7",
}
# zlib-compressed JSON map {relative example path: exact UTF-8 text}; the test
# suite checks every entry against the file under examples/ byte for byte.
_EXAMPLES = (
    "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"
)

# Golden workspaces saved by CIW itself (tests/fixtures/lab/golden). Updating a
# fixture is a deliberate act: regenerate it with write_golden_fixtures and
# record its new digest here in the same change.
GOLDEN_MANIFEST = {
    "golden/energy-accuracy-workspace.json": "cd08fa0a122b27db4d76d27f66e881c58c4bd3b0930f8b3a67d55da7b6ea6ada",
    "golden/numerical-heat-workspace.json": "98ba55e5be14213e6787024576b424cba6d4ba4da0a8cc905ddf652452490fec",
    "golden/oscillator-workspace.json": "68e63248edcb7bb6660734cbf0efe751c53ede53c5029873b4e21fb598a1586d",
}
# The SCR engine that produced the golden numerical-heat workspace (a locked
# offline build of SCR a59aba2 with cargo 1.94.1); T094 compares the retained
# runtime identity with it.
GOLDEN_SCR_ENGINE_SHA256 = "sha256:b9f40b3094ecf4b793676c766ffd559dc69abc4fada564c1bde94e79c6f0b201"
# platform_fingerprint() of the machine that wrote the golden workspaces, whose
# OpenBLAS ran its SkylakeX (AVX-512) kernels. Reopen recomputes the energy
# analysis (LAPACK solves) bit for bit; a platform with a different fingerprint
# may round differently (T094 measures by how much).
GOLDEN_PLATFORM = {"system": "Linux", "machine": "x86_64", "numpy": "2.4.3",
                   "blas": "scipy-openblas 0.3.31.dev", "lapack": "scipy-openblas 0.3.31.dev",
                   "openblas_core": "SkylakeX", "simd_found": ["AVX512_ICL", "AVX512_SPR", "X86_V3", "X86_V4"]}
FIXTURE_BINDING = "ciw-fixtures"


def platform_fingerprint() -> dict:
    """Platform facts that can change the last bits of NumPy/LAPACK results.

    System, NumPy, BLAS and LAPACK builds, SIMD extensions and, where NumPy's
    bundled OpenBLAS names it, the kernel it runs (``openblas_core``; the key is
    absent otherwise).
    """
    import platform
    import numpy as np
    info = {"system": platform.system(), "machine": platform.machine(), "numpy": np.__version__}
    core = openblas_core()
    if core is not None:
        info["openblas_core"] = core
    try:
        config = np.show_config(mode="dicts")
    except (TypeError, ValueError, AttributeError):  # older NumPy without mode="dicts"
        return info
    dependencies = config.get("Build Dependencies", {})
    for name in ("blas", "lapack"):
        entry = dependencies.get(name, {})
        info[name] = f"{entry.get('name')} {entry.get('version')}"
    info["simd_found"] = sorted(config.get("SIMD Extensions", {}).get("found", []))
    return info


def repository_example(name: str) -> bytes | None:
    """Bytes of ``examples/<name>`` through runner.repository_path; None in an installation without examples/."""
    from .runner import repository_path
    path = repository_path("examples", *name.split("/"))
    return path.read_bytes() if path is not None and path.is_file() else None


def example_bytes(name: str) -> bytes:
    """Exact bytes of an embedded CIW example source, checked against its digest."""
    texts = json.loads(zlib.decompress(base64.b64decode("".join(_EXAMPLES))))
    if name not in EXAMPLE_SHA256 or name not in texts:
        raise ValueError(f"Unknown embedded example: {name}")
    raw = texts[name].encode("utf-8")
    if sha256(raw).hexdigest() != EXAMPLE_SHA256[name]:
        raise ValueError(f"Embedded example differs from its recorded digest: {name}")
    return raw


def fixture_root(providers: dict | None = None) -> Path | None:
    """Golden fixture directory: an explicit binding, else the repository root, else none.

    The repository root honours ``CIW_LAB_REPOSITORY_ROOT`` (the clean-room run points it at copies).
    """
    bound = (providers or {}).get(FIXTURE_BINDING)
    if bound is not None:
        return Path(bound) if Path(bound).is_dir() else None
    from .runner import repository_path
    candidate = repository_path("tests", "fixtures", "lab")
    return candidate if candidate is not None and (candidate / "golden").is_dir() else None


class Client:
    """Protocol v1 requests against one session, as the websocket server sends them."""

    def __init__(self, session):
        self.session, self.count = session, 0

    def call(self, kind: str, payload=None) -> dict:
        self.count += 1
        return self.session.handle({"protocol_version": 1, "request_id": f"lab-{self.count}", "type": kind,
                                    "payload": {} if payload is None else payload})

    def ok(self, kind: str, payload=None) -> dict:
        response = self.call(kind, payload)
        if response["type"] != "response":
            raise ValueError(f"Workbench refused {kind}: {response['payload']}")
        return response["payload"]


def source_payload(kind: str, raw: bytes, label: str) -> dict:
    return {"kind": kind, "label": label, "bytes_b64": base64.b64encode(raw).decode("ascii")}


# CIW paths outside the workbench workflows that execute an analysis, a recording
# operation, a provider process or a trusted binding. execution_paths() adds the
# entry points of every workbench workflow kind. Reopening a retained workspace
# must reach none of them.
EXECUTION_PATHS = (
    ("subprocess", None, "Popen"),
    ("ciw.adapters.subprocess", None, "_bounded_process"),
    ("ciw.adapters.subprocess", "PinnedSubprocessAdapter", "__init__"),
    ("ciw.candidate_evidence", None, "_bounded_process"),
    ("ciw.session", None, "execute_operation"),
    ("ciw.adapters.registry", "AdapterRegistry", "execute"),
    ("ciw.instruments", None, "compute_statistics"),
    ("ciw.instruments", None, "compute_spectrum"),
    ("ciw.workbench", "Workbench", "execute"),
    ("ciw.workbench", "Workbench", "replay"),
    ("ciw.workbench", "Workbench", "bind_workflow"),
    ("ciw.workbench", "Workbench", "bind_candidate_adapter"),
    ("ciw.workbench", "Workbench", "execute_candidate"),
    ("ciw.energy_workflow", "EnergyAccuracyWorkflow", "_execute"),
    ("ciw.energy_workflow", "EnergyAccuracyWorkflow", "_step"),
    ("ciw.energy_workflow", "EnergyAccuracyWorkflow", "create_session"),
    ("ciw.energy_workflow", "EnergyAccuracyWorkflow", "replay_session"),
    ("ciw.declared_workload", "DeclaredWorkflow", "_adapters"),
    ("ciw.declared_workload", "DeclaredWorkflow", "_step"),
    ("ciw.declared_workload", "DeclaredWorkflow", "_execute"),
    ("ciw.declared_workload", "DeclaredWorkflow", "create_session"),
    ("ciw.declared_workload", "DeclaredWorkflow", "replay_session"),
)
# Pure recomputation used by validation; counted, not forbidden. Only the energy
# analysis is counted: it is the builtin analysis that reopening the kinds these
# tasks reopen (energy-accuracy, numerical-heat) recomputes. Validators of other
# kinds may recompute commitments or checks in process without being counted.
RECOMPUTATION_PATHS = (("ciw.energy_records", None, "analyze"),)
# Methods through which ciw.workbench runs a workflow kind (where the workflow defines them).
WORKFLOW_METHODS = ("_adapters", "_step", "_execute", "_execute_selected", "create_session", "replay_session")


@lru_cache(maxsize=1)
def workflow_entry_points() -> tuple:
    """(module, class or None, attribute) of every workbench workflow kind's entry points.

    Discovered from ``ciw.workbench._workflow`` for every operation kind, so a
    kind added to the workbench is guarded without editing this list. Methods
    are patched on the class that defines them.
    """
    from ..workbench import OPERATIONS, _workflow
    found = set()
    for kind in sorted(OPERATIONS):
        workflow = _workflow(kind)
        for name in WORKFLOW_METHODS:
            if inspect.ismodule(workflow):
                if callable(getattr(workflow, name, None)):
                    found.add((workflow.__name__, None, name))
                continue
            owner = next((cls for cls in type(workflow).__mro__ if name in vars(cls)), None)
            if owner is not None and owner is not object:
                found.add((owner.__module__, owner.__qualname__, name))
    return tuple(sorted(found, key=lambda entry: (entry[0], entry[1] or "", entry[2])))


def execution_paths() -> tuple:
    """Every guarded execution entry point: EXECUTION_PATHS plus the discovered workflow entry points."""
    return EXECUTION_PATHS + tuple(entry for entry in workflow_entry_points() if entry not in EXECUTION_PATHS)


def path_label(entry) -> str:
    return ".".join(filter(None, entry))


_MISSING = object()


class ExecutionForbidden(RuntimeError):
    """An execution entry point was reached while the guard was active."""


@contextmanager
def execution_guard(refuse: bool = True):
    """Intercept execution entry points; count validation recomputations.

    With ``refuse`` the entry points raise :class:`ExecutionForbidden`;
    otherwise they are only counted and then run. Yields
    ``{"attempts": [...], "recomputations": {...}}``. Every patched attribute is
    restored on exit, including after an exception.
    """
    record = {"attempts": [], "recomputations": {}}
    saved = []

    def owner(module, qualname):
        target = import_module(module)
        return target if qualname is None else getattr(target, qualname)

    def intercept(label, original):
        def run(*args, **kwargs):
            record["attempts"].append(label)
            if refuse:
                raise ExecutionForbidden(f"Execution path reached while guarded: {label}")
            return original(*args, **kwargs)
        return run

    def counting(label, original):
        def count(*args, **kwargs):
            record["recomputations"][label] = record["recomputations"].get(label, 0) + 1
            return original(*args, **kwargs)
        return count

    try:
        for paths, wrap in ((execution_paths(), intercept), (RECOMPUTATION_PATHS, counting)):
            for module, qualname, attribute in paths:
                target = owner(module, qualname)
                # Save the owner's own entry: an inherited method is restored by
                # deleting the shadowing attribute, not by copying it down.
                saved.append((target, attribute, vars(target).get(attribute, _MISSING)))
                setattr(target, attribute, wrap(path_label((module, qualname, attribute)), getattr(target, attribute)))
        yield record
    finally:
        for target, attribute, original in reversed(saved):
            if original is _MISSING:
                delattr(target, attribute)
            else:
                setattr(target, attribute, original)


def analysis_agreement(retained, fresh, tolerance: dict) -> dict:
    """Leaf-by-leaf comparison of a retained and a recomputed analysis.

    Floats agree when ``|retained - fresh| <= abs + rel * |retained|`` (the form
    ``ciw lab verify`` uses); every other leaf, and the structure (keys, list
    lengths), must be equal. ``bit_identical`` is CIW's own test: equal canonical
    JSON. ``max_abs_difference`` is over the floats.
    """
    from ..telemetry import canonical
    counts = {"float_fields": 0, "other_fields": 0, "floats_differing": 0, "floats_outside_tolerance": 0,
              "other_fields_differing": 0, "max_abs_difference": 0.0}

    def walk(old, new):
        if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
            for key in sorted(old):
                walk(old[key], new[key])
        elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            for pair in zip(old, new):
                walk(*pair)
        elif type(old) is float and type(new) is float:
            counts["float_fields"] += 1
            difference = abs(old - new)
            counts["floats_differing"] += old != new
            counts["max_abs_difference"] = max(counts["max_abs_difference"], difference)
            counts["floats_outside_tolerance"] += not difference <= tolerance["abs"] + tolerance["rel"] * abs(old)
        else:
            counts["other_fields"] += 1
            counts["other_fields_differing"] += type(old) is not type(new) or old != new

    walk(retained, fresh)
    counts["bit_identical"] = canonical(retained) == canonical(fresh)
    counts["within_tolerance"] = not counts["floats_outside_tolerance"] and not counts["other_fields_differing"]
    return counts


def retained_energy_analyses(workspace) -> dict:
    """{log digest: retained analysis} of the energy-accuracy bundles of a saved workspace (the first per log)."""
    retained = {}
    bundles = (workspace.get("workbench") or {}).get("bundles") if isinstance(workspace, dict) else None
    for record in bundles if isinstance(bundles, list) else ():
        try:
            if record["kind"] == "energy-accuracy":
                data = record["native"]["steps"][0]["result"]["data"]
                retained.setdefault(data["log_digest"], data)
        except (KeyError, IndexError, TypeError):
            continue
    return retained


@contextmanager
def energy_recomputation_within(retained: dict, tolerance: dict, records: list):
    """Compare each energy-analysis recomputation with the retained analysis to ``tolerance``, not bit for bit.

    Enter it inside :func:`execution_guard`, which still counts every
    recomputation. Each ``ciw.energy_records.analyze`` call recomputes the
    analysis and appends its :func:`analysis_agreement` with the retained analysis
    of the same log (``retained`` maps log digests to them) to ``records``. It
    returns the retained analysis when the recomputation agrees within
    ``tolerance``, so CIW's bit-for-bit comparison passes and every other reopen
    check still decides, and the recomputation otherwise, which CIW refuses. The
    function is restored on exit.
    """
    from .. import energy_records
    recompute = energy_records.analyze

    def analyze(log):
        fresh = recompute(log)
        expected = retained.get(log.get("log_digest")) if isinstance(log, dict) else None
        if expected is None:
            records.append({"retained": False, "bit_identical": False, "within_tolerance": False})
            return fresh
        agreement = analysis_agreement(expected, fresh, tolerance)
        records.append({"retained": True, **agreement})
        return deepcopy(expected) if agreement["within_tolerance"] else fresh

    energy_records.analyze = analyze
    try:
        yield records
    finally:
        energy_records.analyze = recompute


def heat_reference(values, steps: int) -> list[int]:
    """Independent integer Jacobi heat step: u_i += trunc((u_{i-1} - 2 u_i + u_{i+1}) / 4), fixed ends."""
    field = [int(value) for value in values]
    for _ in range(int(steps)):
        previous = field[:]
        for index in range(1, len(field) - 1):
            delta = previous[index - 1] - 2 * previous[index] + previous[index + 1]
            quotient = abs(delta) // 4
            field[index] = previous[index] + (quotient if delta >= 0 else -quotient)
    return field


def _occurrence(prefix: str, seed: str) -> str:
    return prefix + sha256(seed.encode()).hexdigest()[:32]


FABRICATED_SOURCE_TREE = "0" * 40


def fabricated_heat_catalog(values, *, experiment_id: str = "ciw-lab-fabricated-heat-claim",
                            source_tree: str = FABRICATED_SOURCE_TREE) -> dict:
    """A retained-workbench catalog holding one content-consistent numerical-heat bundle.

    Every commitment is recomputed from ``values``, so CIW's reopen validation
    accepts it, but no provider computed those values and its runtime identity
    is fabricated: CIW's pinned SCR revision, module and source root with
    ``source_tree`` (by default an invented tree, :data:`FABRICATED_SOURCE_TREE`;
    T100 also passes CIW's pinned tree to show that a copied pin still matches).
    The bundle digest and verification are computed over that identity, so the
    bundle is sealed with it. Deterministic: no clock, no random occurrence
    identities.
    """
    from ..declared_workload import AUTHORITY, HEAT_DESCRIPTOR, PINS, RESULT_SCHEMA, DeclaredWorkflow, _commit, _verification
    from ..telemetry import _bundle_digest, byte_digest, canonical, digest
    from ..workbench import SCHEMA, _source

    workflow = DeclaredWorkflow("numerical-heat")
    declared = json.loads(example_bytes("declared-workloads/numerical-heat.json"))
    declared["experiment_id"] = experiment_id
    raw = canonical(declared)
    source = workflow._source(raw)
    evidence = byte_digest(raw)
    inputs = struct.pack("<II", source["steps"], len(source["initial_values"])) + b"".join(
        struct.pack("<q", v) for v in source["initial_values"])
    output = b"".join(struct.pack("<q", int(v)) for v in values)
    identities = {"program_identity": _commit("program", [HEAT_DESCRIPTOR]), "input_identity": _commit("input", [inputs]),
                  "output_identity": _commit("output", [output]),
                  "specification_identity": _commit("specification", [HEAT_DESCRIPTOR, b"", inputs])}
    identities["computation_identity"] = _commit("computation", [bytes.fromhex(identities[key]) for key in
                                                 ("program_identity", "input_identity", "output_identity")]
                                                 + [struct.pack("<I", 0)])
    data = {"specification": {"program": HEAT_DESCRIPTOR.hex(), "configuration": "", "input_payload": inputs.hex()},
            **identities, "engine_occurrence": 0, "status": "completed", "exit_code": 0, "output": output.hex(),
            "detail": None, "values": [int(v) for v in values]}

    def step(seed):
        occurrence = _occurrence("execution-", experiment_id + seed)
        result = {"schema": RESULT_SCHEMA, "operation_id": workflow.operation, "execution_ref": occurrence,
                  "input_refs": [evidence], "data": deepcopy(data), "authority": deepcopy(AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": workflow.operation, "data": deepcopy(data)}
        return {"runtime_ref": "scr", "operation_id": workflow.operation, "execution_id": occurrence,
                "input_refs": [evidence], "request": deepcopy(source), "request_sha256": digest(source),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    runtime = {"schema": "ciw.subprocess-runtime.v1", "adapter_version": "fabricated-by-ciw-lab",
               "repository_root": "/fabricated/not-a-provider-checkout", "revision": PINS["numerical-heat"]["revision"],
               "source_tree": source_tree, "module": PINS["numerical-heat"]["module"],
               "source_root": PINS["numerical-heat"]["source_root"], "python_executable": "/fabricated/python",
               "python_sha256": "0" * 64, "python_version": "0.0.0", "dependencies": {},
               "engine": {"sha256": "sha256:" + "0" * 64, "byte_count": 1,
                          "source_binding": "operator_asserted_not_attested"}}
    bundle = {"schema": workflow.schema, "session_id": _occurrence("session-", experiment_id),
              "created_at": "2026-01-01T00:00:00+00:00",
              "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                         "evidence": [{"artifact_ref": evidence, "sha256": evidence,
                                       "bytes_b64": base64.b64encode(raw).decode()}]},
              "configuration": deepcopy(source["configuration"]), "runtimes": {"scr": runtime}, "steps": [step("original")]}
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, step("reproduction"))
    retained = _source(source_payload("numerical-heat", raw, "Fabricated heat claim (ciw.lab counterexample)"))
    record = {"kind": "numerical-heat", "source_id": retained["source_id"], "upstream_bundle_id": None,
              "bundle_id": bundle["bundle_digest"], "native": bundle}
    return {"schema": SCHEMA, "revision": 2, "sources": [retained], "bundles": [record]}


def merge_catalogs(first: dict, second: dict) -> dict:
    """Concatenate two retained-workbench catalogs of the v1 schema."""
    sources, bundles = first["sources"] + second["sources"], first["bundles"] + second["bundles"]
    return {"schema": first["schema"], "revision": len(sources) + len(bundles),
            "sources": deepcopy(sources), "bundles": deepcopy(bundles)}


# A trusted binding that names no checkout: T091 puts it in a session's workbench
# to show that saving and reopening never carry a binding across.
SYNTHETIC_BINDING = {"scr": "/ciw-lab/synthetic-binding/scr", "engine": "/ciw-lab/synthetic-binding/execution-cli"}


def add_provider_content(session, values, *, experiment_id: str = "ciw-lab-provider-kind-bundle") -> str:
    """Give a session a provider-kind numerical-heat bundle and a synthetic trusted binding; returns the bundle id.

    The bundle is :func:`fabricated_heat_catalog` (content-consistent, fabricated
    runtime identity). The binding goes into the workbench's trusted-binding
    table, the state ``Workbench.bind_workflow`` leaves after it has validated a
    real checkout; it points at no checkout and nothing executes through it.
    """
    from ..workbench import Workbench
    fabricated = fabricated_heat_catalog(values, experiment_id=experiment_id)
    session.workbench = Workbench.restore(merge_catalogs(session.workbench.serialize(), fabricated))
    session.workbench._bindings["numerical-heat"] = {role: Path(path) for role, path in SYNTHETIC_BINDING.items()}
    return fabricated["bundles"][0]["bundle_id"]


# ------------------------------------------------------------------ golden bundles

def build_energy_session(directory: Path):
    """Oscillator session with retained oscillator results and an energy original + replay."""
    from ..instruments import make_demo_run
    from ..session import Session
    session = Session(make_demo_run(), Path(directory))
    client = Client(session)
    client.ok("analysis.stats", {})
    revision = client.ok("session.get")["selection"]["revision"]
    client.ok("selection.update", {"expected_revision": revision, "channel": "v", "interval_s": [2.0, 6.0]})
    client.ok("analysis.spectrum", {})
    client.ok("operation.execute", {"operation_id": "statistics.v1", "parameters": {}})
    source = client.ok("source.add", source_payload("energy-accuracy", example_bytes("energy-accuracy/baseline.json"),
                                                    "Synthetic energy-log fixture (examples/energy-accuracy/baseline.json)"))
    original = client.ok("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                "parameters": {"source_id": source["source_id"]}})
    replay = client.ok("bundle.replay", {"bundle_id": original["bundle_id"]})
    return session, client, {"source_id": source["source_id"], "original": original["bundle_id"],
                             "replay": replay["bundle"]["bundle_id"]}


def write_golden_fixtures(directory, *, scr=None, engine=None) -> dict:
    """Write the golden workspaces with the current code; returns their SHA-256 digests.

    The numerical-heat workspace needs a clean SCR checkout at the pinned
    revision and a trusted ``execution-cli``; without them it is not written.
    """
    from ..instruments import make_demo_run
    from ..session import Session
    directory = Path(directory)
    (directory / "golden").mkdir(parents=True, exist_ok=True)
    written = {}
    with tempfile.TemporaryDirectory(prefix="ciw-lab-golden-") as scratch:
        scratch = Path(scratch)
        session, _, _ = build_energy_session(scratch / "energy")
        # The energy workspace keeps only its workbench content; oscillator
        # results live in their own golden workspace.
        energy = Session(make_demo_run(), scratch / "energy-only")
        energy.workbench = session.workbench
        energy.save_workspace(directory / "golden" / "energy-accuracy-workspace.json")
        oscillator = Session(make_demo_run(), scratch / "oscillator")
        oscillator.selection, oscillator.results = session.selection, session.results
        oscillator.executions = session.executions
        oscillator.save_workspace(directory / "golden" / "oscillator-workspace.json")
        if scr is not None and engine is not None:
            heat = Session(make_demo_run(), scratch / "heat")
            heat.workbench.bind_workflow("numerical-heat", {"scr": Path(scr), "engine": Path(engine)})
            client = Client(heat)
            source = client.ok("source.add", source_payload(
                "numerical-heat", example_bytes("declared-workloads/numerical-heat.json"),
                "Declared integer heat workload (examples/declared-workloads/numerical-heat.json)"))
            original = client.ok("operation.execute", {"operation_id": "ciw.numerical-heat.v1",
                                                        "parameters": {"source_id": source["source_id"]}})
            client.ok("bundle.replay", {"bundle_id": original["bundle_id"]})
            heat.save_workspace(directory / "golden" / "numerical-heat-workspace.json")
    for path in sorted((directory / "golden").glob("*.json")):
        written["golden/" + path.name] = sha256(path.read_bytes()).hexdigest()
    return written


# ------------------------------------------------------------ malformed fixtures

_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def canonical_text(value) -> str:
    """Canonical JSON text written from the exchange producer specification, without json.dumps.

    Members sorted by code point, no whitespace, non-ASCII characters literal,
    quote, backslash and control characters escaped as JSON requires, floats
    as their shortest round-trip form; NaN and infinity are refused.
    """
    if value is None:
        return "null"
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Canonical JSON refuses nonfinite numbers")
        return repr(value)
    if isinstance(value, str):
        return '"' + "".join(_ESCAPES.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 else char)
                             for char in value) + '"'
    if isinstance(value, list):
        return "[" + ",".join(canonical_text(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON members need string names")
        return "{" + ",".join(canonical_text(key) + ":" + canonical_text(value[key]) for key in sorted(value)) + "}"
    raise ValueError(f"Canonical JSON cannot encode {type(value).__name__}")


def seal_independently(artifact: dict, field: str) -> dict:
    """Seal an exchange artifact as its producers do (schema NUL canonical JSON), using canonical_text."""
    artifact = deepcopy(artifact)
    artifact.pop(field, None)
    body = canonical_text(artifact).encode("utf-8")
    artifact[field] = "sha256:" + sha256(artifact["schema"].encode("utf-8") + b"\x00" + body).hexdigest()
    return artifact


_sealed = seal_independently


def exchange_result_artifact(**changes) -> dict:
    """A small sealed synthetic result artifact; not a provider output."""
    from ..exchange import RESULT_SCHEMA
    artifact = {"schema": RESULT_SCHEMA, "input_refs": ["example:synthetic-batch"],
                "components": [{"name": "east", "value": 1.2, "unit": "m"}],
                "applicability": "synthetic malformed-fixture base, ciw.lab", "created_at": "2026-09-23T00:00:00Z"}
    artifact.update(changes)
    return _sealed(artifact, "result_id")


def _dumps(value) -> bytes:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def malformed_fixtures() -> dict:
    """Committed malformed fixtures (name -> exact bytes); deterministic and small.

    Each entry names the defect it carries. Oversized inputs and inputs that
    embed a full recording are generated at run time instead of committed.
    """
    log = json.loads(example_bytes("energy-accuracy/baseline.json"))
    base = exchange_result_artifact()
    tampered = deepcopy(base)
    tampered["components"][0]["value"] = 2.4
    wrong_schema = exchange_result_artifact(schema="notation.instrument.result-artifact.v2")
    return {
        "duplicate-key.json": b'{"schema": "notation.instrument.result-artifact.v1", "schema": "ciw.energy-accuracy-log.v1"}\n',
        "nan.json": b'{"schema": "notation.instrument.result-artifact.v1", "components": [{"name": "east", "value": NaN, "unit": "m"}]}\n',
        "infinity.json": b'{"schema": "notation.instrument.result-artifact.v1", "components": [{"name": "east", "value": -Infinity, "unit": "m"}]}\n',
        "overflow.json": b'{"schema": "notation.instrument.result-artifact.v1", "components": [{"name": "east", "value": 1e999, "unit": "m"}]}\n',
        "truncated-energy-log.json": example_bytes("energy-accuracy/baseline.json")[:4096],
        "invalid-utf8.json": b'{"schema": "notation.instrument.result-artifact.v1", "label": "\xff"}\n',
        "deep-nesting.json": b"[" * 20000 + b"]" * 20000 + b"\n",
        "wrong-schema-exchange.json": _dumps(wrong_schema),
        "wrong-schema-energy-log.json": _dumps(dict(log, schema="ciw.energy-accuracy-log.v2")),
        "wrong-type-energy-log.json": _dumps(dict(log, origin=True)),
        "incomplete-workspace.json": _dumps({"workspace_version": 3}),
        "wrong-revision-workbench.json": _dumps({"schema": "ciw.retained-workbench.v1", "revision": 1,
                                                 "sources": [], "bundles": []}),
        "extra-field-energy-log.json": _dumps(dict(log, calibration_certificate="none supplied")),
        "wrong-identity-result.json": _dumps(tampered),
    }


def write_malformed_fixtures(directory) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, raw in sorted(malformed_fixtures().items()):
        (directory / name).write_bytes(raw)
        digests[name] = sha256(raw).hexdigest()
    return digests
