"""T145's Julia path: the oscillator acceptance set of docs/JULIA_SP1.md behind SCR's dispatcher.

Scope: the fixtures (CIW's default oscillator, a mixed initial state with a
negative component, the undamped limit, a tolerance ladder, repeated and
interleaved runs, refusals and channel failures), their plan of worker
sessions, execution through SCR's ``execution.dispatcher.SpecificationDispatcher``
with the Julia worker (:mod:`ciw.lab.julia_worker`) as its ``runner`` in an
isolated interpreter that imports the bound SCR checkout, or straight to the
worker when no SCR checkout is bound, and the analysis of the retained bytes:
decoded outputs against CIW's closed-form oscillator
(:func:`ciw.adapters.oscillator.closed_form`), and SCR's program, input,
specification, output and computation commitments recomputed here from the
exact request and response frames.

Acceptance thresholds are declared below, before any Julia result was seen,
and are componentwise ``|x - x_ref| <= abs + rel * |x_ref|``. The solver's
``reltol`` is a local error control, not a global error bound: the tolerance
ladder records the global errors it produced rather than assuming either.

Non-claims: agreement with the closed form is agreement of two
implementations (OrdinaryDiffEq's Tsit5 and CIW's formula) of one model; it
says nothing about a physical oscillator. Same-host byte equality of repeated
occurrences is not cross-platform agreement, and no platform other than the
one running here executes the worker.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from . import julia_worker as jw

# ---------------------------------------------------------------- declarations
# The fixed timestamp SCR's dispatcher is given (its discipline: caller-supplied, never read from a clock).
EXTRACTED_AT = "2026-09-23T00:00:00Z"
DEFAULT_CONFIGURATION = {"abstol": 1e-10, "reltol": 1e-10, "dt": 1e-3, "dtmax": 12.0, "maxiters": 1_000_000}
TOLERANCE_LADDER = (1e-6, 1e-8, 1e-10, 1e-12)
# Componentwise acceptance thresholds against the closed form at DEFAULT_CONFIGURATION, declared before results.
ACCEPTANCE = {"q": {"abs": 1e-6, "rel": 1e-6}, "v": {"abs": 1e-5, "rel": 1e-6}, "energy": {"abs": 1e-5, "rel": 1e-6}}
UNITS = {"q": "m", "v": "m/s", "energy": "J"}
PHASE_THRESHOLD = 1e-6          # rad, undamped limit
ENERGY_DRIFT_THRESHOLD = 1e-6   # relative to the initial energy, undamped limit
REPLAY_AGREEMENT = 1e-12        # declared agreement of repeated occurrences, in each channel's unit
HALT_MAXITERS = 10              # a valid configuration the solver cannot finish within
STALL_TIMEOUT_S = 2.0
SCR_REPOSITORY = "giasonpooni/Scientific-Computation-Runtime"
# The producer of the independent checks: SciML's solver, run through the CIW-authored worker (the worker's own
# Julia code only sets up the right-hand side and encodes bytes). Its family is ``ordinarydiffeq``.
PRODUCER = "OrdinaryDiffEq.jl Tsit5 (OrdinaryDiffEqTsit5) on Julia"
PRODUCER_PACKAGES = ("OrdinaryDiffEqTsit5", "OrdinaryDiffEqCore", "SciMLBase", "DiffEqBase")


def fixtures() -> dict:
    """The acceptance fixtures: parameters, initial state at t = 0, the requested grid and its declared interval."""
    from ..adapters.oscillator import make_demo_run

    run = make_demo_run()
    model = run["metadata"]["model"]
    default = {"omega_0": model["omega_0_rad_s"], "gamma": model["gamma_s_inv"], "mass": model["mass_kg"],
               "q0": model["initial_q_m"], "v0": model["initial_v_m_s"], "duration": run["metadata"]["duration_s"],
               "times": run["time_s"]}
    return {
        # CIW's default recording: 768 samples at 64 Hz over [0, 12); its channels are the retained reference.
        "ciw-default": default,
        # Nonzero q0 and v0 with a negative component, another mass and frequency: catches state-order and sign errors.
        "mixed-state": {"omega_0": 3.0, "gamma": 0.4, "mass": 2.5, "q0": -0.7, "v0": 2.5, "duration": 12.0,
                        "times": (np.arange(600, dtype=np.float64) / 50.0).tolist()},
        # The undamped limit (gamma = 0) of the default fixture.
        "undamped": dict(default, gamma=0.0),
    }


def reference(fixture: dict, name: str) -> dict:
    """CIW's closed-form values on the fixture grid; the default fixture's are the retained recording's channels."""
    from ..adapters.oscillator import closed_form, make_demo_run

    if name == "ciw-default":
        run = make_demo_run()
        return {channel: np.asarray(run["channels"][channel]["values"]) for channel in UNITS}
    q, v, energy = closed_form(np.asarray(fixture["times"]), omega_0=fixture["omega_0"], gamma=fixture["gamma"],
                               mass=fixture["mass"], q0=fixture["q0"], v0=fixture["v0"])
    return {"q": q, "v": v, "energy": energy}


def _input(fixture: dict) -> bytes:
    return jw.encode_input(fixture["omega_0"], fixture["gamma"], fixture["mass"], fixture["q0"], fixture["v0"],
                           fixture["duration"], fixture["times"])


def _configuration(**changes) -> bytes:
    return jw.encode_configuration(**dict(DEFAULT_CONFIGURATION, **changes))


def _raw_input(fixture: dict, **changes) -> bytes:
    """An input encoded without host validation, so the worker's own refusals are exercised."""
    values = dict(fixture, **changes)
    times = values["times"]
    return (jw._tag(jw.INPUT_SCHEMA) + struct.pack("<6d", values["omega_0"], values["gamma"], values["mass"],
                                                    values["q0"], values["v0"], values["duration"])
            + struct.pack("<I", len(times)) + struct.pack(f"<{len(times)}d", *times))


def _raw_configuration(**controller) -> bytes:
    profile = dict(jw.CONTROLLER, **controller)
    config = DEFAULT_CONFIGURATION
    return (jw._tag(jw.CONFIGURATION_SCHEMA)
            + struct.pack("<4d", config["abstol"], config["reltol"], config["dt"], config["dtmax"])
            + struct.pack("<Q", config["maxiters"]) + struct.pack(f"<{len(profile)}d", *profile.values()))


# Worker-side refusals: request bytes that bypass the host's own validation. Name -> expected refusal code.
WORKER_REFUSALS = {"refuse-nonfinite": "nonfinite_number", "refuse-grid": "grid_not_increasing",
                   "refuse-bounds": "out_of_bounds", "refuse-program": "unknown_operation",
                   "refuse-controller": "configuration_unsupported", "refuse-truncated-input": "input_malformed"}
# Protocol-mock channel failures (never numerical evidence): mode -> the failure code that must end the session.
MOCK_FAILURES = {"truncated": "unexpected_eof", "malformed": "malformed_response", "oversized": "response_too_large",
                 "wrong_request_id": "request_id_mismatch", "unknown_kind": "malformed_response",
                 "exit_after_handshake": "worker_exited"}


def plan() -> list:
    """Worker sessions and requests in execution order (docs/lab/IMPLEMENTATION_TARGETS.md lists them)."""
    cases = fixtures()
    default = cases["ciw-default"]
    program = jw.OPERATION_DESCRIPTOR.hex()

    def request(label, session, fixture, configuration=None, input_payload=None, **extra):
        return {"op": "request", "label": label, "session": session, "fixture": fixture, "program": program,
                "configuration": (configuration or _configuration()).hex(),
                "input": (input_payload or _input(cases[fixture])).hex(), **extra}

    bad_grid = list(default["times"])
    bad_grid[10], bad_grid[11] = bad_grid[11], bad_grid[10]
    steps = [{"op": "start", "label": "start worker-1", "session": "worker-1"},
             request("A1", "worker-1", "ciw-default"),
             request("B", "worker-1", "mixed-state"),
             request("A2", "worker-1", "ciw-default"),
             request("undamped", "worker-1", "undamped")]
    steps += [request(f"reltol-{tolerance:.0e}", "worker-1", "ciw-default",
                      _configuration(abstol=tolerance, reltol=tolerance))
              for tolerance in TOLERANCE_LADDER if tolerance != DEFAULT_CONFIGURATION["reltol"]]
    steps += [request("halted-maxiters", "worker-1", "ciw-default", _configuration(maxiters=HALT_MAXITERS)),
              request("refuse-nonfinite", "worker-1", "ciw-default", input_payload=_raw_input(default, omega_0=math.nan)),
              request("refuse-grid", "worker-1", "ciw-default", input_payload=_raw_input(default, times=bad_grid)),
              request("refuse-bounds", "worker-1", "ciw-default",
                      input_payload=_raw_input(default, gamma=0.6 * default["omega_0"])),
              dict(request("refuse-program", "worker-1", "ciw-default"),
                   program=jw.OPERATION_DESCRIPTOR.replace(b".v1\n", b".v2\n", 1).hex()),
              request("refuse-controller", "worker-1", "ciw-default", _raw_configuration(qmax=5.0)),
              request("refuse-truncated-input", "worker-1", "ciw-default", input_payload=_input(default)[:-8]),
              request("A3", "worker-1", "ciw-default"),
              request("oversized-frame", "worker-1", "ciw-default", fault="oversized_header"),
              request("after-session-end", "worker-1", "ciw-default"),
              {"op": "start", "label": "start worker-2", "session": "worker-2"},
              request("A4", "worker-2", "ciw-default"),
              request("stall-timeout", "worker-2", "ciw-default", fault="stall", timeout=STALL_TIMEOUT_S),
              {"op": "start", "label": "start worker-3", "session": "worker-3"},
              request("crash", "worker-3", "ciw-default", fault="crash"),
              {"op": "start", "label": "start worker-4 (threads 2, expected 1)", "session": "worker-4", "threads": 2,
               "expected_threads": 1}]
    for mode in MOCK_FAILURES:
        session = f"mock-{mode}"
        steps += [{"op": "start", "label": f"start {session}", "session": session, "mock": mode, "replay": "worker-1"},
                  request(session, session, "ciw-default")]
    return steps


# ------------------------------------------------------------------ execution
_SCR_DRIVER = r'''
import json, sys
scr_root, ciw_root = sys.argv[1:3]
sys.path[:0] = [ciw_root, scr_root]
from pathlib import Path
import ciw
if not Path(ciw.__file__).resolve().is_relative_to(Path(ciw_root).resolve()):
    raise SystemExit("the SCR driver imported ciw from another location")
from ciw.lab import julia_worker as jw
from execution.commitments import COMPUTATION_TAG, OUTPUT_TAG, canonical_u32, commit_hex
from execution.dispatcher import SpecificationDispatcher
from execution.engine import ExecutionRefused, ExecutionResult
from execution.specification import ExecutionSpecification
from evidence.types import make_referent
from materials.candidates import make_action_candidate
from materials.specification import PREDICTED

plan = json.loads(sys.stdin.read())
runtime = jw.JuliaRuntime(Path(plan["executable"]), Path(plan["depot"]))
formulation = make_referent(natural_key=jw.OPERATION + " fixtures", kind="formulation")

def dispatch(session, step, program, configuration, input_payload):
    seen = {}

    def runner(spec):
        # The Julia worker below SCR's seam: SCR identities come from SCR's own commitment functions over the exact
        # bytes; a refused request never ran, a halted run has no output, a failed channel raises.
        try:
            occurrence = session.request(spec.program, spec.configuration, spec.input_payload, step.get("timeout"),
                                         step.get("fault"))
        except jw.WorkerFailure as failure:
            seen["failure"] = failure
            raise
        seen["occurrence"] = occurrence
        if occurrence.kind == "refused":
            raise ExecutionRefused(occurrence.payload.decode("utf-8", "replace"))
        common = dict(specification=spec, specification_identity=spec.identity(),
                      program_identity=spec.program_identity(), input_identity=spec.input_identity(),
                      engine_occurrence=occurrence.occurrence)
        if occurrence.kind == "halted":
            return ExecutionResult(status="halted", exit_code=plan["halted_exit_code"], output=None,
                                   output_identity=None, computation_identity=None,
                                   detail=occurrence.payload.decode("utf-8", "replace"), **common)
        output_identity = commit_hex(OUTPUT_TAG, [occurrence.payload])
        computation = commit_hex(COMPUTATION_TAG, [bytes.fromhex(common["program_identity"]),
                                                   bytes.fromhex(common["input_identity"]),
                                                   bytes.fromhex(output_identity), canonical_u32(0)])
        return ExecutionResult(status="completed", exit_code=0, output=occurrence.payload,
                               output_identity=output_identity, computation_identity=computation, detail=None,
                               **common)

    def interpret(candidate, result):
        decoded = jw.decode_output(result.output)
        return {"property": candidate.property, "output_schema": jw.OUTPUT_SCHEMA, "sample_count": decoded["count"],
                "t_last_s": decoded["t"][-1], "q_last_m": decoded["q"][-1], "v_last_m_s": decoded["v"][-1],
                "energy_last_J": decoded["energy"][-1]}

    spec = ExecutionSpecification(program=program, configuration=configuration, input_payload=input_payload)
    candidate = make_action_candidate("model_validation:unspecified", (spec.identity(),), formulation,
                                      "damped_oscillator_trajectory", PREDICTED, {"fixture": step["fixture"]})
    dispatcher = SpecificationDispatcher(spec_for=lambda chosen: spec, interpret=interpret,
                                         extracted_at=plan["extracted_at"], runner=runner)
    try:
        measurement = dispatcher.dispatch(candidate)
        scr = {"dispatch": "measurement", "record_locator": measurement.record_locator,
               "record_raw_content": measurement.record_raw_content, "extraction_method": measurement.extraction_method,
               "extracted_at": measurement.extracted_at, "content": dict(measurement.content)}
    except jw.WorkerFailure as failure:
        scr = {"dispatch": "raised", "exception": type(failure).__name__}
    except ExecutionRefused as refusal:
        scr = {"dispatch": "raised", "exception": type(refusal).__name__}
    except RuntimeError as halted:
        scr = {"dispatch": "raised", "exception": type(halted).__name__, "halted": "execution halted" in str(halted)}
    record = (jw.occurrence_record(seen["occurrence"]) if "occurrence" in seen
              else jw.failure_record(seen["failure"]) if "failure" in seen
              else {"outcome": "not_dispatched", "occurrence": None, "request_frame": None, "response_frame": None})
    record["scr"] = scr
    return record

print(json.dumps(jw.run_plan(runtime, plan["steps"], dispatch)))
'''
HALTED_EXIT_CODE = 3


def run(runtime: jw.JuliaRuntime, steps, scr_root: Path | None = None) -> dict:
    """Execute ``steps``: through SCR's dispatcher in an isolated interpreter when ``scr_root`` is bound, else directly."""
    if scr_root is None:
        return dict(jw.run_plan(runtime, steps), path="direct")
    import ciw

    ciw_root = Path(ciw.__file__).resolve().parent.parent
    payload = {"executable": str(runtime.executable), "depot": str(runtime.depot), "steps": steps,
               "extracted_at": EXTRACTED_AT, "halted_exit_code": HALTED_EXIT_CODE}
    with tempfile.TemporaryDirectory(prefix="ciw-lab-julia-scr-") as directory:
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", _SCR_DRIVER, str(Path(scr_root).resolve()),
                                    str(ciw_root)], input=json.dumps(payload), capture_output=True, text=True,
                                   timeout=900, cwd=directory)
    if completed.returncode:
        raise RuntimeError("SCR dispatch driver failed: " + (completed.stderr.strip().splitlines() or ["no output"])[-1])
    return dict(json.loads(completed.stdout), path="scr")


# ------------------------------------------------------------------- analysis
def scr_commit(tag: str, fields) -> str:
    """SCR's canonical commitment restated here: sha256 of u64-LE-length-prefixed tag, field count and fields."""
    raw = ("scout.execution." + tag + ".v1").encode("ascii")
    data = struct.pack("<Q", len(raw)) + raw + struct.pack("<Q", len(fields))
    return hashlib.sha256(data + b"".join(struct.pack("<Q", len(field)) + field for field in fields)).hexdigest()


def split_request(frame: bytes) -> tuple:
    """(program, configuration, input) from a request frame's payload."""
    payload, fields, at = frame[jw.HEADER.size:], [], 0
    for _ in range(3):
        (size,) = struct.unpack("<Q", payload[at:at + 8])
        fields.append(payload[at + 8:at + 8 + size])
        at += 8 + size
    if at != len(payload):
        raise ValueError("Request payload has trailing bytes")
    return tuple(fields)


def recompute_scr(record: dict) -> dict:
    """SCR's identities recomputed from the retained frames, compared with SCR's own record of the dispatch."""
    program, configuration, input_payload = split_request(bytes.fromhex(record["request_frame"]))
    output = bytes.fromhex(record["response_frame"])[jw.HEADER.size:]
    identities = {"specification": scr_commit("specification", [program, configuration, input_payload]),
                  "program": scr_commit("program", [program]), "input": scr_commit("input", [input_payload]),
                  "output_id": scr_commit("output", [output])}
    identities["computation"] = scr_commit("computation", [bytes.fromhex(identities["program"]),
                                                          bytes.fromhex(identities["input"]),
                                                          bytes.fromhex(identities["output_id"]), struct.pack("<I", 0)])
    lines = dict(line.split(" ", 1) for line in record["scr"]["record_raw_content"].splitlines()[1:] if " " in line)
    mismatched = sorted(key for key, value in identities.items() if lines.get(key) != value)
    if lines.get("exit_code") != "0" or lines.get("engine_occurrence") != str(record["occurrence"]):
        mismatched.append("occurrence_or_exit_code")
    return {"identities": identities, "mismatched": mismatched}


def errors(decoded: dict, expected: dict) -> dict:
    """Largest absolute error per channel and the largest threshold ratio |x - x_ref| / (abs + rel |x_ref|)."""
    worst, ratio = {}, 0.0
    for channel, threshold in ACCEPTANCE.items():
        got, ref = np.asarray(decoded[channel]), np.asarray(expected[channel])
        difference = np.abs(got - ref)
        worst[channel] = float(np.max(difference))
        ratio = max(ratio, float(np.max(difference / (threshold["abs"] + threshold["rel"] * np.abs(ref)))))
    return {"max_abs_error": worst, "max_threshold_ratio": ratio}


def undamped_metrics(decoded: dict, fixture: dict, expected: dict) -> dict:
    """Phase error of (q, v/omega_0) against the closed form and energy drift relative to the initial energy."""
    omega = fixture["omega_0"]
    julia = np.asarray(decoded["q"]) - 1j * np.asarray(decoded["v"]) / omega
    exact = expected["q"] - 1j * expected["v"] / omega
    initial = 0.5 * fixture["mass"] * (fixture["v0"] ** 2 + omega * omega * fixture["q0"] ** 2)
    drift = np.abs(np.asarray(decoded["energy"]) - initial) / initial
    return {"max_phase_error_rad": float(np.max(np.abs(np.angle(julia * np.conj(exact))))),
            "max_relative_energy_drift": float(np.max(drift)), "final_relative_energy_drift": float(drift[-1])}


def analyse(result: dict) -> dict:
    """Decode every completed exchange and compare it with the closed form; recompute SCR identities where SCR ran."""
    cases = fixtures()
    records = {record["label"]: record for record in result["records"]}
    decoded, comparisons, scr, undecodable = {}, {}, {}, {}
    for label, record in records.items():
        if record["op"] != "request" or record.get("outcome") != "completed":
            continue
        try:
            output = jw.decode_output(bytes.fromhex(record["response_frame"])[jw.HEADER.size:])
        except (ValueError, struct.error, UnicodeDecodeError) as exc:
            undecodable[label] = type(exc).__name__   # a completed frame without a valid output is no result
            continue
        fixture = cases[record["fixture"]]
        output["grid_exact"] = output["t"] == [float(t) for t in fixture["times"]]
        decoded[label] = output
        expected = reference(fixture, record["fixture"])
        comparisons[label] = dict(errors(output, expected), naccept=output["naccept"], nreject=output["nreject"],
                                  nf=output["nf"], samples=output["count"])
        if record["fixture"] == "undamped":
            comparisons[label].update(undamped_metrics(output, fixture, expected))
        if record.get("scr", {}).get("dispatch") == "measurement":
            scr[label] = recompute_scr(record)
    return {"records": records, "decoded": decoded, "comparisons": comparisons, "scr": scr,
            "undecodable": undecodable}


def replay_agreement(decoded: dict, labels=("A1", "A2", "A3", "A4")) -> dict:
    """Largest componentwise difference of the repeated occurrences from the first, and distinct output digests."""
    present = [label for label in labels if label in decoded]
    first = decoded[present[0]]
    largest = max((abs(a - b) for label in present[1:] for channel in UNITS
                   for a, b in zip(decoded[label][channel], first[channel])), default=0.0)
    return {"occurrences": present, "max_abs_difference": float(largest)}


def retained_frames(records: dict) -> dict:
    """The exact request and response frames of every request, payloads stored once per distinct byte string."""
    payloads, frames = {}, {}

    def keep(frame_hex):
        if frame_hex is None:
            return None
        raw = bytes.fromhex(frame_hex)
        digest = hashlib.sha256(raw[jw.HEADER.size:]).hexdigest()
        payloads[digest] = raw[jw.HEADER.size:].hex()
        return {"header": raw[:jw.HEADER.size].hex(), "payload_sha256": digest}

    for label, record in records.items():
        if record["op"] == "request":
            frames[label] = {"outcome": record["outcome"], "occurrence": record.get("occurrence"),
                             "request": keep(record.get("request_frame")), "response": keep(record.get("response_frame"))}
    return {"schema": "ciw.lab-julia-frames.v1", "frames": frames, "payloads": dict(sorted(payloads.items()))}


def restore_frames(data: dict) -> dict:
    """Frames rebuilt from :func:`retained_frames`, each payload checked against its digest."""
    def rebuild(part):
        if part is None:
            return None
        payload = bytes.fromhex(data["payloads"][part["payload_sha256"]])
        if hashlib.sha256(payload).hexdigest() != part["payload_sha256"]:
            raise ValueError("Retained Julia frame payload does not match its digest")
        return (bytes.fromhex(part["header"]) + payload).hex()

    return {label: {"outcome": entry["outcome"], "occurrence": entry["occurrence"],
                    "request_frame": rebuild(entry["request"]), "response_frame": rebuild(entry["response"])}
            for label, entry in data["frames"].items()}


def sanitized_handshake(payload_hex: str | None) -> dict | None:
    """A handshake for retention: host paths (project, depot) are replaced by their role."""
    if payload_hex is None:
        return None
    fields = jw.decode_lines(bytes.fromhex(payload_hex), jw.HANDSHAKE_SCHEMA)
    fields.update(project="<bound project>", depot="<bound julia-depot>")
    packages = fields.pop("packages", "")
    fields["packages"] = sorted(item.split("=")[0] + "@" + item.split("=")[2] for item in packages.split(";") if item)
    return fields


def producer_identity(handshake: dict) -> dict:
    """The independent checks' producer: SciML's solver packages and Julia, with their exact versions."""
    versions = dict(item.split("@") for item in handshake["packages"])
    revision = ", ".join(f"{name} {versions[name]}" for name in PRODUCER_PACKAGES if name in versions)
    return {"implementation": PRODUCER,
            "revision": f"{revision}; Julia {handshake['julia_version']} ({handshake['julia_commit'][:12]})",
            "manifest_sha256": handshake["manifest_sha256"]}


# ------------------------------------------------------------------- findings
EXACT = {"abs": 0, "rel": 0}
EXACT_U = {"kind": "roundoff", "value": 0.0, "basis": "exact count, byte or refusal-code comparison"}
# Regression tolerance of every solver-derived value (errors, ratios, step counts). OpenBLAS kernels do not
# enter the Julia computation; Julia's own code generation does: between native (FMA) and generic x86-64 (no
# FMA) code on these fixtures the reported values moved by at most 0.8 % relative (1.7e-14 absolute, at reltol
# 1e-12 where the errors approach rounding), the outputs by at most 5.4e-12 and the step counts not at all
# (docs/lab/IMPLEMENTATION_TARGETS.md, Julia).
SOLVER_TOLERANCE = {"abs": 1e-13, "rel": 0.05}
REFERENCE_U = {"kind": "reference_error", "value": 1e-13,
               "basis": "binary64 rounding of the closed-form reference (values of order 1 to 10) is below 1e-13; "
                        "the measured differences are the solver's global error"}
CLAIMS = {
    "scr": "Julia environment pinned and exercised through the CIW to SCR boundary",
    "pin": "Julia provider pin procedure",
    "default": "Julia Tsit5 agrees with CIW's closed-form oscillator on all 768 q, v and energy samples of the "
               "default fixture",
    "grid": "Every completed Julia output covers exactly the requested sample times",
    "mixed": "Julia Tsit5 agrees with CIW's closed form for a mixed initial state with a negative component",
    "phase": "Julia Tsit5 keeps the undamped oscillator's phase within 1e-6 rad of the closed form",
    "drift": "Julia Tsit5 keeps the undamped oscillator's energy within 1e-6 of its initial value over [0, 12) s",
    "ladder": "Tightening abstol = reltol from 1e-6 to 1e-12 lowers the global error and raises the accepted steps "
              "and function evaluations",
    "bound": "The requested tolerance does not bound the global error of v on the default fixture",
    "replay": "Repeated and interleaved occurrences of the default fixture agree within 1e-12 on one worker and on a "
              "restarted worker",
    "bytes": "Repeated occurrences of the default fixture return byte-identical output on this host",
    "platform": "Julia worker outputs agree between Linux and Windows x86-64 within the declared tolerances",
    "host_refusals": "The host refuses invalid numbers, grids and oversized frames before dispatch",
    "worker_refusals": "The worker refuses invalid requests it receives without running them and keeps serving",
    "halt": "A halted solve returns no output and no SCR measurement",
    "halt_direct": "A halted solve returns no output",
    "channel": "Oversized frames, timeouts, crashes and a wrong environment end the worker session without a result",
    "mock": "Truncated, malformed, oversized and mismatched responses end the session without a result "
            "(protocol mock)",
    "replayed_handshake": "A process replaying a recorded handshake passes the worker handshake comparison",
    "offline": "Retained frames decode without Julia and reproduce the recorded SCR commitments",
    "offline_direct": "Retained frames decode without Julia",
}


def _check(reference, observed, tolerance, comparison="abs_le", kind="exact_arithmetic") -> dict:
    from .evidence import holds

    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


def _refusal(reference, expected, observed) -> dict:
    observed = str(observed) if observed else "none"
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def host_refusals() -> list:
    """Requests the host refuses before dispatch: (reference, expected code, observed code)."""
    times = fixtures()["ciw-default"]["times"]
    cases = [("boolean omega_0", "not_a_number", lambda: jw.encode_input(True, 0.15, 1.0, 1.0, 0.0, 12.0, times)),
             ("NaN initial velocity", "nonfinite_number",
              lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, math.nan, 12.0, times)),
             ("gamma above omega_0 / 2", "out_of_bounds",
              lambda: jw.encode_input(5.0, 2.6, 1.0, 1.0, 0.0, 12.0, times)),
             ("duration above 12 s", "out_of_bounds", lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, 0.0, 12.5, times)),
             ("repeated sample time", "grid_not_increasing",
              lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, 0.0, 12.0, [0.0, 0.5, 0.5, 1.0])),
             ("grid not starting at the initial time 0", "grid_origin",
              lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, 0.0, 12.0, [0.25, 0.5])),
             ("sample at the excluded endpoint", "grid_outside_interval",
              lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, 0.0, 12.0, [0.0, 6.0, 12.0])),
             ("4097 samples", "sample_count_out_of_bounds",
              lambda: jw.encode_input(5.0, 0.15, 1.0, 1.0, 0.0, 12.0, [k / 400.0 for k in range(4097)])),
             ("reltol below 1e-14", "out_of_bounds", lambda: jw.encode_configuration(1e-10, 1e-15, 1e-3, 12.0, 100)),
             ("boolean maxiters", "not_an_integer", lambda: jw.encode_configuration(1e-10, 1e-10, 1e-3, 12.0, True)),
             ("request frame over 65536 bytes", "frame_too_large",
              lambda: jw.frame("request", 1, jw.encode_request(jw.OPERATION_DESCRIPTOR, b"", bytes(70000))))]
    observed = []
    for reference, expected, action in cases:
        try:
            action()
            code = "none"
        except jw.RequestRefusal as refusal:
            code = refusal.code
        observed.append((reference, expected, code))
    return observed


def _outcome(records: dict, label: str) -> str:
    return str(records.get(label, {}).get("outcome") or "missing")


def _measurements(records: dict, labels) -> int:
    return sum(records.get(label, {}).get("scr", {}).get("dispatch") == "measurement" for label in labels)


def _worker_code(record: dict) -> str:
    """The refusal code or halt return code a worker response carries."""
    payload = bytes.fromhex(record["response_frame"])[jw.HEADER.size:]
    schema = jw.REFUSAL_SCHEMA if record["outcome"] == "refused" else jw.HALT_SCHEMA
    fields = jw.decode_lines(payload, schema)
    return fields.get("code") or fields.get("retcode") or "none"


def offline_restore(frames: dict, recorded_scr: dict) -> dict:
    """Decode the retained frames (read back from the artifact) and recompute SCR's commitments; no Julia runs."""
    completed, decoded, mismatched = 0, 0, 0
    for label, record in frames.items():
        if record["outcome"] != "completed":
            continue
        completed += 1
        try:
            jw.decode_output(bytes.fromhex(record["response_frame"])[jw.HEADER.size:])
        except (ValueError, struct.error, UnicodeDecodeError):
            continue
        decoded += 1
        if label in recorded_scr:
            mismatched += len(recompute_scr(dict(record, scr=recorded_scr[label]))["mismatched"])
    return {"completed": completed, "decoded": decoded, "scr_compared": len(recorded_scr),
            "scr_mismatches": mismatched}


def acceptance_findings(result: dict, analysis: dict, offline: dict, scr: dict, identity_fields) -> list:
    """The acceptance set's findings from one executed plan (``scr``: why SCR did or did not host it, its basis)."""
    from .. import __version__
    from .evidence import finding
    from .runner import source_digest

    records, comparisons, decoded = analysis["records"], analysis["comparisons"], analysis["decoded"]
    starts = [record for record in records.values()
              if record["op"] == "start" and not record["session"].startswith("mock-")]
    worker1 = records["start worker-1"]
    handshake = sanitized_handshake(worker1.get("handshake"))
    accepted = [record for record in starts if record["outcome"] == "accepted"]
    mismatches = sum(len(record["comparison"]["mismatches"]) for record in accepted)
    problems = sum(len(record["comparison"]["package_problems"]) for record in accepted)
    sources = sum(len(record["comparison"]["source_problems"]) for record in accepted)
    checker = {"implementation": "ciw.adapters.oscillator.closed_form", "revision": f"ciw {__version__}",
               "source_sha256": source_digest("src/ciw/adapters/oscillator.py")}
    producer = producer_identity(handshake) if handshake else None
    through_scr = result.get("path") == "scr"
    completed = [label for label, record in records.items() if record.get("outcome") == "completed"]
    comparison = worker1["comparison"] or {}
    findings = [finding(
        CLAIMS["pin"], "provenance",
        {"identity_fields": list(identity_fields), "compared_fields": len(comparison.get("compared", [])),
         "packages_verified": comparison.get("packages_verified", 0),
         "package_trees_verified": comparison.get("package_trees_verified", 0), "sessions_accepted": len(accepted)},
        {"checks": [_check("handshake fields, bound executable, runtime tree and system image differing from the "
                           "expected identity and the pinned archive in accepted sessions", mismatches, 0),
                    _check("declared loaded packages whose version or install directory differs from the committed "
                           "Manifest", problems, 0),
                    _check("Manifest package directories in the bound depot whose git tree, recomputed by the host, "
                           "differs from the Manifest's git-tree-sha1", sources, 0),
                    _check("pinned worker sessions accepted (worker-1, worker-2, worker-3)", len(accepted), 3, "ge")],
         "notes": "Each session's handshake is compared field by field, and the host reads the bound executable, "
                  "runtime tree and package directories itself, before any request; the wrong-environment session "
                  "(threads 2 where 1 is declared) is refused and not counted. The depot's precompiled package "
                  "images are trusted from provisioning (neither the host nor Julia 1.10 checks their content)."},
        uncertainty=EXACT_U, tolerance=EXACT)]

    # ---------------------------------------------------------- the SCR boundary
    if through_scr:
        mismatched = sum(len(entry["mismatched"]) for entry in analysis["scr"].values())
        admitted = _measurements(records, completed)
        method = next((records[label]["scr"]["extraction_method"] for label in completed), None)
        basis = dict(scr["basis"], checks=[
            _check("SCR program, input, specification, output and computation identities recomputed from the "
                   "retained frames that differ from SCR's record", mismatched, 0),
            _check("completed exchanges without an SCR DispatchedMeasurement", len(completed) - admitted, 0),
            _check("completed exchanges dispatched through SpecificationDispatcher", admitted, 1, "ge"),
            _check("handshake differences in the dispatching sessions", mismatches, 0)])
        basis["notes"] = {"extraction_method": method,
                          "extraction_method_origin": "fixed by SCR's dispatcher at the pin for every runner; it "
                                                      "does not name Julia or the worker"}
        findings.append(finding(CLAIMS["scr"], "computational_pipeline",
                                {"completed_dispatches": admitted, "identity_differences": mismatched},
                                basis, uncertainty=EXACT_U, tolerance=EXACT))
    else:
        findings.append(finding(CLAIMS["scr"], "computational_pipeline", None,
                                {"notes": "The Julia worker ran and was compared with CIW, but not behind SCR: "
                                          + scr["reason"]}, expected_not_established=True))

    # ---------------------------------------------------------- numerical acceptance
    def independent(reference_text, observed, tolerance):
        return {"independent_check": dict(_check(reference_text, observed, tolerance, "le", "analytic"),
                                          producer=producer, checker=checker)}

    def missing(key, label):
        observed = "undecodable_output" if label in analysis["undecodable"] else _outcome(records, label)
        return finding(CLAIMS[key], "numerical", None,
                       {"checks": [_refusal(f"{label} completed", "completed", observed)]})

    for key, label, samples in (("default", "A1", 768), ("mixed", "B", 600)):
        entry = comparisons.get(label)
        if entry is None or producer is None:
            findings.append(missing(key, label))
            continue
        findings.append(finding(
            CLAIMS[key], "numerical",
            {"max_abs_error": entry["max_abs_error"], "max_threshold_ratio": entry["max_threshold_ratio"],
             "samples": entry["samples"]},
            independent(f"largest |x - x_ref| / (abs + rel |x_ref|) over q, v and energy at {samples} samples, "
                        f"declared thresholds {json.dumps(ACCEPTANCE, sort_keys=True)}",
                        entry["max_threshold_ratio"], 1.0),
            uncertainty=REFERENCE_U, tolerance=SOLVER_TOLERANCE))
    exact = sum(decoded[label]["grid_exact"] for label in decoded)
    findings.append(finding(
        CLAIMS["grid"], "numerical", {"completed_outputs": len(decoded), "exact_grids": exact},
        {"checks": [_check("completed outputs whose time vector differs bitwise from the requested times",
                           len(decoded) - exact, 0),
                    _check("completed outputs decoded", len(decoded), 1, "ge")]},
        uncertainty=EXACT_U, tolerance=EXACT))
    undamped = comparisons.get("undamped")
    if undamped is not None and producer is not None:
        findings.append(finding(
            CLAIMS["phase"], "numerical", {"max_phase_error_rad": undamped["max_phase_error_rad"],
                                           "max_threshold_ratio": undamped["max_threshold_ratio"]},
            independent("largest |arg((q - i v/omega_0) conj(q_ref - i v_ref/omega_0))| over 768 samples",
                        undamped["max_phase_error_rad"], PHASE_THRESHOLD),
            uncertainty=REFERENCE_U, tolerance=SOLVER_TOLERANCE))
        findings.append(finding(
            CLAIMS["drift"], "numerical", {"max_relative_energy_drift": undamped["max_relative_energy_drift"],
                                           "final_relative_energy_drift": undamped["final_relative_energy_drift"]},
            {"checks": [_check("largest |E(t_k) - E(0)| / E(0) over the 768 samples of the undamped fixture",
                               undamped["max_relative_energy_drift"], ENERGY_DRIFT_THRESHOLD, "le", "invariant")]},
            uncertainty={"kind": "roundoff", "value": 1e-15, "basis": "binary64 rounding of energies of order 1 J"},
            tolerance=SOLVER_TOLERANCE))
    else:
        findings += [missing("phase", "undamped"), missing("drift", "undamped")]
    ladder = [(tolerance, "A1" if tolerance == DEFAULT_CONFIGURATION["reltol"] else f"reltol-{tolerance:.0e}")
              for tolerance in TOLERANCE_LADDER]
    if all(label in comparisons for _, label in ladder):
        rows = [comparisons[label] for _, label in ladder]
        global_errors = {channel: [row["max_abs_error"][channel] for row in rows] for channel in UNITS}
        steps = {name: [row[name] for row in rows] for name in ("naccept", "nreject", "nf")}
        worse = max(later - earlier for series in global_errors.values() for earlier, later in zip(series, series[1:]))
        fewer = max(earlier - later for earlier, later in zip(steps["naccept"], steps["naccept"][1:]))
        cheaper = max(earlier - later for earlier, later in zip(steps["nf"], steps["nf"][1:]))
        findings.append(finding(
            CLAIMS["ladder"], "numerical",
            {"reltol": list(TOLERANCE_LADDER), "max_abs_error": global_errors, **steps},
            {"checks": [_check("largest increase of a channel's global error from one rung to the next tighter",
                               worse, 0.0, "signed_le", "self_convergence"),
                        _check("largest decrease of accepted steps from one rung to the next tighter", fewer, -1.0,
                               "signed_le", "self_convergence"),
                        _check("largest decrease of function evaluations from one rung to the next tighter",
                               cheaper, -1.0, "signed_le", "self_convergence")]},
            uncertainty=REFERENCE_U, tolerance=SOLVER_TOLERANCE))
        oracle = reference(fixtures()["ciw-default"], "ciw-default")
        ratios = {channel: [] for channel in ("q", "v")}
        for tolerance, label in ladder:
            for channel in ratios:
                difference = np.abs(np.asarray(decoded[label][channel]) - oracle[channel])
                ratios[channel].append(float(np.max(difference / (tolerance + tolerance * np.abs(oracle[channel])))))
        findings.append(finding(
            CLAIMS["bound"], "numerical", {"reltol": list(TOLERANCE_LADDER), "error_to_tolerance": ratios},
            {"checks": [_check("smallest over the rungs of the largest |v - v_ref| / (abstol + reltol |v_ref|)",
                               min(ratios["v"]), 1.0, "ge", "analytic")],
             "notes": "The requested tolerance is applied to each step's local error estimate, in an RMS norm over "
                      "[q, v] scaled by the step's endpoints (max(|u_prev|, |u|)); it does not control the values "
                      "interpolated at the saved times or the error accumulated over the steps, so a ratio above 1 "
                      "is expected. The RMS norm alone lets one component's scaled estimate reach sqrt(2); these "
                      "data do not separate the causes."},
            counterexample={"statement": "The requested tolerance abstol + reltol*|x| bounds the global error of each "
                                         "state component at the saved times",
                            "witness": {"reltol": TOLERANCE_LADDER[0], "channel": "v",
                                        "error_to_tolerance_ratio": ratios["v"][0]}},
            uncertainty=REFERENCE_U, tolerance=SOLVER_TOLERANCE))
    else:
        rung = next(label for _, label in ladder if label not in comparisons)
        findings += [missing("ladder", rung), missing("bound", rung)]

    # ---------------------------------------------------------- repeated and interleaved runs
    repeated = [label for label in ("A1", "A2", "A3", "A4") if label in decoded]
    agreement = replay_agreement(decoded, repeated) if repeated else {"max_abs_difference": 1.0}
    pairs = {(records[label]["session_id"], records[label]["occurrence"]) for label in repeated}
    sessions = {records[label]["session_id"] for label in repeated}
    findings.append(finding(
        CLAIMS["replay"], "numerical",
        {"occurrences": len(repeated), "sessions": len(sessions), "max_abs_difference": agreement["max_abs_difference"]},
        {"checks": [_check(f"largest |x - x(A1)| over q, v and energy of {', '.join(repeated) or 'none'}",
                           agreement["max_abs_difference"], REPLAY_AGREEMENT),
                    _check("completed occurrences of A (A, B, A, A after a halt and refusals; A on a restarted "
                           "worker)", len(repeated), 4, "ge"),
                    _check("repeated (session, occurrence) pairs among them", len(repeated) - len(pairs), 0),
                    _check("worker sessions among them", len(sessions), 2, "ge")],
         "notes": "Each occurrence builds fresh problem and solver state; B, a halt and six refusals ran between A1 "
                  "and A3 on the same worker."},
        uncertainty=EXACT_U, tolerance={"abs": REPLAY_AGREEMENT, "rel": 0}))
    outputs = {bytes.fromhex(records[label]["response_frame"])[jw.HEADER.size:] for label in repeated}
    findings.append(finding(
        CLAIMS["bytes"], "computational_pipeline", {"occurrences": len(repeated), "distinct_outputs": len(outputs)},
        {"checks": [_check("distinct output byte strings among the repeated occurrences, minus one",
                           len(outputs) - 1, 0), _check("repeated occurrences compared", len(repeated), 2, "ge")],
         "notes": "Same host, binary and depot; this says nothing about another CPU or platform."},
        uncertainty=EXACT_U, tolerance=EXACT))
    findings.append(finding(
        CLAIMS["platform"], "numerical", None,
        {"notes": "Only this Linux x86-64 host ran the worker. The Windows x86-64 archive is pinned "
                  "(src/ciw/lab/julia/julia-runtime.json) but was not provisioned or run, so no Windows output "
                  "exists to compare."}, expected_not_established=True))

    # ---------------------------------------------------------- refusals and failures
    host = host_refusals()
    findings.append(finding(
        CLAIMS["host_refusals"], "computational_pipeline",
        {"refused": sum(code == expected for _, expected, code in host)},
        {"checks": [_refusal(reference_text, expected, code) for reference_text, expected, code in host]},
        uncertainty=EXACT_U, tolerance=EXACT))
    worker_checks = [_refusal(f"{label}: worker refusal code", expected,
                              _worker_code(records[label]) if _outcome(records, label) == "refused"
                              else _outcome(records, label)) for label, expected in WORKER_REFUSALS.items()]
    refused = sum(check["passed"] for check in worker_checks)
    worker_checks.append(_refusal("A3 after the refusals", "completed", _outcome(records, "A3")))
    halted = records.get("halted-maxiters", {})
    halt_checks = [_refusal("maxiters = 10: outcome", "halted", _outcome(records, "halted-maxiters")),
                   _refusal("maxiters = 10: solver return code", "MaxIters",
                            _worker_code(halted) if halted.get("outcome") == "halted" else "none")]
    channel = [("oversized-frame", "frame_too_large"), ("after-session-end", "session_ended"),
               ("stall-timeout", "response_timeout"), ("crash", "worker_exited"),
               ("start worker-4 (threads 2, expected 1)", "environment_mismatch")]
    channel_checks = [_refusal(f"{label}: session end", expected, _outcome(records, label))
                      for label, expected in channel]
    ended = sum(check["passed"] for check in channel_checks)
    if through_scr:
        worker_checks.append(_check("refused requests with an SCR measurement", _measurements(records, WORKER_REFUSALS),
                                    0))
        halt_checks.append(_check("SCR measurements from the halted run", _measurements(records, ["halted-maxiters"]),
                                  0))
        channel_checks.append(_check("SCR measurements from failed exchanges",
                                     _measurements(records, [label for label, _ in channel]), 0))
    findings.append(finding(CLAIMS["worker_refusals"], "computational_pipeline", {"refused": refused},
                            {"checks": worker_checks}, uncertainty=EXACT_U, tolerance=EXACT))
    # Claims that name SCR are made only where SCR ran; the direct path's wording leaves SCR out.
    findings.append(finding(CLAIMS["halt" if through_scr else "halt_direct"], "computational_pipeline",
                            {"halted_without_output": int(halt_checks[0]["passed"] and halt_checks[1]["passed"])},
                            {"checks": halt_checks}, uncertainty=EXACT_U, tolerance=EXACT))
    findings.append(finding(CLAIMS["channel"], "computational_pipeline", {"sessions_ended": ended},
                            {"checks": channel_checks}, uncertainty=EXACT_U, tolerance=EXACT))
    mock_checks = [_refusal(f"mock {mode}: session end", expected, _outcome(records, f"mock-{mode}"))
                   for mode, expected in MOCK_FAILURES.items()]
    findings.append(finding(CLAIMS["mock"], "computational_pipeline",
                            {"sessions_ended": sum(check["passed"] for check in mock_checks)},
                            {"checks": mock_checks,
                             "notes": "A Python stand-in speaking the frame protocol tests the host's failure "
                                      "handling only and supplies no numerical evidence."},
                            uncertainty=EXACT_U, tolerance=EXACT))
    replayed = sum(records.get(f"start mock-{mode}", {}).get("outcome") == "accepted" for mode in MOCK_FAILURES)
    findings.append(finding(
        CLAIMS["replayed_handshake"], "provenance", {"mock_sessions_passing": replayed},
        {"checks": [_check("protocol-mock sessions whose replayed handshake matched the expected identity",
                           replayed, 1, "ge")],
         "notes": "The handshake is what the process declares; its digests are unkeyed and nothing attests it."},
        counterexample={"statement": "A worker whose handshake matches the expected runtime identity runs the pinned "
                                     "Julia environment",
                        "witness": {"process": "Python protocol mock", "handshake": "replayed from worker-1",
                                    "mock_sessions_passing": replayed}},
        uncertainty=EXACT_U, tolerance=EXACT))
    offline_checks = [_check("completed outputs in the retained artifact that do not decode",
                             offline["completed"] - offline["decoded"], 0),
                      _check("completed outputs decoded from the retained artifact", offline["decoded"], 1, "ge")]
    if through_scr:
        offline_checks += [_check("SCR records compared with commitments recomputed from the retained artifact",
                                  offline["scr_compared"], 1, "ge"),
                           _check("SCR commitments recomputed from the retained artifact that differ from SCR's "
                                  "record", offline["scr_mismatches"], 0)]
    findings.append(finding(CLAIMS["offline" if through_scr else "offline_direct"], "computational_pipeline",
                            offline if through_scr else {key: offline[key] for key in ("completed", "decoded")},
                            {"checks": offline_checks}, uncertainty=EXACT_U, tolerance=EXACT))
    return findings
