"""Record adapter for ``run.v1`` projections of retained Julia oscillator results.

The projection shares the oscillator state frame, channels and render layout
so the existing viewport, sample lookup and analysis operations apply to it.
Its provenance must name a simulation origin and the retained bundle; this
adapter never executes Julia and never turns the projection into evidence of
a measurement.
"""
from __future__ import annotations

from .oscillator import _FRAME, _UNITS, compute_spectrum, compute_statistics, validate_run
from .protocol import AdapterRefusal, InstrumentManifest

INSTRUMENT = "julia-oscillator-trajectory.v1"
_PROVENANCE = ("bundle_digest", "result_id", "execution_id", "specification_identity", "computation_identity",
               "verification_id", "worker_runtime_digest")


class JuliaOscillatorRunAdapter:
    manifest = InstrumentManifest(
        instrument_id=INSTRUMENT, role="simulation_projection",
        inputs=("ciw.julia-oscillator-session.v1",), outputs=("run.v1", "statistics.v1", "spectrum.periodogram.v1"),
        units=dict(_UNITS), frames=(_FRAME,),
        sampling={"kind": "uniform", "time_unit": "s", "endpoint": "excluded"},
        normalization={"statistics": "population", "spectrum": "one-sided density; periodic Hann"},
        supported_operations=("statistics.v1", "spectrum.periodogram.v1"),
        determinism={"kind": "same_runtime_declared_tolerance", "dtype": "float64", "origin": "simulation"},
        tolerance_policy={"sampling_rtol": 1e-8, "sampling_atol_spacing_factor": 1e-10,
                          "oracle": "analytic-damped-oscillator.v1 comparison retained in the source bundle",
                          "replay": "declared normalized tolerance; byte identity recorded separately"},
        calibration_requirements={"required": False, "reason": "numerical simulation projection, not a measurement"},
    )

    def validate_run(self, run: dict) -> None:
        if run.get("instrument") != INSTRUMENT:
            raise ValueError("Run instrument does not match the Julia oscillator projection adapter")
        validate_run(run)
        provenance = run["metadata"]["provenance"]
        if provenance.get("origin") != "simulation" or provenance.get("generator") != "ciw.julia_oscillator.project_run":
            raise ValueError("Julia oscillator projections must declare their simulation origin and generator")
        for key in _PROVENANCE:
            if not isinstance(provenance.get(key), str) or not provenance[key]:
                raise ValueError(f"Julia oscillator projection provenance lacks {key}")

    def execute(self, operation_id: str, run: dict, parameters: dict) -> dict:
        self.validate_run(run)
        if parameters.keys() != {"channel", "interval_s"}:
            raise ValueError("Oscillator operations require channel and interval_s only")
        if operation_id == "statistics.v1":
            return compute_statistics(run, parameters["channel"], parameters["interval_s"])
        if operation_id == "spectrum.periodogram.v1":
            return compute_spectrum(run, parameters["channel"], parameters["interval_s"])
        raise AdapterRefusal("unsupported_operation", f"Julia oscillator projection does not support {operation_id}")
