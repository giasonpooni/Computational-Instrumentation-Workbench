"""Offline analysis of an exact retained energy log; never a measurement runner.

A fresh execution or replay recomputes derived quantities from the same sealed
counter/output evidence. It never imports a CUDA/NVML worker, samples hardware,
or runs variational iterations. Historical capture provenance is not attested.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import platform
import re

import numpy as np

from . import energy_records
from .adapters.subprocess import _json
from .core.canonical import canonical, digest, exact_keys
from .pipelines import runner as _runner
from .pipelines.runner import PipelineRunner, RecordProfile, RESULT_SCHEMA, VERIFY_SCHEMA, check_receipts

KIND = "energy-accuracy"
OPERATION = "ciw.energy-accuracy.v1"
SOURCE_LIMIT = 4 * 1024 * 1024
MAX_BYTES = 8 * 1024 * 1024
POLICY = {"operation": "offline_analysis_of_retained_energy_log", "physical_measurement": "not_performed",
          "replay": "fresh_analysis_of_same_retained_measurement", "hardware_provenance": "not_authenticated",
          "state_admission": "not_performed"}
AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed",
             "physical_measurement": "not_performed_by_analysis", "hardware_provenance": "not_authenticated"}
METHOD = "fresh_analysis_of_same_retained_measurement"


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Retained energy analysis binding differs")


def analysis_identity():
    files = ("energy_records.py", "free_energy_math.py", "energy_workflow.py", "pipelines/runner.py")
    code = b"\0".join(name.encode() + b"\0" + Path(__file__).parent.joinpath(name).read_text(encoding="utf-8").encode()
                       for name in files)
    return {"schema": "ciw.energy-analysis-runtime.v1", "profile": "ciw.energy-accuracy-analysis.v1",
            "execution_scope": "cpu_offline_retained_log_analysis", "code_sha256": sha256(code).hexdigest(),
            "source_normalization": "utf8_lf", "python_version": platform.python_version(), "numpy_version": np.__version__}


def _check_runtime(runtime):
    exact_keys(runtime, {"schema", "profile", "execution_scope", "code_sha256", "source_normalization", "python_version", "numpy_version"})
    fixed = {"schema": "ciw.energy-analysis-runtime.v1", "profile": "ciw.energy-accuracy-analysis.v1",
             "execution_scope": "cpu_offline_retained_log_analysis", "source_normalization": "utf8_lf"}
    if any(runtime[key] != value for key, value in fixed.items()):
        raise ValueError("Unsupported retained energy analysis runtime")
    if type(runtime["code_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", runtime["code_sha256"]):
        raise ValueError("Invalid retained analysis implementation identity")
    for key in ("python_version", "numpy_version"):
        if type(runtime[key]) is not str or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime[key]):
            raise ValueError("Invalid retained analysis dependency identity")


def _request(source, evidence):
    return {"schema": "ciw.energy-analysis-request.v1", "evidence_id": evidence,
            "log_digest": source["log_digest"], "measurement_run_id": source["run_id"],
            "analysis_profile": "ciw.energy-accuracy-analysis.v1"}


class EnergyAccuracyWorkflow(PipelineRunner):
    """Offline reanalysis of one retained energy log, sealed, reproduced and replayed by the shared runner."""

    MAX_BYTES = MAX_BYTES
    LABEL = "Energy analysis"
    PROFILE = RecordProfile(RESULT_SCHEMA, VERIFY_SCHEMA, METHOD, AUTHORITY)

    def identity_claims(self, bundle):
        """Reanalysis may reuse a log; one capture occurrence cannot be rebound to changed evidence."""
        data = bundle["steps"][0]["result"]["data"]
        occurrence = bundle["source"]["experiment_id"]
        if occurrence == data["log_digest"]:
            raise ValueError("Retained identity collision")
        return {occurrence: ("retained_energy_log_occurrence", {"log_digest": data["log_digest"], "origin": data["origin"]}),
                data["log_digest"]: ("retained_energy_log", {"run_id": occurrence, "origin": data["origin"]})}

    def __init__(self):
        super().__init__(KIND, {"role": "energy"})
        self.ROLES = frozenset()
        self.SOURCE_SCHEMA = energy_records.SCHEMA

    def parse_source(self, raw):
        if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Energy analysis source must contain 1..4194304 exact retained bytes")
        source = _json(raw)
        energy_records.validate_log(source)
        return source

    def experiment_id(self, source):
        return source["run_id"]

    def configuration(self, source):
        return POLICY

    def step_request(self, source, evidence_id):
        return _request(source, evidence_id)

    def check_data(self, source, data):
        _same(data, energy_records.analyze(source))

    @staticmethod
    def _runtime_projection(runtime):
        return runtime

    def _adapters(self, repositories, expected=None):
        if type(repositories) is not dict or repositories:
            raise ValueError("Offline energy analysis takes no repository or hardware bindings")
        runtime = analysis_identity()
        if expected is not None:
            _same(expected, {"energy": runtime})
        return None, runtime, None

    def _step(self, source, evidence, bound):
        _same(analysis_identity(), bound[1])
        data = energy_records.analyze(source)
        _same(analysis_identity(), bound[1])
        return _runner.seal_step(self.role, self.operation, self.step_request(source, evidence), [evidence], data,
                                 profile=self.PROFILE)

    def _check_envelope(self, bundle):
        value = super()._check_envelope(bundle)
        if len(bundle["created_at"]) > 128:
            raise ValueError("Missing bounded analysis creation metadata")
        return value

    def _check_runtimes(self, runtimes):
        exact_keys(runtimes, {"energy"})
        _check_runtime(runtimes["energy"])

    def _check_receipts(self, bundle):
        # A replay reproduces under the identical analysis runtime, so its
        # receipt names exactly this bundle's runtimes.
        check_receipts(bundle, self.kind, self.PROFILE)
        for receipt in bundle.get("replay_receipts", []):
            if receipt["verification"]["runtime_digest"] != digest(bundle["runtimes"]):
                raise ValueError("Invalid retained energy replay receipt")
