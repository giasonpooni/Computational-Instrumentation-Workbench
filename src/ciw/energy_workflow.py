"""Offline analysis of an exact retained energy log; never a measurement runner.

A fresh execution or replay recomputes derived quantities from the same sealed
counter/output evidence. It never imports a CUDA/NVML worker, samples hardware,
or runs variational iterations. Historical capture provenance is not attested.

The lifecycle (exact source retention, fresh occurrences, save/reopen
validation without a provider, replay receipts) is the shared reference
lifecycle in ``reference_workflow``; this module supplies the energy-specific
source check, analysis, runtime identity, request and policy.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import platform
import re

import numpy as np

from . import energy_records
from . import reference_workflow as base
from .adapters.subprocess import _json
from .declared_workload import RESULT_SCHEMA, VERIFY_SCHEMA
from .telemetry import _keys

KIND = "energy-accuracy"
SCHEMA = "ciw.energy-accuracy-session.v1"
OPERATION = "ciw.energy-accuracy.v1"
ROLE = "energy"
PROFILE = "ciw.energy-accuracy-analysis.v1"
RUNTIME_SCHEMA = "ciw.energy-analysis-runtime.v1"
EXECUTION_SCOPE = "cpu_offline_retained_log_analysis"
REQUEST_SCHEMA = "ciw.energy-analysis-request.v1"
SOURCE_LIMIT = 4 * 1024 * 1024
MAX_BYTES = 8 * 1024 * 1024
POLICY = {"operation": "offline_analysis_of_retained_energy_log", "physical_measurement": "not_performed",
          "replay": "fresh_analysis_of_same_retained_measurement", "hardware_provenance": "not_authenticated",
          "state_admission": "not_performed"}
AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed",
             "physical_measurement": "not_performed_by_analysis", "hardware_provenance": "not_authenticated"}
METHOD = "fresh_analysis_of_same_retained_measurement"
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
# The energy runtime identity predates the shared reference runtime and stays
# flat: its algorithm fields sit beside the schema and scope rather than under
# an ``algorithm`` key, so retained workspaces keep reopening.
FIXED_RUNTIME_KEYS = ("schema", "profile", "execution_scope", "source_normalization")


@lru_cache(maxsize=1)
def _algorithm_identity():
    files = ("energy_records.py", "free_energy_math.py", "reference_workflow.py", "energy_workflow.py")
    return base.algorithm_identity(PROFILE, [Path(__file__).with_name(name) for name in files],
                                   python_version=platform.python_version(), numpy_version=np.__version__)


def analysis_identity():
    return {"schema": RUNTIME_SCHEMA, "execution_scope": EXECUTION_SCOPE, **_algorithm_identity()}


def _check_runtime(runtime):
    expected = analysis_identity()
    _keys(runtime, set(expected))
    if any(runtime[key] != expected[key] for key in FIXED_RUNTIME_KEYS):
        raise ValueError("Unsupported retained energy analysis runtime")
    if type(runtime["code_sha256"]) is not str or not base.CODE_DIGEST.fullmatch(runtime["code_sha256"]):
        raise ValueError("Invalid retained analysis implementation identity")
    for key in ("python_version", "numpy_version"):
        if type(runtime[key]) is not str or not VERSION.fullmatch(runtime[key]):
            raise ValueError("Invalid retained analysis dependency identity")


def _request(source, evidence):
    return {"schema": REQUEST_SCHEMA, "evidence_id": evidence,
            "log_digest": source["log_digest"], "measurement_run_id": source["run_id"],
            "analysis_profile": PROFILE}


class EnergyAccuracyWorkflow(base.ReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    SOURCE_SCHEMA = energy_records.SCHEMA
    result_schema = RESULT_SCHEMA
    verify_schema = VERIFY_SCHEMA
    operation = OPERATION
    role = ROLE
    label = "energy analysis"
    method = METHOD
    ROLES = frozenset()
    MAX_BYTES = MAX_BYTES
    AUTHORITY = AUTHORITY

    def _source(self, raw):
        if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Energy analysis source must contain 1..4194304 exact retained bytes")
        source = _json(raw)
        energy_records.validate_log(source)
        return source

    def _native_data(self, source):
        return energy_records.analyze(source)

    def _runtime_identity(self):
        return analysis_identity()

    def _configuration(self, source):
        return deepcopy(POLICY)

    def _experiment_id(self, source):
        return source["run_id"]

    def _request(self, source, evidence_id):
        return _request(source, evidence_id)

    def _check_runtime(self, runtime):
        _check_runtime(runtime)

    def _check_data(self, result, source):
        # Statuses, reasons, counts, decimal counter strings and structure must
        # match exactly.  The accuracy metrics compare each retained batch with
        # the Gaussian reference, whose conditioning runs through the host's
        # linear-algebra kernels; the resulting roundoff-level errors differ
        # between CPUs without changing any classification, so numbers are
        # compared with the shared binary64 tolerance.
        base.close_data(result["data"], self._native_data(source))


def _verification(bundle, reproduced):
    return EnergyAccuracyWorkflow()._verification(bundle, reproduced)
