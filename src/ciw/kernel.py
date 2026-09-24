"""The frozen CIW kernel surface.

The kernel is the part that should stay expensive: identities, admission,
persistence, reopen/replay, inspection and binding. Scientific work extends
the workbench through declared pipelines and records inside the workspace,
not through new wire verbs, envelopes, workspace formats or source kinds.

Changing any set below is a kernel change: bump ``KERNEL_VERSION``, update
docs/KERNEL.md and justify it in review. ``tests/test_kernel_freeze.py``
checks the dispatcher, the workbench registry and the workspace writer against
these declarations, and the session refuses verbs that are not declared here.
"""
from __future__ import annotations

KERNEL_VERSION = "ciw.kernel.v1"

# Verbs that every client may rely on. New scientific operations are payloads
# of ``operation.execute``; new records are retrieved through the existing
# ``*.get``/``*.list`` and ``experiment.inspect`` verbs.
KERNEL_VERBS = frozenset({
    "session.get", "run.get",
    "source.add", "source.get", "source.list",
    "operation.list", "operation.execute", "execution.list",
    "bundle.list", "bundle.get", "bundle.replay",
    "result.list", "result.get",
    "experiment.inspect",
    "selection.update", "sample.get",
    "workspace.save",
})

# The original oscillator analyses keep their flat result format.
LEGACY_VERBS = frozenset({"analysis.stats", "analysis.spectrum"})

# Read-only projections that grew around individual kinds. They remain for
# existing clients, accept no additions, and are to be subsumed by
# ``experiment.inspect`` and project-graph inspection.
PROJECTION_VERBS = frozenset({
    "spatial.list", "spatial.inspect", "fusion.list",
    "instrument.list", "instrument.inspect",
    "candidate.list", "candidate.get",
})

VERBS = KERNEL_VERBS | LEGACY_VERBS | PROJECTION_VERBS

ENVELOPES = frozenset({
    "ciw.execution.v1", "ciw.operation-result.v1",
    "ciw.workbench-source.v1", "covariance-artifact.v1",
})

# Workspace format 3 is the last format minted because a family of records was
# added. Later kinds are records inside the workspace.
WORKSPACE_FORMATS = (1, 2, 3)
FINAL_WORKSPACE_FORMAT = 3

# Source kinds registered in the shared session when the kernel was frozen.
# A new kind is refused until a project-graph operation exists or an
# investigation is delivered end to end with a held-out measurement; until
# then new work composes these kinds.
FROZEN_KINDS = frozenset({
    "calibrated-observable", "identified-design", "telemetry", "calibrated-window",
    "schematic-assessment", "numerical-heat", "proved-heat", "schematic-companions",
    "bim-quantity", "acquired-dataset", "acquired-calibrated-window", "residual-monitor",
    "measurement-chain", "geometric-circle", "identified-stability", "flat-torus-reference",
    "curved-path-transfer", "covariance-geometry", "mesh-path", "translation-flow",
    "variational-free-energy", "energy-accuracy", "instrument-exchange", "thermal-observer",
    "machine-manifest",
})
SOURCE_ONLY_KINDS = frozenset({"geographic-context"})

# Identities that must never be merged.
DISTINCT_IDENTITIES = ("evidence", "operation", "execution", "result", "verification")


def describe() -> dict:
    """Machine-readable kernel declaration for clients and documentation."""
    return {"kernel_version": KERNEL_VERSION,
            "verbs": {"kernel": sorted(KERNEL_VERBS), "legacy": sorted(LEGACY_VERBS),
                      "projection": sorted(PROJECTION_VERBS)},
            "envelopes": sorted(ENVELOPES), "workspace_formats": list(WORKSPACE_FORMATS),
            "final_workspace_format": FINAL_WORKSPACE_FORMAT,
            "frozen_kinds": sorted(FROZEN_KINDS), "source_only_kinds": sorted(SOURCE_ONLY_KINDS),
            "distinct_identities": list(DISTINCT_IDENTITIES)}
