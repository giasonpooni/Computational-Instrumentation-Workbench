"""Closed vocabularies shared by every scientific record.

A term outside these sets is refused rather than interpreted. Adding a term is
a schema change: it needs a version, a validator and, where it affects retained
records, a registered migration.
"""
from __future__ import annotations

# What a sensor or derived product actually measures. Two observations can only
# be compared when they declare the same observable.
OBSERVABLES = frozenset({
    "intrinsic_distance",      # length of a path within a surface (geodesic arclength)
    "camera_chord",            # straight-line 3-D distance between points, as a camera reconstructs it
    "image_residual",          # reprojection difference in image samples (px)
    "encoder_displacement",    # axis displacement reported by a machine encoder
    "tracker_position",        # 3-D position reported by an external tracker
    "imu_orientation",         # orientation integrated or reported by an IMU
    "filtered_state",          # an estimator output, never a raw observation
    "reconstructed_geometry",  # geometry fitted to observations
})

# The status of an assertion, from README "Claims, assistance and execution authority".
CLAIM_CLASSES = frozenset({"measured", "estimated", "predicted", "computed", "verified", "interpretation",
                           "authorized"})

# How an observation came to exist. Only "physical" can support a measured claim.
ACQUISITION_KINDS = frozenset({"physical", "synthetic", "not_acquired", "planned"})

# Declared solver capabilities. A requirement outside this set is refused.
CAPABILITIES = frozenset({
    "constant_curvature", "arbitrary_metric", "embedded_surfaces", "chart_transitions", "mesh_surfaces",
    "uncertainty_propagation", "near_conjugate", "closed_form_reference", "conjugate_detection",
    "winding_classes", "asynchronous_streams", "correlated_noise", "clock_alignment", "outlier_gating",
})

EXECUTION_MODES = ("explore", "observe", "prepare", "operate")

AGENT_ROLES = frozenset({
    "schematic-retrieval", "provider-resolution", "geometry-reasoning", "experiment-design",
    "numerical-method", "sensor-fusion", "provenance-auditor", "adversarial-test", "documentation",
    "hardware-interface", "safety-reviewer",
})
