"""Every check of the shared covariance artifact validator refuses what it names.

An AST mutation probe dropped each ``if ...: raise`` of ``core.covariance``
in turn; 37 of 39 survived. Each case starts from a valid two-by-two
artifact, changes exactly one thing, reseals the content identity unless
the identity itself is under test, and expects the refusal that names it.
"""
from copy import deepcopy

import pytest

from ciw.core import covariance
from ciw.core.covariance import covariance_identity, create_covariance_artifact, validate_covariance_artifact

DIGEST = "sha256:" + "0" * 64
BASE = create_covariance_artifact(
    matrix=[[1.0, 0.1], [0.1, 2.0]], quantity_ids=["a", "b"], units=["m", "m"], frame="frame:x",
    reference_values=[0.0, 0.0], method="declared", basis={"kind": "observation", "id": "obs:1"},
    provenance={"provider": "test", "source_evidence_ids": [DIGEST], "source_covariance_ids": []},
    assumptions=["none"])


def assign(*path, value):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def no_quantities(artifact):
    for key in ("quantity_ids", "units", "reference_values", "matrix"):
        artifact[key] = []


RESEALED = {
    "extra field": (assign("extra", value=1), "exactly the v1 contract fields"),
    "schema renamed": (assign("schema", value="other"), "Unsupported covariance artifact schema"),
    "no quantities": (no_quantities, "unique ordered quantities"),
    "repeated quantity": (assign("quantity_ids", value=["a", "a"]), "unique ordered quantities"),
    "reference values given as text": (assign("reference_values", value="ab"), "reference_values must match"),
    "one reference value": (assign("reference_values", value=[0.0]), "reference_values must match"),
    "basis given as a list of its names": (assign("basis", value=["kind", "id"]), "basis must declare kind and id"),
    "basis without an id": (assign("basis", value={"kind": "observation"}), "basis must declare kind and id"),
    "basis kind not text": (assign("basis", "kind", value=["observation"]), "Unsupported covariance basis kind"),
    "basis kind unknown": (assign("basis", "kind", value="other"), "Unsupported covariance basis kind"),
    "provenance given as a list of its names": (assign("provenance", value=["provider", "source_evidence_ids", "source_covariance_ids"]), "provenance requires provider"),
    "provenance without covariance sources": (assign("provenance", value={"provider": "test", "source_evidence_ids": [DIGEST]}), "provenance requires provider"),
    "provenance with an unknown field": (assign("provenance", "extra", value=1), "provenance requires provider"),
    "provenance metadata not an object": (assign("provenance", "metadata", value="x"), r"provenance\.metadata must be a JSON object"),
    "method not text": (assign("method", value=5), "method must be a nonempty string"),
    "method blank": (assign("method", value="  "), "method must be a nonempty string"),
    "reference value given as text": (assign("reference_values", 0, value="0"), "reference_values entry must be a finite JSON number, not a boolean"),
    "units given as text": (assign("units", value="ab"), "units must be an array with 2 entries"),
    "one unit": (assign("units", value=["m"]), "units must be an array with 2 entries"),
    "repeated evidence identity": (assign("provenance", "source_evidence_ids", value=[DIGEST, DIGEST]), "content identities must be unique"),
    "evidence identity malformed": (assign("provenance", "source_evidence_ids", value=["abc"]), "must be sha256 content identities"),
    "matrix given as text": (assign("matrix", value="ab"), "matrix shape must match"),
    "matrix with a third row": (assign("matrix", value=[[1.0, 0.1], [0.1, 2.0], [0.0, 0.0]]), "matrix shape must match"),
    "matrix row given as text": (assign("matrix", 1, value="ab"), "matrix must be square"),
    "matrix row with a third entry": (assign("matrix", 1, value=[0.1, 2.0, 0.0]), "matrix must be square"),
    "negative variance": (assign("matrix", value=[[-1.0, 0.0], [0.0, 1.0]]), "variances must be nonnegative"),
    "zero variance beside a covariance": (assign("matrix", value=[[0.0, 0.1], [0.1, 1.0]]), "Zero variance requires an exactly zero covariance row and column"),
    "covariance that overflows its correlation": (assign("matrix", value=[[1e-300, 1e300], [1e300, 1e-300]]), "nonfinite normalized correlation"),
    "indefinite matrix": (assign("matrix", value=[[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    "asymmetric matrix": (assign("matrix", value=[[1.0, 0.1], [0.2, 2.0]]), "symmetric in correlation coordinates"),
}

UNSEALED = {
    "identity not text": (assign("covariance_id", value=5), "content identity mismatch"),
    "identity of other content": (assign("covariance_id", value=DIGEST), "content identity mismatch"),
}


def test_the_base_artifact_is_valid():
    validate_covariance_artifact(deepcopy(BASE), expected_quantity_ids=["a", "b"], expected_units=["m", "m"], expected_frame="frame:x")


@pytest.mark.parametrize("mutate, message", RESEALED.values(), ids=list(RESEALED))
def test_one_resealed_change_is_refused_by_its_own_check(mutate, message):
    artifact = deepcopy(BASE)
    mutate(artifact)
    artifact["covariance_id"] = covariance_identity(artifact)
    with pytest.raises(ValueError, match=message):
        validate_covariance_artifact(artifact)


@pytest.mark.parametrize("mutate, message", UNSEALED.values(), ids=list(UNSEALED))
def test_a_wrong_content_identity_is_refused(mutate, message):
    artifact = deepcopy(BASE)
    mutate(artifact)
    with pytest.raises(ValueError, match=message):
        validate_covariance_artifact(artifact)


def test_an_artifact_must_be_an_object():
    with pytest.raises(ValueError, match="exactly the v1 contract fields"):
        validate_covariance_artifact("x")
    with pytest.raises(ValueError, match="Covariance artifact must be an object"):
        covariance_identity("x")


def test_helper_checks_refuse_what_they_name():
    with pytest.raises(ValueError, match="x must be a finite JSON number"):
        covariance._number(float("inf"), "x")
    with pytest.raises(ValueError, match="x keys must be strings"):
        covariance._json_tree({1: 2}, "x")
