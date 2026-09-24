"""The linear-algebra backend label is informational, bounded and never fails."""

from ciw import numerical_backend
from ciw import reference_workflow as base


def test_label_is_bounded_and_names_openblas_when_readable():
    label = numerical_backend.linear_algebra_backend()
    assert isinstance(label, str) and 1 <= len(label) <= numerical_backend.LIMIT
    assert label == numerical_backend.UNKNOWN or label.startswith("OpenBLAS")
    assert numerical_backend.linear_algebra_backend() == label


def test_unreadable_backend_reports_unknown_without_raising(monkeypatch):
    monkeypatch.setattr(numerical_backend, "_candidate_libraries", lambda: ["/nonexistent/libopenblas.so"])
    assert numerical_backend._describe() == numerical_backend.UNKNOWN
    monkeypatch.setattr(numerical_backend, "_describe", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    numerical_backend.linear_algebra_backend.cache_clear()
    try:
        assert numerical_backend.linear_algebra_backend() == numerical_backend.UNKNOWN
    finally:
        numerical_backend.linear_algebra_backend.cache_clear()


def test_label_is_not_part_of_any_reference_identity():
    assert "numerical_backend" not in base.__file__
    for module in ("thermal_workflow", "machine_workflow", "uncertainty_validation", "energy_workflow", "project_workflow"):
        source = (base.__file__.replace("reference_workflow.py", module + ".py"))
        assert "numerical_backend" not in open(source, encoding="utf-8").read(), module
