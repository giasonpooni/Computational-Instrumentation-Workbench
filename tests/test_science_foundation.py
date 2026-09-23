"""Units, frames/clocks, ledger integrity and Ed25519: the shared scientific state model."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ciw.science import ed25519
from ciw.science._common import Refusal, covariance
from ciw.science.frames import FrameRegistry, rotation_from_quaternion, skew
from ciw.science.ledger import (BODY_SCHEMAS, MIGRATIONS, Ledger, register_body_schema, register_migration,
                                source_evidence)
from ciw.science.units import (Quantity, conversion_factor, convert, convert_covariance, dimension_text, parse_unit)

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("expression, scale, dimension", [
    ("m/s^2", 1.0, "m*s^-2"), ("kg*m^2/s^2", 1.0, "m^2*kg*s^-2"), ("m/sqrt(Hz)", 1.0, "m*s^1/2"),
    ("mm", 1e-3, "m"), ("um", 1e-6, "m"), ("μm", 1e-6, "m"), ("N.m", 1.0, "m^2*kg*s^-2"), ("m^(1/2)", 1.0, "m^1/2"),
    ("kPa", 1e3, "m^-1*kg*s^-2"), ("mrad", 1e-3, "1"), ("deg", math.pi / 180, "1"), ("px", 1.0, "px"),
    ("min", 60.0, "s"), ("cd", 1.0, "cd"), ("1/s", 1.0, "s^-1"),
])
def test_unit_parsing(expression, scale, dimension):
    unit = parse_unit(expression)
    assert unit.scale == pytest.approx(scale) and dimension_text(unit.dimension) == dimension


@pytest.mark.parametrize("expression, code", [("furlong", "unknown_unit"), ("m/(s", "unknown_unit"),
                                              ("degC/s", "offset_unit_compound"), ("", "malformed_record")])
def test_unit_refusals(expression, code):
    with pytest.raises(Refusal) as caught:
        parse_unit(expression)
    assert caught.value.code == code


def test_conversions_offsets_and_pseudo_dimensions():
    assert convert(20.0, "degC", "K") == pytest.approx(293.15)
    assert convert(1.0, "m", "mm") == pytest.approx(1000.0)
    assert conversion_factor("degC", "K") == 1.0  # differences carry no offset
    for source, target in (("px", "m"), ("tick", "s"), ("m", "s")):
        with pytest.raises(Refusal) as caught:
            convert(1.0, source, target)
        assert caught.value.code == "dimension_mismatch"


def test_covariance_scales_with_the_unit_basis():
    cov = np.array([[1e-6, 2e-7], [2e-7, 4e-6]])
    converted = convert_covariance(cov, ["m", "m"], ["mm", "mm"])
    assert np.allclose(converted, cov * 1e6)
    mixed = convert_covariance(cov, ["m", "rad"], ["mm", "mrad"])
    assert np.allclose(mixed, cov * 1e6)
    with pytest.raises(Refusal):
        convert_covariance(cov, ["m"], ["mm"])


def test_quantity_arithmetic_respects_frames_and_offsets():
    a = Quantity(1.0, "m", "coupon")
    assert (a + Quantity(500.0, "mm", "coupon")).value == pytest.approx(1.5)
    with pytest.raises(Refusal) as caught:
        a + Quantity(1.0, "m", "camera")
    assert caught.value.code == "frame_mismatch"
    with pytest.raises(Refusal) as caught:
        Quantity(20.0, "degC") + Quantity(1.0, "K")
    assert caught.value.code == "offset_unit_arithmetic"
    with pytest.raises(Refusal):
        Quantity.from_json({"value": 1.0, "unit": "m", "extra": 1})


def test_covariance_admission_never_repairs():
    for bad, code in (([[1, 0.5], [0.4, 1]], "malformed_covariance"), ([[1, 2], [2, 1]], "malformed_covariance"),
                      ([[1, float("nan")], [float("nan"), 1]], "malformed_record")):
        with pytest.raises(Refusal) as caught:
            covariance(bad, "c", 2)
        assert caught.value.code == code


# ------------------------------------------------------------------ frames and clocks
def cell():
    return FrameRegistry.from_json(json.loads((ROOT / "examples/science/frames/coupon-cell.json").read_text()))


def _expm(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-15:
        return np.eye(3) + skew(phi)
    k = skew(phi / angle)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_transform_chain_and_inverse_round_trip():
    registry = cell()
    resolved = registry.resolve("coupon", "camera", 1000.0, "cell-ptp")
    assert [link["transform_id"] for link in resolved.chain] == ["coupon-in-machine", "camera-in-machine"]
    assert [link["direction"] for link in resolved.chain] == ["forward", "inverse"]
    back = registry.resolve("camera", "coupon", 1000.0, "cell-ptp")
    point = np.array([0.05, 0.0, 0.02])
    mapped, _ = resolved.apply(point)
    assert np.allclose(back.apply(mapped)[0], point, atol=1e-15)


def test_composed_covariance_matches_monte_carlo():
    registry = cell()
    resolved = registry.resolve("tool", "world", 1000.0, "cell-ptp")
    point = np.array([0.01, -0.02, 0.03])
    _, linear = resolved.apply(point)
    rng = np.random.default_rng(0)
    samples = []
    transforms = [registry.transforms["tool-in-machine"], registry.transforms["machine-in-world"]]
    for _ in range(20000):
        x = point
        for item in transforms:
            xi = rng.multivariate_normal(np.zeros(6), item.covariance)
            rotation = _expm(xi[:3])
            x = rotation @ (item.rotation @ x + item.translation) + xi[3:]
        samples.append(x)
    assert np.allclose(np.cov(np.array(samples).T), linear, rtol=0.08, atol=1e-10)


def test_stale_transform_is_refused_not_extrapolated():
    with pytest.raises(Refusal) as caught:
        cell().resolve("coupon", "camera", 8000.0, "cell-ptp")
    assert caught.value.code == "transform_stale"
    assert any(item["transform_id"] == "coupon-in-machine" for item in caught.value.detail["stale"])


def test_clock_alignment_and_clock_confusion():
    registry = cell()
    value, variance, used = registry.align_time(500.0, "camera-clock", "cell-ptp")
    assert value == pytest.approx(12.5 + 1.00002 * 500.0) and variance > 4e-8 and used == ["camera-to-ptp-2026-09"]
    back, _, _ = registry.align_time(value, "cell-ptp", "camera-clock")
    assert back == pytest.approx(500.0)
    with pytest.raises(Refusal) as caught:
        registry.align_time(1.0, "sim", "cell-ptp")
    assert caught.value.code == "clock_unmapped"
    with pytest.raises(Refusal) as caught:
        registry.resolve("coupon", "camera", 1000.0, None)
    assert caught.value.code == "clock_unspecified"
    with pytest.raises(Refusal) as caught:
        registry.add_clock_mapping({"mapping_id": "bad", "source": "sim", "target": "cell-ptp", "offset_s": 0, "rate": 1,
                                    "covariance": [[0, 0], [0, 0]], "valid_from": 0, "valid_until": 1,
                                    "calibration": {"calibration_id": "x", "version": "1"}})
    assert caught.value.code == "clock_domain_mismatch"


def test_ambiguous_chains_and_unknown_frames_are_refused():
    registry = cell()
    registry.add_transform({"transform_id": "camera-in-world", "source": "camera", "target": "world",
                            "rotation": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}, "translation": [0, 0, 0], "unit": "m",
                            "covariance": np.eye(6).tolist(), "clock": "cell-ptp", "estimated_at": 0, "valid_from": 0,
                            "valid_until": 1e5, "calibration": {"calibration_id": "x", "version": "1"}})
    registry.add_transform({"transform_id": "coupon-in-world", "source": "coupon", "target": "world",
                            "rotation": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}, "translation": [0, 0, 0], "unit": "m",
                            "covariance": np.eye(6).tolist(), "clock": "cell-ptp", "estimated_at": 0, "valid_from": 0,
                            "valid_until": 1e5, "calibration": {"calibration_id": "x", "version": "1"}})
    with pytest.raises(Refusal) as caught:
        registry.resolve("coupon", "camera", 1000.0, "cell-ptp")
    assert caught.value.code == "frame_ambiguous"
    via = registry.resolve("coupon", "camera", 1000.0, "cell-ptp", via=["machine"])
    assert [link["transform_id"] for link in via.chain] == ["coupon-in-machine", "camera-in-machine"]
    with pytest.raises(Refusal) as caught:
        registry.resolve("coupon", "nowhere", 1000.0, "cell-ptp")
    assert caught.value.code == "unknown_frame"


def test_rotations_are_validated_not_renormalized():
    registry = cell()
    base = {"transform_id": "t", "source": "tool", "target": "camera", "translation": [0, 0, 0], "unit": "m",
            "covariance": np.zeros((6, 6)).tolist(), "clock": "cell-ptp", "estimated_at": 0, "valid_from": 0,
            "valid_until": 1, "calibration": {"calibration_id": "x", "version": "1"}}
    with pytest.raises(Refusal) as caught:
        registry.add_transform(dict(base, rotation={"quaternion_wxyz": [1.0, 0.1, 0.0, 0.0]}))
    assert caught.value.code == "malformed_rotation"
    with pytest.raises(Refusal) as caught:
        registry.add_transform(dict(base, rotation={"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}))
    assert caught.value.code == "malformed_rotation"
    assert np.allclose(rotation_from_quaternion(np.array([1.0, 0, 0, 0])), np.eye(3))


# ------------------------------------------------------------------ ledger
def test_ledger_append_reopen_and_chain(tmp_path):
    ledger = Ledger.create(tmp_path / "l")
    source = source_evidence(ledger, "spec", b'{"a": 1}', "application/json")
    runtime = ledger.append("ciw.science.runtime-identity.v1", {"runtime": {"python": "3"}}, refs=[source["entry_id"]])
    reopened = Ledger.open(tmp_path / "l")
    assert len(reopened) == 2 and reopened.head() == runtime["entry_id"]
    assert reopened.get_blob(source["body"]["blob"]) == b'{"a": 1}'
    assert reopened.closure([runtime["entry_id"]])[0]["entry_id"] == source["entry_id"]
    assert reopened.verify().ok


@pytest.mark.parametrize("mutation", ["edit", "delete", "reorder", "truncate"])
def test_ledger_tampering_is_detected(tmp_path, mutation):
    ledger = Ledger.create(tmp_path / "l")
    for index in range(4):
        ledger.append("ciw.science.runtime-identity.v1", {"runtime": {"index": index}})
    path = tmp_path / "l" / "ledger.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "edit":
        lines[1] = lines[1].replace(b'"index":1', b'"index":7')
    elif mutation == "delete":
        del lines[1]
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines[-1] = lines[-1][:-5]
    path.write_bytes(b"".join(lines))
    with pytest.raises(Refusal) as caught:
        Ledger.open(tmp_path / "l")
    assert caught.value.code == "ledger_tampered"


def test_ledger_refuses_dangling_refs_unknown_schemas_and_traversal(tmp_path):
    ledger = Ledger.create(tmp_path / "l")
    with pytest.raises(Refusal) as caught:
        ledger.append("ciw.science.runtime-identity.v1", {"runtime": {}}, refs=["sha256:" + "0" * 64])
    assert caught.value.code == "dangling_reference"
    with pytest.raises(Refusal) as caught:
        ledger.append("ciw.science.nonexistent.v1", {})
    assert caught.value.code == "unknown_schema"
    with pytest.raises(Refusal) as caught:
        ledger.append("ciw.science.execution.v1", {"job_id": "j", "solver_id": "s", "status": "maybe"})
    assert caught.value.code == "malformed_entry"
    for identity in ("sha256:../../etc/passwd", "sha256:" + "A" * 64, "md5:" + "0" * 32):
        with pytest.raises(Refusal) as caught:
            ledger.get_blob(identity)
        assert caught.value.code == "malformed_identity"
    with pytest.raises(Refusal):
        Ledger.create(tmp_path / "l")


def test_blob_corruption_is_detected(tmp_path):
    ledger = Ledger.create(tmp_path / "l")
    entry = source_evidence(ledger, "x", b"exact bytes", "text/plain")
    blob = entry["body"]["blob"][7:]
    (tmp_path / "l" / "blobs" / blob[:2] / blob).write_bytes(b"exact bytez")
    with pytest.raises(Refusal) as caught:
        Ledger.open(tmp_path / "l")
    assert caught.value.code == "ledger_tampered"


def test_schema_migration_is_a_view_that_never_rewrites_bytes(tmp_path, monkeypatch):
    monkeypatch.setitem(BODY_SCHEMAS, "ciw.science.test-telemetry.v2", ("telemetry", {"channels", "units"}, None))
    monkeypatch.setattr("ciw.science.ledger.MIGRATIONS", dict(MIGRATIONS))
    register_migration("ciw.science.telemetry.v1", "ciw.science.test-telemetry.v2",
                       lambda body: {**body, "units": {name: "W" for name in body["channels"]}})
    ledger = Ledger.create(tmp_path / "l")
    entry = ledger.append("ciw.science.telemetry.v1", {"channels": ["power"]})
    before = (tmp_path / "l" / "ledger.jsonl").read_bytes()
    view = ledger.view(entry)
    assert view["body_schema"] == "ciw.science.test-telemetry.v2" and view["body"]["units"] == {"power": "W"}
    assert view["stored_body_schema"] == "ciw.science.telemetry.v1"
    assert (tmp_path / "l" / "ledger.jsonl").read_bytes() == before
    with pytest.raises(Refusal):
        register_body_schema("ciw.science.telemetry.v1", "telemetry", {"channels"})


# ------------------------------------------------------------------ Ed25519 (RFC 8032 section 7.1)
RFC8032 = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
]


@pytest.mark.parametrize("secret, public, message, signature", RFC8032)
def test_ed25519_rfc8032_vectors(secret, public, message, signature):
    secret, public, message, signature = map(bytes.fromhex, (secret, public, message, signature))
    assert ed25519.public_key(secret) == public
    assert ed25519.sign(secret, message) == signature
    assert ed25519.verify(public, message, signature)
    assert not ed25519.verify(public, message + b"!", signature)
    assert not ed25519.verify(public, message, signature[:-1] + bytes([signature[-1] ^ 1]))
    assert not ed25519.verify(public[:31], message, signature)
