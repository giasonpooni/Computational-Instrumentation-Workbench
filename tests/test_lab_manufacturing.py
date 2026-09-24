from copy import deepcopy
import hashlib
import importlib.util
import json
import math

import numpy as np
import pytest

from ciw.lab import manufacturing as mfg
from ciw.lab import manufacturing_geometry as geo
from ciw.lab import manufacturing_metrology as met
from ciw.lab import manufacturing_records as rec
from ciw.lab import runner
from ciw import __version__
from ciw.lab.evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, finding
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report

TASK_IDS = [f"T{n}" for n in range(126, 142)]


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    """Run T126-T141 once in queue order with one shared context (memoized studies)."""
    directory = tmp_path_factory.mktemp("lab-manufacturing")
    implementations = section_implementations("manufacturing")
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    ctx = runner.Context(directory)
    reports = {task_id: validate_report(runner.run_task(queue[task_id], implementations[task_id], ctx, {}))
               for task_id in TASK_IDS}
    return directory, reports


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert matches, f"no finding starting with {prefix!r} in {report['task_id']}"
    return matches[0]


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _gap_at(row, radius):
    """Chord-geodesic gap of a cylinder marker pair at another radius (same angles and heights)."""
    return math.hypot(radius * row["dphi_rad"], row["dz_mm"]) - math.hypot(2 * radius * math.sin(row["dphi_rad"] / 2),
                                                                         row["dz_mm"])


def _measurement_record(data):
    """A retention record of kind measurement that is not the schema fixture (synthetic bytes, never retained)."""
    fixture, _ = mfg._schema_fixture()
    record = deepcopy(fixture)
    record.update(record_kind="measurement", instrument=dict(fixture["instrument"], serial="SN-1"),
                  raw=[{"name": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value),
                        "media_type": "application/octet-stream"} for name, value in data.items()])
    record["calibration"] = dict(fixture["calibration"], sha256="c" * 64)
    return record


@pytest.mark.lab_task("T126", "T127", "T128", "T129", "T130", "T131", "T132", "T133", "T134", "T135", "T136", "T137",
                      "T138", "T139", "T140", "T141")
def test_every_task_is_registered_and_reports_honestly(section):
    _, reports = section
    assert set(section_implementations("manufacturing")) == set(TASK_IDS)
    for task_id, report in reports.items():
        # T138 has no measurement to compare and T139 none to retain: both are partial until hardware exists.
        expected = "partial" if task_id in ("T138", "T139") else "completed"
        assert report["state"] == expected, (task_id, report["experiment"])
        assert report["physical_validation_status"]["status"] == "not_established"
        assert not report.get("tests_failed")
        for record in report["findings"]:
            if record["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS:
                assert record["evidence_status"] == "not_established"
            if record["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and not record.get("expected_not_established"):
                assert "regression_tolerance" in record, record["claim"]
                assert record["uncertainty"] is not None, record["claim"]
            assert "|" not in record["claim"], record["claim"]  # claims are table cells in the rendered report
        assert any(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS for f in report["findings"]), task_id
        # Every section task passes its findings through the acceptance-language screen.
        assert rec.screen_acceptance_language(report["findings"])
        assert hasattr(section_implementations("manufacturing")[task_id].run, "__wrapped__")
        for name in ("hypothesis", "mathematical_model", "experiment", "numerical_result", "recommended_next_task"):
            assert isinstance(report[name], str) and report[name].strip()
        # Next steps name forward work, not a queue task that has already run in the same run.
        assert not report["recommended_next_task"].startswith("T"), (task_id, report["recommended_next_task"])
        if task_id in ("T126", "T127", "T128", "T138", "T139"):
            # The measurement route is the capture path, and its physical label waits on a deferred trust anchor;
            # the closed route (execute on hardware, retain through T139, compare with the pair comparator) is gone.
            step = report["recommended_next_task"]
            assert "--capture" in step and "trust anchor" in step and "deferred research question" in step.lower(), task_id
            assert "compare_pair_distances" not in step and "on hardware, retain it through a T139" not in step, task_id
    # The customer-demand boundary item of the section's robotic use cases is recorded, never established.
    demand = [f for f in reports["T141"]["findings"] if f["domain"] == "customer_demand"]
    assert len(demand) == 1 and demand[0]["evidence_status"] == "not_established" and demand[0]["basis"] == {}


@pytest.mark.lab_task("T126")
def test_flat_plate_protocol_is_a_zero_curvature_control(section):
    directory, reports = section
    report = reports["T126"]
    gap = _finding(report, "Flat-plate control: chord and geodesic")
    assert gap["evidence_status"] == "numerically_verified" and gap["value"] <= 1e-9
    phi = _finding(report, "Flat-plate Jacobi transfer")
    assert np.allclose(phi["value"], [[1.0, 240.0], [0.0, 1.0]], atol=1e-9)
    assert _labels(report)["Measured marker chords on the physical plate equal the predicted geodesic distances "
                           "within instrument uncertainty"] == "not_established"
    protocol = rec.validate_protocol(json.loads(
        (directory / "artifacts" / "T126" / "protocol-flat-plate.json").read_text(encoding="utf-8")))
    assert protocol["hardware_measured"] == {"status": "not_acquired", "records": []}
    assert protocol["production_acceptance"] == "outside_system"
    heading = next(q for q in protocol["predicted_quantities"] if q["id"] == "P3")
    assert heading["value"][-1] == pytest.approx(1.2, abs=1e-12)
    assert heading["evidence_status"] == phi["evidence_status"]
    # The geodesic distance is solved for: Newton starts 0.1 rad off the chord direction and converges onto it.
    rows = mfg.plate_study()["pairs"]
    assert min(r["shooting_iterations"] for r in rows) >= 3
    assert max(r["heading_minus_chord_direction_rad"] for r in rows) < 1e-12
    shot = mfg.shoot_geodesic(geo.PLATE, [0.0, 0.0], [30.0, 40.0], 0.0, 100.0)
    assert shot["arclength_mm"] == pytest.approx(50.0, abs=1e-9)
    assert shot["heading_rad"] == pytest.approx(math.atan2(40.0, 30.0), abs=1e-12)


@pytest.mark.lab_task("T126", "T127", "T128", "T129")
def test_protocols_refuse_filled_slots_and_decisions(section):
    directory, reports = section
    for task_id, name in (("T126", "protocol-flat-plate.json"), ("T127", "protocol-rolled-cylinder.json"),
                          ("T128", "protocol-domed-coupon.json"), ("T129", "protocol-surface-scan.json")):
        protocol = json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))
        matrix = rec.protocol_refusal_matrix(protocol)
        assert all(case["observed"] == case["expected"] for case in matrix.values()), matrix
        validated = [f for f in reports[task_id]["findings"] if f["claim"].endswith("refuses malformed variants")]
        # Tape protocols add the layout mutations (two tapes in one laying, a tape left on the specimen, a marker
        # under a tape, a repeat that lays a tape) to the seventeen every protocol has.
        assert validated[0]["value"] == len(matrix) == (17 if task_id == "T129" else 21), task_id
        assert validated[0]["evidence_status"] == "numerically_verified"
        assert all("unsteered" in path["realization"] for path in protocol["paths"] if "realization" in path)
        assert matrix["step_names_undeclared_jig"]["observed"] == "undeclared_device"
    forged = dict(protocol, hardware_measured={"status": "acquired", "records": [{"acquisition": {
        "device": "camera:1", "raw_sha256": "not-a-digest", "acquired_at": "yesterday", "calibration": "CERT"},
        "retention_identity": "sha256:" + "a" * 64}]})
    assert rec.refusal_code(rec.validate_protocol, forged) == "acquisition_malformed"
    uncited = dict(protocol, hardware_measured={"status": "acquired", "records": [{"acquisition": {
        "device": "camera:1", "raw_sha256": "a" * 64, "acquired_at": "2026-09-23T00:00:00Z", "calibration": "CERT"}}]})
    assert rec.refusal_code(rec.validate_protocol, uncited) == "retention_record_missing"
    # A well-formed slot passes the schema check only; the hardware gate is the runner's (T139).
    cited = deepcopy(uncited)
    cited["hardware_measured"]["records"][0]["retention_identity"] = "sha256:" + "a" * 64
    assert rec.validate_protocol(cited)["hardware_measured"]["status"] == "acquired"


def test_protocols_are_executable_as_written(section):
    """Every device a step or path names is declared; the start jig is a fixture; the tube has its own datum scheme."""
    directory, _ = section
    protocols = {task_id: json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))
                 for task_id, name in (("T126", "protocol-flat-plate.json"), ("T127", "protocol-rolled-cylinder.json"),
                                       ("T128", "protocol-domed-coupon.json"), ("T129", "protocol-surface-scan.json"))}
    for task_id, protocol in protocols.items():
        fixtures = {f["id"]: f for f in protocol["fixtures"]}
        declared = set(fixtures) | {i["id"] for i in protocol["instruments"]} | {a["id"] for a in protocol["calibration_artifacts"]}
        datums = {d["id"] for d in protocol["datum_frames"]}
        assert [step["step"] for step in protocol["procedure"]] == list(range(1, len(protocol["procedure"]) + 1))
        for step in protocol["procedure"]:
            assert set(step["uses"]) <= declared and set(step.get("datums", [])) <= datums and step["outputs"], step
        assert all(i["settings"] for i in protocol["instruments"])
        for fmt in protocol["raw_formats"]:
            assert fmt["instrument"] in declared
        tape = [path for path in protocol["paths"] if "realization" in path or "of" in path]
        if tape:
            # The start jig realizes the independent variable, so it is a declared fixture with its slots and tolerance.
            jig = fixtures["JIG-START-01"]
            assert jig["declared_realization_tolerance"]["lateral_mm"] == mfg.EXECUTION_INSERT["lateral_mm"]
            assert jig["declared_realization_tolerance"]["heading_rad"] == mfg.EXECUTION_INSERT["heading_rad"]
            assert jig["declared_reseat_repeatability"]["lateral_mm"] == mfg.JIG_RESEAT["lateral_mm"]
            assert jig["geometry"]["slots_per_insert"] == 1
            assert {slot["slot"] for slot in jig["geometry"]["slots"]} == {path["id"] for path in tape}
            assert all(path["uses"] == ["JIG-START-01"] for path in tape) and set(jig["locates"]) <= datums
            offsets = {slot["slot"]: (slot["lateral_offset_mm"], slot["heading_offset_rad"]) for slot in jig["geometry"]["slots"]}
            for path in tape:
                assert offsets[path["id"]] == (path.get("lateral_offset_mm", 0.0), path.get("heading_offset_rad", 0.0))
            assert any("JIG-START-01" in step["uses"] for step in protocol["procedure"])
        # Raw formats of the section's own captures agree with what T138 reads.
        if task_id != "T129":
            expected = mfg.capture_expectations(protocol["protocol_id"])
            for fmt in protocol["raw_formats"]:
                assert expected[fmt["role"]]["ids"] == fmt["ids"] and expected[fmt["role"]]["schema"] == fmt["schema"]
                assert expected[fmt["role"]]["instrument"] == fmt["instrument"]
                assert expected[fmt["role"]]["frame"] == fmt["frame"]
    # The tube is located on its axis: V-blocks, a cylinder-fit axis A, scribe B, end face C, axis-primary frame.
    cylinder = protocols["T127"]
    assert {f["id"] for f in cylinder["fixtures"]} == {"VB-01", "JIG-START-01"}
    frames = {d["id"]: d for d in cylinder["datum_frames"]}
    assert "cylinder" in frames["A"]["feature"] and "datum_frame_axis" in frames["PART"]["definition"]
    assert "(or" not in json.dumps(cylinder["datum_frames"]) and "normal of A" not in json.dumps(cylinder["datum_frames"])
    assert cylinder["frame_chain"][2]["source"] == mfg.AXIS_FRAME_SOURCE
    # H1 no longer tests a surface distance without an instrument: the film gauge measures it, with its own U.
    h1 = cylinder["acceptance_criteria"][0]
    assert "film" in h1["statement"] and "film" in {i["id"] for i in cylinder["instruments"]}
    assert f"{2 * mfg.GAP_U:.4f}" in h1["test"]
    assert any("film" in step["uses"] for step in cylinder["procedure"])
    # A step that names a device it does not use, or a device the protocol lacks, is refused.
    broken = deepcopy(protocols["T128"])
    broken["fixtures"] = [f for f in broken["fixtures"] if f["id"] != "JIG-START-01"]
    assert rec.refusal_code(rec.validate_protocol, broken) == "undeclared_device"
    unnamed = deepcopy(protocols["T126"])
    unnamed["procedure"][0]["action"] += " Place the V-block."
    assert rec.refusal_code(rec.validate_protocol, unnamed) == "undeclared_device"
    wrong_datum = deepcopy(protocols["T126"])
    wrong_datum["procedure"][2]["datums"] = ["D"]
    assert rec.refusal_code(rec.validate_protocol, wrong_datum) == "undeclared_datum"


def _protocols(directory):
    return {task_id: json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))
            for task_id, name in (("T126", "protocol-flat-plate.json"), ("T127", "protocol-rolled-cylinder.json"),
                                  ("T128", "protocol-domed-coupon.json"), ("T129", "protocol-surface-scan.json"))}


def _dense(line, step=0.25):
    """A polyline resampled every ``step`` mm (for brute-force distances independent of the validator's)."""
    line = np.asarray(line, dtype=float)
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def test_tape_layout_keeps_tapes_targets_and_markers_apart(section):
    """One tape is on a specimen at a time; tape centrelines clear every marker and datum probe point; a layout
    that lays offset tapes together, runs a tape over markers or repeats a laying step is refused."""
    directory, _ = section
    protocols = _protocols(directory)
    for task_id in ("T126", "T127", "T128"):
        protocol = protocols[task_id]
        layout = protocol["tape_layout"]
        assert (layout["tape_width_mm"], layout["target_diameter_mm"], layout["marker_diameter_mm"]) == (3.0, 6.0, 6.0)
        tapes = {p["id"]: p for p in protocol["paths"] if p.get("kind") == "tape"}
        on, placed = set(), set()
        for step in protocol["procedure"]:
            for name in step.get("lays", []):
                assert not on and len(step["lays"]) == 1, (task_id, step["step"])   # one tape at a time
                on.add(name)
            for name in step.get("places", []):
                assert name in on
                placed.add(name)
            for name in step.get("removes", []):
                on.discard(name)
        assert not on and placed == set(tapes), task_id
        # Brute-force clearance on densely resampled centrelines (the validator uses exact segment distances).
        obstacles = [m["position_mm"] for m in protocol["markers"] if m.get("kind") == "specimen marker"]
        obstacles += [p for d in protocol["datum_frames"] for p in d.get("probe_points_mm", [])]
        for name, path in tapes.items():
            dense = _dense(path["centreline_mm"])
            if obstacles:
                nearest = min(float(np.min(np.linalg.norm(dense - np.asarray(o), axis=1))) for o in obstacles)
                assert nearest >= 0.5 * (6.0 + 6.0), (task_id, name, nearest)
        # The offset tapes would overlap if they lay together: that is why each is laid alone.
        names = sorted(tapes)
        closest = min(rec.polyline_distance(tapes[a]["centreline_mm"], tapes[b]["centreline_mm"])
                      for a in names for b in names if a < b)
        assert closest < 3.0 + 6.0, task_id
        # Repeats measure again; they never lay, place or remove anything, and they name the steps they repeat.
        for step in protocol["procedure"]:
            if "repeats" in step:
                assert all(n < step["step"] and not {"lays", "places", "removes"} & set(protocol["procedure"][n - 1])
                           for n in step["repeats"]), step
                assert all(str(n) in step["action"] for n in step["repeats"])
    # The plate tapes run midway between two marker rows: their centrelines are the closed-form lines.
    plate = {p["id"]: np.array(p["centreline_mm"]) for p in protocols["T126"]["paths"]}
    assert np.allclose(plate["N0"][:, 1], 30.0, atol=1e-9) and np.allclose(np.abs(plate["L2"][:, 1] - 30.0), 2.0, atol=1e-4)
    s = plate["H5"][:, 0] - plate["H5"][0, 0]
    assert np.allclose(np.abs(plate["H5"][:, 1] - 30.0), np.tan(0.005) * s, atol=1e-4)
    # The cylinder helices are straight on the development (R phi, z) at 45 degrees.
    helix = np.array(next(p for p in protocols["T127"]["paths"] if p["id"] == "HX45")["centreline_mm"])
    arc = geo.CYLINDER_RADIUS * np.unwrap(np.arctan2(helix[:, 1], helix[:, 0]))
    assert np.allclose(arc - arc[0], helix[:, 2], atol=1e-3)  # points are rounded to 0.1 um
    # The reviewer's layouts are refused: tapes on the plate's marker row, the helix over marker C0-01, offset tapes
    # laid together (their slots 2 mm apart), a tape left on the coupon when the next is laid, a repeat that lays.
    old_plate = deepcopy(protocols["T126"])
    for path in old_plate["paths"]:
        start = (-120.0, 0.0)
        path["centreline_mm"] = mfg.tape_centreline(geo.PLATE, start, 0.0, 240.0, path.get("lateral_offset_mm", 0.0),
                                                    path.get("heading_offset_rad", 0.0))
    assert rec.refusal_code(rec.validate_protocol, old_plate) == "path_over_marker"
    old_helix = deepcopy(protocols["T127"])
    for path in old_helix["paths"]:
        path["centreline_mm"] = mfg.tape_centreline(geo.CYLINDER, (0.0, 0.0), math.radians(45.0), 300.0,
                                                    path.get("lateral_offset_mm", 0.0))
    assert rec.refusal_code(rec.validate_protocol, old_helix) == "path_over_marker"
    marker = next(m for m in protocols["T127"]["markers"] if m["id"] == "C0-01")
    assert rec.point_polyline_distance(marker["position_mm"], old_helix["paths"][0]["centreline_mm"]) < 2.0
    for task_id in ("T126", "T127", "T128"):
        matrix = rec.protocol_refusal_matrix(protocols[task_id])
        for case, code in (("two_offset_tapes_in_one_laying", "jig_slots_overlap"),
                           ("tape_left_on_the_specimen", "tapes_overlap"),
                           ("specimen_marker_under_a_tape", "path_over_marker"),
                           ("repeat_includes_laying", "repeat_includes_laying")):
            assert matrix[case] == {"expected": code, "observed": code}, (task_id, case)
    jig = next(f for f in protocols["T128"]["fixtures"] if f["id"] == "JIG-START-01")
    slots = {slot["slot"]: slot for slot in jig["geometry"]["slots"]}
    assert rec.slot_gap(slots["N"], slots["L"], 20.0) == pytest.approx(2.0)
    assert rec.slot_gap(slots["N"], slots["H"], 20.0) == pytest.approx(0.0)  # the heading slot shares the exit point
    # Two polylines that cross are at distance zero, and parallel ones at their offset.
    assert rec.polyline_distance([[0, 0, 0], [10, 0, 0]], [[5, -5, 0], [5, 5, 0]]) == pytest.approx(0.0)
    assert rec.polyline_distance([[0, 0, 0], [10, 0, 0]], [[0, 3, 0], [10, 3, 0]]) == pytest.approx(3.0)


def test_fixtures_declare_the_datum_targets_their_steps_measure(section):
    directory, _ = section
    for task_id, protocol in _protocols(directory).items():
        fixtures = {f["id"]: f for f in protocol["fixtures"]}
        holder = protocol["fixtures"][0]
        assert len(holder["datum_targets"]) == 4 and all(len(t["position_mm"]) == 3 for t in holder["datum_targets"])
        for step in protocol["procedure"]:
            if "datum targets" in step["action"]:
                assert any(fixtures[name].get("datum_targets") for name in step["uses"] if name in fixtures), step
        # The tracker registration exists only where a tracker is declared (not in MFG-SCAN-01).
        has_tracker = "tracker" in {i["id"] for i in protocol["instruments"]}
        assert has_tracker == any("tracker" in step["uses"] for step in protocol["procedure"]), task_id
        broken = deepcopy(protocol)
        broken["fixtures"][0].pop("datum_targets")
        if has_tracker:
            assert rec.refusal_code(rec.validate_protocol, broken) == "fixture_targets_undeclared"
    # MFG-SCAN-01 lays no tapes, and its repeat step says nothing about tapes.
    scan = _protocols(directory)["T129"]
    repeat = [step for step in scan["procedure"] if "repeats" in step]
    assert len(repeat) == 1 and "tape" not in repeat[0]["action"]
    # A free-text repeat that lists no steps is refused.
    free = deepcopy(scan)
    free["procedure"][repeat[0]["step"] - 1].pop("repeats")
    assert rec.refusal_code(rec.validate_protocol, free) == "procedure_step_malformed"


def test_cylinder_protocol_is_acquirable_in_its_v_blocks(section):
    """The vees sit between the marker rings; datum A is probed between markers on the upper half; the captures
    ask only for markers the camera can see; a tube out of round stops the run instead of being re-seated."""
    directory, _ = section
    protocol = _protocols(directory)["T127"]
    blocks = next(f for f in protocol["fixtures"] if f["id"] == "VB-01")
    assert "100 mm and 200 mm" in blocks["geometry"] and "straight up" in blocks["contacts"]
    rings = set(mfg.CYLINDER_RINGS_MM)
    assert not {100.0, 200.0} & rings
    angles = [round(math.degrees(phi), 9) for phi, _ in mfg.axis_probe_points()["A"]]
    assert all(abs(a) <= mfg.ACCESSIBLE_ARC_DEG and abs(a) % 30.0 == 15.0 for a in angles)
    assert {z for _, z in mfg.axis_probe_points()["A"]} == rings
    datum_a = next(d for d in protocol["datum_frames"] if d["id"] == "A")
    assert datum_a["points"] == len(datum_a["probe_points_mm"]) == 12 and "-75, -15, 45 and 75 deg" in datum_a["feature"]
    photogrammetry = next(f for f in protocol["raw_formats"] if f["role"] == "photogrammetry")
    visible = {m["id"] for m in protocol["markers"] if m["faces_up"]}
    assert set(photogrammetry["ids"]) == visible and len(visible) == 21
    assert {name for pair in protocol["specimen"]["marker_pairs"].values() for name in pair} <= visible
    for marker in protocol["markers"]:
        phi = math.degrees(math.atan2(marker["position_mm"][1], marker["position_mm"][0]))
        assert marker["faces_up"] == (abs(phi) <= 90.0 + 1e-9), marker["id"]
    locate = protocol["procedure"][2]
    assert f"{mfg.roundness_threshold():.3f} mm" in locate["check"] and "otherwise stop and record" in locate["check"]
    assert mfg.roundness_threshold() == pytest.approx(0.1 + 2 * 2 * 0.002)
    assert protocol["specimen"]["declared_roundness_tolerance_mm"] == 0.1
    # The datum scheme recovers a known pose from exactly these probe points.
    assert mfg.axis_frame_study()["pose_error"] < 1e-9


def test_axis_datum_frame_recovers_a_tube_pose():
    pose = met.transform(met.exp_so3([-0.3, 0.2, 0.1]), [5.0, 7.0, -3.0])
    rings = np.array([[100 * math.cos(a), 100 * math.sin(a), z] for z in (50.0, 250.0) for a in np.linspace(0, 2 * math.pi, 9)[:-1]])
    moved = rings @ pose[:3, :3].T + pose[:3, 3]
    point, direction, radius, residual = met.fit_cylinder(moved, pose[:3, 3] + pose[:3, :3] @ [0.5, 0.5, 100.0],
                                                          pose[:3, :3] @ [0.01, -0.02, 1.0], 98.0)
    assert radius == pytest.approx(100.0, abs=1e-9) and np.max(np.abs(residual)) < 1e-9
    scribe = np.array([[100.0, 0.0, 30.0]]) @ pose[:3, :3].T + pose[:3, 3]
    face = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [-5.0, -5.0, 0.0]]) @ pose[:3, :3].T + pose[:3, 3]
    # The axis direction's sign does not matter: z points from the end face into the tube.
    for sign in (1.0, -1.0):
        frame = met.datum_frame_axis(point, sign * direction, scribe, face)
        assert np.max(np.abs(met.pose_difference(frame, pose))) < 1e-9
    study = mfg.axis_frame_study()
    assert study["pose_error"] < 1e-9 and study["scribe_on_axis"] == study["face_parallel_to_axis"] == "datum_degenerate"
    assert study["too_few_points"] == "cylinder_underdetermined"


@pytest.mark.lab_task("T127")
def test_cylinder_protocol_predicts_chord_geodesic_gaps(section):
    _, reports = section
    report = reports["T127"]
    gap_finding = _finding(report, "Rolled-cylinder chord-geodesic gaps")
    gaps = gap_finding["value"]
    # The series check is a signed margin: negative when the alternating-series bound holds with room.
    series = [c for c in gap_finding["basis"]["checks"] if "alternating-series bound" in c["reference"]]
    assert len(series) == 1 and series[0]["comparison"] == "signed_le" and series[0]["observed"] < 0.0
    radius = geo.CYLINDER_RADIUS
    for row in mfg.cylinder_study()["pairs"]:
        step = 1e-4
        numeric = (_gap_at(row, radius + step) - _gap_at(row, radius - step)) / (2 * step)
        assert mfg.gap_radius_derivative(row["dphi_rad"], row["dz_mm"], radius) == pytest.approx(numeric, abs=1e-8)
    assert gaps["circumferential 90 deg"] == pytest.approx(radius * math.pi / 2 - 2 * radius * math.sin(math.pi / 4), rel=1e-12)
    assert gaps["axial 100 mm"] == pytest.approx(0.0, abs=1e-12)
    counter = _finding(report, "A marker chord differs from the surface distance")
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    resolvable = _finding(report, "Minimum circumferential marker separation")["value"]
    assert resolvable["camera"] == pytest.approx(23.86, abs=0.01)
    assert resolvable["cmm"] < resolvable["tracker"] < resolvable["camera"]
    assert _finding(report, "Rolled-cylinder Jacobi transfer equals")["value"] <= 1e-9
    axis = _finding(report, "The axis-primary datum frame")
    assert axis["evidence_status"] == "numerically_verified" and axis["value"]["pose_error"] < 1e-9
    assert _labels(report)["The unrolled-film gauge achieves the declared 0.05 mm on marker-to-marker surface distances"] \
        == "not_established"


@pytest.mark.lab_task("T128")
def test_coupon_protocol_predicts_focal_crossing(section):
    _, reports = section
    report = reports["T128"]
    focal = _finding(report, "Laterally offset routes cross the nominal route")
    assert focal["evidence_status"] == "numerically_verified"
    assert focal["value"] == pytest.approx(153.407, abs=2e-3)
    study = mfg.nominal_study()
    assert study["nonlinear_crossing_mm"] == pytest.approx(154.36, abs=0.01)
    assert study["crossing_extrapolated_mm"] == pytest.approx(focal["value"], abs=0.02)
    orders = _finding(report, "Linearized separation remainder")["value"]
    assert orders["on_axis"] == pytest.approx(3.0, abs=0.1) and orders["off_axis"] == pytest.approx(2.0, abs=0.1)
    signature = _finding(report, "Curvature signature")["value"]
    assert signature["coupon_lateral_2mm"] < 0 < signature["plate_lateral_2mm"]
    assert _labels(report)["The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance"] \
        == "not_established"
    # Same-origin second derivations are always present and always numerically_verified.
    integrator = _finding(report, "The RK4 coupon transfer agrees with ciw's adaptive")
    geometry = _finding(report, "Coupon Christoffel symbols and Gaussian curvature agree with finite-difference")
    assert integrator["evidence_status"] == geometry["evidence_status"] == "numerically_verified"
    assert integrator["value"] < 1e-5 and 0.0 < geometry["value"]["curvature"] < 1e-9
    signature = _finding(report, "Curvature signature")
    assert signature["basis"]["checks"][0]["observed"] > 5.0  # plate - coupon separation over the combined U
    has = {name: importlib.util.find_spec(name) is not None for name in ("scipy", "sympy")}
    for prefix, module in (("scipy DOP853 integrating", "scipy"), ("A sympy derivation", "sympy")):
        record = _finding(report, prefix)
        if has[module]:
            assert record["evidence_status"] == "independently_verified"
            # Producer and checker each carry a real revision: ciw's version and the external package's.
            check = record["basis"]["independent_check"]
            assert check["producer"]["revision"] == f"ciw {__version__}"
            assert check["checker"]["revision"] == __import__(module).__version__
        else:
            assert record["evidence_status"] == "not_established" and record["expected_not_established"] is True


@pytest.mark.lab_task("T128")
def test_coupon_report_wording_does_not_depend_on_optional_modules(section, monkeypatch, tmp_path):
    _, reports = section
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name in ("scipy", "sympy", "mpmath") else real(name, *args))
    result = mfg.independent_coupon_checks(mfg.nominal_study()["length_mm"])
    assert result["scipy"] is None and result["sympy"] is None
    # The fallback curvature comes from the height values, not from the curvature it checks.
    assert 0.0 < result["ciw_geometry"]["max_curvature_difference"] < 1e-9
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    bare = validate_report(runner.run_task(queue["T128"], section_implementations("manufacturing")["T128"],
                                           runner.Context(tmp_path), {}))
    full = reports["T128"]
    assert bare["state"] == "completed" and bare["evidence_status"]["primary"] == "numerically_verified"
    assert [f["claim"] for f in bare["findings"]] == [f["claim"] for f in full["findings"]]
    for name in runner.PROSE_FIELDS:
        assert runner._skeleton(bare[name]) == runner._skeleton(full[name]), name
    changed = {f["claim"] for f, g in zip(bare["findings"], full["findings"]) if f["evidence_status"] != g["evidence_status"]}
    optional = {f["claim"] for f in full["findings"] if f["claim"].startswith(("scipy DOP853", "A sympy derivation"))}
    assert changed <= optional


@pytest.mark.lab_task("T129")
def test_metrology_sampling_design_and_counterexamples(section):
    _, reports = section
    report = reports["T129"]
    design = _finding(report, "Sampling design resolving profile curvature")
    assert design["evidence_status"] == "numerically_verified"
    assert design["value"]["coupon crest"]["window_mm"] == pytest.approx(13.66, abs=0.01)
    assert _finding(report, "Repeat scans needed")["value"] == {"5%": 1, "2%": 5, "1%": 109}
    circle = _finding(report, "The osculating-circle window rule")
    assert circle["counterexample"]["witness"]["quartic_ratio_gauss_over_circle"] == pytest.approx(4.0)
    assert circle["value"] > 1.0
    window = _finding(report, "A smaller fitting window")
    assert window["value"]["rms_at_one_third_window"] > 2 * window["value"]["rms_at_design_window"]
    assert _finding(report, "Monte Carlo curvature error")["evidence_status"] == "numerically_verified"
    assert _finding(report, "A real laser line scanner")["evidence_status"] == "not_established"
    # The design rule is explicit: halve the tolerance between bias and k = 2 noise.
    rule = met.sampling_design(0.01, 1.25e-7, 5e-4, 0.01)
    assert met.curvature_bias_series(rule["window_mm"], 1.25e-7) == pytest.approx(2.5e-4)
    assert 2 * met.curvature_noise_std_continuum(rule["window_mm"], rule["max_spacing_mm"], 0.01) == pytest.approx(2.5e-4)


@pytest.mark.lab_task("T129")
def test_scan_protocol_and_as_built_fit(section):
    directory, reports = section
    report = reports["T129"]
    fit = _finding(report, "The as-built dome fit")
    assert fit["evidence_status"] == "numerically_verified" and fit["uncertainty"]["value"] >= 0.0
    model = _finding(report, "The fit residual flags an elliptical as-built dome")
    assert model["evidence_status"] == "numerically_verified" and "counterexample" in model
    assert _finding(report, "Surface-scan protocol record validates")["value"] == 17
    assert _labels(report)["The formed coupon passes the Gaussian model test and its fitted height and width lie within "
                           "the declared forming tolerances"] == "not_established"
    protocol = rec.validate_protocol(json.loads(
        (directory / "artifacts" / "T129" / "protocol-surface-scan.json").read_text(encoding="utf-8")))
    assert protocol["protocol_id"] == "MFG-SCAN-01" and protocol["hardware_measured"]["records"] == []
    plan = protocol["scan_plan"]
    assert plan["instrument_setup"]["coupon_max_slope_deg"] < plan["instrument_setup"]["max_incidence_deg"]
    assert "model_test" in plan["surface_fit"] and plan["retained_raw_data"]
    study = mfg.as_built_study()
    assert study["recovery_error"] < 1e-8
    covariance = np.array(study["covariance_mm2"])
    assert np.allclose(covariance, covariance.T) and np.linalg.eigvalsh(covariance).min() > 0.0
    # The scan identifies height and width far better than the declared forming tolerances do.
    u = study["standard_uncertainty_mm"]
    assert u["height_mm"] < 0.1 * 0.2 / math.sqrt(3.0) and u["sigma_mm"] < 0.1 * 0.5 / math.sqrt(3.0)
    assert study["elliptical_chi2"] > study["chi2_threshold"] > 1.0 and study["false_alarm_fraction"] <= 0.02
    # The fit itself: exact on noise-free data from the nominal start, refused when underdetermined.
    x, y = np.meshgrid(np.linspace(-60, 140, 21), np.linspace(-100, 100, 21))
    truth = [10.1, 19.8, 0.2, -0.1, 0.0, 0.0, 1e-4]
    params, _, residual = met.fit_dome(x, y, met.dome_surface(truth, x, y), [10.0, 20.0, 0, 0, 0, 0, 0])
    assert np.allclose(params, truth, atol=1e-9) and float(np.max(np.abs(residual))) < 1e-9
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.fit_dome([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [10.0, 20.0, 0, 0, 0, 0, 0])
    assert refused.value.code == "fit_underdetermined"
    # T138 carries the scan-conditioned uncertainty; it is smaller than the tolerance-based one.
    prediction = mfg.separation_prediction()
    assert max(prediction["scan_conditioned_expanded_mm"]) < max(prediction["conditioned_expanded_mm"])


@pytest.mark.lab_task("T130")
def test_artifacts_datum_frames_and_chain_covariance(section):
    _, reports = section
    report = reports["T130"]
    for prefix in ("First-order covariance of the INSTRUMENT->CAD", "The 3-2-1 datum frame", "Gauge-sphere fit",
                   "Step-gauge fit"):
        assert _finding(report, prefix)["evidence_status"] == "numerically_verified", prefix
    assert _finding(report, "The physical gauge sphere")["domain"] == "calibration"
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.datum_frame_321([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0]], [0, 1, 0])
    assert refused.value.code == "datum_degenerate"
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.datum_frame_321([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]], [0, 1, 0])
    assert refused.value.code == "datum_degenerate"
    # Both degenerate datums are refusal checks of the task, not only of this test.
    datum_checks = {c["reference"]: c for c in _finding(report, "The 3-2-1 datum frame")["basis"]["checks"]}
    assert datum_checks["collinear primary datum points (A)"]["observed_refusal"] == "datum_degenerate"
    assert datum_checks["secondary datum direction (B) normal to the primary plane (A)"]["passed"]
    # The chain std's uncertainty is a linearization estimate in mm, far below the std itself.
    chain = _finding(report, "First-order covariance of the INSTRUMENT->CAD")
    assert chain["uncertainty"]["kind"] == "truncation_bound"
    assert 0.0 <= chain["uncertainty"]["value"] < 1e-3 * min(chain["value"]["coupon_far_corner"])
    links = [(met.transform(np.eye(3), [100.0, 0.0, 0.0]), np.diag([1e-4] * 3 + [1e-10] * 3))]
    linear = met.point_covariance(*met.compose_chain(links), np.array([10.0, 0.0, 0.0]))
    assert np.allclose(met.sigma_point_chain_covariance(links, [10.0, 0.0, 0.0]), linear, rtol=1e-6, atol=1e-15)
    pose = met.transform(met.exp_so3([0.1, -0.2, 0.3]), [1.0, 2.0, 3.0])
    assert np.allclose(met.pose_difference(met.exp_se3(np.array([1e-4, 0, 0, 0, 0, 2e-5])) @ pose, pose),
                       [1e-4, 0, 0, 0, 0, 2e-5], atol=1e-8)  # first order: the left Jacobian adds 1e-9


@pytest.mark.lab_task("T131")
def test_gage_rr_recovers_components_and_refuses_unbalanced(section):
    directory, reports = section
    report = reports["T131"]
    recovery = _finding(report, "ANOVA Gage R&R recovers")
    assert recovery["evidence_status"] == "numerically_verified"
    assert recovery["value"]["repeatability"] == pytest.approx(1e-4, rel=0.02)
    spread = _finding(report, "Sampling spread of %GRR")["value"]
    assert spread["0.05"] < spread["true"] < spread["0.95"] and spread["true"] == pytest.approx(23.94, abs=0.01)
    refused = _finding(report, "Gage R&R refuses")
    assert refused["value"] == 2 and refused["evidence_status"] == "numerically_verified"
    spread_finding = _finding(report, "Sampling spread of %GRR")
    spread_checks = spread_finding["basis"]["checks"]
    assert all(check["passed"] for check in spread_checks) and len(spread_checks) == 3
    # The uncertainty is in % (order-statistic intervals of the quantiles), like the value.
    halfwidth = spread_finding["uncertainty"]["value"]
    assert set(halfwidth) == {"0.05", "0.5", "0.95"} and all(0.0 < v < 5.0 for v in halfwidth.values())
    assert mfg._quantile_halfwidth(np.arange(1.0, 2001.0), 0.5) == pytest.approx(0.5 * (1044 - 956), abs=1.0)
    # The procedure of a real study: parts are features, every cell once per replicate round, re-fixturing between.
    procedure = json.loads((directory / "artifacts" / "T131" / "gage-rr-procedure.json").read_text(encoding="utf-8"))
    assert procedure["status"].startswith("plan") and procedure["type1_study"]["readings"] == 25
    # Coupon parts lie on one tape left alone on the coupon (tapes are never on it together).
    assert "alone on the coupon" in procedure["studies"]["MFG-COUPON-01"]["setup"]
    assert not any(" to nominal " in p["feature"] for p in procedure["studies"]["MFG-COUPON-01"]["parts"])
    for protocol_id, study in procedure["studies"].items():
        parts = [p["feature"] for p in study["parts"]]
        assert len(parts) == 10 and len(study["rounds"]) == 3, protocol_id
        for round_ in study["rounds"]:
            assert "re-seat" in round_["before"]
            assert sorted(b["operator"] for b in round_["blocks"]) == ["O1", "O2", "O3"]
            assert all(sorted(b["order"]) == sorted(parts) for b in round_["blocks"])
    plate_pairs = {"-".join(r["pair"]) for r in mfg.plate_study()["pairs"]}
    assert set(mfg.GAGE_PARTS["MFG-FLAT-PLATE-01"]) <= plate_pairs
    assert set(mfg.GAGE_PARTS["MFG-CYLINDER-01"]) == {r["pair"] for r in mfg.cylinder_study()["pairs"]}
    assert _labels(report)["The measurement system is approved for production use"] == "not_established"
    exact = np.zeros((2, 2, 2))
    exact[1] += 1.0
    result = met.gage_rr_anova(exact)
    assert result["ss"]["part"] == pytest.approx(2.0) and result["ss"]["error"] == 0.0


@pytest.mark.lab_task("T132")
def test_placement_curvature_stack_and_radius_counterexample(section):
    _, reports = section
    report = reports["T132"]
    kg = _finding(report, "Geodesic curvature of a 30-to-60 degree")["value"]
    assert kg["min_steering_radius_mm"] == pytest.approx(300 / (math.pi / 6 * math.cos(math.pi / 6)), rel=1e-9)
    stack = _finding(report, "Tolerance stack of the placed course")["value"]
    assert stack["length_max_mm"] == pytest.approx(409.83, abs=0.01)
    counter = _finding(report, "Programming a helix in machine angles")
    assert counter["value"] == pytest.approx(1000 * math.sin(math.atan(1.002) - math.pi / 4), rel=1e-12)
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    assert mfg.placement_deviation(500.0, math.pi / 4, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.lab_task("T133")
def test_winding_clairaut_sensitivity_and_slippage(section):
    _, reports = section
    report = reports["T133"]
    assert _finding(report, "The Clairaut constant is conserved")["evidence_status"] == "numerically_verified"
    # Both regimes grow secularly (linearly): the rate is |M01| / P of the parabolic one-period monodromy.
    rate = _finding(report, "Secular growth rate of the heading-error Jacobi field")
    assert rate["evidence_status"] == "numerically_verified"
    assert rate["value"]["psi50_librating"] == pytest.approx(1.0417, abs=1e-3)
    assert rate["value"]["psi70_circulating"] == pytest.approx(2.9540, abs=1e-3)
    study = mfg.winding_study()
    for regime in study["regimes"].values():
        assert abs(regime["monodromy_trace_minus_2"]) < 1e-6
        assert regime["three_period_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert regime["m01_mm"] == pytest.approx(regime["m01_clairaut_mm"], rel=1e-5)
    assert _finding(report, "The 50 deg winding librates")["evidence_status"] == "numerically_verified"
    slip = _finding(report, "Slippage tendency")["value"]
    assert slip["psi50"]["max_ratio"] == pytest.approx(0.4436, abs=1e-3) and slip["psi70"]["fraction_above_mu"] == 0.0
    counter = _finding(report, "A constant winding angle is not geodesic")
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    turn = _finding(report, "Clairaut sensitivity predicts")["value"]
    assert math.degrees(turn["turn_rad"]) == pytest.approx(115.39, abs=0.01)


@pytest.mark.lab_task("T134")
def test_coating_standoff_and_offset_cusp(section):
    _, reports = section
    report = reports["T134"]
    envelope = _finding(report, "Lateral-error envelope")["value"]
    assert envelope["max_mm"] == pytest.approx(0.4201, abs=1e-4)
    cusp = _finding(report, "A spray standoff beyond the concave radius")
    assert cusp["value"]["min_concave_radius_mm"] < cusp["value"]["spray_standoff_mm"]
    assert cusp["value"]["reversed_segments"] > 0 and "counterexample" in cusp
    assert _labels(report)["The trajectory is safe to execute on a welding or coating robot cell"] == "not_established"
    study = mfg.coating_study()
    assert study["cylinder_control"]["max_difference_mm"] < 1e-12
    # The series check is tight: ray casting and -kappa e^2 / 2 agree to well within 0.1% at every station.
    assert study["standoff_excess"] <= 0.0
    for row in study["standoff"]:
        exact, series = row["standoff_error_exact_mm"], row["standoff_error_series_mm"]
        assert abs(exact - series) <= 1e-3 * abs(series) + 1e-9
    standoff = _finding(report, "Standoff error from a lateral tool offset")
    tight = [c for c in standoff["basis"]["checks"] if c["reference"].startswith("max over stations of abs(ray-cast")]
    assert len(tight) == 1 and "1e-3 abs(series)" in tight[0]["reference"] and tight[0]["passed"]
    # The TCP speed factor is checked pointwise against |t + H dn/ds| and per polyline segment, on the symmetry
    # axis (tau_g = 0) and off it, where the tau_g term is large enough for the pointwise check to resolve.
    offset = _finding(report, "Tool-centre-point path length element")
    assert offset["evidence_status"] == "numerically_verified" and len(offset["basis"]["checks"]) == 3
    assert study["tools"]["welding torch"]["max_H_tau_g"] < 1e-12
    off = study["off_axis_tools"]["welding torch"]
    assert off["max_H_tau_g"] > 1e-2 and off["tau_term_max"] > 100 * 1e-7
    for tools in (study["tools"], study["off_axis_tools"]):
        for tool in tools.values():
            assert tool["pointwise_max_difference"] < 1e-7
        assert tools["welding torch"]["segment_max_difference"] < 2e-3
    u = np.array([-20.0, 5.0])
    t = geo.COUPON.unit_tangent(u, 0.3)
    exact = geo.standoff_error_exact(geo.COUPON, u, geo.lateral_direction3(geo.COUPON, u, t), 0.01, 15.0)
    n = geo.COUPON.normal(u, t)
    assert exact == pytest.approx(-0.5 * float(n @ geo.second_fundamental_form(geo.COUPON, u) @ n) * 1e-4, rel=1e-3)


@pytest.mark.lab_task("T135")
def test_scan_plans_coverage_and_counterexample(section):
    _, reports = section
    report = reports["T135"]
    plans_finding = _finding(report, "Coverage and path length")
    assert plans_finding["evidence_status"] == "numerically_verified"
    plans = plans_finding["value"]
    assert plans["geodesic x1"]["coverage"] < 0.99 < plans["geodesic x0.6"]["coverage"]
    counter = _finding(report, "Geodesic rows at the swath spacing leave gaps")
    assert counter["evidence_status"] == "numerically_verified"
    assert counter["counterexample"]["witness"]["plate_coverage"] == 1.0
    shortest = _finding(report, "Shortest evaluated scan plan")["value"]
    assert shortest["plan"] == "chart-parallel x0.9"
    # Two abutting 2 mm swaths cover a 10 x 4 mm plate exactly; one covers half of it.
    rows = [np.column_stack([np.linspace(0, 10, 11), np.full(11, y), np.zeros(11)]) for y in (0.0, 2.0)]
    sample = geo.area_sample(geo.PLATE, (0.0, 10.0), (-1.0, 3.0), 400, columns=5)
    assert geo.coverage_fraction(sample, rows, 2.0) == pytest.approx(1.0)
    assert geo.coverage_fraction(sample, rows[:1], 2.0) == pytest.approx(0.5, abs=0.02)


@pytest.mark.lab_task("T136", "T137")
def test_rankings_by_calibration_tolerance_and_focus_margin(section):
    _, reports = section
    calibration = _finding(reports["T136"], "Candidate coupon routes ranked by the heading calibration tolerance")
    assert calibration["evidence_status"] == "numerically_verified"
    assert calibration["value"]["ranking"][0] == "fan+0deg"
    table = mfg.calibration_table()
    for row in table["rows"]:
        assert len(row["corner_ratios"]) == 4 and row["derating_converged"]
        assert max(row["corner_ratios"].values()) <= mfg.DERATE_TARGET + 1e-9, row["route"]
        assert row["realized_max_error_mm"] <= mfg.LATERAL_SPEC_MM
        # The evidence: a second computation of the derated corners agrees and stays within the spec.
        assert max(row["independent_corner_ratios"].values()) <= 1.0, row["route"]
        assert all(abs(row["independent_corner_ratios"][c] - v) < 1e-4 for c, v in row["corner_ratios"].items())
    kinds = {c["reference_kind"] for c in calibration["basis"]["checks"]}
    assert "cross_implementation" in kinds
    first = _finding(reports["T136"], "The first-order tolerance allocation exceeds the spec")
    assert first["evidence_status"] == "numerically_verified" and "counterexample" in first
    assert first["value"]["fan+25deg"] > 1.0 and first["value"]["fan+0deg"] < 1.0
    # The focus margin is T024's s_c - L over conjugate points (zeros of j_head): no candidate route has one within
    # its horizon, so every margin is a positive lower bound and the routes tie.
    margin = _finding(reports["T137"], "Candidate coupon routes ranked by focus margin s_c - L")
    assert margin["evidence_status"] == "numerically_verified" and margin["unit"] == "mm"
    routes = [f"fan+{a}deg" for a in mfg.FAN_HEADINGS]
    assert margin["value"]["ranking"] == [sorted(routes)]
    assert all(margin["value"]["lower_bound"][r] for r in routes)
    # A censored margin has no value, as T024 records it; its horizon-dependent lower bound has its own key, so a
    # queue aggregate cannot show a bound as a margin.
    assert all(margin["value"]["focus_margin_mm"][r] is None for r in routes)
    bounds = margin["value"]["focus_margin_lower_bound_mm"]
    assert min(bounds.values()) > 0.0 and bounds["fan+0deg"] == pytest.approx(297.82, abs=0.01)
    assert all(bounds[r] == pytest.approx(margin["value"]["horizon_mm"][r] - _length(reports, r), abs=1e-6) for r in routes)
    rows = mfg.focus_margins(mfg.fan_study()["routes"])
    assert all((row["focus_margin_mm"] is None) == row["margin_is_lower_bound"] for row in rows)
    assert min(margin["value"]["min_j_head_over_s"].values()) > 0.1
    # The ratio that includes focal points (zeros of j_lat) has its own name and ranks the routes.
    focus = _finding(reports["T137"], "Candidate coupon routes ranked by focal clearance ratio")
    # Routes without a focus inside the horizon have lower-bound ratios only: one tied tier, not an order.
    assert focus["value"]["ranking"][0] == ["fan+15deg", "fan+20deg", "fan+25deg"]
    assert focus["value"]["ranking"][-1] == ["fan+0deg"] and all(len(t) == 1 for t in focus["value"]["ranking"][1:])
    assert focus["value"]["unresolved_tier"] == ["fan+15deg", "fan+20deg", "fan+25deg"]
    assert focus["value"]["focal_clearance_ratio"]["fan+0deg"] == pytest.approx(0.7588, abs=1e-4)
    assert not any("focus margin" in f["claim"] and "ratio" not in f["claim"] and f is not margin
                   for f in reports["T137"]["findings"])
    counter = _finding(reports["T137"], "The shortest candidate route has the lowest focal clearance ratio")
    assert counter["evidence_status"] == "numerically_verified"
    assert counter["counterexample"]["witness"]["route"] == "fan+0deg"
    assert "focal clearance ratio" in counter["counterexample"]["statement"]
    # The shortest route's focus margin is censored and positive: no value, a positive lower bound.
    assert counter["value"]["focus_margin_mm"] is None and counter["value"]["focus_margin_lower_bound_mm"] > 0.0
    assert counter["counterexample"]["witness"]["focus_margin_mm"] is None
    variation_finding = _finding(reports["T137"], "The second variation of route length")
    variation = variation_finding["value"]
    assert variation["finite_difference_mm"] == pytest.approx(variation["index_form_mm"], rel=2e-4)
    assert variation_finding["uncertainty"]["value"] == pytest.approx(
        abs(variation["finite_difference_mm"] - variation["index_form_mm"]), abs=1e-12)
    assert _finding(reports["T136"], "The robot, fixture and frame calibration")["evidence_status"] == "not_established"


def _length(reports, route):
    return next(s["length_mm"] for s in mfg.fan_study()["summaries"] if s["route"] == route)


@pytest.mark.lab_task("T136")
def test_ranking_evidence_detects_a_wrong_exact_perturbation(monkeypatch):
    """A wrong exact perturbation still satisfies the derating loop's stop condition but disagrees with the re-evaluation."""
    fan = mfg.fan_study()
    real = geo.separation_nonlinear
    monkeypatch.setattr(geo, "separation_nonlinear", lambda *args, **kwargs: 1.08 * real(*args, **kwargs))
    row = mfg.calibration_ranking({"routes": fan["routes"][:1], "coarse": fan["coarse"]})["rows"][0]
    assert max(row["corner_ratios"].values()) <= mfg.DERATE_TARGET + 1e-9
    assert max(abs(row["independent_corner_ratios"][c] - v) for c, v in row["corner_ratios"].items()) > 1e-2


@pytest.mark.lab_task("T138")
def test_predicted_separation_has_no_measured_counterpart(section):
    _, reports = section
    report = reports["T138"]
    assert report["state"] == "partial"
    predicted = _finding(report, "Predicted separation of the 2 mm offset route")
    assert predicted["evidence_status"] == "numerically_verified"
    assert predicted["value"]["separation_mm"][0] == pytest.approx(2.0, abs=1e-6)
    assert predicted["value"]["separation_mm"][-1] == pytest.approx(-1.0374, abs=1e-4)
    refused = _finding(report, "The comparison refuses")
    assert refused["value"] == 7 and refused["evidence_status"] == "numerically_verified"
    assert "scan_conditioned_expanded_mm" in predicted["value"]
    assert _finding(report, "Measured separation on the coupon agrees")["evidence_status"] == "not_established"
    start = _finding(report, "A 2-sigma start offset of the declared jig")
    assert start["value"]["max_en_without_execution"] > 1.0 >= start["value"]["max_en_open_loop"]
    # The measured branch: prediction re-integrated from a 2-sigma CMM start-pose estimate, conditioned U_p.
    assert 0.0 < start["value"]["max_en_conditioned"] <= 1.0
    conditioned = [c for c in start["basis"]["checks"] if "re-integrated from a start pose" in c["reference"]]
    assert len(conditioned) == 1 and conditioned[0]["passed"] and conditioned[0]["comparison"] == "le"
    assert start["evidence_status"] == "numerically_verified"
    assert "counterexample" in start
    prediction = mfg.separation_prediction()
    assert max(prediction["conditioned_expanded_mm"]) < max(prediction["open_loop_expanded_mm"])
    assert all(value <= 0.1 for value in prediction["linearity"].values())
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_separation({"separation_mm": [1.0], "expanded_uncertainty_mm": [0.1]}, None)
    assert refused.value.code == "measurement_absent"
    assert rec.normalized_error([1.1], [1.0], [0.06], [0.08])[0] == pytest.approx(1.0)


@pytest.mark.lab_task("T138")
def test_comparators_compute_normalized_errors_for_a_measurement_record():
    """The success path of both comparators on a synthetic measurement-kind record (never retained, never cited)."""
    data = {"targets.csv": b"station,separation\n"}
    record = _measurement_record(data)
    predicted = {"stations_mm": [0.0, 100.0, 200.0], "separation_mm": [2.0, 1.0, -1.0],
                 "expanded_uncertainty_mm": [0.08, 0.08, 0.08]}
    result = rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.03, 1.08, -1.0],
                                                "expanded_uncertainty_mm": [0.06, 0.06, 0.06]})
    assert result["normalized_error"] == pytest.approx([0.3, 0.8, 0.0], abs=1e-9) and result["agrees"] is True
    assert result["acquisition"]["raw_sha256"] == hashlib.sha256(rec.raw_manifest(record)).hexdigest()
    disagree = rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.0, 1.2, -1.0],
                                                  "expanded_uncertainty_mm": [0.06, 0.06, 0.06]})
    assert disagree["normalized_error"][1] == pytest.approx(2.0, abs=1e-9) and disagree["agrees"] is False
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.0, 1.0],
                                           "expanded_uncertainty_mm": [0.06, 0.06]})
    assert refused.value.code == "stations_mismatch"
    pairs = mfg.pair_predictions(mfg.plate_study(), mfg.cylinder_study())
    plate = pairs["MFG-FLAT-PLATE-01"]
    measured = {"record": record, "raw_bytes": data, "labels": plate["pairs"],
                "values_mm": [v + 0.01 for v in plate["values_mm"]], "expanded_uncertainty_mm": 0.05}
    result = rec.compare_pair_distances(plate, measured)
    assert result["normalized_error"] == pytest.approx([0.2] * len(plate["pairs"]), abs=1e-9) and result["agrees"]
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_pair_distances(plate, dict(measured, labels=list(reversed(plate["pairs"]))))
    assert refused.value.code == "pairs_mismatch"
    cylinder = pairs["MFG-CYLINDER-01"]
    ninety = cylinder["pairs"].index("circumferential 90 deg")
    assert cylinder["expanded_uncertainty_mm"][ninety] == pytest.approx(
        2 * abs(math.pi / 2 - 2 * math.sin(math.pi / 4)) * 0.1 / math.sqrt(3.0), rel=1e-9)
    assert cylinder["expanded_uncertainty_mm"][cylinder["pairs"].index("axial 100 mm")] == 0.0


def test_capture_reader_refuses_malformed_and_marks_synthetic_bytes():
    expected = mfg.capture_expectations("MFG-FLAT-PLATE-01")["photogrammetry"]
    data = mfg.synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")
    assert data.startswith(rec.SYNTHETIC_CAPTURE_PREFIX)
    parsed = rec.read_capture(data, expected)
    assert parsed["origin"] == "synthetic" and len(parsed["rows"]) == 25 and parsed["frame"] == "CAD"
    # An export declared as a measurement carries no banner; the declaration is the operator's, not authenticated.
    rows = {name: values for name, values in parsed["rows"].items()}
    declared = rec.write_capture(mfg.TARGET_CAPTURE, "MFG-FLAT-PLATE-01", "camera", "CAD", rows, origin="measurement")
    assert rec.read_capture(declared, expected)["origin"] == "measurement"
    assert rec.refusal_code(rec.read_capture, b"\xff\xfe", expected) == "capture_encoding"
    assert rec.refusal_code(rec.read_capture, declared.replace(b"# unit: mm", b"# unit: in"), expected) == "capture_header"
    assert rec.refusal_code(rec.read_capture, declared.replace(b"MFG-FLAT-PLATE-01", b"MFG-OTHER-01"), expected) \
        == "capture_expected"
    study = mfg.capture_reduction_study()
    assert all(case["observed"] == case["expected"] for case in study["refusals"].values()), study["refusals"]
    assert study["start_heading_error_rad"] < 1e-5 and study["conditioned_separation_error_mm"] < 1e-3
    # A cylinder capture: film minus chord recovers the gaps; the chord alone differs from the geodesic (H3).
    cylinder = mfg.compare_captures({role: mfg.synthetic_capture("MFG-CYLINDER-01", role)
                                     for role in ("photogrammetry", "film")})
    gaps = cylinder["comparisons"]["film minus chord (H1)"]
    assert gaps["max_normalized_error"] < 1e-6 and gaps["within_en_1"]
    signature = cylinder["comparisons"]["pair chord against the geodesic distance (H3)"]
    ninety = signature["labels"].index("circumferential 90 deg")
    assert signature["normalized_error"][ninety] > 1.0 and signature["normalized_error"][
        signature["labels"].index("axial 100 mm")] < 1e-6
    with pytest.raises(rec.RecordRefusal) as refused:
        mfg.compare_captures({"photogrammetry": data, "film": mfg.synthetic_capture("MFG-CYLINDER-01", "film")})
    assert refused.value.code == "capture_protocol_mismatch"


def test_operator_captures_are_compared_but_never_hardware_evidence(tmp_path):
    """T138 reads bound captures through ctx.capture, compares them, retains them, and keeps physical claims open."""
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    implementation = section_implementations("manufacturing")["T138"]
    targets, pose = tmp_path / "targets.csv", tmp_path / "pose.csv"
    targets.write_bytes(mfg.synthetic_capture("MFG-COUPON-01", "photogrammetry", 2.05, 3e-4))
    pose.write_bytes(mfg.synthetic_capture("MFG-COUPON-01", "cmm", 2.05, 3e-4))
    ctx = runner.Context(tmp_path / "run", captures={"photogrammetry": targets, "cmm": pose})
    report = validate_report(runner.run_task(queue["T138"], implementation, ctx, {}))
    assert report["state"] == "partial" and report["physical_validation_status"]["status"] == "not_established"
    for record in report["findings"]:
        if record["domain"] in PHYSICAL_DOMAINS:
            assert record["evidence_status"] == "not_established" and record["basis"] == {}
    capture = _finding(report, "Normalized errors of the bound metrology captures")
    assert capture["evidence_status"] == "numerically_verified" and "synthetic_inputs" in capture["origin"]
    assert capture["value"]["origin"] == "synthetic" and capture["basis"]["inputs"]["authenticated"] is False
    conditioned = capture["value"]["comparisons"]["separation, conditioned on the captured start pose"]
    assert conditioned["max_normalized_error"] < 0.05
    assert capture["value"]["start_pose"]["lateral_mm"] == pytest.approx(2.05, abs=1e-6)
    assert capture["value"]["start_pose"]["heading_rad"] == pytest.approx(3e-4, abs=1e-5)
    assert "crossing arclength (H1)" in capture["value"]["comparisons"]
    retained = {a["path"]: a["sha256"] for a in report["generated_artifacts"]}
    digest = hashlib.sha256(targets.read_bytes()).hexdigest()
    assert retained["artifacts/T138/capture-photogrammetry.csv"] == digest
    # The capture is unauthenticated: a physical finding citing its digest cannot pass the runner's gate.
    physical = finding("Measured separation on the coupon", "physical", 1.0, {"acquisition": {
        "device": "photogrammetry camera system:camera:SN-1", "raw_sha256": digest,
        "acquired_at": "2026-09-23T00:00:00Z", "calibration": "CERT-1"}})
    with pytest.raises(EvidenceRefusal):
        runner._gate_physical([physical], ctx)
    # A capture the reader refuses refutes the comparison claim; nothing else changes.
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"x,y\n1,2\n")
    refused = validate_report(runner.run_task(queue["T138"], implementation,
                                              runner.Context(tmp_path / "bad", captures={"photogrammetry": bad}), {}))
    record = _finding(refused, "Normalized errors of the bound metrology captures")
    assert record["evidence_status"] == "not_established" and "expected_not_established" not in record
    assert record["value"] == {"refusal": "capture_header"} and refused["evidence_status"]["primary"] == "not_established"
    # Without a capture the claim is recorded as expected-unestablished (the retained clean-room run).
    plain = validate_report(runner.run_task(queue["T138"], implementation, runner.Context(tmp_path / "plain"), {}))
    assert _finding(plain, "Normalized errors of the bound metrology captures")["expected_not_established"] is True


def _run_t138(tmp_path, name, captures):
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    files = {}
    for role, data in captures.items():
        files[role] = tmp_path / f"{name}-{role}.csv"
        files[role].write_bytes(data)
    ctx = runner.Context(tmp_path / name, captures=files)
    return validate_report(runner.run_task(queue["T138"], section_implementations("manufacturing")["T138"], ctx, {}))


def _declared(data, u=None, only=None):
    """A synthetic capture rewritten as a declared measurement (test bytes, never retained), optionally with new u."""
    parsed = rec.read_capture(data)
    rows = {name: [*values[:-1], (u if (only is None or name in only) else values[-1]) if u is not None else values[-1]]
            for name, values in parsed["rows"].items()}
    return rec.write_capture(parsed["schema"], parsed["protocol"], parsed["instrument"], parsed["frame"], rows,
                             origin="measurement")


def test_zero_uncertainty_capture_refutes_the_comparison_instead_of_blocking(tmp_path):
    """A capture whose u column is 0 would make E_n infinite or undefined: the reader refuses it, and T138 records
    the refused comparison (partial, refuted capture finding) instead of being blocked by a serialization error."""
    plate = mfg.synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")
    expected = mfg.capture_expectations("MFG-FLAT-PLATE-01")["photogrammetry"]
    for zero in (_declared(plate, 0.0), _declared(plate, 0.0, only=("M00", "M01"))):
        assert rec.refusal_code(rec.read_capture, zero, expected) == "capture_value"
    report = _run_t138(tmp_path, "zero", {"photogrammetry": _declared(plate, 0.0)})
    assert report["state"] == "partial"
    record = _finding(report, "Normalized errors of the bound metrology captures")
    assert record["value"] == {"refusal": "capture_value"} and record["evidence_status"] == "not_established"
    assert "expected_not_established" not in record and report["evidence_status"]["primary"] == "not_established"


def test_capture_roles_without_a_comparison_are_retained_not_refuted(tmp_path):
    """Roles a protocol defines for retention only (a start-pose file alone) are parsed and retained, and the
    comparison claim is recorded as expected-unestablished, not refuted."""
    pose = {"N0-S0": [-120.0, 30.0, 0.0, 0.002], "N0-S20": [-100.0, 30.0, 0.0, 0.002]}
    ids = mfg.capture_expectations("MFG-FLAT-PLATE-01")["cmm"]["ids"]
    rows = {name: pose.get(name, [0.0, 0.0, 0.0, 0.002]) for name in ids}
    cases = {"plate-cmm": {"cmm": rec.write_capture(mfg.TARGET_CAPTURE, "MFG-FLAT-PLATE-01", "cmm", "CAD", rows,
                                                    origin="measurement")},
             "coupon-cmm": {"cmm": mfg.synthetic_capture("MFG-COUPON-01", "cmm", 2.05, 3e-4)}}
    for name, captures in cases.items():
        report = _run_t138(tmp_path, name, captures)
        assert report["state"] == "partial", name
        record = _finding(report, "Normalized errors of the bound metrology captures")
        assert record["expected_not_established"] is True and not record["basis"].get("checks"), name
        assert record["value"]["comparisons"] == {} and "cmm" in record["basis"]["notes"], name
        assert report["evidence_status"]["primary"] != "not_established", name
        assert any(a["path"] == "artifacts/T138/capture-cmm.csv" for a in report["generated_artifacts"]), name


def test_crossing_comparison_carries_the_capture_uncertainty():
    """The measured crossing's U comes from the capture's u column through the interpolation; the predicted one
    from the prediction terms of the T140 focal row only."""
    captures = {role: mfg.synthetic_capture("MFG-COUPON-01", role, 2.05, 3e-4, noise_mm=0.01)
                for role in ("photogrammetry", "cmm")}
    base = mfg.compare_captures(captures)["comparisons"]["crossing arclength (H1)"]
    wider = mfg.compare_captures(dict(captures, photogrammetry=_declared(captures["photogrammetry"], 0.05)))
    wider = wider["comparisons"]["crossing arclength (H1)"]
    assert 0.0 < base["expanded_uncertainty_measured_mm"][0] < wider["expanded_uncertainty_measured_mm"][0]
    assert wider["normalized_error"][0] < base["normalized_error"][0]
    focal = mfg.budget_study()["budget"]["coupon focal distance (conditioned on the measured start pose)"]["components"]
    prediction = 2 * math.sqrt(focal["geometry"] ** 2 + focal["execution"] ** 2 + focal["solver"] ** 2)
    assert base["expanded_uncertainty_predicted_mm"][0] == pytest.approx(prediction, rel=1e-12)
    # The propagation formula against finite differences of the interpolated crossing.
    stations, values, sigmas = [0.0, 10.0, 20.0], np.array([2.0, 0.6, -0.9]), np.array([0.03, 0.02, 0.05])
    where, u = mfg._crossing(stations, values, sigmas)
    gradient = []
    for k in (1, 2):
        step = np.zeros(3)
        step[k] = 1e-6
        gradient.append((mfg._crossing(stations, values + step)[0] - mfg._crossing(stations, values - step)[0]) / 2e-6)
    assert u == pytest.approx(math.hypot(gradient[0] * sigmas[1], gradient[1] * sigmas[2]), rel=1e-6)
    assert where == pytest.approx(10.0 + 0.6 / 1.5 * 10.0)


def test_acceptance_criteria_name_only_declared_devices(section):
    directory, _ = section
    protocol = json.loads((directory / "artifacts" / "T126" / "protocol-flat-plate.json").read_text(encoding="utf-8"))
    for text in ("surface distance by tape-measure and laser interferometer equals the geodesic",
                 "the tape measure reading equals the geodesic", "an interferometer confirms the chord"):
        for key in ("statement", "test"):
            broken = deepcopy(protocol)
            broken["acceptance_criteria"][0][key] = text
            assert rec.refusal_code(rec.validate_protocol, broken) == "undeclared_device", (text, key)
    matrix = rec.protocol_refusal_matrix(protocol)
    assert matrix["criterion_names_undeclared_instrument"] == {"expected": "undeclared_device",
                                                                "observed": "undeclared_device"}


@pytest.mark.lab_task("T139")
def test_retention_schema_refusals_and_fixture_boundary(section, tmp_path):
    _, reports = section
    report = reports["T139"]
    schema = _finding(report, "The retention schema refuses")
    assert schema["value"] == 11 and schema["evidence_status"] == "numerically_verified"
    boundary = _finding(report, "A schema fixture (as is or relabelled as a measurement)")
    assert boundary["value"] == 3 and boundary["evidence_status"] == "numerically_verified"
    assert _finding(report, "The retention schema keeps a rank-deficient")["evidence_status"] == "numerically_verified"
    assert _finding(report, "A real measurement with raw bytes")["evidence_status"] == "not_established"
    assert report["state"] == "partial" and "none was bound" in report["experiment"]
    bound = _finding(report, "A bound retention record validates")
    assert bound["expected_not_established"] is True and bound["value"] is None
    assert not any(a["path"].endswith("fixture.txt") for a in report["generated_artifacts"])
    fixture, raw = mfg._schema_fixture()
    assert rec.refusal_code(rec.to_acquisition, fixture, raw) == "fixture_is_not_measurement"
    # Relabelling the fixture does not help: its bytes, serial and zero calibration digest are refused.
    assert rec.refusal_code(rec.to_acquisition, dict(fixture, record_kind="measurement"), raw) == "fixture_is_not_measurement"
    # Synthetic capture bytes are refused as hardware evidence too, whatever record lists them.
    synthetic = mfg.synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")
    listed = _measurement_record({"capture.csv": synthetic})
    assert rec.refusal_code(rec.to_acquisition, listed, {"capture.csv": synthetic}) == "fixture_is_not_measurement"
    assert rec.refusal_code(rec.validate_retention, dict(fixture, clock=dict(fixture["clock"], acquired_at="2026-09-23 00:00"))) \
        == "clock_without_timezone"
    assert rec.refusal_code(rec.validate_retention, dict(fixture, clock=dict(fixture["clock"], acquired_at="2031-01-01T00:00:00Z"))) \
        == "calibration_expired"
    # A record that is not the fixture yields acquisition fields bound to every raw file, yet the runner still refuses
    # a physical finding without a hardware probe and retained bytes: validators alone cannot mint hardware evidence.
    data = {"a.bin": b"bytes of file a", "b.bin": b"bytes of file b"}
    record = deepcopy(fixture)
    record.update(record_kind="measurement", instrument=dict(fixture["instrument"], serial="SN-1"),
                  raw=[{"name": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value),
                        "media_type": "application/octet-stream"} for name, value in data.items()])
    record["calibration"] = dict(fixture["calibration"], sha256="c" * 64)
    acquisition = rec.to_acquisition(record, data)
    assert acquisition["raw_sha256"] == hashlib.sha256(rec.raw_manifest(record)).hexdigest()
    physical = finding("probe", "physical", 1.0, {"acquisition": acquisition})
    with pytest.raises(EvidenceRefusal):
        runner._gate_physical([physical], runner.Context(tmp_path))
    jac = np.random.Generator(np.random.PCG64(1)).normal(0.0, 1000.0, (6, 3))
    lever = deepcopy(fixture)
    lever["frame_chain"][0]["covariance"] = (jac @ np.diag([1e-4] * 3) @ jac.T).tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) is None
    # The scale test: kept by the relative rule although an absolute 1e-12 rule would refuse it; a negative
    # eigenvalue at the same scale is still refused.
    covariance = mfg._rank_deficient_covariance()
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    assert eigenvalues[0] < -mfg.ABSOLUTE_THRESHOLD and eigenvalues[-1] == pytest.approx(1e6, rel=1e-9)
    lever["frame_chain"][0]["covariance"] = covariance.tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) is None
    lever["frame_chain"][0]["covariance"] = mfg._rank_deficient_covariance(-1e3).tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) == "frame_covariance_invalid"


def _run_t139(tmp_path, name, captures):
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    files = {}
    for role, data in captures.items():
        files[role] = tmp_path / f"{name}-{role}.{'json' if role == 'retention' else 'csv'}"
        files[role].write_bytes(data)
    ctx = runner.Context(tmp_path / name, captures=files)
    return validate_report(runner.run_task(queue["T139"], section_implementations("manufacturing")["T139"], ctx, {})), ctx


@pytest.mark.lab_task("T139")
def test_bound_retention_record_is_validated_against_its_raw_bytes(tmp_path):
    """T139 reads a retention record bound with --capture retention=, matches its raw entries to the bytes bound
    under the capture roles by digest, builds the acquisition fields and retains both; it completes only for a
    measurement record matching real (non-synthetic, non-fixture) bytes, and the physical claim stays open."""
    synthetic = mfg.synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")
    declared = _declared(synthetic)                        # test bytes declared as a measurement, never retained
    record = _measurement_record({"targets.csv": declared})
    record["protocol_id"] = "MFG-FLAT-PLATE-01"
    encode = lambda value: json.dumps(value).encode("utf-8")
    report, ctx = _run_t139(tmp_path, "good", {"retention": encode(record), "photogrammetry": declared})
    assert report["state"] == "completed" and "Completed:" in report["experiment"]
    bound = _finding(report, "A bound retention record validates")
    assert bound["evidence_status"] == "numerically_verified" and bound["basis"]["inputs"]["authenticated"] is False
    acquisition = bound["value"]["acquisition"]
    assert acquisition["raw_sha256"] == hashlib.sha256(rec.raw_manifest(record)).hexdigest()
    assert bound["value"]["raw_roles"] == {"targets.csv": "photogrammetry"}
    retained = {a["path"]: a["sha256"] for a in report["generated_artifacts"]}
    assert retained["artifacts/T139/capture-retention.json"] == hashlib.sha256(encode(record)).hexdigest()
    assert retained["artifacts/T139/capture-photogrammetry.csv"] == hashlib.sha256(declared).hexdigest()
    assert retained["artifacts/T139/retention-raw-manifest.json"] == acquisition["raw_sha256"]
    physical = _finding(report, "A real measurement with raw bytes")
    assert physical["evidence_status"] == "not_established" and physical["basis"] == {}
    assert report["physical_validation_status"]["status"] == "not_established"
    # Without a hardware probe the acquisition fields cannot back a physical finding, even with the bytes retained.
    with pytest.raises(EvidenceRefusal):
        runner._gate_physical([finding("Retained plate markers", "physical", 1.0, {"acquisition": acquisition})], ctx)
    # Refused: bytes that do not match, synthetic bytes, the schema fixture, a record that is not JSON.
    fixture, fixture_raw = mfg._schema_fixture()
    listed_synthetic = _measurement_record({"targets.csv": synthetic})
    cases = {"altered": ({"retention": encode(record), "photogrammetry": declared + b"\n"}, "raw_digest_mismatch"),
             "unbound": ({"retention": encode(record)}, "raw_digest_mismatch"),
             "synthetic": ({"retention": encode(listed_synthetic), "photogrammetry": synthetic},
                           "fixture_is_not_measurement"),
             "fixture": ({"retention": encode(fixture), "photogrammetry": fixture_raw["fixture.txt"]},
                         "fixture_is_not_measurement"),
             "text": ({"retention": b"not json"}, "retention_not_json")}
    for name, (captures, code) in cases.items():
        refused, _ = _run_t139(tmp_path, name, captures)
        record_finding = _finding(refused, "A bound retention record validates")
        assert refused["state"] == "partial" and record_finding["value"] == {"refusal": code}, name
        assert record_finding["evidence_status"] == "not_established" and "expected_not_established" not in record_finding
        assert _finding(refused, "A real measurement with raw bytes")["value"] == 0


@pytest.mark.lab_task("T140")
def test_uncertainty_budget_classifies_limiting_terms(section):
    _, reports = section
    report = reports["T140"]
    budget = _finding(report, "Uncertainty budget per predicted quantity")["value"]
    assert budget["cylinder gap, 90 deg pair"]["dominant"] == "instrument"
    # A measured gap is the film surface distance minus the camera chord: both instruments enter.
    assert budget["cylinder gap, 90 deg pair"]["instrument"] == pytest.approx(math.hypot(0.05, mfg.PAIR_U), rel=1e-12)
    assert budget["coupon focal distance (conditioned on the measured start pose)"]["dominant"] == "geometry"
    assert budget["coupon separation at L, 5 mrad heading offset (open-loop start)"]["dominant"] == "execution"
    assert budget["coarse-solver control (6 RK4 steps)"]["dominant"] == "solver"
    # Open loop, each tape is laid after its own seating of the start jig: the execution term carries the insert
    # and laying error and the re-seating of the jig, in quadrature; conditioned rows measure the pose instead.
    for name, row in budget.items():
        if "(open-loop start)" in name:
            terms = row["execution_terms"]
            assert set(terms) == {"insert_and_laying", "jig_reseating"} and min(terms.values()) > 0.0, name
            assert math.hypot(*terms.values()) == pytest.approx(row["execution"], rel=1e-12), name
        else:
            assert "execution_terms" not in row, name
    assert mfg.EXECUTION["lateral_mm"] == pytest.approx(math.hypot(0.05, math.sqrt(2) * 0.01), rel=1e-12)
    assert mfg.EXECUTION["heading_rad"] == pytest.approx(math.hypot(5e-4, math.sqrt(2) * 1e-4), rel=1e-12)
    plate = budget["plate separation at 240 mm, 5 mrad heading offset (open-loop start)"]
    assert plate["value"] == pytest.approx(1.2, abs=1e-12) and 0.0 < plate["geometry"] < 1e-5
    assert _finding(report, "On the coupon, the heading-offset separation")["evidence_status"] == "numerically_verified"
    # Richardson: a fine-step value carries |Q(h) - Q(2h)| / 15, the same as T128 uses for the focal point.
    focal = budget["coupon focal distance (conditioned on the measured start pose)"]
    nominal = mfg.nominal_study()
    assert focal["solver"] == pytest.approx(abs(nominal["focal_mm"] - nominal["focal_h2_mm"]) / 15.0, rel=1e-12)
    counter = _finding(report, "The coupon focal-distance prediction is geometry-limited")
    assert counter["value"] > 0.5 and "counterexample" in counter and "declared" in counter["claim"]
    assert mfg._classify({"instrument": 1.0, "geometry": 1.0, "solver": 0.0})[0] == "mixed (largest: instrument)"
    # The lateral T138 quantity is budgeted open loop too: geometry, not the start pose, limits it at L.
    assert budget["coupon separation at L, 2 mm lateral offset (open-loop start)"]["dominant"] == "geometry"
    assert "declared instrument and start-pose uncertainties" in _finding(report, "On the coupon, the heading-offset")["claim"]
    # Conditioning on the as-built scan (T129) replaces the dome tolerances and shrinks the geometry term.
    scan = _finding(report, "Conditioning on the as-built scan")
    assert scan["evidence_status"] == "numerically_verified"
    for key in ("coupon focal distance", "coupon separation at L, 2 mm lateral offset"):
        declared = budget[f"{key} (conditioned on the measured start pose)"]["geometry"]
        scanned = budget[f"{key} (conditioned on the start pose and the as-built scan)"]["geometry"]
        assert 0.0 < scanned < 0.1 * declared, key


@pytest.mark.lab_task("T141")
def test_production_acceptance_stays_outside_the_system(section):
    _, reports = section
    report = reports["T141"]
    assert report["state"] == "completed"
    api = _finding(report, "No basis establishes a claim filed in an authority domain")
    assert api["evidence_status"] == "numerically_verified"
    assert api["value"]["basis_domain_cases"] == 64 * len(AUTHORITY_DOMAINS) and api["value"]["violations"] == 0
    assert api["value"]["refusals"] == len(api["basis"]["checks"]) - 1
    refs = {c["reference"]: c for c in api["basis"]["checks"]}
    only_section = refs["section screen: 'Coupon lot scrapped', which evidence.finding accepts"]
    assert only_section["passed"] and only_section["observed_refusal"] == "acceptance_outside_authority_domain"
    # The loophole T141 recorded is closed: evidence.finding refuses the statement in a computational domain, and
    # the task records that refusal instead of an established acceptance statement.
    closed = _finding(report, "evidence.finding and validate_finding refuse an acceptance or rejection statement")
    assert closed["evidence_status"] == "numerically_verified" and closed["value"] == len(closed["basis"]["checks"]) == 4
    assert all(c["observed_refusal"] == "authority_outcome_refused" for c in closed["basis"]["checks"])
    assert "loophole" in closed["counterexample"]["statement"]
    assert not [f for f in report["findings"] if f["claim"].startswith("evidence.finding establishes")]
    passing = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
    with pytest.raises(EvidenceRefusal, match="authority outcome"):
        finding("Coupon lot accepted for production", "computational_pipeline", "accepted", {"checks": [passing]})
    # What stays open is recorded: a paraphrase outside both vocabularies is still labelled by its checks.
    paraphrase = _finding(report, "A paraphrased acceptance statement outside both screened vocabularies")
    assert paraphrase["evidence_status"] == "numerically_verified" and paraphrase["value"] == "numerically_verified"
    assert paraphrase["counterexample"]["statement"] == "The lab API cannot mark production acceptance"
    assert finding(mfg.PARAPHRASE, "computational_pipeline", 1.0, {"checks": [passing]})["evidence_status"] \
        == "numerically_verified"
    note = _finding(report, "Domain assignment of free-text claims is machine-checked")
    assert note["evidence_status"] == "not_established" and note["expected_not_established"] is True
    assert "evidence.screen_authority_claim" in note["basis"]["notes"]
    # Both screens: the core one refuses these decisions at finding(); the section one refuses the rest.
    for claim in ("Coupon lot rejected for production", "Coupon lot rejected", "Coupon lot scrapped",
                  "Coupon lot quarantined", "Coupon lot passes acceptance", "Coupon lot passed inspection",
                  "Coupon lot conforms and is released to production"):
        try:
            decision = finding(claim, "computational_pipeline", "ok", {"checks": [passing]})
        except EvidenceRefusal:
            continue
        assert rec.refusal_code(rec.screen_acceptance_language, [decision]) == "acceptance_outside_authority_domain", claim
    topic = finding("Acceptance criteria are hypotheses", "computational_pipeline", 1.0, {"checks": [passing]})
    assert rec.screen_acceptance_language([topic]) == ["Acceptance criteria are hypotheses"]
    assert _finding(report, "Production acceptance of the coupon")["evidence_status"] == "not_established"
    policy = rec.AcceptancePolicy()
    with pytest.raises(rec.RecordRefusal) as refused:
        policy.decide({"part": "coupon-001", "decision": "accept"})
    assert refused.value.code == "production_acceptance_outside_system"
    assert policy.record({"part": "x"})["decision"] == "not_performed"
    for task_id, report in reports.items():
        for record in report["findings"]:
            if record["domain"] == "production_acceptance":
                assert record["evidence_status"] == "not_established", task_id
