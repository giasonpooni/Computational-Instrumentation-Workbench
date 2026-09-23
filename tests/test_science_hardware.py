"""CIWT hardware telemetry is observation-only, loss-accounted, labelled and replayable."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import struct

import pytest

from ciw.science import hardware as hw
from ciw.science._common import Refusal
from ciw.science.frames import FrameRegistry
from ciw.science.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "science" / "hardware"
LAYOUT_RECORD = json.loads((EXAMPLES / "layouts.json").read_text(encoding="utf-8"))
LAYOUTS = hw.validate_layouts(LAYOUT_RECORD)
REGISTRY = json.loads((EXAMPLES / "frames.json").read_text(encoding="utf-8"))
OPTIONS = {"tick_rate_hz": 1e8, "device_clock": "fpga_counter"}
A, B = bytes(range(32)), bytes(range(1, 33))


def packets(count=8, **options):
    return hw.synthetic_packets(LAYOUTS, seed=7, count=count, **options)


def parse(data, **options):
    return hw.parse_stream(data, LAYOUTS, **{**OPTIONS, **options})


def codes(capture):
    return [item["code"] for item in capture.anomalies]


def imu(sequence, ticks, bitstream=A, raw=(1000, -2000, 9810, 123456), **options):
    return hw.encode_packet(1, sequence, ticks, bitstream, LAYOUTS.pack(1, list(raw)), **options)


def fixed_now():
    return datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------- layouts and packets
@pytest.mark.parametrize("mutate,code", [
    (lambda r: r["frames"]["1"]["fields"][0].update(format="x"), "unknown_format"),
    (lambda r: r["frames"]["1"]["fields"][0].update(unit="furlong"), "unknown_unit"),
    (lambda r: r["frames"]["1"]["fields"][0].update(scale=0), "out_of_domain"),
    (lambda r: r["frames"]["1"]["fields"][1].update(name="accel_x"), "duplicate_field"),
    (lambda r: r["frames"].update({"01": r["frames"].pop("1")}), "malformed_record"),
    (lambda r: r["frames"]["2"].update(name="imu"), "duplicate_frame"),
    (lambda r: r.update(schema="ciw.telemetry-layouts.v0"), "unsupported_schema"),
])
def test_layouts_are_validated_not_repaired(mutate, code):
    record = json.loads(json.dumps(LAYOUT_RECORD))
    mutate(record)
    with pytest.raises(Refusal) as caught:
        hw.validate_layouts(record)
    assert caught.value.code == code
    assert LAYOUTS.to_json() == LAYOUT_RECORD


def test_packet_round_trip_decodes_units_and_scales():
    packet = imu(5, 250_000_000)
    magic, version, flags, frame_id, sequence, ticks, bitstream, length = hw.HEADER.unpack_from(packet)
    assert (magic, version, flags, frame_id, sequence, ticks, bitstream, length) == (
        b"CIWT", 1, 0, 1, 5, 250_000_000, A, 10)
    assert len(packet) == hw.HEADER.size + 10 + 4
    capture = parse(packet)
    assert capture.anomalies == () and capture.admissible
    (record,) = capture.records
    assert record["frame"] == "imu" and record["sequence"] == 5 and record["time_s"] == pytest.approx(2.5)
    assert record["bitstream"] == "sha256:" + A.hex()
    assert hw.bitstream_identity(b"image") == "sha256:" + __import__("hashlib").sha256(b"image").hexdigest()
    values = record["values"]
    assert values["accel_x"] == {"raw": 1000, "value": pytest.approx(1.0), "unit": "m/s^2"}
    assert values["accel_z"]["value"] == pytest.approx(9.81)
    assert values["gyro_z"] == {"raw": 123456, "value": pytest.approx(0.123456), "unit": "rad/s"}
    assert capture.summary["acquisition"] == "undeclared"
    assert capture.summary["bytes"] == {"packets": len(packet), "corrupt": 0, "skipped": 0, "truncated": 0}


def test_every_declared_format_decodes():
    payload = LAYOUTS.pack(3, [-7_000_000_000, 2 ** 63, -5, 1500, 2.5])
    capture = parse(hw.encode_packet(3, 0, 0, A, payload) + hw.encode_packet(2, 1, 10, A, LAYOUTS.pack(2, [300.5, 33000, 1])))
    adc, thermal = capture.records
    assert adc["values"]["sample"]["value"] == pytest.approx(-7.0)
    assert adc["values"]["sample_index"]["raw"] == 2 ** 63
    assert adc["values"]["current"] == {"raw": 1500, "value": pytest.approx(1.5), "unit": "mA"}
    assert thermal["values"]["die_temperature"]["value"] == pytest.approx(300.5)
    assert thermal["values"]["supply_voltage"]["value"] == pytest.approx(3.3)


def test_non_finite_payload_values_are_reported_not_emitted():
    capture = parse(hw.encode_packet(2, 0, 0, A, LAYOUTS.pack(2, [float("nan"), 1, 0])))
    assert codes(capture) == ["non_finite_value"]
    assert capture.records[0]["values"]["die_temperature"] == {"raw": None, "value": None, "unit": "K"}
    json.dumps(capture.to_json(), allow_nan=False)


def test_synthetic_streams_are_deterministic_and_labelled():
    first, second = hw.synthetic_stream(LAYOUTS, seed=3, count=30), hw.synthetic_stream(LAYOUTS, seed=3, count=30)
    assert first == second and first != hw.synthetic_stream(LAYOUTS, seed=4, count=30)
    capture = parse(first)
    assert capture.summary["acquisition"] == "synthetic" and capture.summary["packets_decoded"] == 30
    assert capture.summary["packets_per_frame"] == {"imu": 10, "thermal": 10, "adc": 10}
    assert all(record["flags"] & hw.FLAG_SYNTHETIC for record in capture.records)
    with pytest.raises(Refusal) as caught:
        parse(first, acquisition="physical")
    assert caught.value.code == "acquisition_conflict"
    assert parse(imu(0, 0), acquisition="physical").summary["acquisition"] == "physical"
    mixed = parse(imu(0, 0) + imu(1, 1, flags=hw.FLAG_SYNTHETIC))
    assert mixed.summary["acquisition"] == "synthetic" and "synthetic_flag_mixed" in codes(mixed)
    assert not mixed.admissible


# ---------------------------------------------------------------------- sequence accounting
def test_loss_detection_across_u32_wraparound():
    stream = packets(8, start_sequence=2 ** 32 - 3)
    clean = parse(b"".join(stream))
    assert clean.summary["lost"] == 0 and clean.summary["sequence_wraps"] == 1 and clean.anomalies == ()
    kept = [packet for index, packet in enumerate(stream) if index not in (2, 3, 6)]
    capture = parse(b"".join(kept))
    assert capture.summary["lost"] == 3
    assert capture.summary["gaps"] == [[2 ** 32 - 1, 0], [3, 3]]
    gaps = [item for item in capture.anomalies if item["code"] == "sequence_gap"]
    assert [(item["first"], item["last"], item["lost"]) for item in gaps] == [(2 ** 32 - 1, 0, 2), (3, 3, 1)]
    assert capture.summary["reordered"] == 0 and capture.summary["duplicates"] == 0


def test_duplicates_are_reported_and_not_emitted_twice():
    stream = packets(3)
    capture = parse(stream[0] + stream[1] + stream[1] + stream[2])
    assert [record["sequence"] for record in capture.records] == [0, 1, 2]
    (duplicate,) = [item for item in capture.anomalies if item["code"] == "duplicate"]
    assert duplicate["sequence"] == 1 and duplicate["identical"] is True
    assert capture.summary["duplicates"] == 1 and capture.summary["lost"] == 0
    collision = parse(imu(0, 0) + imu(0, 0, raw=(1, 2, 3, 4)))
    assert [item["identical"] for item in collision.anomalies if item["code"] == "duplicate"] == [False]


def test_reordering_fills_the_gap_it_opened():
    stream = packets(5)
    capture = parse(stream[0] + stream[1] + stream[3] + stream[2] + stream[4])
    assert codes(capture) == ["sequence_gap", "reordered"]
    assert capture.anomalies[1]["fills_gap"] is True
    assert capture.summary["lost"] == 0 and capture.summary["gaps"] == [] and capture.summary["reordered"] == 1
    assert [record["late"] for record in capture.records] == [False, False, False, True, False]
    partial = parse(stream[0] + stream[4] + stream[2])
    assert partial.summary["lost"] == 2 and partial.summary["gaps"] == [[1, 1], [3, 3]]


def test_non_monotonic_timestamps_are_flagged():
    capture = parse(imu(0, 500) + imu(1, 400) + imu(2, 400) + imu(3, 900))
    (item,) = [item for item in capture.anomalies if item["code"] == "non_monotonic_timestamp"]
    assert (item["sequence"], item["ticks"], item["previous_ticks"]) == (1, 400, 500)


# ---------------------------------------------------------------------- corruption and framing
def test_crc_corruption_is_detected_and_counted_as_loss():
    stream = packets(4)
    damaged = bytearray(stream[1])
    damaged[hw.HEADER.size + 2] ^= 0xFF
    data = stream[0] + bytes(damaged) + stream[2] + stream[3]
    capture = parse(data)
    assert codes(capture) == ["crc_error", "sequence_gap"]
    assert capture.summary["crc_errors"] == 1 and capture.summary["lost"] == 1
    assert [record["sequence"] for record in capture.records] == [0, 2, 3]
    assert capture.summary["bytes"]["corrupt"] == len(damaged)
    assert sum(capture.summary["bytes"].values()) == len(data)


def test_corrupt_length_field_resynchronizes():
    stream = packets(4)
    damaged = bytearray(stream[1])
    damaged[hw.HEADER.size - 1] ^= 0x01
    capture = parse(stream[0] + bytes(damaged) + stream[2] + stream[3])
    assert {"crc_error", "length_overrun"} & set(codes(capture))
    assert [record["sequence"] for record in capture.records] == [0, 2, 3]


def test_garbage_between_packets_is_skipped_and_counted():
    stream = packets(3)
    data = stream[0] + b"\x00\xffgarbage" + stream[1] + b"CI" + stream[2]
    capture = parse(data)
    bad = [item for item in capture.anomalies if item["code"] == "bad_magic"]
    assert [item["skipped"] for item in bad] == [9, 2]
    assert capture.summary["bytes"]["skipped"] == 11 and capture.summary["packets_decoded"] == 3
    assert capture.summary["lost"] == 0
    assert codes(parse(b"noise only")) == ["bad_magic"]
    assert parse(b"noise only").summary["inadmissible_reasons"] == ["no_packets"]


@pytest.mark.parametrize("cut", [30, -3])
def test_truncated_trailing_packet(cut):
    stream = packets(3)
    data = stream[0] + stream[1] + stream[2][:cut]
    capture = parse(data)
    assert codes(capture) == ["truncated"] and len(capture.records) == 2
    assert capture.summary["bytes"]["truncated"] == len(stream[2][:cut])


def test_unknown_frame_and_length_mismatch_are_accounted():
    data = imu(0, 0) + hw.encode_packet(99, 1, 1, A, b"\x01\x02") + hw.encode_packet(1, 2, 2, A, b"\x00" * 9) + imu(3, 3)
    capture = parse(data)
    assert codes(capture) == ["unknown_frame", "length_mismatch"]
    mismatch = capture.anomalies[1]
    assert (mismatch["expected"], mismatch["declared"]) == (10, 9)
    assert capture.summary["packets_valid"] == 4 and capture.summary["packets_decoded"] == 2
    assert capture.summary["lost"] == 0


def test_unsupported_version_is_skipped():
    capture = parse(imu(0, 0, version=2) + imu(1, 1))
    assert codes(capture) == ["unsupported_version"] and len(capture.records) == 1


def test_bitstream_change_and_mismatch_make_capture_inadmissible():
    first = packets(3, bitstream=A)
    second = packets(3, bitstream=B, start_sequence=3, start_ticks=3000)
    data = b"".join(first + second)
    changed = parse(data)
    assert "bitstream_changed" in codes(changed) and not changed.admissible
    assert changed.summary["inadmissible_reasons"] == ["bitstream_changed"]
    assert [item["packets"] for item in changed.summary["bitstreams"]] == [3, 3]
    mismatch = parse(data, expected_bitstream="sha256:" + A.hex())
    assert codes(mismatch).count("bitstream_mismatch") == 1
    assert set(mismatch.summary["inadmissible_reasons"]) == {"bitstream_mismatch", "bitstream_changed"}
    wrong = parse(b"".join(second), expected_bitstream=A)
    assert wrong.summary["inadmissible_reasons"] == ["bitstream_mismatch"]
    assert parse(b"".join(first), expected_bitstream=A).admissible


def test_clock_alignment_through_frame_registry():
    registry = FrameRegistry.from_json(REGISTRY)
    capture = parse(b"".join(packets(4, ticks_per_packet=50_000_000)), frames=registry, target_clock="utc")
    mapping = REGISTRY["clock_mappings"][0]
    for record in capture.records:
        aligned = record["aligned"]
        assert aligned["clock"] == "utc" and aligned["mappings"] == ["fpga-to-utc"]
        assert aligned["time_s"] == pytest.approx(mapping["offset_s"] + mapping["rate"] * record["time_s"], abs=1e-6)
        assert aligned["variance_s2"] > 0
    assert capture.summary["time_span"]["aligned"]["clock"] == "utc"
    stale = parse(imu(0, 90_000 * 10 ** 8), frames=registry, target_clock="utc")
    assert codes(stale) == ["clock_unaligned"] and stale.records[0]["aligned"] is None
    for options, code in [({"frames": registry}, "malformed_record"),
                          ({"frames": registry, "target_clock": "tai"}, "unknown_clock"),
                          ({"frames": registry, "target_clock": "utc", "device_clock": "cpu"}, "unknown_clock")]:
        with pytest.raises(Refusal) as caught:
            parse(imu(0, 0), **options)
        assert caught.value.code == code


def test_input_and_packet_count_are_bounded(monkeypatch):
    data = b"".join(packets(3))
    monkeypatch.setattr(hw, "MAX_INPUT_BYTES", 100)
    with pytest.raises(Refusal) as caught:
        parse(data)
    assert caught.value.code == "oversized_input"
    monkeypatch.setattr(hw, "MAX_INPUT_BYTES", 1 << 20)
    monkeypatch.setattr(hw, "MAX_PACKETS", 2)
    with pytest.raises(Refusal) as caught:
        parse(data)
    assert caught.value.code == "oversized_input"


def test_resynchronization_work_is_bounded():
    fake = b"".join(hw.HEADER.pack(b"CIWT", 1, 0, 1, index, 0, A, 0xFFFF) + b"\0" * 4 for index in range(200))
    with pytest.raises(Refusal) as caught:
        parse(fake + b"\0" * 70_000)
    assert caught.value.code == "oversized_input"


# ---------------------------------------------------------------------- buffering, replay, retention
def test_capture_buffer_records_every_drop():
    buffer = hw.CaptureBuffer(3)
    assert [buffer.push(item) for item in range(5)] == [None, None, None, 0, 1]
    assert buffer.drain(2) == [2, 3]
    assert buffer.extend(range(5, 10)) == 3
    assert buffer.drain() == [7, 8, 9]
    stats = buffer.stats()
    assert stats["dropped_ranges"] == [[0, 1], [4, 6]] and stats["dropped"] == 5 and not stats["lossless"]
    assert stats["pushed"] == stats["drained"] + stats["dropped"] + stats["size"] == 10
    assert stats["high_water"] == 3
    with pytest.raises(Refusal):
        hw.CaptureBuffer(0)


def test_replay_reproduces_or_diverges():
    data = b"".join(packets(6))
    capture = parse(data)
    assert capture.digest == parse(data).digest
    assert hw.replay(data, LAYOUTS, retained_digest=capture.digest, **OPTIONS)["status"] == "reproduced"
    tampered = bytearray(data)
    tampered[-1] ^= 0x01
    result = hw.replay(bytes(tampered), LAYOUTS, retained_digest=capture.digest, **OPTIONS)
    assert result["status"] == "diverged" and result["replayed_digest"] != capture.digest
    rescaled = json.loads(json.dumps(LAYOUT_RECORD))
    rescaled["frames"]["1"]["fields"][0]["scale"] = 0.002
    assert hw.replay(data, rescaled, retained_digest=capture.digest, **OPTIONS)["status"] == "diverged"


def test_retained_example_replays_on_every_platform():
    reference = json.loads((EXAMPLES / "synthetic-capture.json").read_text(encoding="utf-8"))
    data = (EXAMPLES / reference["capture"]).read_bytes()
    result = hw.replay(data, LAYOUTS, retained_digest=reference["retained_digest"], **reference["parse"])
    assert result["status"] == "reproduced" and result["input_identity"] == reference["input_identity"]
    summary = parse(data).summary
    assert {key: summary[key] for key in reference["expected"]} == reference["expected"]


def test_retain_capture_writes_a_verifiable_entry(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger", now=fixed_now)
    data = b"".join(packets(6)[:2] + packets(6)[3:])
    capture = parse(data)
    entry = hw.retain_capture(ledger, data, capture)
    assert entry["kind"] == "hardware_capture" and entry["blobs"] == [entry["body"]["blob"]]
    assert ledger.get_blob(entry["body"]["blob"]) == data
    summary = entry["body"]["summary"]
    assert summary["acquisition"] == "synthetic" and summary["lost"] == 1 and summary["digest"] == capture.digest
    reopened = Ledger.open(tmp_path / "ledger")
    assert reopened.verify().ok
    receipt = hw.replay_retained(reopened, entry["entry_id"], LAYOUTS, record=True)
    assert receipt["status"] == "reproduced" and reopened.get(receipt["receipt"])["refs"] == [entry["entry_id"]]
    assert Ledger.open(tmp_path / "ledger").verify().ok
    with pytest.raises(Refusal) as caught:
        hw.retain_capture(ledger, data + b"\x00", capture)
    assert caught.value.code == "capture_input_mismatch"
    other = json.loads(json.dumps(LAYOUT_RECORD))
    other["description"] = "different layouts"
    with pytest.raises(Refusal) as caught:
        hw.replay_retained(reopened, entry["entry_id"], other)
    assert caught.value.code == "layouts_mismatch"


# ---------------------------------------------------------------------- control
def test_control_is_always_refused():
    empty = hw.request_control({"op": "write_register", "address": 16, "value": 1}, {})
    assert empty["status"] == "refused" and empty["code"] == "control_path_unbound"
    assert [item["claimed"] for item in empty["prerequisites"]] == [False] * 7
    assert empty["unclaimed"] == list(hw.CONTROL_PREREQUISITES) and empty["device_writes"] == 0
    everything = {name: True for name in hw.CONTROL_PREREQUISITES}
    full = hw.request_control({"op": "load_bitstream"}, {**everything, "note": "please"})
    assert full["status"] == "refused" and full["code"] == "control_path_unbound" and full["all_claimed"]
    assert all(item["claimed"] and not item["verified"] for item in full["prerequisites"])
    assert full["reasons"] == ["control_path_unbound"] and full["unrecognized_context"] == ["note"]
    assert full["control_paths"] == [] and full["device_writes"] == 0
    malformed = hw.request_control("reboot", None)
    assert malformed["status"] == "refused" and {"malformed_command", "malformed_context"} <= set(malformed["reasons"])
    assert struct.calcsize("<4sBBHIQ32sH") == hw.HEADER.size == 54
