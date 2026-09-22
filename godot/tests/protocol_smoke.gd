extends SceneTree
## Run with a live ciw serve: godot --headless --path godot --script res://tests/protocol_smoke.gd
## Two actual WebSocket clients prove selection broadcast delivery to an observer.
const Client = preload("res://scripts/ciw_client.gd")
var _writer = Client.new()
var _observer = Client.new()
var _started := 0
var _runs := 0
var _requested := false
var _broadcast_seen := false
var _sample_seen := false
var _target := 1.5
var _initial_interval: Array = []
var _original_cursor := 0.0
var _restoring := false
var _reconnecting := false
var _expected_revision := -1
var _workbench_requested := false
var _source_count := 0


func _initialize() -> void:
	_started = Time.get_ticks_msec()
	root.add_child(_writer)
	root.add_child(_observer)
	for client in [_writer, _observer]:
		client.run_received.connect(_on_run)
		client.request_failed.connect(func(code, detail): _fail("%s: %s" % [code, detail]))
	_writer.sample_received.connect(_on_sample)
	_observer.selection_received.connect(_on_observer_selection)
	_observer.sample_received.connect(_on_reconnected_sample)
	_observer.snapshot_received.connect(_on_workbench_snapshot)
	_writer.connect_service()
	_observer.connect_service()


func _process(_delta: float) -> bool:
	if Time.get_ticks_msec() - _started > 15000:
		_fail("Timed out waiting for live protocol exchange; run ciw serve first")
	if _broadcast_seen and _sample_seen and not _restoring:
		_restoring = true
		_writer.update_selection({"cursor_s": _original_cursor})
	return false


func _on_run(run: Dictionary) -> void:
	if not run.has("render") or not run.has("channels") or run.time_s.is_empty():
		_fail("run.get did not return the full numerical record and render data")
		return
	if run.render.sample_indices.size() != run.render.trajectory.size():
		_fail("Render points must map to retained sample indices")
		return
	_runs += 1
	if _runs == 2 and not _requested:
		_requested = true
		_original_cursor = float(_writer.selection.cursor_s)
		_initial_interval = _writer.selection.interval_s.duplicate()
		_target = minf(1.5, float(run.metadata.duration_s) * 0.5)
		if is_equal_approx(_target, _original_cursor):
			_target *= 0.5
		_writer.update_selection({"cursor_s": _target})


func _on_observer_selection(selection: Dictionary) -> void:
	if _restoring and not _reconnecting and is_equal_approx(float(selection.cursor_s), _original_cursor):
		_reconnecting = true
		_expected_revision = int(selection.revision)
		_observer.connect_service.call_deferred()
		return
	if _requested and is_equal_approx(float(selection.cursor_s), _target):
		if selection.interval_s != _initial_interval:
			_fail("Cursor update changed the analysis interval")
			return
		_broadcast_seen = true


func _on_sample(sample: Dictionary) -> void:
	if not _requested or _writer.run.is_empty():
		return
	if not is_equal_approx(float(_writer.selection.cursor_s), _target):
		return
	var index := int(sample.sample_index)
	var run: Dictionary = _writer.run
	if sample.run_id != run.run_id or sample.evidence_id != run.evidence_id:
		_fail("Sample provenance does not match run")
		return
	if not is_equal_approx(float(sample.time_s), float(run.time_s[index])):
		_fail("Sample time does not match retained record")
		return
	for channel in ["q", "v", "energy"]:
		if not is_equal_approx(float(sample.values[channel]), float(run.channels[channel].values[index])):
			_fail("Inspection must use backend retained values")
			return
	_sample_seen = true


func _on_reconnected_sample(sample: Dictionary) -> void:
	if not _reconnecting or _observer.status != "ready":
		return
	if int(_observer.selection.revision) != _expected_revision:
		_fail("Reconnect should preserve the authoritative selection revision")
		return
	if sample.run_id != _observer.run.run_id:
		_fail("Reconnected inspection must retain correct run identity")
		return
	if not _workbench_requested:
		_workbench_requested = true
		_source_count = _observer.snapshot.workbench.sources.size()
		var raw := FileAccess.get_file_as_bytes("res://../examples/calibrated-window/source.json")
		_writer._request("source.add", {"kind": "calibrated-window", "label": "Godot protocol source",
			"bytes_b64": Marshalls.raw_to_base64(raw)})
		# Prove the event refreshes immediately, without waiting for the heartbeat.
		_observer._next_heartbeat = Time.get_ticks_msec() + 60000


func _on_workbench_snapshot(value: Dictionary) -> void:
	if _workbench_requested and value.workbench.sources.size() > _source_count:
		print("PASS: full run, response correlation, cross-client broadcast, backend sample, independent interval, same-revision reconnect and live workbench invalidation; cursor restored")
		quit(0)


func _fail(message: String) -> void:
	push_error(message)
	quit(1)
