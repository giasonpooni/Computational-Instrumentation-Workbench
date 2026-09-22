extends Node
## The service owns every scientific value and the shared selection.
## Requests are correlated even when broadcasts arrive between their replies.

signal status_changed(state: String, detail: String)
signal snapshot_received(snapshot: Dictionary)
signal run_received(run: Dictionary)
signal selection_received(selection: Dictionary)
signal sample_received(sample: Dictionary)
signal request_failed(code: String, detail: String)
signal experiment_received(view: Dictionary)
signal artifact_received(artifact: Dictionary)

const ENDPOINT := "ws://127.0.0.1:8765"
const REQUEST_TIMEOUT_MS := 8000
const CURSOR_PERIOD_MS := 100

var endpoint := ENDPOINT
var selection: Dictionary = {}
var snapshot: Dictionary = {}
var run: Dictionary = {}
var online := false
var status := "disconnected"
var _peer := WebSocketPeer.new()
var _pending: Dictionary = {}
var _sequence := 0
var _generation := 0
var _last_state := WebSocketPeer.STATE_CLOSED
var _connect_started := 0
var _last_cursor_sent := 0
var _next_heartbeat := 0
var _queued_selection: Dictionary = {}
var _selection_pending := false
var _refresh_pending := false
var _run_pending := false
var _sample_pending := false
var _wanted_sample_time := -1.0
var _sent_sample_time := -1.0
var _session_id := ""
var _snapshot_dirty := false
var _view_pending := false
var _wanted_bundle := ""
var _artifact_pending := false
var _wanted_result := ""


func connect_service() -> void:
	if _peer.get_ready_state() == WebSocketPeer.STATE_OPEN:
		_peer.close(1000, "Refreshing viewport connection")
	_peer = WebSocketPeer.new()
	# The demo includes full retained values plus render arrays in one JSON packet.
	_peer.inbound_buffer_size = 32 * 1024 * 1024
	_peer.outbound_buffer_size = 1024 * 1024
	_pending.clear()
	_queued_selection.clear()
	_selection_pending = false
	_refresh_pending = false
	_run_pending = false
	_sample_pending = false
	_view_pending = false
	_artifact_pending = false
	_wanted_bundle = ""
	_wanted_result = ""
	_snapshot_dirty = false
	_wanted_sample_time = -1.0
	_sent_sample_time = -1.0
	_generation += 1
	online = false
	_last_state = WebSocketPeer.STATE_CONNECTING
	_connect_started = Time.get_ticks_msec()
	_set_status("connecting", "Connecting to %s" % endpoint)
	var error := _peer.connect_to_url(endpoint)
	if error != OK:
		_disconnect("Connection could not start (%s). Start ciw serve, then reconnect." % error)


func refresh() -> void:
	if not online:
		connect_service()
		return
	_request_snapshot()
	_request_run()


func update_selection(changes: Dictionary) -> void:
	if not online or selection.is_empty():
		return
	for key in changes:
		if key in ["cursor_s", "channel", "interval_s"]:
			_queued_selection[key] = changes[key]


func inspect(time_s: float) -> void:
	_wanted_sample_time = time_s
	_send_sample_if_ready()


func inspect_experiment(bundle_id: String) -> void:
	_wanted_bundle = bundle_id
	if online and not _view_pending and not bundle_id.is_empty():
		_view_pending = true
		_request("experiment.inspect", {"bundle_id": bundle_id})


func inspect_artifact(result_id: String) -> void:
	_wanted_result = result_id
	if online and not _artifact_pending and not result_id.is_empty():
		_artifact_pending = true
		_request("result.get", {"result_id": result_id})


func _process(_delta: float) -> void:
	_peer.poll()
	var state := _peer.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		if _last_state != state:
			online = true
			_set_status("loading", "Connected; fetching the authoritative session")
			_request_snapshot()
			_next_heartbeat = Time.get_ticks_msec() + 5000
		while _peer.get_available_packet_count() > 0:
			var raw := _peer.get_packet().get_string_from_utf8()
			var decoded: Variant = JSON.parse_string(raw)
			if decoded is Dictionary:
				_receive(decoded)
			else:
				_set_status("error", "Service returned invalid JSON; reconnect to refresh")
		var now := Time.get_ticks_msec()
		for request_id in _pending.keys():
			if now - int(_pending[request_id].sent_ms) > REQUEST_TIMEOUT_MS:
				_disconnect("Service response timed out. Retained views are stale; reconnect.")
				break
		if online and now >= _next_heartbeat:
			_request_snapshot()
			_next_heartbeat = now + 5000
		if online and not _selection_pending and not _refresh_pending:
			if not _queued_selection.is_empty() and now - _last_cursor_sent >= CURSOR_PERIOD_MS:
				var payload := _queued_selection.duplicate(true)
				payload.expected_revision = int(selection.get("revision", 0))
				_queued_selection.clear()
				_selection_pending = true
				_last_cursor_sent = now
				_request("selection.update", payload)
	elif state == WebSocketPeer.STATE_CLOSED and _last_state != state:
		_disconnect("Service disconnected. Retained views are stale; start ciw serve and reconnect.")
	elif state == WebSocketPeer.STATE_CONNECTING:
		if Time.get_ticks_msec() - _connect_started > REQUEST_TIMEOUT_MS:
			_disconnect("Connection timed out. Start ciw serve, then reconnect.")
	_last_state = _peer.get_ready_state()


func _request(kind: String, payload: Dictionary = {}) -> String:
	if not online:
		return ""
	_sequence += 1
	var request_id := "godot-%s-%s-%s" % [get_instance_id(), _generation, _sequence]
	_pending[request_id] = {"type": kind, "payload": payload, "sent_ms": Time.get_ticks_msec()}
	var message := {"protocol_version": 1, "request_id": request_id, "type": kind, "payload": payload}
	var error := _peer.send_text(JSON.stringify(message))
	if error != OK:
		_pending.erase(request_id)
		_disconnect("Could not send request. Retained views are stale; reconnect.")
	return request_id


func _request_snapshot() -> void:
	if not _refresh_pending:
		_refresh_pending = true
		_request("session.get")


func _request_run() -> void:
	if not _run_pending:
		_run_pending = true
		_request("run.get")


func _send_sample_if_ready() -> void:
	if online and not _sample_pending and _wanted_sample_time >= 0.0:
		if not is_equal_approx(_wanted_sample_time, _sent_sample_time):
			_sample_pending = true
			_sent_sample_time = _wanted_sample_time
			_request("sample.get", {"time_s": _sent_sample_time})


func _receive(message: Dictionary) -> void:
	if int(message.get("protocol_version", 0)) != 1:
		_set_status("error", "Unsupported service protocol version")
		return
	var kind := str(message.get("type", ""))
	var payload: Variant = message.get("payload", {})
	if not payload is Dictionary:
		_set_status("error", "Malformed service payload")
		return
	if kind == "session.snapshot":
		_apply_snapshot(payload)
		return
	if kind == "selection.changed":
		_apply_selection(payload)
		return
	if kind == "workbench.changed":
		# Coalesce invalidations but remember one received during a read.
		_snapshot_dirty = true
		if not _refresh_pending:
			_snapshot_dirty = false
			_request_snapshot()
		return
	var request_id: Variant = message.get("request_id")
	if request_id == null or not _pending.has(str(request_id)):
		return
	var request: Dictionary = _pending[str(request_id)]
	_pending.erase(str(request_id))
	var request_type := str(request.type)
	if request_type == "selection.update":
		_selection_pending = false
	if request_type == "session.get":
		_refresh_pending = false
	if request_type == "run.get":
		_run_pending = false
	if request_type == "sample.get":
		_sample_pending = false
	if request_type == "experiment.inspect":
		_view_pending = false
		if request.payload.bundle_id != _wanted_bundle:
			inspect_experiment(_wanted_bundle)
			return
	if request_type == "result.get":
		_artifact_pending = false
		if request.payload.result_id != _wanted_result:
			inspect_artifact(_wanted_result)
			return
	if kind == "error":
		var code := str(payload.get("code", "unknown"))
		var detail := str(payload.get("message", "Request failed"))
		request_failed.emit(code, detail)
		if code == "revision_conflict":
			# Do not replay an obsolete user intent over another client's selection.
			_queued_selection.clear()
			_request_snapshot()
		else:
			_set_status("error", "%s: %s" % [code, detail])
		return
	if kind != "response":
		return
	match request_type:
		"session.get":
			_apply_snapshot(payload)
			if _snapshot_dirty:
				_snapshot_dirty = false
				_request_snapshot()
		"experiment.inspect":
			if payload.get("bundle_id", "") == _wanted_bundle:
				experiment_received.emit(payload)
		"result.get":
			artifact_received.emit(payload)
		"run.get":
			if payload.get("run_id", "") != snapshot.get("run", {}).get("run_id", ""):
				_set_status("error", "Run identity changed while loading; reconnect")
				return
			run = payload
			run_received.emit(run)
			_set_status("ready", "Synchronized with the Python service")
			_sent_sample_time = -1.0
			inspect(float(selection.get("cursor_s", 0.0)))
		"selection.update":
			_apply_selection(payload)
		"sample.get":
			if payload.get("run_id", "") == selection.get("run_id", ""):
				if is_equal_approx(float(request.payload.time_s), _wanted_sample_time):
					sample_received.emit(payload)
			_send_sample_if_ready()


func _apply_snapshot(value: Dictionary) -> void:
	if not value.has("session_id") or not value.has("run") or not value.has("selection"):
		_set_status("error", "Incomplete session snapshot")
		return
	var new_session := str(value.session_id)
	var new_run := str(value.run.get("run_id", ""))
	var changed := new_session != _session_id or new_run != str(selection.get("run_id", ""))
	if not changed and int(value.get("workbench", {}).get("revision", 0)) < int(snapshot.get("workbench", {}).get("revision", 0)):
		return
	if changed:
		selection.clear()
		run.clear()
		_queued_selection.clear()
		_sent_sample_time = -1.0
	_session_id = new_session
	snapshot = value
	snapshot_received.emit(snapshot)
	_apply_selection(value.selection)
	# A reconnect can retain the same revision; still refresh stale inspection.
	inspect(float(selection.get("cursor_s", 0.0)))
	if run.is_empty():
		_request_run()
	elif status != "ready":
		_set_status("ready", "Synchronized with the Python service")


func _apply_selection(value: Dictionary) -> void:
	if not value.has("revision") or not value.has("cursor_s"):
		return
	if not selection.is_empty():
		if value.get("run_id", "") != selection.get("run_id", ""):
			_request_snapshot()
			return
		if int(value.revision) <= int(selection.revision):
			return
	selection = value.duplicate(true)
	selection_received.emit(selection)
	inspect(float(selection.cursor_s))


func _set_status(state: String, detail: String) -> void:
	status = state
	status_changed.emit(state, detail)


func _disconnect(detail: String) -> void:
	online = false
	if _peer.get_ready_state() == WebSocketPeer.STATE_OPEN:
		_peer.close(1000, "Viewport disconnected")
	_peer = WebSocketPeer.new()
	_pending.clear()
	_selection_pending = false
	_refresh_pending = false
	_run_pending = false
	_sample_pending = false
	_set_status("disconnected", detail)
