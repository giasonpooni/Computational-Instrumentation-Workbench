extends SceneTree
## UI/protocol fixtures only; numerical projection is tested against real providers in Python.
const View = preload("res://scripts/experiment_view.gd")

class Reader extends "res://scripts/ciw_client.gd":
	var requests: Array = []
	func _request(kind: String, payload: Dictionary = {}) -> String:
		_sequence += 1
		var id := "test-%s" % _sequence
		_pending[id] = {"type": kind, "payload": payload, "sent_ms": Time.get_ticks_msec()}
		requests.append({"id": id, "type": kind, "payload": payload})
		return id

var failures: Array[String] = []


func check(label: String, condition: bool) -> void:
	print("  %s  %s" % ["PASS" if condition else "FAIL", label])
	if not condition:
		failures.append(label)


func reply(reader, id: String, payload: Dictionary) -> void:
	reader._receive({"protocol_version": 1, "request_id": id, "type": "response", "payload": payload})


func snapshot(revision: int, bundles: Array) -> Dictionary:
	return {"session_id": "session", "run": {"run_id": "run"},
		"selection": {"run_id": "run", "revision": 0, "cursor_s": 0},
		"workbench": {"revision": revision, "sources": [], "operations": [], "bundles": bundles}}


func bundle(id: String) -> Dictionary:
	return {"bundle_id": id, "kind": "calibrated-window"}


func projection(id: String) -> Dictionary:
	return {"schema": "ciw.experiment-view.v1", "bundle_id": id, "label": "UI boundary fixture",
		"fusion_context": {"state_kind": "window_feature_posterior", "observability": {"status": "unresolved"}},
		"evidence_id": "evidence", "graph": {"nodes": []},
		"panels": [{"panel_id": "state", "title": "GSIE state", "values": [1.12], "units": ["m"],
			"labels": ["state"], "covariance": [[.84]], "marginal_standard_deviation": [sqrt(.84)],
			"context": {"frame_id": "fixture"}, "provenance": {"result_id": id}}]}


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var reader := Reader.new()
	reader.online = true
	var received: Array = []
	reader.experiment_received.connect(func(value): received.append(value))
	reader.inspect_experiment("old")
	var old: String = reader.requests[-1].id
	reader.inspect_experiment("new")
	check("single outstanding view request", reader.requests.size() == 1)
	reply(reader, old, projection("old"))
	check("old response cannot overwrite new selection", received.is_empty() and reader.requests[-1].payload.bundle_id == "new")
	reply(reader, reader.requests[-1].id, projection("new"))
	check("selected occurrence delivered", received.size() == 1 and received[0].bundle_id == "new")
	var artifacts: Array = []
	reader.artifact_received.connect(func(value): artifacts.append(value))
	reader.inspect_artifact("old-result")
	var old_result: String = reader.requests[-1].id
	reader.inspect_artifact("")
	reply(reader, old_result, {"result_id": "old-result"})
	check("changing context cancels stale artifact delivery", artifacts.is_empty())
	reader._request_snapshot()
	var old_snapshot: String = reader.requests[-1].id
	for i in 4:
		reader._receive({"protocol_version": 1, "type": "workbench.changed", "payload": {"session_id": "session"}})
	check("invalidation while reading stays pending", reader._snapshot_dirty)
	reply(reader, old_snapshot, snapshot(1, []))
	check("pending invalidation triggers a fresh read", reader.requests[-1].type == "session.get" and reader._refresh_pending)
	reply(reader, reader.requests[-1].id, snapshot(2, []))
	reader._apply_snapshot(snapshot(1, []))
	check("catalog cannot regress", reader.snapshot.workbench.revision == 2)

	var view := View.new()
	root.add_child(view)
	view.attach(reader)
	view.apply_snapshot(snapshot(3, [bundle("first")]))
	view.apply_view(projection("first"))
	check("shared occurrence populates native state panel", view._plot.panel.values == [1.12])
	view._follow.button_pressed = false
	view.apply_snapshot(snapshot(4, [bundle("first"), bundle("replay")]))
	check("replay append preserves manual selection", view.selected_bundle == "first" and view._bundle_ids.size() == 2)
	view._follow.button_pressed = true
	view.apply_snapshot(snapshot(5, [bundle("first"), bundle("replay"), bundle("third")]))
	check("follow explicitly selects newest occurrence", view.selected_bundle == "third" and view._plot.panel.is_empty())
	view.apply_view(projection("first"))
	check("panel refuses obsolete projection", view.view.is_empty())
	view.apply_view(projection("third"))
	reader.status_changed.emit("disconnected", "gone")
	check("disconnect marks retained view stale", view._status.text.begins_with("STALE") and not view.view.is_empty())
	var replacement := snapshot(0, [])
	replacement.session_id = "replacement"
	view.apply_snapshot(replacement)
	check("new session clears old science", view.view.is_empty() and view._plot.panel.is_empty() and view._inspector.text.is_empty())
	view.queue_free()
	reader.free()
	if failures.is_empty():
		print("PASS: experiment selection, replay occurrences, invalidation, stale-response and disconnect boundaries")
		quit(0)
	else:
		quit(1)
