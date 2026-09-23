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


func residual_projection(id: String) -> Dictionary:
	# UI fixture only: native scientific values are checked in Python.
	var value := projection(id)
	value.fusion_context = null
	value.object_context = {"object_kind": "residual_sequence",
		"summary": "Declared residual monitor · inter-window covariance unknown · isolation ambiguous",
		"sensor_fusion": "not_performed", "upstream_bundle_ids": ["window-a", "window-b"]}
	value.panels = [{"panel_id": "cusum-positive", "title": "FDIR positive CUSUM",
		"values": [0.5, 2.5], "units": ["1", "1"], "labels": ["1.5 s", "3.5 s"],
		"covariance": null, "marginal_standard_deviation": null,
		"context": {"event_times": [1.5, 3.5], "threshold": 2.0,
			"inter_window_cross_covariance": "unknown", "false_alarm_confidence": "not_declared"},
		"provenance": {"result_id": id + "-fdir", "execution_id": id + "-execution"}}]
	return value


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
	var schematic := projection("third")
	schematic.fusion_context = null
	schematic.object_context = {"object_kind": "declared_schematic", "next_step": "declare sensor quality"}
	schematic.panels = []
	schematic.schematic = {"nodes": [{"id": "sensor", "kind": "measurement", "attrs": {}}], "edges": []}
	view.apply_view(schematic)
	check("schematic view clears prior state plot", view._plot.panel.is_empty() and view._numbers.text.is_empty())
	check("schematic remains a typed object", view._summary.text.contains("declared_schematic") and view._graph.get_root().get_child(0).get_text(0) == "sensor · measurement")
	var numerical := projection("third")
	numerical.fusion_context = null
	numerical.object_context = {"object_kind": "integer_numerical_field"}
	numerical.panels[0].covariance = null
	numerical.panels[0].marginal_standard_deviation = null
	view.apply_view(numerical)
	check("numerical field has no fabricated uncertainty", view._plot.panel.covariance == null and view._summary.text.contains("integer_numerical_field"))
	view.apply_snapshot(snapshot(6, [bundle("third"), {"bundle_id": "monitor", "kind": "residual-monitor"}]))
	view.apply_view(residual_projection("monitor"))
	check("residual monitor preserves event labels and no joint confidence", view._plot.panel.labels == ["1.5 s", "3.5 s"] and view._plot.panel.covariance == null and view._plot.panel.marginal_standard_deviation == null)
	check("monitor exposes declared threshold and ambiguous isolation", view._numbers.text.contains('"threshold":') and view._plot.panel.context.threshold == 2.0 and view._summary.text.contains("isolation ambiguous") and view._summary.text.contains("fusion: not performed"))
	view.apply_snapshot(snapshot(7, [{"bundle_id": "monitor", "kind": "residual-monitor"}, {"bundle_id": "monitor-replay", "kind": "residual-monitor"}]))
	check("monitor replay clears earlier residual provenance while loading", view._plot.panel.is_empty() and view._numbers.text.is_empty())
	view.apply_view(residual_projection("monitor"))
	check("earlier monitor response cannot replace selected replay", view.view.is_empty())
	view.apply_view(residual_projection("monitor-replay"))
	check("replayed residual values remain tied to fresh result identity", view._plot.panel.values == [0.5, 2.5] and view._plot.panel.provenance.result_id == "monitor-replay-fdir")
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
