extends SceneTree
## Display an actual retained Julia oscillator recording through the existing
## oscillator viewport. Runs headless with no service: the recording JSON is a
## run.v1 projection written by `ciw julia-oscillator run`, so this checks that
## the legacy phase, energy and sample views take their values from that record
## without recomputing anything.
##
##   godot --headless --path godot --script res://tests/julia_projection.gd -- /path/to/recording.json

const Main = preload("res://scripts/main.gd")

var _failures: Array[String] = []
var _view
var _run: Dictionary = {}
var _done := false


func _check(name: String, ok: bool, detail: String = "") -> void:
	if ok:
		print("  PASS  %s%s" % [name, ("  " + detail) if detail else ""])
	else:
		_failures.append(name)
		print("  FAIL  %s%s" % [name, ("  " + detail) if detail else ""])


func _initialize() -> void:
	var paths := OS.get_cmdline_user_args()
	if paths.is_empty():
		push_error("Provide an actual julia-oscillator recording.json path after --")
		quit(1)
		return
	var value = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if not value is Dictionary or value.get("instrument") != "julia-oscillator-trajectory.v1":
		push_error("Not a retained Julia oscillator recording: " + paths[0])
		quit(1)
		return
	_run = value
	_view = Main.new()
	root.add_child(_view)


func _metadata_only(run: Dictionary) -> Dictionary:
	var channels := {}
	for name in run.channels.keys():
		channels[name] = {"unit": run.channels[name].unit}
	return {"run_id": run.run_id, "evidence_id": run.evidence_id, "instrument": run.instrument,
			"metadata": run.metadata, "channels": channels}


func _process(_delta: float) -> bool:
	if _done:
		return true
	_done = true
	print("julia oscillator projection check")
	var view = _view
	var count := int(_run.metadata.sample_count)
	var snapshot := {"session_id": "session-julia-projection", "run": _metadata_only(_run),
		"selection": {"run_id": _run.run_id, "channel": "q", "interval_s": [0.0, float(_run.metadata.duration_s)],
					  "cursor_s": 0.0, "coordinate_frame": "oscillator-state", "revision": 0}, "results": []}
	view._on_snapshot(snapshot)
	_check("projection_uses_legacy_viewport", view._view_supported, "no manifest or run_schema gate tripped")
	_check("channels_from_record", view._value_labels.keys() == ["q", "v", "energy"], str(view._value_labels.keys()))
	_check("provenance_is_simulation", _run.metadata.provenance.origin == "simulation", str(_run.metadata.provenance.origin))
	_check("provenance_names_result", str(_run.metadata.provenance.result_id).begins_with("sha256:"), str(_run.metadata.provenance.result_id))
	view._on_run(_run)
	_check("phase_points_from_record", view._phase._positions.size() == count, "%d of %d" % [view._phase._positions.size(), count])
	_check("phase_axes_are_q_v", view._phase._axis_x == "q (m)" and view._phase._axis_y == "v (m/s)",
		   "%s / %s" % [view._phase._axis_x, view._phase._axis_y])
	var first: Vector2 = view._phase._positions[0]
	_check("phase_first_point_is_retained_initial_state",
		   is_equal_approx(first.x, float(_run.channels.q.values[0])) and is_equal_approx(first.y, float(_run.channels.v.values[0])),
		   str(first))
	_check("energy_view_trajectory_from_render", view._energy._points.size() == _run.render.trajectory.size(),
		   "%d points" % view._energy._points.size())
	_check("render_maps_to_retained_samples", _run.render.sample_indices.size() == _run.render.trajectory.size()
		   and int(_run.render.sample_indices[count - 1]) == count - 1, "sample_indices cover the record")
	view._on_selection({"run_id": _run.run_id, "channel": "energy", "interval_s": [0.0, float(_run.metadata.duration_s)],
						"cursor_s": 1.0, "coordinate_frame": "oscillator-state", "revision": 1})
	_check("selection_resolves_energy_channel", view._channel.selected == 2, "selected index %d" % view._channel.selected)
	var index := 64
	view._on_sample({"run_id": _run.run_id, "evidence_id": _run.evidence_id, "sample_index": index,
					 "time_s": float(_run.time_s[index]),
					 "values": {"q": _run.channels.q.values[index], "v": _run.channels.v.values[index], "energy": _run.channels.energy.values[index]},
					 "units": {"q": "m", "v": "m/s", "energy": "J"}})
	_check("sample_cards_show_retained_values", view._value_labels["q"].text.begins_with(("%.6f" % float(_run.channels.q.values[index])).substr(0, 6)),
		   view._value_labels["q"].text)
	_check("energy_marker_follows_sample", view._energy._marker.visible, "marker at retained sample %d" % index)
	if _failures.is_empty():
		print("PASS: the existing viewport displays the retained Julia trajectory without recomputation")
		quit(0)
	else:
		print("FAIL: %s" % ", ".join(_failures))
		quit(1)
	return true
