extends SceneTree
## Generic adapter records must never pass through legacy numerical casts.
## Run headless; fixtures drive the existing UI event boundary directly.

const Main = preload("res://scripts/main.gd")

var _failures: Array[String] = []
var _view
var _done := false


func _check(name: String, ok: bool) -> void:
	print("  %s  %s" % ["PASS" if ok else "FAIL", name])
	if not ok:
		_failures.append(name)


func _legacy_run() -> Dictionary:
	return {
		"run_id": "legacy-run", "evidence_id": "sha256:legacy", "instrument": "legacy.fixture",
		"metadata": {"duration_s": 1.0, "sample_count": 2, "sample_rate_hz": 2.0},
		"time_s": [0.0, 0.5],
		"channels": {"q": {"unit": "m", "values": [1.0, 2.0]},
					 "v": {"unit": "m/s", "values": [3.0, 4.0]}},
	}


func _snapshot(run: Dictionary) -> Dictionary:
	return {"session_id": "adapter-boundary-test", "run": run}


func _initialize() -> void:
	_view = Main.new()
	root.add_child(_view)


func _process(_delta: float) -> bool:
	if _done:
		return true
	_done = true
	var view = _view
	print("generic adapter viewport boundary check")
	var legacy := _legacy_run()
	view._on_snapshot(_snapshot(legacy))
	view._on_run(legacy)
	view._on_status("ready", "test connection")
	var retained_points: PackedVector2Array = view._phase._positions.duplicate()
	_check("legacy_fixture_is_rendered", retained_points.size() == 2 and not view._play.disabled)

	# Either marker advertises the new contract. Its nullable sampling and
	# missing observations must not be interpreted as zero-valued demo data.
	for marker in ["run_schema", "manifest"]:
		var generic := {
			"run_id": "generic-" + marker, "evidence_id": "sha256:generic",
			"instrument": "org.notationsystems.rci",
			"metadata": {"duration_s": 1.0, "sample_count": 1, "sample_rate_hz": null},
			"time_s": [0.0],
			"channels": {"raw": {"unit": "count", "values": [null]},
						 "calibrated": {"unit": "kg", "values": [2.0]}},
		}
		if marker == "run_schema":
			generic[marker] = "run.v1"
		else:
			generic.metadata[marker] = {"manifest_schema": "ciw.instrument-manifest.v1"}
		view._playing = true
		view._on_snapshot(_snapshot(generic))
		view._on_run(generic)
		# A later successful transport status must not undo capability refusal.
		view._on_status("ready", "transport synchronized")
		view._on_selection({"cursor_s": null, "interval_s": null})
		view._on_sample({"values": {"raw": null, "calibrated": 2.0}})
		_check(marker + "_refused", not view._view_supported)
		_check(marker + "_canvases_hidden", not view._phase.visible and not view._energy.visible)
		_check(marker + "_data_not_reinterpreted", view._phase._positions == retained_points)
		_check(marker + "_readouts_cleared", view._value_labels.is_empty() and view._channel.item_count == 0)
		_check(marker + "_interaction_disabled", view._play.disabled and view._apply_interval.disabled and not view._slider.editable and not view._playing)
		_check(marker + "_terminal_message", view._sample_label.text.begins_with("TERMINAL ONLY") and view._detail.text == Main.UNSUPPORTED_VIEW)

	# Reconnecting to a supported recording restores the existing viewport.
	view._on_snapshot(_snapshot(legacy))
	view._on_run(legacy)
	view._on_status("ready", "legacy fixture restored")
	view._on_sample({"sample_index": 0, "time_s": 0.0,
					 "values": {"q": null, "v": 3.0}, "units": {"q": "m", "v": "m/s"}})
	_check("legacy_view_restored", view._view_supported and view._phase.visible and view._energy.visible and not view._play.disabled)
	_check("null_never_becomes_zero", view._value_labels["q"].text == "—")
	_check("finite_value_still_rendered", view._value_labels["v"].text.begins_with("3.000000"))
	if _failures.is_empty():
		print("PASS: generic adapters remain terminal-only without numerical reinterpretation")
		quit(0)
	else:
		print("FAIL: %s" % ", ".join(_failures))
		quit(1)
	return true
