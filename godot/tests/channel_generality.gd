extends SceneTree
## The viewport must take its channels from the record it is shown, not from a
## compiled-in list. Runs headless with no service: snapshots are supplied
## directly, so this checks the client's own handling rather than the protocol.
##
##   godot --headless --path godot --script res://tests/channel_generality.gd

const Main = preload("res://scripts/main.gd")
const PhasePlot = preload("res://scripts/phase_plot.gd")

var _failures: Array[String] = []
var _view
var _done := false

func _check(name: String, ok: bool, detail: String = "") -> void:
	if ok:
		print("  PASS  %s%s" % [name, ("  " + detail) if detail else ""])
	else:
		_failures.append(name)
		print("  FAIL  %s%s" % [name, ("  " + detail) if detail else ""])

func _snapshot(channels: Dictionary, run_id: String) -> Dictionary:
	return {
		"session_id": "session-test",
		"run": {
			"run_id": run_id, "evidence_id": "sha256:test", "instrument": "test.instrument",
			"metadata": {"duration_s": 4.0, "sample_count": 256, "sample_rate_hz": 64.0},
			"channels": channels,
		},
		"selection": {"channel": channels.keys()[0], "interval_s": [0.0, 4.0],
					  "cursor_s": 0.0, "revision": 0},
		"results": [],
	}

func _selector_items(view) -> Array:
	var items: Array = []
	for i in range(view._channel.item_count):
		items.append(view._channel.get_item_text(i))
	return items

func _initialize() -> void:
	## Nodes added here are not ready until the first frame, so the checks run
	## from _process once the scene has built its widgets.
	_view = Main.new()
	root.add_child(_view)


func _process(_delta: float) -> bool:
	if _done:
		return true
	_done = true
	print("channel generality check")
	var view = _view

	# A three-channel record, the shape the demo instrument produces.
	var three := {"q": {"unit": "m"}, "v": {"unit": "m/s"}, "energy": {"unit": "J"}}
	view._on_snapshot(_snapshot(three, "run-three"))
	_check("three_channel_selector", _selector_items(view) == ["q", "v", "energy"],
		   str(_selector_items(view)))
	_check("three_channel_cards", view._value_labels.keys() == ["q", "v", "energy"],
		   str(view._value_labels.keys()))

	# A different instrument: two channels, different names and units.
	var two := {"pressure": {"unit": "Pa"}, "flow": {"unit": "m3/s"}}
	view._on_snapshot(_snapshot(two, "run-two"))
	_check("two_channel_selector", _selector_items(view) == ["pressure", "flow"],
		   str(_selector_items(view)))
	_check("two_channel_cards", view._value_labels.keys() == ["pressure", "flow"],
		   str(view._value_labels.keys()))
	_check("no_stale_cards", not view._value_labels.has("q"),
		   "previous run's channels cleared")

	# Five channels, to show the count is not fixed either.
	var five := {"a": {"unit": "1"}, "b": {"unit": "K"}, "c": {"unit": "Pa"},
				 "d": {"unit": "V"}, "e": {"unit": "A"}}
	view._on_snapshot(_snapshot(five, "run-five"))
	_check("five_channel_selector", _selector_items(view).size() == 5, str(_selector_items(view)))

	# The selection round-trip must resolve against the record's channels.
	view._on_selection({"channel": "c", "interval_s": [0.0, 4.0], "cursor_s": 1.0, "revision": 3})
	_check("selection_resolves_by_record", view._channel.selected == 2,
		   "selected index %d" % view._channel.selected)

	# A sample naming only some channels must not fault the readouts.
	view._on_sample({"sample_index": 7, "time_s": 0.109,
					 "values": {"a": 1.5, "b": 2.5}, "units": {"a": "1", "b": "K"}})
	_check("partial_sample_tolerated", view._value_labels["c"].text == "—",
		   "absent channel shows a dash")
	_check("present_sample_rendered", view._value_labels["a"].text.begins_with("1.5"),
		   view._value_labels["a"].text)

	# The phase portrait takes its two axes from the record, with its units.
	var plot = PhasePlot.new()
	root.add_child(plot)
	plot.set_run({"time_s": [0.0, 0.1, 0.2],
				  "channels": {"pressure": {"unit": "Pa", "values": [1.0, 2.0, 3.0]},
							   "flow": {"unit": "m3/s", "values": [0.5, 0.6, 0.7]}}})
	_check("phase_axes_from_record", plot._axis_x == "pressure (Pa)" and plot._axis_y == "flow (m3/s)",
		   "%s / %s" % [plot._axis_x, plot._axis_y])
	_check("phase_points_from_record", plot._positions.size() == 3,
		   "%d points" % plot._positions.size())

	if _failures.is_empty():
		print("PASS: the viewport builds channels, readouts and phase axes from the record")
		quit(0)
	else:
		print("FAIL: %s" % ", ".join(_failures))
		quit(1)
	return true
