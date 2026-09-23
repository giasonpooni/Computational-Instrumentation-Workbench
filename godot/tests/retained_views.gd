extends SceneTree
## Render actual Python projections. JSON here is display data, never an export.
const View = preload("res://scripts/experiment_view.gd")

class Reader extends "res://scripts/ciw_client.gd":
	func _request(_kind: String, _payload: Dictionary = {}) -> String:
		return ""


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var paths := OS.get_cmdline_user_args()
	if paths.is_empty():
		push_error("Provide actual experiment.inspect JSON paths after --")
		quit(1)
		return
	var reader := Reader.new()
	reader.online = true
	var view := View.new()
	root.add_child(view)
	view.attach(reader)
	var revision := 0
	for path in paths:
		var value = JSON.parse_string(FileAccess.get_file_as_string(path))
		if not value is Dictionary or value.get("schema") != "ciw.experiment-view.v1":
			push_error("Invalid projection: " + path)
			quit(1)
			return
		revision += 1
		view.apply_snapshot({"session_id": "retained-native-views", "workbench": {
			"revision": revision, "sources": [], "operations": [],
			"bundles": [{"bundle_id": value.bundle_id, "kind": value.kind}]}})
		view.apply_view(value)
		if view.view.get("bundle_id") != value.bundle_id or view._summary.text.is_empty():
			push_error("Selected projection was not rendered: " + path)
			quit(1)
			return
		for i in value.panels.size():
			view._select_panel(i)
			if view._plot.panel != value.panels[i] or view._numbers.text.is_empty():
				push_error("Native panel was not rendered: " + path)
				quit(1)
				return
		print("PASS: retained ", value.kind, " occurrence; ", value.panels.size(), " panels")
	view.queue_free()
	reader.free()
	print("PASS: all actual retained projections rendered")
	quit(0)
