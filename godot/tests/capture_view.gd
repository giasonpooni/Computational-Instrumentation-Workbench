extends SceneTree
## Optional visual QA: run with a real graphics driver and a running service.
## godot --path godot --script res://tests/capture_view.gd -- /absolute/path/view.png
var _workbench: Control
var _started := 0
var _capturing := false


func _initialize() -> void:
	_started = Time.get_ticks_msec()
	_workbench = load("res://main.tscn").instantiate()
	root.add_child(_workbench)
	root.size = Vector2i(1280, 850)


func _process(_delta: float) -> bool:
	if Time.get_ticks_msec() - _started > 15000:
		push_error("Visual capture timed out waiting for service and rendering")
		quit(1)
	if not _capturing and _workbench._client.status == "ready":
		_capturing = true
		_capture.call_deferred()
	return false


func _capture() -> void:
	# Let numerical inspection and the complete UI layout settle.
	await create_timer(0.5).timeout
	await RenderingServer.frame_post_draw
	var arguments := OS.get_cmdline_user_args()
	if arguments.is_empty():
		push_error("Pass an absolute output PNG path after --")
		quit(1)
		return
	var image := root.get_texture().get_image()
	if image == null or image.is_empty():
		push_error("No rendered image; capture requires a real graphics driver")
		quit(1)
		return
	var error := image.save_png(arguments[0])
	if error != OK:
		push_error("Could not save capture: %s" % error)
		quit(1)
		return
	print("PASS: rendered viewport saved to %s" % arguments[0])
	quit(0)
