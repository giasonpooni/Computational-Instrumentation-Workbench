extends Control
## Screen coordinates are presentation only. A pick resolves to a retained time.
signal time_picked(time_s: float)

const TEAL := Color("60dfcd")
const MUTED := Color("8d9bac")
var _times: Array = []
var _positions: PackedVector2Array = []
var _screen: PackedVector2Array = []
var _selected := -1
var _stale := true
var _limits := Rect2(-1, -1, 2, 2)
var _axis_x := ""
var _axis_y := ""


func _ready() -> void:
	custom_minimum_size = Vector2(300, 270)
	mouse_default_cursor_shape = Control.CURSOR_CROSS
	resized.connect(queue_redraw)


func set_run(run: Dictionary) -> void:
	_times = run.get("time_s", [])
	_positions.clear()
	## The portrait plots the run's first declared channel against its second;
	## which quantities those are is the record's business, not this view's.
	var channels: Dictionary = run.get("channels", {})
	var names: Array = channels.keys()
	var q: Array = []
	var v: Array = []
	_axis_x = ""
	_axis_y = ""
	if names.size() >= 2:
		var first: Dictionary = channels.get(names[0], {})
		var second: Dictionary = channels.get(names[1], {})
		q = first.get("values", [])
		v = second.get("values", [])
		_axis_x = "%s (%s)" % [str(names[0]), str(first.get("unit", ""))]
		_axis_y = "%s (%s)" % [str(names[1]), str(second.get("unit", ""))]
	var low := Vector2(INF, INF)
	var high := Vector2(-INF, -INF)
	for index in range(mini(q.size(), v.size())):
		var point := Vector2(float(q[index]), float(v[index]))
		_positions.append(point)
		low = low.min(point)
		high = high.max(point)
	if not _positions.is_empty():
		var span := high - low
		span.x = maxf(span.x, 0.001)
		span.y = maxf(span.y, 0.001)
		_limits = Rect2(low - span * 0.1, span * 1.2)
	_selected = -1
	queue_redraw()


func set_sample(index: int) -> void:
	_selected = index
	queue_redraw()


func set_stale(value: bool) -> void:
	_stale = value
	queue_redraw()


func _plot_rect() -> Rect2:
	return Rect2(60, 34, maxf(size.x - 84, 1), maxf(size.y - 86, 1))


func _project(point: Vector2) -> Vector2:
	var area := _plot_rect()
	var fraction := (point - _limits.position) / _limits.size
	return area.position + Vector2(fraction.x, 1.0 - fraction.y) * area.size


func _draw() -> void:
	var font := ThemeDB.fallback_font
	var area := _plot_rect()
	draw_rect(area, Color("0c1521"))
	for step in range(6):
		var fraction := float(step) / 5.0
		var x := area.position.x + fraction * area.size.x
		var y := area.position.y + fraction * area.size.y
		draw_line(Vector2(x, area.position.y), Vector2(x, area.end.y), Color("233043"))
		draw_line(Vector2(area.position.x, y), Vector2(area.end.x, y), Color("233043"))
		draw_string(font, Vector2(x - 15, area.end.y + 21), "%.2f" % (_limits.position.x + _limits.size.x * fraction), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)
		draw_string(font, Vector2(7, y + 4), "%.2f" % (_limits.end.y - _limits.size.y * fraction), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)
	draw_string(font, Vector2(area.position.x, 21), _axis_y, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, MUTED)
	draw_string(font, Vector2(area.end.x - 36, size.y - 8), _axis_x, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, MUTED)
	_screen.clear()
	for point in _positions:
		_screen.append(_project(point))
	if _screen.size() > 1:
		var line_color := Color("537778") if _stale else TEAL
		draw_polyline(_screen, line_color, 1.7, true)
	if _selected >= 0 and _selected < _screen.size():
		var point := _screen[_selected]
		draw_line(Vector2(point.x, area.position.y), Vector2(point.x, area.end.y), Color(0.9, 0.7, 0.38, 0.25))
		draw_line(Vector2(area.position.x, point.y), Vector2(area.end.x, point.y), Color(0.9, 0.7, 0.38, 0.25))
		draw_circle(point, 8, Color(1, 0.76, 0.4, 0.16))
		draw_circle(point, 4, Color("ffcc80"))
	if _positions.is_empty():
		draw_string(font, area.get_center() - Vector2(107, 0), "Awaiting a recorded run", HORIZONTAL_ALIGNMENT_LEFT, -1, 16, MUTED)
	elif _stale:
		draw_string(font, Vector2(area.position.x + 10, area.position.y + 22), "STALE · reconnect to synchronize", HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color("ffcc80"))


func _gui_input(event: InputEvent) -> void:
	if _stale:
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if not _plot_rect().has_point(event.position):
			return
		var best := -1
		var distance := 400.0 # Require a hit within 20 display pixels.
		for index in range(_screen.size()):
			var candidate: float = event.position.distance_squared_to(_screen[index])
			if candidate < distance:
				distance = candidate
				best = index
		if best >= 0 and best < _times.size():
			time_picked.emit(float(_times[best]))
			accept_event()
