extends Control
## Display native points and marginal standard deviations. No fit or interpolation.
var panel: Dictionary = {}


func set_panel(value: Dictionary) -> void:
	panel = value
	queue_redraw()


func _draw() -> void:
	var font := ThemeDB.fallback_font
	var color := Color("60dfcd")
	if panel.is_empty():
		draw_string(font, Vector2(20, 40), "Select a retained experiment", HORIZONTAL_ALIGNMENT_LEFT, -1, 16)
		return
	var values: Array = panel.get("values", [])
	var units: Array = panel.get("units", [])
	if values.is_empty() or units.size() != values.size():
		return
	for unit in units:
		if unit != units[0]:
			draw_string(font, Vector2(20, 40), "Mixed units: inspect the numeric table", HORIZONTAL_ALIGNMENT_LEFT, -1, 16)
			return
	var deviations: Variant = panel.get("marginal_standard_deviation")
	var lower := INF
	var upper := -INF
	for i in values.size():
		var sigma := float(deviations[i]) if deviations is Array else 0.0
		lower = minf(lower, float(values[i]) - sigma)
		upper = maxf(upper, float(values[i]) + sigma)
	if not is_finite(upper - lower):
		draw_string(font, Vector2(20, 40), "Display range overflow: inspect the numeric table", HORIZONTAL_ALIGNMENT_LEFT, -1, 14)
		return
	var extent := maxf(upper - lower, maxf(absf(upper), 1.0) * 0.1)
	lower -= extent * 0.15
	upper += extent * 0.15
	if not is_finite(upper - lower):
		draw_string(font, Vector2(20, 40), "Display range overflow: inspect the numeric table", HORIZONTAL_ALIGNMENT_LEFT, -1, 14)
		return
	var plot := Rect2(72, 28, maxf(size.x - 92, 20), maxf(size.y - 95, 40))
	for j in 5:
		var fraction := float(j) / 4.0
		var y := plot.end.y - fraction * plot.size.y
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), Color("273549"))
		draw_string(font, Vector2(2, y + 4), String.num_scientific(lerpf(lower, upper, fraction)), HORIZONTAL_ALIGNMENT_RIGHT, 62, 12)
	for i in values.size():
		# Equally spaced categorical positions, explicitly labeled. Event times
		# remain labels/context; sample order never implies a temporal trajectory.
		var x := plot.position.x + plot.size.x * (float(i) + 0.5) / values.size()
		var y := plot.end.y - (float(values[i]) - lower) / (upper - lower) * plot.size.y
		if deviations is Array:
			var dy := float(deviations[i]) / (upper - lower) * plot.size.y
			draw_line(Vector2(x, y - dy), Vector2(x, y + dy), color, 2)
			for end in [y - dy, y + dy]:
				draw_line(Vector2(x - 5, end), Vector2(x + 5, end), color, 2)
		draw_circle(Vector2(x, y), 4, color)
		var label := str(panel.labels[i])
		if values.size() > 4:
			label = "row %s" % (i + 1)
		if label.length() > 20:
			label = label.left(17) + "..."
		draw_string(font, Vector2(x - 65, plot.end.y + 22), label, HORIZONTAL_ALIGNMENT_CENTER, 130, 11)
	draw_string(font, Vector2(8, 18), str(units[0]), HORIZONTAL_ALIGNMENT_LEFT, -1, 12)
	draw_string(font, Vector2(72, size.y - 15), "Declared order · points only · " + ("marginal ±1σ" if deviations is Array else "covariance not supplied"), HORIZONTAL_ALIGNMENT_LEFT, -1, 12)
