extends Control

const Client = preload("res://scripts/ciw_client.gd")
const PhasePlot = preload("res://scripts/phase_plot.gd")
const EnergyView = preload("res://scripts/energy_view.gd")
const ExperimentView = preload("res://scripts/experiment_view.gd")
const TEXT := Color("dce6f1")
const MUTED := Color("899cb2")
const TEAL := Color("60dfcd")
const AMBER := Color("ffcc80")

var _client = Client.new()
var _phase = PhasePlot.new()
var _energy = EnergyView.new()
var _experiments = ExperimentView.new()
var _tabs: TabContainer
var _status: Label
var _detail: Label
var _identity: Label
var _run_meta: Label
var _interval_label: Label
var _cursor_label: Label
var _sample_label: Label
var _revision: Label
var _slider: HSlider
var _channel: OptionButton
var _interval_start: SpinBox
var _interval_end: SpinBox
var _apply_interval: Button
var _play: Button
var _view_state: Label
var _value_labels: Dictionary = {}
var _channel_names: Array = []
var _samples_row: HBoxContainer
var _playing := false
var _play_time := 0.0
var _duration := 0.0
var _synchronizing := false
var _identity_run := ""
var _view_supported := true
const UNSUPPORTED_VIEW := "This adapter uses the generic run contract. Inspect raw, calibrated, estimated and refusal states in the terminal; this viewport supports the legacy recording contract only."


func _ready() -> void:
	get_window().min_size = Vector2i(1080, 850)
	_build_theme()
	_build_ui()
	add_child(_client)
	_experiments.attach(_client)
	_client.status_changed.connect(_on_status)
	_client.snapshot_received.connect(_on_snapshot)
	_client.run_received.connect(_on_run)
	_client.selection_received.connect(_on_selection)
	_client.sample_received.connect(_on_sample)
	_client.request_failed.connect(_on_error)
	_phase.time_picked.connect(_pick_time)
	_client.connect_service()


func _build_theme() -> void:
	theme = Theme.new()
	theme.default_font_size = 14
	theme.set_color("font_color", "Label", TEXT)
	theme.set_color("font_color", "Button", TEXT)
	theme.set_color("font_hover_color", "Button", TEAL)
	theme.set_color("font_color", "OptionButton", TEXT)
	theme.set_stylebox("normal", "Button", _box(Color("172537"), Color("30445d"), 6))
	theme.set_stylebox("hover", "Button", _box(Color("20374b"), TEAL, 6))
	theme.set_stylebox("pressed", "Button", _box(Color("254155"), TEAL, 6))
	theme.set_stylebox("disabled", "Button", _box(Color("142031"), Color("263548"), 6))
	theme.set_stylebox("normal", "OptionButton", _box(Color("172537"), Color("30445d"), 6))
	theme.set_stylebox("hover", "OptionButton", _box(Color("20374b"), TEAL, 6))
	theme.set_stylebox("normal", "LineEdit", _box(Color("101c2b"), Color("30445d"), 4))
	theme.set_color("font_color", "LineEdit", TEXT)
	var slider_style := _box(Color("293b50"), Color("293b50"), 2)
	slider_style.content_margin_top = 2
	slider_style.content_margin_bottom = 2
	theme.set_stylebox("slider", "HSlider", slider_style)
	var filled_style := slider_style.duplicate()
	filled_style.bg_color = TEAL
	filled_style.border_color = TEAL
	theme.set_stylebox("grabber_area", "HSlider", filled_style)
	var background := ColorRect.new()
	background.color = Color("080f19")
	background.mouse_filter = Control.MOUSE_FILTER_IGNORE
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(background)


func _box(fill: Color, edge: Color, radius: int) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = fill
	style.border_color = edge
	style.set_border_width_all(1)
	style.set_corner_radius_all(radius)
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	return style


func _label(text: String, font_size: int = 14, color: Color = TEXT) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label


func _panel() -> PanelContainer:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _box(Color("101a28"), Color("273549"), 9))
	return panel


func _button(text: String, action: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.pressed.connect(action)
	return button


func _build_ui() -> void:
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for edge in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + edge, 18)
	add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	margin.add_child(column)
	var header := HBoxContainer.new()
	header.add_theme_constant_override("separation", 16)
	column.add_child(header)
	var heading := VBoxContainer.new()
	heading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(heading)
	heading.add_child(_label("CIW   /   COMPUTATIONAL INSTRUMENTATION WORKBENCH", 12, TEAL))
	heading.add_child(_label("One workspace for scientific instruments.", 29))
	heading.add_child(_label("Experiments · sensor fusion · retained evidence · streaming session updates", 14, MUTED))
	var connection := VBoxContainer.new()
	header.add_child(connection)
	_status = _label("●  CONNECTING", 13, AMBER)
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	connection.add_child(_status)
	var actions := HBoxContainer.new()
	connection.add_child(actions)
	actions.add_child(_button("Refresh", func(): _client.refresh()))
	actions.add_child(_button("Reconnect", func(): _client.connect_service()))
	_detail = _label("Waiting for the local instrument service", 12, MUTED)
	_detail.clip_text = true
	column.add_child(_detail)
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(_tabs)
	_experiments.name = "Experiments"
	_tabs.add_child(_experiments)
	var oscillator := VBoxContainer.new()
	oscillator.name = "Oscillator"
	_tabs.add_child(oscillator)
	column = oscillator
	var context := _panel()
	column.add_child(context)
	var context_rows := VBoxContainer.new()
	context.add_child(context_rows)
	var metadata_row := HBoxContainer.new()
	context_rows.add_child(metadata_row)
	_run_meta = _label("NO RUN  /  start ciw serve in a terminal", 13)
	_run_meta.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	metadata_row.add_child(_run_meta)
	_revision = _label("revision —", 12, MUTED)
	metadata_row.add_child(_revision)
	_identity = _label("Session —   /   Run —   /   Evidence —", 11, MUTED)
	_identity.clip_text = true
	context_rows.add_child(_identity)
	var controls := HBoxContainer.new()
	controls.add_theme_constant_override("separation", 10)
	context_rows.add_child(controls)
	controls.add_child(_label("CHANNEL", 11, MUTED))
	_channel = OptionButton.new()
	## Items come from the run's declared channels, never a compiled-in list.
	_channel.item_selected.connect(func(index: int):
		if not _synchronizing:
			_client.update_selection({"channel": _channel.get_item_text(index)})
	)
	controls.add_child(_channel)
	_interval_label = _label("Analysis interval [—, —) s", 13)
	_interval_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	controls.add_child(_interval_label)
	controls.add_child(_label("EDIT [", 11, MUTED))
	_interval_start = SpinBox.new()
	_interval_end = SpinBox.new()
	for entry in [_interval_start, _interval_end]:
		entry.step = 0.01
		entry.custom_minimum_size.x = 80
		controls.add_child(entry)
	controls.add_child(_label(") s", 12, MUTED))
	_apply_interval = _button("Apply", _set_interval)
	controls.add_child(_apply_interval)
	var panels := HBoxContainer.new()
	panels.add_theme_constant_override("separation", 14)
	panels.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(panels)
	var phase_card := _panel()
	phase_card.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panels.add_child(phase_card)
	var phase_column := VBoxContainer.new()
	phase_card.add_child(phase_column)
	phase_column.add_child(_label("01   PHASE TRAJECTORY", 12, TEAL))
	phase_column.add_child(_label("Position × velocity", 20))
	_phase.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_phase.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	phase_column.add_child(_phase)
	phase_column.add_child(_label("Click a trajectory point to set the shared time cursor", 11, MUTED))
	var energy_card := _panel()
	energy_card.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panels.add_child(energy_card)
	var energy_column := VBoxContainer.new()
	energy_card.add_child(energy_column)
	energy_column.add_child(_label("02   STATE-SPACE SURFACE", 12, TEAL))
	var energy_header := HBoxContainer.new()
	energy_column.add_child(energy_header)
	var energy_title := _label("Energy landscape", 20)
	energy_title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	energy_header.add_child(energy_title)
	_view_state = _label("WAITING", 11, AMBER)
	energy_header.add_child(_view_state)
	energy_column.add_child(_energy)
	energy_column.add_child(_label("Backend geometry · visual scales · drag to orbit / wheel to zoom", 11, MUTED))
	var timeline := _panel()
	column.add_child(timeline)
	var timeline_column := VBoxContainer.new()
	timeline.add_child(timeline_column)
	var cursor_row := HBoxContainer.new()
	timeline_column.add_child(cursor_row)
	_cursor_label = _label("PLAYBACK CURSOR   — s", 13, TEAL)
	_cursor_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	cursor_row.add_child(_cursor_label)
	cursor_row.add_child(_label("Independent of the analysis interval", 12, MUTED))
	_play = _button("Play", _toggle_play)
	cursor_row.add_child(_play)
	_slider = HSlider.new()
	_slider.custom_minimum_size.y = 26
	_slider.step = 0.001
	_slider.value_changed.connect(_slider_changed)
	timeline_column.add_child(_slider)
	_samples_row = HBoxContainer.new()
	_samples_row.add_theme_constant_override("separation", 12)
	column.add_child(_samples_row)
	## One card per declared channel; built when a snapshot names them.
	_sample_label = _label("SAMPLE INSPECTION  /  awaiting backend values", 12, MUTED)
	column.add_child(_sample_label)
	var footer := HBoxContainer.new()
	column.add_child(footer)
	var authority := _label("RETAINED RECORD → INSPECTION     •     RENDER GEOMETRY → PRESENTATION", 10, MUTED)
	authority.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	footer.add_child(authority)
	footer.add_child(_label("VERIFICATION  /  NOT CLAIMED", 10, AMBER))
	_set_interaction(false)


func _rebuild_channels(names: Array) -> void:
	## The record decides which channels exist. Rebuild the selector and the
	## readout cards from it rather than assuming any particular instrument.
	if names == _channel_names:
		return
	_channel_names = names.duplicate()
	_channel.clear()
	for name in _channel_names:
		_channel.add_item(str(name))
	for child in _samples_row.get_children():
		_samples_row.remove_child(child)
		child.queue_free()
	_value_labels.clear()
	for name in _channel_names:
		var sample_card := _panel()
		sample_card.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		_samples_row.add_child(sample_card)
		var contents := HBoxContainer.new()
		sample_card.add_child(contents)
		contents.add_child(_label(str(name).to_upper(), 12, MUTED))
		var value := _label("—", 23)
		value.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		value.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		_value_labels[name] = value
		contents.add_child(value)


func _set_interaction(enabled: bool) -> void:
	_slider.editable = enabled
	_channel.disabled = not enabled
	_apply_interval.disabled = not enabled
	_play.disabled = not enabled
	_interval_start.editable = enabled
	_interval_end.editable = enabled


func _on_status(state: String, detail: String) -> void:
	if not _view_supported:
		state = "unsupported view"
		detail = UNSUPPORTED_VIEW
	var ready := state == "ready"
	_status.text = "●  " + state.to_upper()
	_status.add_theme_color_override("font_color", TEAL if ready else AMBER)
	_detail.text = detail
	_detail.tooltip_text = detail
	_phase.set_stale(not ready)
	_energy.set_stale(not ready)
	_view_state.text = "RECORDED" if ready else ("STALE" if not _identity_run.is_empty() else "WAITING")
	_view_state.add_theme_color_override("font_color", TEAL if ready else AMBER)
	_set_interaction(ready)
	if not ready:
		_stop_playback()
		if not _view_supported:
			_sample_label.text = "TERMINAL ONLY  /  adapter values are not rendered by this viewport"
		elif not _identity_run.is_empty():
			_sample_label.text = "STALE RETAINED INSPECTION  /  reconnect to synchronize"


func _on_snapshot(snapshot: Dictionary) -> void:
	var run: Dictionary = snapshot.run
	var metadata: Dictionary = run.metadata
	# Generic adapters may carry point sampling, missing values and new channel
	# kinds. Do not reinterpret those through the legacy viewport's float casts.
	_view_supported = not run.has("run_schema") and not metadata.has("manifest")
	_phase.visible = _view_supported
	_energy.visible = _view_supported
	if not _view_supported:
		_identity_run = str(run.get("run_id", ""))
		_rebuild_channels([])
		_run_meta.text = str(run.get("instrument", "")) + "  /  terminal inspection available"
		_identity.text = "Session %s   /   Run %s   /   Evidence %s" % [snapshot.get("session_id", ""), _identity_run, run.get("evidence_id", "")]
		_identity.tooltip_text = _identity.text
		_on_status("unsupported view", UNSUPPORTED_VIEW)
		return
	_rebuild_channels(Dictionary(run.get("channels", {})).keys())
	var previous_run := _identity_run
	_identity_run = str(run.run_id)
	_duration = float(metadata.duration_s)
	_slider.max_value = _duration
	_interval_start.max_value = _duration
	_interval_end.max_value = _duration
	_run_meta.text = "%s   /   %s samples   /   %.0f Hz   /   %.2f s" % [str(run.instrument), int(metadata.sample_count), float(metadata.sample_rate_hz), _duration]
	_identity.text = "Session %s   /   Run %s   /   Evidence %s" % [snapshot.session_id, run.run_id, run.evidence_id]
	_identity.tooltip_text = _identity.text
	if previous_run != _identity_run:
		for value in _value_labels.values():
			value.text = "—"
		_sample_label.text = "SAMPLE INSPECTION  /  loading new run values"


func _on_run(run: Dictionary) -> void:
	if not _view_supported:
		return
	_phase.set_run(run)
	_energy.set_run(run)


func _on_selection(selection: Dictionary) -> void:
	if not _view_supported:
		return
	_synchronizing = true
	var cursor := float(selection.cursor_s)
	_slider.set_value_no_signal(cursor)
	if not _playing:
		_play_time = cursor
	_cursor_label.text = "PLAYBACK CURSOR   %.3f s" % cursor
	_revision.text = "revision %s" % int(selection.revision)
	var interval: Array = selection.interval_s
	_interval_label.text = "Analysis interval [%.3f, %.3f) s" % [float(interval[0]), float(interval[1])]
	_interval_start.set_value_no_signal(float(interval[0]))
	_interval_end.set_value_no_signal(float(interval[1]))
	_channel.select(_channel_names.find(str(selection.channel)))
	_synchronizing = false


func _on_sample(sample: Dictionary) -> void:
	if not _view_supported:
		return
	var values: Dictionary = sample.values
	var units: Dictionary = sample.units
	for channel in _value_labels:
		if values.has(channel) and (values[channel] is float or values[channel] is int):
			_value_labels[channel].text = "%.6f %s" % [float(values[channel]), str(units.get(channel, ""))]
		else:
			_value_labels[channel].text = "—"
	_sample_label.text = "BACKEND SAMPLE  %s   /   nearest retained time %.6f s   /   numerical values from the record" % [int(sample.sample_index), float(sample.time_s)]
	_phase.set_sample(int(sample.sample_index))
	_energy.set_sample(int(sample.sample_index))


func _on_error(code: String, detail: String) -> void:
	_stop_playback()
	_detail.text = "%s: %s" % [code, detail]
	if code == "revision_conflict":
		_detail.text = "Another client changed the selection; refreshing the shared revision."


func _slider_changed(value: float) -> void:
	if not _synchronizing:
		_pick_time(value)


func _pick_time(value: float) -> void:
	_stop_playback()
	_client.update_selection({"cursor_s": value})


func _set_interval() -> void:
	_client.update_selection({"interval_s": [_interval_start.value, _interval_end.value]})


func _toggle_play() -> void:
	_playing = not _playing
	if _playing:
		_play_time = float(_client.selection.get("cursor_s", 0.0))
		if _play_time >= _duration:
			_play_time = 0.0
	_play.text = "Pause" if _playing else "Play"


func _stop_playback() -> void:
	_playing = false
	_play.text = "Play"


func _process(delta: float) -> void:
	if _playing and _client.online:
		_play_time = minf(_play_time + delta, _duration)
		_client.update_selection({"cursor_s": _play_time})
		if _play_time >= _duration:
			_stop_playback()
