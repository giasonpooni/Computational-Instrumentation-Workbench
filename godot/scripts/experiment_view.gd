extends VBoxContainer
## One selected bundle drives all panels. Selection is local presentation state.
const Plot = preload("res://scripts/experiment_plot.gd")
var client
var selected_bundle := ""
var selected_result := ""
var view: Dictionary = {}
var _session_id := ""
var _revision := -1
var _bundle_ids: Array[String] = []
var _catalog: ItemList
var _follow: CheckBox
var _summary: Label
var _status: Label
var _panels: OptionButton
var _plot = Plot.new()
var _numbers: TextEdit
var _graph: Tree
var _inspector: TextEdit
var _capabilities: TextEdit


func _ready() -> void:
	add_theme_constant_override("separation", 12)
	var heading := HBoxContainer.new()
	add_child(heading)
	var title := Label.new()
	title.text = "WORKBENCH  /  experiments, schematics and computation"
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	heading.add_child(title)
	_follow = CheckBox.new()
	_follow.text = "Follow new results"
	_follow.button_pressed = true
	heading.add_child(_follow)
	_status = Label.new()
	_status.text = "Waiting for the service"
	add_child(_status)
	_summary = Label.new()
	_summary.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_summary.text = "Run a shared workbench example to populate this view."
	add_child(_summary)
	var body := HSplitContainer.new()
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	add_child(body)
	var sidebar := VBoxContainer.new()
	sidebar.custom_minimum_size.x = 250
	body.add_child(sidebar)
	sidebar.add_child(_label("RETAINED OCCURRENCES"))
	_catalog = ItemList.new()
	_catalog.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_catalog.custom_minimum_size.y = 180
	_catalog.item_selected.connect(func(index: int):
		_follow.button_pressed = false
		_select_bundle(_bundle_ids[index]))
	sidebar.add_child(_catalog)
	sidebar.add_child(_label("SOURCES & BOUND OPERATIONS"))
	_capabilities = _text_box()
	_capabilities.size_flags_vertical = Control.SIZE_EXPAND_FILL
	sidebar.add_child(_capabilities)
	var content := HSplitContainer.new()
	body.add_child(content)
	var scientific := VBoxContainer.new()
	scientific.custom_minimum_size.x = 410
	scientific.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	content.add_child(scientific)
	_panels = OptionButton.new()
	_panels.item_selected.connect(_select_panel)
	scientific.add_child(_panels)
	_plot.custom_minimum_size.y = 180
	_plot.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scientific.add_child(_plot)
	_numbers = _text_box()
	_numbers.custom_minimum_size.y = 88
	scientific.add_child(_numbers)
	scientific.add_child(_label("INSTRUMENT DEPENDENCIES  /  select a result"))
	_graph = Tree.new()
	_graph.hide_root = true
	_graph.custom_minimum_size.y = 110
	_graph.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_graph.item_selected.connect(_select_node)
	scientific.add_child(_graph)
	var evidence := VBoxContainer.new()
	evidence.custom_minimum_size.x = 310
	content.add_child(evidence)
	var actions := HBoxContainer.new()
	evidence.add_child(actions)
	for entry in [["Context", "fusion_context"], ["Raw", "raw_observations"], ["Verification", "verification"]]:
		var button := Button.new()
		button.text = entry[0]
		var field: String = entry[1]
		button.pressed.connect(func():
			selected_result = ""
			client.inspect_artifact("")
			if field == "fusion_context" and view.get("fusion_context") == null:
				_show_json(view.get("object_context", {}))
			elif field == "raw_observations" and view.has("raw_declaration"):
				_show_json(view.raw_declaration)
			else:
				_show_json(view.get(field, {})))
		actions.add_child(button)
	_inspector = _text_box()
	_inspector.size_flags_vertical = Control.SIZE_EXPAND_FILL
	evidence.add_child(_inspector)
	var footer := _label("Read-only retained values · plots preserve separate occurrences · inspection does not replay or admit state")
	footer.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(footer)


func _label(value: String) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", 12)
	return label


func _text_box() -> TextEdit:
	var box := TextEdit.new()
	box.editable = false
	box.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	box.add_theme_font_size_override("font_size", 12)
	box.add_theme_color_override("font_readonly_color", Color("c4d0df"))
	return box


func attach(value) -> void:
	client = value
	client.snapshot_received.connect(apply_snapshot)
	client.experiment_received.connect(apply_view)
	client.artifact_received.connect(func(artifact: Dictionary):
		if not selected_result.is_empty():
			_show_json(artifact))
	client.status_changed.connect(func(state: String, _detail: String):
		_status.text = ("LIVE SESSION" if state == "ready" else "STALE / " + state.to_upper()) + " · catalog revision %s" % _revision)


func apply_snapshot(snapshot: Dictionary) -> void:
	var new_session := str(snapshot.get("session_id", ""))
	var catalog: Dictionary = snapshot.get("workbench", {})
	var revision := int(catalog.get("revision", 0))
	var changed_session := new_session != _session_id
	if not changed_session and revision <= _revision:
		if view.is_empty() and not selected_bundle.is_empty():
			client.inspect_experiment(selected_bundle)
		return
	if changed_session:
		selected_bundle = ""
		selected_result = ""
		view.clear()
		_plot.set_panel({})
		_panels.clear()
		_graph.clear()
		_numbers.text = ""
		_inspector.text = ""
		_summary.text = "Select a retained occurrence."
		client.inspect_artifact("")
		client.inspect_experiment("")
	_session_id = new_session
	_revision = revision
	_catalog.clear()
	_bundle_ids.clear()
	for bundle in catalog.get("bundles", []):
		_bundle_ids.append(str(bundle.bundle_id))
		_catalog.add_item("%02d  %s" % [_bundle_ids.size(), bundle.kind])
		_catalog.set_item_tooltip(_bundle_ids.size() - 1, str(bundle.bundle_id))
	var lines: Array[String] = []
	for source in catalog.get("sources", []):
		lines.append("SOURCE · %s\n%s\n" % [source.kind, source.label])
	for operation in catalog.get("operations", []):
		lines.append("%s · %s" % ["BOUND" if operation.available else "UNAVAILABLE", operation.operation_id])
	_capabilities.text = "\n".join(lines)
	if not _bundle_ids.is_empty():
		if selected_bundle.is_empty() or _follow.button_pressed:
			_select_bundle(_bundle_ids[-1])
		else:
			_catalog.select(_bundle_ids.find(selected_bundle))
	_status.text = ("LIVE SESSION" if client.online else "STALE") + " · catalog revision %s" % _revision


func _select_bundle(bundle_id: String) -> void:
	if selected_bundle == bundle_id and not view.is_empty():
		_catalog.select(_bundle_ids.find(bundle_id))
		return
	selected_bundle = bundle_id
	selected_result = ""
	view.clear()
	_plot.set_panel({})
	_numbers.text = ""
	_panels.clear()
	_graph.clear()
	_inspector.text = ""
	_summary.text = "Loading selected occurrence..."
	_catalog.select(_bundle_ids.find(bundle_id))
	client.inspect_artifact("")
	client.inspect_experiment(bundle_id)


func apply_view(value: Dictionary) -> void:
	if value.get("schema", "") != "ciw.experiment-view.v1" or value.get("bundle_id", "") != selected_bundle:
		return
	view = value
	var context: Dictionary = view.get("object_context", {}) if view.get("fusion_context") == null else view.fusion_context
	if view.get("fusion_context") == null:
		_summary.text = "%s\n%s · %s · fusion: not performed\n%s" % [view.label,
			context.object_kind, context.get("next_step", context.get("summary", "retained native result")), view.bundle_id]
	else:
		_summary.text = "%s\n%s · observability: %s · reconciliation: %s · detection: %s · declared isolability: %s\n%s" % [view.label,
			context.state_kind, context.get("observability", {}).get("status", "unresolved"),
			context.get("reconciliation", {}).get("status", "not_run"),
			context.get("fault_assessment", {}).get("detection", context.get("fault_assessment", {}).get("status", "not_run")),
			context.get("fault_assessment", {}).get("isolability", "not_run"), view.bundle_id]
	_plot.set_panel({})
	_numbers.text = ""
	_panels.clear()
	for panel in view.panels:
		_panels.add_item(panel.title)
	if not view.panels.is_empty():
		_select_panel(0)
	_graph.clear()
	var root := _graph.create_item()
	var roles: Dictionary = {view.evidence_id: "source evidence"}
	for node in view.graph.nodes:
		roles[node.result_id] = node.role.to_upper()
	for node in view.graph.nodes:
		var item := _graph.create_item(root)
		item.set_text(0, node.role.to_upper() + " · " + node.operation_id)
		item.set_metadata(0, node.result_id)
		item.set_tooltip_text(0, node.result_id + "\n" + node.execution_id)
		for reference in node.input_refs:
			var dependency := _graph.create_item(item)
			dependency.set_text(0, "← " + str(roles.get(reference, "external retained input")))
			dependency.set_tooltip_text(0, reference)
			dependency.set_selectable(0, false)
		item.collapsed = true
	if view.get("schematic") != null:
		for node in view.schematic.nodes:
			var item := _graph.create_item(root)
			item.set_text(0, node.id + " · " + node.kind)
			item.set_tooltip_text(0, JSON.stringify(node.attrs))
			item.set_selectable(0, false)
			for edge in view.schematic.edges:
				if edge.src == node.id:
					var link := _graph.create_item(item)
					link.set_text(0, edge.kind + " → " + edge.dst)
					link.set_selectable(0, false)
			item.collapsed = true
	_show_json(context)


func _select_panel(index: int) -> void:
	var panel: Dictionary = view.panels[index]
	_plot.set_panel(panel)
	var rows: Array[String] = []
	for i in panel.values.size():
		rows.append("%s = %s %s" % [panel.labels[i], JSON.stringify(panel.values[i]), panel.units[i]])
	rows.append("Full covariance: " + JSON.stringify(panel.covariance))
	rows.append("Basis: " + JSON.stringify(panel.context))
	rows.append("Source: " + JSON.stringify(panel.provenance))
	_numbers.text = "\n".join(rows)


func _select_node() -> void:
	var item := _graph.get_selected()
	if item == null or item.get_metadata(0) == null:
		return
	selected_result = str(item.get_metadata(0))
	_inspector.text = "Loading retained result..."
	client.inspect_artifact(selected_result)


func _show_json(value: Variant) -> void:
	_inspector.text = JSON.stringify(value, "  ")
