extends Window
## Positions are dataset soma coordinates, activity is live LIF output.
signal experiment_requested(action: String)
const Chart = preload("res://scripts/brain_chart.gd")
var metadata: Dictionary
var positions = PackedVector3Array()
var active_mesh: MultiMesh
var base_mesh: MultiMesh
var viewport: SubViewport
var camera: Camera3D
var info: Label
var rates_label: Label
var chart: Control
var focus_choice: OptionButton
var orbit = 0.0
var elevation = 0.03
var distance = 46.0
var target = Vector3(0,-5,0)
var loaded = false
var shown_active = 0
var last_time = -1.0
var samples = 0
var latest: Dictionary = {}
var filter_group = ""
var filter_indices: Dictionary = {}
var _base_instance: MultiMeshInstance3D
var _active_instance: MultiMeshInstance3D
var arbor_materials: Dictionary = {}
var arbor_instances: Array[MeshInstance3D] = []
var arbors_visible = true
var arbor_button: Button
var profile_choice: OptionButton
var timing_label: Label
var controls_hint: Label

func label(value: String, font_size: int, color = Color("cee0d8")) -> Label:
	var l = Label.new()
	l.text = value
	l.add_theme_font_size_override("font_size",font_size)
	l.add_theme_color_override("font_color",color)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return l

func _ready() -> void:
	title = "Droffel — Brain activity"
	size = Vector2i(1040,780)
	min_size = Vector2i(880,680)
	position = Vector2i(80,50)
	close_requested.connect(hide)
	metadata = JSON.parse_string(FileAccess.get_file_as_string("res://data/brain_metadata.json"))
	var background = ColorRect.new()
	background.color = Color("0a1618")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	background.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(background)
	var layout = HBoxContainer.new()
	layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	layout.add_theme_constant_override("separation",0)
	add_child(layout)
	var left = VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	layout.add_child(left)
	var header = MarginContainer.new()
	for side in ["left","top","right","bottom"]: header.add_theme_constant_override("margin_"+side,18)
	left.add_child(header)
	var heading_box = VBoxContainer.new()
	header.add_child(heading_box)
	heading_box.add_child(label("FLY BRAIN / 3D",23,Color("96d5be")))
	heading_box.add_child(label("MaleCNS · live neural activity",13))
	var controls = HBoxContainer.new()
	heading_box.add_child(controls)
	for entry in [["cns","Whole CNS"],["brain","Brain"],["gf","Escape circuit"]]:
		var b = Button.new()
		b.text = entry[1]
		b.focus_mode = Control.FOCUS_NONE
		b.pressed.connect(_focus.bind(entry[0]))
		controls.add_child(b)
	arbor_button = Button.new()
	arbor_button.text = "Branches: on"
	arbor_button.focus_mode = Control.FOCUS_NONE
	arbor_button.pressed.connect(func():
		arbors_visible = not arbors_visible
		for instance in arbor_instances: instance.visible = arbors_visible
		arbor_button.text = "Branches: on" if arbors_visible else "Branches: off")
	controls.add_child(arbor_button)
	var c = SubViewportContainer.new()
	c.stretch = true
	c.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(c)
	viewport = SubViewport.new()
	viewport.own_world_3d = true
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	c.add_child(viewport)
	var world = Node3D.new()
	viewport.add_child(world)
	camera = Camera3D.new()
	camera.fov = 46
	camera.current = true
	world.add_child(camera)
	var env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color("0a1618")
	camera.environment = env
	var details = MarginContainer.new()
	details.add_theme_constant_override("margin_left",18)
	details.add_theme_constant_override("margin_bottom",15)
	left.add_child(details)
	var detail_box = VBoxContainer.new()
	details.add_child(detail_box)
	info = label("Loading brain activity...",13)
	detail_box.add_child(info)
	detail_box.add_child(label("Activity: cyan → amber → white",12))
	controls_hint = label("Drag to rotate · Scroll to zoom · Space: scare · P: pause",12,Color("839d93"))
	detail_box.add_child(controls_hint)
	var sidebar = PanelContainer.new()
	sidebar.custom_minimum_size.x = 286
	var panel_style = StyleBoxFlat.new()
	panel_style.bg_color = Color("142723")
	panel_style.content_margin_left = 18
	panel_style.content_margin_right = 18
	panel_style.content_margin_top = 20
	panel_style.content_margin_bottom = 18
	sidebar.add_theme_stylebox_override("panel",panel_style)
	layout.add_child(sidebar)
	var scroll = ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	sidebar.add_child(scroll)
	var column = VBoxContainer.new()
	column.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	column.add_theme_constant_override("separation",6)
	scroll.add_child(column)
	column.add_child(label("STIMULUS RESPONSE",15,Color("96d5be")))
	profile_choice = OptionButton.new()
	profile_choice.add_item("Real time")
	profile_choice.add_item("Detailed simulation")
	profile_choice.tooltip_text = "Real time: 2 ms step. Detailed: 0.2 ms step.\nChanging mode resets the simulation."
	profile_choice.item_selected.connect(func(index): experiment_requested.emit("profile_research" if index==1 else "profile_interactive"))
	column.add_child(profile_choice)
	timing_label = label("",12,Color("a4c2b4"))
	column.add_child(timing_label)
	column.add_child(label("Display group",12))
	focus_choice = OptionButton.new()
	for name in ["All active cells","Escape circuit","Smell","Kenyon cells","Touch","Vibration","Heat"]:
		focus_choice.add_item(name)
	focus_choice.item_selected.connect(_filter)
	column.add_child(focus_choice)
	for entry in [["scare","Scare / looming"],["firefly","Moving light"],["odor_a","Odor A"],["odor_b","Odor B"],["touch","Touch"],["vibration","Vibration"],["heat","Heat"],["sequence","Experiment cycle"],["stop_stimuli","Clear stimuli"]]:
		var b = Button.new()
		b.name = "Probe_"+entry[0]
		b.text = entry[1]
		b.add_theme_font_size_override("font_size",14)
		b.focus_mode = Control.FOCUS_NONE
		b.pressed.connect(func(): experiment_requested.emit(entry[0]))
		column.add_child(b)
	rates_label = label("",12)
	column.add_child(rates_label)
	chart = Chart.new()
	chart.custom_minimum_size = Vector2(230,65)
	column.add_child(chart)
	column.add_child(label("Last 12 s · 0–120 Hz\nVision / smell / touch / escape",12,Color("a5bcb0")))
	_build_cloud(world)
	_update_camera()

func _point_material() -> StandardMaterial3D:
	var mat = StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.no_depth_test = true
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	return mat

func _build_cloud(world: Node3D) -> void:
	var data = FileAccess.get_file_as_bytes("res://data/soma_positions_f32.bin").to_float32_array()
	var count = int(data.size()/3)
	positions.resize(count)
	var mesh = QuadMesh.new()
	mesh.size = Vector2(0.065,0.065)
	mesh.material = _point_material()
	base_mesh = MultiMesh.new()
	base_mesh.transform_format = MultiMesh.TRANSFORM_3D
	base_mesh.use_colors = true
	base_mesh.mesh = mesh
	base_mesh.instance_count = int(metadata.visible_somas)
	var visible_index = 0
	for i in range(count):
		var p = Vector3(data[i*3],data[i*3+1],data[i*3+2])
		positions[i] = p
		if p == Vector3.ZERO: continue
		base_mesh.set_instance_transform(visible_index,Transform3D(Basis(),p))
		base_mesh.set_instance_color(visible_index,Color(0.22,0.42,0.43,0.20))
		visible_index += 1
		if i%20000 == 0: await get_tree().process_frame
	_base_instance = MultiMeshInstance3D.new()
	_base_instance.multimesh = base_mesh
	world.add_child(_base_instance)
	var bright = QuadMesh.new()
	bright.size = Vector2(0.17,0.17)
	bright.material = _point_material()
	active_mesh = MultiMesh.new()
	active_mesh.transform_format = MultiMesh.TRANSFORM_3D
	active_mesh.use_colors = true
	active_mesh.mesh = bright
	active_mesh.instance_count = 6002
	active_mesh.visible_instance_count = 0
	_active_instance = MultiMeshInstance3D.new()
	_active_instance.multimesh = active_mesh
	world.add_child(_active_instance)
	_build_arbors(world)
	loaded = true
	update_state(latest)

func _build_arbors(world: Node3D) -> void:
	if not FileAccess.file_exists("res://data/neuron_arbors.json"): return
	var records = JSON.parse_string(FileAccess.get_file_as_string("res://data/neuron_arbors.json"))
	var file = FileAccess.open("res://data/neuron_arbors_f32.bin",FileAccess.READ)
	for record in records:
		file.seek(int(record.byte_offset))
		var raw = file.get_buffer(int(record.segments)*24).to_float32_array()
		var vertices = PackedVector3Array()
		vertices.resize(int(raw.size()/3))
		for i in range(vertices.size()): vertices[i] = Vector3(raw[i*3],raw[i*3+1],raw[i*3+2])
		var arrays = []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = vertices
		var mesh = ArrayMesh.new()
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES,arrays)
		var mat = StandardMaterial3D.new()
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.albedo_color = Color(0.22,0.46,0.43,0.45)
		mat.no_depth_test = true
		var instance = MeshInstance3D.new()
		instance.mesh = mesh
		instance.material_override = mat
		world.add_child(instance)
		arbor_materials[int(record.index)] = mat
		arbor_instances.append(instance)
	file.close()

func update_state(state: Dictionary) -> void:
	latest = state
	if not loaded or state.is_empty() or not visible: return
	var cloud = state.get("cloud",[])
	profile_choice.select(1 if state.get("profile","")=="research" else 0)
	timing_label.text = "Simulation speed: %.2f×" % float(state.get("speed_ratio",1))
	for mat in arbor_materials.values(): mat.albedo_color = Color(0.22,0.46,0.43,0.35)
	for entry in state.get("arbor_rates",[]):
		var idx = int(entry[0])
		if arbor_materials.has(idx):
			var strength = clampf(float(entry[1])/80,0,1)
			arbor_materials[idx].albedo_color = Color(0.22,0.46,0.43,0.35).lerp(Color(1.0,0.75,0.38,0.95),strength)
	var used = 0
	for entry in cloud:
		var idx = int(entry[0])
		if idx<0 or idx>=positions.size() or positions[idx]==Vector3.ZERO: continue
		if not filter_indices.is_empty() and not filter_indices.has(idx): continue
		if used>=6002: break
		var rate = float(entry[1])
		var color = Color("79d6d0").lerp(Color("f5b967"),clampf(rate/60,0,1))
		color = color.lerp(Color("fff1dc"),clampf((rate-60)/80,0,1))
		color.a = 0.75
		active_mesh.set_instance_transform(used,Transform3D(Basis(),positions[idx]))
		active_mesh.set_instance_color(used,color)
		used += 1
	active_mesh.visible_instance_count = used
	shown_active = used
	info.text = "%d active cells%s" % [int(state.get("responding",0))," · PAUSED" if state.get("paused",false) else ""]
	var r = state.get("rates",{})
	rates_label.text = "Vision L/R   %.1f / %.1f Hz\nSmell A/B    %.1f / %.1f Hz\nTouch  %.1f Hz\nVibration  %.1f · Heat  %.1f Hz\nKenyon cells  %.1f Hz\nEscape (GF)  %.1f Hz" % [float(r.get("light_L",0)),float(r.get("light_R",0)),float(r.get("odor_a",0)),float(r.get("odor_b",0)),float(r.get("touch",0)),float(r.get("vibration",0)),float(r.get("heat",0)),float(r.get("KC",0)),float(r.get("GF",0))]
	var sim_time = float(state.get("sim_time",0))
	if sim_time != last_time:
		if sim_time<last_time: chart.history.clear()
		chart.add_sample(r)
		samples += 1
		last_time = sim_time

func _focus(which: String) -> void:
	orbit = 0
	elevation = 0.03
	match which:
		"cns":
			target = Vector3(0,-5,0)
			distance = 46
		"brain":
			var p = metadata.brain_center
			target = Vector3(p[0],p[1],p[2])
			distance = 30
		"gf":
			var p = metadata.gf[0].position
			target = Vector3(p[0],p[1],p[2])
			distance = 17
			focus_choice.select(1)
			_filter(1)
	_update_camera()

func _filter(index: int) -> void:
	filter_indices.clear()
	var names = [[],["loom_L","loom_R","GF"],["odor","ALPN"],["KC","MBON"],["touch"],["vibration"],["heat"]][index]
	for name in names:
		for idx in metadata.groups[name]: filter_indices[int(idx)] = true
	update_state(latest)

func _update_camera() -> void:
	if not camera: return
	camera.position = target+Vector3(sin(orbit)*cos(elevation),sin(elevation),cos(orbit)*cos(elevation))*distance
	camera.look_at(target)

func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		if event.position.x < size.x-290:
			orbit -= event.relative.x*0.007
			elevation = clampf(elevation+event.relative.y*0.006,-1.3,1.3)
			_update_camera()
	if event is InputEventMouseButton and event.pressed and event.position.x<size.x-290:
		if event.button_index==MOUSE_BUTTON_WHEEL_UP: distance = maxf(6,distance-2)
		if event.button_index==MOUSE_BUTTON_WHEEL_DOWN: distance = minf(90,distance+2)
		_update_camera()
	if event is InputEventKey and event.pressed and not event.echo:
		if event.physical_keycode == KEY_SPACE: experiment_requested.emit("scare")
		if event.physical_keycode == KEY_P: experiment_requested.emit("pause")
		if event.physical_keycode == KEY_ESCAPE: hide()

func _process(_delta: float) -> void:
	if viewport:
		viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS if visible else SubViewport.UPDATE_DISABLED
