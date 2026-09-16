extends SceneTree
const Client = preload("res://scripts/sim_client.gd")
const BrainView = preload("res://scripts/brain_view.gd")
const MapView = preload("res://scripts/isaac_map.gd")
var client
var brain
var status: Label
var metrics: Label
var note: Label
var map
var qa = false
var qa_time = 0.0

func _initialize() -> void:
	root.title = "DROFFEL / ISAAC"
	root.size = Vector2i(880,610)
	root.position = Vector2i(40,40)
	root.close_requested.connect(_stop)
	auto_accept_quit = false
	for arg in OS.get_cmdline_user_args():
		if arg=="--qa-isaac": qa = true
	call_deferred("_build")

func _build() -> void:
	var bg = ColorRect.new()
	bg.color = Color("0a1816")
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.add_child(bg)
	var margin = MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left","right","top","bottom"]: margin.add_theme_constant_override("margin_"+side,24)
	root.add_child(margin)
	var col = VBoxContainer.new()
	col.add_theme_constant_override("separation",14)
	margin.add_child(col)
	var title = Label.new()
	title.text = "DROFFEL / ISAAC"
	title.add_theme_font_size_override("font_size",26)
	col.add_child(title)
	status = Label.new()
	status.text = "Open a solo run in Isaac. Press F6 to enable the fly; F7 to stop."
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	col.add_child(status)
	var row = HBoxContainer.new()
	col.add_child(row)
	for entry in [["start","Enable fly"],["stop","STOP"],["brain","Brain activity"],["scare","Scare"]]:
		var b = Button.new()
		b.text = entry[1]
		b.pressed.connect(_action.bind(entry[0]))
		row.add_child(b)
	map = MapView.new()
	map.custom_minimum_size = Vector2(700,310)
	map.size_flags_vertical = Control.SIZE_EXPAND_FILL
	col.add_child(map)
	metrics = Label.new()
	metrics.text = "Waiting for room data..."
	col.add_child(metrics)
	note = Label.new()
	note.text = "F6: enable fly · F7 / Esc: stop"
	note.add_theme_font_size_override("font_size",13)
	col.add_child(note)
	client = Client.new()
	root.add_child(client)
	client.state_updated.connect(_state)
	client.start("127.0.0.1",9886)
	brain = BrainView.new()
	root.add_child(brain)
	brain.title = "Droffel — Isaac brain activity"
	brain.experiment_requested.connect(_probe)
	brain.profile_choice.disabled = true
	brain.profile_choice.hide()
	brain.timing_label.hide()
	brain.controls_hint.text = "Drag to rotate · Scroll to zoom · Space: scare · P: stop"
	for key in ["firefly","sequence"]:
		var b = brain.find_child("Probe_"+key,true,false)
		if b: b.hide()
	brain.show()

func _action(name: String) -> void:
	match name:
		"start": client.send({"cmd":"isaac_control","value":true})
		"stop": client.send({"cmd":"isaac_control","value":false})
		"brain": brain.show()
		"scare": _probe("scare")

func _probe(name: String) -> void:
	if name=="pause": _action("stop")
	else: client.send({"cmd":"isaac_probe","name":name})

func _state(state: Dictionary) -> void:
	brain.update_state(state)
	var g = state.get("isaac",{})
	var m = g.get("metrics",{})
	var mode = {"combat":"combat","clearing_poop":"breaking obstacles","clearing_fire":"extinguishing fire","pickup":"pickup","door":"next room","exit":"next floor","entering":"entering","waiting":"waiting","waiting_vulnerable":"waiting for target","stopped":"stopped"}.get(g.get("mode","waiting"),"waiting")
	note.text = ("No learning" if g.get("memory_mode","")=="fixed" else "Learning enabled")+" · F6: enable fly · F7 / Esc: stop"
	status.text = ("FLY PLAYING · " if g.get("playing",false) else "CONTROL OFF · ")+mode
	if not g.get("connected",false): status.text = "Open a solo run in Isaac, then press F6."
	elif g.get("restart_pending",false): status.text = "New attempt in %.1f s · F7 to cancel" % float(g.get("restart_seconds",0))
	elif g.get("dead",false): status.text = "Run over." if not g.get("armed",false) else "Run over. Waiting for a new attempt..."
	elif g.get("paused",false): status.text = "Transition / pause - control resumes automatically." if g.get("armed",false) else "Game paused. Press F6 to enable the fly."
	if g.get("connected",false) and g.get("mod_version","") != "1.4.1": status.text = "Update required: restart Isaac through START_ISAAC.cmd."
	if state.has("observation"):
		map.observation = state.observation if g.get("connected",false) else {}
		map.queue_redraw()
	metrics.visible = g.get("connected",false)
	metrics.text = "Floor %d · Room %d · Enemies %d\nHealth %.1f + %.1f soul hearts · Rooms visited %d" % [int(g.get("stage",0)),int(g.get("room",0)),int(g.get("enemies",0)),float(g.get("hearts",0))/2,float(g.get("soul",0))/2,int(m.get("rooms_seen",0))]

func _process(delta: float) -> bool:
	qa_time += delta
	if qa and qa_time>8 and brain and brain.loaded:
		qa = false
		_capture_qa()
	return false

func _capture_qa() -> void:
	await RenderingServer.frame_post_draw
	if DisplayServer.get_name() != "headless":
		root.get_viewport().get_texture().get_image().save_png("res://../logs/isaac_dashboard.png")
		brain.viewport.get_texture().get_image().save_png("res://../logs/isaac_brain.png")
		brain.get_texture().get_image().save_png("res://../logs/isaac_brain_window.png")
	print("ISAAC DASHBOARD QA: loaded=",brain.loaded," arbors=",brain.arbor_instances.size()," samples=",brain.samples)
	quit(0 if brain.samples>0 else 1)

func _stop() -> void:
	if client:
		client.send({"cmd":"isaac_control","value":false})
		client.send({"cmd":"quit"})
	quit()
