extends Control
var history: Array = []
const COLORS = [Color("75d7cf"),Color("d9aeed"),Color("f5ca7f"),Color("f88879")]

func add_sample(rates: Dictionary) -> void:
	history.append([maxf(float(rates.get("light_L",0)),float(rates.get("light_R",0))),
		maxf(float(rates.get("odor_a",0)),float(rates.get("odor_b",0))),
		maxf(float(rates.get("touch",0)),float(rates.get("vibration",0))),float(rates.get("GF",0))])
	if history.size()>120: history.pop_front()
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO,size),Color("102320"))
	for y in [0.25,0.5,0.75]:
		draw_line(Vector2(0,size.y*y),Vector2(size.x,size.y*y),Color("29403a"))
	if history.size()<2: return
	for line in range(4):
		var points = PackedVector2Array()
		for i in range(history.size()):
			points.append(Vector2(size.x*float(i)/119.0,size.y-3-clampf(history[i][line]/120.0,0,1)*(size.y-6)))
		draw_polyline(points,COLORS[line],1.6,true)

