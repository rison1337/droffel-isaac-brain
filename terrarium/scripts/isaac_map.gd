extends Control
var observation: Dictionary = {}
func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO,size),Color("102321"))
	if observation.is_empty():
		var font = ThemeDB.fallback_font
		var text = "Room map will appear when a run starts."
		var width = font.get_string_size(text,HORIZONTAL_ALIGNMENT_LEFT,-1,16).x
		draw_string(font,Vector2((size.x-width)/2,size.y/2),text,HORIZONTAL_ALIGNMENT_LEFT,-1,16,Color("839d93"))
		return
	var bounds = observation.get("bounds",[[0,0],[640,400]])
	var lo = Vector2(bounds[0][0],bounds[0][1])-Vector2(40,40)
	var hi = Vector2(bounds[1][0],bounds[1][1])+Vector2(40,40)
	var scale_factor = minf(size.x/(hi.x-lo.x),size.y/(hi.y-lo.y))
	var point = func(p): return (Vector2(p[0],p[1])-lo)*scale_factor
	var origin = observation.get("grid_origin",[0,0])
	var width = int(observation.get("grid_width",15))
	for cell in observation.get("grid",[]):
		if int(cell[1])==0: continue
		var pos = [origin[0]+(int(cell[0])%width)*40,origin[1]+floori(float(cell[0])/width)*40]
		draw_rect(Rect2(point.call(pos)-Vector2.ONE*19*scale_factor,Vector2.ONE*38*scale_factor),Color("304841"))
	for entry in [["pickups",Color("c9c66a"),4],["doors",Color("6ea8e2"),8],["enemies",Color("e98272"),8],["bullets",Color("ffc484"),3]]:
		for entity in observation.get(entry[0],[]): draw_circle(point.call(entity.pos),entry[2]*scale_factor,entry[1])
	for hazard in observation.get("hazards",[]):
		draw_circle(point.call(hazard.pos),maxf(8,float(hazard.get("size",12)))*scale_factor,Color("eeb065"))
	if observation.has("player"):
		var p = point.call(observation.player.pos)
		draw_circle(p,9*scale_factor,Color("a0f4cf"))
		var move = observation.get("applied",[0,0])
		draw_line(p,p+Vector2(move[0],move[1])*35,Color.WHITE,2)
