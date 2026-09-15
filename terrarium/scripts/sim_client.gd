extends Node
## TCP client for the fly-brain sim server (newline-delimited JSON).

signal state_updated(state: Dictionary)
signal connection_changed(connected: bool)

var _tcp := StreamPeerTCP.new()
var _host := "127.0.0.1"
var _port := 9876
var _buf := PackedByteArray()
var _retry := 0.0
var _is_connected := false
var _got_first_state := false


func start(host: String, port: int) -> void:
	_host = host
	# the sim server writes its actual port here (it may fall back from the
	# default when Windows has reserved the port range)
	var f := FileAccess.open("res://data/sim_port.txt", FileAccess.READ)
	if f:
		var txt := f.get_as_text().strip_edges()
		if txt.is_valid_int():
			port = txt.to_int()
	# Test instances can use a separate neural service without touching this file.
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--sim-port="):
			port = arg.trim_prefix("--sim-port=").to_int()
	_port = port
	_tcp.connect_to_host(host, port)


func is_connected_to_sim() -> bool:
	return _is_connected


func send(cmd: Dictionary) -> void:
	if _is_connected:
		# plain newline-delimited UTF-8: put_utf8_string would prepend a 32-bit
		# length prefix the JSON server cannot parse
		_tcp.put_data((JSON.stringify(cmd) + "\n").to_utf8_buffer())


func _process(delta: float) -> void:
	_tcp.poll()
	var status := _tcp.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _is_connected:
			_is_connected = true
			_buf = PackedByteArray()
			connection_changed.emit(true)
		var avail := _tcp.get_available_bytes()
		if avail > 0:
			var res := _tcp.get_partial_data(avail)
			if res[0] == OK:
				_buf.append_array(res[1])
				_drain_lines()
	elif status == StreamPeerTCP.STATUS_CONNECTING:
		pass
	else:
		if _is_connected:
			_is_connected = false
			connection_changed.emit(false)
		_tcp.disconnect_from_host()
		_retry -= delta
		if _retry <= 0.0:
			_retry = 1.0
			_tcp = StreamPeerTCP.new()
			_tcp.connect_to_host(_host, _port)


func _drain_lines() -> void:
	while true:
		var idx := _buf.find(10)  # b'\n'
		if idx < 0:
			return
		var line := _buf.slice(0, idx).get_string_from_utf8()
		_buf = _buf.slice(idx + 1)
		var parsed: Variant = JSON.parse_string(line)
		if parsed is Dictionary:
			if not _got_first_state:
				_got_first_state = true
				print("[sim] receiving brain state (first packet ok)")
			state_updated.emit(parsed)
