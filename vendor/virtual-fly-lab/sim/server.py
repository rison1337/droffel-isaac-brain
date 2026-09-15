"""TCP JSON-lines server: Godot is the client.

Protocol (newline-delimited JSON):
  client -> server: {"cmd": "stim", "name": "light", "value": 0.9, "side": 1}
                    (side: -1 left, 0 both, +1 right)
  server -> client: state dicts from Ports.readout() plus sps/t, ~30 Hz
"""
import json
import queue
import socket
import threading
import time
from pathlib import Path


class SimServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 9876,
                 port_file=None) -> None:
        self.stimuli_q: queue.SimpleQueue = queue.SimpleQueue()
        self._state: dict = {}
        self._state_lock = threading.Lock()
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # no SO_REUSEADDR here: on Windows it silently allows a second bind to the
        # same port and connections then go to the stale server; failing loudly
        # lets the port fallback do its job
        self.port = None
        candidates = [port + i for i in range(10)] + [27182 + i for i in range(10)]
        last_err: OSError | None = None
        for cand in candidates:
            try:
                self._srv.bind((host, cand))
                self.port = cand
                break
            except OSError as e:
                last_err = e
                if getattr(e, "winerror", None) == 10013:
                    print(f"port {cand} blocked (Windows reserved range), trying next...",
                          flush=True)
        if self.port is None:
            raise RuntimeError(f"no bindable port near {port}: {last_err}")
        if port_file is not None:
            try:
                Path(port_file).write_text(str(self.port))
            except OSError as e:
                print(f"could not write port file: {e}", flush=True)
        self._srv.listen(4)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()

    def set_state(self, state: dict) -> None:
        with self._state_lock:
            self._state = state

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._clients_lock:
                self._clients.append(conn)
            threading.Thread(target=self._recv_loop, args=(conn,), daemon=True).start()

    def _recv_loop(self, conn: socket.socket) -> None:
        try:
            f = conn.makefile("r", encoding="utf-8", newline="\n")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.stimuli_q.put(obj)
        except OSError:
            pass
        finally:
            self._drop(conn)

    def _broadcast_loop(self) -> None:
        while True:
            time.sleep(1.0 / 30.0)
            with self._state_lock:
                state = self._state
            if not state:
                continue
            data = (json.dumps(state) + "\n").encode("utf-8")
            with self._clients_lock:
                clients = list(self._clients)
            for c in clients:
                try:
                    c.sendall(data)
                except OSError:
                    self._drop(c)

    def _drop(self, conn: socket.socket) -> None:
        with self._clients_lock:
            if conn in self._clients:
                self._clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass
