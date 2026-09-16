"""Launch local game integration and observer; never terminate the user's game."""
import json
import msvcrt
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from tools.install_isaac_mod import install

ROOT = Path(__file__).resolve().parent
LOGS = ROOT/"logs"


def set_existing_mode(enabled):
    """Switch the existing service and wait for its mode acknowledgement."""
    port=int((ROOT/"data/isaac_sim_port.txt").read_text())
    expected="adaptive" if enabled else "fixed"
    with socket.create_connection(("127.0.0.1",port),timeout=4) as conn:
        conn.sendall((json.dumps({"cmd":"isaac_memory_mode","enabled":enabled})+"\n").encode())
        deadline=time.monotonic()+5
        with conn.makefile("r",encoding="utf-8") as stream:
            while time.monotonic()<deadline:
                line=stream.readline()
                if not line: break
                if json.loads(line).get("isaac",{}).get("memory_mode")==expected: return
    raise RuntimeError("Close the existing Isaac dashboard, then open the requested launcher again.")


def launch_game_if_needed(game_path):
    if "--no-game" in sys.argv:
        return
    tasks = subprocess.check_output(["tasklist","/FI","IMAGENAME eq isaac-ng.exe","/FO","CSV","/NH"],
        creationflags=subprocess.CREATE_NO_WINDOW).decode(errors="replace")
    if '"isaac-ng.exe"' not in tasks.lower():
        subprocess.Popen([str(game_path/"isaac-ng.exe"),"--luadebug"],cwd=game_path)


def stop_orphan_service():
    """Free ports left by a launcher that was interrupted mid-startup.

    This talks only to the fixed local Droffel UI port. Isaac itself is never
    terminated here, so reopening the launcher cannot kill an existing run.
    """
    try:
        with socket.create_connection(("127.0.0.1", 9886), timeout=.5) as conn:
            conn.sendall(b'{"cmd":"quit"}\n')
    except OSError:
        return
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 29876), timeout=.2):
                time.sleep(.1)
        except OSError:
            return


def main():
    LOGS.mkdir(exist_ok=True)
    no_memory="--no-memory" in sys.argv
    lock = (LOGS/"isaac.lock").open("a+b")
    try:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        # The panel may remain open while the game is restarted for a mod update.
        game_path = Path(json.loads((ROOT/"game_paths.json").read_text())["isaac"]["path"])
        lock.close()
        set_existing_mode(not no_memory)
        launch_game_if_needed(game_path)
        return
    game_path = install()
    stop_orphan_service()
    python = ROOT/".venv/Scripts/python.exe"
    godot = ROOT/".tools/godot/Godot_v4.7.2-stable_win64.exe"
    service = view = None
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with (LOGS/"isaac_brain.log").open("w",encoding="utf-8") as brain_log, (LOGS/"isaac_view.log").open("w",encoding="utf-8") as view_log:
        try:
            service_args=[str(python),str(ROOT/"isaac_service.py")]+(["--no-memory"] if no_memory else [])
            service = subprocess.Popen(service_args,cwd=ROOT,env=env,
                stdout=brain_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic()+120
            while "READY ISAAC" not in (LOGS/"isaac_brain.log").read_text(encoding="utf-8",errors="replace"):
                if service.poll() is not None:
                    raise RuntimeError("Isaac brain failed; see logs/isaac_brain.log")
                if time.monotonic()>deadline:
                    raise TimeoutError("Isaac brain startup timeout")
                time.sleep(.25)
            port = int((ROOT/"data/isaac_sim_port.txt").read_text())
            args = [str(godot),"--path",str(ROOT/"terrarium"),"--script","res://scripts/isaac_dashboard.gd",
                "--",f"--sim-port={port}","--observation="+str(ROOT/"data/isaac_observation.json")]
            view = subprocess.Popen(args,cwd=ROOT,stdout=view_log,stderr=subprocess.STDOUT)
            launch_game_if_needed(game_path)
            (LOGS/"isaac_launch_state.json").write_text(json.dumps({"launcher_pid":os.getpid(),
                "backend_pid":service.pid,"godot_pid":view.pid,"port":port,
                "memory_mode":"fixed" if no_memory else "adaptive"},indent=2))
            while view.poll() is None:
                if service.poll() is not None:
                    try: view.wait(timeout=3)
                    except subprocess.TimeoutExpired: raise RuntimeError("Isaac neural service stopped")
                time.sleep(.25)
        finally:
            if service and service.poll() is None:
                try:
                    port = int((ROOT/"data/isaac_sim_port.txt").read_text())
                    with socket.create_connection(("127.0.0.1",port),timeout=1) as conn:
                        conn.sendall(b'{"cmd":"quit"}\n')
                    service.wait(timeout=3)
                except (OSError,ValueError,subprocess.TimeoutExpired): pass
            for child in (view,service):
                if child and child.poll() is None:
                    subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            lock.close()


if __name__=="__main__":
    try: main()
    except Exception as exc:
        (LOGS/"isaac_launcher_error.log").write_text(traceback.format_exc(),encoding="utf-8")
        import tkinter.messagebox
        tkinter.messagebox.showerror("Droffel / Isaac",str(exc)+"\n\n"+str(LOGS))
