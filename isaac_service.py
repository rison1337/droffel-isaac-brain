"""Dedicated full MaleCNS instance for Isaac and the 3D observer."""
import argparse
import json
import math
import os
import queue
import socket
import threading
import time
from pathlib import Path

from droffel_sim import FlyNervousSystem, SimServer, ROOT
from isaac_policy import IsaacPolicy
from isaac_learning import targetable


def write_snapshot(path, value):
    """Publish a complete JSON document, never an in-progress truncated file."""
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value),encoding="utf-8")
    try:
        os.replace(tmp,path)
    except PermissionError:
        # A Windows reader may briefly hold the old file. Keep that complete
        # snapshot and retry with the next report rather than crash the brain.
        pass


class GameBridge:
    def __init__(self, port, token):
        self.token = token
        self.latest, self.received_at = None, -math.inf
        self.action = None
        self.controls = queue.Queue(maxsize=8)
        self.connected = False
        self.error = ""
        self.lock = threading.Lock()
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", port))
        self.server.listen(1)
        threading.Thread(target=self._accept, daemon=True).start()

    def snapshot(self):
        with self.lock:
            return self.latest, self.received_at, self.connected

    def _accept(self):
        while True:
            conn, _ = self.server.accept()
            conn.settimeout(.04)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buf, last_sent = b"", 0.
            try:
                while True:
                    try:
                        part = conn.recv(65536)
                        if not part:
                            break
                        buf += part
                        if len(buf)>1_000_000:
                            break
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            try:
                                msg = json.loads(line)
                            except (ValueError, UnicodeError):
                                continue
                            if not isinstance(msg, dict) or msg.get("token")!=self.token:
                                raise ValueError("Game bridge authentication failed")
                            if msg.get("kind")=="observation" and isinstance(msg.get("player"),dict):
                                msg.pop("token", None)
                                with self.lock:
                                    self.latest, self.received_at, self.connected = msg, time.monotonic(), True
                                    self.error = ""
                    except socket.timeout:
                        pass
                    while not self.controls.empty():
                        msg = self.controls.get_nowait()
                        conn.sendall((json.dumps(dict(msg,token=self.token))+"\n").encode())
                    if self.action and time.monotonic()-last_sent>.033:
                        conn.sendall((json.dumps(dict(self.action,token=self.token))+"\n").encode())
                        last_sent = time.monotonic()
            except (OSError, ValueError) as exc:
                self.error = str(exc)
            finally:
                conn.close()
                with self.lock:
                    self.connected, self.action = False, None
                    self.received_at = -math.inf

    def control(self, value, seq):
        obs, _, connected = self.snapshot()
        if not obs or not connected:
            return
        try:
            self.controls.put_nowait({"kind":"control", "value":bool(value), "seq":seq,
                                      "session":obs["session"], "epoch":obs["epoch"]})
        except queue.Full:
            pass


def playable(obs, age, connected):
    return bool(connected and obs and age<.35 and obs.get("armed") and not obs.get("paused")
                and not obs.get("dead") and obs.get("controls",False) and obs.get("players")==1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds",type=float,default=0)
    ap.add_argument("--no-memory",action="store_true",help="Fixed tactics; never read or write learned experience")
    args = ap.parse_args()
    config = json.loads((ROOT/"data/isaac_connection.json").read_text())
    bridge = GameBridge(config["port"],config["token"])
    fly = FlyNervousSystem(profile="interactive")
    ui = SimServer(port=9886,port_file=ROOT/"data/isaac_sim_port.txt")
    policy = IsaacPolicy(ROOT/"data/isaac_learning.json",learning=not args.no_memory)
    started = now = next_step = last_report = last_plan = time.monotonic()
    running, steps, plan, last_obs = True, 0, None, None
    control_seq = int(time.time()*1000)
    pulses = {}
    metrics = {"observations":0,"action_packets":0,"rooms_seen":0,"distance":0.,"tears":0,"damage":0,"enemy_removals":0}
    previous = None
    (ROOT/"logs").mkdir(exist_ok=True)
    print("READY ISAAC "+str(ui.port),flush=True)
    with (ROOT/"logs/isaac_trace.jsonl").open("w",encoding="utf-8") as trace:
        while running:
            now = time.monotonic()
            while not ui.stimuli_q.empty():
                cmd = ui.stimuli_q.get()
                if not isinstance(cmd,dict):
                    continue
                if cmd.get("cmd")=="quit":
                    running = False
                elif cmd.get("cmd")=="isaac_control":
                    control_seq += 1
                    bridge.control(cmd.get("value",False),control_seq)
                elif cmd.get("cmd")=="isaac_memory_mode" and isinstance(cmd.get("enabled"),bool):
                    if cmd["enabled"] != policy.experience.enabled:
                        control_seq += 1
                        bridge.control(False,control_seq)
                        bridge.action = None
                        policy = IsaacPolicy(ROOT/"data/isaac_learning.json",learning=cmd["enabled"])
                        plan = last_obs = previous = None
                elif cmd.get("cmd")=="isaac_probe" and cmd.get("name") in ("scare","odor_a","odor_b","touch","vibration","heat","stop_stimuli"):
                    name = cmd["name"]
                    if name=="stop_stimuli": pulses.clear()
                    else: pulses[name] = now+2
            obs, received_at, connected = bridge.snapshot()
            active = playable(obs,now-received_at,connected)
            observation_key = lambda o:(o.get("session"),o.get("stage"),o.get("stage_type"),o.get("room"),o.get("epoch")) if o else None
            changed_context = observation_key(obs)!=observation_key(last_obs)
            if obs and now-received_at<.5 and obs is not last_obs and (changed_context or now-last_plan>=.065):
                try:
                    plan = policy.plan(obs)
                    metrics["observations"] += 1
                    metrics["rooms_seen"] = plan["room_count"]
                    metrics["tears"] = obs.get("tears",0)
                    if previous and previous["session"]==obs["session"]:
                        hp = lambda o:o["player"]["hearts"]+o["player"]["soul"]
                        metrics["damage"] += max(0,hp(previous)-hp(obs))
                        if (previous["stage"],previous["room"])==(obs["stage"],obs["room"]):
                            from isaac_policy import distance
                            metrics["distance"] += distance(previous["player"]["pos"],obs["player"]["pos"])
                            old_ids = {e["id"] for e in previous["enemies"]}
                            new_ids = {e["id"] for e in obs["enemies"]}
                            # Disappearances are reported as removals, not proven kills.
                            metrics["enemy_removals"] += len(old_ids-new_ids)
                    previous = obs
                    last_obs, last_plan = obs, now
                except (KeyError,TypeError,ValueError) as exc:
                    print("Bad observation: "+str(exc),flush=True)
                    plan = None
            sensory = policy.encode(plan) if plan and connected and now-received_at<.5 and not obs.get("paused") else {}
            for name, end in list(pulses.items()):
                if now>end:
                    del pulses[name]
                elif name=="scare":
                    sensory.update(loom_L=1.,loom_R=1.)
                else:
                    sensory[name] = 1.
            fly.sense(sensory)
            paused = bool(obs and obs.get("paused"))
            if not paused:
                count = max(0,min(8,int((now-next_step)/fly.brain.dt_s)+1))
                for _ in range(count): fly.step()
                steps += count
                next_step += count*fly.brain.dt_s
                if now-next_step>.15: next_step = now
            else: next_step = now
            if now-last_report>=.1:
                state = fly.state(cloud=True)
                output = policy.decode(plan,state,obs) if active and plan and observation_key(obs)==observation_key(last_obs) else {"move":[0.,0.],"shoot":[0.,0.],"mode":"stopped"}
                if obs and connected:
                    bridge.action = dict(output,kind="action",session=obs["session"],room=obs["room"],
                                         frame=obs["frame"],epoch=obs["epoch"])
                    metrics["action_packets"] += 1
                else: bridge.action = None
                game_state = {"connected":connected,"playing":active,"age":round(min(999.,now-received_at),3),
                    "mode":output["mode"],"move":output["move"],"shoot":output["shoot"],
                    "planned_move":plan["move"] if active and plan else [0.,0.],
                    "planned_shoot":plan["shoot"] if active and plan else [0.,0.],
                    "target":plan.get("target") if active and plan else None,
                    "goal":plan.get("goal") if active and plan else None,
                    "objective":plan.get("objective") if active and plan else None,
                    "skipped_targets":plan.get("skipped_targets",[]) if active and plan else [],
                    "movement_blocked":output.get("movement_blocked",False),
                    "danger":round(plan["danger"],2) if active and plan else 0.,
                    "metrics":dict(metrics),"status":obs.get("status","") if obs else "Start a run in Isaac",
                    "room":obs.get("room",-1) if obs else -1,"stage":obs.get("stage",0) if obs else 0,
                    "hearts":obs["player"]["hearts"] if obs else 0,"soul":obs["player"]["soul"] if obs else 0,
                    "coins":obs["player"].get("coins",0) if obs else 0,
                    "active_item":obs["player"].get("active_item",0) if obs else 0,
                    "active_charge":obs["player"].get("active_charge",0) if obs else 0,
                    "card":obs["player"].get("card",0) if obs else 0,
                    "enemies":sum(targetable(e) for e in obs.get("enemies",[])) if obs else 0,"paused":paused,"dead":obs.get("dead",False) if obs else False,
                    "input_calls":obs.get("input_calls",0) if obs else 0,
                    "shoot_calls":obs.get("shoot_calls",0) if obs else 0,
                    "applied_shoot":obs.get("applied_shoot",[0,0]) if obs else [0,0],
                    "armed":obs.get("armed",False) if obs else False,
                    "restart_pending":obs.get("restart_pending",False) if obs else False,
                    "restart_seconds":obs.get("restart_seconds",0) if obs else 0,
                    "session":obs.get("session","") if obs else "",
                    "frame":obs.get("frame",0) if obs else 0,
                    "hazards":len(obs.get("hazards",[])) if obs else 0,
                    "learning":policy.experience.summary(),
                    "memory_mode":"adaptive" if policy.experience.enabled else "fixed",
                    "mod_version":obs.get("mod_version","1.0") if obs else "",
                    "error":bridge.error}
                sps = steps/max(now-last_report,1e-6)
                state.update(sps=sps,speed_ratio=min(1.,sps*fly.brain.dt_s),paused=paused,isaac=game_state,observation=obs or {})
                ui.set_state(state)
                status = {k:v for k,v in state.items() if k not in ("cloud","observation")}
                write_snapshot(ROOT/"data/isaac_status.json",status)
                if obs:
                    write_snapshot(ROOT/"data/isaac_observation.json",obs)
                trace.write(json.dumps({"wall":round(now-started,2),"game":game_state,"rates":state["rates"],
                                        "sensory":sensory,"gf_spikes":state["gf_spikes"],"sim_time":state["sim_time"]})+"\n")
                trace.flush()
                last_report, steps = now, 0
            if args.seconds and now-started>=args.seconds: running = False
            time.sleep(.001)
    bridge.control(False,control_seq+1)
    bridge.action = None
    policy.experience.finish_window()
    policy.experience.save()
    time.sleep(.1)
    print("STOPPED ISAAC",flush=True)


if __name__=="__main__":
    main()
