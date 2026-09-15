"""Online contextual bandit for the engineered controller, not brain plasticity.

Three combat tactics learn mean reward from observed enemy HP loss, room clears,
player damage and deaths. Exploration is bounded to normal legal game actions.
Spatial costs are keyed by room geometry, never by a grid index shared by rooms.
"""
import hashlib
import json
import math
from pathlib import Path

TACTICS = {
    "balanced": {"distance":190., "risk":4.},
    "kite": {"distance":235., "risk":6.},
    "close": {"distance":145., "risk":3.5},
}
TRAPS = {33,42,44,201,202,203,218,235,236,893}


def targetable(e):
    return e.get("type") not in TRAPS and not e.get("hazard",False) and e.get("hp",1)>0


def layout_key(obs):
    # Decorations and destructible rocks do not change the room's identity.
    geometry = [obs.get("grid_width",15),len(obs.get("grid",[])),obs.get("room_type",1),
                [[i,t] for i,c,t in obs.get("grid",[]) if t in (7,8,9,15,16,25)]]
    return hashlib.sha256(json.dumps(geometry,separators=(",",":")).encode()).hexdigest()[:16]


def cell_index(obs):
    origin=obs.get("grid_origin",[0,0]); p=obs["player"]["pos"]
    x,y=[int(math.floor((p[i]-origin[i])/40+.5)) for i in (0,1)]
    return str(y*obs.get("grid_width",15)+x)


def context_for(obs):
    if obs.get("room_type")==5: return "boss"
    if len(obs.get("bullets",[]))>=3: return "projectiles"
    return "crowd" if len([e for e in obs.get("enemies",[]) if targetable(e)])>=4 else "few_enemies"


class Experience:
    def __init__(self,path=None,enabled=True):
        self.enabled=enabled
        self.path=Path(path) if path and enabled else None
        self.data={"schema":2,"runs":0,"deaths":0,"rooms":0,"damage":0.,"enemy_removals":0,
                   "caution":0.,"updates":0,"stuck_events":0,"contexts":{},"hazard_cells":{}}
        if self.path and self.path.exists():
            try:
                saved=json.loads(self.path.read_text(encoding="utf-8"))
                for key in ("runs","deaths","rooms","damage","enemy_removals","caution"):
                    value=saved.get(key,0)
                    if isinstance(value,(float,int)) and math.isfinite(value): self.data[key]=max(0,value)
                if saved.get("schema")==2:
                    for key in ("updates","stuck_events","contexts","hazard_cells"):
                        self.data[key]=saved.get(key,self.data[key])
            except (ValueError,OSError,TypeError): pass
        self.previous=None
        self.session=None
        self.reward=0.
        self.context=None
        self.tactic="balanced"
        self.window_start=0
        self.changed=False

    def save(self):
        if not self.enabled or not self.path: return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data,indent=2,allow_nan=False),encoding="utf-8")
        try: temp.replace(self.path); self.changed=False
        except PermissionError: pass

    def stats(self,context):
        return self.data["contexts"].setdefault(context,{name:{"n":0,"q":0.} for name in TACTICS})

    def choose(self,context):
        stats=self.stats(context)
        for name in TACTICS:
            if not stats[name]["n"]: return name
        total=sum(v["n"] for v in stats.values())
        return max(TACTICS,key=lambda name:stats[name]["q"]+.7*math.sqrt(math.log(total+2)/(stats[name]["n"]+1)))

    def update(self,context,tactic,reward):
        if not self.enabled: return
        record=self.stats(context)[tactic]
        value=max(-10.,min(4.,float(reward)))
        record["n"]+=1
        record["q"]+=(value-record["q"])/record["n"]
        self.data["updates"]+=1
        self.changed=True

    def finish_window(self):
        if self.context is not None:
            self.update(self.context,self.tactic,self.reward)
            self.save()
        self.context=None
        self.reward=0.

    def spatial_costs(self,obs):
        if not self.enabled: return {}
        return self.data["hazard_cells"].get(layout_key(obs),{})

    def mark_position(self,obs,amount):
        if not self.enabled: return
        layouts=self.data["hazard_cells"]
        key=layout_key(obs)
        if key not in layouts and len(layouts)>=256: layouts.pop(next(iter(layouts)))
        cells=layouts.setdefault(key,{})
        cell=cell_index(obs)
        cells[cell]=min(8.,cells.get(cell,0.)+amount)
        self.changed=True

    def consume(self,obs):
        if not self.enabled: return TACTICS["balanced"]
        prev=self.previous
        self.previous=obs
        if obs.get("armed") and self.session!=obs["session"]:
            self.finish_window()
            self.session=obs["session"]
            self.data["runs"]+=1
            self.save()
        same_run=bool(prev and prev["session"]==obs["session"])
        # Never attribute manual play to the automated controller.
        controlled=bool(same_run and prev.get("armed") and not prev.get("paused") and not prev.get("dead"))
        died=bool(same_run and prev.get("armed") and not prev.get("dead") and obs.get("dead"))
        frame=obs["frame"]
        same_room=bool(same_run and (prev.get("stage"),prev.get("stage_type"),prev["room"])==
                       (obs.get("stage"),obs.get("stage_type"),obs["room"]))
        outcome=0.
        cleared=False
        if controlled or died:
            hp=lambda o:o["player"].get("hearts",0)+o["player"].get("soul",0)
            damage=max(0.,hp(prev)-hp(obs))
            self.data["damage"]+=damage
            if damage:
                outcome-=4*damage
                self.data["caution"]=min(1.5,self.data["caution"]+.1*damage)
                self.mark_position(obs if same_room else prev,damage)
                self.save()
            if same_room:
                old={e["id"]:e for e in prev.get("enemies",[]) if targetable(e)}
                new={e["id"]:e for e in obs.get("enemies",[]) if targetable(e)}
                # Only HP decreases of still-visible enemies are damage evidence.
                dealt=sum(max(0.,old[i].get("hp",0)-new[i].get("hp",0)) for i in old.keys()&new.keys())
                outcome+=min(2.,dealt/15)
                self.data["enemy_removals"]+=len(old.keys()-new.keys())
                if obs.get("clear") and not prev.get("clear"):
                    cleared=True
                    outcome+=2
                    self.data["rooms"]+=1
                    self.data["caution"]=max(0.,self.data["caution"]-.03)
        if died:
            self.data["deaths"]+=1
            outcome-=8
        if self.context is not None:
            self.reward+=outcome
            if controlled and same_room:
                # A tactic that never deals damage must not tie one that makes
                # progress. Paused time and manual play incur no penalty.
                self.reward-=min(30,max(0,frame-prev["frame"]))/180
        combat=obs.get("armed") and not obs.get("paused") and not obs.get("dead") and any(targetable(e) for e in obs.get("enemies",[]))
        if self.context is not None and (not same_room or obs.get("dead") or obs.get("clear") or not obs.get("armed") or frame-self.window_start>=90):
            self.finish_window()
        if combat and self.context is None:
            self.context=context_for(obs)
            self.tactic=self.choose(self.context)
            self.window_start=frame
            self.reward=0.
        if died or cleared: self.save()
        return TACTICS[self.tactic]

    def summary(self):
        return {k:self.data[k] for k in ("runs","deaths","rooms","damage","updates","stuck_events","caution")} | {
            "enabled":self.enabled,"tactic":self.tactic,"context":self.context or "", "remembered_layouts":len(self.data["hazard_cells"])}
