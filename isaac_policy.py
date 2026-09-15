"""Engineered Isaac planner, sensory encoder and causal neural readout.

This is a hybrid controller, not a pretrained fly or a biological game policy.
Object coordinates come from the mod. Direction channels are artificial: visual
L/R encodes horizontal motion; two ORN populations encode vertical motion. The
brain's firing rates supply motor drive; recurrent GF activity boosts evasion.
"""
import heapq
import math
from collections import Counter
from isaac_learning import Experience, targetable


def unit(v):
    size = math.hypot(*v)
    return [v[0]/size, v[1]/size] if size > 1e-6 else [0., 0.]


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


class RoomGrid:
    def __init__(self, obs, clearance=9, costs=None):
        self.origin = obs.get("grid_origin", [0, 0])
        self.width = obs.get("grid_width", 15)
        self.cells = {int(i): (int(c), int(t)) for i, c, t in obs.get("grid", [])}
        self.flying = obs["player"].get("flying", False)
        self.clearance = clearance
        self.costs = costs or {}
        self.hazards = obs.get("hazards",[])
        self.poops = obs.get("poops",[])
        self.player_radius = obs["player"].get("size",10)
        self.hazard_cells = {i for i in self.cells if not self.hazard_safe(self.pos(i))}

    def index(self, p):
        x, y = [int(math.floor((p[i]-self.origin[i])/40+.5)) for i in (0, 1)]
        return y*self.width+x if 0 <= x < self.width and y >= 0 else -1

    def pos(self, idx):
        return [self.origin[0]+(idx % self.width)*40, self.origin[1]+(idx//self.width)*40]

    def walkable(self, idx):
        collision, typ = self.cells.get(idx, (4, 0))
        return idx not in self.hazard_cells and (collision == 0 or (self.flying and collision in (1, 2))) and (self.flying or typ not in (8, 9, 25))

    def hazard_safe(self,p):
        # Flying avoids floor spikes, but it does not make contact with a fire safe.
        return all(distance(p,h["pos"])>h.get("size",12)+self.player_radius+4 for h in self.hazards)

    def safe(self, p, radius=None):
        radius = self.clearance if radius is None else radius
        return self.hazard_safe(p) and all(self.walkable(self.index([p[0]+dx, p[1]+dy]))
                   for dx, dy in [(0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius)])

    def ray(self, a, b, target_cell=None):
        steps = max(1, int(distance(a, b)/12))
        for i in range(1, steps):
            idx = self.index([a[0]+(b[0]-a[0])*i/steps, a[1]+(b[1]-a[1])*i/steps])
            if idx == target_cell:
                continue
            collision, typ = self.cells.get(idx, (4, 0))
            # Floor spikes do not block tears; rocks and walls do.
            if collision not in (0, 1) or typ == 25:
                return False
            if typ == 14 and collision != 0:
                return False
        return True

    def route(self, start, goal):
        # Dijkstra over traversable cells; unreachable objects are not targets.
        source, target = self.index(start), self.index(goal)
        candidates = [i for i in self.cells if self.walkable(i)]
        if not candidates:
            return None
        if not self.walkable(target):
            target = min(candidates, key=lambda i: distance(self.pos(i), goal))
        frontier, costs, parent = [(0., source)], {source: 0.}, {}
        while frontier:
            cost, current = heapq.heappop(frontier)
            if current == target:
                chain = [current]
                while chain[-1] != source:
                    chain.append(parent[chain[-1]])
                chain.reverse()
                next_pos = self.pos(chain[1]) if len(chain)>1 else goal
                return next_pos, cost
            if cost != costs[current]:
                continue
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                x = current % self.width+dx
                nxt = current+dx+dy*self.width
                if not 0 <= x < self.width or not self.walkable(nxt):
                    continue
                new_cost = cost+40+min(120.,self.costs.get(str(nxt),0.)*18)
                if new_cost < costs.get(nxt, math.inf):
                    costs[nxt], parent[nxt] = new_cost, current
                    heapq.heappush(frontier, (new_cost, nxt))
        return None


class IsaacPolicy:
    def __init__(self, memory_path=None, learning=True):
        self.experience = Experience(memory_path,enabled=learning)
        self._reset_run()

    def _reset_run(self):
        self.visits = Counter()
        self.last_room = self.session = self.last_hp = self.target = None
        self.hurt_until = -1
        self.target_since = self.room_entered = 0
        self.ignored = set()
        self.last_move = [0.,0.]
        self.target_health = {}
        self.stuck_anchor = None
        self.reposition_until = -1
        self.firing_lane = 0
        self.item_cooldown_until = -1
        self.use_until = -1
        self.use_request = {}

    @staticmethod
    def danger(pos, obs):
        risk = 0.
        for e in obs.get("enemies", [])+obs.get("hazards", []):
            predicted = [e["pos"][i]+e.get("vel", [0, 0])[i]*5 for i in (0, 1)]
            gap = distance(pos, predicted)-e.get("size", 12)-12
            risk += 9*max(0., 1-gap/90)**2
        for b in obs.get("bullets", []):
            relative = [b["pos"][i]-pos[i] for i in (0, 1)]
            vel = b.get("vel", [0, 0])
            vv = vel[0]**2+vel[1]**2
            t = max(0., min(12., -sum(relative[i]*vel[i] for i in (0, 1))/max(vv, .001)))
            gap = math.hypot(*(relative[i]+vel[i]*t for i in (0, 1)))-b.get("size", 5)-10
            risk += 12*max(0., 1-gap/48)**2
        return min(100., risk)

    def plan(self, obs):
        p = obs["player"]["pos"]
        key = (obs.get("stage"), obs.get("stage_type"), obs["room"])
        if self.session != obs["session"]:
            self._reset_run()
            self.session = obs["session"]
        tactic = self.experience.consume(obs)
        grid = RoomGrid(obs, clearance=9+round(self.experience.data["caution"]*2),
                        costs=self.experience.spatial_costs(obs))
        if key != self.last_room:
            self.visits[key] += 1
            self.last_room, self.target, self.ignored = key, None, set()
            self.last_move, self.room_entered = [0.,0.], obs["frame"]
            self.target_health = {}
            self.stuck_anchor = None
            self.reposition_until = -1
            self.use_until = -1
        hp = obs["player"].get("hearts", 0)+obs["player"].get("soul", 0)
        if self.last_hp is not None and hp < self.last_hp:
            self.hurt_until = obs["frame"]+20
        self.last_hp = hp
        enemies = obs.get("enemies", [])
        threats = [e for e in enemies if targetable(e)]
        targets = [e for e in threats if e.get("vulnerable", True)]
        enemy = min(targets, key=lambda e: distance(p, e["pos"]), default=None)
        clearing_fire = False
        clearing_poop = False
        # A poop tile blocks tears.  Select that tile as a temporary target so
        # the fly opens a firing lane instead of staring at the enemy forever.
        if enemy is not None and not grid.ray(p, enemy["pos"]):
            line = [enemy["pos"][i]-p[i] for i in (0, 1)]
            length = max(1., distance(p, enemy["pos"]))
            blockers = [q for q in grid.poops
                        if 0 < sum((q["pos"][i]-p[i])*line[i] for i in (0, 1)) < length**2
                        and abs((q["pos"][0]-p[0])*line[1]-(q["pos"][1]-p[1])*line[0])/length < 35
                        and grid.ray(p, q["pos"], int(q["id"]))]
            if blockers:
                q = min(blockers, key=lambda item: distance(p, item["pos"]))
                enemy = {"id": -100000-int(q["id"]), "pos": q["pos"],
                         "vel": [0, 0], "size": 14, "hp": 1,
                         "vulnerable": True, "kind": "poop"}
                clearing_poop = True
        if enemy is None and not threats and obs.get("clear"):
            fires = [h for h in obs.get("hazards",[]) if h.get("kind")=="fire"
                     and h.get("destructible",False) and h.get("hp",0)>0
                     and distance(p,h["pos"])<280]
            enemy = min(fires,key=lambda h:distance(p,h["pos"]),default=None)
            clearing_fire = enemy is not None
        # Damage feedback: firing at an unchanged target for four game seconds
        # must provoke a new firing angle, not indefinite stationary shooting.
        if enemy:
            record = self.target_health.get(enemy["id"])
            hp = enemy.get("hp",1)
            if record is None or hp<record[0]-.05:
                self.target_health[enemy["id"]]=(hp,obs["frame"])
            elif obs["frame"]-record[1]>=120 and any(obs.get("applied_shoot",[0,0])):
                self.reposition_until=obs["frame"]+60
                self.firing_lane=(self.firing_lane+1)%4
                self.target_health[enemy["id"]]=(hp,obs["frame"])
        if obs.get("armed") and not obs.get("paused") and any(abs(v)>.2 for v in obs.get("applied",[0,0])):
            if self.stuck_anchor is None or distance(p,self.stuck_anchor[1])>12:
                self.stuck_anchor=(obs["frame"],p[:])
            elif obs["frame"]-self.stuck_anchor[0]>45:
                if self.experience.enabled:
                    self.experience.mark_position(obs,1.)
                    self.experience.data["stuck_events"]+=1
                    self.experience.save()
                self.reposition_until=obs["frame"]+45
                self.firing_lane=(self.firing_lane+1)%4
                self.stuck_anchor=(obs["frame"],p[:])
        else: self.stuck_anchor=None
        shoot, goal, mode = [0., 0.], None, "waiting"
        if enemy:
            ep = [enemy["pos"][i]+enemy.get("vel", [0, 0])[i]*4 for i in (0, 1)]
            target_cell = grid.index(ep) if clearing_poop else None
            delta = [ep[i]-p[i] for i in (0, 1)]
            axis = 0 if abs(delta[0]) >= abs(delta[1]) else 1
            shoot[axis] = 1. if delta[axis]>=0 else -1.
            # Find an accessible firing lane at a useful distance from target.
            options = []
            for lane,direction in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):
                for radius in (tactic["distance"]-50,tactic["distance"],tactic["distance"]+40):
                    point = [ep[i]+direction[i]*radius for i in (0, 1)]
                    if grid.safe(point) and grid.ray(point, ep, target_cell):
                        route = grid.route(p, point)
                        if route:
                            penalty = 350 if obs["frame"]<self.reposition_until and lane!=self.firing_lane else 0
                            options.append((route[1]+distance(p, point)*.2+self.danger(point, obs)*tactic["risk"]*9+penalty, route[0]))
            # Keep an already good firing lane. The old code kept moving even
            # when aligned; its fallback even walked towards unreachable foes.
            aligned = min(abs(delta[0]),abs(delta[1])) < max(12.,enemy.get("size",12))
            safe_distance = tactic["distance"]-60 < distance(p,ep) < tactic["distance"]+75
            fire_shot_from_here = (clearing_fire and aligned and 55 < distance(p,ep) < 300
                                   and grid.ray(p,ep,target_cell))
            alignment_goal = False
            if clearing_fire and not aligned:
                # First move onto the fire's horizontal or vertical line. A
                # diagonal approach can orbit around rocks forever and never
                # produce a valid tear direction.
                align_options = []
                for point in ([p[0], ep[1]], [ep[0], p[1]]):
                    if distance(point, ep) > 55 and grid.safe(point):
                        route = grid.route(p, point)
                        if route:
                            align_options.append((route[1], route[0]))
                if align_options:
                    goal = min(align_options, key=lambda item:item[0])[1]
                    alignment_goal = True
            if aligned and (safe_distance or fire_shot_from_here) and grid.ray(p,ep,target_cell) and self.danger(p,obs)<.4 and obs["frame"]>=self.reposition_until:
                goal = p
            elif not alignment_goal:
                goal = min(options, key=lambda pair: pair[0])[1] if options else None
            if goal is None or not grid.ray(p, ep, target_cell) or not aligned:
                # Enemy is behind a rock/wall or outside a firing lane.
                # Hold fire and route around it instead of firing forever.
                shoot = [0., 0.]
            elif mode == "combat" and distance(goal, p) > 8:
                # Reposition first.  A tear fired while crossing to a new
                # lane inherits the lateral movement and routinely misses.
                shoot = [0., 0.]
            mode = "clearing_poop" if clearing_poop else ("clearing_fire" if clearing_fire else "combat")
        elif obs.get("clear") and not threats:
            options = []
            for pick in obs.get("pickups", []):
                variant, subtype = pick["variant"], pick["subtype"]
                price = int(pick.get("price", 0))
                if price < 0 or pick["id"] in self.ignored or price > obs["player"].get("coins", 0):
                    continue
                if not grid.hazard_safe(pick["pos"]):
                    continue
                if variant==10:
                    if subtype not in (1,2,5,9) or obs["player"].get("hearts", 0)>=obs["player"].get("max_hearts", 0):
                        continue
                # Standard pickups plus grab bags, pills, trinkets and chest
                # variants.  These are all collectible entities; hazards and
                # room decorations are not reported as pickups by the mod.
                elif variant not in (20,30,40,50,51,52,53,54,55,56,57,58,59,69,70,100,300,350):
                    continue
                if variant==300 and obs["player"].get("card",0):
                    continue
                route = grid.route(p, pick["pos"])
                if route:
                    options.append((route[1]-1000+price*8, route[0], ("pickup",pick["id"])))
            for door in obs.get("doors", []):
                if not door["open"] or door.get("type") in (10,13) or door["target"]<0:
                    continue
                route = grid.route(p, door["pos"])
                if route:
                    count = self.visits[(key[0], key[1], door["target"])]
                    point = route[0]
                    if distance(p, door["pos"])<55:
                        # Aim at the doorway center. An outward offset is
                        # outside the grid; route() snaps it back inside and
                        # makes the fly walk away from the exit.
                        point = door["pos"][:]
                    options.append((count*1500+route[1], point, ("door",door["slot"])))
            for idx, ex in enumerate(obs.get("exits", [])):
                route = grid.route(p, ex["pos"])
                if route:
                    options.append((route[1]-200, route[0], ("exit",idx)))
            if options:
                _, goal, chosen = min(options, key=lambda o:o[0])
                if chosen != self.target:
                    self.target, self.target_since = chosen, obs["frame"]
                elif chosen[0]=="pickup" and obs["frame"]-self.target_since>150:
                    self.ignored.add(chosen[1])
                mode = chosen[0]
        # Let the entrance animation settle before selecting another exit.
        if mode in ("door","exit") and obs["frame"]-self.room_entered<15:
            goal, mode = None, "entering"
        if threats and not enemy:
            mode="waiting_vulnerable"
            # Closed Hosts / burrowing enemies: wait at range, never fire into
            # invulnerability. Nearby hazards still drive evasion below.
        wanted = unit([goal[i]-p[i] for i in (0,1)]) if goal and distance(p,goal)>8 else [0.,0.]
        danger = self.danger(p, obs)
        choices = [[0.,0.]]+[unit(v) for v in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]]
        best, best_score = [0.,0.], math.inf
        for move in choices:
            trial = [p[i]+move[i]*24 for i in (0,1)]
            # Exit cells may be outside the ordinary walkable grid.
            leaving = mode=="door" and goal and distance(p,goal)<100 and sum(move[i]*wanted[i] for i in (0,1))>.9
            if not grid.hazard_safe(trial) or (not leaving and not grid.safe(trial)):
                continue
            alignment = sum(move[i]*wanted[i] for i in (0,1))
            score = self.danger(trial, obs)*tactic["risk"]+grid.costs.get(str(grid.index(trial)),0.)*.8-alignment*3+distance(move,self.last_move)*.1
            # Once a valid firing lane is reached ``goal`` is the current
            # position.  In that state movement has no positive objective;
            # the small inertia term above must not make the previous command
            # win forever and carry the player past the target.
            if not any(abs(v) > 1e-6 for v in wanted):
                score += .35 * math.hypot(*move)
            if score < best_score:
                best, best_score = move, score
        self.last_move = best
        hazards = enemies+obs.get("bullets", [])+obs.get("hazards", [])
        threat = min(hazards, key=lambda e:distance(p,e["pos"]), default=None)
        escape = unit([p[i]-threat["pos"][i] for i in (0,1)]) if threat else [0.,0.]
        use_item = False
        use_card = False
        if threats and enemy and obs["frame"] >= self.item_cooldown_until:
            player = obs.get("player", {})
            if player.get("active_item", 0) and player.get("active_charge", 0) >= player.get("active_max_charge", 1):
                use_item = True
            elif player.get("card", 0) and mode in ("combat", "clearing_poop"):
                use_card = True
            if use_item or use_card:
                self.item_cooldown_until = obs["frame"] + 45
                self.use_until = obs["frame"] + 8
                self.use_request = {"use_item":use_item,"use_card":use_card,
                                    "use_id":f'{obs["session"]}:{obs["frame"]}:{"item" if use_item else "card"}'}
        request = self.use_request if obs["frame"] <= self.use_until else {}
        return {"move":best, "shoot":shoot, "use_item":use_item, "use_card":use_card,
                "escape_dir":escape, "danger":danger,
                "hurt":obs["frame"]<self.hurt_until, "mode":mode, "goal":goal,
                "room_count":len(self.visits), "grid":grid, "target":enemy["id"] if enemy else None,
                "repositioning":obs["frame"]<self.reposition_until, **request}

    @staticmethod
    def encode(plan):
        x,y = plan["move"]
        danger = min(1.,plan["danger"]/10)
        return {"light_L":max(0.,-x), "light_R":max(0.,x),
                "odor_a":max(0.,-y), "odor_b":max(0.,y),
                "loom_L":danger, "loom_R":danger, "vibration":min(1.,sum(abs(v) for v in plan["shoot"])),
                "touch":float(plan["hurt"]), "taste":float(plan["mode"]=="pickup")*.5}

    @staticmethod
    def decode(plan, brain, obs):
        r = brain.get("rates", {})
        # Rate thresholds remove spontaneous baseline firing. Zero neural
        # activity means zero movement and shooting (tested by ablation).
        drive = lambda name: max(0.,min(1.,(r.get(name,0.)-2.)/42))
        x = drive("light_R")-drive("light_L")
        y = drive("odor_b")-drive("odor_a")
        movement = [x,y]
        # Input populations can retain/cross-activate activity after a turn.
        # Neural drive may scale or suppress the requested direction; it must
        # not keep walking through the next room on the previous room's signal.
        for i in (0,1):
            wanted = plan["move"][i]
            movement[i] = math.copysign(min(abs(movement[i]),abs(wanted)),wanted) if movement[i]*wanted>0 else 0.
        gf = min(1.,max(0.,(r.get("GF",0.)-5.)/60))*min(1.,plan["danger"]/10)
        if gf>0:
            movement = [(1-.65*gf)*movement[i]+.65*gf*plan["escape_dir"][i] for i in (0,1)]
        magnitude = math.hypot(*movement)
        if magnitude>1:
            movement = [v/magnitude for v in movement]
        # Avoid neural after-discharge walking into a wall after a direction
        # change. This safety projection can veto, never create motor drive.
        p = obs["player"]["pos"]
        destination = [p[i]+movement[i]*20 for i in (0,1)]
        if not plan["grid"].hazard_safe(destination) or (plan["mode"]!="door" and not plan["grid"].safe(destination)):
            alternatives = [[movement[0],0.],[0.,movement[1]],[0.,0.]]
            movement = next((v for v in alternatives if plan["grid"].safe([p[i]+v[i]*20 for i in (0,1)])), [0.,0.])
        # The mechanosensory group has a much lower measured rate than the
        # visual/olfactory groups. Use its calibrated range instead of the
        # generic 42 Hz drive threshold, otherwise valid shots disappear.
        vibration_drive = min(1., max(0., (r.get("vibration",0.)-.1)/2.))
        shooting = plan["shoot"] if vibration_drive>.03 else [0.,0.]
        if plan["mode"] not in ("combat","clearing_fire","clearing_poop"):
            shooting = [0., 0.]
        # In a quiet firing lane, wait for neural permission to fire before
        # advancing. Evasion remains available when danger is immediate.
        if plan["mode"] == "combat" and plan.get("goal") is not None and any(plan["shoot"]) and plan["danger"]<1:
            movement = [0.,0.]
        # Keep a combat tear on its cardinal line.  At high danger, dodge
        # instead of sending a tear that will be displaced by the dodge.
        if plan["mode"] == "combat" and plan.get("goal") is not None and any(plan["shoot"]):
            if plan["danger"] >= 8:
                shooting = [0., 0.]
            else:
                movement = [0., 0.]
        return {"move":[round(v,3) for v in movement], "shoot":shooting,
                "use_item":bool(plan.get("use_item")), "use_card":bool(plan.get("use_card")),
                "use_id":plan.get("use_id",""),
                "mode":plan["mode"], "gf_gain":round(gf,3), "danger":round(plan["danger"],2)}
