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


MOVE_LOOKAHEAD = 20.


def shot_range(obs):
    """Conservative fallback for older bridges without the player's range."""
    return max(40., float(obs['player'].get('tear_range', 260.)) - 15.)


def aimed_shot(obs, target, grid, target_cell=None):
    """Only fire a cardinal tear whose predicted path intersects the target."""
    p = obs['player']['pos']
    speed = max(1., 10 * obs['player'].get('shot_speed', 1.))
    reach = shot_range(obs)
    target_vel = target.get('vel', [0, 0])
    best, miss = [0., 0.], math.inf
    for slot, direction in enumerate([[-1, 0], [0, -1], [1, 0], [0, 1]]):
        inheritance = obs['player'].get('tear_inheritance')
        drift = inheritance[slot] if inheritance else [v*.5 for v in obs['player'].get('vel', [0, 0])]
        velocity = [direction[i]*speed + drift[i] for i in (0, 1)]
        relative = [target['pos'][i]-p[i] for i in (0, 1)]
        # The short prediction horizon avoids projecting erratic enemies far
        # through a wall; long shots use the current target position.
        lead = min(12., distance(p, target['pos']) / speed)
        aim = [target['pos'][i] + target_vel[i]*lead for i in (0, 1)]
        delta = [aim[i]-p[i] for i in (0, 1)]
        length = math.hypot(*velocity)
        along = sum(delta[i]*velocity[i] for i in (0, 1)) / max(1., length)
        cross = abs(delta[0]*velocity[1] - delta[1]*velocity[0]) / max(1., length)
        if (sum(relative[i]*direction[i] for i in (0, 1)) > 0
                and 0 < along < reach and cross <= max(4., target.get('size', 12))
                and grid.ray(p, aim, target_cell) and cross < miss):
            best, miss = direction, cross
    return best


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

    def motion_safe(self, start, end, radius=None, allowed_cells=()):
        """Check the whole move, including reduced neural motor commands.

        Isaac can leave the player inside our conservative wall-clearance
        buffer. Permit movement out of that buffer, but never closer to the
        same obstacle or into a new one. A shorter safe escape must remain
        valid instead of being rejected because it has not cleared it yet.
        """
        radius = self.clearance if radius is None else radius
        offsets = [(0,0),(-radius,0),(radius,0),(0,-radius),(0,radius)]
        def blocked(point):
            indices = {self.index([point[0]+dx,point[1]+dy]) for dx,dy in offsets}
            return {i for i in indices if i not in allowed_cells and not self.walkable(i)}
        initial = blocked(start)
        steps = max(1, math.ceil(distance(start,end)/4))
        for step in range(1,steps+1):
            point = [start[i]+(end[i]-start[i])*step/steps for i in (0,1)]
            touched = blocked(point)
            if touched-initial:
                return False
            if any(distance(point,self.pos(i)) < distance(start,self.pos(i))-1e-6
                   for i in touched):
                return False
            for hazard in self.hazards:
                gap = distance(point,hazard['pos'])
                limit = hazard.get('size',12)+self.player_radius+4
                if gap<=limit and gap<distance(start,hazard['pos'])-1e-6:
                    return False
        return True

    def ray(self, a, b, target_cell=None):
        steps = max(1, int(distance(a, b)/12))
        source_cell = self.index(a)
        for i in range(1, steps):
            idx = self.index([a[0]+(b[0]-a[0])*i/steps, a[1]+(b[1]-a[1])*i/steps])
            if idx == target_cell or idx == source_cell:
                continue
            collision, typ = self.cells.get(idx, (4, 0))
            # Floor spikes do not block tears; rocks and walls do.
            if collision not in (0, 1) or typ == 25:
                return False
            if typ == 14 and collision != 0:
                return False
        return True

    def route(self, start, goal, approach=False):
        # Dijkstra over traversable cells; unreachable objects are not targets.
        source, target = self.index(start), self.index(goal)
        candidates = [i for i in self.cells if self.walkable(i)]
        if not candidates:
            return None
        if not self.walkable(target):
            if not approach:
                return None
            target = min(candidates, key=lambda i: distance(self.pos(i), goal))
            goal = self.pos(target)
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


def firing_routes(grid, start, target, preferred, maximum, minimum=55., target_cell=None):
    """Reachable positions with an unobstructed cardinal firing lane."""
    options = []
    radii = {max(minimum,preferred-50), max(minimum,preferred), min(maximum,preferred+40)}
    for lane,direction in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):
        for radius in sorted(radii):
            if not minimum<=radius<=maximum:
                continue
            point = [target[i]+direction[i]*radius for i in (0,1)]
            if grid.safe(point) and grid.ray(point,target,target_cell):
                route = grid.route(start,point)
                if route:
                    options.append((lane,point,route[0],route[1]))
    return options


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
        self.shot_watch = None
        self.release_until = -1
        self.failed_firing_pos = None
        self.pressed_plates = set()

    @staticmethod
    def danger(pos, obs, velocity=None):
        """Risk along the next 12 game frames, relative to each moving threat.

        Evaluating a whole segment prevents retreat through an enemy to a
        deceptively safe endpoint. Terminal separation breaks ties when every
        candidate starts close to a threat.
        """
        velocity = velocity or [0., 0.]
        radius = obs['player'].get('size', 10)
        risk = 0.
        groups = [(obs.get('enemies', []), 14., 90.),
                  (obs.get('hazards', []), 14., 90.),
                  (obs.get('bullets', []), 30., 32.)]
        for entities, weight, margin in groups:
            for e in entities:
                if e.get('kind') == 'fire' and e.get('destructible', False):
                    continue
                # Non-destructible fire still hurts on contact. Destructible
                # fire is handled as an objective and filtered above.
                reach = 8. if e.get('kind') == 'fire' else margin
                rel = [e['pos'][i]-pos[i] for i in (0, 1)]
                vel = [e.get('vel', [0, 0])[i]-velocity[i] for i in (0, 1)]
                vv = sum(v*v for v in vel)
                t = max(0., min(12., -sum(rel[i]*vel[i] for i in (0, 1))/max(vv, .001)))
                gaps = [math.hypot(*(rel[i]+vel[i]*s for i in (0, 1)))
                        -e.get('size', 12)-radius for s in (t, 12.)]
                close, terminal = [max(0., 1-gap/reach)**2 for gap in gaps]
                risk += weight*(.7*close + .3*terminal)
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
            self.shot_watch = None
            self.release_until = -1
            self.failed_firing_pos = None
            self.pressed_plates = set()
        hp = obs["player"].get("hearts", 0)+obs["player"].get("soul", 0)
        if self.last_hp is not None and hp < self.last_hp:
            self.hurt_until = obs["frame"]+20
        self.last_hp = hp
        enemies = obs.get("enemies", [])
        threats = [e for e in enemies if targetable(e)]
        targets = [e for e in threats if e.get("vulnerable", True)]
        # Prefer an enemy we can hit now over a nearer diagonal target. Risk
        # still considers every enemy, independently of this firing choice.
        enemy = min(targets, key=lambda e: (not any(aimed_shot(obs, e, grid)),
                                           distance(p, e["pos"])), default=None)
        clearing_fire = False
        clearing_poop = False
        clearing_tnt = False
        fire_routes = None
        skipped_targets = []
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
            preferred = min(150., tactic['distance'], max(60.,shot_range(obs)-35.))
            for fire in sorted(fires,key=lambda h:distance(p,h['pos'])):
                routes = firing_routes(grid,p,fire['pos'],preferred,min(175.,shot_range(obs)-5.))
                if routes or any(aimed_shot(obs,fire,grid)):
                    enemy, fire_routes = fire, routes
                    break
                skipped_targets.append(('fire',fire['id']))
            clearing_fire = enemy is not None
        if enemy is None and not enemies and not obs.get("clear") and obs.get("room_type",1)==1:
            plates = [i for i, (_, typ) in grid.cells.items() if typ==20]
            # Trap rooms can enclose their button with bomb rocks. An exposed
            # TNT barrel opens the route; walking at an unreachable plate does
            # nothing. Only choose this objective when all plates are blocked.
            if plates and not any(grid.route(p, grid.pos(i)) for i in plates):
                barrels = [i for i, (collision, typ) in grid.cells.items()
                           if typ==12 and collision!=0]
                if barrels:
                    idx = min(barrels, key=lambda i:distance(p,grid.pos(i)))
                    enemy = {'id':-200000-idx,'pos':grid.pos(idx),'vel':[0,0],
                             'size':14,'hp':1,'kind':'tnt','vulnerable':True}
                    clearing_tnt = True
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
                self.failed_firing_pos = p[:]
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
            target_cell = grid.index(ep) if clearing_poop or clearing_tnt else None
            shoot = aimed_shot(obs, enemy, grid, target_cell)
            # Find an accessible firing lane at a useful distance from target.
            options = []
            preferred_distance = min(tactic['distance'], max(60., shot_range(obs)-35.))
            if clearing_fire:
                preferred_distance = min(150., preferred_distance)
            routes = fire_routes if clearing_fire else firing_routes(
                grid,p,ep,preferred_distance,shot_range(obs)-5.,
                minimum=150. if clearing_tnt else 55.,target_cell=target_cell)
            for lane,point,waypoint,cost in routes:
                penalty = 0.
                if obs['frame']<self.reposition_until:
                    penalty += 350 if lane!=self.firing_lane else 0
                    if self.failed_firing_pos is not None and distance(point,self.failed_firing_pos)<50:
                        penalty += 350
                options.append((cost+distance(p,point)*.2+self.danger(point,obs)*tactic['risk']*9+penalty,waypoint))
            # Keep an already good firing lane. The old code kept moving even
            # when aligned; its fallback even walked towards unreachable foes.
            # aimed_shot already checks the actual tear trajectory, including
            # inherited motion. A separate geometric axis test can veto a
            # valid moving shot (and deadlock just short of a firing lane).
            aligned = any(shoot)
            reach = min(180., shot_range(obs)) if clearing_fire else shot_range(obs)
            safe_distance = preferred_distance-60 < distance(p,ep) < reach
            fire_shot_from_here = (clearing_fire and aligned and 55 < distance(p,ep) < reach
                                   and grid.ray(p,ep,target_cell) and any(shoot))
            alignment_goal = False
            if clearing_fire and not aligned:
                # First move onto the fire's horizontal or vertical line. A
                # diagonal approach can orbit around rocks forever and never
                # produce a valid tear direction.
                align_options = []
                for point in ([p[0], ep[1]], [ep[0], p[1]]):
                    # A missing shot can mean insufficient range or a rock
                    # in the way, even when already on the correct axis.
                    # Never select the current position as an alignment
                    # waypoint: it would override the reachable firing lanes
                    # below and leave both movement and shooting at zero.
                    if (distance(point, p) > 2 and distance(point, ep) > 55
                            and grid.safe(point) and grid.ray(point, ep, target_cell)):
                        route = grid.route(p, point)
                        if route:
                            align_options.append((route[1], route[0]))
                if align_options:
                    goal = min(align_options, key=lambda item:item[0])[1]
                    alignment_goal = True
            # Only HOLD position when safe; firing is independent of moving.
            if aligned and (safe_distance or fire_shot_from_here) and self.danger(p,obs)<.4 and obs["frame"]>=self.reposition_until:
                goal = p
            elif not alignment_goal:
                goal = min(options, key=lambda pair: pair[0])[1] if options else None
            # Keep the predicted shot during retreat/repositioning. Range,
            # obstacles and inherited velocity are checked in aimed_shot and
            # rechecked against the latest observation by the neural decoder.
            if clearing_tnt and distance(p,ep)<150:
                shoot = [0.,0.]
            mode = "clearing_tnt" if clearing_tnt else ("clearing_poop" if clearing_poop else ("clearing_fire" if clearing_fire else "combat"))
        elif not obs.get("clear") and not enemies and obs.get("room_type", 1)==1:
            # Some normal rooms lock every door until a pressure plate is
            # stepped on. They contain no enemy or pickup to lead us there.
            plates = []
            for idx, (_, typ) in grid.cells.items():
                if typ != 20 or idx in self.pressed_plates:
                    continue
                point = grid.pos(idx)
                if distance(p, point) < 10:
                    self.pressed_plates.add(idx)
                    continue
                route = grid.route(p, point) if grid.safe(point) else None
                if route:
                    plates.append((route[1], route[0]))
            if plates:
                goal = min(plates, key=lambda item:item[0])[1]
                mode = "switch"
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
                else:
                    skipped_targets.append(('pickup',pick['id']))
            for door in obs.get("doors", []):
                if not door["open"] or door.get("type") in (10,13) or door["target"]<0:
                    continue
                route = grid.route(p, door["pos"], approach=True)
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
        arrival_radius = 2. if enemy else 8.
        wanted = unit([goal[i]-p[i] for i in (0,1)]) if goal and distance(p,goal)>arrival_radius else [0.,0.]
        danger = self.danger(p, obs)
        gain = min(1.,max(.25,distance(p,goal)/40.)) if goal and danger<.4 and any(wanted) else 1.
        choices = [[0.,0.]]+[unit(v) for v in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]]
        if not any(wanted) and danger < .4:
            choices = [[0., 0.]]
        best, best_score = [0.,0.], math.inf
        for direction in choices:
            move = [v*gain for v in direction]
            trial = [p[i]+move[i]*MOVE_LOOKAHEAD for i in (0,1)]
            # Exit cells may be outside the ordinary walkable grid.
            leaving = mode=="door" and goal and distance(p,goal)<100 and sum(direction[i]*wanted[i] for i in (0,1))>.9
            door_cells = {grid.index(goal)} if leaving else ()
            traversable = grid.motion_safe(p,trial,allowed_cells=door_cells)
            # In a contact emergency the normal clearance buffer can reject
            # every escape vector in a narrow room. Allow the player's centre
            # cell, still respecting walls and hazards, before accepting death
            # by standing still.
            if not traversable and danger >= 6 and not leaving:
                traversable = grid.motion_safe(p,trial,radius=0)
            if not traversable:
                continue
            alignment = sum(direction[i]*wanted[i] for i in (0,1))
            speed = 4. * obs['player'].get('speed', 1.)
            future = [v*speed for v in move]
            # Compare moving trajectories, not only the endpoint. Persistent
            # room costs may break ties but cannot outweigh forward progress.
            risk = .7*self.danger(p, obs, future)+.3*self.danger(trial, obs)
            remembered = min(.6, grid.costs.get(str(grid.index(trial)), 0.)*.1)
            score = risk*tactic['risk']+remembered-alignment*3+distance(direction,self.last_move)*.1
            # Once a valid firing lane is reached ``goal`` is the current
            # position.  In that state movement has no positive objective;
            # the small inertia term above must not make the previous command
            # win forever and carry the player past the target.
            if not any(abs(v) > 1e-6 for v in wanted):
                score += .35 * math.hypot(*move)
            if score < best_score:
                best, best_score = move, score
        self.last_move = unit(best)
        # GF escape activity reinforces the evaluated route, including every
        # projectile/enemy, rather than pushing away from one nearby flame.
        escape = best[:]
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
        # A held input is not evidence that a weapon actually fired. Release
        # it for six game frames if the tear counter stalls, then re-aim.
        # This also lets charge-and-release weapons discharge without changing
        # game stats or spawning projectiles directly.
        if (obs.get('armed') and not obs.get('paused') and obs.get('controls', True)
                and not obs.get('dead') and 'tears' in obs and any(obs.get('applied_shoot', [0,0]))):
            watch_key = (enemy['id'] if enemy else None, tuple(obs['applied_shoot']), obs['tears'])
            if self.shot_watch is None or self.shot_watch[0] != watch_key:
                self.shot_watch = (watch_key, obs['frame'])
            timeout = max(60., 3*(obs['player'].get('max_fire_delay', 10)+1))
            if obs['frame']-self.shot_watch[1] >= timeout:
                self.release_until = obs['frame']+6
                self.shot_watch = None
        else:
            self.shot_watch = None
        releasing = obs['frame'] < self.release_until
        if releasing:
            shoot = [0., 0.]
        return {"move":best, "shoot":shoot, "use_item":use_item, "use_card":use_card,
                "escape_dir":escape, "danger":danger,
                "hurt":obs["frame"]<self.hurt_until, "mode":mode, "goal":goal,
                "room_count":len(self.visits), "grid":grid, "target":enemy["id"] if enemy else None,
                "target_kind": "tnt" if clearing_tnt else ("poop" if clearing_poop else ("fire" if clearing_fire else "enemy")),
                "target_entity":enemy,
                "objective":self.target if mode in ('pickup','door','exit') else None,
                "skipped_targets":skipped_targets,
                "releasing_attack":releasing,
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
        movement = [0., 0.]
        # Input populations can retain/cross-activate activity after a turn.
        # Neural drive may scale or suppress the requested direction; it must
        # not keep walking through the next room on the previous room's signal.
        for i, (negative, positive) in enumerate([('light_L','light_R'), ('odor_a','odor_b')]):
            wanted = plan["move"][i]
            if wanted:
                requested, opposite = (positive,negative) if wanted>0 else (negative,positive)
                # Require the requested population to respond. Antagonist
                # after-discharge reduces gain but cannot cancel a retreat.
                gain = drive(requested)*(1-.25*drive(opposite))
                movement[i] = math.copysign(min(gain,abs(wanted)), wanted)
        gf = min(1.,max(0.,(r.get("GF",0.)-5.)/60))*min(1.,plan["danger"]/10)
        if gf>0:
            movement = [(1-.65*gf)*movement[i]+.65*gf*plan["escape_dir"][i] for i in (0,1)]
        magnitude = math.hypot(*movement)
        if magnitude>1:
            movement = [v/magnitude for v in movement]
        # Avoid neural after-discharge walking into a wall after a direction
        # change. This safety projection can veto, never create motor drive.
        p = obs["player"]["pos"]
        destination = [p[i]+movement[i]*MOVE_LOOKAHEAD for i in (0,1)]
        clearance = 0 if plan['danger'] >= 6 else None
        door_cells = {plan['grid'].index(plan['goal'])} if plan['mode']=='door' and plan.get('goal') and distance(p,plan['goal'])<100 else ()
        requested_motion = any(movement)
        if not plan['grid'].motion_safe(p,destination,clearance,door_cells):
            alternatives = [[movement[0],0.],[0.,movement[1]],[0.,0.]]
            movement = next((v for v in alternatives if plan['grid'].motion_safe(
                p,[p[i]+v[i]*MOVE_LOOKAHEAD for i in (0,1)],clearance,door_cells)),[0.,0.])
        shooting = plan["shoot"] if drive('vibration')>.15 else [0.,0.]
        if plan["mode"] not in ("combat","clearing_fire","clearing_poop","clearing_tnt"):
            shooting = [0., 0.]
        if any(shooting) and 'target_entity' in plan:
            kind = plan['target_kind']
            if kind=='tnt':
                pool = [{'id':i,'pos':plan['grid'].pos(i),'size':14}
                        for i,c,t in obs.get('grid',[]) if t==12 and c!=0]
                target_id = -200000-plan['target']
            else:
                pool = obs.get({'enemy':'enemies','fire':'hazards','poop':'poops'}[kind], [])
                target_id = -100000-plan['target'] if kind=='poop' else plan['target']
            target = next((e for e in pool if e['id']==target_id), None)
            if target is None or not target.get('vulnerable',True) or target.get('hp',1)<=0:
                shooting = [0., 0.]
            else:
                shooting = aimed_shot(obs, target, plan['grid'], int(target_id) if kind in ('poop','tnt') else None)
                if kind=='tnt' and distance(p,target['pos'])<150:
                    shooting = [0.,0.]
        # Movement has already been chosen for safety. Never zero a retreat
        # simply because a target happens to be on the firing line.
        return {"move":[round(v,3) for v in movement], "shoot":shooting,
                "movement_blocked":bool(requested_motion and not any(movement)),
                "use_item":bool(plan.get("use_item")), "use_card":bool(plan.get("use_card")),
                "use_id":plan.get("use_id",""),
                "mode":plan["mode"], "gf_gain":round(gf,3), "danger":round(plan["danger"],2)}
