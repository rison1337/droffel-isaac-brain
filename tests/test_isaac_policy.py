import json
import pytest
from isaac_policy import IsaacPolicy, RoomGrid
from isaac_service import playable


def observation():
    return {"session":"test", "room":1, "frame":100, "stage":1, "stage_type":0,
        "epoch":1, "armed":True,"paused":False,"dead":False,"controls":True,"players":1,
        "player":{"pos":[160,160],"hearts":6,"max_hearts":6,"soul":0,"flying":False},
        "grid_width":11,"grid_origin":[0,0],
        "grid":[[i,4 if i%11 in (0,10) or i//11 in (0,10) else 0,0] for i in range(121)],
        "clear":True,"enemies":[],"bullets":[],"doors":[],"pickups":[],"exits":[]}


def test_brain_activity_is_required_for_actions():
    obs = observation()
    plan = {"move":[1,0],"shoot":[1,0],"danger":0,"hurt":False,
            "mode":"combat","escape_dir":[-1,0],"grid":RoomGrid(obs)}
    assert IsaacPolicy.encode(plan)["light_R"]==1
    silent = IsaacPolicy.decode(plan,{"rates":{}},obs)
    responding = IsaacPolicy.decode(plan,{"rates":{"light_R":60,"vibration":60}},obs)
    assert silent["move"]==[0,0] and silent["shoot"]==[0,0]
    assert responding["move"]==[1,0] and responding["shoot"]==[1,0]


def test_recurrent_gf_response_changes_evasion_direction():
    obs = observation()
    plan = {"move":[1,0],"shoot":[1,0],"danger":10,"hurt":False,
            "mode":"combat","escape_dir":[-1,0],"grid":RoomGrid(obs)}
    quiet = IsaacPolicy.decode(plan,{"rates":{"light_R":60,"vibration":60}},obs)
    escape = IsaacPolicy.decode(plan,{"rates":{"light_R":60,"vibration":60,"GF":100}},obs)
    assert quiet["move"][0]>0 and escape["move"][0]<0


def test_stale_paused_dead_disconnected_and_multiplayer_cannot_play():
    obs = observation()
    assert playable(obs,.1,True)
    assert not playable(obs,.5,True)
    assert not playable(obs,.1,False)
    for key,value in [("paused",True),("dead",True),("players",2),("armed",False),("controls",False)]:
        assert not playable(dict(obs,**{key:value}),.1,True)


def test_route_around_rocks_and_avoid_pits_and_spikes():
    obs = observation()
    for idx in (37,48,59,70): obs["grid"][idx][1]=2
    grid = RoomGrid(obs)
    route = grid.route([80,160],[240,160])
    assert route and route[1]>160
    assert not grid.walkable(48)
    obs["grid"][24] = [24,1,0]
    obs["grid"][25] = [25,0,8]
    grid = RoomGrid(obs)
    assert not grid.walkable(24) and not grid.walkable(25)


def test_bullet_prediction_penalizes_future_collision():
    obs = observation()
    obs["bullets"] = [{"pos":[240,160],"vel":[-8,0],"size":5}]
    assert IsaacPolicy.danger([160,160],obs)>IsaacPolicy.danger([160,100],obs)+1


def test_projectiles_are_more_urgent_than_a_stationary_fire():
    obs = observation()
    obs['hazards'] = [{'kind':'fire','destructible':True,'pos':[200,160],'size':13}]
    calm = IsaacPolicy.danger([200,160], obs)
    obs['bullets'] = [{'pos':[240,160],'vel':[-8,0],'size':5}]
    assert IsaacPolicy.danger([200,160], obs) > calm + 10


def test_fire_aim_accounts_for_player_drift_before_holding_position():
    obs = observation()
    obs['player']['pos'] = [128, 280]
    obs['player']['vel'] = [3, 0]
    obs['player']['tear_inheritance'] = [[0,0],[3,0],[3,0],[3,0]]
    obs['hazards'] = [{'id':55,'kind':'fire','destructible':True,
                      'pos':[120,360],'size':13,'hp':5}]
    plan = IsaacPolicy().plan(obs)
    assert plan['mode'] == 'clearing_fire'
    assert plan['shoot'] == [0., 0.]
    assert plan['goal'] != obs['player']['pos']


def test_stalled_held_attack_is_released_for_charge_weapons():
    obs = observation()
    obs['enemies'] = [{'id':1,'type':10,'hp':10,'size':10,
                      'pos':[320,160],'vel':[0,0],'vulnerable':True}]
    obs['tears'] = 0
    obs['applied_shoot'] = [1,0]
    policy = IsaacPolicy()
    policy.plan(obs)
    released = False
    for frame in range(101, 170):
        obs['frame'] = frame
        plan = policy.plan(obs)
        released = released or plan['releasing_attack']
    assert released


def test_contact_crowd_gets_an_emergency_escape_vector():
    obs = observation()
    obs['player']['pos'] = [320,280]
    obs['clear'] = False
    obs['enemies'] = [{'id':i,'type':10,'hp':10,'size':13,
                       'pos':[320+i*20,280],'vel':[0,0],
                       'vulnerable':True} for i in (-2,-1,1,2)]
    plan = IsaacPolicy().plan(obs)
    assert plan['danger'] > 6 and plan['move'] != [0,0]
    assert any(plan['shoot'])


def test_retreating_from_chaser_keeps_a_valid_defensive_shot():
    obs = observation()
    obs['clear'] = False
    obs['player']['vel'] = [-2, 0]
    obs['enemies'] = [{'id':1,'type':10,'hp':10,'size':13,
                       'pos':[210,160],'vel':[-2,0],'vulnerable':True}]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['danger'] > .4
    assert plan['move'][0] < 0
    assert plan['shoot'] == [1,0]
    action = IsaacPolicy.decode(plan, {'rates':{'light_L':60,'vibration':60}}, obs)
    assert action['move'][0] < 0 and action['shoot'] == [1,0]


def test_inherited_tear_motion_can_hit_an_off_axis_target():
    obs = observation()
    obs['clear'] = False
    obs['player']['vel'] = [0,4]
    obs['player']['tear_inheritance'] = [[0,2],[0,0],[0,2],[0,0]]
    obs['enemies'] = [{'id':1,'type':10,'hp':10,'size':13,
                       'pos':[320,192],'vel':[0,0],'vulnerable':True}]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['shoot'] == [1,0]


def test_fireable_enemy_is_preferred_to_a_nearer_diagonal_enemy():
    obs = observation()
    obs['clear'] = False
    obs['enemies'] = [{'id':i,'type':10,'hp':10,'size':13,'pos':p,
                      'vel':[0,0],'vulnerable':True}
                     for i,p in ((1,[220,210]),(2,[330,160]))]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['target'] == 2
    assert plan['shoot'] == [1,0]
    # The latest observation must still veto a dead/missing target.
    obs['enemies'].pop()
    assert IsaacPolicy.decode(plan, {'rates':{'vibration':60}}, obs)['shoot'] == [0,0]


def test_small_target_alignment_has_no_idle_gap():
    obs = observation()
    obs['player']['pos'] = [160,167]
    obs['hazards'] = [{'id':55,'kind':'fire','destructible':True,
                      'pos':[320,160],'size':4,'hp':5}]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert any(plan['move']) or any(plan['shoot'])


def test_clear_room_seeks_pickups_then_unvisited_exit():
    obs = observation()
    obs["pickups"]=[{"id":1,"pos":[240,160],"variant":20,"subtype":1,"price":0}]
    obs["doors"]=[{"slot":0,"pos":[40,160],"open":True,"target":2,"type":1},
                  {"slot":2,"pos":[360,160],"open":True,"target":3,"type":1}]
    policy = IsaacPolicy()
    assert policy.plan(obs)["mode"]=="pickup"
    obs["pickups"]=[]
    obs["frame"]+=16
    policy.visits[(1,0,2)]=2
    plan = policy.plan(obs)
    assert plan["mode"]=="door" and plan["move"][0]>0


def test_locked_empty_room_seeks_pressure_plate_once_and_resets_on_new_room():
    obs = observation()
    obs['clear'] = False
    obs['grid'][50] = [50,0,20]  # [240,160]
    policy = IsaacPolicy(learning=False)
    plan = policy.plan(obs)
    assert plan['mode'] == 'switch' and plan['move'][0] > 0
    obs['player']['pos'] = [240,160]
    policy.plan(obs)
    obs['player']['pos'] = [160,160]
    assert policy.plan(obs)['mode'] == 'waiting'
    obs['room'] = 2
    assert policy.plan(obs)['mode'] == 'switch'
    # Combat always takes precedence over pressing room buttons.
    obs['enemies'] = [{'id':1,'pos':[320,160],'size':13,'hp':10}]
    assert policy.plan(obs)['mode'] == 'combat'


def test_unreachable_plate_can_be_opened_by_shooting_tnt_at_range():
    obs = observation()
    obs['clear'] = False
    obs['player']['pos'] = [320,320]
    obs['grid'][48] = [48,0,20]  # enclosed plate at [160,160]
    for idx in (37,47,49,59):
        obs['grid'][idx] = [idx,3,2]
    obs['grid'][90] = [90,2,12]  # exposed TNT at [80,320]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['mode'] == 'clearing_tnt' and plan['shoot'] == [-1,0]
    assert IsaacPolicy.decode(plan,{'rates':{'vibration':60}},obs)['shoot'] == [-1,0]
    # Never keep firing at the barrel after the latest snapshot says it broke.
    obs['grid'][90] = [90,0,12]
    assert IsaacPolicy.decode(plan,{'rates':{'vibration':60}},obs)['shoot'] == [0,0]
    obs['grid'][90] = [90,2,12]
    obs['player']['pos'] = [160,320]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['shoot'] == [0,0]  # first retreat outside the blast buffer


def test_near_bottom_door_moves_into_doorway_not_outside_grid():
    obs = observation()
    obs["player"]["pos"] = [320,410]
    obs["doors"] = [{"slot":3,"pos":[320,440],"open":True,"target":2,"type":1}]
    policy = IsaacPolicy()
    policy.plan(dict(obs, doors=[]))
    obs["frame"] += 16
    plan = policy.plan(obs)
    assert plan["mode"] == "door"
    assert plan["goal"] == [320,440]
    assert plan["move"][1] > 0


def test_combat_takes_priority_and_aligned_player_holds_firing_lane():
    obs=observation()
    obs['player']['pos']=[100,160]
    obs['enemies']=[{'id':1,'pos':[320,160],'vel':[0,0],'size':10,'vulnerable':True}]
    obs['doors']=[{'slot':0,'pos':[40,160],'open':True,'target':2,'type':1}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='combat' and plan['shoot']==[1,0]
    assert plan['move']==[0,0]


def test_aligned_firing_lane_stops_after_approach():
    """A prior approach direction must not keep walking while shooting."""
    obs = observation()
    obs['player']['pos'] = [120, 220]
    obs['enemies'] = [{'id': 1, 'pos': [320, 160], 'vel': [0, 0],
                      'size': 10, 'vulnerable': True}]
    policy = IsaacPolicy()
    approach = policy.plan(obs)
    assert approach['move'][1] < 0
    obs['frame'] += 16
    obs['player']['pos'] = [120, 160]
    hold = policy.plan(obs)
    assert hold['shoot'] == [1, 0]
    assert hold['move'] == [0, 0]


def test_previous_direction_cannot_leak_into_room_entry_or_idle():
    obs=observation()
    obs['doors']=[{'slot':0,'pos':[40,160],'open':True,'target':2,'type':1}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='entering'
    assert IsaacPolicy.decode(plan,{'rates':{'light_R':100,'odor_b':100}},obs)['move']==[0,0]


def test_direction_reversal_waits_for_matching_neural_response():
    obs=observation()
    plan={'move':[-1,0],'shoot':[0,0],'danger':0,'mode':'door','grid':RoomGrid(obs),'escape_dir':[0,0]}
    assert IsaacPolicy.decode(plan,{'rates':{'light_R':100}},obs)['move']==[0,0]
    assert IsaacPolicy.decode(plan,{'rates':{'light_L':100}},obs)['move']==[-1,0]


def test_floor_spikes_allow_tears_but_trap_entities_are_not_shooting_targets():
    obs=observation()
    # Spikes are floor cells; they must not become a fake enemy target.
    obs['grid'][49]=[49,0,8]
    grid=RoomGrid(obs)
    assert grid.ray([120,160],[280,160])
    assert not grid.walkable(49)
    obs['enemies']=[{'id':1,'type':202,'pos':[280,160],'vel':[0,0],'size':10,'vulnerable':True}]
    plan=IsaacPolicy().plan(obs)
    assert plan['shoot']==[0.,0.]


def test_learning_memory_increases_caution_after_damage(tmp_path):
    obs=observation()
    path=tmp_path/'learning.json'
    policy=IsaacPolicy(path)
    policy.plan(obs)
    hurt=dict(obs,frame=101,player=dict(obs['player'],hearts=4))
    policy.plan(hurt)
    resumed=IsaacPolicy(path)
    assert resumed.experience.data['damage']==2
    assert resumed.experience.data['caution']>0
    assert resumed.experience.spatial_costs(hurt)
    assert json.loads(path.read_text())['runs']==1


def test_route_avoids_fire_in_a_clear_room_even_with_flight():
    obs=observation()
    obs['hazards']=[{'type':33,'kind':'fire','pos':[200,160],'size':12}]
    for flying in (False,True):
        obs['player']['flying']=flying
        grid=RoomGrid(obs)
        assert not grid.safe([200,160])
        current=[120,160]
        for _ in range(15):
            current,_=grid.route(current,[280,160])
            assert grid.hazard_safe(current)
            if current==[280,160]: break
        assert current==[280,160]
    assert IsaacPolicy.danger([180,160],obs)>IsaacPolicy.danger([80,160],obs)


def test_item_in_a_fire_does_not_lure_player_into_it():
    obs=observation()
    obs['hazards']=[{'type':33,'kind':'fire','pos':[240,160],'size':12}]
    obs['pickups']=[{'id':1,'variant':100,'subtype':1,'pos':[240,160],'price':0}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']!='pickup'
    assert plan['shoot']==[0,0]


def test_door_movement_exception_does_not_override_fire_avoidance():
    obs=observation()
    obs['hazards']=[{'type':33,'kind':'fire','pos':[200,160],'size':12}]
    plan={'move':[1,0],'shoot':[0,0],'danger':0,'mode':'door','grid':RoomGrid(obs),'escape_dir':[0,0]}
    action=IsaacPolicy.decode(plan,{'rates':{'light_R':60}},obs)
    assert action['move']==[0,0]


def test_destructible_fire_is_extinguished_but_not_counted_as_an_enemy():
    obs=observation()
    obs['hazards']=[{'id':55,'kind':'fire','type':33,'pos':[320,160],'size':12,'hp':5,'destructible':True}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='clearing_fire'
    assert plan['shoot']==[1,0]
    assert IsaacPolicy.decode(plan,{'rates':{'vibration':60}},obs)['shoot']==[1,0]
    # A flame that cannot be extinguished by tears remains a navigation hazard.
    obs['hazards'][0]['destructible']=False
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']!='clearing_fire'
    assert plan['shoot']==[0,0]


def test_fire_target_gets_a_cardinal_alignment_goal_before_shooting():
    obs=observation()
    obs['player']['pos']=[280,240]
    obs['hazards']=[{'id':55,'kind':'fire','type':33,'pos':[320,320],'size':12,'hp':5,'destructible':True}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='clearing_fire' and plan['shoot']==[0,0]
    assert plan['goal'] in ([280,320],[320,240])


def test_fire_lane_tolerance_allows_shot_when_player_is_a_few_pixels_off_axis():
    obs=observation()
    obs['player']['pos']=[206.59,285.39]
    obs['hazards']=[{'id':55,'kind':'fire','type':33,'pos':[360,280],'size':13,'hp':4,'destructible':True}]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='clearing_fire'
    assert plan['shoot']==[1,0]


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('gap', [245.9122, 260., 279.])
def test_aligned_fire_beyond_tear_range_is_approached_instead_of_waiting(axis, gap):
    obs = observation()
    obs['grid_width'] = 17
    obs['grid'] = [[i,4 if i%17 in (0,16) or i//17 in (0,16) else 0,0]
                   for i in range(17*17)]
    obs['player']['pos'] = [240.,240.]
    obs['player']['pos'][axis] += gap
    obs['hazards'] = [{'id':55,'kind':'fire','destructible':True,
                      'pos':[240.,240.],'size':13,'hp':5}]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['mode'] == 'clearing_fire'
    assert plan['shoot'] == [0,0]
    assert plan['move'][axis] < 0
    assert plan['goal'] != obs['player']['pos']


def test_aligned_fire_behind_a_rock_routes_around_the_blocker():
    obs = observation()
    obs['player']['pos'] = [360,240]
    obs['hazards'] = [{'id':55,'kind':'fire','destructible':True,
                      'pos':[200,240],'size':13,'hp':5}]
    obs['grid'][73] = [73,3,2]  # rock at [280,240]
    plan = IsaacPolicy(learning=False).plan(obs)
    assert plan['mode'] == 'clearing_fire'
    assert plan['shoot'] == [0,0] and any(plan['move'])
    assert plan['goal'] != obs['player']['pos']


def test_destroyed_fire_is_replaced_by_pickup_goal_without_old_shot():
    obs = observation()
    obs['hazards'] = [{'id':55,'kind':'fire','destructible':True,
                      'pos':[320,160],'size':13,'hp':5}]
    obs['pickups'] = [{'id':99,'pos':[240,160],'variant':100,'subtype':395,'price':0}]
    policy = IsaacPolicy(learning=False)
    old = policy.plan(obs)
    assert old['shoot'] == [1,0]
    obs['hazards'] = []
    obs['frame'] += 2
    assert IsaacPolicy.decode(old,{'rates':{'vibration':60}},obs)['shoot'] == [0,0]
    new = policy.plan(obs)
    assert new['mode'] == 'pickup' and new['move'][0] > 0
    assert new['shoot'] == [0,0]


def test_live_enemies_take_priority_over_fire():
    obs=observation()
    obs['hazards']=[{'id':55,'kind':'fire','type':33,'pos':[320,160],'size':12,'hp':5,'destructible':True}]
    obs['enemies']=[{'id':1,'type':10,'hp':10,'pos':[160,320],'size':12,'vulnerable':True}]
    obs['clear']=False
    plan=IsaacPolicy().plan(obs)
    assert plan['target']==1 and plan['mode']=='combat'


def test_poopy_wall_is_shot_before_enemy_behind_it():
    obs=observation()
    obs['enemies']=[{'id':1,'type':10,'hp':10,'pos':[320,160],'size':12,'vulnerable':True}]
    obs['poops']=[{'id':50,'pos':[240,160],'type':14}]
    obs['grid'][50]=[50,1,14]
    plan=IsaacPolicy().plan(obs)
    assert plan['mode']=='clearing_poop'
    assert plan['shoot']==[1,0]


def test_affordable_shop_item_is_selected_but_unaffordable_one_is_skipped():
    obs=observation()
    obs['player']['coins']=10
    obs['pickups']=[{'id':1,'pos':[240,160],'variant':100,'subtype':1,'price':7}]
    assert IsaacPolicy().plan(obs)['mode']=='pickup'
    obs['player']['coins']=5
    assert IsaacPolicy().plan(obs)['mode']!='pickup'


def test_grab_bag_is_selected_in_a_clear_room():
    obs = observation()
    obs['pickups'] = [{'id': 9, 'pos': [240,160], 'variant': 69,
                      'subtype': 0, 'price': 0}]
    plan = IsaacPolicy().plan(obs)
    assert plan['mode'] == 'pickup'
    assert plan['target'] is None
    obs['pickups'][0]['price']=-1
    assert IsaacPolicy().plan(obs)['mode']!='pickup'


def test_active_item_and_card_are_used_during_combat():
    obs=observation()
    obs['enemies']=[{'id':1,'type':10,'hp':10,'pos':[320,160],'size':10,'vulnerable':True}]
    obs['player'].update(active_item=123,active_charge=6,card=0)
    plan=IsaacPolicy().plan(obs)
    assert plan['use_item'] and not plan['use_card']
    obs['frame'] += 50
    obs['player'].update(active_item=0,active_charge=0,card=42)
    plan=IsaacPolicy().plan(obs)
    assert plan['use_card']


def test_partial_charge_is_not_used_and_request_survives_report_interval():
    obs=observation()
    obs['enemies']=[{'id':1,'type':10,'hp':10,'pos':[320,160],'size':10,'vulnerable':True}]
    obs['player'].update(active_item=123,active_charge=2,active_max_charge=6)
    policy=IsaacPolicy()
    assert not policy.plan(obs)['use_item']
    obs['player']['active_charge']=6
    first=policy.plan(obs)
    obs['frame']+=3
    held=policy.plan(obs)
    assert held['use_item'] and held['use_id']==first['use_id']
    obs['frame']+=10
    assert not policy.plan(obs)['use_item']
