from copy import deepcopy

from isaac_learning import Experience
from isaac_policy import RoomGrid


def observation(frame=0):
    return {"session":"learning-test","frame":frame,"stage":1,"stage_type":0,"room":1,
            "armed":True,"paused":False,"dead":False,"clear":False,
            "player":{"pos":[120,120],"hearts":6,"soul":0},
            "enemies":[{"id":1,"type":10,"hp":90,"pos":[200,120]}],"bullets":[],
            "grid_width":7,"grid_origin":[0,0],
            "grid":[[i,4 if i%7 in (0,6) or i//7 in (0,6) else 0,0] for i in range(49)]}


def test_outcomes_change_tactic_and_preferences_survive_restart(tmp_path):
    path=tmp_path/"experience.json"
    experience=Experience(path)
    state=observation()
    experience.consume(deepcopy(state))
    assert experience.tactic=="balanced"

    # The first tactic takes damage without hurting the enemy.
    state["frame"]=90
    state["player"]["hearts"]=4
    experience.consume(deepcopy(state))
    assert experience.tactic=="kite"
    # Kiting deals damage without taking any.
    state["frame"]=180
    state["enemies"][0]["hp"]=45
    experience.consume(deepcopy(state))
    assert experience.tactic=="close"
    # Closing in takes another hit. Prefer the observed successful tactic.
    state["frame"]=270
    state["player"]["hearts"]=2
    experience.consume(deepcopy(state))
    assert experience.tactic=="kite"
    resumed=Experience(path)
    assert resumed.choose("few_enemies")=="kite"
    assert resumed.data["updates"]==3


def test_death_after_pause_is_saved_once_and_new_run_is_counted(tmp_path):
    path=tmp_path/"experience.json"
    experience=Experience(path)
    state=observation()
    experience.consume(deepcopy(state))
    state["paused"]=True
    experience.consume(deepcopy(state))
    state["dead"]=True
    state["player"]["hearts"]=0
    experience.consume(deepcopy(state))
    experience.consume(deepcopy(state))
    assert Experience(path).data["deaths"]==1
    next_run=observation()
    next_run["session"]="next-run"
    experience.consume(next_run)
    assert Experience(path).data["runs"]==2


def test_manual_play_does_not_train_tactics_or_record_hazards():
    experience=Experience()
    state=observation()
    state["armed"]=False
    experience.consume(deepcopy(state))
    state["frame"]=90
    state["player"]["hearts"]=0
    state["dead"]=True
    experience.consume(state)
    assert experience.data["updates"]==0
    assert experience.data["damage"]==0
    assert experience.data["deaths"]==0
    assert experience.data["hazard_cells"]=={}


def test_remembered_danger_changes_route_and_is_scoped_to_room_geometry(tmp_path):
    state=observation()
    experience=Experience(tmp_path/"experience.json")
    experience.mark_position(state,8)
    experience.save()
    resumed=Experience(experience.path)
    grid=RoomGrid(state,costs=resumed.spatial_costs(state))
    current=[40,120]
    path=[]
    for _ in range(12):
        current,_=grid.route(current,[200,120])
        path.append(current)
        if current==[200,120]: break
    assert path[-1]==[200,120]
    assert [120,120] not in path
    other_room=deepcopy(state)
    other_room["grid"][10]=[10,0,8]
    assert not resumed.spatial_costs(other_room)


def test_disabled_learning_never_reads_or_changes_saved_experience(tmp_path,monkeypatch):
    path=tmp_path/"experience.json"
    path.write_text('{"schema":2,"caution":1.5,"runs":42}')
    original=path.read_bytes()
    def forbid_read(*args,**kwargs):
        raise AssertionError("No-learning mode read saved experience")
    monkeypatch.setattr(type(path),"read_text",forbid_read)
    experience=Experience(path,enabled=False)
    initial=deepcopy(experience.data)
    state=observation()
    experience.consume(deepcopy(state))
    state.update(frame=90,dead=True)
    state["player"]["hearts"]=0
    experience.consume(state)
    experience.mark_position(state,5)
    experience.update("boss","kite",-8)
    experience.finish_window()
    experience.save()
    assert experience.data==initial
    assert experience.tactic=="balanced"
    assert experience.spatial_costs(state)=={}
    assert path.read_bytes()==original
