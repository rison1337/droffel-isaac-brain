import json
from pathlib import Path
from lupa.lua53 import LuaRuntime, lua_type
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lua():
    runtime = LuaRuntime(unpack_returned_tuples=True)
    def plain(value):
        if lua_type(value)!="table": return value
        values=dict(value.items())
        if not values or all(isinstance(k,int) for k in values):
            return [plain(values[k]) for k in sorted(values)]
        return {k:plain(v) for k,v in values.items()}
    runtime.globals().encode_json=lambda obj:json.dumps(plain(obj))
    runtime.globals().decode_json=lambda text:runtime.table_from(json.loads(text),recursive=True)
    runtime.execute('package.preload.json=function() return {encode=encode_json,decode=decode_json} end')
    runtime.execute((ROOT/"tests/isaac_lua_harness.lua").read_text())
    runtime.execute((ROOT/"isaac_mod/main.lua").read_text())
    runtime.execute('callbacks[15](nil,false); callbacks[2](); trigger=295; callbacks[2]()')
    return runtime


def send_action(lua):
    lua.execute('clock=clock+.1; callbacks[2]()')
    observation = next(json.loads(lua.eval('sent['+str(i)+']'))
        for i in range(int(lua.eval('#sent')),0,-1) if lua.eval('sent['+str(i)+']').startswith('{'))
    action = {"kind":"action","token":"test","session":observation["session"],
        "epoch":observation["epoch"],"room":observation["room"],"frame":observation["frame"],"move":[1,0],"shoot":[0,-1]}
    lua.globals().payload=json.dumps(action)
    lua.execute('received[#received+1]=payload; callbacks[2]()')
    return action


def test_lua_53_mod_overrides_only_valid_player_movement_and_shooting(lua):
    send_action(lua)
    assert lua.eval('callbacks[13](nil,player,2,1)')==1
    assert lua.eval('callbacks[13](nil,player,0,6)') is True
    assert lua.eval('callbacks[13](nil,player,2,0)')==0
    assert lua.eval('callbacks[13](nil,nil,2,1)') is None
    assert lua.eval('callbacks[13](nil,player,2,8)') is None


@pytest.mark.parametrize("change",["clock=clock+.4","paused=true","roomIndex=2","frameCount=114; callbacks[1]()",
    "trigger=296; callbacks[2]()","trigger=256; callbacks[2]()","dead=true; callbacks[2]()",
    "numPlayers=2; callbacks[2]()","callbacks[17]()","callbacks[15](nil,false)"])
def test_lua_releases_inputs_on_stale_pause_room_exit_and_stop(lua,change):
    send_action(lua)
    lua.execute(change)
    assert lua.eval('callbacks[13](nil,player,2,1)') is None


def test_stop_epoch_rejects_delayed_arm_command(lua):
    action = send_action(lua)
    lua.execute('trigger=296; callbacks[2]()')
    command = dict(action,kind="control",seq=1,value=True)
    lua.globals().payload=json.dumps(command)
    lua.execute('received[#received+1]=payload; callbacks[2]()')
    assert lua.eval('callbacks[13](nil,player,2,1)') is None


def test_door_transition_keeps_armed_and_resumes_shooting_in_next_room(lua):
    old = send_action(lua)
    # Actual failure sequence: room changes, engine pauses for 0.33s, resumes.
    lua.execute('roomIndex=2; paused=true; controlsEnabled=false; callbacks[19](); callbacks[2]()')
    assert lua.eval('callbacks[13](nil,player,2,1)') is None
    lua.execute('clock=clock+2; callbacks[2]()')  # Also covers long boss intros.
    assert json.loads(lua.eval('sent[#sent]'))['armed'] is True
    lua.execute('paused=false; controlsEnabled=true; frameCount=101; callbacks[1](); callbacks[2]()')
    lua.globals().payload=json.dumps(old)
    lua.execute('received[#received+1]=payload; callbacks[2]()')
    assert lua.eval('callbacks[13](nil,player,2,1)') is None
    new = send_action(lua)
    assert new['room']==2 and new['epoch']>old['epoch']
    assert lua.eval('callbacks[13](nil,player,2,6)')==1
    lua.execute('callbacks[61](nil,{SpawnerEntity=player}); clock=clock+.1; callbacks[2]()')
    snapshot=json.loads(lua.eval('sent[#sent]'))
    assert snapshot['tears']==1 and snapshot['shoot_calls']>0 and snapshot['armed']


@pytest.mark.parametrize('button',['trigger=256','trigger=296','actionTrigger=12'])
def test_manual_stop_during_transition_never_resumes(lua,button):
    send_action(lua)
    lua.execute('roomIndex=2; paused=true; callbacks[19](); callbacks[2]()')
    lua.execute(button+'; callbacks[2](); paused=false; callbacks[2]()')
    send_action(lua)
    assert lua.eval('callbacks[13](nil,player,2,6)') is None


def test_returning_to_same_room_cannot_reuse_old_action(lua):
    old=send_action(lua)
    lua.execute('roomIndex=2; callbacks[19](); roomIndex=1; callbacks[19]()')
    lua.globals().payload=json.dumps(old)
    lua.execute('received[#received+1]=payload; callbacks[2]()')
    assert lua.eval('callbacks[13](nil,player,2,1)') is None


def restart_commands(lua):
    return [lua.eval('sent['+str(i)+']') for i in range(1, int(lua.eval('#sent'))+1)
            if lua.eval('sent['+str(i)+']') == 'restart']


def finish_countdown(lua):
    # Game frames stay frozen on the death screen. The backend keeps replying
    # with zero actions, which confirms that it is ready for another attempt.
    for _ in range(35):
        send_action(lua)


def test_game_over_restarts_with_frozen_frames_and_resumes_without_f6(lua):
    send_action(lua)
    lua.execute('callbacks[16](nil,true); dead=true; paused=true; callbacks[2]()')
    assert not restart_commands(lua)
    finish_countdown(lua)
    assert restart_commands(lua) == ['restart']
    lua.execute('callbacks[17](); dead=false; paused=false; frameCount=0; callbacks[15](nil,false)')
    action = send_action(lua)
    assert action['frame'] == 0
    assert lua.eval('callbacks[13](nil,player,2,1)') == 1
    assert lua.eval('callbacks[13](nil,player,2,6)') == 1


def test_game_over_does_not_restart_after_f7(lua):
    send_action(lua)
    lua.execute('trigger=296; callbacks[2](); callbacks[16](nil,true); dead=true; callbacks[2]()')
    finish_countdown(lua)
    assert not restart_commands(lua)


@pytest.mark.parametrize('button',['trigger=256','trigger=296','actionTrigger=12'])
def test_stop_cancels_pending_new_attempt(lua,button):
    send_action(lua)
    lua.execute('callbacks[16](nil,true); dead=true; callbacks[2]()')
    lua.execute(button+'; callbacks[2]()')
    finish_countdown(lua)
    assert not restart_commands(lua)


def test_new_attempt_is_cancelled_if_brain_stops_replying(lua):
    send_action(lua)
    lua.execute('callbacks[16](nil,true); dead=true; callbacks[2](); clock=clock+4; callbacks[2]()')
    assert not restart_commands(lua)


def test_victory_does_not_restart(lua):
    send_action(lua)
    lua.execute('callbacks[16](nil,false); paused=true; callbacks[2]()')
    finish_countdown(lua)
    assert not restart_commands(lua)


def test_burning_fire_is_observed_even_when_it_is_not_an_enemy(lua):
    lua.execute('''
        fire={Type=33,Variant=0,InitSeed=55,Size=12,HitPoints=5,Position={X=160,Y=160},Velocity={X=0,Y=0},
            EntityCollisionClass=4,CollisionDamage=1,
            IsDead=function() return false end,IsActiveEnemy=function() return false end}
        Isaac.GetRoomEntities=function() return {fire} end
        clock=clock+.1; callbacks[2]()
    ''')
    snapshot=json.loads(lua.eval('sent[#sent]'))
    assert not snapshot['enemies']
    assert snapshot['hazards'][0]['type']==33
    assert snapshot['hazards'][0]['pos']==[160,160]
    assert snapshot['hazards'][0]['destructible'] is True
    lua.execute('fire.Variant=2; clock=clock+.1; callbacks[2]()')
    assert not json.loads(lua.eval('sent[#sent]'))['hazards'][0]['destructible']
    # Extinguished, non-colliding fires must stop blocking the route.
    lua.execute('fire.EntityCollisionClass=0; clock=clock+.1; callbacks[2]()')
    snapshot=json.loads(lua.eval('sent[#sent]'))
    assert not snapshot['hazards']
