clock = 10
paused = false
controlsEnabled = true
dead = false
numPlayers = 1
roomIndex = 1
frameCount = 100
trigger = nil
actionTrigger = nil
received = {}
sent = {}
callbacks = {}
ModCallbacks={MC_POST_GAME_STARTED=15,MC_POST_GAME_END=16,MC_PRE_GAME_EXIT=17,MC_POST_UPDATE=1,MC_POST_RENDER=2,MC_INPUT_ACTION=13,MC_POST_NEW_ROOM=19,MC_POST_TEAR_INIT=39,MC_POST_FIRE_TEAR=61}
InputHook={IS_ACTION_PRESSED=0,IS_ACTION_TRIGGERED=1,GET_ACTION_VALUE=2}
ButtonAction={ACTION_LEFT=0,ACTION_RIGHT=1,ACTION_UP=2,ACTION_DOWN=3,ACTION_SHOOTLEFT=4,ACTION_SHOOTRIGHT=5,ACTION_SHOOTUP=6,ACTION_SHOOTDOWN=7,ACTION_PAUSE=12}
Keyboard={KEY_ESCAPE=256,KEY_F6=295,KEY_F7=296}
EntityType={ENTITY_PLAYER=1,ENTITY_PICKUP=5,ENTITY_PROJECTILE=9,ENTITY_BOMB=4,ENTITY_LASER=7,ENTITY_FIREPLACE=33}
EntityFlag={FLAG_FRIENDLY=1}
GridEntityType={GRID_TRAPDOOR=17,GRID_STAIRS=18}
player={InitSeed=3,Type=1,ControllerIndex=0,Position={X=160,Y=160},Velocity={X=0,Y=0},Size=10,MoveSpeed=1,CanFly=false}
function player:ToPlayer() return self end
function player:IsDead() return dead end
function player:AreControlsEnabled() return controlsEnabled end
function player:GetHearts() return 6 end
function player:GetMaxHearts() return 6 end
function player:GetSoulHearts() return 0 end
room={}
function room:GetDoor() return nil end
function room:GetGridSize() return 1 end
function room:GetGridEntity() return nil end
function room:GetGridCollision() return 0 end
function room:GetGridPosition() return {X=0,Y=0} end
function room:GetGridWidth() return 1 end
function room:GetTopLeftPos() return {X=0,Y=0} end
function room:GetBottomRightPos() return {X=640,Y=400} end
function room:GetType() return 1 end
function room:IsClear() return true end
level={}
function level:GetCurrentRoomIndex() return roomIndex end
function level:GetStage() return 1 end
function level:GetStageType() return 0 end
game={}
function game:GetRoom() return room end
function game:GetLevel() return level end
function game:IsPaused() return paused end
function game:GetFrameCount() return frameCount end
function game:GetNumPlayers() return numPlayers end
function game:GetSeeds() return {GetStartSeed=function() return 123 end} end
function Game() return game end
Isaac={GetPlayer=function() return player end,GetRoomEntities=function() return {} end,
    ExecuteCommand=function(command) sent[#sent+1]=command end,
    DebugString=function() end,RenderText=function() end,GetTime=function() return clock*1000 end}
Input={IsButtonTriggered=function(key) if key==trigger then trigger=nil; return true end return false end,
    IsActionTriggered=function(key) if key==actionTrigger then actionTrigger=nil; return true end return false end}
function RegisterMod() return {AddCallback=function(_,id,fn) callbacks[id]=fn end} end
function include() return {port=29876,token="test"} end
sock={settimeout=function() end,setoption=function() end,connect=function() return 1 end,
    close=function() end,getpeername=function() return "127.0.0.1" end,
    send=function(_,text) sent[#sent+1]=text; return #text end,
    receive=function() if #received>0 then return table.remove(received,1) end return nil,"timeout","" end}
package.preload.socket=function() return {gettime=function() return clock end,tcp=function() return sock end,
    select=function() return {},{sock} end} end
