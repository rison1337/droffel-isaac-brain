-- Droffel: only the first local player's movement and tears are overridden.
-- The only console command is a new run after game-over while control is armed.
local mod = RegisterMod("Droffel Fly Brain", 1)
local json = require("json")
local ok, socket = pcall(require, "socket")
local game = Game()
local config = include("connection")
local client, connecting, incoming, outgoing = nil, false, "", ""
local retryAt, lastActionAt, lastSend = 0, -100, -100
local armed, inRun, action = false, false, nil
local session, lastControl, stopEpoch = "", 0, 0
local frame, inputCalls, tears = 0, 0, 0
local suspended, shootCalls, lastShotFrame = false, 0, -1
local restartPending, restartAt = false, -1
local resumeNextRun, restartIssuedAt, restarts = false, -1, 0
local lastUseId, itemPulseFrame, cardPulseFrame = "", -1, -1
local status = ok and "Waiting for a run / local brain" or "DROFFEL STOPPED: start with START_ISAAC.cmd (--luadebug)"
local function now() return ok and socket.gettime() or Isaac.GetTime()/1000 end
local function stop(reason)
    armed, action = false, nil
    restartPending, resumeNextRun = false, false
    stopEpoch = stopEpoch + 1
    status = reason
    itemPulseFrame, cardPulseFrame = -1, -1
end
local function invalidateAction()
    action = nil
    stopEpoch = stopEpoch + 1
    lastActionAt = now()
    lastSend = -100
    itemPulseFrame, cardPulseFrame = -1, -1
end
local function disconnect()
    if client then client:close() end
    client, connecting, incoming, outgoing = nil, false, "", ""
    retryAt = now()+1
    stop("Disconnected - F6 to resume after reconnect")
end
local function vec(v) return {v.X, v.Y} end
local function fresh()
    return armed and inRun and action and not game:IsPaused() and not Isaac.GetPlayer(0):IsDead()
        and action.session == session and action.room == game:GetLevel():GetCurrentRoomIndex()
        and action.epoch == stopEpoch and now()-lastActionAt < 0.35
        and frame-action.frame >= 0 and frame-action.frame <= 12
end
local function observe()
    local p, room, level = Isaac.GetPlayer(0), game:GetRoom(), game:GetLevel()
    local enemies, bullets, pickups, doors, grid, exits, hazards, poops = {}, {}, {}, {}, {}, {}, {}, {}
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        -- IsActiveEnemy excludes fireplaces. A clear room can still contain
        -- contact damage, so observe burning fires independently of enemies.
        if e.Type == EntityType.ENTITY_FIREPLACE then
            if not e:IsDead() and e.EntityCollisionClass~=0 and e.CollisionDamage>0 then
                hazards[#hazards+1] = {id=e.InitSeed, pos=vec(e.Position), vel=vec(e.Velocity),
                    size=e.Size, type=e.Type, variant=e.Variant, kind="fire", hp=e.HitPoints,
                    destructible=e.Variant==0 or e.Variant==1 or e.Variant==10}
            end
        elseif e:IsActiveEnemy(false) and not e:IsDead() and not e:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
            enemies[#enemies+1] = {id=e.InitSeed, pos=vec(e.Position), vel=vec(e.Velocity), size=e.Size,
                hp=e.HitPoints, max_hp=e.MaxHitPoints, variant=e.Variant, subtype=e.SubType,
                vulnerable=e:IsVulnerableEnemy(), type=e.Type}
        elseif e.Type == EntityType.ENTITY_PROJECTILE then
            bullets[#bullets+1] = {pos=vec(e.Position), vel=vec(e.Velocity), size=e.Size}
        elseif e.Type == EntityType.ENTITY_PICKUP then
            local pick = e:ToPickup()
            pickups[#pickups+1] = {id=e.InitSeed, pos=vec(e.Position), variant=e.Variant,
                subtype=e.SubType, price=pick.Price}
        elseif e.Type == EntityType.ENTITY_BOMB or e.Type == EntityType.ENTITY_LASER then
            bullets[#bullets+1] = {pos=vec(e.Position), vel=vec(e.Velocity), size=50}
        end
    end
    for slot=0,7 do
        local door = room:GetDoor(slot)
        if door then doors[#doors+1] = {slot=slot, pos=vec(door.Position), open=door:IsOpen(),
            locked=door:IsLocked(), target=door.TargetRoomIndex, type=door.TargetRoomType} end
    end
    for i=0,room:GetGridSize()-1 do
        local g = room:GetGridEntity(i)
        local typ = g and g:GetType() or 0
        grid[#grid+1] = {i, room:GetGridCollision(i), typ}
        -- Poop is a destructible grid obstacle.  It is exposed separately so
        -- the planner can aim at it when an enemy is hiding behind it.
        if typ == (GridEntityType.GRID_POOP or 14) and room:GetGridCollision(i)~=0 then
            poops[#poops+1] = {id=i, pos=vec(room:GetGridPosition(i)), type=typ}
        end
        if typ == GridEntityType.GRID_TRAPDOOR or typ == GridEntityType.GRID_STAIRS then
            exits[#exits+1] = {pos=vec(room:GetGridPosition(i)), type=typ}
        end
    end
    local active_item, active_charge, active_max_charge, card, coins = 0, 0, 1, 0, 0
    local tear_range, shot_speed, tear_inheritance = 260, 1, nil
    pcall(function() active_item = p:GetActiveItem() or 0 end)
    pcall(function() active_charge = p:GetActiveCharge() or 0 end)
    pcall(function()
        local item = Isaac.GetItemConfig():GetCollectible(active_item)
        if item then active_max_charge = item.MaxCharges end
    end)
    pcall(function() card = p:GetCard(0) or 0 end)
    pcall(function() coins = p:GetNumCoins() or 0 end)
    pcall(function() tear_range = p.TearRange or tear_range end)
    pcall(function() shot_speed = p.ShotSpeed or shot_speed end)
    pcall(function()
        tear_inheritance = {}
        for _, d in ipairs({Vector(-1,0),Vector(0,-1),Vector(1,0),Vector(0,1)}) do
            local v = p:GetTearMovementInheritance(d)
            tear_inheritance[#tear_inheritance+1] = vec(v)
        end
    end)
    return {kind="observation", token=config.token, session=session, frame=frame, epoch=stopEpoch,
        armed=armed, paused=game:IsPaused(), dead=p:IsDead(), controls=p:AreControlsEnabled(),
        players=game:GetNumPlayers(), room=level:GetCurrentRoomIndex(), stage=level:GetStage(),
        stage_type=level:GetStageType(), room_type=room:GetType(), clear=room:IsClear(),
        player={pos=vec(p.Position), vel=vec(p.Velocity), size=p.Size, speed=p.MoveSpeed,
            hearts=p:GetHearts(), max_hearts=p:GetMaxHearts(), soul=p:GetSoulHearts(), flying=p.CanFly,
            coins=coins, active_item=active_item, active_charge=active_charge, active_max_charge=active_max_charge, card=card,
            tear_range=tear_range, shot_speed=shot_speed, tear_inheritance=tear_inheritance},
        enemies=enemies, bullets=bullets, pickups=pickups, doors=doors, exits=exits, hazards=hazards, poops=poops,
        grid=grid, grid_width=room:GetGridWidth(), grid_origin=vec(room:GetGridPosition(0)),
        bounds={vec(room:GetTopLeftPos()), vec(room:GetBottomRightPos())},
        applied=fresh() and action.move or {0,0}, applied_shoot=fresh() and action.shoot or {0,0},
        input_calls=inputCalls, shoot_calls=shootCalls, tears=tears, last_shot_frame=lastShotFrame,
        mod_version="1.4.0", suspended=suspended, status=status, restarts=restarts,
        restart_pending=restartPending, restart_seconds=restartPending and math.max(0,restartAt-now()) or 0}
end
local function receive(msg)
    if type(msg) ~= "table" or msg.token ~= config.token then return end
    if msg.kind == "control" and type(msg.seq)=="number" and msg.seq>lastControl then
        lastControl = msg.seq
        if msg.value == false then stop("Stopped - F6 to resume")
        elseif msg.session==session and msg.epoch==stopEpoch and inRun and not game:IsPaused()
            and game:GetNumPlayers()==1 and not Isaac.GetPlayer(0):IsDead() then
            armed, status, lastActionAt = true, "Fly control ON", now()
        end
    elseif msg.kind == "action" and msg.session==session and msg.epoch==stopEpoch
        and type(msg.frame)=="number" and msg.frame<=frame and frame-msg.frame<=12
        and msg.room==game:GetLevel():GetCurrentRoomIndex()
        and type(msg.move)=="table" and type(msg.shoot)=="table"
        and (msg.use_item==nil or type(msg.use_item)=="boolean")
        and (msg.use_card==nil or type(msg.use_card)=="boolean")
        and (msg.use_id==nil or (type(msg.use_id)=="string" and #msg.use_id<200)) then
        for _, values in ipairs({msg.move,msg.shoot}) do
            for i=1,2 do
                if type(values[i])~="number" or values[i]~=values[i] or math.abs(values[i])>1 then return end
            end
        end
        if not action or msg.frame >= action.frame then action, lastActionAt = msg, now() end
    end
end
local function pump()
    if not ok then return end
    local t = now()
    if not client and t>=retryAt then
        client = socket.tcp()
        client:settimeout(0)
        client:setoption("tcp-nodelay",true)
        local connected, err = client:connect("127.0.0.1",config.port)
        connecting = not connected
        if err and err~="timeout" and err~="Operation already in progress" then disconnect(); return end
    end
    if not client then return end
    if connecting then
        local _, writable = socket.select({}, {client}, 0)
        if #writable==0 then return end
        if not client:getpeername() then disconnect(); return end
        connecting = false
        status = "Brain connected - F6 to control"
        Isaac.DebugString("Droffel connected on localhost:"..config.port)
    end
    for _=1,8 do
        local line, err, partial = client:receive("*l")
        if line then
            local parsed, msg = pcall(json.decode,incoming..line)
            incoming = ""
            if parsed then receive(msg) end
        else
            incoming = incoming..(partial or "")
            if err=="closed" or #incoming>65536 then disconnect() end
            break
        end
    end
    if not client then return end
    if outgoing=="" and t-lastSend>=1/15 and inRun then
        outgoing = json.encode(observe()).."\n"
        lastSend = t
    end
    if outgoing~="" then
        local sent, err, partial = client:send(outgoing)
        outgoing = outgoing:sub((sent or partial or 0)+1)
        if err=="closed" then disconnect() end
    end
end
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, continued)
    local resume = resumeNextRun and client~=nil and not connecting
    inRun, frame, inputCalls, tears = true, game:GetFrameCount(), 0, 0
    suspended, shootCalls, lastShotFrame, restartPending = false, 0, -1, false
    session = tostring(game:GetSeeds():GetStartSeed())..":"..tostring(now())
    stop("F6: fly control | F7: stop")
    incoming, outgoing = "", ""
    invalidateAction()
    if resume then armed, status = true, "New attempt - fly control ON" end
    Isaac.DebugString("Droffel run started; session="..session)
end)
mod:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, function()
    inRun = false
    if not (resumeNextRun and now()-restartIssuedAt<5) then disconnect() end
end)
mod:AddCallback(ModCallbacks.MC_POST_GAME_END, function(_, gameover)
    if gameover and armed and ok then
        -- Restart only after an actual game-over, never after a victory ending.
        -- The game-over screen can freeze game frames. Use wall-clock time;
        -- render callbacks and the brain heartbeat continue during the delay.
        restartPending, restartAt = true, now()+3
        status = "Game over - starting a new run"
        Isaac.DebugString("Droffel game over; automatic restart scheduled")
    elseif not gameover then stop("Run won - stopped") end
end)
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    frame = game:GetFrameCount()
end)
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    -- A transition pauses the engine but must not disarm the controller.
    -- Reject commands from the old room even if its index is reused later.
    invalidateAction()
    Isaac.DebugString("Droffel room="..game:GetLevel():GetCurrentRoomIndex().." armed="..tostring(armed))
end)
mod:AddCallback(ModCallbacks.MC_POST_FIRE_TEAR, function(_, tear)
    if tear.SpawnerEntity and tear.SpawnerEntity.InitSeed==Isaac.GetPlayer(0).InitSeed then
        tears, lastShotFrame = tears+1, game:GetFrameCount()
    end
end)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    if not inRun then return end
    if Input.IsButtonTriggered(Keyboard.KEY_F7,0) or Input.IsButtonTriggered(Keyboard.KEY_ESCAPE,0)
        or Input.IsActionTriggered(ButtonAction.ACTION_PAUSE,Isaac.GetPlayer(0).ControllerIndex) then
        stop("Stopped - F6 to resume")
    elseif Input.IsButtonTriggered(Keyboard.KEY_F6,0) then
        if armed then stop("Manual control")
        elseif client and not connecting and not game:IsPaused() and game:GetNumPlayers()==1 then
            armed, status, lastActionAt = true, "Fly control ON", now()
        end
    end
    if armed and game:GetNumPlayers()~=1 then
        stop("Multiplayer - stopped")
    end
    local isSuspended = game:IsPaused() or Isaac.GetPlayer(0):IsDead() or not Isaac.GetPlayer(0):AreControlsEnabled()
    if isSuspended~=suspended then
        suspended = isSuspended
        invalidateAction()
        if armed then status = suspended and "Transition / pause - waiting" or "Fly control ON" end
    end
    if armed and not suspended and lastActionAt>0 and now()-lastActionAt>0.7 then stop("Brain timeout - F6 to resume") end
    pump()
    if restartPending and armed and now()>=restartAt then
        if client and not connecting and now()-lastActionAt<1 then
            restartPending, resumeNextRun, restartIssuedAt = false, true, now()
            restarts = restarts+1
            Isaac.DebugString("Droffel restarting after game over; attempt="..restarts)
            Isaac.ExecuteCommand("restart")
            return
        else stop("Brain disconnected - restart cancelled") end
    end
    if resumeNextRun and now()-restartIssuedAt>5 then stop("Restart failed - stopped") end
    if restartPending then status = string.format("New attempt in %.1fs - F7 cancels",math.max(0,restartAt-now())) end
    Isaac.RenderText("DROFFEL  "..(fresh() and "FLY PLAYING" or status), 60, 16, .55, 1, .75, 1)
    Isaac.RenderText("F6 control   F7 stop   Esc pause + stop", 60, 28, .8, .9, .85, 1)
end)
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, button)
    if not fresh() or not entity or entity.Type~=EntityType.ENTITY_PLAYER then return end
    local p = entity:ToPlayer()
    if p.InitSeed~=Isaac.GetPlayer(0).InitSeed or not p:AreControlsEnabled() then return end
    local v
    if button==ButtonAction.ACTION_LEFT then v=math.max(0,-action.move[1])
    elseif button==ButtonAction.ACTION_RIGHT then v=math.max(0,action.move[1])
    elseif button==ButtonAction.ACTION_UP then v=math.max(0,-action.move[2])
    elseif button==ButtonAction.ACTION_DOWN then v=math.max(0,action.move[2])
    elseif button==ButtonAction.ACTION_SHOOTLEFT then v=math.max(0,-action.shoot[1])
    elseif button==ButtonAction.ACTION_SHOOTRIGHT then v=math.max(0,action.shoot[1])
    elseif button==ButtonAction.ACTION_SHOOTUP then v=math.max(0,-action.shoot[2])
    elseif button==ButtonAction.ACTION_SHOOTDOWN then v=math.max(0,action.shoot[2])
    elseif button==ButtonAction.ACTION_ITEM or button==ButtonAction.ACTION_PILLCARD then
        -- Repeat packets and multiple input hooks must produce one press.
        if action.use_id and action.use_id~="" and action.use_id~=lastUseId then
            if action.use_item then itemPulseFrame, lastUseId = frame, action.use_id
            elseif action.use_card then cardPulseFrame, lastUseId = frame, action.use_id end
        end
        v = ((button==ButtonAction.ACTION_ITEM and itemPulseFrame==frame)
            or (button==ButtonAction.ACTION_PILLCARD and cardPulseFrame==frame)) and 1 or 0
    else return end
    inputCalls = inputCalls+1
    if button>=ButtonAction.ACTION_SHOOTLEFT and button<=ButtonAction.ACTION_SHOOTDOWN and v>0 then
        shootCalls = shootCalls+1
    end
    if hook==InputHook.GET_ACTION_VALUE then return v end
    return v>0.2
end)
Isaac.DebugString("Droffel 1.4.0 loaded; socket="..tostring(ok))
