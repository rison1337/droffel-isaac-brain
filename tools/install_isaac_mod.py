"""Install only Droffel's own mod directory; preserve other mods and game files."""
import json
import secrets
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install():
    game = Path(json.loads((ROOT/"game_paths.json").read_text(encoding="utf-8"))["isaac"]["path"])
    if not (game/"isaac-ng.exe").is_file():
        raise FileNotFoundError(game/"isaac-ng.exe")
    config_file = ROOT/"data/isaac_connection.json"
    config_file.parent.mkdir(exist_ok=True)
    if config_file.exists():
        config = json.loads(config_file.read_text())
    else:
        config = {"port":29876,"token":secrets.token_hex(24)}
        config_file.write_text(json.dumps(config),encoding="utf-8")
    target = game/"mods/droffel_fly_brain"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("main.lua","metadata.xml"):
        shutil.copy2(ROOT/"isaac_mod"/name,target/name)
    (target/"connection.lua").write_text('return {port=%d, token="%s"}\n' % (config["port"],config["token"]),encoding="utf-8")
    print("Installed:",target)
    return game


if __name__=="__main__": install()
