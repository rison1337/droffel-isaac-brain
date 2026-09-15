"""Create the curated source tree used for the public Droffel repository."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".publish" / "droffel"


def copy_file(src, dst=None):
    target = OUT / (dst or src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, target)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in ["README.md", "THIRD_PARTY.md", ".gitattributes", "requirements.txt",
                 "requirements-dev.txt", "requirements-gpu.txt", "game_paths.example.json",
                 "droffel_sim.py", "isaac_learning.py", "isaac_policy.py", "isaac_service.py",
                 "launch_isaac.py", "neural_groups.py", "research_brain.py",
                 "START_ISAAC.cmd", "START_ISAAC_NO_MEMORY.cmd"]:
        copy_file(name)
    for directory in ["tools", "tests", "reference_tests", "isaac_mod"]:
        shutil.copytree(ROOT / directory, OUT / directory)
    copy_file("terrarium/project.godot")
    for name in ["isaac_dashboard.gd", "brain_view.gd", "brain_chart.gd", "isaac_map.gd", "sim_client.gd"]:
        copy_file(f"terrarium/scripts/{name}")
    for name in ["__init__.py", "brain.py", "gpu.py", "server.py"]:
        copy_file(f"vendor/virtual-fly-lab/sim/{name}")
    for name in ["build_graph.py", "extract_soma.py"]:
        copy_file(f"vendor/virtual-fly-lab/tools/{name}")
    copy_file("vendor/virtual-fly-lab/LICENSE")
    (OUT / ".github/workflows").mkdir(parents=True)
    (OUT / ".github/workflows/tests.yml").write_text("""name: tests
on:
  push:
  pull_request:
jobs:
  pytest:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -r requirements-dev.txt
      - run: python -m pytest tests -q
""", encoding="utf-8")
    (OUT / ".gitignore").write_text(""".venv/
.tools/
data/
logs/
**/.godot/
terrarium/data/*
!terrarium/data/.gitkeep
vendor/virtual-fly-lab/data/
game_paths.json
isaac_mod/connection.lua
*.pyc
__pycache__/
.pytest_cache/
0916 (1).mp4
""", encoding="utf-8")
    (OUT / "terrarium/data/.gitkeep").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "terrarium/data/.gitkeep").write_text("", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
