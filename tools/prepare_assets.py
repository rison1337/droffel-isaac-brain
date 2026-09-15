"""Build the local MaleCNS graph and the small assets consumed by Godot.

The raw connectome is downloaded only when it is missing.  Generated graph,
soma and morphology files stay out of git because they are large and have
their own source hashes.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GRAPH = ROOT / "vendor" / "virtual-fly-lab" / "data"


def run(script, env=None):
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-morphology", action="store_true")
    args = parser.parse_args()
    if not args.skip_download:
        run(ROOT / "tools" / "download_assets.py")
    required = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    if not required.exists():
        raise FileNotFoundError(f"Missing raw MaleCNS data: {required}")
    env = os.environ.copy()
    env["FLY_DATA_DIR"] = str(RAW)
    GRAPH.mkdir(parents=True, exist_ok=True)
    if not (GRAPH / "graph_malecns_v1.npz").exists():
        run(ROOT / "vendor" / "virtual-fly-lab" / "tools" / "build_graph.py", env)
    if not (GRAPH / "soma_positions_f32.bin").exists():
        run(ROOT / "vendor" / "virtual-fly-lab" / "tools" / "extract_soma.py", env)
    run(ROOT / "tools" / "export_brain_assets.py", env)
    if not args.skip_morphology:
        run(ROOT / "tools" / "download_morphology.py", env)
    print("DROFFEL ASSETS READY")


if __name__ == "__main__":
    main()
