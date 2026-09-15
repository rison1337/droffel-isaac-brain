"""Fetch pinned, public runtime/data assets without changing system settings."""
import hashlib
import json
import time
import http.client
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
NAMES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]


def fetch(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name}", flush=True)
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "droffel-local-lab"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=60) as source:
                    resume = source.status == 206
                    if resume and not source.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected range response")
                    received = offset if resume else 0
                    total = received + int(source.headers.get("Content-Length", "0"))
                    last = received
                    with partial.open("ab" if resume else "wb") as out:
                        while chunk := source.read(4 * 1024 * 1024):
                            out.write(chunk)
                            received += len(chunk)
                            if received - last >= 64 * 1024 * 1024:
                                print(f"  {received / 2**20:.0f} / {total / 2**20:.0f} MiB", flush=True)
                                last = received
                if total and received == total:
                    break
                print(f"Short response at {received} bytes; resuming", flush=True)
            except (OSError, http.client.IncompleteRead) as exc:
                print(f"Retry {attempt + 1}: {exc}", flush=True)
            time.sleep(1)
        else:
            raise RuntimeError(f"Incomplete download after retries: {path.name}")
        partial.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and expected != "sha256:" + digest:
        raise RuntimeError(f"Published hash mismatch: {path.name}")
    print(f"Ready: {path.name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return {"url": url, "bytes": path.stat().st_size, "sha256": digest,
            "published_hash_verified": bool(expected)}


def main():
    records = {}
    # Runtime first, so UI compilation can proceed during the graph download.
    url = "https://api.github.com/repos/godotengine/godot-builds/releases/tags/4.7.2-stable"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "droffel-local-lab"}), timeout=30) as response:
        release = json.load(response)
    asset = next(a for a in release["assets"] if a["name"] == "Godot_v4.7.2-stable_win64.exe.zip")
    archive = ROOT / ".tools" / asset["name"]
    records[asset["name"]] = fetch(asset["browser_download_url"], archive, asset.get("digest"))
    runtime = ROOT / ".tools" / "godot"
    runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (runtime / member.filename).resolve()
            if not target.is_relative_to(runtime.resolve()):
                raise ValueError("Invalid archive path")
        bundle.extractall(runtime)
    print("GODOT READY", flush=True)
    for name in NAMES:
        records[name] = fetch(f"{BASE}/{name}", ROOT / "data" / "raw" / name)
    (ROOT / "data" / "sources.lock.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
