"""Download the openly licensed test models listed in real/open_models.json into real/assets/models/ (not committed).

    python -m real.fetch_models                 # all of them (~12 MB)
    python -m real.fetch_models --only benchy,clamp

Each file is checked against the sha1 in the manifest (a changed upstream file is reported, not used), and
real/assets/models/SOURCES.md records title, author, license and source page for every model fetched.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "open_models.json"
DEST = HERE / "assets" / "models"
UA = "printfarm-sim-model-fetch/1.0 (https://github.com/msannikov03/printfarm-sim)"


def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())["models"]


def model_path(mid: str, dest: Path = DEST) -> Path:
    return dest / f"{mid}.stl"


def available(dest: Path = DEST) -> dict:
    """{id: path} of the manifest models present (with the right sha1) in dest."""
    out = {}
    for m in manifest():
        p = model_path(m["id"], dest)
        if p.exists() and hashlib.sha1(p.read_bytes()).hexdigest() == m["sha1"]:
            out[m["id"]] = p
    return out


def fetch(only=None, dest: Path = DEST, timeout=120) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    got, have = {}, available(dest)
    for m in manifest():
        if only and m["id"] not in only:
            continue
        p = model_path(m["id"], dest)
        if m["id"] in have:
            got[m["id"]] = p
            continue
        req = urllib.request.Request(m["url"], headers={"User-Agent": UA})
        try:
            data = urllib.request.urlopen(req, timeout=timeout).read()
        except OSError as e:
            print(f"  {m['id']}: download failed ({e})", file=sys.stderr)
            continue
        sha = hashlib.sha1(data).hexdigest()
        if sha != m["sha1"]:
            print(f"  {m['id']}: sha1 {sha} does not match the manifest ({m['sha1']}): not used", file=sys.stderr)
            continue
        p.write_bytes(data)
        got[m["id"]] = p
        print(f"  {m['id']}: {len(data) / 1e6:.2f} MB, {m['license']}")
    lines = ["# Models in this folder", "", "Downloaded by `python -m real.fetch_models` from the sources below.", "",
             "| File | Title | Author | License | Source |", "|---|---|---|---|---|"]
    lines += [f"| {model_path(m['id'], dest).name} | {m['title']} | {m['author']} | {m['license']} | {m['page']} |"
              for m in manifest() if model_path(m["id"], dest).exists()]
    (dest / "SOURCES.md").write_text("\n".join(lines) + "\n")
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description="download the open test models (real/open_models.json)")
    ap.add_argument("--only", help="comma-separated model ids")
    ap.add_argument("--dest", default=str(DEST))
    a = ap.parse_args(argv)
    got = fetch(set(a.only.split(",")) if a.only else None, Path(a.dest))
    print(f"{len(got)} model(s) in {a.dest}")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
