"""Slice real models with the BambuStudio command line: models -> one arranged plate -> labelled G-code in a 3MF.

BambuStudio (02.x) slices from the command line when it gets complete machine / process / filament settings. Its
bundled system profiles are layered ("inherits" a parent, "include" template files for the start / end G-code), and
the command line does not resolve those layers itself: fed the raw files it falls back to default values (a 200 mm
bed, no start G-code). `resolve_profile` flattens a profile into one complete JSON first.

    python -m real.bambu out_dir model1.stl model2.3mf ...      # -> out_dir/plate.gcode.3mf (+ plate_1.gcode)

The result is a Bambu project 3MF: the arranged models (3D/*.model, with the build transforms the slicer chose) and
the sliced plate (Metadata/plate_1.gcode, Bambu "; start printing object, unique label id: N" labels, with the label
ids mapped to object names in Metadata/slice_info.config). `real.gcode_objects.extract` reads it directly.

The executable is found from $BAMBU_STUDIO, then the default macOS install, then `bambu-studio` / `BambuStudio` on
the PATH. Nothing is written outside the given output directory (the program's own settings go to out_dir/datadir).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAC_APP = Path("/Applications/BambuStudio.app")
PRINTER = "Bambu Lab P1S 0.4 nozzle"           # the sim's printer: P1S-class, 256 x 256 mm bed
PROCESS = "0.20mm Standard @BBL X1C"           # the P1S's default print profile
FILAMENT = "Bambu PLA Basic @BBL P1S 0.4 nozzle"
# farm defaults on top of the stock profile: no brim (a brim widens the printed footprint past the model's, and a farm
# part should come off the bed without trimming), object labels on (the robot's plan comes from them)
PROCESS_OVERRIDES = {"brim_type": "no_brim", "exclude_object": "1"}
_META = {"inherits", "include", "instantiation", "setting_id", "description"}


def find_bambu() -> Path | None:
    """The BambuStudio executable, or None."""
    env = os.environ.get("BAMBU_STUDIO")
    if env and Path(env).is_file():
        return Path(env)
    exe = MAC_APP / "Contents" / "MacOS" / "BambuStudio"
    if exe.is_file():
        return exe
    for name in ("bambu-studio", "BambuStudio"):
        p = shutil.which(name)
        if p:
            return Path(p)
    return None


def profiles_dir(exe: Path | None = None) -> Path | None:
    """The bundled system profiles (profiles/BBL) next to the executable."""
    exe = exe or find_bambu()
    if exe is None:
        return None
    for d in (exe.parent.parent / "Resources" / "profiles" / "BBL",          # macOS bundle
              exe.parent / "resources" / "profiles" / "BBL",                 # Linux AppImage / Windows layout
              exe.parent.parent / "share" / "BambuStudio" / "profiles" / "BBL"):
        if d.is_dir():
            return d
    return None


def resolve_profile(kind: str, name: str, root: Path) -> dict:
    """Flatten a system profile (kind = machine | process | filament): the parent chain ("inherits"), then the
    "include"d template files, then the profile's own keys, later ones winning."""
    d = json.loads((root / kind / f"{name}.json").read_text())
    out = resolve_profile(kind, d["inherits"], root) if d.get("inherits") else {}
    for inc in d.get("include", []):
        out.update({k: v for k, v in json.loads((root / kind / f"{inc}.json").read_text()).items()
                    if k not in _META and k != "name"})
    out.update({k: v for k, v in d.items() if k not in _META})
    return out


def write_profiles(out_dir: Path, root: Path, printer=PRINTER, process=PROCESS, filament=FILAMENT,
                   process_overrides=None) -> tuple[Path, Path, Path]:
    """Flattened machine / process / filament JSONs in out_dir (names and types kept, as the command line expects)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, name, extra in (("machine", printer, {}), ("process", process, {**PROCESS_OVERRIDES,
                                                                               **(process_overrides or {})}),
                              ("filament", filament, {})):
        d = resolve_profile(kind, name, root)
        d.update(extra)
        d.update(type=kind, name=name, **{"from": "system"})
        p = out_dir / f"{kind}.json"
        p.write_text(json.dumps(d, indent=1))
        paths.append(p)
    return tuple(paths)


def slice_models(models, out_dir, exe=None, arrange=True, orient=False, process_overrides=None,
                 timeout_s=600, log=None) -> Path:
    """Arrange `models` (STL / 3MF paths) on one P1S plate and slice it -> out_dir/plate.gcode.3mf.
    Raises RuntimeError if BambuStudio is missing or slicing fails (its log is in out_dir/bambu.log)."""
    exe = Path(exe) if exe else find_bambu()
    if exe is None:
        raise RuntimeError("BambuStudio not found (set BAMBU_STUDIO)")
    root = profiles_dir(exe)
    if root is None:
        raise RuntimeError(f"no profiles/BBL next to {exe}")
    out_dir = Path(out_dir).resolve()
    machine, process, filament = write_profiles(out_dir / "profiles", root, process_overrides=process_overrides)
    cmd = [str(exe), "--datadir", str(out_dir / "datadir"), "--debug", "2",
           "--arrange", "1" if arrange else "0", "--orient", "1" if orient else "0", "--slice", "0",
           "--load-settings", f"{machine};{process}", "--load-filaments", str(filament),
           "--outputdir", str(out_dir), "--export-3mf", "plate.gcode.3mf", *[str(Path(m).resolve()) for m in models]]
    with open(out_dir / "bambu.log", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=out_dir)
    res = out_dir / "result.json"
    info = json.loads(res.read_text()) if res.exists() else {}
    out = out_dir / "plate.gcode.3mf"
    if r.returncode != 0 or not out.exists() or info.get("return_code", 0) != 0:
        raise RuntimeError(f"BambuStudio failed (rc={r.returncode}, {info.get('error_string')}); see "
                           f"{out_dir / 'bambu.log'}")
    if log:
        print(f"sliced {len(models)} model(s) -> {out}", file=log)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="arrange + slice models on one P1S plate with BambuStudio")
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--orient", action="store_true", help="let BambuStudio pick each model's print orientation")
    ap.add_argument("--no-arrange", action="store_true", help="keep the positions in the input 3MF")
    a = ap.parse_args(argv)
    print(slice_models(a.models, a.out_dir, arrange=not a.no_arrange, orient=a.orient, log=sys.stdout))


if __name__ == "__main__":
    main()
