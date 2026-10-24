"""Pure-Python forward kinematics of the SO-101 (no numpy, no MuJoCo) for the Raspberry Pi arm server.

The kinematic chain (body offsets, joint axes, joint ranges, tool points) is exported once from the
MuJoCo model (real/assets/so101/so101.xml, mujoco_menagerie robotstudio_so101) into real/so101_chain.json:
    PYTHONPATH=. .venv/bin/python -m real.kin_lite export          # needs mujoco (Mac)
The test real/tests/test_pi_stack.py checks this FK against MuJoCo's to 1e-9 m.

Frames and units as in real/so101.py: robot base frame, metres, joint angles in model radians.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CHAIN_JSON = Path(__file__).resolve().parent / "so101_chain.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


# ------------------------------------------------------------------ tiny 3-vector / 3x3 helpers
def _qmat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _axis_angle(a, th):
    x, y, z = a
    c, s, C = math.cos(th), math.sin(th), 1 - math.cos(th)
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(A, v):
    return [A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2] for i in range(3)]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


class Chain:
    def __init__(self, path=CHAIN_JSON):
        d = json.loads(Path(path).read_text())
        self.bodies = d["bodies"]                 # in tree order, each: name, parent, pos, quat, joint|None
        self.points = d["points"]                 # name -> {body, pos}
        self.ranges = {j: tuple(r) for j, r in d["ranges"].items()}
        self.source = d.get("source", "")
        for b in self.bodies:
            b["R"] = _qmat(b["quat"])

    def lo_hi(self):
        return [self.ranges[j][0] for j in JOINTS], [self.ranges[j][1] for j in JOINTS]

    def frames(self, q):
        """{body: (position m, R 3x3)} in the robot base frame for joint vector q (model rad, 5 or 6 values)."""
        qd = {j: (q[i] if i < len(q) else 0.0) for i, j in enumerate(JOINTS)}
        out = {}
        for b in self.bodies:
            if b["parent"] is None:
                p, R = list(b["pos"]), b["R"]
            else:
                pp, Rp = out[b["parent"]]
                p, R = _add(pp, _mv(Rp, b["pos"])), _mm(Rp, b["R"])
            j = b.get("joint")
            if j:
                Rj = _axis_angle(j["axis"], qd[j["name"]])
                jp = j.get("pos", [0.0, 0.0, 0.0])
                # rotation about the joint axis through the joint anchor (in the body frame)
                p = _add(p, _mv(R, _sub(jp, _mv(Rj, jp))))
                R = _mm(R, Rj)
            out[b["name"]] = (p, R)
        return out

    def point(self, q, name="fixed_tip"):
        f = self.frames(q)
        pt = self.points[name]
        p, R = f[pt["body"]]
        return _add(p, _mv(R, pt["pos"]))

    def check_points(self, q) -> dict:
        """Points used by the server's keep-out checks: tips of both jaws, gripper/wrist/elbow origins."""
        f = self.frames(q)
        out = {}
        for name, pt in self.points.items():
            p, R = f[pt["body"]]
            out[name] = _add(p, _mv(R, pt["pos"]))
        for b in ("wrist", "lower_arm", "gripper"):
            out[b] = f[b][0]
        return out


def export(path=CHAIN_JSON):
    """Write the chain JSON from the MuJoCo model (Mac only: needs mujoco)."""
    import mujoco

    from real.so101 import FIXED_TIP, MODEL_XML
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    bodies = []
    for name in BODIES:
        b = m.body(name)
        parent = None if name == "base" else m.body(int(m.body_parentid[b.id])).name
        jid = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b.id]
        joint = None
        if jid:
            j = m.joint(jid[0])
            joint = dict(name=j.name, axis=[float(v) for v in m.jnt_axis[j.id]],
                         pos=[float(v) for v in m.jnt_pos[j.id]])
        bodies.append(dict(name=name, parent=parent, pos=[float(v) for v in m.body_pos[b.id]],
                           quat=[float(v) for v in m.body_quat[b.id]], joint=joint))
    g = m.geom("moving_jaw_sph_tip1")
    gb = m.body(int(m.geom_bodyid[g.id])).name
    points = {"fixed_tip": dict(body="gripper", pos=[float(v) for v in FIXED_TIP]),
              "moving_tip": dict(body=gb, pos=[float(v) for v in m.geom_pos[g.id]])}
    ranges = {j: [float(v) for v in m.joint(j).range] for j in JOINTS}
    d = dict(source=f"exported from {MODEL_XML.relative_to(MODEL_XML.parents[3])} by real/kin_lite.py",
             bodies=bodies, points=points, ranges=ranges)
    Path(path).write_text(json.dumps(d, indent=1))
    return path


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["export"]:
        print("wrote", export())
    else:
        c = Chain()
        q = [float(v) for v in sys.argv[1:]] or [0.0] * 6
        print({k: [round(x * 1000, 1) for x in v] for k, v in c.check_points(q).items()})
