"""Weber-Penn (1995) inspired procedural tree, simplified.

Each species is a parameter set. Levels (0=trunk, 1=main branches, 2=twigs,
3=leaf groups). Per level we control:
  L          base length of branches at this level
  curve      total curve angle along branch (degrees)
  curveV     random curve variance
  segs       number of segments per branch (smoothness)
  branches   how many child branches sprout per parent
  downAngle  pitch-down angle of child from parent direction
  downAngleV variance
  rot        roll rotation between children around parent axis
  rotV       variance
  taper      r_tip / r_base
  lengthV    length random variance fraction

Tree-level:
  scale       overall world height
  shape       0=conical (fir/spruce), 1=spherical (oak), 4=weeping (willow)
  baseSize    fraction of trunk with NO branches (above ground)
  ratio       trunk radius / scale
  leaves      total leaf count target

We project to 3D x,y,z (Y up), output a JSON of { segments, leaves }.
PIL preview to /tmp/wp_*.png

Heavily simplified vs the paper — but the per-level parameter idea is
faithful, and produces visually distinct species silhouettes.
"""
import json, math, os, random
from ipt.render_util import render_tree

from ipt.paths import RESULTS as OUT_DIR


# ---- 3D vector helpers (Y up) ----
def vadd(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def vmul(a, k): return (a[0] * k, a[1] * k, a[2] * k)
def vnorm(a):
    L = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) + 1e-12
    return (a[0] / L, a[1] / L, a[2] / L)

def rot_axis_angle(v, axis, ang):
    """Rodrigues — rotate v around axis by ang radians."""
    ax = vnorm(axis)
    c = math.cos(ang); s = math.sin(ang); k = 1 - c
    x, y, z = v
    ux, uy, uz = ax
    return (
        v[0] * (c + ux * ux * k) + v[1] * (ux * uy * k - uz * s) + v[2] * (ux * uz * k + uy * s),
        v[0] * (uy * ux * k + uz * s) + v[1] * (c + uy * uy * k) + v[2] * (uy * uz * k - ux * s),
        v[0] * (uz * ux * k - uy * s) + v[1] * (uz * uy * k + ux * s) + v[2] * (c + uz * uz * k),
    )

def perpendicular(v):
    """Some vector perpendicular to v."""
    if abs(v[0]) < 0.9:
        return vnorm((1 - v[0] * v[0], -v[0] * v[1], -v[0] * v[2]))
    return vnorm((-v[1] * v[0], 1 - v[1] * v[1], -v[1] * v[2]))


# ---- species presets ----
SPECIES = {
    "douglas_fir": {
        "scale": 10.0, "ratio": 0.018, "baseSize": 0.18, "levels": 3,
        # per level: L, curve, curveV, segs, branches, down, downV, rot, rotV, taper, lengthV
        "lvl": [
            # trunk (level 0): straight up, slight curve, full height
            dict(L=1.0,  curve=4,   curveV=8,   segs=12, branches=42, down=70,  downV=6, rot=137, rotV=15, taper=1.0, lengthV=0.0),
            # main branches: short, slightly drooping (fir feature)
            dict(L=0.45, curve=-10, curveV=20,  segs=6,  branches=30, down=70,  downV=10, rot=170, rotV=30, taper=0.6,  lengthV=0.15),
            # twigs: short flat horizontal at branch tips
            dict(L=0.25, curve=0,   curveV=25,  segs=4,  branches=12, down=55,  downV=20, rot=137, rotV=40, taper=0.6,  lengthV=0.25),
        ],
        "trunk_col": (60, 35, 22),
        "leaf_col":  (22, 80, 35),
        "leaf_per_tip": 35,
        "leaf_cluster_r": 0.25,
        "leaf_size": 0.06,
    },

    "weeping_willow": {
        "scale": 9.0, "ratio": 0.022, "baseSize": 0.20, "levels": 3,
        "lvl": [
            dict(L=1.0,  curve=8,   curveV=10,  segs=10, branches=26, down=50,  downV=8,  rot=137, rotV=20, taper=1.0, lengthV=0.05),
            # main branches arc UP then DROOP DOWN heavily — signature
            dict(L=0.60, curve=-110, curveV=25, segs=8,  branches=18, down=20,  downV=10, rot=140, rotV=40, taper=0.5, lengthV=0.20),
            # drooping twigs
            dict(L=0.45, curve=-50, curveV=30,  segs=5,  branches=6,  down=85,  downV=25, rot=120, rotV=60, taper=0.7,  lengthV=0.40),
        ],
        "trunk_col": (50, 38, 30),
        "leaf_col":  (130, 145, 65),
        "leaf_per_tip": 25,
        "leaf_cluster_r": 0.20,
        "leaf_size": 0.05,
    },

    "black_oak": {
        "scale": 8.5, "ratio": 0.034, "baseSize": 0.25, "levels": 3,
        "lvl": [
            dict(L=1.0,  curve=10,  curveV=20,  segs=8,  branches=14, down=45,  downV=15, rot=110, rotV=40, taper=1.0, lengthV=0.10),
            # broad lateral branches — main canopy spread
            dict(L=0.85, curve=-30, curveV=40,  segs=6,  branches=12, down=55,  downV=20, rot=130, rotV=40, taper=0.6, lengthV=0.30),
            # twigs reach for the light
            dict(L=0.42, curve=-20, curveV=50,  segs=4,  branches=8,  down=40,  downV=30, rot=130, rotV=70, taper=0.6, lengthV=0.40),
        ],
        "trunk_col": (60, 40, 30),
        "leaf_col":  (62, 100, 35),
        "leaf_per_tip": 50,
        "leaf_cluster_r": 0.45,
        "leaf_size": 0.10,
    },
}


def build_tree(species_name, rng=None):
    rng = rng or random.Random(7)
    sp = SPECIES[species_name]
    scale = sp["scale"]
    levels = sp["levels"]
    lvl = sp["lvl"]
    base_radius = scale * sp["ratio"]
    base_size = sp["baseSize"]

    segments = []                 # each: dict p0, p1, r0, r1
    leaves = []                   # each: dict p, r, col

    def emit_branch(level, start, direction, length, radius_base):
        """Build one branch by stepping along direction with curve, spawning
        child branches at each segment endpoint after baseSize fraction."""
        p = lvl[level]
        n_segs = p["segs"]
        seg_len = length / n_segs
        # 'curve' is total angle along the branch in degrees
        per_seg_curve = math.radians((p["curve"] + rng.uniform(-p["curveV"], p["curveV"]))) / n_segs

        # rotation plane: a stable "right" axis perpendicular to direction
        # so the curve happens in one plane (looks natural)
        right = perpendicular(direction)
        cur = start
        cur_dir = direction
        r0 = radius_base

        # spawn count per segment
        n_branches_total = p["branches"]
        # at trunk level, branches start above baseSize fraction of length
        first_seg_with_branches = int(n_segs * base_size) if level == 0 else 0
        spawn_per_seg = n_branches_total / max(1, n_segs - first_seg_with_branches)
        branches_made = 0

        rot_accum = rng.uniform(0, math.pi * 2)

        for seg_i in range(n_segs):
            # curve the direction along this segment
            cur_dir = rot_axis_angle(cur_dir, right, per_seg_curve)
            nxt = vadd(cur, vmul(cur_dir, seg_len))
            # taper radius along branch
            t = (seg_i + 1) / n_segs
            r_tip = radius_base * (1 - t * (1 - p["taper"]))
            segments.append({
                "p0": [round(cur[0], 3), round(cur[1], 3), round(cur[2], 3)],
                "p1": [round(nxt[0], 3), round(nxt[1], 3), round(nxt[2], 3)],
                "r0": round(r0, 4), "r1": round(r_tip, 4),
            })
            r0 = r_tip

            # spawn children at this seg endpoint (after baseSize gating)
            if level + 1 < levels and seg_i >= first_seg_with_branches:
                target_branches_by_now = (seg_i + 1 - first_seg_with_branches) * spawn_per_seg
                while branches_made + 0.5 < target_branches_by_now:
                    branches_made += 1
                    ch = lvl[level + 1]
                    # child length
                    chL = length * ch["L"] * (1 + rng.uniform(-ch["lengthV"], ch["lengthV"]))
                    # child direction: down-angle from parent + roll
                    rot_accum += math.radians(ch["rot"] + rng.uniform(-ch["rotV"], ch["rotV"]))
                    # roll the "right" axis around cur_dir
                    roll_right = rot_axis_angle(right, cur_dir, rot_accum)
                    # pitch DOWN by downAngle around the rolled right axis
                    down_ang = math.radians(ch["down"] + rng.uniform(-ch["downV"], ch["downV"]))
                    child_dir = rot_axis_angle(cur_dir, roll_right, -down_ang)
                    child_dir = vnorm(child_dir)
                    emit_branch(level + 1, nxt, child_dir, chL, r_tip * 0.85)

            cur = nxt

        # leaves at tip (if at deepest level)
        if level == levels - 1:
            for _ in range(sp["leaf_per_tip"]):
                jx = rng.uniform(-1, 1); jy = rng.uniform(-1, 1); jz = rng.uniform(-1, 1)
                L = math.sqrt(jx * jx + jy * jy + jz * jz) + 1e-9
                d = rng.random() ** 0.7 * sp["leaf_cluster_r"]
                pos = (cur[0] + jx / L * d, cur[1] + jy / L * d, cur[2] + jz / L * d)
                # leaf colour with a little jitter
                cr, cg, cb = sp["leaf_col"]
                jitter = rng.uniform(-12, 12)
                lc = (max(0, min(255, cr + int(jitter))),
                      max(0, min(255, cg + int(jitter * 0.8))),
                      max(0, min(255, cb + int(jitter * 0.6))))
                leaves.append({"p": [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)],
                               "r": round(sp["leaf_size"], 4), "col": list(lc)})

    # trunk grows straight up from origin
    emit_branch(0, (0, 0, 0), (0, 1, 0), scale, base_radius)

    return segments, leaves, sp


def render(species_name, out_path, *, seed=7):
    segs, leaves, sp = build_tree(species_name, rng=random.Random(seed))
    print(f"  {species_name}: {len(segs)} segs, {len(leaves)} leaves")
    render_tree(segs, leaves, out_path,
                title=f"Weber-Penn — {species_name}",
                trunk_col=sp["trunk_col"], leaf_col=sp["leaf_col"],
                leaf_radius=3)
    # also save the JSON for later three.js consumption
    json_path = os.path.join(OUT_DIR, f"wp_tree_{species_name}.json")
    json.dump({
        "kind": species_name,
        "segments": segs,
        "leaves": leaves,
        "trunk_col": list(sp["trunk_col"]),
    }, open(json_path, "w"))
    print(f"    JSON {os.path.getsize(json_path) // 1024} KB")


if __name__ == "__main__":
    render("douglas_fir",    "/tmp/wp_douglas_fir.png",    seed=11)
    render("weeping_willow", "/tmp/wp_weeping_willow.png", seed=23)
    render("black_oak",      "/tmp/wp_black_oak.png",      seed=42)
    print("done.")
