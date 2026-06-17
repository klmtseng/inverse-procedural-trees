"""Weber-Penn backend adapter.

Bridges wp_raw.build_tree (which takes a species name + rng) to the flat
parameter-vector interface the DE optimiser uses. Output shape matches
ipt.backends.sca.build_sca2_shape's:
    return segs, tip_positions, scene_w, scene_h, cluster_r
so the rasteriser code path is backend-agnostic.
"""
import math
import random

from ipt.backends.wp_raw import build_tree as _wp_build_tree
from ipt.backends import wp_raw as _wp


# Per-species WP search spaces — bounds reflect the biology each species
# actually exhibits (pine: branches roughly horizontal-down with dense
# needles; broadleaf: main scaffolds arc up + wide twig spread).
WP_BOUNDS = {
    "pine": [
        ("wp_scale",          8.0, 13.0),
        ("wp_ratio",          0.010, 0.022),
        ("wp_baseSize",       0.30, 0.55),
        # trunk
        ("wp0_curve",         0.0, 12.0),
        ("wp0_segs",          8.0, 14.0),
        ("wp0_branches",     20.0, 45.0),
        ("wp0_taper",         0.90, 1.0),
        # main scaffolds: pine branches go DOWN or horizontal-down
        ("wp1_L",             0.30, 0.55),
        ("wp1_curve",       -25.0, -5.0),    # mild downward curve
        ("wp1_segs",          4.0,  8.0),
        ("wp1_branches",     20.0, 35.0),
        ("wp1_down",         60.0, 85.0),    # pine branches near horizontal
        ("wp1_taper",         0.5,  0.75),
        # twigs
        ("wp2_L",             0.20, 0.35),
        ("wp2_branches",      8.0, 16.0),
        ("wp2_down",         45.0, 75.0),
        # foliage — needles dense + small
        ("wp_leaf_per_tip",  30.0, 60.0),
        ("wp_leaf_cluster_r", 0.15, 0.32),
        ("wp_leaf_size",      0.04, 0.08),
    ],
    "broadleaf": [
        ("wp_scale",          6.0, 11.0),
        ("wp_ratio",          0.022, 0.045),
        ("wp_baseSize",       0.15, 0.40),
        ("wp0_curve",         0.0, 20.0),
        ("wp0_segs",          6.0, 14.0),
        ("wp0_branches",     10.0, 20.0),
        ("wp0_taper",         0.85, 1.0),
        ("wp1_L",             0.55, 1.05),
        ("wp1_curve",       -60.0, -10.0),   # arc up = negative curve
        ("wp1_segs",          4.0,  8.0),
        ("wp1_branches",      6.0, 14.0),
        ("wp1_down",         35.0, 75.0),
        ("wp1_taper",         0.4,  0.75),
        ("wp2_L",             0.30, 0.55),
        ("wp2_branches",      4.0, 10.0),
        ("wp2_down",         30.0, 70.0),
        ("wp_leaf_per_tip",  20.0, 50.0),
        ("wp_leaf_cluster_r", 0.20, 0.55),
        ("wp_leaf_size",      0.04, 0.10),
    ],
}


def get_wp_spec(species):
    return WP_BOUNDS[species]


# back-compat default (broadleaf — the path we've validated)
WP_PARAM_SPEC = WP_BOUNDS["broadleaf"]


def _build_wp_species_dict(p, base_color):
    """Make a SPECIES-shaped dict from a flat param dict.

    Reuses douglas_fir's variance knobs (curveV, downV, rot, rotV, lengthV)
    as constants since they only widen distributions and don't move the
    objective much.
    """
    return {
        "scale":     p["wp_scale"],
        "ratio":     p["wp_ratio"],
        "baseSize":  p["wp_baseSize"],
        "levels":    3,
        "lvl": [
            dict(L=1.0,
                 curve=p["wp0_curve"],   curveV=10,
                 segs=int(round(p["wp0_segs"])),
                 branches=int(round(p["wp0_branches"])),
                 down=70, downV=10,
                 rot=137, rotV=15,
                 taper=p["wp0_taper"],
                 lengthV=0.0),
            dict(L=p["wp1_L"],
                 curve=p["wp1_curve"],  curveV=25,
                 segs=int(round(p["wp1_segs"])),
                 branches=int(round(p["wp1_branches"])),
                 down=p["wp1_down"], downV=15,
                 rot=137, rotV=30,
                 taper=p["wp1_taper"],
                 lengthV=0.20),
            dict(L=p["wp2_L"],
                 curve=-15, curveV=30,
                 segs=4,
                 branches=int(round(p["wp2_branches"])),
                 down=p["wp2_down"], downV=25,
                 rot=137, rotV=50,
                 taper=0.6,
                 lengthV=0.35),
        ],
        "trunk_col": (60, 40, 30),
        "leaf_col":  base_color,
        "leaf_per_tip":     int(round(p["wp_leaf_per_tip"])),
        "leaf_cluster_r":   p["wp_leaf_cluster_r"],
        "leaf_size":        p["wp_leaf_size"],
    }


def build_wp_shape(params, rng=None, base_color=(70, 100, 40)):
    """Run Weber-Penn build_tree with parameter overrides, return v5b-bot
    standard shape tuple (segs, tip_positions, scene_w, scene_h, cluster_r).
    """
    rng = rng or random.Random()
    sp = _build_wp_species_dict(params, base_color)

    # Patch WP's SPECIES dict with an ephemeral entry; restore after.
    SLOT = "_wp_adapter_tmp"
    _wp.SPECIES[SLOT] = sp
    try:
        segs, leaves, _ = _wp_build_tree(SLOT, rng=rng)
    finally:
        _wp.SPECIES.pop(SLOT, None)

    # Reshape WP segs (p0/p1 are 3-element lists) into the form the v4
    # rasteriser expects: p0/p1 as tuples is fine.
    segs_out = [{
        "p0": tuple(s["p0"]),
        "p1": tuple(s["p1"]),
        "r0": s["r0"], "r1": s["r1"],
    } for s in segs]

    # tip positions: cluster the WP leaf positions into a smaller set so
    # rasterise_proc_v4's per-tip leaf spam doesn't explode. WP already
    # places leaves at deepest-level branch tips, so we take their world
    # positions and downsample.
    tip_positions = [tuple(l["p"]) for l in leaves]
    if len(tip_positions) > 200:
        stride = len(tip_positions) // 200 + 1
        tip_positions = tip_positions[::stride]

    # scene dims
    if leaves:
        xs = [l["p"][0] for l in leaves]
        zs = [l["p"][2] for l in leaves]
        ys = [l["p"][1] for l in leaves]
        scene_w = max(2.0, max(max(xs) - min(xs), max(zs) - min(zs)) * 1.2)
        scene_h = max(2.0, max(ys) * 1.1)
    else:
        scene_w = params["wp_scale"] * 0.8
        scene_h = params["wp_scale"] * 1.1

    cluster_r = params["wp_leaf_cluster_r"]
    return segs_out, tip_positions, scene_w, scene_h, cluster_r
