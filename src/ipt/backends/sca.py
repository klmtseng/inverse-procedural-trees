"""SCA + post-process pipeline:
  1. Run space colonization (same as procgen_tree_sca.py)
  2. Walk each branch, smooth segments with Catmull-Rom interpolation so
     branches read as curves rather than polylines
  3. At each leaf node, place 25-50 small leaf cards in a small cluster
     for fluffy foliage
"""
import json, math, os, random
from ipt.render_util import render_tree

from ipt.paths import RESULTS as OUT_DIR


# ---- envelopes ----
def envelope_conifer(n, h_total=10.0, base_h=2.5, top_r=0.55, base_r=2.6, rng=None):
    rng = rng or random.Random()
    pts = []
    for _ in range(n):
        # bias toward upper half for narrow-conical fir
        u = rng.random() ** 0.55
        y = base_h + u * (h_total - base_h)
        t = (y - base_h) / max(1e-6, h_total - base_h)
        r_at = base_r * (1 - t) + top_r * t
        # narrow waist near the top for that classic fir shape
        r_at *= 1.0 - (t - 0.4) ** 2 * 0.4 if t > 0.4 else 1.0
        r = math.sqrt(rng.random()) * r_at
        a = rng.uniform(0, 2 * math.pi)
        pts.append((math.cos(a) * r, y, math.sin(a) * r))
    return pts


def envelope_oak(n, h_total=8.5, base_h=2.0, crown_r=3.6, rng=None):
    rng = rng or random.Random()
    pts = []
    crown_cy = (base_h + h_total) / 2
    crown_ry = (h_total - base_h) / 2 * 0.9
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        v = rng.uniform(-1, 1)
        rr = math.sqrt(1 - v * v)
        x = math.cos(a) * rr * crown_r
        z = math.sin(a) * rr * crown_r
        y = crown_cy + v * crown_ry
        f = rng.random() ** 0.4
        pts.append((x * f, crown_cy + (y - crown_cy) * f, z * f))
    return pts


def envelope_tiered_broadleaf(n, h_total=8.0, base_h=1.8,
                               n_tiers=3, tier_r=2.5,
                               tier_taper=0.8, tier_overlap=0.3, rng=None):
    """Mature broadleaf crown as N oblate bulges stacked vertically.

    Each tier i centred at base_h + (i+0.5)*tier_h with horizontal radius
    tier_r * taper**i and small vertical radius — reads as a horizontal
    'umbrella' layer. Stacking n_tiers of these gives the multi-tier
    silhouette of e.g. Phoenix tree / mature elm / pagoda dogwood.
    """
    rng = rng or random.Random()
    n_tiers = max(1, int(n_tiers))
    crown_h = max(0.1, h_total - base_h)
    eff = max(1.0, n_tiers - tier_overlap * (n_tiers - 1))
    tier_h = crown_h / eff
    per_tier = max(1, n // n_tiers)
    pts = []
    for i in range(n_tiers):
        cy = base_h + (i + 0.5) * tier_h * (1 - tier_overlap)
        if i == n_tiers - 1:
            cy = base_h + crown_h - tier_h * 0.5
        r_i = tier_r * (tier_taper ** i)
        ry_i = tier_h * 0.35
        for _ in range(per_tier):
            a = rng.uniform(0, 2 * math.pi)
            v = rng.uniform(-1, 1)
            rr = math.sqrt(1 - v * v)
            x = math.cos(a) * rr * r_i
            z = math.sin(a) * rr * r_i
            y = cy + v * ry_i
            f = rng.random() ** 0.5
            pts.append((x * f, cy + (y - cy) * f, z * f))
    return pts


def envelope_weeping(n, h_total=7.5, base_h=2.5, crown_r=3.0, rng=None):
    """Weeping willow: tear-drop envelope, denser at lower edge."""
    rng = rng or random.Random()
    pts = []
    for _ in range(n):
        # bias toward LOWER part of crown to attract drooping branches
        u = rng.random() ** 0.4
        y = base_h + u * (h_total - base_h)
        # radius bigger at the bottom, tapering toward top
        t = (y - base_h) / max(1e-6, h_total - base_h)
        r_at = crown_r * (1.0 - t * 0.4)
        r = math.sqrt(rng.random()) * r_at
        a = rng.uniform(0, 2 * math.pi)
        pts.append((math.cos(a) * r, y, math.sin(a) * r))
    return pts


def sca(envelope_pts, *, max_iter=200,
        influence_radius=2.0, kill_radius=0.50, step_size=0.40,
        trunk_height=2.5, jitter=0.07,
        gravity=0.0,                 # extra downward bias on growth dir
        rng=None):
    rng = rng or random.Random()
    nodes = [(0.0, 0.0, 0.0, -1)]
    while True:
        cx, cy, cz, _ = nodes[-1]
        any_close = any(
            (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2 < influence_radius ** 2
            for (px, py, pz) in envelope_pts
        )
        if any_close or cy >= trunk_height + 4.0:
            break
        nodes.append((cx, cy + step_size, cz, len(nodes) - 1))

    attractors = list(envelope_pts)

    for _ in range(max_iter):
        if not attractors: break
        node_atts = {}
        for a in attractors:
            best_d2 = influence_radius ** 2
            best_i = -1
            ax, ay, az = a
            for i, (nx, ny, nz, _) in enumerate(nodes):
                d2 = (nx - ax) ** 2 + (ny - ay) ** 2 + (nz - az) ** 2
                if d2 < best_d2:
                    best_d2 = d2; best_i = i
            if best_i >= 0:
                node_atts.setdefault(best_i, []).append(a)
        if not node_atts: break
        new_nodes = []
        for i, atts in node_atts.items():
            nx, ny, nz, _ = nodes[i]
            sx = sy = sz = 0.0
            for (ax, ay, az) in atts:
                dx, dy, dz = ax - nx, ay - ny, az - nz
                L = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
                sx += dx / L; sy += dy / L; sz += dz / L
            L = math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-9
            dx, dy, dz = sx / L, sy / L, sz / L
            dx += rng.uniform(-jitter, jitter)
            dy += rng.uniform(-jitter, jitter) - gravity   # drop bias
            dz += rng.uniform(-jitter, jitter)
            L = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
            dx /= L; dy /= L; dz /= L
            cx, cy, cz = nx + dx * step_size, ny + dy * step_size, nz + dz * step_size
            new_nodes.append((cx, cy, cz, i))
        nodes.extend(new_nodes)
        surviving = []
        for a in attractors:
            ax, ay, az = a
            alive = True
            for (nx, ny, nz, _) in nodes:
                if (nx - ax) ** 2 + (ny - ay) ** 2 + (nz - az) ** 2 < kill_radius ** 2:
                    alive = False; break
            if alive: surviving.append(a)
        attractors = surviving
    return nodes


def build_chain_index(nodes):
    """Map each node → its parent chain back to root, useful for smoothing."""
    n = len(nodes)
    children = [[] for _ in range(n)]
    for i, (_, _, _, p) in enumerate(nodes):
        if p >= 0:
            children[p].append(i)
    return children


def thicken(nodes, *, leaf_radius=0.03, branch_exp=2.3):
    n = len(nodes)
    children = build_chain_index(nodes)
    radius = [0.0] * n
    # post-order
    order = []
    stack = [0]
    in_progress = [False] * n
    while stack:
        v = stack[-1]
        if not in_progress[v]:
            in_progress[v] = True
            for c in children[v]: stack.append(c)
        else:
            stack.pop(); order.append(v)
    for v in order:
        if not children[v]:
            radius[v] = leaf_radius
        else:
            s = sum(radius[c] ** branch_exp for c in children[v])
            radius[v] = s ** (1.0 / branch_exp)
    return radius, children


def smooth_branches(nodes, children, *, smoothing=0.35):
    """Replace each node position with a Catmull-Rom-like blend of its
    parent + own + child positions. Iterate a few times for soft curves."""
    n = len(nodes)
    out = [list(nodes[i][:3]) + [nodes[i][3]] for i in range(n)]
    for _ in range(3):
        new_pos = [list(p[:3]) + [p[3]] for p in out]
        for i in range(1, n):
            p_idx = out[i][3]
            parent = out[p_idx]
            kids = children[i]
            if not kids: continue
            # only smooth interior nodes (not roots / not tips)
            kid = out[kids[0]]
            sx = (parent[0] + kid[0]) / 2
            sy = (parent[1] + kid[1]) / 2
            sz = (parent[2] + kid[2]) / 2
            new_pos[i][0] = out[i][0] * (1 - smoothing) + sx * smoothing
            new_pos[i][1] = out[i][1] * (1 - smoothing) + sy * smoothing
            new_pos[i][2] = out[i][2] * (1 - smoothing) + sz * smoothing
        out = new_pos
    return [(p[0], p[1], p[2], p[3]) for p in out]


def to_segments(nodes, radius):
    segs = []
    for i, (x, y, z, p) in enumerate(nodes):
        if p < 0: continue
        px, py, pz, _ = nodes[p]
        segs.append({
            "p0": [round(px, 3), round(py, 3), round(pz, 3)],
            "p1": [round(x, 3),  round(y, 3),  round(z, 3)],
            "r0": round(radius[p], 4), "r1": round(radius[i], 4),
        })
    return segs


def leaves_with_clusters(nodes, children, *, per_tip, cluster_r, leaf_size, base_col, rng):
    """For each tip (no children), drop N leaf cards in a sphere around it."""
    n = len(nodes)
    out = []
    for i in range(n):
        if children[i]:
            continue
        bx, by, bz, _ = nodes[i]
        for _ in range(per_tip):
            jx = rng.uniform(-1, 1); jy = rng.uniform(-1, 1); jz = rng.uniform(-1, 1)
            L = math.sqrt(jx * jx + jy * jy + jz * jz) + 1e-9
            d = rng.random() ** 0.65 * cluster_r
            p = (bx + jx / L * d, by + jy / L * d, bz + jz / L * d)
            jitter = rng.uniform(-15, 15)
            col = (
                max(0, min(255, base_col[0] + int(jitter))),
                max(0, min(255, base_col[1] + int(jitter * 0.7))),
                max(0, min(255, base_col[2] + int(jitter * 0.5))),
            )
            out.append({"p": [round(p[0], 3), round(p[1], 3), round(p[2], 3)],
                        "r": leaf_size, "col": list(col)})
    return out


# ---- presets ----
def make_fir(rng):
    env = envelope_conifer(280, h_total=10.5, base_h=2.5, top_r=0.5, base_r=2.4, rng=rng)
    nodes = sca(env, max_iter=200, influence_radius=2.0, kill_radius=0.5,
                step_size=0.38, trunk_height=2.5, jitter=0.08, rng=rng)
    radius, children = thicken(nodes, leaf_radius=0.03, branch_exp=2.3)
    nodes = smooth_branches(nodes, children, smoothing=0.30)
    return nodes, radius, children

def make_oak(rng):
    env = envelope_oak(260, h_total=8.5, base_h=2.2, crown_r=3.3, rng=rng)
    nodes = sca(env, max_iter=180, influence_radius=2.3, kill_radius=0.6,
                step_size=0.45, trunk_height=2.2, jitter=0.10, rng=rng)
    radius, children = thicken(nodes, leaf_radius=0.04, branch_exp=2.5)
    nodes = smooth_branches(nodes, children, smoothing=0.40)
    return nodes, radius, children

def make_willow(rng):
    env = envelope_weeping(280, h_total=7.5, base_h=2.5, crown_r=3.2, rng=rng)
    nodes = sca(env, max_iter=180, influence_radius=2.2, kill_radius=0.5,
                step_size=0.40, trunk_height=2.0, jitter=0.10,
                gravity=0.22,           # the weeping bias — branches sag
                rng=rng)
    radius, children = thicken(nodes, leaf_radius=0.025, branch_exp=2.4)
    nodes = smooth_branches(nodes, children, smoothing=0.35)
    return nodes, radius, children


def render(name, builder, leaf_cfg, out_path, *, seed=7):
    rng = random.Random(seed)
    nodes, radius, children = builder(rng)
    segs = to_segments(nodes, radius)
    leaves = leaves_with_clusters(nodes, children,
                                  per_tip=leaf_cfg["per_tip"],
                                  cluster_r=leaf_cfg["cluster_r"],
                                  leaf_size=leaf_cfg["leaf_size"],
                                  base_col=leaf_cfg["col"], rng=rng)
    print(f"  {name}: nodes={len(nodes)}, segs={len(segs)}, leaves={len(leaves)}")
    render_tree(segs, leaves, out_path,
                title=f"SCA+post — {name}",
                trunk_col=leaf_cfg["trunk_col"],
                leaf_col=leaf_cfg["col"], leaf_radius=3)
    json_path = os.path.join(OUT_DIR, f"sca2_tree_{name}.json")
    json.dump({"kind": name, "segments": segs, "leaves": leaves},
              open(json_path, "w"))
    print(f"    JSON {os.path.getsize(json_path) // 1024} KB")


if __name__ == "__main__":
    render("fir",    make_fir,    dict(per_tip=20, cluster_r=0.30, leaf_size=0.05,
                                        col=(35, 92, 50), trunk_col=(60, 38, 22)),
           "/tmp/sca2_fir.png", seed=5)
    render("oak",    make_oak,    dict(per_tip=40, cluster_r=0.55, leaf_size=0.10,
                                        col=(65, 110, 45), trunk_col=(60, 38, 28)),
           "/tmp/sca2_oak.png", seed=15)
    render("willow", make_willow, dict(per_tip=22, cluster_r=0.35, leaf_size=0.05,
                                        col=(140, 155, 75), trunk_col=(55, 38, 28)),
           "/tmp/sca2_willow.png", seed=25)
    print("done.")
