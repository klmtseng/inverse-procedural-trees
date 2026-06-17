"""Inverse procedural tree fitting v3.

Changes vs v2 (`procgen_tree_fit_v2.py`):

  - 8 dist-to-branch bins instead of 4
  - Per-bin colour stored as (mean, std) → leaves sample from
    Gaussian(mean, std) per channel → smooth continuous variation
  - Leaves are rendered as small rotated ELLIPSES (3:1 aspect ratio,
    needle-like for conifers) instead of solid dots; orientation
    randomised per leaf around each cluster centre
  - Objective uses per-bin (mean, std) match instead of just mean
"""
import os, sys, math, json, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import differential_evolution

from ipt.backends.sca import (envelope_conifer, sca, thicken,
                               smooth_branches, to_segments)
from ipt.render_util import render_tree

from ipt.paths import PHOTOS, RESULTS as DATA_DIR


# -------- v3 constants --------
N_DIST_BINS = 8
DIST_BIN_MAX_PX = 60
BRIGHTNESS_THRESHOLDS = [30, 45, 60]


def bucket_brightness(v):
    if v < BRIGHTNESS_THRESHOLDS[0]: return 0
    if v < BRIGHTNESS_THRESHOLDS[1]: return 1
    if v < BRIGHTNESS_THRESHOLDS[2]: return 2
    return 3


# ============================================================
#  Photo analysis: continuous Gaussian per dist bin
# ============================================================
def extract_spatial_model(photo_path, bbox):
    img = Image.open(photo_path).convert("RGBA")
    arr = np.array(img)
    x0, y0, x1, y1 = bbox
    sub_rgb = arr[y0:y1, x0:x1, :3]
    sub_a   = arr[y0:y1, x0:x1, 3]
    sub_op  = sub_a > 100
    sub_lum = sub_rgb.mean(axis=2)
    SH, SW  = sub_rgb.shape[:2]

    branch_mask = sub_op & (sub_lum < 25)
    leaf_mask = sub_op & ~branch_mask
    dist = distance_transform_edt(~branch_mask)
    dist[~sub_op] = 0

    # per-bin (mean, std) RGB
    edges = np.linspace(0, DIST_BIN_MAX_PX, N_DIST_BINS + 1)
    mean_per_bin = np.zeros((N_DIST_BINS, 3), dtype=np.float32)
    std_per_bin  = np.zeros((N_DIST_BINS, 3), dtype=np.float32)
    pop_per_bin  = np.zeros(N_DIST_BINS, dtype=np.int32)
    overall_mean = sub_rgb[leaf_mask].mean(axis=0)
    overall_std = sub_rgb[leaf_mask].std(axis=0)
    for i in range(N_DIST_BINS):
        lo, hi = edges[i], edges[i + 1]
        in_bin = leaf_mask & (dist >= lo) & (dist < hi)
        n = int(in_bin.sum())
        pop_per_bin[i] = n
        if n < 20:
            mean_per_bin[i] = overall_mean
            std_per_bin[i]  = overall_std
        else:
            mean_per_bin[i] = sub_rgb[in_bin].mean(axis=0)
            std_per_bin[i]  = sub_rgb[in_bin].std(axis=0)

    # conditional brightness matrix (kept from v2)
    lum_blur = uniform_filter(sub_lum, size=9)
    own_b = np.array([bucket_brightness(v) for v in sub_lum[leaf_mask]])
    nbr_b = np.array([bucket_brightness(v) for v in lum_blur[leaf_mask]])
    joint = np.zeros((4, 4), dtype=np.float64)
    for o, n_ in zip(own_b, nbr_b):
        joint[n_, o] += 1
    cond = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1)

    # marginal hist (kept)
    op_pixels = sub_rgb[sub_op]
    hist = np.zeros((3, 32), dtype=np.float64)
    for c in range(3):
        h, _ = np.histogram(op_pixels[:, c], bins=32, range=(0, 256))
        s = h.sum()
        if s > 0: h = h / s
        hist[c] = h

    return dict(
        bbox=bbox, mask_bbox_shape=(SH, SW),
        bin_edges=edges, mean_per_bin=mean_per_bin,
        std_per_bin=std_per_bin, pop_per_bin=pop_per_bin,
        cond_matrix=cond, marginal_hist=hist,
    )


# ============================================================
#  Target extraction (same as v2)
# ============================================================
def segment_trees(op, min_w=40, gap_merge=8):
    col_op = op.sum(axis=0).astype(int)
    thr = max(20, int(col_op.max() * 0.10))
    runs = []; in_run = False; start = 0
    for x in range(op.shape[1]):
        if col_op[x] > thr:
            if not in_run: start = x; in_run = True
        elif in_run:
            runs.append((start, x)); in_run = False
    if in_run: runs.append((start, op.shape[1]))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < gap_merge:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [r for r in merged if r[1] - r[0] > min_w]


def extract_target(photo_path, tree_idx, mask_res=128):
    img = Image.open(photo_path).convert("RGBA")
    arr = np.array(img)
    op = arr[:, :, 3] > 100
    runs = segment_trees(op)
    x0, x1 = runs[tree_idx]
    sub_op = op[:, x0:x1]
    rows = np.where(sub_op.sum(axis=1) > max(2, sub_op.shape[1] * 0.05))[0]
    y0, y1 = int(rows.min()), int(rows.max())
    crop_op = op[y0:y1 + 1, x0:x1]
    mask_full = crop_op.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_full).resize((mask_res, mask_res), Image.LANCZOS)
    target_mask = (np.array(mask_img) > 127).astype(np.uint8)
    bbox = (x0, y0, x1, y1)
    spatial = extract_spatial_model(photo_path, bbox)
    crop_pil = Image.fromarray(arr[y0:y1 + 1, x0:x1])
    print(f"  target bbox=({x0},{y0},{x1},{y1})")
    print(f"  per-bin populations: {spatial['pop_per_bin'].tolist()}")
    print(f"  per-bin mean RGB[0]: {spatial['mean_per_bin'][0].astype(int).tolist()}, [-1]: {spatial['mean_per_bin'][-1].astype(int).tolist()}")
    return dict(photo_path=photo_path, tree_idx=tree_idx, bbox=bbox,
                target_mask=target_mask, crop_pil=crop_pil,
                mask_res=mask_res, spatial=spatial)


# ============================================================
#  Param space (10D = 9 shape + 1 noise std scaler)
# ============================================================
PARAM_SPEC = [
    ("influence_radius",  0.8,   3.2),
    ("kill_radius_frac",  0.20,  0.55),
    ("step_size_frac",    0.15,  0.32),
    ("jitter",            0.03,  0.18),
    ("env_h_total",       5.0,  13.0),
    ("env_base_h_frac",   0.18,  0.45),
    ("env_top_r",         0.30,  1.80),
    ("env_base_r",        1.20,  3.80),
    ("leaf_cluster_r",    0.12,  0.55),
    ("color_noise_scale", 0.4,   1.6),   # multiplier on per-bin std
]


def params_to_dict(vec):
    return {n: float(v) for (n, _, _), v in zip(PARAM_SPEC, vec)}


# ============================================================
#  Shape builder (same as v2)
# ============================================================
def build_sca2_shape(params, rng):
    inf_r = params["influence_radius"]
    kill_r = inf_r * params["kill_radius_frac"]
    step = inf_r * params["step_size_frac"]
    jit = params["jitter"]
    h_total = params["env_h_total"]
    base_h = h_total * params["env_base_h_frac"]
    top_r = params["env_top_r"]
    base_r = params["env_base_r"]
    cluster_r = params["leaf_cluster_r"]

    env = envelope_conifer(150, h_total=h_total, base_h=base_h,
                            top_r=top_r, base_r=base_r, rng=rng)
    nodes = sca(env, max_iter=60,
                influence_radius=inf_r, kill_radius=kill_r,
                step_size=step, trunk_height=base_h * 0.9,
                jitter=jit, rng=rng)
    if len(nodes) > 500:
        nodes = nodes[:500]
    radius, children = thicken(nodes, leaf_radius=0.025, branch_exp=2.3)
    nodes = smooth_branches(nodes, children, smoothing=0.30)
    segs = to_segments(nodes, radius)
    tip_positions = [(n[0], n[1], n[2]) for i, n in enumerate(nodes)
                     if not children[i]]
    scene_w = max(base_r, top_r) * 2.4
    scene_h = h_total * 1.1
    return segs, tip_positions, scene_w, scene_h, cluster_r


# ============================================================
#  v3 rasterizer with needle-shaped leaves + Gaussian colour
# ============================================================
def rasterize_proc_v3(segs, tip_positions, cluster_r, scene_w, scene_h,
                       spatial, params, mask_res=128, per_tip=14,
                       leaf_aspect=2.6, leaf_long_frac=0.06, rng=None):
    """Same plumbing as v2 but leaves drawn as rotated ellipses (long axis
    = leaf_long_frac of scene_h, aspect = leaf_aspect) and colours sampled
    from per-bin Gaussian."""
    rng = rng or random.Random()
    img       = Image.new("L", (mask_res, mask_res), 0)
    branch_im = Image.new("L", (mask_res, mask_res), 0)
    leaf_im   = Image.new("L", (mask_res, mask_res), 0)
    draw   = ImageDraw.Draw(img)
    bdraw  = ImageDraw.Draw(branch_im)
    ldraw  = ImageDraw.Draw(leaf_im)
    px_per_unit_x = mask_res / scene_w
    px_per_unit_y = mask_res / scene_h
    def proj(p):
        return (mask_res / 2 + p[0] * px_per_unit_x,
                mask_res - 1 - p[1] * px_per_unit_y)

    # branches — stamp circles at endpoints so joints don't show gaps
    for s in segs:
        x0, y0 = proj(s["p0"]); x1, y1 = proj(s["p1"])
        r0_px = max(1, s["r0"] * px_per_unit_x)
        r1_px = max(1, s["r1"] * px_per_unit_x)
        w = max(1, int(r0_px + r1_px))
        for d in (bdraw, draw):
            d.line([(x0, y0), (x1, y1)], fill=255, width=w)
            d.ellipse([(x0 - r0_px, y0 - r0_px), (x0 + r0_px, y0 + r0_px)], fill=255)
            d.ellipse([(x1 - r1_px, y1 - r1_px), (x1 + r1_px, y1 + r1_px)], fill=255)

    # leaf positions
    leaf_positions = []
    for tip in tip_positions:
        for _ in range(per_tip):
            jx = rng.uniform(-1, 1); jy = rng.uniform(-1, 1); jz = rng.uniform(-1, 1)
            L = math.sqrt(jx ** 2 + jy ** 2 + jz ** 2) + 1e-9
            d = rng.random() ** 0.65 * cluster_r
            wp = (tip[0] + jx / L * d, tip[1] + jy / L * d, tip[2] + jz / L * d)
            px, py = proj(wp)
            leaf_positions.append((px, py))

    # leaf needle size in px
    long_px  = max(3, int(leaf_long_frac * scene_h * px_per_unit_y))
    short_px = max(1, int(long_px / leaf_aspect))

    # rasterize leaves as needles
    for (px, py) in leaf_positions:
        # rotated ellipse: pre-compute corner positions for a rotated
        # rectangle and use draw.polygon (PIL has no rotated-ellipse)
        ang = rng.uniform(0, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        # 4 corners of needle bounding box
        hx = long_px / 2; hy = short_px / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        pts = []
        for cx, cy in corners:
            rx = cx * ca - cy * sa
            ry = cx * sa + cy * ca
            pts.append((px + rx, py + ry))
        ldraw.polygon(pts, fill=255)
        draw.polygon(pts, fill=255)

    # distance-to-branch per leaf pixel (proc-tree pixel space)
    branch_arr = np.array(branch_im) > 127
    if not branch_arr.any():
        leaf_colors = np.tile(spatial["mean_per_bin"].mean(axis=0),
                              (len(leaf_positions), 1)).astype(np.int32)
        return ((np.array(img) > 127).astype(np.uint8),
                branch_arr, np.array(leaf_im) > 127, leaf_colors,
                leaf_positions, long_px, short_px)
    dt = distance_transform_edt(~branch_arr)
    photo_h_px = spatial["bbox"][3] - spatial["bbox"][1]
    scale_factor = photo_h_px / mask_res
    noise_scale = params.get("color_noise_scale", 1.0)
    bin_edges = spatial["bin_edges"]
    leaf_colors = []
    for (px, py) in leaf_positions:
        ix = int(np.clip(px, 0, mask_res - 1))
        iy = int(np.clip(py, 0, mask_res - 1))
        d_photo = dt[iy, ix] * scale_factor
        bin_i = N_DIST_BINS - 1
        for i in range(N_DIST_BINS):
            if d_photo >= bin_edges[i] and d_photo < bin_edges[i + 1]:
                bin_i = i; break
        mean = spatial["mean_per_bin"][bin_i]
        std  = spatial["std_per_bin"][bin_i] * noise_scale
        col = np.clip(np.array([
            rng.gauss(mean[0], std[0]),
            rng.gauss(mean[1], std[1]),
            rng.gauss(mean[2], std[2]),
        ]), 0, 255).astype(np.int32)
        leaf_colors.append(col)
    leaf_colors = np.array(leaf_colors)
    return ((np.array(img) > 127).astype(np.uint8),
            branch_arr, np.array(leaf_im) > 127, leaf_colors,
            leaf_positions, long_px, short_px)


# ============================================================
#  Objective (same structure as v2; bin sig now 8 bins)
# ============================================================
def compute_proc_spatial_signature(branch_arr, leaf_arr, spatial):
    """Mean RGB per dist bin in proc-tree 2D space. Returns (mean[8,3], cond[4,4])."""
    if not branch_arr.any() or not leaf_arr.any():
        return (np.zeros((N_DIST_BINS, 3)), np.eye(4) / 4)
    dt = distance_transform_edt(~branch_arr)
    photo_h_px = spatial["bbox"][3] - spatial["bbox"][1]
    scale_factor = photo_h_px / branch_arr.shape[0]
    bin_edges = spatial["bin_edges"]
    # we render the proc-tree's "colour image" by stamping the photo-mean
    # of the bin each leaf pixel falls in (consistent with how leaves are
    # sampled — we use the BIN mean, not the random Gaussian sample)
    H, W = leaf_arr.shape
    proc_col_img = np.zeros((H, W, 3), dtype=np.float32)
    proc_mean_per_bin = np.zeros((N_DIST_BINS, 3), dtype=np.float32)
    for i in range(N_DIST_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = leaf_arr & (dt * scale_factor >= lo) & (dt * scale_factor < hi)
        if m.sum() > 5:
            proc_col_img[m] = spatial["mean_per_bin"][i]
            proc_mean_per_bin[i] = spatial["mean_per_bin"][i]
        else:
            proc_mean_per_bin[i] = spatial["mean_per_bin"][i]
    proc_lum = proc_col_img.mean(axis=2)
    proc_lum_blur = uniform_filter(proc_lum, size=9)
    own_lum = proc_lum[leaf_arr]
    nbr_lum = proc_lum_blur[leaf_arr]
    if len(own_lum) == 0:
        return (proc_mean_per_bin, np.eye(4) / 4)
    own_b = np.array([bucket_brightness(v) for v in own_lum])
    nbr_b = np.array([bucket_brightness(v) for v in nbr_lum])
    joint = np.zeros((4, 4), dtype=np.float64)
    for o, n_ in zip(own_b, nbr_b):
        joint[n_, o] += 1
    cond = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1)
    return proc_mean_per_bin, cond


def objective_v3(target, proc_mask, branch_arr, leaf_arr, leaf_colors, *,
                  w_sil=1.0, w_col=0.4, w_spat=0.6, w_cond=0.4):
    spatial = target["spatial"]
    t = target["target_mask"]; p = proc_mask
    iou = (t & p).sum() / max((t | p).sum(), 1)
    sil_loss = 1.0 - iou

    if len(leaf_colors) > 0:
        p_hist = np.zeros((3, 32), dtype=np.float64)
        for c in range(3):
            h, _ = np.histogram(leaf_colors[:, c], bins=32, range=(0, 256))
            s = h.sum()
            if s > 0: h = h / s
            p_hist[c] = h
        eps = 1e-6
        col_loss = (((spatial["marginal_hist"] - p_hist) ** 2)
                    / (spatial["marginal_hist"] + p_hist + eps)).sum() / 3
    else:
        col_loss = 2.0

    proc_mean, proc_cond = compute_proc_spatial_signature(branch_arr, leaf_arr, spatial)
    # weight per-bin difference by population (smaller bins matter less)
    pop = spatial["pop_per_bin"].astype(np.float32) + 1
    pop_norm = pop / pop.sum()
    bin_diffs = np.linalg.norm(proc_mean - spatial["mean_per_bin"], axis=1) / 255
    spat_loss = (pop_norm * bin_diffs).sum()

    eps = 1e-4
    t_cond = spatial["cond_matrix"]
    kl = (t_cond * (np.log(t_cond + eps) - np.log(proc_cond + eps))).sum() / 4
    cond_loss = abs(kl)

    if p.sum() < 50:
        deg = 2.0
    else:
        deg = 0.0

    score = (w_sil * sil_loss + w_col * col_loss
             + w_spat * spat_loss + w_cond * cond_loss + deg)
    sub = dict(sil=sil_loss, col=col_loss, spat=spat_loss,
                cond=cond_loss, iou=iou)
    return score, sub


# ============================================================
#  Fit driver (DE)
# ============================================================
def fit_de_v3(target, max_iter=10, popsize=3, seed=7):
    bounds = [(lo, hi) for (_, lo, hi) in PARAM_SPEC]
    trace = []; counter = {"n": 0}
    def fn(vec):
        params = params_to_dict(vec)
        rng = random.Random(seed)
        segs, tips, sw, sh, cr = build_sca2_shape(params, rng)
        mask, b_arr, l_arr, l_cols, _, _, _ = rasterize_proc_v3(
            segs, tips, cr, sw, sh, target["spatial"], params,
            mask_res=target["mask_res"], rng=rng)
        score, sub = objective_v3(target, mask, b_arr, l_arr, l_cols)
        counter["n"] += 1
        trace.append((counter["n"], score, sub))
        return score
    t0 = time.time()
    result = differential_evolution(fn, bounds, maxiter=max_iter,
                                     popsize=popsize, seed=seed,
                                     polish=False, tol=1e-4,
                                     mutation=(0.5, 1.0), recombination=0.7,
                                     init='latinhypercube')
    dt = time.time() - t0
    print(f"  DE v3 finished: {counter['n']} evals in {dt:.1f}s, best={result.fun:.4f}")
    return result.x, result.fun, trace


# ============================================================
#  Human-friendly preview with needle leaves at HIGHER resolution
# ============================================================
def render_preview_needles(target, vec, out_path, *, seed=7,
                            preview_res=560, preview_aspect=1.4):
    """Render the proc tree at preview resolution with needle leaves,
    using each leaf's individually-sampled Gaussian colour."""
    params = params_to_dict(vec)
    rng = random.Random(seed)
    segs, tips, sw, sh, cr = build_sca2_shape(params, rng)
    # rasterize at preview_res for nicer output
    rng2 = random.Random(seed)   # same seed → same leaf positions
    mask, b_arr, l_arr, l_cols, leaf_pos, long_px, short_px = rasterize_proc_v3(
        segs, tips, cr, sw, sh, target["spatial"], params,
        mask_res=preview_res, rng=rng2,
        leaf_long_frac=0.025)    # smaller, more needle-y

    H = preview_res
    W = int(H * preview_aspect)
    img = Image.new("RGB", (W, H), (250, 250, 245))
    draw = ImageDraw.Draw(img)

    px_per_unit_x = preview_res / sw
    px_per_unit_y = preview_res / sh
    cx_off = (W - preview_res) // 2
    def proj(p):
        return (cx_off + preview_res / 2 + p[0] * px_per_unit_x,
                preview_res - 1 - p[1] * px_per_unit_y)

    # branches as bark-brown lines + endpoint circles (no gaps at joints)
    for s in segs:
        x0, y0 = proj(s["p0"]); x1, y1 = proj(s["p1"])
        r0_px = max(1, s["r0"] * px_per_unit_x)
        r1_px = max(1, s["r1"] * px_per_unit_x)
        w = max(1, int(r0_px + r1_px))
        draw.line([(x0, y0), (x1, y1)], fill=(58, 38, 22), width=w)
        draw.ellipse([(x0 - r0_px, y0 - r0_px), (x0 + r0_px, y0 + r0_px)],
                     fill=(58, 38, 22))
        draw.ellipse([(x1 - r1_px, y1 - r1_px), (x1 + r1_px, y1 + r1_px)],
                     fill=(58, 38, 22))

    # render each leaf as a rotated needle ellipse, using ITS colour
    rng3 = random.Random(seed)
    leaf_pos_world = []
    for tip in tips:
        for _ in range(14):
            jx = rng3.uniform(-1, 1); jy = rng3.uniform(-1, 1); jz = rng3.uniform(-1, 1)
            L = math.sqrt(jx ** 2 + jy ** 2 + jz ** 2) + 1e-9
            d = rng3.random() ** 0.65 * cr
            wp = (tip[0] + jx / L * d, tip[1] + jy / L * d, tip[2] + jz / L * d)
            leaf_pos_world.append(wp)
    # depth-sort by Y so back leaves render first (cheap painters' algo)
    sorted_idx = sorted(range(len(leaf_pos_world)),
                        key=lambda i: leaf_pos_world[i][2])
    for i in sorted_idx:
        if i >= len(l_cols): continue
        col = tuple(int(v) for v in l_cols[i])
        wp = leaf_pos_world[i]
        px, py = proj(wp)
        ang = (i * 137 + (i % 7) * 23) % 180 / 180.0 * math.pi
        ca, sa = math.cos(ang), math.sin(ang)
        hx = long_px / 2; hy = short_px / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        pts = []
        for cx, cy in corners:
            rx = cx * ca - cy * sa
            ry = cx * sa + cy * ca
            pts.append((px + rx, py + ry))
        draw.polygon(pts, fill=col)
    img.save(out_path, "PNG", optimize=True)


# ============================================================
#  CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo",    default="tree_pine.png")
    ap.add_argument("--tree-idx", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--seed",     type=int, default=7)
    args = ap.parse_args()

    photo_path = os.path.join(PHOTOS, args.photo)
    target = extract_target(photo_path, args.tree_idx)

    # baseline
    init_vec = [(lo + hi) / 2 for (_, lo, hi) in PARAM_SPEC]
    init_p = params_to_dict(init_vec)
    rng = random.Random(args.seed)
    segs, tips, sw, sh, cr = build_sca2_shape(init_p, rng)
    mask, b_arr, l_arr, l_cols, _, _, _ = rasterize_proc_v3(
        segs, tips, cr, sw, sh, target["spatial"], init_p,
        mask_res=target["mask_res"], rng=rng)
    init_score, init_sub = objective_v3(target, mask, b_arr, l_arr, l_cols)
    print(f"  v3 INIT score = {init_score:.4f} {init_sub}")

    best_vec, best_score, trace = fit_de_v3(target, max_iter=args.max_iter,
                                              seed=args.seed)
    best_p = params_to_dict(best_vec)
    rng = random.Random(args.seed)
    segs, tips, sw, sh, cr = build_sca2_shape(best_p, rng)
    mask, b_arr, l_arr, l_cols, _, _, _ = rasterize_proc_v3(
        segs, tips, cr, sw, sh, target["spatial"], best_p,
        mask_res=target["mask_res"], rng=rng)
    final_score, final_sub = objective_v3(target, mask, b_arr, l_arr, l_cols)
    print(f"  v3 BEST  score = {final_score:.4f} {final_sub}")
    print(f"  v3 improvement: {(1 - final_score / max(init_score, 1e-9)) * 100:.1f}%")

    tag = f"v3_sca2_de_{os.path.splitext(args.photo)[0]}_{args.tree_idx}"
    proc_path = f"/tmp/_v3_proc.png"
    render_preview_needles(target, best_vec, proc_path, seed=args.seed,
                            preview_res=560)

    # composite: original | v2 best | v3 best
    orig = target["crop_pil"].convert("RGB")
    H = 600
    imgs = [("original", orig)]
    if os.path.exists("/tmp/_v2_proc.png"):
        imgs.append(("v2 best (dots)",
                     Image.open("/tmp/_v2_proc.png").convert("RGB")))
    imgs.append((f"v3 best (needles, {final_score:.3f})",
                  Image.open(proc_path).convert("RGB")))
    scaled = [(t, im.resize((int(im.size[0] * H / im.size[1]), H), Image.LANCZOS))
              for t, im in imgs]
    W = sum(im.size[0] for _, im in scaled) + 16 * (len(scaled) + 1)
    canvas = Image.new("RGB", (W, H + 80), (240, 240, 235))
    draw = ImageDraw.Draw(canvas)
    title = (f"v3 spatial+Gaussian+needles  {args.photo}#{args.tree_idx}   "
             f"init→best  {init_score:.4f}→{final_score:.4f}   "
             f"sil={final_sub['sil']:.3f} col={final_sub['col']:.3f} "
             f"spat={final_sub['spat']:.4f} cond={final_sub['cond']:.3f}")
    draw.text((10, 8), title, fill=(15, 15, 15))
    x = 10
    for t, im in scaled:
        canvas.paste(im, (x, 50))
        draw.text((x + 4, 30), t, fill=(15, 15, 15))
        x += im.size[0] + 16
    out = f"/tmp/fit_{tag}.png"
    canvas.save(out, "PNG", optimize=True)
    print(f"  wrote {out}")

    json.dump({
        "tag": tag, "photo": args.photo, "tree_idx": args.tree_idx,
        "init_score": float(init_score), "final_score": float(final_score),
        "improvement_pct": (1 - final_score / max(init_score, 1e-9)) * 100,
        "best_params": best_p,
        "sub_scores": {k: float(v) for k, v in final_sub.items()},
        "n_evals": len(trace),
    }, open(os.path.join(DATA_DIR, f"fit_{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
