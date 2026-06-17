"""Inverse procedural tree fitting v5 — SSIM + Lab + (later) VGG/CLIP.

Changes vs v4 (procgen_tree_fit_v4.py):
  - v5a: 2 new objective terms — 1-SSIM on grayscale + Lab-space percentile
         L1. Captures perceptual structure + colour-space distance the RGB
         chi-squared misses.
  - v5b: optional --backend wp (Weber-Penn) via wp_adapter
  - v5c: optional VGG perceptual loss (torch lazy import)
  - v5d: optional CLIP semantic loss (open_clip lazy import)

All new terms can be zeroed via weight params; v5 reproduces v4 when all
weights are 0.
"""
import os, sys, math, json, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageChops
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import differential_evolution

# v5a perceptual deps
try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.color import rgb2lab
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
# v5c torch — gated so v5a/b can run without it
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
# v5d CLIP — gated similarly
try:
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

from ipt.backends.sca import (envelope_conifer, envelope_tiered_broadleaf,
                               sca, thicken, smooth_branches, to_segments)
from ipt.backends.wp import WP_PARAM_SPEC, get_wp_spec, build_wp_shape

# module-level backend selector; set in main() from --backend
BACKEND = "sca"


def build_shape(params, rng):
    """Dispatch to SCA2 or Weber-Penn based on BACKEND."""
    if BACKEND == "wp":
        # populate the v4 rasteriser's expected keys with WP-friendly
        # defaults so it doesn't re-render along-twig leaves on top of
        # WP's own per-tip leaves.
        params.setdefault("leaves_per_tip",       params.get("wp_leaf_per_tip", 30))
        params.setdefault("leaf_scale",           1.0)
        params.setdefault("n_secondary",          1)
        params.setdefault("secondary_radius",     0.0)
        params.setdefault("trunk_thickness_scale",1.0)
        params.setdefault("leaves_per_unit_length", 0.0)
        params.setdefault("twig_leaf_offset",     0.10)
        params.setdefault("density_prune_frac",   0.0)
        params.setdefault("color_noise_scale",    1.0)
        params.setdefault("twig_radius_threshold",0.05)
        params.setdefault("tier_vertical_squash", 0.4)
        return build_wp_shape(params, rng)
    return build_sca2_shape(params, rng)
from ipt.render_util import render_tree

# spatial-color / target helpers (formerly v3 module)
from ipt.target import (
    N_DIST_BINS, DIST_BIN_MAX_PX, BRIGHTNESS_THRESHOLDS,
    bucket_brightness, extract_spatial_model, extract_target,
    compute_proc_spatial_signature,
)


# v4-specific shape builder honours `lateral_extent_scale`
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
    lat = params.get("lateral_extent_scale", 1.0)
    top_r *= lat
    base_r *= lat

    # When the species exposes n_tiers, use the multi-bulge envelope;
    # otherwise stay on the single conifer ellipsoid (pine path).
    n_tiers_p = params.get("n_tiers", None)
    if n_tiers_p is not None and int(round(n_tiers_p)) >= 2:
        env = envelope_tiered_broadleaf(
            150, h_total=h_total, base_h=base_h,
            n_tiers=int(round(n_tiers_p)),
            tier_r=base_r * 0.9,
            tier_taper=params.get("tier_taper", 0.8),
            tier_overlap=params.get("tier_overlap", 0.3),
            rng=rng)
    else:
        env = envelope_conifer(150, h_total=h_total, base_h=base_h,
                                top_r=top_r, base_r=base_r, rng=rng)
    nodes = sca(env, max_iter=60,
                influence_radius=inf_r, kill_radius=kill_r,
                step_size=step, trunk_height=base_h * 0.9,
                jitter=jit, rng=rng)
    if len(nodes) > 500:
        nodes = nodes[:500]
    # Da Vinci / Murray: k=2 (cross-section conservation, the version
    # Leonardo wrote down in 1490 + measured later in real trees by
    # Shinozaki 1964). v5 used 2.3 (mild fatter trunks); v5b-bot uses
    # the biologically-correct k=2.
    radius, children = thicken(nodes, leaf_radius=0.025, branch_exp=2.0)
    nodes = smooth_branches(nodes, children, smoothing=0.30)
    segs = to_segments(nodes, radius)
    tip_positions = [(n[0], n[1], n[2]) for i, n in enumerate(nodes)
                     if not children[i]]
    scene_w = max(base_r, top_r) * 2.4
    scene_h = h_total * 1.1
    return segs, tip_positions, scene_w, scene_h, cluster_r

from ipt.paths import PHOTOS, LEAF_LIB, RESULTS
DATA_DIR = RESULTS   # back-compat alias for JSON output dir


# =========================================================================
#  v5b-bot: per-species botanical bounds — biology drives the search space
#  instead of letting the optimizer wander all unconstrained.
# =========================================================================
#
# Sources for the numbers:
#   - First-branch frac (= 1 - crown_ratio): USDA Forest Service crown
#     ratio guidelines for mature healthy conifers / broadleaves
#   - H/DBH slenderness: dendrology textbooks (pine 60-100, oak 40-60,
#     poplar 80-120)
#   - Branch insertion angle: tree architecture / Halle 1978
#   - Phyllotaxis: 137.5° (golden angle) for spiral arrangement —
#     applies via 2-level cluster expansion
#   - Murray's law / Da Vinci: branch_exp k=2 (cross-section conservation)
#
# Each species fixes ranges much tighter than the unconstrained v5 search.

GOLDEN_ANGLE_RAD = math.pi * (3 - math.sqrt(5))   # = 137.508° in radians

SPECIES_BOUNDS = {
    "pine": {
        # SCA shape params — pine prefers narrow tall (slender H/DBH 60-100)
        "influence_radius":      (1.0,  2.6),
        "kill_radius_frac":      (0.25, 0.45),
        "step_size_frac":        (0.18, 0.28),
        "jitter":                (0.05, 0.15),
        # Botany-constrained envelope
        "env_h_total":           (8.0, 13.0),    # tall
        # crown_base = (1 - crown_ratio); pine crown_ratio 0.5-0.7
        # → base_h_frac = 0.30 - 0.50
        "env_base_h_frac":       (0.30, 0.50),
        "env_top_r":             (0.40, 1.20),
        "env_base_r":            (1.20, 2.40),   # narrower (pine slender)
        "lateral_extent_scale":  (0.80, 1.40),   # less lateral spread
        # H/DBH: slenderness 60-100 → trunk_thickness very constrained
        # trunk radius from Murray = ~0.10 in scene units → thickness ×0.5
        # gives DBH ≈ 0.10. H=10, DBH=0.10 → ratio 100. Match.
        "trunk_thickness_scale": (0.30, 0.65),
        # Leaf placement
        "leaf_cluster_r":        (0.20, 0.45),
        "color_noise_scale":     (0.5,  1.3),
        "leaf_scale":            (0.6,  1.3),
        "leaves_per_tip":        (8.0,  18.0),
        # Phyllotaxis cluster expansion (137.5° spiral around tip)
        "n_secondary":           (3.0,  6.0),
        "secondary_radius":      (0.12, 0.22),
        # Along-twig density
        "twig_radius_threshold": (0.030, 0.070),
        "leaves_per_unit_length":(0.0,  6.0),
        "twig_leaf_offset":      (0.06, 0.14),
        "density_prune_frac":    (0.0,  0.4),
        # secondary cluster vertical squash; pine = spherical (0.3-0.6)
        "tier_vertical_squash":  (0.30, 0.60),
    },
    "broadleaf": {
        # broadleaf (oak/maple) — stouter, crown more spreading
        "influence_radius":      (1.2,  2.8),
        "kill_radius_frac":      (0.25, 0.50),
        "step_size_frac":        (0.18, 0.30),
        "jitter":                (0.08, 0.20),
        "env_h_total":           (7.0, 11.0),     # shorter
        # broadleaf crown_ratio 0.65-0.85 → base_h_frac 0.15-0.35
        "env_base_h_frac":       (0.15, 0.35),
        "env_top_r":             (1.00, 2.20),    # broader top
        "env_base_r":            (1.80, 3.50),    # broad envelope
        "lateral_extent_scale":  (1.10, 1.70),    # more lateral spread
        # H/DBH 40-60 → stouter trunks
        "trunk_thickness_scale": (0.60, 1.00),
        "leaf_cluster_r":        (0.30, 0.55),
        "color_noise_scale":     (0.6,  1.4),
        "leaf_scale":            (0.8,  1.6),     # bigger leaves
        "leaves_per_tip":        (12.0, 22.0),
        "n_secondary":           (4.0,  7.0),
        "secondary_radius":      (0.15, 0.28),
        "twig_radius_threshold": (0.025, 0.080),
        "leaves_per_unit_length":(2.0,  10.0),
        "twig_leaf_offset":      (0.08, 0.18),
        "density_prune_frac":    (0.0,  0.4),
        # squash secondary cluster vertically → horizontal "umbrella"
        # disks at each main tip; stacking these at different heights
        # reproduces the layered crown of mature broadleaves.
        "tier_vertical_squash":  (0.05, 0.25),
        # multi-tier envelope (broadleaf only — pine keeps single ellipsoid)
        "n_tiers":               (2.0,  4.0),    # int(round()) at use site
        "tier_taper":            (0.55, 0.95),   # upper tiers smaller by this
        "tier_overlap":          (0.10, 0.45),   # how much tiers blend
    },
}


def build_param_spec(species):
    """Return the v5 PARAM_SPEC format from a SPECIES_BOUNDS entry."""
    b = SPECIES_BOUNDS[species]
    spec = [
        ("influence_radius",       *b["influence_radius"]),
        ("kill_radius_frac",       *b["kill_radius_frac"]),
        ("step_size_frac",         *b["step_size_frac"]),
        ("jitter",                 *b["jitter"]),
        ("env_h_total",            *b["env_h_total"]),
        ("env_base_h_frac",        *b["env_base_h_frac"]),
        ("env_top_r",              *b["env_top_r"]),
        ("env_base_r",             *b["env_base_r"]),
        ("leaf_cluster_r",         *b["leaf_cluster_r"]),
        ("color_noise_scale",      *b["color_noise_scale"]),
        ("leaf_scale",             *b["leaf_scale"]),
        ("trunk_thickness_scale",  *b["trunk_thickness_scale"]),
        ("leaves_per_tip",         *b["leaves_per_tip"]),
        ("lateral_extent_scale",   *b["lateral_extent_scale"]),
        ("n_secondary",            *b["n_secondary"]),
        ("secondary_radius",       *b["secondary_radius"]),
        ("twig_radius_threshold",  *b["twig_radius_threshold"]),
        ("leaves_per_unit_length", *b["leaves_per_unit_length"]),
        ("twig_leaf_offset",       *b["twig_leaf_offset"]),
        ("density_prune_frac",     *b["density_prune_frac"]),
        ("tier_vertical_squash",   *b["tier_vertical_squash"]),
    ]
    # broadleaf only — multi-tier crown envelope knobs
    if "n_tiers" in b:
        spec += [
            ("n_tiers",      *b["n_tiers"]),
            ("tier_taper",   *b["tier_taper"]),
            ("tier_overlap", *b["tier_overlap"]),
        ]
    return spec


# default to pine (set by main() based on --species flag)
PARAM_SPEC = build_param_spec("pine")


def params_to_dict(vec):
    return {n: float(v) for (n, _, _), v in zip(PARAM_SPEC, vec)}


# ============================================================
#  Sprite library helpers
# ============================================================
def load_sprite_library(photo_name):
    """Load all PNGs from leaf_lib/<photo>/. Returns list of RGBA np arrays."""
    photo_stem = os.path.splitext(photo_name)[0]
    d = os.path.join(LEAF_LIB, photo_stem)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"no sprite library at {d}. Run leaf_extractor.py first.")
    sprites = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".png"): continue
        sprites.append(np.array(Image.open(os.path.join(d, f)).convert("RGBA")))
    if not sprites:
        raise RuntimeError(f"no sprite files in {d}")
    return sprites


def tint_sprite(sprite_rgba, tint_rgb):
    """Multiply sprite RGB by tint/255, keep alpha."""
    out = sprite_rgba.astype(np.float32).copy()
    f = np.array(tint_rgb, dtype=np.float32) / 128.0   # 128 = neutral tint
    out[:, :, 0] = np.clip(out[:, :, 0] * f[0], 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * f[1], 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] * f[2], 0, 255)
    return out.astype(np.uint8)


def rotate_scale_sprite(sprite_rgba, angle_deg, scale):
    """Rotate (degrees) + scale; returns RGBA PIL image."""
    im = Image.fromarray(sprite_rgba, "RGBA")
    if scale != 1.0:
        ns = max(4, int(im.size[0] * scale))
        im = im.resize((ns, ns), Image.BILINEAR)
    if angle_deg:
        im = im.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    return im


# ============================================================
#  v4 rasterizer (paste sprites instead of polygons)
# ============================================================
def rasterize_proc_v4(segs, tip_positions, cluster_r, scene_w, scene_h,
                      spatial, params, sprites,
                      mask_res=128, per_tip=None, rng=None):
    # honour params['leaves_per_tip'] when caller doesn't override
    if per_tip is None:
        per_tip = int(round(params.get("leaves_per_tip", 14)))
    # ------------ 2-level cluster expansion -----------------
    # Original SCA gives main_tips. Each one is exploded into n_sec
    # secondary cluster centres inside a small ball. Resulting
    # effective_tips = main_tips × n_sec.
    n_sec = max(1, int(round(params.get("n_secondary", 1))))
    sec_r = params.get("secondary_radius", 0.0)
    MAX_TIPS = 70
    if len(tip_positions) > MAX_TIPS:
        stride = len(tip_positions) // MAX_TIPS + 1
        tip_positions = tip_positions[::stride][:MAX_TIPS]
    # v5b-bot: 2-level cluster expansion now uses PHYLLOTAXIS — secondary
    # tips arranged in a Fibonacci spiral (137.5° golden-angle increments)
    # around each main tip, varying radius. This matches how real conifer
    # needles + sub-twigs distribute on a parent stem.
    expanded_tips = []
    rng_sec = rng
    for tip_idx, tip in enumerate(tip_positions):
        expanded_tips.append(tip)
        # phase offset per tip so different tips don't all start at angle 0
        phase = (tip_idx * 1.61803) % (2 * math.pi)
        for k in range(n_sec - 1):
            # angular coord (spiral) + radial coord (square-root)
            theta = phase + (k + 1) * GOLDEN_ANGLE_RAD
            # radius grows as sqrt(k) — emulates sunflower seed packing
            r_norm = math.sqrt((k + 1) / max(1, n_sec - 1))
            d = r_norm * sec_r * (0.8 + rng_sec.uniform(0, 0.4))  # mild jitter
            jx = math.cos(theta) * d
            jz = math.sin(theta) * d
            # vertical jitter — small value squashes secondary expansion
            # into a horizontal umbrella; large keeps it spherical.
            vsq = params.get("tier_vertical_squash", 0.4)
            jy = rng_sec.uniform(-0.5, 0.5) * sec_r * vsq
            expanded_tips.append((tip[0] + jx, tip[1] + jy, tip[2] + jz))
    # hard cap on total leaves to keep eval bounded
    MAX_TOTAL_LEAVES = 3000     # reduced because bigger sprites paste slower
    if len(expanded_tips) * per_tip > MAX_TOTAL_LEAVES:
        per_tip = max(4, MAX_TOTAL_LEAVES // len(expanded_tips))
    tip_positions = expanded_tips
    # store prune fraction for later use during paste
    prune_frac = max(0.0, min(0.95, params.get("density_prune_frac", 0.0)))
    # apply trunk thickness scaling to incoming segments
    trunk_scale = params.get("trunk_thickness_scale", 1.0)
    if trunk_scale != 1.0:
        segs = [dict(s, r0=s["r0"] * trunk_scale, r1=s["r1"] * trunk_scale)
                for s in segs]
    rng = rng or random.Random()
    # canvases:
    #   mask_img: binary silhouette (for sil_loss)
    #   branch_img: branch-only mask (for distance-to-branch)
    #   leaf_img: leaf-only mask (for spatial/cond scoring)
    #   color_img: RGB composite of the proc tree (for histogram + FFT)
    mask_img   = Image.new("L",   (mask_res, mask_res), 0)
    branch_img = Image.new("L",   (mask_res, mask_res), 0)
    leaf_img   = Image.new("L",   (mask_res, mask_res), 0)
    # v5a fix: light bg so dark-green foliage shows contrast (was black,
    # which crushed dark sprites into a black blob visually)
    BG_COL = (220, 230, 240)
    color_img  = Image.new("RGB", (mask_res, mask_res), BG_COL)
    draw = ImageDraw.Draw(mask_img)
    bdraw = ImageDraw.Draw(branch_img)
    cdraw = ImageDraw.Draw(color_img)
    px_per_unit_x = mask_res / scene_w
    px_per_unit_y = mask_res / scene_h
    def proj(p):
        return (mask_res / 2 + p[0] * px_per_unit_x,
                mask_res - 1 - p[1] * px_per_unit_y)

    # branches — colour + mask + branch-only mask, all with endpoint caps
    for s in segs:
        x0, y0 = proj(s["p0"]); x1, y1 = proj(s["p1"])
        r0_px = max(1, s["r0"] * px_per_unit_x)
        r1_px = max(1, s["r1"] * px_per_unit_x)
        w = max(1, int(r0_px + r1_px))
        for d_, fill in ((draw, 255), (bdraw, 255)):
            d_.line([(x0, y0), (x1, y1)], fill=fill, width=w)
            d_.ellipse([(x0 - r0_px, y0 - r0_px), (x0 + r0_px, y0 + r0_px)], fill=fill)
            d_.ellipse([(x1 - r1_px, y1 - r1_px), (x1 + r1_px, y1 + r1_px)], fill=fill)
        cdraw.line([(x0, y0), (x1, y1)], fill=(58, 38, 22), width=w)
        cdraw.ellipse([(x0 - r0_px, y0 - r0_px), (x0 + r0_px, y0 + r0_px)],
                      fill=(58, 38, 22))
        cdraw.ellipse([(x1 - r1_px, y1 - r1_px), (x1 + r1_px, y1 + r1_px)],
                      fill=(58, 38, 22))

    # leaf positions — combined from two sources:
    #   (1) tip clusters: per_tip leaves around each (already-expanded) tip
    #   (2) along-branch: for every TWIG segment (avg_r < twig_threshold)
    #       sprinkle leaves along its length with perpendicular jitter
    leaf_positions = []
    # ---- (1) tip clusters ----
    for tip in tip_positions:
        for _ in range(per_tip):
            jx = rng.uniform(-1, 1); jy = rng.uniform(-1, 1); jz = rng.uniform(-1, 1)
            L = math.sqrt(jx ** 2 + jy ** 2 + jz ** 2) + 1e-9
            d = rng.random() ** 0.65 * cluster_r
            wp = (tip[0] + jx / L * d, tip[1] + jy / L * d, tip[2] + jz / L * d)
            px, py = proj(wp)
            leaf_positions.append((px, py, wp[2]))

    # ---- (2) along-branch leaves on twigs ----
    twig_thresh = params.get("twig_radius_threshold", 0.07)
    leaves_per_unit = params.get("leaves_per_unit_length", 6.0)
    twig_offset = params.get("twig_leaf_offset", 0.10)
    for s in segs:
        avg_r = (s["r0"] + s["r1"]) / 2
        if avg_r > twig_thresh:                  # too thick = trunk/major
            continue
        # 3D segment length
        dx = s["p1"][0] - s["p0"][0]
        dy = s["p1"][1] - s["p0"][1]
        dz = s["p1"][2] - s["p0"][2]
        seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if seg_len < 0.05: continue
        # direction + an arbitrary perpendicular axis
        ux, uy, uz = dx / seg_len, dy / seg_len, dz / seg_len
        # perpendicular: pick world-up cross direction, fall back if parallel
        if abs(uy) < 0.95:
            px_a, py_a, pz_a = -uz, 0, ux
            pl = math.sqrt(px_a*px_a + py_a*py_a + pz_a*pz_a) + 1e-9
            px_a /= pl; py_a /= pl; pz_a /= pl
        else:
            px_a, py_a, pz_a = 1.0, 0.0, 0.0
        n_along = max(2, int(seg_len * leaves_per_unit))
        for j in range(n_along):
            t = (j + 0.5 + rng.uniform(-0.2, 0.2)) / n_along
            t = max(0.0, min(1.0, t))
            # perpendicular offset (random angle around branch axis)
            ang = rng.uniform(0, math.pi * 2)
            ca, sa = math.cos(ang), math.sin(ang)
            # second perpendicular via cross(u, p_a)
            cx = uy * pz_a - uz * py_a
            cy = uz * px_a - ux * pz_a
            cz = ux * py_a - uy * px_a
            off_d = rng.random() * twig_offset
            off_x = (px_a * ca + cx * sa) * off_d
            off_y = (py_a * ca + cy * sa) * off_d
            off_z = (pz_a * ca + cz * sa) * off_d
            wx = s["p0"][0] + dx * t + off_x
            wy = s["p0"][1] + dy * t + off_y
            wz = s["p0"][2] + dz * t + off_z
            px, py = proj((wx, wy, wz))
            leaf_positions.append((px, py, wz))

    # branch distance for color binning (proc tree's own 2D space)
    branch_arr = np.array(branch_img) > 127
    if branch_arr.any():
        dt = distance_transform_edt(~branch_arr)
    else:
        dt = np.zeros_like(branch_arr, dtype=float)
    photo_h_px = spatial["bbox"][3] - spatial["bbox"][1]
    scale_factor = photo_h_px / mask_res
    bin_edges = spatial["bin_edges"]
    noise_scale = params.get("color_noise_scale", 1.0)
    leaf_scale_param = params.get("leaf_scale", 1.0)
    sprite_native_size = sprites[0].shape[0]
    # v4r6: bigger sprites → adjacent leaves overlap and fill micro-gaps.
    # Was 0.20 × cluster_r, now 0.42 × cluster_r.
    sprite_target_px = max(6, int(cluster_r * 0.42 * px_per_unit_x * leaf_scale_param))
    scale_factor_sprite = sprite_target_px / sprite_native_size

    leaf_colors = []
    # depth-sort back→front for painter's algorithm
    leaf_positions.sort(key=lambda t: t[2])
    leaf_mask_arr = np.zeros((mask_res, mask_res), dtype=bool)

    for (px, py, _z) in leaf_positions:
        # v5a-prune: probabilistically drop this leaf (optimizer-controlled
        # density brake; 0 keeps all, 0.6 drops 60%)
        if prune_frac > 0 and rng.random() < prune_frac:
            continue
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

        # pick random sprite + transform + paste
        sp_idx = rng.randrange(len(sprites))
        sp_scale = scale_factor_sprite * rng.uniform(0.8, 1.25)
        sp_angle = rng.uniform(0, 360)
        sp_tinted = tint_sprite(sprites[sp_idx], col)
        sp_im = rotate_scale_sprite(sp_tinted, sp_angle, sp_scale)
        ox = int(px - sp_im.size[0] / 2)
        oy = int(py - sp_im.size[1] / 2)
        color_img.paste(sp_im, (ox, oy), sp_im)
        # also stamp opaque area into the mask + leaf_img
        sp_alpha = np.array(sp_im)[:, :, 3] > 80
        # write region (clipped to canvas)
        ay0 = max(0, oy); ax0 = max(0, ox)
        ay1 = min(mask_res, oy + sp_im.size[1])
        ax1 = min(mask_res, ox + sp_im.size[0])
        if ay1 <= ay0 or ax1 <= ax0: continue
        sy0 = ay0 - oy; sx0 = ax0 - ox
        sy1 = sy0 + (ay1 - ay0); sx1 = sx0 + (ax1 - ax0)
        sub_alpha = sp_alpha[sy0:sy1, sx0:sx1]
        sub_mask = leaf_mask_arr[ay0:ay1, ax0:ax1]
        sub_mask |= sub_alpha
        leaf_mask_arr[ay0:ay1, ax0:ax1] = sub_mask

    # Convert leaf_mask_arr back into PIL for downstream consistency
    leaf_img = Image.fromarray((leaf_mask_arr.astype(np.uint8) * 255), "L")
    # final overall mask: branch ∪ leaf
    full_mask = np.array(mask_img) > 127
    full_mask |= leaf_mask_arr
    return (full_mask.astype(np.uint8),
            branch_arr,
            leaf_mask_arr,
            np.array(leaf_colors, dtype=np.int32) if leaf_colors else np.zeros((0, 3), dtype=np.int32),
            color_img)


# ============================================================
#  FFT term
# ============================================================
def compute_log_power_spectrum(gray_arr, low_freq_band=24):
    """Return log magnitude of 2D rFFT over the central low-freq band.
    gray_arr: 2D float, same dimensions for target + proc."""
    f = np.fft.fft2(gray_arr)
    f = np.fft.fftshift(f)
    mag = np.abs(f)
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    half = low_freq_band
    band = mag[cy - half: cy + half, cx - half: cx + half]
    return np.log1p(band)


def fft_loss(target_color_img_rgb, proc_color_img_rgb, low_band=24):
    """Compare log-power spectra of the LUMA channel of both images."""
    t_lum = target_color_img_rgb.astype(np.float32).mean(axis=2)
    p_lum = proc_color_img_rgb.astype(np.float32).mean(axis=2)
    t_lps = compute_log_power_spectrum(t_lum, low_band)
    p_lps = compute_log_power_spectrum(p_lum, low_band)
    # normalise so DC doesn't dominate
    t_lps /= max(t_lps.max(), 1e-6)
    p_lps /= max(p_lps.max(), 1e-6)
    return float(np.linalg.norm(t_lps - p_lps) / t_lps.size ** 0.5)


# ============================================================
#  v4 objective
# ============================================================
def _erode_1px(m):
    """1-pixel erosion via numpy (4-connected)."""
    H, W = m.shape
    out = m.copy()
    out[1:]   &= m[:-1]
    out[:-1]  &= m[1:]
    out[:, 1:]  &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _dilate_npx(m, n=1):
    """N-pixel 8-connected dilation via repeated PIL MaxFilter."""
    img = Image.fromarray((m.astype(np.uint8) * 255), "L")
    from PIL import ImageFilter as IF
    for _ in range(n):
        img = img.filter(IF.MaxFilter(3))
    return np.array(img) > 127


def _boundary_iou(t, p, band_px=3):
    """Cheng et al. CVPR 2021 — IoU restricted to silhouette outline band.
    Rewards shape match instead of interior fill."""
    # outline = mask XOR eroded mask
    t_in = _erode_1px(t.astype(bool))
    p_in = _erode_1px(p.astype(bool))
    t_outline = t.astype(bool) & ~t_in
    p_outline = p.astype(bool) & ~p_in
    # dilate the outlines to a thicker band so small mis-alignments still
    # produce non-zero overlap
    t_band = _dilate_npx(t_outline, band_px)
    p_band = _dilate_npx(p_outline, band_px)
    inter = (t_band & p_band).sum()
    union = (t_band | p_band).sum()
    return inter / max(union, 1)


def _center_of_mass(m):
    if m.sum() == 0: return (0.0, 0.0)
    ys, xs = np.where(m)
    return (float(ys.mean()), float(xs.mean()))


def _bbox(m):
    if m.sum() == 0: return (0, 0, 0, 0)
    ys, xs = np.where(m)
    return (xs.min(), ys.min(), xs.max(), ys.max())


def _band_signature(mask, color_img, n_bands=5):
    """Per-band (silhouette fraction, mean RGB) — rewards vertical structure."""
    H = mask.shape[0]
    band_h = H // n_bands
    sig = []
    for i in range(n_bands):
        y0 = i * band_h
        y1 = H if i == n_bands - 1 else (i + 1) * band_h
        m = mask[y0:y1]
        c = color_img[y0:y1]
        frac = float(m.sum()) / max(1, m.size)
        if m.sum() > 0:
            rgb_mean = c[m.astype(bool)].mean(axis=0) / 255.0
        else:
            rgb_mean = np.zeros(3)
        sig.append((frac, rgb_mean))
    return sig


def _band_loss(t_mask, t_color, p_mask, p_color, n_bands=5):
    t_sig = _band_signature(t_mask, t_color, n_bands)
    p_sig = _band_signature(p_mask, p_color, n_bands)
    loss = 0.0
    for (tf, trgb), (pf, prgb) in zip(t_sig, p_sig):
        loss += abs(tf - pf) + float(np.abs(trgb - prgb).mean())
    return loss / n_bands


def objective_v4(target, proc_mask, branch_arr, leaf_arr, leaf_colors,
                  proc_color_img, target_color_img, *,
                  w_sil=0.5,  w_col=0.4, w_spat=0.6, w_cond=0.4, w_fft=0.4,
                  w_ssim=0.5, w_lab=0.3,
                  w_biou=1.0, w_dist=0.5, w_aspect=0.3,
                  w_band=0.6,
                  # variant (c): Tversky asymmetric (Salehi 2017). When
                  # tversky=True, sil_loss = 1 - TP/(TP + α*FP + β*FN).
                  # α > β penalises false-positive sprawl harder than gaps.
                  # v2: Tversky DEFAULT (winner of 4-way ablation). α=0.7
                  # > β=0.3 penalises proc sprawl 2.3× harder than gaps.
                  tversky=True, tversky_alpha=0.7, tversky_beta=0.3,
                  # v2: VGG normalised by 30 so raw mse_loss falls in [0,1].
                  vgg_model=None, w_vgg=0.5, vgg_norm=30.0):
    spatial = target["spatial"]
    t = target["target_mask"]; p = proc_mask
    tp = (t & p).sum()
    fp = (~t & p).sum()
    fn = (t & ~p).sum()
    iou = tp / max(tp + fp + fn, 1)
    if tversky:
        denom = tp + tversky_alpha * fp + tversky_beta * fn
        tversky_val = tp / max(denom, 1)
        sil_loss = 1.0 - tversky_val
    else:
        sil_loss = 1.0 - iou

    # === post-2019 IoU improvements ===
    # 1. Boundary IoU (Cheng CVPR 2021): match silhouette OUTLINE not interior
    b_iou = _boundary_iou(t, p, band_px=3)
    biou_loss = 1.0 - b_iou

    # 2. Distance IoU (Zheng AAAI 2020): center-of-mass distance, normalised
    t_cy, t_cx = _center_of_mass(t.astype(bool))
    p_cy, p_cx = _center_of_mass(p.astype(bool))
    H, W = t.shape
    diag = math.sqrt(H * H + W * W)
    dist_loss = math.sqrt((t_cy - p_cy) ** 2 + (t_cx - p_cx) ** 2) / max(diag, 1)

    # 3. Aspect ratio (CIoU variant): bbox height/width ratio mismatch
    t_x0, t_y0, t_x1, t_y1 = _bbox(t.astype(bool))
    p_x0, p_y0, p_x1, p_y1 = _bbox(p.astype(bool))
    t_h = max(1, t_y1 - t_y0); t_w = max(1, t_x1 - t_x0)
    p_h = max(1, p_y1 - p_y0); p_w = max(1, p_x1 - p_x0)
    aspect_loss = abs(math.log(t_h / t_w) - math.log(p_h / p_w)) / 2  # ÷2 normalise

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
    pop = spatial["pop_per_bin"].astype(np.float32) + 1
    pop_norm = pop / pop.sum()
    bin_diffs = np.linalg.norm(proc_mean - spatial["mean_per_bin"], axis=1) / 255
    spat_loss = (pop_norm * bin_diffs).sum()

    eps = 1e-4
    t_cond = spatial["cond_matrix"]
    kl = (t_cond * (np.log(t_cond + eps) - np.log(proc_cond + eps))).sum() / 4
    cond_loss = abs(kl)

    # v4 fft term
    proc_rgb_arr = np.array(proc_color_img)
    f_loss = fft_loss(target_color_img, proc_rgb_arr)

    # v5a SSIM + Lab — gated by HAS_SKIMAGE
    ssim_loss = 0.0
    lab_loss = 0.0
    if HAS_SKIMAGE:
        # SSIM on grayscale (luma)
        t_gray = target_color_img.mean(axis=2)
        p_gray = proc_rgb_arr.mean(axis=2)
        try:
            ssim_val = _ssim(t_gray, p_gray, data_range=255.0,
                              gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False)
        except Exception:
            ssim_val = 0.0
        ssim_loss = 1.0 - max(0.0, min(1.0, float(ssim_val)))

        # Lab-space colour: percentile L1 over opaque pixels
        # use the mask we already have (proc_mask vs target_mask) to
        # restrict to foliage region (skip black background)
        t_op_mask = target_color_img.sum(axis=2) > 10
        p_op_mask = proc_rgb_arr.sum(axis=2) > 10
        if t_op_mask.sum() > 50 and p_op_mask.sum() > 50:
            try:
                t_lab = rgb2lab(target_color_img / 255.0)
                p_lab = rgb2lab(proc_rgb_arr / 255.0)
                pct = [10, 25, 50, 75, 90]
                t_pcts = np.stack([np.percentile(t_lab[t_op_mask][:, c], pct)
                                    for c in range(3)])
                p_pcts = np.stack([np.percentile(p_lab[p_op_mask][:, c], pct)
                                    for c in range(3)])
                # L*: 0..100, a/b: -128..127. Normalise by channel range.
                norm = np.array([100.0, 256.0, 256.0])[:, None]
                lab_loss = float(np.abs((t_pcts - p_pcts) / norm).mean())
            except Exception:
                lab_loss = 0.0

    if p.sum() < 50:
        deg = 2.0
    else:
        deg = 0.0

    # vertical banding: match per-band silhouette mass + mean RGB so
    # the optimizer is pressured to reproduce the photo's tier structure.
    try:
        proc_rgb = np.asarray(proc_color_img)
        band_loss = _band_loss(t.astype(np.uint8), target_color_img,
                                p.astype(np.uint8), proc_rgb, n_bands=5)
    except Exception:
        band_loss = 0.0

    # variant (d) — VGG perceptual loss on RGB
    vgg_loss = 0.0
    if vgg_model is not None and HAS_TORCH:
        try:
            with torch.no_grad():
                t_feat = vgg_extract(vgg_model, target_color_img)
                p_feat = vgg_extract(vgg_model, proc_rgb_arr)
                vgg_raw = float(torch.nn.functional.mse_loss(t_feat, p_feat).item())
                vgg_loss = vgg_raw / vgg_norm   # normalise to [0..1] range
        except Exception as ex:
            vgg_loss = 0.0

    score = (w_sil   * sil_loss     + w_col  * col_loss
             + w_spat * spat_loss   + w_cond * cond_loss
             + w_fft  * f_loss
             + w_ssim * ssim_loss   + w_lab  * lab_loss
             + w_biou * biou_loss   + w_dist * dist_loss
             + w_aspect * aspect_loss
             + w_band * band_loss
             + w_vgg  * vgg_loss + deg)
    sub = dict(sil=sil_loss, col=col_loss, spat=spat_loss,
                cond=cond_loss, fft=f_loss,
                ssim=ssim_loss, lab=lab_loss,
                biou=biou_loss, dist=dist_loss, aspect=aspect_loss,
                band=band_loss,
                vgg=vgg_loss, iou=iou, b_iou=b_iou)
    return score, sub


# ============================================================
#  VGG feature extractor (variant d)
# ============================================================
_VGG_CACHE = {"model": None, "preprocess": None}

def load_vgg():
    if not HAS_TORCH:
        return None
    if _VGG_CACHE["model"] is None:
        # VGG16 features only — through relu_3_3 (index 15)
        weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        full = tv_models.vgg16(weights=weights)
        feats = full.features[:16]    # up to relu_3_3
        feats.eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        _VGG_CACHE["model"] = feats
        _VGG_CACHE["preprocess"] = tv_transforms.Compose([
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
    return _VGG_CACHE["model"]


def vgg_extract(model, rgb_arr):
    """rgb_arr: HxWx3 uint8 numpy → 1×256×28×28 feature tensor."""
    if model is None:
        return None
    x = _VGG_CACHE["preprocess"](rgb_arr.astype(np.uint8)).unsqueeze(0)
    return model(x)


# ============================================================
#  Target colour image (for FFT term)
# ============================================================
def build_target_color_image(target, mask_res=128):
    """Resize the photo crop to mask_res × mask_res, keep RGB only.
    Composite non-opaque pixels onto the SAME light-bg the proc canvas
    uses, so SSIM/Lab/FFT objectives compare apples-to-apples and the
    visual comparison doesn't show "dark on black"."""
    pil = target["crop_pil"]
    arr = np.array(pil.convert("RGBA"))
    rgb = arr[:, :, :3]
    op = arr[:, :, 3] > 100
    BG_COL = np.array([220, 230, 240], dtype=np.uint8)
    rgb_bg = rgb.copy()
    rgb_bg[~op] = BG_COL
    out = Image.fromarray(rgb_bg).resize((mask_res, mask_res), Image.LANCZOS)
    return np.array(out)


# ============================================================
#  Fit driver
# ============================================================
def fit_de_v4(target, sprites, target_color_img, max_iter=8,
               popsize=3, seed=7, obj_kwargs=None):
    bounds = [(lo, hi) for (_, lo, hi) in PARAM_SPEC]
    trace = []; counter = {"n": 0}
    obj_kwargs = obj_kwargs or {}
    def fn(vec):
        params = params_to_dict(vec)
        rng = random.Random(seed)
        segs, tips, sw, sh, cr = build_shape(params, rng)
        mask, b_arr, l_arr, l_cols, p_color = rasterize_proc_v4(
            segs, tips, cr, sw, sh, target["spatial"], params, sprites,
            mask_res=target["mask_res"], rng=rng)
        score, sub = objective_v4(target, mask, b_arr, l_arr, l_cols,
                                    p_color, target_color_img, **obj_kwargs)
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
    print(f"  DE v4 finished: {counter['n']} evals in {dt:.1f}s, best={result.fun:.4f}")
    return result.x, result.fun, trace


# ============================================================
#  Human preview
# ============================================================
def render_preview_v4(target, vec, sprites, out_path, *, seed=7,
                       preview_res=560):
    params = params_to_dict(vec)
    rng = random.Random(seed)
    segs, tips, sw, sh, cr = build_shape(params, rng)
    # render at preview_res
    rng2 = random.Random(seed)
    mask, b_arr, l_arr, l_cols, color_img = rasterize_proc_v4(
        segs, tips, cr, sw, sh, target["spatial"], params, sprites,
        mask_res=preview_res, rng=rng2)
    color_img.save(out_path, "PNG", optimize=True)


# ============================================================
#  CLI
# ============================================================
def main():
    global PARAM_SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo",    default="tree_pine.png")
    ap.add_argument("--tree-idx", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--seed",     type=int, default=7)
    ap.add_argument("--species",  default="pine",
                     choices=list(SPECIES_BOUNDS.keys()),
                     help="botanical preset selecting param bounds")
    ap.add_argument("--backend",  default="sca",
                     choices=["sca", "wp"],
                     help="shape backend: sca (SCA2) or wp (Weber-Penn)")
    # 4-way ablation flags
    ap.add_argument("--w-biou",   type=float, default=1.0,
                     help="(a) boundary-IoU weight (default 1.0; try 1.5)")
    ap.add_argument("--w-cond",   type=float, default=0.4,
                     help="conditional brightness KL weight (default 0.4; "
                          "lower for WP whose leaf bins don't align with SCA)")
    ap.add_argument("--tversky",  action="store_true",
                     help="(c) Tversky asymmetric loss (α=0.7 FP, β=0.3 FN)")
    ap.add_argument("--vgg",      action="store_true",
                     help="(d) VGG perceptual feature loss (loads VGG16)")
    ap.add_argument("--tag",      default="v5b_bot",
                     help="output filename tag (used to separate variant runs)")
    args = ap.parse_args()

    # v5b-bot: install species-specific PARAM_SPEC (or WP spec)
    global BACKEND
    BACKEND = args.backend
    if args.backend == "wp":
        PARAM_SPEC = get_wp_spec(args.species)
        print(f"  backend=wp (Weber-Penn, {args.species} bounds), "
              f"{len(PARAM_SPEC)} params")
    else:
        PARAM_SPEC = build_param_spec(args.species)
        print(f"  species={args.species}, {len(PARAM_SPEC)} params, "
              f"all bounds biologically constrained")

    photo_path = os.path.join(PHOTOS, args.photo)
    target = extract_target(photo_path, args.tree_idx)
    target_color_img = build_target_color_image(target, target["mask_res"])
    sprites = load_sprite_library(args.photo)
    print(f"  loaded {len(sprites)} sprites")

    init_vec = [(lo + hi) / 2 for (_, lo, hi) in PARAM_SPEC]
    init_p = params_to_dict(init_vec)
    rng = random.Random(args.seed)
    segs, tips, sw, sh, cr = build_shape(init_p, rng)
    mask, b_arr, l_arr, l_cols, p_color = rasterize_proc_v4(
        segs, tips, cr, sw, sh, target["spatial"], init_p, sprites,
        mask_res=target["mask_res"], rng=rng)
    init_score, init_sub = objective_v4(target, mask, b_arr, l_arr, l_cols,
                                          p_color, target_color_img)
    print(f"  v4 INIT  score = {init_score:.4f}  {init_sub}")

    # build objective kwargs from ablation flags
    vgg_model = load_vgg() if args.vgg else None
    if args.vgg and vgg_model is None:
        print("  WARN: --vgg requested but torch not available")
    obj_kwargs = dict(w_biou=args.w_biou,
                       w_cond=args.w_cond,
                       tversky=args.tversky,
                       vgg_model=vgg_model)
    print(f"  variant: w_biou={args.w_biou} w_cond={args.w_cond} "
          f"tversky={args.tversky} vgg={vgg_model is not None}")
    best_vec, best_score, trace = fit_de_v4(target, sprites,
                                              target_color_img,
                                              max_iter=args.max_iter,
                                              seed=args.seed,
                                              obj_kwargs=obj_kwargs)
    best_p = params_to_dict(best_vec)
    rng = random.Random(args.seed)
    segs, tips, sw, sh, cr = build_shape(best_p, rng)
    mask, b_arr, l_arr, l_cols, p_color = rasterize_proc_v4(
        segs, tips, cr, sw, sh, target["spatial"], best_p, sprites,
        mask_res=target["mask_res"], rng=rng)
    final_score, final_sub = objective_v4(target, mask, b_arr, l_arr, l_cols,
                                            p_color, target_color_img,
                                            **obj_kwargs)
    print(f"  v4 BEST  score = {final_score:.4f}  {final_sub}")
    print(f"  v4 improvement: {(1 - final_score / max(init_score, 1e-9)) * 100:.1f}%")

    tag = f"{args.tag}_{os.path.splitext(args.photo)[0]}_{args.tree_idx}"
    proc_path = f"/tmp/_{args.tag}_proc.png"
    render_preview_v4(target, best_vec, sprites, proc_path, seed=args.seed)

    # composite: original | v3 | v4
    # Composite photo crop onto same light bg so visual comparison is fair
    _crop_arr = np.array(target["crop_pil"].convert("RGBA"))
    _bg = np.array([220, 230, 240], dtype=np.uint8)
    _rgb = _crop_arr[:, :, :3].copy()
    _rgb[_crop_arr[:, :, 3] <= 100] = _bg
    orig = Image.fromarray(_rgb)
    H = 600
    imgs = [("original", orig)]
    if os.path.exists("/tmp/_v4_proc.png"):
        imgs.append(("v4 best (sprites + FFT)",
                     Image.open("/tmp/_v4_proc.png").convert("RGB")))
    imgs.append((f"v5a best (+SSIM/Lab, {final_score:.3f})",
                  Image.open(proc_path).convert("RGB")))
    scaled = [(t, im.resize((int(im.size[0] * H / im.size[1]), H), Image.LANCZOS))
              for t, im in imgs]
    W = sum(im.size[0] for _, im in scaled) + 16 * (len(scaled) + 1)
    canvas = Image.new("RGB", (W, H + 90), (240, 240, 235))
    draw = ImageDraw.Draw(canvas)
    title = (f"v5a +SSIM/Lab  {args.photo}#{args.tree_idx}   "
             f"init→best  {init_score:.4f}→{final_score:.4f}   "
             f"sil={final_sub['sil']:.3f} col={final_sub['col']:.3f} "
             f"spat={final_sub['spat']:.4f} cond={final_sub['cond']:.3f} "
             f"fft={final_sub['fft']:.3f} ssim={final_sub.get('ssim', 0):.3f} "
             f"lab={final_sub.get('lab', 0):.3f}")
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
