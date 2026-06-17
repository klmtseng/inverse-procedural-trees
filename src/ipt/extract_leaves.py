"""Auto-extract leaf-cluster sprite templates from a tree photo.

Pipeline:
  1. Load `tree_*.png` (chroma-keyed, alpha-tagged).
  2. Build a LEAF mask = opaque ∧ ¬branch (branch = lum < 25, our threshold
     from spatial_color_analysis.py).
  3. Distance transform on the leaf mask → each pixel's distance to the
     nearest NON-leaf pixel (i.e. inside-ness of foliage).
  4. peak_local_max → local maxima of the distance map = cluster centres
     guaranteed to be "deep inside" a leafy region.
  5. Around each centre, crop `sprite_size × sprite_size` window from the
     original RGB + retain alpha from leaf mask, then apply a soft circular
     gradient to fade the rim so paste blends.
  6. K-means on flattened (color-mean) crops → keep one representative per
     cluster. Save N sprites as PNG.

Run via:
    python3 tools/blender/leaf_extractor.py --photo tree_pine.png \\
        --n-templates 10 --sprite-size 32

Output: public/assets/data/leaf_lib/tree_pine/{00..09}.png
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

from ipt.paths import PHOTOS, LEAF_LIB


def find_local_maxima_2d(arr, min_distance=8, threshold_abs=2.0):
    """Lightweight local-maxima finder for a 2D float array.
    Returns list of (y, x) tuples sorted by descending value.
    Picks the brightest pixels, then greedily removes neighbours
    within min_distance to ensure separation."""
    H, W = arr.shape
    mask = arr >= threshold_abs
    candidates = np.argwhere(mask)
    if len(candidates) == 0: return []
    values = arr[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-values)
    candidates = candidates[order]
    chosen = []
    chosen_arr = np.empty((0, 2), dtype=int)
    for y, x in candidates:
        if len(chosen_arr) > 0:
            d2 = ((chosen_arr[:, 0] - y) ** 2 + (chosen_arr[:, 1] - x) ** 2)
            if d2.min() < min_distance ** 2:
                continue
        chosen.append((int(y), int(x)))
        chosen_arr = np.vstack([chosen_arr, [[y, x]]])
        if len(chosen) >= 400:    # cap so K-means stays fast
            break
    return chosen


def build_soft_circular_alpha(size, inner=0.6, outer=0.95):
    """1.0 inside inner*r, smoothstep to 0 at outer*r. Returns uint8 (size, size)."""
    s = size
    cx, cy = (s - 1) / 2, (s - 1) / 2
    r_max = (s - 1) / 2
    ys, xs = np.mgrid[0:s, 0:s]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r_max
    a = np.ones_like(d, dtype=np.float32)
    fade = (d - inner) / max(1e-6, outer - inner)
    fade = np.clip(fade, 0, 1)
    a = (1.0 - fade) ** 2
    a[d >= outer] = 0
    return (a * 255).astype(np.uint8)


def extract_templates(photo_name, n_templates=10, sprite_size=32):
    photo_path = os.path.join(PHOTOS, photo_name)
    img = Image.open(photo_path).convert("RGBA")
    arr = np.array(img)
    W, H = img.size
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    op = alpha > 100
    lum = rgb.mean(axis=2)

    # leaf mask = opaque AND lum >= 25 (skip dark branches)
    leaf_mask = op & (lum >= 25)
    print(f"  {photo_name}: {W}×{H}, opaque {op.sum()}, leaf {leaf_mask.sum()}")

    # distance INSIDE the leaf region
    dist_inside = distance_transform_edt(leaf_mask)
    # local maxima → "deep inside leafy" centres
    peaks = find_local_maxima_2d(dist_inside,
                                  min_distance=sprite_size // 2,
                                  threshold_abs=max(2.0, sprite_size / 8))
    print(f"  found {len(peaks)} candidate cluster centres")
    if len(peaks) < n_templates:
        raise RuntimeError(f"only {len(peaks)} centres, asked for {n_templates}")

    soft_alpha = build_soft_circular_alpha(sprite_size)

    # crop sprite_size around each centre
    crops_rgba = []
    crops_color_summary = []        # flat 12-d vector for clustering
    half = sprite_size // 2
    for (cy, cx) in peaks:
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            continue
        rgb_crop = rgb[y0:y1, x0:x1]
        leaf_crop = leaf_mask[y0:y1, x0:x1]
        # alpha = leaf_mask AND soft circular
        a_crop = (leaf_crop.astype(np.uint16) * soft_alpha).clip(0, 255).astype(np.uint8)
        if a_crop.sum() < soft_alpha.sum() * 0.30:
            continue                # too much transparency, useless crop
        crop_rgba = np.dstack([rgb_crop, a_crop])
        crops_rgba.append(crop_rgba)
        # summary: avg color of 4 quadrants → 12-d vector
        s = sprite_size // 2
        q = []
        for (yy0, yy1, xx0, xx1) in ((0, s, 0, s), (0, s, s, sprite_size),
                                       (s, sprite_size, 0, s), (s, sprite_size, s, sprite_size)):
            q.extend(rgb_crop[yy0:yy1, xx0:xx1].mean(axis=(0, 1)))
        crops_color_summary.append(q)
    print(f"  retained {len(crops_rgba)} crops after edge/alpha filtering")
    if len(crops_rgba) < n_templates:
        raise RuntimeError("not enough valid crops after filtering")

    # K-means cluster the crops by their 12-d color summary
    X = np.array(crops_color_summary)
    km = KMeans(n_clusters=n_templates, n_init=10, random_state=7).fit(X)
    # for each cluster, pick the crop nearest the centroid
    chosen = []
    for k in range(n_templates):
        ids = np.where(km.labels_ == k)[0]
        if len(ids) == 0:
            continue
        centroid = km.cluster_centers_[k]
        d = np.linalg.norm(X[ids] - centroid, axis=1)
        best = ids[d.argmin()]
        chosen.append(best)

    # save sprites
    out_dir = os.path.join(LEAF_LIB, os.path.splitext(photo_name)[0])
    os.makedirs(out_dir, exist_ok=True)
    # clean old contents
    for f in os.listdir(out_dir):
        try: os.unlink(os.path.join(out_dir, f))
        except OSError: pass
    saved = []
    for i, idx in enumerate(chosen):
        sprite = Image.fromarray(crops_rgba[idx], "RGBA")
        # gentle blur to remove pixel jaggies, keep alpha sharp
        rgb_part = sprite.convert("RGB").filter(ImageFilter.GaussianBlur(0.5))
        rgb_arr = np.array(rgb_part)
        sprite_arr = np.dstack([rgb_arr, np.array(sprite)[:, :, 3]])
        Image.fromarray(sprite_arr, "RGBA").save(
            os.path.join(out_dir, f"{i:02d}.png"), "PNG", optimize=True)
        saved.append(f"{i:02d}.png")
    print(f"  saved {len(saved)} sprites to {out_dir}/")
    return out_dir, saved


def build_contact_sheet(out_dir, photo_name):
    """Composite all sprites side by side for visual inspection."""
    sprite_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    if not sprite_files:
        return
    sprites = [Image.open(os.path.join(out_dir, f)).convert("RGBA")
               for f in sprite_files]
    sz = sprites[0].size[0]
    PAD = 8; SCALE = 4
    cell = sz * SCALE + PAD
    cols = min(5, len(sprites))
    rows = (len(sprites) + cols - 1) // cols
    W = cols * cell + PAD
    H = rows * cell + PAD + 30
    out = Image.new("RGB", (W, H), (240, 240, 235))
    draw = ImageDraw.Draw(out)
    draw.text((10, 6), f"leaf templates extracted from {photo_name}  ({len(sprites)} sprites, {sz}×{sz})",
              fill=(15, 15, 15))
    for i, s in enumerate(sprites):
        big = s.resize((sz * SCALE, sz * SCALE), Image.NEAREST)
        col = i % cols; row = i // cols
        x = PAD + col * cell
        y = 30 + PAD + row * cell
        # checkerboard background to show alpha
        cb = Image.new("RGB", big.size, (220, 220, 220))
        for cy in range(0, big.size[1], 8):
            for cx in range(0, big.size[0], 8):
                if ((cy // 8) + (cx // 8)) % 2 == 0:
                    cb.paste((180, 180, 180), (cx, cy, cx + 8, cy + 8))
        cb.paste(big, (0, 0), big)
        out.paste(cb, (x, y))
        draw.text((x + 4, y + sz * SCALE - 16),
                  sprite_files[i], fill=(255, 255, 255))
    out.save("/tmp/leaf_lib_contact_sheet.png", "PNG", optimize=True)
    print("  wrote contact sheet /tmp/leaf_lib_contact_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo",        default="tree_pine.png")
    ap.add_argument("--n-templates",  type=int, default=10)
    ap.add_argument("--sprite-size",  type=int, default=32)
    args = ap.parse_args()
    out_dir, saved = extract_templates(args.photo, args.n_templates,
                                         args.sprite_size)
    build_contact_sheet(out_dir, args.photo)


if __name__ == "__main__":
    main()
