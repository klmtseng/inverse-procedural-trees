"""Shared PIL utilities for rendering procedural tree previews.
Used by procgen_tree_*.py to write debug 2D projections.
"""
from PIL import Image, ImageDraw


def project_xy(p):
    """Trees are stored in 3D (x, y, z) with y-up. Preview is XY orthographic."""
    return (p[0], p[1])


def render_tree(segs, leaves, out_path, *,
                size=(560, 720), title=None,
                trunk_col=(58, 38, 22), leaf_col=(48, 105, 48),
                leaf_radius=4, bg=(252, 250, 245)):
    """Render branches as tapered line segments + leaves as small dots.

    segs:    list of {p0:[x,y,z], p1:[x,y,z], r0:float, r1:float}
    leaves:  list of [x, y, z]  OR  list of {"p":[x,y,z], "r":float, "col":[r,g,b]}
    """
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    W, H = size
    pad = 30
    title_h = 28 if title else 0

    # bbox from segs
    xs, ys = [], []
    for s in segs:
        for p in (s["p0"], s["p1"]):
            xs.append(p[0]); ys.append(p[1])
    if not xs:
        if title:
            draw.text((10, 6), title, fill=(20, 20, 20))
        img.save(out_path); return
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    bw, bh = max(0.1, maxx - minx), max(0.1, maxy - miny)
    avail_w = W - 2 * pad
    avail_h = H - 2 * pad - title_h
    s = min(avail_w / bw, avail_h / bh)
    cx = W / 2 - ((minx + maxx) / 2) * s
    cy = H - pad - (0 - miny) * s   # ground at y=0 sits at H-pad

    def proj(p):
        return (cx + p[0] * s, cy - p[1] * s)

    # draw branches. PIL's draw.line uses SQUARE end caps which leaves
    # V-shaped gaps wherever two segments meet at an angle (very visible on
    # tapered branches). Fix: stamp a filled circle at each endpoint with
    # radius matching the local branch radius — guarantees smooth joints.
    for seg in segs:
        x0, y0 = proj(seg["p0"]); x1, y1 = proj(seg["p1"])
        r0_px = max(1, (seg["r0"]) * s)
        r1_px = max(1, (seg["r1"]) * s)
        w = max(1, int(r0_px + r1_px))
        draw.line([(x0, y0), (x1, y1)], fill=trunk_col, width=w)
        draw.ellipse([(x0 - r0_px, y0 - r0_px), (x0 + r0_px, y0 + r0_px)],
                     fill=trunk_col)
        draw.ellipse([(x1 - r1_px, y1 - r1_px), (x1 + r1_px, y1 + r1_px)],
                     fill=trunk_col)

    # draw leaves
    for lf in leaves:
        if isinstance(lf, dict):
            p = lf["p"]
            r = lf.get("r", leaf_radius) * s
            col = tuple(lf.get("col", leaf_col))
        else:
            p = lf; r = leaf_radius; col = leaf_col
        x, y = proj(p)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=col)

    if title:
        draw.rectangle([(0, 0), (W, title_h)], fill=(245, 240, 230))
        draw.text((10, 6), title, fill=(30, 30, 30))

    img.save(out_path, "PNG", optimize=True)
