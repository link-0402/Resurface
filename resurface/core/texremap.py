"""Move a texture from one UV layout to another (per-triangle resampling).

For every texel of every triangle in the new layout, the matching point of the old
layout is found by barycentric interpolation and the old texture is sampled there.

* COLOR / DATA: bilinear sampling;
* INDEX: nearest sampling, for index / ID maps (FFXIV colorset row maps) whose
  values must never be blended;
* NORMAL: bilinear, and the tangent-space XY (R, G) is rotated by the per-triangle
  change of UV orientation (other channels, which FFXIV reuses for opacity or
  colorset data, are copied as they are).

Images use Blender's convention: pixel row 0 is the bottom, UV v points up.
"""

import numpy as np

from .indexmap import rasterize


def _sample(img, uv, nearest):
    H, W = img.shape[:2]
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5
    if nearest:
        xi = np.clip(np.round(x).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(y).astype(np.int64), 0, H - 1)
        return img[yi, xi]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y0 + 1, 0, H - 1)
    return (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x1c] * fx * (1 - fy)
            + img[y1c, x0c] * (1 - fx) * fy + img[y1c, x1c] * fx * fy)


def _rotations(new_uv, old_uv):
    """Orthogonal part (rotation / reflection) of the old->new UV map per triangle."""
    N = np.stack([new_uv[:, 1] - new_uv[:, 0], new_uv[:, 2] - new_uv[:, 0]], axis=2)
    O = np.stack([old_uv[:, 1] - old_uv[:, 0], old_uv[:, 2] - old_uv[:, 0]], axis=2)
    R = np.tile(np.eye(2), (len(N), 1, 1))
    ok = np.abs(np.linalg.det(O)) > 1e-14
    A = N[ok] @ np.linalg.inv(O[ok])
    u, s, vt = np.linalg.svd(A)
    R[ok] = u @ vt
    return R


def _moved(a, dy, dx):
    """a shifted by (dy, dx) pixels without wrapping around the edges (the vacated border is 0)."""
    r = np.roll(a, (dy, dx), axis=(0, 1))
    if dy:
        r[slice(None, dy) if dy > 0 else slice(dy, None)] = 0
    if dx:
        r[:, slice(None, dx) if dx > 0 else slice(dx, None)] = 0
    return r


def remap(img, new_uv, old_uv, out_w, out_h, mode="COLOR", flip_green=False, padding=8, chunk=1 << 20):
    """img: (H, W, C) float array.  new_uv / old_uv: (n, 3, 2) per triangle corner.
    Returns (out (out_h, out_w, C) float32, filled mask)."""
    C = img.shape[2]
    out = np.zeros((out_h * out_w, C), np.float32)
    filled = np.zeros(out_h * out_w, bool)
    nearest = mode == "INDEX"
    rots = _rotations(new_uv, old_uv) if mode == "NORMAL" else None
    tri, pix = rasterize(new_uv, out_w, out_h, wrap=False)
    # where the new layout overlaps itself, the later triangle wins
    order = np.lexsort((tri, pix))
    ps = pix[order]
    last = order[np.r_[ps[1:] != ps[:-1], True]] if len(ps) else order
    tri, pix = tri[last], pix[last]
    P = new_uv * np.array([out_w, out_h], np.float64)
    for s in range(0, len(tri), chunk):
        t = tri[s:s + chunk]
        X = pix[s:s + chunk] % out_w + 0.5
        Y = pix[s:s + chunk] // out_w + 0.5
        a, b, c = P[t, 0], P[t, 1], P[t, 2]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        l0 = ((b[:, 1] - c[:, 1]) * (X - c[:, 0]) + (c[:, 0] - b[:, 0]) * (Y - c[:, 1])) / d
        l1 = ((c[:, 1] - a[:, 1]) * (X - c[:, 0]) + (a[:, 0] - c[:, 0]) * (Y - c[:, 1])) / d
        w = np.stack([l0, l1, 1.0 - l0 - l1], 1)
        o = np.einsum("nk,nkj->nj", w, old_uv[t])
        vals = _sample(img, o, nearest)
        if rots is not None:
            xy = vals[:, :2] * 2.0 - 1.0
            if flip_green:
                xy[:, 1] = -xy[:, 1]
            xy = np.einsum("nij,nj->ni", rots[t], xy)
            if flip_green:
                xy[:, 1] = -xy[:, 1]
            vals = vals.copy()
            vals[:, :2] = xy * 0.5 + 0.5
        out[pix[s:s + chunk]] = vals
        filled[pix[s:s + chunk]] = True
    out = out.reshape(out_h, out_w, C)
    filled = filled.reshape(out_h, out_w)
    # pad islands outwards so filtering and mip maps do not pull in background
    for _ in range(int(padding)):
        acc = np.zeros_like(out)
        cnt = np.zeros(filled.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sm = _moved(filled, dy, dx)
            acc += _moved(out, dy, dx) * sm[..., None]
            cnt += sm
        grow = (~filled) & (cnt > 0)
        if not grow.any():
            break
        if nearest:
            # keep index values exact: copy one neighbour instead of averaging
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sm = _moved(filled, dy, dx) & grow & (out.sum(-1) == 0)
                out[sm] = _moved(out, dy, dx)[sm]
        else:
            out[grow] = acc[grow] / cnt[grow][:, None]
        filled |= grow
    return out, filled
