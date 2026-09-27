"""FFXIV index (ID) maps: colorset rows for the UV layout, rasterized into a texture.

Dawntrail gear pairs every colorset with an _id texture.  Red selects the row pair
(0x00, 0x11, ... 0xFF for pairs 1-16) and green selects row A (0xFF) or row B (0x00),
so a pixel that no face covers (black) reads as row 1B.

Rows are numbered 0..31 here: 0 = 1A, 1 = 1B, 2 = 2A, ... 31 = 16B.

The automatic mode finds rows in one of two ways: per UV island (from the layout and the
mesh alone), or from the colours of a base colour texture, where every region of one
colour gets a row even inside an island.

UVs wrap like a repeating texture: parts of the layout outside 0..1 land on the
texture where the game samples them.  Images use Blender's convention (pixel row 0
is the bottom, UV v points up).
"""

import numpy as np

from .geom import connected_components, norm, relabel

ROW_NAMES = [f"{p}{c}" for p in range(1, 17) for c in "AB"]
BACKGROUND_ROW = 1                                     # 1B: black, what uncovered texture reads as
AUTO_ROWS = [r for r in range(32) if r != BACKGROUND_ROW]   # 1A, 2A, 2B, 3A, ... 16B


def row_rgb(row):
    """(R, G, B) bytes of a row."""
    pair, is_b = divmod(int(row), 2)
    return (pair * 0x11, 0x00 if is_b else 0xFF, 0x00)


def row_colors():
    """(32, 4) float RGBA of every row."""
    c = np.array([row_rgb(r) + (255,) for r in range(32)], np.float32)
    return c / 255.0


# ------------------------------------------------------------------------------------------
# islands
# ------------------------------------------------------------------------------------------
def weld_ids(P, eps):
    from .source import SourceMesh
    return SourceMesh._weld(np.asarray(P, np.float64), eps)[0]


def uv_islands(P, T, UV, weld=1e-5, uv_tol=1e-5):
    """Island id per triangle.  Two triangles are in the same island when they share an
    edge (vertices welded by position, so meshes split at hard edges still connect)
    and their UVs agree at both ends of it.
    P: (nv, 3) positions, T: (nt, 3) vertex ids, UV: (nt, 3, 2) corner UVs."""
    nt = len(T)
    if nt == 0:
        return np.zeros(0, np.int64)
    W = weld_ids(P, weld)[np.asarray(T, np.int64)]
    k0, k1 = [0, 1, 2], [1, 2, 0]
    a = W[:, k0].ravel()
    b = W[:, k1].ravel()
    ua = UV[:, k0].reshape(-1, 2)
    ub = UV[:, k1].reshape(-1, 2)
    tri = np.repeat(np.arange(nt), 3)
    swap = a > b
    lo = np.where(swap, b, a)
    hi = np.where(swap, a, b)
    ulo = np.where(swap[:, None], ub, ua)
    uhi = np.where(swap[:, None], ua, ub)
    ok = lo != hi
    q = np.round(np.concatenate([ulo, uhi], 1) / uv_tol).astype(np.int64)
    key = np.concatenate([lo[:, None], hi[:, None], q], 1)[ok]
    t = tri[ok]
    if len(key) == 0:
        return np.arange(nt)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    s = inv[order]
    same = s[1:] == s[:-1]
    to = t[order]
    return relabel(connected_components(nt, to[:-1][same], to[1:][same]))


def island_features(isl, P, T, UV):
    """Per island: triangle count, UV area, surface area, major UV extent and
    elongation (minor / major extent).  None of them change when an island is moved,
    rotated or mirrored."""
    n = int(isl.max()) + 1 if len(isl) else 0
    cnt = np.bincount(isl, minlength=n).astype(np.float64)
    e1 = UV[:, 1] - UV[:, 0]
    e2 = UV[:, 2] - UV[:, 0]
    auv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    X = np.asarray(P, np.float64)[np.asarray(T, np.int64)]
    a3 = 0.5 * norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]))
    area_uv = np.bincount(isl, auv, n)
    area_3d = np.bincount(isl, a3, n)
    # exact area moments: a uniform triangle has E[x x^T] = (sum v v^T + s s^T) / 12, s = sum v
    ws = np.maximum(area_uv, 1e-30)
    s = UV.sum(1)
    mx = np.bincount(isl, auv * s[:, 0], n) / 3.0 / ws
    my = np.bincount(isl, auv * s[:, 1], n) / 3.0 / ws
    sxx = (UV[:, :, 0] ** 2).sum(1) + s[:, 0] ** 2
    syy = (UV[:, :, 1] ** 2).sum(1) + s[:, 1] ** 2
    sxy = (UV[:, :, 0] * UV[:, :, 1]).sum(1) + s[:, 0] * s[:, 1]
    cxx = np.bincount(isl, auv * sxx, n) / 12.0 / ws - mx ** 2
    cyy = np.bincount(isl, auv * syy, n) / 12.0 / ws - my ** 2
    cxy = np.bincount(isl, auv * sxy, n) / 12.0 / ws - mx * my
    tr = 0.5 * (cxx + cyy)
    dt = np.sqrt(np.maximum(0.25 * (cxx - cyy) ** 2 + cxy ** 2, 0.0))
    ext1 = np.sqrt(np.maximum(tr + dt, 0.0))
    ext2 = np.sqrt(np.maximum(tr - dt, 0.0))
    elong = ext2 / np.maximum(ext1, 1e-30)
    return np.stack([cnt, area_uv, area_3d, ext1, elong], 1)


def similar_groups(feat, tol, order):
    """Greedy grouping: each island (in `order`) joins the closest group whose first
    member differs from it by less than `tol` on average (relative differences of
    counts, areas and size; plain difference of the elongation).
    Returns (group per island, first member per group)."""
    group = np.full(len(feat), -1, np.int64)
    reps = []
    for i in order:
        if reps:
            R = feat[reps]
            d = np.abs(R - feat[i])
            d[:, :4] /= np.maximum(np.maximum(np.abs(R[:, :4]), np.abs(feat[i, :4])), 1e-12)
            dm = d.mean(1)
            j = int(np.argmin(dm))
            if dm[j] < tol:
                group[i] = j
                continue
        group[i] = len(reps)
        reps.append(i)
    return group, np.array(reps, np.int64)


# ------------------------------------------------------------------------------------------
# rasterizing
# ------------------------------------------------------------------------------------------
def rasterize(UV, W, H, chunk=1 << 20):
    """Pixels whose centres lie inside each triangle, with UVs wrapping around.
    UV: (n, 3, 2).  Returns (triangle index, flat pixel index y * W + x)."""
    UV = np.asarray(UV, np.float64)
    n = len(UV)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    P = UV * np.array([W, H], np.float64)
    lo = P.min(1)
    hi = P.max(1)
    x0 = np.ceil(lo[:, 0] - 0.5).astype(np.int64)
    y0 = np.ceil(lo[:, 1] - 0.5).astype(np.int64)
    bw = np.floor(hi[:, 0] - 0.5).astype(np.int64) - x0 + 1
    bh = np.floor(hi[:, 1] - 0.5).astype(np.int64) - y0 + 1
    A, B, C = P[:, 0], P[:, 1], P[:, 2]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = (bw > 0) & (bh > 0) & (np.abs(d) > 1e-12) & (bw <= 4 * W) & (bh <= 4 * H)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    A, B, C, dd = A[idx], B[idx], C[idx], d[idx]
    x0, y0, bw, bh = x0[idx], y0[idx], bw[idx], bh[idx]
    # barycentric weights as affine functions of the pixel centre: l = a * x + b * y + c
    a0 = (B[:, 1] - C[:, 1]) / dd
    b0 = (C[:, 0] - B[:, 0]) / dd
    c0 = -(a0 * C[:, 0] + b0 * C[:, 1])
    a1 = (C[:, 1] - A[:, 1]) / dd
    b1 = (A[:, 0] - C[:, 0]) / dd
    c1 = -(a1 * C[:, 0] + b1 * C[:, 1])
    cnt = bw * bh
    cum = np.cumsum(cnt)
    out_t, out_p = [], []
    start = 0
    m = len(idx)
    while start < m:
        base = cum[start - 1] if start else 0
        end = max(int(np.searchsorted(cum, base + chunk, side="right")), start + 1)
        c = cnt[start:end]
        loc = np.repeat(np.arange(start, end), c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        w = bw[loc]
        x = x0[loc] + k % w
        y = y0[loc] + k // w
        px = x + 0.5
        py = y + 0.5
        l0 = a0[loc] * px + b0[loc] * py + c0[loc]
        l1 = a1[loc] * px + b1[loc] * py + c1[loc]
        eps = -1e-6
        inside = (l0 >= eps) & (l1 >= eps) & (1.0 - l0 - l1 >= eps)
        out_t.append(idx[loc[inside]])
        out_p.append((y[inside] % H) * W + (x[inside] % W))
        start = end
    return np.concatenate(out_t), np.concatenate(out_p)


def resolve(tri, pix, value, priority, W, H):
    """Label image (H, W) from rasterized triangles, -1 where nothing is drawn.  Where
    triangles overlap, the higher priority wins, then the later triangle."""
    lab = np.full(W * H, -1, np.int64)
    if len(tri):
        order = np.lexsort((tri, priority[tri], pix))
        ps = pix[order]
        last = np.r_[ps[1:] != ps[:-1], True]
        lab[ps[last]] = value[tri[order[last]]]
    return lab.reshape(H, W)


def dilate(lab, n):
    """Grow labelled regions (>= 0) into unlabelled pixels by n pixels (wrapping)."""
    for _ in range(int(n)):
        if not (lab < 0).any():
            break
        new = lab.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.roll(lab, (dy, dx), axis=(0, 1))
            take = (new < 0) & (nb >= 0)
            new[take] = nb[take]
        lab = new
    return lab


def overlap_pairs(tri, pix, isl, n_isl, min_share=0.5):
    """Pairs of islands that cover mostly the same pixels (stacked, e.g. mirrored
    halves sharing texture space): at least `min_share` of the smaller one.
    Returns (pairs a, pairs b, fraction of the covered pixels that several islands use)."""
    empty = np.zeros(0, np.int64)
    if len(tri) == 0:
        return empty, empty, 0.0
    u = np.sort(pix * n_isl + isl[tri])
    u = u[np.r_[True, u[1:] != u[:-1]]]
    p = u // n_isl
    i = u % n_isl
    same = p[1:] == p[:-1]
    n_pix = len(u) - int(same.sum())
    shared_px = len(np.unique(p[1:][same]))
    frac = shared_px / max(n_pix, 1)
    if not same.any():
        return empty, empty, frac
    npx = np.bincount(i, minlength=n_isl)
    keys, shared = np.unique(i[:-1][same] * n_isl + i[1:][same], return_counts=True)
    pa = keys // n_isl
    pb = keys % n_isl
    keep = shared >= min_share * np.minimum(npx[pa], npx[pb])
    return pa[keep], pb[keep], frac


# ------------------------------------------------------------------------------------------
# automatic rows
# ------------------------------------------------------------------------------------------
def auto_rows(P, T, UV, W, H, weld=1e-5, group_similar=False, tolerance=0.1, raster=None, max_rows=31):
    """Row per triangle for the automatic mode, one row per UV island.

    Islands that share texture space always share a row (they cannot differ in the
    texture anyway); with `group_similar`, islands of matching shape do too.  Groups are
    ordered by surface area, largest first, and take the rows 1A, 2A, 2B, 3A, ... 16B.
    With more groups than `max_rows` (at most 31), the most alike groups share a row.
    Returns (row per triangle, draw priority per triangle, stats).  The priority puts
    smaller groups on top where islands partly overlap."""
    isl = uv_islands(P, T, UV, weld)
    n_isl = int(isl.max()) + 1 if len(isl) else 0
    stats = {"islands": n_isl, "groups": 0, "rows": 0, "stacked": 0, "shared": 0.0}
    if n_isl == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), stats
    if raster is None:
        raster = rasterize(UV, W, H)
    tri, pix = raster
    ea, eb, stats["shared"] = overlap_pairs(tri, pix, isl, n_isl)
    stats["stacked"] = int(len(np.unique(np.r_[ea, eb])))
    feat = island_features(isl, P, T, UV)
    if group_similar and n_isl > 1:
        order = np.argsort(-feat[:, 2], kind="stable")
        sim, reps = similar_groups(feat, tolerance, order)
        ea = np.r_[ea, np.arange(n_isl)]
        eb = np.r_[eb, reps[sim]]
    grp = relabel(connected_components(n_isl, ea, eb))
    n_grp = int(grp.max()) + 1
    stats["groups"] = n_grp
    max_rows = max(min(int(max_rows), len(AUTO_ROWS)), 1)
    if n_grp > max_rows:
        grp = fit_rows(grp, feat, max_rows)
        n_grp = int(grp.max()) + 1
    stats["rows"] = n_grp
    garea = np.bincount(grp, feat[:, 2], n_grp)
    rank = np.empty(n_grp, np.int64)
    rank[np.argsort(-garea, kind="stable")] = np.arange(n_grp)
    rows = np.array(AUTO_ROWS, np.int64)[rank]
    return rows[grp[isl]], rank[grp[isl]], stats


def fit_rows(grp, feat, max_rows=31):
    """Merge island groups until at most `max_rows` are left: the most alike groups first
    (triangle count, areas, size and elongation of their largest island), and small groups
    before large ones."""
    n = int(grp.max()) + 1
    first = first_per_key(grp, -feat[:, 2])
    f = feat[first]
    X = np.log(np.maximum(f[:, :4], 1e-30))
    X[:, 1:3] /= 2.0                     # areas: log of a squared size
    X = np.column_stack([X, f[:, 4]])
    return ward_merge(X, np.bincount(grp, feat[:, 2], n), 0.0, max_rows)[grp]


def paint(rows_per_tri, raster, W, H, padding=0, priority=None):
    """RGBA float image (H, W, 4) and the label image for the given triangle rows."""
    tri, pix = raster
    rows = np.asarray(rows_per_tri, np.int64)
    if priority is None:
        priority = np.zeros(len(rows), np.int64)
    return paint_rows(resolve(tri, pix, rows, np.asarray(priority, np.int64), W, H), padding)


def paint_rows(lab, padding=0):
    """RGBA float image (H, W, 4) and the padded label image for a row per texel (-1: none)."""
    lab = dilate(lab, padding)
    img = row_colors()[np.where(lab >= 0, lab, BACKGROUND_ROW)]
    return img, lab


# ------------------------------------------------------------------------------------------
# clustering
# ------------------------------------------------------------------------------------------
def ward_merge(C, w, tol=0.0, max_k=None):
    """Agglomerative (Ward) clustering of weighted points C (n, d): the cheapest merge first
    as long as the two centres are closer than `tol`, then on until at most `max_k`
    clusters are left.  Returns the cluster (0..k-1) of every point."""
    C = np.array(C, np.float64)
    w = np.maximum(np.array(w, np.float64), 1e-12)
    n = len(C)
    if n == 0:
        return np.zeros(0, np.int64)
    max_k = n if max_k is None else max(int(max_k), 1)
    tol2 = float(tol) ** 2
    grp = np.arange(n)
    live = np.ones(n, bool)
    d2 = ((C[:, None] - C[None]) ** 2).sum(-1)
    cost = w[:, None] * w[None] / (w[:, None] + w[None]) * d2
    np.fill_diagonal(cost, np.inf)
    alive = n
    while alive > 1:
        if alive > max_k:
            k = int(np.argmin(cost))
        elif tol2 > 0:
            k = int(np.argmin(np.where(d2 < tol2, cost, np.inf)))
        else:
            break
        i, j = divmod(k, n)
        if not np.isfinite(cost[i, j]) or (alive <= max_k and d2[i, j] >= tol2):
            break
        C[i] = (C[i] * w[i] + C[j] * w[j]) / (w[i] + w[j])
        w[i] += w[j]
        live[j] = False
        grp[grp == j] = i
        alive -= 1
        dd = ((C - C[i]) ** 2).sum(1)
        cc = w * w[i] / (w + w[i]) * dd
        cc[~live] = np.inf
        cc[i] = np.inf
        d2[i], d2[:, i] = dd, dd
        cost[i], cost[:, i] = cc, cc
        cost[j], cost[:, j] = np.inf, np.inf
    return relabel(grp)


def nearest(X, C, k=1, chunk=1 << 18):
    """Indices and distances (n, k) of the k nearest centres C to every row of X, nearest first."""
    X = np.asarray(X, np.float64)
    k = max(min(int(k), len(C)), 1)
    cc = (C ** 2).sum(1)
    idx = np.empty((len(X), k), np.int64)
    dist = np.empty((len(X), k))
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d = np.maximum((x ** 2).sum(1)[:, None] - 2.0 * (x @ C.T) + cc, 0.0)
        if k == 1:
            i = d.argmin(1)[:, None]
            idx[s:s + chunk] = i
            dist[s:s + chunk] = np.sqrt(np.take_along_axis(d, i, 1))
            continue
        if k < len(C):
            i = np.argpartition(d, k - 1, axis=1)[:, :k]
        else:
            i = np.broadcast_to(np.arange(len(C)), d.shape)
        dk = np.take_along_axis(d, i, 1)
        o = np.argsort(dk, axis=1, kind="stable")
        idx[s:s + chunk] = np.take_along_axis(i, o, 1)
        dist[s:s + chunk] = np.sqrt(np.take_along_axis(dk, o, 1))
    return idx, dist


def kmeans(X, k, iters=15, sample=60000, seed=0):
    """k-means++ centres of the rows of X, fitted on a fixed random sample of at most
    `sample` rows (the same input always gives the same centres)."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), sample, replace=False)] if len(X) > sample else np.asarray(X, np.float64)
    k = max(min(int(k), len(S)), 1)
    C = np.empty((k, S.shape[1]))
    C[0] = S[rng.integers(len(S))]
    d2 = ((S - C[0]) ** 2).sum(1)
    m = 1
    while m < k and d2.sum() > 1e-12:
        C[m] = S[rng.choice(len(S), p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((S - C[m]) ** 2).sum(1))
        m += 1
    C = C[:m]
    for _ in range(iters):
        lab = nearest(S, C)[0][:, 0]
        cnt = np.bincount(lab, minlength=m)
        new = np.stack([np.bincount(lab, S[:, j], m) for j in range(S.shape[1])], 1)
        ok = cnt > 0
        new[ok] /= cnt[ok, None]
        new[~ok] = C[~ok]
        if np.allclose(new, C):
            break
        C = new
    return C[np.bincount(nearest(S, C)[0][:, 0], minlength=m) > 0]


# ------------------------------------------------------------------------------------------
# texture regions
# ------------------------------------------------------------------------------------------
WORK_LIMIT = 1024          # texture regions are found at most at this resolution
_N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_N8 = _N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """CIE L*a*b* (D65) of sRGB-encoded colours in 0..1 (last axis)."""
    c = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (c @ _SRGB_TO_XYZ.T) / _WHITE
    e = 216.0 / 24389.0
    f = np.where(t > e, np.cbrt(np.maximum(t, e)), (24389.0 / 27.0 * t + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    """sRGB-encoded colours in 0..1 of CIE L*a*b* colours (clipped to the sRGB gamut)."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], -1)
    t = np.where(f > 6.0 / 29.0, f ** 3, 3.0 * (6.0 / 29.0) ** 2 * (f - 4.0 / 29.0)) * _WHITE
    c = np.clip(t @ np.linalg.inv(_SRGB_TO_XYZ).T, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def work_size(W, H, limit=WORK_LIMIT):
    """Resolution the texture regions are found at for a W x H index map."""
    s = min(1.0, limit / max(W, H))
    return max(int(round(W * s)), 1), max(int(round(H * s)), 1)


def resample(img, w, h):
    """Image (H, W, C) resized to (h, w, C): every new texel is the average of the old
    texels it covers (partly covered ones in proportion), so shrinking never aliases."""
    img = np.asarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float64)
    return _resample_axis(_resample_axis(img, h, 0), w, 1)


def _resample_axis(a, n_out, axis):
    n_in = a.shape[axis]
    if n_in == n_out:
        return a
    if n_in % n_out == 0:
        f = n_in // n_out
        shape = a.shape[:axis] + (n_out, f) + a.shape[axis + 1:]
        return a.reshape(shape).mean(axis + 1, dtype=np.float64).astype(a.dtype)
    if n_out % n_in == 0:
        return np.repeat(a, n_out // n_in, axis)
    # area integral: cumulative sums, read between the (fractional) cell edges
    a = np.moveaxis(a, axis, 0)
    S = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0, dtype=np.float64)], 0)
    e = np.arange(n_out + 1) * (n_in / n_out)
    i = np.minimum(np.floor(e).astype(np.int64), n_in - 1)
    frac = (e - i).reshape((-1,) + (1,) * (a.ndim - 1))
    I = S[i] + frac * a[i]
    out = (I[1:] - I[:-1]) / (n_in / n_out)
    return np.moveaxis(out.astype(a.dtype), 0, axis)


def _shift(a, dy, dx):
    """a[y + dy, x + dx] at (y, x), wrapping like the texture."""
    return np.roll(a, (-dy, -dx), (0, 1))


def smooth_colors(F, M, dom, r, limit, passes=1):
    """Edge-preserving blur: every texel of M takes the average of the texels of M (same
    domain) within a (2r + 1)^2 box whose colour differs from its own by less than `limit`,
    `passes` times.  Grain, grime and noise even out; edges stay sharp."""
    if r <= 0:
        return F
    F = F.astype(np.float32)
    lim2 = float(limit) ** 2
    same = {(dy, dx): M & _shift(M, dy, dx) & (_shift(dom, dy, dx) == dom)
            for dy in range(-r, r + 1) for dx in range(-r, r + 1)}
    for _ in range(passes):
        acc = np.zeros_like(F)
        wt = np.zeros(F.shape[:2], np.float32)
        for (dy, dx), ok0 in same.items():
            Fn = _shift(F, dy, dx)
            ok = ok0 & (((Fn - F) ** 2).sum(-1) < lim2)
            acc += Fn * ok[..., None]
            wt += ok
        F = np.where(wt[..., None] > 0, acc / np.maximum(wt, 1.0)[..., None], F)
    return F.astype(np.float64)


def texel_components(lab, dom):
    """Connected regions (4-neighbours, wrapping) of texels with the same label in the same
    domain.  Returns (region per texel, -1 where lab < 0; texel count per region)."""
    h, w = lab.shape
    M = lab >= 0
    idx = np.arange(h * w).reshape(h, w)
    a, b = [], []
    for dy, dx in _N4[:2]:
        m = M & _shift(M, dy, dx) & (lab == _shift(lab, dy, dx)) & (dom == _shift(dom, dy, dx))
        a.append(idx[m])
        b.append(_shift(idx, dy, dx)[m])
    cc = connected_components(h * w, np.concatenate(a), np.concatenate(b))
    reg = np.full(h * w, -1, np.int64)
    _, inv, area = np.unique(cc[M.ravel()], return_inverse=True, return_counts=True)
    reg[M.ravel()] = inv.ravel()
    return reg.reshape(h, w), area


def _border_pairs(reg, dom):
    """Flat texel ids (p, q) of 4-neighbours in different regions of the same domain."""
    h, w = reg.shape
    idx = np.arange(h * w).reshape(h, w)
    ps, qs = [], []
    for dy, dx in _N4[:2]:
        nreg = _shift(reg, dy, dx)
        m = (reg >= 0) & (nreg >= 0) & (nreg != reg) & (dom == _shift(dom, dy, dx))
        ps.append(idx[m])
        qs.append(_shift(idx, dy, dx)[m])
    return np.concatenate(ps), np.concatenate(qs)


def first_per_key(keys, *sort_by):
    """Index of the first element per key after sorting by (key, *sort_by)."""
    order = np.lexsort(tuple(reversed(sort_by)) + (keys,))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    return order[first]


def fill_within(lab, M, dom, limit):
    """Give texels of M without a label (-1) the label of the nearest labelled texel of the
    same domain, growing one texel per step for at most `limit` steps."""
    lab = lab.copy()
    todo = M & (lab < 0)
    same = [_shift(dom, dy, dx) == dom for dy, dx in _N4]
    for _ in range(int(limit)):
        if not todo.any():
            break
        grown = False
        for (dy, dx), sm in zip(_N4, same):
            nl = _shift(lab, dy, dx)
            take = todo & (nl >= 0) & sm
            if take.any():
                lab[take] = nl[take]
                todo &= ~take
                grown = True
        if not grown:
            break
    return lab


def fill_patches(lab, M, dom):
    """Texels of M without a label (-1) take, per connected patch of them, the most common
    label around the patch (same domain).  Patches with no labelled neighbour stay -1."""
    todo = M & (lab < 0)
    if not todo.any():
        return lab
    reg, _ = texel_components(np.where(todo, 0, -1), dom)
    rs, ls = [], []
    for dy, dx in _N4:
        nl = _shift(lab, dy, dx)
        m = todo & (nl >= 0) & (_shift(dom, dy, dx) == dom)
        rs.append(reg[m])
        ls.append(nl[m])
    r, l = np.concatenate(rs), np.concatenate(ls)
    if len(r) == 0:
        return lab
    nl_ = int(lab.max()) + 1
    key, cnt = np.unique(r * nl_ + l, return_counts=True)
    best = first_per_key(key // nl_, -cnt)
    fill = np.full(int(reg.max()) + 1, -1, np.int64)
    fill[key[best] // nl_] = key[best] % nl_
    lab = lab.copy()
    lab[todo] = fill[reg[todo]]
    return lab


def has_core(reg, n, radius):
    """Regions 0..n-1 that are wider than 2 * radius texels somewhere: at least one texel
    keeps all neighbours within `radius` steps (4-neighbourhood) in its region."""
    core = reg >= 0
    for _ in range(int(radius)):
        nxt = core.copy()
        for dy, dx in _N4:
            nxt &= _shift(core, dy, dx) & (_shift(reg, dy, dx) == reg)
        core = nxt
    out = np.zeros(n, bool)
    out[reg[core]] = True
    return out


def absorb_small(lab, dom, min_area, radius=0, rounds=6):
    """Regions smaller than `min_area` texels, or nowhere wider than 2 * radius texels
    (thin lines, anti-aliased bands along edges), take the label of the larger neighbouring
    region (same domain) they share the longest border with.  Only ever toward a larger
    region, so two small neighbours never swap."""
    for _ in range(rounds):
        reg, area = texel_components(lab, dom)
        small = area < min_area
        if radius > 0:
            small |= ~has_core(reg, len(area), radius)
        if not small.any():
            break
        p, q = _border_pairs(reg, dom)
        a, b = reg.ravel()[p], reg.ravel()[q]
        a, b = np.r_[a, b], np.r_[b, a]
        ok = small[a] & ((area[b] > area[a]) | ((area[b] == area[a]) & (b > a)))
        if not ok.any():
            break
        n = len(area)
        key, votes = np.unique(a[ok] * n + b[ok], return_counts=True)
        ka, kb = key // n, key % n
        best = first_per_key(ka, -votes, -area[kb])
        target = np.full(n, -1, np.int64)
        target[ka[best]] = kb[best]
        M = reg >= 0
        reg_lab = np.empty(n, np.int64)
        reg_lab[reg[M]] = lab[M]
        move = M & (target[np.maximum(reg, 0)] >= 0)
        lab = lab.copy()
        lab[move] = reg_lab[target[reg[move]]]
    return lab


def potts_smooth(lab, cand, cost, M, dom, F, beta, sigma, iters=5):
    """Iterated conditional modes on a contrast-sensitive Potts model.  Every texel of M
    (row-major order, matching `cand`/`cost` (n, k)) picks the candidate label with the least
    colour cost plus `beta` per 8-neighbour of the same domain that has another label;
    neighbours across a colour edge count less, so borders settle on edges."""
    h, w = lab.shape
    ys, xs = np.nonzero(M)
    flat = ys * w + xs
    links = []
    Ff = F.reshape(-1, F.shape[-1])
    for dy, dx in _N8:
        nb = (((ys + dy) % h) * w + (xs + dx) % w).astype(np.int32)
        c2 = ((Ff[flat] - Ff[nb]) ** 2).sum(1)
        wt = np.exp(-c2 / (2.0 * sigma * sigma)) * (dom.ravel()[nb] == dom.ravel()[flat])
        links.append((nb, (wt * (0.7071 if dy and dx else 1.0) * beta).astype(np.float32)))
    cur = lab.ravel().copy()
    rows = np.arange(len(flat))
    for _ in range(iters):
        e = cost.astype(np.float32)
        for nb, wt in links:
            nl = cur[nb]
            e += wt[:, None] * ((nl[:, None] >= 0) & (nl[:, None] != cand))
        new = cand[rows, e.argmin(1)]
        changed = int(np.count_nonzero(new != cur[flat]))
        cur[flat] = new
        if changed <= len(flat) // 1000:
            break
    return cur.reshape(h, w)


def _edge_strength(F, M, dom, p, q, reach=2):
    """Strongest colour step between a texel up to `reach` texels behind p and one up to
    `reach` beyond q (4-neighbours, same domain): how sharp the colour changes at a region
    border, even where the border runs a texel or two beside the actual edge."""
    h, w = M.shape
    py, px = np.divmod(p, w)
    qy, qx = np.divmod(q, w)
    dy = (qy - py + h // 2) % h - h // 2
    dx = (qx - px + w // 2) % w - w // 2
    d0 = dom[py, px]
    best = np.zeros(len(p))
    for j in range(reach + 1):
        ay, ax = (py - j * dy) % h, (px - j * dx) % w
        ok_a = M[ay, ax] & (dom[ay, ax] == d0)
        for k in range(reach + 1):
            by, bx = (qy + k * dy) % h, (qx + k * dx) % w
            ok = ok_a & M[by, bx] & (dom[by, bx] == d0)
            best = np.maximum(best, np.where(ok, norm(F[ay, ax] - F[by, bx]), 0.0))
    return best


def merge_gradients(lab, dom, F, centres, ratio=0.35):
    """Join neighbouring regions whose shared border is a gradual colour change (baked
    shading, a fade) rather than an edge: the median edge strength along the border is below
    `ratio` times the distance between the two classes.  A joined group takes the class of
    its largest region."""
    reg, area = texel_components(lab, dom)
    p, q = _border_pairs(reg, dom)
    if len(p) == 0:
        return lab
    n = len(area)
    M = reg >= 0
    a, b = reg.ravel()[p], reg.ravel()[q]
    strength = _edge_strength(F, M, dom, p, q)
    key, kinv, klen = np.unique(np.minimum(a, b) * n + np.maximum(a, b),
                                return_inverse=True, return_counts=True)
    order = np.lexsort((strength, kinv.ravel()))
    median = strength[order][np.r_[0, np.cumsum(klen)[:-1]] + klen // 2]
    ka, kb = key // n, key % n
    reg_lab = np.empty(n, np.int64)
    reg_lab[reg[M]] = lab[M]
    gap = norm(centres[reg_lab[ka]] - centres[reg_lab[kb]])
    soft = (median < ratio * gap) & (klen >= 3)
    if not soft.any():
        return lab
    grp = connected_components(n, ka[soft], kb[soft])
    big = first_per_key(grp, -area)
    glab = np.empty(n, np.int64)
    glab[grp[big]] = reg_lab[big]
    out = lab.copy()
    out[M] = glab[grp[reg[M]]]
    return out


def class_means(lab, F, n):
    """Mean of F (h, w, c) per label 0..n-1 and texel count per label."""
    M = lab >= 0
    ids = lab[M]
    cnt = np.bincount(ids, minlength=n).astype(np.float64)
    X = F[M]
    means = np.stack([np.bincount(ids, X[:, j], n) for j in range(X.shape[1])], 1)
    return means / np.maximum(cnt, 1.0)[:, None], cnt


def texture_classes(tex, dom, tol=12.0, detail=4.0, max_classes=31):
    """Material classes of the texels in use, from the colours of a base colour texture.

    tex: (h, w, 3 or 4) sRGB colours 0..1; dom: (h, w) texture island of every texel
    (-1: unused).  Colours are compared in CIE Lab: `tol` (delta E) is the smallest colour
    difference that keeps two classes apart.  Regions never cross island borders; regions
    smaller than about `detail` texels across join their surroundings, and regions that
    fade into a neighbour (baked shading) join it.  Texels with alpha below 0.5 (see-through
    lace, holes) take the class of the nearest solid texel.
    Returns (class per texel, -1 where unused; mean Lab colour per class)."""
    h, w = dom.shape
    M = dom >= 0
    lab = np.full((h, w), -1, np.int64)
    if not M.any():
        return lab, np.zeros((0, 3))
    solid = M
    if tex.shape[2] >= 4:
        solid = M & (tex[..., 3] >= 0.5)
        if solid.sum() < 0.05 * M.sum():
            solid = M
    Lab = srgb_to_lab(tex[..., :3])
    F0 = smooth_colors(Lab, solid, dom, 1, 2.0 * tol, 2)        # grain evened out, edges kept
    r = max(int(round(detail / 2.0)), 1)
    F = smooth_colors(F0, solid, dom, r, 2.0 * tol, 2) if r > 1 else F0
    X = F[solid]
    C = kmeans(X, 40)
    wk = np.bincount(nearest(X, C)[0][:, 0], minlength=len(C))
    grp = ward_merge(C, wk, tol, max_classes)
    n = int(grp.max()) + 1
    centres = np.stack([np.bincount(grp, C[:, j] * wk, n) for j in range(3)], 1)
    centres /= np.maximum(np.bincount(grp, wk, n), 1e-12)[:, None]
    cand, dist = nearest(X, centres, 3)
    lab[solid] = cand[:, 0]
    if n > 1:
        lab = potts_smooth(lab, cand, dist / tol, solid, dom, F0, 1.0, tol / 2.0)
    min_area = max(4.0 * detail * detail, 4.0)
    radius = max(int(round(detail / 4.0)), 1)
    lab = absorb_small(lab, dom, min_area, radius)
    if n > 1:
        lab = merge_gradients(lab, dom, F0, class_means(lab, F, n)[0])
        lab = absorb_small(lab, dom, min_area, radius)
        # classes that became alike (e.g. shading joined) merge
        means, cnt = class_means(lab, F, n)
        live = cnt > 0
        g2 = np.full(n, -1, np.int64)
        g2[live] = ward_merge(means[live], cnt[live], tol, max_classes)
        lab = np.where(lab >= 0, g2[np.maximum(lab, 0)], -1)
    # see-through texels: the class around them; islands without solid texels: nearest colour
    lab = fill_patches(lab, M, dom)
    n = int(lab.max()) + 1 if (lab >= 0).any() else 0
    means, cnt = class_means(np.where(solid, lab, -1), Lab, max(n, 1))
    rest = M & (lab < 0)
    if rest.any():
        if n == 0:
            lab[rest] = 0
            n = 1
            means, cnt = class_means(lab, Lab, 1)
        else:
            lab[rest] = nearest(Lab[rest], means)[0][:, 0]
    if (cnt == 0).any():                 # classes seen only through see-through texels
        alt = class_means(lab, Lab, n)[0]
        means[cnt == 0] = alt[cnt == 0]
    return lab, means


def island_majority(cls, dom):
    """Every domain (texture island) takes its most common class."""
    M = (dom >= 0) & (cls >= 0)
    nc = int(cls.max()) + 1
    key, cnt = np.unique(dom[M] * nc + cls[M], return_counts=True)
    kd, kc = key // nc, key % nc
    best = first_per_key(kd, -cnt)
    maj = np.full(int(dom.max()) + 1, -1, np.int64)
    maj[kd[best]] = kc[best]
    return np.where(dom >= 0, maj[np.maximum(dom, 0)], -1)


def upsample_labels(lab, dom_lo, dom_hi, tex, means):
    """Labels found at a lower resolution, carried to the index map's resolution: every
    texel takes the label under it if it lies on the same texture island there, else that of
    the nearest texel of its island, else the class closest to its colour."""
    h, w = lab.shape
    H, W = dom_hi.shape
    iy = np.minimum(((np.arange(H) + 0.5) * h / H).astype(np.int64), h - 1)
    ix = np.minimum(((np.arange(W) + 0.5) * w / W).astype(np.int64), w - 1)
    src = lab[iy[:, None], ix[None, :]]
    M = dom_hi >= 0
    out = np.where(M & (dom_lo[iy[:, None], ix[None, :]] == dom_hi), src, -1)
    out = fill_within(out, M, dom_hi, limit=int(np.ceil(max(H / h, W / w))) * 2 + 2)
    rest = M & (out < 0)
    if rest.any() and len(means):
        ry, rx = np.nonzero(rest)
        out[ry, rx] = nearest(srgb_to_lab(tex[iy[ry], ix[rx], :3]), means)[0][:, 0]
    return out


def texture_rows(P, T, UV, W, H, tex, weld=1e-5, tolerance=12.0, detail=0.004, max_rows=31,
                 whole_islands=False, raster=None):
    """Rows for the automatic mode from a base colour texture.

    tex: (h, w, C) sRGB colours 0..1 at work_size(W, H) (Blender's row order).  Regions of
    one colour get one row, also inside an island; islands that share texture space are
    treated as one.  `detail` is the size of the smallest region kept, as a fraction of the
    texture's longer side.  With `whole_islands`, every island takes the row of its main
    colour.  Classes are ordered by area, largest first: 1A, 2A, 2B, 3A, ... 16B.
    Returns (row per texel (H, W), -1 where unused; row per triangle (its most common row);
    stats, with "colors": [(row, sRGB 0..1, share of the painted area)] by row)."""
    nt = len(T)
    isl = uv_islands(P, T, UV, weld)
    n_isl = int(isl.max()) + 1 if nt else 0
    stats = {"islands": n_isl, "groups": 0, "stacked": 0, "shared": 0.0, "colors": []}
    if raster is None:
        raster = rasterize(UV, W, H)
    if n_isl == 0:
        return np.full((H, W), -1, np.int64), np.zeros(0, np.int64), stats
    tri, pix = raster
    ea, eb, stats["shared"] = overlap_pairs(tri, pix, isl, n_isl)
    stats["stacked"] = int(len(np.unique(np.r_[ea, eb])))
    tisl = relabel(connected_components(n_isl, ea, eb))[isl]     # texture island per triangle
    flat = np.zeros(nt, np.int64)
    h, w = tex.shape[:2]
    dom_hi = resolve(tri, pix, tisl, flat, W, H)
    if (w, h) == (W, H):
        dom_lo = dom_hi
    else:
        tl, pl = rasterize(UV, w, h)
        dom_lo = resolve(tl, pl, tisl, flat, w, h)
    cls, means = texture_classes(tex, dom_lo, tolerance, detail * max(w, h), min(max_rows, len(AUTO_ROWS)))
    if whole_islands:
        cls = island_majority(cls, dom_lo)
    cls = cls if dom_lo is dom_hi else upsample_labels(cls, dom_lo, dom_hi, tex, means)
    M = cls >= 0
    n = len(means)
    area = np.bincount(cls[M], minlength=n)
    order = np.argsort(-area, kind="stable")
    order = order[area[order] > 0]
    row_of = np.full(n, BACKGROUND_ROW, np.int64)
    row_of[order] = np.array(AUTO_ROWS, np.int64)[np.arange(len(order))]
    lab = np.where(M, row_of[np.maximum(cls, 0)], -1)
    stats["groups"] = stats["rows"] = len(order)
    rgb = lab_to_srgb(means)
    total = max(int(area.sum()), 1)
    stats["colors"] = [(int(row_of[c]), tuple(float(v) for v in rgb[c]), area[c] / total) for c in order]
    return lab, triangle_rows(lab, raster, isl), stats


def triangle_rows(lab, raster, isl):
    """Most common row among the texels of every triangle; triangles too small to cover a
    texel take the most common row of their island."""
    tri, pix = raster
    nt = len(isl)
    r = lab.ravel()[pix]
    ok = r >= 0
    out = np.full(nt, -1, np.int64)
    if ok.any():
        key, cnt = np.unique(tri[ok] * 32 + r[ok], return_counts=True)
        best = first_per_key(key // 32, -cnt)
        out[key[best] // 32] = key[best] % 32
    miss = out < 0
    if miss.any() and (~miss).any():
        n_isl = int(isl.max()) + 1
        key, cnt = np.unique(isl[~miss] * 32 + out[~miss], return_counts=True)
        best = first_per_key(key // 32, -cnt)
        maj = np.full(n_isl, -1, np.int64)
        maj[key[best] // 32] = key[best] % 32
        out[miss] = maj[isl[miss]]
    return np.where(out >= 0, out, BACKGROUND_ROW)
