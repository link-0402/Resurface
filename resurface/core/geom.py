"""Vectorized geometry kernels (numpy only)."""

import numpy as np


def dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def norm(a):
    return np.sqrt(dot(a, a))


def normalize(a, eps=1e-30):
    n = norm(a)
    return a / np.maximum(n, eps)[..., None], n


def face_normals(V, F):
    """Unnormalized face normals (length = 2 * area)."""
    a = V[F[:, 0]]
    return np.cross(V[F[:, 1]] - a, V[F[:, 2]] - a)


def closest_point_triangle(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p (broadcasting, last axis = 3).

    Returns (q, bary) where bary holds the weights of a, b, c.
    Region logic follows Ericson, Real-Time Collision Detection, 5.1.5; it degrades
    gracefully on degenerate triangles.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    shape = np.broadcast(d1, d2).shape
    u = np.empty(shape)
    v = np.empty(shape)
    w = np.empty(shape)

    # interior
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    v[...] = vb / denom
    w[...] = vc / denom
    u[...] = 1.0 - v - w

    def setw(mask, uu, vv, ww):
        u[mask] = uu[mask] if isinstance(uu, np.ndarray) else uu
        v[mask] = vv[mask] if isinstance(vv, np.ndarray) else vv
        w[mask] = ww[mask] if isinstance(ww, np.ndarray) else ww

    # Apply regions in reverse priority so the first matching test in Ericson's order wins.
    # edge BC
    e43 = d4 - d3
    e56 = d5 - d6
    m = (va <= 0) & (e43 >= 0) & (e56 >= 0)
    t = e43 / np.where(np.abs(e43 + e56) < 1e-300, 1e-300, e43 + e56)
    setw(m, 0.0, 1.0 - t, t)
    # edge AC
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-300, 1e-300, d2 - d6)
    setw(m, 1.0 - t, 0.0, t)
    # vertex C
    m = (d6 >= 0) & (d5 <= d6)
    setw(m, 0.0, 0.0, 1.0)
    # edge AB
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-300, 1e-300, d1 - d3)
    setw(m, 1.0 - t, t, 0.0)
    # vertex B
    m = (d3 >= 0) & (d4 <= d3)
    setw(m, 0.0, 1.0, 0.0)
    # vertex A
    m = (d1 <= 0) & (d2 <= 0)
    setw(m, 1.0, 0.0, 0.0)

    bary = np.stack([u, v, w], axis=-1)
    q = a * u[..., None] + b * v[..., None] + c * w[..., None]
    return q, bary


def closest_point_segment(p, a, b):
    """Closest points on segments (a, b); returns (q, t) with q = a + t (b - a)."""
    ab = b - a
    l2 = dot(ab, ab)
    t = dot(p - a, ab) / np.where(l2 < 1e-300, 1e-300, l2)
    t = np.clip(t, 0.0, 1.0)
    return a + ab * t[..., None], t


def unique_edges(F):
    """Unique undirected edges of a triangle array.

    Returns (E, inv) where E is (ne, 2) sorted vertex pairs and inv is (nf, 3) mapping
    face-local edge k (F[:, k] -> F[:, (k + 1) % 3]) to its edge id.
    """
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2)
    e.sort(axis=1)
    n = int(e.max()) + 1 if len(e) else 1
    key = e[:, 0].astype(np.int64) * n + e[:, 1]
    ukey, inv = np.unique(key, return_inverse=True)
    E = np.stack([ukey // n, ukey % n], axis=1)
    return E, inv.reshape(-1, 3)


def edge_key(a, b, n):
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return lo * n + hi


def csr_from_pairs(rows, cols, nrows):
    """Group `cols` by `rows` -> (indptr, indices)."""
    order = np.argsort(rows, kind="stable")
    counts = np.bincount(rows, minlength=nrows)
    indptr = np.zeros(nrows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, cols[order]


def connected_components(n, a, b):
    """Label connected components of an undirected graph given edge endpoint arrays."""
    labels = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return labels
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    while True:
        la = labels[a]
        lb = labels[b]
        m = np.minimum(la, lb)
        changed = False
        # hook roots to the smaller label
        new = labels.copy()
        np.minimum.at(new, la, m)
        np.minimum.at(new, lb, m)
        if not np.array_equal(new, labels):
            changed = True
        labels = new
        # pointer jumping
        while True:
            nxt = labels[labels]
            if np.array_equal(nxt, labels):
                break
            labels = nxt
        if not changed:
            break
    return labels


def relabel(labels):
    """Map arbitrary labels to 0..k-1."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(-1)
