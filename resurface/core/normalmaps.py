"""Tell whether a tangent-space normal map's green channel points up or down.

Normal maps come in two conventions: green (+Y) points towards +V in OpenGL style
(Blender, FFXIV) and towards -V in DirectX style (Unreal and most other engines).  Read
with the wrong one, bumps light the wrong way along V.  Inside one UV island that only
looks a little off, but where two islands with differently turned UVs meet, the two
wrong answers disagree, and glossy materials show a hard line along the UV seam.

The check samples the map on both sides of every UV seam, turns each sample into a
surface normal with that side's own tangents, and measures how well the two sides agree,
once as the map is and once with green flipped; the right reading fits clearly better.
Only a mirrored reading can be found this way: turning the whole map (inverting red and
green together) turns both sides of every seam alike.
"""

import numpy as np

from .geom import edge_key, normalize
from .indexmap import weld_ids
from .texremap import _sample

SAMPLES_PER_EDGE = 6
INSET = 0.5          # pixels into each triangle, so the samples stay inside their island
BLUR = 2             # box blur radius in pixels: seam samples sit a pixel or two apart
MIN_SAMPLES = 40     # informative samples needed for a verdict
CLEAR_RATIO = 1.2    # one reading must fit this much better than the other


def sample_blurred(img, uv, r):
    """Bilinear samples of `img` at `uv`, box blurred over (2r+1)^2 pixels.  Blurring only
    at the samples spares whole-image copies of large textures."""
    H, W = img.shape[:2]
    acc = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc = acc + _sample(img, uv + np.array([dx / W, dy / H]), False)
    return acc / (2 * r + 1) ** 2


def triangle_tangents(pos, uv, nrm):
    """Flat tangent (+U direction) and bitangent sign per triangle, repeated for its
    corners; for meshes Blender cannot compute tangents for (n-gons)."""
    e1 = pos[:, 1] - pos[:, 0]
    e2 = pos[:, 2] - pos[:, 0]
    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-20, 1e-20, r)[:, None]
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r
    Bt = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r
    T, _ = normalize(T)
    sign = np.where((np.cross(nrm.mean(1), T) * Bt).sum(-1) < 0.0, -1.0, 1.0)
    return np.repeat(T[:, None], 3, 1), np.repeat(sign[:, None], 3, 1)


def seam_edges(pos, uv, weld=1e-5, uv_tol=1e-6):
    """UV seams: pairs of triangle edges at the same place whose UVs differ.
    pos: (nt, 3, 3) corner positions, uv: (nt, 3, 2) corner UVs.
    Returns (m, 6) int rows (tri_a, i_a, j_a, tri_b, i_b, j_b), both edges running from
    the same point i to the same point j."""
    nt = len(pos)
    if nt == 0:
        return np.zeros((0, 6), np.int64)
    vid = weld_ids(pos.reshape(-1, 3), weld).reshape(nt, 3)
    tri = np.repeat(np.arange(nt), 3)
    i = np.tile([0, 1, 2], nt)
    j = np.tile([1, 2, 0], nt)
    a = vid[tri, i]
    b = vid[tri, j]
    ok = a != b
    tri, i, j, a, b = tri[ok], i[ok], j[ok], a[ok], b[ok]
    key = edge_key(a, b, int(vid.max()) + 1)
    order = np.argsort(key, kind="stable")
    key = key[order]
    start = np.r_[True, key[1:] != key[:-1]]
    count = np.diff(np.r_[np.flatnonzero(start), len(key)])
    # only edges shared by exactly two triangles (non-manifold junctions are skipped)
    first = np.flatnonzero(start)[count == 2]
    e1, e2 = order[first], order[first + 1]
    t1, i1, j1 = tri[e1], i[e1], j[e1]
    t2, i2, j2 = tri[e2], i[e2], j[e2]
    turn = a[e2] != a[e1]
    i2, j2 = np.where(turn, j2, i2), np.where(turn, i2, j2)
    moved = ((np.abs(uv[t1, i1] - uv[t2, i2]).max(1) > uv_tol)
             | (np.abs(uv[t1, j1] - uv[t2, j2]).max(1) > uv_tol))
    return np.stack([t1, i1, j1, t2, i2, j2], 1)[moved]


def _side(tri, i, j, uv, nrm, tan, sign, size, ts):
    """Sample points along the edges (tri: i -> j), moved INSET pixels into the triangle,
    with the tangent frame there.  Returns uv (n, 2), N, T, B (n, 3)."""
    k = 3 - i - j
    w = ts[None, :, None]

    def along(a):
        return a[tri, i][:, None] * (1.0 - w) + a[tri, j][:, None] * w

    U = along(uv)
    inward = (uv[tri, k][:, None] - U) * size
    d, length = normalize(inward)
    U = U + d * np.minimum(INSET, 0.5 * length)[..., None] / size
    N, _ = normalize(along(nrm))
    T = along(tan)
    T, _ = normalize(T - N * (T * N).sum(-1, keepdims=True))
    B = sign[tri, i][:, None, None] * np.cross(N, T)
    return U.reshape(-1, 2), N.reshape(-1, 3), T.reshape(-1, 3), B.reshape(-1, 3)


def _reflection_gap(NA, TA, BA, NB, TB, BB):
    """How far apart the two sides' readings of the same map value move when green is
    mirrored: 0 where both sides' tangents agree (such samples cannot tell)."""
    D = np.array([1.0, -1.0, 1.0])
    FA = np.stack([TA, BA, NA], -1)
    FB = np.stack([TB, BB, NB], -1)
    M = (FA * D) @ np.swapaxes(FA, -1, -2) - (FB * D) @ np.swapaxes(FB, -1, -2)
    return np.sqrt((M ** 2).sum((-2, -1)))


def _normals(xy, N, T, B, flip):
    x = xy[:, 0]
    y = -xy[:, 1] if flip else xy[:, 1]
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    v, _ = normalize(x[:, None] * T + y[:, None] * B + z[:, None] * N)
    return v


def check_green(uv, pos, nrm, tan, sign, img, weld=1e-5):
    """Compare both readings of a tangent-space normal map along the UV seams.

    uv (nt, 3, 2), pos / nrm / tan (nt, 3, 3), sign (nt, 3): per triangle corner, with
    tangents and bitangent signs as Blender computes them for that UV map.
    img: (H, W, >=2) float pixels (row 0 at the bottom); R and G hold the direction and
    the other channels are ignored (the blue channel is often opacity, e.g. in FFXIV).

    Returns a dict: verdict ("UP", "DOWN", "UNCLEAR" or "NO_SEAMS"), seams (edge pairs),
    samples (informative samples used), error_up / error_down (mean squared distance
    between the two sides' normals) and ratio (error_up / error_down; above 1 the map
    fits better with green pointing down).
    """
    H, W = img.shape[:2]
    size = np.array([W, H], np.float64)
    pairs = seam_edges(pos, uv, weld)
    out = {"seams": len(pairs), "samples": 0, "error_up": 0.0, "error_down": 0.0, "ratio": 1.0,
           "verdict": "NO_SEAMS"}
    if len(pairs) == 0:
        return out
    ts = (np.arange(SAMPLES_PER_EDGE) + 0.5) / SAMPLES_PER_EDGE
    UA, NA, TA, BA = _side(pairs[:, 0], pairs[:, 1], pairs[:, 2], uv, nrm, tan, sign, size, ts)
    UB, NB, TB, BB = _side(pairs[:, 3], pairs[:, 4], pairs[:, 5], uv, nrm, tan, sign, size, ts)
    # skip hard edges (the shading breaks there anyway) and samples that cannot tell
    use = ((NA * NB).sum(-1) > np.cos(np.radians(10.0))) & (_reflection_gap(NA, TA, BA, NB, TB, BB) > 0.5)
    out["samples"] = int(use.sum())
    if out["samples"] < MIN_SAMPLES:
        out["verdict"] = "UNCLEAR"
        return out
    ca = sample_blurred(img, UA[use], BLUR)[:, :2] * 2.0 - 1.0
    cb = sample_blurred(img, UB[use], BLUR)[:, :2] * 2.0 - 1.0
    NA, TA, BA, NB, TB, BB = NA[use], TA[use], BA[use], NB[use], TB[use], BB[use]
    err = {}
    for flip in (False, True):
        d = _normals(ca, NA, TA, BA, flip) - _normals(cb, NB, TB, BB, flip)
        err[flip] = float((d ** 2).sum(-1).mean())
    out["error_up"], out["error_down"] = err[False], err[True]
    if err[False] <= 1e-12 and err[True] <= 1e-12:
        out["verdict"] = "UNCLEAR"      # flat map: nothing to compare
        return out
    ratio = err[False] / max(err[True], 1e-12)
    out["ratio"] = ratio
    out["verdict"] = "DOWN" if ratio > CLEAR_RATIO else "UP" if ratio < 1.0 / CLEAR_RATIO else "UNCLEAR"
    return out
