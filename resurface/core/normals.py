"""Robust smooth normals for messy meshes.

Blender's usual fix ("Recalculate Outside" then "Set from Faces") breaks down on game
meshes like cloth with several layers, strings attached at junctions and regions of
flipped faces: faces of different sheets or with opposite winding end up in one
average, which gives dark spots, seams and pinches.  Here:

* vertices are welded by position first, so UV seams and part borders do not show;
* the corners around a welded vertex are grouped into smooth groups: two faces that
  share an edge join when the angle between them is below a threshold, measured
  after accounting for their relative winding (faces wound the other way are
  aligned instead of cancelling out);
* a group's normal is the corner-angle weighted average of its faces, each face
  flipped into the group's orientation;
* optionally the normal field is blurred over the surface to soften lumpy
  (decimated) geometry, without crossing hard edges.

Works on polygon meshes given as loop arrays (corners), so quads and n-gons are fine.
"""

import math

import numpy as np

from .geom import norm


def _weld(V, eps):
    from .source import SourceMesh
    return SourceMesh._weld(V, eps)[0]


class LoopMesh:
    """Minimal polygon mesh in loop (corner) form."""

    def __init__(self, V, loop_vert, loop_start, loop_total):
        self.V = np.asarray(V, np.float64)
        self.loop_vert = np.asarray(loop_vert, np.int64)
        self.loop_start = np.asarray(loop_start, np.int64)
        self.loop_total = np.asarray(loop_total, np.int64)
        nl = len(self.loop_vert)
        self.loop_face = np.repeat(np.arange(len(self.loop_start)), self.loop_total)
        pos = np.arange(nl) - self.loop_start[self.loop_face]
        end = self.loop_start[self.loop_face] + self.loop_total[self.loop_face]
        self.next = np.where(np.arange(nl) + 1 < end, np.arange(nl) + 1, self.loop_start[self.loop_face])
        self.prev = np.where(pos > 0, np.arange(nl) - 1, end - 1)

    def face_normals(self):
        """Newell normals (unnormalised, length = 2 * area for planar polygons)."""
        P = self.V[self.loop_vert]
        Q = self.V[self.loop_vert[self.next]]
        c = np.cross(P, Q)
        n = np.zeros((len(self.loop_start), 3))
        np.add.at(n, self.loop_face, c)
        return n

    def corner_angles(self):
        v = self.V[self.loop_vert]
        a = self.V[self.loop_vert[self.prev]] - v
        b = self.V[self.loop_vert[self.next]] - v
        return np.arctan2(norm(np.cross(a, b)), np.einsum("ij,ij->i", a, b))


def _parity_union_find(n, pairs_a, pairs_b, pairs_s):
    """Union-find with parity.  s = 0: same orientation, 1: opposite.
    Returns (root, parity relative to root) per element."""
    parent = list(range(n))
    parity = [0] * n

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        acc = 0
        for node in reversed(path):
            acc ^= parity[node]
            parity[node] = acc
            parent[node] = root
        return root

    for a, b, s in zip(pairs_a, pairs_b, pairs_s):
        ra = find(a)
        pa = parity[a] if a != ra else 0
        rb = find(b)
        pb = parity[b] if b != rb else 0
        if ra == rb:
            continue
        parent[rb] = ra
        parity[rb] = pa ^ pb ^ s
    roots = [find(i) for i in range(n)]
    par = [parity[i] if roots[i] != i else 0 for i in range(n)]
    return np.array(roots, np.int64), np.array(par, np.int64)


def smooth_loop_normals(mesh, weld_distance=1e-5, angle=math.radians(80.0), blur=0, blur_strength=0.5):
    """Per-loop unit normals for a LoopMesh."""
    nl = len(mesh.loop_vert)
    vid = _weld(mesh.V, weld_distance)
    lv = vid[mesh.loop_vert]
    FN = mesh.face_normals()
    fa = norm(FN)
    FNu = FN / np.maximum(fa, 1e-30)[:, None]
    # every loop is the start of one polygon edge: lv[l] -> lv[next[l]]
    a = lv
    b = lv[mesh.next]
    ok = a != b
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    nkey = int(vid.max()) + 1 if len(vid) else 1
    key = lo * nkey + hi
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(key[idx], kind="stable")]
    ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    # pairs of loops sharing a welded edge (all pairs for non-manifold edges, capped)
    pa, pb = [], []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 2:
            continue
        grp = order[s:e]
        if len(grp) == 2:
            pa.append(int(grp[0]))
            pb.append(int(grp[1]))
        else:
            g = grp[:8].tolist()
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    pa.append(g[i])
                    pb.append(g[j])
    pa = np.array(pa, np.int64)
    pb = np.array(pb, np.int64)
    if len(pa):
        f1 = mesh.loop_face[pa]
        f2 = mesh.loop_face[pb]
        same_dir = a[pa] == a[pb]              # both run the edge the same way -> opposite winding
        par = same_dir.astype(np.int64)
        cosang = np.einsum("ij,ij->i", FNu[f1], FNu[f2]) * np.where(same_dir, -1.0, 1.0)
        join = (cosang > math.cos(angle)) & (f1 != f2)
        pa, pb, par, same_dir = pa[join], pb[join], par[join], same_dir[join]
        # corners at the two edge ends: loop l starts at lv[l]; next[l] sits at lv[next[l]]
        ca1 = pa                                           # f1's corner at vertex a[pa]
        cb1 = mesh.next[pa]                                # f1's corner at vertex b[pa]
        ca2 = np.where(same_dir, pb, mesh.next[pb])        # f2's corner at the same vertex a
        cb2 = np.where(same_dir, mesh.next[pb], pb)
        ua = np.r_[ca1, cb1]
        ub = np.r_[ca2, cb2]
        us = np.r_[par, par]
    else:
        ua = ub = us = np.zeros(0, np.int64)
    root, parity = _parity_union_find(nl, ua.tolist(), ub.tolist(), us.tolist())
    sign = 1.0 - 2.0 * parity
    w = mesh.corner_angles()
    contrib = FNu[mesh.loop_face] * (w * sign)[:, None]
    G = np.zeros((nl, 3))
    np.add.at(G, root, contrib)

    if blur > 0:
        # neighbouring groups along polygon edges, with their relative orientation
        g1 = root
        g2 = root[mesh.next]
        rel = 1.0 - 2.0 * (parity ^ parity[mesh.next])
        keep = g1 != g2
        g1, g2, rel = g1[keep], g2[keep], rel[keep]
        Gu = G / np.maximum(norm(G), 1e-30)[:, None]
        for _ in range(int(blur)):
            acc = np.zeros_like(Gu)
            cnt = np.zeros(len(Gu))
            # only blur across nearly aligned neighbours so hard edges stay hard
            d = np.einsum("ij,ij->i", Gu[g1], Gu[g2]) * rel
            ok2 = d > math.cos(angle)
            np.add.at(acc, g1[ok2], Gu[g2[ok2]] * rel[ok2][:, None])
            np.add.at(acc, g2[ok2], Gu[g1[ok2]] * rel[ok2][:, None])
            np.add.at(cnt, g1[ok2], 1.0)
            np.add.at(cnt, g2[ok2], 1.0)
            has = cnt > 0
            upd = Gu.copy()
            upd[has] = Gu[has] + blur_strength * acc[has] / cnt[has][:, None]
            Gu = upd / np.maximum(norm(upd), 1e-30)[:, None]
        G = Gu

    out = G[root] * sign[:, None]
    ln = norm(out)
    bad = ln < 1e-12
    out[bad] = FNu[mesh.loop_face[bad]]
    return out / np.maximum(norm(out), 1e-30)[:, None]


def flip_islands(mesh, weld_distance=1e-5, max_fraction=0.01):
    """Faces to flip so that tiny regions wound against their surroundings agree
    with them.  A region is flipped when it is smaller than `max_fraction` of the
    connected piece it sits in and larger regions border it with opposite winding.
    Kept deliberately small: bigger inward-facing regions (seam allowances, insides of
    straps) are usually intentional and are visible from inside the garment.
    Returns a boolean mask over polygons."""
    from .geom import connected_components, relabel
    nf = len(mesh.loop_start)
    vid = _weld(mesh.V, weld_distance)
    lv = vid[mesh.loop_vert]
    a = lv
    b = lv[mesh.next]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    nkey = int(vid.max()) + 1 if len(vid) else 1
    key = lo * nkey + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[starts, len(ks)])
    two = starts[cnt == 2]
    l1 = order[two]
    l2 = order[two + 1]
    f1 = mesh.loop_face[l1]
    f2 = mesh.loop_face[l2]
    inconsistent = a[l1] == a[l2]
    area = 0.5 * norm(mesh.face_normals())
    comp = relabel(connected_components(nf, f1[~inconsistent], f2[~inconsistent]))
    piece = relabel(connected_components(nf, f1, f2))
    carea = np.bincount(comp, weights=area)
    parea = np.bincount(piece, weights=area)
    c1 = comp[f1[inconsistent]]
    c2 = comp[f2[inconsistent]]
    cpiece = np.zeros(len(carea), np.int64)
    cpiece[comp] = piece
    small = carea < max_fraction * parea[cpiece]
    flip_c = np.zeros(len(carea), bool)
    flip_c[c1[small[c1] & (carea[c2] > carea[c1])]] = True
    flip_c[c2[small[c2] & (carea[c1] > carea[c2])]] = True
    return flip_c[comp]
