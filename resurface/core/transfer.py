"""Locate every corner of the rebuilt mesh on the source surface.

Each output face corner gets (source face, barycentric weights).  All attributes
(UVs, colors, normals, weights, shape keys, generic attributes) are then plain
barycentric interpolations of the source values at that source face's corners.

Interior vertices use the face they are tracked on.  Vertices on feature chains and
corners touch several source faces (one per sheet / side of a seam); for them the
source face is chosen per output face, from the same patch and on the same side,
so UV seams, split normals and object borders come out exactly as in the source.
"""

import numpy as np

from .geom import closest_point_segment, closest_point_triangle, face_normals, norm


def _unit(a):
    return a / np.maximum(norm(a), 1e-30)[..., None]


def corner_sources(src, rm):
    """Returns (cs_face (nf,3), cs_bary (nf,3,3))."""
    V, F = rm.V, rm.F
    nf = len(F)
    nv = len(V)
    SV, SF = src.V, src.F
    vt = rm.vtype

    vface = np.full(nv, -1, np.int64)
    vbary = np.zeros((nv, 3))
    i0 = np.nonzero(vt == 0)[0]
    if len(i0):
        s0 = rm.vhost[i0]
        tri = SF[s0]
        _, b0 = closest_point_triangle(V[i0], SV[tri[:, 0]], SV[tri[:, 1]], SV[tri[:, 2]])
        vface[i0] = s0
        vbary[i0] = b0

    # chain parameter of every chain vertex
    vt_param = np.zeros(nv)
    i1 = np.nonzero(vt == 1)[0]
    if len(i1):
        seg = rm.vhost[i1]
        _, t = closest_point_segment(V[i1], SV[src.seg_a[seg]], SV[src.seg_b[seg]])
        vt_param[i1] = t

    cs_face = np.full((nf, 3), -1, np.int64)
    cs_bary = np.zeros((nf, 3, 3))
    for k in range(3):
        v = F[:, k]
        m = vt[v] == 0
        cs_face[m, k] = vface[v[m]]
        cs_bary[m, k] = vbary[v[m]]

    # --- chain and corner vertices: pick the source face per output face
    FN = _unit(face_normals(V, F))
    FC = V[F].mean(1)
    SFN = src.face_unit_normals
    SFC = src.face_centroids
    fpatch = rm.fpatch
    spatch = src.fpatch
    todo_f, todo_k = np.nonzero(vt[F] != 0)
    Fl = F.tolist()
    vt_l = vt.tolist()
    host_l = rm.vhost.tolist()
    fp_l = fpatch.tolist()
    for f, k in zip(todo_f.tolist(), todo_k.tolist()):
        v = Fl[f][k]
        P = fp_l[f]
        p = V[v]
        if vt_l[v] == 1:
            seg = host_l[v]
            cand = src.seg_faces(seg)
            cand = cand[spatch[cand] == P]
            e0 = src.seg_a[seg]
            e1 = src.seg_b[seg]
            if len(cand) == 0:
                s = src.nearest_in_patch(P, p)
                cs_face[f, k] = s
                _, b = closest_point_triangle(p, SV[SF[s, 0]], SV[SF[s, 1]], SV[SF[s, 2]])
                cs_bary[f, k] = b
                continue
            if len(cand) > 1:
                u = _unit(SV[e1] - SV[e0])
                df = FC[f] - p
                df = _unit(df - np.dot(df, u) * u)
                ds = SFC[cand] - p
                ds = _unit(ds - (ds @ u)[:, None] * u)
                score = ds @ df + SFN[cand] @ FN[f]
                s = int(cand[np.argmax(score)])
            else:
                s = int(cand[0])
            t = vt_param[v]
            row = SF[s]
            b = np.zeros(3)
            b[int(np.nonzero(row == e0)[0][0])] = 1.0 - t
            b[int(np.nonzero(row == e1)[0][0])] = t
            cs_face[f, k] = s
            cs_bary[f, k] = b
        else:
            c = host_l[v]
            cand = src.vert_faces(c)
            cand = cand[spatch[cand] == P]
            if len(cand) == 0:
                s = src.nearest_in_patch(P, p)
                cs_face[f, k] = s
                _, b = closest_point_triangle(p, SV[SF[s, 0]], SV[SF[s, 1]], SV[SF[s, 2]])
                cs_bary[f, k] = b
                continue
            if len(cand) > 1:
                df = _unit(FC[f] - p)
                ds = _unit(SFC[cand] - p)
                score = ds @ df + SFN[cand] @ FN[f]
                s = int(cand[np.argmax(score)])
            else:
                s = int(cand[0])
            b = np.zeros(3)
            b[int(np.nonzero(SF[s] == c)[0][0])] = 1.0
            cs_face[f, k] = s
            cs_bary[f, k] = b
    return cs_face, cs_bary


class CornerMap:
    """Maps output corners to raw source corners (loops / vertices) with weights."""

    def __init__(self, src, T_loops, T_raw, cs_face, cs_bary):
        tri = src.ftri[cs_face]                         # (nf, 3)
        rc = src.fcorner[cs_face]                       # (nf, 3, 3) raw corner of each source corner
        self.loops = T_loops[tri[..., None], rc]        # (nf, 3, 3) global loop ids
        self.verts = T_raw[tri[..., None], rc]          # (nf, 3, 3) global raw vertex ids
        self.w = cs_bary                                # (nf, 3, 3)
        self.face = cs_face

    def corner(self, values, sel=None):
        """Interpolate a per-loop array (n_loops, d) -> (n_sel_faces, 3, d)."""
        L = self.loops if sel is None else self.loops[sel]
        W = self.w if sel is None else self.w[sel]
        return np.einsum("fcj,fcjd->fcd", W, values[L])

    def point(self, values, sel=None):
        """Interpolate a per-raw-vertex array (n_verts, d) -> (n_sel_faces, 3, d)."""
        Vv = self.verts if sel is None else self.verts[sel]
        W = self.w if sel is None else self.w[sel]
        return np.einsum("fcj,fcjd->fcd", W, values[Vv])

    def nearest_point(self, values, sel=None):
        """Value of the dominant source vertex (for ints / bools)."""
        Vv = self.verts if sel is None else self.verts[sel]
        W = self.w if sel is None else self.w[sel]
        j = np.argmax(W, axis=-1)
        idx = np.take_along_axis(Vv, j[..., None], axis=-1)[..., 0]
        return values[idx]
