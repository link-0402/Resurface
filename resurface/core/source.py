"""Source mesh preparation.

Turns a raw triangle soup (possibly several objects, split vertices, junk fragments,
non-manifold junctions, flipped faces) into a welded "patch complex":

* vertices welded by position,
* degenerate / duplicate faces and tiny floating fragments removed,
* feature edges: boundaries, non-manifold junctions, object/material borders,
  attribute seams (UVs, colors, ...) and optional sharp creases,
* patches: face components that are connected through non-feature edges, each
  consistently oriented,
* feature curves ("chains") between corner vertices.

Everything is numpy based; only the lazily built per-patch BVH uses mathutils.
"""

import math
import numpy as np

from .geom import (
    closest_point_triangle,
    connected_components,
    csr_from_pairs,
    face_normals,
    norm,
    relabel,
    unique_edges,
)


# Why an edge is a feature (bit flags in SourceMesh.ekind)
K_BORDER = 1      # one face
K_JUNCTION = 2    # three or more faces
K_PART = 4        # between objects or materials
K_SEAM = 8        # UV / color discontinuity
K_WINDING = 16    # the two faces have opposite winding
K_SHARP = 32      # crease sharper than the sharp angle
K_SOFT = K_SEAM | K_SHARP   # kinds that may be dissolved when they are only noise


class SourceOptions:
    def __init__(self, **kw):
        self.merge_distance = 1e-5       # absolute weld distance
        self.fragment_faces = 3          # components with <= this many faces are junk
        self.fragment_area = 1e-5        # ... and with less than this fraction of total area
        self.seam_tolerance = 2e-3       # UV/color difference that makes a seam (~1 px at 512)
        self.sharp_angle = None          # degrees; None disables crease detection
        self.sharp_min_edges = 3         # crease chains shorter than this are noise
        self.corner_angle = 60.0         # turning angle (deg) that makes a chain corner
        self.min_feature_size = 0.0      # > 0: dissolve seam/crease noise and flip winding
                                         # islands smaller than this (the target edge length)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


class SourceMesh:
    """Welded, cleaned, feature-annotated source surface."""

    def __init__(self, V_raw, T_raw, T_obj=None, T_mat=None, seam_attrs=(), options=None, log=print,
                 exclude_tris=None):
        self.opts = options or SourceOptions()
        self.log = log
        V_raw = np.asarray(V_raw, dtype=np.float64)
        T_raw = np.asarray(T_raw, dtype=np.int64)
        m = len(T_raw)
        T_obj = np.zeros(m, np.int64) if T_obj is None else np.asarray(T_obj, np.int64)
        T_mat = np.zeros(m, np.int64) if T_mat is None else np.asarray(T_mat, np.int64)
        self.stats = {}

        # ---- weld ---------------------------------------------------------------------
        vid, V = self._weld(V_raw, self.opts.merge_distance)
        F = vid[T_raw]
        ftri = np.arange(m)
        fcorner = np.tile(np.arange(3), (m, 1))
        self.stats["raw_vertices"] = len(V_raw)
        self.stats["raw_triangles"] = m
        self.stats["welded_vertices"] = len(V)
        if exclude_tris is not None:
            keep = ~np.asarray(exclude_tris, bool)
            F, ftri, fcorner = F[keep], ftri[keep], fcorner[keep]
            self.stats["excluded_triangles"] = int((~keep).sum())

        # ---- degenerate faces (repeated vertices) ---------------------------------------
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
        self.stats["degenerate_removed"] = int((~ok).sum())
        F, ftri, fcorner = F[ok], ftri[ok], fcorner[ok]

        # ---- duplicate faces --------------------------------------------------------------
        sf = np.sort(F, axis=1)
        nV = len(V)
        key = (sf[:, 0] * nV + sf[:, 1]) * nV + sf[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        first = np.ones(len(ks), bool)
        first[1:] = ks[1:] != ks[:-1]
        keep_idx = order[first]
        dup_idx = order[~first]
        # Duplicates are not thrown away: they become "twins" of the kept face and are
        # regenerated on the rebuilt surface (backface copies, parts duplicated into
        # another object).  A duplicate with opposite winding marks a double sided face.
        grp_first = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
        rep_of = np.empty(len(F), np.int64)
        rep_of[order] = order[grp_first]
        double = np.zeros(len(F), bool)
        if len(dup_idx):
            rep = rep_of[dup_idx]
            same = self._same_winding(F[dup_idx], F[rep])
            double[rep[~same]] = True
            self._twin_pairs_raw = (ftri[rep], ftri[dup_idx])
        else:
            self._twin_pairs_raw = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        keep_mask = np.zeros(len(F), bool)
        keep_mask[keep_idx] = True
        self.stats["duplicates_removed"] = int(len(dup_idx))
        F, ftri, fcorner, double = F[keep_mask], ftri[keep_mask], fcorner[keep_mask], double[keep_mask]

        # ---- fragments (faces connected through any shared edge) ---------------------------
        area = 0.5 * norm(face_normals(V, F))
        E0, fe0 = unique_edges(F)
        ep, ei = csr_from_pairs(fe0.ravel(), np.repeat(np.arange(len(F)), 3), len(E0))
        cnt0 = np.diff(ep)
        # link consecutive faces of every edge (handles non-manifold fans too)
        rows = np.repeat(np.arange(len(E0)), cnt0)
        pos = np.arange(len(ei)) - ep[rows]
        link = pos > 0
        comp = relabel(connected_components(len(F), ei[np.nonzero(link)[0] - 1], ei[link]))
        ncomp = comp.max() + 1 if len(comp) else 0
        cfaces = np.bincount(comp, minlength=ncomp)
        carea = np.bincount(comp, weights=area, minlength=ncomp)
        total = area.sum()
        # debris = few faces AND (practically) no area; small legit cards survive
        junk = (cfaces <= self.opts.fragment_faces) & (carea < self.opts.fragment_area * total)
        # never delete everything
        if junk.all():
            junk[:] = False
        fkeep = ~junk[comp]
        self.stats["fragments_removed"] = int(junk.sum())
        self.stats["fragment_faces_removed"] = int((~fkeep).sum())
        F, ftri, fcorner, double = F[fkeep], ftri[fkeep], fcorner[fkeep], double[fkeep]

        # drop unreferenced vertices
        used = np.zeros(len(V), bool)
        used[F.ravel()] = True
        remap = np.cumsum(used) - 1
        V = V[used]
        F = remap[F]
        self.raw_to_welded = np.where(used[vid], remap[vid], -1)

        self.V = V
        self.F = F
        self.ftri = ftri
        self.fcorner = fcorner
        self.fobj = T_obj[ftri]
        self.fmat = T_mat[ftri]
        self.double_sided = double
        self._seam_attrs = [np.asarray(a) for a in seam_attrs]

        # ---- connectivity -------------------------------------------------------------------
        self._build_edges()

        # ---- features, orientation, patches ----------------------------------------------
        Lmin = float(self.opts.min_feature_size or 0.0)
        self.ekind = self._basic_features()
        self.efeature = self.ekind != 0
        self._orient_patches(max_island_area=2.0 * Lmin * Lmin)
        if self.opts.sharp_angle is not None:
            self.ekind[self._sharp_features(self.opts.sharp_angle)] |= K_SHARP
            self.efeature = self.ekind != 0
        if Lmin > 0:
            self._simplify_features(Lmin)
            self.efeature = self.ekind != 0
        self.fpatch = self._face_components(len(self.F), self.E, self.fe, self.efeature)
        self.npatch = int(self.fpatch.max()) + 1 if len(self.F) else 0

        # ---- vertex classification and chains --------------------------------------------
        self._classify_vertices()
        self._build_chains()
        self._build_walk()
        self._build_twins(T_raw, T_obj, m)
        self._patch_bvh = {}
        self.stats.update(
            vertices=len(self.V), triangles=len(self.F), patches=self.npatch,
            feature_edges=int(self.efeature.sum()), chains=len(self.chain_start),
            corners=int((self.vtype == 2).sum()),
        )

    # ======================================================================================
    # welding / helpers
    # ======================================================================================
    @staticmethod
    def _weld(V, eps):
        n = len(V)
        if n == 0:
            return np.zeros(0, np.int64), V
        eps = max(float(eps), 1e-12)
        a_list, b_list = [], []
        for off in (0.0, 0.5):
            key = np.floor(V / eps + off).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            # link every vertex to the first vertex of its cell
            order = np.argsort(inv, kind="stable")
            si = inv[order]
            first = np.r_[True, si[1:] != si[:-1]]
            rep = order[np.maximum.accumulate(np.where(first, np.arange(n), 0))]
            a_list.append(order)
            b_list.append(rep)
        lab = connected_components(n, np.concatenate(a_list), np.concatenate(b_list))
        # verify the merged clusters are tight: split members farther than 4*eps from the rep
        lab = relabel(lab)
        cnt = np.bincount(lab)
        acc = np.zeros((cnt.size, 3))
        np.add.at(acc, lab, V)
        W = acc / cnt[:, None]
        return lab, W

    @staticmethod
    def _same_winding(Fa, Fb):
        """True where faces Fa and Fb (same vertex sets) have the same cyclic order."""
        same = np.zeros(len(Fa), bool)
        for r in range(3):
            rb = np.roll(Fb, -r, axis=1)
            same |= (Fa == rb).all(axis=1)
        return same

    def _face_components(self, nf, E, fe, efeature):
        """Face components connected through non-feature edges shared by exactly 2 faces."""
        ef_ptr, ef_idx = csr_from_pairs(fe.ravel(), np.repeat(np.arange(nf), 3), len(E))
        cnt = np.diff(ef_ptr)
        two = np.nonzero((cnt == 2) & ~efeature)[0]
        fa = ef_idx[ef_ptr[two]]
        fb = ef_idx[ef_ptr[two] + 1]
        return relabel(connected_components(nf, fa, fb))

    def _build_edges(self):
        F = self.F
        nf = len(F)
        self.E, self.fe = unique_edges(F)
        ne = len(self.E)
        self.ef_ptr, self.ef_idx = csr_from_pairs(self.fe.ravel(), np.repeat(np.arange(nf), 3), ne)
        # local edge index of each (edge, face) incidence, aligned with ef_idx
        loc = np.tile(np.arange(3), nf)
        _, self.ef_loc = csr_from_pairs(self.fe.ravel(), loc, ne)
        self.ecount = np.diff(self.ef_ptr)
        self.vf_ptr, self.vf_idx = csr_from_pairs(F.ravel(), np.repeat(np.arange(nf), 3), len(self.V))
        self.elen = norm(self.V[self.E[:, 0]] - self.V[self.E[:, 1]])

    def _edge_pairs(self):
        """For manifold edges: arrays (edge, f1, k1, f2, k2)."""
        two = np.nonzero(self.ecount == 2)[0]
        p = self.ef_ptr[two]
        return two, self.ef_idx[p], self.ef_loc[p], self.ef_idx[p + 1], self.ef_loc[p + 1]

    def _corner_values(self, attr, f, vertex):
        """attr: (m_raw, 3, d) per raw triangle corner. Value at the corner of face f holding vertex."""
        F = self.F
        k = np.argmax(F[f] == vertex[:, None], axis=1)
        return attr[self.ftri[f], self.fcorner[f, k]]

    # ======================================================================================
    # features
    # ======================================================================================
    def _basic_features(self):
        """Feature kind per edge (K_* bit flags; 0 = not a feature)."""
        cnt = self.ecount
        kind = np.zeros(len(self.E), np.uint8)
        kind[cnt == 1] |= K_BORDER
        kind[cnt > 2] |= K_JUNCTION
        self.ejump = np.zeros(len(self.E))     # size of the attribute jump across seam edges
        e, f1, k1, f2, k2 = self._edge_pairs()
        if len(e):
            part = (self.fobj[f1] != self.fobj[f2]) | (self.fmat[f1] != self.fmat[f2])
            kind[e[part]] |= K_PART
            jump = np.zeros(len(e))
            a = self.E[e, 0]
            b = self.E[e, 1]
            tol = self.opts.seam_tolerance
            for attr in self._seam_attrs:
                va1 = self._corner_values(attr, f1, a)
                va2 = self._corner_values(attr, f2, a)
                vb1 = self._corner_values(attr, f1, b)
                vb2 = self._corner_values(attr, f2, b)
                d = np.maximum(np.abs(va1 - va2).reshape(len(e), -1).max(1),
                               np.abs(vb1 - vb2).reshape(len(e), -1).max(1))
                jump = np.maximum(jump, d)
            seam = jump > tol
            kind[e[seam]] |= K_SEAM
            self.ejump[e] = jump
        self.stats["boundary_edges"] = int((cnt == 1).sum())
        self.stats["nonmanifold_edges"] = int((cnt > 2).sum())
        self.stats["seam_edges"] = int(((kind & K_SEAM) != 0).sum())
        return kind

    def _inconsistent_edges(self):
        """Non-feature manifold edges whose two faces traverse them in the same direction."""
        F = self.F
        e, f1, k1, f2, k2 = self._edge_pairs()
        usable = ~self.efeature[e]
        e, f1, k1, f2, k2 = e[usable], f1[usable], k1[usable], f2[usable], k2[usable]
        d1 = F[f1, k1] < F[f1, (k1 + 1) % 3]
        d2 = F[f2, k2] < F[f2, (k2 + 1) % 3]
        bad = d1 == d2
        return e, f1, f2, bad

    def _orient_patches(self, max_island=8, max_island_area=0.0):
        """Keep the author's face orientation.  Only small islands whose winding disagrees
        with their surroundings (decimation debris; in game they are holes because of
        back-face culling) are flipped; any remaining winding change becomes a feature
        cut, so every patch is consistently oriented."""
        nf = len(self.F)
        e, f1, f2, bad = self._inconsistent_edges()
        self.stats["inconsistent_edges"] = int(bad.sum())
        if not bad.any():
            self.stats["faces_reoriented"] = 0
            return
        good = ~bad
        comp = relabel(connected_components(nf, f1[good], f2[good]))
        size = np.bincount(comp)
        carea = np.bincount(comp, weights=0.5 * norm(face_normals(self.V, self.F)))
        small = (size <= max_island) | (carea <= max_island_area)
        # islands: small consistent components that touch a larger one through a bad edge
        c1 = comp[f1[bad]]
        c2 = comp[f2[bad]]
        small1 = small[c1] & (carea[c2] > carea[c1])
        small2 = small[c2] & (carea[c1] > carea[c2])
        flip_comp = np.zeros(len(size), bool)
        flip_comp[c1[small1]] = True
        flip_comp[c2[small2]] = True
        # do not flip two neighbouring islands against each other
        both = flip_comp[c1] & flip_comp[c2]
        flip_comp[np.minimum(c1[both], c2[both])] = False
        flip = flip_comp[comp]
        self.stats["faces_reoriented"] = int(flip.sum())
        if flip.any():
            idx = np.nonzero(flip)[0]
            self.F[idx] = self.F[idx][:, [0, 2, 1]]
            self.fcorner[idx] = self.fcorner[idx][:, [0, 2, 1]]
            # the undirected edge set (and so the edge ids) is unchanged
            self._build_edges()
        e, f1, f2, bad = self._inconsistent_edges()
        self.ekind[e[bad]] |= K_WINDING
        self.efeature = self.ekind != 0
        self.stats["winding_cuts"] = int(bad.sum())

    def _sharp_features(self, angle_deg):
        e, f1, k1, f2, k2 = self._edge_pairs()
        N = face_normals(self.V, self.F)
        N = N / np.maximum(norm(N), 1e-30)[:, None]
        cosang = np.einsum("ij,ij->i", N[f1], N[f2])
        cand = np.zeros(len(self.E), bool)
        cand[e[cosang < math.cos(math.radians(angle_deg))]] = True
        cand &= ~self.efeature
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            return cand
        # keep only crease chains with enough edges (decimation noise is isolated)
        lab = relabel(connected_components(len(self.V), self.E[idx, 0], self.E[idx, 1]))
        clab = lab[self.E[idx, 0]]
        size = np.bincount(clab)
        keep = size[clab] >= self.opts.sharp_min_edges
        out = np.zeros(len(self.E), bool)
        out[idx[keep]] = True
        self.stats["sharp_edges"] = int(keep.sum())
        return out

    def _raw_chains(self, feat):
        """Split feature edges into maximal paths between vertices whose feature degree is
        not 2.  Returns ([(edge ids, start vertex, end vertex, closed)], feature degree)."""
        idx = np.nonzero(feat)[0]
        nv = len(self.V)
        a = self.E[idx, 0]
        b = self.E[idx, 1]
        deg = np.bincount(np.r_[a, b], minlength=nv)
        ptr, inc = csr_from_pairs(np.r_[a, b], np.r_[idx, idx], nv)
        Ea = self.E[:, 0].tolist()
        Eb = self.E[:, 1].tolist()
        ptr_l, inc_l, deg_l = ptr.tolist(), inc.tolist(), deg.tolist()
        visited = set()
        out = []

        def walk(start, e):
            edges = []
            cur = start
            while True:
                visited.add(e)
                edges.append(e)
                nxt = Eb[e] if Ea[e] == cur else Ea[e]
                if deg_l[nxt] != 2 or nxt == start:
                    return edges, nxt
                e_next = None
                for j in range(ptr_l[nxt], ptr_l[nxt + 1]):
                    ee = inc_l[j]
                    if ee != e and ee not in visited:
                        e_next = ee
                        break
                if e_next is None:
                    return edges, nxt
                cur, e = nxt, e_next

        for v in np.nonzero((deg > 0) & (deg != 2))[0].tolist():
            for j in range(ptr_l[v], ptr_l[v + 1]):
                if inc_l[j] not in visited:
                    edges, end = walk(v, inc_l[j])
                    out.append((np.array(edges), v, end, False))
        for e in idx.tolist():
            if e not in visited:
                edges, end = walk(Ea[e], e)
                out.append((np.array(edges), Ea[e], end, True))
        return out, deg

    def _simplify_features(self, Lmin):
        """Dissolve seam / crease noise that is smaller than the target resolution.

        * patches smaller than half a target triangle lose their seam/crease borders
          (they merge into their neighbours);
        * seam/crease fragments that dangle into a surface and are shorter than 3/4 of a
          target edge are dropped.
        Borders, junctions, part borders and winding changes are never touched, and
        neither are seams across which the UVs jump far (more than 5% of the texture):
        blending across those would smear the texture."""
        area = 0.5 * norm(face_normals(self.V, self.F))
        merged = 0
        dropped = 0
        near_uv = self.ejump < 0.05
        for _ in range(4):
            changed = False
            feat = self.ekind != 0
            soft = feat & ((self.ekind & ~np.uint8(K_SOFT)) == 0) & near_uv
            P = self._face_components(len(self.F), self.E, self.fe, feat)
            parea = np.bincount(P, weights=area)
            tiny = parea < 0.5 * Lmin * Lmin
            if tiny.any():
                eids = np.unique(self.fe[tiny[P]].ravel())
                eids = eids[soft[eids]]
                if len(eids):
                    self.ekind[eids] = 0
                    merged += int(tiny.sum())
                    changed = True
            feat = self.ekind != 0
            soft = feat & ((self.ekind & ~np.uint8(K_SOFT)) == 0) & near_uv
            chains, deg = self._raw_chains(feat)
            for edges, v0, v1, closed in chains:
                if closed or not soft[edges].all():
                    continue
                if (deg[v0] == 1 or deg[v1] == 1) and self.elen[edges].sum() < 0.75 * Lmin:
                    self.ekind[edges] = 0
                    dropped += 1
                    changed = True
            if not changed:
                break
        self.stats["noise_patches_merged"] = merged
        self.stats["noise_curves_dropped"] = dropped

    # ======================================================================================
    # vertices, chains
    # ======================================================================================
    def _fans(self):
        """Number of face fans around each vertex (fans are joined across non-feature
        manifold edges)."""
        F = self.F
        nf = len(F)
        e, f1, k1, f2, k2 = self._edge_pairs()
        keep = ~self.efeature[e]
        e, f1, k1, f2, k2 = e[keep], f1[keep], k1[keep], f2[keep], k2[keep]
        # corners of f1 at edge endpoints
        u = F[f1, k1]
        w = F[f1, (k1 + 1) % 3]
        c1u = f1 * 3 + k1
        c1w = f1 * 3 + (k1 + 1) % 3
        # in f2 (consistently oriented), the edge runs w -> u at local k2
        g_u_first = F[f2, k2] == u
        c2u = np.where(g_u_first, f2 * 3 + k2, f2 * 3 + (k2 + 1) % 3)
        c2w = np.where(g_u_first, f2 * 3 + (k2 + 1) % 3, f2 * 3 + k2)
        lab = connected_components(nf * 3, np.r_[c1u, c1w], np.r_[c2u, c2w])
        cv = F.ravel()
        pair = np.unique(cv.astype(np.int64) * (nf * 3) + lab)
        return np.bincount(pair // (nf * 3), minlength=len(self.V))

    def _classify_vertices(self):
        nv = len(self.V)
        fe_idx = np.nonzero(self.efeature)[0]
        fa = self.E[fe_idx, 0]
        fb = self.E[fe_idx, 1]
        fdeg = np.bincount(np.r_[fa, fb], minlength=nv)
        self.fdeg = fdeg
        fans = self._fans()
        vtype = np.zeros(nv, np.int8)
        vtype[fdeg > 0] = 1
        corner = (fdeg > 0) & (fdeg != 2)
        corner |= (fdeg == 0) & (fans != 1)
        # degree-2 feature vertices: incident feature edges must share a signature and turn gently
        ptr, inc = csr_from_pairs(np.r_[fa, fb], np.r_[fe_idx, fe_idx], nv)
        deg2 = np.nonzero(fdeg == 2)[0]
        e1 = inc[ptr[deg2]]
        e2 = inc[ptr[deg2] + 1]
        sig_diff = self.ecount[e1] != self.ecount[e2]
        # patch-set signature
        sig_diff |= self._edge_patch_sig(e1) != self._edge_patch_sig(e2)
        # fans must equal number of sheets
        sig_diff |= fans[deg2] != self.ecount[e1]
        # turning angle
        o1 = np.where(self.E[e1, 0] == deg2, self.E[e1, 1], self.E[e1, 0])
        o2 = np.where(self.E[e2, 0] == deg2, self.E[e2, 1], self.E[e2, 0])
        t1 = self.V[deg2] - self.V[o1]
        t2 = self.V[o2] - self.V[deg2]
        c = np.einsum("ij,ij->i", t1, t2) / np.maximum(norm(t1) * norm(t2), 1e-30)
        turn = c < math.cos(math.radians(self.opts.corner_angle))
        corner[deg2[sig_diff | turn]] = True
        vtype[corner] = 2
        self.vtype = vtype
        self.fans = fans

    def _edge_patch_sig(self, eids):
        """Hashable signature of the patches around each edge (sum of hashed patch ids)."""
        # use patch labels of incident faces; combine order-independently
        out = np.zeros(len(eids), np.int64)
        ptr = self.ef_ptr
        for j in range(int(self.ecount[eids].max()) if len(eids) else 0):
            has = self.ecount[eids] > j
            f = self.ef_idx[ptr[eids[has]] + j]
            p = self.fpatch[f].astype(np.int64)
            out[has] += (p * 2654435761) % 1000000007 + 1
        return out

    def _build_chains(self):
        nv = len(self.V)
        fe_idx = np.nonzero(self.efeature)[0]
        fa = self.E[fe_idx, 0]
        fb = self.E[fe_idx, 1]
        ptr, inc = csr_from_pairs(np.r_[fa, fb], np.r_[fe_idx, fe_idx], nv)
        ptr_l = ptr.tolist()
        inc_l = inc.tolist()
        E = self.E
        Ea = E[:, 0].tolist()
        Eb = E[:, 1].tolist()
        vtype = self.vtype.tolist()
        visited = {}
        chains = []
        closed = []

        def walk(start, e0):
            verts = [start]
            edges = []
            cur = start
            e = e0
            while True:
                visited[e] = True
                edges.append(e)
                nxt = Eb[e] if Ea[e] == cur else Ea[e]
                verts.append(nxt)
                if vtype[nxt] == 2 or nxt == start:
                    break
                # continue through the other feature edge
                e_next = None
                for j in range(ptr_l[nxt], ptr_l[nxt + 1]):
                    ee = inc_l[j]
                    if ee != e and ee not in visited:
                        e_next = ee
                        break
                if e_next is None:
                    break
                cur = nxt
                e = e_next
            return verts, edges

        corners = np.nonzero(self.vtype == 2)[0].tolist()
        for c in corners:
            for j in range(ptr_l[c], ptr_l[c + 1]):
                e = inc_l[j]
                if e in visited:
                    continue
                verts, edges = walk(c, e)
                chains.append((verts, edges))
                closed.append(False)
        for e in fe_idx.tolist():
            if e in visited:
                continue
            start = Ea[e]
            verts, edges = walk(start, e)
            is_closed = verts[-1] == start
            chains.append((verts, edges))
            closed.append(is_closed)

        nseg = sum(len(ed) for _, ed in chains)
        seg_a = np.empty(nseg, np.int64)
        seg_b = np.empty(nseg, np.int64)
        seg_edge = np.empty(nseg, np.int64)
        seg_chain = np.empty(nseg, np.int64)
        chain_start = np.empty(len(chains), np.int64)
        chain_nseg = np.empty(len(chains), np.int64)
        vseg = np.full(nv, -1, np.int64)
        vchain = np.full(nv, -1, np.int64)
        pos = 0
        for k, (verts, edges) in enumerate(chains):
            n = len(edges)
            chain_start[k] = pos
            chain_nseg[k] = n
            seg_a[pos:pos + n] = verts[:-1]
            seg_b[pos:pos + n] = verts[1:]
            seg_edge[pos:pos + n] = edges
            seg_chain[pos:pos + n] = k
            pos += n
        self.seg_a, self.seg_b, self.seg_edge, self.seg_chain = seg_a, seg_b, seg_edge, seg_chain
        self.chain_start, self.chain_nseg = chain_start, chain_nseg
        self.chain_closed = np.array(closed, bool)
        self.chain_first = seg_a[chain_start] if len(chains) else np.zeros(0, np.int64)
        self.chain_last = seg_b[chain_start + chain_nseg - 1] if len(chains) else np.zeros(0, np.int64)
        # host segment / chain of type-1 vertices
        t1 = self.vtype[seg_a] == 1
        vseg[seg_a[t1]] = np.nonzero(t1)[0]
        vchain[seg_a[t1]] = seg_chain[t1]
        t1b = (self.vtype[seg_b] == 1) & (vseg[seg_b] < 0)
        vseg[seg_b[t1b]] = np.nonzero(t1b)[0]
        vchain[seg_b[t1b]] = seg_chain[t1b]
        # a closed loop whose start was classified type 1 is fine; type-1 vertices without a
        # chain (should not happen) become corners
        lost = (self.vtype == 1) & (vseg < 0)
        self.vtype[lost] = 2
        self.vseg = vseg
        self.vchain = vchain
        # faces incident to each segment
        self.seg_face_ptr = self.ef_ptr[seg_edge]
        self.seg_face_cnt = self.ecount[seg_edge]
        # number of sheets along each chain
        self.chain_sheets = self.ecount[seg_edge[chain_start]] if len(chains) else np.zeros(0, np.int64)

    def _build_twins(self, T_raw, T_obj, m):
        """Final bookkeeping for duplicate faces (see the duplicate step in __init__).

        twin_face: kept source face, twin_raw: raw triangle of the duplicate,
        twin_cm[:, j]: raw corner of the duplicate that holds local corner j of the kept
        face, twin_same: duplicate has the same winding, twin_obj: its object."""
        rep_raw, dup_raw = self._twin_pairs_raw
        face_of_raw = np.full(m, -1, np.int64)
        face_of_raw[self.ftri] = np.arange(len(self.F))
        s = face_of_raw[rep_raw] if len(rep_raw) else np.zeros(0, np.int64)
        ok = s >= 0
        s, dup_raw = s[ok], dup_raw[ok]
        if len(s):
            tv = self.raw_to_welded[T_raw[dup_raw]]
            cm = np.argmax(tv[:, None, :] == self.F[s][:, :, None], axis=2)
            same = ((cm[:, 1] - cm[:, 0]) % 3) == 1
        else:
            cm = np.zeros((0, 3), np.int64)
            same = np.zeros(0, bool)
        self.twin_face = s
        self.twin_raw = dup_raw
        self.twin_cm = cm
        self.twin_same = same
        self.twin_obj = T_obj[dup_raw] if len(dup_raw) else np.zeros(0, np.int64)
        self.stats["twin_faces"] = int(len(s))

    def _build_walk(self):
        """Face neighbours across non-feature manifold edges (for surface walking)."""
        nf = len(self.F)
        nbr = np.full((nf, 3), -1, np.int64)
        e, f1, k1, f2, k2 = self._edge_pairs()
        keep = ~self.efeature[e]
        nbr[f1[keep], k1[keep]] = f2[keep]
        nbr[f2[keep], k2[keep]] = f1[keep]
        self.fnbr = nbr
        r1 = nbr
        r2 = np.where(r1[:, :, None] >= 0, nbr[np.maximum(r1, 0)], -1).reshape(nf, 9)
        self.ring = np.concatenate([np.arange(nf)[:, None], r1, r2], axis=1)

    # ======================================================================================
    # queries used by the remesher / transfer
    # ======================================================================================
    def patch_bvh(self, p):
        if p not in self._patch_bvh:
            from mathutils.bvhtree import BVHTree
            fids = np.nonzero(self.fpatch == p)[0]
            tree = BVHTree.FromPolygons(self.V.tolist(), self.F[fids].tolist(), all_triangles=True)
            self._patch_bvh[p] = (tree, fids)
        return self._patch_bvh[p]

    def nearest_in_patch(self, p, point):
        tree, fids = self.patch_bvh(p)
        loc, nrm, idx, dist = tree.find_nearest(point)
        if idx is None:
            return int(fids[0])
        return int(fids[idx])

    def hidden_patches(self, max_gap, threshold=0.7, max_samples=20000):
        """Patches that are (almost) entirely covered by another surface facing the same way.

        Such under-layers (common in simulated or duplicated cloth) sit a few mm beneath
        the visible shell and can never be seen: from outside the shell hides them, from
        inside they are back faces.  A face counts as covered when a ray along its normal
        hits a face of another patch with a similar normal within `max_gap`.  Decisions
        are made per patch, so a string or strap crossing a panel never punches holes.

        Returns (hidden patch ids, covered fraction per patch)."""
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        V, F = self.V, self.F
        N = self.face_unit_normals
        A = 0.5 * norm(face_normals(V, F))
        C = self.face_centroids
        tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
        P = self.fpatch
        # a per-patch decision only needs a fraction, so large meshes are sampled
        if len(F) > max_samples:
            sample = np.sort(np.random.default_rng(0).choice(len(F), max_samples, replace=False))
        else:
            sample = np.arange(len(F))
        covered = np.zeros(len(sample), bool)
        eps = max_gap * 1e-4
        ray = tree.ray_cast
        for k, (i, c, n) in enumerate(zip(sample.tolist(), C[sample].tolist(), N[sample].tolist())):
            d = Vector(n)
            loc, nrm, j, dist = ray(Vector(c) + d * eps, d, max_gap)
            if j is not None and j != i and P[j] != P[i] and float(N[j] @ N[i]) > 0.5:
                covered[k] = True
        Ps = P[sample]
        As = A[sample]
        pa = np.bincount(Ps, weights=As, minlength=self.npatch)
        pc = np.bincount(Ps, weights=As * covered, minlength=self.npatch)
        frac = pc / np.maximum(pa, 1e-30)
        return np.nonzero(frac >= threshold)[0], frac

    def seg_faces(self, s):
        start = self.seg_face_ptr[s]
        return self.ef_idx[start:start + self.seg_face_cnt[s]]

    def vert_faces(self, v):
        return self.vf_idx[self.vf_ptr[v]:self.vf_ptr[v + 1]]

    @property
    def face_centroids(self):
        if getattr(self, "_fc", None) is None:
            self._fc = self.V[self.F].mean(1)
        return self._fc

    @property
    def face_unit_normals(self):
        if getattr(self, "_fun", None) is None:
            n = face_normals(self.V, self.F)
            self._fun = n / np.maximum(norm(n), 1e-30)[:, None]
        return self._fun

    def project_faces(self, P, host, normals=None, min_dot=0.0, max_steps=40):
        """Walk-project points onto the source surface starting from host faces.

        The walk only crosses non-feature edges, so points stay on their own patch.
        With `normals`, candidate faces facing away from the point's normal are skipped
        (unless nothing else is in reach); this keeps points from sliding around a
        hairpin fold or onto a flipped sliver.

        Returns (Q, host, bary)."""
        P = np.asarray(P, np.float64)
        host = np.asarray(host, np.int64).copy()
        n = len(P)
        Q = np.empty((n, 3))
        B = np.empty((n, 3))
        active = np.arange(n)
        SV = self.V
        SF = self.F
        FN = self.face_unit_normals if normals is not None else None
        FC = self.face_centroids
        for _ in range(max_steps):
            if len(active) == 0:
                break
            cand = self.ring[host[active]]                   # (k, R)
            valid = cand >= 0
            c = np.maximum(cand, 0)
            tri = SF[c]                                       # (k, R, 3)
            pa = P[active]
            q, bary = closest_point_triangle(pa[:, None, :], SV[tri[..., 0]], SV[tri[..., 1]], SV[tri[..., 2]])
            d = np.sqrt(((q - pa[:, None, :]) ** 2).sum(-1))
            # Tie-break on centroid distance: when the closest point is a shared vertex or
            # edge, every triangle around it is equally close and the walk would stall.
            d = d + 1e-3 * np.sqrt(((FC[c] - pa[:, None, :]) ** 2).sum(-1))
            d[~valid] = np.inf
            if FN is not None:
                facing = np.einsum("krj,kj->kr", FN[c], normals[active]) >= min_dot
                df = np.where(facing, d, np.inf)
                has = np.isfinite(df).any(axis=1)
                d = np.where(has[:, None], df, d)
            best = np.argmin(d, axis=1)
            r = np.arange(len(active))
            nh = cand[r, best]
            Q[active] = q[r, best]
            B[active] = bary[r, best]
            moved = nh != host[active]
            host[active] = nh
            active = active[moved]
        return Q, host, B

    def verify_projection(self, P, Q, host, normals, suspicious, min_dot=0.5):
        """Second opinion for walk results that look stuck.

        For the flagged points, the nearest face of the same patch is looked up in a BVH
        and accepted if it is closer and faces the same way (so a point cannot hop to the
        other side of a fold).  Returns the number of corrected points."""
        idx = np.nonzero(suspicious)[0]
        if len(idx) == 0:
            return 0
        FN = self.face_unit_normals
        fixed = 0
        for i in idx.tolist():
            p = self.fpatch[host[i]]
            tree, fids = self.patch_bvh(p)
            loc, nrm, fi, dist = tree.find_nearest(P[i])
            if fi is None:
                continue
            f = int(fids[fi])
            if normals is not None and float(np.dot(FN[f], normals[i])) < min_dot:
                continue
            cur = float(np.sqrt(((Q[i] - P[i]) ** 2).sum()))
            if dist < cur - 1e-12:
                Q[i] = np.array(loc)
                host[i] = f
                fixed += 1
        return fixed
