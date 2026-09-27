"""Feature-preserving adaptive isotropic remeshing on a patch complex.

Incremental remeshing in the spirit of Botsch & Kobbelt (2004) and Dunyach et al.
(2013): split long edges, collapse short edges, flip edges towards regular valence,
tangential relaxation, re-projection onto the source surface.

Differences from the textbook version, needed for messy game meshes:

* The surface may be non-manifold.  Every edge that is a boundary, a junction of
  3+ sheets, an attribute seam, an object border, a winding change or a crease is
  part of a feature *chain*.  Chain vertices only move along their source curve,
  chain corners never move, and chain edges are resampled like everything else.
* Faces carry a patch id.  Interior vertices are tracked on the source surface of
  their own patch by walking across source triangles, so they can never jump to a
  nearby layer (inner/outer cloth layers, crossing strings, ...).
* Collapses that would connect two different feature chains through an interior
  edge are rejected, so thin strips (strings, straps, fringe) cannot be pinched away.
* The target edge length is a field on the source surface.  It starts uniform and is
  refined wherever the remeshed surface deviates from the source by more than a
  tolerance (wrinkle crests, tight folds), with a bounded gradation.
"""

import math
import time
import numpy as np

from .geom import closest_point_segment, csr_from_pairs, face_normals, norm, unique_edges


class Remesher:
    def __init__(self, src, target_length, min_length=None, log=print):
        self.src = src
        self.log = log
        self.L0 = float(target_length)
        self.Lmin = float(min_length) if min_length else self.L0 * 0.25
        self.hi = 4.0 / 3.0
        self.lo = 4.0 / 5.0
        self.V = src.V.copy()
        self.F = src.F.copy()
        self.fpatch = src.fpatch.copy()
        nv = len(self.V)
        self.vtype = src.vtype.astype(np.int8).copy()
        self.vchain = src.vchain.copy()
        host = np.full(nv, -1, np.int64)
        t0 = np.nonzero(self.vtype == 0)[0]
        host[t0] = src.vf_idx[src.vf_ptr[t0]]
        t1 = self.vtype == 1
        host[t1] = src.vseg[t1]
        t2 = np.nonzero(self.vtype == 2)[0]
        host[t2] = t2
        self.vhost = host
        self.fedge = {}
        for a, b, c in zip(src.seg_a.tolist(), src.seg_b.tolist(), src.seg_chain.tolist()):
            self.fedge[(a, b) if a < b else (b, a)] = c
        self.sL = np.full(len(src.F), self.L0)
        self.sLmin = self._source_resolution()
        self.vn = None
        self.timing = {}
        self._update_sizing()

    def _source_resolution(self):
        """Per source face: the smallest edge length worth refining to.

        There is no surface detail finer than the source triangles themselves, so
        refinement stops at about half the local source triangle size (and never below
        the global minimum).  This keeps low-poly inputs from being subdivided just to
        reproduce their facet creases."""
        s = self.src
        area = 0.5 * norm(face_normals(s.V, s.F))
        size = np.sqrt(area * 4.0 / math.sqrt(3.0))
        # smooth over neighbours so single slivers do not decide
        fa, fb, _ = self._face_pairs()
        for _ in range(3):
            acc = size.copy()
            cnt = np.ones(len(size))
            np.add.at(acc, fa, size[fb])
            np.add.at(acc, fb, size[fa])
            np.add.at(cnt, fa, 1.0)
            np.add.at(cnt, fb, 1.0)
            size = acc / cnt
        return np.maximum(0.5 * size, self.Lmin)

    # ======================================================================================
    # helpers
    # ======================================================================================
    def _tick(self, name, t0):
        self.timing[name] = self.timing.get(name, 0.0) + time.time() - t0

    def _chain_hint_from_corner(self, chain, corner):
        s = self.src
        if s.chain_first[chain] == corner:
            return int(s.chain_start[chain])
        return int(s.chain_start[chain] + s.chain_nseg[chain] - 1)

    def _patch_face_hint(self, v, patch, point):
        """A source face of `patch` next to remesh vertex v (a chain or corner vertex)."""
        s = self.src
        t = self.vtype[v]
        if t == 1:
            cand = s.seg_faces(self.vhost[v])
        elif t == 2:
            cand = s.vert_faces(self.vhost[v])
        else:
            return int(self.vhost[v])
        cand = cand[s.fpatch[cand] == patch]
        if len(cand) == 0:
            return s.nearest_in_patch(patch, point)
        if len(cand) == 1:
            return int(cand[0])
        c = s.V[s.F[cand]].mean(1)
        return int(cand[np.argmin(((c - point) ** 2).sum(1))])

    def _feature_mask(self, E):
        if not self.fedge:
            return np.zeros(len(E), bool)
        n = max(int(E.max()) + 1, 1) if len(E) else 1
        keys = np.fromiter(((a * n + b) for a, b in self.fedge.keys()), np.int64, len(self.fedge))
        ek = E[:, 0].astype(np.int64) * n + E[:, 1]
        return np.isin(ek, keys)

    def _edge_target(self, a, b):
        return 0.5 * (self.vL[a] + self.vL[b])

    # ======================================================================================
    # sizing field
    # ======================================================================================
    def _update_sizing(self):
        s = self.src
        sL = self.sL
        eL = np.full(len(s.E), np.inf)
        np.minimum.at(eL, s.fe.ravel(), np.repeat(sL, 3))
        vLs = np.full(len(s.V), np.inf)
        np.minimum.at(vLs, s.F.ravel(), np.repeat(sL, 3))
        vL = np.empty(len(self.V))
        t0 = self.vtype == 0
        vL[t0] = sL[self.vhost[t0]]
        t1 = self.vtype == 1
        vL[t1] = eL[s.seg_edge[self.vhost[t1]]]
        t2 = self.vtype == 2
        vL[t2] = vLs[self.vhost[t2]]
        self.vL = vL

    def _face_pairs(self):
        if getattr(self, "_fpairs", None) is None:
            s = self.src
            rows = np.repeat(np.arange(len(s.E)), s.ecount)
            pos = np.arange(len(s.ef_idx)) - s.ef_ptr[rows]
            link = np.nonzero(pos > 0)[0]
            fa = s.ef_idx[link - 1]
            fb = s.ef_idx[link]
            C = s.V[s.F].mean(1)
            self._fpairs = (fa, fb, norm(C[fa] - C[fb]))
        return self._fpairs

    def _diffuse_sizing(self, grad, iters=60):
        fa, fb, d = self._face_pairs()
        L = self.sL
        for _ in range(iters):
            new = L.copy()
            np.minimum.at(new, fa, L[fb] + grad * d)
            np.minimum.at(new, fb, L[fa] + grad * d)
            if np.array_equal(new, L):
                break
            L = new
        self.sL = L

    def measure_error(self):
        """Distance of every edge midpoint to the source.

        Chain edges are measured against their source curve, all other edges against
        the source surface of their patch.  Returns (E, err, host_face) where host_face
        is a source face near the measured spot (used to localise refinement)."""
        V, F = self.V, self.F
        nf = len(F)
        E, fe = unique_edges(F)
        mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
        elen = norm(V[E[:, 0]] - V[E[:, 1]])
        eface = np.full(len(E), -1, np.int64)
        eface[fe.ravel()] = np.repeat(np.arange(nf), 3)
        a, b = E[:, 0], E[:, 1]
        ta, tb = self.vtype[a], self.vtype[b]
        err = np.zeros(len(E))
        host = np.full(len(E), -1, np.int64)
        s = self.src

        # --- chain edges: 1D projection onto the source curve
        is_chain = self._feature_mask(E)
        ci = np.nonzero(is_chain)[0]
        if len(ci):
            chain = np.fromiter((self.fedge[(int(x), int(y))] for x, y in zip(a[ci], b[ci])), np.int64, len(ci))
            seg = np.where(ta[ci] == 1, self.vhost[a[ci]], np.where(tb[ci] == 1, self.vhost[b[ci]], -1))
            for j in np.nonzero(seg < 0)[0].tolist():
                corner = self.vhost[a[ci[j]]]
                seg[j] = self._chain_hint_from_corner(chain[j], corner)
            Qc, segc, _ = self.project_chain(mid[ci], seg, chain)
            err[ci] = norm(Qc - mid[ci])
            host[ci] = s.ef_idx[s.seg_face_ptr[segc]]

        # --- everything else: surface walk inside the patch
        oi = np.nonzero(~is_chain)[0]
        if len(oi):
            hint = np.full(len(oi), -1, np.int64)
            m = ta[oi] == 0
            hint[m] = self.vhost[a[oi][m]]
            m = (hint < 0) & (tb[oi] == 0)
            hint[m] = self.vhost[b[oi][m]]
            for j in np.nonzero(hint < 0)[0].tolist():
                i = oi[j]
                hint[j] = self._patch_face_hint(a[i], self.fpatch[eface[i]], mid[i])
            FN = face_normals(V, F)
            FN /= np.maximum(norm(FN), 1e-30)[:, None]
            nrm = FN[eface[oi]]
            P = mid[oi]
            Q, h, _ = s.project_faces(P, hint, normals=nrm, min_dot=0.0)
            d = norm(Q - P)
            s.verify_projection(P, Q, h, nrm, d > 0.25 * elen[oi])
            err[oi] = norm(Q - P)
            host[oi] = h
        return E, err, host

    def refine_sizing(self, eps, grad=0.35):
        """Shrink the sizing field where the surface error exceeds eps."""
        t0 = time.time()
        E, err, host = self.measure_error()
        Le = self._edge_target(E[:, 0], E[:, 1])
        bad = err > eps
        if bad.any():
            req = Le[bad] * np.sqrt(eps / err[bad])
            req = np.maximum(req, self.sLmin[host[bad]])
            np.minimum.at(self.sL, host[bad], req)
            self._diffuse_sizing(grad)
            self._update_sizing()
        self._tick("sizing", t0)
        return int(bad.sum()), float(err.max()) if len(err) else 0.0

    # ======================================================================================
    # split
    # ======================================================================================
    def split_long(self):
        t0 = time.time()
        V, F = self.V, self.F
        nv, nf = len(V), len(F)
        E, fe = unique_edges(F)
        d = norm(V[E[:, 0]] - V[E[:, 1]])
        long = d > self.hi * self._edge_target(E[:, 0], E[:, 1])
        if not long.any():
            self._tick("split", t0)
            return 0
        eid = np.nonzero(long)[0]
        ns = len(eid)
        mid = np.full(len(E), -1, np.int64)
        mid[eid] = nv + np.arange(ns)
        a = E[eid, 0]
        b = E[eid, 1]
        newpos = 0.5 * (V[a] + V[b])
        eface = np.full(len(E), -1, np.int64)
        eface[fe.ravel()] = np.repeat(np.arange(nf), 3)

        fedge = self.fedge
        chain = np.fromiter((fedge.get((x, y), -1) for x, y in zip(a.tolist(), b.tolist())), np.int64, ns)
        ta, tb = self.vtype[a], self.vtype[b]
        ha, hb = self.vhost[a], self.vhost[b]
        host = np.full(ns, -1, np.int64)
        m0 = chain < 0
        sel = m0 & (ta == 0)
        host[sel] = ha[sel]
        sel = m0 & (ta != 0) & (tb == 0)
        host[sel] = hb[sel]
        m1 = ~m0
        sel = m1 & (ta == 1)
        host[sel] = ha[sel]
        sel = m1 & (ta != 1) & (tb == 1)
        host[sel] = hb[sel]
        for i in np.nonzero(host < 0)[0].tolist():
            if chain[i] >= 0:
                corner = ha[i] if ta[i] == 2 else hb[i]
                host[i] = self._chain_hint_from_corner(chain[i], corner)
            else:
                host[i] = self._patch_face_hint(a[i], self.fpatch[eface[eid[i]]], newpos[i])

        Vx = np.concatenate([V, newpos])
        m = long[fe]
        cnt = m.sum(1)
        outF = [F[cnt == 0]]
        outP = [self.fpatch[cnt == 0]]
        P = self.fpatch
        idx = np.nonzero(cnt == 1)[0]
        if len(idx):
            k = np.argmax(m[idx], axis=1)
            r = (k[:, None] + np.arange(3)) % 3
            f = F[idx[:, None], r]
            mm = mid[fe[idx[:, None], r][:, 0]]
            v0, v1, v2 = f[:, 0], f[:, 1], f[:, 2]
            outF += [np.stack([v0, mm, v2], 1), np.stack([mm, v1, v2], 1)]
            outP += [P[idx], P[idx]]
        idx = np.nonzero(cnt == 2)[0]
        if len(idx):
            j = np.argmin(m[idx], axis=1)
            k = (j + 1) % 3
            r = (k[:, None] + np.arange(3)) % 3
            f = F[idx[:, None], r]
            fr = fe[idx[:, None], r]
            m01 = mid[fr[:, 0]]
            m12 = mid[fr[:, 1]]
            v0, v1, v2 = f[:, 0], f[:, 1], f[:, 2]
            d1 = norm(Vx[v0] - Vx[m12])
            d2 = norm(Vx[m01] - Vx[v2])
            u1 = (d1 <= d2)[:, None]
            outF.append(np.stack([m01, v1, m12], 1))
            outF.append(np.where(u1, np.stack([v0, m01, m12], 1), np.stack([v0, m01, v2], 1)))
            outF.append(np.where(u1, np.stack([v0, m12, v2], 1), np.stack([m01, m12, v2], 1)))
            outP += [P[idx]] * 3
        idx = np.nonzero(cnt == 3)[0]
        if len(idx):
            f = F[idx]
            fr = fe[idx]
            m01, m12, m20 = mid[fr[:, 0]], mid[fr[:, 1]], mid[fr[:, 2]]
            v0, v1, v2 = f[:, 0], f[:, 1], f[:, 2]
            outF += [np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1),
                     np.stack([m20, m12, v2], 1), np.stack([m01, m12, m20], 1)]
            outP += [P[idx]] * 4
        self.F = np.concatenate(outF)
        self.fpatch = np.concatenate(outP)
        self.V = Vx
        self.vtype = np.concatenate([self.vtype, np.where(chain >= 0, 1, 0).astype(np.int8)])
        self.vchain = np.concatenate([self.vchain, chain])
        self.vhost = np.concatenate([self.vhost, host])
        self.vL = np.concatenate([self.vL, 0.5 * (self.vL[a] + self.vL[b])])
        if self.vn is not None:
            nn = self.vn[a] + self.vn[b]
            self.vn = np.concatenate([self.vn, nn / np.maximum(norm(nn), 1e-30)[:, None]])
        for i in np.nonzero(chain >= 0)[0].tolist():
            x, y, c = int(a[i]), int(b[i]), int(chain[i])
            mv = nv + i
            del fedge[(x, y)]
            fedge[(x, mv)] = c
            fedge[(y, mv)] = c
        self._tick("split", t0)
        return ns

    # ======================================================================================
    # collapse
    # ======================================================================================
    def collapse_short(self, ratio=None):
        t0 = time.time()
        V, F = self.V, self.F
        nv = len(V)
        E, fe = unique_edges(F)
        d = norm(V[E[:, 0]] - V[E[:, 1]])
        lo_ratio = self.lo if ratio is None else ratio
        Lt = self._edge_target(E[:, 0], E[:, 1])
        cand = np.nonzero(d < lo_ratio * Lt)[0]
        if len(cand) == 0:
            self._tick("collapse", t0)
            return 0
        cand = cand[np.argsort(d[cand] / Lt[cand], kind="stable")]
        Fl = F.tolist()
        Vl = V.tolist()
        vt = self.vtype.tolist()
        vc = self.vchain.tolist()
        vLl = self.vL.tolist()
        vf = [set() for _ in range(nv)]
        for fi, (x, y, z) in enumerate(Fl):
            vf[x].add(fi)
            vf[y].add(fi)
            vf[z].add(fi)
        falive = [True] * len(Fl)
        valive = [True] * nv
        fedge = self.fedge
        fnb = {}
        for (x, y), c in fedge.items():
            fnb.setdefault(x, {})[y] = c
            fnb.setdefault(y, {})[x] = c
        hi = self.hi
        amin2 = (1e-6 * self.Lmin * self.Lmin) ** 2
        COS_LIM = 0.2

        def tri_n(p, q, r):
            ax, ay, az = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            bx, by, bz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
            return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

        def can_remove(r, s):
            t = vt[r]
            if t == 2:
                return False
            if t == 1:
                key = (r, s) if r < s else (s, r)
                c = fedge.get(key)
                if c is None or c != vc[r]:
                    return False
                ts = vt[s]
                return ts == 2 or (ts == 1 and vc[s] == c)
            return True

        def fan_ok(v, faces, p, moving_is_interior):
            """Check faces around v after moving v to p.

            Healthy fans must stay healthy: no face may become degenerate or fold over.
            Fans that are already damaged (degenerate or folded faces, typical right after
            projecting into a cusp) are accepted as long as the move does not add damage,
            so such spots can always be collapsed away."""
            olds = []
            sx = sy = sz = 0.0
            for f in faces:
                a, b, c = Fl[f]
                o = tri_n(Vl[a], Vl[b], Vl[c])
                olds.append(o)
                sx += o[0]; sy += o[1]; sz += o[2]
            ln = math.sqrt(sx * sx + sy * sy + sz * sz)
            use_ref = moving_is_interior and ln > 1e-30
            if use_ref:
                rx, ry, rz = sx / ln, sy / ln, sz / ln
            bad_before = 0
            bad_after = 0
            for f, o in zip(faces, olds):
                tri = Fl[f]
                pts = [p if w == v else Vl[w] for w in tri]
                n = tri_n(pts[0], pts[1], pts[2])
                l2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
                lo_ = o[0] * o[0] + o[1] * o[1] + o[2] * o[2]
                if lo_ < amin2:
                    bad_before += 1
                elif use_ref and o[0] * rx + o[1] * ry + o[2] * rz < COS_LIM * math.sqrt(lo_):
                    bad_before += 1
                if l2 < amin2:
                    bad_after += 1
                elif use_ref:
                    if n[0] * rx + n[1] * ry + n[2] * rz < COS_LIM * math.sqrt(l2):
                        bad_after += 1
                elif lo_ > amin2 and (n[0] * o[0] + n[1] * o[1] + n[2] * o[2]) < COS_LIM * math.sqrt(l2 * lo_):
                    return False
            if bad_after == 0:
                return True
            # an already damaged fan may be collapsed as long as it does not get worse
            return bad_before > 0 and bad_after <= bad_before

        def try_collapse(r, s):
            fr = vf[r]
            fs = vf[s]
            shared = fr & fs
            if not shared:
                return False
            Nr = set()
            for f in fr:
                Nr.update(Fl[f])
            Ns = set()
            for f in fs:
                Ns.update(Fl[f])
            opp = set()
            for f in shared:
                for w in Fl[f]:
                    if w != r and w != s:
                        opp.add(w)
            Nr.discard(r); Nr.discard(s)
            Ns.discard(r); Ns.discard(s)
            if (Nr & Ns) != opp:
                return False
            if vt[r] == 1:
                # the other chain neighbour of r must not already touch s
                for z in fnb.get(r, {}):
                    if z != s and z in Ns:
                        return False
            both_interior = vt[r] == 0 and vt[s] == 0
            if both_interior:
                pr, ps = Vl[r], Vl[s]
                p = [(pr[0] + ps[0]) * 0.5, (pr[1] + ps[1]) * 0.5, (pr[2] + ps[2]) * 0.5]
            else:
                p = Vl[s]
            Ls = vLl[s]
            for w in Nr | Ns:
                q = Vl[w]
                dx, dy, dz = p[0] - q[0], p[1] - q[1], p[2] - q[2]
                lim = hi * 0.5 * (Ls + vLl[w])
                if dx * dx + dy * dy + dz * dz > lim * lim:
                    return False
            rest_r = fr - shared
            if not fan_ok(r, rest_r, p, vt[r] == 0):
                return False
            rest_s = fs - shared
            if both_interior and not fan_ok(s, rest_s, p, True):
                return False
            # apply
            for f in shared:
                falive[f] = False
                for w in Fl[f]:
                    if w != r:
                        vf[w].discard(f)
            for f in rest_r:
                tri = Fl[f]
                tri[tri.index(r)] = s
                fs.add(f)
            vf[r] = set()
            valive[r] = False
            Vl[s] = p
            if both_interior:
                vLl[s] = min(vLl[s], vLl[r])
            if vt[r] == 1:
                nb = fnb.pop(r)
                for w, c in nb.items():
                    del fedge[(r, w) if r < w else (w, r)]
                    fnb[w].pop(r, None)
                    if w != s:
                        fedge[(s, w) if s < w else (w, s)] = c
                        fnb[s][w] = c
                        fnb[w][s] = c
            return True

        count = 0
        Ea = E[:, 0].tolist()
        Eb = E[:, 1].tolist()
        for e in cand.tolist():
            x, y = Ea[e], Eb[e]
            if not (valive[x] and valive[y]):
                continue
            px, py = Vl[x], Vl[y]
            dx, dy, dz = px[0] - py[0], px[1] - py[1], px[2] - py[2]
            lim = lo_ratio * 0.5 * (vLl[x] + vLl[y])
            if dx * dx + dy * dy + dz * dz >= lim * lim:
                continue
            # prefer removing the lower-ranked vertex
            if vt[x] > vt[y]:
                x, y = y, x
            done = False
            if can_remove(x, y):
                done = try_collapse(x, y)
            if not done and can_remove(y, x):
                done = try_collapse(y, x)
            if done:
                count += 1

        self.vL = np.array(vLl)
        self._compact(Fl, falive, Vl, valive)
        self._tick("collapse", t0)
        return count

    def _compact(self, Fl, falive, Vl, valive):
        fa = np.array(falive, bool)
        F = np.array(Fl, np.int64)[fa] if len(Fl) else np.zeros((0, 3), np.int64)
        fpatch = self.fpatch[fa]
        V = np.array(Vl, np.float64)
        nv = len(V)
        used = np.zeros(nv, bool)
        used[F.ravel()] = True
        used &= np.array(valive, bool)
        remap = np.cumsum(used) - 1
        self.F = remap[F]
        self.fpatch = fpatch
        self.V = V[used]
        self.vtype = self.vtype[used]
        self.vchain = self.vchain[used]
        self.vhost = self.vhost[used]
        self.vL = self.vL[used]
        if self.vn is not None:
            self.vn = self.vn[used]
        rm = remap.tolist()
        ul = used.tolist()
        self.fedge = {(rm[a], rm[b]): c for (a, b), c in self.fedge.items() if ul[a] and ul[b]}

    # ======================================================================================
    # flips
    # ======================================================================================
    def flip_edges(self, mode="valence"):
        t0 = time.time()
        V, F = self.V, self.F
        nv, nf = len(V), len(F)
        E, fe = unique_edges(F)
        ne = len(E)
        ep, ei = csr_from_pairs(fe.ravel(), np.repeat(np.arange(nf), 3), ne)
        cnt = np.diff(ep)
        feat = self._feature_mask(E)
        cand = np.nonzero((cnt == 2) & ~feat)[0]
        if len(cand) == 0:
            self._tick("flip", t0)
            return 0
        val = np.bincount(E.ravel(), minlength=nv).astype(np.int64)
        # target valence: interior 6, chain vertex 2 + 2 * sheets, corners ignored
        tgt = np.full(nv, 6, np.int64)
        fidx = np.nonzero(feat)[0]
        sheets = np.zeros(nv, np.int64)
        sheets[E[fidx, 0]] = cnt[fidx]
        sheets[E[fidx, 1]] = cnt[fidx]
        t1 = self.vtype == 1
        tgt[t1] = 2 + 2 * sheets[t1]
        use = self.vtype != 2
        f1 = ei[ep[cand]]
        f2 = ei[ep[cand] + 1]
        a = E[cand, 0]
        b = E[cand, 1]
        c = F[f1].sum(1) - a - b
        dd = F[f2].sum(1) - a - b
        if mode == "valence":
            def dev(v, delta):
                return np.where(use[v], (val[v] + delta - tgt[v]) ** 2, 0)
            before = dev(a, 0) + dev(b, 0) + dev(c, 0) + dev(dd, 0)
            after = dev(a, -1) + dev(b, -1) + dev(c, 1) + dev(dd, 1)
            gain = (before - after).astype(np.float64)
            keep = (gain > 0) & (c != dd)
        else:
            # Delaunay: angles opposite the edge
            def ang(o, p, q):
                u = V[p] - V[o]
                w = V[q] - V[o]
                return np.arctan2(norm(np.cross(u, w)), np.einsum("ij,ij->i", u, w))
            s = ang(c, a, b) + ang(dd, a, b)
            gain = s - math.pi
            keep = (gain > 1e-3) & (c != dd)
        order = cand[keep][np.argsort(-gain[keep], kind="stable")]
        if len(order) == 0:
            self._tick("flip", t0)
            return 0

        Fl = F.tolist()
        Vl = V.tolist()
        vf = [set() for _ in range(nv)]
        for fi, (x, y, z) in enumerate(Fl):
            vf[x].add(fi)
            vf[y].add(fi)
            vf[z].add(fi)
        vall = val.tolist()
        tgtl = tgt.tolist()
        usel = use.tolist()
        amin2 = (1e-6 * self.Lmin * self.Lmin) ** 2

        def tri_n(p, q, r):
            ax, ay, az = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            bx, by, bz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
            return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

        def angle_at(o, p, q):
            ux, uy, uz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
            wx, wy, wz = q[0] - o[0], q[1] - o[1], q[2] - o[2]
            cx, cy, cz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), ux * wx + uy * wy + uz * wz)

        count = 0
        Ea = E[:, 0].tolist()
        Eb = E[:, 1].tolist()
        for e in order.tolist():
            x, y = Ea[e], Eb[e]
            faces = vf[x] & vf[y]
            if len(faces) != 2:
                continue
            g1, g2 = faces
            t = Fl[g1]
            i = t.index(x)
            if t[(i + 1) % 3] == y:
                a_, b_ = x, y
            else:
                a_, b_ = y, x
            if Fl[g1][(Fl[g1].index(a_) + 1) % 3] != b_:
                g1, g2 = g2, g1
            t1_ = Fl[g1]
            t2_ = Fl[g2]
            if t1_[(t1_.index(a_) + 1) % 3] != b_:
                continue
            if t2_[(t2_.index(b_) + 1) % 3] != a_:
                continue
            c_ = t1_[0] + t1_[1] + t1_[2] - a_ - b_
            d_ = t2_[0] + t2_[1] + t2_[2] - a_ - b_
            if c_ == d_:
                continue
            # edge c-d must not exist
            if any(d_ in Fl[f] for f in vf[c_]):
                continue
            if mode == "valence":
                def dv(v, delta):
                    if not usel[v]:
                        return 0
                    q = vall[v] + delta - tgtl[v]
                    return q * q
                before = dv(a_, 0) + dv(b_, 0) + dv(c_, 0) + dv(d_, 0)
                after = dv(a_, -1) + dv(b_, -1) + dv(c_, 1) + dv(d_, 1)
                if after >= before:
                    continue
            else:
                if angle_at(Vl[c_], Vl[a_], Vl[b_]) + angle_at(Vl[d_], Vl[a_], Vl[b_]) <= math.pi + 1e-3:
                    continue
            pa, pb, pc, pd = Vl[a_], Vl[b_], Vl[c_], Vl[d_]
            n1 = tri_n(pa, pb, pc)
            n2 = tri_n(pb, pa, pd)
            m1 = tri_n(pc, pa, pd)
            m2 = tri_n(pd, pb, pc)
            sx, sy, sz = n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]
            l1 = m1[0] * m1[0] + m1[1] * m1[1] + m1[2] * m1[2]
            l2 = m2[0] * m2[0] + m2[1] * m2[1] + m2[2] * m2[2]
            if l1 < amin2 or l2 < amin2:
                continue
            if m1[0] * sx + m1[1] * sy + m1[2] * sz <= 0 or m2[0] * sx + m2[1] * sy + m2[2] * sz <= 0:
                continue
            ln1 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]
            ln2 = n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]
            cos_new = (m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2]) / math.sqrt(l1 * l2)
            if ln1 > amin2 and ln2 > amin2:
                cos_old = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / math.sqrt(ln1 * ln2)
            else:
                cos_old = -1.0
            if cos_new < min(cos_old, 0.7):
                continue
            # apply
            Fl[g1] = [c_, a_, d_]
            Fl[g2] = [d_, b_, c_]
            vf[b_].discard(g1)
            vf[d_].add(g1)
            vf[a_].discard(g2)
            vf[c_].add(g2)
            vall[a_] -= 1
            vall[b_] -= 1
            vall[c_] += 1
            vall[d_] += 1
            count += 1
        self.F = np.array(Fl, np.int64)
        self._tick("flip", t0)
        return count

    # ======================================================================================
    # smoothing and projection
    # ======================================================================================
    def _chain_neighbours(self):
        nv = len(self.V)
        if not self.fedge:
            return np.zeros(nv, np.int64), np.zeros((nv, 3))
        k = np.array(list(self.fedge.keys()), np.int64)
        a, b = k[:, 0], k[:, 1]
        cnt = np.bincount(np.r_[a, b], minlength=nv)
        s = np.zeros((nv, 3))
        V = self.V
        for dim in range(3):
            s[:, dim] = np.bincount(a, weights=V[b, dim], minlength=nv) + np.bincount(b, weights=V[a, dim], minlength=nv)
        return cnt, s

    def _vertex_normals(self):
        V, F = self.V, self.F
        nv = len(V)
        N = face_normals(V, F)
        fr = F.ravel()
        vn = np.empty((nv, 3))
        for dim in range(3):
            vn[:, dim] = np.bincount(fr, weights=np.repeat(N[:, dim], 3), minlength=nv)
        vn /= np.maximum(norm(vn), 1e-30)[:, None]
        return vn, N

    def relax(self, lam=0.6):
        t0 = time.time()
        V, F = self.V, self.F
        nv = len(V)
        vn, N = self._vertex_normals()
        self.vn = vn
        A = 0.5 * norm(N)
        Lf = self.vL[F].mean(1)
        w = A / np.maximum(Lf * Lf, 1e-300)
        C = V[F].mean(1)
        fr = F.ravel()
        wsum = np.bincount(fr, weights=np.repeat(w, 3), minlength=nv)
        acc = np.empty((nv, 3))
        for dim in range(3):
            acc[:, dim] = np.bincount(fr, weights=np.repeat(w * C[:, dim], 3), minlength=nv)
        q = acc / np.maximum(wsum, 1e-300)[:, None]
        dv = q - V
        dv -= np.einsum("ij,ij->i", dv, vn)[:, None] * vn
        t0m = (self.vtype == 0) & (wsum > 0)
        V[t0m] += lam * dv[t0m]
        cnt, s = self._chain_neighbours()
        t1m = (self.vtype == 1) & (cnt == 2)
        V[t1m] += lam * (s[t1m] / 2.0 - V[t1m])
        self._tick("relax", t0)

    def project(self):
        t0 = time.time()
        s = self.src
        V = self.V
        if self.vn is None or len(self.vn) != len(V):
            self.vn, _ = self._vertex_normals()
        i0 = np.nonzero(self.vtype == 0)[0]
        if len(i0):
            P = V[i0]
            Q, host, _ = s.project_faces(P, self.vhost[i0], normals=self.vn[i0], min_dot=0.0)
            s.verify_projection(P, Q, host, self.vn[i0], norm(Q - P) > 0.35 * self.vL[i0])
            V[i0] = Q
            self.vhost[i0] = host
        i1 = np.nonzero(self.vtype == 1)[0]
        if len(i1):
            Q, seg, _ = self.project_chain(V[i1], self.vhost[i1], self.vchain[i1])
            V[i1] = Q
            self.vhost[i1] = seg
        i2 = np.nonzero(self.vtype == 2)[0]
        V[i2] = s.V[self.vhost[i2]]
        self._update_sizing()
        self._tick("project", t0)

    def project_chain(self, P, seg, chain, W=4, max_iter=30):
        s = self.src
        seg = seg.copy()
        n = len(P)
        Q = P.copy()
        T = np.zeros(n)
        start = s.chain_start[chain]
        nseg = s.chain_nseg[chain]
        closed = s.chain_closed[chain]
        offs = np.arange(-W, W + 1)
        active = np.arange(n)
        for _ in range(max_iter):
            if len(active) == 0:
                break
            st = start[active]
            ns_ = nseg[active]
            pos = (seg[active] - st)[:, None] + offs[None, :]
            cl = closed[active][:, None]
            valid = cl | ((pos >= 0) & (pos < ns_[:, None]))
            pos = np.where(cl, pos % ns_[:, None], np.clip(pos, 0, ns_[:, None] - 1))
            sid = st[:, None] + pos
            A = s.V[s.seg_a[sid]]
            B = s.V[s.seg_b[sid]]
            q, t = closest_point_segment(P[active][:, None, :], A, B)
            d2 = ((q - P[active][:, None, :]) ** 2).sum(-1)
            d2[~valid] = np.inf
            best = np.argmin(d2, axis=1)
            r = np.arange(len(active))
            new = sid[r, best]
            Q[active] = q[r, best]
            T[active] = t[r, best]
            at_rim = ((best == 0) | (best == 2 * W)) & (new != seg[active])
            seg[active] = new
            active = active[at_rim]
        return Q, seg, T

    # ======================================================================================
    # driver
    # ======================================================================================
    def stats(self):
        V, F = self.V, self.F
        E, fe = unique_edges(F)
        L = norm(V[E[:, 0]] - V[E[:, 1]])
        Lt = self._edge_target(E[:, 0], E[:, 1])
        N = face_normals(V, F)
        A = 0.5 * norm(N)
        el = np.stack([norm(V[F[:, 1]] - V[F[:, 0]]), norm(V[F[:, 2]] - V[F[:, 1]]), norm(V[F[:, 0]] - V[F[:, 2]])], 1)
        q = 4 * math.sqrt(3) * A / np.maximum((el ** 2).sum(1), 1e-300)
        r = L / Lt
        return {
            "verts": len(V), "tris": len(F),
            "len/target mean": round(float(r.mean()), 3), "len/target std": round(float(r.std()), 3),
            "q_min": round(float(q.min()), 4), "q_p1": round(float(np.percentile(q, 1)), 3),
            "q_med": round(float(np.median(q)), 3), "q<0.3": round(float((q < 0.3).mean()), 4),
            "degenerate": int((el.min(1) < 1e-4 * self.Lmin).sum()),
        }

    def finalize(self):
        """Last cleanup after the final projection: remove edges that collapsed to
        (almost) nothing, then re-flip.  Vertices are not moved any more."""
        n = self.collapse_short(ratio=0.05)
        self.flip_edges("valence")
        return n

    def run(self, iterations=10, polish=3, adaptive_eps=None, progress=None, cancel=None):
        total = iterations + polish
        refine_rounds = set()
        if adaptive_eps:
            refine_rounds = {i for i in range(1, max(iterations - 2, 2), 2)}
        for it in range(iterations):
            ns = self.split_long()
            nc = self.collapse_short()
            nfl = self.flip_edges("valence")
            self.relax()
            self.project()
            msg = f"remeshing {it + 1}/{iterations}: split {ns}, collapse {nc}, flip {nfl}"
            if it in refine_rounds:
                nb, emax = self.refine_sizing(adaptive_eps)
                msg += f", refine {nb} (max err {emax * 1000:.2f} mm)"
            if progress:
                progress((it + 1) / total, msg)
            if cancel and cancel():
                return False
        for it in range(polish):
            nc = self.collapse_short(ratio=0.5)
            nfl = self.flip_edges("valence")
            self.relax(0.5)
            self.project()
            if progress:
                progress((iterations + it + 1) / total, f"polish {it + 1}/{polish}: collapse {nc}, flip {nfl}")
            if cancel and cancel():
                return False
        self.finalize()
        return True
