"""Seam placement for rebuilding UV layouts.

Islands follow the structure of the model instead of face angles (which is what makes
Smart UV Project shred ruched cloth into confetti):

* seams: the old UV island borders (usually the original pattern pieces; their
  layout is what is broken, not their outline), junctions of 3+ sheets, borders
  between objects / materials, winding changes and optionally sharp creases;
* every island is then made a topological disk, the only shape that unwraps flat
  without folding over itself:
  - islands with several boundary loops (rings, tubes) get the shortest cuts joining
    the loops,
  - closed pieces get a slit between their two farthest points,
  - anything else (handles, e.g. a strap or string attached to a panel at both ends)
    is bisected between its two farthest points until the pieces are disks.
"""

import heapq

import numpy as np

from .geom import connected_components, csr_from_pairs, norm, relabel
from .source import K_JUNCTION, K_PART, K_SEAM, K_SHARP, K_WINDING


class _Island:
    """Split-vertex view of a set of source faces (as the unwrapper will see it)."""

    def __init__(self, src, cut, faces):
        F = src.F
        self.faces = faces
        n = len(faces)
        local = np.full(len(F), -1, np.int64)
        local[faces] = np.arange(n)
        eids = np.unique(src.fe[faces].ravel())
        two = eids[(src.ecount[eids] == 2) & ~cut[eids]]
        p = src.ef_ptr[two]
        f1, k1 = src.ef_idx[p], src.ef_loc[p]
        f2, k2 = src.ef_idx[p + 1], src.ef_loc[p + 1]
        inside = (local[f1] >= 0) & (local[f2] >= 0)
        f1, k1, f2, k2 = f1[inside], k1[inside], f2[inside], k2[inside]
        u = F[f1, k1]
        same = F[f2, k2] == u
        c1u = local[f1] * 3 + k1
        c1w = local[f1] * 3 + (k1 + 1) % 3
        c2u = local[f2] * 3 + np.where(same, k2, (k2 + 1) % 3)
        c2w = local[f2] * 3 + np.where(same, (k2 + 1) % 3, k2)
        lab = relabel(connected_components(n * 3, np.r_[c1u, c1w], np.r_[c2u, c2w]))
        self.cls = lab.reshape(n, 3)
        self.nv = int(lab.max()) + 1 if len(lab) else 0
        se = np.stack([self.cls[:, [0, 1]], self.cls[:, [1, 2]], self.cls[:, [2, 0]]], 1).reshape(-1, 2)
        se.sort(axis=1)
        key = se[:, 0] * (self.nv + 1) + se[:, 1]
        _, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        self.E = se[first]
        self.src_e = src.fe[faces].reshape(-1)[first]
        self.cnt = cnt
        self.face_edges = inv.reshape(n, 3)
        self.pos = np.zeros((self.nv, 3))
        self.pos[self.cls.ravel()] = src.V[F[faces].ravel()]
        self.chi = self.nv - len(self.E) + n
        bnd = np.nonzero(cnt == 1)[0]
        self.bnd = bnd
        if len(bnd):
            bl = connected_components(self.nv, self.E[bnd, 0], self.E[bnd, 1])
            self.loop_of = bl
            self.nloops = len(np.unique(bl[np.unique(self.E[bnd].ravel())]))
        else:
            self.loop_of = None
            self.nloops = 0
        inner = cnt == 2
        lengths = norm(self.pos[self.E[:, 0]] - self.pos[self.E[:, 1]])
        self.w = np.where(inner, lengths, np.inf)   # for cuts: through the interior only
        self.w_all = lengths                          # for distances: along any island edge

    @property
    def genus(self):
        return (2 - self.chi - self.nloops) // 2

    @property
    def is_disk(self):
        return self.nloops == 1 and self.chi == 1


def _dijkstra(nv, E, w, sources, targets=None):
    """Returns (dist array, predecessor edge array, first target reached or None)."""
    ptr, idx = csr_from_pairs(np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(len(E)), np.arange(len(E))], nv)
    ptr_l, idx_l = ptr.tolist(), idx.tolist()
    Ea, Eb, wl = E[:, 0].tolist(), E[:, 1].tolist(), w.tolist()
    dist = [np.inf] * nv
    prev = [-1] * nv
    heap = []
    for s in sources:
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    tset = set(targets) if targets is not None else None
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        if tset is not None and v in tset:
            return np.array(dist), np.array(prev), v
        for j in range(ptr_l[v], ptr_l[v + 1]):
            e = idx_l[j]
            if wl[e] == np.inf:
                continue
            u = Eb[e] if Ea[e] == v else Ea[e]
            nd = d + wl[e]
            if nd < dist[u]:
                dist[u] = nd
                prev[u] = e
                heapq.heappush(heap, (nd, u))
    return np.array(dist), np.array(prev), None


def _path_edges(isl, prev, end):
    path = []
    v = end
    E = isl.E
    while v >= 0 and prev[v] >= 0:
        e = prev[v]
        path.append(e)
        v = E[e, 1] if E[e, 0] == v else E[e, 0]
    return np.array(path, np.int64)


def _bisect(isl):
    """Split an island into two halves around its two farthest points.
    Returns the island edges to cut (inner edges between the halves)."""
    c = isl.pos.mean(0)
    a = int(np.argmax(norm(isl.pos - c)))
    da, _, _ = _dijkstra(isl.nv, isl.E, isl.w_all, [a])
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    db, _, _ = _dijkstra(isl.nv, isl.E, isl.w_all, [b])
    fa = np.where(np.isfinite(da), da, 1e30)[isl.cls].mean(1)
    fb = np.where(np.isfinite(db), db, 1e30)[isl.cls].mean(1)
    side = fa < fb
    fe = isl.face_edges
    # inner edges with faces on both sides
    s_edge_a = np.zeros(len(isl.E), bool)
    s_edge_b = np.zeros(len(isl.E), bool)
    s_edge_a[fe[side].ravel()] = True
    s_edge_b[fe[~side].ravel()] = True
    return np.nonzero(s_edge_a & s_edge_b & (isl.cnt == 2))[0]


def _diameter(isl):
    """Approximate geodesic diameter (double sweep) of an island."""
    c = isl.pos.mean(0)
    a = int(np.argmax(norm(isl.pos - c)))
    da, _, _ = _dijkstra(isl.nv, isl.E, isl.w_all, [a])
    fin = np.isfinite(da)
    if not fin.any():
        return 0.0
    b = int(np.argmax(np.where(fin, da, -1)))
    db, _, _ = _dijkstra(isl.nv, isl.E, isl.w_all, [b])
    return float(np.max(np.where(np.isfinite(db), db, 0)))


def compute_uv_cuts(src, keep_old_seams=True, cut_sharp=False, split_long=True, extra_cut=None, max_rounds=400):
    """Seams (bool mask over src.E) that split the model into disk-shaped islands.

    With split_long, islands clearly longer than the longest main panel (long hems,
    flanges going all the way round) are cut in pieces, so a single strip does not
    dictate the scale of the whole layout.  `extra_cut` adds seams that are already
    decided (e.g. cuts at narrow bridges found after a first unwrap)."""
    kinds = K_JUNCTION | K_PART | K_WINDING
    if keep_old_seams:
        kinds |= K_SEAM
    if cut_sharp:
        kinds |= K_SHARP
    cut = (src.ekind & kinds) != 0
    if extra_cut is not None:
        cut |= extra_cut
    nonman = src.ecount != 2
    island = src._face_components(len(src.F), src.E, src.fe, cut | nonman)
    stats = {"islands": 0, "rings_cut": 0, "closed_cut": 0, "bisected": 0, "long_split": 0}
    order = np.argsort(island, kind="stable")
    starts = np.r_[0, np.nonzero(np.diff(island[order]))[0] + 1]
    ends = np.r_[starts[1:], len(order)]
    queue = [order[s:e] for s, e in zip(starts.tolist(), ends.tolist())]
    # the main panels (the islands with at least a quarter of the largest area) set the
    # length scale; only strips clearly longer than all of them are cut
    max_len = thin_len = np.inf
    area = 0.5 * norm(np.cross(src.V[src.F[:, 1]] - src.V[src.F[:, 0]], src.V[src.F[:, 2]] - src.V[src.F[:, 0]]))
    if split_long and queue:
        iarea = np.array([area[f].sum() for f in queue])
        main = [queue[i] for i in np.nonzero(iarea >= 0.25 * iarea.max())[0]]
        main_len = max(_diameter(_Island(src, cut, f)) for f in main)
        max_len = 1.25 * main_len
        thin_len = 0.45 * main_len     # thin strips pack badly when long and lose nothing when cut
    stats["max_length"] = float(max_len)
    rounds = 0
    done = 0
    while queue and rounds < max_rounds:
        faces = queue.pop()
        if len(faces) == 0:
            continue
        isl = _Island(src, cut, faces)
        if len(faces) <= 2:
            done += 1
            continue
        if isl.is_disk:
            if not split_long or len(faces) < 8:
                done += 1
                continue
            d = _diameter(isl)
            thin = area[faces].sum() < 0.02 * d * d
            if d <= (thin_len if thin else max_len):
                done += 1
                continue
            sep = _bisect(isl)
            if len(sep) == 0:
                done += 1
                continue
            rounds += 1
            cut[isl.src_e[sep]] = True
            stats["long_split"] += 1
            sub = src._face_components(len(src.F), src.E, src.fe, cut | nonman)[faces]
            for lab in np.unique(sub):
                queue.append(faces[sub == lab])
            continue
        rounds += 1
        if isl.genus == 0 and isl.nloops >= 2:
            first = isl.loop_of[isl.E[isl.bnd[0], 0]]
            bverts = np.unique(isl.E[isl.bnd].ravel())
            srcs = bverts[isl.loop_of[bverts] == first].tolist()
            tgts = bverts[isl.loop_of[bverts] != first].tolist()
            _, prev, end = _dijkstra(isl.nv, isl.E, isl.w, srcs, tgts)
            path = _path_edges(isl, prev, end) if end is not None else np.zeros(0, np.int64)
            if len(path):
                cut[isl.src_e[path]] = True
                stats["rings_cut"] += 1
                queue.append(faces)
                continue
        elif isl.nloops == 0 and isl.chi == 2:
            c = isl.pos.mean(0)
            p0 = int(np.argmax(norm(isl.pos - c)))
            p1 = int(np.argmax(norm(isl.pos - isl.pos[p0])))
            _, prev, end = _dijkstra(isl.nv, isl.E, isl.w, [p0], [p1])
            path = _path_edges(isl, prev, end) if end is not None else np.zeros(0, np.int64)
            if len(path):
                cut[isl.src_e[path]] = True
                stats["closed_cut"] += 1
                queue.append(faces)
                continue
        # handles or anything unusual: split in two and deal with the halves
        sep = _bisect(isl)
        if len(sep) == 0:
            done += 1
            continue
        cut[isl.src_e[sep]] = True
        stats["bisected"] += 1
        sub = src._face_components(len(src.F), src.E, src.fe, cut | nonman)[faces]
        for lab in np.unique(sub):
            queue.append(faces[sub == lab])
    stats["islands"] = done + len(queue)
    return cut, stats
