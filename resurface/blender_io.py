"""Read Blender mesh objects into flat numpy arrays and write rebuilt meshes back."""

import bpy
import numpy as np
from mathutils import Matrix


class ObjectData:
    """Raw per-object data (object local data expressed in world space)."""

    def __init__(self, obj):
        self.obj = obj
        me = obj.data
        self.name = obj.name
        mw = np.array(obj.matrix_world, dtype=np.float64)
        self.matrix_world = mw
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        self.co_local = co
        self.co = co @ mw[:3, :3].T + mw[:3, 3]

        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tv = np.empty(nt * 3, np.int64)
        tl = np.empty(nt * 3, np.int64)
        tp = np.empty(nt, np.int64)
        me.loop_triangles.foreach_get("vertices", tv)
        me.loop_triangles.foreach_get("loops", tl)
        me.loop_triangles.foreach_get("polygon_index", tp)
        self.tri_verts = tv.reshape(-1, 3)
        self.tri_loops = tl.reshape(-1, 3)
        self.tri_poly = tp
        mat = np.zeros(len(me.polygons), np.int64)
        if len(me.polygons):
            me.polygons.foreach_get("material_index", mat)
        self.tri_mat = mat[tp] if nt else np.zeros(0, np.int64)
        nl = len(me.loops)
        self.n_loops = nl
        self.n_verts = nv

        # corner-domain float attributes: UV maps and colors (point colors mapped to corners)
        loop_vert = np.empty(nl, np.int64)
        me.loops.foreach_get("vertex_index", loop_vert)
        self.loop_vert = loop_vert
        self.uv = {}
        for layer in me.uv_layers:
            a = np.empty(nl * 2, np.float32)
            layer.data.foreach_get("uv", a)
            self.uv[layer.name] = a.reshape(-1, 2).astype(np.float64)
        self.uv_active = me.uv_layers.active.name if me.uv_layers.active else None
        self.uv_render = next((l.name for l in me.uv_layers if l.active_render), None)
        self.colors = {}
        for ca in me.color_attributes:
            n = len(ca.data)
            a = np.empty(n * 4, np.float32)
            ca.data.foreach_get("color", a)
            a = a.reshape(-1, 4).astype(np.float64)
            self.colors[ca.name] = (ca.domain, ca.data_type, a)
        ac = me.color_attributes.active_color
        self.color_active = ac.name if ac else None
        rc = me.color_attributes.default_color_name if hasattr(me.color_attributes, "default_color_name") else None
        self.color_render = rc

        # corner normals (includes custom split normals)
        cn = np.empty(nl * 3, np.float32)
        me.corner_normals.foreach_get("vector", cn)
        self.corner_normals = cn.reshape(-1, 3).astype(np.float64) @ np.linalg.inv(mw[:3, :3])
        self.corner_normals /= np.maximum(np.linalg.norm(self.corner_normals, axis=1), 1e-30)[:, None]
        self.has_custom_normals = me.has_custom_normals

        # weights: dense (nv, ngroups) keyed by group name
        self.group_names = [g.name for g in obj.vertex_groups]
        self.group_locks = [g.lock_weight for g in obj.vertex_groups]
        ng = len(self.group_names)
        W = np.zeros((nv, max(ng, 1)), np.float64)
        if ng:
            for v in me.vertices:
                for g in v.groups:
                    if g.group < ng:
                        W[v.index, g.group] = g.weight
        self.weights = W[:, :ng]

        # shape keys: basis positions and per-key deltas
        self.shape_keys = []
        if me.shape_keys:
            kb = me.shape_keys.key_blocks
            ref = me.shape_keys.reference_key
            base = np.empty(nv * 3, np.float64)
            ref.data.foreach_get("co", base)
            base = base.reshape(-1, 3)
            for key in kb:
                a = np.empty(nv * 3, np.float64)
                key.data.foreach_get("co", a)
                a = a.reshape(-1, 3)
                self.shape_keys.append({
                    "name": key.name,
                    "delta": a - base,
                    "value": key.value,
                    "slider_min": key.slider_min,
                    "slider_max": key.slider_max,
                    "mute": key.mute,
                    "relative_key": key.relative_key.name,
                    "vertex_group": key.vertex_group,
                    "interpolation": key.interpolation,
                    "is_reference": key == ref,
                })
            self.use_relative = me.shape_keys.use_relative

        # generic point / face attributes that are not handled above
        skip = set(self.uv) | set(self.colors) | {"position", "sharp_face", "sharp_edge", "material_index", "custom_normal"}
        self.point_attrs = {}
        self.face_attrs = {}
        self.corner_attrs = {}
        for at in me.attributes:
            if at.name.startswith(".") or at.name in skip or at.is_internal:
                continue
            if at.data_type not in {"FLOAT", "FLOAT2", "FLOAT_VECTOR", "INT", "BOOLEAN", "INT8", "FLOAT_COLOR", "BYTE_COLOR"}:
                continue
            field = {"FLOAT": "value", "INT": "value", "BOOLEAN": "value", "INT8": "value",
                     "FLOAT2": "vector", "FLOAT_VECTOR": "vector",
                     "FLOAT_COLOR": "color", "BYTE_COLOR": "color"}[at.data_type]
            width = {"value": 1, "vector": 3 if at.data_type == "FLOAT_VECTOR" else 2, "color": 4}[field]
            n = len(at.data)
            if at.data_type == "BOOLEAN":
                buf = np.empty(n * width, bool)
            elif at.data_type in {"INT", "INT8"}:
                buf = np.empty(n * width, np.int32)
            else:
                buf = np.empty(n * width, np.float32)
            try:
                at.data.foreach_get(field, buf)
            except Exception:
                continue
            arr = buf.reshape(n, width).astype(np.float64)
            rec = (at.data_type, field, arr)
            if at.domain == "POINT":
                self.point_attrs[at.name] = rec
            elif at.domain == "FACE":
                self.face_attrs[at.name] = rec
            elif at.domain == "CORNER":
                self.corner_attrs[at.name] = rec

        self.materials = [m for m in me.materials]
        self.smooth = np.zeros(len(me.polygons), bool)
        if len(me.polygons):
            me.polygons.foreach_get("use_smooth", self.smooth)


def gather(objects):
    """Collect objects into global arrays for the source mesh.

    Returns dict with V_raw, T_raw, T_obj, T_mat, T_loops (global loop ids),
    seam attribute list (per raw triangle corner), and the per-object data.
    """
    datas = [ObjectData(o) for o in objects]
    v_off = 0
    l_off = 0
    t_off = 0
    Vs, Ts, To, Tm, Tl = [], [], [], [], []
    for i, d in enumerate(datas):
        d.v_offset = v_off
        d.l_offset = l_off
        d.t_offset = t_off
        Vs.append(d.co)
        Ts.append(d.tri_verts + v_off)
        To.append(np.full(len(d.tri_verts), i, np.int64))
        Tm.append(d.tri_mat)
        Tl.append(d.tri_loops + l_off)
        v_off += d.n_verts
        l_off += d.n_loops
        t_off += len(d.tri_verts)
    out = {
        "datas": datas,
        "V_raw": np.concatenate(Vs) if Vs else np.zeros((0, 3)),
        "T_raw": np.concatenate(Ts) if Ts else np.zeros((0, 3), np.int64),
        "T_obj": np.concatenate(To) if To else np.zeros(0, np.int64),
        "T_mat": np.concatenate(Tm) if Tm else np.zeros(0, np.int64),
        "T_loops": np.concatenate(Tl) if Tl else np.zeros((0, 3), np.int64),
        "n_loops": l_off,
        "n_verts": v_off,
    }
    return out


def corner_layer(datas, n_loops, getter, width):
    """Concatenate a per-loop array over objects; objects without the layer get zeros."""
    out = np.zeros((n_loops, width))
    for d in datas:
        a = getter(d)
        if a is not None:
            out[d.l_offset:d.l_offset + d.n_loops] = a
    return out


def seam_attributes(g, use_uv=True, use_color=True):
    """Per raw-triangle-corner attribute arrays whose discontinuities define seams."""
    datas = g["datas"]
    T_loops = g["T_loops"]
    n_loops = g["n_loops"]
    res = []
    if use_uv:
        names = []
        for d in datas:
            for n in d.uv:
                if n not in names:
                    names.append(n)
        for n in names:
            lay = corner_layer(datas, n_loops, lambda d: d.uv.get(n), 2)
            if np.ptp(lay, axis=0).max() > 0:
                res.append(lay[T_loops])
    if use_color:
        names = []
        for d in datas:
            for n, (dom, dt, a) in d.colors.items():
                if dom == "CORNER" and n not in names:
                    names.append(n)
        for n in names:
            lay = corner_layer(datas, n_loops, lambda d: d.colors[n][2] if n in d.colors and d.colors[n][0] == "CORNER" else None, 4)
            if np.ptp(lay, axis=0).max() > 0:
                res.append(lay[T_loops])
    return res


# ==========================================================================================
# writing
# ==========================================================================================
class ObjectResult:
    """Rebuilt geometry and interpolated attributes for one original object."""

    def __init__(self, data):
        self.data = data
        self.positions = None      # (nv, 3) local space
        self.faces = None          # (nf, 3)
        self.material_index = None
        self.uv = {}               # name -> (nf*3, 2)
        self.colors = {}           # name -> (domain, data_type, array)
        self.normals = None        # (nf*3, 3) local space
        self.weights = None        # (nv, ngroups)
        self.shape_keys = []       # list of (spec dict, (nv, 3) local positions)
        self.point_attrs = {}
        self.face_attrs = {}
        self.corner_attrs = {}


def _vertex_average(values, lF, nv):
    """Average per-corner values (nf, 3, d) onto vertices."""
    d = values.shape[-1]
    acc = np.zeros((nv, d))
    np.add.at(acc, lF.ravel(), values.reshape(-1, d))
    cnt = np.bincount(lF.ravel(), minlength=nv).astype(np.float64)
    return acc / np.maximum(cnt, 1)[:, None]


def _face_blocks(g, src, rm, cmap):
    """Per object: the rebuilt faces it owns, plus regenerated twins (duplicates of
    faces that lived in this object, e.g. backface copies)."""
    T_loops, T_raw, T_mat = g["T_loops"], g["T_raw"], g["T_mat"]
    patch_obj = np.zeros(max(src.npatch, 1), np.int64)
    patch_obj[src.fpatch] = src.fobj
    fobj = patch_obj[rm.fpatch]
    blocks = {oi: [] for oi in range(len(g["datas"]))}
    for oi in blocks:
        sel = np.nonzero(fobj == oi)[0]
        if len(sel):
            blocks[oi].append({
                "fv": rm.F[sel], "loops": cmap.loops[sel], "verts": cmap.verts[sel], "w": cmap.w[sel],
                "mat": src.fmat[cmap.face[sel, 0]], "tri": src.ftri[cmap.face[sel, 0]], "tag": 0,
            })
    if len(src.twin_face):
        for tag, o2 in enumerate(np.unique(src.twin_obj).tolist(), start=1):
            ids = np.nonzero(src.twin_obj == o2)[0]
            tmap = np.full(len(src.F), -1, np.int64)
            tmap[src.twin_face[ids]] = ids
            t = tmap[cmap.face]
            sel = np.nonzero((t >= 0).all(1))[0]
            if len(sel) == 0:
                continue
            tt = t[sel]
            raw = src.twin_raw[tt]
            cm = src.twin_cm[tt]
            loops = T_loops[raw[..., None], cm]
            verts = T_raw[raw[..., None], cm]
            w = cmap.w[sel].copy()
            fv = rm.F[sel].copy()
            flip = ~src.twin_same[tt[:, 0]]
            perm = [0, 2, 1]
            for arr in (fv, loops, verts, w):
                arr[flip] = arr[flip][:, perm]
            blocks[o2].append({
                "fv": fv, "loops": loops, "verts": verts, "w": w,
                "mat": T_mat[raw[:, 0]], "tri": raw[:, 0], "tag": tag,
            })
    return blocks


def build_results(g, src, rm, cmap, max_influences=None):
    """Assemble per-object results (numpy only, no Blender data is touched)."""
    datas = g["datas"]
    blocks = _face_blocks(g, src, rm, cmap)
    nvt = len(rm.V)
    out = []
    for oi, d in enumerate(datas):
        bl = blocks[oi]
        if not bl:
            out.append(None)
            continue
        r = ObjectResult(d)
        key = np.concatenate([b["fv"] + b["tag"] * nvt for b in bl])
        used, inv = np.unique(key, return_inverse=True)
        lF = inv.reshape(-1, 3)
        nv = len(used)
        mw = d.matrix_world
        M3 = mw[:3, :3]
        r.positions = (rm.V[used % nvt] - mw[:3, 3]) @ np.linalg.inv(M3).T
        r.faces = lF
        L = np.concatenate([b["loops"] for b in bl]) - d.l_offset
        Vv = np.concatenate([b["verts"] for b in bl]) - d.v_offset
        W = np.concatenate([b["w"] for b in bl])
        tri_raw = np.concatenate([b["tri"] for b in bl])
        r.material_index = np.concatenate([b["mat"] for b in bl])

        def corner(arr):
            return np.einsum("fcj,fcjd->fcd", W, arr[L])

        def point(arr):
            return np.einsum("fcj,fcjd->fcd", W, arr[Vv])

        for name, arr in d.uv.items():
            r.uv[name] = corner(arr).reshape(-1, 2)
        for name, (dom, dt, arr) in d.colors.items():
            if dom == "CORNER":
                r.colors[name] = (dom, dt, corner(arr).reshape(-1, 4))
            else:
                r.colors[name] = (dom, dt, _vertex_average(point(arr), lF, nv))
        n = corner(d.corner_normals).reshape(-1, 3) @ M3
        r.normals = n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]

        # weights: interpolate, keep the strongest influences, renormalise like the source
        if d.weights.shape[1]:
            w = _vertex_average(point(d.weights), lF, nv)
            src_cnt = (d.weights > 0).sum(1)
            kmax = int(src_cnt.max()) if len(src_cnt) else 4
            if max_influences:
                kmax = min(kmax, int(max_influences))
            kmax = max(kmax, 1)
            if w.shape[1] > kmax:
                cut = np.partition(w, -kmax, axis=1)[:, -kmax][:, None]
                w = np.where(w >= cut, w, 0.0)
            w[w < 1e-4] = 0.0
            ssum = d.weights.sum(1)
            weighted = ssum > 0
            normalised = bool(weighted.any()) and np.abs(ssum[weighted] - 1.0).max() < 1e-2
            if normalised:
                t = w.sum(1)
                w = np.where(t[:, None] > 0, w / np.maximum(t, 1e-30)[:, None], w)
            r.weights = w

        for sk in d.shape_keys:
            delta = _vertex_average(point(sk["delta"]), lF, nv)
            r.shape_keys.append((sk, r.positions + delta))

        jmax = np.argmax(W, axis=-1)
        for name, (dt, field, arr) in d.point_attrs.items():
            if dt in {"INT", "BOOLEAN", "INT8"}:
                idx = np.take_along_axis(Vv, jmax[..., None], axis=-1)[..., 0]
                v = np.zeros((nv, arr.shape[1]))
                v[lF.ravel()] = arr[idx].reshape(-1, arr.shape[1])
            else:
                v = _vertex_average(point(arr), lF, nv)
            r.point_attrs[name] = (dt, field, v)
        for name, (dt, field, arr) in d.corner_attrs.items():
            if dt in {"INT", "BOOLEAN", "INT8"}:
                idx = np.take_along_axis(L, jmax[..., None], axis=-1)[..., 0]
                r.corner_attrs[name] = (dt, field, arr[idx].reshape(-1, arr.shape[1]))
            else:
                r.corner_attrs[name] = (dt, field, corner(arr).reshape(-1, arr.shape[1]))
        if d.face_attrs:
            poly = d.tri_poly[tri_raw - d.t_offset]
            for name, (dt, field, arr) in d.face_attrs.items():
                r.face_attrs[name] = (dt, field, arr[poly])
        out.append(r)
    return out


def _set_attr(me, name, domain, data_type, field, values):
    at = me.attributes.get(name)
    if at is None:
        at = me.attributes.new(name, data_type, domain)
    if data_type == "BOOLEAN":
        buf = values.astype(bool).ravel()
    elif data_type in {"INT", "INT8"}:
        buf = np.round(values).astype(np.int32).ravel()
    else:
        buf = values.astype(np.float32).ravel()
    at.data.foreach_set(field, buf)


def create_mesh(r, name):
    """Create a new Blender mesh datablock from an ObjectResult (no weights / keys)."""
    d = r.data
    me = bpy.data.meshes.new(name)
    nv = len(r.positions)
    nf = len(r.faces)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", r.positions.astype(np.float32).ravel())
    me.loops.add(nf * 3)
    me.loops.foreach_set("vertex_index", r.faces.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    for m in d.materials:
        me.materials.append(m)
    if len(d.materials):
        me.polygons.foreach_set("material_index", np.clip(r.material_index, 0, len(d.materials) - 1).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(nf, bool))
    for uname, arr in r.uv.items():
        lay = me.uv_layers.new(name=uname)
        lay.uv.foreach_set("vector", arr.astype(np.float32).ravel())
    if d.uv_active and d.uv_active in me.uv_layers:
        me.uv_layers.active = me.uv_layers[d.uv_active]
    if d.uv_render and d.uv_render in me.uv_layers:
        me.uv_layers[d.uv_render].active_render = True
    for cname, (dom, dt, arr) in r.colors.items():
        ca = me.color_attributes.new(name=cname, type=dt, domain=dom)
        ca.data.foreach_set("color", arr.astype(np.float32).ravel())
    if d.color_active and d.color_active in me.color_attributes:
        me.color_attributes.active_color = me.color_attributes[d.color_active]
    if d.color_render and d.color_render in me.color_attributes:
        me.color_attributes.render_color_index = me.color_attributes.find(d.color_render)
    for aname, (dt, field, arr) in r.point_attrs.items():
        _set_attr(me, aname, "POINT", dt, field, arr)
    for aname, (dt, field, arr) in r.corner_attrs.items():
        _set_attr(me, aname, "CORNER", dt, field, arr)
    for aname, (dt, field, arr) in r.face_attrs.items():
        _set_attr(me, aname, "FACE", dt, field, arr)
    return me


def _set_normals(me, normals):
    if normals is not None and len(me.loops) == len(normals):
        me.normals_split_custom_set(normals.astype(np.float32))


# custom property linking a rebuilt object to its kept original mesh; the second key is
# from when the add-on was called Mesh Rebuild
ORIGINAL_KEYS = ("resurface_original", "mesh_rebuild_original")


def original_name(obj):
    for key in ORIGINAL_KEYS:
        if obj.get(key):
            return obj[key]
    return None


def forget_original(obj):
    for key in ORIGINAL_KEYS:
        if key in obj:
            del obj[key]


def apply_result(obj, r, keep_original=True):
    """Swap the object's mesh for the rebuilt one; restore weights and shape keys."""
    import bmesh
    old = obj.data
    base_name = old.name
    prev = original_name(obj)
    has_prev = bool(prev) and bpy.data.meshes.get(prev) is not None
    new = create_mesh(r, base_name + " (rebuilt)")
    if keep_original and not has_prev:
        # keep the true original; rebuilding a rebuilt object discards the intermediate
        old.use_fake_user = True
        old.name = base_name + " (original)"
        obj[ORIGINAL_KEYS[0]] = old.name
    obj.data = new
    if old.users == 0 and not old.use_fake_user:
        bpy.data.meshes.remove(old)
    new.name = base_name

    # vertex group weights (group indices of the object are unchanged)
    if r.weights is not None and r.weights.shape[1]:
        names = r.data.group_names
        for gname in names:
            if obj.vertex_groups.get(gname) is None:
                obj.vertex_groups.new(name=gname)
        gidx = [obj.vertex_groups[gname].index for gname in names]
        bm = bmesh.new()
        bm.from_mesh(new)
        dl = bm.verts.layers.deform.verify()
        bm.verts.ensure_lookup_table()
        W = r.weights
        nz_v, nz_g = np.nonzero(W > 0)
        verts = bm.verts
        for v, gcol, w in zip(nz_v.tolist(), nz_g.tolist(), W[nz_v, nz_g].tolist()):
            verts[v][dl][gidx[gcol]] = w
        bm.to_mesh(new)
        bm.free()
    _set_normals(new, r.normals)

    # shape keys
    if r.shape_keys:
        created = {}
        for spec, co in r.shape_keys:
            kb = obj.shape_key_add(name=spec["name"], from_mix=False)
            kb.data.foreach_set("co", co.astype(np.float32).ravel())
            created[spec["name"]] = (kb, spec)
        for kb, spec in created.values():
            kb.value = spec["value"]
            kb.slider_min = spec["slider_min"]
            kb.slider_max = spec["slider_max"]
            kb.mute = spec["mute"]
            kb.vertex_group = spec["vertex_group"]
            kb.interpolation = spec["interpolation"]
            rel = created.get(spec["relative_key"])
            if rel is not None:
                kb.relative_key = rel[0]
        if new.shape_keys is not None:
            new.shape_keys.use_relative = getattr(r.data, "use_relative", True)
    new.update()
    return new


# ==========================================================================================
# smooth normals
# ==========================================================================================
def loopmesh_from_objects(objects):
    """Combine mesh objects into one world-space LoopMesh.  Returns (mesh, info) with
    info = [(obj, vertex offset, loop offset, loop count, matrix_world)]."""
    from .core.normals import LoopMesh
    Vs, LV, LS, LT, info = [], [], [], [], []
    v_off = l_off = 0
    for o in objects:
        me = o.data
        nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
        mw = np.array(o.matrix_world, dtype=np.float64)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        ls = np.empty(npoly, np.int32)
        lt = np.empty(npoly, np.int32)
        me.polygons.foreach_get("loop_start", ls)
        me.polygons.foreach_get("loop_total", lt)
        Vs.append(co)
        LV.append(lv.astype(np.int64) + v_off)
        LS.append(ls.astype(np.int64) + l_off)
        LT.append(lt.astype(np.int64))
        info.append((o, v_off, l_off, nl, mw))
        v_off += nv
        l_off += nl
    mesh = LoopMesh(np.concatenate(Vs), np.concatenate(LV), np.concatenate(LS), np.concatenate(LT))
    return mesh, info


def set_loop_normals(me, normals_local):
    """Smooth-shade the mesh, clear old hard edges and apply custom split normals."""
    n = normals_local / np.maximum(np.linalg.norm(normals_local, axis=1), 1e-30)[:, None]
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    sharp = me.attributes.get("sharp_edge")
    if sharp is not None:
        me.attributes.remove(sharp)
    me.normals_split_custom_set(n.astype(np.float32))


def smooth_object_normals(objects, angle, blur=0, fix_winding=True, weld_distance=1e-5):
    """Recompute clean smooth normals for the objects (processed together).
    Returns (flipped faces, loops written)."""
    import bmesh
    from .core.normals import flip_islands, smooth_loop_normals
    flipped = 0
    if fix_winding:
        mesh, info = loopmesh_from_objects(objects)
        flip = flip_islands(mesh, weld_distance)
        if flip.any():
            face_off = 0
            for o, v_off, l_off, nl, mw in info:
                npoly = len(o.data.polygons)
                idx = np.nonzero(flip[face_off:face_off + npoly])[0]
                face_off += npoly
                if len(idx) == 0:
                    continue
                bm = bmesh.new()
                bm.from_mesh(o.data)
                bm.faces.ensure_lookup_table()
                bmesh.ops.reverse_faces(bm, faces=[bm.faces[i] for i in idx.tolist()])
                bm.to_mesh(o.data)
                bm.free()
                flipped += len(idx)
    mesh, info = loopmesh_from_objects(objects)
    N = smooth_loop_normals(mesh, weld_distance, angle, blur)
    for o, v_off, l_off, nl, mw in info:
        n = N[l_off:l_off + nl] @ mw[:3, :3]
        set_loop_normals(o.data, n)
    return flipped, len(N)


# ==========================================================================================
# UV rebuild
# ==========================================================================================
OLD_UV_NAME = "Old UVs"   # must not start with "uv": the MDL exporter treats uv* layers as game channels


def export_uv_layer(me):
    """The UV layer that ends up as game channel 0 (uv0, else the first layer)."""
    if not me.uv_layers:
        return None
    return me.uv_layers.get("uv0") or next((l for l in me.uv_layers if l.name.lower().startswith("uv")), me.uv_layers[0])


def backup_uvs(me):
    """Copy the export UVs into OLD_UV_NAME once (later runs keep the first backup)."""
    src = export_uv_layer(me)
    if src is None:
        src = me.uv_layers.new(name="uv0")
    if me.uv_layers.get(OLD_UV_NAME) is None:
        data = np.empty(len(me.loops) * 2, np.float32)
        src.data.foreach_get("uv", data)
        old = me.uv_layers.new(name=OLD_UV_NAME, do_init=False)
        old.data.foreach_set("uv", data)
        old.active_render = False
    target = export_uv_layer(me)
    me.uv_layers.active = target
    target.active_render = True
    return target


def compute_seam_masks(objects, keep_old_seams=True, cut_sharp=False, sharp_angle=60.0, merge_distance=1e-5,
                       split_long=True, use_existing_seams=False):
    """Seam mask per object (over its mesh edges) from the structural island layout.
    With use_existing_seams, edges already marked as seams are kept as cuts."""
    from .core.source import SourceMesh, SourceOptions
    from .core.uvseams import compute_uv_cuts
    g = gather(objects)
    seams = seam_attributes(g, use_uv=True, use_color=False) if keep_old_seams else []
    src = SourceMesh(
        g["V_raw"], g["T_raw"], g["T_obj"], g["T_mat"], seams,
        SourceOptions(merge_distance=merge_distance, fragment_faces=0,
                      sharp_angle=sharp_angle if cut_sharp else None),
    )
    n = len(src.V) + 1
    src_keys = src.E[:, 0].astype(np.int64) * n + src.E[:, 1]
    edge_keys = []
    for d in g["datas"]:
        me = d.obj.data
        ev = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2).astype(np.int64) + d.v_offset
        w = src.raw_to_welded[ev]
        w.sort(axis=1)
        ok = (w >= 0).all(1)
        edge_keys.append((ok, w[:, 0] * n + w[:, 1]))
    extra = None
    if use_existing_seams:
        extra = np.zeros(len(src.E), bool)
        for d, (ok, ek) in zip(g["datas"], edge_keys):
            seam = np.zeros(len(d.obj.data.edges), bool)
            d.obj.data.edges.foreach_get("use_seam", seam)
            k = ek[ok & seam]
            pos = np.searchsorted(src_keys, k)
            hit = (pos < len(src_keys)) & (src_keys[np.minimum(pos, len(src_keys) - 1)] == k)
            extra[pos[hit]] = True
    cut, stats = compute_uv_cuts(src, keep_old_seams=keep_old_seams, cut_sharp=cut_sharp,
                                 split_long=split_long, extra_cut=extra)
    ck = np.sort(src.E[cut], axis=1)
    keys = np.unique(ck[:, 0].astype(np.int64) * n + ck[:, 1])
    masks = [ok & np.isin(ek, keys) for ok, ek in edge_keys]
    return masks, stats


def find_uv_necks(obj, max_ratio=0.08, min_arc=0.15, max_cuts=8):
    """Seams to add where an unwrapped island pinches: two points of its UV outline
    that are close together but far apart along the outline (a narrow bridge, e.g. a
    hem strip that only touches its panel at one spot).  Returns mesh edge indices."""
    import bmesh
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    uvl = bm.loops.layers.uv.get(export_uv_layer(me).name)
    nf = len(bm.faces)
    island = [-1] * nf
    n_isl = 0
    for f0 in bm.faces:
        if island[f0.index] >= 0:
            continue
        stack = [f0]
        island[f0.index] = n_isl
        while stack:
            f = stack.pop()
            for e in f.edges:
                if e.seam or len(e.link_faces) != 2:
                    continue
                for g in e.link_faces:
                    if island[g.index] < 0:
                        island[g.index] = n_isl
                        stack.append(g)
        n_isl += 1
    faces_of = [[] for _ in range(n_isl)]
    for f in bm.faces:
        faces_of[island[f.index]].append(f)
    cuts = []
    for i, fl in enumerate(faces_of):
        if len(fl) < 20 or len(cuts) >= max_cuts:
            continue
        # outline half-edges (vertex -> next vertex, with uv)
        nxt = {}
        for f in fl:
            for l in f.loops:
                e = l.edge
                inner = (not e.seam) and len(e.link_faces) == 2 and all(island[g.index] == i for g in e.link_faces)
                if not inner:
                    nxt.setdefault(l.vert.index, (l.link_loop_next.vert.index, l[uvl].uv.copy()))
        if len(nxt) < 12:
            continue
        # walk the longest outline loop
        seen = set()
        best_loop = []
        for start in list(nxt.keys()):
            if start in seen:
                continue
            loop = []
            v = start
            while v in nxt and v not in seen:
                seen.add(v)
                loop.append((v, nxt[v][1]))
                v = nxt[v][0]
            if len(loop) > len(best_loop):
                best_loop = loop
        if len(best_loop) < 12:
            continue
        pts = np.array([[p[0], p[1]] for _, p in best_loop])
        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        s = np.r_[0.0, np.cumsum(seg)[:-1]]
        P = seg.sum()
        if P <= 0:
            continue
        kd = KDTree(len(pts))
        for k, p in enumerate(pts):
            kd.insert(Vector((p[0], p[1], 0.0)), k)
        kd.balance()
        best = (max_ratio, None, None)
        radius = max_ratio * min_arc * P * 4
        for k, p in enumerate(pts):
            for (co, j, d) in kd.find_range(Vector((p[0], p[1], 0.0)), radius):
                if j <= k:
                    continue
                arc = abs(s[j] - s[k])
                arc = min(arc, P - arc)
                if arc < min_arc * P:
                    continue
                r = d / arc
                if r < best[0]:
                    best = (r, best_loop[k][0], best_loop[j][0])
        if best[1] is None:
            continue
        # shortest path between the two outline points through the island
        va, vb = best[1], best[2]
        fset = set(f.index for f in fl)
        import heapq
        dist = {va: 0.0}
        prev = {}
        heap = [(0.0, va)]
        while heap:
            d, v = heapq.heappop(heap)
            if v == vb:
                break
            if d > dist.get(v, np.inf):
                continue
            for e in bm.verts[v].link_edges:
                if e.seam or not any(g.index in fset for g in e.link_faces):
                    continue
                u = e.other_vert(bm.verts[v]).index
                nd = d + e.calc_length()
                if nd < dist.get(u, np.inf):
                    dist[u] = nd
                    prev[u] = e.index
                    heapq.heappush(heap, (nd, u))
        if vb not in prev:
            continue
        v = vb
        while v != va:
            ei = prev[v]
            cuts.append(ei)
            v = bm.edges[ei].other_vert(bm.verts[v]).index
    bm.free()
    return cuts


def uv_islands_by_seams(me):
    """Face island id per polygon (islands are separated by seams and non-manifold edges)."""
    from .core.geom import connected_components, relabel
    npoly = len(me.polygons)
    nl = len(me.loops)
    le = np.empty(nl, np.int32)
    me.loops.foreach_get("edge_index", le)
    ls = np.empty(npoly, np.int32)
    lt = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lf = np.repeat(np.arange(npoly), lt)
    seam = np.zeros(len(me.edges), bool)
    me.edges.foreach_get("use_seam", seam)
    order = np.argsort(le, kind="stable")
    se = le[order]
    starts = np.r_[0, np.nonzero(np.diff(se))[0] + 1]
    cnt = np.diff(np.r_[starts, len(se)])
    two = starts[(cnt == 2) & ~seam[se[starts]]]
    a = lf[order[two]]
    b = lf[order[two + 1]]
    return relabel(connected_components(npoly, a, b)), lf


def align_islands_to_old(objects):
    """Rotate (and mirror, if the old layout was mirrored) every new island so it has
    the orientation the same faces had in the old UVs.  Tangent frames then barely
    change, so textures and especially normal maps transfer cleanly."""
    for o in objects:
        me = o.data
        new = export_uv_layer(me)
        old = me.uv_layers.get(OLD_UV_NAME)
        if new is None or old is None:
            continue
        nl = len(me.loops)
        U = np.empty(nl * 2, np.float32)
        new.data.foreach_get("uv", U)
        U = U.reshape(-1, 2).astype(np.float64)
        O = np.empty(nl * 2, np.float32)
        old.data.foreach_get("uv", O)
        O = O.reshape(-1, 2).astype(np.float64)
        isl, lf = uv_islands_by_seams(me)
        li = isl[lf]
        order = np.argsort(li, kind="stable")
        starts = np.r_[0, np.nonzero(np.diff(li[order]))[0] + 1]
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts.tolist(), ends.tolist()):
            idx = order[s:e]
            X = U[idx]
            Y = O[idx]
            cx = X.mean(0)
            cy = Y.mean(0)
            M = (X - cx).T @ (Y - cy)
            if not np.isfinite(M).all() or np.abs(M).sum() < 1e-20:
                continue
            u, sv, vt = np.linalg.svd(M)
            R = u @ vt
            U[idx] = (X - cx) @ R + cx
        new.data.foreach_set("uv", U.astype(np.float32).ravel())


def _uv_triangles(objects, layer_name):
    tris = []
    for o in objects:
        me = o.data
        layer = me.uv_layers.get(layer_name) if layer_name else export_uv_layer(me)
        if layer is None:
            raise ValueError(f"'{o.name}' has no UV layer '{layer_name}'")
        me.calc_loop_triangles()
        tl = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tl)
        uv = np.empty(len(me.loops) * 2, np.float32)
        layer.data.foreach_get("uv", uv)
        tris.append(uv.reshape(-1, 2)[tl.reshape(-1, 3)].astype(np.float64))
    return np.concatenate(tris) if tris else np.zeros((0, 3, 2))


def transfer_texture(objects, image, mode="COLOR", flip_green=False, padding=8, save=True):
    """Resample `image` from the old UV layout ("Old UVs") to the current one.
    Returns (new image, file path or None)."""
    import os
    import bpy
    from .core.texremap import remap
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError(f"image '{image.name}' has no pixel data (is the file missing?)")
    C = image.channels
    buf = np.empty(w * h * C, np.float32)
    image.pixels.foreach_get(buf)
    img = buf.reshape(h, w, C)
    new_uv = _uv_triangles(objects, None)
    old_uv = _uv_triangles(objects, OLD_UV_NAME)
    out, filled = remap(img, new_uv, old_uv, w, h, mode=mode, flip_green=flip_green, padding=padding)
    if C == 4 and not filled.all():
        # outside the islands: keep alpha opaque for colour, neutral normal for normal maps
        empty = ~filled
        if mode == "NORMAL":
            out[empty] = np.array([0.5, 0.5, 1.0, 1.0], np.float32)[:C]
        else:
            out[empty, 3] = 1.0
    base = os.path.splitext(image.name)[0]
    new = bpy.data.images.new(base + "_newuv", w, h, alpha=(C == 4), float_buffer=image.is_float)
    try:
        new.colorspace_settings.name = image.colorspace_settings.name
    except (TypeError, AttributeError):
        pass
    new.alpha_mode = image.alpha_mode
    new.pixels.foreach_set(out.ravel())
    path = None
    src_path = bpy.path.abspath(image.filepath) if image.filepath else ""
    if save and src_path and os.path.isdir(os.path.dirname(src_path)):
        path = os.path.join(os.path.dirname(src_path), base + "_newuv.png")
        new.filepath_raw = path
        new.file_format = "PNG"
        new.save()
    else:
        new.pack()
    return new, path


def uv_stretch_stats(objects):
    """Spread of texel density over the faces (1.0 = perfectly even)."""
    ratios, areas = [], []
    for o in objects:
        me = o.data
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tv = np.empty(nt * 3, np.int32)
        tl = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tv)
        me.loop_triangles.foreach_get("loops", tl)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        uv = np.empty(len(me.loops) * 2, np.float32)
        export_uv_layer(me).data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2).astype(np.float64)
        P = co[tv.reshape(-1, 3)]
        U = uv[tl.reshape(-1, 3)]
        a3 = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        d1 = U[:, 1] - U[:, 0]
        d2 = U[:, 2] - U[:, 0]
        a2 = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        ok = a3 > 1e-14
        ratios.append(np.sqrt(a2[ok] / a3[ok]))
        areas.append(a3[ok])
    r = np.concatenate(ratios)
    a = np.concatenate(areas)
    med = np.median(r)
    rel = r / max(med, 1e-30)
    even = float(a[(rel > 0.8) & (rel < 1.25)].sum() / max(a.sum(), 1e-30))
    return {"within_25pct": even, "p5": float(np.percentile(rel, 5)), "p95": float(np.percentile(rel, 95))}


def rebuild_uvs(context, objects, keep_old_seams=True, cut_sharp=False, sharp_angle=60.0,
                method="MINIMUM_STRETCH", margin=0.004, keep_orientation=True, merge_distance=1e-5):
    """Replace the export UVs with a fresh, non-overlapping layout (old UVs are kept)."""
    import bpy
    masks, stats = compute_seam_masks(objects, keep_old_seams, cut_sharp, sharp_angle, merge_distance,
                                      split_long=False)
    for o, mask in zip(objects, masks):
        backup_uvs(o.data)
        o.data.edges.foreach_set("use_seam", mask)
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    prev_sel = [o for o in context.selected_objects]
    for o in view_layer.objects:
        o.select_set(o in objects)
    view_layer.objects.active = objects[0]

    def unwrap_all():
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.mesh.select_all(action="SELECT")
            try:
                bpy.ops.uv.unwrap(method=method, fill_holes=True, correct_aspect=True, margin=0.0)
            except TypeError:
                bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True, correct_aspect=True, margin=0.0)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    unwrap_all()
    # cut narrow bridges that the first unwrap reveals, then unwrap again
    stats["neck_cuts"] = 0
    for _ in range(4):
        added = 0
        for o in objects:
            cuts = find_uv_necks(o)
            if cuts:
                seam = np.zeros(len(o.data.edges), bool)
                o.data.edges.foreach_get("use_seam", seam)
                seam[np.array(cuts, np.int64)] = True
                o.data.edges.foreach_set("use_seam", seam)
                added += 1
        if not added:
            break
        stats["neck_cuts"] += added
        unwrap_all()

    # with the bridges cut, split strips that are much longer than any main panel
    masks, stats2 = compute_seam_masks(objects, keep_old_seams, cut_sharp, sharp_angle, merge_distance,
                                       split_long=True, use_existing_seams=True)
    stats["long_split"] = stats2["long_split"]
    stats["islands"] = stats2["islands"]
    if stats2["long_split"]:
        for o, mask in zip(objects, masks):
            o.data.edges.foreach_set("use_seam", mask)
        unwrap_all()
    if keep_orientation:
        align_islands_to_old(objects)

    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.average_islands_scale()
        rotate = not keep_orientation
        if rotate:
            # square each island up to its tightest bounding box, then pack in 90 degree steps;
            # free rotation tends to leave long cloth panels lying diagonally
            try:
                bpy.ops.uv.align_rotation(method="AUTO")
            except (AttributeError, TypeError, RuntimeError):
                pass
        try:
            bpy.ops.uv.pack_islands(shape_method="CONCAVE", margin_method="FRACTION", margin=margin,
                                    rotate=rotate, rotate_method="AXIS_ALIGNED")
        except TypeError:
            bpy.ops.uv.pack_islands(margin=margin, rotate=rotate)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        for o in view_layer.objects:
            o.select_set(o in prev_sel)
        view_layer.objects.active = prev_active
    stats.update(uv_stretch_stats(objects))
    return stats


def restore_original(obj):
    """Swap back to the mesh that was kept by apply_result."""
    name = original_name(obj)
    if not name:
        return False
    orig = bpy.data.meshes.get(name)
    if orig is None:
        return False
    cur = obj.data
    base_name = cur.name
    cur.name = base_name + " (rebuilt)"
    obj.data = orig
    orig.use_fake_user = False
    orig.name = base_name
    forget_original(obj)
    if cur.users == 0:
        bpy.data.meshes.remove(cur)
    return True
