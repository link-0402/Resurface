"""UV tools of the UVs tab: Resize Canvas and Index Map Generator (Blender side)."""

import math
import os
import traceback

import bmesh
import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty
from bpy.types import Operator

from .core import canvas as canvas_math
from .core import indexmap as imap
from .operators import show_report

ROW_ATTR = "colorset_row"     # face attribute for manual index maps: 0 = no row, 1 = 1A, 2 = 1B, ... 32 = 16B
LEGACY_PREFIX = "UV_Group"     # vertex groups made by the old Index Map Generator add-on
# tags on images made by the Index Map Generator; the second one is from when
# the add-on was called Resurface
IMAGE_TAGS = ("resurface_index_map", "mesh_rebuild_index_map")


def mesh_targets(context):
    """Meshes the UV tools work on: the ones in Edit Mode, else the selected ones."""
    if context.mode == "EDIT_MESH":
        return [o for o in context.objects_in_mode if o.type == "MESH"]
    return [o for o in context.selected_objects if o.type == "MESH" and len(o.data.polygons)]


def uv_targets(context):
    return [o for o in mesh_targets(context) if o.data.uv_layers.active is not None]


def row_value(name):
    return imap.ROW_NAMES.index(name) + 1


def value_name(v):
    return imap.ROW_NAMES[v - 1] if v > 0 else "No row"


def count(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------------------------------
# reading / writing
# ------------------------------------------------------------------------------------------
class readable_mesh:
    """Mesh data that numpy can read: the object's mesh, or for an object in Edit Mode a
    temporary copy of the edit mesh (Blender blocks reading UV and attribute arrays of a
    mesh in Edit Mode)."""

    def __init__(self, obj):
        self.obj = obj
        self.tmp = None

    def __enter__(self):
        if self.obj.mode != "EDIT":
            return self.obj.data
        self.tmp = bpy.data.meshes.new("_resurface_tmp")
        bmesh.from_edit_mesh(self.obj.data).to_mesh(self.tmp)
        return self.tmp

    def __exit__(self, *exc):
        if self.tmp is not None:
            bpy.data.meshes.remove(self.tmp)
            self.tmp = None


def uv_triangles(objs):
    """World positions, triangles, active-UV corners and the owner (object, polygon) of
    every triangle over all objects, plus the row value of every polygon per object."""
    Ps, Ts, UVs, tobj, tpoly, names, rows = [], [], [], [], [], [], []
    v_off = 0
    for i, o in enumerate(objs):
        active = o.data.uv_layers.active
        with readable_mesh(o) as me:
            rows.append(read_rows(me))
            layer = me.uv_layers.get(active.name) if active is not None else None
            if layer is None:
                continue
            if active.name not in names:
                names.append(active.name)
            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            tv = np.empty(nt * 3, np.int32)
            tl = np.empty(nt * 3, np.int32)
            tp = np.empty(nt, np.int32)
            me.loop_triangles.foreach_get("vertices", tv)
            me.loop_triangles.foreach_get("loops", tl)
            me.loop_triangles.foreach_get("polygon_index", tp)
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            uv = np.empty(len(me.loops) * 2, np.float32)
            layer.data.foreach_get("uv", uv)
        mw = np.array(o.matrix_world, np.float64)
        co = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
        Ps.append(co)
        Ts.append(tv.reshape(-1, 3).astype(np.int64) + v_off)
        UVs.append(uv.reshape(-1, 2).astype(np.float64)[tl.reshape(-1, 3)])
        tobj.append(np.full(nt, i, np.int64))
        tpoly.append(tp.astype(np.int64))
        v_off += len(co)
    if not Ps:
        z = np.zeros(0, np.int64)
        return {"P": np.zeros((0, 3)), "T": np.zeros((0, 3), np.int64), "UV": np.zeros((0, 3, 2)),
                "obj": z, "poly": z, "uv_names": names, "rows": rows}
    return {"P": np.concatenate(Ps), "T": np.concatenate(Ts), "UV": np.concatenate(UVs),
            "obj": np.concatenate(tobj), "poly": np.concatenate(tpoly), "uv_names": names, "rows": rows}


def object_rows(o):
    with readable_mesh(o) as me:
        return read_rows(me)


def read_rows(me):
    """Row value per polygon from the mesh data (0 where there is none)."""
    n = len(me.polygons)
    at = me.attributes.get(ROW_ATTR)
    if at is None or at.domain != "FACE" or at.data_type != "INT":
        return np.zeros(n, np.int64)
    a = np.empty(n, np.int32)
    at.data.foreach_get("value", a)
    return a.astype(np.int64)


def write_rows(o, values):
    """Write the row value of every polygon, in Object or Edit Mode."""
    me = o.data
    values = np.asarray(values, np.int64)
    if o.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        layer = bm.faces.layers.int.get(ROW_ATTR)
        if layer is None:
            layer = bm.faces.layers.int.new(ROW_ATTR)
        for f, v in zip(bm.faces, values.tolist()):
            f[layer] = v
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    else:
        at = me.attributes.get(ROW_ATTR)
        if at is not None and (at.domain != "FACE" or at.data_type != "INT"):
            me.attributes.remove(at)
            at = None
        if at is None:
            at = me.attributes.new(ROW_ATTR, "INT", "FACE")
        at.data.foreach_set("value", values.astype(np.int32))
        me.update()
    _row_cache.pop(o.name, None)


# counts per row for the panel; Edit Mode meshes are cached (reading a BMesh is slow)
_row_cache = {}


def row_counts(objs):
    """{row value: face count} over the objects (value 0 = faces without a row)."""
    total = {}
    for o in objs:
        me = o.data
        if o.mode == "EDIT":
            bm = bmesh.from_edit_mesh(me)
            key = len(bm.faces)
            hit = _row_cache.get(o.name)
            if hit is not None and hit[0] == key:
                counts = hit[1]
            else:
                layer = bm.faces.layers.int.get(ROW_ATTR)
                if layer is None:
                    counts = {0: key}
                else:
                    vals = np.fromiter((f[layer] for f in bm.faces), np.int64, key)
                    u, c = np.unique(vals, return_counts=True)
                    counts = dict(zip(u.tolist(), c.tolist()))
                _row_cache[o.name] = (key, counts)
        else:
            u, c = np.unique(read_rows(me), return_counts=True)
            counts = dict(zip(u.tolist(), c.tolist()))
        for v, c in counts.items():
            total[v] = total.get(v, 0) + c
    return total


def legacy_groups(o):
    return [vg for vg in o.vertex_groups
            if vg.name.startswith(LEGACY_PREFIX) and vg.name[len(LEGACY_PREFIX):] in imap.ROW_NAMES]


@persistent
def _forget_counts(*_args):
    _row_cache.clear()


# ------------------------------------------------------------------------------------------
# Resize Canvas
# ------------------------------------------------------------------------------------------
def canvas_setup(st):
    old = (st.canvas_old_w, st.canvas_old_h)
    new = (st.canvas_new_w, st.canvas_new_h)
    off = canvas_math.canvas_offset(old, new, st.canvas_anchor, (st.canvas_offset_x, st.canvas_offset_y))
    return old, new, off


def _uv_poll(cls, context):
    objs = mesh_targets(context)
    if not objs:
        cls.poll_message_set("Select one or more mesh objects")
        return False
    if not any(o.data.uv_layers.active for o in objs):
        cls.poll_message_set("The meshes have no UV map")
        return False
    return True


class RESURFACE_OT_resize_canvas(Operator):
    bl_idname = "resurface.resize_canvas"
    bl_label = "Resize UV Canvas"
    bl_description = (
        "Move the active UV map so it keeps showing the same pixels after the texture's canvas was cropped "
        "or extended in an image editor (e.g. 2048x4096 to 2048x2048)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _uv_poll(cls, context)

    def execute(self, context):
        st = context.scene.resurface
        old, new, off = canvas_setup(st)
        (su, sv), (tu, tv) = canvas_math.canvas_transform(old, new, off)
        loops = 0
        names = []
        objs = [o for o in mesh_targets(context) if o.data.uv_layers.active]
        for o in objs:
            me = o.data
            layer = me.uv_layers.active
            if layer.name not in names:
                names.append(layer.name)
            if o.mode == "EDIT":
                bm = bmesh.from_edit_mesh(me)
                uvl = bm.loops.layers.uv.get(layer.name)
                for f in bm.faces:
                    if st.canvas_selected_only and not f.select:
                        continue
                    for lp in f.loops:
                        uv = lp[uvl].uv
                        uv.x = uv.x * su + tu
                        uv.y = uv.y * sv + tv
                        loops += 1
                bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
            else:
                a = np.empty(len(me.loops) * 2, np.float32)
                layer.data.foreach_get("uv", a)
                a = a.reshape(-1, 2).astype(np.float64)
                a[:, 0] = a[:, 0] * su + tu
                a[:, 1] = a[:, 1] * sv + tv
                layer.data.foreach_set("uv", a.astype(np.float32).ravel())
                me.update()
                loops += len(me.loops)
        if loops == 0:
            self.report({"WARNING"}, "No faces selected, nothing changed")
            return {"CANCELLED"}
        text = (f"UVs fitted to the {new[0]}x{new[1]} canvas (U x{su:.4g}, V x{sv:.4g})\n"
                f"New canvas starts at x {off[0]}, y {off[1]} of the old {old[0]}x{old[1]}\n"
                f"{loops:,} UVs of '{', '.join(names)}' on {len(objs)} object(s)")
        show_report(self, context, "report_canvas", text)
        return {"FINISHED"}


class RESURFACE_OT_canvas_from_selection(Operator):
    bl_idname = "resurface.canvas_from_selection"
    bl_label = "From Selection"
    bl_description = (
        "Crop at the selected UVs: set the anchor to Top Left and the offset to the top left corner "
        "of the selected faces' UVs (all UVs in Object Mode), in pixels of the old texture"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _uv_poll(cls, context)

    def execute(self, context):
        st = context.scene.resurface
        umin, vmax = math.inf, -math.inf
        for o in mesh_targets(context):
            me = o.data
            layer = me.uv_layers.active
            if layer is None:
                continue
            if o.mode == "EDIT":
                bm = bmesh.from_edit_mesh(me)
                uvl = bm.loops.layers.uv.get(layer.name)
                for f in bm.faces:
                    if f.select:
                        for lp in f.loops:
                            uv = lp[uvl].uv
                            umin = min(umin, uv.x)
                            vmax = max(vmax, uv.y)
            else:
                a = np.empty(len(me.loops) * 2, np.float32)
                layer.data.foreach_get("uv", a)
                a = a.reshape(-1, 2)
                if len(a):
                    umin = min(umin, float(a[:, 0].min()))
                    vmax = max(vmax, float(a[:, 1].max()))
        if not math.isfinite(umin):
            self.report({"WARNING"}, "No faces selected")
            return {"CANCELLED"}

        def px(x):
            r = round(x)
            return int(r) if abs(x - r) < 1e-3 else math.floor(x)

        st.canvas_anchor = "TOP_LEFT"
        st.canvas_offset_x = px(umin * st.canvas_old_w)
        st.canvas_offset_y = px((1.0 - vmax) * st.canvas_old_h)
        self.report({"INFO"}, f"Crop starts at x {st.canvas_offset_x}, y {st.canvas_offset_y}")
        return {"FINISHED"}


# ------------------------------------------------------------------------------------------
# Index Map
# ------------------------------------------------------------------------------------------
def read_texture(img, w, h):
    """Colours of an image at w x h: (h, w, channels) sRGB-encoded 0..1, Blender's row order."""
    sw, sh = img.size
    if sw == 0 or sh == 0:
        raise ValueError(f"image '{img.name}' has no pixel data (is the file missing?)")
    c = img.channels
    buf = np.empty(sw * sh * c, np.float32)
    img.pixels.foreach_get(buf)
    a = imap.resample(buf.reshape(sh, sw, c), w, h)
    del buf
    if c < 3:
        a = np.concatenate([np.repeat(a[..., :1], 3, 2), a[..., 1:]], 2)
    if img.is_float:
        a[..., :3] = imap.linear_to_srgb(a[..., :3])
    return a


def texture_missing(st):
    return st.idx_source == "TEXTURE" and st.idx_texture is None


def auto_detect(st, d, W, H, raster=None):
    """Rows of the automatic mode: (row per texel (H, W), -1 where unused; row per triangle;
    stats)."""
    if raster is None:
        raster = imap.rasterize(d["UV"], W, H)
    if st.idx_source == "TEXTURE":
        tex = read_texture(st.idx_texture, *imap.work_size(W, H))
        return imap.texture_rows(d["P"], d["T"], d["UV"], W, H, tex, weld=st.merge_distance,
                                 tolerance=st.idx_color_tol, detail=st.idx_detail / 100.0,
                                 max_rows=st.idx_max_rows, whole_islands=st.idx_whole_islands, raster=raster)
    rows, prio, stats = imap.auto_rows(d["P"], d["T"], d["UV"], W, H, weld=st.merge_distance,
                                       group_similar=st.idx_group_similar, tolerance=st.idx_similarity,
                                       raster=raster, max_rows=st.idx_max_rows)
    tri, pix = raster
    return imap.resolve(tri, pix, rows, prio, W, H), rows, stats


def color_lines(colors, per_line=2):
    """'1A #A2766C 29%' for every row, a few per line."""
    items = []
    for row, rgb, share in colors:
        r, g, b = (int(round(v * 255)) for v in rgb)
        items.append(f"{imap.ROW_NAMES[row]} #{r:02X}{g:02X}{b:02X} {share * 100:.0f}%")
    return ["   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def polygon_rows(poly, rows, n):
    """Most common row of the triangles of every polygon (-1 for polygons without any)."""
    out = np.full(n, -1, np.int64)
    if len(poly):
        key, cnt = np.unique(poly * 32 + rows, return_counts=True)
        best = imap.first_per_key(key // 32, -cnt)
        out[key[best] // 32] = key[best] % 32
    return out


def _output_path(setting, name):
    """PNG path for the Save To setting (a file or a folder), or None to pack."""
    path = bpy.path.abspath(setting) if setting else ""
    if not path:
        return None
    if path.endswith(("/", "\\")) or os.path.isdir(path):
        safe = "".join("_" if c in '<>:"/\\|?*' else c for c in name)
        path = os.path.join(path, safe + ".png")
    elif not path.lower().endswith(".png"):
        path += ".png"
    return path


def _store_image(name, rgba, W, H, path):
    """Put the pixels into the index map made last time under this name, so materials
    using it update, or into a new image.  Saves it to `path`, or packs it."""
    img = bpy.data.images.get(name)
    if img is not None and (not any(img.get(t) for t in IMAGE_TAGS) or (path and img.packed_file is not None)):
        img = None    # not ours, or packed but now wanted as a file
    for attempt in range(2):
        if img is None:
            img = bpy.data.images.new(name, W, H, alpha=True)
            img[IMAGE_TAGS[0]] = True
        try:
            # colorspace first: changing it can reload the pixels
            img.colorspace_settings.name = "Non-Color"
            if tuple(img.size) != (W, H):
                img.scale(W, H)
            img.pixels.foreach_set(rgba.astype(np.float32).ravel())
            break
        except (RuntimeError, ValueError, TypeError):
            if attempt:
                raise
            img = None    # e.g. its file went missing: start over with a new image
    img.update()
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
    else:
        img.pack()
    return img


class RESURFACE_OT_index_map(Operator):
    bl_idname = "resurface.index_map"
    bl_label = "Generate Index Map"
    bl_description = (
        "Paint an FFXIV index (_id) texture from the active UV map of the selected meshes: every UV island "
        "or texture colour region, or every face with a manual row, gets its colorset row color (red picks "
        "the row pair, green row A or B). Uncovered texture is black, which reads as row 1B"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        st = context.scene.resurface
        if st.idx_mode == "AUTO" and texture_missing(st):
            cls.poll_message_set("Choose the texture to detect the rows from")
            return False
        return _uv_poll(cls, context)

    def execute(self, context):
        import time
        st = context.scene.resurface
        t0 = time.time()
        objs = uv_targets(context)
        W, H = st.idx_width, st.idx_height
        try:
            d = uv_triangles(objs)
            raster = imap.rasterize(d["UV"], W, H)
            if st.idx_mode == "AUTO":
                lab, _, stats = auto_detect(st, d, W, H, raster)
            else:
                face_vals = d["rows"]
                vals = np.zeros(len(d["obj"]), np.int64)
                for i, fv in enumerate(face_vals):
                    sel = d["obj"] == i
                    vals[sel] = fv[d["poly"][sel]]
                rows = np.where(vals > 0, vals - 1, imap.BACKGROUND_ROW)
                prio = (vals > 0).astype(np.int64)      # faces with a row win where they overlap others
                lab = imap.resolve(raster[0], raster[1], rows, prio, W, H)
            n_rows = len(np.unique(lab[lab >= 0]))
            rgba, lab = imap.paint_rows(lab, st.idx_padding)
            active = context.active_object if context.active_object in objs else objs[0]
            name = active.name + "_id"
            img = _store_image(name, rgba, W, H, _output_path(st.idx_output, name))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Index map failed: {e}")
            return {"CANCELLED"}
        auto = st.idx_mode == "AUTO"
        if auto and st.idx_source == "TEXTURE":
            first = f"{count(n_rows, 'row')} from the texture"
        elif auto:
            first = f"{count(stats['islands'], 'island')} on {count(n_rows, 'row')}"
            if stats["groups"] > stats["rows"]:
                first += " (similar ones share)"
        else:
            fv = np.concatenate(face_vals)
            nf = int((fv > 0).sum())
            first = f"{count(len(np.unique(fv[fv > 0])), 'row')} on {count(nf, 'face')}"
            if nf < len(fv):
                first += f", {count(len(fv) - nf, 'face')} without a row (1B)"
        lines = [f"'{img.name}' {W}x{H} in {time.time() - t0:.1f} s: {first}"]
        if auto and st.idx_source == "TEXTURE":
            lines.extend(color_lines(stats["colors"]))
        if auto and stats["stacked"]:
            lines.append(f"{stats['stacked']} stacked islands share their colours" if st.idx_source == "TEXTURE"
                         else f"{stats['stacked']} stacked islands share their row")
        if auto and stats["shared"] > 0.005:
            lines.append(f"UVs overlap on {stats['shared'] * 100:.1f}% of the painted area")
        where = img.filepath_raw if img.packed_file is None else "packed into the .blend"
        lines.append(f"UV map '{', '.join(d['uv_names'])}', {where}")
        text = "\n".join(lines)
        show_report(self, context, "report_index", text)
        return {"FINISHED"}


def _edit_poll(cls, context):
    if context.mode != "EDIT_MESH":
        cls.poll_message_set("Select faces in Edit Mode")
        return False
    return True


class RESURFACE_OT_index_assign(Operator):
    bl_idname = "resurface.index_assign"
    bl_label = "Assign"
    bl_description = "Give the selected faces the chosen colorset row"
    bl_options = {"REGISTER", "UNDO"}

    value: IntProperty(name="Row", default=-1, min=-1, max=32, options={"SKIP_SAVE"})  # type: ignore
    # -1: the row chosen in the panel, 0: remove the row

    @classmethod
    def description(cls, context, props):
        if props and props.value == 0:
            return "Remove the colorset row from the selected faces"
        return cls.bl_description

    @classmethod
    def poll(cls, context):
        return _edit_poll(cls, context)

    def execute(self, context):
        st = context.scene.resurface
        v = self.value if self.value >= 0 else row_value(st.idx_row)
        n = 0
        for o in mesh_targets(context):
            me = o.data
            bm = bmesh.from_edit_mesh(me)
            layer = bm.faces.layers.int.get(ROW_ATTR)
            if layer is None:
                if v == 0:
                    continue
                layer = bm.faces.layers.int.new(ROW_ATTR)
            for f in bm.faces:
                if f.select:
                    f[layer] = v
                    n += 1
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
            _row_cache.pop(o.name, None)
        if n == 0:
            self.report({"WARNING"}, "No faces selected")
            return {"CANCELLED"}
        verb = f"Assigned row {value_name(v)} to" if v else "Removed the row from"
        self.report({"INFO"}, f"{verb} {n:,} faces")
        return {"FINISHED"}


class RESURFACE_OT_index_select(Operator):
    bl_idname = "resurface.index_select"
    bl_label = "Select"
    bl_description = "Select the faces with this row (Shift: add to the selection)"
    bl_options = {"REGISTER", "UNDO"}

    value: IntProperty(name="Row", default=0, min=0, max=32)  # type: ignore
    extend: BoolProperty(name="Extend", default=False)  # type: ignore

    @classmethod
    def poll(cls, context):
        return _edit_poll(cls, context)

    def invoke(self, context, event):
        self.extend = event.shift
        return self.execute(context)

    def execute(self, context):
        if not self.extend:
            bpy.ops.mesh.select_all(action="DESELECT")
        n = 0
        for o in mesh_targets(context):
            me = o.data
            bm = bmesh.from_edit_mesh(me)
            layer = bm.faces.layers.int.get(ROW_ATTR)
            for f in bm.faces:
                if not f.hide and (f[layer] if layer is not None else 0) == self.value:
                    f.select_set(True)
                    n += 1
            bm.select_flush_mode()
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        self.report({"INFO"}, f"Selected {n:,} faces ({value_name(self.value)})")
        return {"FINISHED"}


class RESURFACE_OT_index_clear(Operator):
    bl_idname = "resurface.index_clear"
    bl_label = "Clear Row"
    bl_description = "Remove this row from all faces that have it"
    bl_options = {"REGISTER", "UNDO"}

    value: IntProperty(name="Row", default=1, min=1, max=32)  # type: ignore

    @classmethod
    def poll(cls, context):
        return bool(mesh_targets(context))

    def execute(self, context):
        n = 0
        for o in mesh_targets(context):
            rows = object_rows(o)
            hit = rows == self.value
            if hit.any():
                rows[hit] = 0
                write_rows(o, rows)
                n += int(hit.sum())
        self.report({"INFO"}, f"Cleared row {value_name(self.value)} from {n:,} faces")
        return {"FINISHED"}


class RESURFACE_OT_index_autofill(Operator):
    bl_idname = "resurface.index_autofill"
    bl_label = "Fill Unassigned"
    bl_description = (
        "Give every face without a row the row that Automatic mode would give it (with the settings "
        "below: its UV island's row, or its most common texture colour). Faces that already have a row keep it"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if texture_missing(context.scene.resurface):
            cls.poll_message_set("Choose the texture to detect the rows from")
            return False
        return _uv_poll(cls, context)

    def execute(self, context):
        st = context.scene.resurface
        objs = uv_targets(context)
        try:
            d = uv_triangles(objs)
            _, rows, stats = auto_detect(st, d, st.idx_width, st.idx_height)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Detecting rows failed: {e}")
            return {"CANCELLED"}
        n = 0
        for i, o in enumerate(objs):
            cur = d["rows"][i]
            sel = d["obj"] == i
            auto = polygon_rows(d["poly"][sel], rows[sel], len(cur)) + 1
            fill = (cur == 0) & (auto > 0)
            if fill.any():
                cur[fill] = auto[fill]
                write_rows(o, cur)
                n += int(fill.sum())
        self.report({"INFO"}, f"Filled {n:,} faces ({stats['islands']} islands)")
        return {"FINISHED"}


class RESURFACE_OT_index_convert_groups(Operator):
    bl_idname = "resurface.index_convert_groups"
    bl_label = "Convert Old Row Groups"
    bl_description = (
        "Turn the 'UV_Group..' vertex groups of the old Index Map Generator into face rows and delete them. "
        "The MDL exporter would export those groups as bones and move weight away from the real ones"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.mode != "OBJECT":
            cls.poll_message_set("Switch to Object Mode")
            return False
        return any(legacy_groups(o) for o in mesh_targets(context))

    def execute(self, context):
        faces = groups = 0
        for o in mesh_targets(context):
            vgs = legacy_groups(o)
            if not vgs:
                continue
            row_of = {vg.index: imap.ROW_NAMES.index(vg.name[len(LEGACY_PREFIX):]) for vg in vgs}
            me = o.data
            vert_row = np.full(len(me.vertices), -1, np.int64)
            for v in me.vertices:
                # like the old add-on: the group that comes last in the list wins
                best = -1
                for g in v.groups:
                    if g.group in row_of and g.group > best:
                        best = g.group
                if best >= 0:
                    vert_row[v.index] = row_of[best]
            rows = read_rows(me)
            for p in me.polygons:
                votes = {}
                for vi in p.vertices:
                    r = vert_row[vi]
                    if r >= 0:
                        votes[r] = votes.get(r, 0) + 1
                if votes:
                    rows[p.index] = max(votes, key=votes.get) + 1
                    faces += 1
            write_rows(o, rows)
            for vg in vgs:
                o.vertex_groups.remove(vg)
                groups += 1
        self.report({"INFO"}, f"Converted {groups} vertex groups into rows on {faces:,} faces")
        return {"FINISHED"}


classes = (
    RESURFACE_OT_resize_canvas,
    RESURFACE_OT_canvas_from_selection,
    RESURFACE_OT_index_map,
    RESURFACE_OT_index_assign,
    RESURFACE_OT_index_select,
    RESURFACE_OT_index_clear,
    RESURFACE_OT_index_autofill,
    RESURFACE_OT_index_convert_groups,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    for h in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post, bpy.app.handlers.load_post):
        if _forget_counts not in h:
            h.append(_forget_counts)


def unregister():
    for h in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post, bpy.app.handlers.load_post):
        if _forget_counts in h:
            h.remove(_forget_counts)
    _row_cache.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
