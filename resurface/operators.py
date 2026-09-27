import traceback

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import blender_io
from .pipeline import RebuildJob


def selected_meshes(context):
    return [o for o in context.selected_objects if o.type == "MESH" and len(o.data.polygons)]


def show_report(op, context, prop, text):
    """Keep the result for the panel (one box per tool) and flash it in the status bar."""
    setattr(context.scene.resurface, prop, text)
    op.report({"INFO"}, text.replace("\n", " | "))
    # the sidebar does not notice properties set from Python, e.g. at the end of a modal run
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def format_uv_stats(stats, seconds=None):
    took = f" in {seconds:.1f} s" if seconds is not None else ""
    n = stats["islands"]
    return (f"New UV layout: {n} island{'' if n == 1 else 's'}{took}\n"
            f"Even texel density on {stats['within_25pct'] * 100:.1f}% of the surface\n"
            f"Previous UVs kept in '{blender_io.OLD_UV_NAME}'")


def format_report(rep):
    lines = []
    src = rep.get("source", {})
    rm = rep.get("remesh", {})
    if "written" in rep:
        tris = sum(t for _, _, t in rep["written"])
        verts = sum(v for _, v, _ in rep["written"])
        lines.append(f"Rebuilt {len(rep['written'])} object(s) in {rep.get('seconds', 0):.1f} s")
        lines.append(f"{src.get('raw_triangles', 0):,} -> {tris:,} triangles, {verts:,} vertices")
    if rep.get("hidden_area"):
        lines.append(f"Removed hidden layers: {rep['hidden_area'] * 100:.1f}% of the surface")
    if rep.get("hidden_hint"):
        lines.append(f"Tip: {rep['hidden_hint'] * 100:.0f}% of the surface is a hidden under-layer "
                     "(enable Remove Hidden Layers)")
    for name in rep.get("unchanged", []):
        lines.append(f"'{name}' is entirely hidden or debris, left unchanged")
    if "edge_length" in rep:
        lines.append(f"Base edge length {rep['edge_length'] * 1000:.2f} mm")
    if rm:
        lines.append(f"Triangle quality: median {rm['q_med']:.2f}, worst 1% {rm['q_p1']:.2f}")
    if "normals" in rep:
        flipped = rep["normals"]["flipped"]
        lines.append("Smooth normals" + (f", flipped {flipped:,} faces that pointed the wrong way" if flipped else ""))
    if "uvs" in rep:
        lines.extend(format_uv_stats(rep["uvs"]).split("\n"))
    return "\n".join(lines)


def format_analysis(stats, hidden_area=0.0):
    s = stats
    lines = [
        f"{s['raw_triangles']:,} triangles, {s['raw_vertices']:,} vertices ({s['welded_vertices']:,} after merging)",
        f"Non-manifold edges: {s['nonmanifold_edges']:,}   Border edges: {s['boundary_edges']:,}",
        f"UV/color seam edges: {s['seam_edges']:,}   Winding changes: {s.get('inconsistent_edges', 0):,}",
        f"Degenerate faces: {s['degenerate_removed']:,}   Duplicate/backface copies: {s.get('twin_faces', 0):,} (kept)",
        f"Loose fragments: {s['fragments_removed']:,} ({s['fragment_faces_removed']:,} faces)",
        f"Surface patches: {s['patches']:,}   Feature curves: {s['chains']:,}   Fixed corners: {s['corners']:,}",
    ]
    if hidden_area > 0.005:
        lines.append(f"Hidden under-layers: {hidden_area * 100:.1f}% of the surface (see Remove Hidden Layers)")
    return "\n".join(lines)


class RESURFACE_OT_rebuild(Operator):
    bl_idname = "resurface.rebuild"
    bl_label = "Rebuild Mesh"
    bl_description = (
        "Rebuild the selected meshes with clean, even triangles. Borders, non-manifold junctions, "
        "UV seams and thin strips are kept; UVs, colors, normals, weights and shape keys are transferred. "
        "Optionally smooths the normals and rebuilds the UVs afterwards"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.mode != "OBJECT":
            cls.poll_message_set("Switch to Object Mode")
            return False
        if not selected_meshes(context):
            cls.poll_message_set("Select one or more mesh objects")
            return False
        return True

    def _make_job(self, context):
        return RebuildJob(selected_meshes(context), context.scene.resurface.to_settings())

    def _finish(self, context, rep):
        show_report(self, context, "report_mesh", format_report(rep))

    def execute(self, context):
        job = self._make_job(context)
        try:
            rep = job.run()
        except Exception as e:  # noqa: BLE001 - report any failure to the user
            traceback.print_exc()
            self.report({"ERROR"}, f"Rebuild failed: {e}")
            return {"CANCELLED"}
        self._finish(context, rep)
        return {"FINISHED"}

    def invoke(self, context, event):
        self._job = self._make_job(context)
        self._gen = self._job.steps()
        self._writing = False
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        wm.progress_begin(0, 100)
        context.workspace.status_text_set("Resurface: starting...   Esc to cancel")
        return {"RUNNING_MODAL"}

    def _cleanup(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type == "ESC" and not self._writing:
            self._cleanup(context)
            self.report({"WARNING"}, "Rebuild cancelled, nothing was changed")
            return {"CANCELLED"}
        if event.type != "TIMER":
            # keep the scene untouched while the job runs
            return {"RUNNING_MODAL"}
        try:
            frac, msg = next(self._gen)
        except StopIteration:
            self._cleanup(context)
            self._finish(context, self._job.report)
            return {"FINISHED"}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._cleanup(context)
            self.report({"ERROR"}, f"Rebuild failed: {e}")
            return {"CANCELLED"}
        if msg.startswith("Writing"):
            self._writing = True
        context.window_manager.progress_update(frac * 100)
        busy = "" if self._writing else "   Esc to cancel"
        context.workspace.status_text_set(f"Resurface: {msg} ({frac * 100:.0f}%){busy}")
        return {"RUNNING_MODAL"}


class RESURFACE_OT_analyze(Operator):
    bl_idname = "resurface.analyze"
    bl_label = "Analyze"
    bl_description = "Report what is wrong with the selected meshes, without changing anything"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(selected_meshes(context))

    def execute(self, context):
        from .core.source import SourceMesh, SourceOptions
        import math
        st = context.scene.resurface
        objs = selected_meshes(context)
        try:
            g = blender_io.gather(objs)
            seams = blender_io.seam_attributes(g, use_uv=st.preserve_uv_seams, use_color=st.preserve_color_seams)
            src = SourceMesh(
                g["V_raw"], g["T_raw"], g["T_obj"], g["T_mat"], seams,
                SourceOptions(
                    merge_distance=st.merge_distance,
                    fragment_faces=st.fragment_faces,
                    sharp_angle=math.degrees(st.sharp_angle) if st.preserve_sharp else None,
                    corner_angle=math.degrees(st.corner_angle),
                    seam_tolerance=st.seam_threshold,
                ),
            )
            from .pipeline import auto_gap, find_hidden
            gap = st.hidden_gap if st.hidden_gap > 0 else auto_gap(src)
            _, hidden_area = find_hidden(src, gap)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Analysis failed: {e}")
            return {"CANCELLED"}
        show_report(self, context, "report_mesh", format_analysis(src.stats, hidden_area))
        return {"FINISHED"}


class RESURFACE_OT_restore(Operator):
    bl_idname = "resurface.restore"
    bl_label = "Restore Original"
    bl_description = "Put the original mesh back on the selected rebuilt objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(blender_io.original_name(o) for o in context.selected_objects)

    def execute(self, context):
        n = sum(1 for o in context.selected_objects if blender_io.restore_original(o))
        self.report({"INFO"}, f"Restored {n} object(s)")
        return {"FINISHED"}


class RESURFACE_OT_discard_original(Operator):
    bl_idname = "resurface.discard_original"
    bl_label = "Discard Original"
    bl_description = "Delete the kept original meshes of the selected objects (keeps the rebuilt result)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(blender_io.original_name(o) for o in context.selected_objects)

    def execute(self, context):
        n = 0
        for o in context.selected_objects:
            name = blender_io.original_name(o)
            if not name:
                continue
            me = bpy.data.meshes.get(name)
            if me is not None:
                me.use_fake_user = False
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            blender_io.forget_original(o)
            n += 1
        self.report({"INFO"}, f"Discarded {n} original mesh(es)")
        return {"FINISHED"}


class RESURFACE_OT_smooth_normals(Operator):
    bl_idname = "resurface.smooth_normals"
    bl_label = "Smooth Normals"
    bl_description = (
        "Give the selected meshes clean smooth normals: shading ignores UV seams and part borders, "
        "faces wound the other way are aligned instead of averaged out, only real hard edges stay hard. "
        "Works on any mesh, rebuilt or not"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.mode != "OBJECT":
            cls.poll_message_set("Switch to Object Mode")
            return False
        if not selected_meshes(context):
            cls.poll_message_set("Select one or more mesh objects")
            return False
        return True

    def execute(self, context):
        import time
        st = context.scene.resurface
        objs = selected_meshes(context)
        t0 = time.time()
        try:
            flipped, loops = blender_io.smooth_object_normals(
                objs, st.normals_angle, st.normals_blur, st.fix_winding, st.merge_distance)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Smooth normals failed: {e}")
            return {"CANCELLED"}
        text = f"Smoothed normals on {len(objs)} object(s) in {time.time() - t0:.1f} s"
        if flipped:
            text += f"\nFlipped {flipped:,} faces that pointed the wrong way"
        show_report(self, context, "report_normals", text)
        return {"FINISHED"}


class RESURFACE_OT_rebuild_uvs(Operator):
    bl_idname = "resurface.rebuild_uvs"
    bl_label = "Rebuild UVs"
    bl_description = (
        "Replace the UVs of the selected meshes with a clean layout: islands follow the model's pieces, "
        "every island is unwrapped with minimal stretch and packed without overlaps. "
        "The previous UVs are kept in the 'Old UVs' layer for texture transfer"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.mode != "OBJECT":
            cls.poll_message_set("Switch to Object Mode")
            return False
        if not selected_meshes(context):
            cls.poll_message_set("Select one or more mesh objects")
            return False
        return True

    def execute(self, context):
        import math
        import time
        st = context.scene.resurface
        objs = selected_meshes(context)
        t0 = time.time()
        try:
            stats = blender_io.rebuild_uvs(
                context, objs,
                keep_old_seams=st.uv_keep_old_seams,
                cut_sharp=st.uv_cut_sharp,
                sharp_angle=math.degrees(st.uv_sharp_angle),
                margin=st.uv_margin,
                keep_orientation=st.uv_keep_orientation,
                merge_distance=st.merge_distance,
            )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Rebuild UVs failed: {e}")
            return {"CANCELLED"}
        show_report(self, context, "report_uvs", format_uv_stats(stats, time.time() - t0))
        return {"FINISHED"}


class RESURFACE_OT_transfer_texture(Operator):
    bl_idname = "resurface.transfer_texture"
    bl_label = "Transfer Texture"
    bl_description = (
        "Resample the chosen texture from the old UVs ('Old UVs' layer) onto the current UV layout "
        "of the selected meshes. Saves '<name>_newuv.png' next to the original file"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        st = context.scene.resurface
        if st.tex_image is None:
            cls.poll_message_set("Choose a texture")
            return False
        objs = selected_meshes(context)
        if not objs or not all(o.data.uv_layers.get(blender_io.OLD_UV_NAME) for o in objs):
            cls.poll_message_set("Select meshes that went through Rebuild UVs")
            return False
        return context.mode == "OBJECT"

    def execute(self, context):
        import time
        st = context.scene.resurface
        t0 = time.time()
        try:
            new, path = blender_io.transfer_texture(
                selected_meshes(context), st.tex_image, mode=st.tex_mode,
                flip_green=st.tex_flip_green, padding=st.tex_padding)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Texture transfer failed: {e}")
            return {"CANCELLED"}
        where = path if path else "packed into the .blend"
        show_report(self, context, "report_texture", f"'{new.name}' created in {time.time() - t0:.1f} s ({where})")
        return {"FINISHED"}


class RESURFACE_OT_clear_report(Operator):
    bl_idname = "resurface.clear_report"
    bl_label = "Clear"
    bl_description = "Hide this result"
    bl_options = {"INTERNAL"}

    prop: StringProperty()  # type: ignore

    def execute(self, context):
        st = context.scene.resurface
        if self.prop.startswith("report_") and hasattr(st, self.prop):
            setattr(st, self.prop, "")
        return {"FINISHED"}


classes = (
    RESURFACE_OT_rebuild,
    RESURFACE_OT_analyze,
    RESURFACE_OT_restore,
    RESURFACE_OT_discard_original,
    RESURFACE_OT_smooth_normals,
    RESURFACE_OT_rebuild_uvs,
    RESURFACE_OT_transfer_texture,
    RESURFACE_OT_clear_report,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
