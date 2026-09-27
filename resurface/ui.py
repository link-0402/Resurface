import bpy
from bpy.types import Menu, Panel

from . import uvtools
from .core import canvas as canvas_math
from .core import indexmap as imap


def active_tab(context):
    return context.window_manager.resurface_tab


def objects_text(context, verb, fallback):
    n = sum(1 for o in context.selected_objects if o.type == "MESH")
    return f"{verb} {n} Object{'s' if n != 1 else ''}" if n else fallback


def draw_report(layout, s, prop):
    """The last result of a tool, with a button to hide it."""
    text = getattr(s, prop)
    if not text:
        return
    box = layout.box()
    col = box.column(align=True)
    lines = text.split("\n")
    row = col.row()
    row.label(text=lines[0])
    row.operator("resurface.clear_report", text="", icon="X", emboss=False).prop = prop
    for line in lines[1:]:
        col.label(text=line)


def tab_link(layout, context, tab):
    """A small button that opens another tab."""
    layout.prop_enum(context.window_manager, "resurface_tab", tab, text="", icon="PREFERENCES")


def prepare(layout, context):
    layout.use_property_split = True
    layout.use_property_decorate = False
    return context.scene.resurface


class _Sidebar:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Resurface"


class _TabPanel(_Sidebar):
    tab = ""

    @classmethod
    def poll(cls, context):
        return active_tab(context) == cls.tab


# ==========================================================================================
# tab bar
# ==========================================================================================
# Luci_xiv's pages: (label, icon, URL).
LINKS = (
    ("XIV Mod Archive", "PACKAGE", "https://www.xivmodarchive.com/user/124593"),
    ("GitHub", "SCRIPT", "https://github.com/link-0402/MagicFit"),
    ("Bluesky", "COMMUNITY", "https://bsky.app/profile/xiv-luci.bsky.social"),
    ("Ko-fi", "FUND", "https://ko-fi.com/luci_xiv"),
)


class RESURFACE_MT_links(Menu):
    """Resurface on GitHub, and Luci_xiv's mods, Bluesky and Ko-fi"""
    bl_label = "Links"

    def draw(self, context):
        for label, icon, url in LINKS:
            self.layout.operator("wm.url_open", text=label, icon=icon).url = url


class RESURFACE_PT_tabs(_Sidebar, Panel):
    # The page buttons live in the header.  Blender sorts header-less panels ahead of
    # all others, which would pull the whole sidebar tab above Item / Tool / View.
    bl_label = ""
    bl_options = {"HEADER_LAYOUT_EXPAND"}
    bl_order = 0

    def draw_header(self, context):
        row = self.layout.row()
        row.row(align=True).prop(context.window_manager, "resurface_tab", expand=True)
        # The links at the right end, where Blender puts the presets.  draw_header_preset
        # would end up under the page buttons, which fill the whole header.
        sub = row.row()
        sub.emboss = "NONE"
        sub.ui_units_x = 1
        sub.menu(RESURFACE_MT_links.__name__, text="", icon="URL")

    def draw(self, context):
        pass


# ==========================================================================================
# Mesh tab
# ==========================================================================================
class RESURFACE_PT_mesh(_TabPanel, Panel):
    bl_label = "Rebuild Mesh"
    bl_order = 1
    tab = "MESH"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.row(align=True).prop(s, "size_mode", expand=True)
        if s.size_mode == "RELATIVE":
            col.prop(s, "density")
        else:
            col.prop(s, "edge_length")
        col.prop(s, "adaptive")
        sub = col.column()
        sub.active = s.adaptive
        sub.prop(s, "tolerance")
        sub.prop(s, "min_edge_ratio")
        col.prop(s, "iterations")
        col.prop(s, "remove_hidden")

        col = layout.column(heading="Afterwards")
        row = col.row(align=True)
        row.prop(s, "rebuild_normals")
        tab_link(row, context, "NORMALS")
        row = col.row(align=True)
        row.prop(s, "rebuild_uvs")
        tab_link(row, context, "UV")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("resurface.rebuild", icon="MOD_REMESH", text=objects_text(context, "Rebuild", "Rebuild Mesh"))
        row = layout.row(align=True)
        row.operator("resurface.analyze", icon="VIEWZOOM")
        row.operator("resurface.restore", icon="LOOP_BACK")
        draw_report(layout, s, "report_mesh")


class RESURFACE_PT_mesh_preserve(_Sidebar, Panel):
    bl_label = "Preserve"
    bl_parent_id = "RESURFACE_PT_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column(heading="Keep")
        col.prop(s, "preserve_uv_seams")
        col.prop(s, "preserve_color_seams")
        col.prop(s, "preserve_sharp")
        sub = col.column()
        sub.active = s.preserve_sharp
        sub.prop(s, "sharp_angle")
        layout.prop(s, "seam_threshold")
        layout.prop(s, "clean_features")
        layout.prop(s, "corner_angle")
        layout.label(text="Borders and non-manifold junctions are always kept", icon="INFO")


class RESURFACE_PT_mesh_cleanup(_Sidebar, Panel):
    bl_label = "Cleanup"
    bl_parent_id = "RESURFACE_PT_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        sub = layout.column()
        sub.active = s.remove_hidden
        sub.prop(s, "hidden_gap")
        layout.prop(s, "merge_distance")
        layout.prop(s, "fragment_faces")


class RESURFACE_PT_mesh_output(_Sidebar, Panel):
    bl_label = "Output"
    bl_parent_id = "RESURFACE_PT_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        layout.prop(s, "max_influences")
        layout.prop(s, "keep_original")
        layout.operator("resurface.discard_original", icon="TRASH")


# ==========================================================================================
# Normals tab
# ==========================================================================================
class RESURFACE_PT_normals(_TabPanel, Panel):
    bl_label = "Smooth Normals"
    bl_order = 1
    tab = "NORMALS"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.prop(s, "normals_angle")
        col.prop(s, "normals_blur")
        col.prop(s, "fix_winding")
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("resurface.smooth_normals", icon="NORMALS_VERTEX_FACE",
                     text=objects_text(context, "Smooth", "Smooth Normals"))
        note = "Rebuild Mesh applies these too" if s.rebuild_normals else "Rebuild Mesh can apply these too"
        row = layout.row()
        row.label(text=note, icon="INFO")
        tab_link(row, context, "MESH")
        draw_report(layout, s, "report_normals")


# ==========================================================================================
# UVs tab
# ==========================================================================================
class RESURFACE_PT_uv_rebuild(_TabPanel, Panel):
    bl_label = "Rebuild UVs"
    bl_order = 1
    tab = "UV"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.prop(s, "uv_keep_old_seams")
        col.prop(s, "uv_cut_sharp")
        sub = col.column()
        sub.active = s.uv_cut_sharp
        sub.prop(s, "uv_sharp_angle")
        col.prop(s, "uv_keep_orientation")
        col.prop(s, "uv_margin")
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("resurface.rebuild_uvs", icon="UV", text="Rebuild UVs")
        note = "Rebuild Mesh applies these too" if s.rebuild_uvs else "Rebuild Mesh can apply these too"
        row = layout.row()
        row.label(text=note, icon="INFO")
        tab_link(row, context, "MESH")
        draw_report(layout, s, "report_uvs")


class RESURFACE_PT_uv_transfer(_TabPanel, Panel):
    bl_label = "Transfer Texture"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}
    tab = "UV"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        layout.template_ID(s, "tex_image", open="image.open")
        col = layout.column()
        col.prop(s, "tex_mode")
        if s.tex_mode == "NORMAL":
            col.prop(s, "tex_flip_green")
        col.prop(s, "tex_padding")
        row = layout.row()
        row.scale_y = 1.3
        row.operator("resurface.transfer_texture", icon="IMAGE_DATA")
        layout.label(text="From the 'Old UVs' layer to the current UVs", icon="INFO")
        draw_report(layout, s, "report_texture")


class RESURFACE_PT_uv_canvas(_TabPanel, Panel):
    bl_label = "Resize Canvas"
    bl_order = 3
    bl_options = {"DEFAULT_CLOSED"}
    tab = "UV"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column(align=True)
        col.prop(s, "canvas_old_w", text="Old Size X")
        col.prop(s, "canvas_old_h", text="Y")
        col = layout.column(align=True)
        col.prop(s, "canvas_new_w", text="New Size X")
        col.prop(s, "canvas_new_h", text="Y")

        # 3 x 3 anchor grid like an image editor's canvas size dialog
        split = layout.split(factor=0.4)
        lab = split.row()
        lab.alignment = "RIGHT"
        lab.label(text="Anchor")
        grid = split.column(align=True)
        keys = list(canvas_math.ANCHORS)
        for r in range(3):
            row = grid.row(align=True)
            for key in keys[r * 3:r * 3 + 3]:
                on = s.canvas_anchor == key
                row.prop_enum(s, "canvas_anchor", key, text="", icon="LAYER_ACTIVE" if on else "LAYER_USED")

        col = layout.column(align=True)
        col.prop(s, "canvas_offset_x", text="Offset X")
        col.prop(s, "canvas_offset_y", text="Y")
        row = layout.row()
        row.alignment = "RIGHT"
        row.operator("resurface.canvas_from_selection", icon="RESTRICT_SELECT_OFF")
        sub = layout.column()
        sub.active = context.mode == "EDIT_MESH"
        sub.prop(s, "canvas_selected_only")

        old, new, off = uvtools.canvas_setup(s)
        (su, sv), _ = canvas_math.canvas_transform(old, new, off)
        col = layout.column(align=True)
        col.label(text=f"U x{su:.4g}, V x{sv:.4g}, crop at x {off[0]}, y {off[1]}", icon="INFO")
        row = layout.row()
        row.scale_y = 1.3
        row.operator("resurface.resize_canvas", icon="FULLSCREEN_EXIT")
        draw_report(layout, s, "report_canvas")


class RESURFACE_PT_uv_index(_TabPanel, Panel):
    bl_label = "Index Map Generator"
    bl_order = 4
    bl_options = {"DEFAULT_CLOSED"}
    tab = "UV"

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        row = layout.row(align=True)
        row.prop(s, "idx_mode", expand=True)
        if s.idx_mode == "AUTO":
            self.draw_auto(layout.column(), s)
        else:
            self.draw_manual(context, layout, s)

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "idx_width", text="Size X")
        col.prop(s, "idx_height", text="Y")
        layout.prop(s, "idx_padding")
        layout.prop(s, "idx_output")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("resurface.index_map", icon="IMAGE_RGB")
        draw_report(layout, s, "report_index")

    @staticmethod
    def draw_auto(col, s):
        col.row(align=True).prop(s, "idx_source", expand=True)
        if s.idx_source == "TEXTURE":
            col.template_ID(s, "idx_texture", open="image.open")
            col.prop(s, "idx_color_tol")
            col.prop(s, "idx_detail")
            col.prop(s, "idx_whole_islands")
        else:
            col.prop(s, "idx_group_similar")
            sub = col.column()
            sub.active = s.idx_group_similar
            sub.prop(s, "idx_similarity")
        col.prop(s, "idx_max_rows")

    def draw_manual(self, context, layout, s):
        box = layout.box()
        col = box.column()
        col.use_property_split = False
        row = col.row(align=True)
        row.prop(s, "idx_row", text="")
        row.operator("resurface.index_assign", text="Assign")
        row.operator("resurface.index_assign", text="Remove").value = 0
        if context.mode != "EDIT_MESH":
            col.label(text="Select faces in Edit Mode to assign rows", icon="INFO")

        # rows in use, with buttons to select or clear them
        objs = uvtools.mesh_targets(context)
        counts = uvtools.row_counts(objs) if objs else {}
        lst = col.column(align=True)
        for v in sorted(counts, key=lambda v: (v == 0, v)):
            row = lst.row(align=True)
            if v:
                r, g, b = imap.row_rgb(v - 1)
                row.label(text=f"{uvtools.value_name(v)}   #{r:02X}{g:02X}{b:02X}")
            else:
                row.label(text="No row (1B)")
            row.label(text=f"{counts[v]:,} faces")
            row.operator("resurface.index_select", text="", icon="RESTRICT_SELECT_OFF").value = v
            if v:
                row.operator("resurface.index_clear", text="", icon="X").value = v
            else:
                row.label(text="", icon="BLANK1")

        sub = box.column()
        sub.operator("resurface.index_autofill", icon="AUTO")
        self.draw_auto(sub, s)

        legacy = [vg.name for o in objs for vg in uvtools.legacy_groups(o)]
        if legacy:
            sub = box.column()
            sub.label(text=f"{len(legacy)} old UV_Group vertex groups", icon="ERROR")
            sub.operator("resurface.index_convert_groups", icon="GROUP_VERTEX")


classes = (
    RESURFACE_MT_links,
    RESURFACE_PT_tabs,
    RESURFACE_PT_mesh,
    RESURFACE_PT_mesh_preserve,
    RESURFACE_PT_mesh_cleanup,
    RESURFACE_PT_mesh_output,
    RESURFACE_PT_normals,
    RESURFACE_PT_uv_rebuild,
    RESURFACE_PT_uv_transfer,
    RESURFACE_PT_uv_canvas,
    RESURFACE_PT_uv_index,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
