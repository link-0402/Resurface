import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import PropertyGroup

from .core.indexmap import ROW_NAMES, row_rgb


def _row_items():
    # two columns in the dropdown: A rows and B rows
    items = []
    for heading, first in (("A", 0), ("B", 1)):
        items.append(("", heading, ""))
        for r in range(first, 32, 2):
            note = ", like texture no face covers" if r == 1 else ""
            items.append((ROW_NAMES[r], ROW_NAMES[r],
                          "Colorset row %s (#%02X%02X%02X%s)" % ((ROW_NAMES[r],) + row_rgb(r) + (note,))))
    return items


ROW_ITEMS = _row_items()

TAB_ITEMS = [
    ("MESH", "Mesh", "Rebuild the mesh with clean, even triangles", "MOD_REMESH", 0),
    ("NORMALS", "Normals", "Clean smooth normals", "NORMALS_VERTEX_FACE", 1),
    ("UV", "UVs", "UV layout, texture transfer, canvas resize and index maps", "UV", 2),
]

ANCHOR_ITEMS = [
    ("TOP_LEFT", "Top Left", "The new canvas keeps the old top left corner"),
    ("TOP", "Top", "The new canvas is centred at the old top edge"),
    ("TOP_RIGHT", "Top Right", "The new canvas keeps the old top right corner"),
    ("LEFT", "Left", "The new canvas is centred at the old left edge"),
    ("CENTER", "Center", "The new canvas is centred on the old one"),
    ("RIGHT", "Right", "The new canvas is centred at the old right edge"),
    ("BOTTOM_LEFT", "Bottom Left", "The new canvas keeps the old bottom left corner"),
    ("BOTTOM", "Bottom", "The new canvas is centred at the old bottom edge"),
    ("BOTTOM_RIGHT", "Bottom Right", "The new canvas keeps the old bottom right corner"),
]


class ResurfaceSettings(PropertyGroup):
    # ---- Mesh tab -------------------------------------------------------------------------
    size_mode: EnumProperty(
        name="Resolution",
        items=[
            ("RELATIVE", "Relative", "Base triangle density relative to the input mesh"),
            ("LENGTH", "Edge Length", "Base target edge length in scene units"),
        ],
        default="RELATIVE",
    )  # type: ignore
    density: FloatProperty(
        name="Density",
        description="Base density before adaptive refinement. 1.0 gives about as many "
                    "triangles as the input, lower values give fewer and larger triangles",
        default=0.6, min=0.02, soft_max=4.0, max=20.0,
    )  # type: ignore
    edge_length: FloatProperty(
        name="Edge Length",
        description="Base target edge length",
        default=0.005, min=1e-6, soft_min=1e-4, subtype="DISTANCE", unit="LENGTH", precision=4,
    )  # type: ignore
    adaptive: BoolProperty(
        name="Adaptive Detail",
        description="Use smaller triangles where larger ones would visibly deviate from the "
                    "original surface (tight folds, wrinkles, ridges)",
        default=True,
    )  # type: ignore
    tolerance: FloatProperty(
        name="Tolerance",
        description="Allowed deviation from the original surface, in percent of the base edge length. "
                    "Lower keeps more detail but adds triangles",
        default=4.0, min=0.5, max=50.0, subtype="PERCENTAGE", precision=1,
    )  # type: ignore
    min_edge_ratio: FloatProperty(
        name="Smallest Edge",
        description="Adaptive refinement never makes edges shorter than this percentage of the base edge length",
        default=25.0, min=2.0, max=100.0, subtype="PERCENTAGE", precision=0,
    )  # type: ignore
    iterations: IntProperty(
        name="Iterations",
        description="Remeshing passes. More passes give more even triangles",
        default=10, min=2, max=50,
    )  # type: ignore
    rebuild_normals: BoolProperty(
        name="Smooth Normals",
        description="After rebuilding, give the new meshes clean smooth normals with the settings of the "
                    "Normals tab. Off: the original normals are interpolated onto the new mesh "
                    "(right for hair or anything with edited normals)",
        default=True,
    )  # type: ignore
    rebuild_uvs: BoolProperty(
        name="Rebuild UVs",
        description="After rebuilding, replace the UVs with a clean layout using the settings of the UVs tab. "
                    "Off: the original UV layout is transferred unchanged",
        default=False,
    )  # type: ignore
    preserve_uv_seams: BoolProperty(
        name="UV Seams",
        description="Keep UV seams as edges so the UV layout transfers exactly",
        default=True,
    )  # type: ignore
    preserve_color_seams: BoolProperty(
        name="Color Seams",
        description="Keep edges where corner colors change abruptly",
        default=True,
    )  # type: ignore
    preserve_sharp: BoolProperty(
        name="Sharp Edges",
        description="Keep creases sharper than the angle below as edges (hard-surface models). "
                    "Leave off for cloth and other organic meshes",
        default=False,
    )  # type: ignore
    sharp_angle: FloatProperty(
        name="Sharp Angle", default=math.radians(60.0), min=math.radians(5.0), max=math.radians(179.0),
        subtype="ANGLE",
    )  # type: ignore
    corner_angle: FloatProperty(
        name="Corner Angle",
        description="Border and seam points that turn more sharply than this stay fixed",
        default=math.radians(60.0), min=math.radians(5.0), max=math.radians(179.0), subtype="ANGLE",
    )  # type: ignore
    merge_distance: FloatProperty(
        name="Merge Distance",
        description="Vertices closer than this are treated as one (split normals / UV seams in game meshes). "
                    "Also used by Smooth Normals, Rebuild UVs and Index Map Generator",
        default=1e-5, min=0.0, soft_max=0.001, subtype="DISTANCE", unit="LENGTH", precision=6,
    )  # type: ignore
    fragment_faces: IntProperty(
        name="Remove Fragments",
        description="Delete loose pieces with at most this many triangles (decimation debris). 0 keeps everything",
        default=3, min=0, max=100,
    )  # type: ignore
    max_influences: IntProperty(
        name="Max Weights",
        description="Maximum bone influences per vertex. 0 keeps the maximum found in the input",
        default=0, min=0, max=16,
    )  # type: ignore
    keep_original: BoolProperty(
        name="Keep Original",
        description="Keep the original mesh data so it can be restored",
        default=True,
    )  # type: ignore
    clean_features: BoolProperty(
        name="Ignore Tiny Features",
        description="Dissolve UV seam and crease fragments that are smaller than the new triangles "
                    "(decimation leftovers that would otherwise pin vertices and leave small spikes), "
                    "and flip small patches whose faces point the wrong way",
        default=True,
    )  # type: ignore
    seam_threshold: FloatProperty(
        name="Seam Threshold",
        description="UV or color differences smaller than this are not treated as seams. "
                    "0.002 is about one pixel on a 512 texture",
        default=0.002, min=0.0, soft_max=0.02, precision=4, step=0.01,
    )  # type: ignore
    remove_hidden: BoolProperty(
        name="Remove Hidden Layers",
        description="Delete under-layers that lie just beneath another surface facing the same way "
                    "(duplicated or simulated cloth thickness). They cannot be seen from outside and are "
                    "back faces from inside. Keep this off when the outer layer uses see-through textures "
                    "(lace, mesh fabric), because the layer below is visible there",
        default=False,
    )  # type: ignore
    hidden_gap: FloatProperty(
        name="Layer Gap",
        description="Largest distance between a shell and the hidden layer below it. 0 = automatic "
                    "(1% of the size of the selection)",
        default=0.0, min=0.0, soft_max=0.05, subtype="DISTANCE", unit="LENGTH", precision=4,
    )  # type: ignore

    # ---- Normals tab ----------------------------------------------------------------------
    normals_angle: FloatProperty(
        name="Smooth Angle",
        description="Faces meeting at a sharper angle than this keep a hard edge. "
                    "Faces that are wound the other way are aligned first, not averaged out",
        default=math.radians(80.0), min=math.radians(1.0), max=math.radians(180.0), subtype="ANGLE",
    )  # type: ignore
    normals_blur: IntProperty(
        name="Blur",
        description="Extra smoothing passes over the normals. Softens lumpy, decimated surfaces; "
                    "higher values also soften the shading of real wrinkles",
        default=1, min=0, max=50,
    )  # type: ignore
    fix_winding: BoolProperty(
        name="Fix Flipped Faces",
        description="Also flip tiny groups of faces that point the other way than their surroundings "
                    "(stray decimation debris; in game they are holes because of back-face culling). "
                    "Not needed for smooth shading, and larger inward-facing regions are left alone "
                    "because they are usually meant to be seen from inside",
        default=False,
    )  # type: ignore

    # ---- UVs tab: Rebuild UVs -------------------------------------------------------------
    uv_keep_old_seams: BoolProperty(
        name="Use Old Islands",
        description="Cut along the borders of the old UV islands (usually the original pattern pieces). "
                    "Their outline is usually fine; it is their layout that is broken",
        default=True,
    )  # type: ignore
    uv_cut_sharp: BoolProperty(
        name="Cut Sharp Edges",
        description="Also cut along creases sharper than the angle below (hard-surface models)",
        default=False,
    )  # type: ignore
    uv_sharp_angle: FloatProperty(
        name="Sharp Angle",
        description="Creases sharper than this become seams when Cut Sharp Edges is on",
        default=math.radians(60.0), min=math.radians(5.0), max=math.radians(179.0), subtype="ANGLE",
    )  # type: ignore
    uv_keep_orientation: BoolProperty(
        name="Keep Orientation",
        description="Turn every new island the way the same faces were oriented in the old UVs and pack "
                    "without rotating. Keeps textures, and especially normal maps, easy to transfer",
        default=True,
    )  # type: ignore
    uv_margin: FloatProperty(
        name="Margin",
        description="Space between islands, as a fraction of the texture",
        default=0.004, min=0.0, max=0.1, precision=4, step=0.01,
    )  # type: ignore

    # ---- UVs tab: Transfer Texture --------------------------------------------------------
    tex_image: bpy.props.PointerProperty(
        name="Texture",
        description="Texture painted for the old UVs, to be moved onto the new UV layout",
        type=bpy.types.Image,
    )  # type: ignore
    tex_mode: EnumProperty(
        name="Type",
        items=[
            ("COLOR", "Diffuse or Mask", "Smooth (bilinear) resampling"),
            ("INDEX", "Index / ID", "Nearest resampling, values are never blended (FFXIV _id colorset maps)"),
            ("NORMAL", "Normal Map", "Resample and rotate the tangent-space direction (R, G) with each island; "
                                     "B and A are copied as they are"),
        ],
        default="COLOR",
    )  # type: ignore
    tex_flip_green: BoolProperty(
        name="Green Points Down",
        description="The normal map's green channel points towards -V (DirectX style in Blender's UV space). "
                    "Only matters for islands that were rotated",
        default=False,
    )  # type: ignore
    tex_padding: IntProperty(
        name="Padding",
        description="Pixels to extend each island outwards (prevents seams from filtering)",
        default=3, min=0, max=64, subtype="PIXEL",
    )  # type: ignore

    # ---- UVs tab: Resize Canvas -----------------------------------------------------------
    canvas_old_w: IntProperty(
        name="Old Width", description="Width of the texture the UVs were made for, in pixels",
        default=2048, min=1, soft_max=8192, max=65536, subtype="PIXEL",
    )  # type: ignore
    canvas_old_h: IntProperty(
        name="Old Height", description="Height of the texture the UVs were made for, in pixels",
        default=4096, min=1, soft_max=8192, max=65536, subtype="PIXEL",
    )  # type: ignore
    canvas_new_w: IntProperty(
        name="New Width", description="Width of the cropped or extended texture, in pixels",
        default=2048, min=1, soft_max=8192, max=65536, subtype="PIXEL",
    )  # type: ignore
    canvas_new_h: IntProperty(
        name="New Height", description="Height of the cropped or extended texture, in pixels",
        default=2048, min=1, soft_max=8192, max=65536, subtype="PIXEL",
    )  # type: ignore
    canvas_anchor: EnumProperty(
        name="Anchor",
        description="Where the new canvas sits on the old one. Use the same anchor as in the image "
                    "editor's canvas size dialog",
        items=ANCHOR_ITEMS,
        default="TOP_LEFT",
    )  # type: ignore
    canvas_offset_x: IntProperty(
        name="Offset X",
        description="Extra shift of the new canvas to the right, in pixels of the old texture. "
                    "With the Top Left anchor this is the left edge of a crop rectangle",
        default=0, soft_min=-8192, soft_max=8192, subtype="PIXEL",
    )  # type: ignore
    canvas_offset_y: IntProperty(
        name="Offset Y",
        description="Extra shift of the new canvas downwards, in pixels of the old texture. "
                    "With the Top Left anchor this is the top edge of a crop rectangle",
        default=0, soft_min=-8192, soft_max=8192, subtype="PIXEL",
    )  # type: ignore
    canvas_selected_only: BoolProperty(
        name="Only Selected Faces",
        description="In Edit Mode, only move the UVs of selected faces. In Object Mode whole meshes are changed",
        default=False,
    )  # type: ignore

    # ---- UVs tab: Index Map Generator -----------------------------------------------------
    idx_mode: EnumProperty(
        name="Rows",
        items=[
            ("AUTO", "Automatic", "Rows are found from the UV islands or from a texture's colours, largest area first"),
            ("MANUAL", "Manual", "Rows are assigned to faces by hand in Edit Mode"),
        ],
        default="AUTO",
    )  # type: ignore
    idx_source: EnumProperty(
        name="Detect",
        description="What the automatic rows are based on",
        items=[
            ("ISLANDS", "UV Islands", "Every UV island gets its own row (from the UV layout and the mesh alone)"),
            ("TEXTURE", "Texture", "Every region of one colour in a base colour (diffuse) texture gets its own row, "
                                   "also inside an island. Islands that share texture space are one region"),
        ],
        default="ISLANDS",
    )  # type: ignore
    idx_texture: bpy.props.PointerProperty(
        name="Texture",
        description="Base colour (diffuse, _base / _d) texture painted for the active UV map",
        type=bpy.types.Image,
    )  # type: ignore
    idx_color_tol: FloatProperty(
        name="Color Tolerance",
        description="How different two colours must be to get separate rows (CIE delta E: about 2 is barely "
                    "visible, 10 clearly different). Lower splits more, e.g. baked shading into several rows",
        default=12.0, min=1.0, max=60.0, soft_max=40.0, precision=1,
    )  # type: ignore
    idx_detail: FloatProperty(
        name="Detail Size",
        description="Colour regions smaller than this (percent of the texture's size) join their surroundings: "
                    "print, stitching, dirt. Larger keeps only big areas",
        default=0.4, min=0.0, max=5.0, precision=2, step=5, subtype="PERCENTAGE",
    )  # type: ignore
    idx_whole_islands: BoolProperty(
        name="Whole Islands",
        description="Give every UV island the row of its main colour instead of splitting it by colour. "
                    "Islands of the same colour still share a row",
        default=False,
    )  # type: ignore
    idx_max_rows: IntProperty(
        name="Max Rows",
        description="Use at most this many rows. When more are found, the most alike share one "
                    "(closest colours, or islands of the most similar shape)",
        default=31, min=1, max=31,
    )  # type: ignore
    idx_row: EnumProperty(
        name="Row",
        description="Colorset row to assign to the selected faces",
        items=ROW_ITEMS,
        default="1A",
    )  # type: ignore
    idx_group_similar: BoolProperty(
        name="Match Similar Shapes",
        description="Islands with the same shape and size (mirrored or repeated pieces) share a row. "
                    "Islands that are stacked on the same texture area always share one",
        default=False,
    )  # type: ignore
    idx_similarity: FloatProperty(
        name="Shape Tolerance",
        description="How different two islands may be (average relative difference of triangle count, "
                    "area and extents) and still share a row",
        default=0.1, min=0.0, max=1.0, precision=2, step=1, subtype="FACTOR",
    )  # type: ignore
    idx_width: IntProperty(
        name="Width", description="Width of the index map in pixels",
        default=1024, min=16, soft_max=4096, max=16384, subtype="PIXEL",
    )  # type: ignore
    idx_height: IntProperty(
        name="Height", description="Height of the index map in pixels",
        default=1024, min=16, soft_max=4096, max=16384, subtype="PIXEL",
    )  # type: ignore
    idx_padding: IntProperty(
        name="Padding",
        description="Pixels to extend each island outwards, so filtering and mip maps do not pull in "
                    "the background row at island borders",
        default=4, min=0, max=64, subtype="PIXEL",
    )  # type: ignore
    idx_output: StringProperty(
        name="Save To",
        description="PNG file or folder to save the index map to. Empty: keep it packed in the .blend "
                    "(save it from the Image Editor)",
        default="", subtype="FILE_PATH",
    )  # type: ignore

    # ---- results shown in each panel ------------------------------------------------------
    report_mesh: StringProperty(default="")  # type: ignore
    report_normals: StringProperty(default="")  # type: ignore
    report_uvs: StringProperty(default="")  # type: ignore
    report_texture: StringProperty(default="")  # type: ignore
    report_canvas: StringProperty(default="")  # type: ignore
    report_index: StringProperty(default="")  # type: ignore

    def to_settings(self):
        from .pipeline import RebuildSettings
        return RebuildSettings(
            size_mode=self.size_mode,
            density=self.density,
            edge_length=self.edge_length,
            adaptive=self.adaptive,
            tolerance=self.tolerance / 100.0,
            min_edge_ratio=self.min_edge_ratio / 100.0,
            iterations=self.iterations,
            merge_distance=self.merge_distance,
            fragment_faces=self.fragment_faces,
            preserve_uv_seams=self.preserve_uv_seams,
            preserve_color_seams=self.preserve_color_seams,
            preserve_sharp=self.preserve_sharp,
            sharp_angle=math.degrees(self.sharp_angle),
            corner_angle=math.degrees(self.corner_angle),
            max_influences=self.max_influences,
            keep_original=self.keep_original,
            remove_hidden=self.remove_hidden,
            hidden_gap=self.hidden_gap,
            clean_features=self.clean_features,
            seam_threshold=self.seam_threshold,
            smooth_normals=self.rebuild_normals,
            normals_angle=math.degrees(self.normals_angle),
            normals_blur=self.normals_blur,
            fix_winding=self.fix_winding,
            rebuild_uvs=self.rebuild_uvs,
            uv_keep_old_seams=self.uv_keep_old_seams,
            uv_cut_sharp=self.uv_cut_sharp,
            uv_sharp_angle=math.degrees(self.uv_sharp_angle),
            uv_keep_orientation=self.uv_keep_orientation,
            uv_margin=self.uv_margin,
        )


def register():
    bpy.utils.register_class(ResurfaceSettings)
    bpy.types.Scene.resurface = bpy.props.PointerProperty(type=ResurfaceSettings)
    # UI state only: window manager properties are neither saved nor part of undo
    bpy.types.WindowManager.resurface_tab = EnumProperty(name="Tab", items=TAB_ITEMS, default="MESH")


def unregister():
    del bpy.types.WindowManager.resurface_tab
    del bpy.types.Scene.resurface
    bpy.utils.unregister_class(ResurfaceSettings)
