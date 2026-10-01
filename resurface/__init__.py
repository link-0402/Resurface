"""Resurface: rebuild messy meshes with clean, even topology.

Works on the raw surface instead of a voxel grid, so borders, non-manifold junctions,
UV seams and thin strips (strings, straps, fringe) survive.  UVs, colors, custom
normals, vertex weights and shape keys are transferred onto the new topology.

The sidebar tab has three pages: Mesh (the rebuild), Normals (Smooth Normals, Normal
Map Seams) and UVs (Rebuild UVs, Transfer Texture, Resize Canvas, Index Map Generator).
"""

if "bpy" in locals():
    import importlib
    for _m in ("props", "operators", "uvtools", "ui", "pipeline", "blender_io"):
        if _m in locals():
            importlib.reload(locals()[_m])

import bpy  # noqa: F401

from . import operators, props, ui, uvtools


def register():
    props.register()
    operators.register()
    uvtools.register()
    ui.register()


def unregister():
    ui.unregister()
    uvtools.unregister()
    operators.unregister()
    props.unregister()
