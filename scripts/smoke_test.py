"""Smoke test for an installed Resurface package.

Install the package into a throw-away user profile first, then run the tools on a
UV sphere in background Blender:

    BLENDER_USER_RESOURCES=/tmp/profile blender -b --command extension install-file -r user_default -e Resurface.zip
    BLENDER_USER_RESOURCES=/tmp/profile blender -b --python-exit-code 1 --python scripts/smoke_test.py

Any failure raises, which makes Blender exit with code 1.
"""
import bpy


def check(condition, what):
    if not condition:
        raise RuntimeError(f"Smoke test failed: {what}")
    print(f"ok: {what}")


def run(op, what, **kwargs):
    result = op(**kwargs)
    check(result == {"FINISHED"}, f"{what} {result}")


check(hasattr(bpy.context.scene, "resurface"), "add-on registered (Scene.resurface)")
for name in ("RESURFACE_PT_tabs", "RESURFACE_MT_links", "RESURFACE_PT_uv_index"):
    check(hasattr(bpy.types, name), f"{name} registered")

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
sphere = bpy.context.active_object
sphere.select_set(True)
faces_before = len(sphere.data.polygons)
settings = bpy.context.scene.resurface

run(bpy.ops.resurface.analyze, "Analyze")
check("triangles" in settings.report_mesh, "Analyze report")

run(bpy.ops.resurface.rebuild, "Rebuild Mesh")
mesh = sphere.data
check(len(mesh.polygons) > 100, f"rebuilt mesh has faces ({len(mesh.polygons)})")
check(all(len(p.vertices) == 3 for p in mesh.polygons), "rebuilt mesh is all triangles")
check(len(mesh.uv_layers) > 0, "UVs transferred")

run(bpy.ops.resurface.smooth_normals, "Smooth Normals")
run(bpy.ops.resurface.rebuild_uvs, "Rebuild UVs")
check(mesh.uv_layers.get("Old UVs") is not None, "Old UVs layer kept")

settings.idx_width = settings.idx_height = 256
run(bpy.ops.resurface.index_map, "Index Map Generator")
check(any(img.size[0] == 256 for img in bpy.data.images), "index map image created")

run(bpy.ops.resurface.resize_canvas, "Resize Canvas")

run(bpy.ops.resurface.restore, "Restore Original")
check(len(sphere.data.polygons) == faces_before, "original mesh restored")

print("Smoke test passed")
