"""The rebuild pipeline, as a step generator the modal operator can drive."""

import math
import time

import numpy as np

from . import blender_io
from .core.geom import face_normals, norm
from .core.remesh import Remesher
from .core.source import SourceMesh, SourceOptions
from .core.transfer import CornerMap, corner_sources


class RebuildSettings:
    """Plain settings container (mirrors the add-on's scene properties)."""

    def __init__(self, **kw):
        self.size_mode = "RELATIVE"      # RELATIVE | LENGTH
        self.density = 0.6               # RELATIVE: 1.0 = same triangle count as the input
        self.edge_length = 0.005         # LENGTH: target edge length in scene units
        self.adaptive = True
        self.tolerance = 0.04            # max deviation, as a fraction of the edge length
        self.min_edge_ratio = 0.25       # smallest edge as a fraction of the edge length
        self.iterations = 10
        self.merge_distance = 1e-5
        self.fragment_faces = 3
        self.preserve_uv_seams = True
        self.preserve_color_seams = True
        self.preserve_sharp = False
        self.sharp_angle = 60.0
        self.corner_angle = 60.0
        self.max_influences = 0          # 0 = like the source
        self.keep_original = True
        self.remove_hidden = False       # delete under-layers hidden beneath a same-facing shell
        self.hidden_gap = 0.0            # max shell/under-layer distance; 0 = 1% of the size
        self.clean_features = True       # dissolve seam/crease noise smaller than the triangles
        self.seam_threshold = 0.002      # UV/color difference below this is not a seam
        # afterwards, run the Smooth Normals tool on the result (off: keep the original shading)
        self.smooth_normals = True
        self.normals_angle = 80.0        # degrees
        self.normals_blur = 1            # blur passes
        self.fix_winding = False
        # afterwards, run the Rebuild UVs tool on the result (off: keep the original layout)
        self.rebuild_uvs = False
        self.uv_keep_old_seams = True
        self.uv_cut_sharp = False
        self.uv_sharp_angle = 60.0       # degrees
        self.uv_keep_orientation = True
        self.uv_margin = 0.004
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


def auto_gap(src):
    """Default search distance for hidden layers: 1% of the bounding box diagonal."""
    if len(src.V) == 0:
        return 0.0
    return 0.01 * float(np.linalg.norm(src.V.max(0) - src.V.min(0)))


def find_hidden(src, gap):
    """Raw triangles belonging to hidden under-layer patches (and their duplicates).

    Returns (exclude mask over raw triangles, hidden area fraction)."""
    hidden, frac = src.hidden_patches(gap)
    exclude = np.zeros(src.stats["raw_triangles"], bool)
    if len(hidden) == 0:
        return exclude, 0.0
    hf = np.isin(src.fpatch, hidden)
    exclude[src.ftri[hf]] = True
    if len(src.twin_face):
        exclude[src.twin_raw[hf[src.twin_face]]] = True
    area = 0.5 * norm(face_normals(src.V, src.F))
    return exclude, float(area[hf].sum() / max(area.sum(), 1e-30))


class RebuildJob:
    def __init__(self, objects, settings):
        self.objects = list(objects)
        self.s = settings
        self.report = {}
        self.cancelled = False
        self.modified = False            # Blender data was changed (set right before writing)

    def target_length(self, src, exclude=None):
        """Base edge length, measured on the part of `src` that will be rebuilt."""
        s = self.s
        if s.size_mode == "LENGTH":
            return float(s.edge_length)
        area = 0.5 * norm(face_normals(src.V, src.F))
        keep = np.ones(len(src.F), bool) if exclude is None else ~exclude[src.ftri]
        n = max(int(keep.sum()) * max(s.density, 1e-3), 1.0)
        return math.sqrt(float(area[keep].sum()) / (n * math.sqrt(3.0) / 4.0))

    def steps(self):
        """Yields (fraction, message).  Blender data is only modified in the last step."""
        s = self.s
        t_start = time.time()
        yield 0.0, "Reading meshes"
        g = blender_io.gather(self.objects)
        seams = blender_io.seam_attributes(g, use_uv=s.preserve_uv_seams, use_color=s.preserve_color_seams)
        yield 0.03, "Analysing topology"
        opts = SourceOptions(
            merge_distance=s.merge_distance,
            fragment_faces=s.fragment_faces,
            sharp_angle=s.sharp_angle if s.preserve_sharp else None,
            corner_angle=s.corner_angle,
            seam_tolerance=s.seam_threshold,
            flip_islands=s.clean_features,
        )
        src0 = SourceMesh(g["V_raw"], g["T_raw"], g["T_obj"], g["T_mat"], seams, opts)
        self.report["source"] = dict(src0.stats)
        yield 0.04, "Looking for hidden layers"
        gap = s.hidden_gap if s.hidden_gap > 0 else auto_gap(src0)
        exclude, hidden_area = find_hidden(src0, gap)
        excl = None
        if s.remove_hidden:
            self.report["hidden_area"] = hidden_area
            if exclude.any():
                excl = exclude
        elif hidden_area > 0.02:
            self.report["hidden_hint"] = hidden_area
        L0 = self.target_length(src0, excl)
        self.report["edge_length"] = L0
        if s.clean_features or excl is not None:
            yield 0.045, "Cleaning up features"
            if s.clean_features:
                opts.min_feature_size = L0
            src = SourceMesh(g["V_raw"], g["T_raw"], g["T_obj"], g["T_mat"], seams, opts,
                             exclude_tris=excl)
        else:
            src = src0
        self.report["features"] = {k: src.stats.get(k, 0) for k in
                                   ("noise_patches_merged", "noise_curves_dropped", "faces_reoriented",
                                    "chains", "corners")}
        rm = Remesher(src, L0, min_length=L0 * s.min_edge_ratio)
        eps = s.tolerance * L0 if s.adaptive else None
        for frac, msg in rm.run(max(int(s.iterations), 1), adaptive_eps=eps):
            yield 0.05 + 0.8 * frac, msg
        self.report["remesh"] = rm.stats()
        self.src, self.rm = src, rm      # kept for inspection / debugging
        yield 0.87, "Transferring attributes"
        cs_face, cs_bary = corner_sources(src, rm)
        cmap = CornerMap(src, g["T_loops"], g["T_raw"], cs_face, cs_bary)
        results = blender_io.build_results(g, src, rm, cmap, max_influences=s.max_influences or None)
        yield 0.92, "Writing meshes"
        written = []
        unchanged = []
        new_objs = []
        self.modified = True
        for obj, r in zip(self.objects, results):
            if r is None:
                unchanged.append(obj.name)
                continue
            if s.smooth_normals:
                r.normals = None      # replaced right below
            blender_io.apply_result(obj, r, keep_original=s.keep_original)
            written.append((obj.name, len(r.positions), len(r.faces)))
            new_objs.append(obj)
        self.report["written"] = written
        self.report["unchanged"] = unchanged
        # the optional steps are the standalone tools, run on the new meshes
        if s.smooth_normals and new_objs:
            yield 0.94, "Smoothing normals"
            flipped, _ = blender_io.smooth_object_normals(
                new_objs, math.radians(s.normals_angle), s.normals_blur, s.fix_winding, s.merge_distance)
            self.report["normals"] = {"flipped": flipped}
        if s.rebuild_uvs and new_objs:
            yield 0.95, "Rebuilding UVs"
            import bpy
            self.report["uvs"] = blender_io.rebuild_uvs(
                bpy.context, new_objs,
                keep_old_seams=s.uv_keep_old_seams,
                cut_sharp=s.uv_cut_sharp,
                sharp_angle=s.uv_sharp_angle,
                margin=s.uv_margin,
                keep_orientation=s.uv_keep_orientation,
                merge_distance=s.merge_distance,
            )
        self.report["seconds"] = time.time() - t_start
        yield 1.0, "Done"

    def run(self, progress=None):
        for frac, msg in self.steps():
            if progress:
                progress(frac, msg)
        return self.report
