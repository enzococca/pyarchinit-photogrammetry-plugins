"""MeshLab's filters on a mesh file, in a process of its own.

pymeshlab carries its own native libraries (Qt among them) and some of its filters crash on a
mesh they do not accept (the isotropic remeshing of a non-manifold mesh is a segmentation
fault): the plugin never imports it in the program's process, it runs this file with a Python
that has pymeshlab - the program's own when pymeshlab is installed there, else the environment
the plugin installs - and only the files and the lines below go between the two.

    python -u worker.py JOB.json

This file imports nothing of the program or of the plugin (the environment of the plugin has
only pymeshlab and numpy). The job::

    {"protocol": 1, "input": "in.ply", "output": "out.ply" | null,
     "operations": [{"op": "decimate", "target_faces": 200000, ...}, ...],
     "measure": true, "texture": false, "colors": true}

lengths in the units of the mesh file (the plugin converts the metres of the user). ``texture``:
the texture coordinates of the face corners are written to the output; ``colors``: the vertex
colours. Every message is one line on stdout, the prefix ``@ppg-mesh-tools`` and a JSON object:

* ``{"type": "hello", "protocol": 1, "backend": "pymeshlab", "version": "2025.7.post1"}``
* ``{"type": "progress", "fraction": 0.4, "phase": "run", "op": "decimate", "step": 1, "steps": 1}``
  (phases ``load``, ``run``, ``measure``, ``save``)
* ``{"type": "result", "result": {...}}``: counts before and after, one entry per operation with
  what it did, the measures
* ``{"type": "error", "error": "...", "op": "...", "traceback": "..."}``

Other lines on stdout and stderr are MeshLab's own: the plugin keeps the last ones for the report
of a failure. The exit code is 0 with a result, 1 after an error.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
PREFIX = "@ppg-mesh-tools "
OPERATIONS = ("decimate", "remesh", "smooth", "close_holes", "repair")
# the repair, in this order: what a step leaves is cleaned by the next ones
REPAIR_STEPS = ("duplicate_vertices", "duplicate_faces", "null_faces", "non_manifold_edges",
                "non_manifold_vertices", "unreferenced_vertices")


def emit(message_type: str, **data) -> None:
    """One message of the protocol. The line starts on a line of its own even when MeshLab left a
    line unfinished on stdout."""
    sys.stdout.write("\n" + PREFIX + json.dumps({"type": message_type, **data}) + "\n")
    sys.stdout.flush()


def _plain(value):
    """Numbers, lists and dictionaries of JSON from what pymeshlab returns (numpy arrays and
    scalars, bounding boxes)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


class PyMeshLab:
    """The operations of the plugin with pymeshlab's filters, on the current mesh of a MeshSet."""

    name = "pymeshlab"

    def __init__(self):
        import pymeshlab

        self.ml = pymeshlab
        self.ms = pymeshlab.MeshSet()
        # an absolute length: PureValue since pymeshlab 2023.12, AbsoluteValue before
        self.pure = getattr(pymeshlab, "PureValue", None) or getattr(pymeshlab, "AbsoluteValue")

    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("pymeshlab")
        except Exception:  # noqa: BLE001 - a build without metadata
            return ""

    # ------------------------------------------------------------------ files
    def load(self, path: str) -> None:
        # texture images named in the file and not beside it become placeholders: only the
        # texture coordinates are used, the pages are never read
        self.ms.load_new_mesh(str(path))

    def counts(self) -> tuple[int, int]:
        m = self.ms.current_mesh()
        return int(m.vertex_number()), int(m.face_number())

    def save(self, path: str, texture: bool, colors: bool) -> None:
        options = dict(binary=True, save_vertex_color=bool(colors), save_wedge_texcoord=bool(texture),
                       save_vertex_normal=False, save_vertex_quality=False, save_vertex_flag=False,
                       save_vertex_radius=False, save_face_color=False, save_face_quality=False, save_face_flag=False,
                       save_wedge_color=False, save_wedge_normal=False)
        try:
            self.ms.save_current_mesh(str(path), save_textures=False, **options)
        except self.ml.PyMeshLabException as exc:
            if "not found" not in str(exc):  # an option another version of pymeshlab does not know
                raise
            self.ms.save_current_mesh(str(path), save_textures=False, binary=True, save_vertex_color=bool(colors),
                                      save_wedge_texcoord=bool(texture))

    # ------------------------------------------------------------------ operations
    def decimate(self, op: dict) -> dict:
        """Quadric edge collapse; with ``texture`` the variant that keeps the texture coordinates
        of the face corners (the texture stays valid)."""
        target = int(op["target_faces"])
        common = dict(targetfacenum=target, targetperc=0.0, qualitythr=float(op.get("quality", 0.3)),
                      preserveboundary=bool(op.get("preserve_boundary", True)),
                      boundaryweight=float(op.get("boundary_weight", 1.0)),
                      preservenormal=bool(op.get("preserve_normal", False)),
                      optimalplacement=bool(op.get("optimal_placement", True)),
                      planarquadric=bool(op.get("planar", False)))
        if op.get("texture"):
            self.ms.meshing_decimation_quadric_edge_collapse_with_texture(
                extratcoordw=float(op.get("texture_weight", 1.0)), **common)
        else:
            self.ms.meshing_decimation_quadric_edge_collapse(
                preservetopology=bool(op.get("preserve_topology", True)), autoclean=True, **common)
        return {"target_faces": target, "texture": bool(op.get("texture"))}

    def remesh(self, op: dict) -> dict:
        """Isotropic explicit remeshing to an edge length (it needs a 2-manifold mesh: repaired first)."""
        details = {"repaired": self.repair({}) if op.get("repair_first", True) else {}}
        kw = dict(iterations=int(op.get("iterations", 10)), adaptive=bool(op.get("adaptive", False)),
                  targetlen=self.pure(float(op["edge_length"])), featuredeg=float(op.get("feature_angle", 30.0)))
        if float(op.get("max_distance") or 0.0) > 0:
            kw.update(checksurfdist=True, maxsurfdist=self.pure(float(op["max_distance"])))
        self.ms.meshing_isotropic_explicit_remeshing(**kw)
        details["edge_length"] = float(op["edge_length"])
        return details

    def smooth(self, op: dict) -> dict:
        """Taubin (lambda and mu, no shrinking) or HC Laplacian, ``steps`` times."""
        method, steps = op.get("method", "taubin"), int(op.get("steps", 10))
        if method == "taubin":
            self.ms.apply_coord_taubin_smoothing(lambda_=float(op.get("lambda", 0.5)), mu=float(op.get("mu", -0.53)),
                                                 stepsmoothnum=steps)
        elif method == "hc":
            for _ in range(steps):
                self.ms.apply_coord_hc_laplacian_smoothing()
        else:
            raise ValueError(f"smoothing method {method!r}: taubin or hc")
        return {"method": method, "steps": steps}

    def close_holes(self, op: dict) -> dict:
        """Holes up to ``max_hole_edges`` edges closed (the mesh must be 2-manifold: repaired first);
        with ``select_marked`` only those along the faces of quality 1 of the input."""
        details = {"repaired": self.repair({}) if op.get("repair_first", True) else {}}
        kw = dict(maxholesize=int(op.get("max_hole_edges", 30)), selfintersection=bool(op.get("self_intersection", True)),
                  newfaceselected=False)
        if op.get("select_marked"):  # only the holes the plugin marked (face quality 1), not the outlines
            self.ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
            kw["selected"] = True
        if float(op.get("refine_edge") or 0.0) > 0:
            kw.update(refinehole=True, refineholeedgelen=self.pure(float(op["refine_edge"])))
        out = self.ms.meshing_close_holes(**kw) or {}
        details.update({"closed_holes": int(out.get("closed_holes", 0) or 0), "new_faces": int(out.get("new_faces", 0) or 0)})
        return details

    def repair(self, op: dict) -> dict:
        """Duplicate vertices and faces, faces without area, non-manifold edges (faces removed or
        vertices split) and vertices (split), unused vertices: what each step changed."""
        method = "Split Vertices" if op.get("non_manifold_method") == "split" else "Remove Faces"
        calls = {"duplicate_vertices": self.ms.meshing_remove_duplicate_vertices,
                 "duplicate_faces": self.ms.meshing_remove_duplicate_faces,
                 "null_faces": self.ms.meshing_remove_null_faces,
                 "non_manifold_edges": lambda: self.ms.meshing_repair_non_manifold_edges(method=method),
                 "non_manifold_vertices": lambda: self.ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0),
                 "unreferenced_vertices": self.ms.meshing_remove_unreferenced_vertices}
        out = {}
        for step in REPAIR_STEPS:
            if not op.get(step, True):
                continue
            v0, f0 = self.counts()
            calls[step]()
            v1, f1 = self.counts()
            out[step] = {"vertices": v1 - v0, "faces": f1 - f0}
        return out

    def measure(self) -> dict:
        """Area, volume (closed meshes), edges, topology: in the units of the file."""
        geo = self.ms.get_geometric_measures()
        topo = self.ms.get_topological_measures()
        box = geo.get("bbox")
        out = {"area": float(geo.get("surface_area", 0.0)),
               "volume": float(geo["mesh_volume"]) if "mesh_volume" in geo else None,
               "mean_edge": float(geo.get("avg_edge_length", 0.0)),
               "bbox_min": _plain(box.min()) if box is not None else None,
               "bbox_max": _plain(box.max()) if box is not None else None}
        keys = {"vertices": "vertices_number", "faces": "faces_number", "edges": "edges_number",
                "boundary_edges": "boundary_edges", "components": "connected_components_number",
                "boundary_loops": "number_holes", "genus": "genus", "two_manifold": "is_mesh_two_manifold",
                "non_manifold_edges": "non_two_manifold_edges", "non_manifold_vertices": "non_two_manifold_vertices",
                "unreferenced_vertices": "unreferenced_vertices"}
        for k, name in keys.items():
            out[k] = _plain(topo.get(name))
        out["watertight"] = bool(out["two_manifold"]) and out["boundary_edges"] == 0
        return out


def run_job(job: dict, backend) -> dict:
    """The operations of ``job`` on its input with ``backend``, the messages of the protocol on
    the way; the result (also sent as the last message)."""
    ops = list(job.get("operations") or [])
    for op in ops:
        if op.get("op") not in OPERATIONS:
            raise ValueError(f"unknown operation {op.get('op')!r}: {', '.join(OPERATIONS)}")
    t0 = time.monotonic()
    emit("hello", protocol=PROTOCOL, backend=backend.name, version=backend.version())
    # the share of the bar: reading 10 %, the operations 75 %, measures 5 %, writing 10 %
    where = {"phase": "load", "op": None}  # named in the error message of a failure
    try:
        emit("progress", fraction=0.0, phase="load")
        backend.load(job["input"])
        v0, f0 = backend.counts()
        steps = []
        for k, op in enumerate(ops):
            where.update(phase="run", op=op["op"])
            emit("progress", fraction=0.1 + 0.75 * k / max(len(ops), 1), phase="run", op=op["op"], step=k + 1,
                 steps=len(ops))
            before = backend.counts()
            details = getattr(backend, op["op"])(op) or {}
            after = backend.counts()
            steps.append({"op": op["op"], "vertices_before": before[0], "faces_before": before[1],
                          "vertices": after[0], "faces": after[1], "details": _plain(details)})
        measures = None
        if job.get("measure"):
            where.update(phase="measure", op=None)
            emit("progress", fraction=0.85, phase="measure")
            measures = _plain(backend.measure())
        if job.get("output"):
            where.update(phase="save", op=None)
            emit("progress", fraction=0.9, phase="save")
            backend.save(job["output"], texture=bool(job.get("texture")), colors=bool(job.get("colors", True)))
    except Exception as exc:
        try:
            exc.ppg_where = dict(where)
        except AttributeError:  # an exception of an extension that takes no attribute
            pass
        raise
    v1, f1 = backend.counts()
    result = {"vertices_before": v0, "faces_before": f0, "vertices": v1, "faces": f1, "steps": steps,
              "measures": measures, "backend": backend.name, "version": backend.version(),
              "seconds": round(time.monotonic() - t0, 3)}
    emit("result", result=result)
    return result


def main(argv=None, backend=None) -> int:
    """Run the job of the file named first in ``argv``; ``backend``: another implementation of
    the operations (the tests use one without pymeshlab)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ValueError("usage: worker.py JOB.json")
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if int(job.get("protocol", PROTOCOL)) != PROTOCOL:
            raise ValueError(f"protocol {job.get('protocol')}: this worker speaks {PROTOCOL}")
        backend = backend if backend is not None else PyMeshLab()
        run_job(job, backend)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the plugin as a message
        where = getattr(exc, "ppg_where", None) or {"phase": "start", "op": None}
        emit("error", error=str(exc).strip() or type(exc).__name__, exception=type(exc).__name__, phase=where["phase"],
             op=where["op"], traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # out at once: after a failed filter the destructors of pymeshlab can crash the interpreter
    # on its way out, and the message has gone already
    import os

    os._exit(code)
