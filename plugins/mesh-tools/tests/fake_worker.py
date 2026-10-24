"""A worker that speaks the protocol of ``ppg_mesh_tools/worker.py`` - the very code of it, loaded
from its file - with operations made in numpy instead of pymeshlab, for the tests of the plugin
where pymeshlab is not installed. It runs as the real one does, in a process of its own:

    python -u fake_worker.py JOB.json

``PPG_MESH_TOOLS_FAKE`` changes what it does: ``fail`` (the operation raises MeshLab's message
on a non-manifold mesh), ``crash`` (the process dies as MeshLab does on a segmentation fault),
``sleep`` (the operation takes a minute: for cancelling), ``noise`` (MeshLab's own lines, also
unfinished ones, among the messages).

The operations, simple and predictable: decimate keeps the first faces; remesh reverses the order
of the vertices and drops the texture; smooth raises every vertex by 0.01; close_holes adds a face
when holes of 10 edges or more are allowed (and some faces are marked, when only those count); repair removes duplicate faces; the measures are those
of the faces in numpy.
"""
from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ppg_mesh_tools_worker", HERE.parent / "ppg_mesh_tools" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


class Fake:
    name = "fake"

    def __init__(self):
        from pyarchinit_photogrammetry.io.mesh_ply import MeshData, read_mesh, write_mesh

        self.MeshData, self.read_mesh, self.write_mesh = MeshData, read_mesh, write_mesh
        self.mesh = None
        self.mode = os.environ.get("PPG_MESH_TOOLS_FAKE", "")

    def version(self) -> str:
        return "0.0-fake"

    def load(self, path):
        from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

        self.mesh = self.read_mesh(path)
        faces = PlyMesh(path)._faces
        self.marked = int((faces["quality"] > 0.5).sum()) if "quality" in faces.dtype.names else None
        if self.mode == "noise":
            sys.stdout.write("Warning: MeshLab says something unfinished")  # no newline
            sys.stdout.flush()
            print("\nanother line of MeshLab", flush=True)

    def counts(self):
        return len(self.mesh.vertices), len(self.mesh.faces)

    def _special(self):
        if self.mode == "fail":
            raise RuntimeError("Failed to apply filter\nDetails: Mesh has some not 2-manifold edges, filter requires "
                               "edge manifoldness")
        if self.mode == "crash":
            sys.stdout.flush()
            if hasattr(signal, "SIGSEGV") and os.name != "nt":
                os.kill(os.getpid(), signal.SIGSEGV)
            os._exit(3)
        if self.mode == "sleep":
            time.sleep(60)

    def decimate(self, op):
        self._special()
        keep = np.zeros(len(self.mesh.faces), bool)
        keep[: int(op["target_faces"])] = True
        m = self.mesh.subset_faces(keep)
        if not op.get("texture"):
            m.texcoords, m.texnumber = None, None
        self.mesh = m
        return {"target_faces": int(op["target_faces"]), "texture": bool(op.get("texture"))}

    def remesh(self, op):
        self._special()
        n = len(self.mesh.vertices)
        order = np.arange(n)[::-1]
        where = np.empty(n, np.int64)
        where[order] = np.arange(n)
        self.mesh = self.MeshData(self.mesh.vertices[order], where[self.mesh.faces], None, [], None)
        return {"edge_length": float(op["edge_length"]), "repaired": {}}

    def smooth(self, op):
        self._special()
        v = self.mesh.vertices.copy()
        v["z"] = v["z"] + 0.01
        self.mesh = self.MeshData(v, self.mesh.faces, self.mesh.texcoords, self.mesh.comments, self.mesh.texnumber)
        return {"method": op.get("method"), "steps": op.get("steps")}

    def close_holes(self, op):
        self._special()
        if int(op.get("max_hole_edges", 30)) < 10 or (op.get("select_marked") and not self.marked):
            return {"closed_holes": 0, "new_faces": 0, "repaired": {}}
        f = np.vstack([self.mesh.faces, [[0, 1, 2]]])
        tex = None if self.mesh.texcoords is None else np.vstack([self.mesh.texcoords, np.zeros((1, 6), np.float32)])
        self.mesh = self.MeshData(self.mesh.vertices, f, tex, self.mesh.comments, None)
        return {"closed_holes": 1, "new_faces": 1, "repaired": {}}

    def repair(self, op):
        self._special()
        before = len(self.mesh.faces)
        _, first = np.unique(np.sort(self.mesh.faces, axis=1), axis=0, return_index=True)
        keep = np.zeros(before, bool)
        keep[np.sort(first)] = True
        self.mesh = self.mesh.subset_faces(keep)
        return {"duplicate_faces": {"vertices": 0, "faces": len(self.mesh.faces) - before}}

    def measure(self):
        m = self.mesh
        xyz, f = m.xyz, m.faces
        tri = xyz[f]
        area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, count = np.unique(e, axis=0, return_counts=True)
        boundary = int((count == 1).sum())
        volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6) if not boundary else None
        return {"area": area, "volume": volume, "mean_edge": float(np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2).mean()),
                "bbox_min": xyz.min(axis=0).tolist(), "bbox_max": xyz.max(axis=0).tolist(), "vertices": len(xyz),
                "faces": len(f), "edges": int(len(count)), "boundary_edges": boundary, "components": 1,
                "boundary_loops": 1 if boundary else 0, "genus": 0, "two_manifold": bool((count <= 2).all()),
                "non_manifold_edges": int((count > 2).sum()), "non_manifold_vertices": 0,
                "unreferenced_vertices": int(len(xyz) - len(np.unique(f))), "watertight": boundary == 0}

    def save(self, path, texture, colors):
        m = self.mesh
        fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
        if colors and m.colors is not None:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")]
        v = np.empty(len(m.vertices), fields)
        for c in "xyz":
            v[c] = m.vertices[c]
        if colors and m.colors is not None:
            for c in ("red", "green", "blue"):
                v[c] = m.vertices[c]
            v["alpha"] = 255
        tex = m.texcoords if texture else None
        comments = ["VCGLIB generated"] + (["TextureFile dummy.png"] * len(m.texture_files) if texture else [])
        self.write_mesh(path, self.MeshData(v, m.faces, tex, comments, m.texnumber if texture else None))


if __name__ == "__main__":
    sys.exit(worker.main(sys.argv[1:], backend=Fake()))
