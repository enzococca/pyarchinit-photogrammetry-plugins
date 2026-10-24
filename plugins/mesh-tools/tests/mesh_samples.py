"""Synthetic models for the tests: an excavated surface (a grid with a hole, colours, a vertex
confidence, a texture whose coordinates are the x and y of the corners) and a sphere, in a
project whose chunk has a scale of 2 (internal units to metres)."""
from __future__ import annotations

import numpy as np

SCALE = 2.0


def grid_mesh(n: int = 40, hole: bool = True, texture: bool = True, duplicate_seam: bool = False):
    """A wavy square of side 1 (internal units), ``n`` x ``n`` vertices; a round hole in the middle;
    the texture coordinates of each corner equal to its x and y. ``duplicate_seam``: the vertices of
    the column x = 0.5 repeated for the faces on its right, as a textured model has them along the
    seams of its texture."""
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    xs = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, xs)
    Z = 0.05 * np.sin(6 * X) * np.cos(5 * Y)
    v = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    if hole:
        cen = v[f].mean(axis=1)
        f = f[np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5) > 0.12]
        used = np.unique(f)
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        v, f = v[used], remap[f]
    if duplicate_seam:
        col = np.where(np.isclose(v[:, 0], xs[n // 2]))[0]
        copies = np.arange(len(v), len(v) + len(col))
        v = np.vstack([v, v[col]])
        right = v[f].mean(axis=1)[:, 0] > xs[n // 2]
        lut = np.arange(len(v))
        lut[col] = copies
        f = np.where(right[:, None], lut[f], f)
    arr = np.empty(len(v), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                            ("confidence", "u1")])
    arr["x"], arr["y"], arr["z"] = v.T
    arr["red"] = np.clip(v[:, 0] * 255, 0, 255).astype("u1")
    arr["green"] = np.clip(v[:, 1] * 255, 0, 255).astype("u1")
    arr["blue"] = 100
    arr["confidence"] = (np.arange(len(v)) % 250).astype("u1")
    tex = arr_tex = None
    if texture:
        tex = np.column_stack([arr["x"], arr["y"]])[f].reshape(len(f), 6).astype(np.float32)
        arr_tex = ["TextureFile model_texture.jpg"]
    return MeshData(arr, f.astype(np.int64), tex, arr_tex or [], None)


def sphere_mesh(radius: float = 1.0, subdivisions: int = 4):
    import trimesh

    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    v = np.asarray(s.vertices)
    arr = np.empty(len(v), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = v.T
    arr["red"], arr["green"], arr["blue"] = 200, 150, 100
    return MeshData(arr, np.asarray(s.faces, np.int64))


def project(tmp_path, mesh=None, texture: bool = True, scale: float = SCALE, name: str = "scavo"):
    """A saved project with one chunk (scale ``scale``, a translation) whose model is ``mesh`` (the
    textured grid by default) with its texture image when it has texture coordinates."""
    import cv2

    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh

    mesh = mesh if mesh is not None else grid_mesh(texture=texture)
    doc = ppg.Document()
    chunk = doc.add_chunk("Scavo")
    doc.chunk = chunk
    doc.save(tmp_path / f"{name}.ppg")
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = [500000.0, 4500000.0, 100.0]
    chunk.transform = T
    chunk._set_product("model", write_mesh(chunk.data_dir / "model.ply", mesh))
    if mesh.texcoords is not None:
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :] = (40, 120, 200)
        cv2.imwrite(str(chunk.data_dir / "model_texture.jpg"), img)
        chunk._set_product("model_textured", chunk.data_dir / "model_texture.jpg")
    doc.save()
    return doc, chunk
