"""The proxies modelled in Blender, back into the program.

The panel of the live script in Blender sends a proxy - modelled there by hand, or started by
EMtools for a unit the program has no proxy of - with *Invia al programma*: ``ppg_inbox/<name>.glb``
beside the manifest, Y-up and shifted as the files of the project, and ``<name>.json`` saying what it
is (the name of the object, the id of its node, the shift of the scene). :func:`take` brings each into
the chunk as a proxy of the program, as *Strumenti > Proxy 3D delle US* makes them:

* the PLY in world coordinates, in double precision, beside the others (``proxies/<name>.ply``);
* the volume when the solid is closed - small holes are closed first; an open mesh has no volume,
  and that is said (its volume is written as 0);
* the area in plan, the elevations of its top and of its bottom (the faces looking up and down);
* the unit from the matrix: by the id of the node, else by the name (also with the code of the graph
  EMtools puts before it, ``SCAVO.1.US12``).

An entry of the chunk with the same name is replaced (the proxy built from the DEMs gives way to the
one modelled). The files leave the inbox once taken; one that cannot be read stays, and is said.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .emjson import Matrix, _parsed
from .scene import Folders, from_gltf

INBOX = "ppg_inbox"


def inbox(folders: Folders) -> Path:
    return Path(folders.base) / INBOX


def pending(folders: Folders) -> list[Path]:
    """The descriptions (``.json``) of the proxies waiting, whose glTF is there too."""
    folder = inbox(folders)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.json") if p.with_suffix(".glb").exists())


def close_holes(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The holes of a mesh closed: every loop of border edges (edges of one face only) becomes a fan of
    triangles about a new vertex in its middle, wound as the faces around it. Plain numpy: the
    program carries no networkx, which trimesh's own repair needs."""
    f = np.asarray(faces, np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _u, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = edges[counts[inverse.ravel()] == 1]
    if not len(border):
        return np.asarray(vertices, float), f
    following = {}
    for a, b in border:  # a hole is walked backwards: the new faces run against the border edges
        following.setdefault(int(b), []).append(int(a))
    verts, new = [np.asarray(vertices, float)], []
    count = len(verts[0])
    while following:
        start = next(iter(following))
        loop, v = [start], start
        while True:
            nxt = following[v].pop()
            if not following[v]:
                del following[v]
            if nxt == start or nxt not in following:
                break
            loop.append(nxt)
            v = nxt
        if len(loop) < 3:
            continue
        centre = verts[0][loop].mean(axis=0)
        verts.append(centre[None])
        new += [(loop[i], loop[(i + 1) % len(loop)], count) for i in range(len(loop))]
        count += 1
    return np.vstack(verts), np.vstack([f, np.asarray(new, np.int64).reshape(-1, 3)])


def measure(vertices, faces) -> dict:
    """``closed``, ``volume_m3``, ``area_m2`` (in plan), ``z_top`` and ``z_bottom`` (min, max) of a
    mesh in world coordinates, as the program measures its proxies."""
    import shapely
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64), process=True)
    closed = bool(mesh.is_watertight)
    if not closed:
        v, f = close_holes(mesh.vertices, mesh.faces)
        mended = trimesh.Trimesh(v, f, process=True)
        # holes, not a missing side: a sheet or an open surface closed by its own outline is no solid
        if mended.is_watertight and mended.area - mesh.area < 0.75 * mesh.area:
            mesh, closed = mended, True
    volume = float(abs(mesh.volume)) if closed else 0.0
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    up, down = normals[:, 2] > 0.2, normals[:, 2] < -0.2

    def heights(mask):
        z = v[np.unique(f[mask])][:, 2] if mask.any() else v[:, 2]
        return [float(z.min()), float(z.max())]

    tri = v[f[up] if up.any() else f][:, :, :2]
    cross = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - \
        (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    tri = tri[np.abs(cross) > 1e-12]
    area = 0.0
    if len(tri):
        rings = np.concatenate([tri, tri[:, :1]], axis=1)
        area = float(shapely.union_all(shapely.polygons(rings)).area)
    return {"closed": closed, "volume_m3": volume, "area_m2": area, "z_top": heights(up), "z_bottom": heights(down),
            "vertices": v, "faces": f}


def unit_for(matrix: Matrix | None, name: str, node_id: str | None):
    """The unit of the matrix an object sent from Blender stands for: by the id of its node, else by
    its name (exact, or ending with ``.<unit>``: EMtools puts the code of the graph before it)."""
    if matrix is None:
        return None
    if node_id:
        unit = matrix.unit(str(node_id))
        if unit is not None:
            return unit
    exact = matrix.named(name)
    if exact is not None:
        return exact
    found = [u for u in matrix.units if name.endswith("." + u.name)]
    return max(found, key=lambda u: len(u.name)) if found else None


def _fields(name: str) -> dict:
    """Area, type and number read from the name of an object (``1.US12``, ``SCAVO.1.USM3``, ``US12``)."""
    parts = name.split(".")
    for k in range(len(parts)):
        found = _parsed(".".join(parts[k:]))
        if found:
            return found
    return {}


def take(chunk, folders: Folders, matrix: Matrix | None = None, progress=None) -> dict:
    """The proxies waiting in the inbox into ``chunk.meta["proxies"]``. Returns ``taken`` (the names
    of the entries), ``replaced``, ``open`` (no volume), ``errors`` (``{file: reason}``) and
    ``republish`` (entries that were published in PyArchInit with another geometry)."""
    import trimesh

    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    progress = progress or (lambda *a: None)
    report = {"taken": [], "replaced": [], "open": [], "errors": {}, "republish": []}
    waiting = pending(folders)
    if not waiting:
        return report
    entries = list((chunk.meta or {}).get("proxies") or [])
    folder = chunk.data_dir / "proxies"
    folder.mkdir(parents=True, exist_ok=True)
    for k, desc in enumerate(waiting):
        progress(k / len(waiting), desc.stem)
        glb = desc.with_suffix(".glb")
        try:
            info = json.loads(desc.read_text(encoding="utf-8"))
            loaded = trimesh.load(str(glb), force="mesh")
            if loaded is None or not len(getattr(loaded, "faces", [])):
                raise ValueError("no mesh in the file")
            shift = np.asarray(info.get("shift") or [0.0, 0.0, 0.0], float)
            world = from_gltf(np.asarray(loaded.vertices)) + shift
            m = measure(world, np.asarray(loaded.faces))
        except Exception as exc:  # noqa: BLE001 - the file stays in the inbox, the reason is said
            report["errors"][glb.name] = str(exc)
            continue
        name = str(info.get("name") or desc.stem)
        unit = unit_for(matrix, name, info.get("node_id"))
        fields = _fields(unit.name if unit is not None else name)
        tipo = unit.tipo if unit is not None else (fields.get("tipo") or "US")
        us = unit.us if unit is not None else (fields.get("us") or "")
        area = unit.area if unit is not None else (fields.get("area") or "")
        label = f"{tipo}{us}" if us else re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("._") or "proxy"
        path = folder / f"{label}.ply"
        # double precision: map coordinates in float32 keep half a metre at 4 500 km
        write_mesh_ply(path, m["vertices"], m["faces"], double=True)
        old = next((e for e in entries if e.get("name") == label), None)
        entry = dict(old or {})
        entry.update({"name": label, "us": us or None, "area": area or None, "tipo": tipo,
                      "sito": (unit.data.get("sito") if unit is not None else None) or entry.get("sito"),
                      "file": str(path.relative_to(chunk.data_dir)), "volume_m3": m["volume_m3"],
                      "area_m2": m["area_m2"], "z_top": m["z_top"], "z_bottom": m["z_bottom"], "base": "blender",
                      "closed": m["closed"], "blender_object": name})
        if old is not None:
            entries[entries.index(old)] = entry
            report["replaced"].append(label)
            if old.get("id_media"):
                report["republish"].append(label)
        else:
            entries.append(entry)
        report["taken"].append(label)
        if not m["closed"]:
            report["open"].append(label)
        for f in (glb, desc):
            f.unlink(missing_ok=True)
    chunk.meta["proxies"] = entries
    progress(1.0, "proxy")
    return report


__all__ = ["INBOX", "inbox", "pending", "close_holes", "measure", "unit_for", "take"]
