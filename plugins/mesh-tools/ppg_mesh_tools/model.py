"""MeshLab's filters on the model of a chunk, and the result as a new version of it.

The model (``model.ply``, internal frame of the chunk) is read with the program's own reader
(every vertex property, the texture coordinates of the face corners, the pages), the vertices it
repeats along the seams of its texture welded (MeshLab would take every seam for a border,
smoothing would open it; vertices that only touch at a point stay apart: a non-manifold vertex
split by MeshLab's repair stays split), and written for the worker with
only what MeshLab needs: positions, colours, and the texture coordinates when the operation keeps
them. MeshLab's result comes back the same way and gets the properties it cannot carry
(confidence, any other) from the nearest vertex of the model, or vertex by vertex when the
operation moved the vertices without renumbering them (smoothing, holes closed); colours are
MeshLab's (interpolated on the new vertices).

The texture stays valid after a decimation with the variant of MeshLab that keeps the texture
coordinates, after smoothing (the corners keep theirs) and after the repair (faces removed or
vertices split, each face with its own coordinates); remeshing and closing holes leave it out (new
faces have none): build it again. Closing holes leaves the outline of each separate part - its
longest border - open unless asked: MeshLab is given the faces along the other borders as its
selection (a face quality of 1 in the file it reads), else it would cap a whole excavated surface.
The model before is kept as a version (``keep_active`` first, put back if storing the new one
fails), the new one named after what MeshLab did; lengths are metres of the survey (units of the
model when the chunk has no scale).
"""
from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import runner
from .texts import REPAIR_STEPS, SMOOTHING, italian, length_decimals, number, plural, short, t, title

OPERATIONS = ("decimate", "remesh", "smooth", "close_holes", "repair")
TEMP_PREFIX = ".mesh-tools-"
STALE_HOURS = 6.0  # a work folder older than this was left by a run that died
DEFAULTS = {
    # quadric edge collapse: target_faces 0 = fraction of the faces; borders kept (open surfaces of a
    # survey), topology kept (no hole closed by the decimation); the texture kept when the model has one
    "decimate": {"target_faces": 0, "fraction": 0.5, "quality": 0.3, "preserve_boundary": True, "boundary_weight": 1.0,
                 "preserve_normal": False, "preserve_topology": True, "optimal_placement": True, "planar": False,
                 "texture": True, "texture_weight": 1.0},
    # isotropic remeshing: edge_length 0 = the mean edge of the model (regular triangles of the same size)
    "remesh": {"edge_length": 0.0, "iterations": 10, "adaptive": False, "feature_angle": 30.0, "max_distance": 0.0,
               "repair_first": True},
    # steps 0 = the method's: Taubin 10 (no shrinking), HC 1 (one pass shrinks a little, ten shrink)
    "smooth": {"method": "taubin", "steps": 0, "lambda": 0.5, "mu": -0.53, "keep_borders": True},
    # holes up to max_hole_edges edges; refine_edge > 0: the new faces split to that edge length; the
    # outline of each part (its longest border) stays open: closing it would cap a whole excavated surface
    "close_holes": {"max_hole_edges": 30, "self_intersection": True, "refine_edge": 0.0, "repair_first": True,
                    "keep_outline": True},
    "repair": {"duplicate_vertices": True, "duplicate_faces": True, "null_faces": True, "non_manifold_edges": True,
               "non_manifold_method": "remove", "non_manifold_vertices": True, "unreferenced_vertices": True},
}
SMOOTH_STEPS = {"taubin": 10, "hc": 1}
REPAIR_KEYS = ("duplicate_vertices", "duplicate_faces", "null_faces", "non_manifold_edges", "non_manifold_vertices",
               "unreferenced_vertices")  # the steps of the repair, in the order the worker runs them


# ------------------------------------------------------------------ the model of the chunk
def is_metric(chunk) -> bool:
    """The chunk is in metres: a reference system, a scale, or a metric source (a scan)."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    return (not crs_mod.is_local(chunk.crs) or abs(float(chunk.scale) - 1.0) > 1e-9
            or bool(chunk.meta.get("metric")))


def unit(chunk, stored: bool = False) -> str:
    """``m``, or the units of the model when the chunk has no scale (Italian in a stored name)."""
    if is_metric(chunk):
        return "m"
    return "unità" if stored else t("unità", "units")


def model_path(chunk) -> Path:
    path = chunk.product_path("model") if chunk is not None else None
    if path is None or not path.exists():
        raise RuntimeError(t("Il chunk non ha un modello: costruiscilo prima.", "The chunk has no model: build it first."))
    return path


def texture_pages(chunk) -> list[Path]:
    """The pages of the texture of the model (none for a texture of version 0.5, an OBJ apart)."""
    textured = chunk.product_path("model_textured")
    if textured is None or not textured.exists() or textured.suffix.lower() == ".obj":
        return []
    from pyarchinit_photogrammetry.engine.texturing import texture_pages as pages

    return pages(chunk)


def summary(chunk) -> dict:
    """Faces, vertices and texture pages of the model, read from the header of its file."""
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    path = model_path(chunk)
    try:
        ply = PlyMesh(path)
        faces, vertices = int(ply.n_faces), int(ply.n_vertices)
    except (ValueError, KeyError, OSError):  # not a binary PLY: read it all
        from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

        mesh = read_mesh(path)
        faces, vertices = len(mesh.faces), len(mesh.vertices)
    pages = texture_pages(chunk)
    from pyarchinit_photogrammetry.core import versions

    return {"faces": faces, "vertices": vertices, "pages": len(pages), "textured": bool(pages),
            "metric": is_metric(chunk), "unit": unit(chunk), "name": versions.active_info(chunk, "model")["name"]}


def mean_edge(chunk, sample: int = 200_000) -> float:
    """Mean edge of the model in metres (units of the model), on a sample of its faces."""
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh, read_mesh

    path = model_path(chunk)
    try:  # faces spread over the whole model, and only their vertices, read from the mapped file
        ply = PlyMesh(path)
        n = int(ply.n_faces)
        pick = np.unique(np.linspace(0, n - 1, min(sample, n)).astype(np.int64)) if n else np.zeros(0, np.int64)
        records = getattr(ply, "_faces", None)
        faces = (np.asarray(records[ply.index_field][pick], np.int64) if records is not None
                 else next(ply.face_chunks(sample), np.zeros((0, 3), np.int64)))
        used, inverse = np.unique(faces, return_inverse=True)
        xyz, faces = ply.xyz(used), inverse.reshape(faces.shape)
    except (ValueError, KeyError, OSError):
        mesh = read_mesh(path)
        faces, xyz = mesh.faces[:sample], mesh.xyz
    return _mean_edge(xyz, faces) * float(chunk.scale or 1.0)


def _mean_edge(xyz: np.ndarray, faces: np.ndarray) -> float:
    if not len(faces):
        return 0.0
    tri = xyz[faces]
    return float(np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2).mean())


def _welded(mesh):
    """The vertices repeated along the seams of a texture counted once, faces that become degenerate
    dropped; the number of vertices welded. A seam is a pair of border edges between the same two
    places: only the vertices along one are joined. Vertices at the same place that touch only at a
    point stay apart - the copies of a non-manifold vertex that MeshLab's repair split, two pieces
    meeting at a corner: joined, the repair would be undone by the next operation."""
    from pyarchinit_photogrammetry.engine.cleaning import weld
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    first, place = weld(mesh.xyz)
    n = len(mesh.vertices)
    if len(first) == n or not len(mesh.faces):
        return mesh, 0
    f = np.asarray(mesh.faces, np.int64)
    edges, count = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0,
                             return_counts=True)
    border = edges[count == 1]
    places = np.sort(place[border], axis=1)
    _, again, times = np.unique(places, axis=0, return_inverse=True, return_counts=True)
    on_seam = np.zeros(n, bool)
    on_seam[border[times[again.ravel()] >= 2].ravel()] = True
    label = np.where(on_seam, place, len(first) + np.arange(n))  # off a seam a vertex keeps its own label
    kept, keep, index = np.unique(label, return_index=True, return_inverse=True)
    if len(kept) == n:
        return mesh, 0
    g = index.ravel()[f]
    ok = (g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])
    return (MeshData(mesh.vertices[keep], g[ok], None if mesh.texcoords is None else mesh.texcoords[ok],
                     list(mesh.comments), None if mesh.texnumber is None else mesh.texnumber[ok]),
            int(n - len(kept)))


def _input_mesh(mesh, texture: bool, colors: bool):
    """What MeshLab reads: positions, colours, and the texture coordinates of the corners with the
    pages (their names only: the images are never read)."""
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    v = mesh.vertices
    fields = [(c, v.dtype[c]) for c in "xyz"]
    if colors:
        fields += [(c, np.uint8) for c in ("red", "green", "blue")]
    out = np.empty(len(v), fields)
    for name, _dtype in fields:
        out[name] = v[name]
    if not texture:
        return MeshData(out, mesh.faces)
    names = mesh.texture_files or ["texture.jpg"]
    if mesh.texnumber is not None and len(mesh.texnumber):
        names = names + [f"page_{k}.jpg" for k in range(len(names), int(mesh.texnumber.max()) + 1)]
    return MeshData(out, mesh.faces, mesh.texcoords, [f"TextureFile {n}" for n in names], mesh.texnumber)


def hole_faces(mesh) -> tuple[np.ndarray, int, int]:
    """The faces along the holes of ``mesh`` - every border loop but the longest of each separate
    part, its outline - as a mask; the number of holes and of outlines."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(mesh.faces, np.int64)
    n = len(mesh.vertices)
    mask = np.zeros(len(f), bool)
    if not len(f):
        return mask, 0, 0
    key = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    _, inverse, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = count[inverse.ravel()] == 1
    if not border.any():
        return mask, 0, 0
    edges, faces_on = key[border], owner[border]
    ones = np.ones(len(edges), np.int8)
    loop = connected_components(coo_matrix((ones, (edges[:, 0], edges[:, 1])), shape=(n, n)), directed=False)[1][edges[:, 0]]
    rows = np.concatenate([f[:, 0], f[:, 1]])
    cols = np.concatenate([f[:, 1], f[:, 2]])
    part_of = connected_components(coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n)), directed=False)[1]
    loops, first = np.unique(loop, return_index=True)
    size = np.bincount(loop)[loops]
    part = part_of[edges[first, 0]]
    outline = {}
    for lp, sz, pt in zip(loops, size, part):  # the longest loop of each part
        if pt not in outline or sz > outline[pt][1]:
            outline[pt] = (lp, sz)
    keep = np.array([o[0] for o in outline.values()])
    holes = np.setdiff1d(loops, keep)
    mask[faces_on[np.isin(loop, holes)]] = True
    return mask, int(len(holes)), int(len(keep))


def _write_input(path: Path, mesh, face_quality: np.ndarray | None = None) -> Path:
    """The file MeshLab reads: the PLY of :func:`write_mesh` with, when given, a face quality (1 on
    the faces of the holes to close) that the worker turns into MeshLab's selection."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh

    if face_quality is None:
        return write_mesh(path, mesh)
    v = mesh.vertices
    names = {"f4": "float", "f8": "double", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in mesh.comments]
    header += [f"element vertex {len(v)}"] + [f"property {names[np.dtype(v.dtype[n]).str[1:]]} {n}" for n in v.dtype.names]
    header += [f"element face {len(mesh.faces)}", "property list uchar int vertex_indices"]
    fields = [("n", "u1"), ("v", "<i4", (3,))]
    if mesh.texcoords is not None:
        header.append("property list uchar float texcoord")
        fields += [("nt", "u1"), ("t", "<f4", (6,))]
        if mesh.texnumber is not None:
            header.append("property int texnumber")
            fields.append(("page", "<i4"))
    header += ["property float quality", "end_header"]
    fields.append(("q", "<f4"))
    out = np.empty(len(mesh.faces), fields)
    out["n"], out["v"] = 3, mesh.faces
    if mesh.texcoords is not None:
        out["nt"], out["t"] = 6, mesh.texcoords
        if mesh.texnumber is not None:
            out["page"] = mesh.texnumber
    out["q"] = face_quality
    vout = np.empty(len(v), [(n, np.dtype(v.dtype[n]).newbyteorder("<")) for n in v.dtype.names])
    for n in v.dtype.names:
        vout[n] = v[n]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("utf-8"))
        f.write(vout.tobytes())
        f.write(out.tobytes())
    return Path(path)


def _boundary(faces: np.ndarray, n: int) -> np.ndarray:
    """True for the vertices on an open border (edges of one face only)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    out = np.zeros(n, bool)
    out[uniq[count == 1].ravel()] = True
    return out


def _vertex_normals(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = xyz[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norm > 0, norm, 1.0)


def _rebuild(base, out, same_order: bool, texture: bool, fixed: np.ndarray | None = None):
    """The model from MeshLab's result: every vertex property of ``base`` (by index when
    ``same_order``, else from the nearest vertex), MeshLab's positions (``fixed`` vertices where
    they were) and colours, recomputed normals, the texture coordinates when ``texture``."""
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    if same_order:
        verts = base.vertices.copy()
    else:
        from scipy.spatial import cKDTree

        idx = cKDTree(base.xyz).query(out.xyz, workers=-1)[1]
        verts = base.vertices[idx].copy()
    xyz = out.xyz
    if fixed is not None and fixed.any():
        xyz = xyz.copy()
        xyz[fixed] = base.xyz[fixed]
    for k, c in enumerate("xyz"):
        verts[c] = xyz[:, k]
    names = set(verts.dtype.names or ())
    colours = out.colors
    if not same_order and colours is not None and {"red", "green", "blue"} <= names:
        for k, c in enumerate(("red", "green", "blue")):
            verts[c] = colours[:, k]
    faces = np.asarray(out.faces, np.int64)
    if {"nx", "ny", "nz"} <= names:
        normals = _vertex_normals(xyz, faces)
        for k, c in enumerate(("nx", "ny", "nz")):
            verts[c] = normals[:, k]
    tex = page = None
    if texture:
        if same_order and np.array_equal(faces, base.faces):
            tex, page = base.texcoords, base.texnumber
        else:
            tex = out.texcoords
            page = out.texnumber if base.texnumber is not None else None
    return MeshData(verts, faces, tex, list(base.comments), page)


def _texture_survives(base, out, op: str, sent: bool, same_order: bool) -> bool:
    """Whether the texture coordinates of the result are those of a valid texture."""
    if base.texcoords is None:
        return False
    if op == "smooth":
        return same_order and np.array_equal(np.asarray(out.faces, np.int64), base.faces)
    if not sent or out.texcoords is None or len(out.texcoords) != len(out.faces):
        return False
    if base.texnumber is not None:
        pages = len(base.texture_files) or 1
        return out.texnumber is not None and (not len(out.texnumber) or int(out.texnumber.max()) < pages)
    return True


def _work_folder(chunk) -> Path:
    """A folder for the files of the worker, beside the model (same disk; removed afterwards);
    those left by a run that died are removed first."""
    now = time.time()
    for old in Path(chunk.data_dir).glob(TEMP_PREFIX + "*"):
        try:
            if now - old.stat().st_mtime > STALE_HOURS * 3600:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=chunk.data_dir)).absolute()


# ------------------------------------------------------------------ the operations
def resolve(op: str, params: dict | None, info: dict) -> tuple[dict, dict]:
    """The parameters of ``op`` with the defaults (metres of the survey), and the operation for the
    worker (lengths in the units of the model). ``info``: faces, scale, textured, mean_edge of the
    model. ValueError for a parameter out of range."""
    if op not in OPERATIONS:
        raise ValueError(t("Operazione sconosciuta {0!r}: {1}", "Unknown operation {0!r}: {1}").format(
            op, ", ".join(OPERATIONS)))
    p = dict(DEFAULTS[op])
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in p})
    s = float(info.get("scale") or 1.0)
    if op == "decimate":
        faces = int(info["faces"])
        target = int(p["target_faces"])
        if target <= 0:
            fraction = float(p["fraction"])
            if not 0.0 < fraction < 1.0:
                raise ValueError(t("La frazione delle facce va tra 0 e 1 (esclusi).",
                                   "The fraction of the faces goes between 0 and 1 (excluded)."))
            target = int(round(faces * fraction))
        if target < 4:
            raise ValueError(t("Troppo poche facce: {0}.", "Too few faces: {0}.").format(target))
        if target >= faces:
            raise ValueError(t("Il modello ha già {0} facce: niente da decimare a {1}.",
                               "The model already has {0} faces: nothing to decimate to {1}.").format(
                number(faces), number(target)))
        p["target_faces"] = target
        p["texture"] = bool(p["texture"]) and bool(info.get("textured"))
        w = {"op": op, "target_faces": target, "quality": float(p["quality"]),
             "preserve_boundary": bool(p["preserve_boundary"]), "boundary_weight": float(p["boundary_weight"]),
             "preserve_normal": bool(p["preserve_normal"]), "preserve_topology": bool(p["preserve_topology"]),
             "optimal_placement": bool(p["optimal_placement"]), "planar": bool(p["planar"]), "texture": p["texture"],
             "texture_weight": float(p["texture_weight"])}
    elif op == "remesh":
        edge = float(p["edge_length"]) or float(info.get("mean_edge") or 0.0)
        if edge <= 0:
            raise ValueError(t("Il lato dei triangoli deve essere maggiore di zero.",
                               "The edge of the triangles must be greater than zero."))
        p["edge_length"] = edge
        if int(p["iterations"]) < 1:
            raise ValueError(t("Almeno un'iterazione.", "At least one iteration."))
        w = {"op": op, "edge_length": edge / s, "iterations": int(p["iterations"]), "adaptive": bool(p["adaptive"]),
             "feature_angle": float(p["feature_angle"]), "max_distance": float(p["max_distance"] or 0.0) / s,
             "repair_first": bool(p["repair_first"])}
    elif op == "smooth":
        method = str(p["method"])
        if method not in SMOOTH_STEPS:
            raise ValueError(t("Levigatura {0!r}: taubin o hc.", "Smoothing {0!r}: taubin or hc.").format(method))
        p["steps"] = int(p["steps"]) or SMOOTH_STEPS[method]
        if p["steps"] < 1:
            raise ValueError(t("Almeno un passo.", "At least one step."))
        w = {"op": op, "method": method, "steps": p["steps"], "lambda": float(p["lambda"]), "mu": float(p["mu"])}
    elif op == "close_holes":
        if int(p["max_hole_edges"]) < 3:
            raise ValueError(t("Un buco ha almeno 3 lati.", "A hole has at least 3 edges."))
        w = {"op": op, "max_hole_edges": int(p["max_hole_edges"]), "self_intersection": bool(p["self_intersection"]),
             "refine_edge": float(p["refine_edge"] or 0.0) / s, "repair_first": bool(p["repair_first"]),
             "select_marked": bool(p["keep_outline"])}
    else:
        if p["non_manifold_method"] not in ("remove", "split"):
            raise ValueError(t("Spigoli non-manifold: remove o split.", "Non-manifold edges: remove or split."))
        w = {"op": op, **{k: bool(p[k]) for k in REPAIR_KEYS}, "non_manifold_method": p["non_manifold_method"]}
    return p, w


def version_name(op: str, p: dict, chunk=None) -> str:
    """The name of the new version of the model (Italian: stored in the project, as the program's)."""
    if op == "decimate":
        return f"MeshLab: decimazione a {italian(p['target_faces'])} facce" + (" (texture conservata)" if p.get("texture")
                                                                             else "")
    if op == "remesh":
        u = unit(chunk, stored=True) if chunk is not None else "m"
        return f"MeshLab: rimaglia isotropa, lato {short(p['edge_length'], stored=True)} {u}"
    if op == "smooth":
        steps = int(p["steps"])
        return f"MeshLab: levigatura {SMOOTHING[p['method']][0]}, {steps} {'passo' if steps == 1 else 'passi'}"
    if op == "close_holes":
        return (f"MeshLab: buchi chiusi fino a {italian(int(p['max_hole_edges']))} lati"
                + ("" if p.get("keep_outline", True) else ", contorno compreso"))
    return "MeshLab: topologia riparata"


def _changed(op: str, step: dict, welded: int) -> bool:
    if op in ("decimate", "remesh", "smooth"):
        return True
    counts = (step["vertices"], step["faces"]) != (step["vertices_before"], step["faces_before"])
    details = step.get("details") or {}
    if op == "close_holes":
        repaired = details.get("repaired") or {}
        return counts or int(details.get("new_faces", 0)) > 0 or any(
            v.get("vertices") or v.get("faces") for v in repaired.values())
    return counts or welded > 0 or any(v.get("vertices") or v.get("faces") for v in details.values()
                                       if isinstance(v, dict))


def world_measures(measures: dict | None, chunk, xyz: np.ndarray) -> dict | None:
    """MeshLab's measures (units of the model file) in metres of the survey; the bounding box from
    the vertices in the coordinates of the chunk."""
    if not measures:
        return None
    s = float(chunk.scale or 1.0)
    out = dict(measures)
    out["area"] = float(measures.get("area") or 0.0) * s * s
    volume = measures.get("volume")
    out["volume"] = None if volume is None else abs(float(volume)) * s ** 3
    out["inverted"] = volume is not None and float(volume) < 0
    out["mean_edge"] = float(measures.get("mean_edge") or 0.0) * s
    if len(xyz):
        world = chunk.to_world(np.asarray(xyz, np.float64))
        out["bbox_min"], out["bbox_max"] = world.min(axis=0).tolist(), world.max(axis=0).tolist()
        out["size"] = (world.max(axis=0) - world.min(axis=0)).tolist()
    out["unit"] = unit(chunk)
    return out


def measure(chunk, progress=None) -> dict:
    """Area, volume (closed models), edges and topology of the model in metres; the model does not
    change (the vertices at the seams of the texture are counted once, as the surface is)."""
    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh, write_mesh

    progress = as_progress(progress)
    path = model_path(chunk)
    progress(0.0, t("Lettura del modello", "Reading the model"))
    base, welded = _welded(read_mesh(path))
    folder = _work_folder(chunk)
    try:
        write_mesh(folder / "in.ply", _input_mesh(base, texture=False, colors=False))
        job = {"protocol": 1, "input": str(folder / "in.ply"), "output": None, "operations": [], "measure": True,
               "texture": False, "colors": False}
        result = runner.run(job, folder, progress.sub(0.05, 0.9))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    out = world_measures(result.get("measures"), chunk, base.xyz) or {}
    out["welded"] = welded
    out["pymeshlab"] = result.get("version", "")
    progress(1.0, t("Misure del modello pronte", "Measures of the model ready"))
    return out


def apply(chunk, op: str, params: dict | None = None, progress=None, name: str | None = None,
          keep_version: bool = True, measure_after: bool = True) -> dict:
    """``op`` (decimate, remesh, smooth, close_holes, repair) on the model of ``chunk`` with
    ``params`` (see :data:`DEFAULTS`; lengths in metres), stored as the active model: the one before
    kept as a version (``keep_version``), the new one named ``name`` or after the operation. When
    MeshLab changes nothing (no hole that small, nothing to repair) the model stays as it is
    (``changed`` False). Returns a report with the counts, what MeshLab did, the measures of the
    result (``measure_after``)."""
    from pyarchinit_photogrammetry.core import versions
    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.engine.cleaning import store_model
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    progress = as_progress(progress)
    path = model_path(chunk)
    t0 = time.monotonic()
    progress(0.0, t("Lettura del modello", "Reading the model"))
    original = read_mesh(path)
    textured = original.texcoords is not None and bool(texture_pages(chunk))
    base, welded = _welded(original)
    info = {"faces": len(base.faces), "vertices": len(base.vertices), "scale": float(chunk.scale or 1.0),
            "textured": textured, "mean_edge": _mean_edge(base.xyz, base.faces[:200_000]) * float(chunk.scale or 1.0)
            if op == "remesh" else None}
    p, w = resolve(op, params, info)
    send_texture = textured and (op == "repair" or (op == "decimate" and bool(w["texture"])))
    colors = base.colors is not None
    marks = None
    if op == "close_holes" and w.get("select_marked"):  # the outlines stay open: only the holes are offered
        marks, holes, _outlines = hole_faces(base)
        if not holes:
            report = _unchanged(op, p, original, welded, textured, chunk)
            report["details"] = {"closed_holes": 0, "new_faces": 0, "holes": 0}
            report["seconds"] = round(time.monotonic() - t0, 2)
            progress(1.0, report_text(report))
            return report
    folder = _work_folder(chunk)
    try:
        progress(0.02, t("Preparazione del modello per MeshLab", "Preparing the model for MeshLab"))
        _write_input(folder / "in.ply", _input_mesh(base, send_texture, colors),
                     None if marks is None else marks.astype(np.float32))
        job = {"protocol": 1, "input": str(folder / "in.ply"), "output": str(folder / "out.ply"), "operations": [w],
               "measure": bool(measure_after), "texture": send_texture, "colors": colors}
        result = runner.run(job, folder, progress.sub(0.05, 0.85))
        progress(0.9, t("Lettura del risultato di MeshLab", "Reading MeshLab's result"))
        out = read_mesh(folder / "out.ply")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    step = result["steps"][0]
    report = _unchanged(op, p, original, welded, textured, chunk)
    report.update(changed=_changed(op, step, welded), details=step.get("details") or {},
                  pymeshlab=result.get("version", ""))
    if not report["changed"]:
        report["measures"] = world_measures(result.get("measures"), chunk, base.xyz)
        report["seconds"] = round(time.monotonic() - t0, 2)
        progress(1.0, report_text(report))
        return report
    same_order = op in ("smooth", "close_holes") and len(out.vertices) == len(base.vertices)
    if same_order and op == "close_holes":  # the holes closed with the vertices there were, none moved
        same_order = bool(np.allclose(out.xyz, base.xyz, rtol=0.0, atol=1e-6 * max(1.0, float(np.abs(base.xyz).max()))))
    fixed = _boundary(base.faces, len(base.vertices)) if op == "smooth" and p["keep_borders"] and same_order else None
    keep_texture = textured and _texture_survives(base, out, op, send_texture, same_order)
    new = _rebuild(base, out, same_order, keep_texture, fixed)
    if op in ("decimate", "remesh"):  # vertices no face uses any more (MeshLab keeps those it was given)
        new = new.subset_faces(np.ones(len(new.faces), bool))
    progress(0.95, t("Salvataggio del modello", "Saving the model"))
    kept = versions.keep_active(chunk, "model", move=False) if keep_version else None
    name = (name or "").strip() or version_name(op, p, chunk)
    try:
        store_model(chunk, new, texture_kept=keep_texture)
        meta = chunk.meta.setdefault("model", {})
        # a new list, not the old one extended: the version kept shares the old one in its parameters
        meta["meshlab"] = list(meta.get("meshlab") or []) + [{
            "operation": op, "params": dict(p), "faces_before": report["faces_before"], "faces": int(len(new.faces)),
            "texture_kept": bool(keep_texture), "pymeshlab": result.get("version", ""),
            "when": datetime.now().isoformat(timespec="seconds")}]
        versions.set_active_name(chunk, "model", name)
    except BaseException:
        if kept is not None:  # the model before comes back where it was, as if nothing happened
            versions.restore(chunk, "model", kept)
        raise
    report.update(faces=int(len(new.faces)), vertices=int(len(new.vertices)), texture_kept=bool(keep_texture),
                  measures=world_measures(result.get("measures"), chunk, new.xyz), version=kept["id"] if kept else None,
                  name=name, seconds=round(time.monotonic() - t0, 2))
    progress(1.0, report_text(report))
    return report


# ------------------------------------------------------------------ texts of the results
def _unchanged(op: str, p: dict, original, welded: int, textured: bool, chunk) -> dict:
    """The report of an operation that left the model as it was (the start of every report)."""
    return {"operation": op, "params": p, "changed": False, "faces_before": int(len(original.faces)),
            "vertices_before": int(len(original.vertices)), "faces": int(len(original.faces)),
            "vertices": int(len(original.vertices)), "welded": welded, "textured": textured, "texture_kept": textured,
            "details": {}, "measures": None, "version": None, "name": None, "pymeshlab": "", "unit": unit(chunk)}


def report_text(report: dict) -> str:
    """What an operation did, in a sentence (in the language of the interface)."""
    op, d, p = report["operation"], report.get("details") or {}, report.get("params") or {}
    before, after = number(report["faces_before"]), number(report["faces"])
    if op == "decimate":
        text = t("Modello decimato: {0} → {1} facce", "Model decimated: {0} → {1} faces").format(before, after)
    elif op == "remesh":
        edge = float(p.get("edge_length", 0.0))
        text = t("Modello rimagliato (lato {0} {1}): {2} → {3} facce", "Model remeshed (edge {0} {1}): {2} → {3} faces").format(
            short(edge), report.get("unit", "m"), before, after)
    elif op == "smooth":
        steps = int(p.get("steps", 0))
        text = t("Modello levigato ({0}, {1} {2})", "Model smoothed ({0}, {1} {2})").format(
            t(*SMOOTHING.get(p.get("method"), ("", ""))), steps,
            plural(steps, t("passo", "step"), t("passi", "steps")))
    elif op == "close_holes":
        if not report["changed"]:
            text = t("Nessun buco fino a {0} lati da chiudere: il modello non è cambiato.",
                     "No hole up to {0} edges to close: the model has not changed.").format(p.get("max_hole_edges", ""))
        else:
            n, nf = int(d.get("closed_holes", 0)), int(d.get("new_faces", 0))
            text = (t("Chiuso {0} buco", "{0} hole closed") if n == 1 else t("Chiusi {0} buchi", "{0} holes closed")).format(
                number(n)) + t(" con {0} {1}", " with {0} {1}").format(number(nf), plural(nf, t("faccia", "face"),
                                                                                       t("facce", "faces")))
    elif op == "repair":
        if not report["changed"]:
            text = t("Nessun difetto di topologia: il modello non è cambiato.",
                     "No topology defect: the model has not changed.")
        else:
            parts = []
            for step, c in d.items():
                if step not in REPAIR_STEPS or not isinstance(c, dict):
                    continue
                faces, vertices = abs(int(c.get("faces", 0))), abs(int(c.get("vertices", 0)))
                if step == "non_manifold_edges" and not faces and vertices:
                    step = "non_manifold_edges_split"
                n = faces if step in ("duplicate_faces", "null_faces", "non_manifold_edges") else vertices
                if n:
                    parts.append(f"{t(*REPAIR_STEPS[step])} {number(n)}")
            if report.get("welded"):
                parts.append(t("vertici uniti alle cuciture {0}", "vertices welded at the seams {0}").format(
                    number(report["welded"])))
            text = t("Topologia riparata: {0}", "Topology repaired: {0}").format(", ".join(parts) or "-")
    else:
        text = title(op)
    m = report.get("measures") or {}
    vertices, edges = max(0, int(m.get("non_manifold_vertices") or 0)), max(0, int(m.get("non_manifold_edges") or 0))
    if vertices or edges:  # what MeshLab left (the remeshing can join the vertices a repair split)
        text += t("; restano {0} {1} e {2} {3} non-manifold («Ripara la topologia»)",
                  "; {0} non-manifold {1} and {2} {3} left ('Repair the topology')").format(
            number(vertices), plural(vertices, t("vertice", "vertex"), t("vertici", "vertices")),
            number(edges), plural(edges, t("spigolo", "edge"), t("spigoli", "edges")))
    if report.get("changed") and report.get("textured") and not report.get("texture_kept"):
        text += t("; la texture va ricostruita (Flusso di lavoro > Costruisci texture)",
                  "; the texture has to be built again (Workflow > Build texture)")
    elif report.get("changed") and report.get("texture_kept"):
        text += t("; texture conservata", "; texture kept")
    return text


def measure_rows(m: dict) -> list[tuple[str, str]]:
    """The measures as rows (name, value) in the language of the interface."""
    if not m:
        return []
    u = m.get("unit", "m")
    yes, no = t("sì", "yes"), t("no", "no")

    def count(key):
        v = m.get(key)
        return "–" if v is None or (isinstance(v, (int, float)) and v < 0) else number(v)

    rows = [(t("Facce", "Faces"), count("faces")), (t("Vertici", "Vertices"), count("vertices")),
            (t("Area della superficie", "Surface area"), f"{number(m.get('area', 0.0), 3)} {u}²")]
    if m.get("volume") is not None:
        vol = f"{number(m['volume'], 3)} {u}³"
        if m.get("inverted"):
            vol += t(" (normali verso l'interno)", " (normals pointing inwards)")
        rows.append((t("Volume", "Volume"), vol))
    else:
        rows.append((t("Volume", "Volume"), t("– (il modello non è chiuso)", "– (the model is not closed)")))
    rows += [(t("Chiuso (watertight)", "Closed (watertight)"), yes if m.get("watertight") else no),
             (t("2-manifold", "2-manifold"), yes if m.get("two_manifold") else no),
             (t("Parti separate", "Separate parts"), count("components")),
             (t("Bordi (anelli, il contorno compreso)", "Borders (loops, the outline included)"), count("boundary_loops")),
             (t("Spigoli di bordo", "Border edges"), count("boundary_edges")),
             (t("Spigoli non-manifold", "Non-manifold edges"), count("non_manifold_edges")),
             (t("Vertici non-manifold", "Non-manifold vertices"), count("non_manifold_vertices")),
             (t("Vertici non usati", "Unused vertices"), count("unreferenced_vertices"))]
    if m.get("size"):
        d = max(length_decimals(max(m["size"]) or 1.0), 2)
        rows.append((t("Dimensioni (x × y × z)", "Size (x × y × z)"),
                     " × ".join(number(v, d) for v in m["size"]) + f" {u}"))
    edge = float(m.get("mean_edge") or 0.0)
    rows.append((t("Lato medio dei triangoli", "Mean edge of the triangles"), f"{short(edge)} {u}"))
    return rows


def measures_text(m: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in measure_rows(m))


__all__ = ["DEFAULTS", "OPERATIONS", "apply", "hole_faces", "measure", "measure_rows", "measures_text", "mean_edge", "model_path",
           "report_text", "resolve", "summary", "texture_pages", "unit", "version_name", "world_measures", "is_metric"]
