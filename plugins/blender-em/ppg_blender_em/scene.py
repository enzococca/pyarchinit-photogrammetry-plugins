"""The EM project of the survey: the proxies of the units and the model, as EMtools reads them.

The standard tree of an EM project (``s3dgraphy.project_tree``, the one EMtools 1.6 writes)::

    <project>/EM/<site>.em.json        the matrix: PyArchInit's, EMStudio's - never written here
    <project>/EM/proxies/<unit>.glb    a proxy per unit, named as its node: EMtools pairs by name
    <project>/RB/<chunk>/<chunk>.glb   the photogrammetric survey (reality-based)
    <project>/ppg_scene.json           what the plugin wrote: read by the live script in Blender

With an em.json outside a tree the folders sit beside it (``proxies/``, ``RB/``), as EMtools does.

glTF is Y-up (x east, y up, z south), and Blender turns it Z-up when it imports it. The
coordinates are shifted by whole metres, since a float32 keeps a quarter of a metre at four
thousand kilometres: by the shift of the em.json (its ``geo_position``) when it has one, else by the
lowest corner of what is written, rounded down; once chosen the shift stays (the manifest keeps
it), so an update never moves what Blender already shows.

A file is written again only when what it comes from changed (the PLY of the proxy, the model of
the chunk, the shift, the name, the colour), and never over a file of the same name that another
program wrote or changed since (EMtools exports the proxies modelled in Blender into the same
folder): that is reported and left alone.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .emjson import Matrix, match

MANIFEST = "ppg_scene.json"
FORMAT = "ppg-blender-em"
FORMAT_VERSION = 1
# the colours of the proxies when the matrix gives no epoch colour (as the IFC export of the program)
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


@dataclass
class Folders:
    base: Path  # where the manifest goes: the project, or the folder of the em.json
    proxies: Path
    survey: Path
    standard: bool  # the standard tree of an EM project


def find_project_root(path: str | Path) -> Path | None:
    """The EM project ``path`` belongs to: the nearest folder (itself or above) with an ``EM``
    folder, or whose own name is ``EM`` (then its parent); None when there is none."""
    p = Path(path).resolve()
    p = p if p.is_dir() else p.parent
    for folder in (p, *p.parents):
        if folder.name == "EM":
            return folder.parent
        if (folder / "EM").is_dir():
            return folder
    return None


def project_folders(root: str | Path | None = None, emjson: str | Path | None = None) -> Folders:
    """Where the files go: the standard tree of ``root`` (created when missing), else the project of
    the em.json, else beside the em.json."""
    if root is None and emjson is not None:
        root = find_project_root(emjson)
        if root is None:
            here = Path(emjson).resolve().parent
            return Folders(here, here / "proxies", here / "RB", False)
    if root is None:
        raise ValueError("an EM project folder or an em.json is needed")
    root = Path(root).resolve()
    return Folders(root, root / "EM" / "proxies", root / "RB", True)


def safe_name(name: str) -> str | None:
    """``name`` as a file name, None when it cannot be one without changing it (a separator, a
    character the file system refuses): EMtools names the file of a proxy after its unit."""
    text = str(name or "").strip()
    return None if not text or _UNSAFE.search(text) or text in (".", "..") else text


def _folder_name(label: str) -> str:
    return _UNSAFE.sub("_", str(label or "")).strip(" ._") or "chunk"


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stat(path: Path) -> list:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _source(path: Path | None) -> list:
    """What a file comes from, cheaply: its path, size and modification time."""
    if path is None or not Path(path).exists():
        return []
    st = Path(path).stat()
    return [str(Path(path).resolve()), st.st_size, st.st_mtime_ns]


def to_gltf(xyz: np.ndarray) -> np.ndarray:
    """Z-up coordinates (x east, y north, z up) as glTF wants them (x east, y up, z south)."""
    xyz = np.asarray(xyz, float)
    return np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])


def from_gltf(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, float)
    return np.column_stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]])


def _rgba(color: str | None) -> list[int]:
    text = str(color or "").lstrip("#")
    try:
        return [int(text[i:i + 2], 16) for i in (0, 2, 4)] + [255]
    except ValueError:
        return [200, 200, 200, 255]


def _replace(tmp: Path, path: Path) -> None:
    """``tmp`` in place of ``path`` in one step: Blender never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def linear_colours(rgba) -> np.ndarray:
    """Vertex colours as glTF wants them, linear: the colours of a photo are sRGB, and a viewer that
    follows the standard (Blender does) shows sRGB values written as they are paler than they are."""
    c = np.asarray(rgba, float)[:, :3] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    out = np.asarray(rgba).copy()
    out[:, :3] = np.clip(np.round(lin * 255.0), 0, 255).astype(out.dtype)
    return out


def write_proxy_glb(path: Path, name: str, vertices, faces, shift, color: str | None = None,
                    extras: dict | None = None) -> Path:
    """One proxy as a glTF binary: a node and a mesh named ``name``, a material of ``color``, the
    world ``vertices`` less ``shift``, Y-up."""
    import trimesh

    local = np.asarray(vertices, float) - np.asarray(shift, float)
    mesh = trimesh.Trimesh(to_gltf(local), np.asarray(faces, np.int64), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=_rgba(color), metallicFactor=0.0,
                                                     roughnessFactor=0.8, doubleSided=True))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.metadata = {"ppg": dict(extras or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(scene.export(file_type="glb"))
    _replace(tmp, path)
    return path


def write_survey_glb(chunk, path: Path, name: str, shift, extras: dict | None = None) -> Path:
    """The model of ``chunk`` as a glTF binary with its colours or its texture, shifted. The program
    writes the GLB (``chunk.export_model``) around the centre of the model; here its node takes the
    rest of the way to the shift."""
    import trimesh

    with tempfile.TemporaryDirectory(prefix="ppg-blender-em-") as tmp:
        first = Path(chunk.export_model(Path(tmp) / "model.glb"))
        loaded = trimesh.load(str(first), force="scene")
        centre = np.asarray(((loaded.metadata or {}).get("extras") or {}).get("origin") or (0.0, 0.0, 0.0), float)
        move = np.eye(4)
        move[:3, 3] = to_gltf((centre - np.asarray(shift, float))[None])[0]
        out = trimesh.Scene()
        for k, node in enumerate(loaded.graph.nodes_geometry):
            transform, geom_name = loaded.graph[node]
            label = name if k == 0 else f"{name}.{k:03d}"
            geom = loaded.geometry[geom_name]
            if getattr(geom.visual, "kind", None) == "vertex":
                geom.visual.vertex_colors = linear_colours(geom.visual.vertex_colors)
            out.add_geometry(geom, node_name=label, geom_name=label, transform=move @ transform)
        out.metadata = {"ppg": dict(extras or {})}
        data = out.export(file_type="glb")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(data)
    _replace(tmp_path, path)
    return path


def read_manifest(path: str | Path) -> dict | None:
    """The manifest at ``path`` (a file or the folder holding it), None when there is none."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("format") == FORMAT else None


def _proxy_meshes(chunk) -> list[tuple[dict, Path]]:
    out = []
    for e in (chunk.meta or {}).get("proxies") or []:
        f = chunk.data_dir / str(e.get("file") or "")
        if e.get("file") and f.exists():
            out.append((e, f))
    return out


def _model_path(chunk) -> Path | None:
    p = chunk.product_path("model")
    return p if p is not None and p.exists() else None


def _shift_for(chunk, proxies, matrix: Matrix | None, old: dict | None, survey: bool) -> list[float]:
    if matrix is not None and matrix.shift is not None:
        return [float(v) for v in matrix.shift]
    if old and old.get("shift") is not None:
        return [float(v) for v in old["shift"]]
    import trimesh

    lows = []
    for _e, f in proxies:
        lows.append(np.asarray(trimesh.load(str(f), process=False, force="mesh").vertices).min(0))
    if survey and _model_path(chunk) is not None:
        model = chunk.model
        if model is not None and len(model.vertices):
            lows.append(chunk.to_world(np.asarray(model.vertices)).min(0))
    if not lows:
        return [0.0, 0.0, 0.0]
    return [float(math.floor(v)) for v in np.min(lows, axis=0)]


def epsg_of(crs: str | None) -> str | None:
    m = re.match(r"^\s*EPSG:(\d+)\s*$", str(crs or ""), re.I)
    return m.group(1) if m else None


def export_project(chunk, folders: Folders, matrix: Matrix | None = None, survey: bool = True,
                   node_uuids: dict | None = None, progress=None, cameras: bool | None = None,
                   max_side: int = 2048) -> dict:
    """The proxies of ``chunk`` (``chunk.meta["proxies"]``) and its model into ``folders``, and the
    manifest. ``cameras``: the aligned cameras with their undistorted photos (:mod:`.cameras`, the
    photos ``max_side`` pixels on the long side) when True, none when False, those of the manifest
    before when None (the live update does not undistort photos again). Returns the report:
    ``written``, ``kept`` (unchanged), ``foreign`` (a file of the same name another program wrote or
    changed: left alone), ``removed`` (written here before, gone from the chunk), ``not_in_matrix``,
    ``ambiguous`` (``{proxy: candidates}``), ``unnamed`` (a unit name that cannot be a file name),
    ``cameras`` (``{"written", "kept", "missing", "removed"}``), ``manifest``, ``shift``."""
    import trimesh

    progress = progress or (lambda *a: None)
    old = read_manifest(folders.base) or {}
    previous = {o.get("file"): o for o in old.get("objects") or [] if o.get("file")}
    proxies = _proxy_meshes(chunk)
    shift = _shift_for(chunk, proxies, matrix, old, survey)
    if [float(v) for v in old.get("shift") or []] != shift:
        previous = {k: {**v, "source": None} for k, v in previous.items()}  # another shift: everything again
    report = {"written": [], "kept": [], "foreign": [], "removed": [], "not_in_matrix": [], "ambiguous": {},
              "unnamed": [], "shift": shift}
    objects = []

    def place(rel: str, record: dict, write) -> None:
        """Write the file ``rel`` unless it is unchanged or another program's."""
        path = folders.base / rel
        before = previous.get(rel)
        # the file as written here: the same size and time (cheap: a survey can be large), else the same bytes
        mine = before is not None and path.exists() and (before.get("stat") == _stat(path)
                                                          or before.get("sha1") == sha1(path))
        if path.exists() and not mine:
            report["foreign"].append(record["name"])
            return
        if mine and before.get("source") == record["source"]:
            report["kept"].append(record["name"])
            objects.append({**record, "sha1": before["sha1"], "stat": _stat(path)})
            return
        write(path)
        report["written"].append(record["name"])
        objects.append({**record, "sha1": sha1(path), "stat": _stat(path)})

    matches = match([e for e, _f in proxies], matrix, node_uuids)
    for k, ((entry, f), m) in enumerate(zip(proxies, matches)):
        progress(0.05 + 0.6 * k / max(len(proxies), 1), m.name)
        name = safe_name(m.name)
        if name is None:
            report["unnamed"].append(m.name)
            continue
        if m.unit is None:
            report["not_in_matrix"].append(name)
            if m.problem:
                report["ambiguous"][name] = m.problem
        unit = m.unit
        color = (unit.color if unit is not None and unit.color and unit.color.upper() != "#FFFFFF"
                 else PALETTE[k % len(PALETTE)])
        rel = (folders.proxies / f"{name}.glb").relative_to(folders.base).as_posix()
        attrs = {k2: entry.get(k2) for k2 in ("us", "area", "tipo", "sito", "volume_m3", "area_m2", "z_top", "z_bottom")
                 if entry.get(k2) not in (None, "")}
        record = {"name": name, "kind": "proxy", "file": rel, "color": color, **attrs,
                  "node_id": unit.id if unit is not None else None,
                  "epoch": unit.epoch.name if unit is not None and unit.epoch is not None else None,
                  "graph_id": unit.graph_id if unit is not None else None, "how": m.how,
                  "source": _source(f) + [name, color, shift]}

        def write(path, f=f, name=name, color=color, record=record):
            mesh = trimesh.load(str(f), process=False, force="mesh")
            extras = {k2: v for k2, v in record.items() if k2 not in ("source", "file")}
            write_proxy_glb(path, name, mesh.vertices, mesh.faces, shift, color,
                            {**extras, "crs": chunk.crs, "shift": shift})

        place(rel, record, write)
    if survey and _model_path(chunk) is not None:
        progress(0.7, "rilievo")
        label = _folder_name(chunk.label)
        name = f"Rilievo {chunk.label}"
        rel = (folders.survey / label / f"{label}.glb").relative_to(folders.base).as_posix()
        textured = chunk.product_path("model_textured")
        record = {"name": name, "kind": "survey", "file": rel,
                  "source": _source(_model_path(chunk)) + _source(textured) + [name, shift]}
        place(rel, record, lambda path: write_survey_glb(chunk, path, name, shift,
                                                         {"chunk": chunk.label, "crs": chunk.crs, "shift": shift}))
    kept_files = {o["file"] for o in objects}
    for rel, before in previous.items():  # written here before, gone from the chunk now
        path = folders.base / rel
        if rel in kept_files or not path.exists():
            continue
        if before.get("stat") == _stat(path) or before.get("sha1") == sha1(path):
            path.unlink()
            report["removed"].append(before.get("name") or rel)
    camera_records, resolution = old.get("cameras"), old.get("resolution")
    if [float(v) for v in old.get("shift") or []] != shift and camera_records:
        cameras = True if cameras is None else cameras  # another shift: the cameras move too
    if cameras:
        from .cameras import blender_cameras

        label = _folder_name(chunk.label)
        made = blender_cameras(chunk, folders.base, folders.survey / label / "foto", shift, max_side,
                               old=camera_records, progress=lambda f, m: progress(0.75 + 0.2 * f, m))
        camera_records, resolution = made["cameras"], made["resolution"]
        report["cameras"] = {k: made[k] for k in ("written", "kept", "missing", "removed")}
    elif cameras is False and camera_records:
        for r in camera_records:  # the photos written for Blender go with their cameras
            f = folders.base / str(r.get("file") or "")
            if r.get("file") and f.exists():
                f.unlink()
        report["cameras"] = {"written": [], "kept": [], "missing": [], "removed": [r.get("name") for r in camera_records]}
        camera_records, resolution = [], None
    progress(0.95, "manifest")
    doc = getattr(chunk, "document", None)
    path = folders.base / MANIFEST
    manifest = {
        "format": FORMAT, "version": FORMAT_VERSION, "revision": int(old.get("revision") or 0) + 1,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": str(doc.path) if doc is not None and getattr(doc, "path", None) else None,
        "chunk": chunk.label, "crs": chunk.crs, "epsg": epsg_of(chunk.crs) or (matrix.epsg if matrix else None),
        "shift": shift, "standard_tree": folders.standard,
        "emjson": str(matrix.path) if matrix is not None and matrix.path is not None else None,
        "graph_id": matrix.graph_id if matrix is not None else None,
        "graph_name": matrix.name if matrix is not None else None,
        "objects": objects,
    }
    if camera_records is not None:
        manifest["cameras"], manifest["resolution"] = camera_records, resolution
    unchanged = ("revision", "written")
    if old and {k: v for k, v in old.items() if k not in unchanged} == \
            {k: v for k, v in manifest.items() if k not in unchanged}:
        # nothing changed: the manifest stays as it was, and the live script has nothing to reload
        report["manifest"], report["revision"] = path, int(old.get("revision") or 0)
        return report
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    _replace(tmp, path)
    report["manifest"] = path
    report["revision"] = manifest["revision"]
    progress(1.0, "manifest")
    return report


def drawn_unit(chunk, unit):
    """The polygon of ``unit`` (of the matrix) drawn in the chunk (``chunk.meta["drawings"]``), or None."""
    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    def same(a, b):
        return str(a or "").strip().lower() == str(b or "").strip().lower()

    for d in ((chunk.meta or {}).get("drawings") or {}).get("units") or []:
        try:
            u = UnitPolygon.from_dict(d)
        except Exception:  # noqa: BLE001 - a drawing that does not read is not this unit
            continue
        if same(u.us, unit.us) and same(u.tipo or "US", unit.tipo) and (not u.area or not unit.area
                                                                      or same(u.area, unit.area)):
            return u
    return None


def proxy_start(chunk, unit, shift) -> dict:
    """Where EMtools starts the proxy of ``unit`` (``create_proxy_for_unit``): a box as wide as the
    polygon of the unit drawn in the chunk, in the middle of it, its top on the DEM, in the coordinates
    of Blender (the world less ``shift``); ``{}`` when the unit is not drawn (EMtools puts it at the 3D
    cursor)."""
    poly = drawn_unit(chunk, unit)
    if poly is None or poly.geometry.is_empty:
        return {}
    x0, y0, x1, y1 = poly.geometry.bounds
    size = max(0.1, float(max(x1 - x0, y1 - y0)))
    centre = poly.geometry.representative_point()
    z = None
    dem = chunk.dem_path
    if dem is not None and dem.exists():
        import rasterio

        with rasterio.open(dem) as src:
            value = next(src.sample([(centre.x, centre.y)]))[0]
            if src.nodata is None or value != src.nodata:
                z = float(value)
    shift = np.asarray(shift, float)
    location = [centre.x - shift[0], centre.y - shift[1], (z - size / 2.0 - shift[2]) if z is not None else 0.0]
    return {"location": [round(float(v), 4) for v in location], "size": round(size, 4)}


__all__ = ["Folders", "MANIFEST", "find_project_root", "project_folders", "export_project", "read_manifest",
           "drawn_unit", "proxy_start",
           "write_proxy_glb", "write_survey_glb", "to_gltf", "from_gltf", "safe_name", "epsg_of"]
