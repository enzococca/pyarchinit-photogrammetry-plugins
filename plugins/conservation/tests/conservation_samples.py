"""The synthetic face of the tests, with its truths: a flat wall 6 m long and 3 m high (the masonry plugin's wall mesh,
without lean and bulge) in a project whose chunk has a rotated and scaled transform; its elevation built by the program
at 1 cm/px; then ``elevation.tif`` painted (a stone-like texture, a black crust, a green patch, a white efflorescence,
a dark crack) and ``relief.tif`` carved (a 3 cm deep hole, a groove of 3 mm x 1 cm along the crack, a patch with no
value) with rasterio. Everything in (u, v) metres of the elevation: ``u`` along the wall from its start, ``v`` the
height (world Z)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

WORLD = np.array([712345.678, 4567890.123, 312.5])
CRS = "EPSG:32633"
Z0 = float(WORLD[2])
WALL_LENGTH, WALL_HEIGHT = 6.0, 3.0
RESOLUTION = 0.01
DATE = "2026-05-14T10:00"

# painted on elevation.tif: (u0, v0, u1, v1) and colour
BLACK_CRUST = (0.5, Z0 + 0.5, 1.5, Z0 + 1.3)
BLACK_RGB = (38, 36, 34)
GREEN_PATCH = (4.2, Z0 + 0.4, 5.2, Z0 + 1.0)
GREEN_RGB = (62, 128, 48)
EFFLORESCENCE = (2.0, Z0 + 0.2, 3.0, Z0 + 0.6)
WHITE_RGB = (236, 234, 228)
CRACK_RGB = (22, 20, 18)
# carved in relief.tif
HOLE = (2.5, Z0 + 1.8, 2.9, Z0 + 2.2)
HOLE_DEPTH = 0.03  # metres, away from the viewer
GROOVE = ((0.8, Z0 + 2.0), (2.0, Z0 + 2.6))  # the crack: a line in (u, v)
GROOVE_WIDTH, GROOVE_DEPTH = 0.01, 0.003
NO_RELIEF = (4.0, Z0 + 2.0, 4.8, Z0 + 2.6)  # the relief has no value here


def box(b):
    from shapely.geometry import box as _box

    return _box(*b)


def groove():
    from shapely.geometry import LineString

    return LineString(GROOVE)


def wall_mesh(base=None, step: float = 0.025):
    """The face of a flat wall along +x from ``base`` (world), seen from -y: (vertices, faces)."""
    base = WORLD if base is None else np.asarray(base, np.float64)
    nu, nv = int(round(WALL_LENGTH / step)) + 1, int(round(WALL_HEIGHT / step)) + 1
    s, t = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    verts = base + np.column_stack([s.ravel() * WALL_LENGTH, np.zeros(s.size), t.ravel() * WALL_HEIGHT])
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return verts, faces


def transform():
    from pyarchinit_photogrammetry.core.geometry import compose_similarity

    a, b = np.radians(30.0), np.radians(2.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return compose_similarity(2.5, rz @ rx, WORLD)


def project(folder: Path, label: str = "Muro"):
    """A saved project with one chunk holding the wall as its model (no cameras)."""
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    doc = ppg.Document()
    doc.save(Path(folder) / "conservazione.ppg")
    ch = doc.add_chunk(label)
    ch.crs = CRS
    ch.transform = transform()
    verts, faces = wall_mesh()
    internal = ch.to_internal(np.asarray(verts, np.float64))
    path = write_mesh_ply(ch.data_dir / "model.ply", internal, np.asarray(faces), double=True)
    ch._set_product("model", path)
    ch.meta["metric"] = True
    ch.meta["date"] = DATE
    doc.chunk = ch
    doc.save()
    return doc, ch


def _cells(src):
    """The (u, v) of the centres of the cells of a raster."""
    t = src.transform
    cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
    return t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e


def _inside(u, v, geom):
    import shapely

    return shapely.contains_xy(geom, u, v)


def paint(chunk, entry) -> None:
    """The truths written into the elevation's two rasters."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    ortho, relief = elevation_paths(chunk, entry)
    with rasterio.open(ortho, "r+") as dst:
        u, v = _cells(dst)
        rng = np.random.default_rng(7)
        base = np.array([168, 150, 122], np.float64)
        noise = rng.normal(0.0, 9.0, u.shape)
        rgb = np.clip(base[:, None, None] + noise[None], 0, 255)
        for b, colour in ((BLACK_CRUST, BLACK_RGB), (GREEN_PATCH, GREEN_RGB), (EFFLORESCENCE, WHITE_RGB)):
            m = _inside(u, v, box(b))
            rgb[:, m] = np.array(colour, np.float64)[:, None] + noise[m][None] * 0.3
        m = _inside(u, v, groove().buffer(GROOVE_WIDTH / 2, cap_style="flat"))
        rgb[:, m] = np.array(CRACK_RGB, np.float64)[:, None]
        alpha = np.full((1,) + u.shape, 255.0)
        dst.write(np.concatenate([rgb, alpha]).astype(np.uint8))
    with rasterio.open(relief, "r+") as dst:
        u, v = _cells(dst)
        d = dst.read(1)
        d[_inside(u, v, box(HOLE))] -= HOLE_DEPTH
        d[_inside(u, v, groove().buffer(GROOVE_WIDTH / 2, cap_style="flat"))] -= GROOVE_DEPTH
        d[_inside(u, v, box(NO_RELIEF))] = np.nan
        dst.write(d.astype(np.float32), 1)


def build(folder: Path):
    """``(doc, chunk, entry)``: the project, its chunk and the painted elevation of the wall's face."""
    from pyarchinit_photogrammetry.engine.elevation import build_elevation, plane_from_line, with_heights

    doc, ch = project(folder)
    a, b = WORLD[:2], WORLD[:2] + [WALL_LENGTH, 0.0]
    frame = with_heights(plane_from_line(ch, a, b, side=WORLD[:2] + [3.0, -5.0]))
    entry = build_elevation(ch, frame, RESOLUTION, "Prospetto sud", depth=(-0.5, 0.5),
                            bounds=(0.0, Z0, WALL_LENGTH, Z0 + WALL_HEIGHT))
    paint(ch, entry)
    doc.save()
    return doc, ch, entry


_CACHE: dict = {}


def face(folder: Path):
    """As :func:`build`, built once per session and copied into ``folder`` (the tests change their copy)."""
    import pyarchinit_photogrammetry as ppg

    if "path" not in _CACHE:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="ppg-conservation-face-"))
        doc, ch, entry = build(root)
        _CACHE.update(path=Path(doc.path), root=root)
    src = _CACHE["path"]
    for item in src.parent.iterdir():
        target = Path(folder) / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    doc = ppg.Document.open(Path(folder) / src.name)
    ch = doc.chunk
    return doc, ch, ch.meta["elevations"][0]


def put_units(chunk, eid: str, units: dict, tipo: str = "US", sito: str = "Saggio") -> None:
    """Units drawn on the elevation ``eid`` as the window keeps them (``{us: polygon in (u, v)}``)."""
    frames = chunk.meta.setdefault("drawings", {}).setdefault("frames", {})
    data = frames.setdefault(eid, {})
    data["units"] = [{"wkt": g.wkt, "us": us, "tipo": tipo, "area": "1", "sito": sito, "attributes": {}}
                     for us, g in units.items()]


def cleanup() -> None:
    root = _CACHE.pop("root", None)
    _CACHE.pop("path", None)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)
