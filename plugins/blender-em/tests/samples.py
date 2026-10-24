"""A small excavation for the tests: a DEM in UTM, three unit proxies (US 1, USM 2, US 3 of area 1)
built by the program as its *Proxy 3D delle US* builds them, a coloured model, and the em.json
PyArchInit writes for the site (its composite names, the node_uuid of each form in ``data``, an
epoch per unit, the shift of the project when asked)."""
from __future__ import annotations

import json
import uuid

import numpy as np

ORIGIN = (300000.0, 4500000.0, 100.0)  # the lower left corner of the DEM, its base height
UNITS = [("1", "US", "1"), ("1", "USM", "2"), ("1", "US", "3")]


def project(tmp_path, model: bool = True, units=UNITS):
    import pyarchinit_photogrammetry as ppg
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.proxies import save_proxy, unit_proxy

    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "scavo.ppg")
    chunk.crs = "EPSG:32633"
    x0, y0, z0 = ORIGIN
    rng = np.random.default_rng(3)
    z = (z0 + 0.2 + 0.05 * rng.random((60, 100))).astype(np.float32)
    dem = chunk.data_dir / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", width=100, height=60, count=1, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(x0, y0 + 3.0, 0.05, 0.05), nodata=-9999) as dst:
        dst.write(z, 1)
    chunk._set_product("dem", dem)
    entries = []
    for k, (area, tipo, us) in enumerate(units):
        px = unit_proxy(dem, box(x0 + 0.3 + 1.5 * k, y0 + 0.5, x0 + 1.4 + 1.5 * k, y0 + 2.5), thickness=0.25 + 0.1 * k,
                        us=us, area=area, tipo=tipo)
        f = save_proxy(px, chunk.data_dir / "proxies", "ply")
        entries.append({"name": px.name, "us": us, "area": area, "tipo": tipo, "sito": "Scavo",
                        "file": str(f.relative_to(chunk.data_dir)), "volume_m3": px.volume_m3, "area_m2": px.area_m2,
                        "z_top": list(px.z_top), "z_bottom": list(px.z_bottom), "base": "thickness"})
    chunk.meta["proxies"] = entries
    if model:
        add_model(chunk)
    doc.save()
    return doc


def add_model(chunk, n: int = 24):
    """A coloured grid over the DEM as the model of the chunk (internal frame: world less ORIGIN)."""
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData, write_mesh

    xs = np.linspace(0.0, 5.0, n)
    X, Y = np.meshgrid(xs, np.linspace(0.0, 3.0, n))
    v = np.column_stack([X.ravel(), Y.ravel(), 0.22 + 0.02 * np.sin(X.ravel())])
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])]).astype(np.int64)
    arr = np.empty(len(v), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = v.T
    arr["red"], arr["green"], arr["blue"] = 180, 140, 90
    t = np.eye(4)
    t[:3, 3] = ORIGIN
    chunk.transform = t
    path = chunk.data_dir / "model.ply"
    write_mesh(path, MeshData(arr, faces, None, [], None))
    chunk._set_product("model", path)
    return path


def emjson(path, units=UNITS, shift=None, site: str = "Scavo", container: bool = True, epochs=True) -> dict:
    """The em.json of the site as PyArchInit writes it: ``{node_uuid: (area, tipo, us)}`` returned."""
    nodes, edges, ids = [], [], {}
    if shift is not None:
        nodes.append({"id": f"geo_{site}", "name": "geo_position", "node_type": "geo_position",
                      "data": {"shift_x": shift[0], "shift_y": shift[1], "shift_z": shift[2], "rotation": 0.0}})
    roman = {"id": "ep-roman", "name": "Età romana", "node_type": "EpochNode",
             "data": {"start_time": -100, "end_time": 300, "color": "#CC6633"}}
    late = {"id": "ep-late", "name": "Tardoantico", "node_type": "EpochNode",
            "data": {"start_time": 300, "end_time": 600, "color": "#3366CC"}}
    if epochs:
        nodes += [roman, late]
    for k, (area, tipo, us) in enumerate(units):
        nid = str(uuid.uuid4())
        ids[nid] = (area, tipo, us)
        node_type = "USVs" if tipo == "USVs" else "US"
        nodes.append({"id": nid, "name": f"{area}.{tipo}{us}", "node_type": node_type, "description": f"{tipo} {us}",
                      "data": {"us": us, "area": area, "unita_tipo": tipo, "node_uuid": nid, "sito": site}})
        if epochs:
            edges.append({"id": f"e{k}", "edge_type": "has_first_epoch", "source": nid,
                          "target": "ep-late" if k == 0 else "ep-roman"})
    first, second = list(ids)[:2]
    edges.append({"id": "rel1", "edge_type": "is_after", "source": first, "target": second})
    graph = {"graph_id": site, "name": site, "nodes": nodes, "edges": edges}
    header = {"format": "em.json", "version": "1.0", "schema_version": 2}
    doc = {"header": header, "graphs": {site: graph}, "active_graph_id": site} if container else \
        {"header": header, "graph": graph}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return ids


def look_down(centre, height: float = 3.0, tilt: float = 0.0):
    """``world_from_cam`` (OpenCV axes: x right, y down, z forward) of a camera ``height`` above
    ``centre`` looking down, tilted ``tilt`` radians about the x axis."""
    c, s = np.cos(tilt), np.sin(tilt)
    tilt_m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    r = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]) @ tilt_m  # x east, y south, z down
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(centre, float) + [0.0, 0.0, height]
    return t


def add_cameras(chunk, folder, n: int = 2, width: int = 640, height: int = 480, k1: float = -0.08):
    """``n`` aligned cameras over the excavation with a sensor of radial distortion, and their photos
    (noise, with a white dot where the corner of the first proxy - DOT - is seen, as the lens sees it)."""
    import cv2

    from pyarchinit_photogrammetry.core.camera import Camera, Sensor

    folder.mkdir(parents=True, exist_ok=True)
    f = 0.9 * width
    sensor = Sensor(0, "test", width, height, "OPENCV", [f, f * 1.01, width / 2 + 7.0, height / 2 - 5.0, k1, 0.01, 0.0, 0.0])
    chunk.sensors = {0: sensor}
    t = np.asarray(chunk.transform, float)
    inv = np.linalg.inv(t)
    rng = np.random.default_rng(5)
    chunk.cameras = []
    for k in range(n):
        world = look_down((ORIGIN[0] + 1.5 + 1.2 * k, ORIGIN[1] + 1.5, ORIGIN[2]), 3.0, 0.15 * k)
        internal = inv @ world
        internal[:3, :3] /= np.cbrt(np.linalg.det(internal[:3, :3]))  # a rotation in the frame of the chunk
        cam = Camera(k, f"IMG_{k:04d}.JPG", str(folder / f"IMG_{k:04d}.JPG"), 0, transform=internal)
        image = rng.integers(0, 90, (height, width, 3), dtype=np.uint8)
        px, depth = cam.project(sensor, (inv @ np.append(DOT, 1.0))[:3][None])
        if depth[0] > 0:
            cv2.circle(image, (int(round(px[0, 0] - 0.5)), int(round(px[0, 1] - 0.5))), 3, (255, 255, 255), -1)
        cv2.imwrite(cam.path, image)
        chunk.cameras.append(cam)
    return sensor


DOT = np.array([ORIGIN[0] + 1.4, ORIGIN[1] + 2.5, ORIGIN[2] + 0.22])  # a point of the survey seen by the cameras
