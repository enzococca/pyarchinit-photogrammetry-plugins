"""The plugin through the real pipeline, on synthetic scenes rendered by the program.

Two scenes, each a textured mesh on a printed mat with coded targets (the scale and the frame), photographed by the
program's ray caster (``testing.synthetic.render_mesh``), then aligned, densified, meshed by the program, and the
plugin's elements fitted on the model it made:

* ``colonna``: a column 2 m high, radius 0.30 m at the base, with a known profile (entasis), photographed in three
  rings; the plugin's column from a plan polygon: axis, knots and volume against the truth;
* ``muro``: a wall 4 m wide with a pointed arch (span 1.6 m, an archivolt ring 0.25 m thick standing 15 cm proud of the
  wall) and a cornice of known section along its top, photographed from the front in an arc; the plugin's arch on a
  vertical plane through the archivolt and its cornice along the wall: type, span, rise, thickness, section.

    python make_proof.py [--folder /Users/Shared/ppg_demo_architecture] [--cadquery PYTHON]
                         [--stage all|scenes|process|fit|figures]

Each stage keeps what it made: running again goes on from where it stopped. ``results.json`` and ``LEGGIMI.md`` hold the
numbers; ``figure_*.png`` the figures (a neutral folder, for the guide).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "tests")]

FOLDER = Path("/Users/Shared/ppg_demo_architecture")
SIZE = (1600, 1200)
FOCAL = 1500.0
# the column
H_COL = 2.0


def column_radius(z):
    """The truth: 0.30 m at the base, 0.27 m at the top, swelling over the chord (entasis)."""
    z = np.asarray(z, float)
    return 0.30 - 0.03 * (z / H_COL) ** 2 + 0.008 * np.sin(np.pi * z / H_COL)


# the wall
WALL = {"x0": -2.0, "x1": 2.0, "top": 3.6, "depth": 0.5}
ARCH = {"s": 0.8, "R": 1.6, "vs": 1.4, "t": 0.25, "proud": 0.15}  # equilateral pointed arch, span 1.6
CORNICE = np.array([[0.0, 0.0], [0.04, 0.0], [0.04, 0.03], [0.06, 0.05], [0.10, 0.08], [0.13, 0.13], [0.16, 0.18],
                    [0.18, 0.21], [0.18, 0.25], [0.0, 0.25]])  # (projection, height) on the wall face
CORNICE_Z = WALL["top"] - 0.25


# ====================================================================== scenes
def _texture(rng, n):
    from pyarchinit_photogrammetry.testing.synthetic import _noise_texture

    return _noise_texture(n, n, rng).astype(np.float32)


def _mat(rng, size, texel, targets, marker):
    """A printed mat (square, centred at the origin, z = 0) with ArUco targets at ``targets`` {code: (x, y)}."""
    from pyarchinit_photogrammetry.markers.targets import target_image

    n = int(round(size / texel))
    tex = _texture(rng, n)
    tex = np.clip(tex.mean(axis=2, keepdims=True) * 0.7 + 70, 0, 255)
    tex = np.repeat(tex, 3, axis=2).astype(np.uint8)
    tpx = int(round(marker / texel))
    for code, (tx, ty) in targets.items():
        pat = target_image("aruco_4x4_50", code, tpx)
        quiet = int(tpx * 0.3)
        tile = np.full((tpx + 2 * quiet, tpx + 2 * quiet), 255, np.uint8)
        tile[quiet:quiet + tpx, quiet:quiet + tpx] = pat
        c, r = int((tx + size / 2) / texel), int((size / 2 - ty) / texel)
        r0, c0 = r - tile.shape[0] // 2, c - tile.shape[1] // 2
        tex[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] = tile[..., None]
    half = size / 2
    v = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], np.float64)
    return v, np.array([[0, 1, 2], [0, 2, 3]]), tex


def _noise3d(rng, scales=(0.004, 0.012, 0.04, 0.15), amps=(0.35, 0.3, 0.2, 0.15), tint=(60.0, 105.0, 165.0)):
    """A full-contrast stone-like colour from a 3D value noise at several scales (metres): any face of any shape
    gets texture at every scale, nothing stretched along an axis."""
    perm = rng.permutation(4096)
    tint = np.asarray(tint, float)  # BGR: earthy by default

    def lattice(i, j, k):
        h = perm[(i * 73856093 ^ j * 19349663 ^ k * 83492791) % 4096]
        return (h / 4095.0)

    def octave(p, scale):
        q = p / scale
        f = np.floor(q).astype(np.int64)
        t = q - f
        t = t * t * (3 - 2 * t)
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) \
                        * (t[:, 2] if dz else 1 - t[:, 2])
                    out = out + w * lattice(f[:, 0] + dx, f[:, 1] + dy, f[:, 2] + dz)
        return out

    def colour(p):
        p = np.asarray(p, np.float64) + 100.0  # no negative lattice indices
        g = sum(a * octave(p, s) for s, a in zip(scales, amps))
        g = np.clip((g - 0.5) * 2.6 + 0.5, 0.0, 1.0)
        return np.clip(tint[None, :] * (0.25 + 1.3 * g[:, None]), 0, 255)

    return colour


def _with_backdrop(v, f, colour, rng, radius: float = 7.0, height: float = 6.0):
    """A textured backdrop around the scene (an inward cylinder, as the walls of a room or the trees of a site): the
    silhouettes have something behind them, as in a real survey, not a flat grey that depth maps fill with fins."""
    a = np.linspace(0, 2 * np.pi, 181)[:-1]
    ring = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    n = len(ring)
    bv = np.vstack([np.column_stack([ring, np.full(n, -0.5)]), np.column_stack([ring, np.full(n, height)])])
    j = np.arange(n)
    k = (j + 1) % n
    bf = np.vstack([np.column_stack([j, k + n, k]), np.column_stack([j, j + n, k + n])]) + len(v)
    far = _noise3d(rng, scales=(0.03, 0.1, 0.4, 1.5), tint=(105.0, 135.0, 110.0))

    def both(p):
        out = colour(p)
        m = np.hypot(p[:, 0], p[:, 1]) > radius - 0.5
        if m.any():
            out[m] = far(p[m])
        return out

    return np.vstack([v, bv]), np.vstack([f, bf]), both


def column_scene():
    from pyarchinit_photogrammetry.testing.synthetic import MeshScene

    from arch_samples import _engine

    engine = _engine()
    rng = np.random.default_rng(7)
    z = np.linspace(0, H_COL, 201)
    prof = np.column_stack([column_radius(z), z])
    col = engine.revolved_mesh(prof, segments=360)
    targets = {i: (0.8 * np.cos(2 * np.pi * i / 12), 0.8 * np.sin(2 * np.pi * i / 12)) for i in range(12)}
    mv, mf, mtex = _mat(rng, 2.4, 0.0015, targets, 0.10)
    colour = _noise3d(rng)
    v = np.vstack([mv, col.vertices])
    f = np.vstack([mf, col.faces + len(mv)])
    v, f, colour = _with_backdrop(v, f, colour, rng)
    scene = MeshScene(v.astype(np.float32), f.astype(np.uint32), len(mf), 2.4, 0.0015, mtex, colour,
                      targets={k: (x, y, 0.0) for k, (x, y) in targets.items()})
    from pyarchinit_photogrammetry.core.geometry import look_at

    poses = []
    for height, target_z, count, r in ((0.45, 0.55, 24, 2.2), (1.4, 1.1, 24, 2.2), (2.9, 1.5, 16, 2.0)):
        for k in range(count):
            a = 2 * np.pi * k / count + (0.13 if count == 16 else 0.0)
            eye = np.array([r * np.cos(a), r * np.sin(a), height])
            poses.append(look_at(eye, np.array([0.0, 0.0, target_z]), up=(0, 0, 1)))
    return scene, poses


def wall_mesh():
    """The wall with the opening, the archivolt standing proud of it, the cornice: ``(trimesh, parts)``."""
    import trimesh
    from shapely.geometry import LineString, Polygon

    from arch_samples import _engine

    engine = _engine()
    s, R, vs, t = ARCH["s"], ARCH["R"], ARCH["vs"], ARCH["t"]
    cL, cR = -s + R, s - R
    top = np.arccos((0 - cL) / R)
    aL = np.linspace(np.pi, top, 120)
    aR = np.linspace(np.pi - top, 0, 120)
    arc = np.vstack([np.column_stack([cL + R * np.cos(aL), vs + R * np.sin(aL)]),
                     np.column_stack([cR + R * np.cos(aR), vs + R * np.sin(aR)])[1:]])
    intrados = np.vstack([[-s, 0.0], arc, [s, 0.0]])  # the opening's outline from the ground
    opening = Polygon(np.vstack([intrados, [[s, -1.0], [-s, -1.0]]]))
    x0, x1, top_z = WALL["x0"], WALL["x1"], WALL["top"]
    wall = Polygon([(x0, 0), (x1, 0), (x1, top_z), (x0, top_z)]).difference(opening)
    ring = LineString(intrados).buffer(t, cap_style="flat", join_style="round", quad_segs=32).difference(opening)
    ring = ring.intersection(Polygon([(-9, 0), (9, 0), (9, 9), (-9, 9)]))
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # (x, y=v, z=depth) -> (x, -depth, v)
    parts = []
    w = engine.prism_mesh(np.asarray(wall.exterior.coords), WALL["depth"])
    w.apply_transform(rot)  # y from 0 to -depth: the wall's back towards +y
    w.apply_translation([0, WALL["depth"], 0])
    parts.append(w)
    r = engine.prism_mesh(np.asarray(ring.exterior.coords), ARCH["proud"])
    r.apply_transform(rot)  # y from 0 to -proud: in front of the wall face
    parts.append(r)
    sec = np.column_stack([-CORNICE[:, 0], CORNICE[:, 1]])  # u = up x (+x) = +y: the cornice projects to -y
    path = np.array([[x0, 0.0, CORNICE_Z], [x1, 0.0, CORNICE_Z]])
    parts.append(engine.sweep_mesh(path, sec))
    mesh = trimesh.util.concatenate(parts)
    truth = {"intrados": intrados, "ring_area": float(ring.area), "mesh": mesh}
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.05, max_iter=20)
    return trimesh.Trimesh(v, f, process=False), truth


def wall_scene():
    from pyarchinit_photogrammetry.core.geometry import look_at
    from pyarchinit_photogrammetry.testing.synthetic import MeshScene

    rng = np.random.default_rng(9)
    mesh, _truth = wall_mesh()
    targets = {}
    k = 0
    for x in (-2.6, -1.6, -0.6, 0.6, 1.6, 2.6):
        for y in (-1.0, -2.2):
            targets[k] = (x, y)
            k += 1
    targets[k], targets[k + 1] = (-2.6, 0.5), (2.6, 0.5)
    mv, mf, mtex = _mat(rng, 6.4, 0.003, targets, 0.16)
    colour = _noise3d(rng)
    v = np.vstack([mv, mesh.vertices])
    f = np.vstack([mf, np.asarray(mesh.faces) + len(mv)])
    v, f, colour = _with_backdrop(v, f, colour, rng, radius=8.0)
    scene = MeshScene(v.astype(np.float32), f.astype(np.uint32), len(mf), 6.4, 0.003, mtex, colour,
                      targets={c: (x, y, 0.0) for c, (x, y) in targets.items()})
    poses = []
    for height, target_z, r, count in ((1.0, 1.6, 4.6, 13), (2.4, 2.0, 4.6, 13), (4.2, 2.4, 4.4, 11)):
        for a in np.radians(np.linspace(-60, 60, count)):
            eye = np.array([r * np.sin(a), -r * np.cos(a), height])
            poses.append(look_at(eye, np.array([0.0, 0.0, target_z]), up=(0, 0, 1)))
    for x in (-0.5, 0.0, 0.5):  # under the arch, looking up at the intrados and the cornice
        for zz in (0.8,):
            poses.append(look_at(np.array([x, -2.2, zz]), np.array([x * 0.5, 0.0, 2.6]), up=(0, 0, 1)))
    return scene, poses


def render(name: str, folder: Path) -> None:
    from pyarchinit_photogrammetry.testing.synthetic import render_mesh

    out = folder / name / "foto"
    if out.exists() and len(list(out.glob("*.jpg"))) > 10:
        return
    out.mkdir(parents=True, exist_ok=True)
    scene, poses = column_scene() if name == "colonna" else wall_scene()
    K = np.array([[FOCAL, 0, SIZE[0] / 2], [0, FOCAL, SIZE[1] / 2], [0, 0, 1]])
    for i, pose in enumerate(poses):
        img = render_mesh(scene, pose, K, SIZE, -0.02)
        cv2.imwrite(str(out / f"{name.upper()}_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{name}: photo {i + 1}/{len(poses)}", flush=True)
    with open(folder / name / "target.csv", "w") as fh:
        fh.write("# label,X,Y,Z (LOCAL, metres)\n")
        for code, (x, y, z) in scene.targets.items():
            fh.write(f"{code},{x:.5f},{y:.5f},{z:.5f}\n")


# ====================================================================== processing (the program's own chain)
def process(name: str, folder: Path) -> Path:
    import pyarchinit_photogrammetry as ppg

    project = folder / name / f"{name}.ppg"
    if project.exists():
        doc = ppg.Document.open(project)
        if doc.chunk.product_path("model") is not None and doc.chunk.product_path("model").exists():
            return project
    doc = ppg.Document()
    ch = doc.add_chunk("Colonna" if name == "colonna" else "Facciata")
    doc.save(project)
    ch.add_photos([folder / name / "foto"])
    ch.detect_markers("aruco_4x4_50")
    ch.import_reference(folder / name / "target.csv", crs="LOCAL")
    doc.save()
    times = {}
    t = time.monotonic()
    ch.match_photos(quality="high", progress=_progress("match"))
    ch.align_cameras(progress=_progress("align"))
    ch.update_transform()
    times["align_s"] = time.monotonic() - t
    t = time.monotonic()
    ch.build_depth_maps(quality="high", filter_mode="mild", progress=_progress("depth"))
    ch.build_point_cloud(progress=_progress("cloud"))
    if name == "colonna":  # the mat and the backdrop are not the building
        ch.crop_point_cloud(zmin=0.01, radius=1.5)
    else:
        ch.crop_point_cloud(zmin=0.01, bounds=(-3.0, -1.5, 3.0, 1.5))
    ch.build_model(face_count=300000 if name == "colonna" else 800000, progress=_progress("mesh"))
    times["dense_mesh_s"] = time.monotonic() - t
    ch.meta["proof_times"] = times
    doc.save()
    return project


def _progress(phase):
    last = [-1]

    def cb(fraction, message=""):
        k = int(fraction * 10)
        if k != last[0]:
            last[0] = k
            print(f"  {phase} {fraction * 100:5.1f} % {message}", flush=True)

    return cb


# ====================================================================== the plugin on the models
def fit(folder: Path, cadquery: Path) -> dict:
    import pyarchinit_photogrammetry as ppg

    from ppg_architecture import elements, engine, runner

    elements.WORKER = (Path(cadquery), Path(runner.__file__).with_name("worker.py"))
    results = {}
    # the column
    doc = ppg.Document.open(folder / "colonna" / "colonna.ppg")
    ch = doc.chunk
    for e in elements.elements(ch):
        elements.remove(ch, e["id"])
    square = np.array([[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]])
    col = elements.column(ch, polygon=square, zmin=0.05, zmax=H_COL - 0.05, knots=12, name="Colonna")
    p = col["params"]
    d = np.asarray(p["axis_direction"])
    pt = np.asarray(p["axis_point"])
    zk = np.asarray(p["knot_heights"]) + (pt[2] - 0.0)  # knot heights from the part's base: back to the truth's z
    tilt = float(np.degrees(np.arccos(min(abs(d[2]), 1.0))))
    off = float(np.hypot(pt[0], pt[1]))
    z_true = np.linspace(pt[2], pt[2] + p["height"], 2001)
    vol_true = float(np.trapezoid(np.pi * column_radius(z_true) ** 2, z_true))
    results["colonna"] = {
        "foto": len(ch.cameras), "allineate": len(ch.aligned_cameras), "tempi_s": ch.meta.get("proof_times"),
        "asse_inclinazione_gradi": tilt, "asse_scarto_dall_asse_vero_m": off,
        "nodi_scarto_max_m": float(np.abs(np.asarray(p["knot_radii"]) - column_radius(zk)).max()),
        "nodi_scarto_mediano_m": float(np.median(np.abs(np.asarray(p["knot_radii"]) - column_radius(zk)))),
        "raggio_base": [p["radius_base"], float(column_radius(zk[0]))],
        "raggio_cima": [p["radius_top"], float(column_radius(zk[-1]))],
        "entasi": [p["entasis"], _true_entasis(zk[0], zk[-1])],
        "volume_m3": [col["solid"]["volume"], vol_true],
        "volume_scarto_percento": 100 * (col["solid"]["volume"] / vol_true - 1),
        "scarto_modello_solido": col["fit"],
        "modello_contro_verita": _model_error(ch, _column_truth(), (-0.4, -0.4, 0.05, 0.4, 0.4, H_COL - 0.05))}
    doc.save()
    # the wall: the arch on the archivolt's plane, the cornice along the wall
    doc = ppg.Document.open(folder / "muro" / "muro.ppg")
    ch = doc.chunk
    for e in elements.elements(ch):
        elements.remove(ch, e["id"])
    y = -ARCH["proud"] / 2
    p1, p2 = np.array([-1.8, y]), np.array([1.8, y])
    po, u, vv, _w = elements.plane_of(ch, None, p1, p2)
    v, f = elements.local_mesh(ch)
    guess = engine.guess_pick(engine.mesh_section(v, f, po, u, vv))
    # the point the user clicks in the opening: its middle, 0.8 m above the ground
    rel = np.array([0.0, y, 0.8]) - elements.origin(ch) - po
    pick = (float(rel @ u), float(rel @ vv))
    arch = elements.arch(ch, pick, p1=p1, p2=p2, name="Arco")
    ap = arch["params"]
    s, R = ARCH["s"], ARCH["R"]
    rise = float(np.sqrt(R * R - (R - s) ** 2))
    wall, truth_parts = wall_mesh()
    del wall
    true_mesh = truth_parts["mesh"]
    results["arco"] = {"foto": len(ch.cameras), "allineate": len(ch.aligned_cameras),
                       "tempi_s": ch.meta.get("proof_times"), "punto_uv": list(pick),
                       "punto_proposto_dal_plugin_uv": list(guess), "tipo": [ap["type"], "pointed"],
                       "luce_m": [ap["span"], 2 * s],
                       "freccia_m": [ap["rise"], rise], "raggio_m": [ap.get("radius"), R],
                       "imposta_m": [ap["springing"], ARCH["vs"]], "spessore_m": [ap["thickness"], ARCH["t"]],
                       "candidati_rms_m": {k: c["rms"] for k, c in arch["profile"]["fit"]["candidates"].items()},
                       "scarto_modello_solido": arch["fit"],
                       "modello_contro_verita": _model_error(ch, true_mesh, (-1.1, -0.16, 1.2, 1.1, 0.0, 3.2))}
    path = np.array([[-1.8, 0.0], [1.8, 0.0]])
    # the dialog's defaults: the window found from the cut, 10 cm into the wall
    mould = elements.moulding(ch, path, height=CORNICE_Z, station=1.0, out=-1, name="Cornice")
    mp = mould["params"]
    sec = np.asarray(mould["geometry"]["section"])
    from shapely.geometry import Polygon, box

    # the part of the section in front of the wall face the plugin found (behind it is the back line's slab)
    front = Polygon(sec).intersection(box(-1, -1, mp["wall_u"], 1))
    truth = Polygon(np.column_stack([-CORNICE[:, 0], CORNICE[:, 1]]))
    results["cornice"] = {"sporgenza_m": [mp["projection"], float(CORNICE[:, 0].max())],
                          "altezza_m": [mp["section_height"], float(CORNICE[:, 1].max())],
                          "quote_della_sezione_m": mp["v_range"], "faccia_del_muro_u_m": mp["wall_u"],
                          "finestra_trovata": mp["window_used"],
                          "area_davanti_al_muro_m2": [float(front.area), float(truth.area)],
                          "area_scarto_percento": 100 * (front.area / truth.area - 1),
                          "differenza_simmetrica_m2": float(front.symmetric_difference(truth).area),
                          "hausdorff_profilo_m": float(front.exterior.hausdorff_distance(truth.exterior)),
                          "vertici_sezione": int(len(sec)), "volume_solido_m3": mould["solid"]["volume"],
                          "scarto_modello_solido": mould["fit"],
                          "modello_contro_verita": _model_error(ch, true_mesh, (-1.8, -0.2, CORNICE_Z - 0.01, 1.8, 0.01,
                                                                                WALL["top"] + 0.01))}
    doc.save()
    (folder / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


def _column_truth():
    from arch_samples import _engine

    z = np.linspace(0, H_COL, 401)
    return _engine().revolved_mesh(np.column_stack([column_radius(z), z]), segments=720)


def _model_error(ch, truth_mesh, bbox) -> dict:
    """How far the program's model is from the true surface inside ``bbox`` (world): the photogrammetry's own error,
    which the solid inherits and the plugin cannot take away."""
    from ppg_architecture import elements, engine

    v, _f = elements.local_mesh(ch)
    w = v + elements.origin(ch)
    lo, hi = np.asarray(bbox[:3]), np.asarray(bbox[3:])
    pts = w[np.all((w >= lo) & (w <= hi), axis=1)]
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    d = engine.distances_to_mesh(pts, truth_mesh.vertices, truth_mesh.faces, max_edge=0.05)
    return engine.fit_report(d)


def _true_entasis(z0, z1):
    z = np.linspace(z0, z1, 2001)
    r = column_radius(z)
    return float((r - (r[0] + (r[-1] - r[0]) * (z - z0) / (z1 - z0))).max())


# ====================================================================== figures
def figures(folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarchinit_photogrammetry as ppg

    from ppg_architecture import elements, engine

    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "ppg"})
    # the column's profile: measured, knots, spline, truth
    ch = ppg.Document.open(folder / "colonna" / "colonna.ppg").chunk
    col = elements.entry(ch, "Colonna")
    prof, p = col["profile"], col["params"]
    base = p["axis_point"][2]
    fig, ax = plt.subplots(figsize=(4.2, 6.4))
    zz = np.linspace(0, p["height"], 400)
    ax.plot(column_radius(zz + base), zz, color="#999999", lw=3, label="verità")
    ax.plot(prof["radii"], prof["heights"], ".", ms=3, color="#c0392b", label="profilo misurato (mediana)")
    curve = np.asarray(col["geometry"]["profile"])
    ax.plot(curve[:, 0], curve[:, 1], color="#1f4e79", lw=1.2, label="solido (spline)")
    ax.plot(p["knot_radii"], p["knot_heights"], "o", mfc="white", color="#1f4e79", ms=5, label=f"{p['knots']} nodi")
    ax.set_xlim(0.24, 0.32)
    ax.set_xlabel("raggio (m)")
    ax.set_ylabel("quota dalla base (m)")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("Colonna: profilo misurato e solido")
    fig.tight_layout()
    fig.savefig(folder / "figure_colonna_profilo.png", dpi=150)
    plt.close(fig)
    # the arch's section: the cut of the model, the intrados seen, the fitted ring
    ch = ppg.Document.open(folder / "muro" / "muro.ppg").chunk
    arch = elements.entry(ch, "Arco")
    plane = arch["source"]["plane"]
    o = elements.origin(ch)
    v, f = elements.local_mesh(ch)
    lines = engine.mesh_section(v, f, np.asarray(plane["origin"]) - o, plane["u"], plane["v"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color="#999999", lw=0.8)
    intr = np.asarray(arch["profile"]["intrados"])
    ax.plot(intr[:, 0], intr[:, 1], ".", ms=2, color="#c0392b", label="intradosso visto dal punto")
    ring = engine.outline_polygon(arch["geometry"]["outline"])
    ring = np.vstack([ring, ring[:1]])
    ax.plot(ring[:, 0], ring[:, 1], color="#1f4e79", lw=1.5, label="anello del solido (archi di cerchio)")
    pp = arch["params"]
    ax.axhline(pp["springing"], color="#2e7d32", lw=0.8, ls="--", label="linea d'imposta")
    pick = arch["source"]["pick"]
    ax.plot(*pick, "x", color="black", ms=7, label="punto nell'apertura")
    ax.set_aspect("equal")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("quota (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Arco: sezione del modello e arco a sesto acuto adattato")
    fig.tight_layout()
    fig.savefig(folder / "figure_arco_sezione.png", dpi=150)
    plt.close(fig)
    # the cornice's section: the cut of the model, the solid's section, the truth
    corn = elements.entry(ch, "Cornice")
    src = corn["source"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ln = np.asarray(corn["profile"]["line"])
    ax.plot(ln[:, 0], ln[:, 1], color="#999999", lw=0.8, label="taglio del modello")
    ax.axvline(corn["params"]["wall_u"], color="#2e7d32", lw=0.8, ls="--", label="faccia del muro trovata")
    sec = np.asarray(corn["geometry"]["section"])
    sec = np.vstack([sec, sec[:1]])
    ax.plot(sec[:, 0], sec[:, 1], color="#1f4e79", lw=1.5, label="sezione del solido")
    truth = np.column_stack([-CORNICE[:, 0], CORNICE[:, 1] + (CORNICE_Z - src["height"])])
    truth = np.vstack([truth, truth[:1]])
    ax.plot(truth[:, 0], truth[:, 1], "--", color="#c0392b", lw=1.0, label="verità")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 0.15)
    ax.set_ylim(-0.12, 0.38)
    ax.grid(alpha=0.3)
    ax.set_xlabel("u (m): fuori dal muro verso sinistra")
    ax.set_ylabel("v (m) dalla quota del percorso")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Cornice: sezione normale al percorso")
    fig.tight_layout()
    fig.savefig(folder / "figure_cornice_sezione.png", dpi=150)
    plt.close(fig)
    # the 3D view: the mesh in grey, the solids in colour (ray cast)
    for name, view, cut in (("colonna", (1.0, -1.7, 0.55), True), ("muro", (0.6, -1.05, 0.5), False)):
        ch = ppg.Document.open(folder / name / f"{name}.ppg").chunk
        _render_view(ch, folder / f"figure_{name}_3d.png", view, cut)
    _pane_figure(folder)


def _render_view(ch, path: Path, view=(0.6, -1.05, 0.5), cut: bool = False) -> None:
    """The model in grey and the solids in colour, ray cast from ``view`` (times the size of the scene) towards its
    middle; ``cut``: the model left out on the right half (x beyond the middle), where the solid shows alone."""
    import open3d as o3d
    import trimesh

    from ppg_architecture import elements

    o = elements.origin(ch)
    v, f = elements.local_mesh(ch)
    if cut:
        middle = np.median(v[:, 0])
        f = f[v[f].mean(axis=1)[:, 0] < middle]
    meshes = [(v, f, np.array([0.72, 0.72, 0.72]))]
    for k, e in enumerate(elements.elements(ch)):
        m = elements.solid_mesh(ch, e)
        meshes.append((np.asarray(m.vertices), np.asarray(m.faces), np.asarray(elements.COLOURS[k % 6])))
    del o
    rc = o3d.t.geometry.RaycastingScene()
    for vv, ff, _c in meshes:
        rc.add_triangles(o3d.core.Tensor(vv.astype(np.float32)), o3d.core.Tensor(np.asarray(ff, np.uint32)))
    allv = np.vstack([m[0] for m in meshes])
    centre = (allv.min(0) + allv.max(0)) / 2
    size = float(np.ptp(allv, axis=0).max())
    eye = centre + np.asarray(view, float) * size
    from pyarchinit_photogrammetry.core.geometry import look_at

    pose = look_at(eye, centre, up=(0, 0, 1))
    W, H = 1100, 900
    fpx = 1.15 * W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dirs = np.dstack([(xx + 0.5 - W / 2) / fpx, (yy + 0.5 - H / 2) / fpx, np.ones_like(xx)]).reshape(-1, 3)
    dirs = dirs @ pose[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    org = np.repeat(pose[:3, 3][None], len(dirs), 0)
    ans = rc.cast_rays(o3d.core.Tensor(np.hstack([org, dirs]).astype(np.float32)))
    gid = ans["geometry_ids"].numpy().astype(np.int64)
    hit = np.isfinite(ans["t_hit"].numpy())
    nrm = ans["primitive_normals"].numpy()
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(nrm @ light)
    colours = np.array([m[2] for m in meshes])
    img = np.ones((len(dirs), 3))
    g = np.clip(gid, 0, len(meshes) - 1)
    img[hit] = colours[g[hit]] * shade[hit, None]
    # the solids half transparent over the mesh: where a solid is hit, mix with the mesh behind it
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    del trimesh


def _pane_figure(folder: Path) -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import pyarchinit_photogrammetry as ppg

    QApplication.instance() or QApplication([])
    from ppg_architecture.pane import ArchitecturePane
    from ppg_architecture import elements

    pane = ArchitecturePane()
    pane.resize(560, 760)
    pane.show()
    QApplication.processEvents()
    ch = ppg.Document.open(folder / "muro" / "muro.ppg").chunk
    rows = elements.elements(ch) + elements.elements(ppg.Document.open(folder / "colonna" / "colonna.ppg").chunk)
    pane.set_environment("CadQuery 2.8.0 nell'ambiente del plugin.", False, True)
    pane.set_rows(rows)
    pane.table.selectRow(0)
    QApplication.processEvents()
    pane.grab().save(str(folder / "figure_pannello.png"))
    pane.close()


def leggimi(folder: Path, r: dict) -> None:
    def it(x, d):
        return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")

    def mm(x):
        return "-" if x is None else f"{it(x * 1000, 1)} mm"

    kinds = {"pointed": "a sesto acuto", "semicircular": "a tutto sesto", "segmental": "ribassato",
             "elliptical": "ellittico", "measured": "come misurato"}

    def row(*cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def fit(d, keys=("median", "p90", "max")):
        return " / ".join(mm(d.get(k)) for k in keys)

    c, a, k = r["colonna"], r["arco"], r["cornice"]
    head = ["| misura | plugin | verità |", "|---|---|---|"]
    lines = [
        "# Elementi architettonici: prova sulla catena vera", "",
        "Due scene sintetiche fatte dal programma (`testing.synthetic`: mesh con texture, tappetino con target ArUco "
        "per la scala, uno sfondo con texture tutt'intorno), fotografate dal suo ray caster, allineate, densificate "
        "(mappe di profondità \"high\") e trasformate in mesh dal programma; poi il plugin ha misurato gli elementi "
        "sul modello così ottenuto, con CadQuery nel suo ambiente. Script: "
        "`plugins/architecture/proof/make_proof.py`; "
        "numeri completi in `results.json`.", "",
        "«Modello contro verità» è la distanza del modello del programma dalla superficie vera nella zona "
        "dell'elemento: l'errore della fotogrammetria, che il solido eredita e il plugin non può togliere. "
        "«Scarto» è "
        "la distanza del modello dal solido, quella che il plugin riporta per ogni elemento.", "",
        "## Colonna (alta 2 m, raggio 0,30 m alla base, con entasi nota)", "",
        f"{c['foto']} foto ({c['allineate']} allineate). Colonna dal poligono in pianta, 12 nodi.", "", *head,
        row("inclinazione dell'asse", f"{it(c['asse_inclinazione_gradi'], 4)}°", "0°"),
        row("distanza dell'asse dal vero", mm(c["asse_scarto_dall_asse_vero_m"]), "0"),
        row("raggio alla base", mm(c["raggio_base"][0]), mm(c["raggio_base"][1])),
        row("raggio in cima", mm(c["raggio_cima"][0]), mm(c["raggio_cima"][1])),
        row("entasi", mm(c["entasi"][0]), mm(c["entasi"][1])),
        row("nodi: scarto massimo / mediano", f"{mm(c['nodi_scarto_max_m'])} / {mm(c['nodi_scarto_mediano_m'])}", ""),
        row("volume", f"{it(c['volume_m3'][0], 4)} m³",
            f"{it(c['volume_m3'][1], 4)} m³ ({'+' if c['volume_scarto_percento'] >= 0 else ''}"
            f"{it(c['volume_scarto_percento'], 2)} %)"),
        row("scarto modello-solido (mediana / p90 / max)", fit(c["scarto_modello_solido"]), ""),
        row("modello contro verità (mediana / p90 / max)", fit(c["modello_contro_verita"]), ""), "",
        "## Arco a sesto acuto (luce 1,6 m, ghiera spessa 0,25 m) e cornice", "",
        f"{a['foto']} foto ({a['allineate']} allineate). Arco su un piano verticale per due punti dentro la ghiera "
        f"(sporgente 15 cm dal muro), il punto dato nell'apertura come lo si clicca ({it(a['punto_uv'][0], 2)}; "
        f"{it(a['punto_uv'][1], 2)}); il plugin proponeva ({it(a['punto_proposto_dal_plugin_uv'][0], 2)}; "
        f"{it(a['punto_proposto_dal_plugin_uv'][1], 2)}), anch'esso nell'apertura.", "", *head,
        row("tipo", kinds.get(a["tipo"][0], a["tipo"][0]), kinds[a["tipo"][1]]),
        row("luce", mm(a["luce_m"][0]), mm(a["luce_m"][1])),
        row("freccia", mm(a["freccia_m"][0]), mm(a["freccia_m"][1])),
        row("raggio", mm(a["raggio_m"][0]), mm(a["raggio_m"][1])),
        row("linea d'imposta", mm(a["imposta_m"][0]), mm(a["imposta_m"][1])),
        row("spessore della ghiera", mm(a["spessore_m"][0]), mm(a["spessore_m"][1])),
        row("scarto modello-solido sulle facce curve (mediana / p90)", fit(a["scarto_modello_solido"],
                                                                           ("median", "p90")), ""),
        row("punti sulle due facce piane (esclusi dallo scarto)", a["scarto_modello_solido"].get("excluded_faces"), ""),
        row("modello contro verità (mediana / p90)", fit(a["modello_contro_verita"], ("median", "p90")), ""), "",
        "Cornice lungo il muro (sezione a 1 m dall'inizio del percorso, con i valori predefiniti del dialogo: "
        "finestra trovata dal taglio, 10 cm nel muro):", "", *head,
        row("sporgenza dalla faccia del muro", mm(k["sporgenza_m"][0]), mm(k["sporgenza_m"][1])),
        row("altezza della cornice", mm(k["altezza_m"][0]), mm(k["altezza_m"][1])),
        row("area della sezione davanti al muro", f"{it(k['area_davanti_al_muro_m2'][0] * 1e4, 1)} cm²",
            f"{it(k['area_davanti_al_muro_m2'][1] * 1e4, 1)} cm² (+{it(k['area_scarto_percento'], 1)} %)"),
        row("distanza di Hausdorff dei profili", mm(k["hausdorff_profilo_m"]), ""),
        row("scarto modello-solido (mediana / p90)", fit(k["scarto_modello_solido"], ("median", "p90")), ""),
        row("punti del modello sulla faccia del muro o dietro (esclusi dallo scarto)", k["scarto_modello_solido"].get(
            "excluded_behind"), ""),
        row("modello contro verità (mediana / p90 / max)", fit(k["modello_contro_verita"]), ""), "",
        "La sezione della cornice segue il modello: sporgenza, altezza e area in più vengono dal bordo superiore "
        "del muro "
        "e della cornice, ricostruito dal programma fino a 2 cm più alto (lo si vede solo di scorcio dalle foto più "
        "alte; vedi il massimo di «modello contro verità»), non dal plugin, il cui scarto dal modello resta di pochi "
        "millimetri.", "",
        "Figure: `figure_colonna_3d.png`, `figure_muro_3d.png` (il modello in grigio, i solidi a colori), "
        "`figure_pannello.png`, `figure_colonna_profilo.png`, `figure_arco_sezione.png`, `figure_cornice_sezione.png`.",
        ""]
    (folder / "LEGGIMI.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(FOLDER))
    ap.add_argument("--cadquery", help="the Python of an environment with CadQuery")
    ap.add_argument("--stage", default="all", choices=["all", "scenes", "process", "fit", "figures"])
    ns = ap.parse_args()
    folder = Path(ns.folder)
    folder.mkdir(parents=True, exist_ok=True)
    if ns.stage in ("all", "scenes"):
        for name in ("colonna", "muro"):
            render(name, folder)
    if ns.stage in ("all", "process"):
        for name in ("colonna", "muro"):
            print(f"processing {name}", flush=True)
            process(name, folder)
    if ns.stage in ("all", "fit"):
        r = fit(folder, Path(ns.cadquery))
        leggimi(folder, r)
        print(json.dumps(r, indent=1, ensure_ascii=False))
    if ns.stage in ("all", "figures"):
        figures(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
