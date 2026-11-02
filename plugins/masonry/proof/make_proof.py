"""The plugin through the real pipeline, on a synthetic wall rendered by the program.

A wall 6 m long and 2.6 m high on a printed mat with coded targets (the scale and the frame): below, five courses of
stones of known outlines (the generator of the tests: joints 2.5 cm, recessed 1.5 cm, the stones' edges bevelled),
above them a second unit of rubble with no courses; the face leans 2 degrees towards the viewer and bulges 3 cm
(Gaussian, centred at u = 3 m, h = 1.3 m). Photographed by the program's ray caster (``testing.synthetic.render_mesh``)
in three arcs, aligned, densified and meshed by the program; then the plugin:

* a face on the vertical plane through two points of the plan (as *Faccia su un piano*);
* its shape: lean and bulge against the truth, horizontal sections with their offsets;
* the stones: with the program's stone tool (``archaeo.stones.find_stones``, SAM 2.1 at tiles) on the face's
  elevation when the SAM environment of the program exists on this computer, else the generated outlines are put on
  the face as drawn stones (and ``LEGGIMI.md`` says so); their measures, courses and mortar against the truth;
* two USM drawn on the face with a known relationship (the rubble rests on the coursed wall), the proposal;
* the plate (SVG, PDF), the DXF, the GeoPackages, the IFC (validated).

    python make_proof.py [--folder /Users/Shared/ppg_demo_masonry] [--stage all|scene|process|plugin|figures]
                         [--no-sam]

Each stage keeps what it made. ``results.json`` and ``LEGGIMI.md`` hold the numbers, ``figura_*.png`` the figures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "tests")]

FOLDER = Path("/Users/Shared/ppg_demo_masonry")
SIZE = (1600, 1200)
FOCAL = 1500.0
X0, LENGTH, HEIGHT, DEPTH = -3.0, 6.0, 2.6, 0.6
LEAN_DEG, BULGE, BULGE_AT, BULGE_SIGMA = 2.0, 0.03, (3.0, 1.3), 0.5
JOINT_DEPTH, BEVEL = 0.015, 0.006
STEP = 0.005  # the grid of the face's mesh (m)
TOP_OF_COURSES = 5 * 0.35


# the mat, the noise and the backdrop of the architecture plugin's proof
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


def stones_truth():
    import masonry_samples as S

    return S.stone_pattern(0.0)


def bulge(u, h):
    return BULGE * np.exp(-((u - BULGE_AT[0]) ** 2 + (h - BULGE_AT[1]) ** 2) / (2 * BULGE_SIGMA ** 2))


def _grid_masks():
    """Rasters of the face at ``STEP`` (rows = h from the bottom, cols = u): the stone id (0 = mortar, -1 above the
    courses) and the distance from the stone's edge inside it (m)."""
    from rasterio import features
    from rasterio.transform import from_origin
    from scipy.ndimage import distance_transform_edt

    polys, _truth = stones_truth()
    nu, nh = int(round(LENGTH / STEP)) + 1, int(round(HEIGHT / STEP)) + 1
    tr = from_origin(-STEP / 2, -STEP / 2, STEP, -STEP)  # row 0 at h = 0, growing upwards
    ids = features.rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(nh, nu), transform=tr,
                             fill=0, dtype="int32")
    h = np.arange(nh) * STEP
    ids[h > TOP_OF_COURSES + 1e-9, :] = -1
    inside = distance_transform_edt(ids > 0) * STEP
    return ids, inside


def wall_mesh():
    """The face (a grid with the lean, the bulge and the recessed joints) and the slab's sides and back."""
    ids, inside = _grid_masks()
    nh, nu = ids.shape
    u = np.arange(nu) * STEP
    h = np.arange(nh) * STEP
    U, H = np.meshgrid(u, h)
    relief = np.where(ids > 0, -JOINT_DEPTH * np.clip(1 - inside / BEVEL, 0, 1), np.where(ids == 0, -JOINT_DEPTH, 0.0))
    d = H * np.tan(np.radians(LEAN_DEG)) + bulge(U, H) + relief
    verts = np.column_stack([X0 + U.ravel(), -d.ravel(), H.ravel()])
    idx = np.arange(nh * nu).reshape(nh, nu)
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, e])])
    # the back and the sides of the slab (the face's top edge is joined to the top of the back)
    n = len(verts)
    x1 = X0 + LENGTH
    box = np.array([[X0, DEPTH, 0], [x1, DEPTH, 0], [x1, DEPTH, HEIGHT], [X0, DEPTH, HEIGHT],
                    [X0, -0.15, 0], [x1, -0.15, 0], [X0, -0.15, HEIGHT], [x1, -0.15, HEIGHT]], float)
    bf = np.array([[0, 2, 1], [0, 3, 2],  # back
                   [4, 0, 3], [4, 3, 6],  # left side
                   [5, 2, 1], [5, 7, 2],  # right side
                   [6, 3, 2], [6, 2, 7]]) + n  # top
    return np.vstack([verts, box]), np.vstack([faces, bf])


def scene():
    from pyarchinit_photogrammetry.core.geometry import look_at
    from pyarchinit_photogrammetry.testing.synthetic import MeshScene

    rng = np.random.default_rng(11)
    ids, _inside = _grid_masks()
    polys, _truth = stones_truth()
    tints = rng.uniform(0.75, 1.25, (len(polys) + 1, 3)) * np.array([95.0, 125.0, 160.0])
    stone_noise = _noise3d(rng, scales=(0.003, 0.01, 0.03, 0.08), amps=(0.35, 0.3, 0.2, 0.15), tint=(1.0, 1.0, 1.0))
    mortar_noise = _noise3d(rng, scales=(0.002, 0.006, 0.02, 0.05), tint=(1.0, 1.0, 1.0))
    rubble = _noise3d(rng, scales=(0.004, 0.015, 0.05, 0.12), tint=(80.0, 110.0, 135.0))
    plain = _noise3d(rng)
    nh, nu = ids.shape

    def colour(p):
        out = plain(p)
        u, h = p[:, 0] - X0, p[:, 2]
        on = (np.abs(p[:, 1]) < 0.2) & (u >= 0) & (u <= LENGTH) & (h >= 0) & (h <= HEIGHT)
        if on.any():
            c = np.clip(np.round(u[on] / STEP).astype(int), 0, nu - 1)
            r = np.clip(np.round(h[on] / STEP).astype(int), 0, nh - 1)
            k = ids[r, c]
            q = p[on]
            col = np.empty((len(q), 3))
            st = k > 0
            col[st] = tints[k[st]] * (0.3 + 0.5 * stone_noise(q[st]))
            mo = k == 0
            col[mo] = np.array([205.0, 210.0, 212.0]) * (0.7 + 0.25 * mortar_noise(q[mo]))
            ru = k < 0
            col[ru] = rubble(q[ru])
            out[on] = np.clip(col, 0, 255)
        return out

    targets, k = {}, 0
    for x in (-3.2, -1.6, 0.0, 1.6, 3.2):
        for y in (-0.45, -1.3):  # two rows: the wall's frame is not a line
            targets[k] = (x, y)
            k += 1
    mv, mf, mtex = _mat(rng, 8.6, 0.003, targets, 0.16)
    wv, wf = wall_mesh()
    v = np.vstack([mv, wv])
    f = np.vstack([mf, wf + len(mv)])
    v, f, colour3 = _with_backdrop(v, f, colour, rng, radius=8.0)
    sc = MeshScene(v.astype(np.float32), f.astype(np.uint32), len(mf), 8.6, 0.003, mtex, colour3,
                   targets={c: (x, y, 0.0) for c, (x, y) in targets.items()})
    poses = []
    for height, target_z, r, count in ((0.9, 1.0, 4.2, 13), (1.9, 1.3, 4.2, 13), (3.4, 1.4, 4.0, 11)):
        for a in np.radians(np.linspace(-55, 55, count)):
            eye = np.array([r * np.sin(a), -r * np.cos(a), height])
            poses.append(look_at(eye, np.array([0.0, 0.0, target_z]), up=(0, 0, 1)))
    for x in (-2.2, -0.8, 0.8, 2.2):  # closer, for the stones
        for z in (0.6, 1.6):
            poses.append(look_at(np.array([x, -2.4, z]), np.array([x, 0.0, z]), up=(0, 0, 1)))
    return sc, poses


def render(folder: Path) -> None:
    from pyarchinit_photogrammetry.testing.synthetic import render_mesh

    out = folder / "foto"
    if out.exists() and len(list(out.glob("*.jpg"))) > 30:
        return
    out.mkdir(parents=True, exist_ok=True)
    sc, poses = scene()
    K = np.array([[FOCAL, 0, SIZE[0] / 2], [0, FOCAL, SIZE[1] / 2], [0, 0, 1]])
    for i, pose in enumerate(poses):
        img = render_mesh(sc, pose, K, SIZE, -0.02)
        cv2.imwrite(str(out / f"MURO_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"photo {i + 1}/{len(poses)}", flush=True)
    with open(folder / "target.csv", "w") as fh:
        fh.write("# label,X,Y,Z (LOCAL, metres)\n")
        for code, (x, y, z) in sc.targets.items():
            fh.write(f"{code},{x:.5f},{y:.5f},{z:.5f}\n")


def _progress(phase):
    last = [-1]

    def cb(fraction, message=""):
        k = int(fraction * 10)
        if k != last[0]:
            last[0] = k
            print(f"  {phase} {fraction * 100:5.1f} % {message}", flush=True)

    return cb


def process(folder: Path) -> Path:
    import pyarchinit_photogrammetry as ppg

    project = folder / "muro.ppg"
    if project.exists():
        doc = ppg.Document.open(project)
        if doc.chunk.product_path("model") is not None and doc.chunk.product_path("model").exists():
            return project
    doc = ppg.Document()
    ch = doc.add_chunk("Muro")
    doc.save(project)
    ch.add_photos([folder / "foto"])
    ch.detect_markers("aruco_4x4_50")
    ch.import_reference(folder / "target.csv", crs="LOCAL")
    doc.save()
    times = {}
    t = time.monotonic()
    ch.match_photos(quality="high", progress=_progress("match"))
    ch.align_cameras(progress=_progress("align"))
    ch.update_transform()
    if np.allclose(ch.transform, np.eye(4)):
        raise RuntimeError("the targets did not place the survey: check that at least three, not on a line, are seen")
    times["align_s"] = time.monotonic() - t
    t = time.monotonic()
    ch.build_depth_maps(quality="high", filter_mode="mild", progress=_progress("depth"))
    ch.build_point_cloud(progress=_progress("cloud"))
    ch.crop_point_cloud(zmin=0.01, bounds=(X0 - 0.3, -0.6, X0 + LENGTH + 0.3, 0.2))
    ch.build_model(face_count=1500000, progress=_progress("mesh"))
    times["dense_mesh_s"] = time.monotonic() - t
    ch.meta["proof_times"] = times
    doc.save()
    _drop_workspaces(ch)
    return project


def _drop_workspaces(ch) -> None:
    """The depth maps' workspaces of the dense step: gigabytes not needed once the model is made."""
    for name in ("dense", "mvs", "depth_maps"):
        d = Path(ch.data_dir) / name
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(FOLDER))
    ap.add_argument("--stage", default="all", choices=["all", "scene", "process", "plugin", "figures"])
    ap.add_argument("--no-sam", action="store_true", help="the generated outlines as drawn stones, SAM not used")
    ns = ap.parse_args()
    folder = Path(ns.folder)
    folder.mkdir(parents=True, exist_ok=True)
    if ns.stage in ("all", "scene"):
        render(folder)
    if ns.stage in ("all", "process"):
        process(folder)
    if ns.stage in ("all", "plugin"):
        from proof_plugin import run_plugin

        r = run_plugin(folder, use_sam=not ns.no_sam)
        print(json.dumps(r, indent=1, ensure_ascii=False))
    if ns.stage in ("all", "figures"):
        from proof_plugin import figures

        figures(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
