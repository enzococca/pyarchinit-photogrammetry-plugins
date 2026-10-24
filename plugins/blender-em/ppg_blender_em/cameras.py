"""The cameras of the chunk in Blender, each with its photo behind it, to model on the photos.

Blender's camera is a pinhole: no lens distortion. So each photo is undistorted first, with the
calibration of its sensor (every model of the program, the fisheye ones too, through pycolmap), and
made smaller (2048 pixels on the long side by default): ``RB/<chunk>/foto/<photo>.jpg``. The camera
then gets

* its pose: the camera of the chunk in the world (the similarity of the chunk applied, its scale
  taken out), less the shift of the project;
* its focal length and principal point in pixels of that photo; the live script makes them a focal
  length in millimetres and a shift of the frame of a Blender camera, with the photo as its
  background image.

Looking through a camera in Blender (Numpad 0), the survey and the proxies lie on the photo: a
reconstruction is modelled where the photo shows it. The undistortion map is made once per sensor
and size (on a coarse grid, interpolated), and a photo is written again only when the photo, its
calibration or the size change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAX_SIDE = 2048
_GRID = 8  # pixels between the points of the undistortion map that are projected exactly


def _scale(sensor, max_side: int) -> float:
    return min(1.0, float(max_side) / float(max(sensor.width, sensor.height)))


def output_size(sensor, max_side: int = MAX_SIDE) -> tuple[int, int]:
    s = _scale(sensor, max_side)
    return max(1, int(round(sensor.width * s))), max(1, int(round(sensor.height * s)))


def undistort_map(sensor, max_side: int = MAX_SIDE):
    """``(map_x, map_y, K)``: where each pixel of the undistorted photo (``output_size``) comes from in
    the photo as taken (for ``cv2.remap``), and the pinhole matrix of the undistorted photo. Pixel
    coordinates as COLMAP counts them (the centre of the first pixel at 0.5)."""
    import cv2

    s = _scale(sensor, max_side)
    wo, ho = output_size(sensor, max_side)
    k = sensor.K
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    gw, gh = max(2, wo // _GRID + 2), max(2, ho // _GRID + 2)
    u = np.linspace(0.0, wo, gw)
    v = np.linspace(0.0, ho, gh)
    uu, vv = np.meshgrid(u, v)
    rays = np.column_stack([(uu.ravel() / s - cx) / fx, (vv.ravel() / s - cy) / fy, np.ones(uu.size)])
    src = np.asarray(sensor.project(rays), float).reshape(gh, gw, 2)
    # the coarse grid spans the edges of the photo: resized to the pixel centres (0.5 ... w - 0.5)
    xs = (np.arange(wo) + 0.5) / wo * (gw - 1)
    ys = (np.arange(ho) + 0.5) / ho * (gh - 1)
    gx, gy = (g.astype(np.float32) for g in np.meshgrid(xs, ys))
    map_x = cv2.remap(src[:, :, 0].astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(src[:, :, 1].astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    K = np.array([[fx * s, 0.0, cx * s], [0.0, fy * s, cy * s], [0.0, 0.0, 1.0]])
    return map_x - 0.5, map_y - 0.5, K  # cv2 counts pixels from their centres


def undistort(image, maps):
    """``image`` (as taken, the size of its sensor) undistorted and made smaller with ``maps``."""
    import cv2

    map_x, map_y, _k = maps
    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def world_pose(chunk, camera, shift) -> np.ndarray:
    """``world_from_cam`` of ``camera`` (OpenCV axes) in the world less ``shift``, a rotation without
    the scale of the chunk."""
    t = np.asarray(chunk.transform, float) @ np.asarray(camera.transform, float)
    r = t[:3, :3]
    scale = float(np.cbrt(np.linalg.det(r)))
    out = np.eye(4)
    out[:3, :3] = r / scale
    out[:3, 3] = t[:3, 3] - np.asarray(shift, float)
    return out


def _photo(chunk, camera) -> Path:
    p = Path(camera.path)
    doc = getattr(chunk, "document", None)
    if not p.is_absolute() and doc is not None and getattr(doc, "path", None):
        p = Path(doc.path).parent / p
    return p


def _digest(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()


def blender_cameras(chunk, base: Path, folder: Path, shift, max_side: int = MAX_SIDE, old=None,
                    progress=None) -> dict:
    """The aligned cameras of ``chunk`` as records of the manifest, their photos undistorted into
    ``folder``. ``old``: the records of the manifest before, whose photos are kept when nothing changed.
    Returns ``{"cameras", "resolution", "written", "kept", "missing", "removed"}``."""
    import cv2

    progress = progress or (lambda *a: None)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    before = {r.get("name"): r for r in old or [] if r.get("name")}
    cams = [c for c in chunk.aligned_cameras]
    maps: dict = {}
    out = {"cameras": [], "written": [], "kept": [], "missing": [], "removed": []}
    sizes: dict = {}
    extent = 0.3
    centres = np.array([world_pose(chunk, c, shift)[:3, 3] for c in cams]) if cams else np.zeros((0, 3))
    if len(centres) > 1:  # the cameras drawn at a tenth of the distance between neighbours
        from scipy.spatial import cKDTree

        d, _i = cKDTree(centres).query(centres, k=2)
        extent = float(max(0.05, np.median(d[:, 1]) * 0.4))
    for k, cam in enumerate(cams):
        progress(k / max(len(cams), 1), cam.label)
        sensor = chunk.sensor(cam)
        photo = _photo(chunk, cam)
        if not photo.exists():
            out["missing"].append(cam.label)
            continue
        st = photo.stat()
        key = (sensor.key, max_side)
        source = _digest(str(photo), st.st_size, st.st_mtime_ns, sensor.model, list(map(float, sensor.params)),
                         sensor.width, sensor.height, max_side)
        name = "".join("_" if c in '\\/:*?"<>|' else c for c in Path(cam.label).stem) or f"camera_{cam.key}"
        target = folder / f"{name}.jpg"
        rel = target.relative_to(base).as_posix()
        if key not in maps:
            maps[key] = undistort_map(sensor, max_side)
        kmat = maps[key][2]
        wo, ho = output_size(sensor, max_side)
        prev = before.get(name)
        if prev is not None and prev.get("source") == source and target.exists():
            out["kept"].append(name)
        else:
            image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
            if image is None:
                out["missing"].append(cam.label)
                continue
            if image.shape[1] != sensor.width or image.shape[0] != sensor.height:
                image = cv2.resize(image, (int(sensor.width), int(sensor.height)), interpolation=cv2.INTER_AREA)
            part = target.with_name(target.stem + ".part.jpg")
            cv2.imwrite(str(part), undistort(image, maps[key]), [cv2.IMWRITE_JPEG_QUALITY, 90])
            part.replace(target)
            out["written"].append(name)
        sizes[(wo, ho)] = sizes.get((wo, ho), 0) + 1
        record = {"name": name, "kind": "camera", "file": rel, "photo": str(photo),
                  "matrix": world_pose(chunk, cam, shift).tolist(), "width": wo, "height": ho,
                  "fx": float(kmat[0, 0]), "fy": float(kmat[1, 1]), "cx": float(kmat[0, 2]), "cy": float(kmat[1, 2]),
                  "display_size": extent, "source": source}
        record["sha1"] = _digest(record["matrix"], source)
        out["cameras"].append(record)
    names = {r["name"] for r in out["cameras"]}
    for name, prev in before.items():  # photos written here before, for cameras gone
        if name not in names and prev.get("file"):
            f = Path(base) / prev["file"]
            if f.exists():
                f.unlink()
                out["removed"].append(name)
    out["resolution"] = list(max(sizes, key=sizes.get)) if sizes else None
    progress(1.0, "camere")
    return out


__all__ = ["MAX_SIDE", "output_size", "undistort_map", "undistort", "world_pose", "blender_cameras"]
