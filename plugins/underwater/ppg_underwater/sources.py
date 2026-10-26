"""Camera calibrations and the folders of ROV datasets.

ROVs and research datasets come with the calibration of their camera from Kalibr (the YAML of
``kalibr_calibrate_cameras``: ``camera_model`` pinhole, ``distortion_model`` radtan or
equidistant, ``intrinsics`` fu, fv, pu, pv, ``resolution``) and with the images of a run as a
folder of frames, a table of their times and the logs of the sensors - the raw format of the AQUALOC
dataset (M. Ferrera, V. Creuze, J. Moras, P. Trouvé-Peloux, *AQUALOC: An Underwater Dataset for
Visual-Inertial-Pressure Localization*, The International Journal of Robotics Research, 2019):
``images_sequence_N/frameNNNNNN.png``, ``img_sequence_N.csv`` (``#timestamp [ns], frame_id``) and
``depth_sequence_N.csv`` (``#timestamp [ns], depth [m]``).

A Kalibr calibration becomes the sensor of the photos it fits (same size): the OpenCV model of
COLMAP (radtan: k1, k2, p1, p2) or OPENCV_FISHEYE (equidistant), held fixed in the alignment.
Kalibr, like OpenCV, puts the centre of the first pixel at 0; COLMAP at 0.5: the principal point
moves by half a pixel.
"""
from __future__ import annotations

import re
from pathlib import Path

from .frames import IMAGE_EXTENSIONS
from .texts import t


def _yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:  # the few keys of a Kalibr file, read by hand
        out, current = {}, None
        joined = re.sub(r",\s*\n\s*", ", ", text)
        for line in joined.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not line.startswith(" "):
                current = line.split(":")[0].strip()
                out[current] = {}
                continue
            key, _, value = line.strip().partition(":")
            value = value.strip()
            if value.startswith("["):
                items = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
                try:
                    out[current][key] = [float(v) for v in items]
                except ValueError:
                    out[current][key] = items
            else:
                out[current][key] = value
        return out


def read_kalibr(path, camera: str = "cam0") -> dict:
    """``{"model", "params" (COLMAP), "width", "height"}`` of a camera of a Kalibr YAML."""
    data = _yaml(Path(path))
    cam = data.get(camera) or next((v for k, v in data.items() if isinstance(v, dict) and "intrinsics" in v), None)
    if not cam:
        raise ValueError(t("{0}: nessuna camera Kalibr", "{0}: no Kalibr camera").format(Path(path).name))
    fu, fv, pu, pv = (float(v) for v in cam["intrinsics"][:4])
    w, h = (int(v) for v in cam["resolution"][:2])
    dist = [float(v) for v in cam.get("distortion_coeffs") or []]
    model = str(cam.get("distortion_model", "radtan")).lower()
    cx, cy = pu + 0.5, pv + 0.5
    if model in ("equidistant", "equi"):
        params = [fu, fv, cx, cy] + (dist + [0.0] * 4)[:4]
        name = "OPENCV_FISHEYE"
    elif model in ("radtan", "plumb_bob") or dist:
        params = [fu, fv, cx, cy] + (dist + [0.0] * 4)[:4]
        name = "OPENCV"
    else:
        params = [fu, fv, cx, cy]
        name = "PINHOLE"
    return {"model": name, "params": params, "width": w, "height": h, "source": Path(path).name}


def apply_calibration(chunk, calib: dict, fixed: bool = True) -> int:
    """The calibration as the sensor of the photos of the same size; returns the sensors changed."""
    n = 0
    for sensor in chunk.sensors.values():
        if (int(sensor.width), int(sensor.height)) != (calib["width"], calib["height"]):
            continue
        sensor.model = calib["model"]
        sensor.params = [float(v) for v in calib["params"]]
        sensor.fixed = bool(fixed)
        n += 1
    return n


def find_dataset(folder) -> dict:
    """What a folder of a run holds: ``{"images": folder, "times": table, "depth": table,
    "calibration": yaml}`` (None for what is missing): the subfolder with the most images, the
    table that names image files, the table with a depth or pressure column, a Kalibr YAML."""
    folder = Path(folder)
    best, count = None, 0
    for d in [folder] + [p for p in folder.rglob("*") if p.is_dir()]:
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if n > count:
            best, count = d, n
    times = depth = calib = None
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in (".csv", ".txt") and p.is_file():
            head = p.open(encoding="utf-8", errors="replace").read(400).lower()
            if re.search(r"\.(png|jpe?g|tiff?|bmp)\b", head) and times is None:
                times = p
            elif re.search(r"depth|pressure|profondit", head) and depth is None:
                depth = p
        elif p.suffix.lower() in (".yaml", ".yml") and calib is None:
            if "intrinsics" in p.read_text(encoding="utf-8", errors="replace"):
                calib = p
    for p in sorted(folder.parent.glob("*.yaml")) if calib is None else []:
        text = p.read_text(encoding="utf-8", errors="replace")
        if "intrinsics" in text and "T_cam_imu" not in text:
            calib = p
            break
    return {"images": best, "count": count, "times": times, "depth": depth, "calibration": calib}
