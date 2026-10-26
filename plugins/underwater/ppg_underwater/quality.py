"""Which photos are worth using: sharpness, contrast and veil of every photo, with thresholds.

Turbid water takes three things from a photo, and each has its score here, measured on a copy at
most 1000 px on its long side:

* **quality** - the program's sharpness (``engine.quality.sharpness``: the finest detail over the
  detail a few pixels wide, in the cells of a grid with some texture) over the sharpness of the
  sharpest photos of the chunk (their 90th percentile), at most 1, as *Stima la qualità delle
  immagini* computes it and keeps it (``camera.meta["quality"]``): motion blur, focus;
* **contrast** - the spread of the lightness, between its 2nd and 98th percentiles, 0-1: low in a
  photo taken through a milky water, or too dark;
* **veil** - the light the particles add between the camera and the scene (backscatter, haze): the
  median of the dark channel - the darkest of green and blue (red is absorbed anyway: the
  underwater dark channel prior of Drews et al.) in a window of 15 px - over the brightest
  light of the photo, 0-1: a clear photo has dark pixels in every patch (shadows, crevices), a
  hazy one does not.

A photo is flagged when its quality is under the minimum, its contrast under the minimum or its
veil over the maximum; :func:`exclude` turns the flagged photos off (``camera.enabled``, as
*Disattiva foto* does, so the alignment and the products leave them out) and :func:`include` turns
on again the ones it turned off.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .texts import t

QUALITY_MIN = 0.5
CONTRAST_MIN = 0.12
VEIL_MAX = 0.6
SIDE = 1000


def measure(img: np.ndarray) -> dict:
    """Sharpness, contrast and veil of an image (8-bit BGR or grey)."""
    from pyarchinit_photogrammetry.engine.quality import sharpness

    h, w = img.shape[:2]
    f = min(1.0, SIDE / max(h, w))
    if f < 1:
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lum = cv2.GaussianBlur(grey, (3, 3), 0).astype(np.float32) / 255.0
    p2, p98, p99 = np.percentile(lum, [2, 98, 99.5])
    dark = grey if img.ndim == 2 else np.minimum(img[..., 0], img[..., 1])
    dark = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).astype(np.float32) / 255.0
    veil = float(np.median(dark) / max(p99, 1e-3))
    over = float(np.mean(grey >= 252))
    under = float(np.mean(grey <= 3))
    return {"sharpness": float(sharpness(img)), "contrast": float(p98 - p2), "veil": min(veil, 1.0),
            "overexposed": over, "underexposed": under}


def _read(path) -> np.ndarray | None:
    path = str(path)
    flag = cv2.IMREAD_REDUCED_COLOR_2 if path.lower().endswith((".jpg", ".jpeg")) else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is not None and img.ndim == 3 and np.array_equal(img[..., 0], img[..., 2]) and \
            np.array_equal(img[..., 0], img[..., 1]):
        img = img[..., 0]  # a monochrome photo
    return img


def score_paths(paths, progress=None, workers: int = 4) -> list[dict | None]:
    """The scores of image files (None for an unreadable one); quality relative to the sharpest."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    paths = list(paths)
    out: list[dict | None] = [None] * len(paths)

    def one(i):
        progress.check()
        img = _read(paths[i])
        return i, (measure(img) if img is not None else None)

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, s in pool.map(one, range(len(paths))):
            out[i] = s
            done += 1
            if done % 5 == 0 or done == len(paths):
                progress(done / max(len(paths), 1), t("Valutazione delle foto {0}/{1}", "Scoring the photos {0}/{1}").format(
                    done, len(paths)))
    relative_quality(out)
    return out


def relative_quality(scores: list[dict | None]) -> float:
    """``quality`` of every score: its sharpness over the 90th percentile of all, at most 1."""
    known = [s["sharpness"] for s in scores if s is not None]
    ref = float(np.percentile(known, 90)) if known else 0.0
    for s in scores:
        if s is not None:
            s["quality"] = round(min(1.0, s["sharpness"] / ref), 3) if ref > 0 else 0.0
    return ref


def thresholds(chunk=None, quality: float | None = None, contrast: float | None = None, veil: float | None = None) -> dict:
    """The limits given, else those kept with the chunk, else the defaults."""
    stored = ((chunk.meta.get("underwater") or {}).get("thresholds") or {}) if chunk is not None else {}
    return {"quality": float(quality if quality is not None else stored.get("quality", QUALITY_MIN)),
            "contrast": float(contrast if contrast is not None else stored.get("contrast", CONTRAST_MIN)),
            "veil": float(veil if veil is not None else stored.get("veil", VEIL_MAX))}


def reasons(score: dict | None, limits: dict) -> list[str]:
    """Why a photo is flagged: ``blur``, ``contrast``, ``veil``, ``unreadable`` (empty: usable)."""
    if score is None:
        return ["unreadable"]
    out = []
    if score.get("quality") is not None and score["quality"] < limits["quality"]:
        out.append("blur")
    if score["contrast"] < limits["contrast"]:
        out.append("contrast")
    if score["veil"] > limits["veil"]:
        out.append("veil")
    return out


REASON_TEXT = {"blur": ("mossa o sfocata", "blurred"), "contrast": ("contrasto basso", "low contrast"),
               "veil": ("velata (torbida)", "hazy (turbid)"), "unreadable": ("illeggibile", "unreadable")}


def reason_text(codes) -> str:
    return ", ".join(t(*REASON_TEXT[c]) for c in codes)


def score_chunk(chunk, cams=None, limits: dict | None = None, progress=None) -> dict:
    """Score the photos of ``chunk`` (all by default): ``camera.meta["underwater"]["scores"]``, the
    program's ``meta["sharpness"]`` and ``meta["quality"]`` too, and the flags with the limits.
    Returns ``{"photos", "flagged", "limits", "reference"}``."""
    from .photos import original

    cams = list(chunk.cameras if cams is None else cams)
    limits = limits or thresholds(chunk)
    scores = score_paths([original(c) for c in cams], progress)
    # quality against the sharpest photos of the whole chunk, as the program measures it
    for cam, s in zip(cams, scores):
        if s is not None:
            cam.meta["sharpness"] = s["sharpness"]
    known = [c.meta["sharpness"] for c in chunk.cameras if c.meta.get("sharpness") is not None]
    ref = float(np.percentile(known, 90)) if known else 0.0
    flagged = 0
    for cam, s in zip(cams, scores):
        rec = cam.meta.setdefault("underwater", {})
        if s is not None:
            s["quality"] = round(min(1.0, s["sharpness"] / ref), 3) if ref > 0 else 0.0
            cam.meta["quality"] = s["quality"]
            rec["scores"] = {k: round(float(v), 4) for k, v in s.items()}
        else:
            rec["scores"] = None
        rec["flags"] = reasons(s, limits)
        flagged += bool(rec["flags"])
    chunk.meta.setdefault("underwater", {})["thresholds"] = dict(limits)
    return {"photos": len(cams), "flagged": flagged, "limits": limits, "reference": ref}


def reflag(chunk, limits: dict) -> int:
    """The flags of the photos scored, again with other limits (nothing measured again)."""
    n = 0
    for cam in chunk.cameras:
        rec = cam.meta.get("underwater") or {}
        if "scores" not in rec:
            continue
        rec["flags"] = reasons(rec["scores"], limits)
        n += bool(rec["flags"])
    chunk.meta.setdefault("underwater", {})["thresholds"] = dict(limits)
    return n


def flagged(chunk) -> list:
    return [c for c in chunk.cameras if (c.meta.get("underwater") or {}).get("flags")]


def exclude(chunk) -> int:
    """Turn off the flagged photos; returns how many were turned off now."""
    n = 0
    for cam in flagged(chunk):
        if cam.enabled:
            cam.enabled = False
            cam.meta["underwater"]["excluded"] = True
            n += 1
    return n


def include(chunk) -> int:
    """Turn on again the photos :func:`exclude` turned off."""
    n = 0
    for cam in chunk.cameras:
        rec = cam.meta.get("underwater") or {}
        if rec.pop("excluded", False):
            cam.enabled = True
            n += 1
    return n


def write_csv(chunk, path) -> Path:
    """The scores, flags, times and depths of the photos as a CSV table."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "enabled", "quality", "sharpness", "contrast", "veil", "flags", "time", "depth",
                     "depth_residual", "corrected"])
        for cam in chunk.cameras:
            rec = cam.meta.get("underwater") or {}
            s = rec.get("scores") or {}
            w.writerow([cam.label, int(cam.enabled), s.get("quality", ""), s.get("sharpness", ""), s.get("contrast", ""),
                        s.get("veil", ""), " ".join(rec.get("flags") or []), rec.get("time", ""), rec.get("depth", ""),
                        rec.get("depth_residual", ""), rec.get("colour", "")])
    return path
