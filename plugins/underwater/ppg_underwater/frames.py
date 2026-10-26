"""Photos from videos and from fast image sequences: sharp frames, well spaced by the motion.

An ROV logs images at 10-30 per second and a diver films: hundreds of frames for every one a
reconstruction needs, many blurred by the motion or by the swell. Frames are chosen here by how far
the view has moved since the last frame kept and by how sharp they are, not by the clock alone (an
ROV hovering over a find gives no new frames; one crossing fast gives them close together).

**Image sequences** (:func:`select_sequence`): every frame (or one in ``step``) is reduced to
480 px; its ORB points are matched with those of the frame before and a similarity (rotation,
scale, shift: ``cv2.estimateAffinePartial2D`` with RANSAC) chains the frames of a stretch together.
The motion between two frames is how far the corners and the centre of the first moved in the
second, as a share of the width. From a frame kept, the next one is the sharpest (the program's
sharpness) among the frames that moved by half to all of ``1 - overlap`` of the width, those much
less sharp than their neighbours (under ``min_sharpness`` of the median around them: motion blur)
left out when others are there. Where the frames cannot be matched (a jump, a cloud of sediment, the
lights off) a new stretch starts. ``max_gap`` seconds force a frame where the view barely moves.

**Videos** (:func:`extract_video`): the program's own extraction (``engine.video.extract_frames``:
a frame each time the view has moved by ``1 - overlap``, the sharpest of the moments after), with
the time of every frame: the creation time of the video (the ``mvhd`` box of MP4/MOV files, the
clock of the camera, often local time written as UTC) plus the time in the video.

The frames added to a chunk keep their time (``camera.meta["underwater"]["time"]``) to be matched
with a depth log.
"""
from __future__ import annotations

import json
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .texts import t

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SIDE = 480
MIN_MATCHES = 15


# ------------------------------------------------------------------ sequences
def list_images(folder) -> list[Path]:
    """The images of a folder in their order (natural order of the names: frame2 before frame10)."""
    def key(p: Path):
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]

    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()), key=key)


def read_frame_times(path, images: list[Path] | None = None) -> dict[str, float]:
    """``{image file name: seconds}`` from a table of the frames of a sequence: a time column (Unix
    time in s/ms/us/ns or ISO dates) and a column with the file names, or with the frame numbers (then
    matched to the ``images`` by the number in their name, or by their position)."""
    from .depthlog import _number, _rows, epoch_from_number, parse_datetime, _unit

    rows = _rows(Path(path).read_text(encoding="utf-8-sig", errors="replace"))
    header = None
    if rows and any(re.search(r"[A-Za-z]", c) and not re.search(r"\.(png|jpe?g|tiff?|bmp)$", c.strip(), re.I)
                    for c in rows[0]):
        header = [c.lstrip("#").strip() for c in rows[0]]
        rows = rows[1:]
    rows = [r for r in rows if r and not r[0].lstrip().startswith("#")]
    if not rows:
        return {}
    width = max(len(r) for r in rows)
    name_i = next((i for i in range(width) if any(re.search(r"\.(png|jpe?g|tiff?|bmp)$", r[i].strip(), re.I)
                                                  for r in rows[:5] if len(r) > i)), None)
    time_i = None
    units = [(_unit(h) if header else None) for h in (header or [""] * width)]
    for i in range(width):
        if i == name_i:
            continue
        vals = [r[i] for r in rows[:5] if len(r) > i]
        if vals and all(_number(v) is not None and epoch_from_number(_number(v), units[i])[1] for v in vals):
            time_i = i
            break
        if vals and all(parse_datetime(v) is not None for v in vals):
            time_i = i
            break
    if time_i is None:
        raise ValueError(t("{0}: nessuna colonna con l'ora dei fotogrammi", "{0}: no column with the time of the frames").format(
            Path(path).name))
    index_i = None
    if name_i is None:
        index_i = next((i for i in range(width) if i != time_i and all(
            (_number(r[i]) is not None and float(_number(r[i])).is_integer()) for r in rows[:5] if len(r) > i)), None)
    out: dict[str, float] = {}
    by_number = {}
    if images is not None:
        for k, p in enumerate(images):
            nums = re.findall(r"\d+", p.stem)
            if nums:
                by_number.setdefault(int(nums[-1]), p.name)
    for k, r in enumerate(rows):
        if len(r) <= time_i:
            continue
        v = _number(r[time_i])
        sec = epoch_from_number(v, units[time_i])[0] if v is not None else parse_datetime(r[time_i])
        if sec is None:
            continue
        if name_i is not None and len(r) > name_i:
            out[Path(r[name_i].strip()).name] = sec
        elif images is not None:
            if index_i is not None and len(r) > index_i:
                n = int(float(_number(r[index_i])))
                name = by_number.get(n)
            else:
                name = images[k].name if k < len(images) else None
            if name:
                out[name] = sec
    return out


def _small(path: Path, side: int = SIDE):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    h, w = img.shape
    f = min(1.0, side / w)
    if f < 1:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img


def _features(path: Path):
    from pyarchinit_photogrammetry.engine.quality import sharpness

    img = _small(path)
    if img is None:
        return None
    orb = cv2.ORB_create(1000)
    # local contrast first: the ORB detector finds little in a dim, milky frame
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(img)
    kp, des = orb.detectAndCompute(eq, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return {"size": img.shape[::-1], "pts": pts, "des": des, "sharpness": float(sharpness(img, max_side=SIDE))}


def _step(a: dict, b: dict) -> np.ndarray | None:
    """The 3x3 similarity taking the points of frame ``a`` to frame ``b`` (None: not matched)."""
    if a["des"] is None or b["des"] is None or len(a["pts"]) < MIN_MATCHES or len(b["pts"]) < MIN_MATCHES:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(a["des"], b["des"])
    if len(matches) < MIN_MATCHES:
        return None
    src = a["pts"][[m.queryIdx for m in matches]]
    dst = b["pts"][[m.trainIdx for m in matches]]
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        return None
    out = np.eye(3)
    out[:2] = m
    return out


def change(m: np.ndarray, size) -> float:
    """How far the view moved under ``m`` (3x3, frame to frame): the mean shift of the corners and
    the centre, as a share of the width."""
    w, h = size
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]], float)
    moved = pts @ m.T
    return float(np.mean(np.hypot(moved[:, 0] - pts[:, 0], moved[:, 1] - pts[:, 1])) / w)


def select_sequence(paths, times: dict | None = None, overlap: float = 0.7, min_sharpness: float = 0.6,
                    step: int = 1, max_gap: float | None = None, progress=None, workers: int = 4) -> dict:
    """Frames of an image sequence chosen by motion and sharpness (see the module). ``times``:
    ``{file name: seconds}``. Returns ``{"frames": [{"path", "index", "time", "sharpness",
    "change"}], "analysed", "segments", "unmatched"}``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    paths = [Path(p) for p in paths][::max(1, int(step))]
    n = len(paths)
    if not n:
        return {"frames": [], "analysed": 0, "segments": 0, "unmatched": 0}
    feats: list = [None] * n
    cum = [np.eye(3) for _ in range(n)]
    segment = np.zeros(n, int)
    seg = 0
    unmatched = 0
    with ThreadPoolExecutor(workers) as pool:
        prev = None
        for i, f in enumerate(pool.map(_features, paths)):
            if i % 20 == 0:
                progress(0.9 * i / n, t("Fotogrammi analizzati {0}/{1}", "Frames analysed {0}/{1}").format(i, n))
            feats[i] = None if f is None else {"size": f["size"], "sharpness": f["sharpness"]}
            if f is None:
                seg += 1
                segment[i] = seg
                prev = None
                continue
            m = _step(prev, f) if prev is not None else None
            if prev is not None and m is None:
                seg += 1
                unmatched += 1
            segment[i] = seg
            cum[i] = (m @ cum[i - 1]) if m is not None else np.eye(3)
            prev = f
    sharp = np.array([f["sharpness"] if f else 0.0 for f in feats])
    from scipy.ndimage import median_filter

    local = median_filter(sharp, size=51, mode="nearest")
    eligible = sharp >= min_sharpness * local
    target = max(0.02, 1.0 - overlap)
    stamps = [None if times is None else times.get(p.name) for p in paths]
    kept: list[int] = []

    def moved(k: int, j: int) -> float:
        return change(cum[j] @ np.linalg.inv(cum[k]), feats[j]["size"])

    def best(cands: list[int]) -> int:
        good = [j for j in cands if eligible[j]] or cands
        return max(good, key=lambda j: sharp[j])

    for s in np.unique(segment):
        idx = [i for i in np.flatnonzero(segment == s) if feats[i] is not None]
        if not idx:
            continue
        # the first frame of the stretch: the sharpest before it moved by half the step
        first = [j for j in idx if moved(idx[0], j) <= 0.5 * target] or idx[:1]
        k = best(first)
        kept.append(k)
        pos = idx.index(k)
        while pos < len(idx) - 1:
            within, before, beyond, tail = [], [], None, []
            for j in idx[pos + 1:]:
                c = moved(k, j)
                if c > target:
                    beyond = j
                    break
                before.append(j)
                if c >= 0.5 * target:
                    within.append(j)
                if c >= 0.3 * target:
                    tail.append(j)
            if beyond is None and not within:
                if tail:  # the end of the stretch: one more frame, it moved enough since the last one
                    kept.append(best(tail))
                break
            if not within:  # moving fast: the frame before the step is passed, or the first after it
                within = [before[-1]] if before else [beyond]
            if max_gap and stamps[k] is not None:
                late = list(islice((j for j in idx[pos + 1:] if stamps[j] is not None and stamps[j] - stamps[k] >= max_gap), 3))
                if late and late[0] < within[0]:
                    within = late
            k = best(within)
            kept.append(k)
            pos = idx.index(k)
    kept = sorted(set(kept))
    frames = []
    for j, k in enumerate(kept):
        frames.append({"path": str(paths[k]), "index": int(k * max(1, int(step))), "time": stamps[k],
                       "sharpness": round(float(sharp[k]), 4),
                       "change": round(moved(kept[j - 1], k), 3) if j and segment[kept[j - 1]] == segment[k] else None})
    progress(1.0, t("{0} fotogrammi scelti su {1}", "{0} frames chosen of {1}").format(len(frames), n))
    return {"frames": frames, "analysed": n, "segments": int(len(np.unique(segment))), "unmatched": unmatched}


# ------------------------------------------------------------------ videos
def mp4_creation_time(path) -> float | None:
    """Seconds since 1970 of the creation time in the ``mvhd`` box of an MP4/MOV file (None when
    missing or zero)."""
    epoch_1904 = 2082844800
    try:
        with open(path, "rb") as f:
            size = Path(path).stat().st_size

            def boxes(start: int, end: int):
                pos = start
                while pos + 8 <= end:
                    f.seek(pos)
                    head = f.read(8)
                    if len(head) < 8:
                        return
                    n, kind = struct.unpack(">I4s", head)
                    hdr = 8
                    if n == 1:
                        n = struct.unpack(">Q", f.read(8))[0]
                        hdr = 16
                    elif n == 0:
                        n = end - pos
                    if n < hdr:
                        return
                    yield kind, pos + hdr, pos + n
                    pos += n

            for kind, a, b in boxes(0, size):
                if kind != b"moov":
                    continue
                for k2, a2, _b2 in boxes(a, b):
                    if k2 == b"mvhd":
                        f.seek(a2)
                        version = f.read(4)[0]
                        created = struct.unpack(">Q", f.read(8))[0] if version == 1 else struct.unpack(">I", f.read(4))[0]
                        return float(created - epoch_1904) if created else None
    except (OSError, struct.error, IndexError):
        return None
    return None


def extract_video(video, folder, overlap: float = 0.7, max_frames: int | None = None, start: float | None = None,
                  progress=None) -> dict:
    """Frames of a video (the program's extraction) with their time: ``start`` (seconds since 1970)
    or the creation time of the file, plus the time in the video. Writes ``frames.json`` beside the
    frames. Returns ``{"frames": [{"path", "frame", "time"}], "start"}``."""
    from pyarchinit_photogrammetry.engine.video import extract_frames

    video, folder = Path(video), Path(folder)
    begin = start if start is not None else mp4_creation_time(video)
    frames = extract_frames(video, folder, overlap=overlap, max_frames=max_frames, progress=progress)
    out = []
    for f in frames:
        out.append({"path": str(f["path"]), "frame": int(f["frame"]), "video_time": float(f["time"]),
                    "time": (begin + float(f["time"])) if begin is not None else None})
    (folder / "frames.json").write_text(json.dumps({"video": str(video), "start": begin, "frames": out}, indent=1),
                                        encoding="utf-8")
    return {"frames": out, "start": begin}


def add_frames(chunk, frames: list[dict], source: str = "") -> list:
    """The frames as photos of ``chunk``, with their time and where they came from."""
    paths = [Path(f["path"]) for f in frames]
    before = {Path(c.path).resolve() for c in chunk.cameras}
    cams = chunk.add_photos([p for p in paths if p.resolve() not in before])
    by_path = {Path(f["path"]).resolve(): f for f in frames}
    for cam in cams:
        f = by_path.get(Path(cam.path).resolve())
        if f is None:
            continue
        rec = cam.meta.setdefault("underwater", {})
        if f.get("time") is not None:
            rec["time"] = float(f["time"])
        rec["frame"] = int(f.get("index", f.get("frame", 0)))
        if source:
            rec["source"] = source
        if f.get("sharpness") is not None:
            cam.meta["sharpness"] = float(f["sharpness"])
    return cams
