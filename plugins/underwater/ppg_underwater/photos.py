"""The corrected copies of the photos, and the photos the chunk uses.

Every correction writes a copy of each photo, same name, same format, same size, its EXIF kept
(JPEG), into ``<chunk data>/underwater/colour/`` - under the photo's name in the alignment
(``meta["sfm"]["names"]``, so two photos with the same file name in two folders stay apart), or its
path below the common folder of the photos before the alignment. The originals are never touched.

The orthomosaic, the texture, the depth maps and a new alignment read the photos where the cameras
point (``camera.path``). :func:`use` points them to the copies, or back to the originals - the
paths the plugin kept (``camera.meta["underwater"]["original"]``) and the folder of the alignment
(``meta["sfm"]["image_root"]``) as they were - the same thing the program's *Ricollega le foto*
does, done both ways and without searching folders.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from . import colour
from .texts import t

FOLDER = ("underwater", "colour")


def colour_dir(chunk) -> Path:
    return chunk.data_dir.joinpath(*FOLDER)


def record(cam) -> dict:
    return cam.meta.setdefault("underwater", {})


def original(cam) -> Path:
    """The photo as it was before any copy was used."""
    return Path((cam.meta.get("underwater") or {}).get("original") or cam.path)


def _common_root(paths) -> Path | None:
    paths = [Path(p).resolve() for p in paths]
    if not paths:
        return None
    try:
        return Path(os.path.commonpath([str(p.parent) for p in paths]))
    except ValueError:  # different drives
        return None


def relative_names(chunk, cams) -> dict[int, str]:
    """The name of the copy of every photo: its name in the alignment, else its path below the
    common folder of the photos (POSIX)."""
    names = (chunk.meta.get("sfm") or {}).get("names") or {}
    root = _common_root([original(c) for c in chunk.cameras])
    out = {}
    for cam in cams:
        name = names.get(str(cam.key))
        if not name:
            p = original(cam).resolve()
            try:
                name = p.relative_to(root).as_posix() if root is not None else p.name
            except ValueError:
                name = p.name
        out[cam.key] = str(PurePosixPath(name))
    return out


def corrected(chunk, cam) -> Path | None:
    rel = (cam.meta.get("underwater") or {}).get("copy")
    if not rel:
        return None
    p = colour_dir(chunk) / rel
    return p if p.exists() else None


def read_original(cam) -> np.ndarray:
    """The original photo as stored: 8 or 16 bit, grey or BGR, the EXIF orientation ignored as the
    camera models do."""
    path = original(cam)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise OSError(t("Foto illeggibile: {0}", "Unreadable photo: {0}").format(path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def image_size(path) -> tuple[int, int]:
    """``(width, height)`` of an image from its header (0, 0 when it cannot be read)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return 0, 0


def read_reduced(cam, side: int = colour.WORK_SIDE) -> np.ndarray:
    """A copy of the photo (8 bit, BGR or grey as stored) at most ``side`` pixels on its long side;
    a JPEG is decoded already reduced."""
    path = str(original(cam))
    w, h = image_size(path)
    img = None
    if path.lower().endswith((".jpg", ".jpeg")) and w:
        for flag, f in ((cv2.IMREAD_REDUCED_COLOR_8, 8), (cv2.IMREAD_REDUCED_COLOR_4, 4), (cv2.IMREAD_REDUCED_COLOR_2, 2)):
            if max(w, h) / f >= side:
                img = cv2.imread(path, flag | cv2.IMREAD_IGNORE_ORIENTATION)
                break
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = (img / 257.0 + 0.5).astype(np.uint8) if img.dtype == np.uint16 else \
                    np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img is None:
        raise OSError(t("Foto illeggibile: {0}", "Unreadable photo: {0}").format(path))
    return colour.reduced(img, side)


def write_copy(cam, img: np.ndarray, target: Path) -> None:
    """``img`` written to ``target`` in the format of the original, with its EXIF (JPEG)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    src = original(cam)
    tmp = target.with_name(target.stem + ".part" + target.suffix)
    if target.suffix.lower() in (".jpg", ".jpeg"):
        from PIL import Image

        exif = icc = None
        try:
            with Image.open(src) as im:
                exif = im.info.get("exif")
                icc = im.info.get("icc_profile")
        except Exception:  # noqa: BLE001
            pass
        a = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if a.dtype != np.uint8:
            a = (a / 257.0 + 0.5).astype(np.uint8)
        kwargs = {"quality": 95}
        if exif:
            kwargs["exif"] = exif
        if icc:
            kwargs["icc_profile"] = icc
        Image.fromarray(a).save(tmp, format="JPEG", **kwargs)
    else:
        if not cv2.imwrite(str(tmp), img):
            raise OSError(f"cannot write {target}")
    os.replace(tmp, target)


def _store(chunk, cam, img: np.ndarray, rel: str, method: str) -> None:
    write_copy(cam, img, colour_dir(chunk) / rel)
    rec = record(cam)
    rec["copy"] = rel
    rec["colour"] = method


def _cams(chunk, cams):
    return [c for c in (chunk.cameras if cams is None else cams)]


def _spread(items, n: int):
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def _workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


# ------------------------------------------------------------------ quick correction
def quick(chunk, cams=None, params: colour.QuickParams | None = None, sample: int = 24, progress=None) -> dict:
    """The quick correction of the photos (all by default): copies in the colour folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    params = params or colour.QuickParams()
    cams = _cams(chunk, cams)
    if not cams:
        raise ValueError(t("Nessuna foto da correggere.", "No photo to correct."))
    progress(0.0, t("Correzione rapida: statistiche del rilievo", "Quick correction: statistics of the survey"))
    stats = quick_stats(cams, params, sample)
    names = relative_names(chunk, cams)
    done = [0]

    def one(cam):
        progress.check()
        img = read_original(cam)
        out = colour.quick_correct(_to8(img), params, stats)
        if img.ndim == 2:
            out = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _store(chunk, cam, out, names[cam.key], "quick")
        return cam.key

    start = time.time()
    with ThreadPoolExecutor(_workers()) as pool:
        for _ in pool.map(one, cams):
            done[0] += 1
            progress(0.05 + 0.95 * done[0] / len(cams), t("Correzione rapida {0}/{1}", "Quick correction {0}/{1}").format(
                done[0], len(cams)))
    info = {"method": "quick", "params": params.to_dict(), "stats": stats, "photos": len(cams),
            "seconds": round(time.time() - start, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    chunk.meta.setdefault("underwater", {})["colour"] = info
    return info


def _to8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else (img / 257.0 + 0.5).astype(np.uint8)


def quick_stats(cams, params: colour.QuickParams, sample: int = 24) -> dict:
    images = [colour.reduced(read_reduced(c, 400), 400) for c in _spread(list(cams), sample)]
    return colour.quick_stats(images, params)


# ------------------------------------------------------------------ restoration
def range_source(chunk, depth_folder=None, kind: str = "depth", fov: float | None = None, scale: float = 1.0):
    """Where the ranges come from: the program's depth maps, or a folder of depth images."""
    from .ranges import FileRanges, RangeMaps

    if depth_folder:
        return FileRanges(depth_folder, kind=kind, fov=fov, scale=scale)
    return RangeMaps(chunk)


def _sample(cam, source, side: int = colour.WORK_SIDE):
    img = read_reduced(cam, side)
    if img.ndim == 2:
        img = colour.as_colour(img)
    rng, valid = source(cam, img.shape[:2])
    if rng is None:
        return None
    return cam.key, colour.to_linear(img), rng, valid


def fit(chunk, cams=None, source=None, per: str = "survey", sample: int = 30, progress=None) -> colour.Model:
    """The model of the water fitted on a sample of the photos (``per="survey"``)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    source = source or range_source(chunk)
    cams = [c for c in _cams(chunk, cams) if source.available(c)]
    if not cams:
        raise ValueError(t("Nessuna foto con una mappa di profondità: costruisci prima le mappe di profondità "
                           "(Flusso di lavoro > Mappe di profondità).",
                           "No photo with a depth map: build the depth maps first (Workflow > Depth maps)."))
    picked = _spread(cams, sample)
    samples = []
    for i, cam in enumerate(picked):
        progress(i / len(picked), t("Campioni dell'acqua {0}/{1}", "Samples of the water {0}/{1}").format(i + 1, len(picked)))
        s = _sample(cam, source)
        if s is not None and s[3].mean() > 0.05:
            samples.append(s)
    if not samples:
        raise ValueError(t("Le mappe di profondità non coprono le foto.", "The depth maps do not cover the photos."))
    progress(0.95, t("Coefficienti dell'acqua", "Coefficients of the water"))
    return colour.fit_model(samples, per=per, metric=bool(getattr(source, "metric", False)))


def _exposure(lin, rng, valid, model: colour.Model, amplitude: float) -> float:
    z = np.clip(rng, model.zmin, model.zmax)
    direct = np.maximum(lin - colour.backscatter(z, model.backscatter, model.zref, amplitude), 1e-4)
    light = colour.local_illuminant(direct, z, valid)
    ok = valid & np.all(light > 1e-3, axis=-1)
    if ok.sum() < 50:
        return 0.0
    zn = z[ok] / model.zref
    att = np.column_stack([colour.beta_model(zn, model.beta[c]) * zn for c in range(3)])
    r = np.log(light[ok]) + att - np.asarray(model.white)[None, :]
    return float(np.median(r))


def restore(chunk, cams=None, per: str = "survey", source=None, sample: int = 30, model: colour.Model | None = None,
            progress=None) -> dict:
    """The colours of the photos restored with their ranges (see :mod:`.colour`): copies in the
    colour folder. ``per``: ``survey`` (one model of the water for the photos, fitted on a sample of
    ``sample``) or ``photo`` (one model per photo)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    source = source or range_source(chunk)
    cams = [c for c in _cams(chunk, cams) if source.available(c)]
    skipped = len(_cams(chunk, cams)) - len(cams)
    if not cams:
        raise ValueError(t("Nessuna foto con una mappa di profondità: costruisci prima le mappe di profondità "
                           "(Flusso di lavoro > Mappe di profondità).",
                           "No photo with a depth map: build the depth maps first (Workflow > Depth maps)."))
    start = time.time()
    if per == "survey" and model is None:
        model = fit(chunk, cams, source, per, sample, progress.sub(0.0, 0.3))
    names = relative_names(chunk, cams)
    done = [0]
    sub = progress.sub(0.3 if per == "survey" else 0.0, 0.7 if per == "survey" else 1.0)
    per_photo = {}

    def one(cam):
        progress.check()
        small = _sample(cam, source)
        if small is None:
            return cam.key, None
        _k, lin_s, rng_s, valid_s = small
        if per == "photo":
            m = colour.fit_model([small], per="photo", metric=bool(getattr(source, "metric", False)))
            amp, exp_ = 1.0, m.exposures.get(cam.key, 0.0)
        else:
            m = model
            amp = m.amplitudes.get(cam.key)
            if amp is None:
                amp = colour._amplitude(lin_s, rng_s, valid_s, m) or 1.0
            exp_ = m.exposures.get(cam.key)
            if exp_ is None:
                exp_ = _exposure(lin_s, rng_s, valid_s, m, amp)
        img = read_original(cam)
        grey = img.ndim == 2
        bgr = colour.as_colour(img)
        h, w = bgr.shape[:2]
        rng = cv2.resize(rng_s, (w, h), interpolation=cv2.INTER_LINEAR)
        lin = colour.to_linear(bgr)
        out = colour.restore_linear(lin, rng, m, amp, exp_)
        if img.dtype == np.uint16:
            enc = colour.to_srgb(out).astype(np.uint16) * 257
        else:
            enc = colour.to_srgb(out)
        if grey:
            enc = cv2.cvtColor(enc, cv2.COLOR_BGR2GRAY) if enc.dtype == np.uint8 else enc[..., 1]
        _store(chunk, cam, enc, names[cam.key], "seathru")
        info = {"amplitude": round(float(amp), 4), "exposure": round(float(exp_), 4),
                "covered": round(float(valid_s.mean()), 3)}
        if per == "photo":
            info["model"] = m.to_dict()
        record(cam)["seathru"] = info
        return cam.key, info

    with ThreadPoolExecutor(_workers()) as pool:
        for key, info in pool.map(one, cams):
            done[0] += 1
            if info is not None:
                per_photo[key] = info
            sub(done[0] / len(cams), t("Colori ripristinati {0}/{1}", "Colours restored {0}/{1}").format(done[0], len(cams)))
    result = {"method": "seathru", "per": per, "photos": len(per_photo), "skipped": skipped,
              "seconds": round(time.time() - start, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "ranges": "file" if source.__class__.__name__ == "FileRanges" else "depth_maps"}
    if model is not None:
        result["model"] = model.to_dict()
        result["coefficients"] = model.coefficients()
    chunk.meta.setdefault("underwater", {})["colour"] = result
    return result


def preview(chunk, cam, method: str = "quick", params: colour.QuickParams | None = None, per: str = "survey",
            source=None, side: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """``(before, after)`` of one photo, reduced (BGR 8 bit): the quick correction with the statistics
    of a small sample of the survey, or the restoration with the model of the survey kept from the
    last restoration (else fitted on a small sample)."""
    before = read_reduced(cam, side)
    if method == "quick":
        params = params or colour.QuickParams()
        stats = quick_stats(chunk.cameras, params, 12)
        after = colour.quick_correct(before, params, stats)
        return before, after
    source = source or range_source(chunk)
    if not source.available(cam):
        raise ValueError(t("Questa foto non ha una mappa di profondità.", "This photo has no depth map."))
    small = _sample(cam, source, side)
    _k, lin, rng, valid = small
    if per == "photo":
        m = colour.fit_model([small], per="photo", metric=bool(getattr(source, "metric", False)))
        amp, exp_ = 1.0, m.exposures.get(cam.key, 0.0)
    else:
        stored = ((chunk.meta.get("underwater") or {}).get("colour") or {}).get("model")
        m = colour.Model.from_dict(stored) if stored else fit(chunk, None, source, "survey", 8)
        amp = m.amplitudes.get(cam.key) or colour._amplitude(lin, rng, valid, m) or 1.0
        exp_ = m.exposures.get(cam.key)
        if exp_ is None:
            exp_ = _exposure(lin, rng, valid, m, amp)
    after = colour.to_srgb(colour.restore_linear(lin, rng, m, amp, exp_))
    return colour.as_colour(before), after


# ------------------------------------------------------------------ which photos the chunk uses
def status(chunk) -> dict:
    copies = [c for c in chunk.cameras if corrected(chunk, c) is not None]
    using = [c for c in copies if Path(c.path).resolve() == corrected(chunk, c).resolve()]
    info = (chunk.meta.get("underwater") or {}).get("colour") or {}
    return {"copies": len(copies), "using": len(using), "photos": len(chunk.cameras), "method": info.get("method"),
            "per": info.get("per"), "folder": str(colour_dir(chunk))}


def use(chunk, which: str = "corrected") -> dict:
    """Point the cameras to the corrected copies (``corrected``) or back to the originals
    (``original``). Returns ``{"switched", "missing"}``."""
    sfm = chunk.meta.get("sfm") or {}
    uw = chunk.meta.setdefault("underwater", {})
    switched, missing = 0, 0
    if which == "corrected":
        if "image_root_original" not in uw:
            uw["image_root_original"] = sfm.get("image_root")
        for cam in chunk.cameras:
            copy = corrected(chunk, cam)
            if copy is None:
                missing += 1
                continue
            rec = record(cam)
            if "original" not in rec:
                rec["original"] = cam.path
            if Path(cam.path).resolve() != copy.resolve():
                cam.path = str(copy.resolve())
                switched += 1
        _update_root(chunk, colour_dir(chunk))
        uw["using"] = "corrected"
    elif which == "original":
        for cam in chunk.cameras:
            rec = cam.meta.get("underwater") or {}
            if rec.get("original") and cam.path != rec["original"]:
                cam.path = rec["original"]
                switched += 1
        if sfm and "image_root_original" in uw:
            if uw["image_root_original"] is None:
                sfm.pop("image_root", None)
            else:
                sfm["image_root"] = uw["image_root_original"]
        uw["using"] = "original"
    else:
        raise ValueError(f"which: corrected or original, not {which!r}")
    return {"switched": switched, "missing": missing}


def _update_root(chunk, folder: Path) -> None:
    """``meta["sfm"]["image_root"]`` on the copies when every photo of the alignment is one of them
    under its own name (else the program links the photos where they are)."""
    sfm = chunk.meta.get("sfm")
    if not sfm or not sfm.get("names"):
        return
    folder = folder.resolve()
    by_key = {str(c.key): Path(c.path).resolve() for c in chunk.cameras}
    for k, name in sfm["names"].items():
        if k in by_key and by_key[k] != (folder / name).resolve():
            return
    sfm["image_root"] = str(folder)
