"""The comparison of two campaigns of a face (step 6 of the pane).

Campaign A and campaign B map the same face: in the same chunk (two campaigns on one elevation), or in another chunk
(a later survey) whose elevation lies on the same frame - same frame, resolution, bounds and depth band, so the two
grids coincide; :func:`ensure_elevation` finds it or builds it with the program's ``build_elevation``.

The two chunks must already be in the same system: the program's *Allinea chunk* does that, not the plugin.
:func:`check_alignment` verifies it before anything is compared - the markers seen on both faces within 1 cm, or,
without markers in common, the shift between the two orthophotos measured by phase correlation under 2 px - and the
comparison is refused otherwise, with the shift measured and nothing written.

What :func:`compare` gives: per term the area in A, in B, gained and lost (shapely differences of the (u, v)
polygons); the cracks of B that are new, longer or wider than in A, and those of A not seen again; the treated areas
of A that show decay in B; ``change.tif``, the colour change of the orthophoto in CIELAB (band 1 the signed Delta E:
the size of the change, negative where B is darker; bands 2-4 Delta L, Delta a, Delta b), masked outside the union of
the mapped features of both campaigns grown by 0.5 m; with a second survey whose chunk has a model,
``relief_change.tif``, B's surface on A's grid minus A's relief (cm, positive towards the viewer); the tables (xlsx:
*Confronto per termine*, *Lesioni*, *Interventi*), a PDF page with the two orthophotos and the change, and
``confronto.json``. Everything under ``<chunk data>/conservation/<eid>/compare_<cid A>_<cid B>/``: files, never
records."""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from . import records
from .records import NotReady
from .texts import num, t

LOG = logging.getLogger("ppg_conservation")

MARKER_TOLERANCE = 0.01  # metres: a marker seen on both faces farther apart than this = the surveys are not aligned
PHASE_TOLERANCE = 2.0  # pixels of the elevation's grid
PHASE_MIN_RESPONSE = 0.03  # below this the phase correlation found nothing it can stand for
PHASE_MAX_SIDE = 2048  # the orthophotos are read at most this size for the phase correlation
MIN_SHARED = 0.05  # share of the grid that both orthophotos must cover for the phase correlation
MASK_BUFFER = 0.5  # metres around the mapped features where the colour change is kept
CRACK_NEAR = 0.02  # metres: a crack of B within this distance of a crack of A is the same crack
CRACK_SAME_SHARE = 0.2  # a crack of B with less than this share of its length near cracks of A is new
CRACK_LONGER = 0.02  # metres (and 5 %) more length = longer
CRACK_WIDER = 0.5  # millimetres more opening = wider
BLOCK = 512
ALIGN_POINTER = ("Allinea prima i due rilievi con Flusso di lavoro > Allinea chunk (marker, camere o punti), poi rifai "
                 "il prospetto del secondo.",
                 "Align the two surveys first with Workflow > Align chunks (markers, cameras or points), then make the "
                 "elevation of the second one again.")
CRACK_STATES = {"new": ("nuova", "new"), "longer": ("più lunga", "longer"), "wider": ("più larga", "wider"),
                "longer_wider": ("più lunga e più larga", "longer and wider"), "same": ("invariata", "unchanged"),
                "gone": ("non più vista", "not seen again")}


def _tr(pair) -> str:
    return t(pair[0], pair[1])


# ---------------------------------------------------------------- the frame and the grid
def _close(a, b) -> bool:
    """Equality of two frame dicts (or values in them): numbers within a billionth (relative), the rest exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


SETTINGS = ("blending", "colour_correction", "max_angle", "ghosting", "fill_holes")  # how an orthophoto is painted


def settings_of(entry: dict) -> dict:
    """How the orthophoto of an elevation was painted (the keys of :data:`SETTINGS`): the program's
    ``entry_settings`` (entries made before the settings were stored take the values used then)."""
    try:
        from pyarchinit_photogrammetry.engine.elevation import entry_settings
    except ImportError:  # a program without it: the entry's keys with build_elevation's defaults
        defaults = {"blending": "mosaic", "colour_correction": False, "max_angle": 80.0, "ghosting": False,
                    "fill_holes": True}
        return {k: entry.get(k, defaults[k]) for k in SETTINGS}
    found = entry_settings(entry)
    return {k: found[k] for k in SETTINGS}


def settings_differ(entry_a: dict, entry_b: dict) -> list[str]:
    """The settings of the orthophoto in which two elevations differ, as ``name A -> B``."""
    a, b = settings_of(entry_a), settings_of(entry_b)
    return [f"{k} {a[k]} -> {b[k]}" for k in SETTINGS if not _close(a[k], b[k])]


def same_frame(entry_a: dict, entry_b: dict, settings: bool = False) -> bool:
    """Whether two elevations lie on the same grid: frame, resolution, bounds, depth band and kind; with
    ``settings`` also painted the same way (:data:`SETTINGS`)."""
    grid = (_close(entry_a.get("frame"), entry_b.get("frame")) and _close(entry_a.get("resolution"),
                                                                          entry_b.get("resolution"))
            and entry_a.get("bounds") is not None and _close(entry_a.get("bounds"), entry_b.get("bounds"))
            and _close(entry_a.get("depth"), entry_b.get("depth"))
            and entry_a.get("kind", "elevation") == entry_b.get("kind", "elevation"))
    return grid and (not settings or not settings_differ(entry_a, entry_b))


def find_elevation(other, entry: dict) -> dict | None:
    """The elevation of ``other`` on the same grid as ``entry`` (one painted the same way first), or None."""
    found = [e for e in other.meta.get("elevations") or [] if same_frame(entry, e)]
    return next((e for e in found if not settings_differ(entry, e)), found[0] if found else None)


def ensure_elevation(other, entry: dict, progress=None) -> dict:
    """The elevation of the chunk ``other`` with the frame, resolution, bounds and depth band of ``entry``: found
    when it is there, built with the program's ``build_elevation`` when it is not (the two grids then coincide)."""
    from pyarchinit_photogrammetry.engine.elevation import build_elevation, frame_from_dict

    found = find_elevation(other, entry)
    if found is not None:
        return found
    if not entry.get("bounds"):
        raise NotReady(t("Il prospetto non ha i suoi limiti (fatto prima della 1.5.5): rifallo, poi confronta.",
                         "The elevation has no bounds (made before 1.5.5): make it again, then compare."))
    if entry.get("scale") == "wall":
        raise NotReady(t("Uno sviluppo misurato sul muro non si rifà uguale su un altro rilievo: fai sul secondo "
                         "chunk lo sviluppo sul cilindro, o confronta due campagne dello stesso prospetto.",
                         "A development measured on the wall cannot be made again the same on another survey: make "
                         "the development on the cylinder in the second chunk, or compare two campaigns of the same "
                         "elevation."))
    if other.product_path("model") is None:
        raise NotReady(t(f"Il chunk «{other.label}» non ha un modello: il prospetto si fa sul modello.",
                         f"The chunk «{other.label}» has no model: an elevation is made on the model."))
    how = settings_of(entry)  # painted as A was, so that the colour change is the wall's, not the settings'
    return build_elevation(other, frame_from_dict(entry["frame"]), resolution=float(entry["resolution"]),
                           name=entry.get("name"), depth=tuple(entry["depth"]), bounds=tuple(entry["bounds"]),
                           kind=entry.get("kind", "elevation"), blending=how["blending"],
                           colour_correction=bool(how["colour_correction"]), max_angle=float(how["max_angle"]),
                           ghosting=bool(how["ghosting"]), fill_holes=bool(how["fill_holes"]), progress=progress)


def _grid(chunk, entry: dict):
    """``(transform, (rows, cols))`` of the orthophoto of an elevation."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    path = elevation_paths(chunk, entry)[0]
    if not path.exists():
        raise NotReady(t(f"Manca l'ortofoto del prospetto: {path}", f"The orthophoto of the elevation is missing: "
                                                                     f"{path}"))
    with rasterio.open(path) as src:
        return src.transform, (src.height, src.width)


def _same_grid(chunk, entry_a, other, entry_b) -> None:
    ta, sa = _grid(chunk, entry_a)
    tb, sb = _grid(other, entry_b)
    if sa != sb or not np.allclose(tuple(ta)[:6], tuple(tb)[:6], rtol=0, atol=float(entry_a["resolution"]) / 100):
        raise NotReady(t("I due prospetti non stanno sulla stessa griglia: rifai il secondo con lo stesso piano, la "
                         "stessa risoluzione e gli stessi limiti.",
                         "The two elevations are not on the same grid: make the second one again with the same plane, "
                         "resolution and bounds."))


# ---------------------------------------------------------------- the alignment
def _read_lightness(chunk, entry: dict, factor: int):
    """L (CIELAB, 0-100) and the cells painted, of the orthophoto read ``factor`` times smaller."""
    import cv2
    import rasterio
    from rasterio.enums import Resampling

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    with rasterio.open(elevation_paths(chunk, entry)[0]) as src:
        shape = (max(1, src.height // factor), max(1, src.width // factor))
        data = src.read(out_shape=(src.count,) + shape, resampling=Resampling.average if factor > 1 else
                        Resampling.nearest)
    rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    valid = data[3] > 0 if data.shape[0] >= 4 else np.ones(shape, bool)
    return lab[..., 0], valid


def _markers_uvd(chunk, entry: dict, labels: set) -> dict:
    """``{label: (u, v, d)}`` on the frame of ``entry`` of the markers of ``chunk`` named in ``labels`` that have a
    position, wherever they lie (not only on the face)."""
    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict

    ms = [m for m in chunk.markers if m.position is not None and m.label in labels]
    if not ms:
        return {}
    uvd = frame_from_dict(entry["frame"]).coords(chunk.to_world(np.array([m.position for m in ms], np.float64)))
    return {m.label: np.asarray(p, np.float64) for m, p in zip(ms, uvd)}


def _result(ok, method, shift_px, shift_m, markers, message, **more) -> dict:
    out = {"ok": bool(ok), "method": method, "shift_px": None if shift_px is None else float(shift_px),
           "shift_m": None if shift_m is None else float(shift_m), "markers": int(markers), "message": message}
    out.update(more)
    return out


def check_alignment(chunk, entry: dict, other, other_entry: dict | None = None) -> dict:
    """Whether the chunk ``other`` is in the same system as ``chunk`` on the face ``entry``: ``{"ok", "method":
    "same" | "markers" | "phase" | "system", "shift_px", "shift_m", "markers", "message"}`` (+ ``shift_uv`` in
    metres, where B's content lies with respect to A's, and ``response`` of the phase correlation).

    The markers of this chunk on the face (the program's ``markers_on``) and the markers of ``other`` with the same
    labels, wherever they lie (a marker moved off the face is a refusal, not a marker dropped), decide when there
    are any: ok when the farthest pair is within 1 cm. Without markers in common, the shift of the orthophoto of the
    other chunk's elevation on the same frame (``other_entry``, found when not given) against this one, by phase
    correlation of their lightness on the cells both painted: ok under 2 px. Refused otherwise, with the shift
    measured and the pointer to *Allinea chunk*."""
    import cv2

    from pyarchinit_photogrammetry.engine.elevation import markers_on

    res = float(entry["resolution"])
    if other is chunk and (other_entry is None or other_entry.get("id") == entry.get("id")):
        return _result(True, "same", 0.0, 0.0, 0, t("Stesso rilievo: nessun allineamento da verificare.",
                                                     "The same survey: no alignment to check."))
    if (other.crs or "LOCAL") != (chunk.crs or "LOCAL"):
        return _result(False, "system", None, None, 0,
                       t(f"I due rilievi sono in sistemi diversi ({chunk.crs or 'LOCAL'}, {other.crs or 'LOCAL'}). ",
                         f"The two surveys are in different systems ({chunk.crs or 'LOCAL'}, "
                         f"{other.crs or 'LOCAL'}). ") + _tr(ALIGN_POINTER))
    mine = {label: np.array([u, v, d]) for label, u, v, d in markers_on(chunk, entry)}
    theirs = _markers_uvd(other, entry, set(mine))  # wherever they are: one moved off the face is a refusal
    common = sorted(set(mine) & set(theirs))
    if common:
        gaps = {label: float(np.linalg.norm(theirs[label] - mine[label])) for label in common}
        worst = max(gaps, key=gaps.get)
        gap = gaps[worst]
        shift = np.mean([theirs[k][:2] - mine[k][:2] for k in common], axis=0)
        n = len(common)
        if gap <= MARKER_TOLERANCE:
            return _result(True, "markers", gap / res, gap, n,
                           t(f"Allineati: {n} marker sulle due facce, il più lontano a {num(gap * 100, 1)} cm "
                             f"(limite 1 cm).",
                             f"Aligned: {n} markers on both faces, the farthest {num(gap * 100, 1)} cm apart "
                             f"(limit 1 cm)."), shift_uv=[float(shift[0]), float(shift[1])], gaps=gaps)
        return _result(False, "markers", gap / res, gap, n,
                       t(f"I due rilievi non sono allineati: il marker {worst} è a {num(gap * 100, 1)} cm "
                         f"dall'altro (limite 1 cm, {n} marker in comune). ",
                         f"The two surveys are not aligned: the marker {worst} is {num(gap * 100, 1)} cm from the "
                         f"other (limit 1 cm, {n} markers in common). ") + _tr(ALIGN_POINTER),
                       shift_uv=[float(shift[0]), float(shift[1])], gaps=gaps)
    other_entry = other_entry if other_entry is not None else find_elevation(other, entry)
    if other_entry is None:
        return _result(False, "phase", None, None, 0,
                       t(f"Nessun marker sulle due facce e nessun prospetto di «{other.label}» sullo stesso piano: "
                         f"non si può verificare l'allineamento. ",
                         f"No marker on both faces and no elevation of «{other.label}» on the same plane: the "
                         f"alignment cannot be checked. ") + _tr(ALIGN_POINTER))
    _same_grid(chunk, entry, other, other_entry)
    _tr_, (h, w) = _grid(chunk, entry)
    factor = max(1, int(math.ceil(max(h, w) / PHASE_MAX_SIDE)))
    la, va = _read_lightness(chunk, entry, factor)
    lb, vb = _read_lightness(other, other_entry, factor)
    both = va & vb
    if both.sum() < MIN_SHARED * both.size or min(both.shape) < 16:
        return _result(False, "phase", None, None, 0,
                       t(f"Nessun marker sulle due facce e le ortofoto dei due prospetti hanno troppo poco in comune "
                         f"({num(100 * both.mean(), 0)} %): non si può verificare l'allineamento. Metti dei marker "
                         f"sulla faccia nei due rilievi. ",
                         f"No marker on both faces and the orthophotos of the two elevations have too little in "
                         f"common ({num(100 * both.mean(), 0)} %): the alignment cannot be checked. Put markers on "
                         f"the face in both surveys. ") + _tr(ALIGN_POINTER))
    a = np.where(va, la, la[va].mean() if va.any() else 0.0).astype(np.float64)
    b = np.where(vb, lb, lb[vb].mean() if vb.any() else 0.0).astype(np.float64)
    even = (a.shape[0] - a.shape[0] % 2, a.shape[1] - a.shape[1] % 2)
    a, b = a[:even[0], :even[1]], b[:even[0], :even[1]]  # cv2.phaseCorrelate is half a pixel off on odd sides
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_64F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    dx, dy = dx * factor, dy * factor
    shift_px = float(math.hypot(dx, dy))
    shift_m = shift_px * res
    shift_uv = [float(dx * res), float(-dy * res)]
    if response < PHASE_MIN_RESPONSE:
        return _result(False, "phase", shift_px, shift_m, 0,
                       t(f"Nessun marker sulle due facce e la correlazione di fase delle ortofoto è troppo debole "
                         f"({num(response, 3)}): l'allineamento non si può verificare. Metti dei marker sulla faccia "
                         f"nei due rilievi. ",
                         f"No marker on both faces and the phase correlation of the orthophotos is too weak "
                         f"({num(response, 3)}): the alignment cannot be checked. Put markers on the face in both "
                         f"surveys. ") + _tr(ALIGN_POINTER), shift_uv=shift_uv, response=float(response))
    if shift_px < PHASE_TOLERANCE:
        return _result(True, "phase", shift_px, shift_m, 0,
                       t(f"Allineati: le due ortofoto sono spostate di {num(shift_px, 1)} px "
                         f"({num(shift_m * 100, 1)} cm; limite 2 px), per correlazione di fase.",
                         f"Aligned: the two orthophotos are {num(shift_px, 1)} px apart ({num(shift_m * 100, 1)} cm; "
                         f"limit 2 px), by phase correlation."), shift_uv=shift_uv, response=float(response))
    return _result(False, "phase", shift_px, shift_m, 0,
                   t(f"I due rilievi non sono allineati: le due ortofoto sono spostate di {num(shift_px, 1)} px "
                     f"({num(shift_m * 100, 1)} cm; u {num(shift_uv[0] * 100, 1)} cm, v {num(shift_uv[1] * 100, 1)} "
                     f"cm; limite 2 px). ",
                     f"The two surveys are not aligned: the two orthophotos are {num(shift_px, 1)} px apart "
                     f"({num(shift_m * 100, 1)} cm; u {num(shift_uv[0] * 100, 1)} cm, v {num(shift_uv[1] * 100, 1)} "
                     f"cm; limit 2 px). ") + _tr(ALIGN_POINTER), shift_uv=shift_uv, response=float(response))


# ---------------------------------------------------------------- the vectors
def _mapped(chunk, cid: str) -> list[dict]:
    from .vocabulary import UNNAMED

    return [f for f in records.features(chunk, cid, status="accepted") if f["code"] != UNNAMED]


def _key(f: dict) -> str:
    """The term a feature is compared under: its first ICOMOS-ISCS code (a UNI and an ISCS drawing of one phenomenon
    meet there), its own code otherwise."""
    return (f.get("code_iscs") or [f["code"]])[0]


def _union(geoms):
    from shapely.ops import unary_union

    return unary_union(geoms) if geoms else None


def _area(g) -> float:
    return 0.0 if g is None or g.is_empty else float(g.area)


def _term_name(vocab, code: str) -> str:
    from .texts import lang

    try:
        return vocab.name(code, lang())
    except KeyError:
        return code


def _terms(vocab, fa: list[dict], fb: list[dict]) -> list[dict]:
    from shapely import wkt

    def polygons(found):
        out: dict[str, list] = {}
        for f in found:
            g = wkt.loads(f["geometry"])
            if f["kind"] == "decay" and g.geom_type in ("Polygon", "MultiPolygon"):
                out.setdefault(_key(f), []).append(g)
        return out

    pa, pb = polygons(fa), polygons(fb)
    rows = []
    for code in sorted(set(pa) | set(pb)):
        ua, ub = _union(pa.get(code, [])), _union(pb.get(code, []))
        gained = _area(ub.difference(ua)) if ub is not None and ua is not None else _area(ub)
        lost = _area(ua.difference(ub)) if ua is not None and ub is not None else _area(ua)
        rows.append({"code": code, "name": _term_name(vocab, code), "area_a": _area(ua), "area_b": _area(ub),
                     "gained": gained, "lost": lost, "change": _area(ub) - _area(ua)})
    return rows


def _cracks(fa: list[dict], fb: list[dict]) -> list[dict]:
    """Each crack of B against the cracks of A within ``CRACK_NEAR``: *new* when less than ``CRACK_SAME_SHARE`` of
    its length lies near them (its row names no crack of A); else *longer* / *wider* / both / *same*, the length of
    A being that of each crack of A inside the band of B (a long crack of A crossed by a short one of B is not
    "shorter"). The cracks of A in no such row are *gone*, also when a new crack of B crosses them."""
    from shapely import wkt

    ca = [(f, wkt.loads(f["geometry"])) for f in fa if f["kind"] == "crack"]
    cb = [(f, wkt.loads(f["geometry"])) for f in fb if f["kind"] == "crack"]
    rows, seen = [], set()
    for f, g in cb:
        band = g.buffer(CRACK_NEAR)
        near = [(fa_, ga) for fa_, ga in ca if ga.distance(g) <= CRACK_NEAR]
        zone = _union([ga.buffer(CRACK_NEAR) for _f, ga in near])
        share = (g.intersection(zone).length / g.length) if zone is not None and g.length > 0 else 0.0
        inside = [(fa_, ga.intersection(band).length) for fa_, ga in near]
        inside = [(fa_, length) for fa_, length in inside if length > 0.0]
        wb = f.get("width_mm")
        row = {"id_b": f["id"], "length_b": float(g.length), "width_b": None if wb is None else float(wb)}
        if share < CRACK_SAME_SHARE or not inside:
            row.update(id_a=[], length_a=None, width_a=None, state="new")
        else:
            row["id_a"] = [fa_["id"] for fa_, _l in inside]
            seen.update(row["id_a"])
            la = float(sum(length for _f, length in inside))
            widths = [float(fa_["width_mm"]) for fa_, _l in inside if fa_.get("width_mm") is not None]
            wa = max(widths) if widths else None
            longer = g.length - la > max(CRACK_LONGER, 0.05 * la)
            wider = wa is not None and wb is not None and float(wb) - wa > CRACK_WIDER
            state = "longer_wider" if longer and wider else "longer" if longer else "wider" if wider else "same"
            row.update(length_a=la, width_a=wa, state=state)
        rows.append(row)
    for f, g in ca:
        if f["id"] not in seen:
            rows.append({"id_b": None, "id_a": [f["id"]], "length_a": float(g.length), "length_b": None,
                         "width_a": None if f.get("width_mm") is None else float(f["width_mm"]), "width_b": None,
                         "state": "gone"})
    for r in rows:
        r["state_text"] = _tr(CRACK_STATES[r["state"]])
    return rows


def _treatments(vocab, fa: list[dict], fb: list[dict]) -> list[dict]:
    from shapely import wkt

    from .vocabulary import deteriorates

    decay = []
    for f in fb:
        g = wkt.loads(f["geometry"])
        if f["kind"] != "decay" or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if f["code"] in vocab and not deteriorates(vocab, vocab.term(f["code"])):
            continue
        decay.append((f, g))
    rows = []
    for f in fa:
        g = wkt.loads(f["geometry"])
        if f["kind"] != "treatment" or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        hit = [(fb_, gb) for fb_, gb in decay if gb.intersects(g)]
        inside = _union([gb.intersection(g) for _f, gb in hit])
        area = float(g.area)
        tr = f.get("treatment") or {}
        rows.append({"id": f["id"], "code": f["code"], "name": _term_name(vocab, f["code"]),
                     "technique": tr.get("technique") or f["code"], "done": bool(tr.get("done")), "area": area,
                     "decay_area": _area(inside), "share": _area(inside) / area if area > 0 else 0.0,
                     "terms": sorted({_term_name(vocab, fb_["code"]) for fb_, gb in hit
                                      if gb.intersection(g).area > 0})})
    return rows


# ---------------------------------------------------------------- the rasters
def _mask_shapes(fa: list[dict], fb: list[dict]):
    """The union of the features of both campaigns, grown by ``MASK_BUFFER``; None when nothing is mapped."""
    from shapely import wkt

    from .geometry import band, disc

    geoms = []
    for f in fa + fb:
        g = wkt.loads(f["geometry"])
        if g.geom_type in ("LineString", "MultiLineString"):
            g = band(g, max(float(f.get("width_mm") or 0.0) / 1000.0, 0.005))
        elif g.geom_type == "Point":
            g = disc(g)
        geoms.append(g)
    u = _union(geoms)
    return None if u is None or u.is_empty else u.buffer(MASK_BUFFER)


def _write_change(chunk, entry_a, other, entry_b, shapes, path: Path, progress) -> dict:
    """``change.tif``: the colour change of B's orthophoto against A's, by blocks of rows."""
    import cv2
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    pa, pb = elevation_paths(chunk, entry_a)[0], elevation_paths(other, entry_b)[0]
    stats = []
    with rasterio.open(pa) as sa, rasterio.open(pb) as sb:
        h, w = sa.height, sa.width
        profile = {"driver": "GTiff", "height": h, "width": w, "count": 4, "dtype": "float32",
                   "transform": sa.transform, "compress": "deflate", "nodata": float("nan"), "BIGTIFF": "IF_SAFER"}
        if h >= 256 and w >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(path, "w", **profile) as dst:
            dst.update_tags(ppg_conservation="colour change", units="CIELAB Delta E (CIE76)",
                            sign="band 1 negative where the second campaign is darker", bands="dE, dL, da, db")
            for i, name in enumerate(("dE (signed)", "dL", "da", "db"), start=1):
                dst.set_band_description(i, name)
            for r0 in range(0, h, BLOCK):
                th = min(BLOCK, h - r0)
                win = Window(0, r0, w, th)
                out = np.full((4, th, w), np.nan, np.float32)
                a, b = sa.read(window=win), sb.read(window=win)
                ok = np.ones((th, w), bool)
                if a.shape[0] >= 4:
                    ok &= a[3] > 0
                if b.shape[0] >= 4:
                    ok &= b[3] > 0
                if shapes is not None:
                    ok &= geometry_mask([shapes], out_shape=(th, w), transform=window_transform(win, sa.transform),
                                        invert=True)
                if ok.any():
                    la = cv2.cvtColor(np.moveaxis(a[:3], 0, -1).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
                    lb = cv2.cvtColor(np.moveaxis(b[:3], 0, -1).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
                    d = lb - la
                    de = np.sqrt((d ** 2).sum(axis=-1))
                    signed = np.where(d[..., 0] < 0, -de, de)
                    for k, band in enumerate((signed, d[..., 0], d[..., 1], d[..., 2])):
                        out[k][ok] = band[ok]
                    stats.append(np.abs(signed[ok]))
                dst.write(out, window=win)
                progress(0.3 + 0.3 * (r0 + th) / h, t("Cambiamento dell'ortofoto", "Change of the orthophoto"))
    values = np.concatenate(stats) if stats else np.zeros(0)
    return {"cells": int(values.size), "median": float(np.median(values)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None}


def _write_relief_change(chunk, entry_a, other, path: Path, progress) -> dict | None:
    """``relief_change.tif``: the surface of ``other``'s model on A's grid minus A's relief (cm)."""
    import rasterio
    from rasterio.windows import Window

    from pyarchinit_photogrammetry.engine.elevation import FrameSurface, elevation_paths, frame_from_dict
    from pyarchinit_photogrammetry.engine.visibility import raycaster_for_chunk

    res = float(entry_a["resolution"])
    relief = elevation_paths(chunk, entry_a)[1]
    if not relief.exists():
        return None
    surf = FrameSurface(other, raycaster_for_chunk(other), frame_from_dict(entry_a["frame"]), res, threading.Lock(),
                        tuple(entry_a.get("depth") or (-np.inf, np.inf)), tuple(entry_a["bounds"]))
    vals = []
    with rasterio.open(relief) as src:
        tr = src.transform
        if surf.shape != (src.height, src.width) or abs(surf.origin[0] - tr.c) > res / 100 or \
                abs(surf.origin[1] - tr.f) > res / 100:
            raise NotReady(t("La griglia del prospetto non corrisponde più al suo piano: rifai il prospetto.",
                             "The grid of the elevation no longer matches its plane: make the elevation again."))
        h, w = src.height, src.width
        profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32", "transform": tr,
                   "compress": "deflate", "nodata": float("nan"), "BIGTIFF": "IF_SAFER"}
        if h >= 256 and w >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        tiles = [(r, c) for r in range(0, h, 256) for c in range(0, w, 256)]
        with rasterio.open(path, "w", **profile) as dst:
            dst.update_tags(ppg_conservation="relief change", units="cm", other=other.label,
                            sign="positive where the second survey stands towards the viewer")
            for i, (r0, c0) in enumerate(tiles):
                th, tw = min(256, h - r0), min(256, w - c0)
                win = Window(c0, r0, tw, th)
                diff = ((surf.tile(r0, c0, th, tw) - src.read(1, window=win)) * 100.0).astype(np.float32)
                dst.write(diff[None], window=win)
                ok = np.isfinite(diff)
                if ok.any():
                    vals.append(diff[ok])
                progress(0.6 + 0.25 * (i + 1) / len(tiles), t("Cambiamento del rilievo", "Change of the relief"))
    v = np.concatenate(vals) if vals else np.zeros(0)
    if v.size < 10:
        path.unlink(missing_ok=True)
        return None
    return {"median_cm": float(np.median(v)), "p90_cm": float(np.percentile(np.abs(v), 90)), "cells": int(v.size)}


# ---------------------------------------------------------------- the tables and the page
def _r(value, d: int = 4):
    return None if value is None else round(float(value), d)


def _sheets(r: dict) -> dict:
    yes, no = t("sì", "yes"), t("no", "no")
    return {
        t("Confronto per termine", "Comparison by term"): {
            "headers": [t("Codice", "Code"), t("Termine", "Term"), t("Area A (m²)", "Area A (m²)"),
                        t("Area B (m²)", "Area B (m²)"), t("Guadagnata (m²)", "Gained (m²)"),
                        t("Persa (m²)", "Lost (m²)"), t("Variazione (m²)", "Change (m²)")],
            "rows": [[x["code"], x["name"], _r(x["area_a"]), _r(x["area_b"]), _r(x["gained"]), _r(x["lost"]),
                      _r(x["change"])] for x in r["terms"]]},
        t("Lesioni", "Cracks"): {
            "headers": [t("Lesione in B", "Crack in B"), t("Lesioni in A", "Cracks in A"),
                        t("Lunghezza A (m)", "Length A (m)"), t("Lunghezza B (m)", "Length B (m)"),
                        t("Apertura A (mm)", "Opening A (mm)"), t("Apertura B (mm)", "Opening B (mm)"),
                        t("Esito", "Result")],
            "rows": [[x["id_b"] or "", ", ".join(x["id_a"]), _r(x["length_a"], 3), _r(x["length_b"], 3),
                      _r(x["width_a"], 1), _r(x["width_b"], 1), x["state_text"]] for x in r["cracks"]]},
        t("Interventi", "Interventions"): {
            "headers": [t("Intervento", "Intervention"), t("Tecnica", "Technique"), t("Eseguito", "Done"),
                        t("Area (m²)", "Area (m²)"), t("Con degrado in B (m²)", "With decay in B (m²)"),
                        t("Quota (%)", "Share (%)"), t("Termini in B", "Terms in B")],
            "rows": [[x["id"], x["name"], yes if x["done"] else no, _r(x["area"]), _r(x["decay_area"]),
                      _r(100 * x["share"], 1), "; ".join(x["terms"])] for x in r["treatments"]]},
    }


def _summary_lines(r: dict, ca: dict, cb: dict) -> list[str]:
    lines = [t(f"A: {ca['name']} ({ca['date']}) - B: {cb['name']} ({cb['date']})",
               f"A: {ca['name']} ({ca['date']}) - B: {cb['name']} ({cb['date']})"), r["alignment"]["message"]]
    if r.get("settings_differ"):
        lines.append(t(f"Prospetto B costruito con impostazioni diverse: {', '.join(r['settings_differ'])} (il "
                       f"cambiamento di colore ne risente).",
                       f"Elevation B built with different settings: {', '.join(r['settings_differ'])} (the colour "
                       f"change is affected)."))
    gained = sum(x["gained"] for x in r["terms"])
    lost = sum(x["lost"] for x in r["terms"])
    lines.append(t(f"Degrado: {num(gained, 2)} m² in più, {num(lost, 2)} m² in meno.",
                   f"Decay: {num(gained, 2)} m² more, {num(lost, 2)} m² less."))
    counts = {}
    for c in r["cracks"]:
        counts[c["state_text"]] = counts.get(c["state_text"], 0) + 1
    if counts:
        lines.append(t("Lesioni: ", "Cracks: ") + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".")
    again = [x for x in r["treatments"] if x["decay_area"] > 0]
    if again:
        lines.append(t(f"Aree trattate con degrado in B: {len(again)} su {len(r['treatments'])}.",
                       f"Treated areas with decay in B: {len(again)} of {len(r['treatments'])}."))
    if r.get("relief"):
        lines.append(t(f"Rilievo: mediana {num(r['relief']['median_cm'], 1)} cm, p90 {num(r['relief']['p90_cm'], 1)} "
                       f"cm (positivo = B verso chi guarda).",
                       f"Relief: median {num(r['relief']['median_cm'], 1)} cm, p90 {num(r['relief']['p90_cm'], 1)} "
                       f"cm (positive = B towards the viewer)."))
    return lines


def _read_rgb(chunk, entry, max_side: int = 1500):
    import rasterio
    from rasterio.enums import Resampling

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    with rasterio.open(elevation_paths(chunk, entry)[0]) as src:
        f = max(1, int(math.ceil(max(src.height, src.width) / max_side)))
        shape = (max(1, src.height // f), max(1, src.width // f))
        data = src.read(out_shape=(src.count,) + shape, resampling=Resampling.average if f > 1 else
                        Resampling.nearest)
        b = src.bounds
    img = np.moveaxis(data[:4] if data.shape[0] >= 4 else data[:3], 0, -1)
    return img, (b.left, b.right, b.bottom, b.top)


def _page(chunk, entry_a, other, entry_b, change: Path, lines: list[str], folder: Path) -> dict:
    import rasterio
    from matplotlib.figure import Figure
    from rasterio.enums import Resampling

    ia, extent = _read_rgb(chunk, entry_a)
    ib, _e = _read_rgb(other, entry_b)
    with rasterio.open(change) as src:
        dE = src.read(1, out_shape=ib.shape[:2], resampling=Resampling.nearest)
    finite = np.abs(dE[np.isfinite(dE)])
    lim = max(1.0, float(np.percentile(finite, 99))) if finite.size else 1.0
    wide = (extent[1] - extent[0]) >= 1.5 * (extent[3] - extent[2])
    fig = Figure(figsize=(420 / 25.4, 297 / 25.4))
    rows, cols = (3, 1) if wide else (1, 3)
    grid = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.35]) if wide else \
        fig.add_gridspec(2, cols, height_ratios=[1, 0.35])
    titles = (t("A", "A"), t("B", "B"), t("Cambiamento del colore (ΔE, Lab)", "Colour change (ΔE, Lab)"))
    for k in range(3):
        ax = fig.add_subplot(grid[k, 0] if wide else grid[0, k])
        ax.imshow(ib if k else ia, extent=extent, origin="upper")
        if k == 2:
            im = ax.imshow(np.ma.masked_invalid(dE), extent=extent, origin="upper", cmap="RdBu_r", vmin=-lim,
                           vmax=lim, alpha=0.75)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01,
                         label=t("ΔE (negativo = più scuro in B)", "ΔE (negative = darker in B)"))
        ax.set_title(titles[k], fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    text = fig.add_subplot(grid[rows, :] if wide else grid[1, :])
    text.axis("off")
    text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, wrap=True)
    fig.suptitle(t(f"Confronto delle campagne - {entry_a.get('name', '')}",
                   f"Comparison of the campaigns - {entry_a.get('name', '')}"), fontsize=11)
    out = {}
    try:
        path = folder / "confronto.pdf"
        fig.savefig(path, format="pdf")
        out["pdf"] = path
    except (ValueError, ImportError):  # a matplotlib without its vector backends (an old packaged program)
        path = folder / "confronto.png"
        fig.savefig(path, format="png", dpi=150)
        out["png"] = path
    return out


# ---------------------------------------------------------------- the comparison
def compare_folder(chunk, eid: str, cid_a: str, cid_b: str) -> Path:
    return Path(chunk.data_dir) / "conservation" / eid / f"compare_{cid_a}_{cid_b}"


def compare(chunk, cid_a: str, cid_b: str, folder=None, progress=None, other=None) -> dict:
    """Campaign ``cid_a`` of ``chunk`` against campaign ``cid_b`` of ``other`` (default ``chunk``): see the module.
    The alignment is checked first; NotReady, and nothing written, when it fails. Returns ``{"folder", "alignment",
    "terms", "cracks", "treatments", "change", "change_stats", "relief_change", "relief", "xlsx", "pdf" | "png",
    "json"}``."""
    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.io.tables import write_xlsx

    from .vocabulary import load

    progress = as_progress(progress)
    other = chunk if other is None else other
    ca = records.campaign(chunk, cid_a)
    cb = records.campaign(other, cid_b)
    if other is chunk and cid_a == cid_b:
        raise ValueError(t("Scegli due campagne diverse.", "Choose two different campaigns."))
    entry_a = records._elevation(chunk, ca["elevation"])
    entry_b = records._elevation(other, cb["elevation"])
    if not (other is chunk and entry_a["id"] == entry_b["id"]):
        if not same_frame(entry_a, entry_b):
            raise NotReady(t("Le due campagne non stanno su prospetti con lo stesso piano, la stessa risoluzione e gli "
                             "stessi limiti.",
                             "The two campaigns are not on elevations with the same plane, resolution and bounds."))
        _same_grid(chunk, entry_a, other, entry_b)
    progress(0.02, t("Verifica dell'allineamento", "Checking the alignment"))
    alignment = check_alignment(chunk, entry_a, other, entry_b)
    if not alignment["ok"]:
        raise NotReady(alignment["message"])
    vocab = load(chunk)
    fa, fb = _mapped(chunk, cid_a), _mapped(other, cid_b)
    progress(0.1, t("Aree, lesioni e interventi", "Areas, cracks and interventions"))
    result = {"alignment": alignment, "terms": _terms(vocab, fa, fb), "cracks": _cracks(fa, fb),
              "treatments": _treatments(vocab, fa, fb),
              "settings_differ": [] if entry_b is entry_a else settings_differ(entry_a, entry_b)}
    folder = Path(folder) if folder is not None else compare_folder(chunk, entry_a["id"], cid_a, cid_b)
    folder.mkdir(parents=True, exist_ok=True)
    result["folder"] = folder
    result["change"] = folder / "change.tif"
    result["change_stats"] = _write_change(chunk, entry_a, other, entry_b, _mask_shapes(fa, fb), result["change"],
                                           progress)
    result["relief_change"], result["relief"] = None, None
    if other is not chunk and other.product_path("model") is not None:  # A's surface is its relief.tif
        path = folder / "relief_change.tif"
        stats = _write_relief_change(chunk, entry_a, other, path, progress)
        if stats is not None:
            result["relief_change"], result["relief"] = path, stats
    progress(0.88, t("Tabelle e pagina", "Tables and page"))
    result["xlsx"] = write_xlsx(folder / "confronto.xlsx", _sheets(result),
                                title=t(f"Confronto {ca['name']} - {cb['name']}", f"Comparison {ca['name']} - "
                                                                                   f"{cb['name']}"))
    lines = _summary_lines(result, ca, cb)
    result.update(_page(chunk, entry_a, other, entry_b, result["change"], lines, folder))
    summary = {"made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "elevation": entry_a["id"],
               "a": {"chunk": chunk.label, "campaign": cid_a, "date": ca["date"], "elevation": entry_a["id"]},
               "b": {"chunk": other.label, "campaign": cid_b, "date": cb["date"], "elevation": entry_b["id"]},
               "alignment": {k: v for k, v in alignment.items() if k != "gaps"}, "terms": result["terms"],
               "settings_differ": result["settings_differ"],
               "cracks": result["cracks"], "treatments": result["treatments"], "change": result["change_stats"],
               "relief": result["relief"], "lines": lines,
               "files": {k: result[k].name for k in ("change", "relief_change", "xlsx", "pdf", "png")
                         if result.get(k)}}
    result["json"] = folder / "confronto.json"
    result["json"].write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    result["lines"] = lines
    progress(1.0, t("Confronto fatto", "Comparison done"))
    LOG.info("conservation: comparison %s / %s in %s", cid_a, cid_b, folder)
    return result
