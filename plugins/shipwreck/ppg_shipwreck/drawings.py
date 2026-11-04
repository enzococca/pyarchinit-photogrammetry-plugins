"""The drawings of the hull: body plan, half-breadth plan, profile, plan over the orthophoto, plan of the wreck. A
drawing is a :class:`Drawing` - lines and filled rings by layer, in metres on the sheet - that one function renders with
matplotlib (SVG, PDF, PNG) and another as DXF with a layer per kind. Only measured lines: nothing is faired, and the
sheet says what it shows.

Plans are drawn from above: x to the right and the left of x upwards, so that a frame with x running aft
(``sign`` -1, y to the right of x) is not mirrored; the body plan is seen from aft, port on the left.

The plan of the wreck (:func:`wreck_plan`, "Pianta del relitto") draws the traced timbers of the catalogue over the
hillshade of the hull's relief (or the orthophoto): every polygon outlined and lightly filled in its kind's colour
(:data:`models3d.KIND_COLOURS`) and named, the parts of a strake under a frame, the keelson or a stringer outlined dashed,
the reconstructed parts (any status but ``measured``) hatched and lighter on their own ``RICOSTRUZIONE_<KIND>`` layers,
never mixed with the measured ``LEGNI_<KIND>`` ones. With traced timbers the body plan marks the traced seams and
strake edges at each station (``FASCE_RICALCATE``) and the profile draws the keelson's top (``KEELSON``), the typed keel
hatched (``KEEL``) and the sheer (``SHEER``); a survey without traced timbers is drawn as in 0.2.0."""
from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.archaeo.sections import drawing_scale
from pyarchinit_photogrammetry.core import crs as crs_mod

from . import lines, ship, timbers
from .texts import num, t

KINDS = ("body_plan", "half_breadth", "profile", "plan", "wreck_plan")
KIND_TITLES = {"body_plan": ("Piano di costruzione: sezioni trasversali", "Body plan: cross-sections"),
               "half_breadth": ("Piano delle linee d'acqua: orizzontali", "Half-breadth plan: horizontals"),
               "profile": ("Profilo: longitudinali", "Profile: longitudinals"),
               "plan": ("Pianta dello scafo con le stazioni", "Plan of the hull with the stations"),
               "wreck_plan": ("Pianta del relitto", "Plan of the wreck")}
LAYERS = {"STATIONS": "#111111", "LONGITUDINALS": "#111111", "HORIZONTALS": "#111111", "RAW": "#9a9a9a", "KEEL": "#b02020",
          "LEVEL": "#2060b0", "TIMBERS": "#b06000", "GRID": "#bbbbbb", "Fasce": "#1a8f5a", "KEELSON": "#c0392b",
          "SHEER": "#1f4e79", "FASCE_RICALCATE": "#0b5345"}
WEIGHTS = {"STATIONS": 0.35, "LONGITUDINALS": 0.35, "HORIZONTALS": 0.35, "RAW": 0.13, "KEEL": 0.35, "LEVEL": 0.35,
           "TIMBERS": 0.5, "GRID": 0.18, "Fasce": 0.3, "KEELSON": 0.5, "SHEER": 0.35,
           "FASCE_RICALCATE": 0.3}  # millimetres of paper
TIMBER_WEIGHT = 0.35  # mm of paper: the outlines and lines of the timbers on the plan of the wreck
MEASURED, RECONSTRUCTED = "LEGNI_", "RICOSTRUZIONE_"  # the layers of a kind on the plan of the wreck
HATCH = "////"  # the hatch of a reconstructed part on the sheet (ANSI31 in the DXF)
FILL_ALPHA = 0.3  # the light fill of a timber
PLATE_ALPHA = 0.15  # the lighter fill of the planking and the ceiling: the relief under them stays readable
LINE_SWATCH = "line"  # the "hatch" of a legend entry drawn as a line (seams, the sheer)
POINT_MARK = 0.02  # metres: the arms of the cross of a timber recorded as a point
BACKGROUNDS = ("relief", "ortho", "none")
WRECK_MARGIN = 0.25  # metres around the timbers on the plan of the wreck
SLIVER = 1e-5  # metres: thinner than this a hidden part of a strake is rounding, not a part
CACHE_FIELDS = ("ship_xy", "visible_xy", "frame_key", "solid", "volume")  # fields of a record that no drawing depends on
HATCH_NOTE = ("Parti campite a righe oblique (////): ricostruzione (ipotesi), mai una misura.",
              "Parts hatched with oblique lines (////): reconstruction (hypothesis), never a measure.")
ACROSS_RATIO = 3.0  # a polygon this many times as long as wide is named along its length
NAME_AT = {"frame": 0.85, "line": 0.3, "longitudinal": 0.2}  # where along a timber its name goes
DRAW_ORDER = ("keel", "plank", "ceiling", "frame", "stringer", "keelson", "other", "fastening", "seam")
LAND_MARK = 0.015  # metres: the arms of the cross that marks a land on the body plan
MARGIN = 12.0  # mm of paper around the lines
TITLE_BLOCK = 30.0  # mm of paper under the lines, at least
MIN_SHEET = 210.0  # mm: the title block needs a sheet at least this wide
MAX_RASTER = 6000  # px of the orthophoto on the plan


class Empty(ship.NotReady):
    """A drawing with no lines: the sections asked do not include its kind."""


@dataclass
class Drawing:
    kind: str
    title: str
    lines: list = field(default_factory=list)  # (layer, (N, 2) metres on the sheet, colour or None for the layer's)
    labels: list = field(default_factory=list)  # (x, y, text, colour or None)
    notes: list = field(default_factory=list)
    block: dict = field(default_factory=dict)  # the title block: label -> value
    north: tuple | None = None  # the direction of the north on the sheet, or None
    raster: tuple | None = None  # (RGB array, (xmin, xmax, ymin, ymax))
    axes: tuple = ("", "")
    scale: int = 0
    # (layer, ring (N, 2) metres on the sheet, colour, hatch "" or HATCH, dashed): a ring counter-clockwise is filled
    # lightly (hatched when a hatch is given), a clockwise one (a hole) and a dashed one (a part hidden under another
    # timber, which fills it) are only outlined
    fills: list = field(default_factory=list)
    legend: list = field(default_factory=list)  # (text, colour, hatch) in the title block
    bounds: tuple | None = None  # (xmin, xmax, ymin, ymax) the sheet shows at least
    rotations: dict = field(default_factory=dict)  # index of a label -> its angle on the sheet (degrees)

    def add(self, layer: str, polylines, colour: str | None = None) -> None:
        self.lines += [(layer, np.asarray(p, float), colour) for p in polylines]

    def label(self, x: float, y: float, text: str, colour: str | None = None, angle: float = 0.0) -> None:
        if angle:
            self.rotations[len(self.labels)] = float(angle)
        self.labels.append((float(x), float(y), text, colour))

    @property
    def extent(self) -> tuple:
        pts = [p for _layer, p, _colour in self.lines if len(p)] + [ring for _l, ring, *_rest in self.fills if len(ring)]
        all_points = np.vstack(pts) if pts else np.zeros((1, 2))
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        for box in (self.raster[1] if self.raster is not None else None, self.bounds):
            if box is not None:
                x0, x1, y0, y1 = box
                lo, hi = np.minimum(lo, (x0, y0)), np.maximum(hi, (x1, y1))
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def title_fields(chunk) -> dict:
    """The fields of the title block kept with the chunk, with the defaults of the project."""
    fields = {"site": "", "vessel": chunk.label, "surveyed": "", "drawn": "", "date": date.today().isoformat(), "notes": ""}
    document = getattr(chunk, "document", None)
    if document is not None:
        fields["site"] = str(document.meta.get("sito") or "")
    fields.update({k: v for k, v in chunk.meta.get(ship.KEY, {}).get("title", {}).items() if v})
    return fields


def _block(chunk, frame: ship.Frame) -> dict:
    f = title_fields(chunk)
    block = {t("Sito", "Site"): f["site"], t("Relitto", "Wreck"): f["vessel"], t("Rilievo", "Survey"): f["surveyed"],
             t("Disegno", "Drawn by"): f["drawn"], t("Data", "Date"): f["date"],
             t("Chiglia", "Keel"): t(f"{num(frame.length, 2)} m, pendenza {num(frame.slope * 100, 1)} %",
                                     f"{num(frame.length, 2)} m, slope {num(frame.slope * 100, 1)} %"),
             t("Prua", "Bow"): t(f"a x = {num(frame.bow_x, 2)} m; babordo a y positive",
                                 f"at x = {num(frame.bow_x, 2)} m; port at positive y"),
             "Datum": t(f"z = 0 a quota {num(frame.z0, 3)} m del chunk", f"z = 0 at height {num(frame.z0, 3)} m of the chunk")}
    return {k: v for k, v in block.items() if v}


SHOWS = {"body_plan": "station", "half_breadth": "horizontal", "profile": "longitudinal", "plan": "station"}


def _notes(chunk, data: dict, frame: ship.Frame, kind: str) -> list[str]:
    half = num(data["half_width"], 2)
    notes = [t("Linee di sezione misurate sul modello 3D, senza carenatura; niente di ciò che manca è ricostruito. Tratteggio: "
               "asse del sistema della nave e livello dell'acqua. Legni: dal catalogo, disegnati o importati, non tagliati dal modello.",
               "Section lines measured on the 3D model, not faired; nothing missing is reconstructed. Dashed: the axis of the "
               "ship frame and the water level. Timbers: from the catalogue, drawn or imported, not cut from the model."),
             lines.note(data, raw_layer=kind in ("body_plan", "profile")),
             t(f"Disegno entro ±{half} m dalla linea di chiglia.", f"Drawn within ±{half} m of the keel line.")]
    shown = [s for s in data["sections"] if s["kind"] == SHOWS[kind]]
    empty = lines.without_line(shown)
    if empty:
        notes.append(t(f"Senza linea, fuori dal modello: {', '.join(empty)}.", f"With no line, off the model: {', '.join(empty)}."))
    reached = lines.at_window(shown)
    if reached and kind != "plan":
        notes.append(t(f"Arrivano al limite del disegno, il modello continua oltre: {', '.join(reached)}.",
                       f"They reach the limit of the drawing, the model goes on beyond: {', '.join(reached)}."))
    if not ship.is_scaled(chunk):
        notes.append(t("Modello senza scala nota: le misure non sono metri veri.",
                       "A model without a known scale: the measures are not true metres."))
    if crs_mod.is_local(chunk.crs):
        notes.append(t("Sistema locale: l'alto è l'asse z del modello, non verificato come verticale.",
                       "Local frame: up is the z axis of the model, not verified as vertical."))
    if kind in ("body_plan", "profile", "plan") and _land_lines(chunk):
        notes.append(t("Fasce: gradini rilevati sulle sezioni (croci sul piano di costruzione: i gradini sulle linee); linee delle fasce: "
                       "ipotesi che unisce i gradini fra le stazioni, non una misura.",
                       "Strakes: lands found on the sections (crosses on the body plan: the lands on the lines); land lines: a hypothesis "
                       "joining the lands between the stations, not a measure."))
    extra = title_fields(chunk)["notes"]
    return notes + ([extra] if extra else [])


def _land_lines(chunk) -> list[dict]:
    """The land lines found on the sections (Task 1 of the clinker strakes) when they still match them: else none."""
    from . import clinker

    state = chunk.meta.get(ship.KEY, {}).get("clinker") or {}
    found = [line for line in state.get("lines") or [] if line["points"]]
    if not found or clinker.is_stale(chunk):
        return []
    return found


def _level_line(d: Drawing, data: dict, lo: float, hi: float) -> None:
    water = data.get("water")
    if water:
        d.add("LEVEL", [[[lo, water["level"]], [hi, water["level"]]]])


def _tints(count: int) -> list[str]:
    """A colour for each of ``count`` lines drawn over one another, the first ones blue, the last ones red."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    return [to_hex(colormaps["turbo"](0.05 + 0.9 * i / max(count - 1, 1))) for i in range(count)]


def _of(data: dict, kind: str) -> list[dict]:
    return [s for s in data["sections"] if s["kind"] == kind]


def body_plan(chunk, data: dict, frame: ship.Frame) -> Drawing:
    """Seen from aft: port on the left whichever way x runs."""
    d = Drawing("body_plan", t(*KIND_TITLES["body_plan"]), axes=(t("babordo", "port"), t("tribordo", "starboard")))
    sections = _of(data, "station")
    for s, tint in zip(sections, _tints(len(sections))):
        if "corrected" in s:
            d.add("RAW", [np.column_stack([-np.asarray(p)[:, 0], np.asarray(p)[:, 1]]) for p in s["lines"]])
        drawn = [np.column_stack([-p[:, 0], p[:, 1]]) for p in lines.shown(s)]
        d.add("STATIONS", drawn, tint)
        if drawn:
            longest = max(drawn, key=len)
            top = longest[np.argmax(longest[:, 1])]
            d.label(top[0], top[1], s["name"], tint)
    for line in _land_lines(chunk):  # a cross at each land: left, right, centre, down, up
        for _x, y, z in line["points"]:
            c, a = np.array([-y, z]), LAND_MARK
            d.add("Fasce", [[c - [a, 0], c + [a, 0], c, c - [0, a], c + [0, a]]], LAYERS["Fasce"])
    _traced_lands(chunk, sections, frame, d)
    if d.lines:
        x0, x1, y0, y1 = d.extent
        d.add("KEEL", [[[0.0, y0], [0.0, y1]]])
        _level_line(d, data, x0, x1)
    return d


def _station_ticks(data: dict, y0: float, y1: float, d: Drawing) -> None:
    for s in _of(data, "station"):
        d.add("GRID", [[[s["value"], y0], [s["value"], y1]]])
        d.label(s["value"], y1, s["name"])


def half_breadth(chunk, data: dict, frame: ship.Frame) -> Drawing:
    d = Drawing("half_breadth", t(*KIND_TITLES["half_breadth"]), north=_north(chunk, frame))
    sections = _of(data, "horizontal")
    for s, tint in zip(sections, _tints(len(sections))):
        drawn = [np.column_stack([p[:, 0], p[:, 1] * frame.sign]) for p in map(np.asarray, s["lines"])]
        d.add("HORIZONTALS", drawn, tint)
        if drawn:
            first = max(drawn, key=len)[0]
            d.label(first[0], first[1], f"z {num(s.get('corrected_value', s['value']), 2)}", tint)
    if d.lines:
        x0, x1, y0, y1 = d.extent
        d.add("KEEL", [[[x0, 0.0], [x1, 0.0]]])
        _station_ticks(data, y0, y1, d)
    return d


def profile(chunk, data: dict, frame: ship.Frame) -> Drawing:
    d = Drawing("profile", t(*KIND_TITLES["profile"]))
    sections = _of(data, "longitudinal")
    for s, tint in zip(sections, _tints(len(sections))):
        if "corrected" in s:
            d.add("RAW", [np.asarray(p) for p in s["lines"]])
        drawn = lines.shown(s)
        d.add("LONGITUDINALS", drawn, tint)
        if drawn:
            first = max(drawn, key=len)[0]
            d.label(first[0], first[1], s["name"], tint)
    for line in _land_lines(chunk):
        p = np.asarray(line["points"], float)
        if len(p) > 1:
            d.add("Fasce", [p[:, [0, 2]]], LAYERS["Fasce"])
    if _traced(chunk):
        _profile_timbers(chunk, data, frame, d)
    if d.lines:
        x0, x1, y0, y1 = d.extent
        _station_ticks(data, y0, y1, d)
        _level_line(d, data, x0, x1)
    return d


def _north(chunk, frame: ship.Frame) -> tuple | None:
    """The north on a plan: None in a local frame, which has none."""
    if crs_mod.is_local(chunk.crs):
        return None
    h = np.radians(frame.heading)
    return float(np.cos(h)), float(np.sin(h))


def ortho_in_ship(chunk, frame: ship.Frame, x_range, y_range, resolution: float):
    """The orthophoto on a grid of the sheet (plans: x to the right, the left of x upwards): an ``(H, W, 4)``
    uint8 array and its extent, or None when the chunk has none."""
    path = chunk.orthomosaic_path
    if path is None or not Path(path).exists():
        return None
    import cv2
    import rasterio
    from rasterio.windows import Window

    width = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    height = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    k = max(width, height) / MAX_RASTER
    if k > 1:
        width, height, resolution = int(width / k), int(height / k), resolution * k
    xs = x_range[0] + (np.arange(width) + 0.5) * resolution
    ys = y_range[1] - (np.arange(height) + 0.5) * resolution  # the sheet's y, from the top
    gx, gy = np.meshgrid(xs, ys)
    ship_points = np.column_stack([gx.ravel(), gy.ravel() * frame.sign, np.zeros(gx.size)])
    world = frame.to_world(ship_points)
    with rasterio.open(path) as src:
        inverse = ~src.transform
        cols = inverse.a * world[:, 0] + inverse.b * world[:, 1] + inverse.c
        rows = inverse.d * world[:, 0] + inverse.e * world[:, 1] + inverse.f
        c0, c1 = max(int(cols.min()) - 2, 0), min(int(cols.max()) + 3, src.width)
        r0, r1 = max(int(rows.min()) - 2, 0), min(int(rows.max()) + 3, src.height)
        if c1 <= c0 or r1 <= r0:
            return None
        step = max(1.0, resolution / abs(src.transform.a))
        shape = (max(1, int((r1 - r0) / step)), max(1, int((c1 - c0) / step)))
        bands = (1, 2, 3, 4) if src.count >= 4 else (1, 2, 3)
        image = src.read(bands, window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=(len(bands), *shape))
    image = np.moveaxis(image, 0, -1).astype(np.uint8)
    if image.shape[2] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    # continuous raster coordinates (a pixel's centre at .5) to cv2's (at the integer)
    map_x = ((cols - c0) * shape[1] / (c1 - c0) - 0.5).astype(np.float32).reshape(gx.shape)
    map_y = ((rows - r0) * shape[0] / (r1 - r0) - 0.5).astype(np.float32).reshape(gx.shape)
    out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, (x_range[0], x_range[0] + width * resolution, y_range[1] - height * resolution, y_range[1])


def plan(chunk, data: dict, frame: ship.Frame, scale: int | None = None) -> Drawing:
    d = Drawing("plan", t(*KIND_TITLES["plan"]), north=_north(chunk, frame))
    half = data["half_width"]
    x_lo, x_hi = sorted((frame.x0, frame.x0 + frame.length))
    xs = [s["value"] for s in data["sections"] if s["kind"] == "station"] + [x_lo, x_hi]
    margin = data["margin"]
    x0, x1 = min(xs) - margin, max(xs) + margin
    d.add("KEEL", [[[x_lo, 0.0], [x_hi, 0.0]]])
    for s in _of(data, "station"):
        d.add("STATIONS", [[[s["value"], -half], [s["value"], half]]])
        d.label(s["value"], half, s["name"])
    if chunk.meta.get(ship.KEY, {}).get("timbers"):
        for r in timbers.timbers_of(chunk):
            xy = np.asarray(r["geometry"]["xy"], float)
            p = frame.to_ship(np.column_stack([xy, np.zeros(len(xy))]))[:, :2] * [1, frame.sign]
            if r["geometry"]["type"] == "polygon":
                p = np.vstack([p, p[:1]])
            if len(p) > 1:
                d.add("TIMBERS", [p])
            d.label(p[0, 0], p[0, 1], r["name"] or r["id"], LAYERS["TIMBERS"])
    for line in _land_lines(chunk):
        p = np.asarray(line["points"], float)
        if len(p) > 1:
            d.add("Fasce", [np.column_stack([p[:, 0], p[:, 1] * frame.sign])], LAYERS["Fasce"])
    scale = scale or drawing_scale(x1 - x0, 2 * half)
    raster = ortho_in_ship(chunk, frame, (x0, x1), (-half, half), scale * 0.00025)  # 4 pixels per mm of paper
    if raster is not None:
        d.raster = raster
    else:
        d.notes.append(t("Il chunk non ha un'ortofoto: la pianta mostra solo le stazioni.",
                         "The chunk has no orthophoto: the plan shows the stations only."))
    return d


# ----------------------------------------------------------------------------- the traced timbers (0.3.0)
def _clean_record(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in CACHE_FIELDS}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _traced(chunk) -> bool:
    """Whether the chunk has timbers traced on the relief (records with a plan cache) or a typed keel: the 0.3.0
    additions of the body plan and the profile are drawn only then, so a 0.2.0 survey is drawn as it was."""
    st = chunk.meta.get(ship.KEY, {})
    return bool(st.get("keel")) or any("ship_xy" in r for r in st.get("timbers", []))


def _water(data: dict | None):
    """The correction of the heights when the water is on (as the sections shown), else None."""
    water = (data or {}).get("water")
    return (lambda z: lines.corrected(z, water["level"], water["n"])) if water else (lambda z: np.asarray(z, float))


def _crossings_at(p: np.ndarray, x: float) -> list[float]:
    """The y where the polyline ``p`` (x, y) crosses the line at ``x``."""
    out = []
    for (xa, ya), (xb, yb) in zip(p[:-1], p[1:]):
        if (xa - x) * (xb - x) <= 0 and xa != xb:
            out.append(float(ya + (yb - ya) * (x - xa) / (xb - xa)))
    return out


def _z_on(polylines, y: float) -> float | None:
    """The highest z of the section's polylines (y, z) at ``y`` (the inside of the hull, as the relief sees it)."""
    found = [z for p in polylines for z in _crossings_at(p, y)]
    return max(found) if found else None


def _ship_xy(chunk, record: dict, frame: ship.Frame) -> np.ndarray:
    """A record's geometry in the plan of the frame, read on a COPY of the record: drawing never writes the records'
    caches (it may run in a worker while the window saves or snapshots the chunk's meta)."""
    from . import tracing

    return tracing.ship_xy_of(chunk, dict(record), frame)


def _traced_lands(chunk, sections: list[dict], frame: ship.Frame, d: Drawing) -> None:
    """The traced seams and the edges of the traced strakes at each station (layer ``FASCE_RICALCATE``): their y at the
    station's x, on the section at that y, a cross of :data:`LAND_MARK`."""
    records = [r for r in timbers.timbers_of(chunk) if timbers.is_measured(r) and "ship_xy" in r and
               ((r["kind"] == "seam" and r["geometry"]["type"] == "line") or
                (r["kind"] == "plank" and r["geometry"]["type"] == "polygon"))]
    if not records:
        return
    shapes = []
    for r in records:
        xy = _ship_xy(chunk, r, frame)
        shapes.append(np.vstack([xy, xy[:1]]) if r["geometry"]["type"] == "polygon" else xy)
    a = LAND_MARK
    for s in sections:
        ys = sorted(y for p in shapes for y in _crossings_at(p, s["value"]))
        kept = [y for k, y in enumerate(ys) if k == 0 or y - ys[k - 1] > a / 3.0]  # a seam and a strake's edge: one
        drawn = lines.shown(s)
        for y in kept:
            z = _z_on(drawn, y)
            if z is None:
                continue
            c = np.array([-y, z])
            d.add("FASCE_RICALCATE", [[c - [a, 0], c + [a, 0], c, c - [0, a], c + [0, a]]], LAYERS["FASCE_RICALCATE"])


def _box_of(solid, water) -> np.ndarray:
    """The side view of a solid (x, z): its lowest and highest point at each x of its vertices, a ring
    counter-clockwise (the bottom forward, the top back), the heights through ``water``."""
    v = solid.vertices
    key = np.round(v[:, 0], 6)
    xs = np.unique(key)
    top = np.array([v[key == x, 2].max() for x in xs])
    bottom = np.array([v[key == x, 2].min() for x in xs])
    return np.vstack([np.column_stack([xs, water(bottom)]), np.column_stack([xs[::-1], water(top[::-1])])])


def _profile_timbers(chunk, data: dict, frame: ship.Frame, d: Drawing) -> None:
    """On the profile: the keelson as a box from its solid (:func:`solids.timber_solid`: the hull under it to its top,
    ``KEELSON`` when measured), the typed keel's box hatched (:func:`solids.keel_solid`, ``RICOSTRUZIONE_KEEL``: a
    hypothesis) and the preserved upper edge (``SHEER``: the highest end of each station whose section did not reach
    the edge of the window it was cut in)."""
    from . import models3d, solids

    water = _water(data)
    for r in timbers.timbers_of(chunk):
        if r["kind"] != "keelson" or r["geometry"]["type"] != "polygon":
            continue
        try:
            solid = solids.timber_solid(chunk, dict(r))
        except ship.NotReady:
            continue
        measured = timbers.is_measured(r)
        base = LAYERS["KEELSON"]
        colour, hatch = (base, "") if measured else (models3d.lighter(base), HATCH)
        d.fills.append(("KEELSON" if measured else RECONSTRUCTED + "KEELSON", _box_of(solid, water), colour, hatch, False))
        entry = (f"{timbers.kind_title('keelson')} - {timbers.status_title(timbers.status_of(r))}", colour, hatch)
        if entry not in d.legend:
            d.legend.append(entry)
    try:
        keel = solids.keel_solid(chunk)
    except ship.NotReady:
        keel = None
    if keel is not None:
        ring = _box_of(keel, water)
        colour = models3d.lighter(models3d.KIND_COLOURS["keel"])
        d.fills.append((RECONSTRUCTED + "KEEL", ring, colour, HATCH, False))
        n = len(ring) // 2
        x = float(ring[:n, 0].mean())  # the name in the box, in the keel's own dark colour
        z = 0.5 * (np.interp(x, ring[:n, 0], ring[:n, 1]) + np.interp(x, ring[n:, 0][::-1], ring[n:, 1][::-1]))
        d.label(x, float(z), keel.name, models3d.KIND_COLOURS["keel"])
        d.legend.append((f"{timbers.kind_title('keel')} - {timbers.status_title(keel.status)}", colour, HATCH))
    stations = _of(data, "station")
    runs, run = [], []
    for s in stations:
        ends = [p[[0, -1], 1] for p in lines.shown(s) if len(p)]
        if not ends or s.get("reached"):  # no line, or its highest end is the window's edge: the line breaks
            runs.append(run)
            run = []
            continue
        run.append((s["value"], max(float(e.max()) for e in ends)))
    runs.append(run)
    drawn = [np.array(sorted(r)) for r in runs if len(r) > 1]
    if drawn:
        d.add("SHEER", drawn, LAYERS["SHEER"])
        run = max(drawn, key=len)  # the name midway along its longest span: the stations' names are at their x
        k = int(np.argmax(np.diff(run[:, 0])))
        at = 0.5 * (run[k] + run[k + 1])
        d.label(at[0], at[1], t("orlo conservato", "preserved upper edge"), LAYERS["SHEER"])
        d.legend.append((t("orlo conservato: l'estremo più alto delle stazioni", "preserved upper edge: the highest end of "
                           "the stations"), LAYERS["SHEER"], LINE_SWATCH))
    reached = ", ".join(s["name"] for s in stations if s.get("reached"))
    if reached:
        d.notes.append(t(f"Orlo conservato interrotto alle stazioni che arrivano al limite del disegno: {reached}.",
                         f"Preserved upper edge broken at the stations that reach the limit of the drawing: {reached}.")
                       if drawn else
                       t(f"Orlo conservato non disegnato: le stazioni arrivano al limite del disegno ({reached}).",
                         f"Preserved upper edge not drawn: the stations reach the limit of the drawing ({reached})."))


def _background(chunk, frame: ship.Frame, background: str, x_range, y_range, resolution: float):
    """The raster under the plan of the wreck and a note: the hillshade of the relief on the sheet's grid ((u, v) =
    (x, sign * y) are the sheet's coordinates), the orthophoto, or nothing."""
    if background not in BACKGROUNDS:
        raise ValueError(f"background: one of {', '.join(BACKGROUNDS)}")
    if background == "none":
        return None, None
    if background == "ortho":
        raster = ortho_in_ship(chunk, frame, x_range, y_range, resolution)
        if raster is None:
            return None, t("Il chunk non ha un'ortofoto: la pianta è senza sfondo.",
                           "The chunk has no orthophoto: the plan has no background.")
        return raster, t("Sfondo: ortofoto del chunk.", "Background: the chunk's orthophoto.")
    raster = hillshade_in_ship(chunk, x_range, y_range, resolution)
    if raster is None:
        return None, t("Sfondo: nessuno, il rilievo dello scafo manca o è di un altro sistema della nave.",
                       "Background: none, the relief of the hull is missing or of another ship frame.")
    return raster, t("Sfondo: ombreggiatura del rilievo dello scafo dall'alto.",
                     "Background: hillshade of the relief of the hull from above.")


def hillshade_in_ship(chunk, x_range, y_range, resolution: float):
    """The hillshade of the relief of the hull in grey on the sheet of a plan, read at about ``resolution`` metres
    (never more than :data:`MAX_RASTER` pixels a side): an ``(H, W, 4)`` uint8 array (transparent off the model) and its
    extent, or None without a fresh relief."""
    from . import relief

    if relief.entry(chunk) is None or relief.is_stale(chunk):
        return None
    path = relief.path_of(chunk, "hillshade")
    if not Path(path).exists():
        return None
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        a = src.transform
        inv = ~a
        corners = np.array([inv @ (x, y) for x in x_range for y in y_range])
        c0, r0 = np.floor(corners.min(axis=0)).astype(int)
        c1, r1 = np.ceil(corners.max(axis=0)).astype(int)
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, src.width), min(r1, src.height)
        if c1 <= c0 or r1 <= r0:
            return None
        step = max(1.0, resolution / abs(a.a), max(c1 - c0, r1 - r0) / MAX_RASTER)
        shape = (max(1, int(round((r1 - r0) / step))), max(1, int(round((c1 - c0) / step))))
        z = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=shape).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            z[z == src.nodata] = np.nan
        xa, ya = a @ (c0, r0)
        xb, yb = a @ (c1, r1)
    if ya < yb:  # rows going up the sheet: the image's first row is its top
        z, (ya, yb) = z[::-1], (yb, ya)
    if xa > xb:
        z, (xa, xb) = z[:, ::-1], (xb, xa)
    ok = np.isfinite(z)
    if not ok.any():
        return None
    lo, hi = np.percentile(z[ok], [2.0, 98.0])
    grey = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * 180.0 + 60.0
    image = np.dstack([grey, grey, grey, np.where(ok, 255.0, 0.0)]).astype(np.uint8)
    return image, (float(xa), float(xb), float(yb), float(ya))


def _opened(geom):
    """``geom`` without the slivers of rounding (an opening by :data:`SLIVER`)."""
    return geom.buffer(-SLIVER, join_style="mitre").buffer(SLIVER, join_style="mitre")


def visible_parts(chunk, records, frame: ship.Frame) -> dict:
    """The visible and hidden parts of each measured strake (plank polygon) as :func:`tracing.update_visible` defines
    them - its polygon minus the frames, keelsons and stringers over it - computed here and returned, never written:
    ``{record id: (visible, hidden)}`` shapely geometries in the ship plan. Slivers of rounding along the strake's own
    outline are opened away (they would join the hidden parts into rings around the visible ones)."""
    import shapely

    from . import tracing

    cutters = [_valid(shapely.Polygon(_ship_xy(chunk, r, frame))) for r in records
               if r["kind"] in tracing.CUTTERS and r["geometry"]["type"] == "polygon"]
    over = shapely.union_all(cutters) if cutters else None
    out = {}
    for r in records:
        if r["kind"] not in tracing.CUT or r["geometry"]["type"] != "polygon" or not timbers.is_measured(r):
            continue
        poly = _valid(shapely.Polygon(_ship_xy(chunk, r, frame)))
        if over is None:
            out[r["id"]] = (poly, shapely.Polygon())
            continue
        visible = poly.difference(over)
        out[r["id"]] = (visible, _opened(poly.difference(visible)))
    return out


def _valid(poly):
    return poly if poly.is_valid else poly.buffer(0)


def _polygons(geom) -> list:
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty and g.area > 1e-6]


def _ring_fills(d: Drawing, layer: str, geom, colour: str, hatch: str, dashed: bool) -> None:
    """The rings of ``geom`` (sheet coordinates) as fills, each polygon's outer ring (counter-clockwise) followed by its
    holes (clockwise): :func:`figure` and :func:`write_dxf` fill an outer ring with the holes after it left empty."""
    import shapely

    for poly in _polygons(geom):
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        for ring in [poly.exterior, *poly.interiors]:
            d.fills.append((layer, np.asarray(ring.coords)[:-1], colour, hatch, dashed))


def _keel_ring(chunk) -> np.ndarray | None:
    """The typed keel in plan (a box along the keel line, symmetric about it: the same on the sheet for both bows),
    None when none is typed or a measured keel replaces it."""
    from . import solids

    keel = ship.state(chunk).get("keel")
    if not keel or solids._measured_keel(chunk):
        return None
    h = keel["siding"] / 2.0
    return np.array([[keel["x0"], -h], [keel["x1"], -h], [keel["x1"], h], [keel["x0"], h]])


def _to_sheet(geom, sign: float):
    import shapely

    return shapely.transform(geom, lambda p: p * [1.0, sign])


def _upright(degrees: float) -> float:
    """An angle of text that reads from the left or from below: in (-90, 90]."""
    a = (degrees + 180.0) % 180.0
    return a - 180.0 if a > 90.0 else a


def _along(poly) -> tuple | None:
    """The long axis of a polygon (sheet): (centre, unit vector, length) when it is at least :data:`ACROSS_RATIO`
    times as long as wide, else None."""
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    (long, short) = (e1, e2) if np.hypot(*e1) >= np.hypot(*e2) else (e2, e1)
    length, width = float(np.hypot(*long)), float(np.hypot(*short))
    if width <= 0 or length < ACROSS_RATIO * width:
        return None
    return rect.mean(axis=0), long / length, length


def _name_spot(poly, kind: str, visible=None) -> tuple[float, float, float]:
    """Where a polygon's name goes on the sheet and its angle - a simple rule that keeps the names of a hull apart:
    a strake at the middle of its length (on the visible part between two frames nearest to it), a timber lying
    across the keel line (a frame) near its end farther from the keel line, along it; a longitudinal timber (the
    keelson, a stringer) a fifth of its length from its start, along it; anything else at its representative point."""
    import shapely

    if kind in ("plank", "ceiling"):
        x0, _y0, x1, _y1 = poly.bounds
        mid = 0.5 * (x0 + x1)
        pieces = _polygons(visible) if visible is not None else []
        if pieces:
            best = min(pieces, key=lambda g: (abs(g.centroid.x - mid), -g.area))
            at = best.representative_point()
        else:
            band = poly.intersection(shapely.box(mid - 1e-3, -1e9, mid + 1e-3, 1e9))
            at = band.representative_point() if not band.is_empty else poly.representative_point()
        return at.x, at.y, 0.0
    axis = _along(poly)
    if axis is None:
        at = poly.representative_point()
        return at.x, at.y, 0.0
    centre, u, length = axis
    angle = _upright(np.degrees(np.arctan2(u[1], u[0])))
    if abs(u[1]) > abs(u[0]):  # across the keel line: towards the end farther from it (ties: the sheet's top)
        ends = [centre + u * length / 2.0, centre - u * length / 2.0]
        far = max(ends, key=lambda e: (round(abs(e[1]), 2), e[1]))
        at = centre + (far - centre) * (2.0 * NAME_AT["frame"] - 1.0)
    else:  # along the keel line: from the end at the lower x
        start = centre - u * length / 2.0 * np.sign(u[0] or 1.0)
        at = start + (centre - start) * 2.0 * NAME_AT["longitudinal"]
    return float(at[0]), float(at[1]), angle


def _point_along(p: np.ndarray, fraction: float) -> np.ndarray:
    seg = np.hypot(*np.diff(p, axis=0).T)
    along = np.r_[0.0, np.cumsum(seg)]
    s = fraction * along[-1]
    return np.array([np.interp(s, along, p[:, 0]), np.interp(s, along, p[:, 1])])


def _legend_text(kind: str, status: str) -> str:
    return f"{timbers.kind_title(kind)} - {timbers.status_title(status)}"


def wreck_plan(chunk, scale: int | None = None, background: str = "relief") -> Drawing:
    """The plan of the wreck ("Pianta del relitto"): every record of the catalogue on the sheet of a plan (x to the
    right, ``sign`` * y up, as :func:`plan`). A polygon is outlined and lightly filled in its kind's colour
    (:data:`models3d.KIND_COLOURS`; the planking lighter still, the relief under it stays readable) and named
    (:func:`_name_spot`); a measured strake's parts under a frame, the keelson or a stringer (:func:`visible_parts`)
    outlined dashed; a reconstruction (any status but measured: the typed keel, the strakes of the sections, imported
    planks) lighter and hatched on ``RICOSTRUZIONE_<KIND>``, a measured timber on ``LEGNI_<KIND>``; the typed keel is
    hatched only where no measured timber lies over it (there it is outlined dashed) and named only when mostly seen;
    a strake of the sections is left out where a measured strake has its name. Lines and points drawn in the same
    colours, a line named at :data:`NAME_AT` ``["line"]`` of its length. ``background``: "relief" (the hillshade of the
    relief of the hull), "ortho" (the orthophoto, :func:`ortho_in_ship`) or "none". Legend per kind and status, scale
    bar, north, title block. :class:`Empty` without records or a typed keel. Nothing of the chunk is written."""
    import shapely

    from . import models3d

    ship.check_metric(chunk)
    frame = ship.need_frame(chunk)
    sign = float(frame.sign)
    d = Drawing("wreck_plan", t(*KIND_TITLES["wreck_plan"]), north=_north(chunk, frame))
    records = timbers.timbers_of(chunk)
    parts = visible_parts(chunk, records, frame)
    traced_names = {r["name"] for r in records if r["kind"] == "plank" and r["geometry"]["type"] == "polygon"
                    and timbers.is_measured(r)}
    order = {k: i for i, k in enumerate(DRAW_ORDER)}
    hidden_names, shown = [], {}
    keel_ring = _keel_ring(chunk)
    if keel_ring is not None:
        colour = models3d.lighter(models3d.KIND_COLOURS["keel"])
        keel = shapely.Polygon(keel_ring)
        over = [_valid(shapely.Polygon(_ship_xy(chunk, r, frame))) for r in records
                if r["geometry"]["type"] == "polygon" and timbers.is_measured(r) and len(r["geometry"]["xy"]) >= 3]
        seen = _opened(keel.difference(shapely.union_all(over))) if over else keel
        layer = RECONSTRUCTED + "KEEL"
        _ring_fills(d, layer, _to_sheet(seen, sign), colour, HATCH, False)
        _ring_fills(d, layer, _to_sheet(_opened(keel.difference(seen)), sign), colour, "", True)
        if seen.area >= 0.5 * keel.area:  # under the keelson its name would sit on the keelson's: the legend names it
            at = _to_sheet(seen, sign).representative_point()
            d.label(at.x, at.y, t("chiglia", "keel"), models3d.KIND_COLOURS["keel"])
        shown[("keel", "reconstructed:given")] = (colour, HATCH)
    for r in sorted(records, key=lambda r: order.get(r["kind"], len(order))):
        status = timbers.status_of(r)
        measured = status == "measured"
        if status == "reconstructed:sections" and r["name"] in traced_names:
            hidden_names.append(r["name"])
            continue
        kind, gtype = r["kind"], r["geometry"]["type"]
        base = models3d.KIND_COLOURS.get(kind, models3d.KIND_COLOURS["other"])
        colour, hatch = (base, "") if measured else (models3d.lighter(base), HATCH)
        layer = (MEASURED if measured else RECONSTRUCTED) + kind.upper()
        xy = _ship_xy(chunk, r, frame)
        sheet = xy * [1.0, sign]
        name = r["name"] or r["id"]
        if gtype == "polygon" and len(xy) >= 3:
            poly = _valid(shapely.Polygon(xy))
            visible = None
            if r["id"] in parts:
                visible, hidden = parts[r["id"]]
                _ring_fills(d, layer, _to_sheet(visible, sign), colour, hatch, False)
                _ring_fills(d, layer, _to_sheet(hidden, sign), colour, "", True)
                visible = _to_sheet(visible, sign)
            else:
                _ring_fills(d, layer, _to_sheet(poly, sign), colour, hatch, False)
            x, y, angle = _name_spot(_to_sheet(poly, sign), kind, visible)
            d.label(x, y, name, colour, angle)
        elif gtype == "line" and len(sheet) > 1:
            d.add(layer, [sheet], colour)
            at = _point_along(sheet, NAME_AT["line"])
            d.label(at[0], at[1], name, colour)
            hatch = LINE_SWATCH
        else:
            c, a = sheet[0], POINT_MARK
            d.add(layer, [[c - [a, 0], c + [a, 0]], [c - [0, a], c + [0, a]]], colour)
            d.label(c[0], c[1], name, colour)
            hatch = LINE_SWATCH
        shown.setdefault((kind, status), (colour, hatch))
    if not d.fills and not d.lines:
        raise Empty(t(f"Nessun legno per il disegno «{d.title}».", f"No timbers for the drawing '{d.title}'."))
    statuses = list(timbers.STATUS)
    for (kind, status), (colour, hatch) in sorted(shown.items(), key=lambda kv: (
            order.get(kv[0][0], len(order)), statuses.index(kv[0][1]) if kv[0][1] in statuses else len(statuses))):
        d.legend.append((_legend_text(kind, status), colour, hatch))
    x0, x1, y0, y1 = d.extent
    x0, x1, y0, y1 = x0 - WRECK_MARGIN, x1 + WRECK_MARGIN, y0 - WRECK_MARGIN, y1 + WRECK_MARGIN
    scale = int(scale or drawing_scale(x1 - x0, y1 - y0))
    raster, said = _background(chunk, frame, background, (x0, x1), (y0, y1), scale * 0.00025)  # 4 pixels per mm
    d.raster = raster
    notes = [t("Legni ricalcati sul rilievo dello scafo, nel sistema della nave. Contorno pieno e campitura leggera nel "
               "colore del tipo: legno misurato; contorno a trattini: parte nascosta sotto un'ordinata, il paramezzale o "
               "una serretta.",
               "Timbers traced on the relief of the hull, in the ship frame. Solid outline and a light fill in the kind's "
               "colour: a measured timber; dashed outline: a part hidden under a frame, the keelson or a stringer.")]
    if said:
        notes.append(said)
    if hidden_names:
        notes.append(t(f"Fasce delle sezioni (ipotesi) non disegnate dove una fascia ricalcata ha lo stesso nome: "
                       f"{', '.join(sorted(set(hidden_names)))}.",
                       f"Strakes of the sections (hypothesis) not drawn where a traced strake has the same name: "
                       f"{', '.join(sorted(set(hidden_names)))}."))
    if not ship.is_scaled(chunk):
        notes.append(t("Modello senza scala nota: le misure non sono metri veri.",
                       "A model without a known scale: the measures are not true metres."))
    extra = title_fields(chunk)["notes"]
    d.notes = notes + ([extra] if extra else [])
    d.block = _block(chunk, frame)
    d.bounds = (x0, x1, y0, y1)  # the timbers and the margin around them
    x0, x1, y0, y1 = d.extent
    d.scale = int(scale or drawing_scale(x1 - x0, y1 - y0))
    return d


MAKERS = {"body_plan": body_plan, "half_breadth": half_breadth, "profile": profile}


def make(chunk, kind: str, scale: int | None = None) -> Drawing:
    """One drawing of the sections of the chunk (up to date ones: else :class:`ship.NotReady`); the plan of the wreck
    (:func:`wreck_plan`) needs no sections."""
    if kind == "wreck_plan":
        d = wreck_plan(chunk, scale)
        _hatch_note(d)
        return d
    ship.check_metric(chunk)
    frame = ship.need_frame(chunk)
    data = lines.load(chunk)
    if data is None:
        raise ship.NotReady(t("Calcola prima le sezioni.", "Compute the sections first."))
    if lines.is_stale(chunk, data):
        raise ship.NotReady(t("Le sezioni vengono da un altro sistema della nave, da un altro modello o da un'altra "
                              "georeferenziazione: controlla che la chiglia sia ancora al suo posto e ricalcolale.",
                              "The sections come from another ship frame, another model or another georeference: check "
                              "that the keel is still in its place and compute them again."))
    d = plan(chunk, data, frame, scale) if kind == "plan" else MAKERS[kind](chunk, data, frame)
    if not any(layer not in ("KEEL", "GRID", "LEVEL") for layer, _p, _c in d.lines) and d.raster is None:
        raise Empty(t(f"Nessuna linea per il disegno «{d.title}».", f"No lines for the drawing '{d.title}'."))
    d.notes = _notes(chunk, data, frame, kind) + d.notes
    _hatch_note(d)
    d.block = _block(chunk, frame)
    x0, x1, y0, y1 = d.extent
    d.scale = int(scale or drawing_scale(x1 - x0, y1 - y0))
    return d


def _hatch_note(d: Drawing) -> None:
    """The note on the hatched parts, when a drawing has any (a reconstruction is never presented as measured)."""
    if any(hatch for _l, _r, _c, hatch, _d in d.fills):
        d.notes.append(t(*HATCH_NOTE))


def _bar_length(metres: float) -> float:
    """A round length for the three parts of a scale bar at most a third as long as the drawing."""
    return next((step for step in (100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2) if step * 9 <= metres), 0.1)


def figure(d: Drawing):
    """The sheet of a drawing: the metres keep the drawing's scale exactly on the paper."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
    from matplotlib.patheffects import withStroke

    x0, x1, y0, y1 = d.extent
    k = 1000.0 / d.scale  # mm of paper per metre
    draw_w, draw_h = (x1 - x0) * k, (y1 - y0) * k
    width = max(draw_w + 2 * MARGIN, MIN_SHEET)
    # the notes in the left part of the block, the scale bar and the north in the right one: about a mm for a letter
    wrapped = [row for note in d.notes for row in textwrap.wrap(note, int(0.62 * (width - 2 * MARGIN)))]
    block_h = max(TITLE_BLOCK, 9.0 + 3.4 * (len(d.block) + len(d.legend)) + 3.0 * len(wrapped))
    height = draw_h + MARGIN + 6.0 + block_h
    fig = Figure(figsize=(width / 25.4, height / 25.4))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([(width - draw_w) / 2 / width, block_h / height, draw_w / width, draw_h / height])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    if d.raster is not None:
        image, (a, b, c, e) = d.raster
        ax.imshow(image, extent=(a, b, c, e), interpolation="bilinear", zorder=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    for layer, rings, colour, hatch, dashed in _shapes(d.fills):
        filled = not dashed
        ax.add_patch(PathPatch(_path(rings), facecolor=(*_rgb(colour), _alpha(layer)) if filled else "none",
                               edgecolor=colour, hatch=hatch if filled and hatch else None, ls="--" if dashed else "-",
                               lw=WEIGHTS.get(layer, TIMBER_WEIGHT) / 0.3528, joinstyle="round", clip_on=False,
                               zorder=1.5))
    for layer, p, colour in d.lines:
        if layer == "RAW":  # a sheet shows one line for each section; the raw ones are a layer of the DXF
            continue
        style = "--" if layer in ("KEEL", "LEVEL") else ":" if layer.startswith(RECONSTRUCTED) else "-"
        ax.plot(p[:, 0], p[:, 1], color=colour or LAYERS.get(layer, "#000000"), lw=WEIGHTS.get(layer, TIMBER_WEIGHT) / 0.3528,
                ls=style, solid_joinstyle="round", clip_on=False, zorder=2)
    halo = [withStroke(linewidth=1.2, foreground="white")]
    named = d.kind == "wreck_plan"  # a name on its timber, larger on a larger sheet
    size = float(np.clip(5.0 * width / MIN_SHEET, 5.0, 9.0)) if named else 5
    for i, (x, y, text, colour) in enumerate(d.labels):
        extra = {"rotation": d.rotations[i], "rotation_mode": "anchor"} if i in d.rotations else {}
        ax.annotate(text, (x, y), xytext=(0, 0 if named else 2), textcoords="offset points", ha="center",
                    va="center" if named else "bottom", fontsize=size, annotation_clip=False, zorder=3,
                    color=colour or "#000000", path_effects=halo, **extra)
    fig.text(MARGIN / width, 1 - 7.0 / height, f"{d.title} - 1:{d.scale}", fontsize=10, fontweight="bold")
    if d.kind == "body_plan" and d.axes[0]:
        ax.annotate(d.axes[0], (x0, y1), xytext=(0, 9), textcoords="offset points", fontsize=6, annotation_clip=False)
        ax.annotate(d.axes[1], (x1, y1), xytext=(0, 9), textcoords="offset points", fontsize=6, ha="right", annotation_clip=False)
    # the title block
    left = MARGIN / width
    y = (block_h - 7.0) / height
    for label, value in d.block.items():
        fig.text(left, y, f"{label}: {value}", fontsize=6.5)
        y -= 3.4 / height
    for text, colour, hatch in d.legend:  # a swatch as the timbers are drawn, and what it is
        if hatch == LINE_SWATCH:
            fig.add_artist(Line2D([left, left + 5.0 / width], [y + 0.8 / height] * 2, transform=fig.transFigure,
                                  color=colour, linewidth=1.0))
        else:
            fig.add_artist(Rectangle((left, y - 0.3 / height), 5.0 / width, 2.2 / height, transform=fig.transFigure,
                                     facecolor=(*_rgb(colour), FILL_ALPHA), edgecolor=colour, hatch=hatch or None,
                                     linewidth=0.5))
        fig.text(left + 6.5 / width, y, text, fontsize=6)
        y -= 3.4 / height
    for line in wrapped:
        fig.text(left, y, line, fontsize=5.5, color="#333333")
        y -= 3.0 / height
    # the scale bar, exact on the paper
    bar = _bar_length(x1 - x0)
    bx, by = 1 - (MARGIN + 3 * bar * k) / width, (block_h - 12.0) / height
    for i in range(3):
        fig.add_artist(Rectangle((bx + i * bar * k / width, by), bar * k / width, 1.6 / height, transform=fig.transFigure,
                                 facecolor="black" if i % 2 == 0 else "white", edgecolor="black", linewidth=0.4))
        fig.text(bx + i * bar * k / width, by + 2.4 / height, num(i * bar, 2 if bar < 1 else 0 if bar == int(bar) else 1),
                 fontsize=5, ha="center")
    fig.text(bx + 3 * bar * k / width, by + 2.4 / height, f"{num(3 * bar, 2 if bar < 1 else 0 if bar == int(bar) else 1)} m",
             fontsize=5, ha="center")
    if d.north is not None:
        cx, cy = bx - 12.0 / width, (block_h - 13.0) / height  # beside the scale bar
        tip = (cx + d.north[0] * 7.0 / width, cy + d.north[1] * 7.0 / height)
        tail = (cx - d.north[0] * 7.0 / width, cy - d.north[1] * 7.0 / height)
        fig.add_artist(FancyArrowPatch(tail, tip, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                       linewidth=0.6, color="black"))
        fig.text(tip[0] + d.north[0] * 3.0 / width, tip[1] + d.north[1] * 3.0 / height, "N", fontsize=7, ha="center", va="center")
    return fig


def _rgb(colour: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return to_rgb(colour)


def _alpha(layer: str) -> float:
    return PLATE_ALPHA if layer.endswith(("_PLANK", "_CEILING")) else FILL_ALPHA


def _shapes(fills) -> list:
    """The fills grouped as they were written (:func:`_ring_fills`): an outer ring with the clockwise rings right after
    it on the same layer as its holes; a dashed ring alone. ``[(layer, [rings], colour, hatch, dashed)]``."""
    out = []
    for layer, ring, colour, hatch, dashed in fills:
        ring = np.asarray(ring, float)
        hole = not dashed and _signed_area(ring) < 0
        if hole and out and not out[-1][4] and out[-1][0] == layer and out[-1][2] == colour:
            out[-1][1].append(ring)
        else:
            out.append((layer, [ring], colour, hatch if not hole else "", dashed))
    return out


def _path(rings):
    from matplotlib.path import Path as MplPath

    vertices, codes = [], []
    for ring in rings:
        vertices += [*ring.tolist(), ring[0].tolist()]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(np.asarray(vertices, float), codes)


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def write_dxf(d: Drawing, path) -> Path:
    """DXF in metres on the sheet, a layer per kind of line."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.header["$LWDISPLAY"] = 1
    msp = doc.modelspace()
    colours = {"STATIONS": 7, "LONGITUDINALS": 7, "HORIZONTALS": 7, "RAW": 8, "KEEL": 1, "LEVEL": 5, "TIMBERS": 30,
               "GRID": 9, "TEXT": 7, "Fasce": 3}
    for name, colour in colours.items():
        doc.layers.add(name, color=colour, lineweight=int(WEIGHTS.get(name, 0.18) * 100))
    used = [(layer, colour) for layer, _p, colour in d.lines] + [(layer, colour) for layer, _r, colour, _h, _d in d.fills]
    for name, colour in used:  # the layers of 0.3.0 only when drawn: a 0.2.0 drawing keeps its table
        if name not in doc.layers:
            layer = doc.layers.add(name, color=7, lineweight=int(WEIGHTS.get(name, TIMBER_WEIGHT) * 100))
            layer.rgb = tuple(int(round(v * 255)) for v in _rgb(colour or LAYERS.get(name, "#000000")))
    paper = max(d.scale, 1) / 1000.0  # metres of the drawing per mm of paper
    for layer, ring, _colour, hatch, dashed in d.fills:
        ring = np.asarray(ring, float)
        attribs = {"layer": layer}
        if dashed:
            attribs.update(linetype="DASHED", ltscale=paper * 4.0)
        msp.add_lwpolyline(ring.tolist(), close=True, dxfattribs=attribs)
    for layer, rings, _colour, hatch, dashed in _shapes(d.fills):
        if hatch == HATCH and not dashed:  # the outer ring and its holes: the hatch leaves the holes empty
            h = msp.add_hatch(dxfattribs={"layer": layer})
            h.set_pattern_fill("ANSI31", scale=paper * 2.0 / 3.175)  # a line every 2 mm of paper
            for k, ring in enumerate(rings):
                h.paths.add_polyline_path(ring.tolist(), is_closed=True,
                                          flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL if k == 0 else
                                          ezdxf.const.BOUNDARY_PATH_DEFAULT)
    for layer, p, _colour in d.lines:
        if len(p) > 1:
            msp.add_lwpolyline(p.tolist(), dxfattribs={"layer": layer})
    x0, x1, y0, y1 = d.extent
    size = max(x1 - x0, y1 - y0) * 0.008
    for i, (x, y, text, _colour) in enumerate(d.labels):
        attribs = {"layer": "TEXT", **({"rotation": d.rotations[i]} if i in d.rotations else {})}
        msp.add_text(text, height=size, dxfattribs=attribs).set_placement((x, y))
    msp.add_text(f"{d.title} - 1:{d.scale}", height=size * 2, dxfattribs={"layer": "TEXT"}).set_placement((x0, y1 + size * 6))
    for i, line in enumerate(d.notes):
        msp.add_text(line, height=size, dxfattribs={"layer": "TEXT"}).set_placement((x0, y0 - size * 2 * (i + 2)))
    for i, (text, _colour, hatch) in enumerate(d.legend):
        below = y0 - size * 2 * (len(d.notes) + i + 3)
        msp.add_text(f"{text}{' (////)' if hatch == HATCH else ''}", height=size,
                     dxfattribs={"layer": "TEXT"}).set_placement((x0, below))
    doc.saveas(path)
    return Path(path)


def _signature(chunk, kind: str) -> dict:
    """What a drawing was made from: the sections (with their water and what they were cut from), and for the plan the
    catalogue. It is out of date when any of these changes."""
    data = lines.load(chunk) or {}
    frame = ship.frame_of(chunk)
    source = lines.fingerprint(chunk, frame) if frame is not None else None
    if kind == "wreck_plan":  # no sections on it: the frame and the model, the records, the relief under them
        out = {"source": source, "relief": _relief_signature(chunk) if frame is not None else None}
    else:
        out = {"built": data.get("built"), "water": data.get("water"), "source": source}
    if kind in ("plan", "wreck_plan"):  # the records proper: their plan caches and solids change no drawing
        text = json.dumps([_clean_record(r) for r in timbers.timbers_of(chunk)], sort_keys=True)
        out["timbers"] = hashlib.sha256(text.encode()).hexdigest()
    if kind in ("body_plan", "profile", "wreck_plan") and frame is not None and _traced(chunk):
        out["traced"] = _hash({"timbers": [_clean_record(r) for r in timbers.timbers_of(chunk)],
                               "keel": {k: v for k, v in (ship.state(chunk).get("keel") or {}).items()
                                        if k not in ("solid", "volume")},
                               "relief": _relief_signature(chunk)})
    if kind in ("body_plan", "profile", "plan"):
        found = _land_lines(chunk) if frame is not None else []
        if found:
            out["lands"] = hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()
    return out


def _relief_signature(chunk) -> dict | None:
    from . import relief

    e = relief.entry(chunk)
    if e is None:
        return None
    try:
        mtime = relief.path_of(chunk, "hillshade").stat().st_mtime
    except (OSError, KeyError, ship.NotReady):
        mtime = None
    return {"id": e.get("id"), "stale": relief.is_stale(chunk), "hillshade": mtime}


def out_of_date(chunk) -> list[str]:
    """The drawings written that no longer say what the chunk says."""
    made = chunk.meta.get(ship.KEY, {}).get("drawings") or {}
    return [k for k, v in made.items() if v.get("signature") != _signature(chunk, k)]


def write_all(chunk, kinds=KINDS, scale: int | None = None, folder=None, progress=None) -> list[Path]:
    """Each drawing in ``<chunk data>/shipwreck/drawings`` (or ``folder``) as SVG, PDF, PNG and DXF; a kind with no lines
    is left out when others are made. The drawings written are noted in ``chunk.meta``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    out = Path(folder) if folder else ship.folder(chunk) / "drawings"
    out.mkdir(parents=True, exist_ok=True)
    written, skipped, kept = [], {}, {}
    for i, kind in enumerate(kinds):
        progress(i / len(kinds), t(*KIND_TITLES[kind]))
        try:
            d = make(chunk, kind, scale)
        except Empty as exc:
            skipped[kind] = str(exc)
            continue
        fig = figure(d)
        names = [out / f"{kind}.{suffix}" for suffix in ("svg", "pdf", "png")]
        for p in names:
            fig.savefig(p, dpi=150 if p.suffix == ".png" else None)
        names.append(write_dxf(d, out / f"{kind}.dxf"))
        written += names
        kept[kind] = {"scale": d.scale, "files": [p.name for p in names], "water": lines.is_on(chunk),
                      "signature": _signature(chunk, kind)}
    if not written:
        raise ship.NotReady(" ".join(skipped.values()) or t("Nessun disegno.", "No drawing."))
    if folder is None:
        ship.state(chunk)["drawings"] = kept
    progress(1.0, t("Disegni pronti", "Drawings ready"))
    return written
