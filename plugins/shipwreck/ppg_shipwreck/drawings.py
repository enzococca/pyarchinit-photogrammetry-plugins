"""The drawings of the hull: body plan, half-breadth plan, profile, plan over the orthophoto. A drawing is a
:class:`Drawing` - lines by layer, in metres on the sheet - that one function renders with matplotlib (SVG, PDF, PNG) and
another as DXF with a layer per kind. Only measured lines: nothing is faired, and the sheet says what it shows.

Plans are drawn from above: x to the right and the left of x upwards, so that a frame with x running aft
(``sign`` -1, y to the right of x) is not mirrored; the body plan is seen from aft, port on the left."""
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

KINDS = ("body_plan", "half_breadth", "profile", "plan")
KIND_TITLES = {"body_plan": ("Piano di costruzione: sezioni trasversali", "Body plan: cross-sections"),
               "half_breadth": ("Piano delle linee d'acqua: orizzontali", "Half-breadth plan: horizontals"),
               "profile": ("Profilo: longitudinali", "Profile: longitudinals"),
               "plan": ("Pianta dello scafo con le stazioni", "Plan of the hull with the stations")}
LAYERS = {"STATIONS": "#111111", "LONGITUDINALS": "#111111", "HORIZONTALS": "#111111", "RAW": "#9a9a9a", "KEEL": "#b02020",
          "LEVEL": "#2060b0", "TIMBERS": "#b06000", "GRID": "#bbbbbb"}
WEIGHTS = {"STATIONS": 0.35, "LONGITUDINALS": 0.35, "HORIZONTALS": 0.35, "RAW": 0.13, "KEEL": 0.35, "LEVEL": 0.35,
           "TIMBERS": 0.5, "GRID": 0.18}  # millimetres of paper
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

    def add(self, layer: str, polylines, colour: str | None = None) -> None:
        self.lines += [(layer, np.asarray(p, float), colour) for p in polylines]

    def label(self, x: float, y: float, text: str, colour: str | None = None) -> None:
        self.labels.append((float(x), float(y), text, colour))

    @property
    def extent(self) -> tuple:
        pts = [p for _layer, p, _colour in self.lines if len(p)]
        all_points = np.vstack(pts) if pts else np.zeros((1, 2))
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        if self.raster is not None:
            x0, x1, y0, y1 = self.raster[1]
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
    extra = title_fields(chunk)["notes"]
    return notes + ([extra] if extra else [])


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
    scale = scale or drawing_scale(x1 - x0, 2 * half)
    raster = ortho_in_ship(chunk, frame, (x0, x1), (-half, half), scale * 0.00025)  # 4 pixels per mm of paper
    if raster is not None:
        d.raster = raster
    else:
        d.notes.append(t("Il chunk non ha un'ortofoto: la pianta mostra solo le stazioni.",
                         "The chunk has no orthophoto: the plan shows the stations only."))
    return d


MAKERS = {"body_plan": body_plan, "half_breadth": half_breadth, "profile": profile}


def make(chunk, kind: str, scale: int | None = None) -> Drawing:
    """One drawing of the sections of the chunk (up to date ones: else :class:`ship.NotReady`)."""
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
    d.block = _block(chunk, frame)
    x0, x1, y0, y1 = d.extent
    d.scale = int(scale or drawing_scale(x1 - x0, y1 - y0))
    return d


def _bar_length(metres: float) -> float:
    """A round length for the three parts of a scale bar at most a third as long as the drawing."""
    return next((step for step in (100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2) if step * 9 <= metres), 0.1)


def figure(d: Drawing):
    """The sheet of a drawing: the metres keep the drawing's scale exactly on the paper."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.patheffects import withStroke

    x0, x1, y0, y1 = d.extent
    k = 1000.0 / d.scale  # mm of paper per metre
    draw_w, draw_h = (x1 - x0) * k, (y1 - y0) * k
    width = max(draw_w + 2 * MARGIN, MIN_SHEET)
    # the notes in the left part of the block, the scale bar and the north in the right one: about a mm for a letter
    wrapped = [row for note in d.notes for row in textwrap.wrap(note, int(0.62 * (width - 2 * MARGIN)))]
    block_h = max(TITLE_BLOCK, 9.0 + 3.4 * len(d.block) + 3.0 * len(wrapped))
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
    for layer, p, colour in d.lines:
        if layer == "RAW":  # a sheet shows one line for each section; the raw ones are a layer of the DXF
            continue
        dashed = layer in ("KEEL", "LEVEL")
        ax.plot(p[:, 0], p[:, 1], color=colour or LAYERS[layer], lw=WEIGHTS[layer] / 0.3528, ls="--" if dashed else "-",
                solid_joinstyle="round", clip_on=False, zorder=2)
    halo = [withStroke(linewidth=1.2, foreground="white")]
    for x, y, text, colour in d.labels:
        ax.annotate(text, (x, y), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=5,
                    annotation_clip=False, zorder=3, color=colour or "#000000", path_effects=halo)
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


def write_dxf(d: Drawing, path) -> Path:
    """DXF in metres on the sheet, a layer per kind of line."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.header["$LWDISPLAY"] = 1
    msp = doc.modelspace()
    colours = {"STATIONS": 7, "LONGITUDINALS": 7, "HORIZONTALS": 7, "RAW": 8, "KEEL": 1, "LEVEL": 5, "TIMBERS": 30,
               "GRID": 9, "TEXT": 7}
    for name, colour in colours.items():
        doc.layers.add(name, color=colour, lineweight=int(WEIGHTS.get(name, 0.18) * 100))
    for layer, p, _colour in d.lines:
        if len(p) > 1:
            msp.add_lwpolyline(p.tolist(), dxfattribs={"layer": layer})
    x0, x1, y0, y1 = d.extent
    size = max(x1 - x0, y1 - y0) * 0.008
    for x, y, text, _colour in d.labels:
        msp.add_text(text, height=size, dxfattribs={"layer": "TEXT"}).set_placement((x, y))
    msp.add_text(f"{d.title} - 1:{d.scale}", height=size * 2, dxfattribs={"layer": "TEXT"}).set_placement((x0, y1 + size * 6))
    for i, line in enumerate(d.notes):
        msp.add_text(line, height=size, dxfattribs={"layer": "TEXT"}).set_placement((x0, y0 - size * 2 * (i + 2)))
    doc.saveas(path)
    return Path(path)


def _signature(chunk, kind: str) -> dict:
    """What a drawing was made from: the sections (with their water and what they were cut from), and for the plan the
    catalogue. It is out of date when any of these changes."""
    data = lines.load(chunk) or {}
    frame = ship.frame_of(chunk)
    out = {"built": data.get("built"), "water": data.get("water"),
           "source": lines.fingerprint(chunk, frame) if frame is not None else None}
    if kind == "plan":
        text = json.dumps(timbers.timbers_of(chunk), sort_keys=True)
        out["timbers"] = hashlib.sha256(text.encode()).hexdigest()
    return out


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
