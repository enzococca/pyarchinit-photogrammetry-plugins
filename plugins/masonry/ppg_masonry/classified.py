"""Courses, joints and stones from a drawing of a face classified by its orientation (the program's
:mod:`pyarchinit_photogrammetry.engine.orient_lines`, since 1.7.7).

The drawing is the lines drawn on the face's elevation (its ``(u, v)``; the outlines of the USM drawn there are left
out unless asked) or a file in the same ``(u, v)``. It is classified along the face's horizontal, the ``u`` axis
``((0, 0), (1, 0))``:

* the lines *along* it are the **beds** (the bed joints between two courses), sorted by ``v``;
* a *pair along* it (two parallel lines close together, also a band traced as a polygon) is a **course** of known
  thickness: its two lines are the beds below and above it;
* the lines *across* it are the **joints** (the head joints between two stones of a course); a *pair across* gives
  its two lines as two joints (a wide joint drawn on both sides leaves between them a cell too narrow to be a stone);
* the other directions are left out and counted.

A course lies between a bed and the bed directly above it (over the stretch of ``u`` both cover, less the stretches
where another bed comes between them). Its stones are the cells of the course split by the joints whose ``v`` extent
within the course covers at least ``joint_cover`` (half) of its height: the joint is taken across the whole course.
Cells narrower or lower than ``min_stone`` are not stones.

Nothing enters the project before the check plate: :func:`propose` reads, classifies and makes the stones,
:func:`plate` draws them with the beds and joints, :func:`accept` appends the stones to those the program keeps with
the elevation (``chunk.meta["drawings"]["frames"][<face>]["stones"][<face>]``, WKT in ``(u, v)``), skipping those that
overlap a stone already there by more than :data:`.stones.DUPLICATE_IOU` (so the same drawing taken again adds
nothing). A proposal lives in memory only.

Without the program 1.7.7 :data:`orient_lines` is None and :func:`available` False: the window hides the action and
the command line says what is missing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import walls
from .texts import num, plural, t
from .walls import NotReady

try:  # the program 1.7.7 or later
    from pyarchinit_photogrammetry.engine import orient_lines
except ImportError:  # an older program: the action is hidden
    orient_lines = None

AXIS = ((0.0, 0.0), (1.0, 0.0))  # the face's horizontal in its (u, v)
PLUGIN_DEFAULTS = {
    "joint_cover": 0.5,  # a joint splits a course when it covers this share of the course's height
    "min_stone": 0.05,  # m, cells narrower or lower than this are not stones
    "units": False,  # the outlines of the units drawn on the face read as lines too
}
FILE_SUFFIXES = (".gpkg", ".geojson", ".json", ".dxf")
NEEDS_PROGRAM = ("serve il programma 1.7.7 o successivo", "the program 1.7.7 or later is needed")
COLOURS = {"bed": "#1f77b4", "joint": "#ff7f0e", "unused": "#9a9a9a", "stone": "#c8a165", "edge": "#5b3a1e"}


def available() -> bool:
    """Whether the program classifies drawings (1.7.7 or later)."""
    return orient_lines is not None


def need_program() -> None:
    if orient_lines is None:
        raise NotReady(t(f"Disegno classificato: {NEEDS_PROGRAM[0]}.", f"Classified drawing: {NEEDS_PROGRAM[1]}."))


def _not_ready(exc) -> NotReady:
    """An error of the classification in the language of the interface."""
    italian = getattr(exc, "italian", None)
    return NotReady(t(italian, str(exc)) if italian else str(exc))


def split_params(params: dict | None) -> tuple[dict, dict]:
    """``params`` divided into those of the classification (:data:`orient_lines.DEFAULTS` and ``EXTRA_DEFAULTS``) and
    the plugin's (:data:`PLUGIN_DEFAULTS`), these checked; NotReady for a key neither knows or a value out of range."""
    params = dict(params or {})
    mine = {**PLUGIN_DEFAULTS}
    for key in PLUGIN_DEFAULTS:
        if key in params:
            mine[key] = params.pop(key)
    mine["units"] = bool(mine["units"])
    try:
        mine["joint_cover"] = float(mine["joint_cover"])
        mine["min_stone"] = float(mine["min_stone"])
    except (TypeError, ValueError):
        raise NotReady(t("joint_cover e min_stone sono numeri.", "joint_cover and min_stone are numbers.")) from None
    if not 0.0 < mine["joint_cover"] <= 1.0:
        raise NotReady(t("joint_cover va da 0 (escluso) a 1.", "joint_cover goes from 0 (excluded) to 1."))
    if not 0.0 < mine["min_stone"] < 10.0:
        raise NotReady(t("min_stone va da 0 (escluso) a 10 m.", "min_stone goes from 0 (excluded) to 10 m."))
    return params, mine


# ---------------------------------------------------------------- reading the drawing

def _unit_rings(chunk, eid: str) -> list[np.ndarray]:
    from shapely import wkt

    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    out = []
    for rec in frame_drawings(chunk, eid).get("units") or []:
        try:
            geom = wkt.loads(rec["wkt"])
        except Exception:  # noqa: BLE001 - a broken record is left out, as the program does
            continue
        for pg in getattr(geom, "geoms", [geom]):
            if pg.geom_type == "Polygon" and not pg.is_empty:
                out += [np.asarray(r.coords, float)[:, :2] for r in [pg.exterior, *pg.interiors]]
    return out


def read_lines(chunk, eid: str, source=None, units: bool = False) -> tuple[list[np.ndarray], str]:
    """The lines of the drawing and a text naming it: ``source`` None = the lines drawn on the face's elevation
    (without the outlines of its units unless ``units``), a path = a GeoPackage, GeoJSON or DXF in the face's
    ``(u, v)``."""
    need_program()
    try:
        if source is None or str(source) == "":
            lines = orient_lines.lines_from_elevation(chunk, eid)
            if not units:
                rings = _unit_rings(chunk, eid)
                lines = [ln for ln in lines if not any(r.shape == ln.shape and np.array_equal(r, ln) for r in rings)]
                if not lines:
                    raise NotReady(t("Sulla faccia non c'è nessuna linea disegnata (solo i contorni delle USM).",
                                     "No line is drawn on the face (only the outlines of the USM)."))
            return lines, "face"
        path = Path(source)
        return orient_lines.lines_from_file(path), f"file:{path.name}"
    except orient_lines.OrientLinesError as exc:
        raise _not_ready(exc) from None


# ---------------------------------------------------------------- beds, joints, courses, stones

def _polyline(geom) -> np.ndarray:
    xy = np.asarray(geom.coords, float)[:, :2]
    return xy


def _pair_lines(poly) -> tuple[np.ndarray, np.ndarray]:
    """The two lines of a pair's polygon (corners c0 c1 c2 c3: the lines c0-c1 and c3-c2)."""
    c = np.asarray(poly.exterior.coords, float)[:4, :2]
    return np.array([c[0], c[1]]), np.array([c[3], c[2]])


class _Bed:
    """A bed line as v(u): ``u`` increasing."""

    def __init__(self, xy: np.ndarray, element: str, role: str, pair_width: float | None = None):
        xy = xy[np.argsort(xy[:, 0], kind="stable")]
        keep = np.concatenate([[True], np.diff(xy[:, 0]) > 1e-9])
        self.xy = xy[keep]
        self.element, self.role, self.pair_width = element, role, pair_width
        self.u0, self.u1 = float(self.xy[0, 0]), float(self.xy[-1, 0])

    def v(self, u):
        return np.interp(u, self.xy[:, 0], self.xy[:, 1])

    def points(self, a: float, b: float) -> np.ndarray:
        """The line from u = a to u = b (its vertices between, the ends interpolated)."""
        inner = self.xy[(self.xy[:, 0] > a) & (self.xy[:, 0] < b)]
        return np.vstack([[a, float(self.v(a))], inner, [b, float(self.v(b))]])


def _subtract(intervals: list[tuple[float, float]], cut: tuple[float, float]) -> list[tuple[float, float]]:
    out = []
    for a, b in intervals:
        if cut[1] <= a or cut[0] >= b:
            out.append((a, b))
            continue
        if cut[0] > a:
            out.append((a, cut[0]))
        if cut[1] < b:
            out.append((cut[1], b))
    return out


def _course_pieces(beds: list[_Bed], min_stone: float) -> list[tuple[int, int, float, float]]:
    """``(lower bed, upper bed, u from, u to)``: the stretches where the upper bed is directly above the lower one."""
    out = []
    for i, lo in enumerate(beds):
        for j, hi in enumerate(beds):
            if i == j:
                continue
            a, b = max(lo.u0, hi.u0), min(lo.u1, hi.u1)
            if b - a < min_stone:
                continue
            us = np.linspace(a, b, 9)
            if not np.all(hi.v(us) > lo.v(us)):
                continue
            free = [(a, b)]
            for k, mid in enumerate(beds):
                if k in (i, j):
                    continue
                ka, kb = max(a, mid.u0), min(b, mid.u1)
                if kb - ka <= 1e-9:
                    continue
                km = 0.5 * (ka + kb)
                if lo.v(km) < mid.v(km) < hi.v(km):
                    free = _subtract(free, (ka, kb))
            out += [(i, j, fa, fb) for fa, fb in free if fb - fa >= min_stone]
    return out


def _cutter(joint, band, reach: float):
    """The part of the joint within the course's ``band`` (its longest piece), extended by ``reach`` at both ends;
    None when the joint does not reach the band."""
    from shapely.geometry import LineString

    part = joint.intersection(band)
    pieces = [g for g in getattr(part, "geoms", [part]) if g.geom_type == "LineString" and g.length > 0]
    if not pieces:
        return None
    xy = np.asarray(max(pieces, key=lambda g: g.length).coords, float)[:, :2]
    d0, d1 = xy[0] - xy[1], xy[-1] - xy[-2]
    n0, n1 = np.hypot(*d0), np.hypot(*d1)
    if n0 < 1e-12 or n1 < 1e-12:
        return None
    return LineString(np.vstack([xy[0] + d0 / n0 * reach, xy, xy[-1] + d1 / n1 * reach]))


def propose(chunk, eid: str, source=None, params=None) -> dict:
    """The drawing of the face ``eid`` (``source``: None = the lines drawn on its elevation, a path = a file in its
    ``(u, v)``) classified along the face's ``u`` axis, and what it means for the masonry: ``{"eid", "source",
    "classification", "beds", "joints", "courses", "stones", "skipped", "params", "already"}``.

    * ``beds``: ``[{"element", "role" (bed|pair_lower|pair_upper), "v", "u_from", "u_to"}]`` sorted by ``v``;
    * ``joints``: ``[{"element", "u", "v_from", "v_to", "used"}]`` (``used``: it splits at least one course);
    * ``courses``: ``[{"n", "v", "thickness", "pair", "u_from", "u_to", "stones"}]`` from the lowest; ``pair``: a pair
      along the axis (its thickness = the pair's width), else two beds (the median distance between them);
    * ``stones``: ``[{"id", "course", "polygon" (shapely, (u, v)), "u", "v", "width", "height"}]``;
    * ``skipped``: the elements left out, ``{"other": n, "joints": n}`` (other directions; joints that split no
      course).

    Nothing is written to the project."""
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import polygonize, unary_union

    need_program()
    if walls.elevation(chunk, eid) is None:
        raise NotReady(t(f"Nessun prospetto {eid} nel chunk.", f"No elevation {eid} in the chunk."))
    cparams, mine = split_params(params)
    lines, label = read_lines(chunk, eid, source, mine["units"])
    try:
        result = orient_lines.classify(lines, AXIS, cparams, frame=eid)
    except orient_lines.OrientLinesError as exc:
        raise _not_ready(exc) from None
    beds: list[_Bed] = []
    joints: list[tuple[str, object]] = []
    other = 0
    for e in result.elements:
        if e.group == "along":
            beds.append(_Bed(_polyline(e.geometry), e.id, "bed"))
        elif e.group == "pair_along":
            a, b = _pair_lines(e.geometry)
            lower, upper = (a, b) if a[:, 1].mean() <= b[:, 1].mean() else (b, a)
            beds.append(_Bed(lower, e.id, "pair_lower", e.width))
            beds.append(_Bed(upper, e.id, "pair_upper", e.width))
        elif e.group == "across":
            joints.append((e.id, e.geometry))
        elif e.group == "pair_across":
            for xy in _pair_lines(e.geometry):
                joints.append((e.id, LineString(xy)))
        else:
            other += 1
    min_stone, cover = mine["min_stone"], mine["joint_cover"]
    pieces = _course_pieces(beds, min_stone)
    used = [False] * len(joints)
    raw_courses: dict[tuple[int, int], dict] = {}
    raw_stones = []
    for i, j, a, b in pieces:
        lo, hi = beds[i], beds[j]
        low, up = lo.points(a, b), hi.points(a, b)
        us = np.linspace(a, b, 21)
        gap = hi.v(us) - lo.v(us)
        height = float(np.median(gap))
        if height < min_stone:
            continue
        poly = Polygon(np.vstack([low, up[::-1]])).buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        vmin, vmax = float(low[:, 1].min()), float(up[:, 1].max())
        band = box(a - 1.0, vmin - 0.5 * height, b + 1.0, vmax + 0.5 * height)
        cutters = []
        for k, (_jid, geom) in enumerate(joints):
            gu0, gv0, gu1, gv1 = geom.bounds
            if gu1 < a - min_stone or gu0 > b + min_stone or gv1 <= vmin or gv0 >= vmax:
                continue
            clip = geom.intersection(box(a - min_stone, vmin, b + min_stone, vmax))
            if clip.is_empty:
                continue
            iu0, iv0, iu1, iv1 = clip.bounds
            um = 0.5 * (iu0 + iu1)
            v_lo, v_hi = float(lo.v(um)), float(hi.v(um))
            if v_hi - v_lo <= 0 or min(iv1, v_hi) - max(iv0, v_lo) < cover * (v_hi - v_lo) - 1e-9:
                continue  # it does not cover enough of the course
            if um <= a + min_stone or um >= b - min_stone:
                used[k] = True  # the end of the course
                continue
            c = _cutter(geom, band, 2.0 * height)
            if c is not None:
                cutters.append(c)
                used[k] = True
        if cutters:
            cells = [g for g in polygonize(unary_union([poly.boundary, *cutters]))
                     if poly.contains(g.representative_point())]
        else:
            cells = [poly]
        key = (i, j)
        course = raw_courses.setdefault(key, {"pieces": [], "gaps": [], "pair": (
            lo.element == hi.element and lo.role == "pair_lower" and hi.role == "pair_upper"),
            "width": lo.pair_width})
        course["pieces"].append((a, b))
        course["gaps"].append(gap)
        for cell in cells:
            cu0, cv0, cu1, cv1 = cell.bounds
            if cu1 - cu0 < min_stone or cv1 - cv0 < min_stone or cell.area < 0.5 * min_stone * min_stone:
                continue
            raw_stones.append((key, cell))
    # the courses numbered from the lowest (the mean height of their middle line), the stones course by course
    order = sorted(raw_courses, key=lambda k: float(np.mean([
        0.5 * (beds[k[0]].v(0.5 * (a + b)) + beds[k[1]].v(0.5 * (a + b))) for a, b in raw_courses[k]["pieces"]])))
    order = [k for k in order if any(key == k for key, _c in raw_stones)]
    number = {k: n for n, k in enumerate(order, 1)}
    courses = []
    for k in order:
        c = raw_courses[k]
        gaps = np.concatenate(c["gaps"])
        thickness = float(c["width"]) if c["pair"] and c["width"] is not None else float(np.median(gaps))
        mids = [0.5 * (beds[k[0]].v(0.5 * (a + b)) + beds[k[1]].v(0.5 * (a + b))) for a, b in c["pieces"]]
        courses.append({"n": number[k], "v": float(np.mean(mids)), "thickness": thickness, "pair": bool(c["pair"]),
                        "u_from": float(min(a for a, _b in c["pieces"])),
                        "u_to": float(max(b for _a, b in c["pieces"])), "stones": 0})
    stones = []
    for key, cell in sorted(raw_stones, key=lambda kc: (number.get(kc[0], 0), kc[1].bounds[0])):
        if key not in number:
            continue
        n = number[key]
        courses[n - 1]["stones"] += 1
        u0, v0, u1, v1 = cell.bounds
        c = cell.centroid
        stones.append({"id": len(stones) + 1, "course": n, "polygon": cell, "u": float(c.x), "v": float(c.y),
                       "width": float(u1 - u0), "height": float(v1 - v0)})
    bed_rows = sorted(({"element": bd.element, "role": bd.role, "v": float(bd.v(0.5 * (bd.u0 + bd.u1))),
                        "u_from": bd.u0, "u_to": bd.u1} for bd in beds), key=lambda r: r["v"])
    joint_rows = []
    for (jid, geom), u in zip(joints, used):
        gu0, gv0, gu1, gv1 = geom.bounds
        joint_rows.append({"element": jid, "u": 0.5 * (gu0 + gu1), "v_from": gv0, "v_to": gv1, "used": u,
                           "geometry": geom})
    return {"eid": eid, "source": label, "source_path": None if source in (None, "") else str(source),
            "classification": result, "beds": bed_rows, "joints": joint_rows, "courses": courses,
            "stones": stones, "skipped": {"other": other, "joints": sum(1 for u in used if not u)},
            "params": {**result.params, **mine}, "already": [], "bed_lines": [bd.xy for bd in beds]}


def counts(proposal: dict) -> dict:
    """``{"beds", "joints", "courses", "stones", "pairs", "other", "unused"}`` of a proposal (``pairs``: the courses
    from a pair along the axis; ``unused``: the joints that split no course)."""
    return {"beds": len(proposal["beds"]), "joints": len(proposal["joints"]), "courses": len(proposal["courses"]),
            "stones": len(proposal["stones"]), "pairs": sum(1 for c in proposal["courses"] if c["pair"]),
            "other": proposal["skipped"]["other"], "unused": proposal["skipped"]["joints"]}


def summary_text(proposal: dict) -> str:
    """The proposal for the user: the counts, then a line per course."""
    c = counts(proposal)
    head = t(f"Letti {c['beds']}, giunti {c['joints']}, filari {c['courses']}, pietre {c['stones']}",
             f"Beds {c['beds']}, joints {c['joints']}, courses {c['courses']}, stones {c['stones']}")
    notes = []
    if c["other"]:
        notes.append(t(f"{c['other']} linee in altre direzioni lasciate fuori",
                       f"{c['other']} lines in other directions left out"))
    if c["unused"]:
        notes.append(t(f"{c['unused']} giunti che non dividono nessun filare",
                       f"{c['unused']} joints that split no course"))
    rows = [head + (" (" + "; ".join(notes) + ")" if notes else "") + "."]
    for co in proposal["courses"]:
        kind = t(" (coppia di linee)", " (pair of lines)") if co["pair"] else ""
        n = plural(co["stones"], ("pietra", "pietre"), ("stone", "stones"))
        rows.append(t(f"filare {co['n']}: quota media {num(co['v'], 3)}, spessore {num(co['thickness'] * 100, 1)} cm"
                      f", {n}{kind}",
                      f"course {co['n']}: mean height {num(co['v'], 3)}, thickness "
                      f"{num(co['thickness'] * 100, 1)} cm, {n}{kind}"))
    return "\n".join(rows)


# ---------------------------------------------------------------- the check plate

def plate(chunk, eid: str, proposal: dict, path, size=(2000, 1400)) -> Path:
    """The check plate (PNG or SVG by the suffix of ``path``): the stones proposed filled and numbered, the beds in
    blue, the joints that split a course in orange, the others dashed grey, each course named at its left. Returns
    the path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patches import Polygon as MplPolygon

    path = Path(path)
    if path.suffix.lower() not in (".png", ".svg"):
        raise NotReady(t("La tavola è un PNG o un SVG.", "The plate is a PNG or an SVG."))
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = (max(200, int(v)) for v in size)
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    stones = proposal["stones"]
    font = 6 if len(stones) > 150 else 8
    bounds = []
    for s in stones:
        xy = np.asarray(s["polygon"].exterior.coords)[:, :2]
        bounds.append(s["polygon"].bounds)
        ax.add_patch(MplPolygon(xy, closed=True, facecolor=COLOURS["stone"], alpha=0.45, edgecolor=COLOURS["edge"],
                                linewidth=0.8))
        if len(stones) <= 600:
            p = s["polygon"].representative_point()
            ax.annotate(str(s["id"]), (p.x, p.y), fontsize=font, ha="center", va="center", color="#202020")
    for xy in proposal.get("bed_lines") or []:
        ax.plot(xy[:, 0], xy[:, 1], color=COLOURS["bed"], linewidth=1.4)
        bounds.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
    for jr in proposal["joints"]:
        xy = np.asarray(jr["geometry"].coords)[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=COLOURS["joint" if jr["used"] else "unused"], linewidth=1.2,
                linestyle="-" if jr["used"] else "--")
        bounds.append(jr["geometry"].bounds)
    if bounds:
        b = np.array(bounds)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    else:
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    pad = 0.05 * max(x1 - x0, y1 - y0, 1e-3)
    for co in proposal["courses"]:
        ax.annotate(t(f"filare {co['n']}", f"course {co['n']}"), (co["u_from"] - pad * 0.2, co["v"]), fontsize=font,
                    ha="right", va="center", color=COLOURS["bed"])
    ax.set_xlim(x0 - 2.5 * pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    c = counts(proposal)
    entry = walls.elevation(chunk, eid) or {}
    ax.set_title(t(f"Tavola di controllo - {entry.get('name', eid)}: {c['courses']} filari, {c['stones']} pietre",
                   f"Check plate - {entry.get('name', eid)}: {c['courses']} courses, {c['stones']} stones"))
    ax.legend(handles=[Patch(facecolor=COLOURS["stone"], edgecolor=COLOURS["edge"], alpha=0.45,
                             label=t("pietra proposta", "stone proposed")),
                       Line2D([], [], color=COLOURS["bed"], label=t("letto", "bed")),
                       Line2D([], [], color=COLOURS["joint"], label=t("giunto", "joint")),
                       Line2D([], [], color=COLOURS["unused"], linestyle="--",
                              label=t("giunto che non divide un filare", "joint that splits no course"))],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    return path


# ---------------------------------------------------------------- into the project

def stored(chunk, eid: str) -> list[str]:
    """The stones the program keeps with the elevation ``eid`` (WKT in its ``(u, v)``)."""
    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    return list((frame_drawings(chunk, eid).get("stones") or {}).get(eid) or [])


def put(chunk, eid: str, wkts: list[str]) -> None:
    """The stones of the elevation ``eid`` set to ``wkts`` in the project (the other drawings untouched)."""
    drawings = chunk.meta.setdefault("drawings", {})
    frames = drawings.setdefault("frames", {})
    data = frames.setdefault(eid, {})
    stones = dict(data.get("stones") or {})
    if wkts:
        stones[eid] = list(wkts)
    else:
        stones.pop(eid, None)
    if stones:
        data["stones"] = stones
    else:
        data.pop("stones", None)
    if not data:
        frames.pop(eid, None)
    if not frames:
        drawings.pop("frames", None)
    if not drawings:
        chunk.meta.pop("drawings", None)


def merged(chunk, eid: str, proposal: dict, replace: bool = False) -> tuple[list[str], list[int]]:
    """The stones the elevation would have after :func:`accept` (WKT) and the ids of the proposed stones skipped as
    duplicates of those there (overlapping one by more than :data:`.stones.DUPLICATE_IOU`); nothing written."""
    from shapely import wkt
    from shapely.strtree import STRtree

    from .stones import DUPLICATE_IOU

    if proposal.get("eid") != eid:
        raise NotReady(t("La proposta è di un'altra faccia.", "The proposal is of another face."))
    old = [] if replace else stored(chunk, eid)
    polys = []
    for text in old:
        try:
            g = wkt.loads(text)
        except Exception:  # noqa: BLE001 - a broken outline stays as it is and matches nothing
            continue
        if not g.is_empty:
            polys.append(g)
    tree = STRtree(polys) if polys else None
    out, skipped = list(old), []
    for s in proposal["stones"]:
        p = s["polygon"]
        dup = False
        if tree is not None:
            for k in tree.query(p):
                q = polys[int(k)]
                inter = p.intersection(q).area
                if inter > 0 and inter / p.union(q).area > DUPLICATE_IOU:
                    dup = True
                    break
        if dup:
            skipped.append(s["id"])
        else:
            out.append(p.wkt)
    return out, skipped


def accept(chunk, eid: str, proposal: dict, replace: bool = False) -> int:
    """The proposed stones appended to those of the elevation ``eid`` (``replace``: in place of them), those that
    duplicate a stone already there skipped (their ids in ``proposal["already"]``). Returns how many were added."""
    wkts, skipped = merged(chunk, eid, proposal, replace)
    proposal["already"] = skipped
    added = len(proposal["stones"]) - len(skipped)
    if added or replace:
        put(chunk, eid, wkts)
    return added


def accepted_text(added: int, skipped: int) -> str:
    text = t(f"{added} pietre aggiunte alla faccia", f"{added} stones added to the face")
    if skipped:
        text += t(f"; {skipped} già sulla faccia, saltate", f"; {skipped} already on the face, skipped")
    return text + "."
