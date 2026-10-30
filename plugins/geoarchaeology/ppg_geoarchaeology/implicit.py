"""The implicit model of the deposits, on the side of the program: what the chunk knows (boreholes,
surfaces, drawn units, spot heights, files of contacts and orientations) becomes the job of the worker
(:mod:`.implicit_worker`, in a process of its own: :mod:`.implicit_runner`), and what comes back becomes
surfaces, solids and volumes.

This module runs in the program's process and imports neither LoopStructural nor pandas nor scikit-image.

The data
    * The *series* (:func:`series_of`): the contacts that follow the ground, by the modes of the kriging
      surfaces, in one series in depth space; every other contact a series of its own. ``overrides`` sets
      a unit by hand. With no surfaces yet the mode comes from :func:`surfaces.choose_mode` on the logs.
    * *Cuts and fills* (:func:`cuts_of`): kept as marks in ``chunk.meta["geoarchaeology"]["implicit"]["cuts"]``,
      ``{name: {"kind": "cut", "fills": [fills], "profile": "bowl"}}`` for a cut and ``{name: {"kind": "fill",
      "fills": cut, "profile": None}}`` for a fill. A cut has no material: a unit found in the logs is a fill.
      In a borehole through a fill the base of the deepest fill goes in the ``base`` of the cut and is *not*
      a contact of the layer under it, whose top the cut took away; the holes crossing a fill are inside the
      cut, the others that went as deep are outside. A fill moves up in the column to right under what the holes
      through it see above it: whatever else the order of the logs put over it was cut.
    * The plan of a cut: the US drawn for the cut, else for its fills (:func:`outline`).
    * More data: :func:`read_contacts` (``unit, x, y, z[, kind]``, ``kind`` "wall" for a wall of a cut),
      :func:`read_orientations` (``unit, x, y, z, dip, dip direction``), kept in the registry and copied
      under ``implicit/data/``; the spot heights of the program with a US that is a unit are contacts of it.

The frame
    The grid is that of the surfaces, coarser (:func:`model_frame`) up to :data:`MAX_COLUMNS` columns; the
    voxels are at most a tenth of the thinnest unit and of the shallowest cut, at most :data:`MAX_LEVELS`
    levels (:func:`voxel_step`).

The results (:func:`build`)
    Nothing is touched until the worker has finished: ``implicit/`` and ``model/`` are written aside and
    swapped at the end, so a failure or a cancel leaves them as they were. ``implicit/top_<k>_<unit>.tif``:
    float32, band 1 the elevation, band 2 the jackknife variance (m²), band 3 implicit minus kriging where a
    kriging surface exists. ``model/``: the same files and registry as :mod:`.model`, with ``"method":
    "implicit"``. A kriging :func:`model.build` afterwards replaces the model; the implicit surfaces stay.
    The numbers are an interpretation of the data, and the pane and the figures say so.
"""
from __future__ import annotations

import math
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import boreholes as bh
from . import kriging
from . import model as model_mod
from . import surfaces as sf
from .texts import num, plural, t

FOLDER = "implicit"
DATA = "data"  # the copies of the files of contacts and orientations
PREFIX = "implicit_"  # of the ids of its products
PROFILES = ("flat", "bowl", "v")
DEFAULT_PROFILE = "bowl"
FOLDS = 12
MAX_COLUMNS = 120_000  # columns of the grid when no cell is given
MAX_LEVELS = 200  # levels of voxels
MIN_OUTLINE_CELLS = 3.0  # the narrowest part of a plan, in cells
OUTLINE_BUMPS = 0.05  # share of a plan that may be narrower than the width told: the bumps of the line
VOXEL_SHARE = 0.1  # a voxel is at most this share of the thinnest unit and of the shallowest cut
TOLERANCE = bh.TOLERANCE
VOLUME_COLUMNS = (*model_mod.VOLUME_COLUMNS, "method", "kriging_m3", "range_min", "range_max")
WORKER: tuple | None = None  # (python, script) of another worker: the tests
WALL_WORDS = {"wall", "walls", "parete", "pareti", "muro", "side"}
DIP_NAMES = {"dip", "inclinazione", "angolo", "pendenza", "dip_angle"}
DIRECTION_NAMES = {"dip_direction", "dip_dir", "dipdir", "dip_azimuth", "immersione", "direzione_immersione", "azimut",
                   "azimuth"}
INTERPRETATION = ("Questo modello è un'interpretazione dei dati, non una misura: lontano dai sondaggi dice quello che "
                  "i dati fanno supporre.",
                  "This model is an interpretation of the data, not a measure: away from the boreholes it says what the "
                  "data suggest.")
# every warning of the engine and of this module; test_implicit_plugin checks that none is missing
WARNINGS = {
    "edge_estimated": ("{unit}: nessuno ha disegnato il contorno, stimato dai sondaggi (fino a {limit} m da quelli "
                       "dentro): il volume può essere sovrastimato o sottostimato. Disegna la US del taglio.",
                       "{unit}: nobody drew the outline, estimated from the boreholes (up to {limit} m from those "
                       "inside): the volume may be overstated or understated. Draw the US of the cut."),
    "floor_unmeasured": ("{unit}: tutti i sondaggi sono vicino al bordo, la profondità del fondo non è stata misurata: "
                         "il volume è un'ipotesi, qualunque sia l'intervallo indicato.",
                         "{unit}: all the boreholes are near the edge, the depth of the floor was not measured: the "
                         "volume is a guess, whatever the range says."),
    "base_near_edge": ("{unit}: {rows} misure del fondo vicino al bordo non vengono raggiunte, per non scavare troppo "
                       "al centro.",
                       "{unit}: {rows} measures of the floor next to the edge are not reached, not to dig too much in "
                       "the middle."),
    "base_outside_outline": ("{unit}: una misura del fondo ({x}, {y}) è {distance} m fuori dal contorno: vale come "
                             "se fosse sul bordo.",
                             "{unit}: a measure of the floor ({x}, {y}) is {distance} m outside the outline: it counts "
                             "as if on the edge."),
    "cut_reaches_zmin": ("{unit}: il fondo arriva sotto la base del modello, ed è fermato lì.",
                         "{unit}: the floor goes below the base of the model and is stopped there."),
    "cut_without_data": ("{unit}: nessun dato per collocare il taglio, che è stato lasciato fuori.",
                         "{unit}: no data to place the cut, which was left out."),
    "fill_without_contact": ("{unit}: riempimento di {cut} senza tetto misurato: non si sa fin dove arriva.",
                             "{unit}: fill of {cut} with no measured top: how far it reaches is not known."),
    "series_without_data": ("Serie {series}: senza dati, lasciata fuori.", "Series {series}: no data, left out."),
    "unit_without_contact": ("{unit}: nessun contatto, l'unità non è nel modello.",
                             "{unit}: no contact, the unit is not in the model."),
    "contact_leaves_box": ("{unit}: il contatto esce dall'intervallo di quote del modello, dove manca il suo tetto.",
                           "{unit}: the contact leaves the range of heights of the model, where its top is missing."),
    "contact_shares_no_hole": ("{unit}: nessun sondaggio ha questo contatto insieme agli altri: la sua distanza da "
                               "loro è stimata dai piani dei contatti.",
                               "{unit}: no borehole has this contact with the others: its distance from them is "
                               "estimated from the planes of the contacts."),
    "few_holes": ("Serie {series}: i sondaggi non fanno un piano (pochi, o in fila): di traverso alla fila il contatto "
                  "è preso il meno inclinato possibile.",
                  "Series {series}: the boreholes do not make a plane (few, or in a line): across the line the contact "
                  "is taken as level as possible."),
    "normal_turned_up": ("{unit}: una giacitura era rivolta in basso ed è stata girata.",
                         "{unit}: an orientation pointed down and was turned."),
    "column_contradicted": ("{holes} sondaggi hanno due unità nell'ordine opposto alla colonna.",
                            "{holes} boreholes have two units in the order opposite to the column."),
    "outline_narrow": ("{unit}: il contorno è largo solo {width} m nel punto più stretto ({cells} celle da {cell} m): "
                       "per disegnarlo serve una cella di {use} m al massimo.",
                       "{unit}: the outline is only {width} m wide at its narrowest ({cells} cells of {cell} m): a "
                       "cell of {use} m at most is needed to draw it."),
    "outline_empty": ("{unit}: il contorno disegnato è più piccolo di una cella ed è stato ignorato: serve una cella "
                      "minore.",
                      "{unit}: the drawn outline is smaller than a cell and was ignored: a smaller cell is needed."),
    "voxels_clamped": ("Voxel di {used} m invece di {wanted} m: più di {levels} livelli non si possono fare; le "
                       "unità sottili e i tagli poco profondi sono resi male.",
                       "Voxels of {used} m instead of {wanted} m: no more than {levels} levels can be made; thin "
                       "units and shallow cuts are poorly drawn."),
    "base_at_hole_end": ("{unit}: i sondaggi {holes} finiscono dentro il riempimento: il fondo è più in basso e il volume "
                         "è sottostimato.",
                         "{unit}: the boreholes {holes} end inside the fill: the floor is deeper and the volume is "
                         "understated."),
    "cut_without_fill": ("{unit}: un taglio senza riempimento, lasciato fuori.",
                         "{unit}: a cut without a fill, left out."),
    "holes_without_elevation": ("I sondaggi {holes} non hanno una quota: lasciati fuori.",
                                "The boreholes {holes} have no elevation: left out."),
    "unknown_units": ("Unità sconosciute nei dati aggiunti, lasciate fuori: {units}.",
                      "Unknown units in the added data, left out: {units}."),
    "wall_without_cut": ("{unit}: una riga «wall» di un'unità che non è un taglio né un riempimento: usata come "
                         "contatto.",
                         "{unit}: a 'wall' row of a unit that is neither a cut nor a fill: used as a contact."),
    "orientation_ignored": ("{unit}: giacitura di un riempimento: non serve al modello, lasciata fuori.",
                            "{unit}: orientation of a fill: of no use to the model, left out."),
}
PROFILE_NAMES = {"flat": ("fondo piatto", "flat floor"), "bowl": ("a conca", "bowl"), "v": ("a V", "V-shaped")}


def registry(chunk) -> dict:
    return (chunk.meta.get("geoarchaeology") or {}).get("implicit") or {}


def _state(chunk) -> dict:
    return bh.meta(chunk).setdefault("implicit", {})


def folder(chunk) -> Path:
    return bh.folder(chunk) / FOLDER


def is_product(pid: str) -> bool:
    return str(pid).startswith(PREFIX)


def active_surfaces(chunk) -> list[dict]:
    """The surfaces of the implicit model when it is the model of the chunk, from the top of the column
    down, else nothing: the sections follow the model."""
    if model_mod.method(chunk) != "implicit":
        return []
    return list(registry(chunk).get("surfaces", []))


def _stale(chunk) -> None:
    state = _state(chunk)
    if state.get("made"):
        state["stale"] = True


# ------------------------------------------------------------------ cuts and fills
def _marks(chunk) -> dict:
    return registry(chunk).get("cuts") or {}


def cuts_of(chunk) -> dict:
    """The marks of the chunk, whole: every cut with its fills and a profile, also the cut of a fill
    marked before it. See the module."""
    out: dict = {}
    marks = _marks(chunk)
    for name, mark in marks.items():
        if mark.get("kind") == "cut":
            out[name] = {"kind": "cut", "fills": [], "profile": mark.get("profile") or DEFAULT_PROFILE}
    for name, mark in marks.items():
        if mark.get("kind") == "fill":
            cut = str(mark.get("fills"))
            entry = out.setdefault(cut, {"kind": "cut", "fills": [], "profile": DEFAULT_PROFILE})
            entry["fills"].append(name)
            out[name] = {"kind": "fill", "fills": cut, "profile": None}
    return out


def _store(chunk, marks: dict) -> None:
    _state(chunk)["cuts"] = marks
    _stale(chunk)


def _profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(t("Forma del fondo sconosciuta: {0} ({1})", "Unknown shape of the floor: {0} ({1})")
                         .format(profile, ", ".join(PROFILES)))
    return profile


def mark_cut(chunk, unit: str, profile: str = DEFAULT_PROFILE) -> str:
    """``unit`` is a cut with the floor ``profile``; returns its name. A cut has no material: a unit found
    in the logs is what fills it, and the cut takes the name ``Taglio <unit>``."""
    _profile(profile)
    codes = bh.order(chunk)
    name = bh.resolve_unit(unit, codes) or str(unit).strip()
    if not name:
        raise ValueError(t("Un taglio vuole un nome", "A cut needs a name"))
    marks = cuts_of(chunk)
    if marks.get(name, {}).get("kind") == "fill":
        raise ValueError(t("{0} è già un riempimento di {1}", "{0} is already a fill of {1}")
                         .format(name, marks[name]["fills"]))
    if name in codes and not marks.get(name, {}).get("fills"):
        cut = f"Taglio {name}"
        marks[cut] = {"kind": "cut", "fills": [name], "profile": profile}
        marks[name] = {"kind": "fill", "fills": cut, "profile": None}
        _store(chunk, marks)
        return cut
    entry = marks.setdefault(name, {"kind": "cut", "fills": [], "profile": profile})
    entry["profile"] = profile
    _store(chunk, marks)
    return name


def mark_fill(chunk, fill: str, cut: str) -> str:
    """``fill`` (a unit of the logs) fills ``cut``; returns the name of the unit."""
    codes = bh.order(chunk)
    name = bh.resolve_unit(fill, codes)
    if name is None:
        raise ValueError(t("Unità sconosciuta: {0} (le unità: {1})", "Unknown unit: {0} (the units: {1})")
                         .format(fill, ", ".join(codes)))
    marks = cuts_of(chunk)
    target = bh.resolve_unit(cut, codes) or str(cut).strip()
    if not target or target == name:
        raise ValueError(t("{0} non può riempire {1}", "{0} cannot fill {1}").format(name, target or "?"))
    if marks.get(target, {}).get("kind") == "fill":
        raise ValueError(t("{0} è un riempimento, non un taglio", "{0} is a fill, not a cut").format(target))
    if marks.get(name, {}).get("kind") == "cut":
        raise ValueError(t("{0} è un taglio", "{0} is a cut").format(name))
    old = marks.get(name)
    if old and old["fills"] in marks:
        marks[old["fills"]]["fills"].remove(name)
    marks.setdefault(target, {"kind": "cut", "fills": [], "profile": DEFAULT_PROFILE})["fills"].append(name)
    marks[name] = {"kind": "fill", "fills": target, "profile": None}
    _store(chunk, marks)
    return name


def set_profile(chunk, cut: str, profile: str) -> None:
    marks = cuts_of(chunk)
    if marks.get(cut, {}).get("kind") != "cut":
        raise ValueError(t("{0} non è un taglio", "{0} is not a cut").format(cut))
    marks[cut]["profile"] = _profile(profile)
    _store(chunk, marks)


def unmark(chunk, name: str) -> None:
    """Forget a mark: a fill is a layer again, a cut goes with the marks of its fills."""
    marks = cuts_of(chunk)
    mark = marks.pop(name, None)
    if mark is None:
        return
    if mark["kind"] == "cut":
        for fill in mark["fills"]:
            marks.pop(fill, None)
    elif mark["fills"] in marks:
        marks[mark["fills"]]["fills"].remove(name)
    _store(chunk, marks)


# ------------------------------------------------------------------ what the logs say
@dataclass
class Data:
    """What the job is made of: ``contacts`` rows of the job; for every cut its ``base`` rows, the
    positions ``inside`` and ``outside``, the units seen ``above`` its fills; per unit the ``tables`` of
    its contact and the thicknesses in the logs."""

    contacts: list = field(default_factory=list)
    orientations: list = field(default_factory=list)
    base: dict = field(default_factory=dict)
    inside: dict = field(default_factory=dict)
    outside: dict = field(default_factory=dict)
    above: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    thick: dict = field(default_factory=dict)
    depths: dict = field(default_factory=dict)  # cut -> the depths of its base at the holes
    notes: list = field(default_factory=list)


def _stretches(b) -> list:
    """The intervals of a log by unit, from the top: ``[unit, top, base]``, a repeated unit joined."""
    out: list = []
    for iv in sorted(b.intervals, key=lambda i: i.top):
        if out and out[-1][0] == iv.unit:
            out[-1][2] = max(out[-1][2], iv.base)
        else:
            out.append([iv.unit, iv.top, iv.base])
    return out


def _table(data: Data, unit: str, x: float, y: float, z: float, depth: float) -> None:
    row = data.tables.setdefault(unit, {"xy": [], "z": [], "depth": []})
    row["xy"].append((x, y))
    row["z"].append(z)
    row["depth"].append(depth)


def gather(holes, marks: dict) -> Data:
    """The logs as data of the job (see the module); a hole ending inside a fill has not found the floor:
    its bottom is used only when no other hole of the cut did (:data:`WARNINGS` ``base_at_hole_end``)."""
    data = Data()
    fill_of = {u: m["fills"] for u, m in marks.items() if m["kind"] == "fill"}
    cuts = [u for u, m in marks.items() if m["kind"] == "cut" and m["fills"]]
    ended: dict = {cut: [] for cut in cuts}
    tops: dict = {cut: [] for cut in cuts}
    without = []
    for cut in cuts:
        data.base[cut], data.inside[cut], data.outside[cut], data.above[cut], data.depths[cut] = [], [], [], set(), []
    for b in holes:
        if not b.intervals:
            continue
        if b.z is None:
            without.append(b.id)
            continue
        stretches = _stretches(b)
        truncated = set()
        for cut in cuts:
            own = [i for i, s in enumerate(stretches) if fill_of.get(s[0]) == cut or s[0] == cut]
            if not own:
                continue
            data.inside[cut].append([b.x, b.y])
            data.above[cut].update(s[0] for s in stretches[:own[0]])
            tops[cut].append(stretches[own[0]][1])
            depth = stretches[own[-1]][2]
            row = [b.x, b.y, b.z - depth]
            if own[-1] == len(stretches) - 1 and depth >= b.bottom - TOLERANCE:
                ended[cut].append((b.id, row, depth))
            else:
                data.base[cut].append(row)
                data.depths[cut].append(depth)
                truncated.add(own[-1] + 1)  # its top is the floor of the cut, not the top of that layer
        last = len(stretches) - 1
        for i, (unit, top, base) in enumerate(stretches):
            if i < last or base < b.bottom - TOLERANCE:
                data.thick.setdefault(unit, []).append(base - top)
            if marks.get(unit, {}).get("kind") == "cut" or i in truncated:
                continue
            d = top
            if i > 0 and stretches[i - 1][2] < d - 1e-6:  # a gap in the log: its middle, as in boreholes.contact_data
                d = 0.5 * (stretches[i - 1][2] + d)
            data.contacts.append([unit, float(b.x), float(b.y), float(b.z - d), 1.0, b.id])
            _table(data, unit, float(b.x), float(b.y), float(b.z - d), float(d))
    for cut in cuts:
        reached = min(tops[cut], default=None)
        if reached is not None:
            inside = {tuple(p) for p in data.inside[cut]}
            data.outside[cut] = [[b.x, b.y] for b in holes if b.intervals and (b.x, b.y) not in inside
                                 and b.bottom >= reached]
        if ended[cut] and not data.base[cut]:
            data.base[cut] = [row for _id, row, _d in ended[cut]]
            data.depths[cut] = [d for _id, _row, d in ended[cut]]
        if ended[cut]:
            data.notes.append({"code": "base_at_hole_end", "unit": cut, "holes": ", ".join(i for i, _r, _d in ended[cut])})
    if without:
        data.notes.append({"code": "holes_without_elevation", "holes": ", ".join(without)})
    return data


def _spot_heights(chunk) -> list:
    """The spot heights of the program with a US: ``(kind and US, US, x, y, elevation)``. They live in
    ``chunk.meta["drawings"]["quote"]`` with the fields of ``pyarchinit_quote``; the coordinates are of
    the chunk."""
    out = []
    for q in (chunk.meta.get("drawings") or {}).get("quote") or []:
        us, z = q.get("us_q"), q.get("quota_q")
        if "x" not in q or "y" not in q or us in (None, "") or z in (None, ""):
            continue
        try:
            out.append((f"{q.get('unita_tipo_q') or 'US'} {us}", str(us), float(q["x"]), float(q["y"]), float(z)))
        except (TypeError, ValueError):
            continue
    return out


def add_measures(chunk, data: Data, codes: list, marks: dict) -> None:
    """The contacts and the orientations of the files and the spot heights of the chunk into ``data``. A
    contact of a cut is a point of its floor, a row ``wall`` of a cut or of its fill a point of a wall."""
    known = list(codes) + [u for u in marks if u not in codes]
    unknown: list = []

    def resolve(name):
        unit = bh.resolve_unit(name, known)
        if unit is None and str(name) not in unknown:
            unknown.append(str(name))
        return unit

    def place(unit, x, y, z, wall=False):
        mark = marks.get(unit, {})
        cut = unit if mark.get("kind") == "cut" else mark.get("fills") if mark.get("kind") == "fill" else None
        if cut is not None and (wall or mark.get("kind") == "cut"):
            if cut in data.base:  # a cut without a fill is not in the model, and said so
                data.base[cut].append([x, y, z, "wall"] if wall else [x, y, z])
            return
        if wall:
            data.notes.append({"code": "wall_without_cut", "unit": unit})
        data.contacts.append([unit, x, y, z, 1.0, None])
        _table(data, unit, x, y, z, float("nan"))

    stored = registry(chunk).get("data") or {}
    for r in (stored.get("contacts") or {}).get("rows", []):
        unit = resolve(r["unit"])
        if unit is not None:
            place(unit, float(r["x"]), float(r["y"]), float(r["z"]), r.get("kind") == "wall")
    for drawn, us, x, y, z in _spot_heights(chunk):
        unit = bh.resolve_unit(drawn, known) or resolve(us)
        if unit is not None:
            place(unit, x, y, z)
    for r in (stored.get("orientations") or {}).get("rows", []):
        unit = resolve(r["unit"])
        if unit is None:
            continue
        if marks.get(unit, {}).get("kind") == "fill":
            data.notes.append({"code": "orientation_ignored", "unit": unit})
            continue
        x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        n = dip_normal(float(r["dip"]), float(r["direction"]))
        if marks.get(unit, {}).get("kind") == "cut" and data.inside.get(unit):
            into = np.mean(np.asarray(data.inside[unit], float), axis=0) - (x, y)
            if float(n[:2] @ into) < 0:  # the normal of a wall points into the cut
                n = -n
        data.orientations.append([unit, x, y, z, *(float(v) for v in n)])
    if unknown:
        data.notes.append({"code": "unknown_units", "units": ", ".join(unknown)})


def dip_normal(dip: float, direction: float) -> np.ndarray:
    """The upward normal of a plane dipping ``dip`` degrees towards ``direction`` (degrees clockwise
    from north): east, north, up."""
    d, a = math.radians(dip), math.radians(direction)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def column_of(codes: list, marks: dict, data: Data) -> list:
    """The column top-down, as the job wants it: the order of the units without the cuts, the fills of every
    cut right under what the holes through them see above them, the cut right under its fills. The order of
    the logs may put over a fill a layer that the holes through it do not have: that layer was cut."""
    fill_of = {u: m["fills"] for u, m in marks.items() if m["kind"] == "fill"}
    cuts = [u for u, m in marks.items() if m["kind"] == "cut" and any(f in codes for f in m["fills"])]
    column = [u for u in codes if u not in cuts]
    for cut in sorted(cuts, key=lambda c: max((codes.index(f) for f in marks[c]["fills"] if f in codes), default=0)):
        own = [u for u in column if fill_of.get(u) == cut]
        rest = [u for u in column if fill_of.get(u) != cut]
        above = [rest.index(u) for u in data.above.get(cut, ()) if u in rest]
        at = max(above) + 1 if above else 0
        column = rest[:at] + own + [cut] + rest[at:]
    return [{"unit": u, "kind": "cut" if u in cuts else "fill" if u in fill_of else "layer"} for u in column]


def series_of(chunk, overrides: dict | None = None, data: Data | None = None) -> list[dict]:
    """The series of the job, oldest first: the contacts by depth in one, each contact by elevation its
    own (see the module). The top of the logs joins the series by depth (it is its zero) and is left out
    otherwise: the highest layer takes all that is left under the ground whatever its contact says."""
    holes = bh.load(chunk)
    marks = cuts_of(chunk)
    if data is None:
        data = gather(holes, marks)
        add_measures(chunk, data, bh.order(chunk), marks)
    column = column_of(bh.order(chunk), marks, data)
    layers = [e["unit"] for e in column if e["kind"] == "layer"]
    given = {bh.resolve_unit(k, layers) or str(k): sf.mode_word(v) for k, v in (overrides or {}).items()}
    known = sf.registry(chunk).get("modes") or {}
    ground = bh.dem_path(chunk) is not None or any(b.z is not None for b in holes)
    depth, own = [], []
    for unit in layers:
        table = data.tables.get(unit)
        if table is None:
            continue
        mode = given.get(unit) or known.get(unit)
        if mode is None:
            cd = SimpleNamespace(xy=np.asarray(table["xy"], float).reshape(-1, 2), z=np.asarray(table["z"], float),
                                 depth=np.asarray(table["depth"], float))
            at_surface = bool(np.all(np.abs(cd.depth[np.isfinite(cd.depth)]) < 1e-3)) and np.isfinite(cd.depth).any()
            mode = "ground" if at_surface else sf.choose_mode(cd, ground)[0]
        if mode in ("depth", "ground"):
            depth.append((unit, mode))
        elif unit != layers[0]:
            own.append(unit)
    named = [u for u, mode in depth if mode == "depth" or u != layers[0]]
    series = [{"name": u, "space": "elevation", "units": [u]} for u in own]
    if named:
        series.append({"name": "ground", "space": "depth", "units": [u for u, _m in depth]})
    if not series:  # one layer: its contact is all there is
        seen = [u for u in layers if u in data.tables]
        if seen:
            series = [{"name": seen[0], "space": "elevation", "units": seen[:1]}]
    position = {u: i for i, u in enumerate(layers)}
    return sorted(series, key=lambda s: -max(position[u] for u in s["units"]))


# ------------------------------------------------------------------ the frame
def _narrowest(mask: np.ndarray) -> float:
    """The narrowest part of a plan, in cells: from the ridge of its distance transform, a ridge point
    held by the circle of a wider one saying nothing, and the narrowest circles left out as long as they
    hold less than :data:`OUTLINE_BUMPS` of the plan (the bumps of a hand-drawn line are ridges too)."""
    from scipy import ndimage

    padded = np.pad(mask, 1)
    reach = ndimage.distance_transform_edt(padded)
    rows, cols = np.nonzero(padded & (reach >= ndimage.maximum_filter(reach, size=3)))
    covered = np.zeros(padded.shape, dtype=bool)
    held = []  # the radius of every circle and the cells of the plan only it holds, widest first
    for i in np.argsort(-reach[rows, cols], kind="stable"):
        row, col = rows[i], cols[i]
        if covered[row, col]:
            continue
        radius = float(reach[row, col])
        n = int(radius)
        top, left = max(row - n, 0), max(col - n, 0)
        ys, xs = np.ogrid[top - row:min(row + n + 1, padded.shape[0]) - row,
                          left - col:min(col + n + 1, padded.shape[1]) - col]
        new = (ys ** 2 + xs ** 2 <= radius ** 2) & ~covered[top:top + ys.size, left:left + xs.size]
        covered[top:top + ys.size, left:left + xs.size] |= new
        held.append((radius, int((new & padded[top:top + ys.size, left:left + xs.size]).sum())))
    total = sum(n for _r, n in held)
    so_far = 0
    for radius, n in held:
        so_far += n
        if so_far >= (1.0 - OUTLINE_BUMPS) * total:
            return 2.0 * radius - 1.0
    return 2.0 * held[-1][0] - 1.0


def outline(chunk, grid: sf.Grid, cut: str, fills: list) -> tuple[np.ndarray | None, list]:
    """The plan of a cut on ``grid``, a boolean raster, from the US drawn for it or else for its fills
    (a cell belongs to it when its centre does); None when nobody drew them. With the warnings."""
    from rasterio.features import rasterize
    from shapely.ops import unary_union

    drawn = model_mod.drawn_units(chunk)
    polygons = model_mod.polygons_for(cut, drawn) if drawn else []
    if not polygons:
        polygons = [p for f in fills for p in model_mod.polygons_for(f, drawn)]
    if not polygons:
        return None, []
    shape = unary_union([p.geometry for p in polygons])
    mask = rasterize([(shape, 1)], out_shape=(grid.height, grid.width), transform=grid.transform, fill=0,
                     all_touched=False, dtype="uint8").astype(bool)
    if not mask.any():
        return None, [{"code": "outline_empty", "unit": cut}]
    cells = _narrowest(mask)
    if cells < MIN_OUTLINE_CELLS:
        width = cells * grid.cell
        return mask, [{"code": "outline_narrow", "unit": cut, "width": width, "cells": cells, "cell": grid.cell,
                       "use": width / MIN_OUTLINE_CELLS}]
    return mask, []


def model_frame(chunk, holes, cell: float | None) -> tuple[sf.Grid, int, bool]:
    """``(grid, factor, on the surfaces)``: the grid of the surfaces coarsened by ``factor`` (see
    :func:`model.model_grid`: ``cell``, or within :data:`MAX_COLUMNS`), a grid of the boreholes when
    there are no surfaces."""
    base = sf.grid_of(chunk)
    known = base is not None
    if base is None:
        xy = np.array([[b.x, b.y] for b in holes], float)
        dem = bh.dem_path(chunk)
        own = 0.0
        if dem is not None:
            import rasterio

            with rasterio.open(dem) as src:
                own = abs(src.transform.a)
        base = sf.make_grid(chunk, xy, sf.default_buffer(xy, own), None)
    if not cell:
        cell = base.cell * max(1, int(np.ceil(np.sqrt(base.width * base.height / MAX_COLUMNS))))
    grid, factor = model_mod.model_grid(base, cell)
    return grid, factor, known


def ground_on(chunk, grid: sf.Grid, holes) -> tuple[np.ndarray, np.ndarray | None]:
    """The ground on ``grid`` and the cells with a DEM (None without one): the DEM, filled with the surface
    of the collars where it has no cells, as :func:`surfaces.build` does."""
    dem = sf.dem_on_grid(chunk, grid)
    collars = [b for b in holes if b.z is not None]
    surface = None
    if collars:
        xy = np.array([[b.x, b.y] for b in collars], float)
        z = np.array([b.z for b in collars], float)
        params = sf.registry(chunk).get("params") or {}
        buffer = params.get("buffer") or sf.default_buffer(xy, grid.cell)
        mask = sf.hull_mask(grid, xy, float(buffer))
        if dem is None or not np.isfinite(dem[mask]).all():
            est, _v, _r = kriging.interpolate(xy, z, grid.centres(mask), variance=False, model="auto", nugget=None,
                                              trend="auto", min_points=5)
            surface = np.full((grid.height, grid.width), np.nan)
            surface[mask] = est
    if dem is None:
        if surface is None:
            raise RuntimeError(t("Serve un DEM o la quota dei sondaggi",
                                 "A DEM or the elevations of the boreholes are needed"))
        return surface.astype(np.float32), None
    has_dem = np.isfinite(dem)
    return (dem if surface is None else np.where(has_dem, dem, surface)).astype(np.float32), has_dem


def voxel_step(thinnest: float | None, shallowest: float | None, span: float, notes: list) -> float:
    """The height of a voxel: at most :data:`VOXEL_SHARE` of the thinnest unit (mean thickness, m) and of
    the shallowest cut, and no more than :data:`MAX_LEVELS` levels over ``span``, with a warning then.
    One level is kept for the air over the highest cell."""
    sizes = [v * VOXEL_SHARE for v in (thinnest, shallowest) if v and v > 0]
    fewest = span / (MAX_LEVELS - 1)
    wanted = min(sizes) if sizes else fewest
    if wanted < fewest:
        notes.append({"code": "voxels_clamped", "wanted": wanted, "used": fewest, "levels": MAX_LEVELS})
        return fewest
    return wanted


# ------------------------------------------------------------------ the job
@dataclass
class Plan:
    """What :func:`prepare` knows beyond the job: the grid and its factor from the surfaces' one."""

    grid: sf.Grid
    factor: int
    on_surfaces: bool
    has_dem: np.ndarray | None
    floor_given: bool
    notes: list
    spaces: dict


def prepare(chunk, tmp: Path, cell: float | None = None, folds: int | None = FOLDS, floor: float | None = None,
            overrides: dict | None = None, nelements: int | None = None) -> tuple[dict, Plan]:
    """The job of the chunk, with its arrays in the folder ``tmp``, and the :class:`Plan`."""
    holes = bh.load(chunk)
    if not holes:
        raise RuntimeError(t("Nessun sondaggio nel chunk: importali prima", "No borehole in the chunk: import them first"))
    codes = bh.order(chunk)
    marks = cuts_of(chunk)
    data = gather(holes, marks)
    add_measures(chunk, data, codes, marks)
    if not data.contacts:
        raise RuntimeError(t("Nessun contatto: i sondaggi hanno bisogno di una quota, o il chunk di un DEM",
                             "No contact: the boreholes need an elevation, or the chunk a DEM"))
    notes = list(data.notes)
    notes.extend({"code": "cut_without_fill", "unit": u} for u, m in marks.items() if m["kind"] == "cut" and not m["fills"])
    column = column_of(codes, marks, data)
    series = series_of(chunk, overrides, data)
    grid, factor, on_surfaces = model_frame(chunk, holes, cell)
    ground, has_dem = ground_on(chunk, grid, holes)
    if not np.isfinite(ground).any():
        raise RuntimeError(t("Nessuna cella con il terreno intorno ai sondaggi",
                             "No cell with the ground around the boreholes"))
    tops = sf.tops(chunk)
    last_top = model_mod.coarsen(tops[-1][1], factor) if tops and on_surfaces else None
    zmin = float(floor) if floor is not None else model_mod.default_floor(holes, last_top)
    heights = [r[3] for r in data.contacts] + [r[2] for rows in data.base.values() for r in rows]
    if floor is None:
        margin = max(0.1, 0.05 * (max(heights) - min(heights)))
        zmin = min(zmin, min(heights) - margin) if zmin is not None else min(heights) - margin
    top = float(np.nanmax(ground))
    thin = [float(np.mean(v)) for v in data.thick.values() if v and np.mean(v) > 0]
    shallow = [max(v) for v in data.depths.values() if v]
    dz = voxel_step(min(thin, default=None), min(shallow, default=None), top - zmin, notes)
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "dem.npy", ground, allow_pickle=False)
    cuts = {}
    for i, entry in enumerate(e for e in column if e["kind"] == "cut"):
        cut = entry["unit"]
        mask, said = outline(chunk, grid, cut, marks[cut]["fills"])
        notes.extend(said)
        path = None
        if mask is not None:
            path = str((tmp / f"outline_{i}.npy").absolute())
            np.save(path, mask, allow_pickle=False)
        cuts[cut] = {"base": data.base.get(cut, []), "outline": path, "profile": marks[cut]["profile"],
                     "inside": data.inside.get(cut, []), "outside": data.outside.get(cut, [])}
    job = {"protocol": 1, "output": str((tmp / "run").absolute()), "dem": str((tmp / "dem.npy").absolute()),
           "grid": {"x0": grid.x0, "y0": grid.y0, "cell": grid.cell, "width": grid.width, "height": grid.height},
           "zmin": float(zmin), "zmax": top + dz, "dz": float(dz), "series": series, "contacts": data.contacts,
           "orientations": data.orientations, "column": column,
           "fills": {u: m["fills"] for u, m in marks.items() if m["kind"] == "fill" and u in {e["unit"] for e in column}},
           "cuts": cuts, "options": {"vertical": None, "folds": int(folds) if folds else None}}
    if nelements:
        job["options"]["nelements"] = int(nelements)
    spaces = {u: s["space"] for s in series for u in s["units"]}
    return job, Plan(grid, factor, on_surfaces, has_dem, floor is not None, notes, spaces)


# ------------------------------------------------------------------ the results
def _swap(pairs: list) -> None:
    """Every ``(new, final)`` folder into place, or none: a failure half-way puts the old ones back."""
    done = []
    try:
        for new, final in pairs:
            old = final.with_name(final.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            had = final.exists()
            if had:
                final.rename(old)
            done.append((final, old, had))
            new.rename(final)
    except BaseException:
        for final, old, had in reversed(done):
            shutil.rmtree(final, ignore_errors=True)
            if had:
                old.rename(final)
        raise
    for _final, old, _had in done:
        shutil.rmtree(old, ignore_errors=True)


def _span(values: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    v = values[mask & np.isfinite(values)]
    return (float(v.min()), float(v.max())) if v.size else (None, None)


def _kriging_volumes(chunk) -> dict:
    """The volume of every unit by kriging: the isopachs, the model between the surfaces for the last
    unit (or what the last implicit model kept of it)."""
    out = {}
    previous = model_mod.registry(chunk)
    for r in previous.get("units", []):
        value = r.get("kriging_m3") if previous.get("method") == "implicit" else r.get("volume_m3")
        if value is not None and r.get("cells"):
            out[r["unit"]] = float(value)
    for e in sf.registry(chunk).get("thickness", []):
        out[e["unit"]] = float(e.get("volume_m3", 0.0))
    return out


def _surfaces(chunk, result: dict, run: Path, plan: Plan, out: Path) -> list[dict]:
    """``implicit/top_<k>_<unit>.tif`` in the folder ``out``, and their entries."""
    grid, factor = plan.grid, plan.factor
    shape = (grid.height, grid.width)
    colours = bh.colours(chunk)
    kriged = {e["unit"]: e for e in sf.registry(chunk).get("contacts", [])}
    entries = []
    for k, unit in enumerate(result["units"]):
        name = (result["files"].get("tops") or {}).get(unit)
        if name is None:
            continue
        top = np.load(run / name, allow_pickle=False).astype(np.float64)
        rel = (result["files"].get("variance") or {}).get(unit)
        variance = np.load(run / rel, allow_pickle=False).astype(np.float64) if rel else np.full(shape, np.nan)
        difference = np.full(shape, np.nan)
        e = kriged.get(unit)
        if e is not None and plan.on_surfaces:
            other = model_mod.coarsen(sf.read_band(Path(chunk.data_dir) / e["file"], 1)[0], factor)
            if other.shape == shape:
                difference = top - other
        stem = f"top_{k}_{sf.slug(unit)}"
        sf.write_tif(out / f"{stem}.tif", [top, variance, difference], grid,
                     [f"top of {unit} (m)", "jackknife variance (m2)", "implicit minus kriging (m)"], BIGTIFF="IF_SAFER")
        loo = (result.get("loo") or {}).get(unit) or {}
        entry = {"id": PREFIX + stem, "kind": "surface", "unit": unit, "k": k, "colour": colours.get(unit),
                 "name": t("Tetto di {0} (implicito)", "Top of {0} (implicit)").format(unit), "method": "implicit",
                 "file": str((folder(chunk) / f"{stem}.tif").relative_to(chunk.data_dir)), "stats": sf._stats(top),
                 "variance": sf._stats(variance), "difference": sf._stats(difference)}
        if unit in plan.spaces:
            entry["mode"] = plan.spaces[unit]
        if loo.get("rmse") is not None:
            entry["loocv_rmse"] = float(loo["rmse"])
        entries.append(entry)
    return entries


def _model(chunk, job: dict, result: dict, run: Path, plan: Plan, out: Path, explode: float | None) -> dict:
    """The files of ``model/`` in the folder ``out``, and its registry: the shape of :func:`model.build`
    with ``"method": "implicit"``; a cut, with no material, is a row with its volume range."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    grid = plan.grid
    out.mkdir(parents=True, exist_ok=True)
    area = grid.cell ** 2
    voxel = area * float(job["dz"])
    labels = np.load(run / result["files"]["labels"], allow_pickle=False)
    units = list(result["units"])
    index = {u: i for i, u in enumerate(units)}
    colours = bh.colours(chunk)
    descriptions = {u["code"]: u.get("description", "") for u in bh.units(chunk)}
    kriging_m3 = _kriging_volumes(chunk)
    tops = {u: np.load(run / n, allow_pickle=False) for u, n in (result["files"].get("tops") or {}).items()}
    cuts = (result.get("info") or {}).get("cuts") or {}
    final = model_mod.folder(chunk)
    rows, meshes = [], []
    for entry in job["column"]:
        unit = entry["unit"]
        if entry["kind"] == "cut":
            info = cuts.get(unit) or {}
            span = info.get("volume_range") or {}
            if span:
                rows.append({"unit": unit, "kind": "cut", "description": descriptions.get(unit, ""), "method": "implicit",
                             "cells": int(round(float(info.get("plan_area", 0.0)) / area)),
                             "area_m2": float(info.get("plan_area", 0.0)), "clipped": False, "to_floor": False,
                             "volume_m3": float(span.get(info.get("profile"), max(span.values()))),
                             "range_min": float(min(span.values())), "range_max": float(max(span.values())),
                             "range": {k: float(v) for k, v in span.items()}, "profile": info.get("profile"),
                             "edge": info.get("edge")})
            continue
        row = {"unit": unit, "kind": entry["kind"], "description": descriptions.get(unit, ""), "method": "implicit",
               "kriging_m3": kriging_m3.get(unit), "clipped": False, "clipped_to": [], "cells": 0, "volume_m3": 0.0,
               "area_m2": 0.0, "to_floor": unit == units[-1]}
        rows.append(row)
        solid = (result["files"].get("solids") or {}).get(unit)
        if unit not in index or solid is None:
            continue
        counts = (labels == index[unit]).sum(axis=2)
        mask = counts > 0
        if not mask.any():
            continue
        verts = np.load(run / solid[0], allow_pickle=False)
        faces = np.load(run / solid[1], allow_pickle=False).astype(np.int64)
        rgb = np.array([int(colours.get(unit, "#a0a0a0")[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        name = sf.slug(unit)
        write_mesh_ply(out / f"{name}.ply", verts, faces, np.tile(rgb, (len(verts), 1)), double=True)
        on = mask if plan.has_dem is None else mask & plan.has_dem
        thick = counts[on] * float(job["dz"]) if on.any() else np.zeros(1)
        k = units.index(unit)
        below = tops.get(units[k + 1]) if k + 1 < len(units) else None
        top_lo, top_hi = _span(tops[unit], mask) if unit in tops else (None, None)
        base_lo, base_hi = _span(below, mask) if below is not None else (float(job["zmin"]), float(job["zmin"]))
        row.update(cells=int(mask.sum()), area_m2=float(on.sum() * area), volume_m3=float(counts[on].sum() * voxel),
                   mesh_volume_m3=model_mod.signed_volume(verts, faces), mean_thickness_m=float(thick.mean()),
                   max_thickness_m=float(thick.max()), top_min=top_lo, top_max=top_hi, base_min=base_lo, base_max=base_hi,
                   faces=int(len(faces)), ply=str((final / f"{name}.ply").relative_to(chunk.data_dir)))
        off = mask & ~on
        if off.any():
            row["outside_dem"] = {"cells": int(off.sum()), "area_m2": float(off.sum() * area),
                                  "volume_m3": float(counts[off].sum() * voxel)}
        meshes.append((unit, verts, faces, rgb))
    files = {}
    if meshes:
        model_mod.write_obj(out / "unita.obj", meshes)
        files["obj"] = str((final / "unita.obj").relative_to(chunk.data_dir))
    model_mod.write_volumes(out / "volumi.csv", rows, VOLUME_COLUMNS)
    files["csv"] = str((final / "volumi.csv").relative_to(chunk.data_dir))
    if model_mod.write_display(chunk, meshes, explode, out) is not None:
        files["display"] = str((final / model_mod.DISPLAY).relative_to(chunk.data_dir))
    return {"made": time.strftime("%Y-%m-%d %H:%M"), "cell": grid.cell, "factor": plan.factor, "floor": float(job["zmin"]),
            "floor_given": plan.floor_given, "clip": False, "units": rows, "files": files, "method": "implicit",
            "explode": float(explode or 0.0)}


def _loo_rows(chunk, result: dict) -> list[dict]:
    """The leave-one-out of the contacts beside that of the kriging."""
    kriged = {e["unit"]: e for e in sf.registry(chunk).get("contacts", [])}
    out = []
    for unit, loo in (result.get("loo") or {}).items():
        e = kriged.get(unit) or {}
        out.append({"unit": unit, "rmse": loo.get("rmse"), "mae": loo.get("mae"), "n": loo.get("n"), "nan": loo.get("nan"),
                    "kriging_rmse": e.get("loocv_rmse"), "kriging_points": e.get("points")})
    return out


def build(chunk, cell: float | None = None, folds: int | None = FOLDS, floor: float | None = None,
          explode: float | None = None, overrides: dict | None = None, nelements: int | None = None, progress=None,
          python: Path | None = None, script: Path | None = None) -> dict:
    """The implicit model of the chunk: the job in a temporary folder inside its data, the worker, then
    surfaces, solids, volumes and registries, all at the end. ``cell``: of the grid; ``folds``: groups of
    the leave-one-out, none without; ``floor``: the base of the model; ``explode``: as in
    :func:`model.build`; ``overrides``: ``{unit: "depth"|"elevation"}``. Returns the registry
    ``chunk.meta["geoarchaeology"]["implicit"]``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import implicit_runner

    progress = as_progress(progress)
    if python is None and script is None and WORKER is not None:
        python, script = WORKER
    tmp = bh.folder(chunk) / f"implicit_job_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        progress(0.01, t("Preparazione dei dati", "Preparing the data"))
        job, plan = prepare(chunk, tmp, cell, folds, floor, overrides, nelements)
        result = implicit_runner.run(job, tmp, progress.sub(0.02, 0.86), python=python, script=script)
        progress(0.9, t("Superfici e solidi", "Surfaces and solids"))
        run = Path(result["output"])
        new_surfaces, new_model = tmp / "implicit.new", tmp / "model.new"
        new_surfaces.mkdir()
        entries = _surfaces(chunk, result, run, plan, new_surfaces)
        model_reg = _model(chunk, job, result, run, plan, new_model, explode)
        kept = folder(chunk) / DATA
        if kept.exists():
            shutil.copytree(kept, new_surfaces / DATA)
        _swap([(new_surfaces, folder(chunk)), (new_model, model_mod.folder(chunk))])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    info = result.get("info") or {}
    state = _state(chunk)
    state.update(made=time.strftime("%Y-%m-%d %H:%M"), stale=False, grid=plan.grid.to_dict(), series=job["series"],
                 column=job["column"], surfaces=entries, cut_results=info.get("cuts") or {},
                 warnings=list(plan.notes) + list(info.get("warnings") or []), loo=_loo_rows(chunk, result),
                 versions=result.get("versions") or {}, vertical=info.get("vertical"), dz=job["dz"], zmin=job["zmin"],
                 zmax=job["zmax"], folds=job["options"]["folds"], seconds=result.get("seconds"))
    bh.meta(chunk)["model"] = model_reg
    progress(1.0, t("Modello implicito pronto", "Implicit model ready"))
    return state


def remove(chunk) -> None:
    """Forget the results of the implicit model (and the solids, when they are the model of the chunk).
    Cuts, fills and the data added stay: they are data of the chunk, not results."""
    out = folder(chunk)
    if out.exists():
        for p in out.iterdir():
            if p.is_file():
                p.unlink()
    if model_mod.method(chunk) == "implicit":
        model_mod.remove(chunk)
    state = _state(chunk)
    for key in list(state):
        if key not in ("cuts", "data"):
            state.pop(key)


# ------------------------------------------------------------------ contacts and orientations from files
def _rows(path, extra: dict | None, wanted: tuple) -> tuple[list[dict], list[str]]:
    notes: list = []
    rows, names = bh.read_table(path, notes, extra)
    missing = [f for f in wanted if f not in names]
    if missing:
        raise ValueError(t("{0}: mancano le colonne {1}", "{0}: the columns {1} are missing").format(
            Path(path).name, ", ".join(missing)))
    good = []
    for r in rows:
        if not str(r.get("unit") or "").strip() or any(r.get(f) is None for f in wanted if f != "unit"):
            notes.append(t("riga {0}: incompleta, lasciata fuori", "line {0}: incomplete, left out").format(r["_line"]))
        else:
            good.append(r)
    return good, notes


def read_contacts(path) -> tuple[list[dict], list[str]]:
    """The rows of a file ``unit, x, y, z[, kind]`` (separators and decimal marks as the logs:
    :func:`boreholes.read_table`), ``kind`` "wall" a wall row of a cut, and the notes."""
    rows, notes = _rows(path, None, ("unit", "x", "y", "z"))
    return [{"unit": r["unit"].strip(), "x": r["x"], "y": r["y"], "z": r["z"],
             "kind": "wall" if bh.normalise(r.get("kind") or "") in WALL_WORDS else ""} for r in rows], notes


def read_orientations(path) -> tuple[list[dict], list[str]]:
    """The rows of a file ``unit, x, y, z, dip, dip direction`` (degrees, the direction clockwise from
    north), and the notes."""
    rows, notes = _rows(path, {"dip": DIP_NAMES, "direction": DIRECTION_NAMES}, ("unit", "x", "y", "z", "dip", "direction"))
    good = []
    for r in rows:
        if 0.0 <= r["dip"] <= 180.0:
            good.append({"unit": r["unit"].strip(), "x": r["x"], "y": r["y"], "z": r["z"], "dip": r["dip"],
                         "direction": r["direction"] % 360.0})
        else:
            notes.append(t("riga {0}: inclinazione {1} fuori da 0-180 gradi", "line {0}: dip {1} outside 0-180 degrees")
                         .format(r["_line"], num(r["dip"])))
    return good, notes


def set_data(chunk, contacts=None, orientations=None) -> dict:
    """The rows of the files into the registry and the files under ``implicit/data/``: a file given
    replaces the one of its kind. Returns what was read: ``{"contacts", "orientations", "unknown"
    (units that are in no log and no mark: kept, and said again by the build), "notes"}``."""
    state = _state(chunk)
    stored = state.setdefault("data", {})
    codes = bh.order(chunk)
    known = codes + [u for u in cuts_of(chunk) if u not in codes]
    report: dict = {"contacts": None, "orientations": None, "unknown": [], "notes": []}
    for key, path, reader in (("contacts", contacts, read_contacts), ("orientations", orientations, read_orientations)):
        if not path:
            continue
        rows, notes = reader(path)
        out = folder(chunk) / DATA
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, out / f"{key}{Path(path).suffix}")
        stored[key] = {"file": Path(path).name, "rows": rows}
        report[key] = len(rows)
        report["notes"] += notes
        for r in rows:
            if bh.resolve_unit(r["unit"], known) is None and r["unit"] not in report["unknown"]:
                report["unknown"].append(r["unit"])
    _stale(chunk)
    return report


def data_text(report: dict) -> str:
    lines = []
    if report.get("contacts") is not None:
        lines.append(plural(report["contacts"], t("{0} contatto letto", "{0} contact read"),
                            t("{0} contatti letti", "{0} contacts read")))
    if report.get("orientations") is not None:
        lines.append(plural(report["orientations"], t("{0} giacitura letta", "{0} orientation read"),
                            t("{0} giaciture lette", "{0} orientations read")))
    if report.get("unknown"):
        lines.append(t("Unità sconosciute, per ora fuori dal modello: {0}", "Unknown units, out of the model for now: {0}")
                     .format(", ".join(report["unknown"])))
    return "\n".join(lines + ["- " + n for n in report.get("notes", [])[:15]])


# ------------------------------------------------------------------ what it says
class _Values(dict):
    def __missing__(self, key):
        return "?"


def warning_text(w: dict) -> str:
    """A warning of the engine or of this module in the language of the interface."""
    text = WARNINGS.get(w.get("code"))
    if text is None:
        return str(w.get("code"))
    values = _Values({k: num(v, 2) if isinstance(v, float) else v for k, v in w.items()})
    return t(*text).format_map(values)


def _metres(value) -> str:
    return "-" if value is None else num(value, 2) + " m"


def summary_text(chunk) -> str:
    """The last result: the sentence that it is an interpretation, errors side by side, volumes of the
    cuts as ranges, warnings."""
    reg = registry(chunk)
    if not reg.get("made"):
        return t("Il modello implicito non è ancora stato costruito.", "The implicit model has not been built yet.")
    lines = [t(*INTERPRETATION), t("Costruito il {0}.", "Built on {0}.").format(reg["made"])]
    if reg.get("stale"):
        lines.append(t("I dati sono cambiati: ricostruiscilo.", "The data changed: build it again."))
    g = reg.get("grid") or {}
    if g:
        lines.append(t("{0} x {1} celle di {2} m, voxel di {3} m, esagerazione verticale x{4}.",
                       "{0} x {1} cells of {2} m, voxels of {3} m, vertical exaggeration x{4}.")
                     .format(g["width"], g["height"], num(g["cell"], 2), num(reg.get("dz") or 0.0, 3),
                             num(reg.get("vertical") or 1.0, 1)))
    if reg.get("loo"):
        lines.append(t("Errore leave-one-out per contatto (la verifica lascia fuori i sondaggi a gruppi):",
                       "Leave-one-out error per contact (the check leaves the boreholes out by groups):"))
        for r in reg["loo"]:
            lines.append(t("{0}: implicito {1} ({2} sondaggi), kriging {3} ({4} sondaggi)",
                           "{0}: implicit {1} ({2} boreholes), kriging {3} ({4} boreholes)")
                         .format(r["unit"], _metres(r.get("rmse")), r.get("n", "?"), _metres(r.get("kriging_rmse")),
                                 r.get("kriging_points") or "-"))
    model = model_mod.registry(chunk)
    if model.get("method") == "implicit":
        lines.append(t("Volumi:", "Volumes:"))
        for r in model.get("units", []):
            if r.get("kind") == "cut":
                names = PROFILE_NAMES.get(r.get("profile"), ("", ""))
                lines.append(t("{0}: da {1} a {2} m³ secondo la forma del fondo (scelta: {3}, {4} m³)",
                               "{0}: from {1} to {2} m³ by the shape of the floor (chosen: {3}, {4} m³)")
                             .format(r["unit"], num(r["range_min"], 1), num(r["range_max"], 1), t(*names),
                                     num(r["volume_m3"], 1)))
            else:
                line = f"{r['unit']}: {num(r['volume_m3'], 1)} m³"
                if r.get("kriging_m3") is not None:
                    line += t(" (kriging {0} m³)", " (kriging {0} m³)").format(num(r["kriging_m3"], 1))
                lines.append(line)
    warnings = reg.get("warnings") or []
    if warnings:
        lines.append(t("Avvisi:", "Warnings:"))
        lines += ["- " + warning_text(w) for w in warnings]
    return "\n".join(lines)
