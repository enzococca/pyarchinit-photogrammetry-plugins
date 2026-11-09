"""A drawing classified by its orientation (the program's :mod:`pyarchinit_photogrammetry.engine.orient_lines`, since
1.7.7) read as the timbers of a hull: the lines of a layer of free drawings, or of a file, put in the ship plan (the
plugin's frame, the axis the keel line ``((0, 0), (1, 0))``) and classified there; the pairs across the keel are
*frames* (polygons), the lines along it *seams* (lines), the pairs along it *planks* (polygons); the lines across
left single and the other directions are not timbers and stay out. The side of each comes from its mean y as for a
timber traced by hand (:func:`tracing._side`); the names are the plugin's (``ordinata N`` from the bow, ``comento
B1``, ``fascia T2``).

Nothing enters the catalogue before the check plate: :func:`propose` reads and classifies, :func:`plate` draws it
with the plugin's names, :func:`accept` writes the records the user keeps (``status`` measured, ``source``
``orient:<drawing>:<element id>``); a timber already imported from a classified drawing (same kind and vertices) is
skipped, so importing the same drawing again adds nothing. A proposal lives in memory only: it is not stored in the project.

Without the program 1.7.7 :data:`orient_lines` is None and :func:`available` False: the window hides the action and
the command line says what is missing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import relief, ship, timbers, tracing
from .texts import t

try:  # the program 1.7.7 or later
    from pyarchinit_photogrammetry.engine import orient_lines
except ImportError:  # an older program: the action is hidden
    orient_lines = None

MEANING = {"pair_across": ("frame", "polygon"), "along": ("seam", "line"), "pair_along": ("plank", "polygon")}
"""The groups of the classification that are timbers: kind of the record and type of its geometry."""
KINDS = ("frame", "seam", "plank")
AXIS = ((0.0, 0.0), (1.0, 0.0))  # the keel line in the ship plan
FILE_SUFFIXES = (".gpkg", ".geojson", ".json", ".dxf")
NEEDS_PROGRAM = ("serve il programma 1.7.7 o successivo", "the program 1.7.7 or later is needed")


def available() -> bool:
    """Whether the program classifies drawings (1.7.7 or later)."""
    return orient_lines is not None


def need_program() -> None:
    if orient_lines is None:
        raise ship.NotReady(t(f"Disegno classificato: {NEEDS_PROGRAM[0]}.", f"Classified drawing: {NEEDS_PROGRAM[1]}."))


def _not_ready(exc) -> ship.NotReady:
    """An error of the classification in the language of the interface."""
    italian = getattr(exc, "italian", None)
    return ship.NotReady(t(italian, str(exc)) if italian else str(exc))


def _source(source) -> tuple[str, str]:
    """``("layer", name)`` or ``("file", path)``: a :class:`~pathlib.Path` is a file, a text a layer, a pair as it is."""
    if isinstance(source, tuple) and len(source) == 2 and source[0] in ("layer", "file"):
        return source[0], str(source[1])
    if isinstance(source, Path):
        return "file", str(source)
    return "layer", str(source)


def _world_lines(chunk, kind: str, value: str) -> list[np.ndarray]:
    try:
        if kind == "file":
            return orient_lines.lines_from_file(Path(value), chunk=chunk)
        return orient_lines.lines_from_layer(chunk, value)
    except orient_lines.OrientLinesError as exc:
        raise _not_ready(exc) from None


def _xy(element) -> np.ndarray:
    g = element.geometry
    if g.geom_type == "Polygon":
        return np.asarray(g.exterior.coords, float)[:-1]
    return np.asarray(g.coords, float)


def _names(records: list[dict], frame: ship.Frame) -> None:
    """``ordinata N`` from the bow (both sides of one frame share N, as the proposals name them); the seams and the
    planks numbered on each side from the keel: ``comento B1``, ``fascia T2`` (``C`` on the keel line)."""
    from . import proposals
    from .clinker import SIDE_CODES

    frames = [r for r in records if r["kind"] == "frame"]
    proposals._name_frames(frames, frame.bow_x)
    words = {"seam": t("comento", "seam"), "plank": t("fascia", "strake")}
    for kind, word in words.items():
        for side in ("port", "starboard", "centre"):
            mine = sorted((r for r in records if r["kind"] == kind and r["side"] == side), key=lambda r: abs(r["y"]))
            code = SIDE_CODES[side][0] if side in SIDE_CODES else "C"
            for k, r in enumerate(mine, 1):
                r["name"] = f"{word} {code}{k}"


def propose(chunk, source, params=None) -> dict:
    """The drawing ``source`` (a layer of free drawings by name, a file as a :class:`~pathlib.Path`, or ``("layer",
    name)`` / ``("file", path)``) classified in the ship plan with ``params`` over the program's defaults (also
    ``pair_isolation``: frames drawn as loose lines close together need 1, or drawing them as polygons). Returns
    ``{"classification", "records", "source", "frame_key", "accepted"}``: ``records`` one per element that is a
    timber, ``{"id" (the element's), "group", "kind", "geometry" (line|polygon), "side", "name", "ship_xy", "x", "y",
    "width", "length"}``. Nothing is written in the chunk."""
    need_program()
    frame = ship.need_frame(chunk)
    kind, value = _source(source)
    world = _world_lines(chunk, kind, value)
    plan = [frame.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2] for w in world]
    try:
        result = orient_lines.classify(plan, AXIS, params)
    except orient_lines.OrientLinesError as exc:
        raise _not_ready(exc) from None
    records = []
    for e in result.elements:
        if e.group not in MEANING:
            continue
        rkind, gtype = MEANING[e.group]
        xy = _xy(e)
        records.append({"id": e.id, "group": e.group, "kind": rkind, "geometry": gtype,
                        "side": tracing._side("auto", xy), "name": "", "ship_xy": xy.tolist(),
                        "x": float(e.position), "y": float(xy[:, 1].mean()),
                        "width": None if e.width is None else float(e.width), "length": float(e.length)})
    _names(records, frame)
    label = f"{kind}:{Path(value).name if kind == 'file' else value}"
    return {"classification": result, "records": records, "source": {kind: value}, "label": label,
            "frame_key": relief.frame_key(frame), "accepted": [], "already": []}


def counts(proposal: dict) -> dict:
    """Records per kind (every kind of :data:`KINDS`) and the elements left out (``skipped``)."""
    out = {k: 0 for k in KINDS}
    for r in proposal["records"]:
        out[r["kind"]] += 1
    out["skipped"] = len(proposal["classification"].elements) - len(proposal["records"])
    return out


def plate(chunk, proposal: dict, path) -> Path:
    """The program's check plate of the classification, with the plugin's names on the timbers (the elements left
    out keep their ids). PNG or SVG by the suffix of ``path``."""
    from .texts import _italian

    names = {r["id"]: r["name"] for r in proposal["records"]}
    title = t(f"Disegno classificato - {chunk.label}", f"Classified drawing - {chunk.label}")
    return proposal["classification"].plate(Path(path), language="it" if _italian() else "en", names=names, title=title)


def _key(kind: str, world_xy) -> tuple:
    """A timber of a classified drawing by its kind and its vertices (world x, y to the millimetre, in any order)."""
    pts = np.round(np.asarray(world_xy, float).reshape(-1, 2), 3)
    return kind, frozenset(map(tuple, pts.tolist()))


def imported_keys(chunk) -> set:
    """The :func:`_key` of every timber of the catalogue that came from a classified drawing (``source`` orient:...)."""
    return {_key(r["kind"], r["geometry"]["xy"]) for r in timbers.timbers_of(chunk)
            if str(r.get("source") or "").startswith("orient:")}


def accept(chunk, proposal: dict, ids=None, kinds=None) -> list[dict]:
    """The records ``ids`` of ``proposal`` (element ids; None: all not yet accepted, of ``kinds`` when given) into the
    catalogue: ``status`` measured, ``source`` ``orient:<drawing>:<id>``, ``ship_xy`` as classified, ``geometry`` in
    the chunk's coordinates. A proposal read in another ship frame, an id unknown or already accepted from this
    proposal is refused. A timber already in the catalogue from a classified drawing (same kind and vertices: the same
    drawing imported again) is skipped: its id goes to ``proposal["already"]``. Returns the records written."""
    frame = ship.need_frame(chunk)
    key = relief.frame_key(frame)
    if proposal.get("frame_key") != key:
        raise ship.NotReady(t("Il disegno è stato classificato in un altro sistema della nave: classificalo di nuovo.",
                              "The drawing was classified in another ship frame: classify it again."))
    by_id = {r["id"]: r for r in proposal["records"]}
    done = set(proposal.setdefault("accepted", []))
    if ids is None:
        chosen = [r for r in proposal["records"] if r["id"] not in done and (kinds is None or r["kind"] in kinds)]
    else:
        ids = [str(i) for i in ids]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ship.NotReady(t(f"Elementi che non sono legni del disegno: {', '.join(missing)}.",
                                  f"Elements that are not timbers of the drawing: {', '.join(missing)}."))
        again = [i for i in ids if i in done]
        if again:
            raise ship.NotReady(t(f"Elementi già nel catalogo: {', '.join(again)}.",
                                  f"Elements already in the catalogue: {', '.join(again)}."))
        chosen = [by_id[i] for i in dict.fromkeys(ids)]
    out = []
    there = imported_keys(chunk)
    proposal["already"] = []
    label = proposal.get("label", "")
    for r in chosen:
        world = tracing._world(frame, r["ship_xy"])
        k = _key(r["kind"], world)
        proposal["accepted"].append(r["id"])
        if k in there:
            proposal["already"].append(r["id"])
            continue
        there.add(k)
        out.append(timbers.add(chunk, r["kind"], r["side"], r["name"], timbers.geometry(r["geometry"], world),
                               ship_xy=[[float(x), float(y)] for x, y in r["ship_xy"]], status="measured",
                               chunk=chunk.key, source=f"orient:{label}:{r['id']}" if label else f"orient:{r['id']}",
                               frame_key=key))
    if out:
        tracing.update_visible(chunk)
    return out
