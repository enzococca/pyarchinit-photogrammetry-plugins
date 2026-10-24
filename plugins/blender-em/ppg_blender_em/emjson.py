"""The Extended Matrix of the site as an em.json, read without s3dgraphy.

PyArchInit (*Extended Matrix > Esporta sito in em.json*), EMStudio and EMtools 1.6 write the matrix as an em.json: a
container ``{"header", "graphs": {id: graph}, "active_graph_id"}`` of one or more graphs (one per
saggio when PyArchInit splits the site), or the single graph of the files before the container
and of the snapshots of the live sync, ``{"header", "graph"}``. A graph is a flat list of nodes
``{id, node_type, name, description, data}`` and of edges ``{id, edge_type, source, target}``.

The plugin only reads it: the matrix belongs to who writes it. It needs

* the stratigraphic units: their id (for PyArchInit the ``node_uuid`` of the US form), their name
  (``1.US12``, the composite label of PyArchInit; the name EMtools pairs a proxy object with) and
  the ``us``, ``area`` and ``unita_tipo`` PyArchInit writes into ``data``;
* the epoch of each unit (its ``has_first_epoch`` edge) with its colour, for the colours in Blender;
* the shift of the coordinates of the project (the ``geo_position`` node), when it has one.

:func:`match` pairs the proxies of the chunk (``chunk.meta["proxies"]``: name, us, area, tipo) with
the units: by the ``node_uuid`` of their US form when the program knows it, else by the fields of
the form, else by name; a unit goes to one proxy only, and what stays ambiguous is said.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# the stratigraphic node types of s3dgraphy (s3Dgraphy_node_datamodel.json, stratigraphic_nodes)
# that stand for a unit with a body; BR (continuity) and SE (event) have no proxy
UNIT_TYPES = frozenset({"US", "USVs", "USVn", "SF", "VSF", "RSF", "serSU", "serUSD", "serUSVn", "serUSVs",
                        "serVSF", "USD", "TSU", "UL", "USN", "USNt"})
FIRST_EPOCH = "has_first_epoch"
NOT_STUDIES = ("ShelfGraph", "DTCCorpus")  # the em_collection of the members that hold no units
# a composite label of PyArchInit (area.unita_tipo + us) or a bare one (US12, USVs101)
_NAME = re.compile(r"^(?:(?P<area>[^.]*)\.)?(?P<tipo>[A-Za-z]+)_?(?P<us>[0-9][^\s]*)$")


class EmJsonError(ValueError):
    """A file that is not an em.json this plugin can read, worded for the user."""


@dataclass
class Epoch:
    id: str
    name: str
    start: float | None = None
    end: float | None = None
    color: str | None = None


@dataclass
class Unit:
    """A stratigraphic unit of the matrix."""

    id: str
    name: str
    node_type: str
    description: str = ""
    data: dict = field(default_factory=dict)
    graph_id: str = ""
    epoch: Epoch | None = None

    @property
    def us(self) -> str:
        value = self.data.get("us")
        return str(value).strip() if value not in (None, "") else (_parsed(self.name).get("us") or "")

    @property
    def area(self) -> str:
        value = self.data.get("area")
        return str(value).strip() if value not in (None, "") else (_parsed(self.name).get("area") or "")

    @property
    def tipo(self) -> str:
        """The ``unita_tipo`` of the US form (USM, USR...), else the node type of the matrix."""
        value = self.data.get("unita_tipo")
        if value not in (None, ""):
            return str(value).strip()
        return _parsed(self.name).get("tipo") or self.node_type

    @property
    def node_uuid(self) -> str:
        return str(self.data.get("node_uuid") or self.id)

    @property
    def color(self) -> str | None:
        return self.epoch.color if self.epoch is not None else None


@dataclass
class Matrix:
    """What the plugin reads of an em.json."""

    path: Path | None
    graph_id: str
    name: str
    code: str | None
    units: list[Unit]
    epochs: list[Epoch]
    shift: tuple[float, float, float] | None = None  # the shift of geo_position, when not zero
    epsg: str | None = None
    graphs: list[str] = field(default_factory=list)

    def unit(self, node_id: str) -> Unit | None:
        return next((u for u in self.units if u.id == node_id or u.node_uuid == node_id), None)

    def named(self, name: str) -> Unit | None:
        return next((u for u in self.units if u.name == name), None)


def _parsed(name: str) -> dict:
    m = _NAME.match(str(name or "").strip())
    return m.groupdict() if m else {}


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):  # a name in several languages
        return str(value.get("default") or next(iter(value.values()), "") or "")
    return str(value)


def graph_sections(doc: dict) -> tuple[dict[str, dict], str]:
    """The graphs of an em.json document and the id of the active one: a container
    (``graphs``), the single ``graph`` of the files before it, or a live-sync snapshot
    (``{"doc": ...}``)."""
    if isinstance(doc, dict) and isinstance(doc.get("doc"), dict):  # a snapshot of the sync
        doc = doc["doc"]
    if not isinstance(doc, dict):
        raise EmJsonError("not an em.json document")
    header = doc.get("header") or {}
    if header and header.get("format") not in (None, "em.json"):
        raise EmJsonError(f"not an em.json document (format {header.get('format')!r})")
    if isinstance(doc.get("graphs"), dict):
        graphs = {str(k): v for k, v in doc["graphs"].items() if isinstance(v, dict)}
        active = str(doc.get("active_graph_id") or next(iter(graphs), ""))
    elif isinstance(doc.get("graph"), dict):
        g = doc["graph"]
        graphs = {str(g.get("graph_id") or "graph"): g}
        active = next(iter(graphs))
    else:
        raise EmJsonError("an em.json without graphs")
    # the shelf (the library of assets) and the documentation corpus are members of the container
    # that are not a matrix, told by their marker, never by their id
    graphs = {k: v for k, v in graphs.items() if (v.get("data") or {}).get("em_collection") not in NOT_STUDIES}
    if not graphs:
        raise EmJsonError("an em.json without graphs")
    if active not in graphs:
        active = next(iter(graphs))
    return graphs, active


def read(doc: dict, path: Path | None = None) -> Matrix:
    """The units, epochs and shift of an em.json document (already parsed)."""
    graphs, active = graph_sections(doc)
    units, epochs, shift, epsg = [], {}, None, None
    for graph_id, g in graphs.items():
        nodes = [n for n in g.get("nodes") or [] if isinstance(n, dict) and n.get("id") is not None]
        by_id = {str(n["id"]): n for n in nodes}
        local_epochs = {}
        for n in nodes:
            if n.get("node_type") == "EpochNode":
                data = n.get("data") or {}
                local_epochs[str(n["id"])] = Epoch(str(n["id"]), _text(n.get("name")), _number(data.get("start_time")),
                                                   _number(data.get("end_time")),
                                                   data.get("color") or data.get("fill_color"))
            elif n.get("node_type") == "geo_position" and graph_id == active:
                data = n.get("data") or {}
                xyz = tuple(_number(data.get(k)) or 0.0 for k in ("shift_x", "shift_y", "shift_z"))
                if any(xyz):
                    shift = xyz
                if data.get("epsg") not in (None, "", "NotSet", 0, "0"):
                    epsg = str(data.get("epsg"))
        epochs.update(local_epochs)
        first = {}
        for e in g.get("edges") or []:
            if isinstance(e, dict) and e.get("edge_type") == FIRST_EPOCH and str(e.get("target")) in local_epochs:
                first.setdefault(str(e.get("source")), local_epochs[str(e.get("target"))])
        for nid, n in by_id.items():
            if n.get("node_type") in UNIT_TYPES:
                units.append(Unit(nid, _text(n.get("name")) or nid, str(n["node_type"]), _text(n.get("description")),
                                  dict(n.get("data") or {}), graph_id, first.get(nid)))
    g = graphs[active]
    code = g.get("code") or None
    return Matrix(path, active, _text(g.get("name")) or active, str(code) if code else None, units,
                  list(epochs.values()), shift, epsg, list(graphs))


def load(path: str | Path) -> Matrix:
    """The em.json at ``path`` (see :func:`read`)."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EmJsonError(f"{path.name}: {exc}") from exc
    return read(doc, path.resolve())


@dataclass
class Match:
    """A proxy of the chunk and the unit of the matrix it stands for (None: not in the matrix)."""

    entry: dict
    unit: Unit | None
    how: str  # "node_uuid", "form", "name", "" when not found
    problem: str = ""

    @property
    def name(self) -> str:
        """The name of the proxy in the EM project: the unit's, else the one the program gave it."""
        return self.unit.name if self.unit is not None else str(self.entry.get("name") or "")


def _same(a, b) -> bool:
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def match(entries: list[dict], matrix: Matrix | None, node_uuids: dict | None = None) -> list[Match]:
    """The proxies of the chunk paired with the units of ``matrix``. ``node_uuids``: the ``node_uuid``
    of the US form of each proxy (``{(area, tipo, us): uuid}``), when a PyArchInit database is at
    hand. A unit is claimed by one proxy at most; two candidates for one proxy are a problem, said,
    not a guess."""
    out = [Match(e, None, "") for e in entries]
    if matrix is None:
        return out
    claimed: set[str] = set()

    def claim(m: Match, candidates: list[Unit], how: str) -> bool:
        free = [u for u in candidates if u.id not in claimed]
        if len(free) == 1:
            m.unit, m.how = free[0], how
            claimed.add(free[0].id)
            return True
        if len(free) > 1:
            m.problem = ", ".join(sorted(u.name for u in free))
        return False

    for m in out:  # 1. the node_uuid of the US form
        e = m.entry
        uid = (node_uuids or {}).get((str(e.get("area") or ""), str(e.get("tipo") or "US"), str(e.get("us") or "")))
        if uid:
            claim(m, [u for u in matrix.units if uid in (u.id, u.node_uuid)], "node_uuid")
    for m in out:  # 2. the fields of the form
        if m.unit is not None or not m.entry.get("us"):
            continue
        e = m.entry
        found = [u for u in matrix.units if u.data.get("us") not in (None, "") and _same(u.us, e.get("us"))
                 and _same(u.tipo, e.get("tipo") or "US")
                 and (not e.get("area") or not u.area or _same(u.area, e.get("area")))]
        claim(m, found, "form")
    for m in out:  # 3. the name
        if m.unit is not None or m.problem:
            continue
        e = m.entry
        bare = f"{e.get('tipo') or 'US'}{e.get('us') or ''}" if e.get("us") else str(e.get("name") or "")
        names = {bare, str(e.get("name") or "")} | ({f"{e['area']}.{bare}"} if e.get("area") else set())
        found = [u for u in matrix.units if u.name in names]
        if not found:
            found = [u for u in matrix.units if u.name.endswith("." + bare)]
        claim(m, found, "name")
    return out


__all__ = ["Epoch", "Unit", "Matrix", "Match", "EmJsonError", "UNIT_TYPES", "read", "load", "match",
           "graph_sections"]
