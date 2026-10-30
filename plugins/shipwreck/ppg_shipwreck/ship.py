"""The ship frame, the stations and what the plugin keeps in ``chunk.meta["shipwreck"]``.

Ship coordinates: x along the keel from the start point (the stem unless told otherwise), y to
port, z up, in metres. Up is the world's vertical and the keel line is levelled: x is the direction
of the keel in plan, z the world's height above the datum ``z0``; a keel with a slope of its own
keeps it as the number ``slope``. When the bow is at the start, x runs aft and port lies on the right
of who walks along x, so the frame is a mirror image of a right-handed one (``sign`` -1): plans are
drawn with this in mind, nothing else notices. The frame only reads the chunk: its transform, model
and products are never changed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.core import crs as crs_mod
from pyarchinit_photogrammetry.core import geometry

from .texts import t

KEY = "shipwreck"


class NotReady(ValueError):
    """Something the step needs is missing: the message says what, for the user."""


def state(chunk) -> dict:
    """The plugin's part of ``chunk.meta`` (created when first asked for)."""
    return chunk.meta.setdefault(KEY, {})


def folder(chunk) -> Path:
    """``<chunk data>/shipwreck``: the only folder the plugin writes in by itself."""
    d = chunk.data_dir / "shipwreck"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_metric(chunk) -> None:
    """Refuse a chunk in degrees: a frame in metres cannot be set there."""
    if crs_mod.is_geographic(chunk.crs):
        raise NotReady(t(f"Il chunk è in coordinate geografiche ({chunk.crs}): gradi, non metri. Le misure di uno scafo "
                         "servono in metri: porta il chunk in un sistema proiettato (pannello GNSS > Sistema e quote, "
                         "o Riferimenti) e riprova.",
                         f"The chunk is in geographic coordinates ({chunk.crs}): degrees, not metres. The measures of "
                         "a hull need metres: bring the chunk to a projected system (GNSS pane > System and heights, "
                         "or Reference) and try again."))


def is_scaled(chunk) -> bool:
    """Metres are metres: a projected system, or a local frame with a scale (the program's test)."""
    if not crs_mod.is_local(chunk.crs) or chunk.meta.get("metric") or abs(chunk.scale - 1.0) > 1e-9:
        return True
    return any(m.reference.location for m in chunk.markers) or any(not sb.checkpoint for sb in chunk.scalebars)


# ----------------------------------------------------------------------------------- the frame
@dataclass(frozen=True)
class Frame:
    start: tuple  # world x, y, z of the point where ship x = ``x0``
    end: tuple  # world x, y, z of the other end of the keel line
    bow: str = "start"  # the end of the keel line where the bow is: "start" or "end"
    z0: float | None = None  # world height of ship z = 0 (the start point's when None)
    x0: float = 0.0  # ship x of the start point (not 0 when the start is a chosen station)

    def __post_init__(self):
        object.__setattr__(self, "start", tuple(float(v) for v in self.start))
        object.__setattr__(self, "end", tuple(float(v) for v in self.end))
        object.__setattr__(self, "z0", float(self.start[2] if self.z0 is None else self.z0))
        object.__setattr__(self, "x0", float(self.x0))
        if self.bow not in ("start", "end"):
            raise ValueError("bow: 'start' or 'end'")
        if self.length <= 0:
            raise NotReady(t("I due estremi della chiglia coincidono.", "The two ends of the keel coincide."))

    @property
    def length(self) -> float:
        """The keel line in plan, in metres."""
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    @property
    def slope(self) -> float:
        """The keel's own slope (rise / run along x): kept as a number, not taken out."""
        return (self.end[2] - self.start[2]) / self.length

    @property
    def sign(self) -> int:
        """+1 when port is on the left of who walks along x (x runs forward), -1 on the right."""
        return -1 if self.bow == "start" else 1

    @property
    def heading(self) -> float:
        """The direction of x, degrees clockwise from the world's north (+Y of the chunk)."""
        return float(np.degrees(np.arctan2(self.end[0] - self.start[0], self.end[1] - self.start[1])) % 360.0)

    @property
    def matrix(self) -> np.ndarray:
        """4 x 4 from world to ship coordinates (a mirror when ``sign`` is -1)."""
        dx = (self.end[0] - self.start[0]) / self.length
        dy = (self.end[1] - self.start[1]) / self.length
        m = np.eye(4)
        m[0, :2] = dx, dy
        m[1, :2] = -dy * self.sign, dx * self.sign  # the left of x, or its right
        m[0, 3] = self.x0 - (dx * self.start[0] + dy * self.start[1])
        m[1, 3] = -(m[1, 0] * self.start[0] + m[1, 1] * self.start[1])
        m[2, 3] = -self.z0
        return m

    def to_ship(self, world) -> np.ndarray:
        """World ``(N, 3)`` to ship ``(N, 3)``. Relative to the start point, so that coordinates of
        millions of metres lose nothing."""
        p = np.asarray(world, float).reshape(-1, 3) - np.asarray(self.start)
        m = self.matrix
        out = p.copy()
        out[:, 0] = m[0, 0] * p[:, 0] + m[0, 1] * p[:, 1] + self.x0
        out[:, 1] = m[1, 0] * p[:, 0] + m[1, 1] * p[:, 1]
        out[:, 2] = p[:, 2] + self.start[2] - self.z0
        return out

    def to_world(self, ship) -> np.ndarray:
        """Ship ``(N, 3)`` to world ``(N, 3)``."""
        s = np.asarray(ship, float).reshape(-1, 3)
        m = self.matrix
        x = s[:, 0] - self.x0
        return np.column_stack([self.start[0] + m[0, 0] * x + m[1, 0] * s[:, 1],
                                self.start[1] + m[0, 1] * x + m[1, 1] * s[:, 1], s[:, 2] + self.z0])

    def flipped(self) -> "Frame":
        """The ends swapped: x starts from the other end, the bow is said to be at the new start
        as it was at the old one, the datum stays."""
        return Frame(self.end, self.start, self.bow, self.z0, self.x0)

    def to_dict(self) -> dict:
        return {**asdict(self), "start": list(self.start), "end": list(self.end), "length": self.length,
                "slope": self.slope, "heading": self.heading}

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(d["start"], d["end"], d.get("bow", "start"), d.get("z0"), d.get("x0", 0.0))


def frame_of(chunk) -> Frame | None:
    d = chunk.meta.get(KEY, {}).get("frame")
    return None if d is None else Frame.from_dict(d)


def need_frame(chunk) -> Frame:
    frame = frame_of(chunk)
    if frame is None:
        raise NotReady(t("Imposta prima il sistema della nave.", "Set the ship frame first."))
    return frame


def set_frame(chunk, frame: Frame) -> Frame:
    check_metric(chunk)
    state(chunk)["frame"] = {**frame.to_dict(), "crs": chunk.crs}
    return frame


def world_vertices(chunk) -> np.ndarray:
    """The vertices of the model in world coordinates, in double precision."""
    mesh = _mesh(chunk)
    return geometry.apply(chunk.transform, np.asarray(mesh.vertices, float))


def _mesh(chunk):
    mesh = chunk.model
    if mesh is None:
        raise NotReady(t("Il chunk non ha un modello 3D: costruiscilo prima.",
                         "The chunk has no 3D model: build it first."))
    return mesh


class Surface:
    """The top of the model over a point of the plan: the weighted height of the nearest vertices,
    good to about one edge of the mesh. ``xyz``: the vertices, in any frame with z up."""

    def __init__(self, xyz: np.ndarray):
        from scipy.spatial import cKDTree

        xyz = np.asarray(xyz, float)
        self.origin = xyz[:, :2].mean(axis=0)
        self.z = xyz[:, 2]
        self.tree = cKDTree(xyz[:, :2] - self.origin)
        sample = xyz[:: max(1, len(xyz) // 2000), :2] - self.origin
        self.reach = 4.0 * float(np.median(self.tree.query(sample, k=2)[0][:, 1]))

    def at(self, xy) -> np.ndarray:
        """Heights at ``(N, 2)``; NaN where the model is farther than a few edges."""
        d, i = self.tree.query(np.asarray(xy, float).reshape(-1, 2) - self.origin, k=min(4, len(self.z)))
        d, i = d.reshape(len(d), -1), i.reshape(len(i), -1)
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        out = (w * self.z[i]).sum(axis=1) / w.sum(axis=1)
        out[d[:, 0] > self.reach] = np.nan
        return out


def frame_from_picks(chunk, a, b, bow: str = "start", z0: float | None = None, x0: float = 0.0) -> Frame:
    """The frame through two points of the keel line: world ``(x, y)`` picked on the raster, with the
    heights read from the model, or ``(x, y, z)``."""
    check_metric(chunk)
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) < 3 or len(b) < 3:
        heights = Surface(world_vertices(chunk)).at(np.vstack([a[:2], b[:2]]))
        if not np.isfinite(heights).all():
            raise NotReady(t("Un estremo della chiglia è fuori dal modello.", "An end of the keel lies off the model."))
        a, b = np.r_[a[:2], heights[0]], np.r_[b[:2], heights[1]]
    return Frame(a, b, bow, z0, x0)


def frame_from_markers(chunk, first, second, bow: str = "start", z0: float | None = None, x0: float = 0.0) -> Frame:
    """The frame through two markers (name or key) that have a position."""
    points = []
    for name in (first, second):
        marker = next((m for m in chunk.markers if str(name) in (m.label, str(m.key))), None)
        if marker is None:
            raise NotReady(t(f"Marker {name} non trovato.", f"Marker {name} not found."))
        if marker.position is None:
            raise NotReady(t(f"Il marker {name} non ha una posizione.", f"Marker {name} has no position."))
        points.append(chunk.to_world(np.asarray(marker.position, float)[None])[0])
    return Frame(points[0], points[1], bow, z0, x0)


# -------------------------------------------------------------------------------- the stations
def number(text) -> float:
    """A number of a table: decimal point or comma."""
    return float(str(text).strip().replace(",", "."))


def rows_of(path) -> tuple[list[str], list[list[str]]]:
    """Header (lower case, empty when the first line is data) and rows of a table; separator
    ``;``, tab or comma."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], []
    sep = next((s for s in (";", "\t") if s in lines[0]), ",")
    rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=sep)]
    try:
        number(rows[0][-1])
        return [], rows
    except (ValueError, IndexError):
        pass
    if len(rows[0]) > 1:
        try:
            number(rows[0][1])
            return [], rows
        except ValueError:
            pass
    return [c.lower() for c in rows[0]], rows[1:]


def regular(spacing: float, count: int, start: float = 0.0, prefix: str = "S") -> list[dict]:
    """``count`` stations ``spacing`` metres apart from ``start``: S1, S2... (names of ours: the field's
    numbers come from a table)."""
    if spacing <= 0 or count < 1:
        raise NotReady(t("Servono una distanza positiva e almeno una stazione.",
                         "A positive spacing and at least one station are needed."))
    return [{"name": f"{prefix}{i + 1}", "x": round(float(start + i * spacing), 6)} for i in range(int(count))]


def stations_from_table(path) -> list[dict]:
    """Stations from a table ``name, x`` (F30, 12.4): the names are the field's."""
    header, rows = rows_of(path)
    name_col = next((header.index(k) for k in ("name", "nome", "stazione", "station") if k in header), 0)
    x_col = next((header.index(k) for k in ("x", "dist", "distanza") if k in header), 1)
    out = []
    for i, row in enumerate(rows, 2 if header else 1):
        try:
            out.append({"name": row[name_col], "x": number(row[x_col])})
        except (ValueError, IndexError):
            raise NotReady(t(f"Riga {i} della tabella delle stazioni: serve nome, x.",
                             f"Line {i} of the table of stations: name, x needed.")) from None
    if not out:
        raise NotReady(t("La tabella delle stazioni è vuota.", "The table of stations is empty."))
    return out


def stations_of(chunk) -> list[dict]:
    return list(chunk.meta.get(KEY, {}).get("stations", []))


def _clean(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def set_stations(chunk, stations) -> list[dict]:
    """The stations of the chunk, in order along x; names unique."""
    out = sorted(({"name": _clean(s["name"]), "x": float(s["x"])} for s in stations), key=lambda s: s["x"])
    names = [s["name"] for s in out]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise NotReady(t(f"Stazioni con lo stesso nome: {', '.join(twice)}.",
                         f"Stations with the same name: {', '.join(twice)}."))
    state(chunk)["stations"] = out
    return out
