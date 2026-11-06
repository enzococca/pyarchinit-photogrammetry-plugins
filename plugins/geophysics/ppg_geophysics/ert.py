"""Electrical resistivity tomography (ERT) profiles of a chunk: the readings of a resistivity meter,
the line in the survey, the pseudosection and its filters.

A profile is a record in ``chunk.meta["geophysics"]["ert"]`` (id, name, source, the electrodes and
readings counted, the placement of the line, the filters, the inversion and its fit, the files) and
an archive of arrays ``<chunk data>/geophysics/ert/<id>/profile.npz``:

* the electrodes as read (``file_u``: distance along the line; ``file_x``/``file_y``: world
  coordinates when the file has them; ``file_z``: heights when it has them) and, once the line is
  placed, ``elec_u`` (distance from the start of the line), ``elec_x``, ``elec_y``, ``elec_z``;
* the readings: ``a b m n`` (electrodes counted from 0, -1 at infinity), ``rhoa`` (apparent
  resistivity, ohm m), ``r`` (resistance, ohm), ``err`` (relative error, a fraction), ``u`` (volts),
  ``i`` (amperes), ``ip``, ``k`` (geometric factor); NaN where the file has no value.

Readers of this module (no pyGIMLi): the unified data format of pyGIMLi/BERT (``.ohm``, ``.shm``,
``.dat`` with a count of electrodes on the first line), a plain table of readings (CSV, TXT:
``a,b,m,n,rhoa[,err]``, or resistance, or voltage and current; electrode numbers from 1 or
positions in metres - the Syscal "Spa.1..4" columns too) and a table of electrodes (``x,z`` along the
line, or world ``x,y[,z]``). Other files of the instruments (RES2DINV, ABEM, Syscal binary exports)
are read by pyGIMLi in its worker (:mod:`.ert_plugin`).
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .texts import t

FOLDER = "geophysics/ert"
VALUES = ("rhoa", "r", "err", "u", "i", "ip", "k")
DEFAULT_RULES = {"min_rhoa": None, "max_rhoa": None, "max_error": None, "reciprocal": None, "spikes": None}
SPIKE_NEIGHBOURS = 8


class NeedsGimli(ValueError):
    """The file is not one of the formats this module reads: pyGIMLi's readers are needed."""


@dataclass
class Profile:
    """The electrodes and readings of one ERT line (see the module)."""

    name: str
    file_u: np.ndarray
    a: np.ndarray
    b: np.ndarray
    m: np.ndarray
    n: np.ndarray
    values: dict = field(default_factory=dict)
    file_z: np.ndarray | None = None
    file_x: np.ndarray | None = None
    file_y: np.ndarray | None = None
    id: str = ""
    source: str = ""
    fmt: str = ""
    notes: list = field(default_factory=list)
    elec: dict = field(default_factory=dict)  # elec_u, elec_x, elec_y, elec_z after the placement
    numbered: bool = True  # the electrodes of the file are numbers (not positions in metres)

    @property
    def count(self) -> int:
        return int(len(self.a))

    @property
    def electrodes(self) -> int:
        return int(len(self.file_u))

    def value(self, name: str) -> np.ndarray:
        v = self.values.get(name)
        return np.full(self.count, np.nan) if v is None else np.asarray(v, float)

    def has(self, name: str) -> bool:
        v = self.values.get(name)
        return v is not None and bool(np.isfinite(np.asarray(v, float)).any())

    @property
    def world(self) -> bool:
        """The file gave the electrodes in world coordinates (x, y)."""
        return self.file_x is not None and self.file_y is not None

    @property
    def spacing(self) -> float:
        u = np.sort(np.asarray(self.file_u, float))
        d = np.diff(u)
        d = d[d > 1e-9]
        return float(np.median(d)) if d.size else 1.0

    def placed(self) -> bool:
        return "elec_u" in self.elec

    def positions(self) -> np.ndarray:
        """The distance of each electrode along the line: placed, else as read."""
        return np.asarray(self.elec.get("elec_u", self.file_u), float)

    def heights(self) -> np.ndarray:
        if "elec_z" in self.elec:
            return np.asarray(self.elec["elec_z"], float)
        if self.file_z is not None and np.isfinite(self.file_z).all():
            return np.asarray(self.file_z, float)
        return np.zeros(self.electrodes)

    def apparent(self) -> np.ndarray:
        """The apparent resistivity of each reading: as read, else resistance x geometric factor (the
        factor of the file, else of a flat line)."""
        rhoa = self.value("rhoa")
        if np.isfinite(rhoa).any():
            return rhoa
        r = self.value("r")
        if not np.isfinite(r).any():
            u, i = self.value("u"), self.value("i")
            with np.errstate(divide="ignore", invalid="ignore"):
                r = u / i
        k = self.value("k")
        if not np.isfinite(k).any():
            k = flat_k(self.positions(), self.a, self.b, self.m, self.n)
        return r * k

    def arrays(self) -> dict:
        out = {"file_u": np.asarray(self.file_u, float), "a": self.a, "b": self.b, "m": self.m, "n": self.n}
        for name in ("file_x", "file_y", "file_z"):
            if getattr(self, name) is not None:
                out[name] = np.asarray(getattr(self, name), float)
        for k, v in self.values.items():
            out[k] = np.asarray(v, float)
        for k, v in self.elec.items():
            out[k] = np.asarray(v, float)
        return out


# ------------------------------------------------------------------ geometry of the readings
def flat_k(positions, a, b, m, n) -> np.ndarray:
    """The geometric factors on a flat line (electrodes at ``positions``; -1 = at infinity)."""
    x = np.asarray(positions, float)

    def inv(i, j):
        out = np.zeros(len(i))
        ok = (i >= 0) & (j >= 0)
        out[ok] = 1.0 / np.maximum(np.abs(x[i[ok]] - x[j[ok]]), 1e-9)
        return out

    a, b, m, n = (np.asarray(v, np.int64) for v in (a, b, m, n))
    g = inv(a, m) - inv(b, m) - inv(a, n) + inv(b, n)
    with np.errstate(divide="ignore"):
        return np.where(np.abs(g) > 1e-12, 2 * np.pi / g, np.nan)


def pseudo_positions(prof: Profile) -> tuple[np.ndarray, np.ndarray]:
    """The point of each reading in the pseudosection: the middle of its electrodes along the line and
    a conventional pseudo-depth, 0.17 times the span of its electrodes (about the median depth of
    investigation of Edwards for the common arrays)."""
    x = prof.positions()
    pts = np.stack([prof.a, prof.b, prof.m, prof.n], axis=1)
    pos = np.where(pts >= 0, x[np.clip(pts, 0, None)], np.nan)
    mid = np.nanmean(pos, axis=1)
    span = np.nanmax(pos, axis=1) - np.nanmin(pos, axis=1)
    return mid, 0.17 * span


def array_kind(prof: Profile) -> str:
    """The array of the readings: dipole-dipole, wenner, schlumberger, pole-dipole, pole-pole, mixed."""
    x = prof.positions()
    kinds: dict[str, int] = {}
    for a, b, m, n in zip(prof.a, prof.b, prof.m, prof.n):
        if b < 0 and n < 0:
            k = "pole-pole"
        elif b < 0 or n < 0:
            k = "pole-dipole"
        else:
            ab = sorted((x[a], x[b]))
            mn = sorted((x[m], x[n]))
            if mn[1] <= ab[0] + 1e-9 or mn[0] >= ab[1] - 1e-9:
                k = "dipole-dipole"
            elif ab[0] <= mn[0] and mn[1] <= ab[1]:
                am, mn_, nb = mn[0] - ab[0], mn[1] - mn[0], ab[1] - mn[1]
                k = "wenner" if abs(am - mn_) < 1e-6 * max(1, mn_) + 1e-6 and abs(nb - mn_) < 1e-6 * max(1, mn_) + 1e-6 \
                    else "schlumberger"
            else:
                k = "other"
        kinds[k] = kinds.get(k, 0) + 1
    if not kinds:
        return "other"
    best = max(kinds, key=kinds.get)
    return best if kinds[best] >= 0.9 * sum(kinds.values()) else "mixed"


ARRAY_NAMES = {"dipole-dipole": ("dipolo-dipolo", "dipole-dipole"), "wenner": ("Wenner", "Wenner"),
               "schlumberger": ("Wenner-Schlumberger", "Wenner-Schlumberger"),
               "pole-dipole": ("polo-dipolo", "pole-dipole"), "pole-pole": ("polo-polo", "pole-pole"),
               "mixed": ("misto", "mixed"), "other": ("altro", "other")}


# ------------------------------------------------------------------ reading files
def _numbers(text: str) -> list[float]:
    out = []
    for tok in text.replace(",", " ").replace(";", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def _lines(path: Path) -> list[str]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace").splitlines()


def looks_unified(path: str | Path) -> bool:
    """The unified data format of pyGIMLi/BERT: the first line that is not a comment is the count of
    the electrodes, alone."""
    try:
        for line in _lines(Path(path))[:50]:
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            return bool(re.fullmatch(r"\d+", s))
    except OSError:
        return False
    return False


def read_unified(path: str | Path) -> Profile:
    """The unified data format (``.ohm``, ``.shm``, ``.data``, ``.dat``): electrodes with their token
    line (``# x z`` or ``# x y z``), readings with theirs (``# a b m n rhoa err ...``), electrodes
    numbered from 1 (0 = at infinity); ``err`` is relative, ``ip`` in mrad."""
    lines = _lines(Path(path))
    k = 0
    tokens: list[str] = []

    def next_line():
        nonlocal k, tokens
        while k < len(lines):
            line = lines[k]
            k += 1
            if "#" in line:
                body, comment = line.split("#", 1)
                words = comment.replace("/", " ").split()
                if not body.strip():
                    if words:
                        tokens = [w.lower() for w in words]
                    continue
                line = body
            if line.strip():
                return line
        return None

    first = next_line()
    if first is None or not re.fullmatch(r"\s*\d+\s*", first):
        raise NeedsGimli(f"{path}: not the unified data format")
    count = int(first)
    tokens = []
    sens = []
    for _ in range(count):
        line = next_line()
        if line is None:
            raise ValueError(f"{path}: {count} electrodes announced, fewer found")
        sens.append([float(v) for v in line.split()])
    stoks = tokens or (["x", "z"] if len(sens[0]) == 2 else ["x", "y", "z"])
    sens = np.asarray(sens, float)
    col = {name: i for i, name in enumerate(stoks)}
    x = sens[:, col.get("x", 0)]
    y = sens[:, col["y"]] if "y" in col else None
    z = sens[:, col["z"]] if "z" in col else None
    if z is None and y is not None and len(stoks) == 2:  # "x y" in 2-D is x and height
        z, y = y, None
    line = next_line()
    if line is None:
        raise ValueError(f"{path}: no readings")
    data_count = int(line.split()[0])
    tokens = []
    rows = []
    for _ in range(data_count):
        line = next_line()
        if line is None:
            break
        rows.append([float(v) for v in line.split()])
    if not rows:
        raise ValueError(f"{path}: no readings")
    dtoks = tokens or ["a", "b", "m", "n", "rhoa", "err"][: len(rows[0])]
    width = min(len(r) for r in rows)
    arr = np.asarray([r[:width] for r in rows], float)
    dcol = {name: i for i, name in enumerate(dtoks) if i < width}
    for need in ("a", "b", "m", "n"):
        if need not in dcol:
            raise ValueError(f"{path}: the readings have no column {need!r}")
    idx = {e: arr[:, dcol[e]].astype(np.int64) - 1 for e in ("a", "b", "m", "n")}
    values = {}
    for name in VALUES:
        if name in dcol:
            values[name] = arr[:, dcol[name]]
    if "valid" in dcol:
        keep = arr[:, dcol["valid"]] != 0
        idx = {e: v[keep] for e, v in idx.items()}
        values = {kk: v[keep] for kk, v in values.items()}
    world = y is not None and np.ptp(y) > 0
    along = _along(x, y) if world else x
    return Profile(Path(path).stem, along, idx["a"], idx["b"], idx["m"], idx["n"], values,
                   file_z=z, file_x=x if world else None, file_y=y if world else None, source=str(path),
                   fmt="unified")


def _along(x, y) -> np.ndarray:
    """The distance of each point from the first along the polyline of the points."""
    xy = np.column_stack([x, y]).astype(float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


ALIASES = {
    "a": ("a", "c1", "ca", "ea", "spa.1", "spa1", "elec_a", "electrode_a", "a_x", "xa", "x_a", "a(x)", "c1x"),
    "b": ("b", "c2", "cb", "eb", "spa.2", "spa2", "elec_b", "electrode_b", "b_x", "xb", "x_b", "b(x)", "c2x"),
    "m": ("m", "p1", "pm", "em", "spa.3", "spa3", "elec_m", "electrode_m", "m_x", "xm", "x_m", "m(x)", "p1x"),
    "n": ("n", "p2", "pn", "en", "spa.4", "spa4", "elec_n", "electrode_n", "n_x", "xn", "x_n", "n(x)", "p2x"),
    "rhoa": ("rhoa", "rho_a", "rho", "appres", "app.res", "app_res", "apparent_resistivity", "resistivity", "ra",
             "res", "rho(ohm.m)", "rhoa(ohmm)"),
    "r": ("r", "resistance", "r(ohm)", "rab"),
    "u": ("u", "v", "voltage", "potential", "u(v)", "v(v)"),
    "vp_mv": ("vp", "vp(mv)", "v(mv)", "u(mv)"),
    "i": ("i", "current", "i(a)"),
    "i_ma": ("in", "in(ma)", "i(ma)", "current(ma)"),
    "err": ("err", "error", "stdev", "std", "sd", "dev", "dev.", "q", "err(%)", "error(%)", "dev(%)", "rerr"),
    "ip": ("ip", "m", "chargeability"),
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name.strip().lower()).replace("[", "(").replace("]", ")")


def _columns(header: list[str]) -> dict[str, int]:
    """The column of each field of a table of readings (the first match of its names)."""
    out: dict[str, int] = {}
    names = [_norm(h) for h in header]
    for key, aliases in ALIASES.items():
        if key == "ip" and "m" in names and names.count("m") == 1:  # "m" is the electrode
            aliases = tuple(a for a in aliases if a != "m")
        for i, h in enumerate(names):
            if i in out.values():
                continue
            base = h.split("(")[0]
            if h in aliases or base in aliases or re.sub(r"[%.]", "", base) in aliases:
                out[key] = i
                break
    return out


def _table(path: Path) -> tuple[list[str], list[list[str]]]:
    text = "\n".join(line for line in _lines(path) if line.strip() and not line.lstrip().startswith(("#", "//")))
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delim = dialect.delimiter
    except csv.Error:
        delim = ";" if sample.count(";") > sample.count(",") else ("," if "," in sample else None)
    if delim == " " or delim is None:
        rows = [line.split() for line in text.splitlines()]
    else:
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)]
        if delim != "," and any("," in c for r in rows[1:6] for c in r):  # decimal commas
            rows = [rows[0]] + [[c.replace(",", ".") for c in r] for r in rows[1:]]
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: empty")
    return rows[0], rows[1:]


def _float(text: str) -> float:
    s = str(text).strip()
    if not s or s.lower() in ("nan", "na", "-", "inf", "*"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_table(path: str | Path, spacing: float | None = None, positions: bool | None = None) -> Profile:
    """A table of readings (CSV/TXT with a header): ``a b m n`` and ``rhoa`` (or ``r``, or ``u``/``v``
    and ``i``, or Syscal's ``Vp`` mV and ``In`` mA), ``err`` when present (a fraction; percent when
    its name says ``%``/``dev`` or its values pass 1). The electrodes: numbers from 1 (whole numbers,
    none 0; then ``spacing`` metres apart, 1 by default) or positions along the line in metres
    (``positions`` forces one reading or the other). An empty cell, NaN or a negative number for
    ``b`` or ``n``: an electrode at infinity."""
    path = Path(path)
    header, rows = _table(path)
    cols = _columns(header)
    missing = [k for k in ("a", "b", "m", "n") if k not in cols]
    if missing or not any(k in cols for k in ("rhoa", "r", "u", "vp_mv")):
        raise NeedsGimli(t("{0}: servono le colonne a, b, m, n e rhoa (o r, o u e i); trovate: {1}",
                           "{0}: the columns a, b, m, n and rhoa (or r, or u and i) are needed; found: {1}")
                         .format(path.name, ", ".join(header)))
    arr = np.array([[_float(r[i]) if i < len(r) else np.nan for i in range(len(header))] for r in rows], float)
    ok = np.isfinite(arr[:, cols["a"]]) & np.isfinite(arr[:, cols["m"]])
    arr = arr[ok]
    pos = {k: arr[:, cols[k]] for k in ("a", "b", "m", "n")}
    allpos = np.concatenate([v[np.isfinite(v)] for v in pos.values()])
    integral = bool(np.all(np.abs(allpos - np.round(allpos)) < 1e-9))
    as_numbers = (integral and allpos.min() >= 1) if positions is None else not positions
    notes = []
    if as_numbers:
        count = int(allpos.max())
        step = float(spacing or 1.0)
        file_u = np.arange(count) * step
        idx = {k: np.where(np.isfinite(v) & (v >= 1), np.round(np.nan_to_num(v, nan=0.0)).astype(np.int64) - 1, -1) for k, v in pos.items()}
        notes.append(t("elettrodi numerati, {0:g} m di distanza", "numbered electrodes, {0:g} m apart").format(step))
    else:
        has = [np.isfinite(v) & (v >= 0) if k in ("b", "n") else np.isfinite(v) for k, v in pos.items()]
        uniq = np.unique(np.round(np.concatenate([v[h] for v, h in zip(pos.values(), has)]), 6))
        file_u = uniq.astype(float)
        idx = {}
        for (k, v), h in zip(pos.items(), has):
            j = np.searchsorted(uniq, np.round(np.where(h, v, uniq[0]), 6))
            idx[k] = np.where(h, j, -1).astype(np.int64)
        if spacing:
            notes.append(t("posizioni lette in metri: la distanza scelta non conta",
                           "positions read in metres: the spacing given does not count"))
    values = {}
    if "rhoa" in cols:
        values["rhoa"] = arr[:, cols["rhoa"]]
    if "r" in cols:
        values["r"] = arr[:, cols["r"]]
    if "u" in cols:
        values["u"] = arr[:, cols["u"]]
    elif "vp_mv" in cols:
        values["u"] = arr[:, cols["vp_mv"]] / 1000.0
    if "i" in cols:
        values["i"] = arr[:, cols["i"]]
    elif "i_ma" in cols:
        values["i"] = arr[:, cols["i_ma"]] / 1000.0
    if "err" in cols:
        e = arr[:, cols["err"]]
        name = _norm(header[cols["err"]])
        if "%" in name or name.startswith("dev") or name == "q" or np.nanmax(np.abs(e)) > 1:
            e = e / 100.0
        values["err"] = np.abs(e)
    if "ip" in cols:
        values["ip"] = arr[:, cols["ip"]]
    if "rhoa" not in values and "r" not in values and "u" in values and "i" in values:
        with np.errstate(divide="ignore", invalid="ignore"):
            values["r"] = values["u"] / values["i"]
    prof = Profile(path.stem, file_u, idx["a"], idx["b"], idx["m"], idx["n"], values, source=str(path), fmt="table",
                   notes=notes)
    prof.numbered = bool(as_numbers)
    return prof


def read_electrodes(path: str | Path) -> dict:
    """A table of electrodes: ``x, z`` (or ``distance``/``u``) along the line, or world ``x, y`` and
    ``z`` if present; in the order of their numbers. Returns ``{"u", "z", "x", "y"}`` (None for what
    the file has not)."""
    header, rows = _table(Path(path))
    names = [_norm(h).split("(")[0] for h in header]
    arr = np.array([[_float(c) for c in r[: len(header)]] for r in rows], float)

    def col(*keys):
        for k in keys:
            if k in names:
                return arr[:, names.index(k)]
        return None

    x, y, z = col("x", "east", "e", "easting"), col("y", "north", "easting_y", "northing"), col("z", "h", "height",
                                                                                                "elevation", "quota")
    u = col("u", "distance", "dist", "pos", "position", "d", "s")
    if u is None and x is not None and y is None:
        u, x = x, None
    if u is None and x is not None and y is not None:
        u = _along(x, y)
    if u is None:
        nums = [_numbers(" ".join(r)) for r in rows]
        if nums and all(len(v) >= 2 for v in nums):
            u, z = np.array([v[0] for v in nums]), np.array([v[1] for v in nums])
    if u is None:
        raise ValueError(t("{0}: nessuna colonna di posizione (x, distance, u)", "{0}: no column of position "
                                                                                 "(x, distance, u)").format(Path(path).name))
    return {"u": np.asarray(u, float), "z": None if z is None else np.asarray(z, float),
            "x": None if x is None or y is None else np.asarray(x, float),
            "y": None if x is None or y is None else np.asarray(y, float)}


def with_electrodes(prof: Profile, el: dict) -> Profile:
    """The electrodes of a table (:func:`read_electrodes`) for ``prof``: with numbered electrodes the
    rows of the table in the order of their numbers; with positions read in metres, the row at the
    same distance (within a tenth of the spacing) of each electrode."""
    u_el = np.asarray(el["u"], float)
    if getattr(prof, "numbered", False):
        if len(u_el) < prof.electrodes:
            raise ValueError(t("Il file degli elettrodi ne ha {0}, le letture ne usano {1}.",
                               "The file of the electrodes has {0}, the readings use {1}.")
                             .format(len(u_el), prof.electrodes))
        rows = np.arange(len(u_el))
    else:
        rows = np.array([int(np.argmin(np.abs(u_el - u))) for u in prof.file_u])
        far = np.abs(u_el[rows] - prof.file_u) > 0.1 * prof.spacing
        if far.any():
            raise ValueError(t("{0} elettrodi delle letture non sono nel file degli elettrodi (posizioni diverse).",
                               "{0} electrodes of the readings are not in the file of the electrodes (other "
                               "positions).").format(int(far.sum())))
    prof.file_u = u_el[rows]
    if el.get("z") is not None:
        prof.file_z = np.asarray(el["z"], float)[rows]
    if el.get("x") is not None:
        prof.file_x, prof.file_y = np.asarray(el["x"], float)[rows], np.asarray(el["y"], float)[rows]
    src = Path(el["source"]).name if el.get("source") else "-"
    prof.notes.append(t("elettrodi dal file {0}", "electrodes from the file {0}").format(src))
    return prof


def read_any(path: str | Path, spacing: float | None = None, positions: bool | None = None,
             electrodes: str | Path | None = None) -> Profile:
    """A profile from a file this module reads (:class:`NeedsGimli` for the others)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if looks_unified(path):
        prof = read_unified(path)
    else:
        prof = read_table(path, spacing, positions)
    if electrodes:
        el = read_electrodes(electrodes)
        el["source"] = str(electrodes)
        with_electrodes(prof, el)
    return prof


def from_arrays(name: str, arrays: dict, source: str = "", fmt: str = "pygimli") -> Profile:
    """A profile from the arrays of pyGIMLi's reader (worker ``read`` task: ``sensors`` (N, 2) distance
    and height, ``a b m n``, the values)."""
    sensors = np.asarray(arrays["sensors"], float)
    values = {k: np.asarray(arrays[k], float) for k in VALUES if k in arrays}
    if "err" in values and np.nanmax(np.abs(values["err"])) > 1:
        values["err"] = values["err"] / 100.0
    z = sensors[:, 1] if sensors.shape[1] > 1 and np.ptp(sensors[:, 1]) > 0 else None
    return Profile(name, sensors[:, 0].copy(), *(np.asarray(arrays[k], np.int64) for k in ("a", "b", "m", "n")),
                   values=values, file_z=z, source=source, fmt=fmt)


# ------------------------------------------------------------------ the project
def registry(chunk) -> dict:
    return chunk.meta.setdefault("geophysics", {"grids": []})


def entries(chunk) -> list[dict]:
    return list((chunk.meta.get("geophysics") or {}).get("ert") or [])


def entry_of(chunk, pid: str) -> dict:
    for e in entries(chunk):
        if e["id"] == pid or e["name"] == pid:
            return e
    raise KeyError(t("nessun profilo ERT «{0}»", "no ERT profile {0!r}").format(pid))


def folder(chunk, pid: str) -> Path:
    return Path(chunk.data_dir) / FOLDER / pid


def _free_name(chunk, name: str, own: str = "") -> str:
    taken = {e["name"] for e in entries(chunk) if e["id"] != own}
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def save(chunk, prof: Profile) -> dict:
    """The profile into the chunk: its arrays in ``profile.npz`` and its record (a new id when it has
    none; the record keeps its placement, filters and inversion)."""
    reg = registry(chunk)
    lst = reg.setdefault("ert", [])
    if not prof.id:
        prof.id = datetime.now().strftime("e%Y%m%d%H%M%S%f")[:20]
        prof.name = _free_name(chunk, prof.name)
    out = folder(chunk, prof.id)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "profile.npz", **prof.arrays())
    old = next((e for e in lst if e["id"] == prof.id), None)
    entry = dict(old or {})
    entry.update({"id": prof.id, "name": prof.name, "source": prof.source, "format": prof.fmt,
                  "electrodes": prof.electrodes, "readings": prof.count, "array": array_kind(prof),
                  "spacing": round(prof.spacing, 6), "world": prof.world, "notes": list(prof.notes),
                  "numbered": bool(prof.numbered),
                  "values": [k for k in VALUES if prof.has(k)],
                  "file": str((out / "profile.npz").relative_to(chunk.data_dir)),
                  "updated": datetime.now().isoformat(timespec="seconds")})
    entry.setdefault("created", entry["updated"])
    entry.setdefault("filters", {"rules": dict(DEFAULT_RULES), "manual": []})
    if old is None:
        lst.append(entry)
    else:
        lst[lst.index(old)] = entry
    entry["valid"] = int(valid_mask(prof, entry["filters"]).sum())
    json.dumps(entry)
    return entry


def load(chunk, pid: str) -> Profile:
    e = entry_of(chunk, pid)
    with np.load(Path(chunk.data_dir) / e["file"], allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    prof = Profile(e["name"], arr["file_u"], arr["a"].astype(np.int64), arr["b"].astype(np.int64),
                   arr["m"].astype(np.int64), arr["n"].astype(np.int64),
                   {k: arr[k] for k in VALUES if k in arr}, arr.get("file_z"), arr.get("file_x"), arr.get("file_y"),
                   e["id"], e.get("source", ""), e.get("format", ""), list(e.get("notes") or []),
                   {k: arr[k] for k in ("elec_u", "elec_x", "elec_y", "elec_z") if k in arr},
                   bool(e.get("numbered", True)))
    return prof


def remove(chunk, pid: str) -> None:
    import shutil

    e = entry_of(chunk, pid)
    reg = registry(chunk)
    reg["ert"] = [x for x in reg.get("ert", []) if x["id"] != e["id"]]
    shutil.rmtree(folder(chunk, e["id"]), ignore_errors=True)


def rename(chunk, pid: str, name: str) -> dict:
    e = entry_of(chunk, pid)
    e["name"] = _free_name(chunk, name.strip() or e["name"], e["id"])
    return e


# ------------------------------------------------------------------ placing the line
def line_length(line) -> float:
    xy = np.asarray(line, float)[:, :2]
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def line_point(line, u) -> np.ndarray:
    """The points of the polyline ``line`` at the distances ``u`` from its start (beyond its ends, on
    the prolongation of the first or last segment)."""
    xy = np.asarray(line, float)[:, :2]
    seg = np.diff(xy, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    keep = lens > 1e-12
    xy = np.vstack([xy[:1], xy[1:][keep]])
    seg, lens = seg[keep], lens[keep]
    if not len(lens):
        raise ValueError(t("la linea ha lunghezza zero", "the line has zero length"))
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    u = np.atleast_1d(np.asarray(u, float))
    k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(lens) - 1)
    f = (u - cum[k]) / lens[k]
    return xy[k] + seg[k] * f[:, None]


def line_direction(line, u) -> np.ndarray:
    """The unit direction of the polyline at the distances ``u``."""
    xy = np.asarray(line, float)[:, :2]
    seg = np.diff(xy, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    keep = lens > 1e-12
    seg, lens = seg[keep], lens[keep]
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    u = np.atleast_1d(np.asarray(u, float))
    k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(lens) - 1)
    return seg[k] / lens[k][:, None]


def sample_dem(path, xy) -> np.ndarray:
    """The DEM at the points ``xy`` (bilinear; NaN outside or on no data)."""
    import rasterio

    xy = np.asarray(xy, float)
    with rasterio.open(path) as src:
        inv = ~src.transform
        col, row = inv @ (xy[:, 0], xy[:, 1])
        col, row = np.asarray(col) - 0.5, np.asarray(row) - 0.5
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        out = np.full(len(xy), np.nan)
        lo_c, lo_r = max(int(c0.min()), 0), max(int(r0.min()), 0)
        hi_c, hi_r = min(int(c0.max()) + 2, src.width), min(int(r0.max()) + 2, src.height)
        if hi_c <= lo_c or hi_r <= lo_r:
            return out
        from rasterio.windows import Window

        win = src.read(1, window=Window(lo_c, lo_r, hi_c - lo_c, hi_r - lo_r), masked=True)
        z = win.astype(float).filled(np.nan)
        nod = src.nodata
        if nod is not None:
            z[z == nod] = np.nan
    fc, fr = col - c0, row - r0
    for i in range(len(xy)):
        cc, rr = c0[i] - lo_c, r0[i] - lo_r
        if cc < -1 or rr < -1 or cc >= z.shape[1] or rr >= z.shape[0]:
            continue
        acc, wsum = 0.0, 0.0
        for dr, dc, w in ((0, 0, (1 - fr[i]) * (1 - fc[i])), (0, 1, (1 - fr[i]) * fc[i]),
                          (1, 0, fr[i] * (1 - fc[i])), (1, 1, fr[i] * fc[i])):
            r, c = rr + dr, cc + dc
            if 0 <= r < z.shape[0] and 0 <= c < z.shape[1] and np.isfinite(z[r, c]) and w > 0:
                acc += w * z[r, c]
                wsum += w
        if wsum > 0.25:
            out[i] = acc / wsum
    return out


def place(chunk, pid: str, line=None, first: float = 0.0, spacing: float | None = None, heights: str = "auto",
          z0: float | None = None, method: str = "") -> dict:
    """The line of the profile in the survey: ``line`` (two or more points, world X Y of the chunk, the
    first electrode side first) - None when the file gave the electrodes in world coordinates; the
    first electrode ``first`` metres from the start of the line; the electrodes ``spacing`` metres
    apart (None: as read); the heights from the DEM of the chunk (``heights`` "dem"), the file
    ("file", plus ``z0``), flat at ``z0`` ("flat"), or "auto" (the DEM when the chunk has one, else
    the file, else flat). The inversion done before is to be done again."""
    prof = load(chunk, pid)
    e = entry_of(chunk, pid)
    notes = []
    u_file = np.asarray(prof.file_u, float)
    if prof.world and line is None:
        xy = np.column_stack([prof.file_x, prof.file_y])
        line = xy
        u = _along(prof.file_x, prof.file_y)
        method = method or "file"
    else:
        if line is None or len(line) < 2:
            raise ValueError(t("Servono almeno due punti per la linea.", "At least two points are needed for the line."))
        line = np.asarray(line, float)[:, :2]
        scale = 1.0 if not spacing else float(spacing) / prof.spacing
        u = float(first) + (u_file - u_file.min()) * scale
        xy = line_point(line, u)
        if u.max() > line_length(line) + 0.5 * prof.spacing * scale:
            notes.append(t("gli elettrodi vanno oltre la fine della linea di {0:.2f} m: proseguono sulla sua direzione",
                           "the electrodes go past the end of the line by {0:.2f} m: they follow its direction")
                         .format(u.max() - line_length(line)))
    dem = getattr(chunk, "dem_path", None)
    dem = Path(dem) if dem is not None and Path(dem).exists() else None
    mode = heights
    if mode == "auto":
        mode = "dem" if dem is not None else ("file" if prof.file_z is not None else "flat")
    if mode == "dem":
        if dem is None:
            raise ValueError(t("Il chunk non ha un DEM: scegli le quote del file o una quota fissa.",
                               "The chunk has no DEM: choose the heights of the file or a fixed height."))
        z = sample_dem(dem, xy)
        if not np.isfinite(z).any():
            raise ValueError(t("Gli elettrodi cadono fuori dal DEM: controlla la linea.",
                               "The electrodes fall outside the DEM: check the line."))
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = np.interp(u[bad], u[~bad], z[~bad])
            notes.append(t("{0} elettrodi fuori dal DEM: quote interpolate", "{0} electrodes outside the DEM: heights "
                                                                             "interpolated").format(int(bad.sum())))
    elif mode == "file":
        if prof.file_z is None:
            raise ValueError(t("Il file non ha le quote degli elettrodi.", "The file has no heights of the electrodes."))
        z = np.asarray(prof.file_z, float) + float(z0 or 0.0)
    else:
        z = np.full(len(u), float(z0 or 0.0))
    prof.elec = {"elec_u": u, "elec_x": xy[:, 0], "elec_y": xy[:, 1], "elec_z": z}
    save(chunk, prof)
    e = entry_of(chunk, pid)
    e["placement"] = {"method": method or "points", "line": np.asarray(line, float).round(4).tolist(),
                      "first": float(first), "spacing": float(spacing) if spacing else round(prof.spacing, 6),
                      "heights": mode, "z0": z0, "length": round(float(u.max() - u.min()), 3),
                      "z_range": [round(float(np.min(z)), 3), round(float(np.max(z)), 3)], "notes": notes,
                      "crs": getattr(chunk, "crs", "LOCAL")}
    if e.get("inversion"):
        e["inversion"]["stale"] = True
    return e


# ------------------------------------------------------------------ the filters
def reciprocal_errors(prof: Profile) -> np.ndarray:
    """The reciprocal error of each reading (|R - R'| / mean, R' the reading with the current and
    potential electrodes swapped), NaN without a reciprocal."""
    r = prof.value("r")
    if not np.isfinite(r).any():
        k = prof.value("k")
        if not np.isfinite(k).any():
            k = flat_k(prof.positions(), prof.a, prof.b, prof.m, prof.n)
        r = prof.apparent() / k
    key = {}
    for i, (a, b, m, n) in enumerate(zip(prof.a, prof.b, prof.m, prof.n)):
        key.setdefault((min(a, b), max(a, b), min(m, n), max(m, n)), []).append(i)
    out = np.full(prof.count, np.nan)
    for (a, b, m, n), idx in key.items():
        rec = key.get((m, n, a, b))
        if not rec or (a, b) == (m, n):
            continue
        ra, rb = np.mean(np.abs(r[idx])), np.mean(np.abs(r[rec]))
        if ra + rb > 0:
            out[idx] = abs(ra - rb) / (0.5 * (ra + rb))
    return out


def spike_scores(prof: Profile, valid: np.ndarray | None = None) -> np.ndarray:
    """How far each reading is from its neighbours in the pseudosection: |log10 rhoa - median of the
    neighbours| in robust deviations of the whole (NaN for readings without a value)."""
    rhoa = prof.apparent()
    ok = np.isfinite(rhoa) & (rhoa > 0) & (valid if valid is not None else True)
    x, d = pseudo_positions(prof)
    s = prof.spacing or 1.0
    pts = np.column_stack([x / s, d / s])
    logr = np.where(ok, np.log10(np.where(ok, rhoa, 1.0)), np.nan)
    idx = np.flatnonzero(ok)
    out = np.full(prof.count, np.nan)
    if idx.size < SPIKE_NEIGHBOURS + 2:
        return out
    from scipy.spatial import cKDTree

    tree = cKDTree(pts[idx])
    _dist, nb = tree.query(pts[idx], k=SPIKE_NEIGHBOURS + 1)
    med = np.median(logr[idx][nb[:, 1:]], axis=1)
    dev = logr[idx] - med
    mad = 1.4826 * np.median(np.abs(dev - np.median(dev))) or 1e-6
    out[idx] = np.abs(dev) / mad
    return out


def reasons(prof: Profile, filters: dict) -> dict[str, np.ndarray]:
    """For each rule of ``filters`` (``rules`` and ``manual``), the readings it removes."""
    rules = {**DEFAULT_RULES, **(filters.get("rules") or {})}
    rhoa = prof.apparent()
    out = {"invalid": ~(np.isfinite(rhoa) & (rhoa > 0))}
    if rules.get("min_rhoa") is not None:
        out["min_rhoa"] = np.isfinite(rhoa) & (rhoa < float(rules["min_rhoa"]))
    if rules.get("max_rhoa") is not None:
        out["max_rhoa"] = np.isfinite(rhoa) & (rhoa > float(rules["max_rhoa"]))
    if rules.get("max_error") is not None and prof.has("err"):
        out["max_error"] = np.nan_to_num(prof.value("err"), nan=0.0) > float(rules["max_error"]) / 100.0
    if rules.get("reciprocal") is not None:
        rec = reciprocal_errors(prof)
        out["reciprocal"] = np.nan_to_num(rec, nan=0.0) > float(rules["reciprocal"]) / 100.0
    manual = np.zeros(prof.count, bool)
    for i in filters.get("manual") or []:
        if 0 <= int(i) < prof.count:
            manual[int(i)] = True
    out["manual"] = manual
    if rules.get("spikes") is not None:
        before = ~np.any(np.stack(list(out.values())), axis=0)
        out["spikes"] = np.nan_to_num(spike_scores(prof, before), nan=0.0) > float(rules["spikes"])
    return out


def valid_mask(prof: Profile, filters: dict | None) -> np.ndarray:
    r = reasons(prof, filters or {})
    return ~np.any(np.stack(list(r.values())), axis=0)


def set_filters(chunk, pid: str, rules: dict | None = None, remove_readings=None, restore=None,
                clear_manual: bool = False) -> dict:
    """The filters of the profile changed: ``rules`` (``min_rhoa``, ``max_rhoa`` ohm m, ``max_error`` %,
    ``reciprocal`` %, ``spikes`` robust deviations; None switches a rule off), readings removed by
    hand (indices) or given back. Returns the record with the counts of each rule."""
    e = entry_of(chunk, pid)
    prof = load(chunk, pid)
    f = e.setdefault("filters", {"rules": dict(DEFAULT_RULES), "manual": []})
    if rules is not None:
        f["rules"] = {**DEFAULT_RULES, **{k: (None if v is None else float(v)) for k, v in rules.items()
                                          if k in DEFAULT_RULES}}
    manual = set() if clear_manual else {int(i) for i in f.get("manual") or []}
    manual |= {int(i) for i in (remove_readings or []) if 0 <= int(i) < prof.count}
    manual -= {int(i) for i in (restore or [])}
    f["manual"] = sorted(manual)
    r = reasons(prof, f)
    f["counts"] = {k: int(v.sum()) for k, v in r.items()}
    e["valid"] = int((~np.any(np.stack(list(r.values())), axis=0)).sum())
    if e.get("inversion"):
        e["inversion"]["stale"] = True
    return e


def filter_text(e: dict) -> str:
    counts = (e.get("filters") or {}).get("counts") or {}
    names = {"invalid": ("valori nulli o negativi", "zero or negative values"), "min_rhoa": ("sotto il minimo", "below "
             "the minimum"), "max_rhoa": ("sopra il massimo", "above the maximum"), "max_error": ("errore alto",
             "high error"), "reciprocal": ("reciproci discordanti", "reciprocals that disagree"),
             "spikes": ("picchi", "spikes"), "manual": ("tolte a mano", "removed by hand")}
    parts = [f"{t(*names.get(k, (k, k)))} {v}" for k, v in counts.items() if v]
    return ", ".join(parts)
