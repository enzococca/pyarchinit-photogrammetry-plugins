"""Depth logs and the time of every photo.

A depth log is a series of ``(time, depth)``: the pressure sensor of an ROV, a dive computer, the
depth file of a dataset. Read here:

* CSV and text tables (``,`` ``;`` tab or spaces), with or without a header: a time column - Unix
  time in s, ms, us or ns (told by the header, ``[ns]``, or by the size of the numbers), ISO dates,
  ``hh:mm:ss`` clock times with a date column, seconds or ``mm:ss`` from the start - and a depth
  column in metres or feet, or a pressure column (bar, mbar/hPa, dbar, kPa, Pa, psi) turned into
  depth with the density of the water; the depth file of the AQUALOC dataset
  (``#timestamp [ns], depth [m]``) is one of them;
* UDDF (Universal Dive Data Format, the XML that dive logs such as Subsurface export): the
  waypoints of every dive, ``divetime`` from the start of the dive plus its ``datetime``.

Times are seconds: since 1970-01-01 when the log gives dates (a date without a time zone is read
as UTC, as the times of the photos are), else from the start of the log (``absolute`` False).

The time of a photo comes, in this order, from the plugin's own record (the frames it extracted
from a video or a sequence with their times), from its file name (Unix time in s/ms/us/ns, or a date
and time such as ``20190312_142355_123``), from its EXIF (``DateTimeOriginal`` with
``SubSecTimeOriginal`` and ``OffsetTimeOriginal``). ``offset`` is what to add to the clock of the
camera to read the clock of the log (a camera an hour ahead: ``-3600``).
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

G = 9.80665
SEA_WATER = 1025.0  # kg/m3
FRESH_WATER = 1000.0
SURFACE_HPA = 1013.25  # pressure at the surface when the log gives an absolute pressure

PRESSURE_UNITS = {"pa": 1.0, "hpa": 100.0, "mbar": 100.0, "mb": 100.0, "kpa": 1000.0, "dbar": 1e4,
                  "bar": 1e5, "psi": 6894.757, "atm": 101325.0}
TIME_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0}


@dataclass
class DepthLog:
    """Depths (metres, positive down) at times (seconds, see the module)."""

    times: np.ndarray
    depths: np.ndarray
    absolute: bool
    source: str = ""
    kind: str = "depth"  # or "pressure": the depths were computed from pressures
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        t = np.asarray(self.times, float)
        d = np.asarray(self.depths, float)
        ok = np.isfinite(t) & np.isfinite(d)
        order = np.argsort(t[ok], kind="stable")
        self.times, self.depths = t[ok][order], d[ok][order]

    def __len__(self) -> int:
        return len(self.times)

    @property
    def span(self) -> tuple[float, float]:
        return (float(self.times[0]), float(self.times[-1])) if len(self) else (math.nan, math.nan)

    def smoothed(self, window: float = 1.0) -> np.ndarray:
        """The depths with a running median over ``window`` seconds (the pressure sensor of an ROV
        is noisy at a few centimetres; a median keeps the steps of a descent)."""
        if window <= 0 or len(self) < 3:
            return self.depths.copy()
        dt = float(np.median(np.diff(self.times))) if len(self) > 1 else 1.0
        half = max(1, int(round(window / max(dt, 1e-9) / 2)))
        if half < 1:
            return self.depths.copy()
        from scipy.ndimage import median_filter

        return median_filter(self.depths, size=2 * half + 1, mode="nearest")

    def depth_at(self, times, window: float = 1.0, tolerance: float = 2.0) -> np.ndarray:
        """Depths at ``times`` (seconds on the clock of the log), linearly between the smoothed
        samples; NaN farther than ``tolerance`` seconds from the log or from its nearest sample."""
        times = np.atleast_1d(np.asarray(times, float))
        out = np.full(times.shape, np.nan)
        if not len(self):
            return out
        d = self.smoothed(window)
        inside = (times >= self.times[0] - tolerance) & (times <= self.times[-1] + tolerance)
        if len(self) == 1:
            out[inside] = d[0]
            return out
        i = np.clip(np.searchsorted(self.times, times), 1, len(self) - 1)
        gap = np.minimum(np.abs(times - self.times[i - 1]), np.abs(self.times[i] - times))
        ok = inside & (gap <= max(tolerance, 0.0) + 0.5 * (self.times[i] - self.times[i - 1]))
        out[ok] = np.interp(times[ok], self.times, d)
        return out

    def summary(self) -> dict:
        return {"source": self.source, "samples": int(len(self)), "absolute": bool(self.absolute),
                "start": float(self.times[0]) if len(self) else None,
                "end": float(self.times[-1]) if len(self) else None,
                "min_depth": float(np.min(self.depths)) if len(self) else None,
                "max_depth": float(np.max(self.depths)) if len(self) else None, "kind": self.kind, **self.meta}


# ------------------------------------------------------------------ times
def _utc(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*"
                  r"(Z|[+-]\d{2}:?\d{2})?\s*$")
_DMY = re.compile(r"^\s*(\d{1,2})[/.](\d{1,2})[/.](\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*$")
_CLOCK = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2}(?:[.,]\d+)?)\s*(?:min)?\s*$")
_DATE = re.compile(r"^\s*(?:(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})[/.](\d{1,2})[/.](\d{4}))\s*$")


def _tz(text: str | None) -> timezone | None:
    if not text:
        return None
    if text == "Z":
        return timezone.utc
    sign = -1 if text[0] == "-" else 1
    digits = text[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0)))


def _fraction(text: str | None) -> float:
    return float("0." + text) if text else 0.0


def parse_datetime(text: str) -> float | None:
    """Seconds since 1970 of an ISO (``2019-03-12T14:23:55.12Z``) or day/month/year date and time;
    None when ``text`` is not one."""
    m = _ISO.match(text)
    if m:
        y, mo, d, h, mi, s, frac, tz = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0), tzinfo=_tz(tz))
        return _utc(dt) + _fraction(frac)
    m = _DMY.match(text)
    if m:
        d, mo, y, h, mi, s, frac = m.groups()
        return _utc(datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))) + _fraction(frac)
    return None


def parse_clock(text: str) -> float | None:
    """Seconds of ``hh:mm:ss(.f)`` or ``mm:ss`` (``12:30 min``); None otherwise."""
    m = _CLOCK.match(text)
    if not m:
        return None
    h, mi, s = m.groups()
    return (int(h) * 3600 if h else 0) + int(mi) * 60 + float(s.replace(",", "."))


def parse_date(text: str) -> float | None:
    m = _DATE.match(text)
    if not m:
        return None
    y, mo, d, d2, mo2, y2 = m.groups()
    if y:
        return _utc(datetime(int(y), int(mo), int(d)))
    return _utc(datetime(int(y2), int(mo2), int(d2)))


def epoch_from_number(value: float, unit: str | None = None) -> tuple[float, bool]:
    """``(seconds, absolute)`` of a number of a time column: the unit of the header when given,
    else told by its size (Unix time in ns, us, ms or s; smaller numbers are seconds from a start)."""
    if unit in TIME_UNITS:
        sec = value * TIME_UNITS[unit]
        return sec, sec > 1e8
    a = abs(value)
    if a > 1e17:
        return value * 1e-9, True
    if a > 1e14:
        return value * 1e-6, True
    if a > 1e11:
        return value * 1e-3, True
    if a > 1e8:
        return value, True
    return value, False


# ------------------------------------------------------------------ tables
def _unit(name: str) -> str | None:
    m = re.search(r"[\[(]\s*([^\])]+?)\s*[\])]", name)
    if m:
        return m.group(1).strip().lower().replace("μ", "µ")
    m = re.search(r"[_ ](ns|us|ms|s|sec|min|m|ft|bar|mbar|hpa|kpa|pa|dbar|psi)\s*$", name.strip().lower())
    return m.group(1) if m else None


def _bare(name: str) -> str:
    name = re.sub(r"[\[(].*?[\])]", "", name.lower())
    name = name.replace("à", "a").replace("#", "").strip()
    return re.sub(r"[_\s]+", " ", name).strip()


def _number(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None


def _is_value(text: str) -> bool:
    """A cell of data, not of a header: a number, a date, a clock time."""
    text = text.strip()
    return (_number(text) is not None or parse_datetime(text) is not None or parse_clock(text) is not None
            or parse_date(text) is not None)


def _rows(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sample = "\n".join(lines[:20])
    delimiter = None
    for cand in (";", "\t", ","):
        counts = [ln.count(cand) for ln in lines[:20] if not ln.lstrip().startswith("#") or cand in ln]
        if counts and min(counts) > 0 and len(set(counts)) <= 2:
            delimiter = cand
            break
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
        except csv.Error:
            delimiter = None
    if delimiter in (None, " "):
        return [ln.split() for ln in lines]
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)]


ELAPSED_WORDS = ("sample time", "divetime", "dive time", "elapsed", "runtime", "run time")
TIME_WORDS = ("timestamp", "datetime", "date time", "unix", "epoch") + ELAPSED_WORDS + (
    "time", "ora", "tempo", "seconds", "secondi", "t")
DEPTH_WORDS = ("depth", "profondita", "prof", "sample depth", "z", "d")
PRESSURE_WORDS = ("pressure", "pressione", "press", "p")


def _find(names: list[str], words, exclude=()) -> int | None:
    bare = [_bare(n) for n in names]
    for word in words:
        for i, b in enumerate(bare):
            if i in exclude:
                continue
            if b == word or (len(word) > 2 and word in b.split(" ")) or (len(word) > 3 and word in b):
                return i
    return None


def read_table(path, density: float = SEA_WATER, surface: float | None = None, time_column=None,
               depth_column=None, pressure_unit: str | None = None) -> DepthLog:
    """A depth log from a CSV or text table (see the module). ``surface``: the pressure at the
    surface in hPa for a pressure column (default 1013.25, or ``"first"``-like 0 to use gauge
    pressure: give 0 for a sensor that already reads 0 at the surface). ``time_column`` and
    ``depth_column`` (names or 0-based indices) override the columns found."""
    path = Path(path)
    rows = _rows(path.read_text(encoding="utf-8-sig", errors="replace"))
    if not rows:
        raise ValueError(f"{path.name}: empty")
    header = None
    first = rows[0]
    if any(c.strip() and not _is_value(c.lstrip("#")) for c in first):
        header = [c.lstrip("#").strip() for c in first]
        rows = rows[1:]
    rows = [r for r in rows if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path.name}: no rows")
    width = max(len(r) for r in rows)
    names = header or [f"col{i}" for i in range(width)]

    def index(given, words, exclude=()):
        if given is None:
            return _find(names, words, exclude) if header else None
        if isinstance(given, int) or str(given).isdigit():
            return int(given)
        k = _find(names, (_bare(str(given)),))
        if k is None:
            raise ValueError(f"{path.name}: no column {given!r} in {names}")
        return k

    ti = index(time_column, TIME_WORDS)
    di = index(depth_column, DEPTH_WORDS, exclude={ti} if ti is not None else ())
    pi = None
    if di is None and depth_column is None:
        pi = index(None, PRESSURE_WORDS, exclude={ti} if ti is not None else ())
    if header is None:
        ti = 0 if ti is None else ti
        di = 1 if di is None else di
    if ti is None:
        raise ValueError(t_("{0}: nessuna colonna del tempo fra {1}", "{0}: no time column among {1}").format(path.name, names))
    if di is None and pi is None:
        raise ValueError(t_("{0}: nessuna colonna di profondità o pressione fra {1}",
                            "{0}: no depth or pressure column among {1}").format(path.name, names))
    # a date in a column of its own (Subsurface: date, time, sample time, sample depth)
    date_i = clock_i = None
    if header:
        for i, n in enumerate(names):
            if i != ti and _bare(n) in ("date", "data", "giorno"):
                date_i = i
        if _bare(names[ti]) in ELAPSED_WORDS or any(w in _bare(names[ti]) for w in ELAPSED_WORDS):
            # elapsed time of a dive whose start is in other columns (Subsurface: date, time, sample time)
            clock_i = next((i for i, n in enumerate(names) if i not in (ti, date_i) and _bare(n) in ("time", "ora")),
                           None)
    t_unit = _unit(names[ti]) if header else None
    v_i = di if di is not None else pi
    v_unit = _unit(names[v_i]) if header else None
    times, values = [], []
    absolute = None
    for r in rows:
        if len(r) <= max(ti, v_i):
            continue
        v = _number(r[v_i])
        if v is None:
            continue
        cell = r[ti]
        num = _number(cell)
        sec = None
        if num is not None and not re.search(r"[:/-]\d", cell.strip()[1:]):
            sec, absolute_here = epoch_from_number(num, t_unit)
        else:
            sec = parse_datetime(cell)
            absolute_here = sec is not None
            if sec is None:
                clock = parse_clock(cell)
                if clock is None:
                    continue
                if date_i is not None and clock_i is None and len(r) > date_i and parse_date(r[date_i]) is not None:
                    sec, absolute_here = parse_date(r[date_i]) + clock, True
                else:
                    sec, absolute_here = clock, False
        if not absolute_here and clock_i is not None and date_i is not None and len(r) > max(date_i, clock_i):
            day, start = parse_date(r[date_i]), parse_clock(r[clock_i])
            if day is not None and start is not None:
                sec, absolute_here = day + start + sec, True
        absolute = absolute_here if absolute is None else (absolute and absolute_here)
        times.append(sec)
        values.append(v)
    if not times:
        raise ValueError(t_("{0}: nessuna riga leggibile", "{0}: no readable row").format(path.name))
    values = np.asarray(values, float)
    meta = {"columns": [names[ti], names[v_i]]}
    kind = "depth"
    if di is not None:
        if v_unit in ("ft", "feet", "piedi"):
            values = values * 0.3048
        if np.nanmax(values) <= 0 < np.nanmax(np.abs(values)):
            values = -values  # a log of heights (z = -depth)
            meta["sign"] = "negative values read as depths"
    else:
        kind = "pressure"
        unit = (pressure_unit or v_unit or "").lower()
        if unit not in PRESSURE_UNITS:
            med = float(np.nanmedian(values))
            unit = "pa" if med > 20000 else ("mbar" if med > 200 else "bar")
        pa = values * PRESSURE_UNITS[unit]
        p0 = (SURFACE_HPA if surface is None else float(surface)) * 100.0
        values = (pa - p0) / (density * G)
        meta.update(pressure_unit=unit, surface_hpa=p0 / 100.0, density=density)
    return DepthLog(np.asarray(times, float), values, bool(absolute), source=path.name, kind=kind, meta=meta)


def t_(it: str, en: str) -> str:
    from .texts import t

    return t(it, en)


# ------------------------------------------------------------------ UDDF
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def read_uddf(path) -> DepthLog:
    """The waypoints of the dives of a UDDF file (``divetime`` seconds after the ``datetime`` of
    the dive; ``depth`` metres). Several dives are put one after the other on the same clock."""
    import xml.etree.ElementTree as ET

    path = Path(path)
    root = ET.parse(path).getroot()
    times, depths = [], []
    absolute = True
    dives = [el for el in root.iter() if _local(el.tag) == "dive"]
    for n, dive in enumerate(dives):
        start = None
        for el in dive.iter():
            if _local(el.tag) == "datetime" and el.text:
                start = parse_datetime(el.text.strip())
                break
        if start is None:
            absolute = False
            start = (times[-1] + 60.0) if times else 0.0
        for wp in dive.iter():
            if _local(wp.tag) != "waypoint":
                continue
            dt = d = None
            for el in wp:
                name = _local(el.tag)
                if name == "divetime" and el.text:
                    dt = _number(el.text)
                elif name == "depth" and el.text:
                    d = _number(el.text)
            if dt is not None and d is not None:
                times.append(start + dt)
                depths.append(d)
    if not times:
        raise ValueError(t_("{0}: nessun punto di immersione (waypoint con divetime e depth)",
                            "{0}: no dive waypoint (with divetime and depth)").format(path.name))
    return DepthLog(np.asarray(times), np.asarray(depths), absolute, source=path.name,
                    meta={"dives": len(dives)})


def read_log(path, **kwargs) -> DepthLog:
    """A depth log from a UDDF file or a table, by its content."""
    path = Path(path)
    head = path.read_bytes()[:400].lstrip()
    if path.suffix.lower() in (".uddf", ".xml") or head.startswith(b"<?xml") or b"<uddf" in head.lower():
        return read_uddf(path)
    return read_table(path, **kwargs)


# ------------------------------------------------------------------ photos
_NAME_EPOCH = re.compile(r"(?<!\d)(\d{10})(?:[._](\d{1,9}))?(?!\d)|(?<!\d)(\d{13}|\d{16}|\d{19})(?!\d)")
_NAME_DATE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_T ]?(\d{2})[-_.:h]?(\d{2})[-_.:m]?(\d{2})"
                        r"(?:[-_.,](\d{1,6}))?(?!\d)")


def time_from_name(name: str) -> float | None:
    """Seconds since 1970 read from a file name: Unix time (``1542883311752336025.png``,
    ``1542883311.752.jpg``) or a date and time (``20190312_142355_123``, ``2019-03-12_14-23-55``)."""
    stem = Path(name).stem
    m = _NAME_DATE.search(stem)
    if m:
        y, mo, d, h, mi, s, frac = m.groups()
        try:
            return _utc(datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))) + _fraction(frac)
        except ValueError:
            pass
    for m in _NAME_EPOCH.finditer(stem):
        if m.group(3):
            sec, absolute = epoch_from_number(float(m.group(3)))
        else:
            sec, absolute = float(m.group(1)) + _fraction(m.group(2)), True
        if absolute and 3e8 < sec < 5e9:  # 1979-2128
            return sec
    return None


def time_from_exif(path) -> float | None:
    """Seconds since 1970 of ``DateTimeOriginal`` (+ sub-seconds and the time zone when given) of a
    photo, or ``DateTime``; None without them."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
    except Exception:  # noqa: BLE001 - no EXIF, an unreadable file
        return None
    text = sub.get(36867) or sub.get(36868) or exif.get(306)
    if not text:
        return None
    try:
        dt = datetime.strptime(str(text).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    tz = sub.get(36881) or sub.get(36880)
    try:
        if tz and re.match(r"^[+-]\d{2}:\d{2}$", str(tz).strip()):
            dt = dt.replace(tzinfo=_tz(str(tz).strip()))
    except ValueError:
        pass
    frac = str(sub.get(37521) or sub.get(37522) or "").strip().rstrip("\x00")
    return _utc(dt) + (_fraction(frac) if frac.isdigit() else 0.0)


def photo_time(cam) -> tuple[float | None, str]:
    """``(seconds, where)`` of the moment a photo was taken: ``record`` (the plugin's frames),
    ``name``, ``exif``; ``(None, "")`` when nothing says it."""
    record = (cam.meta.get("underwater") or {}).get("time")
    if record is not None:
        return float(record), "record"
    original = (cam.meta.get("underwater") or {}).get("original") or cam.path
    sec = time_from_name(Path(original).name)
    if sec is not None:
        return sec, "name"
    sec = time_from_exif(original)
    if sec is not None:
        return sec, "exif"
    return None, ""


def photo_times(cams) -> dict[int, tuple[float, str]]:
    out = {}
    for cam in cams:
        sec, where = photo_time(cam)
        if sec is not None:
            out[cam.key] = (sec, where)
    return out


def match(cams, log: DepthLog, offset: float = 0.0, tolerance: float = 2.0, window: float = 1.0,
          times: dict | None = None) -> dict[int, float]:
    """``{camera key: depth}`` of the photos whose time (+ ``offset``) falls on the log. A log
    without dates (seconds from its start) is read from the first photo: its zero is the time of
    the earliest photo, then ``offset``."""
    times = photo_times(cams) if times is None else times
    if not times:
        return {}
    keys = list(times)
    t = np.array([times[k][0] for k in keys], float)
    if not log.absolute:
        t = t - float(np.min(t))
    d = log.depth_at(t + offset, window=window, tolerance=tolerance)
    return {k: float(v) for k, v in zip(keys, d) if np.isfinite(v)}


# ------------------------------------------------------------------ known positions (USBL)
X_WORDS = ("x", "east", "easting", "est", "e")
Y_WORDS = ("y", "north", "northing", "nord", "n")


def read_positions(path) -> dict:
    """Known X, Y positions from a table: by photo (a column of file names or labels) or by time (a
    time column, interpolated at the photos). Returns ``{"by_name": {name: (x, y)}}`` or
    ``{"times": array, "xy": (N, 2), "absolute": bool}``."""
    path = Path(path)
    rows = _rows(path.read_text(encoding="utf-8-sig", errors="replace"))
    if not rows or not any(not _is_value(c.lstrip("#")) for c in rows[0] if c.strip()):
        raise ValueError(t_("{0}: serve un'intestazione (foto o ora, x, y)", "{0}: a header is needed (photo or time, x, y)")
                         .format(path.name))
    names = [c.lstrip("#").strip() for c in rows[0]]
    rows = [r for r in rows[1:] if r and not r[0].lstrip().startswith("#")]
    xi, yi = _find(names, X_WORDS), _find(names, Y_WORDS)
    if xi is None or yi is None:
        raise ValueError(t_("{0}: nessuna colonna x, y fra {1}", "{0}: no x, y columns among {1}").format(path.name, names))
    name_i = _find(names, ("photo", "foto", "image", "immagine", "file", "name", "nome", "label", "camera"), {xi, yi})
    if name_i is not None:
        out = {}
        for r in rows:
            if len(r) > max(name_i, xi, yi) and _number(r[xi]) is not None and _number(r[yi]) is not None:
                out[Path(r[name_i].strip()).name.lower()] = (_number(r[xi]), _number(r[yi]))
        return {"by_name": out}
    ti = _find(names, TIME_WORDS, {xi, yi})
    if ti is None:
        raise ValueError(t_("{0}: né una colonna della foto né una del tempo", "{0}: neither a photo nor a time column")
                         .format(path.name))
    unit = _unit(names[ti])
    times, xy, absolute = [], [], True
    for r in rows:
        if len(r) <= max(ti, xi, yi):
            continue
        x, y = _number(r[xi]), _number(r[yi])
        num = _number(r[ti])
        if num is not None:
            sec, ab = epoch_from_number(num, unit)
        else:
            sec = parse_datetime(r[ti])
            ab = sec is not None
        if sec is None or x is None or y is None:
            continue
        absolute &= ab
        times.append(sec)
        xy.append((x, y))
    order = np.argsort(times)
    return {"times": np.asarray(times)[order], "xy": np.asarray(xy)[order], "absolute": absolute}


def match_positions(cams, path, offset: float = 0.0, tolerance: float = 2.0) -> dict[int, tuple[float, float]]:
    """``{camera key: (x, y)}`` of the photos a positions table gives (see :func:`read_positions`)."""
    pos = read_positions(path)
    out = {}
    if "by_name" in pos:
        for cam in cams:
            for name in (Path(cam.path).name.lower(), cam.label.lower(), Path(str((cam.meta.get("underwater") or {})
                                                                                   .get("original") or "")).name.lower()):
                if name in pos["by_name"]:
                    out[cam.key] = tuple(float(v) for v in pos["by_name"][name])
                    break
        return out
    times = photo_times(cams)
    if not times or not len(pos["times"]):
        return out
    keys = list(times)
    t = np.array([times[k][0] for k in keys])
    if not pos["absolute"]:
        t = t - t.min()
    t = t + offset
    inside = (t >= pos["times"][0] - tolerance) & (t <= pos["times"][-1] + tolerance)
    x = np.interp(t, pos["times"], pos["xy"][:, 0])
    y = np.interp(t, pos["times"], pos["xy"][:, 1])
    for k, ok, xv, yv in zip(keys, inside, x, y):
        if ok:
            out[k] = (float(xv), float(yv))
    return out
