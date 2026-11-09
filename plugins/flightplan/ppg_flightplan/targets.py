"""GNSS targets on the ground: from the file of the GNSS, or from the markers of the chunk, to the special
points of the flight.

A GCP is a point whose coordinates are known, measured in the field. So where the targets are is known before
flying: the drone does not have to look for them, it has to go there. Hence:

1. the file of the targets (name, coordinates, height) enters the plan (``piano["target"]``);
2. on every target a special point is born: a descent with a ladder of photos, the camera straight down.

The CSV is read by hand, as the planner does; from UTM to degrees with the usual formulas; any other system
(Gauss-Boaga, national grids...) through pyproj.

Ported from the author's Photo3D planner (the planning part); the systems other than UTM and the markers of
the chunk are new.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from . import plan as pl
from .geometry import LatLon, distance
from .texts import t

METRES_ABOVE_GROUND = 10.0      # from ten metres an A4 target can be read; the bigger, the better
SITE_MARGIN_M = 80.0            # around the photos, to find the targets of a site after the flight (task 8)

_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996


class TargetError(ValueError):
    pass


def utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) % 60 + 1


def latlon_from_utm(east: float, north: float, zone: int, north_hemisphere: bool = True) -> tuple[float, float]:
    """From UTM metres (WGS84) to degrees."""
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    x = east - 500000.0
    y = north if north_hemisphere else north - 10000000.0
    m = y / _K0
    mu = m / (_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    fi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
           + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
           + (151 * e1 ** 3 / 96) * math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    n1 = _A / math.sqrt(1 - e2 * math.sin(fi1) ** 2)
    t1 = math.tan(fi1) ** 2
    c1 = ep2 * math.cos(fi1) ** 2
    r1 = _A * (1 - e2) / (1 - e2 * math.sin(fi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    fi = fi1 - (n1 * math.tan(fi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lam = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(fi1)
    return math.degrees(fi), (zone - 1) * 6 - 180 + 3 + math.degrees(lam)


def code_of_name(name: str) -> Optional[int]:
    """The number of the coded target, from the name of the point: "T12", "target 3", "GCP_07", "5"."""
    numbers = re.findall(r"\d+", name or "")
    return int(numbers[-1]) if numbers else None


_NAMES = {
    "nome": ("nome", "name", "id", "punto", "point", "label", "etichetta", "target", "gcp", "pt"),
    "codice": ("codice", "code", "numero", "number", "n"),
    "lat": ("lat", "latitude", "latitudine"),
    "lon": ("lon", "lng", "long", "longitude", "longitudine"),
    "est": ("est", "east", "easting", "e", "x"),
    "nord": ("nord", "north", "northing", "y"),
    "quota": ("quota", "alt", "altitude", "altitudine", "z", "h", "elev", "elevation", "height", "ele"),
}


def _number(text: str, decimal_comma: bool) -> Optional[float]:
    s = text.strip().replace(" ", "")
    if decimal_comma:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _rows(text: str) -> tuple[list[list[str]], bool]:
    """The rows of the file split into fields, and whether the comma is the decimal sign."""
    rows = [r.strip() for r in text.replace("\r", "\n").split("\n") if r.strip() and not r.lstrip().startswith("#")]
    if not rows:
        raise TargetError(t("il file è vuoto", "the file is empty"))
    probe = rows[min(1, len(rows) - 1)]
    if ";" in probe:
        return [r.split(";") for r in rows], True
    if "\t" in probe:
        return [r.split("\t") for r in rows], True
    if "," in probe:
        return [r.split(",") for r in rows], False
    return [r.split() for r in rows], False


def _to_degrees(epsg: int):
    """From metres of ``epsg`` to (lat, lon): the planner's formula for UTM WGS84, pyproj for any other system."""
    if _is_utm(epsg):
        zone, north = epsg % 100, epsg < 32700
        return lambda east, north_m: latlon_from_utm(east, north_m, zone, north)
    from pyproj import Transformer

    tr = Transformer.from_crs(_system(epsg), "EPSG:4326", always_xy=True)

    def convert(east: float, north_m: float) -> tuple[float, float]:
        lon, lat = tr.transform(east, north_m)
        return lat, lon
    return convert


def _system(epsg: int):
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        return CRS.from_epsg(epsg)
    except (CRSError, ValueError) as e:
        raise TargetError(t("EPSG:{0} non è un sistema che conosco ({1})",
                            "EPSG:{0} is not a system I know ({1})").format(epsg, e)) from e


def _is_utm(epsg: int) -> bool:
    return 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760


def _metres_swap(raw: list, epsg: int, convert, near: Optional[LatLon], order: Optional[str],
                 warnings: list[str]) -> bool:
    """Without a header, which of the two numbers is the north? ``True`` = the first.

    In UTM the north is the larger of the two (above 9 degrees of latitude it has seven digits): the planner's
    rule. In other grids the east can be the larger (Swiss LV95, the British grid in the south), so the rule
    would throw the points far away without a word. There: ``order`` (``"latlon"`` = north first,
    ``"lonlat"`` = east first) if given; else the order whose points all fall where the system is used (its
    area of use); else, of the orders still possible, the one nearer to ``near``; else the system's own axis
    order, and the doubt is said (``ordine_incerto``)."""
    if order in ("latlon", "lonlat"):
        return order == "latlon"
    if _is_utm(epsg):
        return sum(1 for g in raw if g[1] > g[2]) > len(raw) / 2

    def places(swapped: bool) -> Optional[list[LatLon]]:
        out = []
        for g in raw:
            e, n = (g[2], g[1]) if swapped else (g[1], g[2])
            try:
                lat, lon = convert(e, n)
            except Exception:  # noqa: BLE001 - a projection that fails is an order that does not hold
                return None
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                return None
            out.append((lat, lon))
        return out

    system = _system(epsg)
    area = system.area_of_use
    margin = 0.5                                   # degrees: points just outside the official area still count

    def inside(pts: Optional[list[LatLon]]) -> bool:
        if pts is None:
            return False
        if area is None:
            return True
        return all(area.south - margin <= la <= area.north + margin and area.west - margin <= lo <= area.east + margin
                   for la, lo in pts)

    candidates = {sw: places(sw) for sw in (False, True)}
    possible = [sw for sw in (False, True) if inside(candidates[sw])]
    if len(possible) == 1:
        return possible[0]
    if near and possible:
        return min(possible, key=lambda sw: sum(distance(near, p) for p in candidates[sw]))
    axes = system.axis_info
    north_first = bool(axes) and axes[0].direction.lower() == "north"
    warnings.append("ordine_incerto")
    return north_first


def read_targets_csv(text: str, near: Optional[LatLon] = None, epsg: Optional[int] = None,
                     order: Optional[str] = None) -> dict:
    """The file of the targets, in degrees or in metres (UTM or the system ``epsg``), with or without a header.

    Without a header the order of the columns is understood from the numbers: in UTM the north has seven digits
    and the east six; in degrees - where 45 and 10 could each be the one or the other - the order that brings
    the points nearer to the site (``near``) is chosen, or ``order`` (``"latlon"``, ``"lonlat"``) says it.
    """
    rows, comma = _rows(text)
    header = None
    if all(_number(c, comma) is None for c in rows[0][1:]) and len(rows) > 1:
        header = [re.sub(r"[^a-z]", "", c.strip().lower()) for c in rows[0]]
        rows = rows[1:]
    columns: dict[str, int] = {}
    if header:
        for key, names in _NAMES.items():
            for i, c in enumerate(header):
                if c in names and key not in columns and i not in columns.values():
                    columns[key] = i
    warnings: list[str] = []
    raw = []
    for k, r in enumerate(rows):
        fields = [c.strip() for c in r]
        if header and ("lat" in columns and "lon" in columns or "est" in columns and "nord" in columns):
            def col(key):
                i = columns.get(key)
                return fields[i] if i is not None and i < len(fields) else ""
            name = col("nome") or str(k + 1)
            if "lat" in columns:
                a, b, kind = _number(col("lon"), comma), _number(col("lat"), comma), "gradi"
            else:
                a, b, kind = _number(col("est"), comma), _number(col("nord"), comma), "metri"
            height = _number(col("quota"), comma) if "quota" in columns else None
            code = _number(col("codice"), comma) if "codice" in columns else None
            raw.append((name, a, b, height, kind, int(code) if code is not None else None))
            continue
        numbers = [(i, _number(c, comma)) for i, c in enumerate(fields)]
        if numbers and numbers[0][1] is None:
            name, values = fields[0], [v for _, v in numbers[1:] if v is not None]
        else:
            name, values = str(k + 1), [v for _, v in numbers if v is not None]
            if len(values) >= 4:                       # "12, east, north, height": the first number is the name
                name, values = fields[0], values[1:]
        if len(values) < 2:
            continue
        raw.append((name, values[0], values[1], values[2] if len(values) > 2 else None, None, None))
    raw = [g for g in raw if g[1] is not None and g[2] is not None]
    if not raw:
        raise TargetError(t("nel file non ho trovato coordinate", "I found no coordinates in the file"))

    in_degrees = all(abs(g[1]) <= 180 and abs(g[2]) <= 180 for g in raw)
    without_names = any(g[4] is None for g in raw)
    system = _system(int(epsg)) if epsg is not None else None
    if in_degrees and system is not None and not system.is_geographic:
        raise TargetError(t("i numeri del file sembrano gradi, ma EPSG:{0} è in metri: controlla il sistema",
                            "the numbers of the file look like degrees, but EPSG:{0} is in metres: check the "
                            "system").format(epsg))
    if not in_degrees and system is not None and system.is_geographic:
        raise TargetError(t("i numeri del file sembrano metri, ma EPSG:{0} è in gradi: controlla il sistema",
                            "the numbers of the file look like metres, but EPSG:{0} is in degrees: check the "
                            "system").format(epsg))
    swap = False
    if without_names and in_degrees:
        # (a, b) is (lon, lat) or (lat, lon)? starting from "lat, lon", which is how it is usually written
        if order in ("latlon", "lonlat"):
            swap = order == "latlon"
        elif any(abs(g[1]) > 90 for g in raw):
            swap = False                        # the first is beyond 90: it cannot be a latitude
        elif any(abs(g[2]) > 90 for g in raw):
            swap = True
        elif near:
            def farness(as_lonlat: bool) -> float:
                return sum(distance(near, (g[2], g[1]) if as_lonlat else (g[1], g[2])) for g in raw)
            swap = farness(False) < farness(True)
        else:
            swap = True
            warnings.append("ordine_incerto")
    convert = None
    if in_degrees:
        crs = "EPSG:4326"
        if system is not None and system.to_epsg() != 4326:
            crs, convert = f"EPSG:{epsg}", _to_degrees(int(epsg))    # another datum (ETRS89, Monte Mario...)
    else:
        if epsg is None:
            if near is None:
                raise TargetError(t("coordinate in metri: serve il sistema (EPSG) o un sito a cui riferirle",
                                    "coordinates in metres: the system (EPSG) or a site to refer them to is needed"))
            epsg = (32600 if near[0] >= 0 else 32700) + utm_zone(near[1])
            warnings.append("zona_dedotta")
        crs = f"EPSG:{epsg}"
        convert = _to_degrees(int(epsg))
        if without_names:
            swap = _metres_swap(raw, int(epsg), convert, near, order, warnings)

    found, seen = [], set()
    for name, a, b, height, _kind, code in raw:
        if swap:
            a, b = b, a
        east, north = a, b                              # in degrees: east = lon, north = lat
        lat, lon = convert(east, north) if convert else (north, east)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise TargetError(t("il punto «{0}» esce fuori dal mondo: controlla l'ordine delle colonne",
                                "the point “{0}” falls off the world: check the order of the columns").format(name))
        code = code if code is not None else code_of_name(name)
        if code is not None and code in seen:
            warnings.append("codice_doppio")
        seen.add(code)
        found.append({"nome": name, "codice": code, "lat": round(lat, 8), "lon": round(lon, 8),
                      "est": east, "nord": north, "quota": height})
    return {"crs": crs, "punti": found, "avvisi": _common_warnings(found, near, warnings)}


def _common_warnings(found: list[dict], near: Optional[LatLon], warnings: list[str]) -> list[str]:
    if any(p["quota"] is None for p in found):
        warnings.append("senza_quota")                  # fine to fly over, not as GCPs
    if any(p["codice"] is None for p in found):
        warnings.append("senza_codice")
    if near and min(distance(near, (p["lat"], p["lon"])) for p in found) > 5000:
        warnings.append("lontani_dal_sito")
    return sorted(set(warnings))


def _horizontal(crs: str):
    """The horizontal part of a system (a compound one has heights of its own), as pyproj reads it."""
    from pyproj import CRS

    system = CRS.from_user_input(crs)
    return system.sub_crs_list[0] if system.is_compound else system


def targets_from_markers(chunk, only_with_reference: bool = True) -> dict:
    """The markers of the chunk as targets: those with a reference position (measured in the field), in the
    chunk's system, converted to degrees with pyproj. With ``only_with_reference=False`` also the markers the
    survey placed without a measure (their estimated position, warning ``stimati``), each flagged
    ``"stimato": True`` - kept in the plan with the point, so they are never taken for GCPs - and only when the
    chunk is georeferenced (with the identity transform they are left out, warning ``stimati_saltati``). The name is the
    marker's label, the number its own code or the one in its name."""
    crs = getattr(chunk, "crs", None)
    if not crs or str(crs).upper() == "LOCAL":
        raise TargetError(t("il chunk è in un sistema locale: i suoi marker non hanno una posizione sulla Terra",
                            "the chunk is in a local frame: its markers have no position on the Earth"))
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(_horizontal(crs), CRS.from_epsg(4326), always_xy=True)
    transform = getattr(chunk, "transform", None)
    georeferenced = transform is not None and not np.allclose(np.asarray(transform, dtype=float), np.eye(4))
    warnings: list[str] = []
    found, seen = [], set()
    for m in chunk.markers:
        loc = m.reference.location if m.reference is not None else None
        estimated = False
        if loc is not None and len(loc) >= 2 and loc[0] is not None and loc[1] is not None:
            x, y = float(loc[0]), float(loc[1])
            z = float(loc[2]) if len(loc) > 2 and loc[2] is not None else None
        elif not only_with_reference and m.position is not None:
            if not georeferenced:                # internal coordinates read as the chunk's would be nonsense
                warnings.append("stimati_saltati")
                continue
            x, y, z = (float(v) for v in chunk.to_world(np.asarray(m.position, dtype=float).reshape(1, 3))[0])
            estimated = True
            warnings.append("stimati")
        else:
            continue
        lon, lat = tr.transform(x, y)
        code = m.code if m.code is not None else code_of_name(m.label)
        if code is not None and code in seen:
            warnings.append("codice_doppio")
        seen.add(code)
        target = {"nome": str(m.label), "codice": code, "lat": round(lat, 8), "lon": round(lon, 8),
                  "est": x, "nord": y, "quota": z}
        if estimated:
            target["stimato"] = True             # the survey's own position: to fly over, never a GCP
        found.append(target)
    if not found:
        raise TargetError(t("nessun marker del chunk ha coordinate misurate",
                            "no marker of the chunk has measured coordinates"))
    return {"crs": str(crs), "punti": found, "avvisi": _common_warnings(found, None, warnings)}


def special_point(target: dict, metres: float = METRES_ABOVE_GROUND) -> dict:
    """The special point that photographs a target: free, straight down, with the ladder of photos."""
    return {"id": "target-" + re.sub(r"[^A-Za-z0-9]+", "-", str(target["nome"])).strip("-").lower(),
            "nome": str(target["nome"]), "tipo": "foto", "punto": [target["lat"], target["lon"]], "aggancio": False,
            "discesa": {"modo": "suolo", "metri": float(metres)}, "inclinazione": -90, "scaletta": True,
            "target": str(target["nome"]), **({"stimato": True} if target.get("stimato") else {})}


def put_targets(piano: dict, read: dict, block: Optional[int] = None, metres: float = METRES_ABOVE_GROUND) -> dict:
    """The targets enter the plan, and on each a special point is born in the chosen block (the first active
    one, if not said). Putting them again, those of before go away: no duplicates."""
    blocks = piano.get("blocchi", [])
    for b in blocks:
        if b.get("speciali"):
            b["speciali"] = [sp for sp in b["speciali"] if not sp.get("target")]
            if not b["speciali"]:
                del b["speciali"]
    piano["target"] = {"crs": read["crs"], "punti": read["punti"]}
    if block is None:
        block = next((i for i, b in enumerate(blocks) if pl.active(b)), None)
    if block is None or not 0 <= block < len(blocks):
        return piano                                     # without blocks the targets stay in the plan, for later
    blocks[block].setdefault("speciali", []).extend(special_point(p, metres) for p in read["punti"])
    return piano


def remove_targets(piano: dict) -> dict:
    piano.pop("target", None)
    for b in piano.get("blocchi", []):
        if b.get("speciali"):
            b["speciali"] = [sp for sp in b["speciali"] if not sp.get("target")]
            if not b["speciali"]:
                del b["speciali"]
    return piano

