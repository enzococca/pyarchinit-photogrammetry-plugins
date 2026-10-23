"""Boreholes, cores and trench logs of a chunk.

A borehole is a collar (id, x and y in the chunk's reference system, z of its top, the depth
reached, the kind: auger, core, trench) and its intervals (from and to in metres below the collar,
the unit - a US number or a lithological unit -, a description, a colour, a lithology). They are
read from CSV tables - a collar table and an interval table, or one table with both - whose
columns are recognised by their names in Italian or English; the intervals may also be given by
their elevations (``quota_tetto``, ``quota_letto``). Collars without z take it from the DEM of the
chunk. Everything lives in ``chunk.meta["geoarchaeology"]``: the boreholes, the units in their
order from the top down with a colour each; the CSV files read are copied into
``<chunk data>/geoarchaeology/sources/``.

The order of the units comes from the logs (a unit above another in a borehole lies above it)
and can be set by hand. :func:`contact_data` gives, for the top of every unit, its elevation at
the boreholes that cross it - where a unit is missing from a log its top is the contact between
the units above and below it (zero thickness) - and the bottoms of the boreholes that end above
it (the contact lies deeper there).
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .texts import num, t

FOLDER = "geoarchaeology"
KINDS = ("auger", "core", "trench", "section", "test_pit", "other")
TOLERANCE = 0.005  # m: depths that differ less are the same depth (rounding of the tables)

# column names (normalised: ascii, lowercase, words joined by "_") of each field
ALIASES = {
    "id": ["id", "hole", "hole_id", "holeid", "bhid", "bh", "borehole", "borehole_id", "boring", "core_id",
           "sondaggio", "sondaggi", "id_sondaggio", "nome_sondaggio", "carota", "carotaggio", "trincea", "trench",
           "saggio", "name", "nome", "sigla", "codice", "point", "punto", "station", "location", "location_id"],
    "x": ["x", "east", "easting", "est", "e", "x_coord", "coord_x", "xcoord", "x_utm", "utm_e", "utm_x"],
    "y": ["y", "north", "northing", "nord", "n", "y_coord", "coord_y", "ycoord", "y_utm", "utm_n", "utm_y"],
    "z": ["z", "elevation", "elev", "collar_z", "z_collar", "collar_elevation", "rl", "quota", "quota_bocca",
          "quota_testa", "quota_piano", "quota_slm", "quota_p_c", "h", "height", "altitude", "alt", "ground_level"],
    "depth": ["depth", "total_depth", "max_depth", "final_depth", "eoh", "td", "profondita", "profondita_totale",
              "profondita_raggiunta", "prof", "prof_max", "prof_totale"],
    "kind": ["kind", "type", "tipo", "method", "metodo", "tool", "strumento", "tipologia"],
    "from": ["from", "depth_from", "from_depth", "top", "top_depth", "da", "dal", "prof_da", "profondita_da",
             "tetto", "inizio", "start", "da_prof"],
    "to": ["to", "depth_to", "to_depth", "bottom", "bottom_depth", "base", "a", "al", "prof_a", "profondita_a",
           "letto", "fine", "end", "a_prof"],
    "top_z": ["top_z", "z_top", "z_from", "from_z", "top_elevation", "quota_tetto", "quota_da", "quota_sup",
              "quota_superiore", "z_tetto", "q_tetto"],
    "base_z": ["base_z", "bottom_z", "z_base", "z_bottom", "z_to", "to_z", "bottom_elevation", "quota_letto",
               "quota_base", "quota_a", "quota_inf", "quota_inferiore", "z_letto", "q_letto"],
    "unit": ["unit", "unit_code", "unit_id", "unita", "us", "us_code", "layer", "strato", "stratum", "horizon",
             "orizzonte", "facies", "formation", "litho_unit", "unita_stratigrafica", "codice_unita"],
    "description": ["description", "descrizione", "desc", "descr", "note", "notes", "remarks", "osservazioni"],
    "colour": ["colour", "color", "colore", "munsell", "munsell_colour", "munsell_color", "colore_munsell"],
    "lithology": ["lithology", "litologia", "lito", "lith", "material", "materiale", "texture", "tessitura"],
}
_LOOKUP = {name: key for key, names in ALIASES.items() for name in names}
_DEPTH_FIELDS = ("depth", "from", "to")
# colours of the units without one of their own: earthy, light and dark in turn
PALETTE = ["#d9b77e", "#7d5a3c", "#c9a55c", "#a0522d", "#e6d3a3", "#6b6b4b", "#c27c4c", "#8c9a6b",
           "#b0846a", "#596e79", "#e0c08a", "#704f3c", "#a39e7a", "#8a5a6b", "#cfa98a", "#4f5d4f"]
# lithologies drawn with a hatch in the logs and sections (keywords of the lithology or description)
HATCHES = [(("argill", "clay"), "--"), (("limo", "limos", "silt"), "-."), (("sabbi", "sand"), ".."),
           (("ghiai", "gravel", "ciottol", "pebble", "cobble"), "oo"),
           (("crollo", "pietr", "stone", "rubble", "macerie", "laterizi", "brick"), "**"),
           (("cenere", "carbon", "ash", "charcoal"), "xx"), (("torba", "peat"), "||"),
           (("organic", "organico", "humus"), "\\\\")]


@dataclass
class Interval:
    """A stretch of a log: ``top`` and ``base`` in metres below the collar."""

    top: float
    base: float
    unit: str
    description: str = ""
    colour: str = ""
    lithology: str = ""

    def to_dict(self) -> dict:
        d = {"from": round(float(self.top), 4), "to": round(float(self.base), 4), "unit": self.unit}
        for k in ("description", "colour", "lithology"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Interval":
        return cls(float(d["from"]), float(d["to"]), str(d["unit"]), str(d.get("description") or ""),
                   str(d.get("colour") or ""), str(d.get("lithology") or ""))


@dataclass
class Borehole:
    """A collar and its log. ``z`` is the elevation of the top (None when unknown), ``depth`` the
    depth reached (at least the base of the last interval)."""

    id: str
    x: float
    y: float
    z: float | None = None
    depth: float | None = None
    kind: str = ""
    intervals: list = field(default_factory=list)
    z_source: str = ""  # "file", "dem" or ""
    dem_z: float | None = None  # the DEM at the collar, as a check of z
    source: str = ""

    @property
    def bottom(self) -> float:
        """Depth of the end of the hole."""
        last = max((i.base for i in self.intervals), default=0.0)
        return max(float(self.depth or 0.0), last)

    @property
    def units(self) -> list[str]:
        out = []
        for i in sorted(self.intervals, key=lambda i: i.top):
            if not out or out[-1] != i.unit:
                out.append(i.unit)
        return out

    def to_dict(self) -> dict:
        d = {"id": self.id, "x": float(self.x), "y": float(self.y), "z": None if self.z is None else float(self.z),
             "depth": None if self.depth is None else float(self.depth), "kind": self.kind, "z_source": self.z_source,
             "intervals": [i.to_dict() for i in self.intervals]}
        if self.dem_z is not None:
            d["dem_z"] = float(self.dem_z)
        if self.source:
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Borehole":
        return cls(str(d["id"]), float(d["x"]), float(d["y"]), None if d.get("z") is None else float(d["z"]),
                   None if d.get("depth") is None else float(d["depth"]), str(d.get("kind") or ""),
                   [Interval.from_dict(i) for i in d.get("intervals", [])], str(d.get("z_source") or ""),
                   None if d.get("dem_z") is None else float(d["dem_z"]), str(d.get("source") or ""))


# ------------------------------------------------------------------ storage
def folder(chunk) -> Path:
    return Path(chunk.data_dir) / FOLDER


def meta(chunk) -> dict:
    return chunk.meta.setdefault("geoarchaeology", {})


def load(chunk) -> list[Borehole]:
    return [Borehole.from_dict(d) for d in (chunk.meta.get("geoarchaeology") or {}).get("boreholes", [])]


def get(chunk, bid: str) -> Borehole | None:
    return next((b for b in load(chunk) if b.id == str(bid)), None)


def save(chunk, boreholes: list[Borehole]) -> None:
    m = meta(chunk)
    m["boreholes"] = [b.to_dict() for b in boreholes]
    _refresh_units(chunk, boreholes)
    if m.get("surfaces"):
        m["surfaces"]["stale"] = True  # built from other logs: to build again


# ------------------------------------------------------------------ reading the tables
def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _field_of(column: str) -> tuple[str | None, float]:
    """The field a column holds and the factor to metres (``da_cm`` -> ("from", 0.01))."""
    s = normalise(column)
    if s in _LOOKUP:
        return _LOOKUP[s], 1.0
    m = re.match(r"^(.*?)_(m|mt|metri|metres|meters|cm)$", s)
    if m and m.group(1) in _LOOKUP:
        key = m.group(1)
        return _LOOKUP[key], 0.01 if m.group(2) == "cm" and _LOOKUP[key] in _DEPTH_FIELDS else 1.0
    return None, 1.0


def parse_number(text) -> float | None:
    """A number written with a decimal point or comma (``"12,5"``, ``"1.234,5"``), None when empty."""
    if text is None:
        return None
    s = str(text).strip().replace(" ", "").replace(" ", "")
    if not s or s in ("-", "--", "na", "NA", "nan", "NaN", "null", "None"):
        return None
    if "," in s and "." in s:  # the separator that comes last is the decimal one
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a delimited text file (comma, semicolon, tab or bar; whitespace for a
    table without spaces in its values), in UTF-8 or Windows-1252."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], []
    sample = "\n".join(lines[:50])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: lines[0].count(d) for d in (";", "\t", ",", "|")}
        delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    if delimiter is None:
        rows = [ln.split() for ln in lines]
    else:
        rows = [r for r in csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)]
    header = [h.strip() for h in rows[0]]
    return header, [[c.strip() for c in r] for r in rows[1:] if any(c.strip() for c in r)]


def read_table(path: str | Path) -> tuple[list[dict], dict]:
    """The rows of a table as dictionaries of fields (numbers parsed, depths in metres) and the
    fields found (``{"id": "Sondaggio", ...}``: field -> column)."""
    header, rows = read_csv(path)
    columns: dict[str, tuple[int, float]] = {}
    names: dict[str, str] = {}
    for k, h in enumerate(header):
        key, factor = _field_of(h)
        if key is not None and key not in columns:
            columns[key] = (k, factor)
            names[key] = h
    out = []
    for r in rows:
        rec = {}
        for key, (k, factor) in columns.items():
            value = r[k] if k < len(r) else ""
            if key in ("id", "unit", "description", "colour", "lithology", "kind"):
                rec[key] = value.strip()
            else:
                v = parse_number(value)
                rec[key] = None if v is None else v * factor
        out.append(rec)
    return out, names


def read_tables(paths) -> tuple[dict, dict, list[str], list[str]]:
    """Collars and intervals from one or more tables: ``(collars {id: record}, intervals {id:
    [record]}, kinds of the tables, warnings)``. A table with positions and depths of intervals
    gives both; one with positions only gives collars; one with intervals only gives intervals."""
    collars: dict[str, dict] = {}
    intervals: dict[str, list] = {}
    kinds, warnings = [], []
    for path in paths:
        rows, names = read_table(path)
        name = Path(path).name
        if "id" not in names:
            raise ValueError(t("{0}: manca la colonna del sondaggio (id, sondaggio, hole...)",
                               "{0}: no column of the borehole (id, sondaggio, hole...)").format(name))
        has_xy = "x" in names and "y" in names
        has_iv = ("from" in names and "to" in names) or ("top_z" in names and "base_z" in names)
        if not has_xy and not has_iv:
            raise ValueError(t("{0}: né coordinate (x, y) né intervalli (da, a / from, to)",
                               "{0}: neither coordinates (x, y) nor intervals (from, to)").format(name))
        kinds.append("combined" if has_xy and has_iv else "collars" if has_xy else "intervals")
        for n, rec in enumerate(rows, start=2):
            bid = rec.get("id") or ""
            if not bid:
                continue
            if has_xy and rec.get("x") is not None and rec.get("y") is not None:
                c = {k: rec.get(k) for k in ("x", "y", "z", "depth", "kind")}
                c["source"] = name
                old = collars.get(bid)
                if old is None:
                    collars[bid] = c
                elif abs(old["x"] - c["x"]) > 1e-3 or abs(old["y"] - c["y"]) > 1e-3:
                    warnings.append(t("{0}: {1} ha due posizioni (riga {2}): tenuta la prima",
                                      "{0}: {1} has two positions (line {2}): the first kept").format(name, bid, n))
                else:
                    for k in ("z", "depth", "kind"):
                        if old.get(k) in (None, "") and c.get(k) not in (None, ""):
                            old[k] = c[k]
            if has_iv:
                top, base = rec.get("from"), rec.get("to")
                ztop, zbase = rec.get("top_z"), rec.get("base_z")
                if (top is None or base is None) and (ztop is None or zbase is None):
                    continue
                intervals.setdefault(bid, []).append({
                    "top": top, "base": base, "top_z": ztop, "base_z": zbase, "unit": rec.get("unit", ""),
                    "description": rec.get("description", ""), "colour": rec.get("colour", ""),
                    "lithology": rec.get("lithology", ""), "line": n, "file": name})
    return collars, intervals, kinds, warnings


# ------------------------------------------------------------------ the DEM at points
def raster_values(path: str | Path, x, y, band: int = 1) -> np.ndarray:
    """Bilinear values of a raster at world points; the empty cells around a point are left out,
    NaN where none is valid. Read by windows of 256 cells: a large DEM is never read whole."""
    import rasterio
    from rasterio.windows import Window

    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    out = np.full(x.shape, np.nan)
    with rasterio.open(path) as src:
        inv = ~src.transform
        fc = inv.a * x + inv.b * y + inv.c - 0.5  # in cell centres
        fr = inv.d * x + inv.e * y + inv.f - 0.5
        ok = np.isfinite(fc) & np.isfinite(fr) & (fc > -1) & (fr > -1) & (fc < src.width) & (fr < src.height)
        if not ok.any():
            return out
        c0 = np.floor(fc).astype(np.int64)
        r0 = np.floor(fr).astype(np.int64)
        keys = (r0 // 256) * 1_000_000 + (c0 // 256)
        for key in np.unique(keys[ok]):
            sel = np.nonzero(ok & (keys == key))[0]
            ra, rb = int(r0[sel].min()), int(r0[sel].max()) + 2
            ca, cb = int(c0[sel].min()), int(c0[sel].max()) + 2
            ra_, ca_ = max(ra, 0), max(ca, 0)
            rb_, cb_ = min(rb, src.height), min(cb, src.width)
            if rb_ <= ra_ or cb_ <= ca_:
                continue
            a = src.read(band, window=Window(ca_, ra_, cb_ - ca_, rb_ - ra_), masked=True)
            a = np.ma.filled(a.astype(np.float64), np.nan)
            num = np.zeros(len(sel))
            den = np.zeros(len(sel))
            tr_, tc_ = fr[sel] - r0[sel], fc[sel] - c0[sel]
            for dr, dc, w in ((0, 0, (1 - tr_) * (1 - tc_)), (0, 1, (1 - tr_) * tc_), (1, 0, tr_ * (1 - tc_)),
                              (1, 1, tr_ * tc_)):
                rr, cc = r0[sel] + dr - ra_, c0[sel] + dc - ca_
                inside = (rr >= 0) & (cc >= 0) & (rr < a.shape[0]) & (cc < a.shape[1])
                v = np.full(len(sel), np.nan)
                v[inside] = a[rr[inside], cc[inside]]
                good = np.isfinite(v) & (w > 0)
                num[good] += w[good] * v[good]
                den[good] += w[good]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[sel] = np.where(den > 1e-9, num / den, np.nan)
    return out


def dem_path(chunk) -> Path | None:
    p = getattr(chunk, "dem_path", None)
    return Path(p) if p is not None and Path(p).exists() else None


# ------------------------------------------------------------------ import
def import_files(chunk, paths, replace: bool = False, kind: str = "", progress=None) -> dict:
    """Read the tables ``paths`` (one combined table, or collars and intervals in any order) into
    the chunk; ``replace``: forget the boreholes there were, otherwise those with the same id are
    replaced and the others kept. Returns a report: boreholes, intervals, units, the collars
    whose z came from the DEM, the difference between the given z and the DEM, the warnings."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    paths = [Path(p) for p in paths]
    progress(0.05, "Reading the tables")
    collars, ivs, kinds, warnings = read_tables(paths)
    if not collars:
        raise ValueError(t("Nessun sondaggio con le coordinate nei file letti",
                           "No borehole with coordinates in the files read"))
    for bid in sorted(set(ivs) - set(collars)):
        warnings.append(t("{0}: intervalli senza la posizione del sondaggio: lasciati fuori",
                          "{0}: intervals without the position of the borehole: left out").format(bid))
    dem = dem_path(chunk)
    ids = list(collars)
    xy = np.array([[collars[i]["x"], collars[i]["y"]] for i in ids], float)
    dem_z = raster_values(dem, xy[:, 0], xy[:, 1]) if dem is not None else np.full(len(ids), np.nan)
    progress(0.4, "Collars and intervals")
    new, no_z, from_dem = [], [], 0
    for k, bid in enumerate(ids):
        c = collars[bid]
        z, z_source = c.get("z"), "file"
        dz = float(dem_z[k]) if np.isfinite(dem_z[k]) else None
        if z is None:
            z, z_source = dz, ("dem" if dz is not None else "")
            from_dem += dz is not None
        if z is None:
            no_z.append(bid)
        b = Borehole(bid, float(c["x"]), float(c["y"]), z, c.get("depth"), kind or c.get("kind") or "", [], z_source,
                     dz, c.get("source", ""))
        for rec in ivs.get(bid, []):
            top, base = rec["top"], rec["base"]
            if top is None or base is None:  # given by elevations
                if z is None:
                    warnings.append(t("{0}: intervallo in quote (riga {1}) senza la quota del sondaggio: lasciato fuori",
                                      "{0}: interval by elevations (line {1}) without the elevation of the borehole: "
                                      "left out").format(bid, rec["line"]))
                    continue
                top, base = z - rec["top_z"], z - rec["base_z"]
            unit = str(rec["unit"] or "").strip()
            if not unit:
                warnings.append(t("{0}: intervallo {1}-{2} m senza unità (riga {3}): lasciato fuori",
                                  "{0}: interval {1}-{2} m without a unit (line {3}): left out")
                                .format(bid, num(top), num(base), rec["line"]))
                continue
            if base <= top:
                warnings.append(t("{0}: intervallo {1}-{2} m con la base sopra il tetto (riga {3}): lasciato fuori",
                                  "{0}: interval {1}-{2} m with its base above its top (line {3}): left out")
                                .format(bid, num(top), num(base), rec["line"]))
                continue
            b.intervals.append(Interval(float(top), float(base), unit, rec["description"], rec["colour"],
                                        rec["lithology"]))
        _check_log(b, warnings)
        new.append(b)
    progress(0.7, "Into the project")
    old = [] if replace else [b for b in load(chunk) if b.id not in {n.id for n in new}]
    boreholes = old + new
    save(chunk, boreholes)
    m = meta(chunk)
    src = folder(chunk) / "sources"
    src.mkdir(parents=True, exist_ok=True)
    for p in paths:
        try:
            shutil.copy(p, src / p.name)
        except (OSError, shutil.SameFileError):
            pass
    m["sources"] = sorted(set(m.get("sources", [])) | {p.name for p in paths}) if not replace else [p.name for p in paths]
    m["imported"] = time.strftime("%Y-%m-%d %H:%M")
    try:
        from .export import write_layers

        write_layers(chunk)
    except Exception as exc:  # noqa: BLE001 - the GeoPackage of the layers is a convenience
        warnings.append(f"GeoPackage: {exc}")
    checked = [b for b in new if b.z_source == "file" and b.dem_z is not None]
    dz = np.array([b.z - b.dem_z for b in checked]) if checked else np.zeros(0)
    report = {"files": [p.name for p in paths], "tables": kinds, "boreholes": len(boreholes), "read": len(new),
              "intervals": sum(len(b.intervals) for b in new), "units": order(chunk), "z_from_dem": from_dem,
              "no_z": no_z, "warnings": warnings}
    if dz.size:
        worst = int(np.argmax(np.abs(dz)))
        report["dem_check"] = {"boreholes": int(dz.size), "median_m": float(np.median(dz)),
                               "max_m": float(dz[worst]), "worst": checked[worst].id}
    progress(1.0, f"{len(new)} boreholes, {report['intervals']} intervals")
    return report


def _check_log(b: Borehole, warnings: list) -> None:
    """Intervals sorted, overlaps cut, gaps reported, the depth at least the last base."""
    b.intervals.sort(key=lambda i: (i.top, i.base))
    keep = []
    for iv in b.intervals:
        if keep and iv.top < keep[-1].base - TOLERANCE:
            if iv.base <= keep[-1].base + TOLERANCE:
                warnings.append(t("{0}: intervallo {1}-{2} m dentro il precedente: lasciato fuori",
                                  "{0}: interval {1}-{2} m inside the one before: left out")
                                .format(b.id, num(iv.top), num(iv.base)))
                continue
            warnings.append(t("{0}: intervalli sovrapposti a {1} m: il secondo parte dalla base del primo",
                              "{0}: overlapping intervals at {1} m: the second starts at the base of the first")
                            .format(b.id, num(iv.top)))
            iv.top = keep[-1].base
        elif keep and iv.top > keep[-1].base + TOLERANCE:
            if iv.unit != keep[-1].unit:  # between two units: their contact is put in the middle
                warnings.append(t("{0}: vuoto fra {1} e {2} m (contatto a metà)",
                                  "{0}: gap between {1} and {2} m (contact in the middle)")
                                .format(b.id, num(keep[-1].base), num(iv.top)))
            else:  # inside one unit (a core lost): no contact there
                warnings.append(t("{0}: vuoto fra {1} e {2} m dentro {3} (nessun contatto)",
                                  "{0}: gap between {1} and {2} m inside {3} (no contact)")
                                .format(b.id, num(keep[-1].base), num(iv.top), iv.unit))
        keep.append(iv)
    b.intervals = keep
    last = max((i.base for i in keep), default=0.0)
    if b.depth is not None and last - TOLERANCE <= b.depth < last:
        b.depth = last  # the same depth, rounded differently
    if b.depth is not None and b.depth < last - TOLERANCE:
        warnings.append(t("{0}: profondità {1} m minore della base dell'ultimo intervallo ({2} m)",
                          "{0}: depth {1} m less than the base of the last interval ({2} m)")
                        .format(b.id, num(b.depth), num(last)))
        b.depth = last


def remove(chunk, ids=None) -> int:
    """Forget the boreholes ``ids`` (all when None); returns how many went."""
    bhs = load(chunk)
    keep = [] if ids is None else [b for b in bhs if b.id not in {str(i) for i in ids}]
    save(chunk, keep)
    return len(bhs) - len(keep)


# ------------------------------------------------------------------ units and their order
def unit_order(boreholes: list[Borehole]) -> tuple[list[str], list[str]]:
    """The units from the top down: a unit lying directly above another in a log comes before
    it; the disagreements between logs go to the majority, and a circle of them is broken by
    the mean depth of the units. Returns the order and the conflicts found."""
    from collections import Counter, defaultdict

    seen: list[str] = []
    depth_sum: dict[str, float] = defaultdict(float)
    depth_n: dict[str, int] = defaultdict(int)
    edges: Counter = Counter()
    for b in boreholes:
        units = b.units
        first = {}
        for iv in sorted(b.intervals, key=lambda i: i.top):
            first.setdefault(iv.unit, iv.top)
        for u in units:
            if u not in depth_n:
                seen.append(u)
            depth_sum[u] += first[u]
            depth_n[u] += 1
        for a, c in zip(units[:-1], units[1:]):
            edges[(a, c)] += 1
    mean = {u: depth_sum[u] / max(depth_n[u], 1) for u in seen}
    conflicts = []
    succ: dict[str, set] = defaultdict(set)
    indeg = {u: 0 for u in seen}
    for (a, c), n in edges.items():
        back = edges.get((c, a), 0)
        if back > n or (back == n and mean[a] > mean[c]):
            continue
        if back:
            conflicts.append(t("{0} sopra {1} in {2} sondaggi, sotto in {3}",
                               "{0} above {1} in {2} boreholes, below in {3}").format(a, c, n, back))
        if c not in succ[a]:
            succ[a].add(c)
            indeg[c] += 1
    order, left = [], set(seen)
    while left:
        ready = [u for u in left if indeg[u] == 0]
        if not ready:  # a circle of disagreements: the shallowest unit first
            u = min(left, key=lambda v: (mean[v], seen.index(v)))
            conflicts.append(t("ordine circolare fra le unità: {0} messa prima per la profondità media",
                               "circular order among the units: {0} put first by its mean depth").format(u))
        else:
            u = min(ready, key=lambda v: (mean[v], seen.index(v)))
        order.append(u)
        left.discard(u)
        for c in succ.pop(u, set()):
            indeg[c] -= 1
    return order, conflicts


def order(chunk) -> list[str]:
    return [u["code"] for u in meta(chunk).get("units", [])]


def units(chunk) -> list[dict]:
    """The units in their order: ``{"code", "colour", "description", "boreholes"}``."""
    return [dict(u) for u in meta(chunk).get("units", [])]


def set_order(chunk, codes) -> list[str]:
    """The order of the units given by hand (unknown codes ignored, the missing ones after, in
    the order of the logs); kept until the boreholes change units."""
    have = order(chunk)
    codes = [str(c).strip() for c in codes if str(c).strip() in have]
    codes += [c for c in have if c not in codes]
    m = meta(chunk)
    by = {u["code"]: u for u in m.get("units", [])}
    m["units"] = [by[c] for c in codes]
    m["order_manual"] = True
    if m.get("surfaces") and codes != have:
        m["surfaces"]["stale"] = True  # built in another order
    return codes


def _refresh_units(chunk, boreholes: list[Borehole]) -> None:
    from collections import Counter

    m = meta(chunk)
    auto, conflicts = unit_order(boreholes)
    old = [u["code"] for u in m.get("units", [])]
    if m.get("order_manual") and set(auto) <= set(old):
        codes = [c for c in old if c in auto]
    else:
        codes = auto
        m["order_manual"] = False
    m["order_conflicts"] = conflicts
    old_colours = {u["code"]: u.get("colour") for u in m.get("units", [])}
    out = []
    for k, code in enumerate(codes):
        ivs = [iv for b in boreholes for iv in b.intervals if iv.unit == code]
        desc = Counter(iv.description for iv in ivs if iv.description).most_common(1)
        given = Counter(iv.colour for iv in ivs if iv.colour).most_common(1)
        colour = parse_colour(given[0][0]) if given else None
        colour = colour or old_colours.get(code) or PALETTE[k % len(PALETTE)]
        lith = Counter(iv.lithology for iv in ivs if iv.lithology).most_common(1)
        out.append({"code": code, "colour": colour, "description": desc[0][0] if desc else "",
                    "lithology": lith[0][0] if lith else "",
                    "boreholes": len({b.id for b in boreholes if any(iv.unit == code for iv in b.intervals)})})
    m["units"] = out


def colours(chunk) -> dict[str, str]:
    return {u["code"]: u.get("colour") or PALETTE[k % len(PALETTE)] for k, u in enumerate(meta(chunk).get("units", []))}


def parse_colour(text: str) -> str | None:
    """``#rrggbb`` of a colour written as ``#rrggbb``, ``r,g,b`` (0-255), a name (``brown``) or a
    Munsell notation (``10YR 4/3``, with colour-science); None when it cannot be read."""
    s = str(text or "").strip()
    if not s:
        return None
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", s):
        return "#" + s.lstrip("#").lower()
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if re.fullmatch(r"\s*\d+(\.\d+)?\s*[,; ]\s*\d+(\.\d+)?\s*[,; ]\s*\d+(\.\d+)?\s*", s) and len(nums) == 3:
        r, g, b = (min(255, int(float(v))) for v in nums)
        return f"#{r:02x}{g:02x}{b:02x}"
    if re.match(r"^\s*(\d+(\.\d+)?\s*(R|YR|Y|GY|G|BG|B|PB|P|RP)|N)\s*\d", s, re.IGNORECASE):
        return munsell_to_hex(s)
    try:
        from matplotlib.colors import to_hex

        return to_hex(s.lower())
    except Exception:  # noqa: BLE001 - not a colour name
        return None


def munsell_to_hex(notation: str) -> str | None:
    """sRGB of a Munsell notation (illuminant C to D65), with colour-science; None without it."""
    try:
        import warnings

        import colour

        s = re.sub(r"\s+", " ", notation.strip().upper())
        m = re.match(r"^N\s*(\d+(\.\d+)?)\s*/?\s*0?$", s)
        if m:  # the neutral greys
            s = f"N{m.group(1)}"
        else:
            s = re.sub(r"^(\d+(\.\d+)?)\s*([A-Z]+)\s*", r"\1\3 ", s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xyy = colour.munsell_colour_to_xyY(s)
            xyz = colour.xyY_to_XYZ(xyy)
            c = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]
            rgb = colour.XYZ_to_sRGB(xyz, illuminant=c, chromatic_adaptation_transform="Bradford")
        r, g, b = (int(round(255 * float(np.clip(v, 0, 1)))) for v in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:  # noqa: BLE001 - no colour-science, or a notation it does not know
        return None


def hatch_of(*texts: str) -> str:
    """The matplotlib hatch of a lithology named in ``texts`` ("" for none)."""
    low = " ".join(str(s or "").lower() for s in texts)
    for words, hatch in HATCHES:
        if any(w in low for w in words):
            return hatch
    return ""


# ------------------------------------------------------------------ the contacts at the boreholes
@dataclass
class ContactData:
    """The top of the unit ``order[k]`` at the boreholes: where it is crossed (``z`` elevation,
    ``depth`` below the collar) and where the hole ends above it (``below_*``: the contact lies
    deeper than that bottom)."""

    k: int
    unit: str
    ids: list
    xy: np.ndarray
    z: np.ndarray
    depth: np.ndarray
    below_ids: list
    below_xy: np.ndarray
    below_z: np.ndarray
    below_depth: np.ndarray

    @property
    def at_surface(self) -> bool:
        """The contact is the ground: crossed at the collar of every borehole that has it."""
        return len(self.ids) > 0 and bool(np.all(np.abs(self.depth) < 1e-3))


def contact_data(boreholes: list[Borehole], codes: list[str]) -> tuple[list[ContactData], list[str]]:
    """For the top of every unit of ``codes`` (in order from the top down) its depth at every
    borehole: the top of the first interval of that unit or of one below it (in the middle of a
    gap between logged intervals); boreholes that end before reaching it give their bottom.
    Returns the contacts and the boreholes whose units are not in that order."""
    pos = {u: k for k, u in enumerate(codes)}
    rows = [dict(ids=[], xy=[], z=[], depth=[], bids=[], bxy=[], bz=[], bdepth=[]) for _ in codes]
    notes = []
    for b in boreholes:
        ivs = [iv for iv in sorted(b.intervals, key=lambda i: i.top) if iv.unit in pos]
        if not ivs:
            continue
        p = [pos[iv.unit] for iv in ivs]
        if any(q < r for r, q in zip(p[:-1], p[1:])):
            j = next(j for j in range(1, len(p)) if p[j] < p[j - 1])
            notes.append(t("{0}: {1} sotto {2}, contro l'ordine delle unità", "{0}: {1} below {2}, against the order of "
                           "the units").format(b.id, ivs[j].unit, ivs[j - 1].unit))
        z = np.nan if b.z is None else float(b.z)
        for k in range(len(codes)):
            j = next((j for j in range(len(ivs)) if p[j] >= k), None)
            r = rows[k]
            if j is None:
                r["bids"].append(b.id)
                r["bxy"].append((b.x, b.y))
                r["bdepth"].append(b.bottom)
                r["bz"].append(z - b.bottom)
                continue
            d = ivs[j].top
            if j > 0 and ivs[j - 1].base < d - 1e-6:
                d = 0.5 * (ivs[j - 1].base + d)
            r["ids"].append(b.id)
            r["xy"].append((b.x, b.y))
            r["depth"].append(d)
            r["z"].append(z - d)
    out = []
    for k, code in enumerate(codes):
        r = rows[k]
        out.append(ContactData(k, code, r["ids"], np.asarray(r["xy"], float).reshape(-1, 2), np.asarray(r["z"], float),
                               np.asarray(r["depth"], float), r["bids"], np.asarray(r["bxy"], float).reshape(-1, 2),
                               np.asarray(r["bz"], float), np.asarray(r["bdepth"], float)))
    return out, notes


@dataclass
class Presence:
    """Whether the unit ``order[k]`` is in the logs that can tell: ``present`` where the log has
    it, not where the log goes from a unit above it to one below it; a borehole that ends before
    the place of the unit in the sequence says nothing and is not among ``ids``."""

    k: int
    unit: str
    ids: list
    xy: np.ndarray
    present: np.ndarray

    @property
    def everywhere(self) -> bool:
        return bool(self.present.all())


def presence(boreholes: list[Borehole], codes: list[str]) -> list[Presence]:
    """For every unit of ``codes`` (in order from the top down) the boreholes that have it and
    those that pass it by (a unit missing from a log: zero thickness there)."""
    pos = {u: k for k, u in enumerate(codes)}
    rows = [dict(ids=[], xy=[], present=[]) for _ in codes]
    for b in boreholes:
        here = {pos[iv.unit] for iv in b.intervals if iv.unit in pos}
        if not here:
            continue
        deepest = max(here)
        for k in range(len(codes)):
            if k in here or deepest > k:  # logged, or passed by on the way to a unit below it
                rows[k]["ids"].append(b.id)
                rows[k]["xy"].append((b.x, b.y))
                rows[k]["present"].append(k in here)
    return [Presence(k, code, r["ids"], np.asarray(r["xy"], float).reshape(-1, 2), np.asarray(r["present"], bool))
            for k, (code, r) in enumerate(zip(codes, rows))]


def summary(chunk) -> str:
    bhs = load(chunk)
    lines = [f"{len(bhs)} boreholes, units from the top down: {', '.join(order(chunk)) or '-'}"]
    for b in bhs:
        z = "z ?" if b.z is None else f"z {b.z:.3f}" + (" (DEM)" if b.z_source == "dem" else "")
        lines.append(f"  {b.id}: {b.x:.3f} {b.y:.3f} {z}, {b.bottom:g} m, {' / '.join(b.units) or '-'}")
    return "\n".join(lines)
