"""Boreholes, cores and trench logs of a chunk.

A borehole is a collar (id, x and y in the chunk's reference system, z of its top, the depth
reached, the kind: auger, core, trench) and its intervals (from and to in metres below the collar,
the unit - a US number or a lithological unit -, a description, a colour, a lithology). They are
read from CSV tables - a collar table and an interval table, or one table with both - whose
columns are recognised by their names in Italian or English as they are written in the field
(``Sigla sondaggio``, ``Quota s.l.m. (m)``, ``Codice US``, ``Da (cm)``...); a name that only guesses
its field (``codice``, ``h``, ``tetto`` and ``letto`` holding elevations...) is taken when nothing
clearer is there, and reported. The intervals may also be given by their elevations
(``quota_tetto``, ``quota_letto``). The numbers are read in the context of their file (decimal
comma or point, thousands separators); a file separated by commas whose numbers have unquoted
decimal commas is put back together where that can be done in one way only. Nothing is left out
without a warning that gives the line of the file.

The unit codes are compared in one form (``US 1``, ``US1`` and ``us 1`` are one unit, and so is
``1`` when a single kind of unit has that number); a unit is shown as it is written most often.
Collars without z take it from the DEM of the chunk. Everything lives in
``chunk.meta["geoarchaeology"]``: the boreholes, the units in their order from the top down with a
colour each; the CSV files read are copied into ``<chunk data>/geoarchaeology/sources/``.

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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .texts import num, plural, t

FOLDER = "geoarchaeology"
KINDS = ("auger", "core", "trench", "section", "test_pit", "other")
TOLERANCE = 0.005  # m: depths that differ less are the same depth (rounding of the tables)

# the column names of each field (normalised: ascii, lowercase, words joined by "_") by how sure
# they are: 3 the name says it, 2 usual but generic, 1 a guess - taken only when nothing clearer is
# there, and reported
NAMES = {
    "id": {3: ["sondaggio", "sondaggi", "id_sondaggio", "nome_sondaggio", "carota", "carotaggio", "trincea",
               "saggio", "pozzetto", "trivellata", "foro", "hole", "hole_id", "holeid", "bhid", "bh", "borehole",
               "borehole_id", "boring", "core", "core_id", "trench", "test_pit", "pit", "location_id", "log_id"],
           2: ["id"],
           1: ["codice", "code", "sigla", "nome", "name", "punto", "point", "station", "location"]},
    "x": {3: ["x", "est", "x_coord", "coord_x", "xcoord", "coordinata_x", "x_utm", "utm_e", "utm_x", "east",
              "easting"], 2: ["e"]},
    "y": {3: ["y", "nord", "y_coord", "coord_y", "ycoord", "coordinata_y", "y_utm", "utm_n", "utm_y", "north",
              "northing"], 2: ["n"]},
    "z": {3: ["z", "quota", "quota_bocca", "quota_testa", "quota_piano", "quota_pc", "quota_p_c",
              "quota_piano_campagna", "quota_terreno", "quota_suolo", "elevation", "elev", "collar_z", "z_collar",
              "collar_elevation", "rl", "collar_rl", "altitude", "alt", "ground_level", "ground_elevation",
              "surface_elevation"],
          1: ["h", "height"]},
    "depth": {3: ["profondita", "profondita_totale", "profondita_raggiunta", "profondita_max", "profondita_foro",
                  "prof", "prof_max", "prof_totale", "prof_raggiunta", "fondo_foro", "depth", "total_depth",
                  "max_depth", "final_depth", "hole_depth", "eoh", "td"]},
    "kind": {3: ["tipo", "tipologia", "metodo", "strumento", "tecnica", "kind", "type", "method", "tool"]},
    "from": {3: ["da", "dal", "prof_da", "profondita_da", "da_prof", "inizio", "profondita_tetto", "prof_tetto",
                 "from", "depth_from", "from_depth", "top_depth", "start"]},
    "to": {3: ["a", "al", "prof_a", "profondita_a", "a_prof", "fine", "profondita_letto", "prof_letto",
               "profondita_base", "prof_base", "to", "depth_to", "to_depth", "bottom_depth", "end"]},
    "top_z": {3: ["quota_tetto", "tetto_quota", "quota_da", "da_quota", "quota_sup", "quota_superiore", "z_tetto",
                  "q_tetto", "top_z", "z_top", "z_from", "from_z", "top_elevation", "top_rl", "rl_top", "rl_from"]},
    "base_z": {3: ["quota_letto", "letto_quota", "quota_base", "base_quota", "quota_a", "a_quota", "quota_inf",
                   "quota_inferiore", "quota_fondo", "z_letto", "q_letto", "base_z", "bottom_z", "z_base", "z_bottom",
                   "z_to", "to_z", "bottom_elevation", "base_elevation", "bottom_rl", "base_rl", "rl_bottom",
                   "rl_to"]},
    "unit": {3: ["us", "usm", "us_usm", "unita", "unita_stratigrafica", "unita_litologica", "codice_unita",
                 "strato", "orizzonte", "su", "unit", "unit_code", "unit_id", "us_code", "context", "layer",
                 "stratum", "horizon", "facies", "formation", "litho_unit"],
             2: ["livello"],
             1: ["codice", "code", "sigla"]},
    "description": {3: ["descrizione", "desc", "descr", "description"],
                    2: ["note", "osservazioni", "commento", "commenti", "notes", "remarks", "comment", "comments"]},
    "colour": {3: ["colore", "colore_munsell", "munsell", "colour", "color", "munsell_colour", "munsell_color",
                   "colour_munsell", "color_munsell"]},
    "lithology": {3: ["litologia", "lito", "materiale", "tessitura", "granulometria", "lithology", "lith", "material",
                      "texture"]},
}
# depths or elevations, as the values tell: the top of an interval and its base
AMBIGUOUS = {"tetto": ("from", "top_z"), "top": ("from", "top_z"), "letto": ("to", "base_z"),
             "base": ("to", "base_z"), "bottom": ("to", "base_z")}
_LOOKUP: dict[str, list[tuple[str, int]]] = defaultdict(list)
for _field, _by in NAMES.items():
    for _prio, _names in _by.items():
        for _name in _names:
            _LOOKUP[_name].append((_field, _prio))
ALIASES = {f: sorted({n for names in by.values() for n in names}
                     | {n for n, fs in AMBIGUOUS.items() if f in fs}) for f, by in NAMES.items()}
NUMERIC = ("x", "y", "z", "depth", "from", "to", "top_z", "base_z")
TEXT = ("id", "unit", "description", "colour", "lithology", "kind")
# a unit of length at the end of a name: the values are converted to metres
UNIT_TOKENS = {"m": 1.0, "mt": 1.0, "mts": 1.0, "metro": 1.0, "metri": 1.0, "metre": 1.0, "metres": 1.0,
               "meter": 1.0, "meters": 1.0, "cm": 0.01, "mm": 0.001}
# what an elevation is measured from, at the end of a name: dropped (Quota s.l.m. is the elevation)
ELEVATION_TAILS = (("m", "s", "l", "m"), ("s", "l", "m"), ("slm",), ("m", "a", "s", "l"), ("a", "s", "l"),
                   ("asl",), ("amsl",), ("msl",), ("assoluta",), ("assoluto",), ("ass",), ("abs",), ("absolute",))
# words around the borehole or unit word that do not change it: "Codice US", "N. US", "Sigla sondaggio"
QUALIFIERS = {"n", "nr", "no", "num", "numero", "number", "cod", "codice", "code", "sigla", "id", "nome", "name"}
# the borehole or the unit a field belongs to: "Quota sondaggio", "Descrizione US", "Tipo sondaggio"
SUBJECTS = {"sondaggio", "sondaggi", "foro", "carota", "carotaggio", "hole", "borehole", "us", "unita", "strato",
            "layer", "unit", "intervallo", "interval"}
LABELS = {"id": ("sondaggio", "borehole"), "x": ("x", "x"), "y": ("y", "y"), "z": ("quota", "elevation"),
          "depth": ("profondità raggiunta", "depth reached"), "kind": ("tipo", "kind"), "from": ("da", "from"),
          "to": ("a", "to"), "top_z": ("quota del tetto", "elevation of the top"),
          "base_z": ("quota della base", "elevation of the base"), "unit": ("unità", "unit"),
          "description": ("descrizione", "description"), "colour": ("colore", "colour"),
          "lithology": ("litologia", "lithology")}
_NA = {"-", "--", "na", "n/a", "n.a.", "nd", "n.d.", "n/d", "nan", "null", "none", "?"}
# colours of the units without one of their own: earthy, light and dark in turn
PALETTE = ["#d9b77e", "#7d5a3c", "#c9a55c", "#a0522d", "#e6d3a3", "#6b6b4b", "#c27c4c", "#8c9a6b",
           "#b0846a", "#596e79", "#e0c08a", "#704f3c", "#a39e7a", "#8a5a6b", "#cfa98a", "#4f5d4f"]


def label(field_name: str) -> str:
    """The name of a field in the language of the interface."""
    it, en = LABELS.get(field_name, (field_name, field_name))
    return t(it, en)


@dataclass
class Interval:
    """A stretch of a log: ``top`` and ``base`` in metres below the collar. ``unit`` is the code
    of its unit as shown (one form for the spellings of one unit), ``unit_raw`` the code as
    written in the table when it is another spelling."""

    top: float
    base: float
    unit: str
    description: str = ""
    colour: str = ""
    lithology: str = ""
    unit_raw: str = ""

    def to_dict(self) -> dict:
        d = {"from": round(float(self.top), 4), "to": round(float(self.base), 4), "unit": self.unit}
        for k in ("description", "colour", "lithology", "unit_raw"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Interval":
        return cls(float(d["from"]), float(d["to"]), str(d["unit"]), str(d.get("description") or ""),
                   str(d.get("colour") or ""), str(d.get("lithology") or ""), str(d.get("unit_raw") or ""))


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
    """The boreholes of the chunk. A project of an older version gets its unit codes in one form
    here, once."""
    holes = [Borehole.from_dict(d) for d in (chunk.meta.get("geoarchaeology") or {}).get("boreholes", [])]
    if holes:
        before = [(iv.unit, iv.unit_raw) for b in holes for iv in b.intervals]
        canonicalise(holes)
        if [(iv.unit, iv.unit_raw) for b in holes for iv in b.intervals] != before:
            save(chunk, holes)
    return holes


def get(chunk, bid: str) -> Borehole | None:
    return next((b for b in load(chunk) if b.id == str(bid)), None)


def save(chunk, boreholes: list[Borehole]) -> None:
    m = meta(chunk)
    m["boreholes"] = [b.to_dict() for b in boreholes]
    _refresh_units(chunk, boreholes)
    if m.get("surfaces"):
        m["surfaces"]["stale"] = True  # built from other logs: to build again
    if (m.get("implicit") or {}).get("made"):
        m["implicit"]["stale"] = True


# ------------------------------------------------------------------ reading the tables: the columns
def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _strip_tails(tokens: list[str]) -> tuple[list[str], float]:
    """The words of a column name without what follows the field (``(cm)``, ``s.l.m.``) and the
    factor of the unit of length named there."""
    factor = None
    tokens = list(tokens)
    while len(tokens) > 1:
        tail = next((tl for tl in ELEVATION_TAILS if len(tokens) > len(tl) and tuple(tokens[-len(tl):]) == tl), None)
        if tail is not None:
            tokens = tokens[:-len(tail)]
        elif tokens[-1] in UNIT_TOKENS and factor is None:
            factor = UNIT_TOKENS[tokens.pop()]
        else:
            break
    return tokens, 1.0 if factor is None else factor


def _field_candidates(name: str) -> list[tuple[str, int, float, bool]]:
    """The fields a column may hold: ``(field, how sure 1-3, factor to metres, depth or elevation
    by the values)``."""
    s = normalise(name)
    if not s:
        return []
    tokens, factor = _strip_tails(s.split("_"))
    base = "_".join(tokens)
    if base in AMBIGUOUS:
        depth, elevation = AMBIGUOUS[base]
        return [(depth, 2, factor, True), (elevation, 2, factor, True)]
    if base in _LOOKUP:
        return [(f, p, factor, False) for f, p in _LOOKUP[base]]
    core = list(tokens)  # "Codice US", "N. US", "US n.", "Sigla sondaggio": the borehole or the unit
    while len(core) > 1 and core[0] in QUALIFIERS:
        core = core[1:]
    while len(core) > 1 and core[-1] in QUALIFIERS:
        core = core[:-1]
    if core != tokens:
        hits = [f for f, p in _LOOKUP.get("_".join(core), []) if f in ("id", "unit") and p >= 2]
        if hits:
            return [(f, 3, factor, False) for f in hits]
    core = list(tokens)  # "Quota sondaggio", "Descrizione US", "Tipo sondaggio": a field of one of them
    while len(core) > 1 and core[-1] in SUBJECTS:
        core = core[:-1]
    if core != tokens:
        hits = [(f, p) for f, p in _LOOKUP.get("_".join(core), []) if f not in ("id", "unit")]
        if hits:
            return [(f, p, factor, False) for f, p in hits]
    return []


def _depth_or_elevation(header, cands, rows, notes: list) -> None:
    """``tetto``, ``letto``, ``top``, ``bottom`` and ``base``: depths or elevations, decided by
    the other columns (a ``da`` beside ``tetto`` makes it an elevation) or by the values (the base
    of an interval under its top: elevations). ``cands`` changed in place."""
    amb = [k for k, cs in enumerate(cands) if cs and cs[0][3]]
    if not amb:
        return
    strong = {f for cs in cands for f, p, _fa, a in cs if not a and p >= 2}
    kind = {}  # column -> "depth" | "elevation"
    for k in amb:
        depth, elevation = cands[k][0][0], cands[k][1][0]
        if depth in strong and elevation not in strong:
            kind[k] = "elevation"
        elif elevation in strong and depth not in strong:
            kind[k] = "depth"
    for k in amb:  # by the other end of the interval
        if k in kind:
            continue
        depth = cands[k][0][0]
        other_depth, other_elevation = ("to", "base_z") if depth == "from" else ("from", "top_z")
        if other_depth in strong and other_elevation not in strong:
            kind[k] = "depth"
        elif other_elevation in strong and other_depth not in strong:
            kind[k] = "elevation"
    undecided = [k for k in amb if k not in kind]
    tops = [k for k in undecided if cands[k][0][0] == "from"]
    bases = [k for k in undecided if cands[k][0][0] == "to"]
    why = ""
    if tops and bases:  # by the values: the base under the top is an elevation
        a, b = tops[0], bases[0]
        n = down = 0
        for row in rows:
            va = parse_number(row[a]) if a < len(row) else None
            vb = parse_number(row[b]) if b < len(row) else None
            if va is not None and vb is not None and vb != va:
                n += 1
                down += vb < va
        guess = "elevation" if n and down > n / 2 else "depth"
        for k in tops + bases:
            kind[k] = guess
        why = t(" (la base è sotto il tetto in {0} righe su {1})", " (the base is below the top in {0} rows of {1})") \
            .format(down, n) if guess == "elevation" else \
            t(" (la base è più profonda del tetto in {0} righe su {1})",
              " (the base is deeper than the top in {0} rows of {1})").format(n - down, n)
    for k in amb:
        kind.setdefault(k, next((kind[j] for j in amb if j in kind), "depth"))
    for what in ("elevation", "depth"):
        cols = [k for k in amb if kind[k] == what]
        if cols:
            names = ", ".join(f"'{header[k]}'" for k in cols)
            notes.append((t("{0} lette come quote", "{0} read as elevations") if what == "elevation" else
                          t("{0} lette come profondità", "{0} read as depths")).format(names)
                         + (why if (set(cols) & set(tops + bases)) else ""))
    for k in amb:
        depth, elevation = cands[k][0][0], cands[k][1][0]
        cands[k] = [(elevation if kind[k] == "elevation" else depth, 2, cands[k][0][2], False)]


def _assign(header, cands, notes: list) -> tuple[dict, dict]:
    """The column of every field: the surest names first, the leftmost among equals; a column
    serves one field. Returns ``{field: (column, factor)}`` and ``{field: header}``; the guesses
    and the columns not used are noted."""
    options = sorted(((p, k, f, factor) for k, cs in enumerate(cands) for f, p, factor, _a in cs),
                     key=lambda o: (-o[0], o[1]))
    columns, names, used = {}, {}, set()
    for p, k, f, factor in options:
        if f in columns or k in used:
            continue
        columns[f] = (k, factor)
        names[f] = header[k]
        used.add(k)
        if p <= 1:
            notes.append(t("colonna '{0}' presa come {1}", "column '{0}' taken as the {1}").format(header[k], label(f)))
    unused = [h for k, h in enumerate(header) if k not in used and h.strip()]
    if unused:
        text = t("colonna non usata: {0}", "column not used: {0}") if len(unused) == 1 else \
            t("colonne non usate: {0}", "columns not used: {0}")
        notes.append(text.format(", ".join(f"'{h}'" for h in unused)))
    return columns, names


# ------------------------------------------------------------------ reading the tables: the rows
def _read_rows(path: str | Path) -> tuple[list[str], list[list[str]], list[int], str | None]:
    """Header, rows, the line of the file of every row, and the separator of a delimited text
    file (comma, semicolon, tab or bar; whitespace for a table without spaces in its values), in
    UTF-8 or Windows-1252. Empty lines and lines starting with ``#`` are skipped, and counted."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    content = [(n, ln) for n, ln in enumerate(text.splitlines(), start=1)
               if ln.strip() and not ln.lstrip().startswith("#")]
    if not content:
        return [], [], [], None
    sample = "\n".join(ln for _n, ln in content[:50])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: content[0][1].count(d) for d in (";", "\t", ",", "|")}
        delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    records: list[tuple[int, list[str]]] = []
    if delimiter is None:
        records = [(n, ln.split()) for n, ln in content]
    else:
        reader = csv.reader(io.StringIO("\n".join(ln for _n, ln in content)), delimiter=delimiter)
        start = 0
        for rec in reader:
            first, start = start, reader.line_num
            if any(c.strip() for c in rec):
                records.append((content[min(first, len(content) - 1)][0], [c.strip() for c in rec]))
    if not records:
        return [], [], [], delimiter
    header = [h.strip() for h in records[0][1]]
    return header, [r for _n, r in records[1:]], [n for n, _r in records[1:]], delimiter


def read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a delimited text file (see :func:`_read_rows`)."""
    header, rows, _lines, _d = _read_rows(path)
    return header, rows


_INT = re.compile(r"[-+]?\d+")
_DIGITS = re.compile(r"\d+")


def _splits(tokens: list[str], kinds: list[str], delimiter: str | None, cap: int = 2000) -> list[tuple]:
    """The ways a row with more fields than the header can be put back together: a number split
    by its decimal comma (or by the commas of its thousands) in a file separated by commas, a
    text with the separator inside it, the three numbers of a colour. Returns ``(cost, cells,
    numbers joined)``: the cost counts the separators kept inside texts."""
    n_cols, n_tok = len(kinds), len(tokens)
    glue = {",": ", ", ";": "; ", "|": " | "}.get(delimiter, delimiter or " ")  # the separator given back to a text
    out: list[tuple] = []

    def go(i: int, k: int, cells: list, cost: int, joined: bool) -> None:
        if len(out) > cap:
            return
        if k == n_cols:
            if i == n_tok:
                out.append((cost, tuple(cells), joined))
            return
        room = n_tok - i - (n_cols - k - 1)
        if room < 1:
            return
        kind = kinds[k]
        options = [(1, tokens[i], 0, False)]
        if kind == "num" and delimiter == ",":
            if room >= 2 and _INT.fullmatch(tokens[i]) and _DIGITS.fullmatch(tokens[i + 1]):
                options.append((2, tokens[i] + "," + tokens[i + 1], 0, True))
            for take in range(3, room + 1):  # 4,500,810.5: thousands separated by the delimiter
                group = tokens[i:i + take]
                if (re.fullmatch(r"[-+]?\d{1,3}", group[0]) and all(re.fullmatch(r"\d{3}", g) for g in group[1:-1])
                        and re.fullmatch(r"\d{3}(\.\d+)?", group[-1])):
                    options.append((take, ",".join(group), 0, True))
        elif kind == "colour" and room >= 3 and all(re.fullmatch(r"\d{1,3}", g) for g in tokens[i:i + 3]):
            options.append((3, ",".join(tokens[i:i + 3]), 0, False))
        elif kind == "free":
            options += [(take, glue.join(tokens[i:i + take]), take - 1, False) for take in range(2, room + 1)]
        for take, cell, c, j in options:
            go(i + take, k + 1, cells + [cell], cost + c, joined or j)

    go(0, 0, [], 0, False)
    return out


def _repair(name_cols: dict, n_header: int, rows: list, lines: list, delimiter: str | None,
            notes: list) -> tuple[list, list]:
    """The rows with more fields than the header put back together when that can be done in one
    way only (see :func:`_splits`); the others left out, with their lines."""
    field_of = {k: f for f, (k, _fa) in name_cols.items()}
    kinds = []
    for k in range(n_header):
        f = field_of.get(k)
        kinds.append("num" if f in NUMERIC else "colour" if f == "colour" else
                     "fixed" if f in ("id", "unit", "kind") else "free")
    keep_rows, keep_lines = [], []
    numbers, texts, refused = [], [], []
    for row, line in zip(rows, lines):
        if len(row) <= n_header:
            keep_rows.append(row)
            keep_lines.append(line)
            continue
        found = _splits(row, kinds, delimiter) if len(row) - n_header <= 20 else []
        if found:
            best = min(c for c, _cells, _j in found)
            unique = {cells: j for c, cells, j in found if c == best}
            if len(unique) == 1:
                (cells, joined), = unique.items()
                keep_rows.append(list(cells))
                keep_lines.append(line)
                (numbers if joined else texts).append(line)
                continue
        refused.append((line, len(row), bool(found)))
    if numbers:
        notes.append(plural(len(numbers), t("virgola decimale in un file separato da virgole: {0} riga ricomposta",
                                            "decimal comma in a file separated by commas: {0} row put back together"),
                            t("virgola decimale in un file separato da virgole: {0} righe ricomposte",
                              "decimal comma in a file separated by commas: {0} rows put back together"))
                     + " (" + _lines_text(numbers) + ")")
    if texts:
        notes.append(plural(len(texts), t("il separatore dentro un campo senza virgolette: {0} riga ricomposta",
                                          "the separator inside a field without quotes: {0} row put back together"),
                            t("il separatore dentro un campo senza virgolette: {0} righe ricomposte",
                              "the separator inside a field without quotes: {0} rows put back together"))
                     + " (" + _lines_text(texts) + ")")
    for line, n, ambiguous in refused:
        notes.append((t("riga {0}: {1} campi invece di {2} e più di un modo di ricomporli: lasciata fuori (separa le "
                        "colonne con il punto e virgola, o metti i numeri con la virgola fra virgolette)",
                        "line {0}: {1} fields instead of {2} and more than one way to put them back together: left "
                        "out (separate the columns with semicolons, or quote the numbers with a decimal comma)")
                      if ambiguous else
                      t("riga {0}: {1} campi invece di {2}: lasciata fuori",
                        "line {0}: {1} fields instead of {2}: left out")).format(line, n, n_header))
    return keep_rows, keep_lines


def _lines_text(lines: list[int], most: int = 8) -> str:
    """``riga 3`` or ``righe 3, 7, 9``."""
    shown = ", ".join(str(n) for n in lines[:most]) + (", ..." if len(lines) > most else "")
    return (t("riga {0}", "line {0}") if len(lines) == 1 else t("righe {0}", "lines {0}")).format(shown)


# ------------------------------------------------------------------ reading the tables: the numbers
_GROUP_SPACE = re.compile(r"(?<=\d)[\s   '’](?=\d)")


def _clean_number(text) -> str:
    s = str(text).strip()
    return _GROUP_SPACE.sub("", s).replace(" ", "")


def decimal_mark(cells) -> str | None:
    """The decimal separator of a table, from its numbers: ``","``, ``"."`` or None when they do
    not tell (``1.234`` alone could be either)."""
    votes = {".": 0, ",": 0}
    for cell in cells:
        s = _clean_number(cell)
        if not re.fullmatch(r"[-+]?[\d.,]*\d[\d.,]*", s):
            continue
        has_dot, has_comma = "." in s, "," in s
        if has_dot and has_comma:
            votes[s[max(s.rfind("."), s.rfind(","))]] += 1
        elif has_dot or has_comma:
            sep = "." if has_dot else ","
            if s.count(sep) > 1:
                votes["," if sep == "." else "."] += 1  # separates thousands: the other one is the decimal
            elif len(s.rsplit(sep, 1)[1]) != 3:
                votes[sep] += 1
    if votes["."] == votes[","]:
        return None
    return "." if votes["."] > votes[","] else ","


def _grouped(int_part: str, sep: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d{1,3}(?:" + re.escape(sep) + r"\d{3})+", int_part))


def parse_number(text, decimal: str | None = None) -> float | None:
    """A number written with a decimal point or comma and maybe thousands separators (``"12,5"``,
    ``"1.234,5"``, ``"4,500,810.5"``, ``"4 500 810"``), None when empty or not a number.
    ``decimal``: the decimal separator of the file (:func:`decimal_mark`), which settles
    ``"1.234"``; without it a lone separator is the decimal one."""
    if text is None:
        return None
    s = _clean_number(text)
    if not s or s.lower() in _NA:
        return None
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        dec = s[max(s.rfind("."), s.rfind(","))]
        other = "," if dec == "." else "."
        int_part, frac = s.rsplit(dec, 1)
        if other in frac or not _grouped(int_part, other):
            return None
        s = int_part.replace(other, "") + "." + frac
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        if s.count(sep) > 1:
            if not _grouped(s, sep):
                return None
            s = s.replace(sep, "")
        elif decimal is not None and sep != decimal and _grouped(s, sep):
            s = s.replace(sep, "")  # 1.234 in a file with a decimal comma: thousands
        else:
            s = s.replace(sep, ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def read_table(path: str | Path, notes: list | None = None, extra: dict | None = None) -> tuple[list[dict], dict]:
    """The rows of a table as dictionaries of fields (numbers parsed in the context of the file,
    lengths in metres, ``_line`` the line of the file, ``_bad`` the cells that are not numbers)
    and the fields found (``{"id": "Sondaggio", ...}``: field -> column). ``notes`` receives what
    was guessed or put back together. ``extra``: numeric fields of this table, ``{field: names}``
    with the column names normalised."""
    notes = notes if notes is not None else []
    header, rows, lines, delimiter = _read_rows(path)
    extra = extra or {}
    cands = [_field_candidates(h) + [(f, 3, 1.0, False) for f, known in extra.items() if normalise(h) in known]
             for h in header]
    short = {normalise(h) for h in header}
    if not {"e", "n"} <= short:  # E and N alone are a guess, together the coordinates
        cands = [[(f, 1 if normalise(h) in ("e", "n") else p, fa, a) for f, p, fa, a in cs]
                 for h, cs in zip(header, cands)]
    _depth_or_elevation(header, cands, rows, notes)
    columns, names = _assign(header, cands, notes)
    rows, lines = _repair(columns, len(header), rows, lines, delimiter, notes)
    decimal = decimal_mark(row[k] for f, (k, _fa) in columns.items() if f in NUMERIC or f in extra
                           for row in rows if k < len(row))
    out = []
    for row, line in zip(rows, lines):
        rec: dict = {"_line": line}
        for key, (k, factor) in columns.items():
            value = row[k].strip() if k < len(row) else ""
            if key in TEXT:
                rec[key] = value
                continue
            v = parse_number(value, decimal)
            if v is None and value and value.lower() not in _NA:
                rec.setdefault("_bad", []).append((key, value))
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
        notes: list[str] = []
        rows, names = read_table(path, notes)
        name = Path(path).name
        warnings += [f"{name}: {n}" for n in notes]
        if "id" not in names:
            raise ValueError(t("{0}: manca la colonna del sondaggio (id, sondaggio, hole...)",
                               "{0}: no column of the borehole (id, sondaggio, hole...)").format(name))
        has_xy = "x" in names and "y" in names
        has_iv = ("from" in names and "to" in names) or ("top_z" in names and "base_z" in names)
        if not has_xy and not has_iv:
            raise ValueError(t("{0}: né coordinate (x, y) né intervalli (da, a / from, to)",
                               "{0}: neither coordinates (x, y) nor intervals (from, to)").format(name))
        kinds.append("combined" if has_xy and has_iv else "collars" if has_xy else "intervals")
        in_table, placed, bad_xy = [], set(), set()
        for rec in rows:
            n = rec["_line"]
            bid = rec.get("id") or ""
            if not bid:
                if any(v not in (None, "") for k, v in rec.items() if not k.startswith("_") and k != "id") \
                        or rec.get("_bad"):
                    warnings.append(t("{0}: riga {1} senza il codice del sondaggio: lasciata fuori",
                                      "{0}: line {1} without the code of the borehole: left out").format(name, n))
                continue
            for key, raw in rec.get("_bad", []):
                warnings.append(t("{0}: {1}: {2} '{3}' non è un numero (riga {4})",
                                  "{0}: {1}: {2} '{3}' is not a number (line {4})")
                                .format(name, bid, label(key), raw, n))
                if key in ("x", "y"):
                    bad_xy.add(bid)
            if has_xy:
                if bid not in in_table:
                    in_table.append(bid)
                if rec.get("x") is not None and rec.get("y") is not None:
                    placed.add(bid)
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
                    if any(rec.get(k) not in (None, "") for k in ("from", "to", "top_z", "base_z", "unit",
                                                                   "description", "colour", "lithology")):
                        pair = ("from", "to") if "from" in names else ("top_z", "base_z")
                        missing = t(" e ", " and ").join(f"'{names[k]}'" for k in pair
                                                         if k in names and rec.get(k) is None)
                        warnings.append(t("{0}: {1}: riga {2} senza {3}: lasciata fuori",
                                          "{0}: {1}: line {2} without {3}: left out").format(name, bid, n, missing))
                    continue
                intervals.setdefault(bid, []).append({
                    "top": top, "base": base, "top_z": ztop, "base_z": zbase, "unit": rec.get("unit", ""),
                    "description": rec.get("description", ""), "colour": rec.get("colour", ""),
                    "lithology": rec.get("lithology", ""), "line": n, "file": name})
        for bid in in_table:
            if bid not in placed and bid not in collars and bid not in bad_xy:
                warnings.append(t("{0}: {1} senza x e y: lasciato fuori", "{0}: {1} without x and y: left out")
                                .format(name, bid))
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
    replaced and the others kept. The unit codes of all the boreholes are put in one form. Returns
    a report: boreholes, intervals, units, the collars whose z came from the DEM, the difference
    between the given z and the DEM, the warnings (with the lines of the files)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    paths = [Path(p) for p in paths]
    progress(0.05, "Reading the tables")
    collars, ivs, kinds, warnings = read_tables(paths)
    if not collars:
        raise ValueError(t("Nessun sondaggio con le coordinate nei file letti",
                           "No borehole with coordinates in the files read")
                         + "".join("\n- " + w for w in warnings[:10]))
    for bid in sorted(set(ivs) - set(collars)):
        warnings.append(t("{0}: intervalli senza la posizione del sondaggio: lasciati fuori",
                          "{0}: intervals without the position of the borehole: left out").format(bid))
    dem = dem_path(chunk)
    ids = list(collars)
    xy = np.array([[collars[i]["x"], collars[i]["y"]] for i in ids], float)
    dem_z = raster_values(dem, xy[:, 0], xy[:, 1]) if dem is not None else np.full(len(ids), np.nan)
    progress(0.4, "Collars and intervals")
    new, no_z, from_dem = [], [], 0
    unreadable: dict[tuple[str, str], list[int]] = {}
    munsell_without_library = False
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
            colour = rec["colour"]
            if colour and parse_colour(colour) is None:
                if munsell_notation(colour) and not has_colour_science():
                    munsell_without_library = True
                else:
                    unreadable.setdefault((rec["file"], colour), []).append(rec["line"])
            b.intervals.append(Interval(float(top), float(base), unit, rec["description"], colour, rec["lithology"]))
        _check_log(b, warnings)
        new.append(b)
    for (name, colour), lines in unreadable.items():
        warnings.append(t("{0}: colore '{1}' non riconosciuto ({2}): non usato",
                          "{0}: colour '{1}' not recognised ({2}): not used").format(name, colour, _lines_text(lines)))
    if munsell_without_library:
        warnings.append(t("colori Munsell non convertiti: manca la libreria colour-science (le unità prendono i colori "
                          "della tavolozza)", "Munsell colours not converted: the colour-science library is missing "
                          "(the units take the colours of the palette)"))
    progress(0.7, "Into the project")
    old = [] if replace else [b for b in load(chunk) if b.id not in {n.id for n in new}]
    boreholes = old + new
    merged, ambiguous = canonicalise(boreholes)
    for shown, others in merged:
        warnings.append(t("unità scritte in modi diversi, lette come una sola: {0} ({1})",
                          "units written in different ways, read as one: {0} ({1})").format(shown, ", ".join(others)))
    for code, candidates in ambiguous:
        warnings.append(t("unità '{0}': potrebbe essere {1}, tenuta a parte",
                          "unit '{0}': it could be {1}, kept apart").format(code, t(" o ", " or ").join(candidates)))
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
            if unit_norm(iv.unit) != unit_norm(keep[-1].unit):  # between two units: their contact in the middle
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


# ------------------------------------------------------------------ the unit codes in one form
_US_CODE = re.compile(r"^(US[A-Z]{0,2}|SU)\s*(?:(?:NR|NUM|NO|N)\s*\.?\s*|#\s*)?[-_.:]?\s*(\d.*)$", re.IGNORECASE)


def _token(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    return re.sub(r"^0+(?=\d)", "", s)


def split_unit(code) -> tuple[str | None, str]:
    """``(kind, number)`` of a unit code: ``"US 12"``, ``"us12"``, ``"US n. 012"`` -> ("US", "12");
    ``"USM3"`` -> ("USM", "3"); ``"SU 5"`` -> ("US", "5"); ``"12"`` -> (None, "12"); ``"Sabbia
    gialla"`` -> (None, "sabbia gialla")."""
    s = unicodedata.normalize("NFKD", str(code)).encode("ascii", "ignore").decode().strip()
    m = _US_CODE.match(s)
    if m:
        kind = m.group(1).upper()
        return ("US" if kind == "SU" else kind), _token(m.group(2))
    low = s.lower()
    if re.fullmatch(r"\d+\s*[a-z]?", low):
        return None, _token(low)
    return None, re.sub(r"[^a-z0-9]+", " ", low).strip()


def unit_norm(code) -> str:
    """The form in which unit codes are compared: ``"US 12"`` for US, USM... (any spelling),
    ``"# 12"`` for a plain number, ``"~sabbiagialla"`` for any other code."""
    kind, tok = split_unit(code)
    if kind:
        return f"{kind} {tok}"
    if re.fullmatch(r"\d+[a-z]?", tok):
        return f"# {tok}"
    return "~" + tok.replace(" ", "")


def unit_spellings(codes: list[str]) -> tuple[dict, list, list]:
    """For the unit codes as written (repeated as often as they occur, in the order met): the code
    each one is shown as (the spelling written most often among those of one unit), the spellings
    read as one unit ``[(shown, [others])]`` and the plain numbers that two kinds of unit share
    ``[(number, [the units])]`` - kept apart. A plain number is the unit of the one kind that has
    that number (``1`` is ``US 1``)."""
    counts = Counter(codes)
    first: dict[str, int] = {}
    for i, c in enumerate(codes):
        first.setdefault(c, i)
    key_of = {c: unit_norm(c) for c in counts}
    kinds_of: dict[str, set] = defaultdict(set)
    for c in counts:
        kind, tok = split_unit(c)
        if kind:
            kinds_of[tok].add(kind)
    shared = []
    for c, key in key_of.items():
        if key.startswith("# "):
            tok = key[2:]
            kinds = kinds_of.get(tok, set())
            if len(kinds) == 1:
                key_of[c] = f"{next(iter(kinds))} {tok}"
            elif len(kinds) > 1:
                shared.append((c, [f"{k} {tok}" for k in sorted(kinds)]))
    groups: dict[str, list] = defaultdict(list)
    for c in sorted(counts, key=first.get):
        groups[key_of[c]].append(c)
    shown, merged = {}, []
    for members in groups.values():
        best = max(members, key=lambda c: (counts[c], -first[c]))
        for c in members:
            shown[c] = best
        if len(members) > 1:
            merged.append((best, [c for c in members if c != best]))
    shown_of_key = {key_of[c]: shown[c] for c in counts}
    shared = [(c, [shown_of_key.get(unit_norm(u), u) for u in units]) for c, units in shared]
    return shown, merged, shared


def canonicalise(boreholes: list[Borehole]) -> tuple[list, list]:
    """The unit codes of the intervals in one form (see :func:`unit_spellings`), the code as
    written kept in ``unit_raw``. Returns the spellings merged and the plain numbers kept apart."""
    raw = [iv.unit_raw or iv.unit for b in boreholes for iv in sorted(b.intervals, key=lambda i: i.top)]
    shown, merged, shared = unit_spellings(raw)
    for b in boreholes:
        for iv in b.intervals:
            written = iv.unit_raw or iv.unit
            code = shown.get(written, written)
            iv.unit, iv.unit_raw = code, ("" if code == written else written)
    return merged, shared


def resolve_unit(name, codes) -> str | None:
    """The unit of ``codes`` that ``name`` means, in any spelling (``"us2"`` is ``"US 2"``), a plain
    number meaning the one unit of any kind with that number and the other way round; None when
    there is none or more than one."""
    codes = list(codes)
    key = unit_norm(name)
    same = [c for c in codes if unit_norm(c) == key]
    if len(same) == 1:
        return same[0]
    kind, tok = split_unit(name)
    if kind is None and key.startswith("# "):
        same = [c for c in codes if split_unit(c)[0] is not None and split_unit(c)[1] == tok]
    elif kind is not None:
        same = [c for c in codes if unit_norm(c) == f"# {tok}"]
    else:
        return None
    return same[0] if len(same) == 1 else None


# ------------------------------------------------------------------ units and their order
def unit_order(boreholes: list[Borehole]) -> tuple[list[str], list[str]]:
    """The units from the top down: a unit lying directly above another in a log comes before
    it; the disagreements between logs go to the majority, and a circle of them is broken by
    the mean depth of the units. Returns the order and the conflicts found."""
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
            conflicts.append(t("{0} sopra {1} in {2}, sotto in {3}", "{0} above {1} in {2}, below in {3}").format(
                a, c, plural(n, t("{0} sondaggio", "{0} borehole"), t("{0} sondaggi", "{0} boreholes")), back))
        if c not in succ[a]:
            succ[a].add(c)
            indeg[c] += 1
    order_, left = [], set(seen)
    while left:
        ready = [u for u in left if indeg[u] == 0]
        if not ready:  # a circle of disagreements: the shallowest unit first
            u = min(left, key=lambda v: (mean[v], seen.index(v)))
            conflicts.append(t("ordine circolare fra le unità: {0} messa prima per la profondità media",
                               "circular order among the units: {0} put first by its mean depth").format(u))
        else:
            u = min(ready, key=lambda v: (mean[v], seen.index(v)))
        order_.append(u)
        left.discard(u)
        for c in succ.pop(u, set()):
            indeg[c] -= 1
    return order_, conflicts


def order(chunk) -> list[str]:
    return [u["code"] for u in meta(chunk).get("units", [])]


def units(chunk) -> list[dict]:
    """The units in their order: ``{"code", "colour", "description", "boreholes"}``."""
    return [dict(u) for u in meta(chunk).get("units", [])]


def set_order(chunk, codes) -> list[str]:
    """The order of the units given by hand, in any spelling of their codes (unknown codes
    ignored, the missing ones after, in the order of the logs); kept until the boreholes change
    units."""
    have = order(chunk)
    picked: list[str] = []
    for c in codes:
        u = resolve_unit(str(c).strip(), have)
        if u is not None and u not in picked:
            picked.append(u)
    codes = picked + [c for c in have if c not in picked]
    m = meta(chunk)
    by = {u["code"]: u for u in m.get("units", [])}
    m["units"] = [by[c] for c in codes]
    m["order_manual"] = True
    if m.get("surfaces") and codes != have:
        m["surfaces"]["stale"] = True  # built in another order
    if (m.get("implicit") or {}).get("made") and codes != have:
        m["implicit"]["stale"] = True
    return codes


def _refresh_units(chunk, boreholes: list[Borehole]) -> None:
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
        given = Counter(iv.colour for iv in ivs if iv.colour).most_common()
        colour = next((hx for hx in (parse_colour(c) for c, _n in given) if hx), None)
        colour = colour or old_colours.get(code) or PALETTE[k % len(PALETTE)]
        lith = Counter(iv.lithology for iv in ivs if iv.lithology).most_common(1)
        out.append({"code": code, "colour": colour, "description": desc[0][0] if desc else "",
                    "lithology": lith[0][0] if lith else "",
                    "boreholes": len({b.id for b in boreholes if any(iv.unit == code for iv in b.intervals)})})
    m["units"] = out


def colours(chunk) -> dict[str, str]:
    return {u["code"]: u.get("colour") or PALETTE[k % len(PALETTE)] for k, u in enumerate(meta(chunk).get("units", []))}


# ------------------------------------------------------------------ colours
_HEX = re.compile(r"#?[0-9a-fA-F]{6}")
_RGB_TRIPLE = re.compile(r"\s*(\d{1,3})\s*[,; ]\s*(\d{1,3})\s*[,; ]\s*(\d{1,3})\s*")
_MUNSELL = re.compile(r"(?<![A-Za-z0-9])(?:(\d+(?:[.,]\d+)?)\s*(YR|GY|BG|PB|RP|R|Y|G|B|P)\s*(\d+(?:[.,]\d+)?)\s*/\s*"
                      r"(\d+(?:[.,]\d+)?)|N\s*(\d+(?:[.,]\d+)?)\s*(?:/\s*0?)?)(?![A-Za-z0-9])", re.IGNORECASE)


def _forms(word: str) -> set[str]:
    """An Italian colour adjective with its feminine and plural forms."""
    if word.endswith("co"):
        s = word[:-2]
        return {word, s + "ca", s + "chi", s + "che"}
    if word.endswith(("cio", "gio")):
        s = word[:-1]
        return {word, s + "a", s, s + "e", word[:-2] + "e"}
    if word.endswith("o"):
        s = word[:-1]
        return {word, s + "a", s + "i", s + "e"}
    if word.endswith("e"):
        return {word, word[:-1] + "i"}
    return {word}


# colour words, Italian (with their forms) and English, in the tones of soils: a sensible colour
# for a log written in words ("bruno rossastro", "grigio scuro", "light yellowish brown")
_COLOUR_RGB = {
    "nero": (40, 36, 33), "bianco": (232, 228, 218), "grigio": (138, 135, 128), "marrone": (122, 88, 58),
    "bruno": (106, 76, 52), "castano": (108, 74, 50), "rosso": (152, 62, 46), "giallo": (204, 166, 70),
    "verde": (106, 120, 80), "azzurro": (110, 140, 165), "arancione": (194, 110, 52), "fulvo": (160, 106, 62),
    "arancio": (194, 110, 52), "blu": (78, 100, 132), "rosa": (198, 152, 140), "viola": (112, 82, 110),
    "ocra": (192, 136, 66), "beige": (206, 186, 150), "avana": (142, 108, 72), "nocciola": (142, 110, 77),
    "crema": (230, 216, 182), "ruggine": (140, 76, 45), "tabacco": (110, 80, 48), "cenere": (168, 164, 156),
    "black": (40, 36, 33), "white": (232, 228, 218), "grey": (138, 135, 128), "gray": (138, 135, 128),
    "brown": (122, 88, 58), "red": (152, 62, 46), "yellow": (204, 166, 70), "green": (106, 120, 80),
    "blue": (78, 100, 132), "orange": (194, 110, 52), "olive": (110, 108, 66), "pink": (198, 152, 140),
    "purple": (112, 82, 110), "tan": (190, 160, 120), "buff": (212, 184, 140), "ochre": (192, 136, 66),
    "ocher": (192, 136, 66), "rust": (140, 76, 45), "cream": (230, 216, 182),
}
_ENGLISH_COLOURS = {"black", "white", "grey", "gray", "brown", "red", "yellow", "green", "blue", "orange", "olive",
                    "pink", "purple", "tan", "buff", "ochre", "ocher", "rust", "cream", "beige"}
_TINT_OF = {"rossastro": "rosso", "rossiccio": "rosso", "giallastro": "giallo", "giallognolo": "giallo",
            "grigiastro": "grigio", "verdastro": "verde", "verdognolo": "verde", "brunastro": "bruno",
            "nerastro": "nero", "biancastro": "bianco", "bluastro": "blu", "azzurrognolo": "azzurro",
            "olivastro": "olive", "rosato": "rosa", "violaceo": "viola", "reddish": "red", "yellowish": "yellow",
            "greyish": "grey", "grayish": "grey", "brownish": "brown", "blackish": "black", "whitish": "white",
            "greenish": "green", "bluish": "blue", "pinkish": "pink", "orangish": "orange", "olivish": "olive"}
_BASE_WORDS = {f: (rgb, w in _ENGLISH_COLOURS) for w, rgb in _COLOUR_RGB.items()
               for f in (_forms(w) if w not in _ENGLISH_COLOURS and w not in ("blu", "rosa", "viola", "ocra", "beige",
                                                                              "avana", "nocciola", "crema", "tabacco")
                         else {w})}
_TINT_WORDS = {f: _COLOUR_RGB[base] for w, base in _TINT_OF.items()
               for f in (_forms(w) if not w.endswith("ish") else {w})}
_LIGHT_WORDS = _forms("chiaro") | _forms("pallido") | {"light", "pale"}
_DARK_WORDS = _forms("scuro") | _forms("cupo") | {"dark", "deep"}
_VERY_WORDS = {"molto", "very"}


def colour_words(text) -> str | None:
    """``#rrggbb`` of a colour written in words, Italian or English: a base colour (the first in
    Italian, the last in English), touches of the others and of the ``-astro``/``-ish`` words,
    lighter or darker (``chiaro``, ``scuro``, ``light``, ``dark``, ``molto``/``very`` twice);
    None when no colour word is there."""
    words = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower())
    bases, tints, english = [], [], False
    light = dark = 0
    very = 1
    for w in words:
        if w in _VERY_WORDS:
            very = 2
        elif w in _LIGHT_WORDS:
            light, very = light + very, 1
        elif w in _DARK_WORDS:
            dark, very = dark + very, 1
        elif w in _TINT_WORDS:
            tints.append(_TINT_WORDS[w])
        elif w in _BASE_WORDS:
            rgb, is_english = _BASE_WORDS[w]
            bases.append(rgb)
            english |= is_english
    if not bases and not tints:
        return None
    if bases:
        base, others = (bases[-1], bases[:-1]) if english else (bases[0], bases[1:])
    else:
        base, others, tints = tints[0], [], tints[1:]
    rgb = np.array(base, float)
    for c in list(others) + tints:
        rgb = 0.65 * rgb + 0.35 * np.array(c, float)
    for _ in range(light):
        rgb = 0.6 * rgb + 0.4 * 255.0
    for _ in range(dark):
        rgb = 0.62 * rgb
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def munsell_notation(text) -> str | None:
    """The Munsell notation written in ``text`` (``10YR 4/3``, ``2,5Y 5/2`` with a decimal comma,
    ``N 5/``), in the form colour-science reads; None without one."""
    m = _MUNSELL.search(str(text or ""))
    if not m:
        return None
    if m.group(5) is not None:
        return f"N {m.group(5).replace(',', '.')}/"
    hue, value, chroma = (m.group(i).replace(",", ".") for i in (1, 3, 4))
    return f"{hue}{m.group(2).upper()} {value}/{chroma}"


def has_colour_science() -> bool:
    import importlib.util

    return importlib.util.find_spec("colour") is not None


def parse_colour(text: str) -> str | None:
    """``#rrggbb`` of a colour written as ``#rrggbb``, ``r,g,b`` (0-255), a Munsell notation
    (``10YR 4/3``, also ``2,5Y 5/2``; with colour-science), colour words in Italian or English
    (``bruno rossastro``, ``grigio scuro``, ``light yellowish brown``) or a colour name
    (``sienna``); None when it cannot be read."""
    s = str(text or "").strip()
    if not s:
        return None
    if _HEX.fullmatch(s):
        return "#" + s.lstrip("#").lower()
    m = _RGB_TRIPLE.fullmatch(s)
    if m:
        r, g, b = (min(255, int(v)) for v in m.groups())
        return f"#{r:02x}{g:02x}{b:02x}"
    notation = munsell_notation(s)
    if notation:
        hx = munsell_to_hex(notation)
        if hx:
            return hx
    words = colour_words(s)
    if words:
        return words
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

        s = re.sub(r"\s+", " ", notation.strip().upper().replace(",", "."))
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


# ------------------------------------------------------------------ hatches
# the hatch of every lithology in the logs and sections
HATCH = {"clay": "--", "silt": "-.", "sand": "..", "gravel": "oo", "stones": "**", "mudbrick": "++", "ash": "xx",
         "peat": "||", "organic": "\\\\"}
# lithology -> (nouns, adjectives), whole words, Italian and English
LITHOLOGY_WORDS = {
    "clay": ({"argilla", "argille", "clay", "clays"}, {"argilloso", "argillosa", "argillosi", "argillose", "clayey"}),
    "silt": ({"limo", "limi", "silt", "silts"}, {"limoso", "limosa", "limosi", "limose", "silty"}),
    "sand": ({"sabbia", "sabbie", "sand", "sands"}, {"sabbioso", "sabbiosa", "sabbiosi", "sabbiose", "sandy"}),
    "gravel": ({"ghiaia", "ghiaie", "ghiaietto", "ghiaino", "ciottolo", "ciottoli", "gravel", "gravels", "pebble",
                "pebbles", "cobble", "cobbles"},
               {"ghiaioso", "ghiaiosa", "ghiaiosi", "ghiaiose", "ciottoloso", "ciottolosa", "ciottolosi",
                "ciottolose", "gravelly", "pebbly"}),
    "stones": ({"pietra", "pietre", "pietrame", "pietrisco", "sasso", "sassi", "crollo", "crolli", "maceria", "macerie",
                "laterizio", "laterizi", "mattone", "mattoni", "tegola", "tegole", "coppo", "coppi", "stone", "stones",
                "rubble", "brick", "bricks", "tile", "tiles", "collapse", "debris"},
               {"pietroso", "pietrosa", "pietrosi", "pietrose", "stony"}),
    "mudbrick": ({"mudbrick", "mudbricks", "adobe", "adobes"}, set()),
    "ash": ({"cenere", "ceneri", "carbone", "carboni", "carbonella", "ash", "ashes", "charcoal"},
            {"cineritico", "cineritica", "cineritici", "cineritiche", "carbonioso", "carboniosa", "carboniosi",
             "carboniose", "ashy"}),
    "peat": ({"torba", "torbe", "peat"}, {"torboso", "torbosa", "torbosi", "torbose", "peaty"}),
    "organic": ({"humus", "topsoil"}, {"organico", "organica", "organici", "organiche", "humico", "humica", "organic"}),
}
LITHOLOGY_PHRASES = [(("mattone", "crudo"), "mudbrick"), (("mattoni", "crudi"), "mudbrick"),
                     (("mattone", "di", "fango"), "mudbrick"), (("mattoni", "di", "fango"), "mudbrick"),
                     (("terra", "cruda"), "mudbrick"), (("mud", "brick"), "mudbrick"), (("mud", "bricks"), "mudbrick"),
                     (("terreno", "vegetale"), "organic"), (("terra", "vegetale"), "organic"),
                     (("suolo", "vegetale"), "organic")]
_NOUNS = {w: k for k, (nouns, _a) in LITHOLOGY_WORDS.items() for w in nouns}
_ADJECTIVES = {w: k for k, (_n, adjectives) in LITHOLOGY_WORDS.items() for w in adjectives}
_ENGLISH_WORDS = {"clay", "clays", "clayey", "silt", "silts", "silty", "sand", "sands", "sandy", "gravel", "gravels",
                  "gravelly", "pebble", "pebbles", "pebbly", "cobble", "cobbles", "stone", "stones", "stony", "rubble",
                  "brick", "bricks", "tile", "tiles", "collapse", "debris", "mudbrick", "mudbricks", "adobe", "adobes",
                  "mud", "ash", "ashes", "ashy", "charcoal", "peat", "peaty", "topsoil", "organic"}


def lithology_of(text) -> str | None:
    """The lithology a text names (a key of :data:`HATCH`): the head noun decides - the first
    lithology noun in Italian (``ghiaia sabbiosa``: gravel), the last of a compound in English
    (``sandy gravel``: gravel, ``mudbrick collapse``: stones) -, an adjective only when no noun is
    there; whole words only. None when none is named."""
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    tokens = re.findall(r"[a-z]+|[^\sa-z]", s)
    items = []  # (start, end, lithology, a noun, English)
    i = 0
    while i < len(tokens):
        phrase = next(((words, kind) for words, kind in LITHOLOGY_PHRASES
                       if tuple(tokens[i:i + len(words)]) == words), None)
        if phrase is not None:
            words, kind = phrase
            items.append((i, i + len(words), kind, True, words[0] in _ENGLISH_WORDS))
            i += len(words)
            continue
        w = tokens[i]
        if w in _NOUNS:
            items.append((i, i + 1, _NOUNS[w], True, w in _ENGLISH_WORDS))
        elif w in _ADJECTIVES:
            items.append((i, i + 1, _ADJECTIVES[w], False, w in _ENGLISH_WORDS))
        i += 1
    nouns = [k for k, it in enumerate(items) if it[3]]
    if nouns:
        k = nouns[0]
        if items[k][4]:  # English: the last noun of the compound
            while k + 1 < len(items) and items[k + 1][3] and items[k + 1][4] and items[k + 1][0] == items[k][1]:
                k += 1
        return items[k][2]
    return items[0][2] if items else None


def hatch_of(*texts: str) -> str:
    """The matplotlib hatch of the lithology named in ``texts``: the first text that names one
    (the lithology column, then the description) decides (:func:`lithology_of`); "" for none."""
    for text in texts:
        kind = lithology_of(text)
        if kind:
            return HATCH[kind]
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
