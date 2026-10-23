"""Geophysical grids: the data model and the file formats.

A grid is a regular array of values (one band, or several: the time or depth slices of a GPR
survey) with its cell size and its position: in the coordinate system of the project when it
is georeferenced, else in the local metres of the survey grid (origin at the lower-left corner,
x along the first row of readings). Formats read: Surfer GRD (ASCII ``DSAA``, binary ``DSBB`` of
Surfer 6 and ``DSRB`` of Surfer 7), ASCII XYZ / CSV (``x y value``, with or without a header,
the Geoplot, Bartington and Terrasurveyor exports), any raster rasterio opens (GeoTIFF, ERDAS,
ASCII grid ``.asc``), and a folder of rasters or images as the slices of a GPR survey. Every grid
of the project is kept as a GeoTIFF (float32, NaN where no reading) in ``<chunk data>/geophysics/``
and listed in ``chunk.meta["geophysics"]["grids"]``.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

KINDS = ("magnetometry", "resistivity", "gpr", "em", "susceptibility", "other")
UNITS = {"magnetometry": "nT", "resistivity": "ohm m", "gpr": "", "em": "mS/m", "susceptibility": "SI", "other": ""}
FOLDER = "geophysics"
NODATA = 1.70141e38  # Surfer's blank value
BLANK = 1e38  # a reading this large is Surfer's blank, not a reading
DUMMIES = (2047.5, -9999.0, 32767.0, -32768.0)  # Geoscan's dummy reading and the usual no-data numbers
MAX_CELLS = 200_000_000  # an XYZ table whose readings are not on a lattice would make a grid this large


@dataclass
class Grid:
    """``values`` (bands, rows, cols) float32 with NaN where no reading; row 0 at the TOP (as a
    raster); ``cell`` metres; ``origin`` the top-left corner (x, y) in ``crs`` (a projected
    system, or "LOCAL" for the survey metres); ``rotation`` degrees counter-clockwise of the
    grid's x axis from the east (0 for a north-up raster). ``local_cell``: the cell in the metres
    of the survey (0: ``cell``), what a placement with a free scale starts from again; ``notes``:
    how the readings became the grid (spacings, interpolation between traverses, dummies)."""

    name: str
    values: np.ndarray
    cell: float
    origin: tuple[float, float]
    crs: str = "LOCAL"
    rotation: float = 0.0
    kind: str = "other"
    units: str = ""
    slices: list[str] = field(default_factory=list)  # a label per band (depth or time of a GPR slice)
    source: str = ""
    filters: list[dict] = field(default_factory=list)
    id: str = ""
    local_cell: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape[1], self.values.shape[2]

    @property
    def bands(self) -> int:
        return self.values.shape[0]

    def survey_cell(self) -> float:
        """The cell in the survey's own metres (before a placement with a free scale)."""
        return float(self.local_cell or self.cell)

    def transform(self):
        """The rasterio affine of the grid (rotation included)."""
        from rasterio.transform import Affine

        a = np.radians(self.rotation)
        c, s = np.cos(a), np.sin(a)
        return Affine(self.cell * c, self.cell * s, self.origin[0], self.cell * s, -self.cell * c, self.origin[1])

    def corners(self) -> np.ndarray:
        """The four corners (x, y) of the grid: top-left, top-right, bottom-right, bottom-left."""
        t = self.transform()
        h, w = self.shape

        def at(c, r):
            return (t.a * c + t.b * r + t.c, t.d * c + t.e * r + t.f)

        return np.array([at(0, 0), at(w, 0), at(w, h), at(0, h)], float)

    def bounds(self) -> tuple[float, float, float, float]:
        c = self.corners()
        return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())

    def georeferenced(self) -> bool:
        return self.crs != "LOCAL"

    def stats(self, band: int = 0) -> dict:
        v = self.values[band]
        ok = np.isfinite(v)
        if not ok.any():
            return {"count": 0}
        x = v[ok]
        p = np.percentile(x, [2, 50, 98])
        return {"count": int(ok.sum()), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
                "std": float(x.std()), "p2": float(p[0]), "median": float(p[1]), "p98": float(p[2])}

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "units": self.units, "cell": self.cell,
                "origin": [float(self.origin[0]), float(self.origin[1])], "crs": self.crs, "rotation": self.rotation,
                "rows": self.shape[0], "cols": self.shape[1], "bands": self.bands, "slices": list(self.slices),
                "source": self.source, "filters": list(self.filters), "local_cell": self.survey_cell(),
                "notes": dict(self.notes)}


def like(grid: Grid, values: np.ndarray, **changes) -> Grid:
    """A grid with the metadata of ``grid`` and other ``values`` (and ``changes``)."""
    fields = {"name": grid.name, "values": values, "cell": grid.cell, "origin": grid.origin, "crs": grid.crs,
              "rotation": grid.rotation, "kind": grid.kind, "units": grid.units, "slices": list(grid.slices),
              "source": grid.source, "filters": list(grid.filters), "id": grid.id, "local_cell": grid.local_cell,
              "notes": dict(grid.notes)}
    fields.update(changes)
    return Grid(**fields)


# ------------------------------------------------------------------------------ Surfer GRD
def read_grd(path: str | Path) -> Grid:
    """A Surfer grid: ASCII ``DSAA``, binary ``DSBB`` (Surfer 6) or ``DSRB`` (Surfer 7). Surfer
    stores the rows from the bottom (y min) up: they are flipped so that row 0 is the top."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"DSAA":
            text = magic + f.read()
            tokens = text.decode("latin-1").split()
            nx, ny = int(tokens[1]), int(tokens[2])
            xlo, xhi, ylo, yhi = (float(v) for v in tokens[3:7])
            zlo, zhi = float(tokens[7]), float(tokens[8])
            vals = np.array([float(v) for v in tokens[9:9 + nx * ny]], np.float32).reshape(ny, nx)
            blank = 1.70141e38
        elif magic == b"DSBB":
            nx, ny = struct.unpack("<hh", f.read(4))
            xlo, xhi, ylo, yhi, zlo, zhi = struct.unpack("<6d", f.read(48))
            vals = np.frombuffer(f.read(4 * nx * ny), "<f4").reshape(ny, nx).astype(np.float32)
            blank = 1.70141e38
        elif magic == b"DSRB":
            f.read(4)  # size of the header section
            f.read(4)  # version
            vals = None
            blank = 1.70141e38
            nx = ny = 0
            xlo = ylo = 0.0
            xhi = yhi = 0.0
            while True:
                tag = f.read(4)
                if len(tag) < 4:
                    break
                size = struct.unpack("<i", f.read(4))[0]
                data = f.read(size)
                if tag == b"GRID":
                    ny, nx = struct.unpack("<ii", data[:8])
                    xlo, ylo, xsize, ysize, zlo, zhi, rotation, blank = struct.unpack("<8d", data[8:72])
                    xhi, yhi = xlo + (nx - 1) * xsize, ylo + (ny - 1) * ysize
                elif tag == b"DATA":
                    vals = np.frombuffer(data, "<f8").reshape(ny, nx).astype(np.float32)
            if vals is None:
                raise ValueError(f"{path.name}: no DATA section in the Surfer 7 grid")
        else:
            raise ValueError(f"{path.name}: not a Surfer grid (DSAA, DSBB or DSRB)")
    vals = np.array(vals, np.float32)
    vals[(vals >= blank * 0.99) | ~np.isfinite(vals)] = np.nan
    cell_x = (xhi - xlo) / max(nx - 1, 1)
    cell_y = (yhi - ylo) / max(ny - 1, 1)
    cell = float(cell_x if abs(cell_x - cell_y) < 1e-9 * max(cell_x, 1) else cell_x)
    # Surfer's x/y are node centres: the top-left corner of the top-left cell
    grid = Grid(path.stem, vals[::-1][None], cell, (xlo - cell / 2, yhi + cell_y / 2), "LOCAL", 0.0, source=str(path))
    if abs(cell_x - cell_y) > 1e-6 * max(cell_x, 1):
        grid = resample_square(grid, cell_x, cell_y)
    return grid


def north_up(grid: Grid) -> Grid:
    """``grid`` without rotation: a turned grid resampled (bilinear) onto a north-up grid of the
    same cell over the box of its corners, NaN where the grid does not reach; a north-up grid as it is."""
    if abs((grid.rotation + 180.0) % 360.0 - 180.0) < 1e-9:
        return grid
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    x0, y0, x1, y1 = grid.bounds()
    w = max(1, int(np.ceil((x1 - x0) / grid.cell - 1e-6)))
    h = max(1, int(np.ceil((y1 - y0) / grid.cell - 1e-6)))
    dst_tr = Affine(grid.cell, 0.0, x0, 0.0, -grid.cell, y1)
    crs = rasterio.crs.CRS.from_epsg(3857)  # the same frame on both sides: any projected CRS does
    out = np.full((grid.bands, h, w), np.nan, np.float32)
    for b in range(grid.bands):
        reproject(source=grid.values[b].astype(np.float32), destination=out[b], src_transform=grid.transform(),
                  src_crs=crs, dst_transform=dst_tr, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return like(grid, out, origin=(x0, y1), rotation=0.0)


def write_grd(grid: Grid, path: str | Path, band: int = 0, binary: bool = True) -> Path:
    """A Surfer 6 grid (binary ``DSBB``, or ASCII ``DSAA``) of one band, Surfer's blank for NaN.

    A Surfer grid has no rotation: a turned grid (placed with a rotation) is resampled onto a
    north-up grid of the same cell that covers it (:func:`north_up`), the cells outside it blank."""
    path = Path(path)
    grid = north_up(grid)
    v = np.array(grid.values[band][::-1], np.float64)  # Surfer: rows from the bottom
    ny, nx = v.shape
    xlo, ylo = grid.origin[0] + grid.cell / 2, grid.origin[1] - grid.cell * ny + grid.cell / 2
    xhi, yhi = xlo + (nx - 1) * grid.cell, ylo + (ny - 1) * grid.cell
    ok = np.isfinite(v)
    zlo, zhi = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (0.0, 0.0)
    out = np.where(ok, v, NODATA)
    if binary:
        with open(path, "wb") as f:
            f.write(b"DSBB")
            f.write(struct.pack("<hh", nx, ny))
            f.write(struct.pack("<6d", xlo, xhi, ylo, yhi, zlo, zhi))
            f.write(out.astype("<f4").tobytes())
    else:
        lines = ["DSAA", f"{nx} {ny}", f"{xlo:.6f} {xhi:.6f}", f"{ylo:.6f} {yhi:.6f}", f"{zlo:.6f} {zhi:.6f}"]
        for row in out:
            lines.append(" ".join(f"{x:.6g}" for x in row))
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# ------------------------------------------------------------------------------ XYZ / CSV
_DECIMAL_COMMA = re.compile(r"\d,\d")


def _separator(sample: list[str]) -> str:
    """How the columns of a table are separated: semicolons (then a comma is a decimal comma),
    tabs, spaces (also ``x, y, v`` and the decimal commas of ``12,5 3,25 45,1``), or commas."""
    if any(";" in ln for ln in sample):
        return "semicolon"
    if any("\t" in ln for ln in sample):
        return "tab"
    if sample and sum(1 for ln in sample if len(ln.split()) >= 3) >= len(sample) / 2:
        return "space"
    return "comma"


def _fields(line: str, sep: str) -> list[str]:
    if sep == "semicolon":
        parts = line.split(";")
    elif sep == "tab":
        parts = line.split("\t")
    elif sep == "space":
        parts = [p.rstrip(",") for p in line.split()]
    else:
        parts = line.split(",")
    parts = [p.strip() for p in parts if p.strip()]
    return parts if sep == "comma" else [p.replace(",", ".") for p in parts]  # decimal commas


def _step(a: np.ndarray) -> float | None:
    """The spacing of the distinct coordinates of readings taken on a lattice (the median of the
    positive differences), None with a single coordinate."""
    u = np.unique(np.round(a, 6))
    if len(u) < 2:
        return None
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(np.median(d)) if d.size else None


def _mean_on(cells: np.ndarray, v: np.ndarray, size: int) -> np.ndarray:
    """The mean of the readings ``v`` of every cell index (NaN where none)."""
    sums = np.zeros(size)
    cnt = np.zeros(size)
    np.add.at(sums, cells, v)
    np.add.at(cnt, cells, 1)
    out = np.full(size, np.nan)
    out[cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
    return out


def _check_size(w: int, h: int, name: str) -> None:
    if w * h > MAX_CELLS:
        raise ValueError(f"{name}: the readings would make a grid of {w} x {h} cells - they are not on a regular "
                         "grid; give the cell")


def read_xyz(path: str | Path, cell: float | None = None, columns: tuple[int, int, int] = (0, 1, 2),
             dummy: float | list[float] | None = None) -> Grid:
    """``x y value`` lines (a header or comment lines allowed) put on a regular grid.

    * The columns are separated by spaces, tabs, commas or semicolons; decimal commas are read when
      the columns are separated by semicolons, tabs or spaces (``12,5;3,25;45,1``, ``12,5 3,25 45,1``).
    * Dummy readings are no readings: Geoscan's 2047.5, -9999, 32767, -32768, Surfer's blank
      (1.70141e38 and anything as large) and ``dummy`` (a number or a list of numbers).
    * The spacing of the readings is found along x and along y. When the two are equal it is the
      cell. When they differ (readings every 0.125 m along traverses 0.5 m apart) the cell is the
      finer one and the rows (or columns) between two traverses are interpolated linearly from the
      two traverses beside them - never beyond the first and the last traverse, never across a
      traverse missing; ``notes["interpolated"]`` says so. ``cell``: a cell of your own instead (the
      spacing of the traverses, say): the readings falling in each cell are averaged.
    * The cells are centred on the readings: the grid starts half a spacing before the first reading,
      so cells of 0.5 m over readings every 0.125 m from 0.0625 m start at 0 and hold four readings
      each, as the survey's own cells."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        raw = [ln.strip() for ln in f]
    sample = [ln for ln in raw if any(ch.isdigit() for ch in ln)][:200]
    sep = _separator(sample)
    rows, comma = [], False
    for line in raw:
        parts = _fields(line, sep)
        if len(parts) <= max(columns):
            continue
        try:
            rows.append([float(parts[c]) for c in columns])
        except ValueError:
            continue  # a header or a text line
        comma = comma or (sep != "comma" and _DECIMAL_COMMA.search(line) is not None)
    if not rows:
        raise ValueError(f"{path.name}: no numeric x y value lines")
    pts = np.array(rows, float)
    v = pts[:, 2]
    bad = ~np.isfinite(v) | (np.abs(v) >= BLANK)
    for d in list(DUMMIES) + ([] if dummy is None else [float(x) for x in np.atleast_1d(dummy)]):
        bad |= np.abs(v - d) <= 1e-6 * max(1.0, abs(d))
    pts = pts[~bad]
    if not len(pts):
        raise ValueError(f"{path.name}: only dummy readings")
    x, y, v = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = _step(x), _step(y)
    dx, dy = dx or dy or 1.0, dy or dx or 1.0
    notes = {"readings": int(len(pts)), "spacing_x": dx, "spacing_y": dy}
    if bad.any():
        notes["dummies"] = int(bad.sum())
    if comma:
        notes["decimal_comma"] = True
    same = abs(dx - dy) <= 0.01 * max(dx, dy)
    if cell is None and not same:
        grid = _between_traverses(x, y, v, dx, dy, path.name)
        grid.name, grid.source = path.stem, str(path)
        grid.notes = {**notes, **grid.notes}
        return grid
    cell = float(cell) if cell else min(dx, dy)
    # the first cell edge half a spacing before the first reading (half a cell when the cell is finer)
    x0 = x.min() - min(dx, cell) / 2.0
    y1 = y.max() + min(dy, cell) / 2.0
    cols = np.floor((x - x0) / cell + 1e-9).astype(int)
    rws = np.floor((y1 - y) / cell + 1e-9).astype(int)
    w, h = int(cols.max()) + 1, int(rws.max()) + 1
    _check_size(w, h, path.name)
    vals = _mean_on(rws * w + cols, v, h * w).astype(np.float32)
    if cell > min(dx, dy) * 1.01:
        notes["averaged"] = True
    return Grid(path.stem, vals.reshape(1, h, w), float(cell), (float(x0), float(y1)), "LOCAL", 0.0, source=str(path),
                notes=notes)


def _between_traverses(x: np.ndarray, y: np.ndarray, v: np.ndarray, dx: float, dy: float, name: str) -> Grid:
    """Readings closer along the traverses than the traverses are apart: a grid of the finer
    spacing, the rows (or columns) between two neighbouring traverses interpolated linearly."""
    rows_between = dx < dy  # the traverses run along x, one per row
    along, across = (x, y) if rows_between else (y, x)
    f, c = min(dx, dy), max(dx, dy)
    ia = np.rint((along - along.min()) / f).astype(int)
    it = np.rint((across - across.min()) / c).astype(int)
    na, nt = int(ia.max()) + 1, int(it.max()) + 1
    _check_size(na, int(round((across.max() - across.min()) / f)) + 1, name)
    trav = _mean_on(it * na + ia, v, nt * na).reshape(nt, na)  # a row per traverse
    n_out = int(round((across.max() - across.min()) / f)) + 1
    pos = np.arange(n_out) * f / c  # every output line in traverses from the first one
    t0 = np.clip(np.floor(pos + 1e-9).astype(int), 0, nt - 1)
    frac = np.clip(pos - t0, 0.0, 1.0)
    t1 = np.clip(t0 + 1, 0, nt - 1)
    with np.errstate(invalid="ignore"):
        out = (1.0 - frac)[:, None] * trav[t0] + frac[:, None] * trav[t1]  # NaN beside a missing reading
    on = frac < 1e-6
    out[on] = trav[t0[on]]  # the lines of the traverses: their own readings
    # (n_out lines across, na cells along) -> raster rows from the top (largest y)
    vals = out[::-1] if rows_between else out.T[::-1]
    h, w = vals.shape
    x0 = x.min() - f / 2.0
    y1 = y.min() + (h - 1) * f + f / 2.0
    notes = {"interpolated": "rows" if rows_between else "columns", "traverse_spacing": c, "factor": c / f,
             "traverses": nt}
    return Grid(name, vals[None].astype(np.float32), float(f), (float(x0), float(y1)), "LOCAL", 0.0, notes=notes)


# ------------------------------------------------------------------------------ rasters
def read_raster(path: str | Path, kind: str = "other") -> Grid:
    """Any raster rasterio opens: its bands as the bands of the grid, its CRS when it has one
    (else LOCAL), the rotation of its transform."""
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        vals = src.read(masked=True).astype(np.float32)
        vals = np.ma.filled(vals, np.nan)
        if src.nodata is not None and np.isfinite(src.nodata):
            vals[vals == src.nodata] = np.nan
        t = src.transform
        cell = float(np.hypot(t.a, t.d))
        rotation = float(np.degrees(np.arctan2(t.d, t.a)))
        crs = src.crs.to_string() if src.crs else "LOCAL"
        slices = [d for d in src.descriptions if d] if src.count > 1 else []
    return Grid(path.stem, vals, cell, (float(t.c), float(t.f)), crs, rotation, kind=kind, slices=slices,
                source=str(path))


def natural_key(name: str) -> list:
    """The order of names with numbers as people count: ``slice_2`` before ``slice_10``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def read_slices(folder: str | Path, labels: list[str] | None = None, kind: str = "gpr") -> Grid:
    """The slices of a GPR survey: the rasters or images of a folder (in the natural order of
    their names - ``slice_2`` before ``slice_10`` - one per band; images without a world file take
    the first raster's position, or local metres of 1 per pixel), with a label per slice (default:
    the file names)."""
    folder = Path(folder)
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".asc",
                                                                            ".grd")), key=lambda p: natural_key(p.name))
    if not files:
        raise ValueError(f"{folder}: no raster or image slices")
    grids = []
    for p in files:
        try:
            grids.append(read_grd(p) if p.suffix.lower() == ".grd" else read_raster(p, kind))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{p.name}: {exc}") from exc
    base = grids[0]
    stack = []
    for g in grids:
        v = g.values[0] if g.bands >= 1 else g.values
        if v.shape != base.shape:
            import cv2

            v = cv2.resize(v.astype(np.float32), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
        stack.append(v.astype(np.float32))
    return Grid(folder.name, np.stack(stack), base.cell, base.origin, base.crs, base.rotation, kind=kind,
                slices=list(labels or [p.stem for p in files]), source=str(folder))


def read_any(path: str | Path, kind: str = "other", cell: float | None = None,
             dummy: float | list[float] | None = None) -> Grid:
    """A grid from any file the plugin reads (by the extension) or a folder of slices; ``cell``
    and ``dummy`` for the XYZ tables (:func:`read_xyz`)."""
    p = Path(path)
    if p.is_dir():
        return read_slices(p, kind=kind)
    ext = p.suffix.lower()
    if ext == ".grd":
        g = read_grd(p)
    elif ext in (".xyz", ".csv", ".txt", ".dat"):
        g = read_xyz(p, cell=cell, dummy=dummy)
    else:
        g = read_raster(p, kind)
    g.kind = kind
    g.units = g.units or UNITS.get(kind, "")
    return g


def resample_square(grid: Grid, cell_x: float, cell_y: float) -> Grid:
    """A grid with rectangular cells brought to square cells of the smaller side."""
    import cv2

    cell = min(cell_x, cell_y)
    h, w = grid.shape
    new_w, new_h = int(round(w * cell_x / cell)), int(round(h * cell_y / cell))
    out = np.stack([cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for b in grid.values])
    grid.values, grid.cell = out.astype(np.float32), cell
    return grid


# ------------------------------------------------------------------------------ in the project
def write_geotiff(grid: Grid, path: str | Path, crs: str | None = None) -> Path:
    """The grid as a GeoTIFF (float32, NaN no data, a band per slice with its label), in
    ``crs`` (the grid's; none for a local grid)."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    crs = crs if crs is not None else grid.crs
    profile = {"driver": "GTiff", "height": grid.shape[0], "width": grid.shape[1], "count": grid.bands,
               "dtype": "float32", "transform": grid.transform(), "nodata": float("nan"), "compress": "deflate"}
    if crs and crs != "LOCAL":
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid.values.astype(np.float32))
        for i, label in enumerate(grid.slices[:grid.bands]):
            dst.set_band_description(i + 1, str(label))
        dst.update_tags(ppg_kind=grid.kind, ppg_units=grid.units, ppg_name=grid.name)
    return path


def registry(chunk) -> dict:
    return chunk.meta.setdefault("geophysics", {"grids": []})


def save_grid(chunk, grid: Grid) -> dict:
    """The grid into the chunk: its GeoTIFF in ``<data>/geophysics/<id>.tif`` and its entry in
    the registry (a new id when it has none, else the file and the entry replaced)."""
    reg = registry(chunk)
    if not grid.id:
        grid.id = datetime.now().strftime("g%Y%m%d%H%M%S%f")[:20]
    folder = chunk.data_dir / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = write_geotiff(grid, folder / f"{grid.id}.tif", None if grid.crs == "LOCAL" else grid.crs)
    entry = {**grid.info(), "file": str(path.relative_to(chunk.data_dir)), "stats": grid.stats(),
             "updated": datetime.now().isoformat(timespec="seconds")}
    grids = reg.setdefault("grids", [])
    for k, e in enumerate(grids):
        if e["id"] == grid.id:
            grids[k] = entry
            break
    else:
        grids.append(entry)
    json.dumps(entry)  # serialisable
    return entry


def entries(chunk) -> list[dict]:
    return list((chunk.meta.get("geophysics") or {}).get("grids") or [])


def entry_of(chunk, grid_id: str) -> dict:
    for e in entries(chunk):
        if e["id"] == grid_id or e["name"] == grid_id:
            return e
    raise KeyError(f"no geophysical grid {grid_id!r}")


def _survey_cell_of(e: dict) -> float:
    """The cell in survey metres of an entry; an entry of version 0.1.0 has none: the cell, or the
    cell divided by the scale of its last placement with a free scale."""
    if e.get("local_cell"):
        return float(e["local_cell"])
    cell = float(e["cell"])
    for f in reversed(e.get("filters") or []):
        if f.get("name") == "georef" and f.get("scale"):
            return cell / float(f["scale"])
    return cell


def load_grid(chunk, grid_id: str) -> Grid:
    """A grid of the chunk from its GeoTIFF and its entry."""
    e = entry_of(chunk, grid_id)
    g = read_raster(chunk.data_dir / e["file"], e.get("kind", "other"))
    g.id, g.name, g.units, g.crs = e["id"], e["name"], e.get("units", ""), e.get("crs", g.crs)
    g.cell, g.origin, g.rotation = float(e["cell"]), tuple(e["origin"]), float(e.get("rotation", 0.0))
    g.slices, g.source, g.filters = list(e.get("slices") or []), e.get("source", ""), list(e.get("filters") or [])
    g.local_cell, g.notes = _survey_cell_of(e), dict(e.get("notes") or {})
    return g


def remove_grid(chunk, grid_id: str) -> None:
    e = entry_of(chunk, grid_id)
    reg = registry(chunk)
    reg["grids"] = [x for x in reg.get("grids", []) if x["id"] != e["id"]]
    p = chunk.data_dir / e["file"]
    if p.exists():
        p.unlink()
