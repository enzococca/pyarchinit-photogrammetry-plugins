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


@dataclass
class Grid:
    """``values`` (bands, rows, cols) float32 with NaN where no reading; row 0 at the TOP (as a
    raster); ``cell`` metres; ``origin`` the top-left corner (x, y) in ``crs`` (a projected
    system, or "LOCAL" for the survey metres); ``rotation`` degrees counter-clockwise of the
    grid's x axis from the east (0 for a north-up raster)."""

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

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape[1], self.values.shape[2]

    @property
    def bands(self) -> int:
        return self.values.shape[0]

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
                "source": self.source, "filters": list(self.filters)}


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


def write_grd(grid: Grid, path: str | Path, band: int = 0, binary: bool = True) -> Path:
    """A Surfer 6 grid (binary ``DSBB``, or ASCII ``DSAA``) of one band, blanks for NaN."""
    path = Path(path)
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
_SEP = re.compile(r"[,\t; ]+")


def read_xyz(path: str | Path, cell: float | None = None, columns: tuple[int, int, int] = (0, 1, 2)) -> Grid:
    """``x y value`` lines (a header line allowed; ``,`` ``;`` tabs or spaces): the readings
    put on a regular grid of ``cell`` (default: the smallest spacing found), the mean of the
    readings of a cell, NaN where none."""
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p for p in _SEP.split(line.strip()) if p != ""]
            if len(parts) <= max(columns):
                continue
            try:
                rows.append([float(parts[c]) for c in columns])
            except ValueError:
                continue  # a header or a text line
    if not rows:
        raise ValueError(f"{path.name}: no numeric x y value lines")
    pts = np.array(rows, float)
    x, y, v = pts[:, 0], pts[:, 1], pts[:, 2]
    if cell is None:
        cell = _spacing(x, y)
    x0, y1 = x.min() - cell / 2, y.max() + cell / 2
    cols = np.floor((x - x0) / cell).astype(int)
    rws = np.floor((y1 - y) / cell).astype(int)
    w, h = int(cols.max()) + 1, int(rws.max()) + 1
    sums = np.zeros(h * w)
    cnt = np.zeros(h * w)
    np.add.at(sums, rws * w + cols, v)
    np.add.at(cnt, rws * w + cols, 1)
    vals = np.full(h * w, np.nan, np.float32)
    vals[cnt > 0] = (sums[cnt > 0] / cnt[cnt > 0]).astype(np.float32)
    return Grid(path.stem, vals.reshape(1, h, w), float(cell), (float(x0), float(y1)), "LOCAL", 0.0, source=str(path))


def _spacing(x: np.ndarray, y: np.ndarray) -> float:
    """The cell of readings taken along lines: the smallest common spacing of the distinct
    coordinates (the median of the positive differences, on both axes)."""
    best = []
    for a in (x, y):
        u = np.unique(np.round(a, 6))
        if len(u) > 1:
            d = np.diff(u)
            d = d[d > 1e-9]
            if d.size:
                best.append(float(np.median(d)))
    if not best:
        return 1.0
    return float(min(best))


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


def read_slices(folder: str | Path, labels: list[str] | None = None, kind: str = "gpr") -> Grid:
    """The slices of a GPR survey: the rasters or images of a folder (sorted by name, one per
    band; images without a world file take the first raster's position, or local metres of 1 per
    pixel), with a label per slice (default: the file names)."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".asc", ".grd"))
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


def read_any(path: str | Path, kind: str = "other", cell: float | None = None) -> Grid:
    """A grid from any file the plugin reads (by the extension) or a folder of slices."""
    p = Path(path)
    if p.is_dir():
        return read_slices(p, kind=kind)
    ext = p.suffix.lower()
    if ext == ".grd":
        g = read_grd(p)
    elif ext in (".xyz", ".csv", ".txt", ".dat"):
        g = read_xyz(p, cell=cell)
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


def load_grid(chunk, grid_id: str) -> Grid:
    """A grid of the chunk from its GeoTIFF and its entry."""
    e = entry_of(chunk, grid_id)
    g = read_raster(chunk.data_dir / e["file"], e.get("kind", "other"))
    g.id, g.name, g.units, g.crs = e["id"], e["name"], e.get("units", ""), e.get("crs", g.crs)
    g.cell, g.origin, g.rotation = float(e["cell"]), tuple(e["origin"]), float(e.get("rotation", 0.0))
    g.slices, g.source, g.filters = list(e.get("slices") or []), e.get("source", ""), list(e.get("filters") or [])
    return g


def remove_grid(chunk, grid_id: str) -> None:
    e = entry_of(chunk, grid_id)
    reg = registry(chunk)
    reg["grids"] = [x for x in reg.get("grids", []) if x["id"] != e["id"]]
    p = chunk.data_dir / e["file"]
    if p.exists():
        p.unlink()
