"""The resistivity section of an inverted ERT profile: from the cells of pyGIMLi's mesh to a raster in
the frame of the profile, its colours, its files, its figures and its place in the 3D view.

The frame of a section is that of the program's elevations: ``u`` metres along the line from its
start (to the right), ``v`` the elevation (up); a GeoTIFF without a CRS whose transform maps the
pixels to (u, v). The blanked cells - not resolved by the readings - are NaN in the data and
transparent in the picture: by the coverage (the cumulative sensitivity, log10, more than ``drop``
below its 95th percentile) or by the depth-of-investigation index of Oldenburg and Li (above
``doi_max``, when the inversion computed it).

* :func:`rasterise` - the cells of the mesh on a regular grid (every pixel the value of the cell it
  falls in);
* :func:`colours` - the RGBA picture, log10 of the resistivity between two limits, a palette;
* :func:`write` - ``section-<stamp>.tif`` (float32, ohm m), ``section_view-<stamp>.tif`` (RGBA for the
  2D view), named in the record;
* :func:`figure` - PNG/SVG/PDF of the section or of the pseudosection with axes, electrodes and
  colour bar (Matplotlib, Agg);
* :func:`section_mesh` and :func:`scene` - the section as vertical coloured quads under the line in
  world coordinates, and the PLY of the 3D view of the program: the sections of the chunk, the
  ground cut away along the line (the model of the chunk, else its DEM), the program's proxies of the
  stratigraphic units, the units and the cores of the geoarchaeology plugin when the chunk has them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import ert as ert_mod
from .texts import t

PALETTES = {
    # the usual ramp of the resistivity sections (blue conductive - red, purple resistive)
    "rainbow": [(0.0, (0, 0, 128)), (0.15, (0, 70, 255)), (0.32, (0, 210, 255)), (0.48, (40, 200, 60)),
                (0.62, (255, 240, 0)), (0.78, (255, 120, 0)), (0.9, (210, 0, 0)), (1.0, (120, 0, 90))],
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37))],
    "earth": [(0.0, (0, 60, 48)), (0.25, (1, 133, 113)), (0.5, (245, 245, 245)), (0.75, (191, 129, 45)),
              (1.0, (84, 48, 5))],
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
}
PALETTE_NAMES = {"rainbow": ("Arcobaleno (classica)", "Rainbow (classic)"), "viridis": ("Viridis", "Viridis"),
                 "earth": ("Terra", "Earth"), "grey": ("Grigi", "Greys")}
DEFAULT_BLANK = {"method": "coverage", "drop": 2.5, "doi_max": 0.4}
MAX_PIXELS = 1_500_000
MAX_3D_CELLS = 60_000


# ------------------------------------------------------------------ the grid
def blanked(result: dict, blank: dict | None = None) -> np.ndarray:
    """The cells of the mesh left out of the section (True), by ``blank`` (:data:`DEFAULT_BLANK`)."""
    blank = {**DEFAULT_BLANK, **(blank or {})}
    model = np.asarray(result["model"], float)
    out = ~np.isfinite(model) | (model <= 0)
    method = blank.get("method", "coverage")
    if method == "doi" and "doi" in result:
        out |= np.asarray(result["doi"], float) > float(blank["doi_max"])
    elif method in ("coverage", "doi"):
        cov = np.asarray(result["coverage"], float)
        ok = np.isfinite(cov)
        if ok.any():
            out |= ~ok | (cov < np.percentile(cov[ok], 95) - float(blank["drop"]))
    return out


def rasterise(result: dict, cell: float | None = None, blank: dict | None = None) -> dict:
    """The cells of the mesh on a regular grid of the (u, v) frame: ``values`` (rows from the top,
    ohm m, NaN outside the mesh and on blanked cells), ``coverage``, ``cell``, ``origin`` (u of the
    left edge, v of the top edge)."""
    from matplotlib.tri import Triangulation

    nodes = np.asarray(result["nodes"], float)
    tris = np.asarray(result["cells"], np.int64)
    owner = np.asarray(result.get("cell_of", np.arange(len(tris))), np.int64)
    model = np.asarray(result["model"], float)
    cov = np.asarray(result.get("coverage", np.zeros(len(model))), float)
    hide = blanked(result, blank)
    u0, v0 = nodes.min(axis=0)
    u1, v1 = nodes.max(axis=0)
    if cell is None:
        spacing = float((result.get("fit") or {}).get("spacing") or 0) or (u1 - u0) / 50.0
        cell = spacing / 4.0
        while (u1 - u0) * (v1 - v0) / cell ** 2 > MAX_PIXELS:
            cell *= 1.25
    cell = float(cell)
    w = max(1, int(np.ceil((u1 - u0) / cell)))
    h = max(1, int(np.ceil((v1 - v0) / cell)))
    uu = u0 + (np.arange(w) + 0.5) * cell
    vv = v1 - (np.arange(h) + 0.5) * cell
    U, V = np.meshgrid(uu, vv)
    tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    k = tri.get_trifinder()(U.ravel(), V.ravel()).reshape(h, w)
    inside = k >= 0
    c = np.where(inside, owner[np.clip(k, 0, None)], 0)
    values = np.where(inside & ~hide[c], model[c], np.nan)
    coverage = np.where(inside, cov[c], np.nan)
    return {"values": values.astype(np.float32), "coverage": coverage.astype(np.float32), "cell": cell,
            "origin": (float(u0), float(v1)), "shape": (h, w)}


def limits(values: np.ndarray) -> tuple[float, float]:
    """The ends of the colour ramp in ohm m: the 2nd and 98th percentiles of the values shown, on a
    log scale."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return 1.0, 1000.0
    lo, hi = np.percentile(np.log10(v), [2, 98])
    if hi - lo < 0.05:
        lo, hi = lo - 0.1, hi + 0.1
    return float(10 ** lo), float(10 ** hi)


def ramp(x: np.ndarray, palette: str) -> np.ndarray:
    stops = PALETTES.get(palette, PALETTES["rainbow"])
    pos = np.array([s[0] for s in stops])
    out = np.zeros(x.shape + (3,), np.uint8)
    for k in range(3):
        out[..., k] = np.interp(x, pos, [s[1][k] for s in stops]).astype(np.uint8)
    return out


def colours(values: np.ndarray, lo: float, hi: float, palette: str = "rainbow") -> np.ndarray:
    """RGBA (h, w, 4): log10 of ``values`` between ``lo`` and ``hi`` (ohm m), transparent where NaN."""
    ok = np.isfinite(values) & (values > 0)
    x = np.clip((np.log10(np.where(ok, values, lo)) - np.log10(lo)) / max(np.log10(hi) - np.log10(lo), 1e-9), 0, 1)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    rgba[..., :3] = ramp(x, palette)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def ticks(lo: float, hi: float) -> list[float]:
    """Round values between ``lo`` and ``hi`` for a log colour bar: 1, 2, 5 times the powers of ten."""
    out = []
    mantissas = (1, 2, 5) if np.log10(hi / lo) > 0.9 else (1, 1.5, 2, 3, 5, 7)
    for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1):
        for m in mantissas:
            v = m * 10.0 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    if len(out) > 7:
        out = [v for v in out if abs(np.log10(v) - round(np.log10(v))) < 1e-9] or out[:: max(1, len(out) // 6)]
    return out


def number(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if v >= 10 else f"{v:g}"


# ------------------------------------------------------------------ the files
def inversion_path(chunk, pid: str) -> Path:
    """The result of the inversion the record uses (``inversion-<run>.npz``)."""
    e = ert_mod.entry_of(chunk, pid)
    rel = ((e.get("inversion") or {}).get("files") or {}).get("result")
    return Path(chunk.data_dir) / rel if rel else ert_mod.folder(chunk, e["id"]) / "inversion.npz"


def load_result(chunk, pid: str) -> dict:
    e = ert_mod.entry_of(chunk, pid)
    inv = e.get("inversion") or {}
    path = inversion_path(chunk, e["id"])
    if not path.exists():
        ert_mod.restore(chunk, e["id"])
    if not path.exists():
        raise KeyError(t("Il profilo {0} non è ancora invertito.", "The profile {0} is not inverted yet.").format(e["name"]))
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["fit"] = dict(inv.get("fit") or {})
    return out


def write(chunk, pid: str, blank: dict | None = None, palette: str | None = None, lims=None) -> dict:
    """The files of the section of the profile from its inversion (written again with other blanking,
    palette or limits), under new names every time - ``section-<stamp>.tif`` (float32, ohm m) and
    ``section_view-<stamp>.tif`` (RGBA) - so that Undo finds the files of the drawing it takes back; the
    record of the inversion updated (``display``, ``files``)."""
    from datetime import datetime

    import rasterio
    from rasterio.transform import from_origin

    e = ert_mod.entry_of(chunk, pid)
    inv = e["inversion"]
    disp = dict(inv.get("display") or {})
    if blank is not None:
        disp["blank"] = {**DEFAULT_BLANK, **blank}
    disp.setdefault("blank", dict(DEFAULT_BLANK))
    if palette:
        disp["palette"] = palette
    disp.setdefault("palette", "rainbow")
    res = load_result(chunk, pid)
    grid = rasterise(res, blank=disp["blank"])
    values = grid["values"]
    if lims is not None:
        disp["limits"] = [float(lims[0]), float(lims[1])]
    elif not disp.get("limits") or blank is not None:
        disp["limits"] = list(limits(values))
    lo, hi = disp["limits"]
    out = ert_mod.folder(chunk, e["id"])
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    sec_path, view = out / f"section-{stamp}.tif", out / f"section_view-{stamp}.tif"
    u0, vtop = grid["origin"]
    cell = grid["cell"]
    tr = from_origin(u0, vtop, cell, cell)
    h, w = values.shape
    prof = {"driver": "GTiff", "width": w, "height": h, "transform": tr, "compress": "deflate"}
    with rasterio.open(sec_path, "w", count=1, dtype="float32", nodata=np.nan, **prof) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, "resistivity (ohm m)")
        dst.update_tags(ppg_frame="ert-section", ppg_profile=e["id"], ppg_name=e["name"], ppg_units="ohm m",
                        ppg_axes="u = metres along the line, v = elevation")
    rgba = colours(values, lo, hi, disp["palette"])
    with rasterio.open(view, "w", count=4, dtype="uint8", **prof) as dst:
        dst.write(np.moveaxis(rgba, -1, 0))
        dst.update_tags(ppg_frame="ert-section", ppg_profile=e["id"])
    shown = np.isfinite(values)
    disp["shown_m2"] = round(float(shown.sum()) * cell * cell, 3)
    disp["cell"] = cell
    disp["bounds"] = [u0, vtop - h * cell, u0 + w * cell, vtop]
    inv["display"] = disp
    inv["files"] = {**(inv.get("files") or {}), "section": str(sec_path.relative_to(chunk.data_dir)),
                    "view": str(view.relative_to(chunk.data_dir)),
                    "result": str(inversion_path(chunk, e["id"]).relative_to(chunk.data_dir))}
    return inv


def view_path(chunk, pid: str) -> Path | None:
    e = ert_mod.entry_of(chunk, pid)
    rel = ((e.get("inversion") or {}).get("files") or {}).get("view")
    p = Path(chunk.data_dir) / rel if rel else None
    if p is not None and not p.exists():
        ert_mod.restore(chunk, e["id"])
    return p if p is not None and p.exists() else None


def read_section(chunk, pid: str) -> tuple[np.ndarray, tuple, float]:
    import rasterio

    e = ert_mod.entry_of(chunk, pid)
    with rasterio.open(Path(chunk.data_dir) / e["inversion"]["files"]["section"]) as src:
        v = src.read(1).astype(np.float64)
        tr = src.transform
    return v, (tr.c, tr.f), float(tr.a)


def write_cells_csv(chunk, pid: str, path: str | Path) -> Path:
    """The cells of the inversion mesh: their centre in the frame of the section (u, v) and in the
    world (x, y, z), resistivity, its log10, the coverage and whether they are blanked."""
    e = ert_mod.entry_of(chunk, pid)
    res = load_result(chunk, pid)
    prof = ert_mod.load(chunk, pid)
    line = (e.get("placement") or {}).get("line")
    nodes = np.asarray(res["nodes"], float)
    tris = np.asarray(res["cells"], np.int64)
    owner = np.asarray(res.get("cell_of", np.arange(len(tris))), np.int64)
    model = np.asarray(res["model"], float)
    hide = blanked(res, ((e.get("inversion") or {}).get("display") or {}).get("blank"))
    cu = np.zeros(len(model))
    cv = np.zeros(len(model))
    cnt = np.zeros(len(model))
    cen = nodes[tris].mean(axis=1)
    np.add.at(cu, owner, cen[:, 0])
    np.add.at(cv, owner, cen[:, 1])
    np.add.at(cnt, owner, 1)
    cu, cv = cu / np.maximum(cnt, 1), cv / np.maximum(cnt, 1)
    xy = ert_mod.line_point(line, cu) if line and prof.placed() else np.full((len(cu), 2), np.nan)
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "u", "v", "x", "y", "z", "rho_ohm_m", "log10_rho", "coverage", "blanked"])
        cov = np.asarray(res.get("coverage", np.full(len(model), np.nan)), float)
        for i in range(len(model)):
            w.writerow([i, round(cu[i], 4), round(cv[i], 4), round(xy[i, 0], 4), round(xy[i, 1], 4), round(cv[i], 4),
                        round(model[i], 4), round(np.log10(model[i]), 5) if model[i] > 0 else "",
                        round(cov[i], 4), int(hide[i])])
    return path


def write_xyz(chunk, pid: str, path: str | Path) -> Path:
    """The pixels of the section with a value: ``u v x y z rho`` per line (space separated)."""
    e = ert_mod.entry_of(chunk, pid)
    v, (u0, vtop), cell = read_section(chunk, pid)
    rows, cols = np.nonzero(np.isfinite(v))
    uu = u0 + (cols + 0.5) * cell
    zz = vtop - (rows + 0.5) * cell
    line = (e.get("placement") or {}).get("line")
    xy = ert_mod.line_point(line, uu) if line else np.full((len(uu), 2), np.nan)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# u v x y z rho_ohm_m\n")
        for i in range(len(uu)):
            f.write(f"{uu[i]:.3f} {zz[i]:.3f} {xy[i, 0]:.3f} {xy[i, 1]:.3f} {zz[i]:.3f} {v[rows[i], cols[i]]:.4g}\n")
    return path


def write_geotiff(chunk, pid: str, path: str | Path) -> Path:
    import shutil

    e = ert_mod.entry_of(chunk, pid)
    src = Path(chunk.data_dir) / e["inversion"]["files"]["section"]
    path = Path(path)
    shutil.copyfile(src, path)
    frame = {"profile": e["name"], "frame": "u = metres along the line from its start, v = elevation (m)",
             "line": (e.get("placement") or {}).get("line"), "crs": (e.get("placement") or {}).get("crs"),
             "units": "ohm m"}
    path.with_suffix(".frame.json").write_text(json.dumps(frame, indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------ figures
def figure(chunk, pid: str, path: str | Path, kind: str = "section", dpi: int = 200) -> Path:
    """A figure of the section (``kind`` "section") or of the pseudosection ("pseudosection"): axes in
    metres, the electrodes, a log colour bar in ohm m, the fit; PNG, SVG or PDF by the extension."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.figure import Figure

    e = ert_mod.entry_of(chunk, pid)
    prof = ert_mod.load(chunk, pid)
    path = Path(path)
    disp = ((e.get("inversion") or {}).get("display") or {})
    pal = disp.get("palette", "rainbow")
    stops = PALETTES.get(pal, PALETTES["rainbow"])
    cmap = LinearSegmentedColormap.from_list(pal, [(p, tuple(c / 255 for c in rgb)) for p, rgb in stops]) \
        .with_extremes(bad=(1, 1, 1, 0))
    x_el, z_el = prof.positions(), prof.heights()
    if kind == "section":
        v, (u0, vtop), cell = read_section(chunk, pid)
        h, w = v.shape
        rows = np.flatnonzero(np.isfinite(v).any(axis=1))
        width, height = w * cell, ((rows[-1] - rows[0] + 1) * cell if rows.size else h * cell)
    else:
        valid = ert_mod.valid_mask(prof, e.get("filters"))
        x, d = ert_mod.pseudo_positions(prof)
        width = float(np.nanmax(x_el) - np.nanmin(x_el)) or 1.0
        height = float(np.nanmax(d)) or 1.0
    ratio = height / max(width, 1e-9)
    exaggeration = 1.0 if ratio >= 0.25 else max(1.0, round(2 * 0.25 / ratio) / 2)
    fig_h = min(max(0.85 * 11 * ratio * exaggeration + 1.4, 3.0), 8.5)
    fig = Figure(figsize=(11, fig_h), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if kind == "section":
        lo, hi = disp.get("limits") or limits(v)
        im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, norm=LogNorm(lo, hi), interpolation="nearest",
                       extent=(u0, u0 + w * cell, vtop - h * cell, vtop), origin="upper")
        if rows.size:
            ax.set_ylim(vtop - (rows[-1] + 1) * cell, max(vtop - rows[0] * cell, float(np.max(z_el))) + cell)
        ax.plot(x_el, z_el, "v", color="black", ms=3, label=t("elettrodi", "electrodes"))
        ax.plot(x_el, z_el, "-", color="black", lw=0.6)
        ax.set_ylabel(t("quota (m)", "elevation (m)"))
        fit = (e.get("inversion") or {}).get("fit") or {}
        title = t("{0}: resistività (inversione, χ² {1:.2f}, RMS {2:.1f} %)", "{0}: resistivity (inversion, χ² {1:.2f}, "
                  "RMS {2:.1f} %)").format(e["name"], fit.get("chi2", float("nan")), fit.get("rrms", float("nan")))
    else:
        rhoa = prof.apparent()
        ok = valid & np.isfinite(rhoa) & (rhoa > 0)
        lo, hi = limits(rhoa[ok])
        im = ax.scatter(x[ok], -d[ok], c=rhoa[ok], cmap=cmap, norm=LogNorm(lo, hi), s=14, marker="s", linewidths=0)
        if (~ok).any():
            ax.scatter(x[~ok], -d[~ok], c="#9e9e9e", s=10, marker="x", label=t("letture tolte", "readings removed"))
            ax.legend(loc="lower right", fontsize=8)
        ax.plot(x_el, np.zeros_like(x_el), "v", color="black", ms=3)
        ax.set_ylabel(t("pseudoprofondità (m)", "pseudo-depth (m)"))
        title = t("{0}: pseudosezione della resistività apparente ({1} letture)",
                  "{0}: pseudosection of the apparent resistivity ({1} readings)").format(e["name"], int(ok.sum()))
    ax.set_aspect(exaggeration)
    if exaggeration > 1:
        title += t(" - esagerazione verticale x{0:g}", " - vertical exaggeration x{0:g}").format(exaggeration)
    ax.set_xlabel(t("distanza lungo la linea (m)", "distance along the line (m)"))
    ax.set_title(title, fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    bar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, aspect=25)
    marks = ticks(lo, hi)
    bar.set_ticks(marks)
    bar.set_ticklabels([number(m) for m in marks])
    bar.ax.minorticks_off()
    bar.set_label(t("resistività (Ω·m)", "resistivity (Ω·m)") if kind == "section" else
                  t("resistività apparente (Ω·m)", "apparent resistivity (Ω·m)"))
    fig.savefig(path, dpi=dpi)
    return path


# ------------------------------------------------------------------ 3D
def section_mesh(chunk, pid: str, max_cells: int = MAX_3D_CELLS):
    """The section as vertical quads in world coordinates (x, y from the line, z = v), each with its
    colour: vertices (V, 3), faces (F, 3), colours (V, 3). None without an inversion or a line."""
    e = ert_mod.entry_of(chunk, pid)
    line = (e.get("placement") or {}).get("line")
    if not line or not (e.get("inversion") or {}).get("files"):
        return None
    v, (u0, vtop), cell = read_section(chunk, pid)
    lo, hi = ((e["inversion"].get("display") or {}).get("limits")) or limits(v)
    pal = (e["inversion"].get("display") or {}).get("palette", "rainbow")
    step = 1
    while np.isfinite(v[::step, ::step]).sum() > max_cells:
        step += 1
    vs = v[::step, ::step]
    c = cell * step
    rows, cols = np.nonzero(np.isfinite(vs))
    if not len(rows):
        return None
    rgb = colours(vs, lo, hi, pal)[rows, cols, :3]
    ua, ub = u0 + cols * c, u0 + (cols + 1) * c
    za, zb = vtop - rows * c, vtop - (rows + 1) * c
    pa, pb = ert_mod.line_point(line, ua), ert_mod.line_point(line, ub)
    n = len(rows)
    verts = np.empty((n, 4, 3))
    verts[:, 0, :2], verts[:, 0, 2] = pa, za
    verts[:, 1, :2], verts[:, 1, 2] = pb, za
    verts[:, 2, :2], verts[:, 2, 2] = pb, zb
    verts[:, 3, :2], verts[:, 3, 2] = pa, zb
    base = (np.arange(n) * 4)[:, None]
    faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]])
    cols_v = np.repeat(rgb, 4, axis=0)
    return verts.reshape(-1, 3), faces.astype(np.int64), cols_v.astype(np.uint8)


def _side(line, xy) -> tuple[np.ndarray, np.ndarray]:
    """For points ``xy``: the signed distance from the polyline (positive on its left) and the
    distance along it of the nearest point."""
    P = np.asarray(line, float)[:, :2]
    best = np.full(len(xy), np.inf)
    sign = np.zeros(len(xy))
    along = np.zeros(len(xy))
    cum = 0.0
    for p0, p1 in zip(P[:-1], P[1:]):
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-12:
            continue
        f = np.clip(((xy - p0) @ d) / (L * L), 0, 1)
        q = p0 + f[:, None] * d
        dist = np.hypot(*(xy - q).T)
        closer = dist < best
        cross = d[0] * (xy[:, 1] - p0[1]) - d[1] * (xy[:, 0] - p0[0])
        best = np.where(closer, dist, best)
        sign = np.where(closer, np.sign(cross), sign)
        along = np.where(closer, cum + f * L, along)
        cum += L
    return sign * best, along


MAX_TERRAIN_FACES = 300_000


def _model_mesh(chunk, max_faces: int | None = None):
    """The model of the chunk for the scene (internal frame: vertices, faces, colours or None), read
    without writing anything: the program's own preview when it is up to date, the model itself when
    it is small, else a copy decimated in memory. (The program's ``display_model`` would write a
    preview of the size asked, which its own 3D view would then reuse.)"""
    from pyarchinit_photogrammetry.io.mesh_ply import face_count, read_mesh

    max_faces = MAX_TERRAIN_FACES if max_faces is None else max_faces
    model = chunk.product_path("model")
    if model is None or not Path(model).exists():
        return None
    model = Path(model)
    preview = model.parent / "model_preview.ply"
    if preview.exists() and preview.stat().st_mtime >= model.stat().st_mtime:
        m = read_mesh(preview)
        return np.asarray(m.xyz, float), np.asarray(m.faces, np.int64), m.colors
    n = face_count(model)
    if n is not None and n > max_faces:
        from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh, cluster_decimate

        mesh = PlyMesh(model)
        V, F, C = cluster_decimate(mesh.xyz(), mesh.face_chunks, mesh.colors(), max_faces)
        return np.asarray(V, float), np.asarray(F, np.int64), C
    m = read_mesh(model)
    return np.asarray(m.xyz, float), np.asarray(m.faces, np.int64), m.colors


def _terrain(chunk, kind: str = "auto"):
    """The ground of the chunk for the 3D scene in world coordinates: the model (:func:`_model_mesh`),
    else the DEM as a mesh of at most 400 x 400 cells. None when there is neither."""
    if kind in ("auto", "model"):
        try:
            m = _model_mesh(chunk)
        except Exception:  # noqa: BLE001 - an unreadable model: the DEM
            m = None
        if m is not None and len(m[1]):
            xyz, faces, c = m
            world = chunk.to_world(np.asarray(xyz, float))
            rgb = np.asarray(c)[:, :3].astype(np.uint8) if c is not None and len(c) == len(world) else \
                np.tile(np.array([[170, 150, 120]], np.uint8), (len(world), 1))
            return world, np.asarray(faces, np.int64), rgb
    if kind in ("auto", "dem"):
        dem = getattr(chunk, "dem_path", None)
        if dem is not None and Path(dem).exists():
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(dem) as src:
                f = max(1.0, max(src.width, src.height) / 400.0)
                h, w = int(src.height / f), int(src.width / f)
                z = src.read(1, out_shape=(h, w), masked=True, resampling=Resampling.average).astype(float).filled(np.nan)
                tr = src.transform @ src.transform.scale(src.width / w, src.height / h)
            cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
            xs, ys = tr @ (cols, rows)
            V = np.column_stack([np.ravel(xs), np.ravel(ys), z.ravel()])
            idx = np.arange(h * w).reshape(h, w)
            q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
            F = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            F = F[np.isfinite(V[F, 2]).all(axis=1)]
            zf = V[:, 2]
            lo, hi = np.nanpercentile(zf, [2, 98]) if np.isfinite(zf).any() else (0, 1)
            g = np.clip((np.nan_to_num(zf, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
            rgb = np.column_stack([150 + 60 * g, 140 + 50 * g, 110 + 40 * g]).astype(np.uint8)
            return V, F, rgb
    return None


def _cut(world, faces, rgb, lines, side: str, margin: float):
    """The faces of the ground on one side of every line (``side`` "left", "right"), within ``margin``
    metres of a line, or all of them ("all")."""
    if side == "all" or not lines:
        return world, faces, rgb
    cen = world[faces].mean(axis=1)[:, :2]
    keep = np.zeros(len(faces), bool)
    for line in lines:
        s, along = _side(line, cen)
        length = ert_mod.line_length(line)
        inside = (np.abs(s) <= margin) & (along > -margin) & (along < length + margin)
        keep |= inside & ((s > 0) if side == "left" else (s < 0))
    return world, faces[keep], rgb


def _units(chunk):
    """The solids of the units of the geoarchaeology plugin (their PLY for the 3D view), in the
    internal frame of the chunk: (vertices, faces, colours) or None."""
    reg = ((chunk.meta.get("geoarchaeology") or {}).get("model") or {})
    rel = (reg.get("files") or {}).get("display")
    p = Path(chunk.data_dir) / rel if rel else None
    if p is None or not p.exists():
        return None
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    m = read_mesh(p)
    rgb = m.colors if m.colors is not None and len(m.colors) == len(m.xyz) else \
        np.tile(np.array([[160, 160, 160]], np.uint8), (len(m.xyz), 1))
    return np.asarray(m.xyz, float), np.asarray(m.faces, np.int64), np.asarray(rgb, np.uint8)[:, :3]


# the colours the geoarchaeology plugin gives the units without one of their own (its boreholes.PALETTE)
UNIT_PALETTE = ["#d9b77e", "#7d5a3c", "#c9a55c", "#a0522d", "#e6d3a3", "#6b6b4b", "#c27c4c", "#8c9a6b",
                "#b0846a", "#596e79", "#e0c08a", "#704f3c", "#a39e7a", "#8a5a6b", "#cfa98a", "#4f5d4f"]
# the colours of the program's proxies of the units, when its gui.scene cannot be imported
PROXY_COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49), (145, 30, 180)]


def _hex(colour: str | None, default=(150, 150, 150)) -> tuple[int, int, int]:
    s = str(colour or "").strip().lstrip("#")
    if len(s) != 6:
        return default
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return default


def _proxies(chunk):
    """The program's proxies of the stratigraphic units (``chunk.meta["proxies"]``, PLY in world
    coordinates under ``<data>/proxies/``), each in the colour the program's 3D view gives it, in
    world coordinates: (vertices, faces, colours, count) or None."""
    entries = chunk.meta.get("proxies") or []
    if not entries:
        return None
    try:
        from pyarchinit_photogrammetry.gui.scene import PROXY_COLORS

        palette = [tuple(int(v) for v in c) for c in np.asarray(PROXY_COLORS)]
    except Exception:  # noqa: BLE001 - no window: the first colours, copied
        palette = PROXY_COLOURS
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    V, F, C, n, count = [], [], [], 0, 0
    for i, entry in enumerate(entries):
        p = Path(chunk.data_dir) / str(entry.get("file") or "")
        if not entry.get("file") or not p.exists():
            continue
        try:
            m = read_mesh(p)
        except Exception:  # noqa: BLE001 - an unreadable proxy is left out
            continue
        xyz = np.asarray(m.xyz, float)
        V.append(xyz)
        F.append(np.asarray(m.faces, np.int64) + n)
        C.append(np.tile(np.array([palette[i % len(palette)]], np.uint8), (len(xyz), 1)))
        n += len(xyz)
        count += 1
    if not V:
        return None
    return np.vstack(V), np.vstack(F), np.vstack(C), count


def _cores(chunk, lines, margin: float, width: float):
    """The cores of the geoarchaeology plugin near the lines as square prisms, every interval in its
    own colour when it has one (``#rrggbb``), else in the colour of its unit (the unit's own, else the
    one the geoarchaeology plugin gives it), in world coordinates."""
    g = chunk.meta.get("geoarchaeology") or {}
    holes = g.get("boreholes") or []
    if not holes:
        return None
    colour = {u.get("code"): _hex(u.get("colour"), _hex(UNIT_PALETTE[k % len(UNIT_PALETTE)]))
              for k, u in enumerate(g.get("units") or [])}
    dem = getattr(chunk, "dem_path", None)
    V, F, C = [], [], []
    n = 0
    hw = width / 2
    for b in holes:
        xy = np.array([[float(b["x"]), float(b["y"])]])
        if lines and min(abs(_side(line, xy)[0][0]) for line in lines) > margin:
            continue
        z = b.get("z")
        if z is None and dem is not None and Path(dem).exists():
            z = float(ert_mod.sample_dem(dem, xy)[0])
        if z is None or not np.isfinite(z):
            continue
        for iv in b.get("intervals") or []:
            top, base = float(z) - float(iv["from"]), float(z) - float(iv["to"])
            x, y = xy[0]
            corners = np.array([[x - hw, y - hw], [x + hw, y - hw], [x + hw, y + hw], [x - hw, y + hw]])
            verts = np.vstack([np.column_stack([corners, np.full(4, top)]), np.column_stack([corners, np.full(4, base)])])
            quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
            for q in quads:
                F.append([n + q[0], n + q[1], n + q[2]])
                F.append([n + q[0], n + q[2], n + q[3]])
            V.append(verts)
            rgb = _hex(iv.get("colour"), colour.get(iv.get("unit"), (150, 150, 150)))
            C.append(np.tile(np.array([rgb], np.uint8), (8, 1)))
            n += 8
    if not V:
        return None
    return np.vstack(V), np.asarray(F, np.int64), np.vstack(C)


SCENE = "vista_3d.ply"


def scene(chunk, pids=None, terrain: str = "auto", side: str = "left", units: bool = True, cores: bool = True,
          lift: float = 0.0) -> tuple[Path | None, dict]:
    """The PLY of the 3D view (``<data>/geophysics/ert/vista_3d.ply``, the internal frame of the
    chunk): the sections of ``pids`` (all the inverted ones by default), the ground (``terrain``
    "auto", "model", "dem" or "none") cut away on the ``side`` of the lines ("left", "right"; "all"
    keeps it whole) within a margin, the program's proxies of the stratigraphic units, the units and
    the cores of the geoarchaeology plugin (``units``: the proxies and the units). ``lift``:
    metres the sections are raised (to see them over an opaque ground). Returns the file (None
    without a section) and what is in it."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    if pids is None:
        pids = [e["id"] for e in ert_mod.entries(chunk) if (e.get("inversion") or {}).get("files")]
    parts, lines, what = [], [], {"sections": [], "terrain": None, "units": False, "proxies": 0, "cores": 0}
    for pid in pids:
        m = section_mesh(chunk, pid)
        if m is None:
            continue
        v, f, c = m
        v = v.copy()
        v[:, 2] += float(lift)
        parts.append((chunk.to_internal(v), f, c))
        e = ert_mod.entry_of(chunk, pid)
        lines.append(e["placement"]["line"])
        what["sections"].append(e["name"])
    if not parts:
        return None, what
    length = max(ert_mod.line_length(line) for line in lines)
    margin = max(0.5 * length, 10.0)
    if terrain != "none":
        g = _terrain(chunk, terrain)
        if g is not None:
            world, faces, rgb = _cut(*g, lines, side, margin)
            if len(faces):
                used = np.unique(faces)
                remap = np.full(len(world), -1, np.int64)
                remap[used] = np.arange(len(used))
                parts.append((chunk.to_internal(world[used]), remap[faces], rgb[used]))
                what["terrain"] = terrain if terrain != "auto" else "model/dem"
    if units:
        px = _proxies(chunk)
        if px is not None:
            parts.append((chunk.to_internal(px[0]), px[1], px[2]))
            what["proxies"] = px[3]
        u = _units(chunk)
        if u is not None:
            parts.append(u)
            what["units"] = True
    if cores:
        c = _cores(chunk, lines, margin, max(0.01 * length, 0.08))
        if c is not None:
            v, f, col = c
            parts.append((chunk.to_internal(v), f, col))
            what["cores"] = int(len(f) // 12)
    V, F, C, n = [], [], [], 0
    for v, f, c in parts:
        V.append(np.asarray(v, float))
        F.append(np.asarray(f, np.int64) + n)
        C.append(np.asarray(c, np.uint8)[:, :3])
        n += len(v)
    out = Path(chunk.data_dir) / ert_mod.FOLDER
    out.mkdir(parents=True, exist_ok=True)
    path = write_mesh_ply(out / SCENE, np.vstack(V), np.vstack(F), np.vstack(C), double=True)
    return path, what
