"""Stratigraphic columns of the boreholes, drawn with matplotlib.

One borehole: its column at a scale (1:10 to 1:1000, chosen to fit an A4 page when not given)
with the elevations on the left and the depths on the right when the collar has an elevation
(the depths alone otherwise), and the description and colour of every interval beside it.
Several: a panel of columns side by side, on a common elevation axis when every collar has an
elevation (on depth otherwise), the tops of the same unit joined between neighbouring columns by
dashed lines, a legend of the units. The units take their colours (the Munsell colour of the log
when given) and the hatch of the lithology the log names (:func:`boreholes.hatch_of`: clay, silt,
sand, gravel, stones, mudbrick, ash, peat, organic).

SVG and PNG in ``<chunk data>/geoarchaeology/logs/`` unless another path is given: the column of
one borehole is ``<borehole>.svg``, a panel ``colonne_<the boreholes>.svg``, so a panel never
replaces another. A vector format this installation cannot write is skipped and noted, the PNG
kept (:mod:`figures`).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from . import boreholes as bh
from .figures import Written, save
from .texts import num, t

LOG_SCALES = (10, 20, 50, 100, 200, 500, 1000)
MAX_HEIGHT_MM = 250.0  # the height of the column on an A4 page


def folder(chunk) -> Path:
    return bh.folder(chunk) / "logs"


def panel_name(ids) -> str:
    """The file name of the panel of ``ids``: ``colonne_S5-S4-S3`` (shortened with a hash of the
    whole list when long)."""
    import hashlib

    from .surfaces import slug

    joined = "-".join(slug(i) for i in ids)
    if len(joined) > 60:
        joined = joined[:48] + "-" + hashlib.sha1("\n".join(map(str, ids)).encode()).hexdigest()[:8]
    return "colonne_" + joined


def _scale(height_m: float, scale: float | None) -> float:
    if scale:
        return float(scale)
    return float(next((s for s in LOG_SCALES if height_m * 1000.0 / s <= MAX_HEIGHT_MM), LOG_SCALES[-1]))


def draw(chunk, ids=None, path: str | Path | None = None, vertical: str = "auto", scale: float | None = None,
         correlate: bool = True) -> Written:
    """The log of one borehole, or the panel of several (``ids``: all when None, in their order);
    ``vertical``: "elevation", "depth" or "auto" (elevations when every collar has one).
    ``path``: the file to write (its suffix .svg, .png or .pdf chooses the format), SVG and PNG in
    the logs folder when None. Returns the files written, with ``notes`` on the formats this
    installation cannot write (the PNG is written then)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    every = bh.load(chunk)
    if ids:
        by = {b.id: b for b in every}
        missing = [i for i in ids if str(i) not in by]
        if missing:
            raise ValueError(t("Sondaggi sconosciuti: {0}", "Unknown boreholes: {0}").format(", ".join(map(str, missing))))
        holes = [by[str(i)] for i in ids]
    else:
        holes = every
    holes = [b for b in holes if b.intervals]
    if not holes:
        raise ValueError(t("Nessun sondaggio con intervalli da disegnare", "No borehole with intervals to draw"))
    if vertical == "auto":
        vertical = "elevation" if all(b.z is not None for b in holes) else "depth"
    if vertical == "elevation" and any(b.z is None for b in holes):
        raise ValueError(t("Un sondaggio senza quota: disegna in profondità", "A borehole without elevation: draw by depth"))
    single = len(holes) == 1
    colours = bh.colours(chunk)
    units = {u["code"]: u for u in bh.units(chunk)}

    def y(b, depth):
        return b.z - depth if vertical == "elevation" else depth

    if vertical == "elevation":
        hi = max(b.z for b in holes)
        lo = min(b.z - b.bottom for b in holes)
    else:
        hi, lo = 0.0, max(b.bottom for b in holes)
    span = abs(hi - lo) or 1.0
    scale = _scale(span, scale)
    k = 1000.0 / scale
    col_w, gap = (18.0, 0.0) if single else (12.0, 22.0)
    text_w = 85.0 if single else 70.0
    left, bottom, top = 24.0, 10.0, 22.0
    right = (14.0 if vertical == "elevation" else 6.0) + text_w
    plot_w = col_w + (0 if single else (len(holes) - 1) * (col_w + gap))
    plot_h = max(span * k, 30.0)
    W, H = left + plot_w + right, bottom + plot_h + top
    fig = Figure(figsize=(W / 25.4, H / 25.4))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([left / W, bottom / H, plot_w / W, plot_h / H])
    ax.set_xlim(0, plot_w)
    if vertical == "elevation":
        ax.set_ylim(hi - plot_h / k, hi)
        ax.set_ylabel(t("Quota (m)", "Elevation (m)"), fontsize=7)
    else:
        ax.set_ylim(plot_h / k, 0.0)
        ax.set_ylabel(t("Profondità (m)", "Depth (m)"), fontsize=7)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    data, _notes = bh.contact_data(holes, bh.order(chunk))
    contact = {(bid, cd.k): d for cd in data for bid, d in zip(cd.ids, cd.depth)}
    xs = []
    for i, b in enumerate(holes):
        x0 = i * (col_w + gap)
        xs.append(x0)
        for iv in b.intervals:
            colour = colours.get(iv.unit, "#bbbbbb")
            hatch = bh.hatch_of(iv.lithology, iv.description) or None
            y0, y1 = sorted((y(b, iv.top), y(b, iv.base)))
            ax.add_patch(Rectangle((x0, y0), col_w, y1 - y0, facecolor=colour, edgecolor="black", linewidth=0.5,
                                   hatch=hatch, zorder=2))
            if (iv.base - iv.top) * k >= 3.0:
                written = (iv.unit_raw or iv.unit) if single else iv.unit
                ax.text(x0 + col_w / 2, (y0 + y1) / 2, written, ha="center", va="center", fontsize=6, zorder=3,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "linewidth": 0})
            if single:
                words = f"{num(iv.top, 2)}-{num(iv.base, 2)} m  {iv.unit_raw or iv.unit}" + (
                    f": {iv.description}" if iv.description else "")
                if iv.colour:
                    words += f" ({iv.colour})"
                ax.text(col_w + (16.0 if vertical == "elevation" else 4.0), (y0 + y1) / 2,
                        "\n".join(textwrap.wrap(words, 52)), ha="left", va="center", fontsize=5.5, clip_on=False)
        yb = y(b, b.bottom)
        ax.plot([x0 - 1.5, x0 + col_w + 1.5], [yb, yb], color="black", linewidth=1.0, zorder=4, clip_on=False)
        label = b.id if b.z is None else f"{b.id}\n{num(b.z, 2)} m"
        ax.annotate(label, (x0 + col_w / 2, y(b, 0.0)), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold", annotation_clip=False)
    if correlate and not single:
        for i in range(len(holes) - 1):
            a, c = holes[i], holes[i + 1]
            for cd in data:
                da, dc = contact.get((a.id, cd.k)), contact.get((c.id, cd.k))
                if da is None or dc is None or cd.k == 0:
                    continue
                ax.plot([xs[i] + col_w, xs[i + 1]], [y(a, da), y(c, dc)], color=colours.get(cd.unit, "#555555"),
                        linestyle="--", linewidth=0.7, zorder=1)
    if vertical == "elevation" and single:
        twin = ax.twinx()
        lo_, hi_ = ax.get_ylim()
        b = holes[0]
        twin.set_ylim(b.z - lo_, b.z - hi_)
        twin.tick_params(labelsize=6)
        twin.set_ylabel(t("Profondità (m)", "Depth (m)"), fontsize=7)
        for side in ("top", "bottom", "left"):
            twin.spines[side].set_visible(False)
    title = (t("Sondaggio {0}", "Borehole {0}").format(holes[0].id) if single
             else t("Colonne stratigrafiche ({0} sondaggi)", "Stratigraphic columns ({0} boreholes)").format(len(holes)))
    fig.text(left / W, 1 - 5 / H, f"{title} - 1:{scale:g}", fontsize=8, va="top")
    if not single:
        present = list(dict.fromkeys(iv.unit for b in holes for iv in sorted(b.intervals, key=lambda i: i.top)))
        order = [u for u in bh.order(chunk) if u in present] + [u for u in present if u not in bh.order(chunk)]
        handles = []
        for u in order:
            info = units.get(u, {})
            hatch = bh.hatch_of(info.get("lithology", ""), info.get("description", "")) or None
            handles.append(Patch(facecolor=colours.get(u, "#bbbbbb"), edgecolor="black", linewidth=0.4, hatch=hatch,
                                 label=u + (f" - {info['description'][:30]}" if info.get("description") else "")))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((left + plot_w + 6) / W, 1 - top / H),
                   fontsize=6, frameon=False, title=t("Unità", "Units"), title_fontsize=7)
    if path is None:
        from .surfaces import slug

        out = folder(chunk)
        out.mkdir(parents=True, exist_ok=True)
        base = out / (slug(holes[0].id) if single else panel_name([b.id for b in holes]))
        targets = [base.with_suffix(".svg"), base.with_suffix(".png")]
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        targets = [path] if path.suffix.lower() in (".svg", ".png", ".pdf") else [path.with_suffix(".svg"),
                                                                                    path.with_suffix(".png")]
    return save(fig, targets, png_dpi=200)
