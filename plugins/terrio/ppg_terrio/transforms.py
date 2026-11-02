"""Coordinate transformations through terrio that never fall back to a rougher one in silence.

From terrio 0.6 its CRS engine (``terrio-crs``) reads no Zstandard-compressed TIFF grids in the
default build, and a transformation whose grid is missing or unreadable goes on with the next
operation - possibly a *ballpark* one, which ignores the datum shift (about 100 m between OSGB36 and
WGS 84) - with no more than a Python warning. terrio 0.6 refuses that with ``only_best`` (and
``ONLY_BEST=YES`` among the transformer options of ``warp``): the plugin always asks for it. An
older terrio has no such option; its result is checked against the program's PROJ (pyproj) on one
point instead.

* :func:`transform_points` - ``terrio.transform_points`` with the best operation or an error;
* :func:`warp` - ``terrio.warp`` with the best operation or an error;
* :func:`check_best` - the points of an area must go by the best operation (for steps where terrio
  reprojects by itself, such as the web map tiles).

The errors are :class:`TransformUnavailable`, which says what to do: allow the network (the grids
come from the PROJ CDN) or put the grid in a folder terrio searches.
"""
from __future__ import annotations

import inspect

import numpy as np

from .texts import t

# beyond this the result of an older terrio is not the transformation PROJ uses: a ballpark one,
# which ignores the datum shift (tens to hundreds of metres); different grids of the same shift
# (a Helmert transformation against a grid) stay well within it
PYPROJ_TOLERANCE_M = 5.0


class TransformUnavailable(RuntimeError):
    """The best transformation between two systems cannot be used here: its grid is missing or
    unreadable (terrio would take a rougher one, possibly a ballpark)."""


def has_only_best(terrio) -> bool:
    """terrio offers ``only_best`` (0.6 and later)."""
    try:
        return "only_best" in inspect.signature(terrio.transform_points).parameters
    except (TypeError, ValueError):  # a compiled function without a signature
        return tuple(int(v) for v in str(terrio.__version__).split(".")[:2] if v.isdigit()) >= (0, 6)


def _refusal(src: str, dst: str, what: str, error) -> TransformUnavailable:
    return TransformUnavailable(t(
        "{0}: la trasformazione migliore da {1} a {2} non si può usare (la sua griglia manca o non si legge) e terrio "
        "ne userebbe una più approssimativa, anche di decine di metri. Attiva la rete per scaricare la griglia dal "
        "CDN di PROJ, o mettila in una cartella delle griglie. ({3})",
        "{0}: the best transformation from {1} to {2} cannot be used (its grid is missing or unreadable) and terrio "
        "would take a rougher one, even tens of metres off. Allow the network to download the grid from the PROJ "
        "CDN, or put it in a folder of grids. ({3})").format(what or t("Trasformazione", "Transformation"),
                                                             src, dst, error))


def _refused(error, warping: bool = False) -> bool:
    """terrio refused the rougher operations (with ``only_best``): a warp says only that its points
    failed to transform."""
    text = str(error)
    keys = ("ONLY_BEST", "ALLOW_BALLPARK", "missing grid") + (("failed to transform",) if warping else ())
    return any(k in text for k in keys)


def _pyproj(xs, ys, src: str, dst: str, zs=None):
    from pyproj import CRS, Transformer

    try:
        from pyarchinit_photogrammetry.core.crs import to_pyproj
    except ImportError:  # an older program
        to_pyproj = CRS.from_user_input
    tr = Transformer.from_crs(to_pyproj(src), to_pyproj(dst), always_xy=True)
    if zs is None:
        return tr.transform(xs, ys)
    return tr.transform(xs, ys, zs)


def _metres_apart(a, b, crs: str) -> float:
    """The distance between the points ``a`` and ``b`` (x, y) of ``crs``, in metres."""
    from pyproj import CRS, Geod

    c = CRS.from_user_input(crs)
    if c.is_compound:
        c = c.sub_crs_list[0]
    if c.is_geographic:
        return float(abs(Geod(ellps="WGS84").inv(a[0], a[1], b[0], b[1])[2]))
    factor = c.axis_info[0].unit_conversion_factor if c.axis_info else 1.0
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) * factor)


def check_against_pyproj(xs, ys, out, src: str, dst: str, what: str = "") -> None:
    """The first finite point of ``out`` (the x, y terrio gave for ``xs``, ``ys``) within
    :data:`PYPROJ_TOLERANCE_M` of the program's PROJ; else :class:`TransformUnavailable`."""
    xs, ys = np.asarray(xs, np.float64).ravel(), np.asarray(ys, np.float64).ravel()
    ox, oy = np.asarray(out[0], np.float64).ravel(), np.asarray(out[1], np.float64).ravel()
    finite = np.nonzero(np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ox) & np.isfinite(oy))[0]
    if not len(finite):
        return
    i = int(finite[0])
    try:
        px, py = _pyproj(xs[i], ys[i], src, dst)
    except Exception:  # PROJ cannot make it either: nothing to compare with
        return
    if not (np.isfinite(px) and np.isfinite(py)):
        return
    apart = _metres_apart((ox[i], oy[i]), (px, py), dst)
    if apart > PYPROJ_TOLERANCE_M:
        raise _refusal(src, dst, what, t("{0} m dalla trasformazione di PROJ", "{0} m from PROJ's transformation")
                       .format(f"{apart:.1f}"))


def transform_points(terrio, xs, ys, src: str, dst: str, zs=None, what: str = ""):
    """``terrio.transform_points(xs, ys, src, dst, zs)`` by the best operation: ``only_best`` with
    terrio 0.6 and later, the check against pyproj before; :class:`TransformUnavailable` else."""
    kw = {} if zs is None else {"zs": zs}
    if has_only_best(terrio):
        try:
            return terrio.transform_points(xs, ys, src, dst, only_best=True, **kw)
        except Exception as exc:
            if _refused(exc):
                raise _refusal(src, dst, what, exc) from exc
            raise
    out = terrio.transform_points(xs, ys, src, dst, **kw)
    check_against_pyproj(xs, ys, out, src, dst, what)
    return out


def check_best(terrio, bounds, src: str, dst: str, what: str = "") -> None:
    """The corners and the centre of ``bounds`` (``min_x, min_y, max_x, max_y`` in ``src``) go to
    ``dst`` by the best operation, else :class:`TransformUnavailable`."""
    x0, y0, x1, y1 = (float(v) for v in bounds)
    xs = [x0, x1, x0, x1, (x0 + x1) / 2]
    ys = [y0, y0, y1, y1, (y0 + y1) / 2]
    transform_points(terrio, xs, ys, src, dst, what=what)


def warp(terrio, source, *, t_srs: str, te=None, src_crs: str | None = None, what: str = "", **kw):
    """``terrio.warp(source, t_srs=..., te=..., ...)`` by the best operation: ``ONLY_BEST=YES``
    among the transformer options with terrio 0.6 and later; with an older terrio the corners of
    ``te`` checked against pyproj when ``src_crs`` (the source's system) is known."""
    if te is not None:
        kw["te"] = te
    if has_only_best(terrio):
        options = dict(kw.pop("to", None) or {})
        options["ONLY_BEST"] = "YES"
        if te is not None and src_crs:
            check_best(terrio, te, t_srs, src_crs, what)  # the clear message, before terrio's own
        try:
            return terrio.warp(source, t_srs=t_srs, to=options, **kw)
        except Exception as exc:
            if _refused(exc, warping=True):
                raise _refusal(src_crs or str(source), t_srs, what, exc) from exc
            raise
    if te is not None and src_crs:
        check_best(terrio, te, t_srs, src_crs, what)
    return terrio.warp(source, t_srs=t_srs, **kw)
