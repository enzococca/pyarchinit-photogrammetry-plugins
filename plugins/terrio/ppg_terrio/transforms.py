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

Why a grid was not used terrio says in warnings (from 0.9: which grid, and that it is outside the
folders of grids, named by a path, too large, not downloaded), its error being often no more than
"cannot transform". The warnings issued during a call that may refuse a grid are recorded
(:func:`listen`): what they say of the grid goes into the refusal (:func:`words`), every other one is
issued again as it was (:func:`again`). Nothing is remembered from one call to the next: terrio
says it again every time it refuses.

From terrio 0.9 ``errcheck=False`` gives the points that cannot be moved back at infinity instead of
refusing them all (:func:`has_errcheck`). :func:`transform_points` asks for it only with ``partial``,
for a caller that leaves those points out and counts them (a cloud across the edge of a grid): any
other one needs an answer for every point and keeps the refusal. No point moved is no result: the
refusal of always, then.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np

from .texts import t

# beyond this the result of an older terrio is not the transformation PROJ uses: a ballpark one,
# which ignores the datum shift (tens to hundreds of metres); different grids of the same shift
# (a Helmert transformation against a grid) stay well within it
PYPROJ_TOLERANCE_M = 5.0
REASON_LENGTH = 300  # characters of terrio's own words kept in a message, at most


class TransformUnavailable(RuntimeError):
    """The best transformation between two systems cannot be used here: its grid is missing or
    unreadable (terrio would take a rougher one, possibly a ballpark). ``reason``: what terrio said
    of it in its warnings (which grid, and why it was not read), "" when it said nothing."""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


def has_only_best(terrio) -> bool:
    """terrio offers ``only_best`` (0.6 and later)."""
    try:
        return "only_best" in inspect.signature(terrio.transform_points).parameters
    except (TypeError, ValueError):  # a compiled function without a signature
        return tuple(int(v) for v in str(terrio.__version__).split(".")[:2] if v.isdigit()) >= (0, 6)


def has_errcheck(terrio) -> bool:
    """terrio offers ``errcheck`` (0.9 and later): with ``errcheck=False`` the points that cannot be
    moved come back at infinity. Told by the keyword alone: a signature that cannot be read is one
    without it (the points are then refused all together, as before)."""
    try:
        return "errcheck" in inspect.signature(terrio.transform_points).parameters
    except (TypeError, ValueError):
        return False


def finite(out) -> np.ndarray:
    """Which points of ``out`` (the arrays of coordinates terrio gave) have numbers in every one."""
    return np.logical_and.reduce([np.isfinite(np.asarray(v, np.float64)).ravel() for v in out])


def listen(call):
    """``call()`` with the warnings issued meanwhile recorded instead of shown: its result (``None``
    when it failed), its error (``None`` when it went well) and the warnings, for :func:`words` and
    :func:`again`. The window is the call and no more: the recording is of the whole process, so a
    warning of another thread in that time is recorded with terrio's (and issued again, or - when
    it is a ``UserWarning`` and a grid was refused - quoted beside terrio's). ``catch_warnings`` is not
    safe between threads: two such windows that overlap can put the filters of the process back wrong."""
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter("always")
        try:
            return call(), None, list(heard)
        except Exception as exc:
            return None, exc, list(heard)


def again(said) -> None:
    """Issue the recorded warnings ``said`` again, each with its own category, file and line: what
    another part of the program would have seen is not swallowed."""
    for w in said:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)


def words(said, but: str = "") -> tuple[str, list]:
    """What terrio said in the recorded warnings ``said``, as one line of :data:`REASON_LENGTH`
    characters at most: the text of each ``UserWarning`` (terrio's are) once, one after the other,
    without what ``but`` - the text of the error - says already. And the warnings that are not in
    the line (of another category, or left without room), for :func:`again`."""
    line, told, rest = "", set(), []
    for w in said:
        text = " ".join(str(w.message).split())
        if not text or not issubclass(w.category, UserWarning):
            rest.append(w)
        elif text in told or text in but:
            continue
        elif not line:
            line = text if len(text) <= REASON_LENGTH else text[:REASON_LENGTH - 3] + "..."
            told.add(text)
        elif len(line) + 2 + len(text) <= REASON_LENGTH:
            line += "; " + text
            told.add(text)
        else:
            rest.append(w)
    return line, rest


def _pass_on(heard, said) -> None:
    """The warnings of a call that went well: to whoever asked for them (the list ``said``), else
    issued again."""
    if said is None:
        again(heard)
    else:
        said.extend(heard)


def _refusal(src: str, dst: str, what: str, error, reason: str = "") -> TransformUnavailable:
    return TransformUnavailable(t(
        "{0}: la trasformazione migliore da {1} a {2} non si può usare (la sua griglia manca o non si legge) e terrio "
        "ne userebbe una più approssimativa, anche di decine di metri. Attiva la rete per scaricare la griglia dal "
        "CDN di PROJ, o mettila in una cartella delle griglie. ({3})",
        "{0}: the best transformation from {1} to {2} cannot be used (its grid is missing or unreadable) and terrio "
        "would take a rougher one, even tens of metres off. Allow the network to download the grid from the PROJ "
        "CDN, or put it in a folder of grids. ({3})").format(what or t("Trasformazione", "Transformation"),
                                                             src, dst, f"{error}; {reason}" if reason else error),
        reason)


def _refused(error, warping: bool = False) -> bool:
    """terrio refused the rougher operations (with ``only_best``): a warp says only that its points
    failed to transform."""
    text = str(error)
    keys = ("ONLY_BEST", "ALLOW_BALLPARK", "missing grid") + (("failed to transform",) if warping else ())
    return any(k in text for k in keys)


def _settle(out, error, heard, src: str, dst: str, what: str, warping: bool = False, said=None):
    """What :func:`listen` heard of a call of terrio asked for the best operation: its result, or its
    refusal as a :class:`TransformUnavailable` with terrio's own words about the grid. A refusal is an error
    that says so (:func:`_refused`) or one that came with a warning about a grid: terrio 0.9 says
    "cannot transform" of a point whose grid it did not read, and why in the warning. Any other
    error is terrio's own, raised as it is, and so is always a refusal for memory (terrio 0.9's
    ``ResourceLimitError``). The warnings no message uses are issued again (given to ``said``, a
    list, when the call went well and the caller asked for them)."""
    if error is None:
        _pass_on(heard, said)
        return out
    reason, rest = words(heard, but=str(error))
    if not isinstance(error, MemoryError) and (_refused(error, warping) or "grid" in reason):
        again(rest)
        raise _refusal(src, dst, what, error, reason) from error
    again(heard)
    raise error


def _strict(call, src: str, dst: str, what: str, warping: bool = False, said=None):
    """The result of ``call()`` - terrio asked for the best operation - or its refusal (:func:`_settle`)."""
    return _settle(*listen(call), src, dst, what, warping, said)


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


def check_against_pyproj(xs, ys, out, src: str, dst: str, what: str = "", said=()) -> None:
    """The first finite point of ``out`` (the x, y terrio gave for ``xs``, ``ys``) within
    :data:`PYPROJ_TOLERANCE_M` of the program's PROJ; else :class:`TransformUnavailable`, with what
    terrio warned of meanwhile (``said``, recorded by :func:`listen`)."""
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
        reason, rest = words(said)
        again(rest)
        raise _refusal(src, dst, what, t("{0} m dalla trasformazione di PROJ", "{0} m from PROJ's transformation")
                       .format(f"{apart:.1f}"), reason)


def transform_points(terrio, xs, ys, src: str, dst: str, zs=None, what: str = "", said=None, partial: bool = False):
    """``terrio.transform_points(xs, ys, src, dst, zs)`` by the best operation: ``only_best`` with
    terrio 0.6 and later, the check against pyproj before; :class:`TransformUnavailable` else, with
    terrio's own words about the grid. ``said``: a list that takes the warnings of a transformation
    that went well, for a caller that may still refuse its result (they are issued again else).

    One point that cannot be moved (outside a grid, outside what a projection takes) refuses them
    all. ``partial``, with a terrio that has ``errcheck``: such points come back at infinity in every
    coordinate (:func:`finite` tells them) and the others moved, for a caller that leaves them out
    and says how many; when no point at all was moved, the refusal of always. With a terrio without
    the keyword ``partial`` changes nothing."""
    kw = {} if zs is None else {"zs": zs}
    if has_only_best(terrio):
        def strict():
            return terrio.transform_points(xs, ys, src, dst, only_best=True, **kw)

        if not (partial and has_errcheck(terrio)):
            return _strict(strict, src, dst, what, said=said)
        out, error, heard = listen(lambda: terrio.transform_points(xs, ys, src, dst, only_best=True, errcheck=False,
                                                                   **kw))
        if error is None:
            moved = finite(out)
            if moved.size and not moved.any():  # no result: terrio asked as always says why, in its error
                told = {(w.category, str(w.message)) for w in heard}
                out, error, more = listen(strict)
                heard += [w for w in more if (w.category, str(w.message)) not in told]
        return _settle(out, error, heard, src, dst, what, said=said)
    out, error, heard = listen(lambda: terrio.transform_points(xs, ys, src, dst, **kw))
    if error is not None:
        again(heard)
        raise error
    check_against_pyproj(xs, ys, out, src, dst, what, said=heard)
    _pass_on(heard, said)
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
        return _strict(lambda: terrio.warp(source, t_srs=t_srs, to=options, **kw), src_crs or str(source), t_srs,
                       what, warping=True)
    if te is not None and src_crs:
        check_best(terrio, te, t_srs, src_crs, what)
    return terrio.warp(source, t_srs=t_srs, **kw)
