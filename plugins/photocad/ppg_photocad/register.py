"""Where PhotoCAD's drawing lies on the image it was made from: registration.

PhotoCAD's page is not the image: it is the box of the SUBJECT it found (the real test on Khutm: a 2400 x 1198 px
image whose content filled rows 55-981, a page of 300 x 115.172 mm), and its coordinates are millimetres from the
lower-left corner of that page whatever ``units`` asks (``$INSUNITS`` says which). So the drawing is downloaded at its
natural size in mm and placed in two steps:

1. the page on the image's content box (the cells that are not background: white, as the renders paint what has no
   data), with one scale for both axes - the proportions must agree within 2 %;
2. the offset (+-24 px) and the scale (+-3 %) refined by bringing the drawn lines onto the edges of the image: the
   mean strength of the image's gradient under the points of the lines (divided by its maximum) is maximised,
   first on a smoothed gradient and a grid of 2 px, then on a sharper one down to a fortieth of a pixel (a gradient has its peak between the pixels of a step: no half-pixel bias, no ties).

Refused: an optimum on the border of the search (searched once more three times as far first); a score at the
optimum under 1.3 times the median score of the placements +-40 px around it (no peak: on a textured wall any
placement fits a little - Khutm 1.9, random lines 1.04); and, as before, a median distance from the Canny edges over
2 px or under 60 % of the points within 2 px (on masonry these alone refuse nothing: a diagnostic).
"""
from __future__ import annotations

import numpy as np

from .texts import t

ASPECT_TOLERANCE = 0.02
SHIFT = 24  # px searched around the content box (Khutm: 12.5 px; a page is never exactly the content box)
SCALE_RANGE = 0.03
TRUNCATE = 8.0
MAX_RESIDUAL = 2.0
PEAK_RATIO = 1.3  # the score at the optimum over the median of its surroundings (review of 2026-11-04)
PEAK_AROUND = 40  # px
MIN_NEAR = 0.6
WHITE = 250

_UNITS_MM = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 7: 1_000_000.0, 8: 0.0000254, 9: 0.0254,
             10: 914.4, 13: 0.001, 14: 100.0}


def unit_to_mm(code) -> float:
    """Millimetres in a drawing unit of ``$INSUNITS`` (unitless taken as mm, as PhotoCAD writes)."""
    return _UNITS_MM.get(int(code or 0), 1.0)


def read_mm(dxf_path) -> tuple[list[dict], float]:
    """The paths of a DXF in millimetres (:func:`drawing.paths`), and the factor applied."""
    import ezdxf

    from .drawing import paths

    factor = unit_to_mm(ezdxf.readfile(str(dxf_path)).header.get("$INSUNITS", 0))
    items = paths(dxf_path, tolerance=0.05 / factor)
    for it in items:
        it["xy"] = it["xy"] * factor
    return items, factor


def grey_of(image_bytes: bytes) -> np.ndarray:
    import cv2

    a = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError("the image cannot be read")
    return a


def content_box(grey: np.ndarray, white: int = WHITE, share: float = 0.002):
    """``(c0, r0, c1, r1)`` (pixel edges, end exclusive) of what is not white, ignoring rows and columns with
    fewer than ``share`` of their pixels content (JPEG specks); None for a blank image."""
    content = grey < white
    cols = np.nonzero(content.sum(axis=0) > max(1, share * grey.shape[0]))[0]
    rows = np.nonzero(content.sum(axis=1) > max(1, share * grey.shape[1]))[0]
    if not len(cols) or not len(rows):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def edge_distance(grey: np.ndarray) -> np.ndarray:
    """For every pixel the distance (px) to the nearest Canny edge."""
    import cv2

    edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 50, 150)
    if not edges.any():
        return np.full(grey.shape, TRUNCATE * 4, np.float32)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)


def _points(items: list[dict], step_mm: float, limit: int, seed: int = 7) -> np.ndarray:
    """Points along the lines of the drawing every ``step_mm`` (the closed ones closed)."""
    out = []
    for it in items:
        xy = it["xy"]
        if it.get("closed"):
            xy = np.vstack([xy, xy[:1]])
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / step_mm)))
            f = np.arange(n)[:, None] / n
            out.append(a + f * (b - a))
        out.append(xy[-1:])
    pts = np.vstack(out) if out else np.empty((0, 2))
    if len(pts) > limit:
        pts = pts[np.random.default_rng(seed).choice(len(pts), limit, replace=False)]
    return pts


def gradient(grey: np.ndarray, sigma: float) -> np.ndarray:
    """The gradient's strength of the image blurred by ``sigma``, 0-1 (divided by its maximum, not clipped: a clipped
    ridge has a flat top, and the search would slide along it)."""
    import cv2

    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    return (m / max(float(m.max()), 1e-6)).astype(np.float32)


def _sample(dist: np.ndarray, rows: np.ndarray, cols: np.ndarray, exact, outside: float = TRUNCATE):
    """``dist`` at pixel-edge coordinates (row, col): nearest (``exact`` False), bilinear (True) or cubic (``exact``
    = the cubic spline coefficients of ``dist``, :func:`scipy.ndimage.spline_filter`: a two-pixel peak keeps a
    peak, where bilinear would give it a flat top); ``outside`` beyond the image."""
    h, w = dist.shape
    r, c = rows - 0.5, cols - 0.5  # the pixel centres
    inside = (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
    out = np.full(r.shape, outside, np.float64)
    if isinstance(exact, np.ndarray):
        from scipy.ndimage import map_coordinates

        out[inside] = map_coordinates(exact, [r[inside], c[inside]], order=3, prefilter=False)
    elif exact:
        from scipy.ndimage import map_coordinates

        out[inside] = map_coordinates(dist, [r[inside], c[inside]], order=1)
    else:
        out[inside] = dist[np.rint(r[inside]).astype(int), np.rint(c[inside]).astype(int)]
    return out


def register(items_mm: list[dict], page_mm, image_bytes: bytes, search_px: float | None = None) -> dict:
    """The placement of the drawing (mm, page coordinates) on the image: ``{"ok", "problems", "scale" (px per mm),
    "dx", "dy", "box", "page_mm", "aspect_ratio", "residual_px", "near_share", "points"}``; image pixel-edge
    coordinates of a point (x, y) mm: ``col = cx + scale (x - pw/2) + dx``, ``row = cy - scale (y - ph/2) + dy``
    with ``(cx, cy)`` the centre of the content box. ``search_px``: the offset searched (default :data:`SHIFT`; widened
    3 times once when the optimum lies on its border)."""
    grey = grey_of(image_bytes)
    problems: list[str] = []
    out = {"ok": False, "problems": problems, "image_size": [int(grey.shape[1]), int(grey.shape[0])]}
    if not items_mm:
        problems.append(t("il disegno è vuoto", "the drawing is empty"))
        return out
    allp = np.vstack([it["xy"] for it in items_mm])
    pw, ph = (list(page_mm or []) + [None, None])[:2]
    if not (pw and ph and float(pw) > 0 and float(ph) > 0):
        pw, ph = float(allp[:, 0].max()), float(allp[:, 1].max())
    pw, ph = float(pw), float(ph)
    out["page_mm"] = [pw, ph]
    box = content_box(grey)
    if box is None:
        problems.append(t("l'immagine non ha contenuto", "the image has no content"))
        return out
    c0, r0, c1, r1 = box
    bw, bh = c1 - c0, r1 - r0
    out["box"] = list(box)
    aspect = (bw / bh) / (pw / ph)
    out["aspect_ratio"] = float(aspect)
    if abs(aspect - 1) > ASPECT_TOLERANCE:
        problems.append(t("la pagina di PhotoCAD ({0:g} x {1:g} mm) non ha le proporzioni del contenuto "
                          "dell'immagine ({2} x {3} px): {4:+.1f} %",
                          "PhotoCAD's page ({0:g} x {1:g} mm) does not have the proportions of the image's content "
                          "({2} x {3} px): {4:+.1f} %").format(pw, ph, bw, bh, (aspect - 1) * 100))
        return out
    s0 = (bw / pw + bh / ph) / 2
    cx, cy = (c0 + c1) / 2, (r0 + r1) / 2
    from scipy.ndimage import spline_filter

    smooth, sharp = gradient(grey, 2.5), gradient(grey, 1.0)
    spline = spline_filter(sharp, order=3)
    coarse = _points(items_mm, 1.0 / s0, 4000)
    fine = _points(items_mm, 0.5 / s0, 20000)

    shift0 = float(search_px or SHIFT)

    def search(shift, srange, step):
        best = (-np.inf, s0, 0.0, 0.0)

        def better(v, s, dx, dy):  # ties (flat tops) go to the smaller move from the content box
            if v > best[0] + 1e-9:
                return True
            return v > best[0] - 1e-9 and abs(s / s0 - 1) * 100 + abs(dx) + abs(dy) < \
                abs(best[1] / s0 - 1) * 100 + abs(best[2]) + abs(best[3])

        shifts = np.arange(-shift, shift + 1e-9, step)
        for s in s0 * (1 + np.arange(-srange, srange + 1e-9, 0.005)):
            cols0 = cx + s * (coarse[:, 0] - pw / 2)
            rows0 = cy - s * (coarse[:, 1] - ph / 2)
            for dy in shifts:
                for dx in shifts:
                    v = float(_sample(smooth, rows0 + dy, cols0 + dx, True, 0.0).mean())
                    if better(v, s, dx, dy):
                        best = (v, s, dx, dy)
        for step_s, span, fstep in ((0.001, 0.008, 0.5), (0.0002, 0.001, 0.1), (0.00005, 0.0002, 0.025)):
            best = (-np.inf, *best[1:])  # another measure (the sharp gradient): compared among itself
            _v, s1, dx1, dy1 = best
            for s in s1 * (1 + np.arange(-span, span + 1e-9, step_s)):
                cols0 = cx + s * (fine[:, 0] - pw / 2)
                rows0 = cy - s * (fine[:, 1] - ph / 2)
                for dy in dy1 + np.arange(-4 * fstep, 4 * fstep + 1e-9, fstep):
                    for dx in dx1 + np.arange(-4 * fstep, 4 * fstep + 1e-9, fstep):
                        v = float(_sample(sharp, rows0 + dy, cols0 + dx, spline, 0.0).mean())
                        if better(v, s, dx, dy):
                            best = (v, s, dx, dy)
        return best

    def on_border(best, shift, srange):
        _v, s, dx, dy = best
        return max(abs(dx), abs(dy)) >= shift - 1.0 or abs(s / s0 - 1) >= srange - 0.002

    best = search(shift0, SCALE_RANGE, 2.0)
    widened = on_border(best, shift0, SCALE_RANGE)
    at_border = False
    if widened:  # the optimum at the edge of the search: once more, three times as far and twice the scale range
        best = search(3 * shift0, 2 * SCALE_RANGE, 4.0)
        at_border = on_border(best, 3 * shift0, 2 * SCALE_RANGE)
    _v, s, dx, dy = best
    dist = edge_distance(grey)
    d = _sample(dist, cy - s * (fine[:, 1] - ph / 2) + dy, cx + s * (fine[:, 0] - pw / 2) + dx, True)
    residual, near = float(np.median(d)), float((d <= 2.0).mean())
    peak2d, peak_x, peak_y, around = _peak(sharp, coarse, s, dx, dy, cx, cy, pw, ph)
    peak = min(peak2d, peak_x, peak_y)
    out.update(scale=float(s), dx=float(dx), dy=float(dy), centre=[float(cx), float(cy)], residual_px=residual,
               near_share=near, points=int(len(fine)), scale_change=float(s / s0 - 1), peak_ratio=peak,
               peak_ratio_2d=peak2d, peak_ratio_x=peak_x, peak_ratio_y=peak_y, peak_around=around, widened=bool(widened), at_border=bool(at_border))
    if at_border:
        problems.append(t("il miglior accordo è al bordo della ricerca (spostamento {0:+.1f}, {1:+.1f} px, scala "
                          "{2:+.1f} %): il disegno è più lontano di quanto si cerca", "the best match lies on the border "
                          "of the search (offset {0:+.1f}, {1:+.1f} px, scale {2:+.1f} %): the drawing is further "
                          "than the search reaches").format(dx, dy, (s / s0 - 1) * 100))
    if peak < PEAK_RATIO:
        problems.append(t("nessun accordo netto: il punteggio al punto trovato è {0:.2f} volte la mediana dei dintorni "
                          "(+-{1} px), ne serve almeno {2}", "no clear match: the score at the place found is {0:.2f} "
                          "times the median around it (+-{1} px), at least {2} is needed").format(
            peak, PEAK_AROUND, PEAK_RATIO))
    if residual > MAX_RESIDUAL or near < MIN_NEAR:
        problems.append(t("il disegno non combacia con i bordi dell'immagine (scarto mediano {0:.1f} px, {1:.0f} % "
                          "dei punti entro 2 px)", "the drawing does not match the edges of the image (median "
                          "residual {0:.1f} px, {1:.0f} % of the points within 2 px)").format(residual, near * 100))
    if problems:
        return out
    out["ok"] = True
    return out


def _peak(score, pts, s, dx, dy, cx, cy, pw, ph) -> tuple[float, float, float, float]:
    """The score (mean gradient under the lines) at the optimum over the median score of the placements around it,
    ``(ratio, ratio along x, ratio along y, median around)``: around = +-:data:`PEAK_AROUND` px every 4 px (the
    optimum's own 5 x 5 px left out); along x/y = the same row/column only. A real match is a peak in both
    directions; a texture that fits anywhere is a plain (random lines over a wall: about 1.0), and coursed masonry
    shifted along its courses still fits the bed joints - a ridge, high across it but flat along it."""
    cols0 = cx + s * (pts[:, 0] - pw / 2)
    rows0 = cy - s * (pts[:, 1] - ph / 2)

    def at(ox, oy):
        return float(_sample(score, rows0 + dy + oy, cols0 + dx + ox, True, 0.0).mean())

    centre = at(0, 0)
    steps = [o for o in range(-PEAK_AROUND, PEAK_AROUND + 1, 4)]
    grid, along_x, along_y = [], [], []
    for oy in steps:
        for ox in steps:
            if abs(ox) <= 2 and abs(oy) <= 2:
                continue
            v = at(ox, oy)
            grid.append(v)
            if oy == 0:
                along_x.append(v)
            if ox == 0:
                along_y.append(v)

    def ratio(values):
        m = float(np.median(values))
        if m <= 1e-12:
            return 999.0 if centre > 1e-12 else 0.0
        return min(999.0, centre / m)

    return ratio(grid), ratio(along_x), ratio(along_y), float(np.median(grid))


def similarity(reg: dict, image: dict) -> tuple[float, float, float]:
    """``(m, tx, ty)``: a point (x, y) mm of the drawing is at ``(tx + m x, ty + m y)`` in the frame of the image
    (coordinates of the chunk, or (u, v) of the elevation)."""
    p, h = float(image["pixel"]), float(image["height"])
    s, (cx, cy), (pw, ph) = reg["scale"], reg["centre"], reg["page_mm"]
    tx = image["x0"] + p * (cx + reg["dx"] - s * pw / 2)
    ty = image["y0"] + p * (h - (cy + reg["dy"]) - s * ph / 2)
    return p * s, tx, ty
