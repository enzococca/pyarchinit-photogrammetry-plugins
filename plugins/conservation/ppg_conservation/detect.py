"""Proposals of the decay mapping: what the image and the relief of an elevation suggest, never more than a proposal.

Every detector reads a window of the elevation (``bounds`` = ``(u0, v0, u1, v1)`` in its metres, None = the whole face)
and returns a list of proposals ``{"kind", "code" | None, "geometry" (shapely, in (u, v)), "width_mm", "depth_mm",
"notes", "coverage"}``: ``notes`` say which detector and with which parameters, ``coverage`` the share of the window
where the relief has a value. A window with no surface at all (no relief and no opaque pixel) gives ``[]``.

- :func:`sam_areas`: segments of SAM 2.1 (the program's ``ai.sam.auto_units`` on ``elevation.tif``), with no term;
  :func:`like_this` finds the segments that look like one the user named.
- :func:`cracks`: valleys of the relief (Hessian of the smoothed depth at a few scales, a Frangi-like line measure,
  both sides higher) joined with the dark thin lines of the orthophoto (black-hat on L, near Canny edges), thinned,
  followed into polylines; joints between drawn stones are not cracks.
- :func:`material_loss`: local depressions of the relief under a smooth reference of the wall around them.
- :func:`colour_classes`: k-means on Lab of the orthophoto; the user names the classes.

The relief is ``d`` in metres, positive towards the viewer, NaN where the model is missing: a groove or a hole is a
negative ``d``. Only cracks (``ISCS.CRACK``) and material loss (``ISCS.MISSING_PART``, ``ISCS.ALVEOLIZATION``) come
with a term, said as "proposto"; the others are segments for the user to name."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CRACK = "ISCS.CRACK"
MISSING_PART = "ISCS.MISSING_PART"
ALVEOLIZATION = "ISCS.ALVEOLIZATION"
BIG_FACE = 4.0  # m2: SAM on a coarser copy and the "normale" thoroughness above it
BIG_FACE_RESOLUTION = 0.005  # m/px: the finest pixel SAM gets on a big face (memory of SAM on MPS)
JOINT_DISTANCE = 0.01  # m: a crack within this of a joint between two stones lies on the joint
JOINT_GAP = 0.05  # m: two stones nearer than this have a joint between them
JOINT_SHARE = 0.6  # a crack longer than this share on a joint is dropped
MAX_CRACK_WIDTH_MM = 12.0  # mm: a wider dark or hollow line is a joint between stones, not a crack
REFERENCE_SCALE = 1.0  # m: the scale of the local reference of the losses (its gaussian sigma is half of it)
WIDE_SHARE = 0.2  # a depression over this share of the window is no loss of its own: maybe the shape of the wall
PIXEL_BUDGET = 30_000_000  # pixels of the working resolution a detector reads at once; beyond: by tiles
TILE_OVERLAP = 0.2  # m: the overlap of two tiles
VALLEY_DEPTH = 0.0005  # m: the least depth of a valley of the relief (scale-normalised curvature and both sides)
DARK_CONTRAST = 12.0  # L units: the least black-hat of a dark thin line
SUGGESTIONS = {"green": ("ISCS.BIOLOGICAL_COLONIZATION",), "bright": ("ISCS.EFFLORESCENCE", "ISCS.DEPOSIT"),
               "dark": ("ISCS.CRUST.BLACK_CRUST", "ISCS.DISCOLOURATION.STAINING")}


def _t(it: str, en: str) -> str:
    from .texts import t

    return t(it, en)


def _num(value, decimals: int = 2) -> str:
    from .texts import num

    return num(value, decimals)


def _progress(progress):
    from pyarchinit_photogrammetry.core.progress import as_progress

    return as_progress(progress)


@dataclass
class Window:
    """A window of the elevation's two rasters on the same grid: ``rgb`` (H, W, 3) uint8, ``opaque`` (H, W) bool,
    ``relief`` (H, W) float64 with NaN, ``transform`` (rasterio Affine of the window), ``res`` (m/px), ``bounds``
    the window clipped to the rasters, ``ortho``/``relief_path`` the files."""
    rgb: np.ndarray
    opaque: np.ndarray
    relief: np.ndarray
    transform: object
    res: float
    bounds: tuple
    ortho: str
    relief_path: str

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.relief).mean()) if self.relief.size else 0.0

    @property
    def empty(self) -> bool:
        return not (self.opaque.any() or np.isfinite(self.relief).any())

    def uv(self, rows, cols):
        """``(u, v)`` of the centres of the cells ``(rows, cols)``."""
        t = self.transform
        return t.c + (np.asarray(cols, float) + 0.5) * t.a, t.f + (np.asarray(rows, float) + 0.5) * t.e

    def rc(self, u, v):
        """Fractional ``(row, col)`` of the points ``(u, v)`` (cell centres at whole numbers)."""
        t = self.transform
        return (np.asarray(v, float) - t.f) / t.e - 0.5, (np.asarray(u, float) - t.c) / t.a - 0.5


def read_window(chunk, entry, bounds=None) -> Window | None:
    """The window ``bounds`` (None: the whole face) of the elevation's orthophoto and relief, or None when it lies
    outside the rasters."""
    import rasterio
    import rasterio.errors
    from rasterio.windows import Window as RWindow
    from rasterio.windows import from_bounds

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    ortho, relief_path = (str(p) for p in elevation_paths(chunk, entry))
    with rasterio.open(ortho) as src:
        full = src.bounds
        bx = (full.left, full.bottom, full.right, full.top) if bounds is None else (
            max(bounds[0], full.left), max(bounds[1], full.bottom), min(bounds[2], full.right),
            min(bounds[3], full.top))
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            return None
        w = from_bounds(*bx, transform=src.transform).round_offsets().round_lengths()
        w = w.intersection(RWindow(0, 0, src.width, src.height))
        if w.width < 1 or w.height < 1:
            return None
        data = src.read(window=w)
        transform = src.window_transform(w)
        res = abs(src.res[0])
    rgb = np.moveaxis(data[:3], 0, -1).copy() if data.shape[0] >= 3 else np.repeat(data[:1], 3, 0).transpose(1, 2, 0)
    opaque = data[3] > 0 if data.shape[0] >= 4 else np.ones(data.shape[1:], bool)
    relief = np.full(opaque.shape, np.nan)
    try:
        with rasterio.open(relief_path) as src:
            rw = from_bounds(*bx, transform=src.transform).round_offsets().round_lengths()
            rw = rw.intersection(RWindow(0, 0, src.width, src.height))
            d = src.read(1, window=rw, masked=True).astype(np.float64).filled(np.nan)
        if d.shape != opaque.shape:
            import cv2

            d = cv2.resize(d, (opaque.shape[1], opaque.shape[0]), interpolation=cv2.INTER_NEAREST)
        relief = d
    except (FileNotFoundError, rasterio.errors.RasterioIOError):  # no relief: the image detectors still work
        pass
    out = (transform.c, transform.f + transform.e * opaque.shape[0], transform.c + transform.a * opaque.shape[1],
           transform.f)
    return Window(rgb, opaque, relief, transform, float(res), out, ortho, relief_path)


def lab(rgb: np.ndarray) -> np.ndarray:
    """Lab of ``rgb`` uint8 (L 0-100, a and b signed) as float32."""
    import cv2

    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _polygons(mask: np.ndarray, win: Window, min_area: float = 0.0) -> list:
    """The polygons of ``mask`` in (u, v) (``rasterio.features.shapes``), those of at least ``min_area`` m2."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    out = []
    if not mask.any():
        return out
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=win.transform, connectivity=8):
        if value:
            g = shape(geom).buffer(0)
            if not g.is_empty and g.area >= min_area:
                out.append(g)
    return out


def _proposal(kind, code, geometry, notes, coverage, width_mm=None, depth_mm=None) -> dict:
    return {"kind": kind, "code": code, "geometry": geometry, "width_mm": width_mm, "depth_mm": depth_mm,
            "notes": notes, "coverage": round(float(coverage), 4)}


# ---------------------------------------------------------------- SAM


SAM_TILE, SAM_OVERLAP, SAM_MAX_TILES = 1024, 128, 400


def sam_pixel(win: Window, resolution: float | None = None) -> float:
    """The pixel (m) SAM gets on the window: ``resolution`` when given (never finer than the raster); else on a
    window over ``BIG_FACE`` m2 ``BIG_FACE_RESOLUTION`` at the finest; else the finest that keeps at most
    ``SAM_MAX_TILES`` tiles, as the program's ``auto_units`` chooses it."""
    bx = win.bounds
    if resolution is None:
        area = (bx[2] - bx[0]) * (bx[3] - bx[1])
        if area > BIG_FACE:
            resolution = BIG_FACE_RESOLUTION
        else:
            side = max(bx[2] - bx[0], bx[3] - bx[1])
            per_axis = max(1, int(np.floor(np.sqrt(SAM_MAX_TILES))))
            resolution = side / (per_axis * (SAM_TILE - SAM_OVERLAP) + SAM_OVERLAP)
    return max(float(resolution), win.res)


def sam_areas(chunk, entry, bounds=None, thoroughness: str = "normale", resolution: float | None = None,
              progress=None, segmenter=None) -> list[dict]:
    """The segments of SAM 2.1 in the window, with no term (``code`` None), at the pixel of :func:`sam_pixel` (the
    notes say which). ``segmenter``: the
    program's SAM worker (``ai.sam_env.make_segmenter()`` when None)."""
    from pyarchinit_photogrammetry.ai.sam import auto_units

    progress = _progress(progress)
    win = read_window(chunk, entry, bounds)
    if win is None or win.empty:
        return []
    bx = win.bounds
    resolution = sam_pixel(win, resolution)
    if segmenter is None:
        from pyarchinit_photogrammetry.ai.sam_env import make_segmenter

        segmenter = make_segmenter()
    units = auto_units(None, segmenter, path=win.ortho, bounds=bx, resolution=resolution, thoroughness=thoroughness,
                       tile=SAM_TILE, overlap=SAM_OVERLAP, exclusive=True, max_tiles=SAM_MAX_TILES,
                       progress=progress.sub(0, 0.95))
    res = resolution
    notes = _t(f"proposto da SAM 2.1: accuratezza {thoroughness}, pixel {_num(res * 1000, 1)} mm",
               f"proposed by SAM 2.1: thoroughness {thoroughness}, pixel {_num(res * 1000, 1)} mm")
    out = [_proposal("decay", None, u.geometry, notes, win.coverage) for u in units
           if u.geometry is not None and not u.geometry.is_empty]
    progress(1.0, _t(f"SAM: {len(out)} segmenti", f"SAM: {len(out)} segments"))
    return out


def _feature_images(rgb: np.ndarray) -> np.ndarray:
    """The per-pixel quantities of the descriptor, computed once for a window: ``(H, W, 5)`` float64 = L, a, b,
    squared Sobel gradient of L, and the number of the 8 neighbours brighter than the pixel."""
    import cv2

    labimg = lab(rgb)
    L = labimg[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    pad = np.pad(L, 1, mode="edge")
    h, w = L.shape
    brighter = np.zeros(L.shape, np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                brighter += pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] > L
    return np.dstack([labimg.astype(np.float64), (gx * gx + gy * gy).astype(np.float64), brighter.astype(np.float64)])


def _descriptors(images: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """The descriptors of the regions ``1..n`` of the index image (0 = none) in one pass: ``(n, 15)`` = Lab mean and
    standard deviation, gradient energy of L, an 8-bin histogram of the neighbours brighter than each pixel
    (LBP-like); NaN rows for regions under 4 pixels."""
    flat = index.ravel()
    keep = flat > 0
    idx = flat[keep]
    vals = images.reshape(-1, images.shape[2])[keep]
    count = np.bincount(idx, minlength=n + 1)[1:].astype(np.float64)
    out = np.full((n, 15), np.nan)
    ok = count >= 4
    if not ok.any():
        return out
    c = np.where(ok, count, 1.0)
    for k in range(3):
        s = np.bincount(idx, weights=vals[:, k], minlength=n + 1)[1:]
        s2 = np.bincount(idx, weights=vals[:, k] ** 2, minlength=n + 1)[1:]
        mean = s / c
        out[:, k] = mean
        out[:, 3 + k] = np.sqrt(np.maximum(s2 / c - mean * mean, 0.0))
    out[:, 6] = np.sqrt(np.bincount(idx, weights=vals[:, 3], minlength=n + 1)[1:] / c)
    bins = np.clip(vals[:, 4].astype(np.int64) - 1, 0, 7)
    hist = np.bincount(idx * 8 + bins, minlength=(n + 1) * 8).reshape(n + 1, 8)[1:].astype(np.float64)
    out[:, 7:] = hist / c[:, None]
    out[~ok] = np.nan
    return out


# the least spread of each component of the descriptor: Lab mean, Lab spread, gradient energy, the 8 bins (a share)
SPREAD_FLOORS = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0] + [0.1] * 8)


def like_this(chunk, entry, segments: list, reference, radius: float = 1.0) -> list[int]:
    """The indices of ``segments`` (polygons in (u, v)) that look like ``reference``: descriptor (Lab mean and spread,
    Sobel gradient energy, an 8-bin LBP-like histogram, from the orthophoto inside each polygon) within ``radius``
    of the reference's, each component standardised by its spread over the segments (Euclidean distance), never by
    less than its ``SPREAD_FLOORS`` (a component that hardly changes between the segments - the histogram of a
    uniform texture - would else turn its noise into distance). The window is read and its per-pixel quantities
    computed once; the segments are rasterised once into an index image (where they overlap, the later one), the
    reference into its own mask."""
    from rasterio.features import rasterize
    from shapely.ops import unary_union

    from .geometry import as_geometry

    reference = as_geometry(reference)
    segments = [as_geometry(s) for s in segments]
    if not segments:
        return []
    u0, v0, u1, v1 = unary_union(segments + [reference]).bounds
    win = read_window(chunk, entry, (u0, v0, u1, v1))
    if win is None:
        return []
    images = _feature_images(win.rgb)
    shape = win.opaque.shape
    index = rasterize([(g, i + 1) for i, g in enumerate(segments) if not g.is_empty], out_shape=shape,
                      transform=win.transform, fill=0, dtype="int32")
    index[~win.opaque] = 0
    ref_index = rasterize([(reference, 1)], out_shape=shape, transform=win.transform, fill=0, dtype="int32")
    ref_index[~win.opaque] = 0
    ref = _descriptors(images, ref_index, 1)[0]
    if np.isnan(ref).any():
        return []
    table = _descriptors(images, index, len(segments))
    have = [i for i in range(len(segments)) if not np.isnan(table[i]).any()]
    if not have:
        return []
    table = table[have]
    spread = np.maximum(table.std(axis=0), SPREAD_FLOORS)
    dist = np.sqrt((((table - ref) / spread) ** 2).sum(axis=1))
    return [i for i, d in zip(have, dist) if d <= radius]


# ---------------------------------------------------------------- relief


def _smooth(d: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware gaussian: (smoothed, share of the kernel with values)."""
    from scipy import ndimage

    w = np.isfinite(d).astype(np.float64)
    num = ndimage.gaussian_filter(np.where(w > 0, d, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(w, sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan), den


def _valleys(d: np.ndarray, res: float, scales_mm) -> np.ndarray:
    """The valleys of the depth: for each scale, the Hessian of the relief smoothed at that scale; a pixel is a
    valley where the largest principal curvature is positive (a trough of ``d``) and dominates the other
    (Frangi-like), its scale-normalised depth is at least ``VALLEY_DEPTH`` and both sides across it, along the
    principal direction, stand higher than its centre by ``VALLEY_DEPTH``."""
    from scipy import ndimage

    out = np.zeros(d.shape, bool)
    if not np.isfinite(d).any():
        return out
    sigmas = sorted({round(max(0.7, s / 1000.0 / res), 2) for s in scales_mm})
    for sigma in sigmas:
        g, cover = _smooth(d, sigma)
        filled = np.where(np.isfinite(g), g, np.nanmedian(g))
        hyy = ndimage.gaussian_filter(filled, sigma * 0.5, order=(2, 0), mode="nearest")
        hxx = ndimage.gaussian_filter(filled, sigma * 0.5, order=(0, 2), mode="nearest")
        hxy = ndimage.gaussian_filter(filled, sigma * 0.5, order=(1, 1), mode="nearest")
        half = (hxx + hyy) / 2.0
        root = np.sqrt(((hxx - hyy) / 2.0) ** 2 + hxy ** 2)
        l1, l2 = half + root, half - root  # l1 the largest
        s2 = sigma * sigma
        ratio = np.abs(l2) / np.maximum(l1, 1e-12)
        strength = s2 * np.sqrt(l1 * l1 + l2 * l2)
        frangi = np.exp(-ratio ** 2 / 0.5) * (1.0 - np.exp(-(strength / VALLEY_DEPTH) ** 2 / 2.0))
        cand = (l1 > 0) & (s2 * l1 >= VALLEY_DEPTH) & (frangi > 0.4) & (cover > 0.9) & np.isfinite(d)
        if not cand.any():
            continue
        rows, cols = np.nonzero(cand)
        # the eigenvector of l1: across the valley
        ex = hxy[rows, cols]
        ey = l1[rows, cols] - hxx[rows, cols]
        small = np.hypot(ex, ey) < 1e-15
        ex = np.where(small, np.where(hxx[rows, cols] >= hyy[rows, cols], 1.0, 0.0), ex)
        ey = np.where(small, np.where(hxx[rows, cols] >= hyy[rows, cols], 0.0, 1.0), ey)
        n = np.hypot(ex, ey)
        ex, ey = ex / n, ey / n
        k = max(1.5, 2.0 * sigma)
        centre = filled[rows, cols]
        sides = []
        for s in (-k, k):
            sides.append(ndimage.map_coordinates(filled, [rows + s * ey, cols + s * ex], order=1, mode="nearest"))
        both = (sides[0] - centre >= VALLEY_DEPTH) & (sides[1] - centre >= VALLEY_DEPTH)
        out[rows[both], cols[both]] = True
    return out


def _dark_lines(L: np.ndarray, opaque: np.ndarray, res: float, scales_mm) -> np.ndarray:
    """The dark thin lines of the orthophoto: black-hat of L (a kernel wider than the largest scale) above
    ``DARK_CONTRAST`` and the noise, the components that touch Canny edges."""
    import cv2

    k = max(5, 2 * int(round(max(scales_mm) / 1000.0 / res)) + 3)
    k += 1 - k % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    Lf = np.where(opaque, L, np.median(L[opaque]) if opaque.any() else 0).astype(np.float32)
    bh = cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, kernel)
    vals = bh[opaque]
    if vals.size == 0:
        return np.zeros(L.shape, bool)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    dark = (bh > max(DARK_CONTRAST, med + 6 * mad)) & opaque
    if not dark.any():
        return dark
    edges = cv2.Canny(np.clip(Lf * 2.55, 0, 255).astype(np.uint8), 40, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
    n, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    touch = np.zeros(n, bool)
    touch[np.unique(labels[edges & dark])] = True
    touch[0] = False
    return touch[labels]


def _skeleton_lines(mask: np.ndarray, spur: float) -> list:
    """Polylines (in pixel coordinates (col, row)) of the thinned ``mask``: the skeleton's pixels joined to their
    4-neighbours, and diagonally when no 4-path joins them, merged into lines; end branches shorter than ``spur``
    pixels at a junction pruned."""
    import cv2
    import shapely
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge

    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    if not sk.any():
        return []
    p = np.pad(sk, 1)
    segs = []
    rows, cols = np.nonzero(sk)
    r, c = rows + 1, cols + 1
    for dr, dc in ((0, 1), (1, 0)):
        m = p[r + dr, c + dc]
        segs += [((x, y), (x + dc, y + dr)) for x, y in zip(cols[m], rows[m])]
    for dc in (1, -1):
        m = p[r + 1, c + dc] & ~p[r, c + dc] & ~p[r + 1, c]
        segs += [((x, y), (x + dc, y + 1)) for x, y in zip(cols[m], rows[m])]
    if not segs:
        return []
    lines = list(shapely.get_parts(linemerge(MultiLineString(segs))))
    for _ in range(3):
        ends = {}
        for ln in lines:
            for pt in (ln.coords[0], ln.coords[-1]):
                ends[pt] = ends.get(pt, 0) + 1
        kept = []
        pruned = False
        for ln in lines:
            a, b = ends[ln.coords[0]], ends[ln.coords[-1]]
            if ln.length < spur and min(a, b) == 1 and max(a, b) >= 3:
                pruned = True
                continue
            kept.append(ln)
        lines = kept
        if not pruned or not lines:
            break
        lines = list(shapely.get_parts(linemerge(MultiLineString([list(ln.coords) for ln in lines]))))
    return lines


def _joints(chunk, entry):
    """The joints between the stones drawn on the face (within ``JOINT_DISTANCE`` of the space between two stones
    nearer than ``JOINT_GAP``), or None when the face has no stones."""
    import shapely
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    stones = (frame_drawings(chunk, entry["id"]).get("stones") or {}).get(entry["id"]) or []
    polys = []
    for text in stones:
        try:
            g = shapely.from_wkt(text) if isinstance(text, str) else text
        except Exception:  # noqa: BLE001 - a broken record is no stone
            continue
        if g is not None and not g.is_empty:
            polys.append(g.buffer(0))
    if not polys:
        return None
    tree = STRtree(polys)
    zones = []
    for i, g in enumerate(polys):
        for j in tree.query(g.buffer(JOINT_GAP)):
            j = int(j)
            if j <= i:
                continue
            gap = g.distance(polys[j])
            if gap > JOINT_GAP:
                continue
            r = gap / 2.0 + JOINT_DISTANCE
            zone = g.buffer(r).intersection(polys[j].buffer(r))
            if not zone.is_empty:
                zones.append(zone)
    return unary_union(zones) if zones else unary_union([])


def _profile_width(d: np.ndarray, rows, cols, nx, ny, reach: float) -> np.ndarray:
    """The width (pixels) at half depth of the relief's profile across a line at ``(rows, cols)`` along the normal
    ``(nx, ny)`` (column, row), sampled every quarter pixel up to ``reach`` on each side; NaN where the profile
    has no value or no depth."""
    from scipy import ndimage

    t = np.arange(-reach, reach + 1e-9, 0.25)
    rr = rows[:, None] + t[None] * ny[:, None]
    cc = cols[:, None] + t[None] * nx[:, None]
    filled = np.where(np.isfinite(d), d, np.nan)
    prof = ndimage.map_coordinates(np.nan_to_num(filled, nan=0.0), [rr.ravel(), cc.ravel()], order=1,
                                   mode="nearest").reshape(rr.shape)
    have = ndimage.map_coordinates(np.isfinite(filled).astype(float), [rr.ravel(), cc.ravel()], order=1,
                                   mode="constant", cval=0.0).reshape(rr.shape)
    out = np.full(len(rows), np.nan)
    outer = np.abs(t) >= reach * 0.6
    mid = np.abs(t) <= 1.5
    for i in range(len(rows)):
        if (have[i] < 0.999).any():
            continue
        base = float(np.median(prof[i, outer]))
        j0 = np.nonzero(mid)[0]
        jmin = j0[np.argmin(prof[i, j0])]
        depth = base - prof[i, jmin]
        if depth < VALLEY_DEPTH:
            continue
        below = prof[i] < base - depth / 2.0
        lo = jmin
        while lo > 0 and below[lo - 1]:
            lo -= 1
        hi = jmin
        while hi < len(t) - 1 and below[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(t) - 1:
            continue
        # the crossings of the half level, interpolated between the samples
        level = base - depth / 2.0

        def cross(k0, k1):
            p0, p1 = prof[i, k0], prof[i, k1]
            f = (level - p0) / (p1 - p0) if p1 != p0 else 0.5
            return t[k0] + f * (t[k1] - t[k0])

        out[i] = cross(hi, hi + 1) - cross(lo - 1, lo)
    return out


def cracks(chunk, entry, bounds=None, min_length: float = 0.10, scales_mm=(2, 5, 10, 20),
           progress=None, max_width_mm: float | None = MAX_CRACK_WIDTH_MM, stats: dict | None = None) -> list[dict]:
    """Cracks proposed in the window (``code`` ``ISCS.CRACK``, said as "proposto"): the valleys of the relief at the
    scales ``scales_mm`` joined with the dark thin lines of the orthophoto, thinned, followed into polylines,
    simplified by a pixel, those shorter than ``min_length`` m dropped. ``width_mm``: the median along the line of
    the dark width across it (distance transform of the dark mask: ``2 d - 1`` pixels) and, where the relief has a
    value, of the width at half depth of its profile. A line lying for more than ``JOINT_SHARE`` of its length on a
    joint between two stones drawn on the face is dropped and counted; so is a line wider than ``max_width_mm``
    (None: no limit), which on a stone wall is a mortar joint, not a crack. ``stats`` (a dict, when given) gets
    ``on_joints``, ``wide`` (the lines dropped for each reason) and ``stones`` (whether the face has stones drawn)."""
    import cv2
    from shapely.geometry import LineString

    progress = _progress(progress)
    win = read_window(chunk, entry, bounds)
    if win is None or win.empty:
        return []
    res = win.res
    progress(0.05, _t("lesioni: valli del rilievo", "cracks: valleys of the relief"))
    valleys = _valleys(win.relief, res, scales_mm)
    progress(0.4, _t("lesioni: linee scure dell'immagine", "cracks: dark lines of the image"))
    L = lab(win.rgb)[..., 0]
    dark = _dark_lines(L, win.opaque, res, scales_mm)
    mask = valleys | dark
    progress(0.6, _t("lesioni: scheletro", "cracks: skeleton"))
    lines = _skeleton_lines(mask, spur=max(5.0, 0.3 * min_length / res))
    dist = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    joints = _joints(chunk, entry)
    reach = max(4.0, 2.5 * max(scales_mm) / 1000.0 / res)
    found, on_joints, wide = [], 0, 0
    for ln in lines:
        ln = ln.simplify(1.0)
        xy = np.asarray(ln.coords)
        u, v = win.uv(xy[:, 1], xy[:, 0])  # the skeleton's pixels (col, row): their cells' centres
        geom = LineString(np.column_stack([u, v]))
        if geom.length < min_length:
            continue
        if joints is not None and not joints.is_empty and geom.intersection(joints).length > JOINT_SHARE * geom.length:
            on_joints += 1
            continue
        # samples every pixel along the line
        n = max(2, int(np.ceil(ln.length)) + 1)
        pts = [ln.interpolate(s) for s in np.linspace(0, ln.length, n)]
        pc = np.array([[p.x, p.y] for p in pts])
        tang = np.gradient(pc, axis=0)
        tn = np.hypot(tang[:, 0], tang[:, 1])
        tn[tn == 0] = 1.0
        nx, ny = -tang[:, 1] / tn, tang[:, 0] / tn
        rows = np.clip(np.round(pc[:, 1]).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(np.round(pc[:, 0]).astype(int), 0, mask.shape[1] - 1)
        dark_w = dist[rows, cols]
        dark_w = (2.0 * dark_w[dark_w > 0] - 1.0) * res
        relief_w = _profile_width(win.relief, pc[:, 1], pc[:, 0], nx, ny, reach) * res
        relief_w = relief_w[np.isfinite(relief_w)]
        widths = np.concatenate([dark_w, relief_w])
        width = float(np.median(widths) * 1000.0) if widths.size else None
        if max_width_mm is not None and width is not None and width > max_width_mm:
            wide += 1  # a joint between stones, not a crack
            continue
        source = _t("rilievo e immagine", "relief and image") if relief_w.size and dark_w.size else (
            _t("solo dal rilievo", "relief only") if relief_w.size else _t("solo dall'immagine", "image only"))
        notes = _t(f"lesione proposta: valli del rilievo a {', '.join(str(s) for s in scales_mm)} mm e linee scure "
                   f"dell'immagine, lunghezza minima {_num(min_length * 100, 0)} cm; apertura {source}",
                   f"crack proposed: valleys of the relief at {', '.join(str(s) for s in scales_mm)} mm and dark "
                   f"lines of the image, least length {_num(min_length * 100, 0)} cm; opening {source}")
        found.append(_proposal("crack", CRACK, geom, notes, win.coverage, width_mm=width))
    if joints is None:
        extra = _t("le pietre del prospetto non sono disegnate: le proposte possono comprendere i giunti",
                   "the stones of the elevation are not drawn: the proposals may include joints")
    else:
        extra = _t(f"{on_joints} linee sui giunti tra le pietre tolte", f"{on_joints} lines on joints between the "
                   "stones dropped")
    if max_width_mm is not None:
        extra += "; " + _t(f"{wide} linee più larghe di {_num(max_width_mm, 0)} mm tolte (giunti?)",
                           f"{wide} lines wider than {_num(max_width_mm, 0)} mm dropped (joints?)")
    for f in found:
        f["notes"] += "; " + extra
    if stats is not None:
        stats.update(on_joints=on_joints, wide=wide, stones=joints is not None)
    progress(1.0, _t(f"lesioni: {len(found)} proposte; {extra}", f"cracks: {len(found)} proposals; {extra}"))
    return found


def surface_fit(u, v, d, order: int = 1) -> tuple[np.ndarray, float]:
    """The robust polynomial surface of the samples, ``order`` 1 (a plane ``d = a u + b v + c``) or 2 (with ``u²``,
    ``u v``, ``v²``): least squares, then twice again without those farther than 3 MAD from the median residual (at
    least 0.5 mm); ``(coefficients, MAD)``, the coefficients in the order of :func:`_terms`."""
    A = _terms(np.asarray(u, float), np.asarray(v, float), order)
    keep = np.ones(len(u), bool)
    coef, mad = np.zeros(A.shape[1]), 0.0
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        r = d - A @ coef
        centre = float(np.median(r[keep]))  # a big hole pulls the first surface: the bulk of the wall is the median
        mad = float(np.median(np.abs(r[keep] - centre))) * 1.4826
        keep = np.abs(r - centre) <= max(3.0 * mad, 0.0005)
        if keep.sum() < A.shape[1] + 1:
            break
    return coef, mad


def _terms(u, v, order: int) -> np.ndarray:
    cols = [u, v, np.ones_like(u)]
    if order >= 2:
        cols += [u * u, u * v, v * v]
    return np.column_stack(cols)


def plane_fit(u, v, d) -> tuple[np.ndarray, float]:
    """The robust plane ``d = a u + b v + c`` of the samples (:func:`surface_fit` of order 1)."""
    return surface_fit(u, v, d, 1)


def _local_reference(d: np.ndarray, res: float, sigma: float) -> np.ndarray:
    """A smooth reference of the relief ``d`` (NaN where it has no value): at each pixel the value of the
    gaussian-weighted local surface of the second order of the values around it (sigma ``sigma`` m), so that a slope,
    the curve of a bulge or the edge of the window bring no bias; fitted on a grid of blocks of about ``sigma / 8``
    (block means weighted by their counts), each pixel given the surface of its block at its own offset. The weighted
    moments about each block come from separable filters (the gaussian times the powers of the offset). NaN where
    the values within reach weigh less than 2 % of a full kernel."""
    from scipy import ndimage

    H, W = d.shape
    f = max(1, int(sigma / res / 8.0))
    h, w = -(-H // f), -(-W // f)
    pad = np.full((h * f, w * f), np.nan)
    pad[:H, :W] = d
    blocks = pad.reshape(h, f, w, f)
    ok = np.isfinite(blocks)
    count = ok.sum(axis=(1, 3)).astype(np.float64)
    total = np.where(ok, blocks, 0.0).sum(axis=(1, 3))
    z = np.where(count > 0, total / np.maximum(count, 1.0), 0.0)
    s = sigma / (res * f)
    radius = int(np.ceil(3.0 * s))
    off = np.arange(-radius, radius + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (off / s) ** 2)
    kern /= kern.sum()
    powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]  # the terms 1, x, y, x², x y, y² of the surface

    def moment(a, px, py):  # sum over the kernel of a * dx^px * dy^py (x along the columns)
        out = ndimage.correlate1d(a, kern * off ** px, axis=1, mode="constant", cval=0.0)
        return ndimage.correlate1d(out, kern * off ** py, axis=0, mode="constant", cval=0.0)

    cache = {}

    def m(px, py):
        if (px, py) not in cache:
            cache[(px, py)] = moment(count, px, py)
        return cache[(px, py)]

    A = np.empty((h, w, 6, 6))
    b = np.empty((h, w, 6))
    for i, (pi, qi) in enumerate(powers):
        for j, (pj, qj) in enumerate(powers):
            A[..., i, j] = m(pi + pj, qi + qj)
        b[..., i] = moment(count * z, pi, qi)
    S0 = A[..., 0, 0]
    scale = np.array([1.0, s, s, s * s, s * s, s * s])  # the unknowns in kernel units: a well-conditioned system
    An = A / scale[:, None] / scale[None, :]
    bn = b / scale
    An += np.eye(6) * (1e-9 * np.maximum(S0, 1e-12))[..., None, None]
    coef = np.full((h, w, 6), np.nan)
    good = S0 >= 0.02 * f * f
    if good.any():
        try:
            coef[good] = np.linalg.solve(An[good], bn[good][..., None])[..., 0] / scale
        except np.linalg.LinAlgError:  # a degenerate cell somewhere: the weighted mean there instead
            coef[good] = 0.0
            coef[good, 0] = b[good][:, 0] / S0[good]
    # each pixel: the surface of its block evaluated at its offset from the block's centre (in blocks)
    rows, cols = np.arange(H), np.arange(W)
    br, bc = rows // f, cols // f
    dy = (rows - (br * f + (f - 1) / 2.0)) / f
    dx = (cols - (bc * f + (f - 1) / 2.0)) / f
    c = coef[br[:, None], bc[None, :]]
    DX, DY = dx[None, :], dy[:, None]
    return c[..., 0] + c[..., 1] * DX + c[..., 2] * DY + c[..., 3] * DX * DX + c[..., 4] * DX * DY + \
        c[..., 5] * DY * DY


def loss_residual(win: Window, min_depth: float, scale: float = REFERENCE_SCALE) -> tuple[np.ndarray, str]:
    """The relief of the window against its reference (negative = behind it) and the words of the notes. A window
    whose shorter side is under ``2 scale`` has its robust plane as the reference (the wall around a hole within
    it). A larger one: a robust surface of the second order over the window (a batter, a broad bulge), then a local
    reference (:func:`_local_reference`, sigma ``scale / 2``) of what is left, made twice: the second time without
    the depressions the first found (deeper than half ``min_depth`` under it, grown by two pixels), so that a hole
    does not pull its own reference. Where the second reference has no value (inside a depression wider than its
    reach) the surface of the window is the reference."""
    from scipy import ndimage

    d = win.relief
    ok = np.isfinite(d)
    rows, cols = np.nonzero(ok)
    u, v = win.uv(rows, cols)
    pick = np.arange(len(u))
    if len(pick) > 200_000:
        pick = np.random.default_rng(0).choice(len(u), 200_000, replace=False)
    bx = win.bounds
    local = min(bx[2] - bx[0], bx[3] - bx[1]) >= 2.0 * scale
    order = 2 if local else 1
    uc, vc = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0  # centred: a well-conditioned second order
    coef, _mad = surface_fit(u[pick] - uc, v[pick] - vc, d[rows[pick], cols[pick]], order)
    r0 = np.full(d.shape, np.nan)
    r0[rows, cols] = d[rows, cols] - _terms(u - uc, v - vc, order) @ coef
    if not local:
        return r0, _t("piano robusto della finestra", "robust plane of the window")
    first = _local_reference(r0, win.res, scale / 2.0)
    cand = np.nan_to_num(r0 - first, nan=0.0) < -min_depth / 2.0
    cand = ndimage.binary_dilation(cand, iterations=2) & ok
    second = _local_reference(np.where(cand, np.nan, r0), win.res, scale / 2.0)
    resid = np.where(np.isfinite(second), r0 - second, r0)
    words = _t(f"superficie locale del muro senza le depressioni, scala {_num(scale, 2)} m",
               f"local surface of the wall without the depressions, scale {_num(scale, 2)} m")
    return resid, words


def material_loss(chunk, entry, bounds=None, min_depth: float = 0.02, min_area: float = 0.01,
                  progress=None, scale: float = REFERENCE_SCALE, stats: dict | None = None) -> list[dict]:
    """Losses of material proposed in the window: the relief against the wall's reference around it
    (:func:`loss_residual`: a robust surface over the window, then a local one at the scale ``scale`` m with the
    depressions left out), depressions deeper than ``min_depth`` m as connected components of at least ``min_area``
    m2, polygons with ``depth_mm`` = their median depth. ``ISCS.MISSING_PART`` over 0.05 m2; ``ISCS.ALVEOLIZATION``
    for the components under 0.02 m2 when the window holds at least five of them; the others with no term. A
    component larger than a quarter of ``scale²`` or than ``WIDE_SHARE`` of the window gets no term and the note
    "da verificare" (``check`` True): at that size the reference cannot tell a loss from the shape of the wall.
    ``stats`` (a dict, when given) gets ``check``, the number of those."""
    import cv2

    progress = _progress(progress)
    win = read_window(chunk, entry, bounds)
    if win is None or win.empty:
        return []
    ok = np.isfinite(win.relief)
    if ok.sum() < 7:
        return []
    progress(0.1, _t("mancanze: superficie di riferimento", "losses: reference surface"))
    resid, words = loss_residual(win, min_depth, scale)
    progress(0.5, _t("mancanze: depressioni", "losses: depressions"))
    deep = np.nan_to_num(resid, nan=0.0) < -min_depth
    n, labels, cc_stats, _c = cv2.connectedComponentsWithStats(deep.astype(np.uint8), connectivity=8)
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape

    cell = win.res * win.res
    big = np.zeros(n, bool)
    big[1:] = cc_stats[1:, cv2.CC_STAT_AREA] * cell >= min_area
    labels = np.where(big[labels], labels, 0).astype(np.int32)
    ks = np.nonzero(big)[0]
    depths = dict(zip(ks.tolist(), (np.asarray(ndimage.median(-np.nan_to_num(resid), labels, ks)) * 1000.0
                                    ).reshape(-1).tolist())) if len(ks) else {}
    comps = []  # one vectorisation of all the components
    for geom, value in shapes(labels, mask=labels > 0, transform=win.transform, connectivity=8):
        g = shape(geom).buffer(0)
        if not g.is_empty:
            comps.append((g, float(depths[int(value)])))
    too_big = min(scale * scale / 4.0, WIDE_SHARE * float(ok.sum()) * cell)
    small = [g.area < 0.02 for g, _ in comps]
    alveoli = sum(small) >= 5
    found = []
    for (g, depth), is_small in zip(comps, small):
        check = ""
        if g.area > too_big:
            code = None
            check = _t("; da verificare: può essere la forma del muro", "; to check: it may be the shape of the wall")
        elif g.area > 0.05:
            code = MISSING_PART
        elif alveoli and is_small:
            code = ALVEOLIZATION
        else:
            code = None
        notes = _t(f"mancanza proposta: più di {_num(min_depth * 100, 1)} cm sotto il muro intorno ({words}), "
                   f"area minima {_num(min_area, 3)} m²", f"loss proposed: more than {_num(min_depth * 100, 1)} cm "
                   f"behind the wall around it ({words}), least area {_num(min_area, 3)} m²") + check
        found.append({**_proposal("decay", code, g, notes, win.coverage, depth_mm=round(depth, 1)),
                      "check": bool(check)})
    if stats is not None:
        stats["check"] = sum(1 for f in found if f["check"])
    progress(1.0, _t(f"mancanze: {len(found)} proposte", f"losses: {len(found)} proposals"))
    return found


# ---------------------------------------------------------------- colours


def looks(lab_mean) -> str | None:
    """``green`` when a < -8, ``bright`` when L > 80 and chroma < 10, ``dark`` when L < 35, else None."""
    L, a, b = (float(x) for x in lab_mean)
    if a < -8:
        return "green"
    if L > 80 and math.hypot(a, b) < 10:
        return "bright"
    if L < 35:
        return "dark"
    return None


def colour_classes(chunk, entry, bounds=None, k: int = 5, progress=None, min_area: float = 0.005) -> list[dict]:
    """Classes of colour in the window: k-means (``cv2.kmeans``, 3 attempts) on Lab of the opaque pixels; each class
    cleaned (opening 3 px, components of at least ``min_area`` m2) and vectorised: ``{"class", "lab_mean", "share",
    "polygons", "looks", "suggest" (codes the dialog proposes), "notes", "coverage"}``, the largest first."""
    import cv2

    progress = _progress(progress)
    win = read_window(chunk, entry, bounds)
    if win is None or win.empty or not win.opaque.any():
        return []
    labimg = lab(win.rgb)
    data = labimg[win.opaque].astype(np.float32)
    k = int(max(1, min(k, len(data))))
    sample = data
    if len(data) > 200_000:
        sample = data[np.random.default_rng(0).choice(len(data), 200_000, replace=False)]
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    progress(0.1, _t("colori: k-means", "colours: k-means"))
    _c, _l, centres = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # nearest centre by ||c||^2 - 2 x.c (the ||x||^2 term is the same for every centre): N x k, never N x k x 3
    score = (centres * centres).sum(axis=1)[None, :] - 2.0 * (data @ centres.T)
    label = np.full(win.opaque.shape, -1, np.int32)
    label[win.opaque] = score.argmin(axis=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    total = float(win.opaque.sum())
    out = []
    for i in range(k):
        m = label == i
        share = float(m.sum()) / total
        clean = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        polys = _polygons(clean, win, min_area=min_area)
        mean = [round(float(x), 2) for x in centres[i]]
        what = looks(mean)
        notes = _t(f"classe di colore {i + 1} di {k} (k-means su Lab), pulita con apertura di 3 px, aree da "
                   f"{_num(min_area, 3)} m²", f"colour class {i + 1} of {k} (k-means on Lab), cleaned by a 3 px "
                   f"opening, areas from {_num(min_area, 3)} m²")
        out.append({"class": i, "lab_mean": mean, "share": round(share, 4), "polygons": polys, "looks": what,
                    "suggest": list(SUGGESTIONS.get(what, ())), "notes": notes, "coverage": round(win.coverage, 4)})
    out.sort(key=lambda c: -c["share"])
    progress(1.0, _t(f"colori: {k} classi", f"colours: {k} classes"))
    return out


# ---------------------------------------------------------------- big windows


def stones_drawn(chunk, entry) -> bool:
    """Whether the face has stones drawn (the program's *Pietre* on the elevation): without them the joints between
    the stones cannot be told from cracks."""
    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    return bool((frame_drawings(chunk, entry["id"]).get("stones") or {}).get(entry["id"]))


def window_extent(chunk, entry, bounds=None) -> tuple[tuple, float, int] | None:
    """``(bounds clipped to the orthophoto, its pixel in m, the pixels of the window)``, or None outside it."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    ortho, _relief = elevation_paths(chunk, entry)
    with rasterio.open(str(ortho)) as src:
        full = src.bounds
        res = abs(src.res[0])
    bx = (full.left, full.bottom, full.right, full.top) if bounds is None else (
        max(bounds[0], full.left), max(bounds[1], full.bottom), min(bounds[2], full.right), min(bounds[3], full.top))
    if bx[2] <= bx[0] or bx[3] <= bx[1]:
        return None
    return bx, float(res), int(round((bx[2] - bx[0]) / res)) * int(round((bx[3] - bx[1]) / res))


def tiles(bounds, res: float, budget: int = PIXEL_BUDGET, overlap: float = TILE_OVERLAP) -> list[tuple]:
    """The tiles of the window ``bounds``: ``(core, window)`` pairs, the cores a partition of it, each window its core
    grown by ``overlap`` (within ``bounds``) and of at most ``budget`` pixels of ``res`` m."""
    u0, v0, u1, v1 = bounds
    side = math.sqrt(budget) * res - 2.0 * overlap
    if side <= 0:
        raise ValueError(f"a budget of {budget} pixels is smaller than the overlap of the tiles")
    nu, nv = max(1, math.ceil((u1 - u0) / side)), max(1, math.ceil((v1 - v0) / side))
    du, dv = (u1 - u0) / nu, (v1 - v0) / nv
    out = []
    for j in range(nv):
        for i in range(nu):
            core = (u0 + i * du, v0 + j * dv, u0 + (i + 1) * du, v0 + (j + 1) * dv)
            win = (max(u0, core[0] - overlap), max(v0, core[1] - overlap), min(u1, core[2] + overlap),
                   min(v1, core[3] + overlap))
            out.append((core, win))
    return out


def _merge(pieces: list[dict], res: float) -> list[dict]:
    """The proposals of the tiles, each already clipped to its tile's core, merged where they meet: the areas of the
    same kind that touch in one polygon (the term, depth and notes of the largest piece; a piece "to check" leaves
    the whole with no term), the lines joined end to end (snapped within two pixels; the opening the length-weighted
    mean of the pieces)."""
    import shapely
    from shapely.ops import linemerge, unary_union

    out = []
    areas = [p for p in pieces if p["geometry"].geom_type in ("Polygon", "MultiPolygon")]
    lines = [p for p in pieces if p["geometry"].geom_type in ("LineString", "MultiLineString")]
    out += [p for p in pieces if p not in areas and p not in lines]
    for kind in sorted({p["kind"] for p in areas}):
        group = [p for p in areas if p["kind"] == kind]
        for g in shapely.get_parts(unary_union([p["geometry"].buffer(res * 0.01) for p in group]).buffer(
                -res * 0.01)):
            if g.is_empty:
                continue
            mine = [p for p in group if p["geometry"].intersects(g)]
            if not mine:
                continue
            first = max(mine, key=lambda p: p["geometry"].area)
            depths = [(p["geometry"].area, p["depth_mm"]) for p in mine if p.get("depth_mm") is not None]
            depth = round(sum(a * x for a, x in depths) / sum(a for a, _ in depths), 1) if depths else None
            check = [p for p in mine if p.get("check")]
            code = None if check else first["code"]
            notes = check[0]["notes"] if check else first["notes"]
            out.append({**first, "code": code, "geometry": g, "depth_mm": depth, "notes": notes})
    for code in sorted({str(p["code"]) for p in lines}):
        group = [p for p in lines if str(p["code"]) == code]
        joined = unary_union([p["geometry"] for p in group])
        joined = unary_union(shapely.snap(joined, joined, 2.0 * res))  # the overlaps noded once
        merged = linemerge(joined) if joined.geom_type == "MultiLineString" else joined
        for ln in shapely.get_parts(merged):
            if ln.is_empty:
                continue
            near = [p for p in group if p["geometry"].distance(ln) <= res]
            first = max(near, key=lambda p: p["geometry"].length) if near else group[0]
            widths = [(p["geometry"].length, p["width_mm"]) for p in near if p.get("width_mm") is not None]
            width = sum(a * x for a, x in widths) / sum(a for a, _ in widths) if widths else None
            out.append({**first, "geometry": ln, "width_mm": width})
    return out


def tiled(detector, chunk, entry, bounds=None, budget: int = PIXEL_BUDGET, overlap: float = TILE_OVERLAP,
          progress=None, **kwargs) -> list[dict]:
    """``detector`` (:func:`cracks` or :func:`material_loss`) on the window ``bounds``, at once when it holds at most
    ``budget`` pixels, else by :func:`tiles` with an overlap of ``overlap`` m: each tile's proposals clipped to its
    core and merged across the tiles (:func:`_merge`). A ``stats`` dict in ``kwargs`` gets the tiles' counts summed
    and ``tiles``."""
    import shapely
    from shapely.geometry import box

    progress = _progress(progress)
    extent = window_extent(chunk, entry, bounds)
    if extent is None:
        return []
    bx, res, pixels = extent
    if pixels <= budget:
        return detector(chunk, entry, bounds, progress=progress, **kwargs)
    parts = tiles(bx, res, budget, overlap)
    stats = kwargs.pop("stats", None)
    pieces = []
    for k, (core, win) in enumerate(parts):
        mine = {} if stats is not None else None
        extra = {"stats": mine} if stats is not None else {}
        found = detector(chunk, entry, win, progress=progress.sub(k / len(parts), 0.95 / len(parts)),
                         **kwargs, **extra)
        if stats is not None:
            for key, value in mine.items():
                stats[key] = (stats.get(key, 0) + value) if isinstance(value, (int, float)) and \
                    not isinstance(value, bool) else value
        cell = box(*core)
        for p in found:
            g = p["geometry"].intersection(cell)
            if g.is_empty:
                continue
            want = "Polygon" if p["geometry"].geom_type.endswith("Polygon") else "LineString"
            pieces += [{**p, "geometry": x} for x in shapely.get_parts(g) if x.geom_type == want and not x.is_empty]
    out = _merge(pieces, res)
    if stats is not None:
        stats["tiles"] = len(parts)
    progress(1.0, _t(f"{len(parts)} tasselli: {len(out)} proposte", f"{len(parts)} tiles: {len(out)} proposals"))
    return out


DETECTORS = ("sam", "cracks", "loss", "colours")

__all__ = ["CRACK", "MISSING_PART", "ALVEOLIZATION", "BIG_FACE", "BIG_FACE_RESOLUTION", "SUGGESTIONS", "DETECTORS",
           "MAX_CRACK_WIDTH_MM", "REFERENCE_SCALE", "PIXEL_BUDGET", "TILE_OVERLAP", "Window", "read_window", "lab",
           "sam_pixel", "sam_areas", "like_this", "cracks", "material_loss", "loss_residual", "plane_fit",
           "surface_fit", "looks", "colour_classes", "stones_drawn", "window_extent", "tiles", "tiled"]
