"""A fake PhotoCAD and the projects the tests send to it.

:class:`FakePhotoCAD` answers as the API is documented and as photocad.ai behaved on the real test: ``POST
/convert`` (multipart ``image_file``, ``Idempotency-Key``) -> 202 ``{id, status: processing}``, ``GET
/conversions/{id}`` processing then complete (``width_mm``, ``height_mm`` = the page, the SUBJECT's box),
``GET /conversions/{id}/files/{format}``, ``GET /account``, ``DELETE /conversions/{id}``, errors as ``{"errors":
[{code, title, detail}]}``. The drawing is made from the image it received: the outline of every dark blob
(closed, layer ``OUTLINE``) and an open line along the top of the first one (``DETAIL``), in millimetres from the
lower-left corner of the subject's box - so the tests see where a known square of the orthophoto comes back after
the plugin's registration.

The projects: a chunk in EPSG:32633 with an orthophoto (white, black squares at known pixels) and a DEM; an
elevation of a wall (its orthophoto in (u, v) metres, black squares, a flat relief).
"""
from __future__ import annotations

import email.parser
import email.policy
import io
import json
import re
import urllib.error
import urllib.parse
from pathlib import Path

import numpy as np

KEY = "pcad_live_TESTKEY0123456789abcdef"
BASE = "https://photocad.ai/api/v1"

CRS = "EPSG:32633"
X0, Y1, RES = 500000.0, 4000004.0, 0.01  # upper-left corner of the orthophoto, metres per pixel
W, H = 600, 400
SQUARE = (200, 100, 60)  # col, row of the upper-left pixel, side in pixels


class FakeKeyring:
    """The system keychain as a dictionary (``get/set/delete_password``)."""

    priority = 1

    def __init__(self):
        self.store: dict = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, password):
        self.store[(service, user)] = password

    def delete_password(self, service, user):
        if (service, user) not in self.store:
            raise KeyError("no such password")
        del self.store[(service, user)]


class _Answer(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _error(url, code, slug, title, detail="", headers=None):
    body = json.dumps({"errors": [{"code": slug, "title": title, "detail": detail}]}).encode()
    from email.message import Message

    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(url, code, title, msg, io.BytesIO(body))


def dark_boxes(image_bytes: bytes):
    """``([(c0, r0, c1, r1), ...], w, h)``: the boxes of the dark blobs of the image (pixel edges, exclusive end),
    from left to right."""
    import cv2
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("L"))
    n, _labels, stats, _c = cv2.connectedComponentsWithStats((a < 128).astype(np.uint8), connectivity=8)
    boxes = sorted((int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, area in stats[1:] if area > 4)
    return boxes, a.shape[1], a.shape[0]


NATURAL = 0.125  # mm per pixel of the image: the natural size of the drawing (2400 px -> 300 mm, as at Khutm)


class FakePhotoCAD:
    """The opener given to the client instead of urllib: ``fake(request, timeout)``.

    It behaves as photocad.ai did on the real test (Khutm, 2026-11-04): the page of the drawing is the box of the
    SUBJECT (here: of the dark blobs), not the image; the coordinates are millimetres from the lower-left corner of
    that page at a natural size (:data:`NATURAL`), whatever ``units`` says ($INSUNITS 4); ``longest`` scales the page
    so that its long side is ``longest`` x 1000 (the unit asked, written in mm). ``page_pad`` (left, bottom, right,
    top, in pixels) makes the subject box larger than the blobs, as a photo's subject never matches our content box
    exactly; ``garbage`` draws outlines that are not in the image. ``Idempotency-Key``: a POST with a key seen
    before gives the same conversion again, not charged."""

    def __init__(self, credits: int = 5, polls: int = 2, fail: dict | None = None, status_failed: bool = False,
                 free: int = 5, account: str = "real", page_pad=(0, 0, 0, 0), garbage: bool = False,
                 open_lines: bool = False, ring: bool = False, tiny: bool = False, gap_px: float = 0.5,
                 honour_keys: bool = True, boxes_px=None, offset_px=(0, 0), random_lines: int = 0):
        self.credits = credits
        self.free = free  # free conversions left
        self.account = account  # "real": the shape photocad.ai gives (2026-11-04); "flat": an older guess
        self.polls = polls  # "processing" answers before "complete"
        # "METHOD path-prefix" -> list of codes to answer first: an HTTP status, 0 a connection broken before the
        # server saw the request, -1 a connection broken after the server did what was asked
        self.fail = dict(fail or {})
        self.status_failed = status_failed
        self.page_pad = tuple(page_pad)
        self.garbage = garbage
        # as PhotoCAD draws stones: each side an open polyline, the ends 0.5 px short of the corners (a 1 px gap)
        self.open_lines = open_lines
        self.honour_keys = honour_keys  # False: a key seen before is ignored (a POST sent again is charged again)
        self.boxes_px = boxes_px  # the outlines to draw, in pixels of the image (else: the dark blobs)
        self.offset_px = tuple(offset_px)  # the drawing moved inside its page (a placement the search cannot reach)
        self.random_lines = random_lines  # instead of the outlines, random polylines over the page
        self.gap_px = gap_px  # 0: the sides chained end to end exactly, as at Khutm
        self.ring = ring  # four open lines around a white rectangle (background: never a stone)
        self.tiny = tiny  # a 4 x 4 px closed loop inside the first blob (smaller than the minimum area)
        self.requests: list[tuple[str, str, dict]] = []  # method, path?query, headers
        self.images: dict[str, bytes] = {}
        self.options: dict[str, dict] = {}
        self.keys: dict[str, str] = {}  # idempotency key -> conversion id
        self._seen: dict[str, int] = {}
        self.deleted: list[str] = []
        self.n = 0
        self.charged = 0

    def calls(self, method: str, prefix: str = "") -> list[str]:
        return [p for m, p, _ in self.requests if m == method and p.startswith(prefix)]

    def __call__(self, request, timeout):
        url = request.full_url
        assert url.startswith(BASE), url
        parts = urllib.parse.urlsplit(url)
        path = parts.path[len(urllib.parse.urlsplit(BASE).path):]
        query = dict(urllib.parse.parse_qsl(parts.query))
        method = request.get_method()
        headers = {k.lower(): v for k, v in request.header_items()}
        self.requests.append((method, path + ("?" + parts.query if parts.query else ""), headers))
        drop_after = False
        for rule, codes in self.fail.items():
            m, prefix = rule.split(" ", 1)
            if m == method and path.startswith(prefix) and codes:
                code = codes.pop(0)
                if code == 0:
                    raise urllib.error.URLError("connection reset by peer")
                if code == -1:
                    drop_after = True
                    break
                slugs = {401: "unauthorized", 402: "out_of_credits", 413: "payload_too_large",
                         422: "unprocessable", 429: "rate_limited", 500: "server_error"}
                extra = {"Retry-After": "2"} if code == 429 else None
                raise _error(url, code, slugs.get(code, "error"), f"Error {code}", f"detail {code}", extra)
        answer = self._answer(url, path, query, method, headers, request)
        if drop_after:
            raise urllib.error.URLError("connection reset by peer")
        return answer

    def _answer(self, url, path, query, method, headers, request):
        if headers.get("authorization") != f"Bearer {KEY}":
            raise _error(url, 401, "unauthorized", "Unauthorized", "Invalid API key")
        if method == "GET" and path == "/account":
            if self.account == "flat":
                return _Answer(json.dumps({"credits_remaining": self.credits}).encode())
            return _Answer(json.dumps({"credits": {"total": self.credits, "subscription": 0, "api": self.credits},
                                       "plan": "free", "ceiling": {"limit": None, "used": 0},
                                       "free_conversions": {"granted": 5, "remaining": self.free}}).encode())
        if method == "POST" and path == "/convert":
            key = headers.get("idempotency-key")
            if key and key in self.keys and self.honour_keys:
                return _Answer(json.dumps({"id": self.keys[key], "status": "processing"}).encode(), 202)
            ctype = headers.get("content-type", "")
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                f"Content-Type: {ctype}\r\n\r\n".encode() + request.data)
            fields = {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                fields[name] = part.get_payload(decode=True)
            self.n += 1
            cid = f"conv_{self.n:04d}"
            if key:
                self.keys[key] = cid
            self.images[cid] = fields["image_file"]
            self.options[cid] = {k: v.decode() for k, v in fields.items() if k != "image_file"}
            self.charged += 1
            if self.free > 0:
                self.free -= 1
            else:
                self.credits -= 1
            return _Answer(json.dumps({"id": cid, "status": "processing"}).encode(), 202)
        m = re.fullmatch(r"/conversions/([^/]+)(?:/files/([a-z]+))?", path)
        if m is None:
            raise _error(url, 404, "not_found", "Not found")
        cid, fmt = m.groups()
        if method == "DELETE":
            self.deleted.append(cid)
            return _Answer(b"", 204)
        if cid not in self.images:
            raise _error(url, 404, "not_found", "Not found")
        if fmt is None:
            n = self._seen[cid] = self._seen.get(cid, 0) + 1
            boxes, _w, _h = self._boxes(cid) if self.images[cid] else ([], 0, 0)
            if self.status_failed or (n > self.polls and not boxes):
                return _Answer(json.dumps({"id": cid, "status": "failed", "credits_remaining": self.credits,
                                           "error": {"code": "general_error",
                                                     "detail": "No drawing found in the image"}}).encode())
            if n <= self.polls:
                return _Answer(json.dumps({"id": cid, "status": "processing"}).encode())
            page = self.page(cid)
            return _Answer(json.dumps({
                "id": cid, "status": "complete", "credits_remaining": self.credits, "title": "Squares",
                "width_mm": round((page[2] - page[0]) * NATURAL, 3), "height_mm": round((page[3] - page[1]) * NATURAL, 3),
                "files": {f: f"{BASE}/conversions/{cid}/files/{f}" for f in ("dxf", "dwg", "svg", "pdf", "geometry")},
            }).encode())
        return _Answer(self.file(cid, fmt, query))

    def _boxes(self, cid):
        if self.boxes_px is None:
            return dark_boxes(self.images[cid])
        w, h = _size(self.images[cid])
        return list(self.boxes_px), w, h

    def page(self, cid):
        """The subject box in image pixels (c0, r0, c1, r1, edges), padded."""
        boxes, w, h = self._boxes(cid)
        c0 = min(b[0] for b in boxes) - self.page_pad[0]
        r1 = max(b[3] for b in boxes) + self.page_pad[1]
        c1 = max(b[2] for b in boxes) + self.page_pad[2]
        r0 = min(b[1] for b in boxes) - self.page_pad[3]
        return c0, r0, c1, r1

    def drawing(self, cid, query):
        """Closed outlines of the blobs and an open line along the top of the first one, in the units PhotoCAD
        writes (mm from the page's lower-left corner; ``longest`` x 1000 on the long side when given)."""
        boxes, w, h = self._boxes(cid)
        pc0, pr0, pc1, pr1 = self.page(cid)
        s = NATURAL
        if "longest" in query:
            s = float(query["longest"]) * 1000.0 / max(pc1 - pc0, pr1 - pr0)

        ox, oy = self.offset_px

        def mm(c, r):
            return ((c + ox - pc0) * s, (pr1 - r - oy) * s)

        outlines = []
        for c0, r0, c1, r1 in boxes:
            if self.garbage:  # somewhere else: shifted by a third of the page and mirrored
                c0, c1 = pc1 - (c1 - pc0) + (pc1 - pc0) // 3, pc1 - (c0 - pc0) + (pc1 - pc0) // 3
            outlines.append([mm(c0, r0), mm(c1, r0), mm(c1, r1), mm(c0, r1)])
        c0, r0, c1, _r1 = boxes[0]
        line = [mm(c0, r0), mm((c0 + c1) / 2, r0), mm(c1, r0)]
        self._open = []
        if self.ring:  # a white rectangle inside the page, between the blobs
            a0, a1 = pc0 + 0.45 * (pc1 - pc0), pc0 + 0.75 * (pc1 - pc0)
            b0, b1 = pr0 + 0.05 * (pr1 - pr0), pr0 + 0.35 * (pr1 - pr0)
            self._open += _sides([mm(a0, b0), mm(a1, b0), mm(a1, b1), mm(a0, b1)], 0.5 * s)
        if self.tiny:
            outlines.append([mm(c0 + 10, r0 + 10), mm(c0 + 14, r0 + 10), mm(c0 + 14, r0 + 14), mm(c0 + 10, r0 + 14)])
        if self.open_lines:
            for o in list(outlines):
                self._open += _sides(o, self.gap_px * s)
            outlines = []
        if self.random_lines:
            rng = np.random.default_rng(11)
            outlines = []
            for _ in range(self.random_lines):
                start = rng.uniform((pc0, pr0), (pc1, pr1))
                pts = start + np.cumsum(rng.normal(0, 25, (4, 2)), axis=0)
                pts = np.clip(pts, (pc0, pr0), (pc1, pr1))
                self._open.append([mm(c, r) for c, r in pts])
        return outlines, line, (pc1 - pc0) * s, (pr1 - pr0) * s

    def file(self, cid: str, fmt: str, query: dict) -> bytes:
        outlines, line, pw, ph = self.drawing(cid, query)
        if fmt == "dxf":
            import ezdxf

            doc = ezdxf.new("R2000")
            doc.header["$INSUNITS"] = 4  # always millimetres, whatever "units" asked
            doc.layers.add("OUTLINE")
            doc.layers.add("DETAIL")
            msp = doc.modelspace()
            for outline in outlines:
                msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": "OUTLINE"})
            msp.add_lwpolyline(line, close=False, dxfattribs={"layer": "DETAIL"})
            for side in self._open:
                msp.add_lwpolyline(side, close=False, dxfattribs={"layer": "OUTLINE"})
            out = io.StringIO()
            doc.write(out)
            return out.getvalue().encode()
        if fmt == "svg":
            polys = "".join('<polygon points="' + " ".join(f"{x:.4f},{ph - y:.4f}" for x, y in o)
                            + '" fill="none" stroke="black"/>' for o in outlines)
            return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw}mm" height="{ph}mm" '
                    f'viewBox="0 0 {pw} {ph}">{polys}</svg>').encode()
        if fmt == "pdf":
            return b"%PDF-1.4\n% fake\n"
        if fmt == "geometry":
            return json.dumps({"units": "mm", "width": pw, "height": ph, "paths": outlines + [line]}).encode()
        raise ValueError(fmt)


def _size(image_bytes: bytes):
    from PIL import Image

    w, h = Image.open(io.BytesIO(image_bytes)).size
    return w, h


def _sides(corners, gap):
    """The four sides of a box as open lines, each ``gap`` short of the corners at both ends; with no gap the end of
    each side a few 1e-9 mm off the start of the next (float noise: what the placement leaves at Khutm)."""
    out = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (b - a) / np.hypot(*(b - a))
        end = b - u * gap if gap else b + 3e-9
        out.append([tuple(a + u * gap), tuple(end)])
    return out


# ---------------------------------------------------------------- projects
def _write_rgb(path: Path, rgb: np.ndarray, transform, crs) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.full(rgb.shape[1:], 255, np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=rgb.shape[2], height=rgb.shape[1], count=4, dtype="uint8",
                       crs=crs, transform=transform) as dst:
        dst.write(np.concatenate([rgb, alpha[None]]).astype(np.uint8))


SQUARE2 = (400, 220, 40)


def square_image(w: int, h: int, square=SQUARE) -> np.ndarray:
    """White, with black squares (``square``: one ``(col, row, side)`` or a list of them)."""
    rgb = np.full((3, h, w), 255, np.uint8)
    for c, r, side in ([square] if isinstance(square[0], int) else square):
        rgb[:, r:r + side, c:c + side] = 0
    return rgb


def square_world(square=SQUARE, x0=X0, y1=Y1, res=RES):
    """The world box of the square: xmin, ymin, xmax, ymax."""
    c, r, side = square
    return x0 + c * res, y1 - (r + side) * res, x0 + (c + side) * res, y1 - r * res


def project(tmp_path: Path, w: int = W, h: int = H, square=SQUARE):
    """A saved project whose chunk has an orthophoto with a black square and a DEM with a mound."""
    import rasterio
    from rasterio.transform import from_origin

    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    ch = doc.add_chunk("Scavo")
    doc.chunk = ch
    doc.save(tmp_path / "scavo.ppg")
    ch.crs = CRS
    tr = from_origin(X0, Y1, RES, RES)
    ortho = ch.data_dir / "orthomosaic.tif"
    _write_rgb(ortho, square_image(w, h, square), tr, CRS)
    ch._set_product("orthomosaic", ortho)
    dem = ch.data_dir / "dem.tif"
    yy, xx = np.mgrid[0:h // 2, 0:w // 2]
    z = 100.0 + 0.5 * np.exp(-(((xx - w / 4) / 40.0) ** 2 + ((yy - h / 4) / 30.0) ** 2))
    with rasterio.open(dem, "w", driver="GTiff", width=w // 2, height=h // 2, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0, Y1, RES * 2, RES * 2), nodata=float("nan")) as dst:
        dst.write(z.astype(np.float32), 1)
    ch._set_product("dem", dem)
    doc.save()
    return doc, ch


# the elevation: a vertical wall facing south, u eastwards from ORIGIN, v = height, d towards the viewer (south)
E_ORIGIN = np.array([500001.0, 4000002.0, 0.0])
E_U0, E_V0, E_RES = 0.0, 100.0, 0.005  # lower-left corner of the elevation's orthophoto in (u, v), metres per pixel
E_W, E_H = 400, 300
E_SQUARE = (100, 50, 80)
E_SQUARE2 = (260, 160, 60)
E_DEPTH = 0.02  # the flat relief: the surface 2 cm in front of the plane


def add_elevation(ch, w=E_W, h=E_H, square=E_SQUARE) -> dict:
    """An elevation written by hand in the format of the program (orthophoto and relief in (u, v))."""
    import rasterio
    from rasterio.transform import from_origin

    from pyarchinit_photogrammetry.engine.elevation import Plane

    frame = Plane(E_ORIGIN, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    folder = ch.data_dir / "elevations" / "e1"
    tr = from_origin(E_U0, E_V0 + h * E_RES, E_RES, E_RES)
    _write_rgb(folder / "orthophoto.tif", square_image(w, h, square), tr, None)
    with rasterio.open(folder / "relief.tif", "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       transform=tr, nodata=float("nan")) as dst:
        dst.write(np.full((1, h, w), E_DEPTH, np.float32))
    entry = {"id": "e1", "name": "Prospetto sud", "kind": "elevation", "crs": ch.crs, "frame": frame.to_dict(),
             "depth": [-0.5, 0.5], "resolution": E_RES, "bounds": [E_U0, E_V0, E_U0 + w * E_RES, E_V0 + h * E_RES],
             "v_is_height": True, "orthophoto": "elevations/e1/orthophoto.tif", "relief": "elevations/e1/relief.tif",
             "size": [w, h]}
    ch.meta.setdefault("elevations", []).append(entry)
    return entry


def elevation_square(square=E_SQUARE, h=E_H):
    """The (u, v) box of the square of the elevation: umin, vmin, umax, vmax."""
    c, r, side = square
    v_top = E_V0 + h * E_RES
    return E_U0 + c * E_RES, v_top - (r + side) * E_RES, E_U0 + (c + side) * E_RES, v_top - r * E_RES


# ---------------------------------------------------------------- a textured wall (Khutm-like)
def masonry(w: int = 1200, h: int = 800, seed: int = 5):
    """A dry-stone wall: courses of blocks with a stone texture, dark joints of 4 px. ``(rgb (3, h, w), boxes)``:
    the blocks as (c0, r0, c1, r1) pixel edges - what PhotoCAD would draw."""
    import cv2

    rng = np.random.default_rng(seed)
    grey = np.full((h, w), 55.0)
    boxes = []
    r = 2
    while r < h - 40:
        ch = int(rng.integers(55, 110))
        r1 = min(h - 2, r + ch)
        c = 2
        while c < w - 40:
            cw = int(rng.integers(80, 220))
            c1 = min(w - 2, c + cw)
            boxes.append((c, r, c1, r1))
            grey[r:r1, c:c1] = rng.uniform(130, 205)
            c = c1 + 4
        r = r1 + 4
    texture = cv2.GaussianBlur(rng.normal(0, 30, (h, w)), (0, 0), 1.2) + cv2.GaussianBlur(rng.normal(0, 60, (h, w)),
                                                                                         (0, 0), 6)
    stone = grey > 100
    grey = np.where(stone, grey + texture, grey + texture * 0.3)
    rgb = np.clip(np.stack([grey * 1.05, grey, grey * 0.85]), 0, 255).astype(np.uint8)
    return rgb, boxes


def masonry_project(tmp_path: Path):
    """A project whose orthophoto (1 cm, 1200 x 800 px: no zoom) is the textured wall: ``(doc, chunk, boxes)``."""
    import rasterio  # noqa: F401
    from rasterio.transform import from_origin

    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    ch = doc.add_chunk("Muro")
    doc.chunk = ch
    doc.save(tmp_path / "muro.ppg")
    ch.crs = CRS
    rgb, boxes = masonry()
    ortho = ch.data_dir / "orthomosaic.tif"
    _write_rgb(ortho, rgb, from_origin(X0, Y1, RES, RES), CRS)
    ch._set_product("orthomosaic", ortho)
    doc.save()
    return doc, ch, boxes
