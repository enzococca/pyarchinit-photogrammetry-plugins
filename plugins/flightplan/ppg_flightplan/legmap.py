"""A map for every leg: the chunk's orthophoto underneath (grey without one), the route of that leg alone on
top, with the takeoff, the first and the last point, the shots, and the line of numbers; and the 1920 x 1080
preview the library of DJI Fly shows for a mission.

The map is drawn in Web Mercator (EPSG:3857), as the planner drew it on its tiles. The background is a
callable (:data:`Background`) that gives the image of a WGS84 box at a size, its rows linear in Web Mercator:
:func:`orthophoto_background` reads the chunk's orthomosaic; nothing is ever fetched from the network.

Adapted from the author's Photo3D planner: framing, drawing, captions and the DJI Fly preview are the planner's;
the satellite tiles it fetched are replaced by the chunk's orthophoto.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from .geometry import LatLon, densify
from .specials import num
from .texts import t

COLOURS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1", "#6d4c41", "#c0ca33"]
GREY = (214, 220, 226)
WIDTH, HEIGHT = 1400, 1000
BAND = 64                      # the strip of text at the bottom
MARGIN = 0.84                  # the share of the frame the leg fills
MIN_GROUND_M_PER_PX = 0.05     # the closest the map zooms in on a tiny area

# The background of a map: (west, south, east, north) in WGS84 degrees, width and height in pixels -> the
# image of that box (rows linear in Web Mercator), or None for the grey.
Background = Callable[[tuple[float, float, float, float], int, int], Optional[Image.Image]]

_R = 6378137.0


def _merc(lat: float, lon: float) -> tuple[float, float]:
    """Web Mercator metres (EPSG:3857) of a point."""
    lat = max(-85.0, min(85.0, lat))
    return _R * math.radians(lon), _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _unmerc(x: float, y: float) -> LatLon:
    return math.degrees(2 * math.atan(math.exp(y / _R)) - math.pi / 2), math.degrees(x / _R)


def grey_background(bounds, width: int, height: int) -> Image.Image:
    """No background: the grey of the planner's missing tiles."""
    return Image.new("RGB", (width, height), GREY)


def _overview_for(src, metres_per_px: float) -> Optional[int]:
    """The overview level of the raster closest to (not coarser than) the resolution needed; None = full."""
    try:
        factors = src.overviews(1)
        cell = abs(src.res[0]) if src.crs is not None and src.crs.is_projected else None
    except Exception:  # noqa: BLE001 - a raster that does not say: read it whole
        return None
    if not factors or not cell:
        return None
    level = None
    for k, f in enumerate(factors):
        if cell * f <= metres_per_px:
            level = k
    return level


def _to_rgb8(data, valid):
    """Bands of any type to 8 bits: uint8 as it is, anything else stretched between the 2nd and 98th percentile
    of the valid cells (an orthophoto of reflectance or temperatures)."""
    import numpy as np

    if data.dtype == np.uint8:
        return data
    out = np.zeros(data.shape, dtype=np.uint8)
    for b in range(data.shape[0]):
        v = data[b][valid & np.isfinite(data[b])]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, (2, 98))
        scaled = (data[b].astype("float64") - lo) / max(hi - lo, 1e-12) * 255.0
        out[b] = np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)
    return out


def orthophoto_background(chunk) -> Optional[Background]:
    """The chunk's orthomosaic as the background of the maps; None when the chunk has none (or the file says
    nowhere where it is). The window covering the leg is warped to Web Mercator at the image size, from the
    overview of the right resolution; cells without data stay grey."""
    path = getattr(chunk, "orthomosaic_path", None)
    if not path or not Path(path).is_file():
        return None

    def background(bounds, width: int, height: int) -> Optional[Image.Image]:
        import numpy as np
        import rasterio
        from pyproj import Transformer
        from rasterio.enums import MaskFlags, Resampling
        from rasterio.transform import from_bounds
        from rasterio.vrt import WarpedVRT

        west, south, east, north = bounds
        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x0, y0 = to_merc.transform(west, south)
        x1, y1 = to_merc.transform(east, north)
        ground = (x1 - x0) / width * math.cos(math.radians((south + north) / 2))
        with rasterio.open(path) as probe:
            if probe.crs is None:
                return None
            level = _overview_for(probe, ground)
        kwargs = {} if level is None else {"overview_level": level}
        with rasterio.open(path, **kwargs) as src:
            # without nodata, alpha or mask, the warp is asked for an alpha: else the cells outside the raster
            # would read as valid black
            all_valid = MaskFlags.all_valid in src.mask_flag_enums[0]
            with WarpedVRT(src, crs="EPSG:3857", transform=from_bounds(x0, y0, x1, y1, width, height),
                           width=width, height=height, resampling=Resampling.bilinear, add_alpha=all_valid) as vrt:
                bands = [1, 1, 1] if src.count < 3 else [1, 2, 3]
                data = vrt.read(bands)
                valid = vrt.dataset_mask() > 0
        if not valid.any():
            return None
        rgb = _to_rgb8(data, valid)
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = GREY
        image[valid] = np.moveaxis(rgb, 0, -1)[valid]
        return Image.fromarray(image, "RGB")

    background.attribution = t("sfondo: ortofoto del progetto", "background: the project's orthophoto")
    return background


def _font(size: int) -> ImageFont.ImageFont:
    names = ["/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    try:  # the program always has matplotlib, and with it DejaVu
        import matplotlib

        names.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
    except Exception:  # noqa: BLE001
        pass
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an old Pillow
        return ImageFont.load_default()


def caption(piano: dict, leg) -> tuple[str, str]:
    """The two lines of text of the map: who the leg is, and its numbers (``leg`` a
    :class:`~ppg_flightplan.plan.PlanLeg`)."""
    from .plan import plan_camera

    p = leg.route
    b = piano["blocchi"][leg.block_index]
    cam = plan_camera(piano)
    battery_minutes = float(piano.get("minutiBatteria", 18))
    high = bool(b.get("alta", piano.get("alta", False)))
    shot = (t("scatto ogni {0} s", "a shot every {0} s").format(num(p.interval)) if p.interval
            else t("uno scatto per punto", "one shot per point"))
    line1 = (f"{piano.get('nome', 'piano')} · {b.get('nome', '')} · "
             + t("tratta {0} di {1}", "leg {0} of {1}").format(leg.number, leg.total))
    mission = ((piano.get("djiFly") or {}).get(leg.key) or {}).get("nomeDji")
    if mission:
        line1 += f" · DJI Fly {mission}"
    line2 = t("quota {0} m · camera {1}° · {2:.1f} m/s · {3} · {4} waypoint · {5} scatti · {6:.1f} km · "
              "{7:.0f} min · {8:.1f} batterie · GSD {9:.2f} cm/px",
              "height {0} m · camera {1}° · {2:.1f} m/s · {3} · {4} waypoints · {5} shots · {6:.1f} km · "
              "{7:.0f} min · {8:.1f} batteries · GSD {9:.2f} cm/px").format(
        num(p.height), num(p.pitch), p.effective_speed, shot, len(p.points), p.shots(), p.length / 1000,
        p.minutes(), p.minutes() / max(battery_minutes, 1), cam.gsd_cm(p.height, high))
    return line1, line2


def _encode(img: Image.Image, fmt: str = "JPEG") -> bytes:
    out = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(out, "JPEG", quality=86, optimize=True)
    else:
        img.save(out, "PNG", optimize=True)
    return out.getvalue()


def draw_leg(piano: dict, index: int, width: int = WIDTH, height: int = HEIGHT,
             background: Optional[Background] = None) -> Image.Image:
    """The map of the leg in position ``index`` (order of flight, from 0) as a Pillow image: its route alone,
    the others barely sketched in white to see where it is compared with the rest."""
    from .plan import plan_legs

    legs = plan_legs(piano)
    if not 0 <= index < len(legs):
        raise IndexError(t("il piano ha {0} tratte", "the plan has {0} legs").format(len(legs)))
    leg = legs[index]
    p = leg.route
    b = piano["blocchi"][leg.block_index]
    colour = COLOURS[leg.block_index % len(COLOURS)]

    framed = list(p.points) + ([leg.takeoff] if leg.takeoff else []) + ([p.poi] if p.poi else [])
    xs, ys = zip(*(_merc(la, lo) for la, lo in framed))
    mid_lat = sum(q[0] for q in framed) / len(framed)
    k = math.cos(math.radians(mid_lat))           # ground metres per Mercator metre
    dx, dy = max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)
    scale = min(width * MARGIN / dx, (height - BAND) * MARGIN / dy)       # pixels per Mercator metre
    scale = min(scale, k / MIN_GROUND_M_PER_PX)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2 - BAND / 2 / scale    # the leg in the middle of the space above the band

    north_west = _unmerc(cx - width / 2 / scale, cy + height / 2 / scale)
    south_east = _unmerc(cx + width / 2 / scale, cy - height / 2 / scale)
    bounds = (north_west[1], south_east[0], south_east[1], north_west[0])
    img = (background or grey_background)(bounds, width, height) or grey_background(bounds, width, height)
    img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    drawing = ImageDraw.Draw(img, "RGBA")

    def px(q: LatLon) -> tuple[float, float]:
        x, y = _merc(q[0], q[1])
        return (x - cx) * scale + width / 2, (cy - y) * scale + height / 2

    for other in legs:  # the other legs of the plan, barely sketched
        if other is not leg and len(other.route.points) >= 2:
            drawing.line([px(q) for q in other.route.points], fill=(255, 255, 255, 90), width=2)
    if b.get("tipo", "griglia") == "griglia" and len(b.get("area") or []) >= 3:  # the perimeter of the block
        area = [px(tuple(q)) for q in b["area"]]
        drawing.line(area + [area[0]], fill=(255, 255, 255, 160), width=2)
    line = [px(q) for q in p.points]
    if len(line) >= 2:
        drawing.line(line, fill=(0, 0, 0, 130), width=8)
        drawing.line(line, fill=colour, width=4)
    shots = p.points if p.mode == "per_punto" else densify(p.points, max(p.effective_step, 1.0))
    every = max(1, len(shots) // 400)
    for n, q in enumerate(shots):
        if n % every == 0:
            x, y = px(q)
            drawing.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 255, 255, 235), outline=colour)

    def circle(q: LatLon, radius: float, fill, edge, text: Optional[str] = None) -> None:
        x, y = px(q)
        drawing.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=edge, width=2)
        if text:
            drawing.text((x, y), text, fill="black", font=_font(int(radius * 1.1)), anchor="mm")

    if leg.takeoff:
        x0, y0 = px(leg.takeoff)
        x1, y1 = px(p.points[0])
        drawing.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, 190), width=2)
    if p.poi:
        circle(p.poi, 7, "#ff9800", "white")
    circle(p.points[0], 10, "#43a047", "white", "1")
    circle(p.points[-1], 10, "white", colour, str(len(p.points)))
    if leg.takeoff:
        circle(leg.takeoff, 15, "white", "black", "H")

    line1, line2 = caption(piano, leg)
    drawing.rectangle([0, height - BAND, width, height], fill=(0, 0, 0, 160))
    drawing.text((14, height - BAND + 8), line1, fill="white", font=_font(24))
    drawing.text((14, height - BAND + 40), line2, fill="white", font=_font(15))
    metres_per_px = k / scale
    bar = 20
    for candidate in (1000, 500, 200, 100, 50, 20):
        if candidate / metres_per_px <= width / 4:
            bar = candidate
            break
    lpx = bar / metres_per_px
    drawing.rectangle([14, height - BAND - 22, 14 + lpx, height - BAND - 14], fill="white", outline="black")
    drawing.text((14, height - BAND - 46), f"{bar} m", fill="white", font=_font(16), stroke_width=2,
                 stroke_fill="black")
    attribution = getattr(background, "attribution", "") if background else ""
    if attribution:
        drawing.text((width - 12, height - BAND - 12), attribution, fill="white", font=_font(13), anchor="rs",
                     stroke_width=2, stroke_fill="black")
    drawing.text((width - 12, height - BAND - 32),
                 t("H decollo · 1 primo punto · {0} ultimo punto", "H takeoff · 1 first point · {0} last point")
                 .format(len(p.points)),
                 fill="white", font=_font(13), anchor="rs", stroke_width=2, stroke_fill="black")
    return img


def leg_image(piano: dict, index: int, width: int = WIDTH, height: int = HEIGHT, fmt: str = "JPEG",
              background: Optional[Background] = None) -> bytes:
    """The map of the leg ``index``, encoded (JPEG or PNG)."""
    return _encode(draw_leg(piano, index, width, height, background), fmt)


def dji_preview(piano: dict, index: int, number: Optional[int] = None,
                background: Optional[Background] = None) -> bytes:
    """The preview for the mission library of DJI Fly (``map_preview/<uuid>/<uuid>.jpg``, 1920 x 1080 as its
    own): the map of the leg with a large label at the top, number and name, readable in the thumbnail too. So
    the placeholder mission, which keeps the name DJI Fly gave it, is recognised from the image."""
    from .plan import plan_legs

    img = draw_leg(piano, index, 1920, 1080, background)
    leg = plan_legs(piano)[index]
    n = index + 1 if number is None else number
    line1 = f"{n} · {leg.route.name}"
    line2 = f"{len(leg.route.points)} waypoint · {num(leg.route.height)} m · {leg.route.minutes():.0f} min"
    mission = ((piano.get("djiFly") or {}).get(leg.key) or {}).get("nomeDji")
    if mission:
        line2 += f" · DJI Fly {mission}"
    drawing = ImageDraw.Draw(img, "RGBA")
    f1, f2 = _font(84), _font(40)
    w1, w2 = drawing.textlength(line1, font=f1), drawing.textlength(line2, font=f2)
    w = max(w1, w2) + 80
    h = 84 + 40 + 70
    x0, y0 = (1920 - w) / 2, 70.0   # at the top, where the route does not usually pass
    drawing.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=28, fill=(0, 0, 0, 170),
                              outline=(255, 255, 255, 200), width=3)
    drawing.text((1920 / 2, y0 + 24), line1, fill="white", font=f1, anchor="ma")
    drawing.text((1920 / 2, y0 + 24 + 84 + 18), line2, fill=(255, 255, 255, 230), font=f2, anchor="ma")
    return _encode(img)


def leg_images(piano: dict, **kw) -> list[tuple[str, bytes]]:
    """One image per leg, with the file name of the leg."""
    from .plan import plan_legs

    extension = "png" if kw.get("fmt", "JPEG").upper() == "PNG" else "jpg"
    return [(f"{leg.file_name}.{extension}", leg_image(piano, k, **kw)) for k, leg in enumerate(plan_legs(piano))]


def zip_images(piano: dict, background: Optional[Background] = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in leg_images(piano, background=background):
            z.writestr(name, data)
    return buffer.getvalue()
