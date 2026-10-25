"""A synthetic trench for the tests: an orthophoto of 8 x 6 m at 5 mm/px (RGB and alpha) of brown ground
with bright "stones" - discs the fake engine of the worker finds - one of them across the seam of two
tiles, one dimmer than the others, one in the corner without data; a DEM of the same area."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CRS = "EPSG:32633"
X0, Y1 = 500000.0, 4500006.0  # the top left corner
WIDTH, HEIGHT = 8.0, 6.0  # metres
RES = 0.005  # m/px
# (x, y, radius, brightness): x and y in metres from the top left corner, y going down
STONES = (
    (1.0, 1.0, 0.20, 250),
    (2.5, 1.2, 0.20, 250),  # across the seam of the first two tiles of 256 px at 1 cm/px
    (5.0, 2.0, 0.25, 245),
    (3.0, 4.0, 0.15, 250),
    (6.5, 4.5, 0.18, 215),  # dimmer: a lower score
    (1.5, 3.5, 0.10, 250),
)
NO_DATA = (7.4, 5.4)  # the corner beyond this (x, y) has no data: a stone there is not seen
HIDDEN = (7.7, 5.7, 0.12, 250)


def orthophoto(path: Path, crs: str = CRS) -> Path:
    from pyarchinit_photogrammetry.engine.dem import write_geotiff

    h, w = int(round(HEIGHT / RES)), int(round(WIDTH / RES))
    rng = np.random.default_rng(7)
    ground = 90 + rng.normal(0, 12, (h, w))
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) * RES
    rgb = np.stack([ground * 1.2, ground, ground * 0.7])
    for x, y, r, v in STONES + (HIDDEN,):
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        rgb[:, disc] = v
    alpha = np.where((xx > NO_DATA[0]) & (yy > NO_DATA[1]), 0, 255)
    img = np.clip(np.concatenate([rgb, alpha[None]]), 0, 255).astype(np.uint8)
    return write_geotiff(path, img, (X0, Y1), RES, crs)


def dem(path: Path, crs: str = CRS) -> Path:
    from pyarchinit_photogrammetry.engine.dem import write_geotiff

    h, w = int(round(HEIGHT / 0.02)), int(round(WIDTH / 0.02))
    yy, xx = np.mgrid[0:h, 0:w] * 0.02
    z = 100.0 + 0.05 * xx - 0.02 * yy
    return write_geotiff(path, z.astype(np.float32), (X0, Y1), 0.02, crs, nodata=-9999.0)


def world(x: float, y: float) -> tuple[float, float]:
    """Metres from the top left corner (y down) -> coordinates of the chunk."""
    return X0 + x, Y1 - y


def project(folder: Path, with_dem: bool = False):
    """A saved project with one chunk, its orthophoto (and its DEM): ``(document, chunk)``."""
    import pyarchinit_photogrammetry as ppg

    folder.mkdir(parents=True, exist_ok=True)
    doc = ppg.Document()
    ch = doc.add_chunk("Saggio")
    doc.meta["sito"] = "Saggio di prova"
    doc.save(folder / "scavo.ppg")
    ch.crs = CRS
    ch._set_product("orthomosaic", orthophoto(ch.data_dir / "orthomosaic.tif"))
    if with_dem:
        ch._set_product("dem", dem(ch.data_dir / "dem.tif"))
    doc.save()
    return doc, ch
