"""Fake terrain for the tests: tiles of the public DEM coded as the terrarium format wants, and small
GeoTIFFs of heights written with rasterio. All on the neutral place of the tests (45.0 N 10.0 E)."""
import numpy as np
from PIL import Image

# The design area of the tests: about 470 x 670 m around 45.0 N 10.0 E (inside one tile of zoom 13).
AREA = [(44.9970, 9.9970), (44.9970, 10.0030), (45.0030, 10.0030), (45.0030, 9.9970)]


def fake_tile(height_m: float) -> Image.Image:
    """A tile all at the same height, coded as the terrarium format wants."""
    v = int(round((height_m + 32768.0) * 256))
    r, g, b = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF
    return Image.new("RGB", (256, 256), (r, g, b))


def tile_rising_north(z, x, y) -> Image.Image:
    """Higher to the north: about ten metres between the bottom and the top of the area."""
    img = Image.new("RGB", (256, 256))
    pixel = img.load()
    for row in range(256):
        q = 270.0 - row * 0.2                      # row 0 is the north
        v = int(round((q + 32768.0) * 256))
        for col in range(256):
            pixel[col, row] = ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
    return img


def write_dem(path, heights, *, x0, y0, step, crs="EPSG:4326", nodata=None):
    """A float32 GeoTIFF: ``heights`` rows from the north, (x0, y0) the top-left corner of the first cell."""
    import rasterio
    from rasterio.transform import from_origin

    a = np.asarray(heights, dtype="float32")
    profile = dict(driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype="float32",
                   crs=crs, transform=from_origin(float(x0), float(y0), float(step), float(step)))
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)
    return path
