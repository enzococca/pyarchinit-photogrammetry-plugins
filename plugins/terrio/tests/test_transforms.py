"""Transformations through terrio never fall back in silence: from terrio 0.6 a transformation
whose grid is missing goes on with a rougher one (even a ballpark: about 100 m between OSGB36 and
WGS 84) with only a warning; the plugin asks for the best one or says why it cannot (no network:
the grid of OSTN15 is not on the computer)."""
import numpy as np
import pytest

terrio = pytest.importorskip("terrio")

from ppg_terrio import transforms  # noqa: E402

LONDON = (530000.0, 180000.0)  # British National Grid (EPSG:27700): OSTN15 grid, else a Helmert or a ballpark


@pytest.fixture
def no_grids(tmp_path):
    terrio.set_grid_search_paths([str(tmp_path)])
    terrio.set_network_enabled(False)
    yield
    terrio.set_grid_search_paths(None)
    terrio.set_network_enabled(None)


needs_only_best = pytest.mark.skipif(not transforms.has_only_best(terrio), reason="only_best comes with terrio 0.6")


@needs_only_best
def test_a_missing_grid_is_an_error_not_a_rougher_transformation(no_grids):
    with pytest.raises(transforms.TransformUnavailable, match="EPSG:27700"):
        transforms.transform_points(terrio, [LONDON[0]], [LONDON[1]], "EPSG:27700", "EPSG:4326", what="Prova")
    # what the plugin no longer accepts: terrio alone goes on, warning only
    with pytest.warns(UserWarning, match="missing grid"):
        terrio.transform_points([LONDON[0]], [LONDON[1]], "EPSG:27700", "EPSG:4326")
    # a transformation without grids is not touched
    lon, lat = transforms.transform_points(terrio, [430000.0], [4554000.0], "EPSG:32633", "EPSG:4326")[:2]
    assert 14 < lon[0] < 15 and 41 < lat[0] < 42


@needs_only_best
def test_a_warp_never_takes_a_ballpark(no_grids):
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    r = terrio.Raster.from_array(a, (LONDON[0], 10.0, 0.0, LONDON[1], 0.0, -10.0), crs="EPSG:27700", nodata=-1)
    with pytest.raises(transforms.TransformUnavailable):  # terrio alone: a ballpark, 100 m off
        transforms.warp(terrio, r, t_srs="EPSG:4326", resampling="bilinear")
    with pytest.raises(transforms.TransformUnavailable):
        transforms.warp(terrio, r, t_srs="EPSG:4326", te=(-0.13, 51.50, -0.12, 51.51), src_crs="EPSG:27700")
    utm = terrio.Raster.from_array(a, (430000.0, 10.0, 0.0, 4554000.0, 0.0, -10.0), crs="EPSG:32633", nodata=-1)
    out = transforms.warp(terrio, utm, t_srs="EPSG:4326", src_crs="EPSG:32633")
    assert out.width > 0


class _OldTerrio:
    """A terrio before 0.6 (no ``only_best``) whose transformation is ``off`` metres east."""

    __version__ = "0.5.0"

    def __init__(self, off):
        self.off = off

    def transform_points(self, xs, ys, src_crs, dst_crs, zs=None):
        from pyproj import Transformer

        x, y = Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform(np.asarray(xs), np.asarray(ys))
        return (np.asarray(x) + self.off, np.asarray(y))


def test_an_older_terrio_is_checked_against_proj():
    assert not transforms.has_only_best(_OldTerrio(0.0))
    good = transforms.transform_points(_OldTerrio(0.0), [430000.0], [4554000.0], "EPSG:32633", "EPSG:32632")
    assert np.isfinite(good[0]).all()
    with pytest.raises(transforms.TransformUnavailable):
        transforms.transform_points(_OldTerrio(120.0), [430000.0], [4554000.0], "EPSG:32633", "EPSG:32632")
    with pytest.raises(transforms.TransformUnavailable):
        transforms.check_best(_OldTerrio(120.0), (430000, 4554000, 431000, 4555000), "EPSG:32633", "EPSG:32632")
