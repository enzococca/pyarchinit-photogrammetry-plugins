"""Transformations through terrio never fall back in silence: from terrio 0.6 a transformation
whose grid is missing goes on with a rougher one (even a ballpark: about 100 m between OSGB36 and
WGS 84) with only a warning; the plugin asks for the best one or says why it cannot (no network:
the grid of OSTN15 is not on the computer), in terrio's own words when it gave any: from 0.9 terrio
says in a warning which grid it did not read and why, the error being no more than "cannot transform"."""
import struct
import sys
import warnings

import numpy as np
import pytest

terrio = pytest.importorskip("terrio")

from ppg_terrio import transforms  # noqa: E402

LONDON = (530000.0, 180000.0)  # British National Grid (EPSG:27700): OSTN15 grid, else a Helmert or a ballpark
BY_PATH = ("grid named by a path is not read: grid search paths are set, so grids are read only from them "
           "(/griglie/geoide.gtx)")  # what terrio 0.9 says of such a grid
NO_POINT = "coordinate reference system: cannot transform (36.5, 36, 100)"  # and its error then
BALLPARK = "ballpark coordinate operation used between different ellipsoids; datum shift is unknown"


@pytest.fixture
def no_grids(tmp_path):
    terrio.set_grid_search_paths([str(tmp_path)])
    terrio.set_network_enabled(False)
    yield
    terrio.set_grid_search_paths(None)
    terrio.set_network_enabled(None)


needs_only_best = pytest.mark.skipif(not transforms.has_only_best(terrio), reason="only_best comes with terrio 0.6")


def _seen(recorded):
    """The warnings recorded around a step, each with its category, file and line (without those of objects the
    interpreter clears meanwhile, which come when they come)."""
    return [(str(w.message), w.category, w.filename, w.lineno) for w in recorded
            if not issubclass(w.category, ResourceWarning)]


def _texts(recorded):
    return [w[0] for w in _seen(recorded)]


@needs_only_best
def test_a_missing_grid_is_an_error_not_a_rougher_transformation(no_grids):
    def refused():
        with pytest.raises(transforms.TransformUnavailable, match="EPSG:27700") as exc:
            transforms.transform_points(terrio, [LONDON[0]], [LONDON[1]], "EPSG:27700", "EPSG:4326", what="Prova")
        return str(exc.value)

    with warnings.catch_warnings(record=True) as let_out:
        warnings.simplefilter("always")
        first, second = refused(), refused()
    assert "needs a missing grid" in first, "the reason in terrio's own words"
    assert second == first, "and the same at the second refusal"
    assert not _texts(let_out), "what terrio warned of is in the message: not on the console as well"
    # what the plugin no longer accepts: terrio alone goes on, warning only (at every call, 0.6.1 as 0.9.0)
    for _ in range(2):
        with pytest.warns(UserWarning, match="missing grid"):
            terrio.transform_points([LONDON[0]], [LONDON[1]], "EPSG:27700", "EPSG:4326")
    # a transformation without grids is not touched
    lon, lat = transforms.transform_points(terrio, [430000.0], [4554000.0], "EPSG:32633", "EPSG:4326")[:2]
    assert 14 < lon[0] < 15 and 41 < lat[0] < 42


@needs_only_best
def test_a_warp_never_takes_a_ballpark(no_grids):
    """terrio's error of a warp says only that the points failed to transform: why is in its warning."""
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    r = terrio.Raster.from_array(a, (LONDON[0], 10.0, 0.0, LONDON[1], 0.0, -10.0), crs="EPSG:27700", nodata=-1)
    with warnings.catch_warnings(record=True) as let_out:
        warnings.simplefilter("always")
        with pytest.raises(transforms.TransformUnavailable, match="needs a missing grid"):  # terrio alone: 100 m off
            transforms.warp(terrio, r, t_srs="EPSG:4326", resampling="bilinear")
        with pytest.raises(transforms.TransformUnavailable, match="needs a missing grid"):
            transforms.warp(terrio, r, t_srs="EPSG:4326", te=(-0.13, 51.50, -0.12, 51.51), src_crs="EPSG:27700")
    assert not _texts(let_out)
    utm = terrio.Raster.from_array(a, (430000.0, 10.0, 0.0, 4554000.0, 0.0, -10.0), crs="EPSG:32633", nodata=-1)
    out = transforms.warp(terrio, utm, t_srs="EPSG:4326", src_crs="EPSG:32633")
    assert out.width > 0


@needs_only_best
def test_a_grid_terrio_does_not_read_is_named_with_the_reason(no_grids, tmp_path):
    """With the folders of grids set, terrio 0.9 reads grids from them only: of a grid named by its path it says so
    in a warning, and its error is no more than "cannot transform". The refusal holds both, the same every time.
    terrio 0.6 reads such a grid: nothing is refused. What terrio does by itself decides which of the two is checked."""
    grid = tmp_path / "geoide" / "locale.gtx"  # 20 m everywhere
    grid.parent.mkdir()
    grid.write_bytes(struct.pack(">ddddii", 35.0, 35.0, 0.25, 0.25, 9, 9) + np.full((9, 9), 20.0, ">f4").tobytes())
    point = ([36.5], [36.0], "EPSG:4979", f"+proj=longlat +datum=WGS84 +geoidgrids={grid} +vunits=m +no_defs")
    with warnings.catch_warnings(record=True) as alone:
        warnings.simplefilter("always")
        try:
            read = terrio.transform_points(*point, zs=[100.0], only_best=True)
        except ValueError as exc:
            read = None
            assert "cannot transform" in str(exc) and "grid" not in str(exc)
    with warnings.catch_warnings(record=True) as let_out:
        warnings.simplefilter("always")
        if read is not None:
            assert not alone and read[2][0] == pytest.approx(80.0)
            assert transforms.transform_points(terrio, *point, zs=[100.0])[2][0] == pytest.approx(80.0)
        else:
            assert alone and all("grid" in str(w.message) for w in alone), "terrio says which grid and why"
            refusals = []
            for _ in range(2):
                with pytest.raises(transforms.TransformUnavailable, match="cannot transform") as exc:
                    transforms.transform_points(terrio, *point, zs=[100.0], what="Le quote")
                refusals.append(exc.value)
            message = str(refusals[0])
            assert all(str(w.message) in message for w in alone) and str(grid) in message
            assert refusals[0].reason and refusals[0].reason in message
            assert str(refusals[1]) == message
    assert not _texts(let_out)


class _Terrio:
    """A terrio with ``only_best`` that warns of ``said`` and then raises ``error`` as a ``kind`` (returns the
    points as they are when there is none); ``line``: where it warned."""

    __version__ = "0.9.0"

    def __init__(self, said=(), error="", kind=ValueError):
        self.said, self.error, self.kind, self.line = said, error, kind, 0

    def transform_points(self, xs, ys, src_crs, dst_crs, zs=None, only_best=False):
        for w in self.said:
            self.line = sys._getframe().f_lineno + 1
            warnings.warn(w, stacklevel=1)
        if self.error:
            raise self.kind(self.error)
        return tuple(np.asarray(v, np.float64) for v in ((xs, ys) if zs is None else (xs, ys, zs)))


def _heights(fake, what="Le quote"):
    return transforms.transform_points(fake, [36.5], [36.0], "EPSG:4979", "EPSG:4326+3855", zs=[100.0], what=what)


def _from(fake, *said):
    """What ``fake`` warned of, as :func:`_seen` gives it: each of ``said`` with its category, this file, its line."""
    return [(str(w), type(w) if isinstance(w, Warning) else UserWarning, __file__, fake.line) for w in said]


def test_what_terrio_says_of_a_grid_is_in_the_refusal():
    """The error names no grid, the warning does: a refusal of the plugin with both, a path inside it kept as it is
    (the message is read on the user's own computer), the same at the second refusal, nothing on the console."""
    fake = _Terrio((BY_PATH,), NO_POINT)
    refusals = []
    with warnings.catch_warnings(record=True) as let_out:
        warnings.simplefilter("always")
        for _ in range(2):
            with pytest.raises(transforms.TransformUnavailable) as exc:
                _heights(fake)
            refusals.append(exc.value)
    message = str(refusals[0])
    assert message.startswith("Le quote: la trasformazione migliore da EPSG:4979 a EPSG:4326+3855 non si può usare")
    assert message.endswith(f"({NO_POINT}; {BY_PATH})")
    assert refusals[0].reason == BY_PATH and str(refusals[1]) == message
    assert isinstance(refusals[0].__cause__, ValueError) and not _texts(let_out)


def test_warnings_that_no_message_uses_are_issued_again():
    """What is heard around a call of terrio and goes into no message is issued again with its category, file and
    line: of a transformation that went well, of an error that is no refusal of a grid (a refusal for memory is
    never one, whatever was said of a grid meanwhile), and - beside a refusal - whatever is not a word of terrio
    about it (a DeprecationWarning)."""
    old = DeprecationWarning("zs will be required")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fake = _Terrio((BALLPARK, old))
        assert _heights(fake)[2][0] == 100.0
    assert _seen(seen) == _from(fake, BALLPARK, old)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fake = _Terrio((BALLPARK, old), "coordinate reference system: cannot transform (NaN, 36, 1)")
        with pytest.raises(ValueError, match="NaN") as exc:
            _heights(fake)
    assert not isinstance(exc.value, transforms.TransformUnavailable), "terrio's own error, as it is"
    assert _seen(seen) == _from(fake, BALLPARK, old)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fake = _Terrio((old, BY_PATH), NO_POINT)
        with pytest.raises(transforms.TransformUnavailable, match="named by a path"):
            _heights(fake)
    assert _seen(seen) == _from(fake, old)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fake = _Terrio((BY_PATH,), "resource limit: the points would take 9 GiB", MemoryError)
        with pytest.raises(MemoryError, match="9 GiB"):
            _heights(fake)
    assert _seen(seen) == _from(fake, BY_PATH)


def test_the_words_of_terrio_are_one_line_of_reasonable_length():
    long = "grid file is not read:\n    larger than 2147483648 bytes (" + "/cartella" * 100 + "/geoide.tif)"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(transforms.TransformUnavailable) as exc:
            _heights(_Terrio((long, long, "grid download failed"), NO_POINT))
    reason = exc.value.reason
    assert reason.startswith("grid file is not read: larger than 2147483648 bytes (/cartella/cartella")
    assert len(reason) == transforms.REASON_LENGTH and reason.endswith("...") and "\n" not in str(exc.value)
    assert str(exc.value).endswith(f"({NO_POINT}; {reason})")
    assert _texts(seen) == ["grid download failed"], "what found no room is not swallowed"
    with pytest.raises(transforms.TransformUnavailable) as exc:  # each sentence once, one after the other
        _heights(_Terrio(("grid a.tif is not read", "grid a.tif is not read", "grid b.tif is not read"), NO_POINT))
    assert exc.value.reason == "grid a.tif is not read; grid b.tif is not read"


class _OldTerrio:
    """A terrio before 0.6 (no ``only_best``) whose transformation is ``off`` metres east and warns of ``said``."""

    __version__ = "0.5.0"

    def __init__(self, off, said=()):
        self.off, self.said = off, said

    def transform_points(self, xs, ys, src_crs, dst_crs, zs=None):
        from pyproj import Transformer

        for w in self.said:
            warnings.warn(w, stacklevel=1)
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
    # what it warned of: in the refusal when its result is refused, issued again when it is taken
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(transforms.TransformUnavailable, match="datum shift is unknown") as exc:
            transforms.transform_points(_OldTerrio(120.0, (BALLPARK,)), [430000.0], [4554000.0], "EPSG:32633",
                                        "EPSG:32632")
        assert exc.value.reason == BALLPARK and not _texts(seen)
        transforms.transform_points(_OldTerrio(0.0, (BALLPARK,)), [430000.0], [4554000.0], "EPSG:32633", "EPSG:32632")
    assert _texts(seen) == [BALLPARK]
