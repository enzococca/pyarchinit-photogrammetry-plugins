"""The detectors on the synthetic face, against its truths: the groove found as one crack of its length and opening,
the hole as a missing part of its depth and area, the green patch as a "green" colour class, SAM's segments (the
program's fake SAM) with no term, the segments that look like the one named, nothing in a window with no surface,
and a crack on the joint between two drawn stones left out."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the detectors need the program")

import conservation_samples as S  # noqa: E402
from ppg_conservation import detect  # noqa: E402

Z0 = S.Z0


@pytest.fixture
def face(tmp_path):
    return S.face(tmp_path)


def _near(found, geom, distance=0.05):
    return [f for f in found if f["geometry"].distance(geom) < distance]


def test_cracks_find_the_groove(face):
    doc, ch, entry = face
    found = detect.cracks(ch, entry)
    assert len(found) == 1, [(f["geometry"].length, f["geometry"].centroid.wkt) for f in found]
    crack = found[0]
    truth = S.groove()
    assert crack["kind"] == "crack" and crack["code"] == "ISCS.CRACK"
    assert crack["geometry"].hausdorff_distance(truth) < 0.03
    assert abs(crack["geometry"].length - truth.length) <= 0.05 * truth.length
    assert abs(crack["width_mm"] - S.GROOVE_WIDTH * 1000) <= 1.0
    assert "lesione proposta" in crack["notes"] and "giunti" in crack["notes"]
    assert 0.9 < crack["coverage"] < 1.0
    # a window around the groove only: the same line
    u0, v0, u1, v1 = truth.bounds
    near = detect.cracks(ch, entry, bounds=(u0 - 0.2, v0 - 0.2, u1 + 0.2, v1 + 0.2))
    assert len(near) == 1 and near[0]["geometry"].hausdorff_distance(truth) < 0.03
    assert near[0]["coverage"] == 1.0


def test_a_wider_crack_gives_its_opening(face):
    """A crack 3 px (3 cm) wide and 5 mm deep, painted dark and carved, on a clean part of the face."""
    import rasterio
    from shapely.geometry import LineString

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    line = LineString([(3.25, Z0 + 0.8), (3.65, Z0 + 1.7)])
    width, depth = 0.03, 0.005
    band = line.buffer(width / 2, cap_style="flat")
    ortho, relief = elevation_paths(ch, entry)
    for path in (ortho, relief):
        with rasterio.open(path, "r+") as dst:
            u, v = S._cells(dst)
            m = S._inside(u, v, band)
            if path == ortho:
                rgb = dst.read()
                rgb[:3, m] = np.array(S.CRACK_RGB, np.uint8)[:, None]
                dst.write(rgb)
            else:
                d = dst.read(1)
                d[m] -= depth
                dst.write(d, 1)
    u0, v0, u1, v1 = line.bounds
    # wider than the default maximum (a joint on a stone wall): kept only when the maximum is lifted
    found = detect.cracks(ch, entry, bounds=(u0 - 0.2, v0 - 0.2, u1 + 0.2, v1 + 0.2), max_width_mm=None)
    assert len(found) == 1
    assert abs(found[0]["geometry"].length - line.length) <= 0.05 * line.length
    assert abs(found[0]["width_mm"] - width * 1000) <= 1.0, found[0]["width_mm"]


def test_a_crack_on_a_joint_between_drawn_stones_is_dropped(face):
    doc, ch, entry = face
    line = S.groove()
    above = line.buffer(0.25, single_sided=True).difference(line.buffer(0.005))
    below = line.buffer(-0.25, single_sided=True).difference(line.buffer(0.005))
    eid = entry["id"]
    ch.meta.setdefault("drawings", {}).setdefault("frames", {}).setdefault(eid, {})["stones"] = {
        eid: [above.wkt, below.wkt]}
    said = []
    found = detect.cracks(ch, entry, progress=lambda f, m="": said.append(m))
    assert not _near(found, line)
    assert any("1 linee sui giunti" in m for m in said)


def test_material_loss_finds_the_hole(face):
    doc, ch, entry = face
    found = detect.material_loss(ch, entry)
    hole = S.box(S.HOLE)
    assert len(found) == 1
    loss = found[0]
    assert loss["kind"] == "decay" and loss["code"] == "ISCS.MISSING_PART"
    assert abs(loss["depth_mm"] - S.HOLE_DEPTH * 1000) <= 3.0
    assert abs(loss["geometry"].area - hole.area) <= 0.10 * hole.area
    assert loss["geometry"].intersection(hole).area >= 0.9 * hole.area
    assert "muro intorno" in loss["notes"]
    # a window tight around the hole (the hole is 44 % of it): the wall's plane is still the plane of the rest
    tight = detect.material_loss(ch, entry, bounds=hole.buffer(0.1).bounds)
    assert len(tight) == 1 and abs(tight[0]["depth_mm"] - 30.0) <= 3.0
    assert abs(tight[0]["geometry"].area - hole.area) <= 0.10 * hole.area


def _bulge(chunk, entry, height: float = 0.2) -> None:
    """A bulge of the whole face: ``height`` m at the middle, nothing at the edges (a product of two parabolas: no
    surface of the second order fits it)."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    _o, relief = elevation_paths(chunk, entry)
    with rasterio.open(relief, "r+") as dst:
        u, v = S._cells(dst)
        su = (u - S.WALL_LENGTH / 2) / (S.WALL_LENGTH / 2)
        sv = (v - Z0 - S.WALL_HEIGHT / 2) / (S.WALL_HEIGHT / 2)
        d = dst.read(1)
        d += (height * (1 - su ** 2) * (1 - sv ** 2)).astype(np.float32)
        dst.write(d, 1)


def test_material_loss_on_a_bulging_face(face):
    """The wall bulges by 20 cm over the window: only the hole is proposed, with its depth against the wall around
    it, not against a plane of the whole window."""
    doc, ch, entry = face
    _bulge(ch, entry)
    found = detect.material_loss(ch, entry)
    hole = S.box(S.HOLE)
    assert len(found) == 1, [(round(f["geometry"].area, 3), f["depth_mm"]) for f in found]
    assert found[0]["code"] == "ISCS.MISSING_PART"
    assert abs(found[0]["depth_mm"] - S.HOLE_DEPTH * 1000) <= 3.0, found[0]["depth_mm"]
    assert found[0]["geometry"].intersection(hole).area >= 0.9 * hole.area


def test_a_loss_too_big_for_its_reference_gets_no_term(face):
    """A depression of 2 x 1 m, 5 cm deep: no term and the note "da verificare"."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    _o, relief = elevation_paths(ch, entry)
    sink = S.box((3.3, Z0 + 0.3, 5.3, Z0 + 1.3))
    with rasterio.open(relief, "r+") as dst:
        u, v = S._cells(dst)
        d = dst.read(1)
        d[S._inside(u, v, sink)] -= 0.05
        dst.write(d, 1)
    found = detect.material_loss(ch, entry)
    big = [f for f in found if f["geometry"].intersects(sink)]
    assert len(big) == 1 and big[0]["code"] is None and "da verificare" in big[0]["notes"]
    assert any(f["code"] == "ISCS.MISSING_PART" and f["geometry"].intersects(S.box(S.HOLE)) for f in found)


def test_wide_dark_bands_are_joints_not_cracks(face):
    """A dark band 4 cm wide (a mortar joint) is dropped and counted; the groove (1 cm) stays."""
    import rasterio
    from shapely.geometry import LineString

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    band = LineString([(3.3, Z0 + 1.6), (5.6, Z0 + 1.6)]).buffer(0.02, cap_style="flat")
    ortho, _relief = elevation_paths(ch, entry)
    with rasterio.open(ortho, "r+") as dst:
        u, v = S._cells(dst)
        m = S._inside(u, v, band)
        rgb = dst.read()
        rgb[:3, m] = np.array(S.CRACK_RGB, np.uint8)[:, None]
        dst.write(rgb)
    stats = {}
    found = detect.cracks(ch, entry, stats=stats)
    assert len(found) == 1 and found[0]["geometry"].hausdorff_distance(S.groove()) < 0.03
    assert stats["wide"] >= 1 and stats["stones"] is False
    assert "più larghe di 12 mm" in found[0]["notes"]
    lifted = detect.cracks(ch, entry, max_width_mm=None)
    assert any(f["geometry"].distance(band) < 0.01 for f in lifted)


def test_a_big_window_by_tiles(face):
    """With a budget of 30 000 pixels the face (180 000) runs in 15 tiles, whose edges cross the hole and the groove:
    the same hole, of the same depth and area, and the groove as one line."""
    doc, ch, entry = face
    stats = {}
    found = detect.tiled(detect.material_loss, ch, entry, budget=30_000)
    hole = S.box(S.HOLE)
    assert len(found) == 1 and abs(found[0]["depth_mm"] - 30.0) <= 3.0
    assert abs(found[0]["geometry"].area - hole.area) <= 0.10 * hole.area
    assert len(detect.tiles(detect.window_extent(ch, entry)[0], 0.01, 400_000)) > 1
    cracks = detect.tiled(detect.cracks, ch, entry, budget=30_000, stats=stats)
    assert stats["tiles"] > 1
    assert len(cracks) == 1, [c["geometry"].wkt[:80] for c in cracks]
    assert abs(cracks[0]["geometry"].length - S.groove().length) <= 0.05 * S.groove().length
    assert abs(cracks[0]["width_mm"] - S.GROOVE_WIDTH * 1000) <= 1.0


def test_alveolization_from_many_small_holes(face):
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    _o, relief = elevation_paths(ch, entry)
    with rasterio.open(relief, "r+") as dst:
        d = dst.read(1)
        t = dst.transform
        for k in range(6):  # six cavities of 12 x 12 cm, 2.5 cm deep, in a row low on the wall
            c0 = int((0.3 + 0.25 * k - t.c) / t.a)
            r0 = int((Z0 + 2.9 - t.f) / t.e)
            d[r0:r0 + 12, c0:c0 + 12] -= 0.025
        dst.write(d, 1)
    found = detect.material_loss(ch, entry, bounds=(0.0, Z0 + 2.6, 2.0, Z0 + 3.0))
    assert len(found) == 6 and all(f["code"] == "ISCS.ALVEOLIZATION" for f in found)
    assert all(abs(f["depth_mm"] - 25) <= 3 for f in found)


def test_colour_classes_find_the_green_patch(face):
    doc, ch, entry = face
    classes = detect.colour_classes(ch, entry)
    assert len(classes) == 5
    green = [c for c in classes if c["looks"] == "green"]
    assert len(green) == 1
    from shapely.ops import unary_union

    patch = S.box(S.GREEN_PATCH)
    union = unary_union(green[0]["polygons"])
    assert union.intersection(patch).area >= 0.95 * patch.area
    assert abs(union.area - patch.area) <= 0.05 * patch.area
    assert green[0]["suggest"] == ["ISCS.BIOLOGICAL_COLONIZATION"]
    looks = {c["looks"] for c in classes}
    assert {"bright", "dark"} <= looks
    assert abs(sum(c["share"] for c in classes) - 1.0) < 1e-3


@pytest.fixture
def fake_sam(monkeypatch):
    monkeypatch.setenv("PPG_SAM_FAKE", "1")
    from pyarchinit_photogrammetry.ai.sam_env import ExternalSam2Segmenter

    seg = ExternalSam2Segmenter(python=sys.executable)
    yield seg
    seg._proc.kill()


def test_sam_areas_with_the_fake_sam(face, fake_sam):
    doc, ch, entry = face
    window = (0.5, Z0 + 0.5, 2.5, Z0 + 2.5)
    found = detect.sam_areas(ch, entry, bounds=window, segmenter=fake_sam)
    assert found and all(f["code"] is None and f["kind"] == "decay" for f in found)
    from shapely.geometry import box

    assert all(box(*window).buffer(0.02).contains(f["geometry"]) for f in found)
    assert "SAM 2.1" in found[0]["notes"] and "pixel 10,0 mm" in found[0]["notes"]


def test_sam_on_a_big_face_is_coarser_and_normal(face, monkeypatch):
    import pyarchinit_photogrammetry.ai.sam as sam

    doc, ch, entry = face
    seen = {}

    def spy(chunk, segmenter, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(sam, "auto_units", spy)
    detect.sam_areas(ch, entry, segmenter=object())  # 18 m2
    assert seen["resolution"] == max(entry["resolution"], detect.BIG_FACE_RESOLUTION)
    assert seen["thoroughness"] == "normale"
    detect.sam_areas(ch, entry, bounds=(0, Z0, 1, Z0 + 1), segmenter=object())  # 1 m2: as auto_units chooses
    assert seen["resolution"] == entry["resolution"]  # 1 m on 400 tiles: finer than the raster, so the raster's
    detect.sam_areas(ch, entry, resolution=0.002, thoroughness="alta", segmenter=object())  # asked: never finer
    assert seen["resolution"] == entry["resolution"] and seen["thoroughness"] == "alta"
    detect.sam_areas(ch, entry, resolution=0.02, thoroughness="alta", segmenter=object())  # asked
    assert seen["resolution"] == 0.02 and seen["thoroughness"] == "alta"


def test_like_this_picks_the_same_colour(face):
    doc, ch, entry = face
    b, g, w = S.BLACK_CRUST, S.GREEN_PATCH, S.EFFLORESCENCE
    segments = [S.box((g[0] + 0.05, g[1] + 0.05, 4.65, g[3] - 0.05)),           # 0 green
                S.box((b[0] + 0.05, b[1] + 0.05, 0.95, b[3] - 0.05)),           # 1 black
                S.box((4.75, g[1] + 0.05, g[2] - 0.05, g[3] - 0.05)),           # 2 green
                S.box((w[0] + 0.05, w[1] + 0.05, w[2] - 0.05, w[3] - 0.05)),    # 3 white
                S.box((3.2, Z0 + 0.2, 3.8, Z0 + 0.8)),                          # 4 stone
                S.box((1.05, b[1] + 0.05, b[2] - 0.05, b[3] - 0.05)),           # 5 black
                S.box((3.2, Z0 + 1.2, 3.8, Z0 + 1.7))]                          # 6 stone
    reference = S.box((g[0] + 0.3, g[1] + 0.1, g[0] + 0.6, g[1] + 0.4))
    assert detect.like_this(ch, entry, segments, reference) == [0, 2]
    black = S.box((b[0] + 0.2, b[1] + 0.2, b[0] + 0.5, b[1] + 0.5))
    assert detect.like_this(ch, entry, segments, black) == [1, 5]


def test_like_this_reads_the_window_once(face, monkeypatch):
    """The per-pixel quantities are computed once per call, whatever the number of segments, and the segments are
    described from one index image."""
    doc, ch, entry = face
    calls = {"images": 0, "regions": []}
    images, descriptors = detect._feature_images, detect._descriptors

    def count_images(rgb):
        calls["images"] += 1
        return images(rgb)

    def count_regions(imgs, index, n):
        calls["regions"].append(n)
        return descriptors(imgs, index, n)

    monkeypatch.setattr(detect, "_feature_images", count_images)
    monkeypatch.setattr(detect, "_descriptors", count_regions)
    g = S.GREEN_PATCH
    segments = [S.box((g[0] + 0.2 * k + 0.01, g[1] + 0.05, g[0] + 0.2 * k + 0.19, g[1] + 0.5)) for k in range(5)]
    found = detect.like_this(ch, entry, segments, S.box((g[0] + 0.3, g[1] + 0.1, g[0] + 0.6, g[1] + 0.4)))
    assert calls["images"] == 1 and calls["regions"] == [1, 5]
    assert found == list(range(5))


def test_a_window_with_no_surface_gives_nothing(face, fake_sam):
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    outside = (7.0, Z0, 8.0, Z0 + 1.0)  # beyond the rasters
    empty = (3.2, Z0 + 1.0, 3.9, Z0 + 1.6)  # made transparent and with no relief here
    ortho, relief = elevation_paths(ch, entry)
    for path, band, value in ((ortho, 4, 0), (relief, 1, np.nan)):
        with rasterio.open(path, "r+") as dst:
            t = dst.transform
            c0, c1 = int((empty[0] - t.c) / t.a) - 2, int((empty[2] - t.c) / t.a) + 2
            r0, r1 = int((empty[3] - t.f) / t.e) - 2, int((empty[1] - t.f) / t.e) + 2
            a = dst.read(band)
            a[r0:r1, c0:c1] = value
            dst.write(a, band)
    for window in (outside, empty):
        assert detect.cracks(ch, entry, bounds=window) == []
        assert detect.material_loss(ch, entry, bounds=window) == []
        assert detect.colour_classes(ch, entry, bounds=window) == []
        assert detect.sam_areas(ch, entry, bounds=window, segmenter=fake_sam) == []
    # with no relief but an opaque image the image still speaks: the coverage says 0
    nothing = S.box(S.NO_RELIEF).bounds
    classes = detect.colour_classes(ch, entry, bounds=nothing)
    assert classes and classes[0]["coverage"] == 0.0
    assert detect.material_loss(ch, entry, bounds=nothing) == []
    assert os.environ.get("PPG_SAM_FAKE") == "1"
