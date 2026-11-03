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
    assert "piano del muro" in loss["notes"]
    # a window tight around the hole (the hole is 44 % of it): the wall's plane is still the plane of the rest
    tight = detect.material_loss(ch, entry, bounds=hole.buffer(0.1).bounds)
    assert len(tight) == 1 and abs(tight[0]["depth_mm"] - 30.0) <= 3.0
    assert abs(tight[0]["geometry"].area - hole.area) <= 0.10 * hole.area


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
    assert "SAM 2.1" in found[0]["notes"]


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
    detect.sam_areas(ch, entry, bounds=(0, Z0, 1, Z0 + 1), segmenter=object())  # 1 m2: SAM chooses
    assert seen["resolution"] is None
    detect.sam_areas(ch, entry, resolution=0.002, thoroughness="alta", segmenter=object())  # asked
    assert seen["resolution"] == 0.002 and seen["thoroughness"] == "alta"


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
