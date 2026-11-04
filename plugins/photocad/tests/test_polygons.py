"""PhotoCAD draws stones as open polylines (Khutm: 1993 entities, 6 closed): the lines of a conversion noded, their
near-touching ends snapped (2 px of the image), polygonized; faces under the minimum area and faces of background
(white) dropped; kept in ``polygons.gpkg`` and as a second layer; *Converti in US / USM* and the command line take
them by default."""
from __future__ import annotations

import fakecad
import numpy as np
import pytest

from ppg_photocad import client, project

PX = fakecad.RES / 2


@pytest.fixture
def open_fake(monkeypatch):
    f = fakecad.FakePhotoCAD(credits=9, open_lines=True, ring=True, tiny=True)
    monkeypatch.setattr(client, "_urlopen", f)
    return f


def test_open_lines_become_stone_polygons(tmp_path, open_fake):
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    rec = project.convert(ch, "ortho", fakecad.KEY)
    lines = project.features(ch, rec)
    assert not [g for g, _a in lines if g.geom_type == "Polygon"]  # nothing closed in what PhotoCAD gave
    stones = project.stone_features(ch, rec)
    assert len(stones) == 2 and rec["counts"]["stones"] == 2  # the background ring and the tiny loop dropped
    boxes = sorted(tuple(g.bounds) for g, _a in stones)
    truth = sorted([fakecad.square_world(), fakecad.square_world(fakecad.SQUARE2)])
    for b, tr in zip(boxes, truth):
        assert np.abs(np.asarray(b) - tr).max() <= 1.5 * PX
    assert all(a["area_m2"] > 0 and a["source"] == "polygonize" for _g, a in stones)
    assert (project.folder_of(ch, rec) / rec["files"]["polygons"]).name == "polygons.gpkg"
    dropped = rec["polygonize"]
    assert dropped["small"] >= 1 and dropped["background"] >= 1 and dropped["snap_m"] == pytest.approx(2 * PX)


def test_units_from_the_stones_by_default(tmp_path, open_fake):
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    rec = project.convert(ch, "ortho", fakecad.KEY)
    polygons, attrs, skipped = project.unit_polygons(project.unit_features(ch, rec), tipo="USM", start=1)
    assert len(polygons) == 2 and skipped == 0 and [a["us"] for a in attrs] == ["1", "2"]
    polygons, _a, skipped = project.unit_polygons(project.unit_features(ch, rec, "lines"), tipo="US")
    assert polygons == [] and skipped > 0
    # a conversion with closed outlines and no stones: the lines, as before
    fake = fakecad.FakePhotoCAD(credits=9)
    client._urlopen = fake
    rec2 = project.convert(ch, "ortho", fakecad.KEY)
    assert len(project.unit_polygons(project.unit_features(ch, rec2), "US")[0]) == 2


def test_command_line_units_from(tmp_path, open_fake, monkeypatch, capsys):
    import pyarchinit_photogrammetry as ppg

    from ppg_photocad import PhotoCADPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    app.document = None
    plugin = PhotoCADPlugin()
    plugin.activate(Context("photocad", tmp_path / "plugin"))
    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    doc, ch = fakecad.project(tmp_path / "p", square=[fakecad.SQUARE, fakecad.SQUARE2])
    path = str(doc.path)
    assert plugin.command("convert", path, "--source", "ortho", "--yes") == 0
    rid = project.records(ppg.Document.open(path).chunk)[0]["id"]
    assert plugin.command("units", path, rid, "--from", "lines") == 0 and "0 unità" in capsys.readouterr().out
    assert plugin.command("units", path, rid, "--tipo", "USM") == 0
    assert len(ppg.Document.open(path).chunk.meta["drawings"]["units"]) == 2
    assert plugin.command("polygonize", path, rid, "--snap-px", "3") == 0
    assert "2" in capsys.readouterr().out


def test_lines_chained_end_to_end_are_noded(tmp_path, monkeypatch):
    """At Khutm the polylines chain exactly end to start; after the placement their ends differ by 1e-16 and were
    not noded (3 faces out of 1993 lines): the lines go on a grid first."""
    fake = fakecad.FakePhotoCAD(open_lines=True, gap_px=0.0)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    rec = project.convert(ch, "ortho", fakecad.KEY)
    assert rec["counts"]["stones"] == 2 and rec["polygonize"]["connectors"] == 0
    rec = project.record(ch, rec["id"])
    assert project.polygonize(ch, rec, snap_px=0.0) == 2


def test_joint_slivers_are_counted_and_left_out():
    from shapely.geometry import box

    assert project.is_sliver(box(0, 0, 10, 0.1)) and not project.is_sliver(box(0, 0, 10, 4))
