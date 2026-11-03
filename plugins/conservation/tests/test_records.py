"""Campaigns and features on the synthetic elevation: dates, extents, the checks of a feature, units linked, totals of
a campaign, the world geometry of a feature (also where the relief has no value), photos, save and reopen, a campaign
removed with its files."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the records live on the program's elevations")

import conservation_samples as S  # noqa: E402
from ppg_conservation import geometry, records  # noqa: E402
from ppg_conservation.records import NotReady  # noqa: E402

Z0 = S.Z0


@pytest.fixture
def face(tmp_path):
    return S.face(tmp_path)


def test_campaigns_of_a_face(face):
    doc, ch, entry = face
    eid = entry["id"]
    c = records.new_campaign(ch, eid, surveyor="E. Cocca", method="rilievo fotogrammetrico")
    assert c["date"] == "2026-05-14" and c["name"] == "Campagna 2026-05-14" and c["elevation"] == eid
    assert len(c["id"]) == 8 and c["classes"] == {"cc": None, "uc": None, "reason": ""} and c["made"]
    assert records.store(ch)["features"] == [] and records.campaigns(ch, eid) == [c]
    assert records.campaigns(ch, "other") == [] and records.campaign(ch, c["id"]) is c
    records.set_campaign(ch, c["id"], weather="sereno", classes={"cc": 2, "reason": "lesioni"})
    assert c["weather"] == "sereno" and c["classes"] == {"cc": 2, "uc": None, "reason": "lesioni"}
    with pytest.raises(ValueError):
        records.set_campaign(ch, c["id"], classes={"cc": 5})
    with pytest.raises(ValueError):
        records.set_campaign(ch, c["id"], elevation="x")
    with pytest.raises(NotReady):
        records.campaign(ch, "nothing")
    with pytest.raises(NotReady):
        records.new_campaign(ch, "nothing")
    ch.meta.pop("date")  # no date and no photos: today
    c2 = records.new_campaign(ch, eid, name="Seconda")
    assert c2["date"] == date.today().isoformat() and c2["name"] == "Seconda"
    assert records.assessment(ch, "2026-05-14") is None
    records.set_assessment(ch, "2026-05-14", 2, reason="riparazione", surveyor="E. Cocca")
    assert records.assessment(ch, "2026-05-14") == {"rc": 2, "reason": "riparazione", "surveyor": "E. Cocca"}
    with pytest.raises(ValueError):
        records.set_assessment(ch, "2026-05-14", 4)


def _painted_area(ch, entry, bounds, dark=60):
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    with rasterio.open(elevation_paths(ch, entry)[0]) as src:
        r = src.read(1)
        u, v = S._cells(src)
    u0, v0, u1, v1 = bounds
    near = (u > u0 - 0.05) & (u < u1 + 0.05) & (v > v0 - 0.05) & (v < v1 + 0.05)
    return float(((r < dark) & near).sum()) * entry["resolution"] ** 2


def test_features_and_their_extent(face):
    doc, ch, entry = face
    c = records.new_campaign(ch, entry["id"])
    crust = records.add_feature(ch, c["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST),
                                intensity="high")
    assert abs(crust["extent"] - _painted_area(ch, entry, S.BLACK_CRUST)) / crust["extent"] < 0.01
    assert crust["code_uni"] == "UNI.CROSTA" and crust["geometry"].startswith("POLYGON")
    assert crust["status"] == "accepted" and crust["source"] == "drawn" and len(crust["id"]) == 8
    crack = records.add_feature(ch, c["id"], "crack", "ISCS.CRACK", S.groove().wkt, width_mm=3.0)
    assert crack["extent"] == pytest.approx(S.groove().length) and crack["code_uni"] == "UNI.FESSURAZIONE"
    note = records.add_feature(ch, c["id"], "note", "ISCS.CRUST", "POINT (3 313.5)", notes="campione")
    assert note["extent"] is None
    # the short code of a sub-type is stored as its full code
    assert records.add_feature(ch, c["id"], "decay", "ISCS.BLACK_CRUST",
                               S.box((0, Z0, 0.1, Z0 + 0.1)))["code"] == "ISCS.CRUST.BLACK_CRUST"
    with pytest.raises(ValueError, match="LINESTRING"):
        records.add_feature(ch, c["id"], "crack", "ISCS.CRACK", S.box(S.BLACK_CRUST))
    with pytest.raises(ValueError, match="crack"):
        records.add_feature(ch, c["id"], "decay", "ISCS.CRACK", S.box(S.BLACK_CRUST))
    with pytest.raises(ValueError, match="treatment"):
        records.add_feature(ch, c["id"], "decay", "INT.CLEANING", S.box(S.BLACK_CRUST))
    with pytest.raises(ValueError, match="POINT"):
        records.add_feature(ch, c["id"], "note", "ISCS.CRUST", S.box(S.BLACK_CRUST))
    with pytest.raises(ValueError):
        records.add_feature(ch, c["id"], "decay", "ISCS.CRUST", S.box(S.BLACK_CRUST), intensity="huge")
    with pytest.raises(KeyError):
        records.add_feature(ch, c["id"], "decay", "ISCS.NOTHING", S.box(S.BLACK_CRUST))
    with pytest.raises(NotReady):
        records.add_feature(ch, "nothing", "decay", "ISCS.CRUST", S.box(S.BLACK_CRUST))
    treat = records.add_feature(ch, c["id"], "treatment", "INT.CLEANING", S.box(S.BLACK_CRUST),
                                treatment={"technique": "INT.CLEANING", "materials": "impacchi", "done": False})
    assert treat["treatment"]["materials"] == "impacchi" and treat["code_uni"] is None
    # editing: the geometry changes the extent
    records.update_feature(ch, crust["id"], geometry=S.box((0.5, Z0 + 0.5, 1.0, Z0 + 1.3)), notes="ridotta")
    assert crust["extent"] == pytest.approx(0.4) and crust["notes"] == "ridotta"
    with pytest.raises(ValueError):
        records.update_feature(ch, crust["id"], geometry=S.groove())
    with pytest.raises(ValueError):
        records.update_feature(ch, crust["id"], id="x")
    assert len(records.features(ch, c["id"])) == 5 and len(records.features(ch, eid=entry["id"])) == 5
    prop = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH), status="proposed",
                               source="proposed:colour")
    assert records.features(ch, c["id"], status="proposed") == [prop]
    records.accept(ch, [prop["id"]])
    assert prop["status"] == "accepted" and prop["source"] == "proposed:colour"
    other = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH), status="proposed")
    records.discard(ch, [other["id"]])
    with pytest.raises(NotReady):
        records.feature(ch, other["id"])
    records.remove_feature(ch, note["id"])
    assert note["id"] not in [f["id"] for f in records.features(ch)]


def test_units_linked_to_a_feature(face):
    from shapely.geometry import box

    doc, ch, entry = face
    c = records.new_campaign(ch, entry["id"])
    f = records.add_feature(ch, c["id"], "decay", "ISCS.EROSION", box(1.0, Z0 + 1.0, 2.0, Z0 + 2.0))
    S.put_units(ch, entry["id"], {"10": box(1.5, Z0 + 0.5, 2.5, Z0 + 2.5),  # half of the feature
                                  "11": box(1.98, Z0 + 1.0, 3.0, Z0 + 2.0)})  # 2 %
    links = records.link_units(ch, entry, f["id"])
    assert links == [("1", "10", "US")] and f["links"]["us"] == [["1", "10", "US"]]
    crack = records.add_feature(ch, c["id"], "crack", "ISCS.CRACK", S.groove())
    assert records.link_units(ch, entry, crack["id"]) == []  # a line has no area


def test_totals_of_a_campaign(face):
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    doc, ch, entry = face
    c = records.new_campaign(ch, entry["id"])
    records.add_feature(ch, c["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH))
    records.add_feature(ch, c["id"], "decay", "ISCS.MISSING_PART", S.box(S.HOLE), depth_mm=30.0)
    records.add_feature(ch, c["id"], "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    records.add_feature(ch, c["id"], "crack", "ISCS.CRACK.HAIR_CRACK", "LINESTRING (3 313, 3.5 313)", width_mm=0.05)
    records.add_feature(ch, c["id"], "treatment", "INT.CLEANING", S.box(S.BLACK_CRUST))
    records.add_feature(ch, c["id"], "decay", "ISCS.EFFLORESCENCE", S.box(S.EFFLORESCENCE), status="proposed")
    tot = records.totals(ch, c["id"])
    assert tot["area_by_code"] == pytest.approx({"ISCS.CRUST.BLACK_CRUST": 0.8, "ISCS.ALGA": 0.6,
                                                 "ISCS.MISSING_PART": 0.16})
    assert tot["area_by_family"] == pytest.approx({"discoloration_deposit": 0.8, "biological": 0.6,
                                                   "material_loss": 0.16})
    assert tot["length_by_code"] == pytest.approx({"ISCS.CRACK": S.groove().length, "ISCS.CRACK.HAIR_CRACK": 0.5})
    with rasterio.open(elevation_paths(ch, entry)[1]) as src:
        face_area = float(np.isfinite(src.read(1)).sum()) * entry["resolution"] ** 2
    assert tot["face_area"] == pytest.approx(face_area)
    assert tot["share_of_face"] == pytest.approx(1.56 / face_area)
    assert tot["crack_max_width_mm"] == 3.0 and tot["loss_max_depth_mm"] == 30.0


def test_lift_on_the_relief(face):
    from shapely.geometry import box

    doc, ch, entry = face
    frame = geometry.frame_of(ch, entry)
    # over the hole: inner points lie 3 cm behind the plane, the outline on it
    hole = box(2.55, Z0 + 1.85, 2.85, Z0 + 2.15)
    verts, tris, on_plane = geometry.lift(ch, entry, hole, step=0.05)
    d = frame.coords(verts)[:, 2]
    assert on_plane == 0 and verts.shape[1] == 3 and tris.shape[1] == 3 and len(tris) > 0
    assert (d < -0.02).sum() >= 9 and d.min() == pytest.approx(-S.HOLE_DEPTH, abs=1e-4)
    uv = frame.coords(verts)[:, :2]
    a = uv[tris[:, 1]] - uv[tris[:, 0]]
    b = uv[tris[:, 2]] - uv[tris[:, 0]]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    assert (cross > 0).all()  # anticlockwise in (u, v): facing the viewer
    assert cross.sum() / 2 == pytest.approx(hole.area, rel=1e-9)
    # over the patch with no relief: the vertices go on the plane and are counted (review focus 1)
    inside = box(4.1, Z0 + 2.1, 4.7, Z0 + 2.5)
    verts, tris, on_plane = geometry.lift(ch, entry, inside)
    assert on_plane == len(verts) and np.allclose(frame.coords(verts)[:, 2], 0.0)
    across = box(3.5, Z0 + 2.1, 4.5, Z0 + 2.5)
    verts, tris, on_plane = geometry.lift(ch, entry, across)
    assert 0 < on_plane < len(verts)
    # a whole feature outside the elevation: on the plane, nothing raised
    verts, tris, on_plane = geometry.lift(ch, entry, box(10, Z0, 10.2, Z0 + 0.2))
    assert on_plane == len(verts)
    # a crack's band and a note's disc
    band = geometry.band(S.groove(), 0.001)
    assert band.area == pytest.approx(S.groove().length * 0.005, rel=1e-6)
    assert geometry.band(S.groove(), 0.02).area == pytest.approx(S.groove().length * 0.02, rel=1e-6)
    disc = geometry.disc(S.groove().interpolate(0.5))
    assert disc.area == pytest.approx(np.pi * 0.05 ** 2, rel=0.02)
    verts, tris, _ = geometry.lift(ch, entry, band)
    assert len(tris) and np.isfinite(verts).all()
    # a multipolygon with a hole of its own
    ring = box(0.2, Z0 + 0.2, 1.2, Z0 + 1.2).difference(box(0.5, Z0 + 0.5, 0.8, Z0 + 0.8))
    verts, tris, _ = geometry.lift(ch, entry, ring.union(box(5.0, Z0 + 2.8, 5.3, Z0 + 2.9)))
    uv = frame.coords(verts)[:, :2]
    a, b = uv[tris[:, 1]] - uv[tris[:, 0]], uv[tris[:, 2]] - uv[tris[:, 0]]
    assert (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2 == pytest.approx(ring.area + 0.03, rel=1e-9)


def test_photos_of_a_feature(face):
    doc, ch, entry = face
    c = records.new_campaign(ch, entry["id"])
    f = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH))
    assert records.photos_of(ch, entry, f) == []  # no aligned camera in the sample


def test_records_kept_and_removed(face, tmp_path):
    import pyarchinit_photogrammetry as ppg

    doc, ch, entry = face
    eid = entry["id"]
    c = records.new_campaign(ch, eid)
    keep = records.new_campaign(ch, eid, name="Altra")
    f = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH))
    g = records.add_feature(ch, keep["id"], "crack", "ISCS.CRACK", S.groove())
    records.set_assessment(ch, c["date"], 1)
    doc.save()
    again = ppg.Document.open(doc.path)
    ch2 = again.chunk
    assert records.campaigns(ch2, eid) == records.campaigns(ch, eid)
    assert records.feature(ch2, f["id"]) == f and records.assessment(ch2, c["date"])["rc"] == 1
    folder = records.campaign_folder(ch, eid, c["id"])
    assert folder == ch.data_dir / "conservation" / eid / c["id"]
    folder.mkdir(parents=True)
    (folder / "tavola.pdf").write_bytes(b"%PDF")
    records.remove_campaign(ch, c["id"])
    assert not folder.exists() and records.campaigns(ch, eid) == [keep]
    assert [x["id"] for x in records.features(ch)] == [g["id"]]
    other = records.campaign_folder(ch, eid, keep["id"])
    other.mkdir(parents=True)
    records.remove_campaign(ch, keep["id"], files=False)
    assert other.exists() and records.features(ch) == []
