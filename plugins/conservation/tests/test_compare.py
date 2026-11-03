"""The comparison of two campaigns of a face: per term the area gained and lost, the cracks new, longer or wider, the
treated areas that show decay again; the change of the orthophoto in Lab and, with a second survey, the change of the
relief; the check that the two surveys are in the same system (markers, else phase correlation of the orthophotos)
and nothing written when they are not."""
from __future__ import annotations

import numpy as np
import pytest

import conservation_samples as S
from ppg_conservation import compare, records
from ppg_conservation.records import NotReady

Z0 = S.Z0


def _two_campaigns(ch, entry):
    """Campaign A and campaign B on the same face: in B the black crust has grown by 0.2 m2 and a new crack."""
    a = records.new_campaign(ch, entry["id"], date="2026-05-14", name="A")
    b = records.new_campaign(ch, entry["id"], date="2027-05-14", name="B")
    u0, v0, u1, v1 = S.BLACK_CRUST
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    records.add_feature(ch, b["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box((u0, v0, u1 + 0.25, v1)))
    records.add_feature(ch, a["id"], "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    records.add_feature(ch, b["id"], "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    new = records.add_feature(ch, b["id"], "crack", "ISCS.CRACK", f"LINESTRING (3.5 {Z0 + 0.3}, 3.9 {Z0 + 1.1})",
                              width_mm=1.0)
    clean = records.add_feature(ch, a["id"], "treatment", "INT.CLEANING", S.box(S.EFFLORESCENCE),
                                treatment={"technique": "INT.CLEANING", "done": True})
    eu0, ev0, eu1, ev1 = S.EFFLORESCENCE
    records.add_feature(ch, b["id"], "decay", "ISCS.EFFLORESCENCE", S.box((eu0, ev0, eu0 + 0.5, ev1)))
    return a, b, new, clean


def test_two_campaigns_of_one_face(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    a, b, new, clean = _two_campaigns(ch, entry)
    r = compare.compare(ch, a["id"], b["id"])
    assert r["alignment"]["ok"] and r["alignment"]["method"] == "same"
    crust = next(row for row in r["terms"] if row["code"] == "ISCS.CRUST.BLACK_CRUST")
    assert crust["gained"] == pytest.approx(0.2, rel=0.05)
    assert crust["lost"] == pytest.approx(0.0, abs=1e-6)
    assert crust["area_a"] == pytest.approx(0.8, rel=0.01) and crust["area_b"] == pytest.approx(1.0, rel=0.01)
    states = {row["id_b"]: row["state"] for row in r["cracks"]}
    assert states[new["id"]] == "new"
    assert list(states.values()).count("same") == 1
    treated = next(row for row in r["treatments"] if row["id"] == clean["id"])
    assert treated["decay_area"] == pytest.approx(0.2, rel=0.05)
    folder = ch.data_dir / "conservation" / entry["id"] / f"compare_{a['id']}_{b['id']}"
    assert r["folder"] == folder
    for key in ("change", "xlsx", "json"):
        assert r[key].exists() and r[key].parent == folder
    assert r.get("pdf") or r.get("png")
    assert r["relief_change"] is None  # one survey: no change of the relief
    import rasterio
    from openpyxl import load_workbook

    with rasterio.open(r["change"]) as src, rasterio.open(ch.data_dir / entry["orthophoto"]) as ortho:
        assert src.transform == ortho.transform and src.shape == ortho.shape and src.dtypes[0] == "float32"
        dE = src.read(1)
    assert np.nanmax(np.abs(dE)) < 1e-3  # the same orthophoto: no change of colour
    assert np.isnan(dE[0, -1])  # far from every feature (more than 0.5 m): masked
    sheets = load_workbook(r["xlsx"]).sheetnames
    assert sheets == ["Confronto per termine", "Lesioni", "Interventi"]
    # the comparison is files, never records
    assert set(ch.meta["conservation"]) == {"campaigns", "features", "assessments", "legend", "local_terms"}


def test_second_survey_on_the_same_grid_and_its_relief(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc, offset=0.02)
    built = compare.ensure_elevation(other, entry)
    import rasterio

    with rasterio.open(ch.data_dir / entry["orthophoto"]) as a, \
            rasterio.open(other.data_dir / built["orthophoto"]) as b:
        assert a.transform == b.transform and a.shape == b.shape
    assert compare.ensure_elevation(other, entry)["id"] == built["id"]  # found, not built again
    assert len(other.meta["elevations"]) == 1
    S.copy_orthophoto(ch, entry, other, built)
    a = records.new_campaign(ch, entry["id"], date="2026-05-14")
    b = records.new_campaign(other, built["id"], date="2027-05-14")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    records.add_feature(other, b["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    r = compare.compare(ch, a["id"], b["id"], other=other)
    assert r["alignment"]["ok"] and r["alignment"]["method"] == "phase"
    assert r["alignment"]["shift_px"] < 0.5
    with rasterio.open(r["relief_change"]) as src:
        d = src.read(1)
    assert np.nanmedian(d) == pytest.approx(2.0, abs=0.5)  # cm, B minus A, towards the viewer
    assert r["relief"]["median_cm"] == pytest.approx(2.0, abs=0.5)


def test_a_survey_not_aligned_is_refused_and_nothing_is_written(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc, shift_u=0.05)
    built = compare.ensure_elevation(other, entry)
    S.copy_orthophoto(ch, entry, other, built, shift_u=0.05)
    check = compare.check_alignment(ch, entry, other)
    assert not check["ok"] and check["method"] == "phase"
    assert check["shift_px"] == pytest.approx(5.0, abs=0.5)
    assert check["shift_m"] == pytest.approx(0.05, abs=0.005)
    assert check["shift_uv"][0] == pytest.approx(0.05, abs=0.005) and abs(check["shift_uv"][1]) < 0.005
    assert "Allinea chunk" in check["message"]
    a = records.new_campaign(ch, entry["id"], date="2026-05-14")
    b = records.new_campaign(other, built["id"], date="2027-05-14")
    records.add_feature(ch, a["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    with pytest.raises(NotReady, match="Allinea chunk"):
        compare.compare(ch, a["id"], b["id"], other=other)
    assert not (ch.data_dir / "conservation" / entry["id"] / f"compare_{a['id']}_{b['id']}").exists()
    assert not (ch.data_dir / "conservation").exists()


def test_markers_on_both_faces_decide(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    other = S.second_chunk(doc)
    built = compare.ensure_elevation(other, entry)  # its orthophoto is empty: the markers decide
    for label, u, v in (("T1", 1.0, Z0 + 0.5), ("T2", 5.0, Z0 + 2.5)):
        S.put_marker(ch, label, u, v)
        S.put_marker(other, label, u, v, du=0.004)
    check = compare.check_alignment(ch, entry, other)
    assert check["ok"] and check["method"] == "markers" and check["markers"] == 2
    assert check["shift_m"] == pytest.approx(0.004, abs=1e-4)
    far = S.second_chunk(doc, label="Muro 2028")
    compare.ensure_elevation(far, entry)
    for label, u, v in (("T1", 1.0, Z0 + 0.5), ("T2", 5.0, Z0 + 2.5)):
        S.put_marker(far, label, u, v, du=0.05)
    check = compare.check_alignment(ch, entry, far)
    assert not check["ok"] and check["method"] == "markers"
    assert check["shift_m"] == pytest.approx(0.05, abs=1e-3)
    assert "Allinea chunk" in check["message"]
    assert built["id"] != entry["id"] or other is not ch
