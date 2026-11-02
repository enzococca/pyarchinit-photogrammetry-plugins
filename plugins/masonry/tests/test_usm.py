"""The USM of a face: the relationships the geometry proposes (si appoggia a, taglia, copre) with their evidence, the
dialog that lets the user keep some of them, and their saving through the program's bridge in a PyArchInit database
made in the temporary folder."""
from __future__ import annotations

import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, stones, usm, walls  # noqa: E402

Z0 = S.WORLD[2]
EXPECTED = {("USM 11", "Si appoggia a", "USM 10"), ("USM 13", "Copre", "USM 11"), ("USM 12", "Taglia", "USM 10")}


def _face(tmp_path, with_stones=True):
    doc, ch = S.project(tmp_path, S.wall(lean=False, bulge=False))
    doc.meta["sito"] = "Saggio"
    w = walls.new_wall(ch, "Muro")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.STONE_WALL, 0.0],
                             side=S.WORLD[:2] + [3.0, -5.0], resolution=0.02)
    units, notch = S.usm_layout(Z0)
    polys = S.stone_pattern(Z0, hole=notch)[0] if with_stones else []
    S.put_drawings(ch, entry["id"], polys, units)
    if with_stones:
        stones.measure_face(ch, w["id"], entry["id"])
    return doc, ch, w, entry


def test_the_three_rules(tmp_path):
    doc, ch, w, entry = _face(tmp_path)
    found = usm.propose(ch, w["id"], entry["id"])
    assert {(p["unit"], p["relation"], p["other"]) for p in found} == EXPECTED
    by = {(p["unit"], p["other"]): p for p in found}
    assert "si interrompe contro USM 12" in by[("USM 12", "USM 10")]["evidence"]
    assert "sta sopra USM 11" in by[("USM 13", "USM 11")]["evidence"]
    assert "100 % della sua larghezza" in by[("USM 11", "USM 10")]["evidence"]
    assert "USM 13 copre USM 11" in usm.proposal_text(by[("USM 13", "USM 11")])


def test_without_the_stones_a_cut_is_not_proposed(tmp_path):
    doc, ch, w, entry = _face(tmp_path, with_stones=False)
    found = usm.propose(ch, w["id"], entry["id"])
    assert not [p for p in found if p["relation"] == "Taglia"]
    assert ("USM 12", "Si appoggia a", "USM 10") in {(p["unit"], p["relation"], p["other"]) for p in found}
    assert all("taglia" in p["note"] for p in found)


def test_the_dialog_keeps_what_the_user_ticks(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ppg_masonry.pane import RelationsDialog

    doc, ch, w, entry = _face(tmp_path)
    found = usm.propose(ch, w["id"], entry["id"])
    assert len(found) == 3
    dialog = RelationsDialog(found)
    assert dialog.list.count() == 3 and all(dialog.list.item(i).checkState() == Qt.CheckState.Unchecked
                                            for i in range(3))
    dialog.list.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.list.item(2).setCheckState(Qt.CheckState.Checked)
    assert dialog.chosen() == [found[0], found[2]]
    assert found[1]["evidence"] in dialog.list.item(1).text()


def test_saved_in_pyarchinit(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.pyarchinit import PyArchInit
    from pyarchinit_photogrammetry.pyarchinit.bridge import parse_relationships
    from pyarchinit_photogrammetry.pyarchinit.demo_db import create_demo_database

    home = tmp_path / "pyarchinit_5"
    us_list = tuple(("1", us, "US", f"Strato {us}", "", "I", "1", "") for us in ("1", "2"))
    try:
        cfg = create_demo_database(home, site="Saggio", us_list=us_list, name="scavo.sqlite", demo_content=False)
    except Exception as exc:  # noqa: BLE001 - no SpatiaLite on this computer
        pytest.skip(f"no PyArchInit database here: {exc}")
    cfg.write_text(
        "{'DATABASE': 'scavo.sqlite', 'SERVER': 'sqlite', 'HOST': '', 'USER': '', 'PASSWORD': '', 'PORT': '', "
        f"'THUMB_PATH': '{tmp_path}/media/thumb/', 'THUMB_RESIZE': '{tmp_path}/media/resize/', 'SITE_SET': ''}}")
    monkeypatch.setenv("PYARCHINIT_HOME", str(home))
    assert str(home).startswith(str(tmp_path))
    pa = PyArchInit.connect(media_root=tmp_path / "media")
    doc, ch, w, entry = _face(tmp_path)
    found = usm.propose(ch, w["id"], entry["id"])
    keep = [p for p in found if (p["unit"], p["relation"]) in (("USM 12", "Taglia"), ("USM 13", "Copre"))]
    media_before = len(pa.db.query("media_table"))
    r = usm.save(pa, ch, w["id"], entry["id"], keep)
    assert sorted(r["created"]) == ["USM 10", "USM 11", "USM 12", "USM 13"] and r["relations"] == 4
    rec12 = pa.us_record("Saggio", "1", "12", "USM")
    assert rec12["d_stratigrafica"] == "Muratura 12"
    assert parse_relationships(rec12["rapporti"]) == [["Taglia", "10", "1", "Saggio"]]
    assert parse_relationships(pa.us_record("Saggio", "1", "10", "USM")["rapporti"]) == [
        ["Tagliato da", "12", "1", "Saggio"]]
    assert parse_relationships(pa.us_record("Saggio", "1", "11", "USM")["rapporti"]) == [
        ["Coperto da", "13", "1", "Saggio"]]
    assert len(pa.db.query("media_table")) == media_before + 2
    assert set(walls.elevation(ch, entry["id"])["media"]) == {"dxf", "svg"}
    # again: nothing new, the same media records
    r2 = usm.save(pa, ch, w["id"], entry["id"], keep)
    assert r2["created"] == [] and r2["relations"] == 0 and len(pa.db.query("media_table")) == media_before + 2
    assert walls.results(ch, w["id"], entry["id"])["usm"]["accepted"][0]["relation"] in ("Taglia", "Copre")
