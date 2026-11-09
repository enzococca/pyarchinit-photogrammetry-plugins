"""Courses, joints and stones from a drawing of a face classified by orientation (the program's
``engine.orient_lines``, 1.7.7): an ashlar wall drawn by its beds and joints gives its courses and stones; nothing
written before the accept; the same drawing taken again adds no stone twice; the command line; the window's action one
step of *Annulla*, its check plate run for real, hidden on an older program (0.1.3)."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, stones, walls  # noqa: E402

pytest.importorskip("pyarchinit_photogrammetry.engine.orient_lines", reason="the program 1.7.7 classifies drawings")

from ppg_masonry import classified  # noqa: E402

Z0 = S.WORLD[2]
U0 = 0.5  # the drawing starts half a metre along the face


@pytest.fixture
def face(tmp_path):
    doc, ch = S.project(tmp_path, S.wall(lean=False, bulge=False))
    w = walls.new_wall(ch, "Muro")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.STONE_WALL, 0.0],
                             side=S.WORLD[:2] + [3.0, -5.0], resolution=0.05)
    S.put_lines(ch, entry["id"], S.bed_and_joint_lines(Z0, U0))
    doc.save()
    return doc, ch, w, entry


def _wkts(ch, eid):
    return classified.stored(ch, eid)


def test_ashlar_drawing_gives_courses_and_stones(face, tmp_path):
    doc, ch, w, entry = face
    eid = entry["id"]
    prop = classified.propose(ch, eid)
    c = classified.counts(prop)
    assert (c["beds"], c["courses"], c["stones"], c["pairs"], c["other"], c["unused"]) == (7, 6, 30, 0, 0, 0), c
    assert c["joints"] == 26  # 24 joints and the two ends of the wall, each one line through all the courses
    assert [co["stones"] for co in prop["courses"]] == [5] * 6
    for k, co in enumerate(prop["courses"]):
        assert co["n"] == k + 1 and not co["pair"]
        assert abs(co["v"] - (Z0 + (k + 0.5) * S.DRAWN_PITCH)) < 0.01
        assert abs(co["thickness"] - S.DRAWN_PITCH) < 0.01
    beds = [b["v"] for b in prop["beds"]]
    assert np.allclose(beds, Z0 + S.DRAWN_PITCH * np.arange(7), atol=0.01)
    # the stones: the cells between the beds split by the joints, 0.55 m wide (the halves at the ends of the odd
    # courses), the courses numbered from the lowest
    widths = sorted(round(s["width"] / S.DRAWN_JOINTS * 4) / 4 for s in prop["stones"])
    assert widths == [0.5] * 3 + [1.0] * 24 + [1.5] * 3, widths
    assert all(abs(s["height"] - S.DRAWN_PITCH) < 0.02 for s in prop["stones"])
    assert [s["course"] for s in prop["stones"]] == [k for k in range(1, 7) for _ in range(5)]
    assert all(U0 - 0.01 < s["polygon"].bounds[0] and s["polygon"].bounds[2] < U0 + S.DRAWN_LENGTH + 0.01
               for s in prop["stones"])
    union = sum(s["polygon"].area for s in prop["stones"])
    assert abs(union - S.DRAWN_LENGTH * S.DRAWN_COURSES * S.DRAWN_PITCH) < 0.02  # the wall, no gap nor overlap
    # the plugin's own courses on them agree
    rows = [stones.measure(s["polygon"], s["id"]) for s in prop["stones"]]
    found = [x for x in stones.find_courses(rows) if x["counts"]]
    assert len(found) == 6 and all(x["horizontal"] and x["stones"] == 5 for x in found)
    text = classified.summary_text(prop)
    assert text.startswith("Letti 7, giunti 26, filari 6, pietre 30.") and "filare 6: quota media" in text
    # the check plate, PNG and SVG
    png = classified.plate(ch, eid, prop, tmp_path / "tavola.png")
    svg = classified.plate(ch, eid, prop, tmp_path / "tavola.svg")
    assert png.stat().st_size > 10_000 and "filare 6" in svg.read_text()
    with pytest.raises(walls.NotReady, match="PNG o un SVG"):
        classified.plate(ch, eid, prop, tmp_path / "tavola.pdf")
    # the top course traced as a band (a pair along the axis): a course of known thickness above the sixth
    lines = S.bed_and_joint_lines(Z0, U0, ends=False)
    top = Z0 + S.DRAWN_COURSES * S.DRAWN_PITCH
    band = np.array([(U0, top), (U0 + S.DRAWN_LENGTH, top), (U0 + S.DRAWN_LENGTH, top + 0.12), (U0, top + 0.12),
                     (U0, top)])
    S.put_lines(ch, eid, lines[:S.DRAWN_COURSES] + lines[S.DRAWN_COURSES + 1:] + [band])
    prop = classified.propose(ch, eid)
    c = classified.counts(prop)
    assert (c["courses"], c["stones"], c["pairs"]) == (7, 31, 1), c
    assert prop["courses"][-1]["pair"] and abs(prop["courses"][-1]["thickness"] - 0.12) < 1e-6
    assert prop["courses"][-2]["stones"] == 5  # the sixth course closed by the band's lower line


def test_nothing_written_before_accept(face, tmp_path):
    doc, ch, w, entry = face
    eid = entry["id"]
    units, _notch = S.usm_layout(Z0)
    data = ch.meta["drawings"]["frames"][eid]
    data["units"] = [{"wkt": g.wkt, "us": us, "tipo": "USM", "area": "1", "sito": "Saggio", "attributes": {}}
                     for us, g in units.items()]
    before = json.dumps(ch.meta, sort_keys=True, default=str)
    prop = classified.propose(ch, eid)
    classified.plate(ch, eid, prop, tmp_path / "tavola.png")
    classified.merged(ch, eid, prop)
    assert json.dumps(ch.meta, sort_keys=True, default=str) == before and _wkts(ch, eid) == []
    # the outlines of the USM are left out, unless asked
    assert classified.counts(prop)["stones"] == 30
    with_units = classified.propose(ch, eid, params={"units": True})
    assert classified.counts(with_units)["beds"] > 7
    # the same lines from a file in the face's (u, v)
    path = tmp_path / "corsi.geojson"
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": ln.tolist()}}
             for ln in S.bed_and_joint_lines(Z0, U0)]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    from_file = classified.propose(ch, eid, path)
    assert classified.counts(from_file)["stones"] == 30 and from_file["source"] == "file:corsi.geojson"
    # what cannot be done is said
    with pytest.raises(walls.NotReady, match="Nessun prospetto"):
        classified.propose(ch, "nessuno")
    with pytest.raises(walls.NotReady, match="joint_cover"):
        classified.propose(ch, eid, params={"joint_cover": 0})
    with pytest.raises(walls.NotReady):  # a parameter of the classification out of range, in Italian
        classified.propose(ch, eid, params={"min_length": -1})
    other = dict(prop, eid="altra")
    with pytest.raises(walls.NotReady, match="altra faccia"):
        classified.accept(ch, eid, other)
    ch.meta["drawings"]["frames"][eid]["lines"] = []
    with pytest.raises(walls.NotReady, match="solo i contorni delle sue unità"):
        classified.propose(ch, eid)
    assert _wkts(ch, eid) == []


def test_accept_twice_adds_no_duplicates(face):
    doc, ch, w, entry = face
    eid = entry["id"]
    hand = S._stone(5.0, Z0 + 0.1, 5.4, Z0 + 0.3)  # a stone drawn by hand elsewhere on the face: kept
    classified.put(ch, eid, [hand.wkt])
    prop = classified.propose(ch, eid)
    assert classified.accept(ch, eid, prop) == 30 and prop["already"] == []
    assert len(_wkts(ch, eid)) == 31 and _wkts(ch, eid)[0] == hand.wkt
    again = classified.propose(ch, eid)
    assert classified.accept(ch, eid, again) == 0 and len(again["already"]) == 30
    assert len(_wkts(ch, eid)) == 31
    # the drawing redrawn a few mm off (another seed of the hand): still the same stones
    S.put_lines(ch, eid, S.bed_and_joint_lines(Z0, U0, seed=9))
    redrawn = classified.propose(ch, eid)
    assert classified.accept(ch, eid, redrawn) == 0 and len(_wkts(ch, eid)) == 31
    # a stone taken away comes back, only that one
    classified.put(ch, eid, _wkts(ch, eid)[:-1])
    assert classified.accept(ch, eid, classified.propose(ch, eid)) == 1 and len(_wkts(ch, eid)) == 31
    # the program's stone list reads them, no duplicate among them; the plugin measures 6 courses on them
    items, set_aside = stones.stone_items(ch, eid)
    assert len(items) == 31 and set_aside["merged"] == [] and set_aside["inside"] == []
    r = stones.measure_face(ch, w["id"], eid)
    assert r["courses"] == 6
    # in place of the face's stones
    assert classified.accept(ch, eid, classified.propose(ch, eid), replace=True) == 30
    assert len(_wkts(ch, eid)) == 30 and hand.wkt not in _wkts(ch, eid)


def test_cli(face, capsys, tmp_path):
    import pyarchinit_photogrammetry as ppg
    from ppg_masonry import MasonryPlugin

    doc, ch, w, entry = face
    p = str(doc.path)
    plate = tmp_path / "tavola.png"
    plugin = MasonryPlugin()
    assert plugin.command("classified", p, "Muro", "--plate", str(plate)) == 0
    out = capsys.readouterr().out
    assert "Letti 7, giunti 26, filari 6, pietre 30." in out and plate.exists() and "--accept" in out
    assert _wkts(ppg.Document.open(p).chunk, entry["id"]) == []  # nothing saved without --accept
    assert plugin.command("classified", p, "Muro", "--accept", "--joint-cover", "0.5", "--param",
                          "pair_isolation=1.5") == 0
    assert "30 pietre aggiunte alla faccia." in capsys.readouterr().out
    assert len(_wkts(ppg.Document.open(p).chunk, entry["id"])) == 30
    assert plugin.command("classified", p, "Muro", entry["name"], "--accept") == 0  # again: nothing twice
    assert "0 pietre aggiunte alla faccia; 30 già sulla faccia, saltate." in capsys.readouterr().out
    assert len(_wkts(ppg.Document.open(p).chunk, entry["id"])) == 30
    assert plugin.command("classified", p, "Muro", "--accept", "--replace") == 0
    assert "30 pietre aggiunte alla faccia." in capsys.readouterr().out
    assert len(_wkts(ppg.Document.open(p).chunk, entry["id"])) == 30
    assert plugin.command("classified", p, "Muro", "--min-stone", "0.3") == 0  # courses lower than the smallest stone
    assert "filari 0, pietre 0" in capsys.readouterr().out
    assert plugin.command("classified", p, "Muro", "--param", "nessuno=1") == 1
    assert plugin.command("classified", p, "Muro", "--param", "pair_isolation") == 1
    assert plugin.command("classified", p, "Muro", "--file", str(tmp_path / "manca.dxf")) == 1
    assert "non c'è" in capsys.readouterr().out


def test_cli_and_propose_without_the_program(face, capsys, monkeypatch):
    from ppg_masonry import MasonryPlugin

    doc, ch, w, entry = face
    monkeypatch.setattr(classified, "orient_lines", None)
    assert not classified.available()
    assert MasonryPlugin().command("classified", str(doc.path), "Muro") == 1
    assert "serve il programma 1.7.7 o successivo" in capsys.readouterr().out
    with pytest.raises(walls.NotReady, match="1.7.7"):
        classified.propose(ch, entry["id"])


def test_the_action_is_hidden_without_the_program(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ppg_masonry.pane import MasonryPane

    QApplication.instance() or QApplication([])
    assert not MasonryPane().buttons["classified"].isHidden()
    monkeypatch.setattr(classified, "orient_lines", None)
    assert MasonryPane().buttons["classified"].isHidden()


@pytest.fixture
def window(tmp_path, monkeypatch, face):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ppg_masonry import MasonryPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    doc, ch, w, entry = face
    app.document = doc
    win = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    win.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(win, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(win, "_info", lambda *a, **k: None)
    monkeypatch.setattr(win, "_warn", lambda *a, **k: None)
    monkeypatch.setattr(win, "choice", lambda *a, **k: 0, raising=False)
    plugin = MasonryPlugin()
    plugin.activate(Context("masonry", tmp_path / "plugin", window=win))
    said = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(f"{kind}: {text}"))

    def wait():
        win.runner.wait()
        QApplication.processEvents()
        assert not errors, "\n".join(errors)

    try:
        yield win, ch, plugin, entry, wait, said
    finally:
        plugin.deactivate()
        win.setWindowModified(False)
        win.close()
        win.deleteLater()
        app.document = None


def test_the_action_in_the_window_is_one_undo_step(window, monkeypatch):
    win, ch, plugin, entry, wait, said = window
    eid = entry["id"]
    pane = plugin.pane
    assert not pane.buttons["classified"].isHidden()
    plugin.refresh_pane()
    pane.select_face(eid)
    plugin._requested("show")  # the face shown: the window keeps its stones and draws them
    wait()
    assert win.drawings_frame == eid
    hand = S._stone(5.0, Z0 + 0.1, 5.4, Z0 + 0.3)
    win.set_stones(eid, [hand])
    asked, plates = {}, []

    def answer(title, fields, intro=""):
        asked[title] = {f.name: f for f in fields}
        return {f.name: f.default for f in fields if f.kind != "heading"}

    monkeypatch.setattr(plugin, "_ask", answer)
    monkeypatch.setattr(plugin, "_check_plate",
                        lambda plate, prop, dup, present: plates.append((plate, dup, present)) or False)
    entries = len(win.history.entries)
    plugin._requested("classified")
    wait()
    assert not said, said
    (plate, dup, present), = plates
    assert plate.exists() and plate.parent.name == "classificato" and (dup, present) == (0, 1)
    fields = asked["Corsi e giunti da un disegno classificato"]
    assert fields["pair_isolation"].default == 1.5 and fields["joint_cover"].default == 0.5
    assert fields["source"].default == "face" and fields["units"].default is False
    assert len(win.frame_stones[eid]) == 31 and len(_wkts(ch, eid)) == 31  # the window and the project agree
    assert len(win.history.entries) == entries + 1
    assert "Pietre" in win.ortho_view.layers
    win.history.undo()
    assert len(win.frame_stones[eid]) == 1 and _wkts(ch, eid) == [hand.wkt]
    win.history.redo()
    assert len(win.frame_stones[eid]) == 31 and len(_wkts(ch, eid)) == 31
    # the same drawing again: nothing added, said, no entry of Annulla
    entries = len(win.history.entries)
    plugin._requested("classified")
    wait()
    assert plates[-1][1:] == (30, 31)
    assert len(_wkts(ch, eid)) == 31 and len(win.history.entries) == entries
    assert any("30 già sulla faccia" in m for m in said), said
    # cancelled at the plate: nothing
    monkeypatch.setattr(plugin, "_check_plate", lambda *a: None)
    win.set_stones(eid, [hand])
    entries = len(win.history.entries)
    plugin._requested("classified")
    wait()
    assert _wkts(ch, eid) == [hand.wkt] and len(win.history.entries) == entries


def test_the_check_plate_dialog(window, tmp_path):
    """The real dialog, offscreen: the plate, the summary, the duplicates said, the box to replace; Cancel = None."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QLabel

    win, ch, plugin, entry, wait, said = window
    prop = classified.propose(ch, entry["id"])
    plate = classified.plate(ch, entry["id"], prop, tmp_path / "tavola.png")
    seen = {}

    def press(tick=False, ok=True):
        def run():
            dlg = next((d for d in QApplication.topLevelWidgets() if isinstance(d, QDialog) and d.isVisible()), None)
            if dlg is None:  # not shown yet
                QTimer.singleShot(20, run)
                return
            box = dlg.findChild(QCheckBox)
            seen["box"] = (box.text(), box.isChecked(), box.isEnabled())
            seen["texts"] = [lab.text() for lab in dlg.findChildren(QLabel)]
            seen["image"] = any(not lab.pixmap().isNull() for lab in dlg.findChildren(QLabel))
            if tick:
                box.setChecked(True)
            buttons = dlg.findChild(QDialogButtonBox)
            which = QDialogButtonBox.StandardButton.Ok if ok else QDialogButtonBox.StandardButton.Cancel
            seen["ok"] = buttons.button(QDialogButtonBox.StandardButton.Ok).text()
            buttons.button(which).click()

        QTimer.singleShot(0, run)

    press()
    assert plugin._check_plate(plate, prop, 0, 0) is False
    assert seen["box"] == ("Sostituisci le 0 pietre della faccia", False, False) and seen["image"]
    assert seen["ok"] == "Aggiungi le pietre"
    assert any(x.startswith("Letti 7, giunti 26, filari 6, pietre 30.") for x in seen["texts"])
    press(tick=True)
    assert plugin._check_plate(plate, prop, 4, 12) is True
    assert seen["box"] == ("Sostituisci le 12 pietre della faccia", False, True)
    assert any("4 pietre proposte sono già sulla faccia" in x for x in seen["texts"])
    press(ok=False)
    assert plugin._check_plate(plate, prop, 0, 12) is None
    assert _wkts(ch, entry["id"]) == []


def test_a_bare_accept_is_still_an_error_for_usm(face, capsys):
    # --accept takes no value for classified; for usm it needs one (as before 0.1.3): never every proposal by mistake
    import pyarchinit_photogrammetry as ppg
    from ppg_masonry import MasonryPlugin

    doc, ch, w, entry = face
    p = str(doc.path)
    plugin = MasonryPlugin()
    with pytest.raises(SystemExit) as exc:
        plugin.command("usm", p, "Muro", "--accept")
    assert exc.value.code == 2 and "usm --accept" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        plugin.command("classified", p, "Muro", "--accept", "1,3")
    assert exc.value.code == 2 and "takes no value" in capsys.readouterr().err
    assert _wkts(ppg.Document.open(p).chunk, entry["id"]) == []  # nothing saved by either
    assert plugin.command("classified", p, "Muro", "--accept", "all") == 0
    assert len(_wkts(ppg.Document.open(p).chunk, entry["id"])) == 30
