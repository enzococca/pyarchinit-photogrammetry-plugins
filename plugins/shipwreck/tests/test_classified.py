"""A drawing classified by orientation (the program's ``engine.orient_lines``, 1.7.7) read as timbers along the keel:
pairs across = frames, lines along = seams, pairs along = strakes; nothing into the catalogue before the check plate
and the accept; the window's action one step of *Annulla*; the program's water level of the chunk as the default of
the plugin's *Livello e rifrazione* (0.5.0)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from hulls import DRAWN_FRAMES, draw_timbers, project

from ppg_shipwreck import lines, ship, timbers

pytest.importorskip("pyarchinit_photogrammetry.engine.orient_lines", reason="the program 1.7.7 classifies drawings")

from ppg_shipwreck import classified  # noqa: E402


@pytest.fixture
def hull(tmp_path):
    doc, ch, ends = project(tmp_path, heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    doc.save()
    return doc, ch, ship.frame_of(ch)


def test_frames_and_seams_from_a_drawing(hull, tmp_path):
    doc, ch, frame = hull
    draw_timbers(ch, frame)
    prop = classified.propose(ch, "Rilievo")
    c = classified.counts(prop)
    assert (c["frame"], c["seam"], c["plank"], c["skipped"]) == (8, 4, 1, 1)
    frames = sorted((r for r in prop["records"] if r["kind"] == "frame"), key=lambda r: (r["x"], r["side"]))
    assert all(r["geometry"] == "polygon" and abs(r["width"] - 0.08) < 1e-3 for r in frames)
    assert [r["side"] for r in frames[:2]] == ["port", "starboard"]
    assert [round(r["x"], 2) for r in frames[::2]] == [round(x + 0.04, 2) for x in DRAWN_FRAMES]
    # named from the bow (x = 0 with the bow at the start): both sides of one frame share the number
    assert [r["name"] for r in frames[:2]] == ["ordinata 1", "ordinata 1"] and frames[-1]["name"] == "ordinata 4"
    seams = {r["name"]: r for r in prop["records"] if r["kind"] == "seam"}
    assert set(seams) == {"comento B1", "comento B2", "comento T1", "comento T2"}
    assert abs(seams["comento B1"]["y"] - 0.3) < 1e-6 and abs(seams["comento T2"]["y"] + 1.0) < 1e-6
    plank = next(r for r in prop["records"] if r["kind"] == "plank")
    assert plank["side"] == "port" and plank["name"] == "fascia B1" and abs(plank["width"] - 0.25) < 1e-3

    records = classified.accept(ch, prop)
    assert len(records) == 13 and len(timbers.timbers_of(ch)) == 13
    rec = next(r for r in records if r["name"] == "comento B1")
    assert rec["kind"] == "seam" and rec["geometry"]["type"] == "line" and rec["status"] == "measured"
    assert rec["source"].startswith("orient:") and rec["chunk"] == ch.key
    # the geometry in the chunk's coordinates lies on the drawn seam, back through the frame
    back = frame.to_ship(np.column_stack([np.asarray(rec["geometry"]["xy"]), np.zeros(2)]))
    assert np.allclose(back[:, 1], 0.3, atol=1e-6)
    measured = {r["id"]: r for r in timbers.measure(ch)}
    f1 = next(r for r in records if r["kind"] == "frame")
    assert abs(measured[f1["id"]]["width"] - 0.08) < 0.01
    assert rec["source"] == f"orient:layer:Rilievo:{prop['accepted'][records.index(rec)]}"
    with pytest.raises(ship.NotReady, match="già nel catalogo"):
        classified.accept(ch, prop, [prop["accepted"][0]])
    assert classified.accept(ch, prop) == []  # all accepted already


def test_the_same_drawing_imported_again_adds_nothing(hull):
    from pyarchinit_photogrammetry.core import free_drawings

    doc, ch, frame = hull
    draw_timbers(ch, frame)
    assert len(classified.accept(ch, classified.propose(ch, "Rilievo"))) == 13
    again = classified.propose(ch, "Rilievo")
    assert classified.accept(ch, again) == [] and len(again["already"]) == 13
    assert len(timbers.timbers_of(ch)) == 13
    # explicit ids too
    ids = [r["id"] for r in again["records"]][:2]
    fresh = classified.propose(ch, "Rilievo")
    assert classified.accept(ch, fresh, ids) == [] and fresh["already"] == ids
    # the drawing grown by one seam: only that one comes in
    w = frame.to_world(np.array([[3.5, 1.2, 0.0], [6.5, 1.2, 0.0]]))
    free_drawings.add(ch, "line", w, layer="Rilievo")
    grown = classified.propose(ch, "Rilievo")
    new = classified.accept(ch, grown)
    assert [r["kind"] for r in new] == ["seam"] and len(timbers.timbers_of(ch)) == 14
    # a timber taken out of the catalogue comes back with the next import
    timbers.remove(ch, [new[0]["id"]])
    assert len(classified.accept(ch, classified.propose(ch, "Rilievo"))) == 1


def test_nothing_imported_before_accept(hull):
    doc, ch, frame = hull
    draw_timbers(ch, frame)
    before = json.dumps(ch.meta, sort_keys=True, default=str)
    prop = classified.propose(ch, ("layer", "Rilievo"), {"pair_isolation": 1.0})  # seams stay seams beside frames
    assert prop["records"] and timbers.timbers_of(ch) == []
    assert json.dumps(ch.meta, sort_keys=True, default=str) == before
    # only the kinds and the ids chosen
    seams = classified.accept(ch, prop, kinds=["seam"])
    assert {r["kind"] for r in seams} == {"seam"} and len(seams) == 4
    frame_ids = [r["id"] for r in prop["records"] if r["kind"] == "frame"][:2]
    assert [r["kind"] for r in classified.accept(ch, prop, frame_ids)] == ["frame", "frame"]
    with pytest.raises(ship.NotReady, match="non sono legni"):
        classified.accept(ch, prop, ["999"])
    # a proposal of another ship frame is refused
    ship.set_frame(ch, frame.flipped())
    with pytest.raises(ship.NotReady, match="altro sistema della nave"):
        classified.accept(ch, prop)
    with pytest.raises(ship.NotReady, match="nessuna linea"):
        classified.propose(ch, "Non c'è")


def test_frames_drawn_as_loose_lines_close_together_need_pair_isolation_1(hull):
    doc, ch, frame = hull
    close = tuple(4.0 + 0.25 * k for k in range(8))  # 0.12 sided, 0.13 between
    draw_timbers(ch, frame, loose_frames=True, frames=close, sided=0.12)
    loose = classified.counts(classified.propose(ch, "Rilievo"))
    paired = classified.counts(classified.propose(ch, "Rilievo", {"pair_isolation": 1.0}))
    assert loose["frame"] < 16 <= paired["frame"]


def test_plate_written(hull, tmp_path):
    doc, ch, frame = hull
    draw_timbers(ch, frame)
    prop = classified.propose(ch, "Rilievo")
    png = classified.plate(ch, prop, tmp_path / "tavola.png")
    assert png.exists() and png.stat().st_size > 10_000
    svg = classified.plate(ch, prop, tmp_path / "tavola.svg")
    text = svg.read_text(encoding="utf-8")
    assert "ordinata 1" in text and "comento B1" in text and "fascia B1" in text
    assert timbers.timbers_of(ch) == []


def test_drawing_from_a_file(hull, tmp_path):
    """A GeoJSON in the chunk's system: the same timbers as from the layer."""
    doc, ch, frame = hull
    draw_timbers(ch, frame, layer="F")
    from pyarchinit_photogrammetry.core import free_drawings

    feats = []
    for f in free_drawings.features(ch, "F"):
        xy = [list(map(float, p[:2])) for p in f["xyz"]]
        if f["kind"] == "polygon":
            feats.append({"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [xy + [xy[0]]]}})
        else:
            feats.append({"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": xy}})
    path = tmp_path / "disegno.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": ch.crs}},
                                "features": feats}))
    from pathlib import Path

    a = classified.counts(classified.propose(ch, Path(path)))
    b = classified.counts(classified.propose(ch, "F"))
    assert a == b


def test_water_level_from_the_chunk_when_none_set(hull):
    from pyarchinit_photogrammetry.engine import water

    doc, ch, frame = hull
    assert lines.water_of(ch) is None
    water.set_level(ch, frame.z0 - 0.4, source="hand", n=1.33)
    w = lines.water_of(ch)
    assert w["from"] == "chunk" and not w["on"] and not lines.is_on(ch)
    assert abs(w["level"] + 0.4) < 1e-9 and abs(w["level_world"] - (frame.z0 - 0.4)) < 1e-9 and w["n"] == 1.33
    from ppg_shipwreck import ShipwreckPlugin

    assert "chunk" in ShipwreckPlugin.states(ch)["water"]
    # the plugin's own level wins
    lines.set_water(ch, -0.2, 1.34, False)
    w = lines.water_of(ch)
    assert "from" not in w and abs(w["level"] + 0.2) < 1e-9


def test_a_water_level_the_program_cannot_read_is_no_default(hull):
    doc, ch, frame = hull
    ch.meta["water"] = {"level": 11.0, "built_in": "not a matrix", "x": 0.0, "y": 0.0}
    assert lines.water_of(ch) is None and not lines.is_on(ch)
    ch.meta["water"] = {"level": "nan"}
    assert lines.water_of(ch) is None


def test_cli_classified(hull, capsys, tmp_path):
    from ppg_shipwreck import ShipwreckPlugin

    doc, ch, frame = hull
    draw_timbers(ch, frame)
    doc.save()
    plugin = ShipwreckPlugin()
    p = str(doc.path)
    plate = tmp_path / "tavola.png"
    assert plugin.command("classified", p, "--layer", "Rilievo", "--plate", str(plate)) == 0
    out = capsys.readouterr().out
    assert "Ordinate 8, comenti 4, fasce 1" in out and plate.exists() and "--accept" in out
    import pyarchinit_photogrammetry as ppg

    assert timbers.timbers_of(ppg.Document.open(p).chunk) == []  # not saved without --accept
    assert plugin.command("classified", p, "--layer", "Rilievo", "--pair-isolation", "1.5", "--accept",
                          "--kinds", "frame,plank") == 0
    out = capsys.readouterr().out
    assert "9 legni importati" in out
    got = timbers.timbers_of(ppg.Document.open(p).chunk)
    assert sorted({r["kind"] for r in got}) == ["frame", "plank"] and len(got) == 9
    assert plugin.command("classified", p, "--layer", "Rilievo", "--accept") == 0  # again: the frames and plank skipped
    out = capsys.readouterr().out
    assert "4 legni importati; 9 già nel catalogo da un disegno classificato, saltati" in out
    assert plugin.command("classified", p, "--layer", "Rilievo", "--accept") == 0
    assert "0 legni importati; 13 già nel catalogo" in capsys.readouterr().out
    assert len(timbers.timbers_of(ppg.Document.open(p).chunk)) == 13
    assert plugin.command("classified", p, "--layer", "Nessuno") == 1
    assert "nessuna linea" in capsys.readouterr().out
    assert plugin.command("classified", p, "--layer", "Rilievo", "--kinds", "keel", "--accept") == 1


def test_cli_without_the_program_says_what_is_needed(hull, capsys, monkeypatch):
    from ppg_shipwreck import ShipwreckPlugin

    doc, ch, frame = hull
    monkeypatch.setattr(classified, "orient_lines", None)
    assert ShipwreckPlugin().command("classified", str(doc.path), "--layer", "Rilievo") == 1
    assert "serve il programma 1.7.7 o successivo" in capsys.readouterr().out
    with pytest.raises(ship.NotReady, match="1.7.7"):
        classified.propose(ch, "Rilievo")


@pytest.fixture
def window(tmp_path, monkeypatch, hull):
    from PySide6.QtWidgets import QApplication

    from ppg_shipwreck import ShipwreckPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    doc, ch, frame = hull
    draw_timbers(ch, frame)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(w, "_info", lambda *a, **k: None)
    monkeypatch.setattr(w, "_warn", lambda *a, **k: None)
    monkeypatch.setattr(w, "choice", lambda *a, **k: 0, raising=False)
    plugin = ShipwreckPlugin()
    plugin.activate(Context("shipwreck", tmp_path / "plugin", window=w))
    said = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(f"{kind}: {text}"))

    def wait():
        w.runner.wait()
        QApplication.processEvents()
        assert not errors, "\n".join(errors)

    try:
        yield w, ch, plugin, wait, said
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
        w.deleteLater()
        app.document = None


def test_the_action_in_the_window_is_one_undo_step(window, monkeypatch):
    w, ch, plugin, wait, said = window
    assert not plugin.pane.buttons["classified"].isHidden()
    asked, plates = {}, []

    def answer(title, fields, intro=""):
        asked[title] = {f.name: f for f in fields}
        return {**{f.name: f.default for f in fields if f.kind != "heading"}, "source": "layer:Rilievo"}

    monkeypatch.setattr(plugin, "_ask", answer)
    monkeypatch.setattr(plugin, "_check_plate", lambda plate, counts: plates.append((plate, counts)) or ["frame", "seam"])
    plugin._requested("classified")
    wait()
    assert not said, said
    (plate, counts), = plates
    assert plate.exists() and counts["frame"] == 8
    assert asked["Disegno classificato"]["pair_isolation"].default == 1.5
    assert ("layer:Rilievo", "Layer: Rilievo") in asked["Disegno classificato"]["source"].choices
    got = timbers.timbers_of(ch)
    assert len(got) == 12 and {r["kind"] for r in got} == {"frame", "seam"}
    w.history.undo()
    assert timbers.timbers_of(ch) == []
    w.history.redo()
    assert len(timbers.timbers_of(ch)) == 12
    # the same drawing again: nothing added, said, no entry of Annulla
    entries = len(w.history.entries)
    plugin._requested("classified")
    wait()
    assert len(timbers.timbers_of(ch)) == 12 and len(w.history.entries) == entries
    assert any("12 già nel catalogo" in m for m in said), said
    # cancelled at the plate: nothing
    monkeypatch.setattr(plugin, "_check_plate", lambda plate, counts: None)
    entries = len(w.history.entries)
    plugin._requested("classified")
    wait()
    assert len(timbers.timbers_of(ch)) == 12 and len(w.history.entries) == entries


def test_the_action_is_hidden_without_the_program(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ppg_shipwreck.pane import ShipwreckPane

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(classified, "orient_lines", None)
    assert ShipwreckPane().buttons["classified"].isHidden()


def test_the_check_plate_dialog(window, tmp_path):
    """The real dialog, offscreen: a box per kind with its count, unticked kinds left out, Cancel = None."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QLabel

    w, ch, plugin, wait, said = window
    prop = classified.propose(ch, "Rilievo")
    plate = classified.plate(ch, prop, tmp_path / "tavola.png")
    counts = classified.counts(prop)
    seen = {}

    def press(untick=(), ok=True):
        def run():
            dlg = next((d for d in QApplication.topLevelWidgets() if isinstance(d, QDialog) and d.isVisible()), None)
            if dlg is None:  # not shown yet
                QTimer.singleShot(20, run)
                return
            boxes = dlg.findChildren(QCheckBox)
            seen["boxes"] = [(b.text(), b.isChecked(), b.isEnabled()) for b in boxes]
            seen["image"] = any(not lab.pixmap().isNull() for lab in dlg.findChildren(QLabel))
            for b in boxes:
                if any(b.text().startswith(u) for u in untick):
                    b.setChecked(False)
            buttons = dlg.findChild(QDialogButtonBox)
            which = QDialogButtonBox.StandardButton.Ok if ok else QDialogButtonBox.StandardButton.Cancel
            buttons.button(which).click()

        QTimer.singleShot(0, run)

    press()
    assert plugin._check_plate(plate, counts) == ["frame", "seam", "plank"]
    assert seen["boxes"] == [("ordinata: 8", True, True), ("comento: 4", True, True),
                             ("corso di fasciame: 1", True, True)] and seen["image"]
    press(untick=("comento",))
    assert plugin._check_plate(plate, counts) == ["frame", "plank"]
    press(ok=False)
    assert plugin._check_plate(plate, counts) is None
    press(untick=("ordinata", "comento", "corso"))
    assert plugin._check_plate(plate, counts) is None  # nothing ticked: nothing to import
    assert timbers.timbers_of(ch) == []
