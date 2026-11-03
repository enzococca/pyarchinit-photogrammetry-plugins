"""The plugin's side: the EN 16096 suggestions on the synthetic face, and the pane in the program's window (offscreen,
the preferences in a temporary folder) - a campaign, areas, a crack and a note drawn through the context, a treatment
from a decay feature, the classes dialog with its suggestion, deletion from the view, layers redrawn and taken away,
a campaign removed and brought back by the window's undo, the English texts."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import conservation_samples as S  # noqa: E402
from ppg_conservation import assess, records  # noqa: E402

Z0 = S.Z0


def _box(u0, v0, u1, v1):
    return S.box((u0, v0, u1, v1))


def test_suggestions_on_the_synthetic_face(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    c = records.new_campaign(ch, entry["id"], surveyor="E. Cocca")
    cid = c["id"]
    assert assess.suggest_cc(ch, cid)[0] == 0
    face = records.face_area(ch, entry)
    # a tenth of the face dark (a black crust on 8.5 %, with the hole below 10 %): CC 1 by the share
    side = float(np.sqrt(0.085 * face))
    crust = records.add_feature(ch, cid, "decay", "ISCS.CRUST.BLACK_CRUST", _box(0.2, Z0 + 0.2, 0.2 + side,
                                                                                    Z0 + 0.2 + side))
    cc, rule = assess.suggest_cc(ch, cid)
    assert cc == 1 and "CC 1" in rule and "8,5 %" in rule
    # a patina is no symptom: the share does not change
    records.add_feature(ch, cid, "decay", "ISCS.PATINA", _box(4.0, Z0 + 0.1, 5.0, Z0 + 0.3))
    assert assess.suggest_cc(ch, cid)[0] == 1
    # with a crack of 3 mm: CC 2 by the crack
    crack = records.add_feature(ch, cid, "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    cc, rule = assess.suggest_cc(ch, cid)
    assert cc == 2 and "fessura di 3,0 mm" in rule
    records.remove_feature(ch, crack["id"])
    # with the hole 3 cm deep: CC 2 by the loss
    records.add_feature(ch, cid, "decay", "ISCS.MISSING_PART", S.box(S.HOLE), depth_mm=30.0)
    cc, rule = assess.suggest_cc(ch, cid)
    assert cc == 2 and "3,0 cm" in rule
    # the thresholds of the project and those given
    assert assess.suggest_cc(ch, cid, thresholds={"loss_cc3_cm": 2.5})[0] == 3
    records.store(ch)["thresholds"] = {"loss_cc2_cm": 4.0}
    assert assess.suggest_cc(ch, cid)[0] == 1
    records.store(ch).pop("thresholds")
    # a detachment on more than 5 % of the face: CC 3
    records.update_feature(ch, crust["id"], code="ISCS.SCALING")
    cc, rule = assess.suggest_cc(ch, cid)
    assert cc == 3 and "distacco" in rule
    # a proposed feature is not counted
    records.discard(ch, [f["id"] for f in records.features(ch, cid)])
    records.add_feature(ch, cid, "crack", "ISCS.CRACK", S.groove(), width_mm=8.0, status="proposed")
    assert assess.suggest_cc(ch, cid)[0] == 0
    # UC: the highest urgency of the features
    assert assess.suggest_uc(ch, cid)[0] is None
    records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH), urgency=1)
    records.add_feature(ch, cid, "decay", "ISCS.EFFLORESCENCE", S.box(S.EFFLORESCENCE), urgency=2)
    uc, why = assess.suggest_uc(ch, cid)
    assert uc == 2 and "UC 2" in why
    # the wordings, the treatments, the words of the forms, the report
    assert assess.labels("en")["cc"][2] == "Moderately strong symptoms"
    assert assess.labels("en")["uc"][3] == "Urgent and immediate"
    assert assess.labels("en")["rc"][1] == "Maintenance / preventive conservation"
    assert assess.labels("it")["cc"][0] == "Nessun sintomo"
    assert assess.usual_treatments("ISCS.CRUST.BLACK_CRUST") == ["INT.CLEANING"]
    assert assess.usual_treatments("UNI.CROSTA") == ["INT.CLEANING"]
    assert assess.usual_treatments("ISCS.ALGA") == ["INT.BIOCIDE", "INT.CLEANING"]
    assert assess.usual_treatments("LOCAL.OPEN_JOINT") == ["INT.REPOINTING"]
    assert assess.usual_treatments("ISCS.DAMAGE") == []
    assert [assess.condition_word(n) for n in range(4)] == ["buono", "discreto", "mediocre", "cattivo"]
    assert assess.condition_word(1, "en") == "fair"
    assert assess.condition_word(2, thesaurus=[("1", "Ottimo"), ("2", "Mediocre")]) == "Mediocre"
    assert assess.condition_word(3, thesaurus=["A", "B", "C", "D"]) == "D"
    records.set_campaign(ch, cid, classes={"cc": 2, "uc": 2, "reason": "efflorescenze"})
    records.set_assessment(ch, c["date"], 1)
    lines = assess.report_lines(ch, cid, "it")
    heads = [x for x in lines if x[:2] in ("1.", "2.", "3.", "4.", "5.", "6.")]
    assert heads == ["1. Oggetto", "2. Fonti", "3. Rilievo", "4. Componente, sintomi e classi",
                     "5. Rischio e raccomandazioni", "6. Sintesi"]
    text = "\n".join(lines)
    assert "CC 2 - Sintomi moderatamente forti" in text and "RC 1 - Manutenzione" in text and "E. Cocca" in text
    assert "Prospetto sud" in text
    assert "Condition class: CC 2 - Moderately strong symptoms" in "\n".join(assess.report_lines(ch, cid, "en"))


def _window(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, entry = S.face(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    errors, infos = [], []
    monkeypatch.setattr(w, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(w, "_info", lambda title, text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "_warn", lambda text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "choice", lambda *a, **k: 0, raising=False)
    return w, doc, ch, entry, errors, infos


def _fids(view, layer):
    lay = view.layers.get(layer)
    return [] if lay is None else [a.get("fid") for _i, _l, a in lay["features"]]


def test_the_pane_in_the_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from ppg_conservation import ConservationPlugin
    from ppg_conservation.pane import ConservationPane
    from pyarchinit_photogrammetry.plugins import Context

    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    eid = entry["id"]
    view = w.ortho_view
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))
        pane = plugin.pane
        assert isinstance(pane, ConservationPane)
        dock = w.panes.groups["right"].dock(pane)
        assert dock is not None and dock.objectName().startswith("pane_plugin_conservation")
        assert dock.windowTitle() == "Conservazione"
        assert pane.minimumSizeHint().width() <= 340
        assert pane.elevations.count() == 1 and pane.elevation_id() == eid and pane.campaigns.count() == 0
        assert "Nuova campagna" in pane.l_state.text()
        # the dialogs answered by the test: the defaults, changed by ``answers``
        asked, answers = [], {}

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append((title, intro, [f.name for f in fields]))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        drawn = {}
        for kind in ("polygon", "line", "point"):
            monkeypatch.setattr(plugin.context, "request_" + kind,
                                lambda done, cancelled=None, hint="", k=kind: done(np.asarray(drawn[k], float)))
        # a campaign made from the pane: the elevation shown
        answers.update(surveyor="E. Cocca", method="ispezione diretta")
        plugin._requested("new_campaign")
        c = records.campaigns(ch, eid)[0]
        assert asked[-1][0] == "Nuova campagna" and c["date"] == "2026-05-14" and c["surveyor"] == "E. Cocca"
        assert pane.campaign_id() == c["id"] and w._raster_version == ("elevation", eid) and w.isWindowModified()
        answers.clear()
        # the term missing: said, nothing drawn
        drawn["polygon"] = list(S.box(S.BLACK_CRUST).exterior.coords)[:-1]
        plugin._requested("area")
        assert "Scegli un termine" in said[-1] and records.features(ch, c["id"]) == []
        # an area with the term chosen: a feature, a polygon in its layer, a row in the table
        assert pane.choose_term("ISCS.CRUST.BLACK_CRUST")
        answers.update(intensity="high", notes="crosta sul basamento")
        plugin._requested("area")
        crust = records.features(ch, c["id"])[0]
        assert crust["code"] == "ISCS.CRUST.BLACK_CRUST" and crust["intensity"] == "high"
        assert crust["extent"] == pytest.approx(1.0 * 0.8) and crust["notes"] == "crosta sul basamento"
        assert asked[-1][0] == "Elemento" and "intensity" in asked[-1][2]
        assert _fids(view, "Degrado") == [crust["id"]] and pane.table.rowCount() == 1
        attrs = view.layers["Degrado"]["features"][0][2]
        assert attrs["code"] == "ISCS.CRUST.BLACK_CRUST"
        answers.clear()
        # a crack term needs a line
        assert pane.choose_term("ISCS.CRACK")
        plugin._requested("area")
        assert "Linea" in said[-1] and len(records.features(ch, c["id"])) == 1
        drawn["line"] = list(S.groove().coords)
        answers.update(width_mm=3.0)
        plugin._requested("line")
        crack = [f for f in records.features(ch, c["id"]) if f["kind"] == "crack"][0]
        assert crack["width_mm"] == 3.0 and _fids(view, "Lesioni") == [crack["id"]]
        assert "width_mm" in asked[-1][2]
        answers.clear()
        # a note at a point
        assert pane.choose_term("ISCS.EFFLORESCENCE")
        drawn["point"] = [[2.5, Z0 + 0.4]]
        plugin._requested("point")
        note = [f for f in records.features(ch, c["id"]) if f["kind"] == "note"][0]
        assert note["code"] == "ISCS.EFFLORESCENCE" and _fids(view, "Note") == [note["id"]]
        assert "ISCS.EFFLORESCENCE" in plugin.context.settings.get("recent", [])
        # a treatment from the decay feature chosen: its geometry, the technique suggested
        pane.select(crust["id"])
        plugin._requested("to_treat")
        treat = [f for f in records.features(ch, c["id"]) if f["kind"] == "treatment"][0]
        assert treat["code"] == "INT.CLEANING" and treat["geometry"] == crust["geometry"]
        assert "suggerimento" in asked[-1][1] and treat["treatment"]["done"] is False
        assert _fids(view, "Interventi") == [treat["id"]]
        # a treatment drawn as a line, its materials remembered
        drawn["line"] = list(S.groove().coords)
        answers.update(technique="INT.GROUTING", materials="malta di calce idraulica", done=True)
        plugin._requested("treat_line")
        sealed = [f for f in records.features(ch, c["id"]) if f["code"] == "INT.GROUTING"][0]
        assert sealed["treatment"]["materials"] == "malta di calce idraulica" and sealed["treatment"]["done"]
        assert "malta di calce idraulica" in plugin.context.settings["materials"]
        answers.clear()
        # the classes: the suggestion and its rule in the dialog, the class the user's
        answers.update(cc=2, uc=1, reason="fessura aperta")
        plugin._requested("classes")
        title, intro, names = asked[-1]
        assert "CC 2" in intro and "fessura di 3,0 mm" in intro and "suggerimento del plugin" in intro
        assert records.campaign(ch, c["id"])["classes"] == {"cc": 2, "uc": 1, "reason": "fessura aperta"}
        answers.clear()
        answers.update(rc=2)
        plugin._requested("assessment")
        assert records.assessment(ch, c["date"])["rc"] == 2
        answers.clear()
        assert "CC 2" in pane.summary.toPlainText() and "RC 2" in pane.summary.toPlainText()
        # the table: the filter by family
        i = pane.family.findData("crack_deformation")
        pane.family.setCurrentIndex(i)
        visible = [r for r in range(pane.table.rowCount()) if not pane.table.isRowHidden(r)]
        assert len(visible) == 1
        pane.family.setCurrentIndex(0)
        # deleted with the window's Seleziona + Canc: the record goes
        view.select_items([view.layers["Lesioni"]["features"][0][0]])
        assert w.delete_selected(confirm=False) == 1
        assert not [f for f in records.features(ch, c["id"]) if f["kind"] == "crack"]
        assert _fids(view, "Lesioni") == []
        # the layers come back when the elevation is loaded again, and go when the view leaves it
        view.load(str(ch.data_dir / entry["orthophoto"]), "ortho")
        assert _fids(view, "Degrado") == [crust["id"]]
        plugin.context.show_frame((0, 0, 10, 10), 0.01)
        assert "Degrado" not in view.layers and "Interventi" not in view.layers
        # Review Focus 2: a campaign removed while shown; the other's features stay; the undo brings it back
        plugin._requested("show")
        assert _fids(view, "Degrado") == [crust["id"]]
        answers.update(name="Seconda", date="2027-06-01")
        plugin._requested("new_campaign")
        answers.clear()
        c2 = records.campaigns(ch, eid)[1]
        assert pane.campaign_id() == c2["id"] and "Degrado" not in view.layers
        assert pane.choose_term("ISCS.ALGA")
        drawn["polygon"] = list(S.box(S.GREEN_PATCH).exterior.coords)[:-1]
        plugin._requested("area")
        alga = records.features(ch, c2["id"])[0]
        pane.select_campaign(c["id"])
        n_first = len(records.features(ch, c["id"]))
        assert _fids(view, "Degrado") == [crust["id"]]
        monkeypatch.setattr(plugin, "_confirm", lambda text: True)
        plugin._requested("remove_campaign")
        assert [x["id"] for x in records.campaigns(ch)] == [c2["id"]] and records.features(ch, c["id"]) == []
        assert crust["id"] not in _fids(view, "Degrado") and _fids(view, "Degrado") == [alga["id"]]
        assert records.features(ch, c2["id"]) == [alga]
        w.history.undo()
        assert {x["id"] for x in records.campaigns(ch)} == {c["id"], c2["id"]}
        assert len(records.features(ch, c["id"])) == n_first and records.features(ch, c2["id"])[0]["id"] == alga["id"]
        pane.select_campaign(c["id"])
        assert crust["id"] in _fids(view, "Degrado")
        w.history.redo()
        assert [x["id"] for x in records.campaigns(ch)] == [c2["id"]]
        # nothing to compare with: said
        plugin._requested("compare")
        assert "Nessun'altra campagna" in said[-1]
        assert len(said) == 3, said  # the term missing, the crack drawn as an area, the comparison: nothing else
        assert not errors, errors
        plugin.deactivate()
        plugin.context.release()
        assert w.panes.dock(pane) is None
    finally:
        w.setWindowModified(False)
        w.close()


def test_proposals_edits_and_treated_areas(tmp_path, monkeypatch):
    """Proposals accepted one by one and all together, discarded; a feature edited with its form; a treated area; the
    photos of a feature in a project without aligned photos; each change one entry of the window's undo."""
    pytest.importorskip("PySide6")
    from ppg_conservation import ConservationPlugin
    from pyarchinit_photogrammetry.plugins import Context

    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    view = w.ortho_view
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))
        pane = plugin.pane
        asked, answers = [], {}

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append((title, intro, [f.name for f in fields]))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._requested("new_campaign")
        cid = records.campaigns(ch, entry["id"])[0]["id"]
        # proposals (as Task 3's detectors will add them): drawn faint in their own layer, not counted
        props = [records.add_feature(ch, cid, "decay", code, S.box(b), status="proposed", source="test")
                 for code, b in (("ISCS.CRUST.BLACK_CRUST", S.BLACK_CRUST), ("ISCS.ALGA", S.GREEN_PATCH),
                                 ("ISCS.EFFLORESCENCE", S.EFFLORESCENCE))]
        plugin.refresh_pane()
        assert sorted(_fids(view, "Proposte")) == sorted(p["id"] for p in props)
        assert _fids(view, "Degrado") == [] and pane.table.rowCount() == 3
        # nothing chosen: said
        pane.table.clearSelection()
        plugin._requested("accept")
        assert "Scegli un elemento" in said[-1]
        n_undo = len(w.history.entries)
        pane.select(props[0]["id"])
        plugin._requested("accept")
        assert records.feature(ch, props[0]["id"])["status"] == "accepted"
        assert _fids(view, "Degrado") == [props[0]["id"]]
        assert len(w.history.entries) == n_undo + 1 and w.history.entries[-1].text.startswith("Conservazione")
        # an accepted feature is no proposal
        pane.select(props[0]["id"])
        plugin._requested("accept")
        assert "non sono proposte" in said[-1]
        # one discarded, the rest accepted together
        pane.select(props[1]["id"])
        plugin._requested("discard")
        assert [f["id"] for f in records.features(ch, cid)] == [props[0]["id"], props[2]["id"]]
        plugin._requested("accept_all")
        assert {f["status"] for f in records.features(ch, cid)} == {"accepted"}
        plugin._requested("accept_all")
        assert "non ha proposte" in said[-1]
        assert "Proposte" not in view.layers or _fids(view, "Proposte") == []
        # the form again on a feature: its term (of the same kind), the intensity, the cause
        pane.select(props[2]["id"])
        answers.update(intensity="medium", cause="risalita capillare")
        plugin._requested("edit")
        title, _intro, names = asked[-1]
        assert title == "Elemento" and "code" in names and "photos" not in names
        rec = records.feature(ch, props[2]["id"])
        assert rec["intensity"] == "medium" and rec["cause"] == "risalita capillare"
        answers.clear()
        # the photos: none aligned in this project
        plugin._requested("photos")
        assert "Nessuna foto" in said[-1]
        # a treated area drawn, then its form edited
        monkeypatch.setattr(plugin.context, "request_polygon",
                            lambda done, cancelled=None, hint="": done(
                                np.asarray(list(S.box(S.EFFLORESCENCE).exterior.coords)[:-1], float)))
        answers.update(technique="INT.DESALINATION")
        plugin._requested("treat_area")
        treat = [f for f in records.features(ch, cid) if f["kind"] == "treatment"][0]
        assert treat["extent"] == pytest.approx(0.4) and _fids(view, "Interventi") == [treat["id"]]
        answers.update(done=True, by="Restauro srl")
        pane.select(treat["id"])
        plugin._requested("edit")
        assert asked[-1][0] == "Intervento"
        assert records.feature(ch, treat["id"])["treatment"]["done"] is True
        assert records.feature(ch, treat["id"])["treatment"]["by"] == "Restauro srl"
        answers.clear()
        # the undo takes back the last change only
        w.history.undo()
        assert records.feature(ch, treat["id"])["treatment"]["done"] is False
        assert len(said) == 4, said  # nothing chosen, no proposal, none left, no photo
        assert not errors, errors
        plugin.deactivate()
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()


def test_forms_photos_and_safe_removals(tmp_path, monkeypatch):
    """A photo opened from the feature form centred on the feature; a number that cannot be read asks the form again
    and keeps the drawing; accepted features discarded only when confirmed; a campaign removed keeps its folder; a
    script's changes are not pushed twice; the campaign edited; favourites; the term tree keeps what the user opened."""
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from ppg_conservation import ConservationPlugin, pane as pane_module
    from pyarchinit_photogrammetry.engine.elevation import world_of
    from pyarchinit_photogrammetry.plugins import Context

    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))
        pane = plugin.pane
        # the forms answered in turn by the list ``replies`` (the defaults when it is empty)
        asked, replies = [], []

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append((title, intro, {f.name: f.default for f in fields}))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(replies.pop(0) if replies else {})
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._requested("new_campaign")
        c = records.campaigns(ch, entry["id"])[0]
        # the campaign edited
        replies.append({"name": "Rilievo di maggio", "weather": "sereno", "method": "da ponteggio"})
        plugin._requested("edit_campaign")
        c = records.campaign(ch, c["id"])
        assert c["name"] == "Rilievo di maggio" and c["weather"] == "sereno"
        assert "da ponteggio" in plugin.context.settings["methods"]
        assert "Rilievo di maggio" in pane.campaigns.currentText()
        # a crack whose opening is no number: the form again with what was typed, then the drawing kept
        assert pane.choose_term("ISCS.CRACK")
        monkeypatch.setattr(plugin.context, "request_line",
                            lambda done, cancelled=None, hint="": done(np.asarray(S.groove().coords, float)))
        replies += [{"width_mm": "tre", "notes": "da rivedere"}, {"width_mm": "2,5"}]
        plugin._requested("line")
        assert [a[0] for a in asked[-2:]] == ["Elemento", "Elemento"]
        assert "Non è un numero" in asked[-1][1] and asked[-1][2]["width_mm"] == "tre"
        assert asked[-1][2]["notes"] == "da rivedere"
        crack = [f for f in records.features(ch, c["id"]) if f["kind"] == "crack"][0]
        assert crack["width_mm"] == 2.5 and crack["notes"] == "da rivedere"
        # the photos of the feature form (the plugin's own _ask): the one chosen opened at the feature's point
        monkeypatch.setattr(plugin, "_ask", ConservationPlugin._ask.__get__(plugin))
        camera = SimpleNamespace(key=987, label="IMG_0001.JPG", enabled=False, aligned=False)
        ch.cameras.append(camera)
        photo = {"key": 987, "label": "IMG_0001.JPG", "distance": 3.0, "angle": 10.0, "px": 100.0, "py": 80.0}
        monkeypatch.setattr(plugin, "_photos", lambda *a, **k: [photo])
        opened = []
        monkeypatch.setattr(w, "open_photo", lambda cam, focus=None: opened.append((cam, focus)))

        def form(parent, title, fields, intro, photos, open_photo):
            open_photo(photos[0])  # *apri*, then the form confirmed as it is
            return {f.name: f.default for f in fields if f.kind != "heading"}

        monkeypatch.setattr(pane_module, "feature_dialog", form)
        pane.select(crack["id"])
        plugin._requested("edit")
        assert records.feature(ch, crack["id"])["photos"][0]["key"] == 987
        cam, focus = opened[-1]
        assert cam is camera and focus is not None
        point = S.groove().representative_point()
        expected = world_of(entry, point.x, point.y, None, ch)
        if expected is None:
            expected = world_of(entry, point.x, point.y, 0.0)
        assert np.allclose(ch.to_world(np.asarray(focus, float)[None])[0], expected, atol=1e-6)
        plugin._requested("photos")
        assert len(opened) == 2 and opened[-1][1] is not None
        ch.cameras.remove(camera)
        monkeypatch.setattr(plugin, "_ask", answer)
        # accepted features discarded only when confirmed; proposals without asking
        prop = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box(S.GREEN_PATCH), status="proposed")
        plugin.refresh_pane()
        confirmations = []
        monkeypatch.setattr(plugin, "_confirm", lambda text: confirmations.append(text) or False)
        pane.select(prop["id"])
        plugin._requested("discard")
        assert confirmations == [] and records.features(ch, c["id"], status="proposed") == []
        pane.select(crack["id"])
        plugin._requested("discard")
        assert "non sono proposte" in confirmations[-1] and records.feature(ch, crack["id"])
        monkeypatch.setattr(plugin, "_confirm", lambda text: confirmations.append(text) or True)
        plugin._requested("discard")
        assert not records.features(ch, c["id"])
        # a script's changes: the script's own entry of the undo, none of the plugin's
        n = len(w.history.entries)
        w._script_running = True
        try:
            replies.append({"notes": "dallo script"})
            plugin._requested("edit_campaign")
        finally:
            w._script_running = False
        assert len(w.history.entries) == n and records.campaign(ch, c["id"])["notes"] == "dallo script"
        # favourites: on and off, at the top of the chooser
        assert pane.choose_term("ISCS.EFFLORESCENCE")
        plugin._requested("favourite")
        assert plugin.context.settings["favourites"] == ["ISCS.EFFLORESCENCE"]
        top = pane.terms.topLevelItem(0)
        assert top.childCount() == 1 and top.child(0).data(0, 0x0100) == "ISCS.EFFLORESCENCE"

        def uni_group():
            return next(pane.terms.topLevelItem(i) for i in range(pane.terms.topLevelItemCount())
                        if pane.terms.topLevelItem(i).text(0) == "UNI 11182")

        # the tree as the user opened it: neither a refresh nor a new favourite closes it
        uni_group().setExpanded(True)
        plugin.refresh_pane()
        assert uni_group().isExpanded()
        plugin._requested("favourite")
        assert plugin.context.settings["favourites"] == [] and uni_group().isExpanded()
        # a campaign removed keeps its folder (the files are the exports'; the undo brings the records back)
        folder = records.campaign_folder(ch, entry["id"], c["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tavola.pdf").write_bytes(b"%PDF")
        plugin._requested("remove_campaign")
        assert "restano nella cartella" in confirmations[-1] and (folder / "tavola.pdf").exists()
        assert records.campaigns(ch) == []
        w.history.undo()
        assert records.campaign(ch, c["id"])["name"] == "Rilievo di maggio"
        assert not said, said
        assert not errors, errors
        plugin.deactivate()
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()


def test_the_proposals_dialog(tmp_path, monkeypatch):
    """*Proposte...*: the groove proposed as a crack and the hole as a missing part in the background, in the layer
    "Proposte", with their source and the detector's notes; the colour classes named by the user (only the green
    one, as biological colonization); SAM's areas (the program's fake SAM) with the chosen term and refused without
    one; *Come questo* on SAM's proposals; a window with no surface said; an area chosen as the window clips the
    proposals; every run one entry of the undo."""
    pytest.importorskip("PySide6")
    import sys

    from shapely import wkt

    from ppg_conservation import ConservationPlugin
    from pyarchinit_photogrammetry.ai.sam_env import ExternalSam2Segmenter
    from pyarchinit_photogrammetry.plugins import Context

    monkeypatch.setenv("PPG_SAM_FAKE", "1")
    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    view = w.ortho_view
    plugin = ConservationPlugin()
    segmenter = ExternalSam2Segmenter(python=sys.executable)
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))
        monkeypatch.setattr(plugin, "_segmenter", lambda: segmenter)
        pane = plugin.pane
        answers = {}

        def answer(title, fields, intro="", photos=None, **_k):
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            if title == "Classi di colore":  # the user names the green class only
                for f in fields:
                    if "verde" in f.label:
                        values[f.name] = "ISCS.BIOLOGICAL_COLONIZATION"
                        assert f.choices[1][0] == "ISCS.BIOLOGICAL_COLONIZATION"  # suggested first
            else:
                values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._requested("new_campaign")
        cid = records.campaigns(ch, entry["id"])[0]["id"]

        def propose(**values):
            answers.clear()
            answers.update(values)
            n = len(said)
            plugin._requested("propose")
            w.runner.wait()
            return said[n:]

        n_undo = len(w.history.entries)
        out = propose(detector="cracks", window="face")
        assert errors == [], errors
        cracks = records.features(ch, cid, status="proposed")
        assert len(cracks) == 1 and cracks[0]["code"] == "ISCS.CRACK" and cracks[0]["source"] == "proposed:cracks"
        assert abs(cracks[0]["extent"] - S.groove().length) <= 0.05 * S.groove().length
        assert abs(cracks[0]["width_mm"] - S.GROOVE_WIDTH * 1000) <= 1.0
        assert "lesione proposta" in cracks[0]["notes"] and "rilievo sul" in cracks[0]["notes"]
        assert "1 proposte aggiunte" in out[-1]
        assert _fids(view, "Proposte") == [cracks[0]["id"]]
        assert len(w.history.entries) == n_undo + 1
        # the hole as a missing part
        propose(detector="loss", window="face")
        loss = [f for f in records.features(ch, cid) if f["source"] == "proposed:material_loss"]
        assert len(loss) == 1 and loss[0]["code"] == "ISCS.MISSING_PART" and loss[0]["depth_mm"] == 30.0
        # the colour classes: only the class named by the user, as proposals
        propose(detector="colours", window="face")
        named = [f for f in records.features(ch, cid) if f["source"] == "proposed:colours"]
        assert named and {f["code"] for f in named} == {"ISCS.BIOLOGICAL_COLONIZATION"}
        green = sum(wkt.loads(f["geometry"]).area for f in named)
        assert abs(green - S.box(S.GREEN_PATCH).area) <= 0.05 * S.box(S.GREEN_PATCH).area
        # SAM with no term: the segments stay "to be named", and cannot be accepted so
        out = propose(detector="sam", window="face")
        sam = [f for f in records.features(ch, cid) if f["source"] == "proposed:sam"]
        assert sam and {f["code"] for f in sam} == {"PPG.UNNAMED"} and "da nominare" in out[-1]
        assert "SAM 2.1" in sam[0]["notes"]
        assert sorted(_fids(view, "Proposte")) == sorted(f["id"] for f in records.features(ch, cid,
                                                                                           status="proposed"))
        pane.select(sam[0]["id"])
        plugin._requested("accept")
        assert "dai un termine prima di accettare" in said[-1]
        assert records.feature(ch, sam[0]["id"])["status"] == "proposed"
        plugin._requested("accept_all")
        assert "dai un termine prima di accettare" in said[-1]
        assert records.features(ch, cid, status="accepted") == []
        # with a term chosen in the list it is used only when asked for
        assert pane.choose_term("ISCS.ALGA")
        propose(detector="sam", window="face")
        assert {f["code"] for f in records.features(ch, cid) if f["source"] == "proposed:sam"} == {"PPG.UNNAMED"}
        propose(detector="sam", window="face", use_term=True)
        named = [f for f in records.features(ch, cid) if f["source"] == "proposed:sam" and f["code"] == "ISCS.ALGA"]
        assert len(named) == len(sam)
        # Come questo (in the background): the unnamed proposals like the chosen area take its term, one undo entry
        records.discard(ch, [f["id"] for f in records.features(ch, cid) if f["source"] == "proposed:sam"])
        g, b = S.GREEN_PATCH, S.BLACK_CRUST
        boxes = [(g[0] + 0.05, g[1] + 0.05, 4.65, g[3] - 0.05), (4.75, g[1] + 0.05, g[2] - 0.05, g[3] - 0.05),
                 (b[0] + 0.05, b[1] + 0.05, 0.95, b[3] - 0.05), (3.2, Z0 + 0.2, 3.8, Z0 + 0.8)]
        segs = [records.add_feature(ch, cid, "decay", "PPG.UNNAMED", S.box(x), status="proposed",
                                    source="proposed:sam") for x in boxes]
        ref = records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box((4.4, g[1] + 0.1, 4.7, g[1] + 0.4)))
        plugin.refresh_pane()
        pane.select(ref["id"])
        n_undo = len(w.history.entries)
        plugin._requested("like")
        w.runner.wait()
        assert [records.feature(ch, s["id"])["code"] for s in segs] == ["ISCS.ALGA", "ISCS.ALGA", "PPG.UNNAMED",
                                                                         "PPG.UNNAMED"]
        assert "2 aree proposte" in said[-1] and len(w.history.entries) == n_undo + 1
        assert records.totals(ch, cid)["area_by_code"].get("PPG.UNNAMED") is None
        records.discard(ch, [s["id"] for s in segs[2:]])
        # a window with no surface: an area outside the face chosen as the window
        away = records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box((7.0, Z0, 8.0, Z0 + 1.0)))
        plugin.refresh_pane()
        pane.select(away["id"])
        n = len(records.features(ch, cid))
        out = propose(detector="cracks", window="selected")
        assert "non ha superficie" in out[-1] and len(records.features(ch, cid)) == n
        # the area chosen as the window clips the proposals: losses around the hole only
        around = records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box(S.HOLE).buffer(0.1))
        plugin.refresh_pane()
        pane.select(around["id"])
        propose(detector="loss", window="selected")
        inside = [f for f in records.features(ch, cid) if f["source"] == "proposed:material_loss"]
        assert len(inside) == 2 and wkt.loads(inside[-1]["geometry"]).within(S.box(S.HOLE).buffer(0.1001))
        assert errors == [], errors
    finally:
        segmenter._proc.kill()
        w.setWindowModified(False)
        w.close()


def test_english_texts(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.gui import i18n

    QApplication.instance() or QApplication([])
    from ppg_conservation import texts
    from ppg_conservation.pane import ConservationPane

    monkeypatch.setenv("PPG_LANG", "en")
    i18n.set_current("en")
    try:
        assert texts.t("Nuova campagna...", "New campaign...") == "New campaign..."
        pane = ConservationPane()
        assert pane.buttons["new_campaign"].text() == "New campaign..."
        assert pane.buttons["classes"].text() == "Classes CC and UC..."
        assert pane.table.horizontalHeaderItem(0).text() == "Term"
        assert assess.class_text("uc", 3, "en") == "UC 3 - Urgent and immediate"
    finally:
        i18n.set_current(None)
