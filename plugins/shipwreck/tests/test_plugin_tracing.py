"""The relief and tracing step in the window (offscreen, the forms answered by the test): the relief built and shown
as an elevation, the hillshade over it, the proposals and the traced timbers as layers of the raster view only while
the relief is shown, accepted, rejected (also by deleting the feature), added and corrected by hand, flagged loose,
each change one step of *Modifica > Annulla*, and the items of the workspace tree."""
from __future__ import annotations

import numpy as np
import pytest
from hulls import framed_project

from ppg_shipwreck import relief, ship, timbers

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ppg_shipwreck import ShipwreckPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, ends = framed_project(tmp_path, heading_deg=35, x_range=(3.0, 9.0))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    errors, infos = [], []
    monkeypatch.setattr(w, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(w, "_info", lambda title, text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "_warn", lambda text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "choice", lambda *a, **k: 0, raising=False)

    def wait():
        w.runner.wait()
        QApplication.processEvents()
        assert not errors, "a task of the window failed:\n" + "\n".join(errors) + "\nsaid: " + "\n".join(infos)

    plugin = ShipwreckPlugin()
    plugin.activate(Context("shipwreck", tmp_path / "plugin", window=w))
    said = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(f"{kind}: {text}"))
    answers = {}

    def answer(title, fields, intro=""):
        out = {f.name: f.default for f in fields if f.kind != "heading"}
        out.update(answers.get(title, {}))
        return out

    monkeypatch.setattr(plugin, "_ask", answer)
    try:
        yield w, ch, plugin, wait, answers, said
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
        w.deleteLater()
        app.document = None


def _features(view, layer):
    lay = view.layers.get(layer)
    return [] if lay is None else lay["features"]


def _select(view, layer, ppg_ids):
    items = [f[0] for f in _features(view, layer) if f[2].get("ppg_id") in ppg_ids]
    view.select_items(items)
    return items


def _ready(plugin, wait, answers, propose=True):
    """The relief built (5 mm) and, with ``propose``, the proposals read, through the pane."""
    answers["Rilievo dello scafo"] = {"cell": 5.0}
    plugin._requested("relief_build")
    wait()
    if propose:
        plugin._requested("proposals_build")
        wait()


def _plain_frame(plugin, ch):
    xyz = ship.world_vertices(ch)
    lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    plugin.context.show_frame((lo[0], lo[1], hi[0], hi[1]), 0.01, "piano")


def test_the_relief_built_and_the_proposals_read(window):
    from ppg_shipwreck import PROPOSALS_LAYER

    w, ch, plugin, wait, answers, _said = window
    pane = plugin.pane
    names = ("relief_build", "relief_show", "proposals_build", "trace_accept", "trace_reject", "trace_correct",
             "trace_add", "trace_names", "trace_loose")
    assert all(n in pane.buttons for n in names)
    assert pane.buttons["relief_build"].text() == "Rilievo dello scafo..."
    assert pane.buttons["trace_loose"].text() == "Sciolto"
    assert "Non costruito" in pane.states["relief"].text()
    _ready(plugin, wait, answers, propose=False)
    entry = relief.entry(ch)
    assert entry is not None and w.drawings_frame == entry["id"]
    assert ship.state(ch)["relief"]["cell"] == pytest.approx(0.005)
    assert any(i["label"] == "Relitto: rilievo" for i in plugin._tree_items(ch)[0]["children"])
    plugin._requested("proposals_build")
    wait()
    items = ship.state(ch)["proposals"]["items"]
    drawn = _features(w.ortho_view, PROPOSALS_LAYER)
    assert items and len(drawn) == len(items)
    assert {f[2]["ppg_id"] for f in drawn} == {i["id"] for i in items}
    assert drawn[0][0].pen().style().name == "DashLine"
    assert "Proposte:" in pane.states["relief"].text()
    assert any(i["label"] == f"Proposte ({len(items)})" for i in plugin._tree_items(ch)[0]["children"])
    w.history.undo()  # the reading itself is a step
    assert "proposals" not in ship.state(ch) and not _features(w.ortho_view, PROPOSALS_LAYER)


def test_accept_reject_add_correct_loose_and_undo(window, monkeypatch):
    from ppg_shipwreck import PROPOSALS_LAYER, TRACED_LAYER

    w, ch, plugin, wait, answers, said = window
    view = w.ortho_view
    _ready(plugin, wait, answers)
    items = ship.state(ch)["proposals"]["items"]

    # accepted from the selection on the view: a record, drawn in the traced layer, one undo step
    frame_item = next(i for i in items if i["kind"] == "frame")
    _select(view, PROPOSALS_LAYER, {frame_item["id"]})
    plugin._requested("trace_accept")
    records = timbers.timbers_of(ch)
    assert [r["source"] for r in records] == [frame_item["id"]]
    traced = _features(view, TRACED_LAYER)
    assert len(traced) == 1 and traced[0][2]["ppg_id"] == records[0]["id"]
    assert traced[0][1] is not None and traced[0][1].text() == records[0]["name"]
    assert len(_features(view, PROPOSALS_LAYER)) == len(items) - 1
    assert plugin.pane.table.rowCount() == 1
    w.history.undo()
    assert timbers.timbers_of(ch) == []
    assert frame_item["id"] in {i["id"] for i in ship.state(ch)["proposals"]["items"]}
    assert len(_features(view, PROPOSALS_LAYER)) == len(items) and not _features(view, TRACED_LAYER)
    w.history.redo()
    assert [r["source"] for r in timbers.timbers_of(ch)] == [frame_item["id"]]

    # a proposal deleted on the view is rejected
    plank = next(i for i in ship.state(ch)["proposals"]["items"] if i["kind"] == "plank")
    _select(view, PROPOSALS_LAYER, {plank["id"]})
    assert w.delete_selected(confirm=False) == 1
    prop = ship.state(ch)["proposals"]
    assert plank["id"] not in {i["id"] for i in prop["items"]} and prop["rejected"]
    w.history.undo()
    assert plank["id"] in {i["id"] for i in ship.state(ch)["proposals"]["items"]}
    assert plank["id"] in {f[2]["ppg_id"] for f in _features(view, PROPOSALS_LAYER)}

    # a timber traced by hand on the relief: the view gives (u, v), the record has the chunk's coordinates
    uv = np.array([[5.0, 0.20], [5.6, 0.20], [5.6, 0.30], [5.0, 0.30]])
    monkeypatch.setattr(plugin.context, "request_polygon", lambda finished, *a, **k: finished(uv) or True)
    answers["Legno ricalcato"] = {"kind": "stringer", "side": "auto", "name": "S9", "geometry": "polygon"}
    plugin._requested("trace_add")
    added = timbers.timbers_of(ch)[-1]
    assert added["name"] == "S9" and added["kind"] == "stringer" and added["status"] == "measured"
    frame = ship.frame_of(ch)
    xy = np.asarray(added["geometry"]["xy"])
    back = relief.uv_of_ship(frame.to_ship(np.column_stack([xy, np.zeros(len(xy))]))[:, :2], frame.sign)
    assert np.abs(back - uv).max() < 1e-6
    assert added["id"] in {f[2]["ppg_id"] for f in _features(view, TRACED_LAYER)}
    n = len(timbers.timbers_of(ch))
    w.history.undo()
    assert len(timbers.timbers_of(ch)) == n - 1
    w.history.redo()
    assert len(timbers.timbers_of(ch)) == n

    # the view's selection wins over a row left selected in the table
    plugin.pane.table.selectRow(0)  # the frame accepted first
    assert plugin.pane.selected_ids() == [records[0]["id"]]
    _select(view, TRACED_LAYER, {added["id"]})
    moved = uv + [0.0, 0.05]
    monkeypatch.setattr(plugin.context, "request_polygon", lambda finished, *a, **k: finished(moved) or True)
    plugin._requested("trace_correct")
    assert not said, said
    rec = next(r for r in timbers.timbers_of(ch) if r["id"] == added["id"])
    assert rec["corrected"]
    assert np.asarray(rec["ship_xy"]) == pytest.approx(relief.ship_of_uv(moved, frame.sign), abs=1e-9)

    # flagged loose and back (the frame of the table untouched)
    plugin.pane.table.selectRow(0)
    _select(view, TRACED_LAYER, {added["id"]})
    plugin._requested("trace_loose")
    rec = next(r for r in timbers.timbers_of(ch) if r["id"] == added["id"])
    assert rec["loose"] and rec["kind"] == "other"
    assert not next(r for r in timbers.timbers_of(ch) if r["id"] == records[0]["id"]).get("loose")
    _select(view, TRACED_LAYER, {added["id"]})
    plugin._requested("trace_loose")
    rec = next(r for r in timbers.timbers_of(ch) if r["id"] == added["id"])
    assert not rec["loose"] and rec["kind"] == "stringer"

    # a traced timber deleted on the view leaves the catalogue
    _select(view, TRACED_LAYER, {added["id"]})
    assert w.delete_selected(confirm=False) == 1
    assert added["id"] not in {r["id"] for r in timbers.timbers_of(ch)}
    w.history.undo()
    assert added["id"] in {r["id"] for r in timbers.timbers_of(ch)}
    assert any(i["label"].startswith("Legni (2): ") for i in plugin._tree_items(ch)[0]["children"])


def test_names_that_rename_nothing_make_no_undo_step(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from ppg_shipwreck import tracing

    w, ch, plugin, wait, answers, _said = window
    _ready(plugin, wait, answers)
    fr = next(i for i in ship.state(ch)["proposals"]["items"] if i["kind"] == "frame")
    tracing.accept(ch, [fr["id"]])
    record = timbers.timbers_of(ch)[0]
    record["frame_key"] = None  # a plan cache of another frame: refreshed by the renaming, no change of the user's
    table = tmp_path / "nomi.csv"
    table.write_text("nome;x\nF99;500\n")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(table), ""))
    entries = len(w.history.entries)
    plugin._requested("trace_names")
    assert record["frame_key"] is not None and record["name"] == fr["name"]
    assert len(w.history.entries) == entries


def test_a_rebuild_follows_the_new_entry(window):
    from ppg_shipwreck import TRACED_LAYER

    w, ch, plugin, wait, answers, _said = window
    _ready(plugin, wait, answers)
    plugin._requested("trace_accept")  # none selected: the form, every proposal from score 0.5
    assert timbers.timbers_of(ch)
    old = relief.entry(ch)["id"]
    plugin._requested("relief_build")
    wait()
    new = relief.entry(ch)["id"]
    assert new != old and w.drawings_frame == new
    assert old not in ((ch.meta.get("drawings") or {}).get("frames") or {})
    assert {f[2]["ppg_id"] for f in _features(w.ortho_view, TRACED_LAYER)} == {r["id"] for r in timbers.timbers_of(ch)}


def test_another_raster_takes_the_relief_layers_away(window):
    from ppg_shipwreck import PROPOSALS_LAYER, TRACED_LAYER

    w, ch, plugin, wait, answers, _said = window
    view = w.ortho_view
    _ready(plugin, wait, answers)
    first = ship.state(ch)["proposals"]["items"][0]["id"]
    _select(view, PROPOSALS_LAYER, {first})
    plugin._requested("trace_accept")
    _plain_frame(plugin, ch)
    assert w.drawings_frame is None
    assert not _features(view, PROPOSALS_LAYER) and not _features(view, TRACED_LAYER)
    assert _features(view, "Legni del relitto")  # the plan's timbers, in the chunk's coordinates
    answers["Mostra il rilievo"] = {"kind": "photo"}
    plugin._requested("relief_show")
    assert _features(view, PROPOSALS_LAYER) and _features(view, TRACED_LAYER)
    assert not _features(view, "Legni del relitto")  # never in (u, v)


def test_the_overlay_shown_removed_kept_and_read_at_the_views_size(window, monkeypatch):
    import rasterio

    from ppg_shipwreck import overlay as overlay_mod
    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    w, ch, plugin, wait, answers, _said = window
    view = w.ortho_view
    _ready(plugin, wait, answers, propose=False)
    answers["Mostra il rilievo"] = {"kind": "hillshade"}
    plugin._requested("relief_show")
    assert plugin.overlay is not None and plugin.overlay.shown()
    assert plugin.overlay.item.zValue() == -0.5
    # an edit redraws the layers: the overlay is not read again
    hillshade = str(relief.path_of(ch, "hillshade"))
    real_open, opened = rasterio.open, []

    def counting(path, *a, **k):
        if str(path) == hillshade:
            opened.append(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr(rasterio, "open", counting)
    plugin.draw_layer()
    assert plugin.overlay.shown() and opened == []
    # the relief at a sixth of its size in the view: the raster read at that size, not whole
    reads = []
    real_read = overlay_mod._read

    def recording(path, factor):
        values, tr = real_read(path, factor)
        reads.append(values.shape)
        return values, tr

    monkeypatch.setattr(overlay_mod, "_read", recording)
    full = relief.load(ch, "hillshade")[0].shape
    assert full[0] * full[1] > 1_500_000
    view.load(elevation_paths(ch, relief.entry(ch))[0], "ortho", max_side=400)
    assert plugin.overlay.shown() and reads
    assert reads[-1][0] * reads[-1][1] < full[0] * full[1] / 10
    # gone with another raster, back with the relief, removed with "photo"
    _plain_frame(plugin, ch)
    assert not plugin.overlay.shown()
    plugin._requested("relief_show")
    assert plugin.overlay.shown()
    answers["Mostra il rilievo"] = {"kind": "photo"}
    plugin._requested("relief_show")
    assert not plugin.overlay.shown()


def test_the_overlay_stretches_to_grey(tmp_path):
    from ppg_shipwreck.overlay import grey

    values = np.linspace(0.0, 100.0, 101).reshape(1, -1)
    values[0, 0] = np.nan
    rgba = grey(values)
    assert rgba.shape == (1, 101, 4)
    assert rgba[0, 0, 3] == 0 and (rgba[0, 1:, 3] == 255).all()
    assert rgba[0, 2, 0] == 0 and rgba[0, -2, 0] == 255  # 2 and 98 % the ends of the ramp
    assert (rgba[0, 1:, 0] == rgba[0, 1:, 1]).all() and (rgba[0, 1:, 1] == rgba[0, 1:, 2]).all()
