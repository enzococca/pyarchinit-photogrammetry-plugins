"""The 0.3.0 chain from the command line and from the pane's step 8: the relief built and exported, the proposals
read, listed, accepted, rejected and named, the typed keel, the timbers in 3D, the OBJ/GLB/IFC, the plan of the wreck,
the status with the counts; the version and the index entry."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from hulls import framed_project

from ppg_shipwreck import ship, solids, timbers

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import pyarchinit_photogrammetry as ppg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "shipwreck" / "ppg_shipwreck-0.3.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-shipwreck\nVersion: 0.3.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\nshipwreck = ppg_shipwreck:ShipwreckPlugin\n")
    return folder


def test_the_reconstruction_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch, ends = framed_project(tmp_path, heading_deg=35, x_range=(3.0, 9.0))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    doc.save()
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    monkeypatch.setenv("PPG_PLUGINS_ENVIRONMENT", "0")
    p = str(doc.path)
    cli = CliRunner()

    def run(*args, code=0):
        result = cli.invoke(app, ["plugin", "shipwreck", *args])
        assert result.exit_code == code, result.output
        return result.output

    assert "Costruisci prima il rilievo" in run("propose", p, code=1)
    out = run("relief", "build", p, "--cell", "10")
    assert "celle di 10,0 mm" in out
    shade = tmp_path / "ombra.tif"
    run("relief", "export", p, "--kind", "hillshade", "--out", str(shade))
    run("relief", "export", p, "--kind", "height", "--out", str(tmp_path / "quote.tif"), "--world")
    assert shade.exists() and (tmp_path / "quote.tif").exists()
    out = run("propose", p, "--min-step", "8", "--opening", "0.3", "--top-hat", "20")
    assert "Proposte:" in out and "ordinate" in out
    listed = run("trace", "list", p)
    items = ship.state(ppg.Document.open(p).chunk)["proposals"]["items"]
    assert all(i["id"] in listed for i in items)
    seams = [i for i in items if i["kind"] == "seam"]
    stray = next((i for i in seams if "?" in i["name"]), None)
    run("trace", "accept", p, "--kind", "frame")
    run("trace", "accept", p, "--kind", "keelson")
    run("trace", "accept", p, "--kind", "plank")
    if stray is not None:
        run("trace", "reject", p, stray["id"])
    first = next(i for i in seams if "?" not in i["name"])
    run("trace", "accept", p, first["id"], "--name", "comento X1")
    ch = ppg.Document.open(p).chunk
    records = timbers.timbers_of(ch)
    assert {"frame", "keelson", "plank", "seam"} <= {r["kind"] for r in records}
    assert any(r["name"] == "comento X1" for r in records)
    assert stray is None or stray["id"] not in {i["id"] for i in ship.state(ch)["proposals"]["items"]}
    frames = sorted((r for r in records if r["kind"] == "frame"), key=lambda r: float(np.mean(np.asarray(r["ship_xy"])[:, 0])))
    table = tmp_path / "nomi.csv"
    x = float(np.mean(np.asarray(frames[0]["ship_xy"])[:, 0]))
    table.write_text(f"nome;x\nF1;{x:.3f}\n".replace(".", ","))
    assert "1" in run("trace", "names", p, "--table", str(table))
    assert any(r["name"] == "F1" for r in timbers.timbers_of(ppg.Document.open(p).chunk))
    assert "Costruisci prima i legni in 3D" in run("export3d", p, "--out", str(tmp_path / "x.obj"), code=1)
    run("keel", p, "--siding", "120", "--moulding", "150")
    keel = ship.state(ppg.Document.open(p).chunk)["keel"]
    assert keel["siding"] == pytest.approx(0.12) and keel["moulding"] == pytest.approx(0.15)
    out = run("solids", p, "--thickness", "20")
    assert "costruiti" in out
    ch = ppg.Document.open(p).chunk
    assert ship.state(ch)["solids"]["count"] > 0 and not solids.is_stale(ch)
    obj, glb = tmp_path / "out" / "legni.obj", tmp_path / "out" / "legni.glb"
    run("export3d", p, "--out", str(obj))
    run("export3d", p, "--out", str(glb), "--world")
    assert obj.exists() and obj.with_suffix(".mtl").exists() and glb.exists()
    assert (obj.parent / "timbers_3d.csv").exists()
    assert ".obj, .glb" in run("export3d", p, "--out", str(tmp_path / "x.stl"), code=1)
    plan = tmp_path / "pianta"
    out = run("plan", p, "--background", "relief", "--scale", "20", "--out", str(plan))
    assert {f.name for f in plan.iterdir()} >= {"wreck_plan.pdf", "wreck_plan.png", "wreck_plan.svg", "wreck_plan.dxf"}
    status = run("status", p)
    state = json.loads(status)
    built = ship.state(ppg.Document.open(p).chunk)["solids"]["count"]
    assert f"Legni 3D: {built}" in state["relief"]
    run("keel", p, "--none")
    assert "keel" not in ship.state(ppg.Document.open(p).chunk)
    assert "da rifare" in json.loads(run("status", p))["relief"]  # the keel was in the build: the solids are stale


def test_a_cancelled_build_is_flagged_and_an_edit_makes_the_solids_stale(tmp_path):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    from ppg_shipwreck import proposals, relief, tracing

    _doc, ch, ends = framed_project(tmp_path, heading_deg=35, x_range=(3.0, 9.0))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if i["kind"] in ("frame", "keelson")])
    calls = []

    def stop(fraction, message):
        calls.append(message)
        if len(calls) > 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        solids.build(ch, progress=Progress(stop))
    assert ship.state(ch)["solids"]["cancelled"] is True
    out = solids.build(ch)
    assert out["built"] and not ship.state(ch)["solids"].get("cancelled") and not solids.is_stale(ch)
    timbers.timbers_of(ch)[0]["name"] = "F99"
    assert solids.is_stale(ch)


def test_version_and_index_entry():
    import ppg_shipwreck

    assert ppg_shipwreck.__version__ == "0.3.0"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == "0.3.0"
    entry = json.loads((ROOT / "entry.json").read_text())
    assert entry["min_app"] == "1.7.1"
    for text in (entry["description"], entry["i18n"]["it"]["description"]):
        assert len(text) <= 370
    assert re.search(r"3D", entry["description"]) and re.search(r"3D", entry["i18n"]["it"]["description"])


# ------------------------------------------------------------------------------------ the pane
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
    said, chosen = [], []
    monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(f"{kind}: {text}"))
    answers = {}

    def answer(title, fields, intro=""):
        out = {f.name: f.default for f in fields if f.kind != "heading"}
        out.update(answers.get(title, {}))
        return out

    def choose(title, text, options):
        chosen.append(text)
        return answers.get(("choice", title), 0)

    monkeypatch.setattr(plugin, "_ask", answer)
    monkeypatch.setattr(plugin, "_choose", choose)
    try:
        yield w, ch, plugin, wait, answers, said, chosen
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
        w.deleteLater()
        app.document = None


def _features(view, layer):
    lay = view.layers.get(layer)
    return [] if lay is None else lay["features"]


def test_the_pane_builds_the_timbers_exports_them_and_draws_the_plan(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from ppg_shipwreck import PROPOSALS_LAYER, TRACED_LAYER, relief

    w, ch, plugin, wait, answers, said, chosen = window
    pane = plugin.pane
    names = {"solids_build": "Legni 3D...", "keel_set": "Chiglia (dimensioni)...", "model_export": "Esporta 3D...",
             "wreck_plan": "Pianta del relitto..."}
    for name, text in names.items():
        assert pane.buttons[name].text() == text
    answers["Rilievo dello scafo"] = {"cell": 10.0}
    plugin._requested("relief_build")
    wait()
    plugin._requested("proposals_build")
    wait()
    answers["Accetta le proposte"] = {"kind": "all", "score": 0.5}
    plugin._requested("trace_accept")
    assert timbers.timbers_of(ch)

    answers["Chiglia (dimensioni)"] = {"siding": 120.0, "moulding": 150.0}
    plugin._requested("keel_set")
    assert ship.state(ch)["keel"]["siding"] == pytest.approx(0.12)
    w.history.undo()  # the keel is a step of the tracing too
    assert "keel" not in ship.state(ch)
    w.history.redo()
    assert ship.state(ch)["keel"]["moulding"] == pytest.approx(0.15)

    answers["Legni in 3D"] = {"thickness": 20.0}
    plugin._requested("solids_build")
    wait()
    st = ship.state(ch)["solids"]
    assert st["count"] > 0 and "Legni 3D" in pane.states["relief"].text()
    tree = plugin._tree_items(ch)[0]["children"]
    assert any(i["label"] == f"Legni 3D ({st['count']})" for i in tree)

    out = tmp_path / "esporta" / "legni.glb"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    answers["Esporta 3D"] = {"system": "ship", "tables": True}
    plugin._requested("model_export")
    wait()
    assert out.exists() and (out.parent / "timbers_3d.xlsx").exists()
    assert not [s for s in said if s.startswith("warning")]

    # a record changed after the build: the export warns, the pane says so
    timbers.timbers_of(ch)[0]["name"] = "F77"
    plugin._changed()
    assert solids.is_stale(ch) and "da rifare" in pane.states["relief"].text()
    plugin._requested("model_export")
    wait()
    assert any("più vecchi" in s for s in said)

    plugin._requested("wreck_plan")
    wait()
    folder = ship.folder(ch) / "drawings"
    assert (folder / "wreck_plan.pdf").exists() and "wreck_plan" in ship.state(ch)["drawings"]

    # a stale relief: the pane offers to build it again before the plan
    ship.state(ch)["relief"]["model"] = "another"
    assert relief.is_stale(ch)
    answers[("choice", "Pianta del relitto")] = None  # cancel
    plugin._requested("wreck_plan")
    assert chosen and "rilievo" in chosen[-1].lower()

    # deactivate takes the relief layers away
    plugin._requested("relief_show")
    assert _features(w.ortho_view, TRACED_LAYER) or _features(w.ortho_view, PROPOSALS_LAYER)
    plugin.deactivate()
    assert not _features(w.ortho_view, TRACED_LAYER) and not _features(w.ortho_view, PROPOSALS_LAYER)


def test_a_cancelled_build_says_interrupted_in_the_tree(window):
    w, ch, plugin, wait, answers, said, chosen = window
    ship.state(ch)["solids"] = {"built": "2026-11-04T00:00:00+00:00", "count": 3, "refused": [], "cancelled": True,
                                "frame_key": None}
    labels = [i["label"] for i in plugin._tree_items(ch)[0]["children"]]
    assert any(label.startswith("Legni 3D (3)") and "interrotto" in label for label in labels)
    assert "interrotto" in plugin.states(ch)["relief"]


def test_a_mixed_delete_is_one_undo_entry(window):
    from ppg_shipwreck import PROPOSALS_LAYER, TRACED_LAYER

    w, ch, plugin, wait, answers, said, chosen = window
    answers["Rilievo dello scafo"] = {"cell": 10.0}
    plugin._requested("relief_build")
    wait()
    plugin._requested("proposals_build")
    wait()
    from ppg_shipwreck import tracing

    items = ship.state(ch)["proposals"]["items"]
    fr = next(i for i in items if i["kind"] == "frame")
    before = len(w.history.entries)
    tracing.accept(ch, [fr["id"]])
    plugin.draw_layer()
    view = w.ortho_view
    record = timbers.timbers_of(ch)[0]
    plank = next(i for i in ship.state(ch)["proposals"]["items"] if i["kind"] == "plank")
    picked = [f[0] for f in _features(view, PROPOSALS_LAYER) if f[2].get("ppg_id") == plank["id"]]
    picked += [f[0] for f in _features(view, TRACED_LAYER) if f[2].get("ppg_id") == record["id"]]
    view.select_items(picked)
    assert w.delete_selected(confirm=False) == 2
    assert len(w.history.entries) == before + 1
    w.history.undo()
    assert record["id"] in {r["id"] for r in timbers.timbers_of(ch)}
    assert plank["id"] in {i["id"] for i in ship.state(ch)["proposals"]["items"]}
