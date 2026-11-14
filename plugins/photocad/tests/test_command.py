"""The command line: ``ppg plugin photocad key|credit|convert|list|units|export|remove``, on the fake
server; the key typed without echo, never printed; the confirmation as a prompt unless ``--yes``."""
from __future__ import annotations

import json

import fakecad
import pytest

from ppg_photocad import client


@pytest.fixture
def plugin(tmp_path):
    from ppg_photocad import PhotoCADPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    app.document = None
    p = PhotoCADPlugin()
    p.activate(Context("photocad", tmp_path / "plugin"))
    return p


def test_the_plugin_registers_actions_and_command(plugin):
    titles = [a.title for a in plugin.context.contributions.actions]
    assert titles[:3] == ["Vettorializza l'ortofoto...", "Vettorializza il DEM...", "Vettorializza il prospetto mostrato..."]
    assert "Chiave API PhotoCAD..." in titles and "Converti in US / USM..." in titles
    assert list(plugin.context.contributions.commands) == ["photocad"]


def test_key_commands(plugin, monkeypatch, capsys, _no_network_no_keychain, tmp_path):
    import getpass

    monkeypatch.setattr(getpass, "getpass", lambda prompt="": fakecad.KEY)
    assert plugin.command("key", "status") == 0 and "nessuna chiave" in capsys.readouterr().out.lower()
    assert plugin.command("key", "set") == 0
    out = capsys.readouterr().out
    assert fakecad.KEY not in out and _no_network_no_keychain.store
    assert plugin.command("key", "status") == 0
    out = capsys.readouterr().out
    assert "portachiavi" in out and fakecad.KEY not in out and "…cdef" in out
    fake = fakecad.FakePhotoCAD(credits=3)
    monkeypatch.setattr(client, "_urlopen", fake)
    assert plugin.command("credit") == 0 and "Crediti: 3, conversioni gratuite: 5" in capsys.readouterr().out
    assert plugin.command("key", "forget") == 0 and not _no_network_no_keychain.store
    assert plugin.command("credit") == 1 and "PHOTOCAD_API_KEY" in capsys.readouterr().out
    for path in (tmp_path / "plugin").rglob("*"):  # the plugin's settings never hold it
        assert not path.is_file() or fakecad.KEY.encode() not in path.read_bytes()


def test_convert_list_units_export_remove(plugin, monkeypatch, capsys, tmp_path):
    import builtins

    import pyarchinit_photogrammetry as ppg

    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    fake = fakecad.FakePhotoCAD(credits=9)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path / "p", square=[fakecad.SQUARE, fakecad.SQUARE2])
    fakecad.add_elevation(ch, square=[fakecad.E_SQUARE, fakecad.E_SQUARE2])
    doc.save()
    path = str(doc.path)
    # the prompt declined: nothing sent
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    assert plugin.command("convert", path, "--source", "ortho", "--out", str(tmp_path / "a.gpkg")) == 1
    out = capsys.readouterr().out
    assert "photocad.ai" in out and "9" in out and fake.calls("POST") == []
    # accepted at the prompt
    monkeypatch.setattr(builtins, "input", lambda prompt="": "s")
    assert plugin.command("convert", path, "--source", "ortho", "--out", str(tmp_path / "a.gpkg")) == 0
    assert (tmp_path / "a.gpkg").exists() and len(fake.calls("POST")) == 1
    # --yes: no prompt; the elevation in 3D, a part of the plan
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("asked"))
    assert plugin.command("convert", path, "--source", "elevation:e1", "--3d", "--out", str(tmp_path / "e.dxf"),
                          "--yes", "--simplified", "--remove-background") == 0
    assert (tmp_path / "e.dxf").exists() and fake.options["conv_0002"] == {"detailed": "false",
                                                                           "remove_background": "true"}
    x0, y0, x1, y1 = fakecad.square_world()
    assert plugin.command("convert", path, "--source", "ortho", "--bbox", f"{x0 - 1.0},{y0 - 1.0},{x1 + 1.0},{y1 + 1.0}",
                          "--out", str(tmp_path / "b.svg"), "--yes") == 0
    assert (tmp_path / "b.svg").exists()
    assert plugin.command("convert", path, "--source", "elevation:zz", "--yes") == 1
    assert "zz" in capsys.readouterr().out
    capsys.readouterr()
    assert plugin.command("list", path, "--json") == 0
    recs = json.loads(capsys.readouterr().out)
    assert [r["source"] for r in recs] == ["ortho", "elevation:e1", "ortho"]
    # units: on the plan, on the elevation's drawings
    assert plugin.command("units", path, recs[0]["id"], "--tipo", "US", "--start", "5") == 0
    assert "2" in capsys.readouterr().out
    assert plugin.command("units", path, recs[1]["id"], "--tipo", "USM") == 0
    again = ppg.Document.open(path).chunk
    plan = again.meta["drawings"]["units"]
    assert [u["us"] for u in plan] == ["5", "6"] and {u["tipo"] for u in plan} == {"US"}
    frame = again.meta["drawings"]["frames"]["e1"]["units"]
    assert len(frame) == 2 and {u["tipo"] for u in frame} == {"USM"}
    # export and remove
    assert plugin.command("export", path, recs[2]["id"], str(tmp_path / "c.gpkg")) == 0
    assert (tmp_path / "c.gpkg").exists()
    assert plugin.command("remove", path, recs[2]["id"]) == 0
    assert plugin.command("remove", path, recs[2]["id"]) == 1
    for p in tmp_path.rglob("*"):
        assert not p.is_file() or fakecad.KEY.encode() not in p.read_bytes(), p


def test_fetch_accept_places_a_soft_drawing_refused_for_its_residual(plugin, monkeypatch, capsys, tmp_path):
    """``convert`` refuses a drawing on a soft image and says to look at the overlay; ``fetch --accept`` places it
    without sending anything again."""
    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    fake = fakecad.FakePhotoCAD(credits=3, soft_px=4)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path / "p")
    doc.save()
    path = str(doc.path)
    assert plugin.command("convert", path, "--source", "ortho", "--yes") == 1
    out = capsys.readouterr().out
    assert "overlay.jpg" in out and "--accept" in out
    capsys.readouterr()
    assert plugin.command("list", path, "--json") == 0
    (rec,) = json.loads(capsys.readouterr().out)
    assert plugin.command("fetch", path, "--id", rec["id"]) == 1          # without --accept: still refused
    assert plugin.command("fetch", path, "--id", rec["id"], "--accept", "--out", str(tmp_path / "s.gpkg")) == 0
    assert (tmp_path / "s.gpkg").exists() and len(fake.calls("POST")) == 1


def test_units_on_an_elevation_go_through_the_api_and_need_the_elevation(plugin, monkeypatch, capsys, tmp_path):
    """Since program 1.10.0 the units of a conversion of an elevation are kept with it through
    ``context.add_units(elevation=)`` (each with its ``ppg_id``, read back by ``context.elevation_drawings``); once
    the elevation is removed (its drawings with it) the command says so and adds nothing. A program without the API
    gets them written into the elevation's drawings as before."""
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.engine.elevation import remove_elevation

    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    monkeypatch.setattr(client, "_urlopen", fakecad.FakePhotoCAD(credits=9))
    doc, ch = fakecad.project(tmp_path / "p", square=[fakecad.SQUARE, fakecad.SQUARE2])
    fakecad.add_elevation(ch, square=[fakecad.E_SQUARE, fakecad.E_SQUARE2])
    doc.save()
    path = str(doc.path)
    assert plugin.command("convert", path, "--source", "elevation:e1", "--yes") == 0
    capsys.readouterr()
    assert plugin.command("list", path, "--json") == 0
    (rec,) = json.loads(capsys.readouterr().out)
    assert plugin.command("units", path, rec["id"], "--tipo", "USM") == 0
    again = ppg.Document.open(path)
    ppg.app.document = again
    units = plugin.context.elevation_drawings("e1", chunk=again.chunk)["units"]
    assert len(units) == 2 and all(u["id"] for u in units) and {u["tipo"] for u in units} == {"USM"}
    assert not again.chunk.meta["drawings"].get("units")  # nothing among the plan's
    # the elevation removed: its drawings go with it, the units are refused
    assert remove_elevation(again.chunk, "e1")
    again.save()
    capsys.readouterr()
    assert plugin.command("units", path, rec["id"], "--tipo", "USM") == 1
    assert "e1" in capsys.readouterr().out
    assert "frames" not in (ppg.Document.open(path).chunk.meta.get("drawings") or {})
    ppg.app.document = None


def test_units_on_an_elevation_with_an_older_program(plugin, tmp_path):
    """Without ``shown_elevation`` (a program before 1.10.0) the units go straight into the elevation's drawings."""
    import pyarchinit_photogrammetry as ppg  # noqa: F401
    from ppg_photocad import project

    doc, ch = fakecad.project(tmp_path / "p")
    fakecad.add_elevation(ch)

    class OldContext:  # the methods of a 1.9 context the plugin uses here
        def add_units(self, *a, **k):
            raise AssertionError("the plan's add_units used for an elevation")

    plugin.context = OldContext()
    rec = {"id": "c1", "frame": "e1"}
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    orig = project.unit_features, project.unit_polygons
    try:
        project.unit_features = lambda chunk, r, source=None: []
        project.unit_polygons = lambda feats, tipo, start: ([poly], [{}], 0)
        assert plugin.make_units(ch, rec, "USM") == (1, 0)
    finally:
        project.unit_features, project.unit_polygons = orig
    assert len(ch.meta["drawings"]["frames"]["e1"]["units"]) == 1
