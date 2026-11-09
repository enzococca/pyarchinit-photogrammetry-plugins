"""Plans kept in the project (meta, files, undo of a removal), the chunk's system at the boundary, the exports of a
project's plan (terrain refreshed, the DEM's takeoff height for a flat plan) and the command line, with and
without a project, on a local chunk too."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import synth

from ppg_flightplan import design, project

CRS = "EPSG:32632"


def _plugin(tmp_path):
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    from ppg_flightplan import FlightPlanPlugin

    app.document = None
    plugin = FlightPlanPlugin()
    plugin.activate(Context("flightplan", tmp_path / "plugin"))
    return plugin


def _out(capsys) -> dict:
    text = capsys.readouterr().out
    return json.loads(text[text.index("{"):])


def _bbox(w=300.0, h=200.0) -> str:
    lats = [p[0] for p in synth.area_latlon(w, h)]
    lons = [p[1] for p in synth.area_latlon(w, h)]
    return f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"


def _coords(w=300.0, h=200.0) -> str:
    return ";".join(f"{lat},{lon}" for lat, lon in synth.area_latlon(w, h))


def test_round_trip_crs(tmp_path):
    _, ch = synth.chunk(tmp_path)
    xy = np.array(synth.rectangle_xy(812.3, 404.7))
    latlons = project.to_wgs84(ch, xy)
    assert all(40 < lat < 43 and 8 < lon < 10 for lat, lon in latlons)
    back = project.to_chunk(ch, latlons)
    assert np.abs(back - xy).max() < 0.01
    _, local = synth.chunk(tmp_path / "locale", crs=None)
    with pytest.raises(ValueError):
        project.to_chunk(local, latlons)
    with pytest.raises(ValueError):
        project.to_wgs84(local, xy)


def test_plans_in_the_meta(tmp_path):
    import copy

    doc, ch = synth.chunk(tmp_path, dem=100.0)
    piano = design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"]
    a = project.add_plan(ch, piano, "Primo")
    b = project.add_plan(ch, piano, "Secondo")
    assert project.active_plan(ch)["id"] == b and project.active_plan(ch)["name"] == "Secondo"
    project.set_active(ch, a)
    assert project.active_plan(ch)["id"] == a
    old = project.entry(ch, a)["stamp"]
    changed = copy.deepcopy(project.entry(ch, a)["piano"])
    changed["blocchi"][0]["lato"] = 60
    assert project.update_plan(ch, a, changed) != old

    rec = project.coverage_for(ch, a)
    files = [ch.data_dir / f for f in rec["files"].values()]
    assert all(f.is_file() for f in files) and rec["surface"] == "dem"
    assert project.coverage_for(ch, a) is rec                  # current: not made again
    snapshot = copy.deepcopy(ch.meta["flightplan"])            # what the program's undo keeps
    project.remove_plan(ch, a)
    assert project.entry(ch, a) is None and project.active_plan(ch)["id"] == b
    assert not any(f.exists() for f in files)
    assert (project.folder(ch) / ".removed" / a / "copertura_rgba.tif").is_file()

    ch.meta["flightplan"] = snapshot                           # the undo puts the meta back
    project.restore_files(ch)
    assert all(f.is_file() for f in files)
    assert project.coverage_record(ch, a) is not None
    # and the other way round: a plan whose addition is undone leaves its files in .removed
    snapshot.pop("coverage")
    ch.meta["flightplan"] = {"plans": [p for p in snapshot["plans"] if p["id"] != a], "active": b}
    assert project.restore_files(ch)["removed"] == [a]
    assert not (project.folder(ch) / a).exists()


def test_plans_coming_in_are_signed(tmp_path):
    from ppg_flightplan import plan as pl

    _, ch = synth.chunk(tmp_path, dem=100.0)
    piano = design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"]
    b = piano["blocchi"][0]
    points = pl.route_of_block(b, piano)[0].points
    b["seguiTerreno"], b["quote"] = True, [60.0] * len(points)      # a plan of the planner: no signature
    plan_file = project.standalone_save(tmp_path / "vecchio.json", piano)
    loaded = project.standalone_load(plan_file)
    assert loaded["blocchi"][0].get(design.HEIGHTS_SIGNATURE)
    id_ = project.add_plan(ch, piano)
    kept = project.entry(ch, id_)["piano"]["blocchi"][0]
    assert kept.get(design.HEIGHTS_SIGNATURE) and design.HEIGHTS_SIGNATURE not in b    # the caller's dict untouched
    kept["orientamento"] = 180                     # same points in reverse order: caught thanks to the signature
    assert design.blocks_with_old_heights(project.entry(ch, id_)["piano"]) == [0]


def test_per_leg_files_next_to_the_path(tmp_path):
    _, ch = synth.chunk(tmp_path, dem=100.0)
    id_ = project.add_plan(ch, design.design(synth.area_latlon(300, 200), "modello3d", 2.0)["piano"])
    files = project.export(ch, id_, "kmz", tmp_path / "out" / "volo.kmz")
    assert len(files) >= 3 and all(f.parent == tmp_path / "out" and f.suffix == ".kmz" for f in files)
    one = project.export(ch, id_, "litchi", tmp_path / "out" / "litchi.zip")
    names = zipfile.ZipFile(one[0]).namelist()
    assert len([n for n in names if n.endswith(".csv")]) == len(files) and "LEGGIMI.txt" in names


def test_takeoff_ground_and_coverage_ground(tmp_path):
    _, ch = synth.chunk(tmp_path, dem=lambda x, y: 50.0 + (x - synth.X0) * 0.01)
    piano = design.design(synth.area_latlon(300, 200), "orto", 2.0, takeoff=synth.area_latlon(300, 200)[0])["piano"]
    assert project.takeoff_ground(ch, piano) == pytest.approx(50.0 - 1.5, abs=0.1)
    piano["quotaDecollo"] = 77.0
    assert project.takeoff_ground(ch, piano) == 77.0
    assert project.dem_name(ch) == "dem.tif"


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("flightplan")
    assert info is not None and info.active, info and info.error
    assert "flightplan" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "flightplan"] == [".kmz", ".plan", ".csv", ".gpkg",
                                                                                 ".xlsx"]
    manager.unload()


def test_exporter_writes_the_active_plan(tmp_path):
    from ppg_flightplan import FlightPlanPlugin

    _, ch = synth.chunk(tmp_path, dem=100.0)
    project.add_plan(ch, design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"])
    plugin = FlightPlanPlugin()
    files = plugin._exporter("gpkg")(ch, tmp_path / "piano.gpkg", None)
    assert files == [tmp_path / "piano.gpkg"] and files[0].is_file()


def test_command_design_export_coverage(tmp_path, capsys):
    import sqlite3

    doc, ch = synth.chunk(tmp_path, dem=100.0)
    path = str(doc.path)
    plugin = _plugin(tmp_path)
    assert plugin.command("cameras") == 0 and "mini4pro" in capsys.readouterr().out
    assert plugin.command("design", path, "--bbox", _bbox(), "--goal", "orto", "--value", "2") == 0
    designed = _out(capsys)
    assert designed["totale"]["scatti"] > 0 and designed["id"]
    assert plugin.command("show", path) == 0 and '"scatti"' in capsys.readouterr().out

    assert plugin.command("coverage", path) == 0
    rec = _out(capsys)
    assert rec["surface"] == "dem" and rec["stats"]["cells"] > 0
    import pyarchinit_photogrammetry as ppg

    reopened = ppg.Document.open(path).chunk
    assert all((reopened.data_dir / f).is_file() for f in rec["files"].values())
    assert project.coverage_record(reopened, designed["id"]) is not None       # saved in the project

    out = tmp_path / "out" / "legs.zip"
    assert plugin.command("export", path, "--format", "kmz-zip", "--out", str(out)) == 0
    names = zipfile.ZipFile(out).namelist()
    assert any(n.endswith(".kmz") for n in names) and any(n.endswith(".kml") for n in names)
    gp = tmp_path / "out" / "piano.gpkg"
    assert plugin.command("export", path, "--format", "gpkg", "--out", str(gp)) == 0
    with sqlite3.connect(gp) as con:
        tables = {r[0] for r in con.execute("select table_name from gpkg_contents")}
        reference = con.execute("select riferimento_quote from ppg_scatti").fetchone()
    assert {"ppg_area", "ppg_strisciate", "ppg_scatti"} <= tables
    # a flat plan: Z from the project DEM at the takeoff (the first waypoint here), named by its file only
    assert "dem.tif" in reference[0] and "100 m" in reference[0] and str(tmp_path) not in reference[0]
    kml = tmp_path / "out" / "piano.kml"
    assert plugin.command("export", path, "--format", "kml", "--out", str(kml)) == 0
    text = kml.read_text(encoding="utf-8")
    assert "<altitudeMode>absolute</altitudeMode>" in text and "dem.tif" in text and str(tmp_path) not in text
    capsys.readouterr()

    assert plugin.command("list", path) == 0
    listed = _out(capsys)
    assert len(listed["piani"]) == 1 and listed["piani"][0]["copertura"]["cells"] == rec["stats"]["cells"]
    assert plugin.command("use", path, "nessuno") == 1 and "nessuno" in capsys.readouterr().out
    assert plugin.command("remove", path, designed["id"]) == 0
    capsys.readouterr()
    assert plugin.command("list", path) == 0 and _out(capsys)["piani"] == []


def test_export_refreshes_the_terrain(tmp_path, capsys):
    doc, ch = synth.chunk(tmp_path, dem=lambda x, y: 100.0 + (x - synth.X0) * 0.1)
    path = str(doc.path)
    plugin = _plugin(tmp_path)
    assert plugin.command("design", path, "--coords", _coords(), "--terrain") == 0
    id_ = _out(capsys)["id"]
    import pyarchinit_photogrammetry as ppg

    doc2 = ppg.Document.open(path)
    piano = project.entry(doc2.chunk, id_)["piano"]
    assert piano["blocchi"][0].get("quote") and piano["fonteQuote"]["sorgente"] == "dem"
    piano["blocchi"][0]["lato"] = 60                          # changed after its heights
    project.update_plan(doc2.chunk, id_, piano)
    doc2.save()
    assert design.blocks_with_old_heights(piano) == [0]
    out = tmp_path / "x.plan"
    assert plugin.command("export", path, "--format", "plan", "--out", str(out)) == 0
    fresh = project.entry(ppg.Document.open(path).chunk, id_)["piano"]
    assert design.blocks_with_old_heights(fresh) == []           # redone on the DEM and saved
    items = json.loads(out.read_text())["mission"]["items"]
    heights = {round(i["params"][6], 1) for i in items if i.get("command") == 16}
    assert len(heights) > 3                                       # one height per point, following the slope


def test_stale_heights_refused_with_a_message(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    plan_file = tmp_path / "piano.json"
    piano = design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"]
    b = piano["blocchi"][0]
    b["seguiTerreno"] = True
    b["quote"] = [60.0, 61.0]                                     # not the block's points: stale
    project.standalone_save(plan_file, piano)
    assert plugin.command("export", "--plan", str(plan_file), "--format", "litchi", "--out",
                          str(tmp_path / "x.csv")) == 1
    text = capsys.readouterr().out
    assert "ricalcola il terreno" in text and not (tmp_path / "x.csv").exists()


def test_command_without_project(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    plan_file = tmp_path / "plan.json"
    assert plugin.command("design", "--coords", _coords(), "--plan", str(plan_file), "--camera", "air3",
                          "--name", "Senza progetto") == 0
    assert plan_file.is_file() and json.loads(plan_file.read_text())["camera"] == "air3"
    capsys.readouterr()
    out = tmp_path / "x.csv"
    assert plugin.command("export", "--plan", str(plan_file), "--format", "litchi", "--out", str(out)) == 0
    assert out.is_file() and out.read_text().startswith("latitude,longitude,altitude(m)")
    capsys.readouterr()
    assert plugin.command("show", "--plan", str(plan_file)) == 0 and '"scatti"' in capsys.readouterr().out
    assert plugin.command("coverage", "--plan", str(plan_file)) == 1
    assert plugin.command("design", "--coords", _coords()) == 1          # neither a project nor a file
    assert "PROJECT.ppg" in capsys.readouterr().out


def test_manual_camera_and_targets(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    plan_file = tmp_path / "plan.json"
    assert plugin.command("design", "--coords", _coords(), "--plan", str(plan_file),
                          "--manual", "Mia camera,13.2,8.8,8.8,5472") == 0
    piano = json.loads(plan_file.read_text())
    assert piano["cameraPersonale"]["nome"] == "Mia camera" and piano["camera"].startswith("manual-")
    capsys.readouterr()
    lat, lon = synth.area_latlon(100, 100)[0]
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    csv = tmp_path / "target.csv"
    csv.write_text(f"nome,est,nord,quota\nT1,{x:.3f},{y:.3f},101.2\n", encoding="utf-8")
    assert plugin.command("targets", "--plan", str(plan_file), "--csv", str(csv), "--epsg", "32632") == 0
    assert _out(capsys)["target"] == 1
    piano = json.loads(plan_file.read_text())
    assert piano["target"]["punti"][0]["nome"] == "T1"
    assert any(sp.get("target") == "T1" for sp in piano["blocchi"][0]["speciali"])
    assert plugin.command("targets", "--plan", str(plan_file), "--remove") == 0
    assert "target" not in json.loads(plan_file.read_text())


def test_area_from_a_file_and_from_the_chunks_system(tmp_path, capsys):
    doc, ch = synth.chunk(tmp_path)
    path = str(doc.path)
    plugin = _plugin(tmp_path)
    ring = [[lon, lat] for lat, lon in synth.area_latlon(300, 200)]
    area = tmp_path / "area.geojson"
    area.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}]}))
    assert plugin.command("design", path, "--area", str(area)) == 0
    from_file = _out(capsys)["totale"]["scatti"]
    xy = ";".join(f"{x},{y}" for x, y in synth.rectangle_xy(300, 200))
    assert plugin.command("design", path, "--xy", xy) == 0
    assert _out(capsys)["totale"]["scatti"] == from_file
    reopened = ppg_chunk(path)
    assert len(project.plans(reopened)) == 2 and project.active_plan(reopened)["piano"]["crs"] == CRS
    assert plugin.command("design", path, "--xy", xy, "--coords", _coords()) == 1


def test_djifly_zip(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    plan_file = tmp_path / "plan.json"
    assert plugin.command("design", "--coords", _coords(), "--plan", str(plan_file)) == 0
    capsys.readouterr()
    legs = len(json.loads(plan_file.read_text())["blocchi"])
    missions = ",".join(f"{k:08X}-0000-4000-8000-000000000000" for k in range(legs + 2))
    out = tmp_path / "pronte.zip"
    assert plugin.command("djifly", "--plan", str(plan_file), "--zip", str(out), "--missions", missions) == 0
    names = zipfile.ZipFile(out).namelist()
    assert any(n.endswith(".kmz") for n in names) and any(n.startswith("map_preview/") for n in names)


def test_djifly_folder_remembers_the_missions(tmp_path, capsys, monkeypatch):
    from ppg_flightplan import djifly

    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    doc, ch = synth.chunk(tmp_path, dem=100.0)
    path = str(doc.path)
    plugin = _plugin(tmp_path)
    assert plugin.command("design", path, "--coords", _coords()) == 0
    capsys.readouterr()
    waypoint = tmp_path / "telecomando" / "waypoint"
    slots = [f"{k:08X}-0000-4000-8000-000000000000" for k in range(2)]
    for s in slots:
        (waypoint / s).mkdir(parents=True)
    assert plugin.command("djifly", path, "--folder", str(waypoint), "--missions", ",".join(slots)) == 0
    assert (waypoint / slots[0] / f"{slots[0]}.kmz").is_file()
    assert (waypoint / "map_preview" / slots[0] / f"{slots[0]}.jpg").is_file()
    given = project.active_plan(ppg_chunk(path))["piano"]["djiFly"]
    assert [v["cartella"] for v in given.values()] == slots[:len(given)]
    # a mission of the user in the slot: nothing written without --overwrite
    import io

    from ppg_flightplan import export_dji
    from ppg_flightplan.camera import camera
    from ppg_flightplan.route import Route

    many = Route("utente", [(45.0 + k * 1e-4, 10.0) for k in range(5)], 50.0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("wpmz/template.kml", export_dji.template_kml(many, camera("mini4pro"), 0, author="utente"))
        z.writestr("wpmz/waylines.wpml", export_dji.waylines_wpml(many, camera("mini4pro")))
    (waypoint / slots[0] / f"{slots[0]}.kmz").write_bytes(buffer.getvalue())
    assert plugin.command("djifly", path, "--folder", str(waypoint), "--missions", ",".join(slots)) == 1
    assert "--overwrite" in capsys.readouterr().out


def test_local_chunk(tmp_path, capsys):
    doc, ch = synth.chunk(tmp_path, crs=None)
    path = str(doc.path)
    plugin = _plugin(tmp_path)
    assert plugin.command("design", path, "--coords", _coords()) == 0
    assert "crs" not in project.active_plan(ppg_chunk(path))["piano"]
    capsys.readouterr()
    assert plugin.command("coverage", path) == 1
    assert "serve un sistema di riferimento" in capsys.readouterr().out
    assert plugin.command("export", path, "--format", "gpkg", "--out", str(tmp_path / "x.gpkg")) == 1
    assert "serve un sistema di riferimento" in capsys.readouterr().out
    assert plugin.command("export", path, "--format", "litchi", "--out", str(tmp_path / "x.csv")) == 0
    assert plugin.command("design", path, "--xy", "0,0;10,0;10,10") == 1


def ppg_chunk(path):
    import pyarchinit_photogrammetry as ppg

    return ppg.Document.open(path).chunk
