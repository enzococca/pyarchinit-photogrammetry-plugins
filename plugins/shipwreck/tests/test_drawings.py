"""The catalogue, the drawings and the plugin: measures from geometry and model, room and space, exports, the sheets
(scale, what they say, a plan that is not a mirror image, DXF layers), the command line through the program's ``ppg
plugin``, the pane in the program's window."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from hulls import D, half_breadth, project

import pyarchinit_photogrammetry as ppg  # noqa: F401
from ppg_shipwreck import drawings, lines, ship, timbers


def _ready(tmp_path, heading=37.0, bow="end", water=False, **kw):
    doc, ch, ends = project(tmp_path, heading_deg=heading, **kw)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], bow))
    ship.set_stations(ch, ship.regular(2.0, 5, start=2.0))
    lines.build(ch, longitudinals=[0.0, 0.5], horizontals=[-0.5, -1.0], half_width=3.0)
    if water:
        lines.set_water(ch, -0.5, on=True)
    return doc, ch, ship.frame_of(ch)


def _world(frame, *points):
    return frame.to_world(np.array([[x, y, 0.0] for x, y in points]))[:, :2]


def test_measures_from_the_geometry_and_the_model(tmp_path):
    _doc, ch, frame = _ready(tmp_path)
    timbers.add(ch, "keel", "centre", "K", timbers.geometry("line", _world(frame, (2.0, 0.0), (10.0, 0.0))))
    s = half_breadth(6.0) * 0.5
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("line", _world(frame, (6.0, 0.0), (6.0, s))), station="S3")
    timbers.add(ch, "plank", "port", "P", timbers.geometry("polygon", _world(frame, (3.0, 0.1), (5.0, 0.1), (5.0, 0.4), (3.0, 0.4))))
    timbers.add(ch, "fastening", "port", "N", timbers.geometry("point", _world(frame, (6.0, 0.5))))
    rows = {r["name"]: r for r in timbers.measure(ch)}
    assert rows["K"]["length"] > 8.0 and rows["K"]["z_min"] == pytest.approx(-D, abs=0.01)  # the keel follows the bottom
    # along the frame, on the curve of the section: z = D (y^2 / b^2 - 1) from y = 0 to b / 2
    b = half_breadth(6.0)
    y = np.linspace(0.0, s, 2000)
    truth = np.hypot(np.diff(y), np.diff(D * (y**2 / b**2 - 1.0))).sum()
    assert rows["F1"]["length"] == pytest.approx(truth, abs=0.01)
    assert rows["F1"]["width"] is None and rows["F1"]["x"] == pytest.approx(6.0)
    assert rows["P"]["length"] == pytest.approx(2.0, abs=1e-6) and rows["P"]["width"] == pytest.approx(0.3, abs=1e-6)
    assert rows["N"]["length"] is None
    assert rows["N"]["z_max"] == pytest.approx(D * (0.25 / b**2 - 1.0), abs=0.01)
    above = {r["name"]: r for r in timbers.measure(ch, datum=-D)}
    assert above["K"]["z_min"] == pytest.approx(0.0, abs=0.01)


def test_the_water_also_corrects_the_heights_of_the_timbers(tmp_path):
    _doc, ch, frame = _ready(tmp_path, water=True)
    timbers.add(ch, "keel", "centre", "K", timbers.geometry("line", _world(frame, (5.0, 0.0), (7.0, 0.0))))
    r = timbers.measure(ch)[0]
    assert r["z_min_water"] == pytest.approx(-0.5 - (-0.5 - r["z_min"]) * 1.34, abs=1e-6)


def test_room_and_space(tmp_path):
    _doc, ch, frame = _ready(tmp_path)
    for name, x, side in (("F1", 3.0, "port"), ("F2", 3.4, "port"), ("F3", 3.9, "port"), ("F1", 3.0, "starboard"), ("M", 3.2, "centre")):
        timbers.add(ch, "frame", side, name, timbers.geometry("point", _world(frame, (x, 0.3 if side == "port" else -0.3))))
    timbers.add(ch, "plank", "port", "P", timbers.geometry("point", _world(frame, (3.1, 0.3))))
    spacing = timbers.frame_spacing(timbers.measure(ch))
    assert spacing["port"]["spaces"] == pytest.approx([0.2, 0.2, 0.5])
    assert spacing["port"]["median"] == pytest.approx(0.2)
    assert [f["name"] for f in spacing["starboard"]["frames"]] == ["F1", "M"]
    stations = timbers.stations_from_catalogue(ch)
    assert {s["name"]: s["x"] for s in stations}["F1"] == pytest.approx(3.0)
    assert len(stations) == 4


def test_table_of_timbers_and_exports(tmp_path):
    import rasterio  # noqa: F401 - the program's stack
    from openpyxl import load_workbook

    from pyarchinit_photogrammetry.io import gpkg

    _doc, ch, frame = _ready(tmp_path)
    table = tmp_path / "t.csv"
    table.write_text("tipo;lato;nome;x1;y1;x2;y2;note\nframe;port;F36;4;0;4;0,5;rotta\nframe;starboard;F36;4;0;4;-0,5;\n", encoding="utf-8")
    assert timbers.import_csv(ch, table) == 2
    (tmp_path / "p.csv").write_text("kind,side,name,wkt\nkeelson,centre,KS,\"POINT (%f %f)\"\n" % tuple(_world(frame, (5.0, 0.0))[0]))
    timbers.import_csv(ch, tmp_path / "p.csv")
    rows = timbers.measure(ch)
    assert rows[0]["notes"] == "rotta" and rows[0]["x"] == pytest.approx(4.0)
    with open(timbers.export_any(ch, tmp_path / "a.csv")) as f:
        assert len(list(csv.DictReader(f))) == 3
    book = load_workbook(timbers.export_any(ch, tmp_path / "a.xlsx"))
    assert book.sheetnames == ["Legni", "Maglia"] and book["Legni"]["B2"].value == "ordinata"
    path = timbers.export_any(ch, tmp_path / "a.gpkg")
    assert len(gpkg.read_table(path, "timbers_line")) == 2 and len(gpkg.read_table(path, "timbers_point")) == 1
    with pytest.raises(ship.NotReady):
        timbers.export_any(ch, tmp_path / "a.txt")
    (tmp_path / "bad.csv").write_text("tipo;nome;x;y\nvela;A;1;2\n", encoding="utf-8")
    with pytest.raises(ship.NotReady, match="Riga 2"):
        timbers.import_csv(ch, tmp_path / "bad.csv")
    assert len(timbers.timbers_of(ch)) == 3  # a table with a bad line adds nothing


def test_the_drawings_are_written_with_their_notes(tmp_path):
    _doc, ch, frame = _ready(tmp_path, water=True)
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("line", _world(frame, (6.0, 0.0), (6.0, 0.5))))
    ch.meta[ship.KEY]["title"] = {"site": "Prova", "vessel": "Scafo A"}
    written = drawings.write_all(ch)
    names = {p.name for p in written}
    for kind in drawings.KINDS:
        assert {f"{kind}.svg", f"{kind}.pdf", f"{kind}.png", f"{kind}.dxf"} <= names
    assert all(p.parent == ch.data_dir / "shipwreck" / "drawings" for p in written)
    svg = (written[0].parent / "body_plan.svg").read_text()
    assert "1,34" in svg and "Prova" in svg and "carenatura" in svg
    d = drawings.make(ch, "body_plan")
    assert d.scale == 20
    # raw line kept under the corrected one, the level drawn
    assert {"STATIONS", "RAW", "LEVEL", "KEEL"} == {layer for layer, _p, _c in d.lines}
    import ezdxf

    doc = ezdxf.readfile(written[0].parent / "body_plan.dxf")
    used = {e.dxf.layer for e in doc.modelspace()}
    assert {"STATIONS", "RAW", "LEVEL", "TEXT"} <= used
    plan = ezdxf.readfile(written[0].parent / "plan.dxf")
    assert "TIMBERS" in {e.dxf.layer for e in plan.modelspace()}
    assert ch.meta[ship.KEY]["drawings"]["profile"]["water"] is True


def _sheet_text(ch, kind) -> str:
    fig = drawings.figure(drawings.make(ch, kind))
    return " ".join(x.get_text() for x in fig.texts)


def test_a_sheet_says_what_it_shows(tmp_path):
    _doc, ch, frame = _ready(tmp_path, bow="start")
    text = _sheet_text(ch, "body_plan")
    assert "±3,00 m" in text and "Prua" in text and "x = 0,00" in text and "Datum" in text
    assert "Tratteggio" in text and "non tagliati dal modello" in text
    assert "layer RAW" not in text and "verticale" not in text
    lines.set_water(ch, -0.5, on=True)
    assert "layer RAW" in _sheet_text(ch, "body_plan") and "layer RAW" not in _sheet_text(ch, "half_breadth")
    ch.crs = "LOCAL"
    ship.state(ch)["frame"]["crs"] = "LOCAL"
    lines.build(ch, half_width=1.0)
    ship.set_stations(ch, [{"name": "F99", "x": 40.0}, {"name": "S", "x": 6.0}])
    lines.build(ch, longitudinals=[0.0], horizontals=[-0.5], half_width=1.0)
    text = _sheet_text(ch, "body_plan").replace("\n", " ")
    assert "F99" in text and "non verificato come verticale" in text and "continua oltre: S" in text
    assert drawings.make(ch, "body_plan").north is None


def test_long_notes_stay_on_the_sheet(tmp_path):
    _doc, ch, frame = _ready(tmp_path)
    ship.state(ch)["title"] = {"notes": "parola " * 80}
    fig = drawings.figure(drawings.make(ch, "body_plan"))
    width = fig.get_size_inches()[0] * fig.dpi
    renderer = fig.canvas.get_renderer()
    assert all(x.get_window_extent(renderer).x1 <= width for x in fig.texts)


def test_drawings_out_of_date_are_said(tmp_path):
    from ppg_shipwreck import ShipwreckPlugin

    _doc, ch, frame = _ready(tmp_path)
    drawings.write_all(ch)
    assert drawings.out_of_date(ch) == [] and "Da rifare" not in ShipwreckPlugin.states(ch)["drawings"]
    lines.set_water(ch, -0.5, on=True)  # the water
    assert set(drawings.out_of_date(ch)) == set(drawings.KINDS) - {"wreck_plan"} and "Da rifare" in ShipwreckPlugin.states(ch)["drawings"]
    lines.set_water(ch, -0.5, on=False)
    assert drawings.out_of_date(ch) == []
    timbers.add(ch, "frame", "port", "F", timbers.geometry("point", _world(frame, (6.0, 0.5))))  # the plan only
    assert drawings.out_of_date(ch) == ["plan"]
    drawings.write_all(ch)
    ship.set_frame(ch, frame.flipped())  # the frame
    assert set(drawings.out_of_date(ch)) == set(drawings.KINDS)
    ship.set_frame(ch, frame)
    lines.build(ch, half_width=2.5)  # sections cut again
    assert set(drawings.out_of_date(ch)) == set(drawings.KINDS)


def test_a_side_against_its_y_is_said(tmp_path):
    from ppg_shipwreck import ShipwreckPlugin

    _doc, ch, frame = _ready(tmp_path)
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("point", _world(frame, (6.0, 0.5))))
    assert timbers.measure(ch)[0]["warning"] == ""
    ship.set_frame(ch, frame.flipped())  # x from the other end: the sides stay
    assert timbers.measure(ch)[0]["warning"] == ""
    ship.set_frame(ch, frame.other_bow())  # the bow at the other end: they change
    assert "babordo" in timbers.measure(ch)[0]["warning"]
    assert "F1" in ShipwreckPlugin.states(ch)["timbers"]
    path = timbers.export_any(ch, tmp_path / "a.csv")
    assert "dall'altro lato" in path.read_text()


def test_a_catalogue_without_a_model(tmp_path):
    from ppg_shipwreck import ShipwreckPlugin

    _doc, ch, frame = _ready(tmp_path)
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("line", _world(frame, (6.0, 0.5), (6.0, 1.0))))
    (ch.data_dir / "model.ply").unlink()
    row = timbers.measure(ch)[0]
    assert row["x"] == pytest.approx(6.0) and row["length"] == pytest.approx(0.5) and row["z_max"] is None
    assert timbers.stations_from_catalogue(ch) == [{"name": "F1", "x": pytest.approx(6.0)}]
    assert ShipwreckPlugin.states(ch)["timbers"].startswith("1 legno.")


def test_a_kind_gone_from_the_catalogue_leaves_the_geopackage(tmp_path):
    from pyarchinit_photogrammetry.io import gpkg

    _doc, ch, frame = _ready(tmp_path)
    first = timbers.add(ch, "fastening", "port", "N", timbers.geometry("point", _world(frame, (6.0, 0.5))))
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("line", _world(frame, (6.0, 0.5), (6.0, 1.0))))
    path = timbers.export_any(ch, tmp_path / "a.gpkg")
    assert len(gpkg.read_table(path, "timbers_point")) == 1
    timbers.remove(ch, [first["id"]])
    timbers.export_any(ch, path)
    assert gpkg.read_table(path, "timbers_point") == [] and len(gpkg.read_table(path, "timbers_line")) == 1


def test_the_orthophoto_on_the_plan_lands_on_its_pixels(tmp_path):
    """One bright pixel of a coarse orthophoto: on the sheet its centre is where its world position is."""
    import rasterio
    from rasterio.transform import from_origin

    _doc, ch, frame = _ready(tmp_path, heading=90.0, bow="end")
    centre = frame.to_world(np.array([[6.0, 0.0, 0.0]]))[0]
    cell, n = 0.1, 41
    x0, y1 = centre[0] - (n / 2) * cell, centre[1] + (n / 2) * cell  # the pixel n // 2 has its centre on the point
    image = np.zeros((3, n, n), np.uint8)
    image[:, n // 2, n // 2] = 255
    path = ch.data_dir / "orthomosaic.tif"
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=ch.crs,
                       transform=from_origin(x0, y1, cell, cell)) as dst:
        dst.write(image)
    ch._set_product("orthomosaic", path)
    rgba, (a, b, c, e) = drawings.ortho_in_ship(ch, frame, (4.0, 8.0), (-2.0, 2.0), 0.01)
    weight = rgba[..., 0].astype(float)
    rows, cols = np.indices(weight.shape)
    x = a + (cols + 0.5) * (b - a) / rgba.shape[1]
    y = e - (rows + 0.5) * (e - c) / rgba.shape[0]
    assert (weight * x).sum() / weight.sum() == pytest.approx(6.0, abs=0.006)
    assert (weight * y).sum() / weight.sum() == pytest.approx(0.0, abs=0.006)


def test_the_scale_is_exact_on_the_paper(tmp_path):
    _doc, ch, frame = _ready(tmp_path)
    d = drawings.make(ch, "body_plan", scale=50)
    fig = drawings.figure(d)
    ax = fig.axes[0]
    width_mm = ax.get_position().width * fig.get_size_inches()[0] * 25.4
    x0, x1, _y0, _y1 = d.extent
    assert width_mm == pytest.approx((x1 - x0) * 1000.0 / 50, rel=1e-6)


def test_a_plan_is_never_a_mirror_image(tmp_path):
    """On the sheet the left of x is up, whichever way port lies: the world's north of an x to the east."""
    for bow in ("start", "end"):
        (tmp_path / bow).mkdir()
        _doc, ch, frame = _ready(tmp_path / bow, heading=90.0, bow=bow)  # x to the east: north is up on the sheet
        d = drawings.make(ch, "half_breadth")
        assert d.north == pytest.approx((0.0, 1.0), abs=1e-9)
        world = np.array([[*frame.to_world(np.array([[6.0, 0.0, 0.0]]))[0][:2], 0.0]]) + [0.0, 1.0, 0.0]
        assert frame.to_ship(world)[0, 1] * frame.sign > 0


def test_a_chunk_with_a_stale_or_missing_input_is_told(tmp_path):
    _doc, ch, frame = _ready(tmp_path)
    ship.set_frame(ch, frame.flipped())
    with pytest.raises(ship.NotReady, match="ricalcolale"):
        drawings.write_all(ch)
    drawings_folder = ch.data_dir / "shipwreck" / "drawings"
    assert not drawings_folder.exists() or not list(drawings_folder.iterdir())


def test_a_drawing_with_no_lines_is_left_out(tmp_path):
    doc, ch, ends = project(tmp_path)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1]))
    ship.set_stations(ch, [{"name": "S", "x": 6.0}])
    lines.build(ch, half_width=3.0)
    names = {p.stem for p in drawings.write_all(ch)}
    assert names == {"body_plan", "plan"}


def test_the_plan_over_the_orthophoto(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    _doc, ch, frame = _ready(tmp_path, heading=30.0)
    # an orthophoto whose east half is red: on the sheet the port (+y) and starboard sides of an x to the
    # north-north-east show which side is which
    centre = frame.to_world(np.array([[6.0, 0.0, 0.0]]))[0]
    cell, n = 0.02, 1000
    x0, y1 = centre[0] - n * cell / 2, centre[1] + n * cell / 2
    image = np.zeros((3, n, n), np.uint8)
    image[0, :, n // 2:] = 255
    image[1, :, : n // 2] = 255
    path = ch.data_dir / "orthomosaic.tif"
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=ch.crs,
                       transform=from_origin(x0, y1, cell, cell)) as dst:
        dst.write(image)
    ch._set_product("orthomosaic", path)
    d = drawings.make(ch, "plan", scale=20)
    assert d.raster is not None
    rgba, (a, b, c, e) = d.raster
    rows, cols = rgba.shape[:2]
    # the point of the sheet at (6 m, +1 m on the sheet = left of x): west of the keel when x runs north-north-east
    col = int((6.0 - a) / (b - a) * cols)
    row = int((e - 1.0) / (e - c) * rows)
    assert rgba[row, col, 1] == 255 and rgba[row, col, 0] == 0
    row = int((e + 1.0) / (e - c) * rows)
    assert rgba[row, col, 0] == 255 and rgba[row, col, 1] == 0


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "shipwreck" / "ppg_shipwreck-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-shipwreck\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\nshipwreck = ppg_shipwreck:ShipwreckPlugin\n")
    return folder


def test_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch, ends = project(tmp_path, heading_deg=37.0)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    monkeypatch.setenv("PPG_PLUGINS_ENVIRONMENT", "0")
    p = str(doc.path)

    def run(*args):
        result = CliRunner().invoke(app, ["plugin", "shipwreck", args[0], p, *args[1:]])
        assert result.exit_code == 0, result.output
        return result.output

    run("frame", "--start", ",".join(map(str, ends[0])), "--end", ",".join(map(str, ends[1])), "--bow", "end")
    run("stations", "--spacing", "2", "--count", "5", "--first", "2")
    run("sections", "--longitudinals", "0;0.5", "--horizontals", "-0.5", "--half-width", "3")
    run("water", "--level", "-0.5", "--on")
    table = tmp_path / "t.csv"
    table.write_text("kind,side,name,x,y\nframe,port,F36,6,0.5\n")
    assert "F36" in run("timbers", "--table", str(table))
    run("drawings", "--scale", "50")
    out = tmp_path / "legni.xlsx"
    run("export", "--out", str(out))
    assert out.exists()
    assert "sezioni" in run("status")
    # the commands of the guide, with lists and coordinates that begin with a minus
    run("sections", "--longitudinals", "0;1;-1", "--horizontals", "-0.2;-0.5", "--half-width", "3")
    assert len(lines.sections_of(ppg.Document.open(p).chunk, "horizontal")) == 2
    local = ship.Frame.from_dict(ppg.Document.open(p).chunk.meta["shipwreck"]["frame"])
    run("frame", "--start", f"-1.5,2,{local.z0}", "--end", f"-3.5,9,{local.z0}")
    assert ship.frame_of(ppg.Document.open(p).chunk).start[:2] == (-1.5, 2.0)
    again = ppg.Document.open(p).chunk
    assert len(ship.stations_of(again)) == 5 and lines.is_on(again) and len(timbers.timbers_of(again)) == 1
    assert (again.data_dir / "shipwreck" / "drawings" / "body_plan.pdf").exists()
    bad = CliRunner().invoke(app, ["plugin", "shipwreck", "water", p])
    assert bad.exit_code == 1


def test_the_pane_in_the_window(tmp_path):
    from PySide6.QtWidgets import QApplication

    from ppg_shipwreck import ShipwreckPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    doc, ch, ends = project(tmp_path, heading_deg=37.0)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    plugin = ShipwreckPlugin()
    answers = {}
    original = dialogs.ParamDialog.ask
    dialogs.ParamDialog.ask = classmethod(lambda cls, parent, title, fields, intro="": answers.pop("next", None))
    try:
        context = Context("shipwreck", tmp_path / "plugin", window=w)
        plugin.activate(context)
        pane = plugin.pane
        assert w.panes.groups["right"].dock(pane) is not None
        assert "Non impostato" in pane.states["frame"].text()
        # a step that misses what it needs says so
        messages = []
        import ppg_shipwreck

        plugin._message = lambda kind, text: messages.append(text)
        pane.requested.emit("sections")
        assert "sistema della nave" in messages[-1]
        # the keel traced on the view with the program's tool
        pane.requested.emit("frame_pick")
        assert w.ortho_view.tr is not None  # an empty frame around the model: there is no raster
        middle = ch.to_world(np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
        plugin._set_picked(ch, middle[:, :2])
        frame = ship.frame_of(ch)
        assert frame.length == pytest.approx(8.0)
        assert "Chiglia 8,00 m" in pane.states["frame"].text()
        pane.requested.emit("frame_flip")  # the bow stays
        assert ship.frame_of(ch).flipped().start == frame.start and ship.frame_of(ch).bow != frame.bow
        pane.requested.emit("frame_bow")  # the sides change
        assert ship.frame_of(ch).bow == frame.bow
        pane.requested.emit("frame_flip")
        pane.requested.emit("frame_bow")
        assert ship.frame_of(ch) == frame
        answers["next"] = {"spacing": 2.0, "count": 3, "start": 0.0}
        pane.requested.emit("stations_regular")
        assert "3 stazioni" in pane.states["stations"].text()
        answers["next"] = {"longitudinals": "0", "horizontals": "-0,5", "half_width": 3.0}
        pane.requested.emit("sections")
        w.runner.wait()
        assert "5 sezioni" in pane.states["sections"].text()
        answers["next"] = {"level": -0.5, "n": 1.34, "on": True}
        pane.requested.emit("water")
        assert "correzione accesa" in pane.states["water"].text()
        answers["next"] = {"kind": "frame", "side": "port", "name": "F36", "station": "", "notes": ""}
        plugin._timber("line", ch.to_world(np.array([[2.0, 0.0, 0.0], [2.0, 0.2, 0.0]]))[:, :2])
        assert pane.table.rowCount() == 1 and pane.table.item(0, 3).text() == "F36"
        assert ppg_shipwreck.LAYER in w.ortho_view.layers
        labels = [i.get("label") for i in plugin._tree_items(ch)[0]["children"]]
        assert any("Legni (1)" in x for x in labels) and any("Sezioni" in x for x in labels)
        # deleted on the view, gone from the catalogue
        plugin._features_deleted([{"id": "T1"}])
        assert not timbers.timbers_of(ch) and pane.table.rowCount() == 0
    finally:
        dialogs.ParamDialog.ask = original
        context.release()
        w.setWindowModified(False)
        w.close()


def test_the_land_lines_on_the_drawings(tmp_path):
    import ezdxf
    from hulls import clinker_project

    from ppg_shipwreck import clinker

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))  # sign -1: the plan's y is the ship's -y
    ship.set_stations(ch, ship.regular(1.0, 5, start=4.01))
    lines.build(ch, longitudinals=[0.5], half_width=3.0)
    for kind in ("body_plan", "profile", "plan"):  # before the lands: no layer, no note
        d = drawings.make(ch, kind)
        assert "Fasce" not in {layer for layer, _p, _c in d.lines}
        assert not any("ipotesi" in n for n in d.notes)
    clinker.detect_lands(ch)
    found = ch.meta["shipwreck"]["clinker"]["lines"]
    frame = ship.need_frame(ch)
    for kind in ("body_plan", "profile", "plan"):
        d = drawings.make(ch, kind)
        drawn = [(p, c) for layer, p, c in d.lines if layer == "Fasce"]
        assert drawn and all(c == drawings.LAYERS["Fasce"] for _p, c in drawn)
        assert drawings.LAYERS["Fasce"] not in {drawings.LAYERS[k] for k in drawings.LAYERS if k != "Fasce"}
        notes = " ".join(d.notes).lower()
        assert "fasce: gradini rilevati sulle sezioni" in notes and "linee delle fasce: ipotesi" in notes
        if kind == "plan":
            assert len(drawn) == len(found)
            for line in found:
                p = np.asarray(line["points"])
                want = np.column_stack([p[:, 0], p[:, 1] * frame.sign])
                assert any(q.shape == want.shape and np.abs(q - want).max() < 1e-9 for q, _c in drawn)
        if kind == "profile":
            assert any(np.abs(q - np.asarray(line["points"])[:, [0, 2]]).max() < 1e-9
                       for q, _c in drawn for line in found if q.shape == (len(line["points"]), 2))
        if kind == "body_plan":  # the land points at each station, port on the left
            centres = np.array([q.mean(axis=0) for q, _c in drawn])
            lands = np.array([[-y, z] for line in found for _x, y, z in line["points"]])
            assert len(centres) == len(lands)
            assert np.abs(np.sort(centres, axis=0) - np.sort(lands, axis=0)).max() < 1e-9
    written = drawings.write_all(ch, kinds=("plan",))
    dxf = ezdxf.readfile(next(p for p in written if p.suffix == ".dxf"))
    assert "Fasce" in {e.dxf.layer for e in dxf.modelspace()}
    # the sections cut again elsewhere: the lands are stale, the layer goes
    ship.set_stations(ch, ship.regular(1.0, 4, start=4.51))
    lines.build(ch, longitudinals=[0.5], half_width=3.0)
    assert clinker.is_stale(ch)
    d = drawings.make(ch, "plan")
    assert "Fasce" not in {layer for layer, _p, _c in d.lines} and not any("ipotesi" in n for n in d.notes)


# ----------------------------------------------------------------------------- the plan of the wreck (0.3.0)
WRECK_STATIONS = (4.275, 5.175, 6.075, 6.975)  # between the frames of the framed hull (0.9 + 0.45 k)


@pytest.fixture(scope="module")
def _wreck(tmp_path_factory):
    import copy

    from hulls import framed_project

    from ppg_shipwreck import proposals, relief, solids, tracing

    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("wreck"), heading_deg=35, x_range=(3.0, 9.0))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))  # sign -1: the plan's y is the ship's -y
    relief.build(ch, cell=0.01)
    proposals.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"]])
    solids.set_keel(ch, 0.12, 0.15)
    ship.set_stations(ch, [{"name": f"S{k + 1}", "x": x} for k, x in enumerate(WRECK_STATIONS)])
    lines.build(ch, longitudinals=[0.0, 0.5], half_width=2.0)
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def wreck(_wreck):
    import copy

    ch, snapshot = _wreck
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _sheet_polygon(ch, record):
    import shapely

    from ppg_shipwreck import tracing

    frame = ship.need_frame(ch)
    return shapely.Polygon(tracing.ship_xy_of(ch, record, frame) * [1.0, frame.sign])


def test_the_wreck_plan_outlines_fills_and_names_every_timber(wreck):
    import shapely

    ch = wreck
    d = drawings.make(ch, "wreck_plan")
    assert d.kind == "wreck_plan" and d.title == "Pianta del relitto" and d.scale > 0
    polygons = [r for r in timbers.timbers_of(ch) if r["geometry"]["type"] == "polygon"]
    assert {r["kind"] for r in polygons} >= {"frame", "plank", "keelson"}
    texts = [text for _x, _y, text, _c in d.labels]
    for r in polygons:
        poly = _sheet_polygon(ch, r).buffer(0.001)
        layer = f"LEGNI_{r['kind'].upper()}"
        mine = [ring for lay, ring, _c, _h, _dashed in d.fills if lay == layer and
                poly.contains(shapely.Point(np.asarray(ring).mean(axis=0)))]
        assert mine, r["name"]
        assert texts.count(r["name"]) == sum(1 for q in polygons if q["name"] == r["name"])
        x, y = next((x, y) for x, y, text, _c in d.labels if text == r["name"] and poly.contains(shapely.Point(x, y)))
    from ppg_shipwreck import models3d

    measured = [(lay, c, h) for lay, _ring, c, h, _dashed in d.fills if lay.startswith("LEGNI_")]
    assert all(c == models3d.KIND_COLOURS[lay[6:].lower()] and h == "" for lay, c, h in measured)
    seams = [r for r in timbers.timbers_of(ch) if r["kind"] == "seam"]
    assert seams and {"LEGNI_SEAM"} <= {layer for layer, _p, _c in d.lines}
    legend = [text for text, _c, _h in d.legend]
    assert any(text.startswith("ordinata") for text in legend) and any("ipotesi" in text or "ricostruito" in text
                                                                     for text in legend)
    assert d.north is not None and any("Sfondo" in n for n in d.notes)
    assert d.raster is not None  # the hillshade of the relief


def test_the_typed_keel_is_hatched_and_the_four_files_written(wreck, tmp_path):
    import ezdxf

    ch = wreck
    d = drawings.make(ch, "wreck_plan")
    keel = [(ring, c, h) for lay, ring, c, h, _dashed in d.fills if lay == "RICOSTRUZIONE_KEEL"]
    assert len(keel) == 1 and keel[0][2] == "////"
    from ppg_shipwreck import models3d

    assert keel[0][1] == models3d.lighter(models3d.KIND_COLOURS["keel"])
    k = ship.state(ch)["keel"]
    ring = np.asarray(keel[0][0])
    assert ring[:, 0].min() == pytest.approx(k["x0"]) and ring[:, 1].max() == pytest.approx(0.06)
    assert any("parti tratteggiate: ricostruzione (ipotesi)" in n.lower() for n in d.notes)
    written = drawings.write_all(ch, kinds=("wreck_plan",), folder=tmp_path / "out")
    assert {p.name for p in written} == {"wreck_plan.svg", "wreck_plan.pdf", "wreck_plan.png", "wreck_plan.dxf"}
    msp = ezdxf.readfile(tmp_path / "out" / "wreck_plan.dxf").modelspace()
    hatches = [e for e in msp if e.dxftype() == "HATCH"]
    assert any(e.dxf.layer == "RICOSTRUZIONE_KEEL" and e.dxf.pattern_name == "ANSI31" for e in hatches)
    assert not any(e.dxf.layer.startswith("LEGNI_") for e in hatches)  # a measured timber is never hatched
    closed = [e for e in msp if e.dxftype() == "LWPOLYLINE" and e.closed]
    assert {"LEGNI_FRAME", "LEGNI_PLANK", "LEGNI_KEELSON", "RICOSTRUZIONE_KEEL"} <= {e.dxf.layer for e in closed}
    assert any(e.dxf.linetype == "DASHED" for e in closed if e.dxf.layer == "LEGNI_PLANK")
    assert "chiglia" in (tmp_path / "out" / "wreck_plan.svg").read_text()
    # written into the chunk's folder by default, with the others: noted and up to date
    drawings.write_all(ch)
    assert "wreck_plan" in ch.meta[ship.KEY]["drawings"] and drawings.out_of_date(ch) == []
    timbers.add(ch, "other", "port", "X", timbers.geometry("point", _world(ship.need_frame(ch), (6.0, 0.4))))
    assert "wreck_plan" in drawings.out_of_date(ch)


def test_the_section_strakes_hidden_where_a_traced_strake_has_their_name(wreck):
    from ppg_shipwreck import clinker

    ch = wreck
    frame = ship.need_frame(ch)
    traced = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and timbers.is_measured(r))
    outline = _world(frame, (4.0, 0.3), (6.0, 0.3), (6.0, 0.5), (4.0, 0.5))
    for name in (traced["name"], "fascia B9"):
        timbers.add(ch, "plank", "port", name, timbers.geometry("polygon", outline), notes=clinker.CATALOGUE_NOTES[0],
                    status="reconstructed:sections")
    d = drawings.make(ch, "wreck_plan")
    texts = [text for _x, _y, text, _c in d.labels]
    assert texts.count(traced["name"]) == 1 and texts.count("fascia B9") == 1
    hypotheses = [(ring, c, h) for lay, ring, c, h, dashed in d.fills if lay == "RICOSTRUZIONE_PLANK" and not dashed]
    assert hypotheses and all(h == "////" for _r, _c, h in hypotheses)
    assert len({tuple(np.round(np.asarray(r).mean(axis=0), 6)) for r, _c, _h in hypotheses}) >= 1
    assert any("fascia B9" in text or "ipotesi dalle sezioni" in text for text, _c, _h in d.legend)
    assert any(traced["name"] in n for n in d.notes)  # the hidden one is said


def test_the_hidden_parts_of_a_strake_are_dashed(wreck):
    import shapely

    ch = wreck
    d = drawings.make(ch, "wreck_plan")
    frames = shapely.union_all([_sheet_polygon(ch, r) for r in timbers.timbers_of(ch)
                                if r["kind"] in ("frame", "keelson", "stringer")]).buffer(0.002)
    dashed = [shapely.LinearRing(ring) for lay, ring, _c, _h, dashed in d.fills if lay == "LEGNI_PLANK" and dashed]
    solid = [shapely.Polygon(ring) for lay, ring, _c, _h, dashed in d.fills if lay == "LEGNI_PLANK" and not dashed]
    assert dashed and solid
    assert all(p.difference(frames).length < 1e-6 for p in dashed)  # outlines under a frame or the keelson
    assert sum(p.length for p in dashed) > 5.0 and all(p.intersection(frames).area < 0.5 * p.area for p in solid)
    from ppg_shipwreck import tracing

    planks = [r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and timbers.is_measured(r)]
    visible = sum(abs(shapely.Polygon(ring).area) * (1 if shapely.LinearRing(ring).is_ccw else -1)
                  for r in planks for ring in r["visible_xy"])
    filled = sum(p.area * (1 if p.exterior.is_ccw else -1) for p in solid)
    assert filled == pytest.approx(visible, rel=1e-6)  # the filled parts are the visible ones
    assert tracing.CUT == ("plank",)


def test_the_wreck_plan_is_never_a_mirror_image(wreck):
    """Port on the sheet's sign * y side, as on :func:`drawings.plan`, for both bows."""
    import shapely

    ch = wreck
    frame = ship.need_frame(ch)
    for f in (frame, frame.other_bow()):
        ship.set_frame(ch, f)
        d = drawings.wreck_plan(ch, background="none")
        assert d.raster is None
        for r in [r for r in timbers.timbers_of(ch) if r["kind"] == "frame"][:3]:
            w = np.asarray(r["geometry"]["xy"], float)
            want = f.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2] * [1.0, f.sign]
            poly = shapely.Polygon(want)
            assert any(shapely.Polygon(ring).symmetric_difference(poly).area < 1e-7
                       for lay, ring, _c, _h, _d in d.fills if lay == "LEGNI_FRAME")
        from ppg_shipwreck import tracing

        planks = [r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and timbers.is_measured(r)]
        assert {r["side"] for r in planks} == {"port", "starboard"}
        for r in planks:  # port (positive ship y) up on the sheet when sign is 1, down when the frame is a mirror
            ship_y = float(tracing.ship_xy_of(ch, r, f)[:, 1].mean())
            x, y = next((x, y) for x, y, text, _c in d.labels if text == r["name"])
            assert np.sign(y) == f.sign * np.sign(ship_y)


def test_the_traced_strakes_on_the_body_plan(wreck):
    from hulls import seam_y

    ch = wreck
    d = drawings.make(ch, "body_plan")
    marks = [p for layer, p, _c in d.lines if layer == "FASCE_RICALCATE"]
    assert len(marks) >= 8 * len(WRECK_STATIONS)
    centres = np.array([p[2] for p in marks])  # left, right, centre, down, up
    arms = np.array([np.ptp(p[:, 0]) for p in marks])
    assert np.allclose(arms, 2 * drawings.LAND_MARK)
    data = lines.load(ch)
    for s in [s for s in data["sections"] if s["kind"] == "station"]:
        for y in seam_y(s["value"]):
            for side in (1.0, -1.0):
                near = np.abs(centres[:, 0] - (-side * y))
                assert near.min() < 0.01, (s["name"], y, side)
    assert drawings.LAYERS["FASCE_RICALCATE"] != drawings.LAYERS["Fasce"]
    assert not any(layer == "FASCE_RICALCATE" for layer, _p, _c in drawings.make(ch, "profile").lines)


def test_keelson_keel_and_sheer_on_the_profile(wreck):
    from hulls import KEELSON_HEIGHT, KEELSON_X

    ch = wreck
    d = drawings.make(ch, "profile")
    layers = {layer for layer, _p, _c in d.lines}
    assert {"KEELSON", "SHEER", "LONGITUDINALS"} <= layers
    top = np.vstack([p for layer, p, _c in d.lines if layer == "KEELSON"])
    assert top[:, 0].min() > max(KEELSON_X[0], 3.0) - 0.05 and top[:, 0].max() < min(KEELSON_X[1], 9.0) + 0.05
    assert float(np.median(top[:, 1])) == pytest.approx(-1.5 + KEELSON_HEIGHT, abs=0.015)
    keel = [(ring, h) for layer, ring, _c, h, _dashed in d.fills if layer == "KEEL"]
    assert len(keel) == 1 and keel[0][1] == "////"
    ring = np.asarray(keel[0][0])
    assert np.ptp(ring[:, 1]) == pytest.approx(0.15 + np.ptp(ring[len(ring) // 2:, 1]), abs=0.02)
    assert ring[:, 1].max() < float(np.median(top[:, 1]))  # under the keelson
    sheer = next(p for layer, p, _c in d.lines if layer == "SHEER")
    data = lines.load(ch)
    for s in [s for s in data["sections"] if s["kind"] == "station"]:
        ends = [q[[0, -1], 1] for q in lines.shown(s)]
        want = max(float(e.max()) for e in ends)
        assert np.interp(s["value"], sheer[:, 0], sheer[:, 1]) == pytest.approx(want, abs=1e-9)
    assert any("parti tratteggiate: ricostruzione (ipotesi)" in n.lower() for n in d.notes)
    import ezdxf

    out = drawings.write_dxf(d, Path(ship.folder(ch)) / "profile_test.dxf")
    msp = ezdxf.readfile(out).modelspace()
    assert {"KEELSON", "SHEER"} <= {e.dxf.layer for e in msp}
    assert any(e.dxftype() == "HATCH" and e.dxf.layer == "KEEL" for e in msp)
    out.unlink()


def test_a_survey_without_traced_timbers_draws_as_before(tmp_path):
    """0.2.0 projects: no new layer, fill or legend on the four drawings, no wreck plan without timbers."""
    _doc, ch, frame = _ready(tmp_path, water=True)
    for kind in ("body_plan", "half_breadth", "profile", "plan"):
        d = drawings.make(ch, kind)
        assert d.fills == [] and d.legend == []
        assert not {layer for layer, _p, _c in d.lines} & {"KEELSON", "SHEER", "FASCE_RICALCATE"}
        assert not any("parti tratteggiate" in n.lower() for n in d.notes)
    with pytest.raises(drawings.Empty):
        drawings.make(ch, "wreck_plan")
    names = {p.stem for p in drawings.write_all(ch)}
    assert "wreck_plan" not in names and "profile" in names
    import ezdxf

    msp = ezdxf.readfile(ch.data_dir / "shipwreck" / "drawings" / "profile.dxf")
    assert not {"KEELSON", "SHEER", "FASCE_RICALCATE"} & {layer.dxf.name for layer in msp.layers}
