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
    lines.build(ch, half_width=2.5)  # sections cut again: the plan of the wreck draws none of them
    assert set(drawings.out_of_date(ch)) == set(drawings.KINDS) - {"wreck_plan"}


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
KEEL_SIDING = 0.16  # wider than the keelson (0.12): the keel is seen beside it


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
    solids.set_keel(ch, KEEL_SIDING, 0.15)
    ship.set_stations(ch, [{"name": f"S{k + 1}", "x": x} for k, x in enumerate(WRECK_STATIONS)])
    lines.build(ch, longitudinals=[0.0, 0.5], half_width=2.5)  # wider than the hull: no section reaches the window
    sections = ship.folder(ch) / "sections.json"
    return ch, copy.deepcopy(ch.meta[ship.KEY]), sections, sections.read_text()


@pytest.fixture
def wreck(_wreck):
    import copy

    ch, snapshot, sections, text = _wreck
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    sections.write_text(text)  # a test that sets the water or cuts the sections again leaves the next one alone
    return ch


def _sheet_polygon(ch, record):
    import shapely

    from ppg_shipwreck import tracing

    frame = ship.need_frame(ch)
    return shapely.Polygon(tracing.ship_xy_of(ch, record, frame) * [1.0, frame.sign])


def _filled(d, layer, hatched=None):
    """The filled area of a layer (outer rings minus their holes), hatched or not."""
    import shapely

    out = shapely.Polygon()
    for lay, ring, _c, hatch, dashed in d.fills:
        if lay != layer or dashed or (hatched is not None and bool(hatch) != hatched):
            continue
        ring = shapely.Polygon(ring)
        out = out.union(ring) if shapely.LinearRing(ring.exterior).is_ccw else out.difference(ring)
    return out


HATCH_TEXT = "campite a righe oblique (////): ricostruzione (ipotesi)"


def test_the_wreck_plan_outlines_fills_and_names_every_timber(wreck):
    import copy

    import shapely

    from ppg_shipwreck import models3d

    ch = wreck
    before = copy.deepcopy(ch.meta[ship.KEY])
    d = drawings.make(ch, "wreck_plan")
    assert ch.meta[ship.KEY] == before  # drawing writes nothing, not even the plan caches
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
        inside = [(x, y) for x, y, text, _c in d.labels if text == r["name"] and poly.contains(shapely.Point(x, y))]
        assert inside, f"the name of {r['name']} is not on it"
    measured = [(lay, c, h) for lay, _ring, c, h, _dashed in d.fills if lay.startswith("LEGNI_")]
    assert all(c == models3d.KIND_COLOURS[lay[6:].lower()] and h == "" for lay, c, h in measured)
    assert [r for r in timbers.timbers_of(ch) if r["kind"] == "seam"]
    assert "LEGNI_SEAM" in {layer for layer, _p, _c in d.lines}
    legend = {text: hatch for text, _c, hatch in d.legend}
    assert legend["ordinata - misurato"] == "" and legend["chiglia - ricostruito: dimensioni date"] == "////"
    assert legend["comento - misurato"] == drawings.LINE_SWATCH  # a line, not a box
    assert d.north is not None and any("Sfondo" in n for n in d.notes)
    assert d.raster is not None  # the hillshade of the relief
    frames = [i for i, (_x, _y, text, _c) in enumerate(d.labels) if text.startswith("ordinata")]
    assert frames and all(abs(abs(d.rotations[i]) - 90.0) < 5.0 for i in frames)  # along the frames


def test_no_two_names_overlap_on_the_wreck_plan(wreck):
    ch = wreck
    fig = drawings.figure(drawings.make(ch, "wreck_plan"))
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    boxes = [(a.get_text(), a.get_window_extent(renderer)) for a in ax.texts]
    assert len(boxes) == len(drawings.make(ch, "wreck_plan").labels) > 20
    clashes = [(a, b) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
               if min(ba.x1, bb.x1) > max(ba.x0, bb.x0) and min(ba.y1, bb.y1) > max(ba.y0, bb.y0)]
    assert clashes == []


def test_the_typed_keel_is_hatched_only_where_seen_and_the_four_files_written(wreck, tmp_path):
    import ezdxf
    import shapely

    from ppg_shipwreck import models3d

    ch = wreck
    d = drawings.make(ch, "wreck_plan")
    keel = [(ring, c, h, dashed) for lay, ring, c, h, dashed in d.fills if lay == "RICOSTRUZIONE_KEEL"]
    colour = models3d.lighter(models3d.KIND_COLOURS["keel"])
    assert keel and all(c == colour for _r, c, _h, _d in keel)
    k = ship.state(ch)["keel"]
    hatched = _filled(d, "RICOSTRUZIONE_KEEL", hatched=True)
    covered = shapely.union_all([shapely.Polygon(r) for r, _c, _h, dashed in keel if dashed])
    whole = shapely.box(k["x0"], -KEEL_SIDING / 2, k["x1"], KEEL_SIDING / 2)
    assert hatched.union(covered).area == pytest.approx(whole.area, rel=1e-3)
    keelson = next(_sheet_polygon(ch, r) for r in timbers.timbers_of(ch) if r["kind"] == "keelson")
    assert hatched.intersection(keelson).area < 1e-6  # never hatched through the measured keelson
    assert hatched.area > 0.2 * whole.area and covered.intersection(keelson).area > 0.5 * keelson.intersection(whole).area
    assert "chiglia" not in [text for _x, _y, text, _c in d.labels]  # under the keelson: named by the legend only
    assert any(HATCH_TEXT in n.lower() for n in d.notes)
    assert not any("tratteggiate" in n.lower() for n in d.notes)
    written = drawings.write_all(ch, kinds=("wreck_plan",), folder=tmp_path / "out")
    assert {p.name for p in written} == {"wreck_plan.svg", "wreck_plan.pdf", "wreck_plan.png", "wreck_plan.dxf"}
    msp = ezdxf.readfile(tmp_path / "out" / "wreck_plan.dxf").modelspace()
    hatches = [e for e in msp if e.dxftype() == "HATCH"]
    assert any(e.dxf.layer == "RICOSTRUZIONE_KEEL" and e.dxf.pattern_name == "ANSI31" for e in hatches)
    assert not any(e.dxf.layer.startswith("LEGNI_") for e in hatches)  # a measured timber is never hatched
    closed = [e for e in msp if e.dxftype() == "LWPOLYLINE" and e.closed]
    assert {"LEGNI_FRAME", "LEGNI_PLANK", "LEGNI_KEELSON", "RICOSTRUZIONE_KEEL"} <= {e.dxf.layer for e in closed}
    assert any(e.dxf.linetype == "DASHED" for e in closed if e.dxf.layer == "LEGNI_PLANK")
    assert any(e.dxftype() == "TEXT" and abs(abs(e.dxf.rotation) - 90.0) < 5.0 for e in msp)  # the frames' names
    assert "comento" in (tmp_path / "out" / "wreck_plan.svg").read_text()
    # written into the chunk's folder by default, with the others: noted and up to date
    drawings.write_all(ch)
    assert "wreck_plan" in ch.meta[ship.KEY]["drawings"] and drawings.out_of_date(ch) == []
    lines.set_water(ch, -0.5, on=True)  # the sections' water: not on the plan of the wreck
    assert "wreck_plan" not in drawings.out_of_date(ch)
    timbers.add(ch, "other", "port", "X", timbers.geometry("point", _world(ship.need_frame(ch), (6.0, 0.4))))
    assert "wreck_plan" in drawings.out_of_date(ch)


def test_the_section_strakes_hidden_where_a_traced_strake_has_their_name(wreck):
    import shapely

    from ppg_shipwreck import clinker

    ch = wreck
    frame = ship.need_frame(ch)
    traced = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and timbers.is_measured(r))
    for name, x0, x1 in ((traced["name"], 7.0, 8.0), ("fascia B9", 4.0, 6.0)):
        outline = _world(frame, (x0, 0.3), (x1, 0.3), (x1, 0.5), (x0, 0.5))
        timbers.add(ch, "plank", "port", name, timbers.geometry("polygon", outline), notes=clinker.CATALOGUE_NOTES[0],
                    status="reconstructed:sections")
    d = drawings.make(ch, "wreck_plan")
    texts = [text for _x, _y, text, _c in d.labels]
    assert texts.count(traced["name"]) == 1 and texts.count("fascia B9") == 1
    hypotheses = [(ring, h) for lay, ring, _c, h, dashed in d.fills if lay == "RICOSTRUZIONE_PLANK"]
    assert hypotheses and all(h == "////" for _r, h in hypotheses)
    drawn = _filled(d, "RICOSTRUZIONE_PLANK")
    b9 = shapely.box(4.0, 0.3 * frame.sign, 6.0, 0.5 * frame.sign).normalize()
    assert drawn.symmetric_difference(b9).area < 1e-6  # B9 whole, nothing at x 7-8 where the traced name is
    assert "corso di fasciame - ipotesi dalle sezioni" in [text for text, _c, _h in d.legend]
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
    # a frame ending a cell short of the strake's outline leaves a visible corner of a few millimetres beside it
    assert sum(p.length for p in dashed) > 5.0 and all(p.intersection(frames).area < 0.5 * p.area for p in solid
                                                       if p.area > 1e-4)
    planks = [_sheet_polygon(ch, r) for r in timbers.timbers_of(ch) if r["kind"] == "plank" and timbers.is_measured(r)]
    visible = shapely.union_all(planks).difference(frames.buffer(-0.002))
    assert _filled(d, "LEGNI_PLANK").symmetric_difference(visible).area < 1e-4  # the filled parts are the visible ones


def test_holes_are_left_empty_on_the_sheet_and_in_the_hatch(tmp_path):
    """A timber with another lying wholly inside it: a hole in its fill, in the figure's path and in the DXF hatch."""
    import ezdxf
    import shapely

    d = drawings.Drawing("wreck_plan", "t", scale=20)
    geom = shapely.box(0, 0, 2, 1).difference(shapely.box(0.8, 0.4, 1.2, 0.6))
    drawings._ring_fills(d, "RICOSTRUZIONE_PLANK", geom, "#88aacc", drawings.HATCH, False)
    shapes = drawings._shapes(d.fills)
    assert len(shapes) == 1 and len(shapes[0][1]) == 2
    fig = drawings.figure(d)
    fig.canvas.draw()
    ax = fig.axes[0]
    pixels = np.asarray(fig.canvas.buffer_rgba())
    height = pixels.shape[0]

    def tinted(x, y):
        px, py = ax.transData.transform((x, y))
        return pixels[int(height - py), int(px), :3].min() < 250

    assert tinted(0.3, 0.5) and not tinted(1.0, 0.5)  # the body filled, the hole left white
    msp = ezdxf.readfile(drawings.write_dxf(d, tmp_path / "h.dxf")).modelspace()
    (hatch,) = [e for e in msp if e.dxftype() == "HATCH"]
    assert len(hatch.paths) == 2


def test_the_wreck_plan_is_never_a_mirror_image(wreck):
    """Port on the sheet's sign * y side, as on :func:`drawings.plan`, for both bows."""
    import shapely

    from ppg_shipwreck import tracing

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
    import shapely
    from hulls import KEELSON_HEIGHT, KEELSON_X

    from ppg_shipwreck import models3d

    ch = wreck
    d = drawings.make(ch, "profile")
    assert {"SHEER", "LONGITUDINALS"} <= {layer for layer, _p, _c in d.lines}
    (keelson,) = [ring for layer, ring, _c, h, _d in d.fills if layer == "KEELSON" and h == ""]
    n = len(keelson) // 2
    bottom, top = keelson[:n], keelson[n:]
    assert top[:, 0].min() > max(KEELSON_X[0], 3.0) - 0.05 and top[:, 0].max() < min(KEELSON_X[1], 9.0) + 0.05
    assert float(np.median(top[:, 1])) == pytest.approx(-1.5 + KEELSON_HEIGHT, abs=0.015)
    assert float(np.median(bottom[:, 1])) == pytest.approx(-1.5, abs=0.015)  # its body: from the hull up
    keel = [(ring, h) for layer, ring, _c, h, _dashed in d.fills if layer == "RICOSTRUZIONE_KEEL"]
    assert len(keel) == 1 and keel[0][1] == "////"
    ring = np.asarray(keel[0][0])
    m = len(ring) // 2
    assert np.ptp(ring[:, 1]) == pytest.approx(0.15 + np.ptp(ring[m:, 1]), abs=0.02)
    assert shapely.Polygon(ring).intersection(shapely.Polygon(keelson)).area < 1e-4  # the keel under the keelson
    (label,) = [(x, y, c) for x, y, text, c in d.labels if text == "chiglia"]
    assert label[2] == models3d.KIND_COLOURS["keel"] and shapely.Polygon(ring).buffer(1e-6).contains(
        shapely.Point(label[0], label[1]))
    sheer = next(p for layer, p, _c in d.lines if layer == "SHEER")
    data = lines.load(ch)
    stations = [s for s in data["sections"] if s["kind"] == "station"]
    assert not any(s.get("reached") for s in stations) and len(sheer) == len(stations)
    for s in stations:
        ends = [q[[0, -1], 1] for q in lines.shown(s)]
        want = max(float(e.max()) for e in ends)
        assert np.interp(s["value"], sheer[:, 0], sheer[:, 1]) == pytest.approx(want, abs=1e-9)
    assert "orlo conservato" in [text for _x, _y, text, _c in d.labels]
    assert any(h == drawings.LINE_SWATCH and "orlo conservato" in text for text, _c, h in d.legend)
    assert any(HATCH_TEXT in n.lower() for n in d.notes)
    import ezdxf

    out = drawings.write_dxf(d, Path(ship.folder(ch)) / "profile_test.dxf")
    msp = ezdxf.readfile(out).modelspace()
    assert {"KEELSON", "SHEER", "RICOSTRUZIONE_KEEL"} <= {e.dxf.layer for e in msp}
    assert any(e.dxftype() == "HATCH" and e.dxf.layer == "RICOSTRUZIONE_KEEL" for e in msp)
    assert not any(e.dxftype() == "HATCH" and e.dxf.layer == "KEELSON" for e in msp)
    out.unlink()


def test_the_sheer_breaks_where_a_section_reached_the_window(wreck):
    ch = wreck
    ship.set_stations(ch, [{"name": f"S{k + 1}", "x": x} for k, x in enumerate((3.375, 3.825, 4.275, 6.075))])
    lines.build(ch, longitudinals=[0.0], half_width=2.0)  # narrower than the hull amidships
    stations = [s for s in lines.load(ch)["sections"] if s["kind"] == "station"]
    reached = [s for s in stations if s.get("reached")]
    assert reached and len(stations) - len(reached) >= 2
    d = drawings.make(ch, "profile")
    sheer = [p for layer, p, _c in d.lines if layer == "SHEER"]
    assert sheer
    xs = np.concatenate([p[:, 0] for p in sheer])
    assert not any(np.isclose(xs, s["value"]).any() for s in reached)
    assert all(np.isclose(xs, s["value"]).any() for s in stations if not s.get("reached"))
    names = ", ".join(s["name"] for s in reached)
    assert any("Orlo conservato interrotto" in n and names in n for n in d.notes)
    lines.build(ch, longitudinals=[0.0], half_width=1.0)  # every station at the window: no line, and it is said
    d = drawings.make(ch, "profile")
    assert not [p for layer, p, _c in d.lines if layer == "SHEER"]
    assert any("Orlo conservato non disegnato" in n for n in d.notes)


def test_a_survey_without_traced_timbers_draws_as_before(tmp_path):
    """0.2.0 projects: no new layer, fill or legend on the four drawings, no wreck plan without timbers."""
    _doc, ch, frame = _ready(tmp_path, water=True)
    for kind in ("body_plan", "half_breadth", "profile", "plan"):
        d = drawings.make(ch, kind)
        assert d.fills == [] and d.legend == [] and d.rotations == {}
        assert not {layer for layer, _p, _c in d.lines} & {"KEELSON", "SHEER", "FASCE_RICALCATE"}
        assert not any("righe oblique" in n.lower() for n in d.notes)
    with pytest.raises(drawings.Empty):
        drawings.make(ch, "wreck_plan")
    names = {p.stem for p in drawings.write_all(ch)}
    assert "wreck_plan" not in names and "profile" in names
    import ezdxf

    doc = ezdxf.readfile(ch.data_dir / "shipwreck" / "drawings" / "profile.dxf")
    assert not {"KEELSON", "SHEER", "FASCE_RICALCATE"} & {layer.dxf.name for layer in doc.layers}


def test_a_keel_wholly_under_the_keelson_says_so(wreck):
    from ppg_shipwreck import solids

    ch = wreck
    solids.set_keel(ch, 0.10, 0.15)  # narrower than the keelson (0.12): no part of it shows
    d = drawings.make(ch, "wreck_plan")
    assert any("interamente sotto il paramezzale" in n for n in d.notes)


def test_the_profile_says_what_it_could_not_draw(wreck, monkeypatch):
    from ppg_shipwreck import solids

    ch = wreck

    def refused(chunk, record):
        raise ship.NotReady("no")

    monkeypatch.setattr(solids, "timber_solid", refused)
    monkeypatch.setattr(solids, "keel_solid", lambda chunk: (_ for _ in ()).throw(ship.NotReady("no")))
    d = drawings.make(ch, "profile")
    assert any(n.startswith("Non disegnati: paramezzale") and "chiglia" in n for n in d.notes)


# ----------------------------------------------------------------------------- the reconstruction on the drawings (0.4.0)
@pytest.fixture(scope="module")
def _rebuilt(tmp_path_factory):
    """The project of the reconstruction's tests (both sides lost over 8.4-9.3, port lost aft of 6.5), its meta before
    and after the whole reconstruction is built."""
    import copy

    from test_reconstruct import HOLE, _wreck_project

    from ppg_shipwreck import reconstruct

    ch = _wreck_project(tmp_path_factory.mktemp("rebuilt"), port_from=HOLE)
    before = copy.deepcopy(ch.meta[ship.KEY])
    reconstruct.build(ch)
    return ch, before, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def rebuilt(_rebuilt):
    import copy

    ch, _before, after = _rebuilt
    ch.meta[ship.KEY] = copy.deepcopy(after)
    return ch


def _dxf_entities(path) -> list:
    """What a DXF draws: type, layer, linetype and points of each entity (no header, no time stamps)."""
    import ezdxf

    out = []
    for e in ezdxf.readfile(path).modelspace():
        kind = e.dxftype()
        if kind == "LWPOLYLINE":
            pts = [tuple(round(v, 9) for v in p[:2]) for p in e.get_points()]
        elif kind == "HATCH":
            pts = [(round(float(v2[0]), 9), round(float(v2[1]), 9)) for path in e.paths for v2 in path.vertices]
        elif kind == "TEXT":
            pts = [e.dxf.text, round(e.dxf.insert.x, 9), round(e.dxf.insert.y, 9)]
        else:
            pts = []
        out.append((kind, e.dxf.layer, e.dxf.get("linetype", ""), pts))
    return out


def _png_hash(d) -> str:
    import hashlib
    import io

    buf = io.BytesIO()
    drawings.figure(d).savefig(buf, format="png", dpi=60, metadata={"Software": None})
    return hashlib.sha256(buf.getvalue()).hexdigest()


def test_the_reconstruction_hatched_and_lighter_on_the_wreck_plan(rebuilt, tmp_path):
    import copy

    import ezdxf
    import shapely

    from ppg_shipwreck import reconstruct
    from ppg_shipwreck.texts import num

    ch = rebuilt
    before = copy.deepcopy(ch.meta[ship.KEY])
    d = drawings.make(ch, "wreck_plan")
    assert ch.meta[ship.KEY] == before  # drawing writes nothing
    records = [r for r in reconstruct.records_of(ch) if r["geometry"]["type"] == "polygon"]
    statuses = {r["status"] for r in records}
    assert statuses == {"reconstructed:mirror", "reconstructed:extension", "reconstructed:spacing"}
    sign = ship.need_frame(ch).sign
    hatches = {"reconstructed:mirror": "////", "reconstructed:extension": "\\\\\\\\", "reconstructed:spacing": "xxxx"}
    assert drawings.STATUS_HATCHES == hatches
    for r in records:  # every reconstruction record a fill hatched as its status, lighter, on its own layer
        layer = "RICOSTRUZIONE_" + r["kind"].upper()
        assert layer in ("RICOSTRUZIONE_PLANK", "RICOSTRUZIONE_FRAME") or r["kind"] not in ("plank", "frame")
        poly = shapely.Polygon(np.asarray(r["ship_xy"]) * [1.0, sign]).buffer(0)
        mine = [i for i, (lay, ring, _c, hatch, dashed) in enumerate(d.fills) if lay == layer
                and hatch == hatches[r["status"]] and not dashed
                and shapely.Polygon(ring).buffer(0).symmetric_difference(poly).area < 1e-3 * poly.area]
        assert mine, r["name"]
        assert all(i in d.faint for i in mine)
    assert not any(lay.startswith("LEGNI_") for i, (lay, *_rest) in enumerate(d.fills) if i in d.faint)
    # the faired hull: its plan a dashed outline only (no fill, no hatch) on RICOSTRUZIONE_HULL, smoothed
    rings = [(ring, hatch, dashed) for lay, ring, _c, hatch, dashed in d.fills if lay == "RICOSTRUZIONE_HULL"]
    assert rings and all(dashed and not hatch for _r, hatch, dashed in rings)
    hull = shapely.union_all([shapely.Polygon(r) for r, _h, _d in rings if shapely.LinearRing(r).is_ccw])
    assert hull.area == pytest.approx(reconstruct.state(ch)["loft"]["area"], rel=0.05)
    assert sum(len(r) for r, _h, _d in rings) < 400  # smoothed: no saw-tooth along the cells
    # a legend entry per how present, its swatch as on the sheet, nothing for the others
    legend = {text: hatch for text, _c, hatch in d.legend}
    texts = [text for text, _c, _h in d.legend]
    for status in ("reconstructed:mirror", "reconstructed:extension", "reconstructed:spacing", "reconstructed:fairing"):
        assert texts.count(timbers.status_title(status)) == 1, status
        assert legend[timbers.status_title(status)] == hatches.get(status, drawings.DASH_SWATCH)
    assert timbers.status_title("reconstructed:imported") not in " ".join(texts)
    # both ends open: "carenatura aperta" on the keel line just past the first and last stations
    stations = reconstruct.state(ch)["loft"]["stations"]
    open_at = sorted(x for x, y, text, _c in d.labels if text == "carenatura aperta")
    assert len(open_at) == 2 and open_at[0] < min(stations) - 0.05 and open_at[1] > max(stations) + 0.05
    assert all(y == 0.0 for _x, y, text, _c in d.labels if text == "carenatura aperta")
    # one note on the hatches, naming the three of the reconstruction
    hatch_notes = [n for n in d.notes if "////" in n]
    assert len(hatch_notes) == 1 and "\\\\\\\\" in hatch_notes[0] and "xxxx" in hatch_notes[0]
    assert "----" not in hatch_notes[0]  # no imported plank on this plan: not named
    assert not any(n.startswith("Ricostruzione (ipotesi): campitura") for n in d.notes)
    totals = reconstruct.state(ch)["table"]["totals"]
    said = f"misurato {num(totals['measured']['area'], 2)} m², ricostruito {num(totals['reconstructed']['area'], 2)} m² (ipotesi)"
    assert any(said in n for n in d.notes), d.notes
    assert not any("oltre l'ultima stazione" in n.lower() for n in d.notes)  # no end given: the loft stops at the stations
    assert not any("dai chunk" in n for n in d.notes)  # one chunk
    path = drawings.write_dxf(d, tmp_path / "wreck.dxf")
    msp = ezdxf.readfile(path).modelspace()
    patterns = {(e.dxf.layer, e.dxf.pattern_name) for e in msp if e.dxftype() == "HATCH"}
    for status, pattern in (("reconstructed:mirror", "ANSI31"), ("reconstructed:extension", "ANSI32"),
                            ("reconstructed:spacing", "ANSI37")):
        layers = {"RICOSTRUZIONE_" + r["kind"].upper() for r in records if r["status"] == status}
        assert {(lay, pattern) for lay in layers} <= patterns, status
    assert not any(lay == "RICOSTRUZIONE_HULL" for lay, _p in patterns)
    assert any(e.dxf.linetype == "DASHED" for e in msp if e.dxftype() == "LWPOLYLINE"
               and e.dxf.layer == "RICOSTRUZIONE_HULL")
    drawings.figure(d)  # the sheet draws
    # written and noted: out of date when the reconstruction changes
    drawings.write_all(ch, kinds=("wreck_plan", "body_plan", "profile"))
    assert drawings.out_of_date(ch) == []
    reconstruct.state(ch)["timbers"] = reconstruct.records_of(ch)[1:]
    assert {"wreck_plan", "body_plan", "profile"} <= set(drawings.out_of_date(ch))


def test_the_loft_past_the_last_station_and_the_chunks_said(rebuilt):
    from ppg_shipwreck import reconstruct

    ch = rebuilt
    reconstruct.set_ends(ch, stem=-0.3, stern=12.3)
    reconstruct.loft(ch)
    d = drawings.make(ch, "wreck_plan")
    from ppg_shipwreck import models3d

    assert not any(text == "carenatura aperta" for _x, _y, text, _c in d.labels)
    stations = reconstruct.state(ch)["loft"]["stations"]
    lighter = models3d.lighter(models3d.lighter(models3d.HULL_COLOUR))
    beyond = [ring for lay, ring, c, _h, dashed in d.fills if lay == "RICOSTRUZIONE_HULL" and dashed and c == lighter]
    assert beyond and all(ring[:, 0].min() >= max(stations) - 1e-6 or ring[:, 0].max() <= min(stations) + 1e-6
                          for ring in beyond)
    assert any("oltre l'ultima stazione la carenatura è un'ipotesi della sola larghezza" in n.lower() for n in d.notes)
    reconstruct.state(ch)["members"] = {"chunks": [{"chunk": ch.key, "label": "Scavo A"},
                                                   {"chunk": "k2", "label": "Scavo B"}]}
    d = drawings.make(ch, "wreck_plan")
    assert any("dai chunk: Scavo A, Scavo B" in n for n in d.notes)


def test_the_loft_on_the_body_plan_and_the_extended_sheer_on_the_profile(rebuilt):
    import copy

    from hulls import clinker_section
    from test_reconstruct import LOST_BOTH

    from ppg_shipwreck import reconstruct

    ch = rebuilt
    d = drawings.make(ch, "body_plan")
    hull = [p for lay, p, _c in d.lines if lay == "RICOSTRUZIONE_HULL"]
    assert hull
    lost = [s["value"] for s in lines.load(ch)["sections"] if s["kind"] == "station" and LOST_BOTH[0] < s["value"] < LOST_BOTH[1]]
    assert lost
    pts = np.vstack(hull)
    assert pts[:, 0].min() < -0.5 and pts[:, 0].max() > 0.5  # both sides: neither has a section in the lost span
    stations = [s["value"] for s in lines.load(ch)["sections"] if s["kind"] == "station"]
    for y, z in pts[::7]:  # on the hull at one of the stations (the loft also fills gaps beside mirrored timbers)
        best = min(abs(z - np.interp(abs(y), sec[sec[:, 0] >= 0][:, 0], sec[sec[:, 0] >= 0][:, 1]))
                   for sec in map(clinker_section, stations))
        assert best < 0.03, (y, z, best)
    assert not any("niente di ciò che manca è ricostruito" in n for n in d.notes)
    assert any(n.startswith("Linee di sezione misurate sul modello 3D; a trattini la carenatura (ipotesi)") for n in d.notes)
    assert any("Senza linea del modello, carenatura (ipotesi): S17, S18." in n for n in d.notes), d.notes
    assert not any(n.startswith("Senza linea, fuori dal modello") for n in d.notes)
    # the top strakes (T5, B5) are not extended on this hull: no extended upper edge, never a lower strake's
    p = drawings.make(ch, "profile")
    assert not [q for lay, q, _c in p.lines if lay == "RICOSTRUZIONE_HULL"]
    assert drawings._top_strakes(ch, timbers.timbers_of(ch)) == {"port": "fascia B5", "starboard": "fascia T5"}
    # a piece of the top strake extended: its upper edge, where it has a height
    st = reconstruct.state(ch)
    piece = copy.deepcopy(next(r for r in st["timbers"] if r["status"] == "reconstructed:extension"
                               and r["side"] == "starboard"))
    piece.update(id="R99", name="fascia T5")
    piece["edges"]["upper"][3][2] = None  # a point without a height breaks the line
    st["timbers"].append(piece)
    p = drawings.make(ch, "profile")
    sheer = [q for lay, q, _c in p.lines if lay == "RICOSTRUZIONE_HULL"]
    assert len(sheer) == 2
    upper = np.array([[x, z] for x, _y, z in piece["edges"]["upper"] if z is not None])
    assert sum(len(q) for q in sheer) == len(upper)
    for x, z in np.vstack(sheer):
        assert np.min(np.hypot(upper[:, 0] - x, upper[:, 1] - z)) < 1e-9
    assert any(text.startswith("orlo prolungato") for text, _c, _h in p.legend)


def test_without_the_reconstruction_the_drawings_of_0_3_0(_rebuilt, tmp_path):
    """``reconstruction=False`` on the built project, and the default on the project before the build: the same
    DXF entities and the same PNG, for the plan of the wreck, the body plan and the profile."""
    import copy

    ch, before, after = _rebuilt
    try:
        for kind in ("wreck_plan", "body_plan", "profile"):
            ch.meta[ship.KEY] = copy.deepcopy(before)
            old = drawings.make(ch, kind)
            ch.meta[ship.KEY] = copy.deepcopy(after)
            off = drawings.make(ch, kind, reconstruction=False)
            on = drawings.make(ch, kind)
            a = _dxf_entities(drawings.write_dxf(old, tmp_path / f"{kind}_old.dxf"))
            b = _dxf_entities(drawings.write_dxf(off, tmp_path / f"{kind}_off.dxf"))
            c = _dxf_entities(drawings.write_dxf(on, tmp_path / f"{kind}_on.dxf"))
            assert a == b, kind
            if kind != "profile":  # the profile draws only the top strake's extension: none on this hull
                assert a != c, kind  # the reconstruction is drawn when asked
            assert _png_hash(old) == _png_hash(off), kind
            assert off.faint == set() and not any(lay.endswith("_HULL") for lay, *_r in off.fills + off.lines)
    finally:
        ch.meta[ship.KEY] = copy.deepcopy(after)


def test_imported_planks_have_their_own_hatch(rebuilt, tmp_path):
    """A plank imported from the Clinker Boat Tool: hatched with horizontal lines (----, ANSI33 in the DXF) when the
    reconstruction is drawn, named in the note on the hatches; as 0.3.0 (////) without it."""
    import ezdxf
    import shapely

    from ppg_shipwreck import houdini

    ch = rebuilt
    plank = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and r["geometry"]["type"] == "polygon")
    plank["notes"], plank["status"] = houdini.NOTE[0].format("tavole.obj"), "reconstructed:imported"
    assert timbers.status_of(plank) == "reconstructed:imported"
    sign = ship.need_frame(ch).sign
    poly = shapely.Polygon(np.asarray(plank["ship_xy"]) * [1.0, sign]).buffer(0)

    def hatches(d):
        return {hatch for lay, ring, _c, hatch, dashed in d.fills if lay == "RICOSTRUZIONE_PLANK" and not dashed
                and shapely.Polygon(ring).buffer(0).intersection(poly).area > 0.5 * poly.area}

    d = drawings.make(ch, "wreck_plan")
    assert drawings.IMPORTED_HATCH == "----" and hatches(d) == {"----"}
    note = next(n for n in d.notes if "////" in n)
    assert "(----)" in note and "Houdini" in note
    legend = {text: hatch for text, _c, hatch in d.legend}
    assert legend[f"{timbers.kind_title('plank')} - {timbers.status_title('reconstructed:imported')}"] == "----"
    path = drawings.write_dxf(d, tmp_path / "wreck_plan.dxf")
    patterns = {e.dxf.pattern_name for e in ezdxf.readfile(path).modelspace() if e.dxftype() == "HATCH"}
    assert "ANSI33" in patterns
    assert hatches(drawings.make(ch, "wreck_plan", reconstruction=False)) == {"////"}
