"""The fixes of 0.1.2, found by writing the user guide: the tables as people write them (column
names of the field, decimal commas in a comma-separated file, thousands separators, the real line
numbers, rows half filled, colours in words), the unit codes in one form everywhere, the hatches by
the head noun of the lithology, the corrections of the stack on a diverging ramp, the Italian
texts, the panels of columns that keep their own file, the section through the boreholes in the
order they were chosen, the plugin's layers in the QGIS project of a chunk in a local system."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_geoarchaeology import project

import pyarchinit_photogrammetry as ppg
from ppg_geoarchaeology import boreholes as bh
from ppg_geoarchaeology import export, logs, model, overlay
from ppg_geoarchaeology import surfaces as sf


def write(folder: Path, name: str, text: str) -> Path:
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return p


def local_chunk(folder: Path):
    """A chunk in a local reference system (the default of a new project), no DEM."""
    folder.mkdir(parents=True, exist_ok=True)
    doc = ppg.Document()
    ch = doc.add_chunk("Locale")
    doc.save(folder / "locale.ppg")
    return doc, ch


def warned(report: dict, *words: str) -> bool:
    """Whether one warning of the import has all ``words`` (case-insensitive)."""
    return any(all(w.lower() in m.lower() for w in words) for m in report["warnings"])


# ------------------------------------------------------------------ 1. column names
def test_column_names_written_in_the_field(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    collars = write(tmp_path, "collari.csv", "Sigla sondaggio;Est;Nord;Quota s.l.m. (m);Prof. (m)\n"
                                             "S1;500010;4500010;100,50;3,00\nS2;500020;4500010;100,60;2,50\n")
    intervals = write(tmp_path, "intervalli.csv",
                      "N. sondaggio;Da (cm);A (cm);Codice US;Descrizione US;Colore Munsell\n"
                      "S1;0;30;US 1;humus;10YR 3/2\nS1;30;300;US 2;argilla;2.5Y 5/2\n"
                      "S2;0;40;US 1;humus;10YR 3/2\nS2;40;250;US 2;argilla;2.5Y 5/2\n")
    rep = bh.import_files(ch, [collars, intervals])
    assert rep["read"] == 2 and rep["intervals"] == 4 and rep["units"] == ["US 1", "US 2"]
    s1 = bh.get(ch, "S1")
    assert s1.z == pytest.approx(100.5) and s1.depth == pytest.approx(3.0)
    assert [(iv.top, iv.base) for iv in s1.intervals] == [(0.0, pytest.approx(0.3)), (pytest.approx(0.3), 3.0)]
    assert s1.intervals[1].description == "argilla" and s1.intervals[1].colour == "2.5Y 5/2"
    # "US n." and "N. US" are the unit, "Sigla" with "sondaggio" beside it is not the borehole
    other = write(tmp_path, "altro.csv", "Sondaggio;US n.;Sigla;da;a\nS1;3;x;0;1\nS1;4;y;1;2\n")
    rows, names = bh.read_table(other)
    assert names["id"] == "Sondaggio" and names["unit"] == "US n."
    rows, names = bh.read_table(write(tmp_path, "n_us.csv", "sondaggio;da;a;N. US\nS1;0;1;5\n"))
    assert names["unit"] == "N. US" and rows[0]["unit"] == "5"
    # the English context, US/USM, a "livello" when nothing clearer is there
    for header in ("Context No.", "US/USM", "Livello"):
        rows, names = bh.read_table(write(tmp_path, "u.csv", f"Trench;From;To;{header}\nT1;0;1;101\n"))
        assert names["id"] == "Trench" and names["unit"] == header and rows[0]["unit"] == "101"
    notes = []
    rows, names = bh.read_table(write(tmp_path, "l.csv", "sondaggio;da;a;us;livello\nS1;0;1;US 1;3\n"), notes)
    assert names["unit"] == "us" and any("livello" in n for n in notes)  # the clearer name wins, the other reported


def test_ambiguous_columns_are_resolved_and_reported(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    # "id" is the number of the row, "codice" the unit: the borehole is "sondaggio"
    collars = write(tmp_path, "c.csv", "sondaggio;x;y;quota\nS1;500010;4500010;100\n")
    intervals = write(tmp_path, "i.csv", "id;sondaggio;da;a;codice\n1;S1;0;1;A\n2;S1;1;2;B\n")
    rep = bh.import_files(ch, [collars, intervals])
    assert rep["units"] == ["A", "B"] and [iv.unit for iv in bh.get(ch, "S1").intervals] == ["A", "B"]
    assert warned(rep, "codice", "unità")
    # tetto and letto holding elevations (the base under the top): read as elevations, said so
    t = write(tmp_path, "t.csv", "sondaggio;x;y;quota;tetto;letto;us\n"
                                 "T1;500030;4500010;100;100;99,7;US 1\nT1;500030;4500010;100;99,7;98,5;US 2\n")
    rep = bh.import_files(ch, [t])
    t1 = bh.get(ch, "T1")
    assert [(round(iv.top, 6), round(iv.base, 6)) for iv in t1.intervals] == [(0.0, 0.3), (0.3, 1.5)]
    assert warned(rep, "tetto", "quote")
    # the same names holding depths stay depths
    d = write(tmp_path, "d.csv", "sondaggio;x;y;quota;tetto;letto;us\nD1;500040;4500010;100;0;0,4;US 1\n")
    bh.import_files(ch, [d])
    assert bh.get(ch, "D1").intervals[0].base == pytest.approx(0.4)
    # elevations in centimetres; "h" is the elevation only when nothing better is there, and said so
    cm = write(tmp_path, "cm.csv", "sondaggio;x;y;quota (cm);profondita (cm)\nC1;500050;4500010;10050;250\n")
    bh.import_files(ch, [cm])
    c1 = bh.get(ch, "C1")
    assert c1.z == pytest.approx(100.5) and c1.depth == pytest.approx(2.5)
    h = write(tmp_path, "h.csv", "sondaggio;x;y;h\nH1;500060;4500010;101,2\n")
    rep = bh.import_files(ch, [h])
    assert bh.get(ch, "H1").z == pytest.approx(101.2) and warned(rep, "'h'", "quota")


# ------------------------------------------------------------------ 2. no silent failures
def test_decimal_commas_in_a_file_separated_by_commas(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    p = write(tmp_path, "virgole.csv", "sondaggio,x,y,quota,profondita\n"
                                       "S1,500010,5,4500010,0,100,1,3,0\n"
                                       "S2,500020,4500010,5,100,2,3\n")
    rep = bh.import_files(ch, [p])
    s1 = bh.get(ch, "S1")
    assert (s1.x, s1.y, s1.z, s1.depth) == (pytest.approx(500010.5), pytest.approx(4500010.0),
                                            pytest.approx(100.1), pytest.approx(3.0))
    assert warned(rep, "virgol", "decimal")
    # a row that can be put together in more than one way is left out, with its line
    assert bh.get(ch, "S2") is None and warned(rep, "riga 3")


def test_thousands_separators_in_the_context_of_the_file(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    comma = write(tmp_path, "migliaia.csv", "sondaggio;x;y;quota\n"
                                            "A1;500.100,5;4.500.810;100,25\nA2;500 120;4 500 820;100,5\n")
    point = write(tmp_path, "thousands.csv", "hole_id;x;y;elevation\nB1;500,140.5;4,500,830;100.75\n")
    rep = bh.import_files(ch, [comma, point])
    a1, a2, b1 = bh.get(ch, "A1"), bh.get(ch, "A2"), bh.get(ch, "B1")
    assert (a1.x, a1.y, a1.z) == (pytest.approx(500100.5), pytest.approx(4500810.0), pytest.approx(100.25))
    assert (a2.x, a2.y, a2.z) == (pytest.approx(500120.0), pytest.approx(4500820.0), pytest.approx(100.5))
    assert (b1.x, b1.y, b1.z) == (pytest.approx(500140.5), pytest.approx(4500830.0), pytest.approx(100.75))
    assert rep["read"] == 3
    # a coordinate that is not a number: never a collar lost without a word
    bad = write(tmp_path, "bad.csv", "sondaggio;x;y\nZ1;cinquecento;4500010\n")
    with pytest.raises(ValueError):
        bh.import_files(ch, [bad])
    mixed = write(tmp_path, "mixed.csv", "sondaggio;x;y;da;a;us\nZ1;cinquecento;4500010;0;1;US 1\n"
                                         "Z2;500200;4500010;0;1;US 1\n")
    rep = bh.import_files(ch, [mixed])
    assert bh.get(ch, "Z1") is None and warned(rep, "Z1", "cinquecento", "riga 2")


def test_real_line_numbers_and_rows_half_filled(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    p = write(tmp_path, "righe.csv", "# campagna 2026\n\nsondaggio;x;y;da;a;us\n"
                                     "S1;500010;4500010;0;1;US 1\n\n"
                                     "S1;500010;4500010;1;;US 2\n"
                                     "# nota\n"
                                     "S1;500010;4500010;2;1,5;US 3\n")
    rep = bh.import_files(ch, [p])
    assert warned(rep, "riga 6") and warned(rep, "riga 8", "base sopra il tetto")
    assert not warned(rep, "riga 4")


def test_colours_written_in_words_and_with_a_decimal_comma(tmp_path):
    marrone = bh.parse_colour("marrone")
    assert marrone is not None
    r, g, b = (int(marrone[i:i + 2], 16) for i in (1, 3, 5))
    assert r > g > b
    grigio, scuro = bh.parse_colour("grigio"), bh.parse_colour("grigio scuro")
    assert grigio and scuro and sum(int(scuro[i:i + 2], 16) for i in (1, 3, 5)) < sum(
        int(grigio[i:i + 2], 16) for i in (1, 3, 5))
    rossastro = bh.parse_colour("bruno rossastro")
    r, g, b = (int(rossastro[i:i + 2], 16) for i in (1, 3, 5))
    assert r > g and r > b
    assert bh.parse_colour("light yellowish brown") is not None
    if bh.munsell_to_hex("2.5Y 5/2") is not None:  # colour-science
        assert bh.parse_colour("2,5Y 5/2") == bh.parse_colour("2.5Y 5/2")
    doc, ch = project(tmp_path, dem=False)
    p = write(tmp_path, "colori.csv", "sondaggio;x;y;da;a;us;colore\n"
                                      "S1;500010;4500010;0;1;US 1;grigio scuro\n"
                                      "S1;500010;4500010;1;2;US 2;xyzzy\n")
    rep = bh.import_files(ch, [p])
    assert warned(rep, "xyzzy", "riga 3") and not warned(rep, "grigio scuro")


# ------------------------------------------------------------------ 3. the unit codes in one form
def test_unit_codes_in_one_form_everywhere(tmp_path):
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    doc, ch = project(tmp_path, dem=False)
    collars = write(tmp_path, "c.csv", "sondaggio;x;y;quota\n" + "".join(
        f"{n};{500000 + 10 * k};{4500000 + 7 * (k % 2)};100\n" for k, n in enumerate(["A", "B", "C", "D"])))
    rows = {"A": ["US 1", "US 2", "US 3"], "B": ["US1", "us 2", "US3"], "C": ["1", "2", "3"],
            "D": ["us 1", "US 02", "US 3"]}
    text = "sondaggio;da;a;us\n" + "".join(f"{bid};{k};{k + 1};{u}\n" for bid, us in rows.items()
                                          for k, u in enumerate(us))
    rep = bh.import_files(ch, [collars, write(tmp_path, "i.csv", text)])
    assert rep["units"] == ["US 1", "US 2", "US 3"]
    assert all(b.units == ["US 1", "US 2", "US 3"] for b in bh.load(ch))
    assert warned(rep, "US1", "US 1")  # the spellings read as one unit are reported
    written = {(r["sondaggio"], r["unita"]): r["unita_scritta"]
               for r in export.read_attributes(export.layers_path(ch), "intervalli")}
    assert written[("B", "US 1")] == "US1" and written[("C", "US 3")] == "3"  # the code as written is kept
    assert bh.set_order(ch, ["us3", "US 2", "1"]) == ["US 3", "US 2", "US 1"]
    bh.set_order(ch, ["US 1", "US 2", "US 3"])
    reg = sf.build(ch, mode="auto, us2=depth")
    assert reg["modes"]["US 2"] == "depth" and [e["unit"] for e in reg["contacts"]] == ["US 1", "US 2", "US 3"]
    assert sf.unit_of("us 3", bh.order(ch)) == "US 3" and sf.unit_of("3", bh.order(ch)) == "US 3"
    ch.meta["drawings"] = {"units": [UnitPolygon(box(499990, 4499990, 500040, 4500020), us="2", tipo="US").to_dict()]}
    out = model.build(ch)
    assert {r["unit"]: r["clipped"] for r in out["units"]}["US 2"]
    assert logs.draw(ch, ["A", "B", "C", "D"])
    # a plain number shared by two kinds stays apart, and is reported
    (tmp_path / "two").mkdir()
    doc2, ch2 = project(tmp_path / "two", dem=False)
    t = write(tmp_path / "two", "t.csv", "sondaggio;x;y;da;a;us\nP;0;0;0;1;US 4\nP;0;0;1;2;USM 4\nQ;5;0;0;1;4\n")
    rep = bh.import_files(ch2, [t])
    assert "4" in rep["units"] and warned(rep, "4", "US 4", "USM 4")


# ------------------------------------------------------------------ 4. hatches
def test_hatch_of_the_head_noun():
    assert bh.hatch_of("ghiaia sabbiosa") == "oo"
    assert bh.hatch_of("limo argilloso") == "-."
    assert bh.hatch_of("sabbia limosa") == ".."
    assert bh.hatch_of("sandy gravel") == "oo" and bh.hatch_of("silty clay") == "--"
    assert bh.hatch_of("clayey silt") == "-."
    assert bh.hatch_of("concrezioni carbonatiche") == ""  # no "carbon" inside a word
    assert bh.hatch_of("argilla con concrezioni carbonatiche") == "--"
    assert bh.hatch_of("cenere e carboni") == "xx"
    mud = bh.hatch_of("mattone crudo, cenere, battuti")
    assert mud and mud not in ("xx", "**") and bh.hatch_of("mudbricks") == mud
    assert bh.hatch_of("mudbrick collapse and colluvium") == "**"  # English: the last noun of the compound
    assert bh.hatch_of("crollo di mattoni crudi") == "**"
    assert bh.hatch_of("terreno vegetale (humus)") == "\\\\"
    # the lithology first, then the description
    assert bh.hatch_of("ghiaia", "sabbia limosa") == "oo"
    assert bh.hatch_of("", "sabbia limosa") == ".." and bh.hatch_of("misto", "argilla") == "--"


# ------------------------------------------------------------------ 5. the corrections on a diverging ramp
def test_the_correction_of_the_stack_on_a_diverging_ramp():
    values = np.array([[-2.0, -0.5, 0.0, 0.3, 1.0]])
    lo, hi = overlay.limits(values, symmetric=True)
    assert lo == -hi and hi > 0
    palette, zero, symmetric = overlay.style("surface", 3)
    assert palette == "diverging" and symmetric and not zero
    rgba = overlay.render(np.array([[-1.0, 0.0, 1.0]]), lo, hi, palette)
    assert tuple(rgba[0, 0, :3]) != tuple(rgba[0, 1, :3]) != tuple(rgba[0, 2, :3])  # raised, unmoved, lowered
    assert overlay.style("surface", 2)[1] and overlay.style("thickness", 1)[1]  # variance, thickness: from zero


# ------------------------------------------------------------------ 6. Italian texts
def test_italian_texts():
    from ppg_geoarchaeology import GeoarchaeologyPlugin

    reg = {"grid": {"width": 10, "height": 10, "cell": 0.5}, "thickness": [],
           "contacts": [{"name": "Tetto di US 5", "method": "kriging", "trend": "none", "loocv_rmse": 0.274,
                         "variogram": {"model": "spherical"}, "mode": "elevation", "mode_reason": {"rule": "given"},
                         "least_error": {"model": "exponential", "trend": "none", "loocv_rmse": 0.271}}]}
    text = GeoarchaeologyPlugin.surfaces_text(reg)
    assert "il esponenziale" not in text and "esponenziale dava 0,271 m" in text
    one = GeoarchaeologyPlugin.import_text({"read": 1, "boreholes": 1, "intervals": 1, "units": ["US 1"],
                                            "z_from_dem": 1})
    assert one.startswith("1 sondaggio letto (1 nel chunk), 1 intervallo;") and "1 quota presa dal DEM" in one


# ------------------------------------------------------------------ 7. the smaller ones
def _three(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    t = write(tmp_path, "t.csv", "sondaggio;x;y;quota;da;a;us\n" + "".join(
        f"S{k};{500000 + 10 * k};4500000;100;{a};{b};{u}\n" for k in (1, 2, 3)
        for a, b, u in ((0, 1, "US 1"), (1, 2, "US 2"))))
    bh.import_files(ch, [t])
    return doc, ch


def test_every_panel_of_columns_keeps_its_file(tmp_path):
    doc, ch = _three(tmp_path)
    first = logs.draw(ch, ["S1", "S2"])
    second = logs.draw(ch, ["S2", "S3"])
    assert first[0] != second[0] and all(p.exists() for p in first + second)
    assert logs.draw(ch, ["S1"])[0].stem == "S1"


def test_the_section_through_the_boreholes_in_the_order_chosen(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QApplication

    from ppg_geoarchaeology.pane import GeoarchaeologyPane

    QApplication.instance() or QApplication([])
    doc, ch = _three(tmp_path)
    pane = GeoarchaeologyPane()
    pane.set_state(bh.load(ch), [], [], "")
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in (2, 0, 1):
        pane.t_holes.selectionModel().select(pane.t_holes.model().index(row, 0), flags)
    assert pane.selected_boreholes() == ["S3", "S1", "S2"]
    pane.t_holes.selectionModel().select(pane.t_holes.model().index(0, 0),
                                         QItemSelectionModel.SelectionFlag.Deselect
                                         | QItemSelectionModel.SelectionFlag.Rows)
    assert pane.selected_boreholes() == ["S3", "S2"]


def test_a_local_chunk_has_its_layers_in_the_qgis_project(tmp_path):
    doc, ch = local_chunk(tmp_path)
    t = write(tmp_path, "t.csv", "sondaggio;x;y;quota;da;a;us\n" + "".join(
        f"L{k};{10 * k};{3 * (k % 2)};100;{a};{b};{u}\n" for k in (1, 2, 3, 4)
        for a, b, u in ((0, 1, "US 1"), (1, 2, "US 2"))))
    bh.import_files(ch, [t])
    sf.build(ch)
    rasters, vectors = export.qgis_entries(ch)
    assert rasters and {v["table"] for v in vectors} == {"sondaggi"}


# ------------------------------------------------------------------ 8. an installation without the vector backends
def test_figures_without_the_svg_and_pdf_backends(tmp_path, monkeypatch, capsys):
    """The packaged application up to 1.3.0 carries only Agg, the backend of the PNG: the SVG and
    the PDF are skipped, the PNG written, the user told (and an older SVG of that name removed)."""
    import sys

    from matplotlib import backend_bases

    from ppg_geoarchaeology import GeoarchaeologyPlugin, section
    from pyarchinit_photogrammetry.plugins import Context

    modules = {"svg": "svg", "svgz": "svg", "pdf": "pdf", "eps": "ps", "ps": "ps"}
    for fmt, mod in modules.items():  # loaded again, as the first time a file of that format is written
        monkeypatch.setitem(backend_bases._default_backends, fmt, f"matplotlib.backends.backend_{mod}")
    for mod in set(modules.values()):  # ModuleNotFoundError, as in the application
        monkeypatch.setitem(sys.modules, f"matplotlib.backends.backend_{mod}", None)
    doc, ch = project(tmp_path)
    t = write(tmp_path, "t.csv", "sondaggio;x;y;da;a;us\n" + "".join(
        f"S{k};{500000 + 20 * k};4500040;{a};{b};{u}\n" for k in (1, 2, 3)
        for a, b, u in ((0, 1, "US 1"), (1, 2, "US 2"))))
    bh.import_files(ch, [t])
    doc.save()
    old = logs.folder(ch) / "S1.svg"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("<svg/>")
    drawn = logs.draw(ch, ["S1"])
    assert [p.suffix for p in drawn] == [".png"] and drawn[0].stat().st_size > 1000 and not old.exists()
    assert len(drawn.notes) == 1 and "SVG" in drawn.notes[0] and "PNG" in drawn.notes[0]
    pdf = logs.draw(ch, ["S1", "S2"], path=tmp_path / "colonne.pdf")
    assert list(pdf) == [tmp_path / "colonne.png"] and pdf[0].exists() and "PDF" in pdf.notes[0]
    e = section.build(ch, section.through(ch, ["S1", "S3"]), name="Senza SVG")
    assert set(e["files"]) == {"png", "dxf", "csv"} and "SVG" in e["warnings"][0]
    assert "warnings" not in section.entries(ch)[-1]  # said to the user, not kept in the project
    assert drawn.notes[0] == "SVG non disponibile in questa installazione: scritto solo il PNG"
    # the command line says it too
    plugin = GeoarchaeologyPlugin()
    plugin.activate(Context("geoarchaeology", tmp_path / "plugin"))
    capsys.readouterr()
    assert plugin.command("logs", str(doc.path), "S2", "S3") == 0
    out = capsys.readouterr().out
    assert "colonne_S2-S3.png" in out and "colonne_S2-S3.svg" not in out and "SVG non disponibile" in out


# ------------------------------------------------------------------ 9. the model of the units in the 3D view
def test_the_model_of_the_units_in_the_3d_view(tmp_path, monkeypatch):
    """After *Modello 3D...* the units are in the program's 3D view, each in its colour, written in
    the frame the view expects (the internal one of the chunk, taken to the world by its
    transform); the pane shows them again and opens their folder."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.core import geometry
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    qapp = QApplication.instance() or QApplication([])
    doc, ch = project(tmp_path)
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    T = geometry.identity()  # an internal frame that is not the world: rotated, scaled, moved
    T[:3, :3] = 2.0 * np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [500050.0, 4500040.0, 100.0]
    ch.transform = T
    t = write(tmp_path, "t.csv", "sondaggio;x;y;da;a;us;colore\n" + "".join(
        f"S{k};{500000 + 20 * k};4500040;{a};{b};{u};{col}\n" for k in (1, 2, 3)
        for a, b, u, col in ((0, 1, "US 1", "#8b5a2b"), (1, 2, "US 2", "#c9a55c"))))
    bh.import_files(ch, [t])
    sf.build(ch)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    monkeypatch.setattr(w, "_warn", lambda text: None)
    monkeypatch.setattr(w, "_info", lambda title, text: None)
    plugin = GeoarchaeologyPlugin()
    messages, opened = [], []
    try:
        plugin.activate(Context("geoarchaeology", tmp_path / "plugin", window=w))
        pane = plugin.pane
        assert not pane.b_model_view.isEnabled() and not pane.b_model_folder.isEnabled()
        monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
            lambda parent, title, fields, intro="": {"cell": None, "clip": True, "last": True, "floor": None,
                                                     "explode": 0.5}))
        monkeypatch.setattr(plugin, "_run", lambda title, job, done=None: done(job(None)) if done else job(None))
        monkeypatch.setattr(plugin, "_message", lambda kind, text: messages.append(text))
        monkeypatch.setattr(plugin, "_open_file", lambda path: opened.append(Path(path)))
        w.show_view(w.ortho_view)
        plugin.model_dialog()
        reg = model.registry(ch)
        shown = ch.data_dir / reg["files"]["display"]
        assert shown.name == "unita_vista.ply" and reg["explode"] == 0.5
        # the display file: the units of the model, in the internal frame, the upper one raised by the gap
        units = [read_mesh(ch.data_dir / r["ply"]) for r in reg["units"] if r.get("ply")]
        assert len(units) == 2
        expected = np.vstack([m.xyz + [0.0, 0.0, 0.5 * (len(units) - 1 - k)] for k, m in enumerate(units)])
        both = read_mesh(shown)
        assert np.allclose(ch.to_world(both.xyz), expected, atol=1e-6)
        assert len(both.faces) == sum(len(m.faces) for m in units)
        assert {tuple(c) for c in both.colors} == {(0x8B, 0x5A, 0x2B), (0xC9, 0xA5, 0x5C)}
        # the 3D view shows them, where they are in the world, each in its colour
        assert w.tabs.currentWidget() is w.model_view and w.model_view.paths == {"model": str(shown)}
        sd = w.model_view.sd
        assert np.allclose(sd.mesh_vertices.astype(np.float64) + sd.offset, expected, atol=0.3)  # float32 at 4.5e6
        assert {tuple(c) for c in sd.mesh_colors} == {(0x8B, 0x5A, 0x2B), (0xC9, 0xA5, 0x5C)}
        assert w.model_view.version_label.text() == "Unità geoarcheologiche (2 unità)"
        assert "Mostra nella vista 3D" in messages[-1] and "Apri la cartella" in messages[-1]
        # the pane: the units again, and their folder
        assert pane.b_model_view.isEnabled() and pane.b_model_folder.isEnabled()
        w.show_view(w.ortho_view)
        pane.b_model_view.click()
        assert w.tabs.currentWidget() is w.model_view
        pane.b_model_folder.click()
        assert opened == [model.folder(ch)]
        # a model of 0.1.1 has no file for the view: it is written when the units are shown
        shown.unlink()
        reg["files"].pop("display")
        assert plugin.show_model() and shown.exists() and model.registry(ch)["files"]["display"]
        assert np.allclose(ch.to_world(read_mesh(shown).xyz), expected, atol=1e-6)
        qapp.processEvents()
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
