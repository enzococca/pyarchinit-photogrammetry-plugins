"""The outputs of a campaign on the synthetic face: the plate (SVG, PDF) with the legend in both glossaries and the
classes, the tables (Excel and CSV) with the records' extents, the DXF (a layer per family, a block per term), the two
GeoPackages (unfolded and 3D in the chunk's system), the QGIS styles and project, PyArchInit (media of the elevation,
the condition of the linked US forms written only by ``save``)."""
from __future__ import annotations

import csv
import sqlite3

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the outputs live on the program's elevations")

import conservation_samples as S  # noqa: E402
from ppg_conservation import assess, outputs, records  # noqa: E402
from ppg_conservation.records import NotReady  # noqa: E402

Z0 = S.Z0
DECAY = ("ISCS.CRUST.BLACK_CRUST", "UNI.PATINA_BIOLOGICA", "ISCS.EFFLORESCENCE", "ISCS.MISSING_PART")


def campaign(tmp_path):
    """The face with a campaign of six features (the painted patches, the hole, the groove, a cleaning of the crust),
    a note and a proposal (never in the outputs); the classes CC 2, UC 1, the RC 2 of the date."""
    doc, ch, entry = S.face(tmp_path)
    c = records.new_campaign(ch, entry["id"], surveyor="E. Cocca", method="rilievo fotogrammetrico")
    records.set_campaign(ch, c["id"], classes={"cc": 2, "uc": 1, "reason": "lesione aperta"})
    records.set_assessment(ch, c["date"], 2, reason="riparazione")
    cid = c["id"]
    feats = {
        "crust": records.add_feature(ch, cid, "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST),
                                     intensity="high", urgency=2),
        "patina": records.add_feature(ch, cid, "decay", "UNI.PATINA_BIOLOGICA", S.box(S.GREEN_PATCH)),
        "efflo": records.add_feature(ch, cid, "decay", "ISCS.EFFLORESCENCE", S.box(S.EFFLORESCENCE),
                                     intensity="low"),
        "hole": records.add_feature(ch, cid, "decay", "ISCS.MISSING_PART", S.box(S.HOLE), depth_mm=30.0),
        "crack": records.add_feature(ch, cid, "crack", "ISCS.CRACK", S.groove(), width_mm=3.0, cause="assestamento"),
        "clean": records.add_feature(ch, cid, "treatment", "INT.CLEANING", S.box(S.BLACK_CRUST),
                                     treatment={"technique": "INT.CLEANING", "materials": "acqua deionizzata",
                                                "date": "2026-06-01", "done": True, "by": "Restauri srl"}),
        "note": records.add_feature(ch, cid, "note", "ISCS.CRUST", f"POINT (3 {Z0 + 1.5})", notes="campione"),
        "proposed": records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box((5.4, Z0 + 2.0, 5.8, Z0 + 2.4)),
                                        status="proposed", source="proposed:colours"),
    }
    doc.save()
    return doc, ch, entry, c, feats


@pytest.fixture
def made(tmp_path):
    return campaign(tmp_path)


def test_the_plate(made, tmp_path):
    doc, ch, entry, c, feats = made
    out = outputs.plate(ch, c["id"], tmp_path / "tav")
    assert set(out) >= {"svg", "pdf", "scale"} and out["scale"] == 20
    assert out["svg"].stat().st_size > 10_000 and out["pdf"].read_bytes()[:4] == b"%PDF"
    svg = out["svg"].read_text(encoding="utf-8")
    # the legend: both glossaries' names of the terms present, the plugin's colours said as such
    for word in ("Black crust", "crosta", "Biological colonization", "patina biologica", "Efflorescence",
                 "efflorescenza", "Missing part", "mancanza", "Crack", "fessurazione", "pulitura", "Cleaning"):
        assert word in svg, word
    assert "ISCS.CRUST.BLACK_CRUST" in svg and "ISCS.ALGA" not in svg  # a proposal is not part of the campaign
    assert "colori e retini del plugin" in svg
    # the title block: the campaign and its classes
    for word in (entry["name"], c["name"], c["date"], "E. Cocca", "rilievo fotogrammetrico", "1:20",
                 "CC 2 - Sintomi moderatamente forti", "UC 1 - Medio termine", "RC 2"):
        assert word in svg, word
    assert "3 mm" in svg  # the opening of the crack beside it
    assert "<pattern" in svg  # the hatches
    # the default folder is the campaign's
    default = outputs.plate(ch, c["id"], formats=("svg",))
    assert default["svg"].parent == records.campaign_folder(ch, entry["id"], c["id"])


def test_the_tables(made, tmp_path):
    from openpyxl import load_workbook

    doc, ch, entry, c, feats = made
    out = outputs.tables(ch, c["id"], tmp_path / "tab")
    wb = load_workbook(out["xlsx"])
    assert wb.sheetnames == ["Elementi", "Aree per termine", "Aree per famiglia", "Lesioni", "Interventi",
                             "Rapporto EN 16096"]
    rows = list(wb["Elementi"].values)
    head = rows[0]
    assert len(rows) - 1 == 7  # the accepted features, not the proposal
    for col in ("id", "Tipo", "Codice ISCS", "Termine ISCS", "Codice UNI", "Termine UNI", "Famiglia", "Estensione",
                "Intensità", "Apertura (mm)", "Profondità (mm)", "Causa", "Urgenza", "Stato", "Fonte", "Foto", "US",
                "Note"):
        assert col in head, col
    by_id = {r[head.index("id")]: r for r in rows[1:]}
    crust = by_id[feats["crust"]["id"]]
    assert crust[head.index("Codice ISCS")] == "ISCS.CRUST.BLACK_CRUST"
    assert crust[head.index("Termine ISCS")] == "Black crust" and crust[head.index("Codice UNI")] == "UNI.CROSTA"
    assert crust[head.index("Termine UNI")] == "crosta"
    assert crust[head.index("Estensione")] == pytest.approx(feats["crust"]["extent"], abs=1e-4)
    patina = by_id[feats["patina"]["id"]]
    assert "ISCS.ALGA" in patina[head.index("Codice ISCS")] and patina[head.index("Termine UNI")] == "patina biologica"
    areas = list(wb["Aree per termine"].values)
    ah = areas[0]
    assert len(areas) - 1 == len(DECAY)
    got = {r[ah.index("Codice")]: r[ah.index("Area (m²)")] for r in areas[1:]}
    for key in ("crust", "patina", "efflo", "hole"):
        assert got[feats[key]["code"]] == pytest.approx(feats[key]["extent"], abs=1e-4)
    family = {r[0]: r for r in list(wb["Aree per famiglia"].values)[1:]}
    assert family["discoloration_deposit"][3] == pytest.approx(feats["crust"]["extent"] + feats["efflo"]["extent"],
                                                               abs=1e-4)
    cracks = list(wb["Lesioni"].values)
    assert len(cracks) == 2 and cracks[1][cracks[0].index("Lunghezza (m)")] == pytest.approx(S.groove().length,
                                                                                              abs=1e-4)
    assert cracks[1][cracks[0].index("Apertura (mm)")] == 3.0
    treat = list(wb["Interventi"].values)
    assert len(treat) == 2 and "acqua deionizzata" in treat[1] and "sì" in treat[1]
    report = [r[0] for r in list(wb["Rapporto EN 16096"].values)[1:]]
    assert [x or "" for x in report] == assess.report_lines(ch, c["id"])
    with open(out["csv_features"], encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 8
    with open(out["csv_areas"], encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == len(DECAY) + 1


def test_the_dxf(made, tmp_path):
    import ezdxf

    doc, ch, entry, c, feats = made
    path = outputs.dxf(ch, c["id"], tmp_path / "d" / "campagna.dxf")
    d = ezdxf.readfile(path)
    names = {layer.dxf.name for layer in d.layers}
    assert {"DEGRADO_DISCOLORATION_DEPOSIT", "DEGRADO_BIOLOGICAL", "DEGRADO_MATERIAL_LOSS", "LESIONI", "INTERVENTI",
            "NOTE", "BORDO"} <= names
    from ppg_conservation.vocabulary import legend

    colour = legend(ch)["ISCS.CRACK"]["colour"]
    assert d.layers.get("LESIONI").rgb == tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
    msp = d.modelspace()
    closed = [e for e in msp.query("LWPOLYLINE") if e.closed and e.dxf.layer.startswith("DEGRADO_")]
    assert len(closed) == 4
    assert len(msp.query('LWPOLYLINE[layer=="LESIONI"]')) == 1 and len(msp.query('POINT[layer=="NOTE"]')) == 1
    assert len([e for e in msp.query('LWPOLYLINE[layer=="INTERVENTI"]') if e.closed]) == 1
    inserts = list(msp.query("INSERT"))
    assert len(inserts) == 7
    attribs = {(i.get_attrib_text("CODE"), i.get_attrib_text("NAME_IT"), i.get_attrib_text("NAME_EN")) for i in inserts}
    assert ("ISCS.CRUST.BLACK_CRUST", "crosta nera (UNI: crosta)", "Black crust") in attribs
    assert any(code == "ISCS.CRACK" for code, *_ in attribs)
    lines = msp.query('LWPOLYLINE[layer=="LESIONI"]')[0]
    assert np.allclose([p[:2] for p in lines.get_points()], S.GROOVE)


def test_the_geopackages(made, tmp_path):
    from shapely import wkt

    from pyarchinit_photogrammetry.io import gpkg

    doc, ch, entry, c, feats = made
    out = outputs.geopackages(ch, c["id"], tmp_path / "g", "campagna")
    assert out["gpkg"].name == "campagna.gpkg" and out["gpkg_world"].name == "campagna_mondo.gpkg"
    count = {"degrado": 4, "lesioni": 1, "interventi": 1, "note": 1}
    for table, n in count.items():
        rows = gpkg.read_table(out["gpkg"], table)
        assert len(rows) == n, table
        for geom, attrs in rows:
            world = wkt.loads(attrs["geom_world"])
            assert world.has_z and np.allclose(np.asarray(world.bounds[:2]), S.WORLD[:2], atol=10)
    crust = next(a for g, a in gpkg.read_table(out["gpkg"], "degrado") if a["id"] == feats["crust"]["id"])
    assert crust["code"] == "ISCS.CRUST.BLACK_CRUST" and crust["code_uni"] == "UNI.CROSTA"
    assert crust["extent"] == pytest.approx(feats["crust"]["extent"], abs=1e-6)
    con = sqlite3.connect(out["gpkg"])
    assert {r[0] for r in con.execute("SELECT srs_id FROM gpkg_geometry_columns")} == {-1}
    con.close()
    con = sqlite3.connect(out["gpkg_world"])
    assert set(con.execute("SELECT srs_id, z FROM gpkg_geometry_columns")) == {(32633, 1)}
    con.close()
    for geom, attrs in gpkg.read_table(out["gpkg_world"], "degrado"):
        assert geom.has_z and abs(np.asarray(geom.exterior.coords)[:, 2] - Z0).max() < 3.5
    # the hole's polygon lies on the plane of the face (d = 0): its world points have their (u, v) back
    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict

    frame = frame_from_dict(entry["frame"])
    hole = next(g for g, a in gpkg.read_table(out["gpkg_world"], "degrado") if a["id"] == feats["hole"]["id"])
    uvd = frame.coords(np.asarray(hole.exterior.coords))
    assert np.allclose(uvd[:, 2], 0, atol=1e-6) and np.allclose(uvd[:, :2].min(0), S.HOLE[:2], atol=1e-6)


def test_qgis_styles_and_project(made, tmp_path, monkeypatch, caplog):
    from pyarchinit_photogrammetry.io import qgis_link

    from ppg_conservation.vocabulary import legend

    doc, ch, entry, c, feats = made
    styles = outputs.qgis_styles(tmp_path / "s", {"degrado": DECAY, "lesioni": ["ISCS.CRACK"]}, legend(ch))
    assert set(styles) == {"degrado", "lesioni"} and styles["degrado"].name == "degrado.qml"
    text = styles["degrado"].read_text(encoding="utf-8")
    assert all(f'value="{code}"' in text for code in DECAY) and "categorizedSymbol" in text
    assert "<layerGeometryType>1</layerGeometryType>" in styles["lesioni"].read_text(encoding="utf-8")

    calls = {}

    def build(project, crs, layers, rasters, title="", qgis=None, extra_vectors=None, **k):
        calls["build"] = (project, rasters, extra_vectors, title)
        layers_xml = "".join(
            f'<maplayer><datasource>{v["path"]}|layername={v["table"]}</datasource><layername>{v["name"]}</layername>'
            f'<renderer-v2 type="singleSymbol"><symbols/></renderer-v2></maplayer>' for v in extra_vectors)
        project.write_text(f"<qgis><projectlayers>{layers_xml}</projectlayers></qgis>", encoding="utf-8")
        return {"written": True}

    monkeypatch.setattr(qgis_link, "find_qgis", lambda: None)
    with pytest.raises(NotReady):
        outputs.qgis_project(ch, c["id"], tmp_path / "q")
    monkeypatch.setattr(qgis_link, "find_qgis", lambda: "QGIS")
    monkeypatch.setattr(qgis_link, "build_project", build)
    monkeypatch.setattr(qgis_link, "open_in_qgis", lambda project, q=None: calls.setdefault("open", project))
    project = outputs.qgis_project(ch, c["id"], tmp_path / "q")
    assert calls["open"] == project and project.suffix == ".qgs"
    _p, rasters, vectors, title = calls["build"]
    assert rasters and rasters[0]["name"] == entry["name"] and c["name"] in title
    ours = {v["table"]: v for v in vectors if v["path"].endswith("_conservazione.gpkg")}
    assert set(ours) == {"degrado", "lesioni", "interventi", "note"}
    assert any(v["table"] == "unita" for v in vectors)  # the program's own layers of the elevation
    for table in ours:
        assert (project.parent / f"{table}.qml").exists()
    xml = project.read_text(encoding="utf-8")
    assert xml.count('type="categorizedSymbol"') == 4 and "ISCS.CRUST.BLACK_CRUST" in xml
    # a project where the styles could not go: said in the log, not silent
    import logging

    monkeypatch.setattr(qgis_link, "build_project", lambda project, *a, **k: project.write_text("<qgis/>"))
    with caplog.at_level(logging.WARNING, logger="ppg_conservation"):
        outputs.qgis_project(ch, c["id"], tmp_path / "q2", open_it=False)
    assert "0 of the 4 layers" in caplog.text


def test_the_whole_export(made):
    from ppg_conservation.vocabulary import UNNAMED

    doc, ch, entry, c, feats = made
    seg = records.add_feature(ch, c["id"], "decay", UNNAMED, S.box((5.4, Z0 + 0.1, 5.8, Z0 + 0.3)),
                              status="proposed", source="proposed:sam")
    seg["status"] = "accepted"  # as an older project might hold it: still never exported
    out = outputs.export_campaign(ch, c["id"])
    assert out["unnamed"] == 1 and out["unnamed_proposed"] == 0
    assert outputs.unnamed_text(ch, c["id"]) == "1 elementi da nominare esclusi"
    records.add_feature(ch, c["id"], "decay", UNNAMED, S.box((5.4, Z0 + 0.4, 5.8, Z0 + 0.6)), status="proposed")
    assert "1 proposte ancora da nominare" in outputs.unnamed_text(ch, c["id"])
    assert UNNAMED not in out["plate_svg"].read_text(encoding="utf-8")
    assert len(outputs.table_data(ch, c["id"])["Elementi"]["rows"]) == 7
    folder = records.campaign_folder(ch, entry["id"], c["id"])
    for key in ("plate_svg", "plate_pdf", "xlsx", "csv_features", "csv_areas", "dxf", "gpkg", "gpkg_world"):
        assert out[key].exists() and out[key].parent == folder, key
    assert {p.name for p in folder.glob("*.qml")} == {"degrado.qml", "lesioni.qml", "interventi.qml", "note.qml"}


class FakePyArchInit:
    """What the plugin asks of the program's bridge, recorded (the masonry plugin's fake, with the US forms)."""

    def __init__(self, words=None):
        self.forms, self.media, self.updates = {}, [], []
        self.words = words or []

    def us_record(self, sito, area, us, unita_tipo=None):
        return self.forms.get((str(area), str(us)))

    def update_us(self, id_us, **fields):
        self.updates.append((id_us, fields))
        for rec in self.forms.values():
            if rec["id_us"] == id_us:
                rec.update(fields)
        return 1

    def thesaurus(self, code, table="pottery_table", lingua=None):
        return list(self.words) if (code, table) == ("stato_di_conservazione", "us_table") else []

    def register_elevation_media(self, chunk, entry, paths, sito=None, link_us=None):
        for link in link_us or []:  # as the program's bridge: a link to a form that is not there stops everything
            if (str(link[0]), str(link[1])) not in self.forms:
                raise KeyError(f"US {link[1]} (area {link[0]}) not found in site {sito}")
        self.media.append((entry, paths, sito, link_us))
        entry["media"] = {"dxf": 11, "svg": 12}
        return [11, 12]


def test_pyarchinit_save(made, tmp_path):
    from ppg_conservation import pyarchinit

    doc, ch, entry, c, feats = made
    S.put_units(ch, entry["id"], {"5": S.box((0.0, Z0, 3.0, Z0 + 3.0)), "6": S.box((3.0, Z0, 6.0, Z0 + 3.0))})
    for f in records.features(ch, c["id"]):
        records.link_units(ch, entry, f["id"])
    pa = FakePyArchInit()
    pa.forms[("1", "5")] = {"id_us": 50, "osservazioni": "muro in blocchi"}
    entry.setdefault("media", {"dxf": 1, "svg": 2})  # the program's own drawings, published before
    with pytest.raises(NotReady):  # no site: nothing written
        pyarchinit.save(pa, ch, c["id"])
    assert pa.media == [] and pa.updates == []
    out = pyarchinit.save(pa, ch, c["id"], sito="Saggio", folder=tmp_path / "pa")
    assert out["media"] == [11, 12] and entry["media"] == {"dxf": 1, "svg": 2}  # the program's media left alone
    shadow, paths, sito, link = pa.media[0]
    assert sito == "Saggio" and "conservazione" in shadow["name"] and shadow["id"] == entry["id"]
    assert paths["dxf"].exists() and paths["svg"].exists() and paths["svg"].suffix == ".svg"
    assert [tuple(x) for x in link] == [("1", "5", "US")]  # the unit without its form is not linked
    assert c["media"] == {"dxf": 11, "svg": 12}
    assert out["us"] == [("1", "5", "US", 50)] and out["missing"] == [("1", "6", "US")]
    (id_us, fields), = pa.updates
    assert id_us == 50 and fields["stato_di_conservazione"] == "mediocre"
    obs = fields["osservazioni"]
    assert obs.startswith("muro in blocchi\n[conservazione 2026-05-14] ") and "crosta nera" in obs
    assert "efflorescenza" in obs and "m²" in obs and "patina" not in obs  # the terms of this unit only
    # saved again: the line of the campaign replaced, not repeated; the database's words when it has four
    pa.words = [("1", "Ottimo"), ("2", "Buono"), ("3", "Mediocre"), ("4", "Pessimo")]
    pyarchinit.save(pa, ch, c["id"], sito="Saggio", folder=tmp_path / "pa")
    fields = pa.updates[-1][1]
    assert fields["stato_di_conservazione"] == "Mediocre"
    assert fields["osservazioni"].count("[conservazione 2026-05-14]") == 1
    assert pa.media[-1][0]["media"] == {"dxf": 11, "svg": 12}  # the same media records again
    # the words chosen in the dialog win
    pyarchinit.save(pa, ch, c["id"], sito="Saggio", words=("A", "B", "C", "D"), folder=tmp_path / "pa")
    assert pa.updates[-1][1]["stato_di_conservazione"] == "C"
    # no class chosen: the observations only
    records.set_campaign(ch, c["id"], classes={"cc": None})
    pyarchinit.save(pa, ch, c["id"], sito="Saggio", folder=tmp_path / "pa")
    assert "stato_di_conservazione" not in pa.updates[-1][1]


def test_pyarchinit_round_trip_on_a_demo_database(made, tmp_path):
    """The program's own bridge on a PyArchInit 5 database made by the program: media (DXF, SVG, PDF) of a
    "Prospetto" document, the linked US form written, nothing else."""
    from ppg_conservation import pyarchinit
    from pyarchinit_photogrammetry.pyarchinit import demo_db
    from pyarchinit_photogrammetry.pyarchinit.bridge import PyArchInit

    doc, ch, entry, c, feats = made
    cfg = demo_db.create_demo_database(tmp_path / "home", demo_content=False)
    pa = PyArchInit.connect(config_path=cfg, media_root=tmp_path / "media")
    sito = demo_db.DEMO_SITE
    pa.create_us(sito, "1", "5", osservazioni="muro in blocchi")  # US 6 is drawn on the face but has no form
    S.put_units(ch, entry["id"], {"5": S.box((0.0, Z0, 3.0, Z0 + 3.0)), "6": S.box((3.0, Z0, 6.0, Z0 + 3.0))},
                sito=sito)
    for f in records.features(ch, c["id"]):
        records.link_units(ch, entry, f["id"])
    before = len(pa.db.query("media_table"))
    out = pyarchinit.save(pa, ch, c["id"], sito=sito)
    assert out["missing"] == [("1", "6", "US")] and [x[:3] for x in out["us"]] == [("1", "5", "US")]
    assert len(out["media"]) == 3 and set(c["media"]) == {"dxf", "svg", "pdf"}
    media = pa.db.query("media_table")
    assert len(media) == before + 3
    assert {m["filepath"].rsplit(".", 1)[-1] for m in media[-3:]} == {"dxf", "svg", "pdf"}
    rec = pa.us_record(sito, "1", "5")
    assert rec["stato_di_conservazione"] == "mediocre"
    assert rec["osservazioni"].startswith("muro in blocchi\n[conservazione 2026-05-14] crosta nera")
    docs = pa.db.query("documentazione_table", sito=sito, tipo_documentazione="Prospetto")
    assert [d["nome_doc"] for d in docs] == [f"{entry['name']} - conservazione 2026-05-14"]
    assert not entry.get("media")  # the elevation's own media record untouched
    pyarchinit.save(pa, ch, c["id"], sito=sito)  # again: the same records
    assert len(pa.db.query("media_table")) == before + 3
    pdf = c["media"]["pdf"]
    entry["name"] = "Prospetto rinominato"  # renamed: still the same media records, the PDF too
    pyarchinit.save(pa, ch, c["id"], sito=sito)
    assert len(pa.db.query("media_table")) == before + 3 and c["media"]["pdf"] == pdf
    assert pa.us_record(sito, "1", "5")["osservazioni"].count("[conservazione") == 1


def test_outputs_from_the_pane(tmp_path, monkeypatch):
    """The output buttons of the pane: plate, tables, drawings (and their 3D tables in the plan's QGIS project), QGIS,
    PyArchInit written only on *Salva*, the exporter of the plate."""
    pytest.importorskip("PySide6")
    from test_plugin import _window

    from ppg_conservation import ConservationPlugin
    from pyarchinit_photogrammetry.io import qgis_link
    from pyarchinit_photogrammetry.plugins import Context

    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    _d, _c, _e, c, feats = campaign(tmp_path / "copy")  # the same records, put into the window's chunk
    ch.meta["conservation"] = _c.meta["conservation"]
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        assert plugin._qgis_entries in w.qgis_providers
        said, asked, answers = [], [], {}
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append((title, intro, {f.name: f.default for f in fields}))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin.refresh_pane()
        assert plugin.pane.campaign_id() == c["id"]
        folder = records.campaign_folder(ch, entry["id"], c["id"])
        assert plugin._qgis_entries() == ([], [])  # nothing written yet
        plugin._requested("plate")
        w.runner.wait()
        assert asked[-1][2]["folder"] == str(folder) and "1:20" in said[-1]
        assert any(p.suffix == ".pdf" for p in folder.iterdir())
        answers["folder"] = str(tmp_path / "tab")
        plugin._requested("tables")
        w.runner.wait()
        assert (tmp_path / "tab").exists() and ".xlsx" in said[-1]
        answers["folder"] = str(tmp_path / "dis")
        plugin._requested("drawings")
        w.runner.wait()
        answers.clear()
        assert ".dxf" in said[-1] and "_mondo.gpkg" in said[-1]
        rasters, vectors = plugin._qgis_entries()  # from the folder the user chose
        assert rasters == [] and {v["table"] for v in vectors} == {"degrado", "lesioni", "interventi", "note"}
        assert all(v["path"].startswith(str(tmp_path / "dis")) for v in vectors)
        # QGIS
        calls = {}
        monkeypatch.setattr(qgis_link, "find_qgis", lambda: None)
        plugin._requested("qgis")
        assert "QGIS non trovato" in said[-1]
        monkeypatch.setattr(qgis_link, "find_qgis", lambda: "QGIS")

        def build(project, crs, layers, rasters, title="", qgis=None, extra_vectors=None, **k):
            calls["build"] = extra_vectors
            project.write_text("<qgis/>", encoding="utf-8")
            return {"written": True}

        monkeypatch.setattr(qgis_link, "build_project", build)
        monkeypatch.setattr(qgis_link, "open_in_qgis", lambda project, q=None: calls.setdefault("open", project))
        plugin._requested("qgis")
        w.runner.wait()
        assert calls["open"].parent == folder / "qgis" and "Progetto QGIS" in said[-1]
        # PyArchInit: not connected; cancelled; saved
        w.pa = None
        plugin._requested("pyarchinit")
        assert "Collegati prima a PyArchInit" in said[-1]
        pa = FakePyArchInit()
        pa.forms[("1", "5")] = {"id_us": 50, "osservazioni": ""}
        w.pa = pa
        S.put_units(ch, entry["id"], {"5": S.box((0.0, Z0, 3.0, Z0 + 3.0))})
        for f in records.features(ch, c["id"]):
            records.link_units(ch, entry, f["id"])
        monkeypatch.setattr(plugin, "_ask", lambda *a, **k: None)
        plugin._requested("pyarchinit")
        w.runner.wait()
        assert pa.media == [] and pa.updates == []  # nothing written without Salva
        monkeypatch.setattr(plugin, "_ask", answer)
        answers.update(sito="Saggio", w2="degradato")
        plugin._requested("pyarchinit")
        w.runner.wait()
        assert "US 5" in asked[-1][1] and asked[-1][2]["w2"] == "mediocre"
        assert pa.updates[-1][1]["stato_di_conservazione"] == "degradato" and "US 5: aggiornata" in said[-1]
        assert records.campaign(ch, c["id"])["media"] == {"dxf": 11, "svg": 12}
        w.history.undo()  # the media ids kept in the campaign are one entry of the undo
        assert "media" not in records.campaign(ch, c["id"])
        # the exporter of File > Esporta
        pdf = plugin._export_plate(ch, tmp_path / "esportata.pdf")
        assert pdf == tmp_path / "esportata.pdf" and pdf.read_bytes()[:4] == b"%PDF"
        assert not errors, errors
        plugin.deactivate()
        assert plugin._qgis_entries not in w.qgis_providers
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()


def _plugin_folder(root):
    folder = root / "plugins"
    dist = folder / "conservation" / "ppg_conservation-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-conservation\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\n"
                                           "conservation = ppg_conservation:ConservationPlugin\n")
    return folder


def test_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.cli import app

    doc, ch, entry = S.face(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()
    project = str(doc.path)

    def run(*args):
        return cli.invoke(app, ["plugin", "conservation", *args])

    r = run("list", project)
    assert r.exit_code == 0 and "Nessuna campagna" in r.output, r.output
    r = run("list", str(tmp_path / "nessuno.ppg"))
    assert r.exit_code == 1 and "Progetto non aperto" in r.output, r.output
    r = run("plate", project)
    assert r.exit_code == 1 and "Nessuna campagna" in r.output, r.output
    r = run("campaign", project, "--new", "--surveyor", "E. Cocca", "--method", "drone", "--name", "Prima")
    assert r.exit_code == 0 and "Prima" in r.output and "2026-05-14" in r.output, r.output
    r = run("map", project, "--term", "ISCS.CRUST.BLACK_CRUST", "--wkt", S.box(S.BLACK_CRUST).wkt,
            "--intensity", "high")
    assert r.exit_code == 0 and "decay" in r.output, r.output
    wkt_file = tmp_path / "patina.wkt"
    wkt_file.write_text(S.box(S.GREEN_PATCH).wkt)
    r = run("map", project, "--elevation", entry["name"], "--campaign", "Prima", "--term", "UNI.PATINA_BIOLOGICA",
            "--wkt", str(wkt_file))
    assert r.exit_code == 0, r.output
    r = run("map", project, "--term", "INT.CLEANING", "--wkt", S.box(S.BLACK_CRUST).wkt, "--done")
    assert r.exit_code == 0 and "treatment" in r.output, r.output
    r = run("map", project, "--term", "ISCS.NOTHING", "--wkt", "POINT (1 313)")
    assert r.exit_code == 1, r.output
    r = run("detect", "cracks", project, "--bounds", f"0.6,{Z0 + 1.8},2.2,{Z0 + 2.8}", "--accept")
    assert r.exit_code == 0 and "accettate" in r.output, r.output
    ch = ppg.Document.open(project).chunk
    c = records.campaigns(ch)[0]
    feats = records.features(ch, c["id"])
    cracks = [f for f in feats if f["kind"] == "crack"]
    assert cracks and all(f["status"] == "accepted" and f["source"] == "proposed:cracks" for f in cracks)
    assert {f["code"] for f in feats} >= {"ISCS.CRUST.BLACK_CRUST", "UNI.PATINA_BIOLOGICA", "INT.CLEANING"}
    assert next(f for f in feats if f["kind"] == "treatment")["treatment"]["done"] is True
    r = run("plate", project, "--folder", str(tmp_path / "tav"), "--scale", "25")
    assert r.exit_code == 0 and "1:25" in r.output and (tmp_path / "tav" / "tavola.pdf").exists(), r.output
    r = run("export", project, "--folder", str(tmp_path / "exp"))
    assert r.exit_code == 0, r.output
    assert len(list((tmp_path / "exp").glob("*.gpkg"))) == 2 and len(list((tmp_path / "exp").glob("*.qml"))) == 4
    r = run("legend", project, "--set", "ISCS.CRUST.BLACK_CRUST=#112233:xx", "--qml", str(tmp_path / "qml"))
    assert r.exit_code == 0 and "#112233" in r.output and (tmp_path / "qml" / "degrado.qml").exists(), r.output
    r = run("list", project)
    assert r.exit_code == 0 and "Prima" in r.output, r.output
