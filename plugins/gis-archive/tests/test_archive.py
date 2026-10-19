"""Archivio GIS on a small synthetic project: a sloping DEM and an orthomosaic in UTM, units,
spot heights, stones, a section and a marker drawn; written into a GeoPackage (rasters too), a
SpatiaLite file and, when a server is given (PPG_TEST_POSTGIS_URL), PostGIS."""
from __future__ import annotations

import json
import os
import sqlite3
import struct

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from ppg_gis_archive import archive as arc  # noqa: E402

CRS = "EPSG:32637"
X0, Y0, RES, SIZE = 300000.0, 3960000.0, 0.05, 200  # 10 m x 10 m


def _project(tmp_path, name="scavo"):
    import pyarchinit_photogrammetry as ppg
    from rasterio.transform import from_origin
    from shapely.geometry import box

    doc = ppg.Document()
    doc.meta["sito"] = "Tell di prova"
    ch = doc.add_chunk()
    doc.chunk = ch
    doc.save(tmp_path / f"{name}.ppg")
    ch.crs = CRS
    transform = from_origin(X0, Y0 + SIZE * RES, RES, RES)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    dem = (100.0 + 0.02 * cols + 0.01 * rows).astype(np.float32)  # rises to the west... a gentle slope
    dem[:5, :5] = -32767
    dem_path = ch.data_dir / "dem.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", width=SIZE, height=SIZE, count=1, dtype="float32", crs=CRS,
                       transform=transform, nodata=-32767) as dst:
        dst.write(dem, 1)
    ortho_path = ch.data_dir / "orthomosaic.tif"
    rgba = np.zeros((4, SIZE, SIZE), np.uint8)
    rgba[0], rgba[1], rgba[2], rgba[3] = (cols % 256), (rows % 256), 120, 255
    with rasterio.open(ortho_path, "w", driver="GTiff", width=SIZE, height=SIZE, count=4, dtype="uint8", crs=CRS,
                       transform=transform, photometric="RGB") as dst:
        dst.write(rgba)
    ch._set_product("dem", dem_path)
    ch._set_product("orthomosaic", ortho_path)
    unit = box(X0 + 2, Y0 + 2, X0 + 4, Y0 + 5)
    ch.meta["drawings"] = {
        "units": [{"wkt": unit.wkt, "sito": "Tell di prova", "area": "1", "us": "12", "tipo": "US",
                   "attributes": {"stratigraph_index_us": 2, "disegnatore": "EC"}, "layer": "US disegnate"}],
        "quote": [{"x": X0 + 3, "y": Y0 + 3, "quota_q": 100.07, "sito_q": "Tell di prova", "area_q": "1", "us_q": "12",
                   "unita_misu_q": "m", "precisione_q": 0.0012, "rilievo_originale": "Fotogrammetria (±1,2 mm)"},
                  {"x": X0 + 6, "y": Y0 + 7, "quota_q": 100.18, "area_q": "1", "us_q": "13"}]}
    ch.meta["stones"] = {"Pietre": [box(X0 + 7, Y0 + 1, X0 + 7.3, Y0 + 1.2).wkt, box(X0 + 8, Y0 + 1, X0 + 8.2, Y0 + 1.4).wkt]}
    (tmp_path / "sezioni").mkdir(exist_ok=True)
    (tmp_path / "sezioni" / "sezioni.json").write_text(json.dumps(
        [{"label": "A-A'", "p1": [X0 + 1, Y0 + 5], "p2": [X0 + 9, Y0 + 5], "length": 8.0, "svg": "sezioni/A.svg"}]))
    ch.add_marker("target 1", position=np.array([X0 + 5, Y0 + 5, 100.1]), reference=[X0 + 5.001, Y0 + 5, 100.1])
    doc.save()
    return doc


def test_geopackage_with_rasters(tmp_path):
    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    out = tmp_path / "archivio.gpkg"
    report = arc.archive(doc, str(out), interval=0.5)
    t = report["tables"]
    assert t == {"ppg_quote": 2, "ppg_us": 1, "ppg_pietre": 2, "ppg_sezioni": 1, "ppg_marker": 1,
                 "ppg_curve": t["ppg_curve"], "ppg_raster": 2} and t["ppg_curve"] > 3
    quote = gpkg.read_table(out, "ppg_quote")
    geom, values = quote[0]
    assert (geom.x, geom.y) == (X0 + 3, Y0 + 3) and values["quota"] == pytest.approx(100.07)
    assert values["progetto"] == "scavo" and values["sito"] == "Tell di prova" and values["us"] == "12"
    unit = gpkg.read_table(out, "ppg_us")[0]
    assert unit[0].geom_type == "MultiPolygon" and unit[1]["area_m2"] == pytest.approx(6.0)
    assert unit[1]["stratigraph_index_us"] == 2
    marker = gpkg.read_table(out, "ppg_marker")[0][1]
    assert marker["errore_m"] == pytest.approx(0.001, abs=1e-6) and marker["controllo"] == 1
    levels = sorted({v["quota"] for _g, v in gpkg.read_table(out, "ppg_curve")})
    assert all(abs((lv * 2) - round(lv * 2)) < 1e-6 for lv in levels)  # every 0.5 m
    # the rasters in the GeoPackage, read back by GDAL
    tables = {r["kind"]: r["table"] for r in report["rasters"]}
    assert tables == {"dem": "PPG_DEM_SCAVO_CHUNK_1", "ortofoto": "PPG_ORTOFOTO_SCAVO_CHUNK_1"}  # as rasterio writes them
    with rasterio.open(str(out), table=tables["dem"]) as src:
        back = src.read(1)
        assert src.crs.to_epsg() == 32637 and back.shape == (SIZE, SIZE)
        assert back[100, 100] == pytest.approx(100.0 + 0.02 * 100 + 0.01 * 100, abs=1e-3)
    with rasterio.open(str(out), table=tables["ortofoto"]) as src:
        assert src.count >= 3 and abs(int(src.read(1)[10, 20]) - 20) <= 4  # JPEG tiles where opaque
    footprint = gpkg.read_table(out, "ppg_raster")
    assert {v["tipo"] for _g, v in footprint} == {"dem", "ortofoto"}
    assert footprint[0][0].bounds == pytest.approx((X0, Y0, X0 + SIZE * RES, Y0 + SIZE * RES))
    # archiving again replaces the rows of this project; another project is added beside it
    arc.archive(doc, str(out), what=["quote", "dem"], interval=0.5)
    assert len(gpkg.read_table(out, "ppg_quote")) == 2
    (tmp_path / "altro").mkdir()
    other = _project(tmp_path / "altro", name="altro")
    arc.archive(other, str(out), what=["quote"])
    projects = sorted(v["progetto"] for _g, v in gpkg.read_table(out, "ppg_quote"))
    assert projects == ["altro", "altro", "scavo", "scavo"]


def test_rasters_reduced_and_tables_of_others_left_alone(tmp_path):
    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    out = tmp_path / "ridotto.gpkg"
    report = arc.archive(doc, str(out), what=["dem"], max_side=50)
    with rasterio.open(str(out), table=report["rasters"][0]["table"]) as src:
        assert (src.width, src.height) == (50, 50) and src.res == pytest.approx((RES * 4, RES * 4))
        assert src.read(1)[25, 25] == pytest.approx(100.0 + 0.03 * 100, abs=0.05)
    # a table of the same name that is not a raster is not dropped: the raster stays a reference
    con = sqlite3.connect(out)
    con.execute('CREATE TABLE "PPG_ORTOFOTO_SCAVO_CHUNK_1" (note TEXT)')
    con.commit()
    con.close()
    report = arc.archive(doc, str(out), what=["ortofoto"])
    assert report["rasters"][0]["table"] == "" and "non e' un raster" in report["warnings"][0]
    con = sqlite3.connect(out)
    assert con.execute('SELECT COUNT(*) FROM "PPG_ORTOFOTO_SCAVO_CHUNK_1"').fetchone()[0] == 0
    con.close()
    rows = [v for _g, v in gpkg.read_table(out, "ppg_raster")]
    assert sorted(v["tipo"] for v in rows) == ["ortofoto"]  # the DEM row of this project replaced as asked


def test_a_file_that_is_not_a_geopackage_is_left_alone(tmp_path):
    doc = _project(tmp_path)
    other = tmp_path / "appunti.gpkg"
    other.write_text("not a database")
    with pytest.raises(arc.ArchiveError, match="non e' un GeoPackage"):
        arc.archive(doc, str(other))
    assert other.read_text() == "not a database"
    notes = tmp_path / "appunti.sqlite"
    notes.write_text("not a database either")
    from pyarchinit_photogrammetry.pyarchinit import database

    if database.spatialite_path():
        with pytest.raises(arc.ArchiveError, match="non e' un database SQLite"):
            arc.archive(doc, f"spatialite:{notes}")
        assert notes.read_text() == "not a database either"


def test_spatialite(tmp_path):
    from pyarchinit_photogrammetry.pyarchinit import database

    if not database.spatialite_path():
        pytest.skip("mod_spatialite not found")
    doc = _project(tmp_path)
    out = tmp_path / "archivio.sqlite"
    report = arc.archive(doc, f"spatialite:{out}", interval=0.5)
    assert report["tables"]["ppg_quote"] == 2 and report["tables"]["ppg_us"] == 1
    assert all(r["table"] == "" for r in report["rasters"])  # rasters only as references
    assert any("riferimento" in w for w in report["warnings"])
    target = arc.SpatialiteTarget(out)
    try:
        con = target.con
        assert con.execute("SELECT COUNT(*) FROM ppg_quote WHERE progetto = 'scavo'").fetchone()[0] == 2
        assert con.execute("SELECT GeometryType(geom), ST_Srid(geom) FROM ppg_us").fetchone() == ("MULTIPOLYGON", 32637)
        assert con.execute("SELECT ROUND(ST_Area(geom), 3) FROM ppg_us").fetchone()[0] == pytest.approx(6.0)
    finally:
        target.close()
    arc.archive(doc, f"spatialite:{out}", what=["quote"])
    con = sqlite3.connect(out)
    assert con.execute("SELECT COUNT(*) FROM ppg_quote").fetchone()[0] == 2
    con.close()


def test_postgis_raster_tiles_encoding():
    from rasterio.transform import from_origin

    data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    wkb = arc.raster_wkb(data, from_origin(X0, Y0, RES, RES), 32637, nodata=-1.0)
    head = struct.unpack("<BHHddddddiHH", wkb[:61])
    assert head[0] == 1 and head[2] == 1 and head[3:5] == (RES, -RES) and head[5:7] == (X0, Y0)
    assert head[9:] == (32637, 4, 3)
    band, nodata = wkb[61], struct.unpack("<f", wkb[62:66])[0]
    assert band == 10 | 0x40 and nodata == -1.0  # 32BF, with a nodata value
    assert np.array_equal(np.frombuffer(wkb[66:], "<f4").reshape(3, 4), data[0])
    rgba = np.zeros((4, 2, 2), np.uint8)
    wkb = arc.raster_wkb(rgba, from_origin(X0, Y0, RES, RES), 32637)
    assert len(wkb) == 61 + 4 * (1 + 1 + 4)  # 4 bands of 8BUI: flag, nodata byte, 4 pixels


@pytest.mark.skipif(not os.environ.get("PPG_TEST_POSTGIS_URL"), reason="no PostGIS server (PPG_TEST_POSTGIS_URL)")
def test_postgis(tmp_path):
    """Into a schema of its own, dropped at the end: nothing else of the server is touched."""
    import uuid

    from sqlalchemy import create_engine, text

    url = os.environ["PPG_TEST_POSTGIS_URL"]
    schema = f"ppg_test_{uuid.uuid4().hex[:8]}"
    engine = create_engine(url.replace("postgresql://", "postgresql+psycopg2://"))
    with engine.begin() as c:
        try:
            with c.begin_nested():
                c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_raster"))
        except Exception:
            pass  # a server without the raster extension: the rasters stay references
    try:
        doc = _project(tmp_path)
        target = url + ("&" if "?" in url else "?") + f"schema={schema}"
        report = arc.archive(doc, target, interval=0.5)
        assert report["kind"] == "PostGIS" and report["tables"]["ppg_quote"] == 2 and report["tables"]["ppg_us"] == 1
        tables = {r["kind"]: r["table"] for r in report["rasters"]}
        with engine.connect() as c:
            assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.ppg_quote")).scalar() == 2
            assert c.execute(text(f"SELECT ST_SRID(geom) FROM {schema}.ppg_us")).scalar() == 32637
            assert c.execute(text(f"SELECT ST_Area(geom) FROM {schema}.ppg_us")).scalar() == pytest.approx(6.0)
            if tables.get("dem"):  # PostGIS raster available on the server
                point = f"ST_SetSRID(ST_Point({X0 + 5.0 + RES / 2}, {Y0 + 5.0 - RES / 2}), 32637)"
                v = c.execute(text(f"SELECT ST_Value(rast, {point}) FROM {schema}.{tables['dem']} "
                                   f"WHERE ST_Intersects(rast, {point})")).scalar()
                assert v == pytest.approx(100.0 + 0.02 * 100 + 0.01 * 100, abs=1e-3)
                columns = c.execute(text("SELECT srid, num_bands, scale_x FROM raster_columns "
                                         "WHERE r_table_schema = :s AND r_table_name = :t"),
                                    {"s": schema, "t": tables["ortofoto"]}).one()
                assert columns == (32637, 4, pytest.approx(RES))  # the constraints QGIS reads
        # archiving again replaces the rows and makes the raster tables again
        arc.archive(doc, target, interval=0.5)
        with engine.connect() as c:
            assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.ppg_quote")).scalar() == 2
            assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.ppg_raster")).scalar() == 2
            if tables.get("dem"):
                assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.{tables['dem']}")).scalar() == 1  # 200 px: one tile
    finally:
        with engine.begin() as c:
            c.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        engine.dispose()


def test_the_plugin_in_the_program(tmp_path, monkeypatch):
    """The plugin found by its entry point: its action, command and export format; the command
    archives a project into a GeoPackage."""
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app
    from pyarchinit_photogrammetry.plugins import Manager

    m = Manager(tmp_path / "plugins")
    m.load()
    info = m.get("gis-archive")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in m.actions if a.plugin == "gis-archive"] == ["Salva in un database GIS..."]
    assert "gis-archive" in m.commands and any(e.suffix == ".gpkg" for e in m.exporters)
    doc = _project(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(tmp_path / "plugins"))
    out = tmp_path / "riga.gpkg"
    r = CliRunner().invoke(app, ["plugin", "gis-archive", str(doc.path), str(out), "--what", "quote,us,dem",
                                 "--rasters", "reference"])
    assert r.exit_code == 0, r.output
    assert "ppg_quote: 2" in r.output and "solo riferimento" in r.output and out.exists()
