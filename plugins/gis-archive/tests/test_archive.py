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

    tmp_path.mkdir(parents=True, exist_ok=True)
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
    assert sorted(v["tipo"] for v in rows) == ["dem", "ortofoto"], "only the row of the orthomosaic is replaced"
    dem_row = next(v for v in rows if v["tipo"] == "dem")
    assert dem_row["tabella"] == "PPG_DEM_SCAVO_CHUNK_1" and dem_row["risoluzione"] == pytest.approx(RES * 4)
    assert (dem_row["larghezza"], dem_row["altezza"]) == (50, 50), "the size of the reduced copy"


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
    engine = create_engine(arc.engine_url(url)[0])  # the driver the program uses (psycopg2 or psycopg 3)
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
        # the orthomosaic alone: the row of the DEM stays; another project with the same file name
        # gets rows and raster tables of its own
        arc.archive(doc, target, what=["ortofoto"], max_side=100)
        other = _project(tmp_path / "altro")
        again = arc.archive(other, target, what=["quote", "dem"])
        with engine.connect() as c:
            assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.ppg_raster WHERE tipo = 'dem'")).scalar() == 2
            assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.ppg_quote")).scalar() == 4
            if tables.get("ortofoto"):
                res = c.execute(text(f"SELECT risoluzione FROM {schema}.ppg_raster WHERE tipo = 'ortofoto'")).scalar()
                assert res == pytest.approx(RES * 2)
            if tables.get("dem"):
                assert again["rasters"][0]["table"] not in (tables["dem"], "")
                assert c.execute(text(f"SELECT COUNT(*) FROM {schema}.{tables['dem']}")).scalar() == 1
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


def test_postgresql_driver_of_the_program(monkeypatch):
    """PostGIS takes the driver the program uses: psycopg 3 where psycopg2 is missing (the
    Windows ARM64 application); a driver named in the address is kept."""
    from pyarchinit_photogrammetry.pyarchinit import database

    monkeypatch.setattr(database, "postgres_driver", lambda: "psycopg")
    monkeypatch.delenv("PGPASSWORD", raising=False)
    url, schema = arc.engine_url("postgresql://enzo@localhost:5432/scavi?schema=tell", "segreta")
    assert url.drivername == "postgresql+psycopg" and schema == "tell" and url.password == "segreta"
    assert "schema" not in url.query and url.database == "scavi"
    url, schema = arc.engine_url("postgres://enzo@localhost/scavi")
    assert url.drivername == "postgresql+psycopg" and schema == "public" and url.password is None
    monkeypatch.setattr(database, "postgres_driver", lambda: "psycopg2")
    monkeypatch.setenv("PGPASSWORD", "dall-ambiente")
    url, _ = arc.engine_url("postgresql://enzo@localhost/scavi")
    assert url.drivername == "postgresql+psycopg2" and url.password == "dall-ambiente"
    url, _ = arc.engine_url("postgresql+psycopg://enzo@localhost/scavi")
    assert url.drivername == "postgresql+psycopg", "a driver named in the address is kept"


def test_two_projects_with_the_same_file_name(tmp_path):
    """Two projects called scavo.ppg in two folders never replace each other's rows or raster
    copies; each one archived again replaces only its own; a copy made with Save as is another
    project; a project that lost its id finds its rows again by its path."""
    from pyarchinit_photogrammetry.io import gpkg

    out = tmp_path / "scavi.gpkg"
    a = _project(tmp_path / "trincea_a")
    b = _project(tmp_path / "trincea_b")
    ra = arc.archive(a, str(out), what=["quote", "dem"])
    rb = arc.archive(b, str(out), what=["quote", "dem"])
    ida, idb = a.meta["gis_archive"]["id"], b.meta["gis_archive"]["id"]
    assert ida != idb and ra["project_changed"] and a.meta["gis_archive"]["file"] == str(a.path)
    rows = [v for _g, v in gpkg.read_table(out, "ppg_quote")]
    assert len(rows) == 4 and {v["progetto"] for v in rows} == {"scavo"}
    assert sorted(v["progetto_id"] for v in rows) == sorted([ida, ida, idb, idb])
    assert {v["chunk_id"] for v in rows} == {0} and {v["progetto_file"] for v in rows} == {str(a.path), str(b.path)}
    ta, tb = ra["rasters"][0]["table"], rb["rasters"][0]["table"]
    assert ta == "PPG_DEM_SCAVO_CHUNK_1" and tb == f"PPG_DEM_SCAVO_CHUNK_1_{idb[:8].upper()}"
    # each archived again: its own rows and its own copy, nothing of the other
    a.chunk.meta["drawings"]["quote"] = a.chunk.meta["drawings"]["quote"][:1]
    again = arc.archive(a, str(out), what=["quote", "dem"])
    assert not again["project_changed"] and again["rasters"][0]["table"] == ta
    assert arc.archive(b, str(out), what=["dem"])["rasters"][0]["table"] == tb
    rows = [v for _g, v in gpkg.read_table(out, "ppg_quote")]
    assert sorted(v["progetto_id"] for v in rows) == sorted([ida, idb, idb])
    footprints = [v for _g, v in gpkg.read_table(out, "ppg_raster")]
    assert sorted(v["tabella"] for v in footprints) == sorted([ta, tb])
    with rasterio.open(str(out), table=tb) as src:
        assert src.width == SIZE
    # Save as: the copy is another project, the original keeps its rows
    a.save()
    a.save(tmp_path / "copia" / "scavo.ppg")
    copy = arc.archive(a, str(out), what=["quote"])
    idc = a.meta["gis_archive"]["id"]
    assert idc not in (ida, idb) and copy["project_changed"]
    assert sorted(v["progetto_id"] for _g, v in gpkg.read_table(out, "ppg_quote")) == sorted([ida, idb, idb, idc])
    # an id never saved with its project: the rows written from the same file are found again
    import pyarchinit_photogrammetry as ppg

    b2 = ppg.Document.open(b.path)
    assert "gis_archive" not in b2.meta  # b was never saved after its archive
    arc.archive(b2, str(out), what=["quote"])
    assert b2.meta["gis_archive"]["id"] == idb
    assert sorted(v["progetto_id"] for _g, v in gpkg.read_table(out, "ppg_quote")) == sorted([ida, idb, idb, idc])


def test_rows_of_version_0_1_0(tmp_path):
    """Rows written by version 0.1.0 (no progetto_id): those of this project are replaced at its
    next archive; those of another project with the same file name - their rasters lie in its
    folder - stay, and the report says so."""
    from shapely.geometry import Point, box

    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path / "nostro")
    out = tmp_path / "vecchio.gpkg"
    old = {"progetto": "TEXT", "chunk": "TEXT", "sito": "TEXT", "archiviato": "TEXT"}
    quote = {**old, "quota": "REAL"}
    rows = [(Point(X0 + 1, Y0 + 1), {"progetto": "scavo", "chunk": "Chunk 1", "quota": 1.0}),
            (Point(X0 + 2, Y0 + 2), {"progetto": "scavo", "chunk": "Chunk 1", "quota": 2.0}),
            (Point(X0 + 3, Y0 + 3), {"progetto": "altro", "chunk": "Chunk 1", "quota": 3.0})]
    gpkg.write_table(out, "ppg_quote", "POINT", quote, rows, CRS)
    raster = {**old, "tipo": "TEXT", "file": "TEXT", "tabella": "TEXT"}
    footprint = box(X0, Y0, X0 + 10, Y0 + 10)
    gpkg.write_table(out, "ppg_raster", "POLYGON", raster, [
        (footprint, {"progetto": "scavo", "chunk": "Chunk 1", "tipo": "dem", "file": str(doc.chunk.dem_path),
                     "tabella": ""})], CRS)
    report = arc.archive(doc, str(out), what=["quote", "dem"], rasters="reference")
    values = [v for _g, v in gpkg.read_table(out, "ppg_quote")]
    assert sorted(v["progetto"] for v in values) == ["altro", "scavo", "scavo"]
    assert all(v["progetto_id"] == doc.meta["gis_archive"]["id"] for v in values if v["progetto"] == "scavo")
    assert len(gpkg.read_table(out, "ppg_raster")) == 1 and not any("0.1.0" in w for w in report["warnings"])
    # rows of 0.1.0 of another scavo.ppg, whose DEM lies in its own folder: kept
    out2 = tmp_path / "di_un_altro.gpkg"
    gpkg.write_table(out2, "ppg_quote", "POINT", quote, rows[:2], CRS)
    gpkg.write_table(out2, "ppg_raster", "POLYGON", raster, [
        (footprint, {"progetto": "scavo", "chunk": "Chunk 1", "tipo": "dem",
                     "file": str(tmp_path / "altrove" / "scavo.ppg.files" / "chunk_0" / "dem.tif"), "tabella": ""})], CRS)
    report = arc.archive(doc, str(out2), what=["quote", "dem"], rasters="reference")
    assert len(gpkg.read_table(out2, "ppg_quote")) == 4 and len(gpkg.read_table(out2, "ppg_raster")) == 2
    assert any("0.1.0" in w and "altrove" in w for w in report["warnings"])


def test_sections_once_per_project(tmp_path):
    """The sections belong to the project: archiving its chunks one after the other writes them
    once (no chunk), and the duplicates of version 0.1.0 go."""
    from shapely.geometry import LineString

    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    second = doc.add_chunk("Saggio 2")
    out = tmp_path / "sezioni.gpkg"
    old = {"progetto": "TEXT", "chunk": "TEXT", "sito": "TEXT", "archiviato": "TEXT", "nome": "TEXT"}
    gpkg.write_table(out, "ppg_sezioni", "LINESTRING", old, [
        (LineString([(X0, Y0), (X0 + 1, Y0)]), {"progetto": "scavo", "chunk": c, "nome": "A-A'"})
        for c in ("Chunk 1", "Saggio 2")], CRS)
    arc.archive(doc, str(out), what=["sezioni"], chunk=doc.chunk)
    arc.archive(doc, str(out), what=["sezioni"], chunk=second)
    rows = gpkg.read_table(out, "ppg_sezioni")
    assert len(rows) == 1 and rows[0][1]["nome"] == "A-A'" and rows[0][1]["chunk"] is None
    assert rows[0][1]["chunk_id"] is None and rows[0][1]["progetto_id"] == doc.meta["gis_archive"]["id"]


def test_stones_on_wall_elevations_stay_out(tmp_path):
    """Stones drawn on the elevation of a wall are in the frame of the elevation, not of the map:
    they are not written into the map table, and the summary says so."""
    from shapely.geometry import box

    from ppg_gis_archive import summary
    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    ch = doc.chunk
    ch.meta["elevations"] = [{"id": "ab12cd34", "name": "Prospetto", "orthophoto": "elevations/ab12cd34/elevation.tif",
                              "relief": "elevations/ab12cd34/relief.tif"}]
    ch.meta["stones"] = {"orthomosaic.tif": [box(X0 + 1, Y0 + 1, X0 + 1.2, Y0 + 1.3).wkt],
                         "elevation.tif": [box(0.5, 0.2, 0.8, 0.5).wkt, box(1.5, 0.2, 1.7, 0.4).wkt]}
    out = tmp_path / "pietre.gpkg"
    report = arc.archive(doc, str(out), what=["pietre"])
    rows = gpkg.read_table(out, "ppg_pietre")
    assert len(rows) == 1 and rows[0][1]["gruppo"] == "orthomosaic.tif" and report["tables"]["ppg_pietre"] == 1
    note = next(w for w in report["warnings"] if "prospetto" in w)
    assert "2 pietre" in note and "elevation.tif" in note and "2 pietre" in summary(report)


def test_the_archive_id_from_the_command_line(tmp_path, capsys):
    """The command line writes the archive id into the project file (nothing else changes), so
    archiving again replaces the rows instead of adding them."""
    import json

    from ppg_gis_archive import GisArchivePlugin
    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    before = json.loads(doc.path.read_text())
    out = tmp_path / "riga.gpkg"
    plugin = GisArchivePlugin()
    for _ in range(2):
        assert plugin.command(str(doc.path), str(out), "--what", "quote,dem", "--rasters", "reference") == 0
    after = json.loads(doc.path.read_text())
    assert after["chunks"] == before["chunks"] and after["meta"]["sito"] == before["meta"]["sito"]
    assert after["meta"]["gis_archive"]["file"] == str(doc.path)
    rows = gpkg.read_table(out, "ppg_quote")
    assert len(rows) == 2 and {v["progetto_id"] for _g, v in rows} == {after["meta"]["gis_archive"]["id"]}
    assert len(gpkg.read_table(out, "ppg_raster")) == 1
    assert "ppg_quote: 2" in capsys.readouterr().out


def test_the_exporter_runs_in_the_background(tmp_path):
    """File > Esporta runs the exporter in a worker thread: it must not touch the window (the
    window stored its drawings before)."""
    import threading

    from ppg_gis_archive import GisArchivePlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    doc = _project(tmp_path)
    plugin = GisArchivePlugin()
    plugin.activate(Context("gis-archive", tmp_path / "_data", window=object()))
    previous, app.document = app.document, doc
    errors = []

    def run():
        try:
            plugin.export_gpkg(doc.chunk, tmp_path / "esportato.gpkg", lambda *a: None)
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    try:
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(120)
    finally:
        app.document = previous
    assert not errors, errors
    assert (tmp_path / "esportato.gpkg").exists() and doc.meta["gis_archive"]["id"]
