"""The terrio plugin on a small synthetic survey: a sloping ground with a wall and a tree as the
dense cloud of a chunk in UTM, a DEM and an orthomosaic. Ground classification, import of a
cloud written by terrio (new chunk and in place of the dense cloud), E57 and EPT exports, web
tiles of the orthomosaic, a reference raster kept as a version, the plugin in the program."""
from __future__ import annotations

import os

import numpy as np
import pytest

terrio = pytest.importorskip("terrio")

from ppg_terrio import clouds, environment, rasters  # noqa: E402

CRS = "EPSG:32637"
ORIGIN = np.array([300000.0, 3960000.0, 100.0])
SIZE = 40.0  # metres of ground


def _cloud(rng):
    """Internal coordinates (metres from the origin): ground, a wall 2 m high and a tree 4 m high."""
    n = 6000
    gx, gy = rng.uniform(0, SIZE, n), rng.uniform(0, SIZE, n)
    ground = np.column_stack([gx, gy, 0.02 * gx + rng.normal(0, 0.01, n)])
    wall = np.column_stack([rng.uniform(10, 10.4, 800), rng.uniform(5, 25, 800), rng.uniform(0.3, 2.0, 800) + 0.2])
    tree = np.column_stack([rng.normal(30, 0.8, 600), rng.normal(30, 0.8, 600), rng.uniform(2.5, 4.0, 600) + 0.6])
    xyz = np.vstack([ground, wall, tree])
    rgb = rng.integers(0, 255, (len(xyz), 3)).astype(np.uint8)
    kinds = np.array([0] * n + [1] * 800 + [2] * 600)
    return xyz, rgb, kinds


def _project(tmp_path):
    import pyarchinit_photogrammetry as ppg
    import rasterio
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    doc = ppg.Document()
    chunk = doc.add_chunk("rilievo")
    doc.chunk = chunk
    doc.save(tmp_path / "scavo.ppg")
    chunk.crs = CRS
    chunk.transform = np.eye(4)
    chunk.transform[:3, 3] = ORIGIN
    xyz, rgb, kinds = _cloud(rng)
    chunk.point_cloud = PointCloud(xyz, rgb)
    chunk.meta["kinds"] = kinds.tolist()
    res, n = 0.5, int(SIZE / 0.5)
    transform = from_origin(ORIGIN[0], ORIGIN[1] + SIZE, res, res)
    dem = (ORIGIN[2] + 0.02 * np.arange(n)[None, :] * res).repeat(n, 0).astype(np.float32)
    with rasterio.open(chunk.data_dir / "dem.tif", "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs=CRS, transform=transform, nodata=-9999.0) as dst:
        dst.write(dem, 1)
    chunk._set_product("dem", chunk.data_dir / "dem.tif")
    rgba = rng.integers(0, 255, (4, n, n)).astype(np.uint8)
    rgba[3] = 255
    with rasterio.open(chunk.data_dir / "orthomosaic.tif", "w", driver="GTiff", width=n, height=n, count=4, dtype="uint8",
                       crs=CRS, transform=transform, photometric="RGB") as dst:
        dst.write(rgba)
    chunk._set_product("orthomosaic", chunk.data_dir / "orthomosaic.tif")
    doc.save()
    return doc


def test_ground_classification_keeps_manual_labels(tmp_path):
    from pyarchinit_photogrammetry.ai.classify import set_class

    doc = _project(tmp_path)
    chunk = doc.chunk
    kinds = np.array(chunk.meta["kinds"])
    manual = np.zeros(len(kinds), bool)
    manual[np.nonzero(kinds == 1)[0][:50]] = True  # a bit of the wall labelled by hand as masonry
    set_class(chunk, manual, 64, manual=True)
    seen = []
    report = clouds.ground(chunk, "smrf", cell=1.0, noise=True, vegetation=True, progress=lambda f, m: seen.append(f))
    codes = chunk.point_cloud.classification
    assert seen[-1] == 1.0 and report["points"] == len(codes) and report["counts"][2] > 0
    assert (codes[kinds == 0] == 2).mean() > 0.95, "the ground is class 2"
    assert (codes[kinds == 2] != 2).mean() > 0.95 and (codes[kinds == 2] == 5).mean() > 0.9, "the tree is high vegetation"
    assert (codes[manual] == 64).all(), "hand labels are kept"
    assert chunk.meta["terrio_ground"]["method"] == "smrf" and "2" in chunk.meta["terrio_ground"]["counts"]
    for method in ("csf", "pmf"):
        report = clouds.ground(chunk, method, cell=1.0, resolution=0.5)
        assert (chunk.point_cloud.classification[kinds == 0] == 2).mean() > 0.9, method


def test_import_new_chunk_and_in_place(tmp_path):
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    world = chunk.to_world(chunk.point_cloud.xyz)
    laz = tmp_path / "rilievo.copc.laz"
    terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
                                 "classification": np.full(len(world), 2, np.uint8),
                                 "red": np.full(len(world), 200 * 257, np.uint16),
                                 "green": np.zeros(len(world), np.uint16),
                                 "blue": np.zeros(len(world), np.uint16)}, crs=CRS).write(str(laz))
    new = clouds.import_cloud(doc, laz, label="laser")
    assert new is not chunk and new.label == "laser" and new.crs == CRS
    pc = new.point_cloud
    assert len(pc) == len(world) and (pc.classification == 2).all() and (pc.rgb[:, 0] == 200).all()
    assert np.allclose(new.to_world(pc.xyz), world, atol=0.002) and new.meta["metric"]
    assert new.meta["point_cloud_source"].startswith("terrio:")
    # in place of the dense cloud of the first chunk: the previous one becomes a version
    back = clouds.import_cloud(doc, laz, chunk=chunk, max_points=1000)
    assert back is chunk and 500 <= len(chunk.point_cloud) <= 1500
    assert len(versions.versions(chunk, "point_cloud")) == 1
    assert np.allclose(chunk.to_world(chunk.point_cloud.xyz)[:, 2].mean(), world[:, 2].mean(), atol=0.5)


def test_import_reprojects_into_the_chunk_crs(tmp_path):
    from pyarchinit_photogrammetry.core.crs import transform_points

    doc = _project(tmp_path)
    chunk = doc.chunk
    world = chunk.to_world(chunk.point_cloud.xyz)[:500]
    lonlat = transform_points(world, CRS, "EPSG:4326")
    geo = tmp_path / "geo.las"
    terrio.PointCloud.from_dict({"x": lonlat[:, 0], "y": lonlat[:, 1], "z": lonlat[:, 2]}, crs="EPSG:4326").write(str(geo))
    clouds.import_cloud(doc, geo, chunk=chunk)
    assert np.allclose(chunk.to_world(chunk.point_cloud.xyz), world, atol=0.01)
    local = doc.add_chunk("locale")
    local.point_cloud = chunk.point_cloud
    with pytest.raises(ValueError):
        clouds.import_cloud(doc, geo, chunk=local)


def test_import_between_european_frames_goes_at_the_epoch_of_the_survey(tmp_path):
    """A cloud in ETRF2014 into a chunk in ETRS89 / UTM 33N (ETRF2000): through ITRF2020 at the epoch of the survey of
    the chunk, as the program converts points that carry no epoch (from 1.6.1), not by PROJ's null shift through the
    ETRS89 ensemble (8-9 cm off in Italy in 2026); with an older program the plain conversion of before."""
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.core import crs as crs_mod
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    frames = pytest.importorskip("pyarchinit_photogrammetry.gnss.frames")
    if not hasattr(frames, "Legacy"):
        pytest.skip("the program has no reference frames and epochs (before 1.6.1)")
    doc = ppg.Document()
    chunk = doc.add_chunk("Roma")
    chunk.crs = "EPSG:25833"
    chunk.transform = np.eye(4)
    chunk.transform[:3, 3] = [291000.0, 4640000.0, 50.0]
    rng = np.random.default_rng(1)
    chunk.point_cloud = PointCloud(rng.uniform(0, 30, (300, 3)), rng.integers(0, 255, (300, 3)).astype(np.uint8))
    world = chunk.to_world(chunk.point_cloud.xyz)
    epoch = frames.survey_epoch(chunk)
    etrf2014 = crs_mod.transform_points(world, chunk.crs, "EPSG:8403", epoch=epoch)
    cloud = tmp_path / "etrf2014.las"
    terrio.PointCloud.from_dict({"x": etrf2014[:, 0], "y": etrf2014[:, 1], "z": etrf2014[:, 2]},
                                crs="EPSG:8403").write(str(cloud))
    clouds.import_cloud(doc, cloud, chunk=chunk)
    stored = terrio.read_points(str(cloud)).to_dict()  # degrees kept to 1e-7 in the file: up to 6 mm
    stored = np.column_stack([stored["x"], stored["y"], stored["z"]])
    expected = crs_mod.transform_points(stored, "EPSG:8403", chunk.crs, epoch=epoch)
    assert np.abs(chunk.to_world(chunk.point_cloud.xyz) - expected).max() < 0.001
    null = crs_mod.by_proj(stored, "EPSG:8403", chunk.crs)
    assert np.linalg.norm((null - expected)[:, :2], axis=1).min() > 0.03  # what the plain conversion would have given


def test_export_e57_and_ept(tmp_path):
    doc = _project(tmp_path)
    chunk = doc.chunk
    e57 = clouds.export_cloud(chunk, tmp_path / "out" / "nuvola.e57")
    back = terrio.read_points(str(e57)).to_dict()
    assert len(back["x"]) == len(chunk.point_cloud)
    assert np.allclose(back["x"].min(), ORIGIN[0], atol=0.01) and back["red"].max() > 255
    ept = clouds.export_cloud(chunk, tmp_path / "nuvola.ept")
    assert (ept / "ept.json").exists() and ept.is_dir()
    assert terrio.open_points(str(ept / "ept.json")).point_count == len(chunk.point_cloud)


def test_orthomosaic_tiles(tmp_path):
    doc = _project(tmp_path)
    for suffix in (".pmtiles", ".mbtiles"):
        out = rasters.export_tiles(doc.chunk, tmp_path / f"orto{suffix}")
        tiles = terrio.open_raster(str(out))
        assert out.stat().st_size > 0 and tiles.width > 0 and "3857" in str(tiles.crs)
    with pytest.raises(ValueError):
        rasters.export_tiles(doc.chunk, tmp_path / "orto.tif")


def test_reference_raster_as_a_version(tmp_path):
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    rng = np.random.default_rng(1)
    big = tmp_path / "regione.tif"  # a wider DEM at 2 m, as a COG on the web would be
    n = 200
    terrio.Raster.from_array((50 + rng.random((n, n))).astype(np.float32),
                             (ORIGIN[0] - 150, 2.0, 0.0, ORIGIN[1] + 250, 0.0, -2.0), crs=CRS, nodata=-9999.0).write(
        str(big), driver="COG")
    report = rasters.reference_raster(chunk, big, product="dem", margin=10.0, resolution=1.0)
    assert report["path"].exists() and report["width"] == report["height"] == 60
    assert report["bounds"] == (ORIGIN[0] - 10, ORIGIN[1] - 10, ORIGIN[0] + SIZE + 10, ORIGIN[1] + SIZE + 10)
    kept = versions.versions(chunk, "dem")
    assert len(kept) == 1 and kept[0]["name"] == "Riferimento: regione.tif" and kept[0]["params"]["source"] == str(big)
    assert versions.version_file(chunk, "dem", kept[0]["id"]) == report["path"]
    assert chunk.dem_path.name == "dem.tif" and chunk.dem_path.parent == chunk.data_dir, "the active DEM is untouched"
    local = doc.add_chunk("locale")
    with pytest.raises(RuntimeError):
        rasters.reference_raster(local, big)


def test_choose_wheel():
    files = [{"filename": f"terrio-0.3.2-cp310-abi3-{p}.whl", "url": p} for p in
             ("macosx_10_12_x86_64", "macosx_11_0_arm64", "manylinux_2_17_x86_64.manylinux2014_x86_64",
              "manylinux_2_28_aarch64", "win_amd64", "win_arm64")] + [{"filename": "terrio-0.3.2.tar.gz"}]
    assert environment.choose_wheel(files, ["macosx_14_0_arm64", "macosx_11_0_arm64"])["url"] == "macosx_11_0_arm64"
    assert environment.choose_wheel(files, ["manylinux2014_x86_64"])["url"].startswith("manylinux_2_17")
    assert environment.choose_wheel(files, ["win_arm64"])["url"] == "win_arm64"
    assert environment.choose_wheel(files, ["win_arm64"], python=(3, 9)) is None, "abi3 from 3.10 on"
    assert environment.choose_wheel(files, ["linux_riscv64"]) is None
    assert environment.platform_tags() and environment.choose_wheel(files) is not None


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("terrio")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "terrio"] == [
        "Importa nuvola di punti...", "Classifica il terreno (SMRF, CSF, PMF)...", "Pulisci la nuvola (LOF, DBSCAN)...",
        "Colori e valori da un raster...", "Valori dei punti nella vista 3D...", "Quote ellissoidiche e sul geoide...",
        "Raster di riferimento dal web...", "Installa terrio..."]
    assert "terrio" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "terrio"] == [".e57", ".ept", ".pmtiles", ".mbtiles"]
    assert environment.lib_dir() == tmp_path / "plugins" / "_data" / "terrio" / "lib"
    assert environment.available() == terrio.__version__


def test_command(tmp_path, capsys):
    from ppg_terrio import TerrioPlugin

    doc = _project(tmp_path)
    plugin = TerrioPlugin()
    assert plugin.command("version") == 0 and terrio.__version__ in capsys.readouterr().out
    assert plugin.command("ground", str(doc.path), "--method", "csf", "--resolution", "0.5") == 0
    out = capsys.readouterr().out
    assert "Filtro CSF" in out and "2 Terreno: " in out, "class names in the language of the program"
    assert plugin.command("ground", str(doc.path), "--method", "pmf") == 0
    capsys.readouterr()
    import pyarchinit_photogrammetry as ppg

    params = ppg.Document.open(doc.path).chunk.meta["terrio_ground"]
    assert (params["method"], params["slope"], params["threshold"]) == ("pmf", 1.0, 0.15), "PMF's own defaults"
    assert plugin.command("export", str(doc.path), str(tmp_path / "n.e57")) == 0 and (tmp_path / "n.e57").exists()
    assert plugin.command("export", str(doc.path), str(tmp_path / "o.pmtiles")) == 0 and (tmp_path / "o.pmtiles").exists()
    assert plugin.command("import", str(doc.path), str(tmp_path / "n.e57"), "--label", "da E57", "--crs", CRS) == 0
    again = ppg.Document.open(doc.path)
    assert again.chunk.label == "rilievo", "the active chunk stays"
    new = next(c for c in again.chunks if c.label == "da E57")
    assert new.crs == CRS and len(new.point_cloud) == len(doc.chunk.point_cloud)
    assert plugin.command("import", str(doc.path), str(tmp_path / "n.e57"), "--label", "attivo", "--crs", CRS,
                          "--activate") == 0
    assert ppg.Document.open(doc.path).chunk.label == "attivo"
    assert plugin.command("ground", str(doc.path), "--chunk", "nessuno") == 2
    assert plugin.command("export", str(doc.path), str(tmp_path / "o.tif"), "--chunk", "rilievo") == 1
    assert "error" in capsys.readouterr().out
    assert os.environ.get("PPG_LANG") == "it"


def test_version_gate():
    """terrio from 0.3.2 up to the 0.5 releases is accepted, older and newer ones refused; the
    steps of 0.2.0 need 0.5."""
    inside = lambda v: environment.OLDEST <= environment._key(v) < environment.NEWEST  # noqa: E731
    assert inside("0.3.2") and inside("0.4.0") and inside("0.5.0") and inside("0.5.9")
    assert not inside("0.3.1") and not inside("0.6.0") and not inside("1.0.0")
    assert environment._key("0.4.2") < environment.FEATURES_SINCE <= environment._key("0.5.0")
    assert environment._key(environment.TERRIO_VERSION) >= environment.OLDEST


def test_import_from_a_zip(tmp_path):
    """A zipped LAZ (terrio 0.4 opens zipped point clouds) comes in like the file itself."""
    import zipfile

    doc = _project(tmp_path)
    chunk = doc.chunk
    world = chunk.to_world(chunk.point_cloud.xyz)
    laz = tmp_path / "rilievo.laz"
    terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2]}, crs=CRS).write(str(laz))
    zipped = tmp_path / "rilievo.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.write(laz, laz.name)
    new = clouds.import_cloud(doc, zipped, label="zip")
    assert new.label == "zip" and len(new.point_cloud) == len(world)
    assert np.allclose(new.to_world(new.point_cloud.xyz), world, atol=0.002)


def test_ground_changes_only_unclassified_and_ground_points(tmp_path):
    """Only classes 0, 1 and 2 change (and 3, 4, 5 with the vegetation): buildings, walls,
    water, noise, the classes of an imported lidar and the hand labels stay; the noise is
    left out of the filters."""
    doc = _project(tmp_path)
    chunk = doc.chunk
    kinds = np.array(chunk.meta["kinds"])
    pc = chunk.point_cloud
    codes = np.ones(len(pc), np.uint8)
    ground = np.nonzero(kinds == 0)[0]
    codes[ground[:300]] = 0  # never classified
    codes[ground[300:400]] = 9  # water of an imported lidar
    codes[ground[400:450]] = 18  # high noise
    codes[ground[450:500]] = 7  # low noise
    wall = np.nonzero(kinds == 1)[0]
    codes[wall] = 64  # masonry, classified by the program (not by hand)
    tree = np.nonzero(kinds == 2)[0]
    codes[tree[:200]] = 4  # medium vegetation of an earlier classification
    codes[tree[200:250]] = 6  # building
    pc.classification = codes
    chunk.point_cloud = pc
    pc = chunk.point_cloud
    pc.xyz[ground[450:500], 2] -= 5.0  # the low noise lies well below the ground
    chunk.point_cloud = pc
    report = clouds.ground(chunk, "smrf", cell=1.0)
    after = chunk.point_cloud.classification
    fixed = ~np.isin(codes, (0, 1, 2))
    assert (after[fixed] == codes[fixed]).all(), "other classes stay"
    assert report["kept"] == int(fixed.sum())
    assert (after[ground[:300]] == 2).mean() > 0.95, "never classified points become ground"
    assert (after[tree[250:]] == 1).all(), "the rest is class 1"
    assert (after[ground[500:]] == 2).mean() > 0.95, "the low noise is no ground: it was left out"
    # with the vegetation: 3, 4 and 5 may change, the others still stay
    report = clouds.ground(chunk, "smrf", cell=1.0, vegetation=True)
    after = chunk.point_cloud.classification
    other = ~np.isin(codes, (0, 1, 2, 3, 4, 5))
    assert (after[other] == codes[other]).all()
    assert (after[tree[:200]] == 5).mean() > 0.9, "the earlier medium vegetation is classified again"
    assert (after[wall] == 64).all(), "a wall classified as masonry does not become vegetation"
    assert report["kept"] == int(other.sum())


def test_the_imported_chunk_becomes_the_active_one(tmp_path, monkeypatch):
    """A cloud imported into a new chunk: the new chunk is the active one (``context.activate_chunk``
    with the program 1.3.4, the window's own method before)."""
    from ppg_terrio import TerrioPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    doc = _project(tmp_path)
    first = doc.chunk
    world = first.to_world(first.point_cloud.xyz)
    laz = tmp_path / "laser.copc.laz"
    terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2]}, crs=CRS).write(str(laz))
    activated = []

    class Window:
        def __init__(self):
            self.doc = doc

        def run_task(self, title, job, done=None, refresh=True, save=True):
            result = job(lambda *a: None)
            if done is not None:
                done(result)

        def _activate_chunk(self, chunk):  # what the window does when the user picks a chunk
            activated.append(chunk)
            doc.chunk = chunk

    monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
        lambda parent, title, fields, intro="": {f.name: f.default for f in fields} | {"file": str(laz),
                                                                                        "label": "laser"}))
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: None)
    plugin = TerrioPlugin()
    plugin.activate(Context("terrio", tmp_path / "_data", window=Window()))
    previous, app.document = app.document, doc
    try:
        plugin.import_dialog()
    finally:
        app.document = previous
    new = doc.chunks[-1]
    assert new is not first and new.label == "laser"
    assert doc.chunk is new and activated == [new]


def test_ground_dialog_has_the_defaults_of_each_filter(tmp_path, monkeypatch):
    """The window gives PMF its own defaults (slope 1.0, initial distance 0.15) and the
    vegetation option says what it does to walls, sections and spoil heaps."""
    from ppg_terrio import TerrioPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    doc = _project(tmp_path)
    seen = {}

    def ask(parent, title, fields, intro=""):
        seen.update({f.name: f for f in fields}, intro=intro)
        return {f.name: f.default for f in fields} | {"method": "pmf"}

    class Window:
        def run_task(self, title, job, done=None, refresh=True, save=True):
            result = job(lambda *a: None)
            if done is not None:
                done(result)

    monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(ask))
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: None)
    plugin = TerrioPlugin()
    plugin.activate(Context("terrio", tmp_path / "_data", window=Window()))
    previous, app.document = app.document, doc
    try:
        plugin.ground_dialog()
    finally:
        app.document = previous
    assert seen["slope"].default == 0.15 and seen["threshold"].default == 0.5
    assert seen["pmf_slope"].default == 1.0 and seen["pmf_distance"].default == 0.15
    assert "muri" in seen["vegetation"].tip and "sezioni" in seen["vegetation"].tip
    assert "0, 1 e 2" in seen["intro"]
    params = doc.chunk.meta["terrio_ground"]
    assert (params["method"], params["slope"], params["threshold"]) == ("pmf", 1.0, 0.15)


def test_texts_in_the_language_of_the_program(tmp_path):
    """Class names and the name of a reference version follow the language of the program."""
    from ppg_terrio import summary_ground
    from pyarchinit_photogrammetry.gui import i18n

    doc = _project(tmp_path)
    report = {"method": "smrf", "points": 1234, "changed": 1200, "kept": 34, "counts": {2: 1000, 64: 34}, "labels": {}}
    assert "2 Terreno: 1.000" in summary_ground(report) and "1.234 punti" in summary_ground(report)
    assert "64 Struttura muraria: 34" in summary_ground(report)
    big = tmp_path / "regionale.tif"
    terrio.Raster.from_array(np.full((50, 50), 50.0, np.float32), (ORIGIN[0] - 10, 2.0, 0.0, ORIGIN[1] + 60, 0.0, -2.0),
                             crs=CRS, nodata=-9999.0).write(str(big), driver="COG")
    try:
        i18n.set_current("en")
        text = summary_ground(report)
        assert "2 Terrain: 1,000" in text and "64 Masonry structure: 34" in text and "Filter SMRF" in text
        report = rasters.reference_raster(doc.chunk, big, product="dem", margin=0.0)
        assert report["name"] == "Reference: regionale.tif"
    finally:
        i18n.set_current(None)
    assert os.environ.get("PPG_LANG") == "it"
