"""The plugin with the fake engine of its worker (``PPG_SAMGEO_FAKE``: the bright blobs of an image are the
objects, numpy and SciPy in place of PyTorch), on a synthetic trench (``samples.py``): the texts given to the
models, the tiles of a run, the rasters read and stretched, the stones found and the one cut by the seam of
two tiles joined, the files and the record of a run, the failures of the worker and cancelling, the classes
of the dense cloud, the environment installed with uv, the pane and the layer in the window, the units, the
command line."""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import samples
from ppg_samgeo import client, environment, runs, segment
from ppg_samgeo.texts import model_text, slug


def _program_has(name: str) -> bool:
    from pyarchinit_photogrammetry.plugins.api import Context

    return hasattr(Context, name)


@pytest.fixture
def fake(monkeypatch):
    """The worker runs with this Python, its fake engine answering (as if PyTorch were there)."""
    monkeypatch.setenv("PPG_SAMGEO_FAKE", "1")
    monkeypatch.setattr(environment, "worker_python", lambda feature="samgeo": Path(sys.executable))
    monkeypatch.setattr(environment, "where", lambda feature="samgeo": "local")
    return monkeypatch


def _stone_at(objects, x, y):
    """The objects whose centroid is within 5 cm of the stone at (x, y) of the samples."""
    wx, wy = samples.world(x, y)
    return [o for o in objects if math.hypot(o["geometry"].centroid.x - wx, o["geometry"].centroid.y - wy) < 0.05]


# ------------------------------------------------------------------ the plugin and its texts
def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("samgeo")
    assert info is not None and info.active, info and info.error
    titles = [a.title for a in manager.actions if a.plugin == "samgeo"]
    assert titles == ["Pannello Samgeo", "Installa i modelli (PyTorch, Grounding DINO, SAM 2.1)..."]
    assert "samgeo" in manager.commands
    manager.unload()


def test_what_the_models_read():
    assert model_text("pietre") == "stone."
    assert model_text("  Pietre ,  mattoni ") == "stone. brick."
    assert model_text("tegole e coppi") == "roof tile."
    assert model_text("roof tiles") == "roof tile."  # the English entry of the pane
    assert model_text("attrezzi e secchi") == "bucket. shovel. wheelbarrow."
    assert model_text("Amphora handle; stone") == "amphora handle. stone."  # any other text as written
    assert model_text("stone. stone") == "stone."
    assert model_text("") == ""
    assert slug("tegole e coppi") == "tegole_e_coppi" and slug("Città 2") == "citta_2" and slug("??") == "samgeo"


# ------------------------------------------------------------------ the plan and the rasters
def test_the_tiles_cover_the_area_and_overlap():
    from shapely.geometry import box
    from shapely.ops import unary_union

    area = (0.0, 0.0, 8.0, 6.0)
    tiles = segment.tiles_of(area, 0.01, 256, 64)
    assert len(tiles) == 12  # 4 x 3
    assert all(math.isclose(t[2] - t[0], 2.56) and math.isclose(t[3] - t[1], 2.56) for t in tiles)
    assert unary_union([box(*t) for t in tiles]).equals(box(*area))
    assert math.isclose(tiles[0][2] - tiles[1][0], 0.64)  # 64 px in common
    assert segment.tiles_of((0.0, 0.0, 1.0, 1.0), 0.01, 256, 64) == [(0.0, 0.0, 1.0, 1.0)]  # smaller than a tile


def test_the_plan_of_a_run(tmp_path):
    from shapely.geometry import box

    doc, ch = samples.project(tmp_path / "p")
    p = segment.plan(ch.orthomosaic_path)
    assert p.resolution == 0.01 and p.native == pytest.approx(0.005)  # not finer than 1 cm/px by default
    assert p.size == (800, 600) and len(p.tiles) == 1 and p.crs is not None
    assert segment.plan(ch.orthomosaic_path, resolution=0.001).resolution == pytest.approx(0.005)  # never finer
    assert segment.plan(ch.orthomosaic_path, resolution=0.02).size == (400, 300)
    x0, y1 = samples.world(0, 0)
    inside = [(x0 + 1, y1 - 1), (x0 + 3, y1 - 1), (x0 + 3, y1 - 2), (x0 + 1, y1 - 2)]
    p = segment.plan(ch.orthomosaic_path, polygon=inside, tile=64, overlap=16)
    assert p.bounds == pytest.approx((x0 + 1, y1 - 2, x0 + 3, y1 - 1)) and p.polygon.area == pytest.approx(2.0)
    assert all(p.polygon.intersects(box(*t)) for t in p.tiles)
    bowtie = [(x0 + 1, y1 - 1), (x0 + 3, y1 - 3), (x0 + 3, y1 - 1), (x0 + 1, y1 - 3)]  # crossing itself: two parts
    p = segment.plan(ch.orthomosaic_path, polygon=bowtie)
    assert p.polygon.geom_type == "MultiPolygon" and p.polygon.area == pytest.approx(2.0)
    clipped = segment.plan(ch.orthomosaic_path, bounds=(x0 - 5, y1 - 2, x0 + 2, y1 + 5))
    assert clipped.bounds == pytest.approx((x0, y1 - 2, x0 + 2, y1))  # within the raster
    with pytest.raises(ValueError):
        segment.plan(ch.orthomosaic_path, bounds=(x0 - 9, y1 - 9, x0 - 1, y1 - 1))
    with pytest.raises(ValueError):
        segment.plan(ch.orthomosaic_path, polygon=[(x0, y1), (x0 + 1, y1), (x0 + 2, y1)])  # no area


def test_tiles_read_as_rgb_or_stretched_grey(tmp_path):
    doc, ch = samples.project(tmp_path / "p", with_dem=True)
    assert segment.stretch(ch.orthomosaic_path, (samples.X0, samples.Y1 - 6, samples.X0 + 8, samples.Y1)) is None
    tb = (samples.X0 + 7, samples.Y1 - 6, samples.X0 + 8, samples.Y1 - 5)  # the corner without data
    img, valid, tr = segment.read_tile(ch.orthomosaic_path, tb, 0.01)
    assert img.shape == (100, 100, 3) and img.dtype == np.uint8 and tr.a == pytest.approx(0.01)
    assert not valid[-1, -1] and valid[0, 0] and not img[-1, -1].any()
    levels = segment.stretch(ch.dem_path, (samples.X0, samples.Y1 - 6, samples.X0 + 8, samples.Y1))
    assert len(levels) == 1 and 99.8 < levels[0][0] < 99.95 and 100.3 < levels[0][1] < 100.4  # 2 and 98 %
    img, valid, _tr = segment.read_tile(ch.dem_path, (samples.X0, samples.Y1 - 2, samples.X0 + 8, samples.Y1), 0.04,
                                        levels)
    assert img.shape == (50, 200, 3) and valid.all()
    assert (img[..., 0] == img[..., 1]).all() and img[0, 0, 0] < img[0, -1, 0]  # grey, rising to the east


# ------------------------------------------------------------------ finding
def test_the_stones_are_found_and_the_one_across_the_seam_once(tmp_path, fake):
    doc, ch = samples.project(tmp_path / "p")
    plan = segment.plan(ch.orthomosaic_path, tile=256, overlap=64)
    assert len(plan.tiles) == 12
    seen = []
    with client.Segmenter(progress=lambda f, m: seen.append(f)) as seg:
        assert seg.engine == "fake" and seg.device == "fake"
        found = segment.find(seg, plan, "stone.", progress=lambda f, m: seen.append(f))
    assert len(found) == len(samples.STONES)  # the stone in the corner without data is not seen
    for x, y, r, _v in samples.STONES:
        (o,) = _stone_at(found, x, y)
        assert o["geometry"].area == pytest.approx(math.pi * r * r, rel=0.06)
        assert o["geometry"].geom_type == "Polygon" and o["label"] == "stone"
    (seam,) = _stone_at(found, 2.5, 1.2)  # cut by the border of the first tile, whole in the second
    assert seam["geometry"].bounds[2] - seam["geometry"].bounds[0] == pytest.approx(0.4, abs=0.02)
    dim = _stone_at(found, 6.5, 4.5)[0]
    assert dim["score"] < 0.9 < min(o["score"] for o in found if o is not dim)
    assert [o["score"] for o in found] == sorted((o["score"] for o in found), reverse=True)
    assert seen == sorted(seen) and seen[-1] == 1.0
    with client.Segmenter() as seg:
        assert len(segment.find(seg, plan, "stone.", threshold=0.9)) == len(samples.STONES) - 1
        small = segment.find(seg, plan, "stone.", min_area=0.05, max_area=0.15)
        assert sorted(round(o["geometry"].area, 1) for o in small) == [0.1, 0.1, 0.1, 0.1]
        # a polygon: only what lies inside it, wherever its tiles reach
        x0, y1 = samples.world(0, 0)
        p = segment.plan(ch.orthomosaic_path, polygon=[(x0 + 0.5, y1 - 0.5), (x0 + 3.2, y1 - 0.5),
                                                       (x0 + 3.2, y1 - 1.6), (x0 + 0.5, y1 - 1.6)], tile=128, overlap=32)
        inside = segment.find(seg, p, "stone.")
        assert len(inside) == 2 and _stone_at(inside, 1.0, 1.0) and _stone_at(inside, 2.5, 1.2)


def test_objects_cut_by_tiles_joined_and_duplicates_dropped():
    from shapely.geometry import box

    tiles = [(0.0, 0.0, 10.0, 10.0), (8.0, 0.0, 18.0, 10.0)]
    found = [(0, {"geometry": box(7, 4, 10, 6), "score": 0.5, "quality": 0.9, "label": "wall"}),  # cut at 10
             (1, {"geometry": box(8, 4, 12, 6), "score": 0.7, "quality": 0.8, "label": "wall"}),  # cut at 8
             (0, {"geometry": box(8.5, 1, 9.2, 2), "score": 0.6, "quality": 0.9, "label": "stone"}),
             (1, {"geometry": box(9.0, 1, 9.8, 2), "score": 0.6, "quality": 0.9, "label": "stone"}),  # another one
             (1, {"geometry": box(14, 7, 16, 9), "score": 0.9, "quality": 0.9, "label": "stone"}),
             (1, {"geometry": box(14.1, 7, 16, 9), "score": 0.4, "quality": 0.9, "label": "stone"})]  # its double
    merged = segment.merge(found, tiles)
    assert len(merged) == 4
    wall = next(o for o in merged if o["label"] == "wall")
    assert wall["geometry"].equals(box(7, 4, 12, 6)) and wall["score"] == 0.7 and wall["quality"] == 0.9
    assert sorted(o["geometry"].area for o in merged if o["label"] == "stone") == pytest.approx([0.7, 0.8, 4.0])
    assert next(o for o in merged if o["geometry"].area == pytest.approx(4.0))["score"] == 0.9
    assert segment.merge([], tiles) == []


def test_failures_of_the_worker_and_cancelling(tmp_path, fake):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    img = np.zeros((64, 64, 3), np.uint8)
    fake.setenv("PPG_SAMGEO_FAKE", "fail")
    with client.Segmenter() as seg, pytest.raises(client.WorkerError, match="the fake engine refused the image"):
        seg.segment(img, "stone.")
    fake.setenv("PPG_SAMGEO_FAKE", "crash")
    with client.Segmenter() as seg, pytest.raises(client.WorkerError, match="si è chiuso inaspettatamente.*codice 3"):
        seg.segment(img, "stone.")
    fake.setenv("PPG_SAMGEO_FAKE", "slow")  # two seconds a tile: cancelled in the middle of one
    seg = client.Segmenter()
    progress = Progress()
    threading.Timer(0.5, progress.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        seg.segment(img, "stone.", progress=progress)
    assert time.monotonic() - t0 < 1.8 and seg._proc.poll() is not None  # the worker stopped at once
    # an engine the worker does not know
    fake.delenv("PPG_SAMGEO_FAKE")
    with pytest.raises(client.WorkerError, match="Modelli: .*unknown engine"):
        client.Segmenter(engine="nothing")


def test_a_download_broken_off_is_tried_again_without_xet(fake):
    """Hugging Face's Xet storage broke off the weights (seen on a Mac): the worker is started again once with
    plain HTTP downloads, and when that fails too the message says to try again."""
    fake.setenv("PPG_SAMGEO_FAKE", "xet")
    with client.Segmenter() as seg:
        assert seg.engine == "fake"
    assert client.download_failed("CAS Client Error: Format error: I/O error: error decoding response body")
    assert "riprova" in client._hint("File reconstruction error") and not client.download_failed("unknown engine")
    fake.setattr(client, "download_failed", lambda text: False)  # no second try: the error with its hint
    with pytest.raises(client.WorkerError, match="CAS Client Error"):
        client.Segmenter()


def test_the_download_of_the_weights_in_the_progress_window(fake):
    """The first run downloads the weights: how far, in percent and megabytes, before the models answer."""
    fake.setenv("PPG_SAMGEO_FAKE", "download")
    said = []
    with client.Segmenter(progress=lambda fraction, message: said.append((round(fraction, 2), message))) as seg:
        assert seg.engine == "fake"
    assert [f for f, _m in said] == [0.0, 0.25, 0.5, 0.75]
    assert said[1][1] == "Download dei modelli: 25% (250 di 1.000 MB)"


def test_the_files_downloaded_and_reported(tmp_path, monkeypatch):
    """The worker's download: the sizes from a dry run, the bytes arriving in the cache reported while they
    come, nothing when all is in the cache already or the sizes cannot be known."""
    import types

    sys.path.insert(0, str(Path(client.WORKER).parent))
    try:
        import worker
    finally:
        sys.path.pop(0)
    blobs = tmp_path / "models--a--b" / "blobs"
    blobs.mkdir(parents=True)
    local = tmp_path / "models--a--b" / "snapshots" / "c" / "w.pt"

    class Bar:  # tqdm, as huggingface_hub's bars are made
        def __init__(self, *args, **kwargs):
            pass

        def update(self, n=1):
            pass

    def snapshot_download(repo, allow_patterns=None, dry_run=False, tqdm_class=None, **kw):
        if dry_run:
            return [types.SimpleNamespace(file_size=3000, will_download=cached is None, local_path=str(local))]
        bar = tqdm_class(unit="B", desc="Downloading bytes", total=0) if counted else None
        for k in range(3):
            if bar is not None:  # the bytes as they arrive
                bar.update(1000)
            else:  # the bytes written into the cache
                with open(blobs / "w.incomplete", "ab") as f:
                    f.write(b"x" * 1000)
            time.sleep(0.3)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "tqdm.auto", types.SimpleNamespace(tqdm=Bar))
    cached = None
    for counted in (True, False):
        said = []
        worker.download([("a/b", ["*.pt"])], lambda done, total: said.append((done, total)), every=0.1)
        assert said[0] == (0, 3000) and said[-1] == (3000, 3000) and any(0 < d < 3000 for d, _t in said)
        assert all(total == 3000 for _d, total in said) and [d for d, _t in said] == sorted(d for d, _t in said)
    said.clear()
    cached = True
    worker.download([("a/b", ["*.pt"])], lambda done, total: said.append((done, total)), every=0.1)
    assert said == []

    def offline(*a, **kw):
        raise OSError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=offline))
    worker.download([("a/b", ["*.pt"])], lambda done, total: said.append((done, total)), every=0.1)
    assert said == []


def test_a_worker_that_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "where", lambda feature="samgeo": None)
    with pytest.raises(environment.EnvironmentMissing, match="ppg plugin samgeo install"):
        client.Segmenter()
    bad = tmp_path / "worker.py"
    bad.write_text("import sys\nsys.stderr.write('No module named torch\\n')\nsys.exit(1)\n")
    with pytest.raises(client.WorkerError, match="installa di nuovo i modelli") as err:
        client.Segmenter(python=Path(sys.executable), script=bad)
    assert "No module named torch" in err.value.details


# ------------------------------------------------------------------ the runs
def _a_run(ch, text="pietre", **plan_args):
    with client.Segmenter() as seg:
        plan = segment.plan(ch.orthomosaic_path, **plan_args)
        found = segment.find(seg, plan, model_text(text))
    run = runs.write(ch, plan, found, {"text": text, "prompt": model_text(text), "engine": "open", "raster": "orthomosaic"})
    runs.add(ch, run)
    return run


def test_a_run_its_files_and_its_record(tmp_path, fake):
    import rasterio

    from pyarchinit_photogrammetry.engine import rastercalc
    from pyarchinit_photogrammetry.io import gpkg

    doc, ch = samples.project(tmp_path / "p")
    run = _a_run(ch)
    assert (run["id"], run["name"], run["count"], run["tiles"]) == ("pietre", "pietre", 6, 1)
    assert run["files"] == {"gpkg": "samgeo/pietre.gpkg", "mask": "samgeo/pietre_mask.tif"} and run["visible"]
    assert runs.records(ch) == [run] and runs.layer(run) == "samgeo: pietre"
    rows = gpkg.read_table(ch.data_dir / "samgeo" / "pietre.gpkg", "samgeo")
    assert [a["n"] for _g, a in rows] == [1, 2, 3, 4, 5, 6] and all(a["prompt"] == "pietre" for _g, a in rows)
    assert all(g.geom_type == "MultiPolygon" and a["area_m2"] == pytest.approx(g.area, abs=1e-3) for g, a in rows)
    with rasterio.open(ch.data_dir / run["files"]["mask"]) as src:
        mask = src.read(1)
        assert src.tags()["ppg_name"] == "samgeo: pietre" and src.res == (0.01, 0.01) and src.nodata == 0
        assert sorted(np.unique(mask).tolist()) == [0, 1, 2, 3, 4, 5, 6]
        r, c = src.index(*samples.world(5.0, 2.0))
        n = int(mask[r, c])
    assert next(a for _g, a in rows if a["n"] == n)["area_m2"] == pytest.approx(math.pi * 0.25 ** 2, rel=0.06)
    # the raster calculator reads the mask by itself, named by its tag
    (found,) = [r for r in rastercalc.rasters(ch) if r.name == "samgeo_pietre_mask"]
    assert found.label == "samgeo: pietre"
    # another run of the same text: a name of its own; saved with the project
    second = _a_run(ch, resolution=0.02)
    assert (second["id"], second["name"], second["resolution"]) == ("pietre_2", "pietre 2", 0.02)
    assert second["colour"] != run["colour"]
    # a long text: its id cut to a length, the number of a second run kept
    long_text = "a long description of archaeological stones"
    first_id = runs.identity(ch, long_text)[1]
    runs.add(ch, {"id": first_id, "name": long_text})
    assert runs.identity(ch, long_text) == (f"{long_text} 2", f"{first_id}_2") and len(first_id) == 40
    runs.remove(ch, first_id)
    doc.save()
    import pyarchinit_photogrammetry as ppg

    again = ppg.Document.open(doc.path).chunk
    assert [r["name"] for r in runs.records(again)] == ["pietre", "pietre 2"]
    # objects deleted: out of the GeoPackage, zero in the mask
    assert runs.remove_objects(ch, run, {n, 99}) == 1 and run["count"] == 5
    assert n not in [a["n"] for _g, a in runs.features(ch, run)]
    with rasterio.open(ch.data_dir / run["files"]["mask"]) as src:
        assert n not in np.unique(src.read(1)).tolist() and len(np.unique(src.read(1))) == 6
    assert runs.remove_objects(ch, run, {n}) == 0
    # a GeoPackage of the runs, a layer each
    out = runs.export(ch, tmp_path / "out.gpkg")
    assert [len(gpkg.read_table(out, t)) for t in ("pietre", "pietre_2")] == [5, 6]
    assert len(gpkg.read_table(runs.export(ch, tmp_path / "one.gpkg", ["pietre_2"]), "pietre_2")) == 6
    with pytest.raises(ValueError):
        runs.export(ch, tmp_path / "none.gpkg", ["nothing"])
    # removed: files and record
    assert runs.remove(ch, "pietre")["name"] == "pietre"
    assert not (ch.data_dir / "samgeo" / "pietre.gpkg").exists() and not (ch.data_dir / "samgeo" / "pietre_mask.tif").exists()
    assert [r["id"] for r in runs.records(ch)] == ["pietre_2"] and runs.remove(ch, "pietre") is None
    runs.remove(ch, "pietre_2")
    assert "samgeo" not in ch.meta


def test_classes_of_the_dense_cloud(tmp_path, fake):
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    doc, ch = samples.project(tmp_path / "p")
    run = _a_run(ch)
    with pytest.raises(RuntimeError, match="nuvola densa"):
        runs.classify(ch, run)
    xs, ys = np.meshgrid(np.arange(0.02, 8.0, 0.04), np.arange(0.02, 6.0, 0.04))
    wx, wy = samples.world(0, 0)
    xyz = np.column_stack([wx + xs.ravel(), wy - ys.ravel(), np.full(xs.size, 100.0)])
    classes = np.ones(len(xyz), np.uint8)
    manual = np.zeros(len(xyz), bool)
    centre = np.argmin(np.hypot(xyz[:, 0] - samples.world(1.0, 1.0)[0], xyz[:, 1] - samples.world(1.0, 1.0)[1]))
    classes[centre], manual[centre] = 6, True  # labelled by hand: kept
    ch.point_cloud = PointCloud(xyz, classification=classes, manual=manual)
    result = runs.classify(ch, run)
    assert result["name"] == "samgeo: pietre" and result["code"] == runs.FIRST_CLASS
    pc = ch.point_cloud
    assert ch.meta["classes"][str(result["code"])] == "samgeo: pietre" and pc.classification[centre] == 6
    expected = sum(math.pi * r * r for _x, _y, r, _v in samples.STONES) / 0.04 ** 2
    assert result["points"] == pytest.approx(expected, rel=0.1)
    assert int((pc.classification == result["code"]).sum()) == result["points"]
    # again with one stone: the same class, the points of the others back to unclassified
    (o,) = [a for g, a in runs.features(ch, run) if g.contains(_point(*samples.world(5.0, 2.0)))]
    again = runs.classify(ch, run, {o["n"]})
    assert again["code"] == result["code"] and again["points"] < result["points"] / 3
    assert int((ch.point_cloud.classification == result["code"]).sum()) == again["points"]


def _point(x, y):
    from shapely.geometry import Point

    return Point(x, y)


# ------------------------------------------------------------------ the environment
@pytest.mark.skipif(os.name == "nt", reason="the fake uv is a shell script")
def test_the_environment_installed_with_uv(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Progress

    monkeypatch.setattr(environment, "local_available", lambda feature="samgeo": False)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"python": "3.12.9", "versions": {"torch": "9.1.0", "transformers": "9.2.0",
                                                                   "sam2": "9.3.0"}, "device": "cuda"}))
    log = tmp_path / "uv.log"
    monkeypatch.setenv("ANSWER", str(answer))
    monkeypatch.setenv("UVLOG", str(log))
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n"
                  "echo \"$@\" >> \"$UVLOG\"\n"
                  "if [ \"$1\" = venv ]; then\n"
                  "  for last; do :; done\n"
                  "  mkdir -p \"$last/bin\"\n"
                  "  printf '#!/bin/sh\\ncat \"$ANSWER\"\\n' > \"$last/bin/python\"\n"
                  "  chmod +x \"$last/bin/python\"\n"
                  "  exit 0\n"
                  "fi\n"
                  "if [ \"$1\" = pip ]; then echo '  --torch-backend <BACKEND>'; echo ' + torch==9.1.0'; "
                  "[ -n \"$FAIL\" ] && exit 1; exit 0; fi\n"
                  "exit 2\n")
    uv.chmod(0o755)
    env = tmp_path / "geoai-env"
    monkeypatch.setenv(environment.ENV_VAR, str(env))
    assert environment.where() is None and "non sono installati" in environment.status_text()
    seen = []
    assert environment.install(Progress(lambda f, m: seen.append((f, m))), uv=uv) == env
    info = environment.installed_info()
    assert info["features"] == ["samgeo"] and info["device"] == "cuda" and info["versions"]["torch"] == "9.1.0"
    assert environment.where() == "environment" and environment.worker_python() == environment.env_python(env)
    assert "PyTorch 9.1.0" in environment.status_text() and "GPU NVIDIA" in environment.status_text()
    assert seen[-1] == (1.0, "Modelli pronti (PyTorch 9.1.0, GPU NVIDIA (CUDA))")
    assert [f for f, _m in seen] == sorted(f for f, _m in seen)
    install = [line for line in log.read_text().splitlines() if line.startswith("pip install --python")][-1]
    assert "--torch-backend auto" in install and all(p in install.split() for p in ("torch", "transformers", "sam2"))
    # the features of another plugin already there are kept when this one installs again
    info["features"] = ["geoai", "samgeo"]
    (env / environment.MARKER).write_text(json.dumps(info))
    monkeypatch.setitem(environment.FEATURES, "geoai", {"packages": ["geoai-py"], "modules": ["geoai"]})
    answer.write_text(json.dumps({"python": "3.12.9", "versions": {"torch": "9.1.0", "geoai": "1"}, "device": "cpu"}))
    environment.install(uv=uv)
    assert environment.installed_info()["features"] == ["geoai", "samgeo"]
    assert "geoai-py" in log.read_text().splitlines()[-1].split()
    # a failure leaves no environment that looks ready
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError, match="Installazione non riuscita"):
        environment.install(uv=uv)
    assert environment.installed_info() == {} and environment.where() is None
    # installed, but the models do not import
    monkeypatch.delenv("FAIL")
    answer.write_text(json.dumps({"python": "3.12.9", "versions": {}, "errors": {"sam2": "ImportError: nope"}}))
    with pytest.raises(RuntimeError, match="sam2: ImportError: nope"):
        environment.install(uv=uv)
    assert environment.where() is None


# ------------------------------------------------------------------ in the window
@pytest.fixture
def window(tmp_path, monkeypatch, fake):
    """The program's window on the synthetic trench, with the plugin."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_samgeo import SamgeoPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")  # the window alone: the plugin is activated below
    qapp = QApplication.instance() or QApplication([])
    doc, ch = samples.project(tmp_path / "p")
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))  # a modal error box would stop the test
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    plugin = SamgeoPlugin()
    plugin.activate(Context("samgeo", tmp_path / "plugin", window=w))
    try:
        yield w, plugin, ch, qapp, errors
    finally:
        plugin.deactivate()
        plugin.context.release()  # as the program's plugin manager does
        w.setWindowModified(False)
        w.close()


def _find(w, plugin, qapp, **params):
    values = {**plugin.pane.values(), **params}
    plugin.find(values)
    w.runner.wait()
    for _ in range(3):
        qapp.processEvents()


def test_the_pane_and_the_layer_of_a_run(window, monkeypatch, tmp_path):
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.io import gpkg

    w, plugin, ch, qapp, errors = window
    pane, v = plugin.pane, w.ortho_view
    from PySide6.QtWidgets import QDockWidget

    assert w.findChild(QDockWidget, "pane_plugin_samgeo") is not None  # a dock of its own
    assert pane.c_raster.itemText(0) == "Ortofoto" and pane.c_raster.itemData(0) == "orthomosaic"
    assert pane.c_text.currentText() == "pietre" and pane.values()["prompt"] == "stone."
    assert "PyTorch di questo Python" in pane.l_status.text() and not pane.b_install.isVisibleTo(pane)
    plugin.context.show_raster()
    w.setWindowModified(False)
    _find(w, plugin, qapp)
    assert not errors
    (run,) = runs.records(ch)
    assert run["count"] == 6 and run["raster_label"] == "Ortofoto" and run["device"] == "fake" and run["seconds"] >= 0
    layer = runs.layer(run)
    assert len(v.features(layer)) == 6 and {f["n"] for f in v.features(layer)} == set(range(1, 7))
    assert v.features(layer)[0]["prompt"] == "pietre" and w.isWindowModified()
    assert pane.table.rowCount() == 1 and pane.table.item(0, 1).text() == "6" and pane.current_run() == "pietre"
    assert "6 proposte nel layer «samgeo: pietre»" in pane.l_last.text()
    # the masks of the runs are not rasters to look in
    assert [pane.c_raster.itemData(k) for k in range(pane.c_raster.count())] == ["orthomosaic"]
    # in the workspace
    (item,) = plugin._tree_items(ch)
    assert item["label"] == "Samgeo" and item["children"][0]["label"] == "pietre (6)"
    # another raster in the view: the layer is drawn again on it; hidden, it stays hidden
    w._load_raster("ortho")
    assert len(v.features(layer)) == 6
    plugin.set_visible("pietre", False)
    assert layer not in v.layers and not run["visible"]
    w._load_raster("ortho")
    assert layer not in v.layers
    plugin.zoom("pietre")  # shown again to be zoomed to
    assert run["visible"] and len(v.features(layer)) == 6
    # a proposal deleted on the view: out of the run too
    v.select_items([v.layers[layer]["features"][0][0]])
    w.delete_selected(confirm=False)
    for _ in range(3):
        qapp.processEvents()
    assert run["count"] == 5 and len(runs.features(ch, run)) == 5 and len(v.features(layer)) == 5
    assert pane.table.item(0, 1).text() == "5"
    # the part shown, and a polygon drawn (the program lends its drawing)
    _find(w, plugin, qapp, area="view")
    assert [r["name"] for r in runs.records(ch)] == ["pietre", "pietre 2"]
    x0, y1 = samples.world(0, 0)
    monkeypatch.setattr(plugin.context, "request_polygon", lambda done, cancelled=None, hint="": done(
        np.array([(x0 + 0.5, y1 - 0.5), (x0 + 3.2, y1 - 0.5), (x0 + 3.2, y1 - 1.6), (x0 + 0.5, y1 - 1.6)])) or True)
    pane.c_text.setCurrentText("mattoni, pietre")  # written by the user: both given to the models in English
    _find(w, plugin, qapp, area="polygon")
    third = runs.records(ch)[-1]
    assert (third["name"], third["prompt"], third["count"], third["area"]) == ("mattoni, pietre", "brick. stone.", 2,
                                                                                "polygon")
    assert third["polygon"].startswith("POLYGON") and len(v.features(runs.layer(third))) == 2
    # a GeoPackage of the runs
    out = plugin.export(tmp_path / "samgeo.gpkg")
    assert len(gpkg.read_table(out, "pietre")) == 5 and len(gpkg.read_table(out, "mattoni_pietre")) == 2
    # removed: the layer and the files go
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    plugin.remove_dialog("pietre_2")
    assert [r["id"] for r in runs.records(ch)] == ["pietre", "mattoni_pietre"] and "samgeo: pietre 2" not in v.layers
    assert not (ch.data_dir / "samgeo" / "pietre_2.gpkg").exists() and pane.table.rowCount() == 2
    assert not errors


@pytest.mark.skipif(not _program_has("add_units"), reason="context.add_units comes with the program 1.5.0")
def test_proposals_become_units(window, monkeypatch):
    from pyarchinit_photogrammetry.gui import dialogs

    w, plugin, ch, qapp, errors = window
    v = w.ortho_view
    plugin.context.show_raster()
    _find(w, plugin, qapp)
    run = runs.records(ch)[0]
    layer = runs.layer(run)
    before, entries = len(w.units), w.history.index
    asked = {}

    def ask(parent, title, fields, intro=""):
        asked.update({f.name: (f.default, [c[0] for c in f.choices or []]) for f in fields})
        return {"which": "selected", "tipo": "US"}

    monkeypatch.setattr(dialogs.ParamDialog, "ask", ask)
    v.select_items([v.layers[layer]["features"][k][0] for k in (0, 1)])
    plugin.units_dialog("pietre")
    assert asked["which"] == ("selected", ["selected", "all"]) and asked["tipo"][0] == "US"
    assert len(w.units) == before + 2 and w.history.index == entries + 1
    unit = w.units[-1]
    assert unit.tipo == "US" and unit.us is None and unit.sito == "Saggio di prova"
    assert unit.attributes["samgeo_prompt"] == "pietre" and 0.5 < unit.attributes["samgeo_score"] <= 1.0
    assert run["units"] == 2 and len(run["units_made"]) == 2 and run["visible"]  # some left: the layer stays
    outlines = [f for f in v.features(layer) if f["us"]]  # those made units drawn as outlines
    assert sorted(f["n"] for f in outlines) == run["units_made"]
    asked.clear()
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda parent, title, fields, intro="": asked.update(
        {f.name: (f.default, [c[0] for c in f.choices or []]) for f in fields}) or {"which": "left", "tipo": "USM"})
    plugin.units_dialog("pietre")
    assert asked["which"] == ("left", ["left", "all"])  # the rest, never a unit twice
    assert len(w.units) == before + 6 and w.units[-1].tipo == "USM" and len(set(map(id, w.units))) == before + 6
    assert run["units"] == 6 and not run["visible"] and layer not in v.layers  # all of them: the layer goes
    w.undo()
    assert len(w.units) == before + 2 and runs.records(ch)[0]["id"] == "pietre"  # the run stays
    assert not errors


def test_the_command_line(tmp_path, fake, capsys):
    import pyarchinit_photogrammetry as ppg
    from ppg_samgeo import SamgeoPlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = samples.project(tmp_path / "p")
    plugin = SamgeoPlugin()
    plugin.activate(Context("samgeo", tmp_path / "plugin"))
    project = str(doc.path)
    assert plugin.command("status") == 0 and "PyTorch di questo Python" in capsys.readouterr().out
    args = ["segment", project, "pietre", "--resolution", "0.02", "--min-area", "0.05"]
    if _program_has("add_units"):
        args.append("--units")
    assert plugin.command(*args) == 0
    out = capsys.readouterr().out
    assert "1 tiles at 0.02 m/px" in out and "«pietre»: 5 oggetti su Ortofoto, 2 cm/px" in out
    again = ppg.Document.open(project).chunk
    (run,) = runs.records(again)
    assert run["count"] == 5 and run["resolution"] == 0.02 and run["raster"] == "orthomosaic"
    if _program_has("add_units"):
        assert run["units"] == 5 and len(again.meta["drawings"]["units"]) == 5
    assert plugin.command("list", project) == 0 and "pietre: «pietre»: 5 oggetti" in capsys.readouterr().out
    assert plugin.command("list", project, "--json") == 0 and json.loads(capsys.readouterr().out)[0]["id"] == "pietre"
    assert plugin.command("export", project, str(tmp_path / "out.gpkg"), "--run", "pietre") == 0
    assert (tmp_path / "out.gpkg").exists()
    assert plugin.command("export", project, str(tmp_path / "x.gpkg"), "--run", "nothing") == 1
    x0, y1 = samples.world(0, 0)
    assert plugin.command("segment", project, "stone", "--area", f"{x0},{y1 - 2},{x0 + 3},{y1}", "--json") == 0
    out = capsys.readouterr().out
    second = json.loads(out[out.index("\n{") + 1:])
    assert second["name"] == "stone" and second["count"] == 2 and second["area"] == "bounds"
    assert plugin.command("segment", project, "stone", "--raster", "nothing") == 1
    assert plugin.command("segment", project, "stone", "--area", "1,2") == 1
    assert plugin.command("remove", project, "stone") == 0
    assert [r["id"] for r in runs.records(ppg.Document.open(project).chunk)] == ["pietre"]
    assert plugin.command("remove", project, "stone") == 1
    plugin.deactivate()
