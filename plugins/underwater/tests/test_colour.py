"""The colours under water: the physical restoration on images made with the image formation model
(known attenuation and backscatter), the quick correction, the ranges read from depth maps in the
program's convention, the corrected copies and the photos the chunk uses."""
from __future__ import annotations

import numpy as np
import pytest

import samples
from ppg_underwater import colour


def _survey(n: int = 5, h: int = 200, w: int = 260):
    data = []
    for i in range(n):
        j = samples.texture(h, w, seed=i)
        z = samples.ranges(h, w, near=1.0 + 0.3 * i, far=5.0 + 0.5 * i, seed=i)
        lin = samples.underwater(j, z, exposure=1.0 + 0.15 * (i - 2), seed=i)
        data.append((i, colour.to_linear(samples.srgb(lin)), z, np.ones(z.shape, bool), j))
    return data


def test_srgb_round_trip():
    a = np.arange(256, dtype=np.uint8)
    assert np.array_equal(colour.to_srgb(colour.to_linear(a)), a)


def test_the_water_is_found_and_taken_away():
    data = _survey()
    model = colour.fit_model([d[:4] for d in data])
    # the coefficients of the water (per unit of range: the ranges here are metres)
    beta_b = np.array([model.backscatter[c][1] for c in range(3)]) / model.zref
    beta_d = np.array([colour.beta_model(1.0, model.beta[c]) for c in range(3)]) / model.zref
    assert np.allclose(beta_b[:2], samples.BETA_B[:2], rtol=0.15), beta_b  # red: almost no backscatter to fit
    assert np.allclose(beta_d, samples.BETA_D, rtol=0.2, atol=0.01), beta_d
    zz = np.linspace(model.zmin, model.zmax, 40)[:, None]
    for key, lin, z, valid, j in data:
        exposure = 1.0 + 0.15 * (key - 2)
        fitted = colour.backscatter(zz, model.backscatter, model.zref, model.amplitudes[key])[:, 0, :]
        true = exposure * samples.B_INF * (1.0 - np.exp(-samples.BETA_B * zz))
        assert np.abs(fitted - true).max() < 0.02, np.abs(fitted - true).max(axis=0)
        out = colour.restore_linear(lin, z, model, model.amplitudes[key], model.exposures[key])
        ok = j.min(axis=-1) > 0.03  # not the shadows
        restored = np.log(np.maximum(out[ok], 1e-4) / j[ok])
        raw = np.log(np.maximum(lin[ok], 1e-4) / j[ok])
        # what is left of the water: a gain per channel, nearly the same at every range
        assert np.all(restored.std(axis=0)[:2] < 0.07) and restored.std(axis=0)[2] < 0.3, restored.std(axis=0)
        assert np.all(restored.std(axis=0) < 0.6 * raw.std(axis=0))
        # the red of the far pixels is no longer darker than the red of the near ones
        zk = z[ok]
        slope_raw = np.polyfit(zk, raw[:, 2], 1)[0]
        slope = np.polyfit(zk, restored[:, 2], 1)[0]
        assert slope_raw < -0.3 and abs(slope) < 0.1, (slope_raw, slope)


def test_one_model_per_photo():
    key, lin, z, valid, j = _survey(1)[0]
    model = colour.fit_model([(key, lin, z, valid)], per="photo")
    out = colour.restore_linear(lin, z, model, 1.0, model.exposures[key])
    ok = j.min(axis=-1) > 0.03
    restored = np.log(np.maximum(out[ok], 1e-4) / j[ok])
    assert restored.std(axis=0)[2] < 0.3 and model.per == "photo"
    # the model kept in the project and read back
    again = colour.Model.from_dict(model.to_dict())
    assert np.allclose(colour.restore_linear(lin, z, again, 1.0, again.exposures[key]), out)
    assert set(again.coefficients()) == {"R", "G", "B"}


def test_quick_correction_balances_a_green_blue_photo():
    j = samples.texture(120, 160, seed=3)
    z = np.full((120, 160), 5.0, np.float32)
    img = samples.srgb(samples.underwater(j, z))
    before = colour.colour_cast(img)
    assert before["r_g"] < 0.5  # red taken by the water
    params = colour.QuickParams()
    stats = colour.quick_stats([img], params)
    out = colour.quick_correct(img, params, stats)
    after = colour.colour_cast(out)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert abs(after["r_g"] - 1.0) < 0.15 and abs(after["b_g"] - 1.0) < 0.15, after
    # CLAHE and a monochrome photo
    out2 = colour.quick_correct(img, colour.QuickParams(clahe=True), stats)
    assert out2.shape == img.shape
    grey = img[..., 1]
    assert colour.quick_correct(grey, params, stats).shape == grey.shape


# ------------------------------------------------------------------ ranges from the program's depth maps
@pytest.fixture
def chunk_with_depth_map(tmp_path, monkeypatch):
    """A chunk with one photo of a pinhole camera and a depth map of the undistorted photo (a plane
    at z = 2, a hole in a corner), as the program writes them."""
    import cv2
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.engine import confidence
    from pyarchinit_photogrammetry.engine.dense import write_colmap_array

    photo = tmp_path / "foto" / "a.jpg"
    photo.parent.mkdir()
    cv2.imwrite(str(photo), np.full((80, 100, 3), 128, np.uint8))
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    (cam,) = chunk.add_photos([photo])
    sensor = chunk.sensor(cam)
    sensor.model, sensor.params = "PINHOLE", [100.0, 100.0, 50.0, 40.0]
    chunk.meta["sfm"] = {"names": {str(cam.key): "a.jpg"}}
    depth = np.full((80, 100), 2.0, np.float32)
    depth[:10, :10] = 0  # a hole
    path = tmp_path / "a.jpg.geometric.bin"
    write_colmap_array(path, depth)
    monkeypatch.setattr(confidence, "depth_maps", lambda ch: [{"name": "a.jpg", "path": path, "K": (100.0, 100.0, 50.0,
                                                                                                    40.0),
                                                               "pose": np.eye(4)[:3], "size": (100, 80)}])
    return doc, chunk, cam


def test_ranges_follow_the_rays_of_the_photo(chunk_with_depth_map):
    from ppg_underwater.ranges import RangeMaps

    doc, chunk, cam = chunk_with_depth_map
    maps = RangeMaps(chunk)
    rng, valid = maps(cam, (80, 100))
    # the centre of the first pixel at (0.5, 0.5) in COLMAP's pixels: its ray
    x, y = (12.5 - 50) / 100, (20.5 - 40) / 100
    assert rng[20, 12] == pytest.approx(2.0 * np.sqrt(1 + x * x + y * y), rel=1e-5)
    assert rng[40, 50] == pytest.approx(2.0 * np.sqrt(1 + 0.005 ** 2 + 0.005 ** 2), rel=1e-5)
    assert not valid[0, 0] and valid[40, 50] and rng[0, 0] > 2.0  # the hole filled from its neighbours
    # a grid coarser than the photo, and a chunk in metres (a levelled one): the ranges in metres
    small, _ = maps(cam, (40, 50))
    assert small.shape == (40, 50)
    from pyarchinit_photogrammetry.core import geometry

    chunk.transform = geometry.compose_similarity(0.5, np.eye(3), np.zeros(3))
    rng2, _ = RangeMaps(chunk)(cam, (80, 100))
    assert rng2[40, 50] == pytest.approx(0.5 * rng[40, 50], rel=1e-5)


def test_corrected_copies_and_the_photos_in_use(tmp_path):
    from PIL import Image

    import pyarchinit_photogrammetry as ppg
    from ppg_underwater import photos

    folder = tmp_path / "foto"
    folder.mkdir()
    for i in range(3):
        img = samples.srgb(samples.underwater(samples.texture(60, 80, seed=i), np.full((60, 80), 3.0, np.float32)))
        exif = Image.Exif()
        exif[0x0110] = "UWCam"
        Image.fromarray(img[..., ::-1]).save(folder / f"IMG_{i}.JPG", exif=exif.tobytes())
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    cams = chunk.add_photos(sorted(folder.iterdir()))
    chunk.meta["sfm"] = {"names": {str(c.key): f"IMG_{i}.JPG" for i, c in enumerate(cams)}, "image_root": str(folder)}
    info = photos.quick(chunk)
    assert info["photos"] == 3 and info["method"] == "quick"
    copy = photos.colour_dir(chunk) / "IMG_0.JPG"
    with Image.open(copy) as im:
        assert im.size == (80, 60) and im.getexif().get(0x0110) == "UWCam"  # the EXIF kept
    res = photos.use(chunk, "corrected")
    assert res == {"switched": 3, "missing": 0}
    assert all(photos.colour_dir(chunk) in __import__("pathlib").Path(c.path).parents for c in cams)
    assert chunk.meta["sfm"]["image_root"] == str(photos.colour_dir(chunk).resolve())
    assert photos.status(chunk)["using"] == 3
    # a new correction writes over the copies in use; back to the originals
    photos.quick(chunk)
    assert photos.original(cams[0]) == folder / "IMG_0.JPG"
    res = photos.use(chunk, "original")
    assert res["switched"] == 3 and cams[0].path == str(folder / "IMG_0.JPG")
    assert chunk.meta["sfm"]["image_root"] == str(folder) and photos.status(chunk)["using"] == 0


def test_the_points_seen_twice_give_the_differences_of_attenuation():
    """Four photos of the same bottom from different ranges: the points they share give the difference of
    attenuation between red, green and blue whatever the colours of the scene."""
    import cv2

    h, w = 220, 300
    grey = samples.blobs(h, w + 150, seed=11).astype(np.float32) / 255.0
    tint = cv2.resize(np.random.default_rng(4).uniform(0.4, 1.0, (4, 6, 3)).astype(np.float32), (w + 150, h))
    scene = np.clip(grey[..., None] * tint, 0.01, 0.95)
    holes = cv2.dilate((np.random.default_rng(5).uniform(size=grey.shape) < 0.004).astype(np.uint8),
                       np.ones((3, 3), np.uint8)) > 0
    scene[holes] = 0.004  # the shadows a real bottom has
    data = []
    for k, (shift, near) in enumerate(((0, 1.2), (50, 2.8), (100, 1.6), (150, 3.4))):
        j = scene[:, shift:shift + w]
        z = (near + 0.6 * np.linspace(0, 1, h)[:, None] * np.ones((1, w))).astype(np.float32)
        lin = colour.to_linear(samples.srgb(samples.underwater(j, z, seed=k)))
        data.append((k, lin, z, np.ones(z.shape, bool)))
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    model = colour.fit_model(data, pairs=pairs)
    assert model.method == "multiview" and model.observations > 150
    beta = np.array([colour.beta_model(1.0, model.beta[c]) for c in range(3)]) / model.zref
    diff = beta - beta[1]
    true = samples.BETA_D - samples.BETA_D[1]
    assert np.allclose(diff, true, atol=0.06), (diff, true)


def test_photos_restored_with_depth_images_made_elsewhere(tmp_path):
    """The restoration of the photos of a chunk with depth images in metres beside them (as the Sea-thru
    dataset gives them): copies written, the water taken away, the model kept in the project."""
    import cv2

    import pyarchinit_photogrammetry as ppg
    from ppg_underwater import photos
    from ppg_underwater.ranges import FileRanges

    folder, depths = tmp_path / "foto", tmp_path / "depth"
    folder.mkdir()
    depths.mkdir()
    grey = samples.blobs(160, 420, seed=21).astype(np.float32) / 255.0
    tint = cv2.resize(np.random.default_rng(6).uniform(0.4, 1.0, (4, 8, 3)).astype(np.float32), (420, 160))
    scene = np.clip(grey[..., None] * tint, 0.01, 0.95)
    holes = cv2.dilate((np.random.default_rng(7).uniform(size=grey.shape) < 0.004).astype(np.uint8),
                       np.ones((3, 3), np.uint8)) > 0
    scene[holes] = 0.004
    for k, (shift, near) in enumerate(((0, 1.5), (40, 3.0), (80, 2.0), (120, 3.5), (160, 1.8))):
        j = scene[:, shift:shift + 240]
        z = (near + 0.8 * np.linspace(1, 0, 160)[:, None] * np.ones((1, 240))).astype(np.float32)
        cv2.imwrite(str(folder / f"IMG_{k}.png"), samples.srgb(samples.underwater(j, z, seed=k)))
        cv2.imwrite(str(depths / f"depthIMG_{k}.tif"), z)
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    cams = chunk.add_photos(sorted(folder.iterdir()))
    source = FileRanges(depths, kind="range")
    info = photos.restore(chunk, source=source)
    assert info["photos"] == 5 and info["attenuation"].startswith("multiview") and "R" in info["coefficients"]
    before = colour.colour_cast(cv2.imread(str(cams[3].path)))
    after = colour.colour_cast(cv2.imread(str(photos.corrected(chunk, cams[3]))))
    assert before["r_g"] < 0.6 and abs(after["r_g"] - 1.0) < abs(before["r_g"] - 1.0) / 2, (before, after)
    model = colour.Model.from_dict(chunk.meta["underwater"]["colour"]["model"])
    assert model.method == "multiview" and len(model.reference) == 3
    # one model per photo, the differences between the channels from the survey
    info = photos.restore(chunk, per="photo", source=source)
    assert info["photos"] == 5 and "seathru" in cams[0].meta["underwater"] and \
        cams[0].meta["underwater"]["seathru"]["model"]["method"] == "multiview (survey)"
