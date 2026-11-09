"""Cameras: the planner's list as it is, and the three ways to a camera that is not in it - a photo's EXIF,
the calibrated sensor of a chunk, typed by hand. No sensor value is ever made up."""
import pytest
from PIL import Image

from ppg_flightplan.camera import (CAMERAS, Camera, CameraUnknown, camera, camera_from_exif, camera_from_sensor,
                                   manual_camera)


def test_the_list_is_unique_and_positive():
    ids = [c.id for c in CAMERAS]
    assert len(ids) == len(set(ids)) == 20
    for c in CAMERAS:
        assert c.width_mm > 0 and c.height_mm > 0 and c.focal_mm > 0 and c.pixels > 0
        assert c.min_interval > 0 and c.min_interval_high > 0
        assert c.pixels_high is None or c.pixels_high > c.pixels
    assert camera("mini4pro").gsd_cm(110, True) == pytest.approx(1.95, abs=0.02)
    assert camera("nessuna") is CAMERAS[0]


def _jpeg(path, width, height, focal=None, f35=None):
    im = Image.new("RGB", (width, height), (120, 120, 120))
    exif = Image.Exif()
    exif[0x010F] = "Prova"
    exif[0x0110] = "Camera di prova"
    sub = {}
    if focal is not None:
        sub[0x920A] = focal                          # FocalLength
    if f35 is not None:
        sub[0xA405] = f35                            # FocalLengthIn35mmFilm
    if sub:
        exif.get_ifd(0x8769).update(sub)
    im.save(path, exif=exif.tobytes())
    return path


def test_camera_from_exif(tmp_path):
    c = camera_from_exif(_jpeg(tmp_path / "a.jpg", 4032, 3024, focal=6.7, f35=24))
    assert c.width_mm == pytest.approx(10.05, abs=0.01)          # 36 * 6.7 / 24
    assert c.height_mm == pytest.approx(7.54, abs=0.01)
    assert c.focal_mm == pytest.approx(6.7) and c.pixels == 4032
    assert c.pixels_high is None and c.min_interval == 2.0
    assert "Camera di prova" in c.name
    with pytest.raises(CameraUnknown):
        camera_from_exif(_jpeg(tmp_path / "b.jpg", 4032, 3024, focal=6.7))
    with pytest.raises(CameraUnknown):
        camera_from_exif(_jpeg(tmp_path / "c.jpg", 4032, 3024, f35=24))
    portrait = camera_from_exif(_jpeg(tmp_path / "d.jpg", 3024, 4032, focal=6.7, f35=24))
    assert portrait.width_mm == pytest.approx(10.05, abs=0.01) and portrait.pixels == 4032


def test_camera_from_sensor():
    from pyarchinit_photogrammetry.core.camera import Sensor

    s = Sensor(key=1, label="prova", width=4000, height=3000, model="PINHOLE",
               params=[3000.0, 3000.0, 2000.0, 1500.0], focal_length_mm=6.0)
    c = camera_from_sensor(s)
    assert c.width_mm == pytest.approx(8.0) and c.height_mm == pytest.approx(6.0)
    assert c.focal_mm == 6.0 and c.pixels == 4000
    with pytest.raises(CameraUnknown):
        camera_from_sensor(Sensor(key=2, label="senza", width=4000, height=3000, model="PINHOLE",
                                  params=[3000.0, 3000.0, 2000.0, 1500.0]))


def test_manual_camera():
    c = manual_camera("Mia camera", 13.2, 8.8, 8.8, 5472, min_interval=2.5)
    assert c.gsd_cm(100, False) == pytest.approx(13.2 * 100 * 100 / (8.8 * 5472))
    assert Camera.from_json(c.to_json()) == c
    for bad in (dict(width_mm=0), dict(height_mm=-1), dict(focal_mm=0), dict(pixels=0), dict(min_interval=0)):
        args = dict(name="x", width_mm=13.2, height_mm=8.8, focal_mm=8.8, pixels=5472) | bad
        with pytest.raises(ValueError):
            manual_camera(**args)
    # a custom camera stored in the plan wins when its id is chosen
    assert camera(c.id, c.to_json()) == c
    assert camera("mini4pro", c.to_json()).id == "mini4pro"


def test_every_listed_camera_comes_back_from_json():
    for c in CAMERAS:
        assert Camera.from_json(c.to_json()) == c


def test_a_broken_camera_of_the_plan_is_refused_not_replaced():
    """A camera stored on purpose in the plan never becomes the Mini 4 Pro in silence: strips, step and GSD
    would be planned with another sensor."""
    from ppg_flightplan import plan as pl

    good = manual_camera("Mia camera", 13.2, 8.8, 8.8, 5472).to_json()
    broken = dict(good, focaleMm=0)
    with pytest.raises(CameraUnknown, match="Mia camera"):
        camera(good["id"], broken)
    with pytest.raises(CameraUnknown):
        camera(good["id"], {k: v for k, v in good.items() if k != "larghezzaMm"})
    piano = {"camera": good["id"], "cameraPersonale": broken, "blocchi": []}
    with pytest.raises(CameraUnknown):
        pl.plan_camera(piano)
    # an id of the list that is not the plan's own camera still works as before
    assert camera("mini4pro", broken).id == "mini4pro"


def test_every_sensor_value_is_checked(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="pixels_high"):
        Camera.from_json(dict(camera("mini4pro").to_json(), pixelAlta=-8000))
    # the EXIF path checks the camera it makes: an absurd tag does not give a negative sensor
    import pyarchinit_photogrammetry.io.exif as exif

    monkeypatch.setattr(exif, "read_exif", lambda path: {"focal_length_mm": -6.7, "focal_length_35mm": 24,
                                                         "width": 4032, "height": 3024, "make": "X", "model": "Y"})
    with pytest.raises(ValueError, match="width_mm"):
        camera_from_exif(tmp_path / "x.jpg")
