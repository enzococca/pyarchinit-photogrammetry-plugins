"""The camera of the drone: footprint, ground resolution, shortest interval between shots, and the codes the
DJI WPML names drone and camera with.

``CAMERAS`` is the planner's list as it is (values from the spec sheets, ported from the author's Photo3D
planner). A camera that is not in the list is never made up: it comes from a photo's EXIF
(:func:`camera_from_exif`), from the calibrated sensor of a chunk (:func:`camera_from_sensor`) or is typed
by the user (:func:`manual_camera`); a plan stores it under ``cameraPersonale`` (:meth:`Camera.to_json`).

``width_mm`` is the sensor's LONG side (across the strips), ``height_mm`` the short one (along them),
``pixels`` the long side in pixels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .texts import t


class CameraUnknown(ValueError):
    """A camera that cannot be told from what is known (a photo without its focal lengths, a sensor without
    its focal length in mm)."""


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    width_mm: float
    height_mm: float
    focal_mm: float
    pixels: int = 4000
    pixels_high: Optional[int] = None
    min_interval: float = 2.0
    min_interval_high: float = 5.0
    # the codes the WPML names drone and camera with: 68 is what DJI Fly writes for the Mini 4 Pro
    drone_enum: int = 68
    drone_sub_enum: int = 0
    payload: Optional[int] = None
    enterprise: bool = False
    # the megapixels of the high-resolution shot (48, 50, 100): only for the label
    megapixels_high: Optional[int] = None
    # the Mobile SDK does not know it: the KMZ is flown only from DJI Fly, copied into the remote controller
    fly_only: bool = False
    # not a DJI: flown with the maker's app (or QGroundControl, Litchi), with the files of export_formats;
    # here the profile is for the height, the step between shots and the GSD
    app: str = ""

    @property
    def group(self) -> str:
        return "altro" if self.app else ("fly" if self.fly_only else "sdk")

    def footprint(self, height: float) -> float:
        return self.width_mm / self.focal_mm * height

    def footprint_short(self, height: float) -> float:
        return self.height_mm / self.focal_mm * height

    def pixels_for(self, high: bool) -> int:
        return (self.pixels_high or self.pixels) if high else self.pixels

    def min_interval_for(self, high: bool) -> float:
        return self.min_interval_high if (high and self.pixels_high) else self.min_interval

    def gsd_cm(self, height: float, high: bool) -> float:
        return self.width_mm * height * 100.0 / (self.focal_mm * self.pixels_for(high))

    def to_json(self) -> dict:
        """The keys of the planner (``nome``, ``altaRisoluzione``, ``gruppo``...) and the sensor's values, so
        that a camera stored in a plan (``cameraPersonale``) comes back as it was (:meth:`from_json`)."""
        return {"id": self.id, "nome": self.name, "altaRisoluzione": self.pixels_high is not None,
                "intervalloMinimo": self.min_interval, "intervalloMinimoAlta": self.min_interval_high,
                "megapixelAlta": self.megapixels_high, "enterprise": self.enterprise, "soloFly": self.fly_only,
                "gruppo": self.group, "app": self.app,
                "larghezzaMm": self.width_mm, "altezzaMm": self.height_mm, "focaleMm": self.focal_mm,
                "pixel": self.pixels, "pixelAlta": self.pixels_high, "codiceDrone": self.drone_enum,
                "sottocodiceDrone": self.drone_sub_enum, "payload": self.payload}

    @staticmethod
    def from_json(d: dict) -> "Camera":
        """The camera of :meth:`to_json`; ValueError when a sensor value is missing or not positive."""
        try:
            cam = Camera(id=str(d["id"]), name=str(d.get("nome") or d["id"]), width_mm=float(d["larghezzaMm"]),
                         height_mm=float(d["altezzaMm"]), focal_mm=float(d["focaleMm"]), pixels=int(d["pixel"]),
                         pixels_high=None if d.get("pixelAlta") is None else int(d["pixelAlta"]),
                         min_interval=float(d.get("intervalloMinimo", 2.0)),
                         min_interval_high=float(d.get("intervalloMinimoAlta", 5.0)),
                         drone_enum=int(d.get("codiceDrone", 68)), drone_sub_enum=int(d.get("sottocodiceDrone", 0)),
                         payload=None if d.get("payload") is None else int(d["payload"]),
                         enterprise=bool(d.get("enterprise", False)),
                         megapixels_high=None if d.get("megapixelAlta") is None else int(d["megapixelAlta"]),
                         fly_only=bool(d.get("soloFly", False)), app=str(d.get("app") or ""))
        except (KeyError, TypeError) as e:
            raise ValueError(f"camera: missing or wrong value ({e})") from None
        _check(cam)
        return cam


def _check(cam: Camera) -> None:
    for f in ("width_mm", "height_mm", "focal_mm", "pixels", "min_interval", "min_interval_high"):
        if not getattr(cam, f) > 0:
            raise ValueError(t("camera «{0}»: {1} deve essere maggiore di zero",
                               "camera '{0}': {1} must be greater than zero").format(cam.name, f))


CAMERAS: list[Camera] = [
    Camera("mini4pro", "DJI Mini 4 Pro", 9.6, 7.2, 6.7, 4032, 8064, megapixels_high=48),
    Camera("mini3", "DJI Mini 3 / Mini 3 Pro", 9.6, 7.2, 6.7, 4032, 8064, megapixels_high=48),
    Camera("mavic3e", "DJI Mavic 3E / 3M", 17.3, 13.0, 12.29, 5280, None, 2.0, 5.0, 77, 0, 66, True),
    Camera("mavic3t", "DJI Mavic 3T (grandangolo)", 6.4, 4.8, 4.4, 8000, None, 3.0, 5.0, 77, 1, 67, True),
    Camera("m30", "DJI Matrice 30 / 30T", 6.4, 4.8, 4.5, 4000, None, 2.0, 5.0, 67, 0, 52, True),
    Camera("m4e", "DJI Matrice 4E / 4T", 17.3, 13.0, 12.29, 5280, None, 2.0, 5.0, 99, 0, None, True),
    Camera("m300p1", "DJI Matrice 300 / 350 + P1 (35 mm)", 35.9, 24.0, 35.0, 8192, None, 2.0, 5.0, 60, 0, None, True),
    # DJI Fly flies these with waypoints, but the Mobile SDK does not know them: the KMZ is copied into the
    # remote controller. The file is the Mini 4 Pro's, code 68: tried on the Mini 4 Pro, reported working on
    # the Mini 5 Pro and the Mavic 3 Pro, taken as good on the others. The file does not choose the lens: on
    # drones with several cameras the wide one is set by hand. Sensors: 1" of 50 MP in 4:3 (13.2 x 9.9 mm,
    # 24 mm equiv. -> 9.2 mm); Mavic 4 Pro 4/3 in 3:2 (17.3 x 11.5 mm), 28 mm equiv. with a 72 degree field
    # -> 14.3 mm; the 5 s at 100 MP are a cautious estimate.
    Camera("mini5pro", "DJI Mini 5 Pro", 13.2, 9.9, 9.2, 4096, 8192, megapixels_high=50, fly_only=True),
    Camera("air3", "DJI Air 3 (grandangolo)", 9.6, 7.2, 6.7, 4032, 8064, megapixels_high=48, fly_only=True),
    Camera("air3s", "DJI Air 3S (grandangolo)", 13.2, 9.9, 9.2, 4096, 8192, megapixels_high=50, fly_only=True),
    Camera("mavic3", "DJI Mavic 3 / Classic / Pro (Hasselblad)", 17.3, 13.0, 12.29, 5280, fly_only=True),
    Camera("mavic4pro", "DJI Mavic 4 Pro (Hasselblad 28 mm)", 17.3, 11.5, 14.3, 6144, 12288, megapixels_high=100,
           fly_only=True),
    # Not DJI: flown with the maker's app, QGroundControl or Litchi, with the files of export_formats. Sensors
    # from the spec sheets, focal lengths from the 35 mm equivalent; the shortest intervals are cautious.
    # 1", 20 MP, 29 mm equiv.
    Camera("autel_evo2pro", "Autel EVO II Pro (V3)", 13.2, 8.8, 10.6, 5472, app="Autel Explorer"),
    # 1", 20 MP, 29 mm equiv.
    Camera("autel_liteplus", "Autel EVO Lite+", 13.2, 8.8, 10.6, 5472, app="Autel Sky"),
    # 1/2.4", 21 MP, 26 mm equiv.
    Camera("anafi", "Parrot Anafi / Anafi USA (grandangolo)", 5.8, 4.3, 4.3, 5344, app="FreeFlight 6"),
    # 1/2", 48 MP, 24 mm equiv.
    Camera("anafi_ai", "Parrot Anafi Ai", 6.4, 4.8, 4.4, 4000, 8000, megapixels_high=48, app="FreeFlight 7"),
    # 1/2.3", 12 MP, 20 mm equiv.
    Camera("skydio2", "Skydio 2 / 2+", 6.2, 4.6, 3.6, 4056, app="Skydio"),
    # DJI drones that the Mobile SDK v5 does not know and DJI Fly does not fly: the Mavic 2 Enterprise Advanced
    # goes with DJI Pilot, the Phantom 4 Multispectral with DJI GS Pro (iPad only); both are given the area in
    # KML. M2EA: visible camera 1/2" of 48 MP (8000 x 6000), 24 mm equiv. -> 4.4 mm. P4M: RGB camera 1/2.9" of
    # 2.08 MP (1600 x 1300), focal length 5.74 mm; the sensor comes from the GSD DJI declares, H/18.9 cm/px ->
    # 3.04 um pixels -> 4.86 x 3.95 mm. The shortest intervals are cautious.
    Camera("m2ea", "DJI Mavic 2 Enterprise Advanced (visibile)", 6.4, 4.8, 4.4, 8000, None, 3.0, app="DJI Pilot"),
    Camera("p4m", "DJI Phantom 4 Multispectral (RGB)", 4.86, 3.95, 5.74, 1600, app="DJI GS Pro (iPad)"),
    Camera("altro", "Altro drone (24 mm equivalenti)", 9.6, 7.2, 6.7, 4000),
]


def camera(id_: str, custom: Optional[dict] = None) -> Camera:
    """The camera ``id_``: the plan's own camera (``custom``, :meth:`Camera.to_json`) when its id is that one,
    else the one of the list; an unknown id gives the first of the list, as the planner does.

    A plan's own camera that is broken (a value missing or not positive) raises :class:`CameraUnknown`: it was
    chosen on purpose, and planning strips, steps and GSD with another sensor without a word would be wrong.
    """
    if custom and str(custom.get("id")) == str(id_):
        try:
            return Camera.from_json(custom)
        except ValueError as e:
            raise CameraUnknown(t("la camera del piano «{0}» non è valida: {1}",
                                  "the plan's camera '{0}' is not valid: {1}")
                                .format(custom.get("nome") or id_, e)) from None
    for c in CAMERAS:
        if c.id == id_:
            return c
    return CAMERAS[0]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:40] or "camera"


# The program's convention for the 35 mm equivalent: 36 mm on the long side of the image.
FULL_FRAME_LONG_MM = 36.0


def camera_from_exif(path) -> Camera:
    """The camera of a photo from its EXIF: the focal length and the 35 mm equivalent focal length are both
    needed (sensor long side = 36 x focal / focal35, the short side in the proportion of the image);
    :class:`CameraUnknown` when either is missing. The interval is the planner's default (2 s)."""
    from pyarchinit_photogrammetry.io.exif import read_exif

    info = read_exif(path)
    focal, f35 = info.get("focal_length_mm"), info.get("focal_length_35mm")
    if not focal or not f35:
        names = ((t("la focale", "the focal length"), focal),
                 (t("la focale equivalente 35 mm", "the 35 mm equivalent focal length"), f35))
        missing = [n for n, v in names if not v]
        raise CameraUnknown(t("{0}: nell'EXIF manca {1}: la camera non si ricava dalla foto",
                              "{0}: the EXIF lacks {1}: the camera cannot be told from the photo")
                            .format(Path(path).name, t(" e ", " and ").join(missing)))
    width, height = int(info["width"]), int(info["height"])
    long_px, short_px = max(width, height), min(width, height)
    width_mm = FULL_FRAME_LONG_MM * float(focal) / float(f35)
    label = " ".join(x for x in (info.get("make"), info.get("model")) if x) or Path(path).name
    return Camera(id=f"exif-{_slug(label)}-{long_px}", name=f"{label} (EXIF)", width_mm=round(width_mm, 4),
                  height_mm=round(width_mm * short_px / long_px, 4), focal_mm=float(focal), pixels=long_px)


def camera_from_sensor(sensor) -> Camera:
    """The camera of a calibrated sensor of a chunk: long side in mm = focal length in mm x long side in
    pixels / focal length in pixels. :class:`CameraUnknown` without the focal length in mm."""
    focal_mm = getattr(sensor, "focal_length_mm", None)
    if not focal_mm:
        raise CameraUnknown(t("sensore «{0}»: manca la focale in mm, la dimensione del sensore non si ricava",
                              "sensor '{0}': the focal length in mm is missing, the sensor size cannot be told")
                            .format(getattr(sensor, "label", "")))
    f_px = float(sensor.focal_px)
    long_px, short_px = max(int(sensor.width), int(sensor.height)), min(int(sensor.width), int(sensor.height))
    if f_px <= 0 or long_px <= 0:
        raise CameraUnknown(t("sensore «{0}»: focale in pixel o dimensioni non valide",
                              "sensor '{0}': focal length in pixels or size not valid")
                            .format(getattr(sensor, "label", "")))
    width_mm = float(focal_mm) * long_px / f_px
    label = str(getattr(sensor, "label", "") or f"sensore {getattr(sensor, 'key', '')}")
    return Camera(id=f"sensor-{getattr(sensor, 'key', 0)}-{_slug(label)}", name=f"{label} (sensore calibrato)",
                  width_mm=round(width_mm, 4), height_mm=round(width_mm * short_px / long_px, 4),
                  focal_mm=float(focal_mm), pixels=long_px)


def manual_camera(name: str, width_mm: float, height_mm: float, focal_mm: float, pixels: int,
                  min_interval: float = 2.0) -> Camera:
    """A camera typed by the user; ValueError unless every value is greater than zero."""
    cam = Camera(id=f"manual-{_slug(name)}", name=str(name), width_mm=float(width_mm), height_mm=float(height_mm),
                 focal_mm=float(focal_mm), pixels=int(pixels), min_interval=float(min_interval),
                 min_interval_high=float(min_interval))
    _check(cam)
    return cam

