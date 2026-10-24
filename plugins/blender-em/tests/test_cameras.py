"""The cameras of the chunk for Blender: the pose in the world without the scale of the chunk, the
photo undistorted so that a pinhole sees in it what the lens saw, the photos kept and taken away."""
from __future__ import annotations

import numpy as np
import pytest
from samples import DOT, ORIGIN, add_cameras, look_down, project

from ppg_blender_em import cameras, scene


def _dot(image):
    """Where the white dot of a photo is (pixel coordinates counted as COLMAP does)."""
    grey = image.astype(float).sum(axis=2)
    ys, xs = np.nonzero(grey > 600)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def test_the_pose_in_the_world(tmp_path):
    doc = project(tmp_path, model=False)
    chunk = doc.chunk
    c, s = np.cos(0.4), np.sin(0.4)
    t = np.eye(4)
    t[:3, :3] = 2.5 * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # the frame of an alignment: turned, scaled
    t[:3, 3] = ORIGIN
    chunk.transform = t
    add_cameras(chunk, tmp_path / "foto", n=1)
    shift = [300000.0, 4500000.0, 99.0]
    pose = cameras.world_pose(chunk, chunk.cameras[0], shift)
    expected = look_down((ORIGIN[0] + 1.5, ORIGIN[1] + 1.5, ORIGIN[2]), 3.0, 0.0)
    assert pose[:3, :3] == pytest.approx(expected[:3, :3], abs=1e-9)
    assert pose[:3, 3] == pytest.approx(expected[:3, 3] - shift, abs=1e-6)


def test_the_photo_undistorted(tmp_path):
    import cv2

    doc = project(tmp_path, model=False)
    chunk = doc.chunk
    sensor = add_cameras(chunk, tmp_path / "foto", n=2)
    maps = cameras.undistort_map(sensor, max_side=320)  # half the size of the photo
    assert cameras.output_size(sensor, 320) == (320, 240)
    k = maps[2]
    for cam in chunk.cameras:
        out = cameras.undistort(cv2.imread(cam.path), maps)
        assert out.shape == (240, 320, 3)
        # the pinhole of the undistorted photo sees the point where the photo shows it
        world_from_cam = chunk.transform @ cam.transform
        p = (np.linalg.inv(world_from_cam) @ np.append(DOT, 1.0))[:3]
        pin = (k @ (p / p[2]))[:2]
        assert np.linalg.norm(_dot(out) - pin) < 0.8
        # and the photo as taken shows it elsewhere, in its own pixels: the lens bends it (k1 = -0.08)
        assert np.linalg.norm(_dot(cv2.imread(cam.path)) - 2 * pin) > 1.5


def test_cameras_in_the_manifest(tmp_path):
    doc = project(tmp_path)
    chunk = doc.chunk
    add_cameras(chunk, tmp_path / "foto", n=2)
    folders = scene.project_folders(root=tmp_path / "Scavo")
    first = scene.export_project(chunk, folders, cameras=True, max_side=320)
    assert first["cameras"]["written"] == ["IMG_0000", "IMG_0001"]
    manifest = scene.read_manifest(folders.base)
    assert manifest["resolution"] == [320, 240]
    record = manifest["cameras"][0]
    assert record["file"] == "RB/Chunk 1/foto/IMG_0000.jpg" and (folders.base / record["file"]).exists()
    assert record["width"] == 320 and record["fx"] == pytest.approx(0.9 * 640 / 2)
    assert record["cx"] == pytest.approx((320 + 7.0) / 2) and record["cy"] == pytest.approx((240 - 5.0) / 2)
    assert np.asarray(record["matrix"])[:3, 3] == pytest.approx(
        [ORIGIN[0] + 1.5 - first["shift"][0], ORIGIN[1] + 1.5 - first["shift"][1], ORIGIN[2] + 3.0 - first["shift"][2]])
    # the live update keeps them, without undistorting again
    again = scene.export_project(chunk, folders)
    assert "cameras" not in again and scene.read_manifest(folders.base)["cameras"] == manifest["cameras"]
    # asked again with nothing changed: the photos are kept
    third = scene.export_project(chunk, folders, cameras=True, max_side=320)
    assert third["cameras"]["kept"] == ["IMG_0000", "IMG_0001"] and third["revision"] == manifest["revision"]
    # a camera gone: its photo goes
    chunk.cameras = chunk.cameras[:1]
    fourth = scene.export_project(chunk, folders, cameras=True, max_side=320)
    assert fourth["cameras"]["removed"] == ["IMG_0001"] and not (folders.base / "RB/Chunk 1/foto/IMG_0001.jpg").exists()
    # without cameras: none in Blender, no photo left
    scene.export_project(chunk, folders, cameras=False)
    assert scene.read_manifest(folders.base)["cameras"] == [] and not list((folders.base / "RB/Chunk 1/foto").glob("*.jpg"))
