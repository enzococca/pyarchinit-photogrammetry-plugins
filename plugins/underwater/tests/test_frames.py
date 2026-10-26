"""Usable photos and frames: the scores of blurred, hazy and flat photos, flagged and excluded; frames
of a fast sequence chosen by motion and sharpness; the times of videos; Kalibr calibrations and the
folders of ROV datasets."""
from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

import samples
from ppg_underwater import frames, quality, sources


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def test_scores_tell_blurred_hazy_and_flat_photos(tmp_path):
    base = cv2.cvtColor(samples.blobs(300, 400, seed=1), cv2.COLOR_GRAY2BGR)
    sharp = _write(tmp_path / "sharp.png", base)
    blurred = _write(tmp_path / "blurred.png", cv2.GaussianBlur(base, (0, 0), 4))
    hazy_img = (0.3 * base.astype(np.float32) + np.array([150, 140, 60], np.float32)).clip(0, 255).astype(np.uint8)
    hazy = _write(tmp_path / "hazy.png", hazy_img)
    flat = _write(tmp_path / "flat.png", (base.astype(np.float32) * 0.08 + 10).astype(np.uint8))
    scores = quality.score_paths([sharp, blurred, hazy, flat])
    s, b, h, f = scores
    limits = quality.thresholds()
    assert quality.reasons(s, limits) == []
    assert "blur" in quality.reasons(b, limits) and b["quality"] < 0.5
    assert "veil" in quality.reasons(h, limits) and h["veil"] > s["veil"] + 0.3
    assert "contrast" in quality.reasons(f, limits) and f["contrast"] < 0.12
    assert quality.reasons(None, limits) == ["unreadable"]


def test_flagged_photos_excluded_and_included_again(tmp_path):
    import pyarchinit_photogrammetry as ppg

    base = cv2.cvtColor(samples.blobs(240, 320, seed=2), cv2.COLOR_GRAY2BGR)
    paths = [_write(tmp_path / "f" / f"{i}.png", base if i % 3 else cv2.GaussianBlur(base, (0, 0), 5)) for i in range(6)]
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    chunk.add_photos(paths)
    rep = quality.score_chunk(chunk)
    assert rep["photos"] == 6 and rep["flagged"] == 2
    assert all(c.meta["quality"] is not None and "scores" in c.meta["underwater"] for c in chunk.cameras)
    assert quality.exclude(chunk) == 2 and sum(c.enabled for c in chunk.cameras) == 4
    # other limits: flags again from the scores kept
    assert quality.reflag(chunk, {"quality": 0.0, "contrast": 0.0, "veil": 1.0}) == 0
    assert quality.include(chunk) == 2 and all(c.enabled for c in chunk.cameras)
    out = quality.write_csv(chunk, tmp_path / "foto.csv")
    assert out.read_text().splitlines()[0].startswith("photo,enabled,quality")


@pytest.fixture
def sequence(tmp_path):
    """A camera moving over the bottom at 4 px a frame (320 px wide frames), hovering 40 frames, moving on;
    one frame in five blurred; their times in a table, as an ROV logs them."""
    world = samples.blobs(260, 1700, seed=5)
    xs = list(range(0, 400, 4)) + [400] * 40 + list(range(404, 1300, 4))
    folder = tmp_path / "seq"
    rows = ["#timestamp [ns], frame_id"]
    for i, x in enumerate(xs):
        img = world[10:250, x:x + 320]
        if i % 5 == 2:
            img = cv2.blur(img, (1, 15))  # moved: blurred along the motion
        _write(folder / f"frame{i:06d}.png", img)
        rows.append(f"{1542883311000000000 + i * 50_000_000},frame{i:06d}.png")
    (tmp_path / "img_sequence.csv").write_text("\n".join(rows) + "\n")
    return folder, xs


def test_frames_chosen_by_motion_and_sharpness(sequence, tmp_path):
    folder, xs = sequence
    imgs = frames.list_images(folder)
    times = frames.read_frame_times(tmp_path / "img_sequence.csv", imgs)
    assert len(times) == len(xs) and times["frame000001.png"] == pytest.approx(1542883311.05)
    res = frames.select_sequence(imgs, times, overlap=0.8)
    kept = [int(f["index"]) for f in res["frames"]]
    assert res["segments"] == 1 and res["unmatched"] == 0
    assert all(k % 5 != 2 for k in kept)  # never a blurred frame
    shifts = np.diff([xs[k] for k in kept])
    step = 0.2 * 320
    assert shifts.max() <= step + 8 and np.median(shifts) >= 0.5 * step - 8, shifts
    hover = [k for k in kept if 100 <= k < 140]
    assert len(hover) <= 1  # hovering gives no new frames
    assert all(f["time"] is not None for f in res["frames"])
    # added to a chunk with their time
    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    cams = frames.add_frames(chunk, res["frames"], source="seq")
    assert len(cams) == len(kept) and cams[0].meta["underwater"]["time"] == pytest.approx(times[f"frame{kept[0]:06d}.png"])
    from ppg_underwater.depthlog import photo_time

    assert photo_time(cams[1])[1] == "record"


def test_a_break_starts_a_new_stretch(tmp_path):
    a = samples.blobs(240, 900, seed=7)
    b = samples.blobs(240, 900, seed=8)
    paths = [_write(tmp_path / "s" / f"{i:03d}.png", a[:, 4 * i:4 * i + 320]) for i in range(30)]
    paths += [_write(tmp_path / "s" / f"{30 + i:03d}.png", b[:, 4 * i:4 * i + 320]) for i in range(30)]
    res = frames.select_sequence(paths, overlap=0.8)
    assert res["segments"] == 2 and res["unmatched"] == 1
    idx = [f["index"] for f in res["frames"]]
    assert any(i < 30 for i in idx) and any(i >= 30 for i in idx)


def test_the_creation_time_of_a_video(tmp_path):
    created = 1717236000  # 2024-06-01 10:00:00 UTC
    mvhd = struct.pack(">I4sB3sII", 108, b"mvhd", 0, b"\x00\x00\x00", created + 2082844800, created + 2082844800) + b"\x00" * 88
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s4sI", 16, b"ftyp", b"isom", 0)
    path = tmp_path / "GX010001.MP4"
    path.write_bytes(ftyp + struct.pack(">I4s", 16, b"free") + b"\x00" * 8 + moov)
    assert frames.mp4_creation_time(path) == created
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 20)
    assert frames.mp4_creation_time(tmp_path / "x.mp4") is None


def test_kalibr_calibration_and_the_folder_of_a_run(tmp_path):
    import pyarchinit_photogrammetry as ppg

    yaml_path = tmp_path / "camera_calib.yaml"
    yaml_path.write_text("cam0:\n  cam_overlaps: []\n  camera_model: pinhole\n  distortion_coeffs: [-0.1255945656257394, "
                         "0.053221287232781606, 9.94070021080493e-05,\n    9.550660927242349e-05]\n"
                         "  distortion_model: radtan\n  intrinsics: [543.33, 542.40, 489.03, 305.39]\n"
                         "  resolution: [968, 608]\n  rostopic: /camera/image_raw\n")
    calib = sources.read_kalibr(yaml_path)
    assert calib["model"] == "OPENCV" and calib["width"] == 968
    assert calib["params"][:4] == pytest.approx([543.33, 542.40, 489.53, 305.89])  # centre of the first pixel at 0.5
    assert calib["params"][4] == pytest.approx(-0.1255945656257394)
    run = tmp_path / "raw_data"
    for i in range(3):
        _write(run / "images_sequence_1" / f"frame{i:06d}.png", samples.blobs(608, 968, seed=i))
    (run / "img_sequence_1.csv").write_text("#timestamp [ns], frame_id\n1,frame000000.png\n")
    (run / "depth_sequence_1.csv").write_text("#timestamp [ns], depth [m]\n1,10.0\n")
    found = sources.find_dataset(run)
    assert found["images"].name == "images_sequence_1" and found["count"] == 3
    assert found["times"].name == "img_sequence_1.csv" and found["depth"].name == "depth_sequence_1.csv"
    assert found["calibration"] == yaml_path
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    chunk.add_photos(sorted((run / "images_sequence_1").iterdir()))
    assert sources.apply_calibration(chunk, calib) == 1
    (sensor,) = chunk.sensors.values()
    assert sensor.model == "OPENCV" and sensor.fixed and sensor.params[2] == pytest.approx(489.53)
