"""The KMZ inside DJI Fly: finding the waypoint folder, reading the placeholder missions, copying every leg in
place of one of them with the right name, or making the folders to drag. Everything in temporary folders: no
test ever writes into a real DJI Fly folder or device."""
import io
import re
import sys
import zipfile
from pathlib import Path

import pytest

from ppg_flightplan import djifly
from ppg_flightplan import export_dji as dji
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import camera
from ppg_flightplan.djifly import DjiFlyError
from ppg_flightplan.route import Route
from planfixtures import reference_plan

UUID1 = "11111111-1111-4111-8111-111111111111"
UUID2 = "22222222-2222-4222-8222-222222222222"


def _placeholder(created_ms: int) -> bytes:
    """A mission of two waypoints, as a placeholder saved in DJI Fly would be (created at ``created_ms``)."""
    p = Route(name="fittizia", points=[(45.0, 10.0), (45.0005, 10.0)], height=30.0)
    return dji.kmz(p, camera("mini4pro"), now_ms=created_ms)


@pytest.fixture
def dji_device(tmp_path, monkeypatch):
    """A fake device: <root>/Android/data/dji.go.v5/files/waypoint with two placeholder missions, the first
    created before the second. Not under /Volumes: the "only /Volumes on the Mac" rule is switched off."""
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    root = tmp_path / "DJI RC 2"
    waypoint = root / djifly.SUBFOLDER
    placeholder = _placeholder(1_699_000_000_000)
    for uuid, ms in ((UUID1, 1_699_000_000_000), (UUID2, 1_699_000_100_000)):
        (waypoint / uuid).mkdir(parents=True)
        (waypoint / uuid / f"{uuid}.kmz").write_bytes(_placeholder(ms))
    return {"root": root, "waypoint": waypoint, "uuid1": UUID1, "uuid2": UUID2, "placeholder": placeholder}


@pytest.fixture
def test_plan():
    piano = reference_plan()
    assert len(pl.plan_legs(piano)) >= 2
    return piano


def test_the_waypoint_folder_is_found_from_any_level(dji_device, tmp_path):
    root, waypoint = dji_device["root"], dji_device["waypoint"]
    assert djifly.normalise(str(root)) == waypoint
    assert djifly.normalise(str(root / "Android")) == waypoint
    assert djifly.normalise(str(root / "Android" / "data" / "dji.go.v5")) == waypoint
    assert djifly.normalise(str(waypoint) + " ") == waypoint
    with pytest.raises(DjiFlyError):
        djifly.normalise(str(tmp_path / "non-esiste"))
    with pytest.raises(DjiFlyError):
        djifly.normalise(str(tmp_path))          # it exists, but without the structure of DJI Fly
    assert isinstance(djifly.candidates(), list)  # depends on the machine: enough that it does not explode


def test_the_placeholder_missions_are_read_in_the_order_they_were_created(dji_device):
    m = djifly.missions(dji_device["waypoint"])
    assert [x.folder for x in m] == [dji_device["uuid1"], dji_device["uuid2"]]
    assert all(x.points == 2 and x.author == dji.AUTHOR and x.size > 0 for x in m)
    assert m[0].to_json()["cartella"] == dji_device["uuid1"]
    assert djifly.kmz_points(b"non sono uno zip") == (None, None)


def test_our_kmz_is_ours_whether_written_here_or_by_the_planner(dji_device):
    m = djifly.missions(dji_device["waypoint"])[0]
    assert m.ours and m.to_json()["nostra"] is True
    with zipfile.ZipFile(io.BytesIO(dji_device["placeholder"])) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    entries["wpmz/template.kml"] = entries["wpmz/template.kml"].replace(dji.AUTHOR.encode(), b"Photo3D")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for n, c in entries.items():
            z.writestr(n, c)
    _, author = djifly.kmz_points(buffer.getvalue())
    assert author == "Photo3D" and author in djifly.OUR_AUTHORS
    assert not djifly.Mission("x", 0.0, author="DJI Fly").ours


def test_every_leg_ends_in_place_of_its_mission_with_the_right_name(dji_device, test_plan):
    waypoint, u1, u2 = dji_device["waypoint"], dji_device["uuid1"], dji_device["uuid2"]
    legs = dji.kmz_per_leg(test_plan)
    assert len(legs) >= 2
    two = legs[:2]
    outcomes = djifly.copy(waypoint, two, [u2, u1])
    assert [e["contenitore"] for e in outcomes] == [u2, u1]
    # same content (points, author), but with the creation time of the placeholder
    for u, (name, data) in ((u2, two[0]), (u1, two[1])):
        written = (waypoint / u / f"{u}.kmz").read_bytes()
        assert djifly.kmz_info(written)[:2] == djifly.kmz_info(data)[:2]
        assert djifly.kmz_info(written)[2] != djifly.kmz_info(data)[2]
    assert sorted(p.name for p in (waypoint / u1).iterdir()) == [f"{u1}.kmz"]   # no file more
    read = next(m for m in djifly.missions(waypoint) if m.folder == u2)
    assert read.points == djifly.kmz_points(two[0][1])[0] and read.points > 2
    # the errors speak clearly: too few containers, doubled, missing, odd names
    with pytest.raises(DjiFlyError, match="servono 2 contenitori"):
        djifly.copy(waypoint, two, [u1])
    with pytest.raises(DjiFlyError, match="due tratte"):
        djifly.copy(waypoint, two, [u1, u1])
    with pytest.raises(DjiFlyError, match="non c'è la missione"):
        djifly.copy(waypoint, two, [u1, "spuria-che-non-esiste"])
    with pytest.raises(DjiFlyError, match="non valido"):
        djifly.copy(waypoint, two, [u1, "../fuori"])
    with pytest.raises(DjiFlyError, match="nessuna tratta"):
        djifly.copy(waypoint, [], [])


def test_the_ready_folders_to_drag(dji_device, test_plan):
    u1, u2 = dji_device["uuid1"], dji_device["uuid2"]
    legs = dji.kmz_per_leg(test_plan)[:2]
    data = djifly.zip_containers(legs, [u1, u2])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == sorted([f"{u1}/{u1}.kmz", f"{u2}/{u2}.kmz", "LEGGIMI.txt"])
        assert z.read(f"{u1}/{u1}.kmz") == legs[0][1]
        assert "waypoint" in z.read("LEGGIMI.txt").decode()


def test_mtp_devices_exist_only_on_windows():
    assert djifly.is_mtp("mtp://DJI RC 2/Internal shared storage") and not djifly.is_mtp("/Volumes/SD")
    if not sys.platform.startswith("win"):
        assert djifly.mtp_devices() == []


def test_the_copy_reads_back_and_checks(dji_device):
    waypoint, u1 = dji_device["waypoint"], dji_device["uuid1"]
    outcomes = djifly.copy(waypoint, [("a.kmz", b"x" * 100)], [u1])
    assert outcomes[0]["byte"] == 100 and outcomes[0]["letti"] == 100 and outcomes[0]["verificato"] is True


def test_the_temporary_copy_of_idroid_is_not_the_remote_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    fake = tmp_path / "Library/Containers/com.leonboettger.idroid/Data/tmp/waypoint"
    (fake / "AAA").mkdir(parents=True)
    (fake / "AAA" / "AAA.kmz").write_bytes(b"x")
    assert djifly.is_local_copy(fake)
    assert not djifly.is_local_copy(tmp_path / "DJI RC 2/Android/data/dji.go.v5/files/waypoint")
    with pytest.raises(DjiFlyError, match="iDroid"):
        djifly.copy(fake, [("a.kmz", b"x")], ["AAA"])


def test_the_previews_end_in_map_preview_and_in_the_zip(dji_device):
    waypoint, u1, u2 = dji_device["waypoint"], dji_device["uuid1"], dji_device["uuid2"]
    two = [("a.kmz", b"x" * 10), ("b.kmz", b"y" * 20)]
    outcomes = djifly.copy(waypoint, two, [u1, u2], [b"JPG1", None])
    assert (waypoint / "map_preview" / u1 / f"{u1}.jpg").read_bytes() == b"JPG1"
    assert outcomes[0]["anteprima"].endswith(f"{u1}.jpg") and "anteprima" not in outcomes[1]
    assert not (waypoint / "map_preview" / u2).exists()
    with zipfile.ZipFile(io.BytesIO(djifly.zip_containers(two, [u1, u2], [b"JPG1", b"JPG2"]))) as z:
        names = z.namelist()
        assert f"{u1}/{u1}.kmz" in names and f"map_preview/{u1}/{u1}.jpg" in names
        assert f"map_preview/{u2}/{u2}.jpg" in names
        assert "map_preview" in z.read("LEGGIMI.txt").decode()


def test_the_name_in_dji_fly_is_the_creation_time_of_the_template(dji_device):
    m = djifly.missions(dji_device["waypoint"])
    assert all(x.created and x.dji_name for x in m)
    assert re.fullmatch(r"\d\d/\d\d/\d{4} \d\d:\d\d:\d\d", m[0].dji_name)
    assert m[0].to_json()["nomeDji"] == m[0].dji_name
    assert djifly.kmz_info(b"non sono uno zip") == (None, None, None)


def test_the_idroid_copy_is_listed_but_only_to_read(tmp_path, monkeypatch, dji_device):
    fake = tmp_path / "Library/Containers/com.leonboettger.idroid/Data/tmp/waypoint"
    fake.mkdir(parents=True)
    u1 = dji_device["uuid1"]
    (fake / u1).mkdir()
    (fake / u1 / f"{u1}.kmz").write_bytes(dji_device["placeholder"])
    monkeypatch.setattr(djifly, "IDROID_COPIES", [str(fake)])
    monkeypatch.setattr(djifly, "_path_patterns", lambda: [])
    assert djifly.candidates() == [fake] and djifly.is_local_copy(fake)
    assert [m.folder for m in djifly.missions(fake)] == [u1]


def test_the_kmz_written_into_the_placeholder_inherits_its_creation_time(dji_device, test_plan):
    waypoint, u1 = dji_device["waypoint"], dji_device["uuid1"]
    before = djifly.kmz_info((waypoint / u1 / f"{u1}.kmz").read_bytes())[2]
    assert before
    ours = dji.kmz_per_leg(test_plan)[0]
    assert djifly.kmz_info(ours[1])[2] != before
    djifly.copy(waypoint, [ours], [u1])
    after = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    assert djifly.kmz_info(after) == (len(pl.plan_legs(test_plan)[0].route.points), dji.AUTHOR, before)
    # in the template too there are createTime and updateTime, aligned
    with zipfile.ZipFile(io.BytesIO(after)) as z:
        text = z.read("wpmz/template.kml").decode()
    assert (re.search(r"<wpml:createTime>(\d+)<", text).group(1) == re.search(r"<wpml:updateTime>(\d+)<", text).group(1)
            == str(int(round(before * 1000))))
    # with the time given explicitly, and in the zip
    explicit = 1_700_000_000.0
    djifly.copy(waypoint, [ours], [u1], None, [explicit])
    assert djifly.kmz_info((waypoint / u1 / f"{u1}.kmz").read_bytes())[2] == explicit
    with zipfile.ZipFile(io.BytesIO(djifly.zip_containers([ours], [u1], None, [explicit]))) as z:
        assert djifly.kmz_info(z.read(f"{u1}/{u1}.kmz"))[2] == explicit
    assert djifly.with_create_time(b"non sono uno zip", 1.0) == b"non sono uno zip"


def test_on_the_mac_only_volumes_is_a_device(tmp_path, monkeypatch):
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", True)
    assert djifly.is_local_copy(tmp_path / "Downloads/waypoint")
    assert not djifly.is_local_copy(Path("/Volumes/SD/Android/data/dji.go.v5/files/waypoint"))
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    assert not djifly.is_local_copy(tmp_path / "Downloads/waypoint")
