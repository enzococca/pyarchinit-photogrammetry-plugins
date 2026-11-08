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
    return {"root": root, "waypoint": waypoint, "uuid1": UUID1, "uuid2": UUID2, "placeholder": placeholder,
            "backup": tmp_path / "progetto" / "flightplan" / "djifly_backup"}


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
    outcomes = djifly.copy(waypoint, two, [u2, u1], backup_dir=dji_device["backup"])
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
        djifly.copy(waypoint, two, [u1], backup_dir=dji_device["backup"])
    with pytest.raises(DjiFlyError, match="due tratte"):
        djifly.copy(waypoint, two, [u1, u1], backup_dir=dji_device["backup"])
    with pytest.raises(DjiFlyError, match="non c'è la missione"):
        djifly.copy(waypoint, two, [u1, "spuria-che-non-esiste"], backup_dir=dji_device["backup"])
    with pytest.raises(DjiFlyError, match="non valido"):
        djifly.copy(waypoint, two, [u1, "../fuori"], backup_dir=dji_device["backup"])
    with pytest.raises(DjiFlyError, match="nessuna tratta"):
        djifly.copy(waypoint, [], [], backup_dir=dji_device["backup"])


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
    outcomes = djifly.copy(waypoint, [("a.kmz", b"x" * 100)], [u1], backup_dir=dji_device["backup"])
    assert outcomes[0]["byte"] == 100 and outcomes[0]["letti"] == 100 and outcomes[0]["verificato"] is True


def test_the_temporary_copy_of_idroid_is_not_the_remote_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    fake = tmp_path / "Library/Containers/com.leonboettger.idroid/Data/tmp/waypoint"
    (fake / "AAA").mkdir(parents=True)
    (fake / "AAA" / "AAA.kmz").write_bytes(b"x")
    assert djifly.is_local_copy(fake)
    assert not djifly.is_local_copy(tmp_path / "DJI RC 2/Android/data/dji.go.v5/files/waypoint")
    with pytest.raises(DjiFlyError, match="iDroid"):
        djifly.copy(fake, [("a.kmz", b"x")], ["AAA"], backup_dir=tmp_path / "copie")


def test_the_previews_end_in_map_preview_and_in_the_zip(dji_device):
    waypoint, u1, u2 = dji_device["waypoint"], dji_device["uuid1"], dji_device["uuid2"]
    two = [("a.kmz", b"x" * 10), ("b.kmz", b"y" * 20)]
    outcomes = djifly.copy(waypoint, two, [u1, u2], [b"JPG1", None], backup_dir=dji_device["backup"])
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
    djifly.copy(waypoint, [ours], [u1], backup_dir=dji_device["backup"])
    after = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    assert djifly.kmz_info(after) == (len(pl.plan_legs(test_plan)[0].route.points), dji.AUTHOR, before)
    # in the template too there are createTime and updateTime, aligned
    with zipfile.ZipFile(io.BytesIO(after)) as z:
        text = z.read("wpmz/template.kml").decode()
    assert (re.search(r"<wpml:createTime>(\d+)<", text).group(1) == re.search(r"<wpml:updateTime>(\d+)<", text).group(1)
            == str(int(round(before * 1000))))
    # with the time given explicitly, and in the zip
    explicit = 1_700_000_000.0
    djifly.copy(waypoint, [ours], [u1], None, [explicit], backup_dir=dji_device["backup"])
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


# ------------------------------------------------------------------ nothing of the user is ever lost

UUID3 = "33333333-3333-4333-8333-333333333333"


def _mission(n: int, author: str, created_ms: int) -> bytes:
    """A KMZ of ``n`` waypoints written by ``author`` (DJI Fly, for a mission of the user or a placeholder)."""
    p = Route(name="m", points=[(45.0 + 0.0003 * i, 10.0 + 0.0002 * (i % 2)) for i in range(n)], height=30.0)
    cam = camera("mini4pro")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("wpmz/template.kml", dji.template_kml(p, cam, created_ms, author=author))
        z.writestr("wpmz/waylines.wpml", dji.waylines_wpml(p, cam))
    return buffer.getvalue()


def _add_slot(waypoint, uuid, data):
    (waypoint / uuid).mkdir(parents=True, exist_ok=True)
    if data is not None:
        (waypoint / uuid / f"{uuid}.kmz").write_bytes(data)


def test_the_replaced_file_is_kept_next_to_the_project(dji_device):
    waypoint, u1, backup = dji_device["waypoint"], dji_device["uuid1"], dji_device["backup"]
    old = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    leg = _mission(2, dji.AUTHOR, 1_700_000_000_000)
    outcome = djifly.copy(waypoint, [("a.kmz", leg)], [u1], backup_dir=backup)[0]
    kept = Path(outcome["copia"])
    assert kept.parent == backup.resolve() and kept.read_bytes() == old
    assert re.fullmatch(rf"{u1}-\d{{8}}-\d{{6}}\.kmz", kept.name)
    assert outcome["stato"] == djifly.SLOT_OURS
    # a second copy in the same second does not overwrite the first copy
    written = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    again = djifly.copy(waypoint, [("b.kmz", leg)], [u1], backup_dir=backup)[0]
    assert Path(again["copia"]).read_bytes() == written and kept.read_bytes() == old
    assert again["copia"] != outcome["copia"]


def test_a_mission_of_the_user_is_not_touched_without_confirmation(dji_device):
    waypoint, u1, backup = dji_device["waypoint"], dji_device["uuid1"], dji_device["backup"]
    mine = _mission(7, "DJI Fly", 1_698_000_000_000)
    _add_slot(waypoint, UUID3, mine)
    before = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    with pytest.raises(djifly.SlotsInUse) as err:
        djifly.copy(waypoint, [("a.kmz", b"x" * 10), ("b.kmz", b"y" * 10)], [u1, UUID3], backup_dir=backup)
    assert [s["contenitore"] for s in err.value.slots] == [UUID3]
    assert err.value.slots[0]["stato"] == djifly.SLOT_MISSION and err.value.slots[0]["punti"] == 7
    assert "non le ho toccate" in str(err.value)
    # nothing written, not even in the safe slot
    assert (waypoint / u1 / f"{u1}.kmz").read_bytes() == before
    assert (waypoint / UUID3 / f"{UUID3}.kmz").read_bytes() == mine
    # confirmed: written, and the mission is kept
    outcomes = djifly.copy(waypoint, [("a.kmz", b"x" * 10), ("b.kmz", b"y" * 10)], [u1, UUID3],
                           backup_dir=backup, overwrite=True)
    assert outcomes[1]["stato"] == djifly.SLOT_MISSION and Path(outcomes[1]["copia"]).read_bytes() == mine


def test_the_states_of_the_slots_for_the_pane(dji_device):
    waypoint = dji_device["waypoint"]
    _add_slot(waypoint, UUID3, _mission(2, "DJI Fly", 1_698_000_000_000))
    _add_slot(waypoint, "vuota", None)
    _add_slot(waypoint, "rotta", b"non sono uno zip")
    states = {s["contenitore"]: s for s in djifly.slot_states(waypoint, [dji_device["uuid1"], UUID3, "vuota", "rotta"])}
    assert states[dji_device["uuid1"]]["stato"] == djifly.SLOT_OURS
    assert states[UUID3]["stato"] == djifly.SLOT_PLACEHOLDER and states[UUID3]["nomeDji"]
    assert states["vuota"]["stato"] == djifly.SLOT_EMPTY
    assert states["rotta"]["stato"] == djifly.SLOT_MISSION          # a file that cannot be read is protected
    # a placeholder of DJI Fly and an empty slot are written without asking
    outcomes = djifly.copy(waypoint, [("a.kmz", b"x"), ("b.kmz", b"y")], [UUID3, "vuota"],
                           backup_dir=dji_device["backup"])
    assert [o["stato"] for o in outcomes] == [djifly.SLOT_PLACEHOLDER, djifly.SLOT_EMPTY]
    assert outcomes[1]["copia"] is None
    with pytest.raises(DjiFlyError, match="non c'è la missione"):
        djifly.slot_states(waypoint, ["manca"])


def test_the_copies_need_their_folder_outside_the_waypoints(dji_device):
    waypoint, u1 = dji_device["waypoint"], dji_device["uuid1"]
    with pytest.raises(DjiFlyError, match="cartella per le copie"):
        djifly.copy(waypoint, [("a.kmz", b"x")], [u1])
    with pytest.raises(DjiFlyError, match="dentro la cartella dei waypoint"):
        djifly.copy(waypoint, [("a.kmz", b"x")], [u1], backup_dir=waypoint / "copie")
    assert not (waypoint / "copie").exists()


def test_the_replaced_preview_is_kept_too(dji_device):
    waypoint, u1, backup = dji_device["waypoint"], dji_device["uuid1"], dji_device["backup"]
    (waypoint / "map_preview" / u1).mkdir(parents=True)
    (waypoint / "map_preview" / u1 / f"{u1}.jpg").write_bytes(b"OLD")
    outcome = djifly.copy(waypoint, [("a.kmz", b"x")], [u1], [b"NEW"], backup_dir=backup)[0]
    assert Path(outcome["copiaAnteprima"]).read_bytes() == b"OLD"
    assert (waypoint / "map_preview" / u1 / f"{u1}.jpg").read_bytes() == b"NEW"


def _fake_powershell(device_waypoint: Path, partial: bool = False):
    """The two PowerShell scripts of the MTP copy, played on a folder: fetch the slots' KMZ, copy the legs."""
    import json
    import shutil

    calls = []

    def run(script, environment=None, **kw):
        a = json.loads(environment["PPG_FLIGHTPLAN_ARGS"])
        calls.append(script)
        if script is djifly._PS_FETCH:
            sizes = {}
            for c in a["pairs"]:
                mission = device_waypoint / c["container"]
                if not mission.is_dir():
                    raise DjiFlyError(f"the device has no mission {c['container']}")
                if (mission / c["file"]).is_file():
                    sizes[c["file"]] = (mission / c["file"]).stat().st_size
                    data = (mission / c["file"]).read_bytes()
                    # the shell's copy still running when the script reads it: only part of the file
                    (Path(a["backupFolder"]) / c["file"]).write_bytes(data[:len(data) // 2] if partial else data)
            return "ok\n" + json.dumps(sizes)
        elif script is djifly._PS_COPY:
            for c in a["pairs"]:
                shutil.copy(Path(a["tempFolder"]) / c["file"], device_waypoint / c["container"] / c["file"])
        else:
            raise AssertionError("unknown script")
        return "ok"
    return run, calls


def test_the_mtp_copy_keeps_backups_refuses_missions_and_keeps_the_creation_time(dji_device, monkeypatch,
                                                                                  test_plan):
    waypoint, u1, backup = dji_device["waypoint"], dji_device["uuid1"], dji_device["backup"]
    run, calls = _fake_powershell(waypoint)
    monkeypatch.setattr(djifly, "_powershell", run)
    mine = _mission(7, "DJI Fly", 1_698_000_000_000)
    _add_slot(waypoint, UUID3, mine)
    path = "mtp://DJI RC 2/Internal shared storage"
    leg = dji.kmz_per_leg(test_plan)[0]
    with pytest.raises(djifly.SlotsInUse):
        djifly.copy_mtp(path, [leg, leg], [u1, UUID3], backup_dir=backup)
    assert calls == [djifly._PS_FETCH]                                  # nothing was sent to the device
    assert (waypoint / UUID3 / f"{UUID3}.kmz").read_bytes() == mine
    before = djifly.kmz_info((waypoint / u1 / f"{u1}.kmz").read_bytes())[2]
    old = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    outcome = djifly.copy_mtp(path, [leg], [u1], backup_dir=backup)[0]
    assert Path(outcome["copia"]).read_bytes() == old and outcome["stato"] == djifly.SLOT_OURS
    written = (waypoint / u1 / f"{u1}.kmz").read_bytes()
    assert djifly.kmz_info(written)[2] == before                       # the name in DJI Fly stays the same
    assert outcome["file"].startswith(path)
    djifly.copy_mtp(path, [leg], [u1], [1_700_000_000.0], backup_dir=backup)
    assert djifly.kmz_info((waypoint / u1 / f"{u1}.kmz").read_bytes())[2] == 1_700_000_000.0
    outcomes = djifly.copy_mtp(path, [leg, leg], [u1, UUID3], backup_dir=backup, overwrite=True)
    assert Path(outcomes[1]["copia"]).read_bytes() == mine
    with pytest.raises(DjiFlyError, match="non valido"):
        djifly.copy_mtp("mtp://senza-memoria", [leg], [u1], backup_dir=backup)


def test_a_backup_cut_short_by_the_device_never_lets_the_mission_be_replaced(dji_device, monkeypatch):
    """CopyHere is asynchronous: a backup shorter than the file on the device is not a copy, so the slot is not
    replaced - not even with overwrite=True - and the short file is not taken for a mission of the user."""
    waypoint, backup = dji_device["waypoint"], dji_device["backup"]
    mine = _mission(7, "DJI Fly", 1_698_000_000_000)
    _add_slot(waypoint, UUID3, mine)
    run, calls = _fake_powershell(waypoint, partial=True)
    monkeypatch.setattr(djifly, "_powershell", run)
    path = "mtp://DJI RC 2/Internal shared storage"
    for overwrite in (False, True):
        with pytest.raises(DjiFlyError, match="incompleta") as err:
            djifly.copy_mtp(path, [("a.kmz", b"NEW")], [UUID3], backup_dir=backup, overwrite=overwrite)
        assert not isinstance(err.value, djifly.SlotsInUse)
    assert calls == [djifly._PS_FETCH, djifly._PS_FETCH]                  # nothing ever sent to the device
    assert (waypoint / UUID3 / f"{UUID3}.kmz").read_bytes() == mine
    assert not list(backup.glob(f"{UUID3}-*"))                           # no truncated file passed off as a copy


def test_a_fetch_that_does_not_say_the_sizes_is_refused(dji_device, monkeypatch):
    waypoint, u1 = dji_device["waypoint"], dji_device["uuid1"]
    run, _ = _fake_powershell(waypoint)

    def without_sizes(script, environment=None, **kw):
        out = run(script, environment)
        return "ok" if script is djifly._PS_FETCH else out
    monkeypatch.setattr(djifly, "_powershell", without_sizes)
    with pytest.raises(DjiFlyError, match="incompleta"):
        djifly.copy_mtp("mtp://D/S", [("a.kmz", b"NEW")], [u1], backup_dir=dji_device["backup"], overwrite=True)



def test_a_write_that_fails_says_the_slot_its_copy_and_what_was_done(dji_device, monkeypatch):
    waypoint, u1, u2, backup = dji_device["waypoint"], dji_device["uuid1"], dji_device["uuid2"], dji_device["backup"]
    real = Path.write_bytes

    def failing(self, data):
        if self.name == f"{u2}.kmz" and self.parent.name == u2:
            raise OSError("device gone")
        return real(self, data)
    monkeypatch.setattr(Path, "write_bytes", failing)
    with pytest.raises(djifly.WriteFailed) as err:
        djifly.copy(waypoint, [("a.kmz", b"x"), ("b.kmz", b"y")], [u1, u2], backup_dir=backup)
    e = err.value
    assert e.slot == u2 and Path(e.backup).is_file() and str(e.backup) in str(e) and u2 in str(e)
    assert [o["contenitore"] for o in e.outcomes] == [u1]
    assert "device gone" in str(e)


def test_a_size_of_zero_from_the_device_is_not_trusted(dji_device, monkeypatch):
    """Some MTP shells report 0 bytes: an empty "copy" must never let the mission be replaced."""
    import json

    waypoint, backup = dji_device["waypoint"], dji_device["backup"]
    mine = _mission(7, "DJI Fly", 1_698_000_000_000)
    _add_slot(waypoint, UUID3, mine)
    run, calls = _fake_powershell(waypoint)

    def zero(script, environment=None, **kw):
        if script is djifly._PS_FETCH:
            a = json.loads(environment["PPG_FLIGHTPLAN_ARGS"])
            for c in a["pairs"]:
                (Path(a["backupFolder"]) / c["file"]).write_bytes(b"")
            return "ok\n" + json.dumps({c["file"]: 0 for c in a["pairs"]})
        return run(script, environment)
    monkeypatch.setattr(djifly, "_powershell", zero)
    with pytest.raises(DjiFlyError, match="incompleta"):
        djifly.copy_mtp("mtp://D/S", [("a.kmz", b"NEW")], [UUID3], backup_dir=backup, overwrite=True)
    assert calls == [] and (waypoint / UUID3 / f"{UUID3}.kmz").read_bytes() == mine
    assert not list(backup.glob(f"{UUID3}-*"))


def test_a_powershell_that_does_not_end_is_a_dji_fly_error(monkeypatch):
    import subprocess

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))
    monkeypatch.setattr(djifly.subprocess, "run", slow)
    with pytest.raises(DjiFlyError, match="PowerShell"):
        djifly._powershell(djifly._PS_FETCH, {"PPG_FLIGHTPLAN_ARGS": "{}"})
    assert djifly.FETCH_DEADLINE_S < djifly.POWERSHELL_TIMEOUT_S
    assert '$a.deadlineSeconds' in djifly._PS_FETCH and "ExtendedProperty('System.Size')" in djifly._PS_FETCH


def test_the_fetch_is_given_its_deadline(dji_device, monkeypatch):
    import json

    seen = []
    run, _ = _fake_powershell(dji_device["waypoint"])

    def spy(script, environment=None, **kw):
        seen.append((json.loads(environment["PPG_FLIGHTPLAN_ARGS"]), kw))
        return run(script, environment)
    monkeypatch.setattr(djifly, "_powershell", spy)
    djifly.copy_mtp("mtp://D/S", [("a.kmz", b"NEW")], [dji_device["uuid1"]], backup_dir=dji_device["backup"])
    assert seen[0][0]["deadlineSeconds"] == djifly.FETCH_DEADLINE_S
    assert seen[1][1]["timeout"] > 0                     # the copy script gets time for every leg
