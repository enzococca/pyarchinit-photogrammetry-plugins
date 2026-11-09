"""The KMZ inside DJI Fly: the waypoint folder on the remote controller or the phone, the placeholder missions
that act as containers, the copy with the right name.

DJI Fly does not import files: it keeps every mission in ``Android/data/dji.go.v5/files/waypoint/<uuid>/
<uuid>.kmz`` (RC 2, RC Pro and Android phones; on iPhone in ``DJI Fly/wayline_mission``, which cannot be reached
from the computer). A mission of any kind is created in DJI Fly and its file is replaced with the leg's,
renamed with the name of the folder. This module does that copy-paste-rename: the folder is found by itself
when the device is visible as a path (Linux through gvfs, an SD card, a pasted path), on Windows through the
shell for the MTP devices; where there is no path (a Mac with the RC 2) the ready folders to drag are made.

Ported from the author's Photo3D planner: paths, checks and the order of operations unchanged.
"""
from __future__ import annotations

import datetime
import glob
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from .texts import t

SUBFOLDER = Path("Android") / "data" / "dji.go.v5" / "files" / "waypoint"
IOS_FOLDER = "DJI Fly/wayline_mission"
MTP_PREFIX = "mtp://"
# The authors of a KMZ that is already ours: this plugin, and the planner it was ported from.
OUR_AUTHORS = ("pyarchinit-photogrammetry", "Photo3D")


@dataclass(frozen=True)
class Mission:
    """A mission saved by DJI Fly: the folder and its KMZ of the same name. ``created`` is the createTime of
    the template (seconds): DJI Fly does not let missions be renamed and calls them with that time, so
    ``dji_name`` is the name read in the library."""
    folder: str
    modified: float
    points: Optional[int] = None
    author: Optional[str] = None
    size: int = 0
    created: Optional[float] = None

    @property
    def dji_name(self) -> Optional[str]:
        if self.created is None:
            return None
        return datetime.datetime.fromtimestamp(self.created).strftime("%d/%m/%Y %H:%M:%S")

    @property
    def ours(self) -> bool:
        """Written by this plugin or by the planner it comes from (not a placeholder of DJI Fly)."""
        return self.author in OUR_AUTHORS

    def to_json(self) -> dict:
        """The planner's keys (``cartella``, ``nomeDji``...): a plan's ``djiFly`` entries keep them."""
        d = asdict(self)
        return {"cartella": d["folder"], "modificato": d["modified"], "punti": d["points"], "autore": d["author"],
                "dimensione": d["size"], "creata": d["created"], "nomeDji": self.dji_name, "nostra": self.ours}


class DjiFlyError(ValueError):
    pass


# ------------------------------------------------------------------ where the folder is

def _path_patterns() -> list[str]:
    s = str(SUBFOLDER)
    if sys.platform.startswith("linux"):
        return [f"/run/user/*/gvfs/mtp:host=*/*/{s}", f"/run/media/*/*/{s}", f"/media/*/*/{s}", f"/media/*/{s}"]
    if sys.platform == "darwin":
        return [f"/Volumes/*/{s}"]
    if sys.platform.startswith("win"):
        return [f"{letter}:/{s}" for letter in "DEFGHIJKLMNOPQRSTUVWXYZ"]
    return []


# The temporary copy iDroid (an MTP app for the Mac) keeps of the device's folders.
IDROID_COPIES = ["~/Library/Containers/com.leonboettger.idroid/Data/tmp/waypoint"]


def candidates() -> list[Path]:
    """The waypoint folders reachable as a path, without the user looking for them. On the Mac also the
    temporary copy of iDroid, good only to read the names of the missions."""
    found: list[Path] = []
    for pattern in _path_patterns():
        for p in sorted(glob.glob(pattern)):
            if Path(p).is_dir():
                found.append(Path(p))
    for p in IDROID_COPIES:
        folder = Path(os.path.expanduser(p))
        if folder.is_dir() and folder not in found:
            found.append(folder)
    return found


def normalise(path: str) -> Path:
    """Accepts the root of the device, ``Android``, ``.../files`` or the waypoint folder itself."""
    base = Path(os.path.expanduser(path.strip()))
    if not base.is_dir():
        raise DjiFlyError(t("la cartella non esiste: {0}", "the folder does not exist: {0}").format(base))
    if base.name == "waypoint":
        return base
    parts = SUBFOLDER.parts
    for k in range(len(parts)):
        candidate = base.joinpath(*parts[k:])
        if candidate.is_dir():
            return candidate
    raise DjiFlyError(t("sotto {0} non c'è la cartella dei waypoint di DJI Fly ({1})",
                        "under {0} there is no waypoint folder of DJI Fly ({1})").format(base, SUBFOLDER))


# ------------------------------------------------------------------ the missions inside

def kmz_points(data: bytes) -> tuple[Optional[int], Optional[str]]:
    """How many waypoints the KMZ has, and who wrote it (wpml:author): one of :data:`OUR_AUTHORS` if it is
    already ours."""
    points, author, _ = kmz_info(data)
    return points, author


def kmz_info(data: bytes) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """Waypoints, author and creation time (seconds, from the createTime in milliseconds) of the KMZ."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            lines = next((n for n in names if n.endswith("waylines.wpml")), None)
            model = next((n for n in names if n.endswith("template.kml")), None)
            points = None
            if lines:
                root = ET.fromstring(z.read(lines))
                points = sum(1 for e in root.iter() if e.tag.endswith("}Placemark") or e.tag == "Placemark")
            author, created = None, None
            if model:
                root = ET.fromstring(z.read(model))
                el = next((e for e in root.iter() if e.tag.endswith("}author")), None)
                author = (el.text or "").strip() if el is not None else None
                el = next((e for e in root.iter() if e.tag.endswith("}createTime")), None)
                try:
                    created = int((el.text or "").strip()) / 1000.0 if el is not None else None
                except ValueError:
                    created = None
            return points, author, created
    except (zipfile.BadZipFile, ET.ParseError, KeyError, OSError):
        return None, None, None


def with_create_time(data: bytes, created: float) -> bytes:
    """The same KMZ with ``createTime`` and ``updateTime`` (milliseconds) set to the time ``created`` (seconds):
    written in place of a placeholder mission, DJI Fly keeps calling it with the time it was created at, and
    the name stays the one matched with the leg."""
    ms = str(int(round(created * 1000)))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries = [(i, z.read(i.filename)) for i in z.infolist()]
    except zipfile.BadZipFile:
        return data
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for info, content in entries:
            if info.filename.endswith((".kml", ".wpml")):
                text = content.decode("utf-8", "replace")
                text = re.sub(r"(<wpml:createTime>)\d+(</wpml:createTime>)", rf"\g<1>{ms}\g<2>", text)
                text = re.sub(r"(<wpml:updateTime>)\d+(</wpml:updateTime>)", rf"\g<1>{ms}\g<2>", text)
                content = text.encode("utf-8")
            z.writestr(info.filename, content)
    return buffer.getvalue()


def missions(folder: Path) -> list[Mission]:
    """The missions saved in a waypoint folder, in the order they were created."""
    found: list[Mission] = []
    for sub in Path(folder).iterdir():
        if not sub.is_dir():
            continue
        kmz = sub / f"{sub.name}.kmz"
        if not kmz.is_file():
            continue
        st = kmz.stat()
        points, author, created = kmz_info(kmz.read_bytes())
        found.append(Mission(sub.name, st.st_mtime, points, author, st.st_size, created))
    # in the order they were created: so the first placeholder takes the first leg
    found.sort(key=lambda m: (m.created if m.created is not None else m.modified))
    return found


# ------------------------------------------------------------------ the copy

def _check(legs: list[tuple[str, bytes]], containers: list[str]) -> None:
    if not legs:
        raise DjiFlyError(t("nessuna tratta da copiare: il piano non ha blocchi accesi",
                            "no leg to copy: the plan has no active block"))
    if len(containers) < len(legs):
        raise DjiFlyError(t("servono {0} contenitori, ne hai scelti {1}: crea in DJI Fly una missione fittizia per "
                            "ogni tratta",
                            "{0} containers are needed, you chose {1}: create a placeholder mission in DJI Fly for "
                            "every leg").format(len(legs), len(containers)))
    if len(set(containers)) != len(containers):
        raise DjiFlyError(t("lo stesso contenitore è scelto per due tratte",
                            "the same container is chosen for two legs"))
    for c in containers:
        if not c or "/" in c or "\\" in c or c in (".", ".."):
            raise DjiFlyError(t("nome di contenitore non valido: {0!r}", "container name not valid: {0!r}").format(c))


LOCAL_COPIES = ("/Library/Containers/com.leonboettger.idroid/", "/Library/Containers/", "/Library/Caches/")
ONLY_VOLUMES = sys.platform == "darwin"   # on the Mac a device or a card are only under /Volumes


def is_local_copy(folder: Path) -> bool:
    """The MTP apps of the Mac (iDroid and the like) keep a temporary copy of the device's folders inside their
    container, and the user can download the waypoint folder onto the disk: writing there does not touch the
    remote controller. On the Mac a device or a card are only under /Volumes: all the rest is a local copy."""
    text = str(folder)
    if any(piece in text for piece in LOCAL_COPIES) and "/tmp/" in text:
        return True
    return ONLY_VOLUMES and not text.startswith("/Volumes/")


PREVIEWS = "map_preview"

# What a slot of DJI Fly holds, before a leg is written into it (codes stored and shown by the pane):
SLOT_EMPTY = "vuota"            # the mission folder without its KMZ
SLOT_PLACEHOLDER = "fittizia"   # a placeholder saved in DJI Fly for this purpose: two waypoints at most
SLOT_OURS = "nostra"            # a leg written before by this plugin or by the planner it comes from
SLOT_MISSION = "missione"       # anything else: a mission of the user (or a file that cannot be read)
SLOT_OTHER_PLAN = "altro_piano"  # a leg written from here (or by the planner) for another plan: asked before
SAFE_SLOTS = (SLOT_EMPTY, SLOT_PLACEHOLDER, SLOT_OURS)


def _for_this_plan(state: dict, container: str, own) -> dict:
    """A slot written from here is "ours" only when this plan wrote it (``own``: the containers of the plan's
    ``djiFly`` entries); a leg of another plan or site is asked before it is replaced. ``own`` None: as before,
    every leg written from here is ours. Two waypoints at most are a placeholder whoever wrote them."""
    small = state.get("punti") is not None and state["punti"] <= PLACEHOLDER_MAX_POINTS
    if own is not None and state.get("stato") == SLOT_OURS and container not in own and not small:
        return dict(state, stato=SLOT_OTHER_PLAN)
    return state
PLACEHOLDER_MAX_POINTS = 2


def slot_state(data: Optional[bytes]) -> dict:
    """What a slot holds, from the bytes of its KMZ (None: no file): ``stato`` (one of the ``SLOT_*`` codes),
    ``punti``, ``autore``, ``creata`` and ``nomeDji``."""
    if data is None:
        return {"stato": SLOT_EMPTY, "punti": None, "autore": None, "creata": None, "nomeDji": None}
    points, author, created = kmz_info(data)
    if author in OUR_AUTHORS:
        state = SLOT_OURS
    elif points is not None and points <= PLACEHOLDER_MAX_POINTS:
        state = SLOT_PLACEHOLDER
    else:
        state = SLOT_MISSION
    name = Mission("", 0.0, created=created).dji_name
    return {"stato": state, "punti": points, "autore": author, "creata": created, "nomeDji": name}


def slot_states(folder: Path, containers: list[str], own=None) -> list[dict]:
    """What every chosen slot of a waypoint folder holds (``contenitore`` + :func:`slot_state`), for the pane to
    show before copying; DjiFlyError if a slot does not exist. With ``own`` (the containers this plan wrote) a leg
    written from here for another plan is :data:`SLOT_OTHER_PLAN`."""
    out = []
    for container in containers:
        dest = Path(folder) / container
        if not dest.is_dir():
            raise DjiFlyError(t("in {0} non c'è la missione {1}", "in {0} there is no mission {1}")
                              .format(folder, container))
        file = dest / f"{container}.kmz"
        state = {"contenitore": container, **slot_state(file.read_bytes() if file.is_file() else None)}
        out.append(_for_this_plan(state, container, own))
    return out


class SlotsInUse(DjiFlyError):
    """Some chosen slots hold a mission of the user: nothing was written. ``slots`` says which (the entries of
    :func:`slot_states`); the copy goes on only with ``overwrite`` naming them (or True), after the user confirmed."""

    def __init__(self, slots: list[dict]):
        self.slots = slots
        names = ", ".join(s.get("nomeDji") or s["contenitore"] for s in slots)
        super().__init__(t("le missioni {0} di DJI Fly non sono fittizie né tratte di questo piano: non le ho toccate. "
                           "Se vanno davvero sostituite, conferma la sovrascrittura (una copia resta comunque).",
                           "the DJI Fly missions {0} are neither placeholders nor legs of this plan: I left them "
                           "alone. If they really are to be replaced, confirm the overwrite (a copy is kept anyway).")
                         .format(names))


def not_agreed(busy: list[dict], overwrite) -> list[dict]:
    """The slots of ``busy`` (missions of the user) that may not be overwritten: all with ``overwrite`` False, none
    with True, else those whose container is not in ``overwrite`` (the containers the user saw listed as missions
    and agreed to overwrite: a slot that became a mission after the question is not among them)."""
    if overwrite is True:
        return []
    if not overwrite:
        return list(busy)
    if isinstance(overwrite, (str, bytes)):
        raise TypeError("overwrite: True, False or a collection of container names, not one string")
    agreed = {str(c) for c in overwrite}
    return [s for s in busy if s["contenitore"] not in agreed]


class WriteFailed(DjiFlyError):
    """Writing into a slot failed halfway (a device gone, a full card): ``slot``, the copy of what was there
    (``backup``, None if the slot was empty) and the ``outcomes`` of the slots done before, for the pane to say
    what to put back and where it is."""

    def __init__(self, slot: str, backup, outcomes: list[dict], cause: Exception):
        self.slot, self.backup, self.outcomes = slot, backup, list(outcomes)
        where = (t("la copia di quel che c'era è in {0}", "the copy of what was there is in {0}").format(backup)
                 if backup else t("lo slot era vuoto", "the slot was empty"))
        super().__init__(t("scrittura della missione {0} non riuscita ({1}): {2}; {3} tratte scritte prima",
                           "writing mission {0} failed ({1}): {2}; {3} legs written before")
                         .format(slot, cause, where, len(self.outcomes)))


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _backup_folder(backup_dir, waypoint_folder: Optional[Path] = None) -> Path:
    """The folder of the copies of what is replaced: required, outside the waypoint folder (DJI Fly scans every
    folder there)."""
    if backup_dir is None:
        raise DjiFlyError(t("serve una cartella per le copie delle missioni sostituite",
                            "a folder for the copies of the replaced missions is needed"))
    backup = Path(backup_dir).expanduser().resolve()
    if waypoint_folder is not None:
        root = Path(waypoint_folder).expanduser().resolve()
        if backup == root or root in backup.parents:
            raise DjiFlyError(t("la cartella delle copie non può stare dentro la cartella dei waypoint di DJI Fly",
                                "the folder of the copies cannot be inside the waypoint folder of DJI Fly"))
    backup.mkdir(parents=True, exist_ok=True)
    return backup


def _keep(backup: Path, container: str, data: bytes, suffix: str, stamp: str) -> Path:
    """Writes the copy ``<container>-<date>.<suffix>`` and reads it back: no copy, no overwrite."""
    target = backup / f"{container}-{stamp}.{suffix}"
    k = 2
    while target.exists():
        target = backup / f"{container}-{stamp}-{k}.{suffix}"
        k += 1
    target.write_bytes(data)
    if target.read_bytes() != data:
        raise DjiFlyError(t("la copia di {0} non è riuscita: non sostituisco la missione",
                            "the copy of {0} did not succeed: the mission is not replaced").format(container))
    return target


def copy(folder: Path, legs: list[tuple[str, bytes]], containers: list[str],
         previews: Optional[list[Optional[bytes]]] = None,
         created: Optional[list[Optional[float]]] = None, *, backup_dir=None,
         overwrite=False, own=None) -> list[dict]:
    """Writes ``container/container.kmz`` for every leg, in order, and if there are previews also
    ``map_preview/container/container.jpg``, the image the library of DJI Fly shows for the mission. The
    containers must already be there: they are the placeholder missions saved by DJI Fly.

    Nothing is ever lost: before a file is replaced it is copied into ``backup_dir`` (``<container>-<date>.kmz``
    and ``.jpg``; required, outside the waypoint folder), and the outcome says where (``copia``). A slot that
    holds a mission of the user (neither a placeholder nor a leg written from here) stops the whole copy before
    anything is written (:class:`SlotsInUse`), unless ``overwrite`` is True or names its container (a collection of
    the containers the user agreed to overwrite: :func:`not_agreed`); with ``own`` (the containers this plan wrote
    before) a leg written from here for another plan stops it too."""
    _check(legs, containers)
    folder = Path(folder)
    if is_local_copy(folder):
        raise DjiFlyError(t("{0} è una copia sul Mac (iDroid o una cartella scaricata), non il telecomando: scrivere "
                            "qui non cambia nulla sul dispositivo e i file scaricati si mischiano. Usa «Scarica le "
                            "cartelle pronte» e trascina le cartelle nel dispositivo con iDroid o OpenMTP.",
                            "{0} is a copy on the Mac (iDroid or a downloaded folder), not the remote controller: "
                            "writing here changes nothing on the device and the downloaded files get mixed. Use "
                            "'Download the ready folders' and drag the folders into the device with iDroid or "
                            "OpenMTP.").format(folder))
    backup = _backup_folder(backup_dir, folder)
    states = slot_states(folder, containers[:len(legs)], own)
    busy = [s for s in states if s["stato"] not in SAFE_SLOTS]
    refused = not_agreed(busy, overwrite)
    if refused:
        raise SlotsInUse(refused)
    stamp = _stamp()
    outcomes = []
    for (name, data), container, state in zip(legs, containers, states):
        dest = folder / container
        file = dest / f"{container}.kmz"
        kept = None
        if file.is_file():
            kept = _keep(backup, container, file.read_bytes(), "kmz", stamp)
        # the name DJI Fly shows is the creation time of the placeholder: it is kept in the new KMZ
        time = created[len(outcomes)] if created and len(outcomes) < len(created) else None
        if time is None:
            time = state["creata"]
        if time:
            data = with_create_time(data, time)
        # written over DJI Fly's file: on mounted MTP devices (gvfs) the rename does not always succeed, so no
        # temporary file
        try:
            file.write_bytes(data)
        except OSError as e:
            raise WriteFailed(container, kept, outcomes, e) from e
        # read back from the device: on mounted MTP paths the write can half succeed
        try:
            written = file.stat().st_size
        except OSError:
            written = -1
        outcome = {"tratta": name, "contenitore": container, "file": str(file), "byte": len(data),
                   "letti": written, "verificato": written == len(data), "stato": state["stato"],
                   "copia": str(kept) if kept else None}
        preview = previews[len(outcomes)] if previews and len(outcomes) < len(previews) else None
        if preview:
            preview_folder = folder / PREVIEWS / container
            preview_file = preview_folder / f"{container}.jpg"
            if preview_file.is_file():
                outcome["copiaAnteprima"] = str(_keep(backup, container, preview_file.read_bytes(), "jpg", stamp))
            try:
                preview_folder.mkdir(parents=True, exist_ok=True)
                preview_file.write_bytes(preview)
            except OSError as e:
                raise WriteFailed(container, kept, outcomes + [outcome], e) from e
            outcome["anteprima"] = str(preview_file)
        outcomes.append(outcome)
    return outcomes


README = ("Trascina le cartelle delle missioni in Android/data/dji.go.v5/files/waypoint del telecomando o del "
          "telefono, sostituendo quelle con lo stesso nome (le missioni fittizie salvate in DJI Fly); le cartelle "
          "dentro map_preview vanno in waypoint/map_preview, sempre sostituendo: sono le anteprime che la libreria "
          "di DJI Fly mostra, col numero e il nome della tratta.\nDrag the mission folders into "
          "Android/data/dji.go.v5/files/waypoint on the controller or phone, replacing the ones with the same name "
          "(the placeholder missions saved in DJI Fly); the folders inside map_preview go into waypoint/map_preview, "
          "again replacing: they are the previews DJI Fly's library shows, with the leg number and name.\n")


def zip_containers(legs: list[tuple[str, bytes]], containers: list[str],
                   previews: Optional[list[Optional[bytes]]] = None,
                   created: Optional[list[Optional[float]]] = None) -> bytes:
    """The ready folders, ``<container>/<container>.kmz`` and ``map_preview/<container>/<container>.jpg``, to
    drag into the waypoint folder with iDroid, OpenMTP or Android File Transfer when the device is not seen as
    a path."""
    _check(legs, containers)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for k, ((name, data), container) in enumerate(zip(legs, containers)):
            time = created[k] if created and k < len(created) else None
            z.writestr(f"{container}/{container}.kmz", with_create_time(data, time) if time else data)
            if previews and k < len(previews) and previews[k]:
                z.writestr(f"{PREVIEWS}/{container}/{container}.jpg", previews[k])
        z.writestr("LEGGIMI.txt", README)
    return buffer.getvalue()


# ------------------------------------------------------------------ Windows: the MTP devices through the shell

_PS_FIND = r'''
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject Shell.Application
$out = New-Object System.Collections.ArrayList
foreach ($dev in $shell.NameSpace(17).Items()) {
  if (-not $dev.IsFolder) { continue }
  $f = $null; try { $f = $dev.GetFolder } catch { continue }
  if ($null -eq $f) { continue }
  foreach ($store in $f.Items()) {
    if (-not $store.IsFolder) { continue }
    $wp = $null
    try {
      $cur = $store
      foreach ($name in @('Android', 'data', 'dji.go.v5', 'files', 'waypoint')) {
        $cur = $cur.GetFolder.ParseName($name)
        if ($null -eq $cur) { break }
      }
      $wp = $cur
    } catch { $wp = $null }
    if ($null -eq $wp) { continue }
    $missions = New-Object System.Collections.ArrayList
    foreach ($m in $wp.GetFolder.Items()) {
      if (-not $m.IsFolder) { continue }
      $k = $m.GetFolder.ParseName($m.Name + '.kmz')
      if ($null -ne $k) { [void]$missions.Add(@{ folder = $m.Name; modified = $k.ModifyDate.ToString('o'); size = $k.Size }) }
    }
    [void]$out.Add(@{ device = $dev.Name; storage = $store.Name; missions = $missions })
  }
}
ConvertTo-Json -InputObject $out -Depth 6 -Compress
'''

_PS_FETCH = r'''
$ErrorActionPreference = 'Stop'
$a = $env:PPG_FLIGHTPLAN_ARGS | ConvertFrom-Json
$shell = New-Object -ComObject Shell.Application
$wp = $null
foreach ($dev in $shell.NameSpace(17).Items()) {
  if (-not $dev.IsFolder -or $dev.Name -ne $a.device) { continue }
  foreach ($store in $dev.GetFolder.Items()) {
    if (-not $store.IsFolder -or $store.Name -ne $a.storage) { continue }
    $cur = $store
    foreach ($name in @('Android', 'data', 'dji.go.v5', 'files', 'waypoint')) {
      $cur = $cur.GetFolder.ParseName($name); if ($null -eq $cur) { break }
    }
    $wp = $cur
  }
}
if ($null -eq $wp) { throw "device $($a.device) / $($a.storage) not found, or without the waypoint folder" }
$dst = $shell.NameSpace($a.backupFolder)
$sizes = @{}
# one deadline for the whole fetch, shorter than the time the program gives the script
$deadline = (Get-Date).AddSeconds([int]$a.deadlineSeconds)
foreach ($c in $a.pairs) {
  $mission = $wp.GetFolder.ParseName($c.container)
  if ($null -eq $mission) { throw "the device has no mission $($c.container)" }
  $item = $mission.GetFolder.ParseName($c.file)
  if ($null -eq $item) { continue }
  # System.Size is the reliable size on WPD/MTP namespaces; FolderItem.Size (32 bits) may say 0
  $size = $null
  try { $size = $item.ExtendedProperty('System.Size') } catch { $size = $null }
  if ($null -eq $size -or [long]$size -le 0) { $size = $item.Size }
  $size = [long]$size
  $sizes[$c.file] = $size
  # the mission's KMZ copied to the computer before anything is replaced. CopyHere is asynchronous and the file
  # appears as soon as the copy starts: wait until it has the size it has on the device and the shell let it go
  $dst.CopyHere($item, 16)
  $local = Join-Path $a.backupFolder $c.file
  $done = $false
  while (-not $done -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    try {
      if ($size -gt 0 -and (Test-Path -LiteralPath $local) -and (Get-Item -LiteralPath $local).Length -eq $size) {
        $fs = [System.IO.File]::Open($local, 'Open', 'Read', 'None'); $fs.Close(); $done = $true
      }
    } catch { }
  }
  if (-not $done) { throw "the copy of $($c.file) from the device did not end" }
}
'ok'
ConvertTo-Json -InputObject $sizes -Compress
'''

_PS_COPY = r'''
$ErrorActionPreference = 'Stop'
$a = $env:PPG_FLIGHTPLAN_ARGS | ConvertFrom-Json
$shell = New-Object -ComObject Shell.Application
$wp = $null
foreach ($dev in $shell.NameSpace(17).Items()) {
  if (-not $dev.IsFolder -or $dev.Name -ne $a.device) { continue }
  foreach ($store in $dev.GetFolder.Items()) {
    if (-not $store.IsFolder -or $store.Name -ne $a.storage) { continue }
    $cur = $store
    foreach ($name in @('Android', 'data', 'dji.go.v5', 'files', 'waypoint')) { $cur = $cur.GetFolder.ParseName($name); if ($null -eq $cur) { break } }
    $wp = $cur
  }
}
if ($null -eq $wp) { throw "device $($a.device) / $($a.storage) not found, or without the waypoint folder" }
$src = $shell.NameSpace($a.tempFolder)
foreach ($c in $a.pairs) {
  $dest = $wp.GetFolder.ParseName($c.container)
  if ($null -eq $dest) { throw "the device has no mission $($c.container)" }
  $item = $src.ParseName($c.file)
  # 16 = yes to all: the file with the same name is replaced without questions
  $dest.GetFolder.CopyHere($item, 16)
  $t = 0
  do { Start-Sleep -Milliseconds 500; $t += 1; $ok = $dest.GetFolder.ParseName($c.file) } while ($null -eq $ok -and $t -lt 120)
  if ($null -eq $ok) { throw "the copy of $($c.file) did not succeed" }
}
'ok'
'''


# The fetch from the device ends within FETCH_DEADLINE_S (all the slots together); the program waits for any
# PowerShell script POWERSHELL_TIMEOUT_S at least.
FETCH_DEADLINE_S = 150
POWERSHELL_TIMEOUT_S = 180
COPY_SECONDS_PER_LEG = 60          # _PS_COPY waits up to a minute for every file to appear on the device


def _powershell(script: str, environment: Optional[dict] = None, timeout: float = POWERSHELL_TIMEOUT_S) -> str:
    env = dict(os.environ)
    if environment:
        env.update(environment)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-Command", script], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DjiFlyError(t("PowerShell non ha finito in {0:.0f} s: il dispositivo non risponde? Nulla è stato "
                            "sostituito se la copia dal dispositivo non era finita",
                            "PowerShell did not finish in {0:.0f} s: is the device answering? Nothing was replaced "
                            "if the copy from the device had not ended").format(timeout)) from None
    if r.returncode != 0:
        raise DjiFlyError((r.stderr or r.stdout).strip()
                          or t("PowerShell non ha risposto", "PowerShell did not answer"))
    return r.stdout


def mtp_devices() -> list[dict]:
    """On Windows: remote controllers and phones connected by USB that have the waypoint folder. Every entry has
    ``percorso`` in the form ``mtp://<device>/<storage>``, to pass to :func:`copy_mtp`."""
    if not sys.platform.startswith("win"):
        return []
    try:
        text = _powershell(_PS_FIND).strip()
    except (OSError, subprocess.SubprocessError, DjiFlyError):
        return []
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    entries = []
    for d in data:
        found = d.get("missions") or []
        if isinstance(found, dict):
            found = [found]
        entries.append({"percorso": f"{MTP_PREFIX}{d['device']}/{d['storage']}",
                        "dispositivo": d["device"], "memoria": d["storage"],
                        "missioni": [{"cartella": m["folder"], "modificato": m.get("modified"),
                                      "dimensione": m.get("size", 0), "punti": None, "autore": None}
                                     for m in found]})
    return entries


def _device_of(path: str) -> tuple[str, str]:
    rest = path[len(MTP_PREFIX):]
    if "/" not in rest:
        raise DjiFlyError(t("percorso MTP non valido: {0}", "MTP path not valid: {0}").format(path))
    device, storage = rest.split("/", 1)
    return device, storage


def _sizes_of(answer) -> Optional[dict]:
    """The sizes on the device of the files fetched (the last line of the fetch script: a JSON object), or None
    when the script did not say them."""
    for line in reversed(str(answer or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                found = json.loads(line)
            except ValueError:
                return None
            return {str(k): int(v) for k, v in found.items()} if isinstance(found, dict) else None
    return None


def copy_mtp(path: str, legs: list[tuple[str, bytes]], containers: list[str],
             created: Optional[list[Optional[float]]] = None, *, backup_dir=None,
             overwrite=False, own=None) -> list[dict]:
    """Copies onto the folders of the Windows MTP device (``mtp://<device>/<storage>``), as :func:`copy` does:
    the KMZ of every chosen slot is first brought to the computer, into ``backup_dir`` (``<container>-<date>.kmz``),
    and read there - so a mission of the user stops the copy (:class:`SlotsInUse`, unless ``overwrite`` allows it,
    :func:`not_agreed`) and the
    placeholder's creation time goes into the new KMZ (the name in DJI Fly stays the one matched with the leg)."""
    _check(legs, containers)
    device, storage = _device_of(path)
    backup = _backup_folder(backup_dir)
    stamp = _stamp()
    with tempfile.TemporaryDirectory(prefix="ppg-djifly-") as tmp:
        fetched = Path(tmp) / "dal-dispositivo"
        fetched.mkdir()
        files = [{"container": c, "file": f"{c}.kmz"} for c in containers[:len(legs)]]
        answer = _powershell(_PS_FETCH, {"PPG_FLIGHTPLAN_ARGS": json.dumps(
            {"device": device, "storage": storage, "backupFolder": str(fetched), "pairs": files,
             "deadlineSeconds": FETCH_DEADLINE_S})})
        sizes = _sizes_of(answer)
        fetched_data = {}
        for c in containers[:len(legs)]:
            old = fetched / f"{c}.kmz"
            data = old.read_bytes() if old.is_file() else None
            # the script names a file only when the slot has one on the device; a size of 0 says nothing
            on_device = sizes is None or f"{c}.kmz" in sizes or data is not None
            expected = sizes.get(f"{c}.kmz") if sizes is not None else None
            if expected is not None and expected <= 0:
                expected = None
            # a copy that is not whole is no copy: never replace the slot (overwrite or not), never call it a mission
            if on_device and (data is None or expected is None or len(data) != expected):
                raise DjiFlyError(t("la copia della missione {0} portata dal dispositivo è incompleta "
                                    "({1} byte di {2}): non sostituisco nulla, riprova",
                                    "the copy of mission {0} brought from the device is incomplete "
                                    "({1} bytes of {2}): nothing is replaced, try again")
                                  .format(c, 0 if data is None else len(data), expected or "?"))
            fetched_data[c] = data
        states, kept = [], []
        for c in containers[:len(legs)]:
            data = fetched_data[c]
            states.append(_for_this_plan({"contenitore": c, **slot_state(data)}, c, own))
            kept.append(_keep(backup, c, data, "kmz", stamp) if data is not None else None)
        busy = [st for st in states if st["stato"] not in SAFE_SLOTS]
        refused = not_agreed(busy, overwrite)
        if refused:
            raise SlotsInUse(refused)
        outgoing = Path(tmp) / "verso-il-dispositivo"
        outgoing.mkdir()
        pairs = []
        for k, ((name, data), container) in enumerate(zip(legs, containers)):
            time = created[k] if created and k < len(created) else None
            if time is None:
                time = states[k]["creata"]
            if time:
                data = with_create_time(data, time)
            file = f"{container}.kmz"
            (outgoing / file).write_bytes(data)
            pairs.append({"container": container, "file": file, "leg": name, "bytes": len(data)})
        arguments = json.dumps({"device": device, "storage": storage, "tempFolder": str(outgoing), "pairs": pairs})
        _powershell(_PS_COPY, {"PPG_FLIGHTPLAN_ARGS": arguments},
                    timeout=max(POWERSHELL_TIMEOUT_S, COPY_SECONDS_PER_LEG * len(pairs) + 60))
    return [{"tratta": c["leg"], "contenitore": c["container"],
             "file": f"{path}/{SUBFOLDER.as_posix()}/{c['container']}/{c['file']}", "byte": c["bytes"],
             "stato": st["stato"], "copia": str(k) if k else None}
            for c, st, k in zip(pairs, states, kept)]


def is_mtp(path: str) -> bool:
    return path.startswith(MTP_PREFIX)
