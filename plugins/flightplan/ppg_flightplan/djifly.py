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


def copy(folder: Path, legs: list[tuple[str, bytes]], containers: list[str],
         previews: Optional[list[Optional[bytes]]] = None,
         created: Optional[list[Optional[float]]] = None) -> list[dict]:
    """Writes ``container/container.kmz`` for every leg, in order, and if there are previews also
    ``map_preview/container/container.jpg``, the image the library of DJI Fly shows for the mission. The
    containers must already be there: they are the placeholder missions saved by DJI Fly."""
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
    outcomes = []
    for (name, data), container in zip(legs, containers):
        dest = folder / container
        if not dest.is_dir():
            raise DjiFlyError(t("in {0} non c'è la missione {1}", "in {0} there is no mission {1}")
                              .format(folder, container))
        file = dest / f"{container}.kmz"
        # the name DJI Fly shows is the creation time of the placeholder: it is kept in the new KMZ
        time = created[len(outcomes)] if created and len(outcomes) < len(created) else None
        if time is None and file.is_file():
            try:
                time = kmz_info(file.read_bytes())[2]
            except OSError:
                time = None
        if time:
            data = with_create_time(data, time)
        # written over DJI Fly's file: on mounted MTP devices (gvfs) the rename does not always succeed, so no
        # temporary file
        file.write_bytes(data)
        # read back from the device: on mounted MTP paths the write can half succeed
        try:
            written = file.stat().st_size
        except OSError:
            written = -1
        outcome = {"tratta": name, "contenitore": container, "file": str(file), "byte": len(data),
                   "letti": written, "verificato": written == len(data)}
        preview = previews[len(outcomes)] if previews and len(outcomes) < len(previews) else None
        if preview:
            preview_folder = folder / PREVIEWS / container
            preview_folder.mkdir(parents=True, exist_ok=True)
            (preview_folder / f"{container}.jpg").write_bytes(preview)
            outcome["anteprima"] = str(preview_folder / f"{container}.jpg")
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


def _powershell(script: str, environment: Optional[dict] = None) -> str:
    env = dict(os.environ)
    if environment:
        env.update(environment)
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-Command", script], capture_output=True, text=True, env=env, timeout=180)
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


def copy_mtp(path: str, legs: list[tuple[str, bytes]], containers: list[str]) -> list[dict]:
    """Copies onto the folders of the Windows MTP device (``mtp://<device>/<storage>``)."""
    _check(legs, containers)
    rest = path[len(MTP_PREFIX):]
    if "/" not in rest:
        raise DjiFlyError(t("percorso MTP non valido: {0}", "MTP path not valid: {0}").format(path))
    device, storage = rest.split("/", 1)
    with tempfile.TemporaryDirectory(prefix="ppg-djifly-") as tmp:
        pairs = []
        for (name, data), container in zip(legs, containers):
            file = f"{container}.kmz"
            Path(tmp, file).write_bytes(data)
            pairs.append({"container": container, "file": file, "leg": name, "bytes": len(data)})
        arguments = json.dumps({"device": device, "storage": storage, "tempFolder": tmp, "pairs": pairs})
        _powershell(_PS_COPY, {"PPG_FLIGHTPLAN_ARGS": arguments})
    return [{"tratta": c["leg"], "contenitore": c["container"],
             "file": f"{path}/{SUBFOLDER.as_posix()}/{c['container']}/{c['file']}", "byte": c["bytes"]} for c in pairs]


def is_mtp(path: str) -> bool:
    return path.startswith(MTP_PREFIX)
