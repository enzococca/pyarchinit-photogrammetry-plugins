"""Spirula Studio run by the plugin: the executable found, the dataset laid out as Spirula reads a
COLMAP reconstruction, the training and the meshing as child processes with their progress.

Spirula Studio (https://github.com/harry7557558/spirula-studio, GPL-3) is one executable,
``spirula``: ``spirula train <preset> --data <dataset> [--flag value ...]`` trains the splats,
``spirula mesh <run>`` extracts a mesh from them, ``spirula <file.ply>`` opens the viewer. It
reads a COLMAP model (``sparse/0``, text or binary, every camera model: it undistorts by
itself) with the photos in ``images/`` and the masks in ``masks/<image name>.png``; it writes
the run into ``<prefix>/<name>/`` with ``config.json``, ``splat.ply`` and, after meshing, the
mesh files beside it. Every hundred steps it prints ``step N/TOTAL (P%)  splats S  [elapsed
... | ETA ...]`` (in English with ``--lang en``), ``Training complete`` at the end.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.progress import Cancelled, as_progress

from .texts import num, t

PRESETS = ("3dgs", "360-camera", "in-the-wild", "centered-object", "hdr", "synthetic", "meshing")
QUALITIES = ("low", "medium", "high", "ultra")
ITERATIONS = {"low": 15000, "medium": 30000, "high": 50000, "ultra": 80000}  # what Spirula's --quality sets
SPLAT_CAPS = {"low": 200_000, "medium": 1_000_000, "high": 3_000_000, "ultra": 10_000_000}
MESH_FORMATS = ("ply", "obj", "gltf", "glb", "stl")
MESH_COLOURS = ("none", "vertex", "texture")
MAX_SEED_POINTS = 2_000_000
STEP = re.compile(r"step\s+(\d+)\s*/\s*(\d+)\s*\(\s*(\d+)%\)")
SPLATS = re.compile(r"splats\s+(\d+)")
OUTPUT = re.compile(r"Output directory:\s*(.+)")
VIEWER = re.compile(r"[Vv]iewer at\s+(https?://[^\s)]+)")  # "Viewer at http://0.0.0.0:7007/ (forward the port ...)"
PERCENT = re.compile(r"(\d{1,3})%")
EXE = "spirula.exe" if sys.platform == "win32" else "spirula"


class SpirulaMissing(RuntimeError):
    """Spirula Studio was not found; the message says where to get it."""


# ---------------------------------------------------------------------------- the executable
def resolve(path) -> Path:
    """The executable from what the user gave: the ``spirula`` file, the macOS application bundle
    or the folder of the unzipped release."""
    p = Path(path).expanduser()
    if p.suffix == ".app":
        return p / "Contents" / "MacOS" / "spirula"
    if p.is_dir():
        return p / EXE
    return p


def candidates() -> list[Path]:
    """Where Spirula may be: ``PPG_SPIRULA``, the PATH, the usual folders of the installers."""
    out = []
    if os.environ.get("PPG_SPIRULA"):
        out.append(resolve(os.environ["PPG_SPIRULA"]))
    found = shutil.which("spirula")
    if found:
        out.append(Path(found))
    home = Path.home()
    if sys.platform == "darwin":
        out += [resolve("/Applications/Spirula Studio.app"), resolve(home / "Applications" / "Spirula Studio.app")]
    elif sys.platform == "win32":
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if base:
                out += [Path(base) / "Programs" / "Spirula Studio" / EXE, Path(base) / "Spirula Studio" / EXE,
                        Path(base) / "spirula-studio" / EXE]
    out += [home / ".local" / "share" / "spirula-studio" / EXE, home / "spirula-studio" / EXE,
            home / ".local" / "bin" / EXE, Path("/opt/spirula-studio") / EXE]
    return out


def find_spirula(setting: str | None = None) -> Path | None:
    """The ``spirula`` executable: the plugin setting first, then :func:`candidates`."""
    for p in ([resolve(setting)] if setting else []) + candidates():
        try:
            if p.is_file() and os.access(p, os.X_OK):
                return p
        except OSError:
            continue
    return None


def require(setting: str | None = None) -> Path:
    exe = find_spirula(setting)
    if exe is None:
        raise SpirulaMissing("Spirula Studio not found: `ppg plugin spirula install` downloads it (Plugin > Spirula > "
                             "Installa Spirula Studio in the window), or give the plugin its path (Plugin > Spirula > "
                             "Dove sta Spirula Studio, PPG_SPIRULA, `ppg plugin spirula where --set PATH`); releases: "
                             "https://github.com/harry7557558/spirula-studio/releases")
    return exe


# ---------------------------------------------------------------------------- the dataset
def check_aligned(chunk) -> None:
    """RuntimeError, with what to do, unless the photos of ``chunk`` are aligned: Spirula trains
    from the COLMAP model of the alignment and the poses of the photos."""
    model = chunk.product_path("reconstruction")
    if model is None or not Path(model).exists() or not chunk.aligned_cameras:
        raise RuntimeError(t("Le foto del chunk {0} non sono allineate: allineale prima (Flusso di lavoro > Allinea "
                             "foto, o ppg align). Spirula addestra dal modello dell'allineamento.",
                             "The photos of the chunk {0} are not aligned: align them first (Workflow > Align "
                             "photos, or ppg align). Spirula trains from the model of the alignment.")
                           .format(chunk.label))


def spirula_dir(chunk) -> Path:
    return chunk.data_dir / "spirula"


def dataset_dir(chunk) -> Path:
    return spirula_dir(chunk) / "dataset"


def runs_dir(chunk) -> Path:
    return spirula_dir(chunk) / "runs"


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_folder(target: Path, link: Path) -> str:
    """``link`` leading to ``target``: a symbolic link, a junction on Windows, else a copy of the
    folder (Spirula wants the photos inside the dataset). Returns how it was made."""
    _remove(link)
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return "junction"
        except OSError:
            pass
    shutil.copytree(target, link)
    return "copy"


def write_seed_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None) -> Path:
    """A binary PLY of points with colours, as Spirula seeds its splats from."""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = np.asarray(xyz, np.float32).T
    colours = np.asarray(rgb, np.uint8) if rgb is not None else np.full((n, 3), 128, np.uint8)
    arr["red"], arr["green"], arr["blue"] = colours.T
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}", "property float x", "property float y",
              "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(arr.tobytes())
    return path


def prepare_dataset(chunk, masks: bool = True, seed: str = "dense", max_seed_points: int = MAX_SEED_POINTS,
                    progress=None) -> dict:
    """The dataset Spirula trains from, in ``<chunk data>/spirula/dataset``: ``sparse/0`` with the
    COLMAP model of the chunk as text, ``images`` leading to the folder of the photos (the image
    names of the model are relative to it), ``masks`` laid out as COLMAP names them when the chunk
    has masks, ``seed.ply`` from the dense cloud (``seed="dense"``; ``"sparse"``: the tie points of
    the model) so that the splats start from it."""
    from pyarchinit_photogrammetry.engine.sfm import load_reconstruction, photo_folder

    progress = as_progress(progress)
    rec = load_reconstruction(chunk)
    folder = dataset_dir(chunk)
    _remove(folder)
    sparse = folder / "sparse" / "0"
    sparse.mkdir(parents=True)
    progress(0.1, t("Modello COLMAP", "COLMAP model"))
    rec.write_text(str(sparse))
    photos = Path(photo_folder(chunk))
    report = {"folder": folder, "images": int(rec.num_reg_images()), "photos": photos,
              "images_link": _link_folder(photos, folder / "images"), "masks": False, "seed": None, "seed_points": 0}
    if masks:
        from pyarchinit_photogrammetry.engine.masks import colmap_mask_folder

        names = {int(k): v for k, v in ((chunk.meta.get("sfm") or {}).get("names") or {}).items()}
        if names and colmap_mask_folder(chunk, names, folder / "masks") is not None:
            report["masks"] = True
    if seed == "dense":
        pc = chunk.point_cloud
        if pc is not None and len(pc):
            progress(0.5, t("Punti di partenza", "Seed points"))
            xyz, rgb = pc.xyz, pc.rgb
            if len(xyz) > max_seed_points:
                keep = np.sort(np.random.default_rng(0).choice(len(xyz), max_seed_points, replace=False))
                xyz, rgb = xyz[keep], (None if rgb is None else rgb[keep])
            report["seed"] = write_seed_ply(folder / "seed.ply", xyz, rgb)
            report["seed_points"] = int(len(xyz))
    progress(1.0, t("Dati per Spirula: {0} foto", "Dataset: {0} photos").format(report["images"]))
    return report


def suggested_divisor(chunk, target_megapixels: float = 4.0) -> int:
    """The resolution divisor that brings the photos to about ``target_megapixels`` (1 when they
    are already small): Spirula trains at the resolution of the files it is given."""
    sizes = [s.width * s.height for s in chunk.sensors.values()] or [0]
    mp = max(sizes) / 1e6
    return max(1, int(round(np.sqrt(mp / target_megapixels))))


# ---------------------------------------------------------------------------- the processes
def _popen(cmd):
    return subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, close_fds=False)


def _run(cmd, progress, on_line=None, log=None) -> tuple[int, list[str]]:
    """``cmd`` with its output read line by line (``on_line``, ``log``); the process is ended when
    the user cancels (:class:`Cancelled` out of ``progress``). Returns the exit code and the
    last lines."""
    proc = _popen(cmd)
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            lines.append(line)
            if len(lines) > 400:
                del lines[0]
            if log is not None:
                log.info("spirula: %s", line)
            try:
                if on_line is not None:
                    on_line(line)
                progress.check()
            except Cancelled:
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.wait()
    return int(proc.returncode or 0), lines


def train(chunk, spirula, preset: str = "3dgs", quality: str = "medium", iterations: int | None = None,
          divisor: float = 0.0, sh_degree: int = 3, seed: str = "dense", masks: bool = True, viewer: bool = False,
          port: int = 7007, open_browser: bool = True, name: str | None = None, extra=(), progress=None,
          log=None) -> dict:
    """Train the Gaussian splats of the chunk with Spirula: the dataset is laid out
    (:func:`prepare_dataset`), ``spirula train`` runs in ``<chunk data>/spirula/runs/<name>``
    with the preset and the quality (``iterations`` overrides the quality's), the photos reduced
    by ``divisor`` (0: their own size), ``sh_degree`` spherical harmonics, the web viewer of
    Spirula on ``port`` when ``viewer`` (opened in the browser as soon as Spirula announces it,
    unless ``open_browser`` is False; its URL is ``viewer`` in the result); ``extra``: more
    ``--flag value`` pairs. The splat becomes the product ``splat`` of the chunk, its run is in
    ``chunk.meta["splat"]``."""
    progress = as_progress(progress)
    check_aligned(chunk)
    exe = resolve(spirula)
    if not exe.is_file():
        raise SpirulaMissing(f"{exe}: not the spirula executable")
    if preset not in PRESETS:
        raise ValueError(f"preset {preset!r}: one of {', '.join(PRESETS)}")
    if quality not in QUALITIES:
        raise ValueError(f"quality {quality!r}: one of {', '.join(QUALITIES)}")
    ds = prepare_dataset(chunk, masks=masks, seed=seed, progress=progress.sub(0.0, 0.05))
    runs = runs_dir(chunk)
    runs.mkdir(parents=True, exist_ok=True)
    name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    cmd = [exe, "train", preset, "--data", ds["folder"], "--output-dir-prefix", runs, "--output-dir-name", name,
           "--quality", quality, "--sh-degree", int(sh_degree), "--disable-viewer", "0" if viewer else "1",
           "--keep-viewer-alive", "0"]
    if viewer:
        cmd += ["--viewer-port", int(port)]
    if iterations:
        cmd += ["--num-iterations", int(iterations)]
    if divisor and float(divisor) > 1:
        cmd += ["--train-resolution-divisor", float(divisor)]
    if ds["seed"] is not None:
        cmd += ["--seed-pointcloud", "seed.ply"]
    if not ds["masks"]:
        cmd += ["--load-masks", "0"]
    cmd += list(extra) + ["--lang", "en"]
    state = {"step": 0, "total": int(iterations or ITERATIONS[quality]), "splats": 0, "out": "", "viewer": ""}
    started = time.time()

    def on_line(line: str) -> None:
        m = VIEWER.search(line)
        if m and viewer and not state["viewer"]:
            state["viewer"] = url = _local_url(m.group(1))
            progress(0.05, t("Spirula: visualizzatore web su {0}", "Spirula: web viewer at {0}").format(url))
            if open_browser:
                open_url(url)
        m = STEP.search(line)
        if m:
            state["step"], state["total"] = int(m.group(1)), int(m.group(2))
            s = SPLATS.search(line)
            if s:
                state["splats"] = int(s.group(1))
            text = t("Spirula: passo {0} di {1}", "Spirula: step {0} of {1}").format(num(state["step"]), num(state["total"]))
            if state["splats"]:
                text += t(", {0} splat", ", {0} splats").format(num(state["splats"]))
            progress(0.05 + 0.93 * state["step"] / max(state["total"], 1), text)
        m = OUTPUT.search(line)
        if m:
            state["out"] = m.group(1).strip()

    progress(0.05, t("Spirula: {0} foto, preset {1}, qualita' {2}", "Spirula: {0} photos, preset {1}, quality {2}")
             .format(ds["images"], preset, quality))
    code, lines = _run(cmd, progress, on_line, log)
    if code != 0:
        raise RuntimeError(f"spirula train ended with code {code}:\n" + "\n".join(lines[-12:]))
    run_dir = Path(state["out"]) if state["out"] and Path(state["out"]).is_dir() else runs / name
    splat = run_dir / "splat.ply"
    if not splat.exists():
        found = sorted(run_dir.rglob("splat.ply"))
        if not found:
            raise RuntimeError(f"spirula wrote no splat.ply in {run_dir}:\n" + "\n".join(lines[-12:]))
        splat = found[-1]
    chunk._set_product("splat", splat)
    frame = scene_transform(run_dir)
    chunk.meta["splat"] = {
        "world_from_train": frame,  # the splats are trained in a frame of Spirula's: this 4 x 4 brings them back
        "file": chunk.products["splat"], "run": os.path.relpath(run_dir, chunk.data_dir), "preset": preset,
        "quality": quality, "iterations": state["total"], "steps": state["step"], "splats": state["splats"],
        "sh_degree": int(sh_degree), "resolution_divisor": float(divisor or 0), "seed": "dense" if ds["seed"] else "sparse",
        "seed_points": ds["seed_points"], "masks": ds["masks"], "photos": ds["images"],
        "created": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - started, 1),
        "spirula": str(exe), "command": " ".join(str(c) for c in cmd)}
    progress(1.0, t("Splat: {0} splat dopo {1} passi", "Splat: {0} splats after {1} steps")
             .format(num(state["splats"]), num(state["step"])) if state["splats"] else f"Splat: {splat.name}")
    return dict(chunk.meta["splat"], path=splat, run_dir=run_dir, viewer=state["viewer"] or None)


def _local_url(url: str) -> str:
    """The URL Spirula announces its web viewer at, as this computer reaches it: ``0.0.0.0``
    (every interface) becomes ``localhost``, the port stays."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme or "http", netloc, parts.path or "/", "", ""))


def browser_command(url: str) -> list[str] | None:
    """The command that opens ``url`` in the browser of this computer (``None`` on Windows,
    where :func:`os.startfile` does it, and where no opener is found)."""
    if sys.platform == "win32":
        return None
    if sys.platform == "darwin":
        return ["/usr/bin/open", url]
    opener = shutil.which("xdg-open")
    return [opener, url] if opener else None


def open_url(url: str) -> bool:
    """Open ``url`` in the browser, detached from the program and without a fork (the program
    may have SpatiaLite loaded); False when no browser can be started."""
    try:
        if sys.platform == "win32":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        cmd = browser_command(url)
        if cmd is None:
            return False
        subprocess.Popen(cmd, close_fds=False, start_new_session=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def scene_transform(run_dir: Path) -> list | None:
    """The 4 x 4 matrix (rows) that brings the splats of a run from the frame Spirula trained them
    in back into the frame of the COLMAP model (``scene_transform.json`` of the run: Spirula
    recentres and reorients the scene before training); ``None`` when the run has none."""
    path = Path(run_dir) / "scene_transform.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        matrix = data["world_from_train"]["matrix_4x4"]
        return [[float(v) for v in row] for row in matrix]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def keep_model_version(chunk, ply: Path, name: str, params: dict | None = None) -> dict:
    """A mesh kept among the versions of the model of the chunk (*Versioni* in the window), the
    active model untouched."""
    vid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = chunk.data_dir / "versions" / "model" / vid
    folder.mkdir(parents=True)
    dst = folder / "model.ply"
    shutil.copy2(ply, dst)
    entry = {"id": vid, "name": name, "created": datetime.now().isoformat(timespec="seconds"),
             "params": dict(params or {}), "files": {"model": str(dst.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault("model", []).append(entry)
    return entry


def mesh(chunk, spirula, formats=("ply",), colour: str = "vertex", texture_size: int = 0, run=None, progress=None,
         log=None) -> dict:
    """A mesh from the splats of the chunk with ``spirula mesh`` (the last run, or ``run``): the
    files are written beside ``splat.ply``; a PLY is also kept as a version of the model."""
    progress = as_progress(progress)
    exe = resolve(spirula)
    if not exe.is_file():
        raise SpirulaMissing(f"{exe}: not the spirula executable")
    formats = [f for f in formats if f in MESH_FORMATS]
    if not formats:
        raise ValueError(f"formats: among {', '.join(MESH_FORMATS)}")
    if colour not in MESH_COLOURS:
        raise ValueError(f"colour {colour!r}: one of {', '.join(MESH_COLOURS)}")
    info = chunk.meta.get("splat") or {}
    run_dir = Path(run) if run else (chunk.data_dir / info["run"] if info.get("run") else None)
    if run_dir is None or not run_dir.is_dir():
        raise RuntimeError("train the splats first (Plugin > Spirula > Addestra Gaussian splatting)")
    cmd = [exe, "mesh", run_dir, "--data", dataset_dir(chunk), "--output", run_dir / "mesh", "--format",
           ",".join(formats), "--color", colour, "--texture-size", int(texture_size), "--lang", "en"]
    before = {p for p in run_dir.rglob("mesh*") if p.is_file()}

    def on_line(line: str) -> None:
        m = PERCENT.search(line)
        if m:
            progress(min(0.95, int(m.group(1)) / 100), f"Spirula mesh: {line.strip()[:80]}")  # Spirula's own words

    progress(0.02, t("Spirula mesh: {0}, colore {1}", "Spirula mesh: {0}, colour {1}").format(", ".join(formats), colour))
    code, lines = _run(cmd, progress, on_line, log)
    if code != 0:
        raise RuntimeError(f"spirula mesh ended with code {code}:\n" + "\n".join(lines[-12:]))
    files = sorted(p for p in run_dir.rglob("mesh*")  # beside the splat, or in its checkpoint folder
                   if p.is_file() and (p not in before or p.stat().st_mtime >= time.time() - 3600))
    if not files:
        raise RuntimeError(f"spirula mesh wrote no mesh file in {run_dir}:\n" + "\n".join(lines[-12:]))
    by_name = {p.name: p for p in sorted(files, key=lambda p: len(p.parts), reverse=True)}  # the shallowest of a name
    files = sorted(by_name.values())
    chunk.meta.setdefault("splat", {})["mesh"] = [os.path.relpath(p, chunk.data_dir) for p in files]
    ply = next((p for p in files if p.suffix.lower() == ".ply"), None)
    version = None
    if ply is not None:
        version = keep_model_version(chunk, ply, f"Mesh dallo splat (Spirula, {run_dir.name})",
                                     {"source": "spirula", "run": run_dir.name, "colour": colour})
    progress(1.0, f"Mesh: {', '.join(p.name for p in files)}")
    return {"files": files, "version": version, "run_dir": run_dir}


def open_viewer(spirula, path) -> None:
    """Open ``path`` (a splat or a mesh) in Spirula Studio, detached from the program."""
    exe = resolve(spirula)
    kw = {"close_fds": False, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform != "win32":
        kw["start_new_session"] = True
    subprocess.Popen([str(exe), str(path)], **kw)
