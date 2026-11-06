"""The steps of an ERT profile that need the worker or write its products: import (pyGIMLi's readers
for the files of the instruments), synthetic readings, the inversion and its section, the exports.
No window here: the plugin's interface and its command line call these."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from . import ert as ert_mod
from . import ert_environment as environment
from . import ert_runner as runner
from . import ert_section as section
from .texts import t

# another worker (the tests): its Python and script instead of pyGIMLi's environment
WORKER: Path | None = None
WORKER_PYTHON: Path | None = None
DEFAULT_PARAMS = {"lam": 20.0, "rel_error": 0.03, "abs_error_u": None, "use_file_errors": True, "robust": False,
                  "blocky": False, "z_weight": 1.0, "max_iter": 20, "para_depth": None, "para_dx": 0.33,
                  "para_max_cell": None, "quality": 33.0, "doi": "coverage", "topography_k": True}


def _worker():
    if WORKER is not None:
        return Path(WORKER_PYTHON or sys.executable), Path(WORKER)
    return environment.worker_python(), None


def work_folder(chunk, pid: str | None = None) -> Path:
    base = Path(chunk.data_dir) / ert_mod.FOLDER
    return (base / pid / "work") if pid else (base / "_work")


def run(job: dict, folder: Path, progress=None, timeout: float | None = None) -> tuple[dict, dict]:
    python, script = _worker()
    res = runner.run(job, folder, progress, python=python, script=script, timeout=timeout)
    return res, runner.load(res)


# ------------------------------------------------------------------ import
def import_profile(chunk, path, name: str = "", spacing: float | None = None, positions: bool | None = None,
                   electrodes=None, progress=None) -> dict:
    """A file of a resistivity meter into the chunk as a new profile: the unified data format and
    tables of readings read here, the other formats of the instruments by pyGIMLi in its worker."""
    path = Path(path)
    try:
        prof = ert_mod.read_any(path, spacing, positions, electrodes)
    except ert_mod.NeedsGimli as first:
        if progress is not None:
            progress(0.1, t("Lettura con pyGIMLi", "Reading with pyGIMLi"))
        folder = work_folder(chunk)
        try:
            res, arrays = run({"protocol": 1, "task": "read", "source": str(path.absolute()),
                               "output": str(folder / "read.npz")}, folder, progress)
        except environment.GimliMissing as exc:
            raise ValueError(f"{first}\n{t('Gli altri formati li legge pyGIMLi', 'The other formats are read by pyGIMLi')}: "
                             f"{exc}") from exc
        finally:
            (folder / "read.npz").unlink(missing_ok=True)
        prof = ert_mod.from_arrays(path.stem, arrays, str(path), "pygimli")
        if electrodes:
            el = ert_mod.read_electrodes(electrodes)
            el["source"] = str(electrodes)
            ert_mod.with_electrodes(prof, el)
    if name.strip():
        prof.name = name.strip()
    return ert_mod.save(chunk, prof)


def simulate_profile(chunk, name: str, count: int = 48, spacing: float = 1.0, scheme: str = "dd", model=None,
                     slope: float = 0.0, noise: float = 0.02, seed: int = 1337, progress=None) -> dict:
    """Synthetic readings of a model on ``count`` electrodes ``spacing`` metres apart (pyGIMLi's
    forward modelling) as a new profile: a demonstration, a check of the inversion."""
    x = np.arange(int(count)) * float(spacing)
    sensors = np.column_stack([x, slope * x])
    model = model or {"background": 50.0, "layers": [[3.0, 200.0]], "blocks": [[0.42 * x[-1], 0.58 * x[-1], 1.5, 3.5,
                                                                                 2000.0]]}
    folder = work_folder(chunk)
    try:
        res, arrays = run({"protocol": 1, "task": "simulate", "sensors": sensors.tolist(), "model": model,
                           "scheme": scheme, "noise": float(noise), "seed": int(seed),
                           "output": str(folder / "simulate.npz")}, folder, progress)
    finally:
        (folder / "simulate.npz").unlink(missing_ok=True)
    prof = ert_mod.from_arrays(name, arrays, "pyGIMLi " + scheme, "synthetic")
    if "k" in prof.values:
        del prof.values["k"]  # the factors of pyGIMLi's mesh: the inversion makes its own
    prof.notes.append(t("letture sintetiche di pyGIMLi: {0}", "synthetic readings of pyGIMLi: {0}").format(json.dumps(model)))
    return ert_mod.save(chunk, prof)


# ------------------------------------------------------------------ inversion
def inversion_input(chunk, pid: str) -> tuple[dict, np.ndarray]:
    prof = ert_mod.load(chunk, pid)
    e = ert_mod.entry_of(chunk, pid)
    valid = ert_mod.valid_mask(prof, e.get("filters"))
    sensors = np.column_stack([prof.positions(), prof.heights()])
    arrays = {"sensors": sensors, "a": prof.a, "b": prof.b, "m": prof.m, "n": prof.n, "valid": valid.astype(np.int8)}
    for k in ("rhoa", "r", "err", "u", "i"):
        if prof.has(k):
            arrays[k] = prof.value(k)
    return arrays, valid


def invert_profile(chunk, pid: str, params: dict | None = None, blank: dict | None = None, palette: str | None = None,
                   progress=None, timeout: float | None = None) -> dict:
    """The inversion of the profile in pyGIMLi's worker: the result (mesh, resistivity, coverage)
    in ``inversion-<run>.npz``, the section written (:func:`.ert_section.write`), the parameters, the
    fit and the versions in the record and in ``inversion-<run>.json`` beside it (the paradata). Every
    run has files of its own, named in the record: Undo takes the record of the run before back and
    finds its files (the unused ones go when the project is opened again, :func:`.ert.settle`).
    ``timeout``: seconds before a silent or endless worker is stopped (None: no limit)."""
    e = ert_mod.entry_of(chunk, pid)
    pid = e["id"]
    params = {**DEFAULT_PARAMS, **(params or {})}
    if params.get("doi") == "oldenburg" and blank is None:
        blank = {"method": "doi"}
    arrays, valid = inversion_input(chunk, pid)
    if valid.sum() < 4:
        raise ValueError(t("Restano {0} letture valide: allenta i filtri.", "{0} valid readings left: loosen the "
                                                                            "filters.").format(int(valid.sum())))
    folder = work_folder(chunk, pid)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "input.npz", **arrays)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    out = ert_mod.folder(chunk, pid) / f"inversion-{run_id}.npz"
    tmp = folder / "result.npz"
    try:
        res, result = run({"protocol": 1, "task": "invert", "input": str(folder / "input.npz"), "params": params,
                           "output": str(tmp)}, folder, progress, timeout)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
        (folder / "input.npz").unlink(missing_ok=True)
    fit = dict(res.get("fit") or {})
    prof_entry = ert_mod.entry_of(chunk, pid)
    old_display = (prof_entry.get("inversion") or {}).get("display") or {}
    rel = lambda p: str(p.relative_to(chunk.data_dir))  # noqa: E731
    prof_entry["inversion"] = {"params": params, "fit": fit, "versions": res.get("versions", {}),
                               "date": datetime.now().isoformat(timespec="seconds"), "stale": False, "run": run_id,
                               "readings_used": int(fit.get("readings", valid.sum())),
                               "display": {k: v for k, v in old_display.items() if k in ("palette",)},
                               "files": {"result": rel(out)}}
    section.write(chunk, pid, blank=blank or old_display.get("blank"), palette=palette)
    para = write_paradata(chunk, pid, ert_mod.folder(chunk, pid) / f"inversion-{run_id}.json")
    prof_entry["inversion"]["files"]["paradata"] = rel(para)
    return prof_entry


def paradata(chunk, pid: str) -> dict:
    e = ert_mod.entry_of(chunk, pid)
    inv = e.get("inversion") or {}
    return {"profile": e["name"], "source": e.get("source"), "format": e.get("format"), "electrodes": e.get("electrodes"),
            "readings": e.get("readings"), "array": e.get("array"), "placement": e.get("placement"),
            "filters": e.get("filters"), "inversion": {k: inv.get(k) for k in ("params", "fit", "versions", "date")},
            "display": inv.get("display"),
            "software": {"inversion": "pyGIMLi (Rücker, Günther & Wagner 2017), ERTManager",
                         "plugin": "ppg-geophysics " + _version()}}


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except ImportError:
        return ""


def write_paradata(chunk, pid: str, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(paradata(chunk, pid), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------ exports
def export_lines_gpkg(chunk, path, pids=None) -> Path:
    """GeoPackage: the lines of the placed profiles (table ``profili_ert``) and their electrodes
    (``elettrodi_ert``, with the height used); other tables of the file are kept."""
    from shapely.geometry import LineString, Point

    from pyarchinit_photogrammetry.io import gpkg

    crs = chunk.crs if getattr(chunk, "crs", "LOCAL") != "LOCAL" else None
    lines, points = [], []
    for e in ert_mod.entries(chunk):
        if pids is not None and e["id"] not in pids:
            continue
        pl = e.get("placement")
        if not pl:
            continue
        fit = (e.get("inversion") or {}).get("fit") or {}
        lines.append((LineString([tuple(p) for p in pl["line"]]),
                      {"nome": e["name"], "elettrodi": int(e.get("electrodes", 0)), "letture": int(e.get("readings", 0)),
                       "valide": int(e.get("valid", 0)), "dispositivo": e.get("array", ""),
                       "spaziatura_m": float(pl.get("spacing") or e.get("spacing") or 0), "quote": pl.get("heights", ""),
                       "lunghezza_m": float(pl.get("length", 0)), "chi2": fit.get("chi2"), "rms_pc": fit.get("rrms")}))
        prof = ert_mod.load(chunk, e["id"])
        for k in range(prof.electrodes):
            points.append((Point(float(prof.elec["elec_x"][k]), float(prof.elec["elec_y"][k])),
                           {"profilo": e["name"], "elettrodo": k + 1, "distanza_m": float(prof.elec["elec_u"][k]),
                            "quota_m": float(prof.elec["elec_z"][k])}))
    gpkg.write_table(path, "profili_ert", "LINESTRING", {"nome": "TEXT", "elettrodi": "INTEGER", "letture": "INTEGER",
                                                         "valide": "INTEGER", "dispositivo": "TEXT", "spaziatura_m": "REAL",
                                                         "quote": "TEXT", "lunghezza_m": "REAL", "chi2": "REAL",
                                                         "rms_pc": "REAL"}, lines, crs, "profili ERT")
    gpkg.write_table(path, "elettrodi_ert", "POINT", {"profilo": "TEXT", "elettrodo": "INTEGER", "distanza_m": "REAL",
                                                      "quota_m": "REAL"}, points, crs, "elettrodi ERT")
    return Path(path)


EXPORTS = {".tif": "section GeoTIFF", ".tiff": "section GeoTIFF", ".csv": "cells", ".xyz": "pixels", ".png": "figure",
           ".svg": "figure", ".pdf": "figure", ".gpkg": "line", ".json": "paradata", ".ohm": "readings"}


def write_unified(chunk, pid: str, path) -> Path:
    """The readings kept by the filters in the unified data format (``.ohm``), with the electrodes at
    their distance and height: pyGIMLi, BERT and other programs read it."""
    prof = ert_mod.load(chunk, pid)
    e = ert_mod.entry_of(chunk, pid)
    valid = ert_mod.valid_mask(prof, e.get("filters"))
    tokens = ["a", "b", "m", "n"] + [k for k in ("rhoa", "r", "err", "u", "i", "ip") if prof.has(k)]
    if "rhoa" not in tokens:
        tokens.insert(4, "rhoa")
    rhoa = prof.apparent()
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{prof.electrodes}\n# x z\n")
        for u, z in zip(prof.positions(), prof.heights()):
            f.write(f"{u:.4f}\t{z:.4f}\n")
        f.write(f"{int(valid.sum())}\n# {' '.join(tokens)}\n")
        for i in np.flatnonzero(valid):
            row = [str(int(prof.a[i]) + 1), str(int(prof.b[i]) + 1), str(int(prof.m[i]) + 1), str(int(prof.n[i]) + 1)]
            for k in tokens[4:]:
                v = rhoa[i] if k == "rhoa" else prof.value(k)[i]
                row.append(f"{v:.6g}")
            f.write("\t".join(row) + "\n")
        f.write("0\n")
    return path


def export(chunk, pid: str, path, pseudosection: bool = False) -> Path:
    """One export of the profile by the extension of ``path``: ``.tif`` the section (float32, ohm m, frame
    u-v, with ``.frame.json``), ``.csv`` the cells of the mesh, ``.xyz`` the pixels of the section with
    world coordinates, ``.png``/``.svg``/``.pdf`` the figure of the section (of the pseudosection with
    ``pseudosection``), ``.gpkg`` the line and the electrodes, ``.json`` the paradata, ``.ohm`` the
    readings kept."""
    path = Path(path)
    ext = path.suffix.lower()
    e = ert_mod.entry_of(chunk, pid)
    needs_inv = ext in (".tif", ".tiff", ".csv", ".xyz") or (ext in (".png", ".svg", ".pdf") and not pseudosection)
    if needs_inv and not (e.get("inversion") or {}).get("files"):
        raise ValueError(t("Il profilo {0} non è ancora invertito.", "The profile {0} is not inverted yet.").format(e["name"]))
    if ext in (".tif", ".tiff"):
        return section.write_geotiff(chunk, pid, path)
    if ext == ".csv":
        return section.write_cells_csv(chunk, pid, path)
    if ext == ".xyz":
        return section.write_xyz(chunk, pid, path)
    if ext in (".png", ".svg", ".pdf"):
        return section.figure(chunk, pid, path, "pseudosection" if pseudosection else "section")
    if ext == ".gpkg":
        return export_lines_gpkg(chunk, path, [e["id"]])
    if ext == ".json":
        return write_paradata(chunk, pid, path)
    if ext == ".ohm":
        return write_unified(chunk, pid, path)
    raise ValueError(t("Formato non riconosciuto: {0} (tif, csv, xyz, png, svg, pdf, gpkg, json, ohm)",
                       "Format not known: {0} (tif, csv, xyz, png, svg, pdf, gpkg, json, ohm)").format(ext))


def export_all(chunk, pid: str, folder) -> list[Path]:
    """Every export of the profile into ``folder`` (named after the profile)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    e = ert_mod.entry_of(chunk, pid)
    stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in e["name"])[:60] or "ert"
    out = [export(chunk, pid, folder / f"{stem}_pseudosezione.png", pseudosection=True),
           export(chunk, pid, folder / f"{stem}_letture.ohm"), export(chunk, pid, folder / f"{stem}_paradati.json")]
    if e.get("placement"):
        out.append(export(chunk, pid, folder / f"{stem}_linea.gpkg"))
    if (e.get("inversion") or {}).get("files"):
        out += [export(chunk, pid, folder / f"{stem}_sezione.tif"), export(chunk, pid, folder / f"{stem}_celle.csv"),
                export(chunk, pid, folder / f"{stem}_sezione.xyz"), export(chunk, pid, folder / f"{stem}_sezione.png"),
                export(chunk, pid, folder / f"{stem}_sezione.svg")]
    return out


def summary_line(e: dict) -> str:
    """One line on a profile, in the language of the interface."""
    text = t("{0}: {1} elettrodi, {2} letture ({3}), {4} valide", "{0}: {1} electrodes, {2} readings ({3}), {4} valid") \
        .format(e["name"], e.get("electrodes", 0), e.get("readings", 0), t(*ert_mod.ARRAY_NAMES.get(e.get("array", "other"),
                                                                                                     ("?", "?"))),
                e.get("valid", e.get("readings", 0)))
    pl = e.get("placement")
    text += t("; linea {0:.1f} m", "; line {0:.1f} m").format(pl["length"]) if pl else t("; da posizionare", "; to place")
    inv = e.get("inversion")
    if inv and inv.get("fit"):
        f = inv["fit"]
        text += t("; inversione χ² {0:.2f}, RMS {1:.1f} %", "; inversion χ² {0:.2f}, RMS {1:.1f} %").format(
            f.get("chi2", float("nan")), f.get("rrms", float("nan")))
        if inv.get("stale"):
            text += t(" (da rifare)", " (to do again)")
    return text


def summary(chunk) -> str:
    lines = [summary_line(e) for e in ert_mod.entries(chunk)]
    return "\n".join(lines) if lines else t("nessun profilo ERT", "no ERT profile")
