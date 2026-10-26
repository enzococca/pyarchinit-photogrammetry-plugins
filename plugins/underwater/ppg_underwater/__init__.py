"""Underwater - a plugin of pyarchinit-photogrammetry for surveys under water.

1. **Colours restored.** Water absorbs red first and the suspended particles add a bluish veil that
   grows with the distance (backscatter). With the range of every pixel from the program's depth
   maps, the colours are restored by the physical model of Sea-thru (Akkaynak and Treibitz, CVPR
   2019; :mod:`.colour`, :mod:`.ranges`); before the alignment, or without depth maps, a quick
   correction (red channel compensated, white balance, contrast). Corrected copies of the photos,
   used by the orthomosaic and the texture when the plugin points the cameras to them
   (:mod:`.photos`).
2. **Usable photos and frames.** Blurred, low-contrast and hazy photos scored and flagged with
   thresholds, excluded with one click (:mod:`.quality`); sharp, well-spaced frames from videos and
   from fast image sequences (an ROV at 20 images a second), chosen by motion and sharpness
   (:mod:`.frames`); Kalibr calibrations (:mod:`.sources`).
3. **Depth instead of GPS.** Depth logs (ROV pressure sensors, dive computers as CSV, UDDF) matched
   to the photos by time (:mod:`.depthlog`) level and scale the chunk, ``Z = -depth``, in a local
   frame; known positions (USBL) place it (:mod:`.levelling`).

The *Subacqueo* pane and the *Plugin > Subacqueo* menu in the window, the products under each chunk
in the workspace; ``ppg plugin underwater ...`` from the command line; *File > Esporta > Foto
subacquee (CSV)*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import number, t

__version__ = "0.1.0"


def _uw(chunk) -> dict:
    return chunk.meta.setdefault("underwater", {})


def log_path(chunk) -> Path:
    return chunk.data_dir / "underwater" / "depth_log.npz"


def save_log(chunk, log) -> None:
    import numpy as np

    path = log_path(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=log.times, depths=log.depths)
    _uw(chunk)["log"] = {**log.summary(), "file": path.name}


def load_log(chunk):
    import numpy as np

    from .depthlog import DepthLog

    info = (chunk.meta.get("underwater") or {}).get("log")
    path = log_path(chunk)
    if not info or not path.exists():
        return None
    data = np.load(path)
    return DepthLog(data["times"], data["depths"], bool(info.get("absolute", True)), source=info.get("source", ""),
                    kind=info.get("kind", "depth"))


def match_depths(chunk, log, offset: float = 0.0, tolerance: float = 2.0) -> dict:
    """The depths of the log on the photos of ``chunk``, by time; kept in the cameras."""
    from . import depthlog

    found = depthlog.match(chunk.cameras, log, offset, tolerance=tolerance)
    for cam in chunk.cameras:
        rec = cam.meta.setdefault("underwater", {})
        if cam.key in found:
            rec["depth"] = round(found[cam.key], 4)
        else:
            rec.pop("depth", None)
    timed = len(depthlog.photo_times(chunk.cameras))
    _uw(chunk)["depths"] = {"offset": float(offset), "matched": len(found), "timed": timed, "photos": len(chunk.cameras)}
    return {"matched": len(found), "timed": timed, "photos": len(chunk.cameras)}


LOW_COVERAGE = 0.4  # below it, most ranges of the restoration are guessed from the nearest surface


def low_coverage_text(covered: float) -> str:
    return t("Le mappe di profondità coprono solo il {0} % di ogni foto (mediana): dove mancano, la distanza viene "
             "dalla superficie più vicina e i colori sono solo plausibili. Con foto ravvicinate costruiscile con il "
             "motore «PatchMatch multi-vista» (Flusso di lavoro > Costruisci nuvola densa..., «Solo le mappe di "
             "profondità») e ripristina di nuovo.",
             "The depth maps cover only {0} % of each photo (median): where they are missing, the range comes from "
             "the nearest surface and the colours are only plausible. For close-range photos build them with the "
             "\"PatchMatch multi-view\" engine (Workflow > Build dense cloud..., \"Depth maps only\") and restore "
             "again.").format(number(100 * covered, 0))


# ------------------------------------------------------------------ summaries
def colour_text(chunk) -> str:
    from . import photos

    s = photos.status(chunk)
    if not s["copies"]:
        return t("Nessuna foto corretta.", "No corrected photo.")
    method = {"seathru": "Sea-thru", "quick": t("rapida", "quick")}.get(s["method"], s["method"] or "")
    text = t("{0} foto corrette ({1}); in uso: {2}.", "{0} corrected photos ({1}); in use: {2}.").format(
        s["copies"], method, t("le corrette", "the corrected ones") if s["using"] else t("le originali", "the originals"))
    info = (chunk.meta.get("underwater") or {}).get("colour") or {}
    coeffs = info.get("coefficients")
    if info.get("attenuation"):
        text += " " + (t("Attenuazione dai punti visti due volte.", "Attenuation from the points seen twice.")
                       if str(info["attenuation"]).startswith("multiview")
                       else t("Attenuazione dall'illuminante (poche foto in comune).",
                              "Attenuation from the illuminant (few photos in common)."))
    if coeffs and info.get("per") == "survey":
        unit = "/m" if (info.get("model") or {}).get("metric") else ""
        parts = [f"{c}: βD {number(v['beta_D'], 3)}{unit}, βB {number(v['beta_B'], 3)}{unit}" for c, v in coeffs.items()]
        text += "\n" + t("Acqua: ", "Water: ") + "; ".join(parts)
    return text


def depth_texts(chunk) -> tuple[str, str]:
    from .levelling import levelled

    uw = chunk.meta.get("underwater") or {}
    log = uw.get("log")
    if log:
        lt = t("Registro {0}: {1} campioni, {2}-{3} m.", "Log {0}: {1} samples, {2}-{3} m.").format(
            log.get("source", ""), log.get("samples", 0), number(log.get("min_depth") or 0, 2),
            number(log.get("max_depth") or 0, 2))
        d = uw.get("depths")
        if d:
            lt += " " + t("{0} foto su {1} con la profondità (scarto {2} s).",
                          "{0} photos of {1} with a depth (offset {2} s).").format(d["matched"], d["photos"],
                                                                                  number(d["offset"], 2))
    else:
        lt = t("Nessun registro di profondità.", "No depth log.")
    lev = uw.get("levelling")
    if not lev:
        return lt, t("Chunk non messo in bolla.", "Chunk not levelled.")
    state = "" if levelled(chunk) else t(" (non più valido: allineamento rifatto o chunk spostato)",
                                         " (no longer valid: alignment done again or chunk moved)")
    src = {"depths": t("profondità", "depths"), "bars": t("barre di scala", "scale bars"), "given": t("data", "given")}
    lv = t("In bolla su {0} camere: scala dalle {1}, scarto quadratico {2} cm (max {3} cm){4}.",
           "Levelled on {0} cameras: scale from the {1}, RMS {2} cm (max {3} cm){4}.").format(
        lev["cameras"], src.get(lev.get("scale_from"), lev.get("scale_from")), number(lev["rms"] * 100, 1),
        number(lev["max"] * 100, 1), state)
    if lev.get("tilt_std_deg") is not None:
        lv += " " + t("Incertezza della verticale {0}°.", "Uncertainty of the vertical {0}°.").format(
            number(lev["tilt_std_deg"], 2))
    return lt, lv


class UnderwaterPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Subacqueo", "Underwater")
        self.description = t("Rilievi sott'acqua: colori ripristinati, foto e fotogrammi utilizzabili, profondità "
                             "invece del GPS", "Underwater surveys: colours restored, usable photos and frames, "
                                               "depth instead of GPS")
        self.context = None
        self.pane = None
        self._busy_signal = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Pannello Subacqueo", "Underwater pane"), self.show_pane)
        context.add_action(t("Valuta le foto", "Score the photos"), lambda: self.score(None), processing=True)
        context.add_action(t("Fotogrammi da un video...", "Frames from a video..."), self.video_dialog, processing=True)
        context.add_action(t("Fotogrammi da una sequenza di immagini...", "Frames from an image sequence..."),
                           self.sequence_dialog, processing=True)
        context.add_action(t("Correggi i colori delle foto", "Correct the colours of the photos"),
                           lambda: self.correct(None), processing=True)
        context.add_action(t("Importa un registro di profondità...", "Import a depth log..."), lambda: self.import_log(None),
                           processing=True)
        context.add_action(t("Metti in bolla e in scala con le profondità", "Level and scale with the depths"),
                           lambda: self.level(None), processing=True)
        context.add_command("underwater", self.command,
                            help="underwater surveys: status, score, exclude, include, frames-sequence, frames-video, "
                                 "calibration, quick, restore, use, depth, offset, level, csv")
        context.add_exporter(t("Foto subacquee (CSV)", "Underwater photos (CSV)"), ".csv", self._export_csv)
        if context.window is not None:
            self._install_pane(context.window)
            context.on_refresh(self.refresh_pane)
            context.add_tree_items(self._tree_items)

    def deactivate(self) -> None:
        signal, self._busy_signal = self._busy_signal, None
        if signal is not None:
            try:
                signal.disconnect(self._busy)
            except (RuntimeError, TypeError):
                pass
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import UnderwaterPane

        self.pane = p = UnderwaterPane()
        self.context.add_pane(p, t("Subacqueo", "Underwater"), "plugin", area="right")
        p.video_requested.connect(self.video_dialog)
        p.sequence_requested.connect(self.sequence_dialog)
        p.calibration_requested.connect(self.calibration_dialog)
        p.score_requested.connect(self.score)
        p.thresholds_changed.connect(self.reflag)
        p.exclude_requested.connect(self.exclude)
        p.include_requested.connect(self.include)
        p.csv_requested.connect(self.csv_dialog)
        p.preview_requested.connect(self.preview)
        p.correct_requested.connect(self.correct)
        p.use_requested.connect(self.use)
        p.folder_requested.connect(self.open_folder)
        p.log_requested.connect(self.import_log)
        p.match_requested.connect(self.match)
        p.offset_requested.connect(self.estimate_offset)
        p.level_requested.connect(self.level)
        p.positions_requested.connect(self.positions_dialog)
        signal = getattr(getattr(window, "runner", None), "busy_changed", None)
        if signal is not None:
            signal.connect(self._busy)
            self._busy_signal = signal
        self.refresh_pane()

    def _busy(self, busy: bool) -> None:
        if self.pane is not None:
            self.pane.set_busy(bool(busy))

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        return None if doc is None else doc.chunk

    def _message(self, kind: str, text: str, details: str = "") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text, details)

    def _status(self, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            return
        log = getattr(window, "log_pane", None)
        if log is not None:
            log.info(text)
        window.statusBar().showMessage(text, 15000)

    def _changed(self, photos: bool = False) -> None:
        """The chunk changed: to be saved, shown again (the photos pane when photos changed)."""
        self.context.mark_modified()
        window = self.context.window
        if photos and window is not None and hasattr(window, "refresh_all"):
            window.refresh_all()  # calls refresh_pane through on_refresh
        else:
            self.refresh_pane()
            self.context.refresh_tree()

    def _need_chunk(self):
        chunk = self._chunk()
        if chunk is None or not chunk.cameras:
            self._message("warning", t("Il chunk attivo non ha foto.", "The active chunk has no photos."))
            return None
        return chunk

    def _open_file(self, title: str, filters: str) -> str | None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self.context.window, title, "", filters)
        return path or None

    def refresh_pane(self) -> None:
        from . import quality
        from .quality import reason_text

        if self.pane is None:
            return
        chunk = self._chunk()
        if chunk is None:
            self.pane.set_photos([], "")
            self.pane.set_colour("")
            self.pane.set_depth("", "")
            self.pane.set_cameras([])
            return
        self.pane.set_limits(quality.thresholds(chunk))
        scored = [c for c in chunk.cameras if (c.meta.get("underwater") or {}).get("scores") is not None]
        flagged = quality.flagged(chunk)
        excluded = [c for c in chunk.cameras if (c.meta.get("underwater") or {}).get("excluded")]
        if scored:
            text = t("{0} foto valutate su {1}: {2} segnate, {3} escluse.",
                     "{0} photos scored of {1}: {2} flagged, {3} excluded.").format(len(scored), len(chunk.cameras),
                                                                                   len(flagged), len(excluded))
        else:
            text = t("{0} foto, non ancora valutate.", "{0} photos, not scored yet.").format(len(chunk.cameras))
        rows = []
        for cam in sorted(flagged, key=lambda c: ((c.meta["underwater"].get("scores") or {}).get("quality") or 0)):
            s = cam.meta["underwater"].get("scores") or {}
            rows.append({"label": cam.label, "quality": s.get("quality"), "contrast": s.get("contrast"),
                         "veil": s.get("veil"), "why": reason_text(cam.meta["underwater"]["flags"]),
                         "enabled": cam.enabled})
        self.pane.set_photos(rows[:500], text)
        self.pane.set_cameras([(c.key, c.label) for c in chunk.cameras])
        self.pane.set_colour(colour_text(chunk))
        lt, lv = depth_texts(chunk)
        self.pane.set_depth(lt, lv)
        d = (chunk.meta.get("underwater") or {}).get("depths")
        if d:
            self.pane.set_offset(d.get("offset", 0.0))

    # ------------------------------------------------------------------ workspace
    def _tree_items(self, chunk) -> list:
        from . import photos, quality
        from .levelling import levelled

        uw = chunk.meta.get("underwater") or {}
        if not uw and not any("underwater" in c.meta for c in chunk.cameras):
            return []
        children = []
        s = photos.status(chunk)
        if s["copies"]:
            children.append({"label": t("Foto corrette: {0} ({1})", "Corrected photos: {0} ({1})").format(
                s["copies"], t("in uso", "in use") if s["using"] else t("non in uso", "not in use")),
                "icon": "photo", "activate": self.show_pane,
                "actions": [(t("Usa le foto corrette", "Use the corrected photos"), lambda: self.use("corrected")),
                            (t("Usa le originali", "Use the originals"), lambda: self.use("original")),
                            (t("Apri la cartella", "Open the folder"), self.open_folder)]})
        flagged = quality.flagged(chunk)
        if flagged:
            children.append({"label": t("Foto segnate: {0}", "Flagged photos: {0}").format(len(flagged)),
                             "icon": "photo", "activate": self.show_pane,
                             "actions": [(t("Escludi le segnate", "Exclude the flagged"), self.exclude),
                                         (t("Riattiva le escluse", "Include again"), self.include)]})
        if uw.get("log"):
            d = uw.get("depths") or {}
            children.append({"label": t("Profondità: {0} foto", "Depths: {0} photos").format(d.get("matched", 0)),
                             "icon": "dem", "tooltip": uw["log"].get("source", ""), "activate": self.show_pane})
        if uw.get("levelling"):
            lev = uw["levelling"]
            ok = levelled(chunk)
            children.append({"label": (t("In bolla: ±{0} cm", "Levelled: ±{0} cm").format(number(lev["rms"] * 100, 1))
                                       if ok else t("Bolla da rifare", "Levelling to redo")),
                             "icon": "dem", "activate": self.show_pane,
                             "actions": [(t("Metti in bolla di nuovo", "Level again"), lambda: self.level(None))]})
        if not children:
            return []
        return [{"label": t("Subacqueo", "Underwater"), "icon": "plugin", "children": children, "expanded": True,
                 "activate": self.show_pane}]

    # ------------------------------------------------------------------ 1. photos and frames
    def score(self, limits: dict | None) -> None:
        from . import quality

        chunk = self._need_chunk()
        if chunk is None:
            return
        limits = limits or quality.thresholds(chunk)

        def done(rep):
            self._changed()
            self._status(t("{0} foto valutate, {1} segnate.", "{0} photos scored, {1} flagged.").format(
                rep["photos"], rep["flagged"]))

        self.context.run(t("Valutazione delle foto", "Scoring the photos"),
                         lambda p: quality.score_chunk(chunk, limits=limits, progress=p), done)

    def reflag(self, limits: dict) -> None:
        from . import quality

        chunk = self._chunk()
        if chunk is None or not any("scores" in (c.meta.get("underwater") or {}) for c in chunk.cameras):
            if chunk is not None:
                _uw(chunk)["thresholds"] = dict(limits)
            return
        quality.reflag(chunk, limits)
        self._changed()

    def exclude(self) -> None:
        from . import quality

        chunk = self._need_chunk()
        if chunk is None:
            return
        n = quality.exclude(chunk)
        self._changed(photos=True)
        self._status(t("{0} foto escluse (disattivate).", "{0} photos excluded (disabled).").format(n))

    def include(self) -> None:
        from . import quality

        chunk = self._need_chunk()
        if chunk is None:
            return
        n = quality.include(chunk)
        self._changed(photos=True)
        self._status(t("{0} foto riattivate.", "{0} photos enabled again.").format(n))

    def csv_dialog(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import quality

        chunk = self._need_chunk()
        if chunk is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.context.window, t("Tabella delle foto", "Table of the photos"),
                                              "foto_subacquee.csv", "CSV (*.csv)")
        if path:
            quality.write_csv(chunk, path)
            self._status(t("Tabella scritta: {0}", "Table written: {0}").format(path))

    def _export_csv(self, chunk, path, progress) -> None:
        from . import quality

        progress(0.5, t("Foto subacquee", "Underwater photos"))
        quality.write_csv(chunk, path)

    def sequence_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        chunk = self._chunk()
        if chunk is None:
            return
        v = ParamDialog.ask(self.context.window, t("Fotogrammi da una sequenza", "Frames from a sequence"), [
            Field("folder", t("Cartella delle immagini", "Folder of the images"), "", kind="dir"),
            Field("times", t("Tabella dei tempi (facoltativa)", "Table of the times (optional)"), "", kind="file_open",
                  filter="CSV (*.csv *.txt)", tip=t("Ora di ogni fotogramma (nome o numero, ora); altrimenti dal nome "
                                                     "del file o dall'EXIF", "Time of every frame (name or number, time); "
                                                                            "else from the file name or the EXIF")),
            Field("overlap", t("Sovrapposizione fra fotogrammi", "Overlap between frames"), 0.8, minimum=0.3,
                  maximum=0.97, decimals=2),
            Field("min_sharpness", t("Nitidezza minima (rispetto ai vicini)", "Least sharpness (against the neighbours)"),
                  0.6, minimum=0.0, maximum=1.0, decimals=2),
            Field("step", t("Analizza un fotogramma ogni", "Analyse one frame in"), 1, minimum=1, maximum=100),
            Field("max_gap", t("Al più ogni (s, 0 = nessun limite)", "At most every (s, 0 = no limit)"), 0.0,
                  minimum=0.0, maximum=3600.0, decimals=1)],
            intro=t("I fotogrammi vengono scelti per quanto si è spostata la vista e per nitidezza; restano dove sono "
                    "e diventano foto del chunk attivo con la loro ora.",
                    "Frames are chosen by how far the view moved and by sharpness; they stay where they are and become "
                    "photos of the active chunk with their time."))
        if not v or not v["folder"]:
            return
        self._sequence(chunk, v)

    def _sequence(self, chunk, v: dict) -> None:
        from . import frames

        folder = Path(v["folder"])

        def job(progress):
            imgs = frames.list_images(folder)
            times = frames.read_frame_times(v["times"], imgs) if v.get("times") else None
            if times is None:
                from .depthlog import time_from_name

                found = {p.name: time_from_name(p.name) for p in imgs}
                times = {k: s for k, s in found.items() if s is not None} or None
            res = frames.select_sequence(imgs, times, overlap=v["overlap"], min_sharpness=v["min_sharpness"],
                                         step=int(v["step"]), max_gap=v["max_gap"] or None, progress=progress)
            return res

        def done(res):
            cams = frames.add_frames(chunk, res["frames"], source=folder.name)
            self._changed(photos=True)
            self._message("info", t("{0} fotogrammi scelti su {1} ({2} tratti), aggiunti come foto.",
                                    "{0} frames chosen of {1} ({2} stretches), added as photos.").format(
                len(cams), res["analysed"], res["segments"]))

        self.context.run(t("Fotogrammi da una sequenza", "Frames from a sequence"), job, done)

    def video_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        chunk = self._chunk()
        if chunk is None:
            return
        v = ParamDialog.ask(self.context.window, t("Fotogrammi da un video", "Frames from a video"), [
            Field("video", t("Video", "Video"), "", kind="file_open", filter="Video (*.mp4 *.mov *.m4v *.avi *.mkv *.mts)"),
            Field("overlap", t("Sovrapposizione fra fotogrammi", "Overlap between frames"), 0.8, minimum=0.3,
                  maximum=0.97, decimals=2),
            Field("start", t("Inizio del video (facoltativo, AAAA-MM-GG hh:mm:ss)", "Start of the video (optional, "
                                                                                  "YYYY-MM-DD hh:mm:ss)"), "",
                  tip=t("Altrimenti l'ora di creazione scritta nel file", "Else the creation time written in the file"))],
            intro=t("I fotogrammi vengono scritti in una cartella accanto al video e aggiunti al chunk attivo con la "
                    "loro ora.", "The frames are written to a folder beside the video and added to the active chunk "
                                 "with their time."))
        if not v or not v["video"]:
            return
        from . import frames
        from .depthlog import parse_datetime

        video = Path(v["video"])
        start = parse_datetime(v["start"]) if v.get("start") else None
        folder = video.with_name(f"{video.stem}_fotogrammi")

        def done(res):
            cams = frames.add_frames(chunk, res["frames"], source=video.name)
            self._changed(photos=True)
            self._status(t("{0} fotogrammi aggiunti da {1}.", "{0} frames added from {1}.").format(len(cams), video.name))

        self.context.run(t("Fotogrammi da un video", "Frames from a video"),
                         lambda p: frames.extract_video(video, folder, overlap=v["overlap"], start=start, progress=p), done)

    def calibration_dialog(self) -> None:
        from . import sources

        chunk = self._need_chunk()
        if chunk is None:
            return
        path = self._open_file(t("Calibrazione Kalibr", "Kalibr calibration"), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            calib = sources.read_kalibr(path)
        except (ValueError, KeyError, OSError) as exc:
            self._message("warning", str(exc))
            return
        n = sources.apply_calibration(chunk, calib)
        self._changed(photos=True)
        self._message("info" if n else "warning", t("Calibrazione {0} ({1} {2}x{3}) data a {4} sensori, tenuta fissa.",
                                                    "Calibration {0} ({1} {2}x{3}) given to {4} sensors, held fixed.").format(
            calib["source"], calib["model"], calib["width"], calib["height"], n))

    # ------------------------------------------------------------------ 2. colours
    def _quick_params(self, v: dict):
        from .colour import QuickParams

        return QuickParams(alpha=v.get("alpha", 1.0), blue=v.get("blue", 0.0), balance=v.get("balance", True),
                           clahe=v.get("clahe", False))

    def preview(self, v: dict | None) -> None:
        from . import photos

        chunk = self._need_chunk()
        if chunk is None or self.pane is None:
            return
        v = v or self.pane.colour_values()
        key = v.get("camera")
        cam = next((c for c in chunk.cameras if c.key == key), chunk.cameras[0])

        def done(pair):
            if self.pane is not None:
                self.pane.set_preview(*pair)

        def job(progress):
            progress(0.1, t("Anteprima", "Preview"))
            return photos.preview(chunk, cam, v["method"], self._quick_params(v), v.get("per", "survey"))

        self.context.run(t("Anteprima dei colori", "Preview of the colours"), job, done)

    def correct(self, v: dict | None) -> None:
        from . import photos

        chunk = self._need_chunk()
        if chunk is None:
            return
        v = v or (self.pane.colour_values() if self.pane is not None else {"method": "seathru", "per": "survey"})
        if v["method"] == "seathru":
            try:
                from .ranges import RangeMaps

                RangeMaps(chunk)
            except RuntimeError:
                self._message("warning", t("Il ripristino fisico vuole le mappe di profondità: allinea le foto e "
                                           "costruiscile (Flusso di lavoro > Costruisci nuvola densa..., «Solo le "
                                           "mappe di profondità»), oppure usa la correzione rapida.",
                                           "The physics-based restoration needs the depth maps: align the photos and "
                                           "build them (Workflow > Build dense cloud..., \"Depth maps only\"), or use "
                                           "the quick correction."))
                return

            def job(progress):
                return photos.restore(chunk, per=v.get("per", "survey"), progress=progress)
        else:
            params = self._quick_params(v)

            def job(progress):
                return photos.quick(chunk, params=params, progress=progress)

        def done(info):
            self._changed()
            self._status(t("{0} foto corrette in {1} s: «Usa le foto corrette» le dà a ortofoto e texture.",
                           "{0} photos corrected in {1} s: 'Use the corrected photos' gives them to the orthomosaic "
                           "and the texture.").format(info["photos"], number(info.get("seconds", 0), 0)))
            if info.get("method") == "seathru" and info.get("covered", 1.0) < LOW_COVERAGE:
                self._message("warning", low_coverage_text(info["covered"]))

        self.context.run(t("Colori delle foto", "Colours of the photos"), job, done)

    def use(self, which: str) -> None:
        from . import photos

        chunk = self._need_chunk()
        if chunk is None:
            return
        res = photos.use(chunk, which)
        self._changed(photos=True)
        if which == "corrected":
            text = t("{0} foto puntano alle copie corrette ({1} senza copia): ortofoto, texture e mappe di profondità "
                     "le useranno.", "{0} photos point to the corrected copies ({1} without a copy): orthomosaic, "
                                     "texture and depth maps will use them.").format(res["switched"], res["missing"])
        else:
            text = t("{0} foto puntano di nuovo agli originali.", "{0} photos point to the originals again.").format(
                res["switched"])
        self._status(text)

    def open_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from . import photos

        chunk = self._chunk()
        if chunk is None:
            return
        folder = photos.colour_dir(chunk)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------ 3. depths
    def import_log(self, v: dict | None) -> None:
        from . import depthlog

        chunk = self._need_chunk()
        if chunk is None:
            return
        v = v or (self.pane.depth_values() if self.pane is not None else {"density": depthlog.SEA_WATER, "offset": 0.0})
        path = self._open_file(t("Registro di profondità", "Depth log"),
                               t("Registri (*.csv *.txt *.tsv *.uddf *.xml);;Tutti i file (*)",
                                 "Logs (*.csv *.txt *.tsv *.uddf *.xml);;All files (*)"))
        if not path:
            return
        try:
            log = depthlog.read_log(path, density=v["density"]) if not path.lower().endswith((".uddf", ".xml")) \
                else depthlog.read_log(path)
        except (ValueError, OSError) as exc:
            self._message("warning", str(exc))
            return
        save_log(chunk, log)
        res = match_depths(chunk, log, v.get("offset", 0.0))
        self._changed()
        if not res["matched"]:
            self._message("warning", t("Nessuna foto cade nel tempo del registro ({0} foto con l'ora su {1}): controlla "
                                       "lo scarto di tempo, o stimalo dopo l'allineamento.",
                                       "No photo falls in the time of the log ({0} photos with a time of {1}): check "
                                       "the time offset, or estimate it after the alignment.").format(
                res["timed"], res["photos"]))
        else:
            self._status(t("{0} foto su {1} con la profondità.", "{0} photos of {1} with a depth.").format(
                res["matched"], res["photos"]))

    def match(self, v: dict | None) -> None:
        chunk = self._need_chunk()
        if chunk is None:
            return
        log = load_log(chunk)
        if log is None:
            self._message("warning", t("Importa prima un registro di profondità.", "Import a depth log first."))
            return
        v = v or (self.pane.depth_values() if self.pane is not None else {"offset": 0.0})
        res = match_depths(chunk, log, v.get("offset", 0.0))
        self._changed()
        self._status(t("{0} foto su {1} con la profondità.", "{0} photos of {1} with a depth.").format(
            res["matched"], res["photos"]))

    def estimate_offset(self, v: dict | None) -> None:
        from . import levelling

        chunk = self._need_chunk()
        if chunk is None:
            return
        log = load_log(chunk)
        if log is None:
            self._message("warning", t("Importa prima un registro di profondità.", "Import a depth log first."))
            return
        v = v or (self.pane.depth_values() if self.pane is not None else {"offset": 0.0})

        def job(progress):
            return levelling.offset_search(chunk, log, around=v.get("offset", 0.0), span=900.0, progress=progress)

        def done(res):
            import math

            if not math.isfinite(res["rms"]):
                self._message("warning", t("Nessuno scarto fa accordare le profondità con le camere: controlla che il "
                                           "registro e le foto siano dello stesso giorno.",
                                           "No offset makes the depths agree with the cameras: check that the log and "
                                           "the photos are of the same day."))
                return
            match_depths(chunk, log, res["offset"])
            if self.pane is not None:
                self.pane.set_offset(res["offset"])
            self._changed()
            self._message("info", t("Scarto di tempo stimato: {0} s (scarto quadratico delle profondità {1} cm su {2} "
                                    "camere). Le profondità sono state associate di nuovo.",
                                    "Time offset estimated: {0} s (RMS of the depths {1} cm on {2} cameras). The depths "
                                    "were matched again.").format(number(res["offset"], 2), number(res["rms"] * 100, 1),
                                                                  res["cameras"]))

        self.context.run(t("Stima dello scarto di tempo", "Estimating the time offset"), job, done)

    def level(self, v: dict | None) -> None:
        from . import levelling

        chunk = self._need_chunk()
        if chunk is None:
            return
        v = v or (self.pane.depth_values() if self.pane is not None else {"scale": "depths"})
        scale = "bars" if v.get("scale") == "bars" else None
        positions = (chunk.meta.get("underwater") or {}).get("positions")
        pos = {int(k): tuple(p) for k, p in (positions or {}).get("cameras", {}).items()} if positions else None
        try:
            rep = levelling.level(chunk, scale=scale, positions=pos, positions_crs=(positions or {}).get("crs"))
        except ValueError as exc:
            self._message("warning", str(exc))
            return
        self._changed(photos=True)
        self._message("info", t("Chunk in bolla e in scala su {0} camere: scarto quadratico {1} cm, massimo {2} cm, {3} "
                                "fuori misura. Z = -profondità, in metri; X, Y e la direzione restano arbitrarie{4}. "
                                "Ricostruisci DEM e ortofoto.",
                                "Chunk levelled and scaled on {0} cameras: RMS {1} cm, largest {2} cm, {3} outliers. "
                                "Z = -depth, in metres; X, Y and the heading stay arbitrary{4}. Build the DEM and the "
                                "orthomosaic again.").format(
            rep["cameras"], number(rep["rms"] * 100, 1), number(rep["max"] * 100, 1), rep["outliers"],
            t(" (posti dalle posizioni note)", " (placed by the known positions)") if rep.get("positions") else ""))

    def positions_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import depthlog

        chunk = self._need_chunk()
        if chunk is None:
            return
        v = ParamDialog.ask(self.context.window, t("Posizioni note", "Known positions"), [
            Field("path", t("Tabella", "Table"), "", kind="file_open", filter="CSV (*.csv *.txt)"),
            Field("crs", t("Sistema di riferimento (EPSG, vuoto = locale)", "Reference system (EPSG, empty = local)"),
                  "")],
            intro=t("Posizioni X, Y di alcune camere (USBL, foto posizionate a mano): per nome della foto o per ora "
                    "(interpolate). Dopo la messa in bolla danno la direzione e la posizione orizzontale.",
                    "X, Y positions of some cameras (USBL, photos placed by hand): by photo name or by time "
                    "(interpolated). After the levelling they give the heading and the horizontal position."))
        if not v or not v["path"]:
            return
        try:
            found = depthlog.match_positions(chunk.cameras, v["path"],
                                             offset=((chunk.meta.get("underwater") or {}).get("depths") or {}).get("offset", 0.0))
        except (ValueError, OSError) as exc:
            self._message("warning", str(exc))
            return
        crs = v["crs"].strip()
        if crs and crs.isdigit():
            crs = f"EPSG:{crs}"
        _uw(chunk)["positions"] = {"source": Path(v["path"]).name, "crs": crs or None,
                                   "cameras": {str(k): list(p) for k, p in found.items()}}
        self._changed()
        self._status(t("{0} camere con una posizione nota: «Metti in bolla e in scala» le usa.",
                       "{0} cameras with a known position: 'Level and scale' uses them.").format(len(found)))

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        from . import colour, depthlog, frames, levelling, photos, quality, sources

        parser = argparse.ArgumentParser(prog="ppg plugin underwater", description="Underwater surveys.")
        sub = parser.add_subparsers(dest="what", required=True)

        def project(p):
            p.add_argument("project", help="the .ppg project")
            p.add_argument("--chunk", help="label of the chunk (the active one by default)")
            return p

        project(sub.add_parser("status", help="what the plugin knows of the chunk"))
        p = project(sub.add_parser("score", help="sharpness, contrast and veil of the photos, flagged with thresholds"))
        p.add_argument("--quality", type=float)
        p.add_argument("--contrast", type=float)
        p.add_argument("--veil", type=float)
        p.add_argument("--exclude", action="store_true", help="disable the flagged photos")
        project(sub.add_parser("exclude", help="disable the flagged photos"))
        project(sub.add_parser("include", help="enable again the photos the plugin disabled"))
        p = project(sub.add_parser("frames-sequence", help="frames of an image sequence chosen by motion and sharpness"))
        p.add_argument("folder")
        p.add_argument("--times", help="table of the times of the frames")
        p.add_argument("--overlap", type=float, default=0.8)
        p.add_argument("--min-sharpness", type=float, default=0.6)
        p.add_argument("--step", type=int, default=1)
        p.add_argument("--max-gap", type=float)
        p.add_argument("--dry-run", action="store_true", help="only list the frames chosen")
        p = project(sub.add_parser("frames-video", help="frames of a video with their time"))
        p.add_argument("video")
        p.add_argument("--folder")
        p.add_argument("--overlap", type=float, default=0.8)
        p.add_argument("--start", help="start of the video, YYYY-MM-DD hh:mm:ss (else the creation time of the file)")
        p = project(sub.add_parser("calibration", help="a Kalibr calibration as the sensor of the photos"))
        p.add_argument("yaml")
        p.add_argument("--free", action="store_true", help="not held fixed in the alignment")
        p = project(sub.add_parser("quick", help="quick colour correction (no depth)"))
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--blue", type=float, default=0.0)
        p.add_argument("--no-balance", action="store_true")
        p.add_argument("--no-stretch", action="store_true")
        p.add_argument("--clahe", action="store_true")
        p.add_argument("--use", action="store_true", help="point the cameras to the copies")
        p = project(sub.add_parser("restore", help="colours restored with the ranges (Sea-thru)"))
        p.add_argument("--per", choices=("survey", "photo"), default="survey")
        p.add_argument("--sample", type=int, default=30)
        p.add_argument("--depth-folder", help="depth images in metres made elsewhere (one per photo)")
        p.add_argument("--fov", type=float, help="horizontal field of view of the photos (with --depth-folder)")
        p.add_argument("--use", action="store_true", help="point the cameras to the copies")
        p = project(sub.add_parser("use", help="which photos the cameras point to"))
        p.add_argument("which", choices=("corrected", "original"))
        p = project(sub.add_parser("depth", help="import a depth log (CSV, UDDF) and match it to the photos by time"))
        p.add_argument("log")
        p.add_argument("--offset", type=float, default=0.0, help="seconds added to the camera clock")
        p.add_argument("--fresh", action="store_true", help="fresh water (pressure logs)")
        p.add_argument("--surface", type=float, help="pressure at the surface in hPa (pressure logs)")
        p.add_argument("--tolerance", type=float, default=2.0)
        p = project(sub.add_parser("offset", help="estimate the time offset from the aligned cameras"))
        p.add_argument("--span", type=float, default=900.0)
        p.add_argument("--around", type=float, default=0.0)
        p = project(sub.add_parser("level", help="level, scale and place the chunk by the depths"))
        p.add_argument("--scale", help="'bars' or metres per internal unit (default: from the depths)")
        p.add_argument("--positions", help="known X, Y positions (USBL): by photo name or by time")
        p.add_argument("--crs", help="CRS of the positions (EPSG:...)")
        p = project(sub.add_parser("csv", help="table of the photos: scores, times, depths"))
        p.add_argument("output")
        a = parser.parse_args(list(args))
        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}")
            return 2
        try:
            if a.what == "status":
                print(colour_text(chunk))
                print("\n".join(depth_texts(chunk)))
                fl = quality.flagged(chunk)
                print(f"flagged photos: {len(fl)}")
                return 0
            if a.what == "score":
                limits = quality.thresholds(chunk, a.quality, a.contrast, a.veil)
                rep = quality.score_chunk(chunk, limits=limits, progress=print_progress)
                print(f"{rep['photos']} photos scored, {rep['flagged']} flagged with {limits}")
                for cam in quality.flagged(chunk):
                    s = cam.meta["underwater"]["scores"] or {}
                    print(f"  {cam.label}: quality {s.get('quality')}, contrast {s.get('contrast')}, veil {s.get('veil')}"
                          f" - {' '.join(cam.meta['underwater']['flags'])}")
                if a.exclude:
                    print(f"{quality.exclude(chunk)} photos disabled")
            elif a.what == "exclude":
                print(f"{quality.exclude(chunk)} photos disabled")
            elif a.what == "include":
                print(f"{quality.include(chunk)} photos enabled again")
            elif a.what == "frames-sequence":
                imgs = frames.list_images(a.folder)
                times = frames.read_frame_times(a.times, imgs) if a.times else None
                res = frames.select_sequence(imgs, times, overlap=a.overlap, min_sharpness=a.min_sharpness, step=a.step,
                                             max_gap=a.max_gap, progress=print_progress)
                print(f"{len(res['frames'])} frames chosen of {res['analysed']} ({res['segments']} stretches, "
                      f"{res['unmatched']} breaks)")
                if a.dry_run:
                    for f in res["frames"]:
                        print(f"  {Path(f['path']).name}  sharpness {f['sharpness']}  moved {f['change']}")
                    return 0
                cams = frames.add_frames(chunk, res["frames"], source=Path(a.folder).name)
                print(f"{len(cams)} photos added")
            elif a.what == "frames-video":
                video = Path(a.video)
                folder = Path(a.folder) if a.folder else video.with_name(f"{video.stem}_fotogrammi")
                res = frames.extract_video(video, folder, overlap=a.overlap,
                                           start=depthlog.parse_datetime(a.start) if a.start else None,
                                           progress=print_progress)
                cams = frames.add_frames(chunk, res["frames"], source=video.name)
                print(f"{len(cams)} frames added (start {res['start']})")
            elif a.what == "calibration":
                calib = sources.read_kalibr(a.yaml)
                print(f"{calib['model']} {calib['params']} -> {sources.apply_calibration(chunk, calib, not a.free)} sensors")
            elif a.what == "quick":
                params = colour.QuickParams(alpha=a.alpha, blue=a.blue, balance=not a.no_balance, stretch=not a.no_stretch,
                                            clahe=a.clahe)
                info = photos.quick(chunk, params=params, progress=print_progress)
                print(f"{info['photos']} photos corrected in {info['seconds']} s -> {photos.colour_dir(chunk)}")
                if a.use:
                    print(photos.use(chunk, "corrected"))
            elif a.what == "restore":
                source = photos.range_source(chunk, a.depth_folder, fov=a.fov)
                info = photos.restore(chunk, per=a.per, source=source, sample=a.sample, progress=print_progress)
                print(f"{info['photos']} photos restored in {info['seconds']} s ({info['skipped']} without ranges)")
                if info.get("covered", 1.0) < LOW_COVERAGE:
                    print(low_coverage_text(info["covered"]))
                if info.get("coefficients"):
                    print(json.dumps(info["coefficients"], indent=1))
                if a.use:
                    print(photos.use(chunk, "corrected"))
            elif a.what == "use":
                print(photos.use(chunk, a.which))
            elif a.what == "depth":
                kwargs = {}
                if not a.log.lower().endswith((".uddf", ".xml")):
                    kwargs = {"density": depthlog.FRESH_WATER if a.fresh else depthlog.SEA_WATER, "surface": a.surface}
                log = depthlog.read_log(a.log, **kwargs)
                save_log(chunk, log)
                res = match_depths(chunk, log, a.offset, a.tolerance)
                print(f"log: {json.dumps(log.summary(), default=str)}")
                print(f"{res['matched']} photos of {res['photos']} with a depth ({res['timed']} with a time)")
            elif a.what == "offset":
                log = load_log(chunk)
                if log is None:
                    print("import a depth log first")
                    return 2
                res = levelling.offset_search(chunk, log, around=a.around, span=a.span, progress=print_progress)
                print(f"offset {res['offset']:.2f} s, RMS {res['rms']:.4f} m on {res['cameras']} cameras")
                match_depths(chunk, log, res["offset"])
            elif a.what == "level":
                scale = None if a.scale is None else ("bars" if a.scale == "bars" else float(a.scale))
                pos = None
                if a.positions:
                    pos = depthlog.match_positions(chunk.cameras, a.positions,
                                                   offset=((chunk.meta.get("underwater") or {}).get("depths") or {}).get(
                                                       "offset", 0.0))
                crs = f"EPSG:{a.crs}" if a.crs and a.crs.isdigit() else a.crs
                rep = levelling.level(chunk, scale=scale, positions=pos, positions_crs=crs)
                print(json.dumps({k: v for k, v in rep.items() if k not in ("residuals", "transform")}, indent=1,
                                 default=str))
            elif a.what == "csv":
                print(quality.write_csv(chunk, a.output))
                return 0
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}")
            return 1
        doc.save()
        return 0
