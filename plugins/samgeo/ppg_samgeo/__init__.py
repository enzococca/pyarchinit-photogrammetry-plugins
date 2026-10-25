"""Samgeo - a plugin of pyarchinit-photogrammetry: what you write, found in the orthomosaic.

A text - *pietre*, *mattoni*, *tegole e coppi*, or any English phrase - is looked for in the
orthomosaic of the chunk or in any raster the raster calculator reads (a relief visualisation of the
DEM, for mounds and ditches, experimentally): the recipe of segment-geospatial's ``LangSAM``, Grounding
DINO finding the boxes of the text and SAM 2.1 drawing the mask in each box, tile by tile. Every run
is a layer of proposals on the 2D view, checked there (what is wrong is deleted with the program's
*Seleziona*), and then made into stratigraphic units (``context.add_units``: in the live QGIS link,
numbered by *Proponi US e rapporti*), classes of the dense cloud or a GeoPackage; its mask is a
GeoTIFF the raster calculator reads.

The *Samgeo* pane and the *Plugin > Samgeo* menu in the window, the runs in the workspace;
``ppg plugin samgeo install|status|segment|list|export|remove`` from the command line. PyTorch never
runs in the program's process: the models run in a worker (:mod:`.worker`) with a Python of the
environment the AI plugins share (:mod:`.environment`), installed on request.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import environment
from .texts import model_text, number, plural, t

__version__ = "0.2.0"
TEXT_THRESHOLD = 0.25  # how sure Grounding DINO must be that a phrase names a box


class SamgeoPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "Samgeo"
        self.description = t("Trova nell'ortofoto quello che scrivi (pietre, mattoni, tegole...): proposte da "
                             "controllare, poi US, classi della nuvola o GeoPackage",
                             "Find in the orthomosaic what you write (stones, bricks, tiles...): proposals to check, "
                             "then units, classes of the cloud or a GeoPackage")
        self.context = None
        self.pane = None
        self._hooked: set[str] = set()  # the layers whose deletions the program tells the plugin
        self._busy_signal = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Pannello Samgeo", "Samgeo pane"), self.show_pane,
                           tip=t("Cerca con un testo nell'ortofoto; le ricerche del chunk",
                                 "Look for a text in the orthomosaic; the runs of the chunk"))
        context.add_action(t("Installa i modelli del motore selezionato...",
                             "Install the selected engine models..."), self.install_dialog,
                           processing=True,
                           tip=t("Python 3.12 e PyTorch in un ambiente dei plugin AI, con uv (circa 2 GB)",
                                 "Python 3.12 and PyTorch in an environment of the AI plugins, with uv (about 2 GB)"))
        context.add_command("samgeo", self.command,
                            help="what a text names, found in the orthomosaic: install, status, segment, list, export, "
                                 "remove")
        if context.window is not None:
            self._install_pane(context.window)
            # the layers of the runs go with the raster the view leaves: drawn again on the next one
            context.on_raster_loaded(self._raster_loaded)
            # the pane follows the active chunk: another project, chunk, processing step or undo
            context.on_refresh(self.refresh_pane)
            context.add_tree_items(self._tree_items)

    def deactivate(self) -> None:
        signal, self._busy_signal = self._busy_signal, None
        if signal is not None:
            try:
                signal.disconnect(self._busy)
            except (RuntimeError, TypeError):  # already gone with the window
                pass
        view = self._view()
        chunk = self._chunk()
        if view is not None and chunk is not None:
            from .runs import layer, records

            for run in records(chunk):  # the runs stay in the project
                if layer(run) in getattr(view, "layers", {}):
                    view.remove_layer(layer(run))
        self._hooked = set()  # the callbacks of on_features_deleted: the context disconnects them
        if self.pane is not None:
            self.pane.e_token.clear()
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import SamgeoPane

        self.pane = SamgeoPane()
        self.context.add_pane(self.pane, "Samgeo", "sam", area="right")
        self.pane.find_requested.connect(self.find)
        self.pane.install_requested.connect(self.install_dialog)
        self.pane.engine_changed.connect(self.refresh_pane)
        self.pane.visibility_changed.connect(self.set_visible)
        self.pane.zoom_requested.connect(self.zoom)
        self.pane.units_requested.connect(self.units_dialog)
        self.pane.classes_requested.connect(self.classes)
        self.pane.export_requested.connect(self.export_dialog)
        self.pane.remove_requested.connect(self.remove_dialog)
        signal = getattr(getattr(window, "runner", None), "busy_changed", None)
        if signal is not None:  # one step at a time: the buttons wait while the program works
            signal.connect(self._busy)
            self._busy_signal = signal
        self.refresh_pane()
        self._draw_all()

    def _busy(self, busy: bool) -> None:
        if self.pane is not None:
            self.pane.set_busy(bool(busy))

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    def refresh_pane(self) -> None:
        from .runs import records

        if self.pane is None:
            return
        feature = "sam3" if self.pane.engine() == "sam3" else "samgeo"
        self.pane.set_status(environment.status_text(feature), can_install=environment.where(feature) is None)
        chunk = self._chunk()
        self.pane.set_rasters([(r["name"], r["label"]) for r in self._rasters(chunk)])
        self.pane.set_runs(records(chunk))
        for run in records(chunk):
            self._hook(run)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        return None if doc is None else doc.chunk

    def _view(self):
        window = self.context.window if self.context is not None else None
        return getattr(window, "ortho_view", None) if window is not None else None

    def _message(self, kind: str, text: str, details: str = "") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text, details)

    def _status(self, text: str) -> None:
        window = self.context.window if self.context else None
        if self.pane is not None:
            self.pane.set_last(text)
        if window is None:
            return
        log = getattr(window, "log_pane", None)
        if log is not None:
            log.info(text)
        window.statusBar().showMessage(text, 15000)

    def _rasters(self, chunk) -> list[dict]:
        """The rasters of the chunk to look in: the orthomosaic first, the masks of the runs left out."""
        from .runs import folder

        if chunk is None:
            return []
        try:
            found = self.context.rasters(chunk)
        except Exception as exc:  # noqa: BLE001 - an unreadable raster never takes the pane down
            self.context.log.warning("the rasters of the chunk: %s", exc)
            return []
        own = folder(chunk).resolve()
        names = {"orthomosaic": t("Ortofoto", "Orthomosaic"), "dem": "DEM"}
        out = []
        for r in found:
            try:
                if Path(r["path"]).resolve().is_relative_to(own):
                    continue
            except (OSError, ValueError):
                continue
            out.append({**r, "label": names.get(r["name"], r["label"])})
        out.sort(key=lambda r: (r["name"] != "orthomosaic", r["source"] != "product"))
        return out

    def _hook(self, run: dict) -> None:
        """The program tells the plugin the proposals the user deletes in the layer of ``run``."""
        from .runs import layer

        name = layer(run)
        if name not in self._hooked:
            self.context.on_features_deleted(name, lambda feats, name=name: self._deleted(name, feats))
            self._hooked.add(name)

    def _draw_all(self) -> None:
        from .view import draw_all

        chunk = self._chunk()
        if chunk is not None:
            draw_all(self._view(), chunk)

    def _raster_loaded(self) -> None:
        """Another raster, an empty frame or nothing in the 2D view: the layers of the runs went with
        the old one, drawn again on the new one."""
        if self.pane is not None:
            self._draw_all()

    def _changed(self, chunk) -> None:
        """The runs of ``chunk`` changed: the project is to be saved, the pane and the workspace show it."""
        self.context.mark_modified()
        self.refresh_pane()
        self.context.refresh_tree()

    def _deleted(self, name: str, features) -> None:
        """Proposals deleted on the view (the program's *Seleziona* and Canc): out of their run too."""
        from .runs import layer, records, remove_objects

        chunk = self._chunk()
        run = next((r for r in records(chunk) if layer(r) == name), None)
        numbers = {int(f.get("n") or 0) for f in features} - {0}
        if run is None or not numbers:
            return
        gone = remove_objects(chunk, run, numbers)
        view = self._view()
        if view is not None and any(int(f.get("n") or 0) in numbers for f in view.features(name)):
            from PySide6.QtCore import QTimer

            from .view import draw

            # the other parts of an object deleted in one of them: once the program is done with the deletion
            QTimer.singleShot(0, lambda: draw(self._view(), chunk, run))
        if gone:
            self._changed(chunk)
            self._status(t("{0}: {1} proposte eliminate, ne restano {2}", "{0}: {1} proposals deleted, {2} left").format(
                run["name"], gone, run["count"]))

    # ------------------------------------------------------------------ the environment
    def _selected_feature(self) -> str:
        return "sam3" if self.pane is not None and self.pane.engine() == "sam3" else "samgeo"

    def _ready(self, then, feature: str = "samgeo") -> bool:
        """Offer the installation of the selected engine, then resume the pending run."""
        if environment.where(feature) is not None:
            return True
        window = self.context.window
        if window is None:
            print(environment.status_text(feature))
            return False
        from pyarchinit_photogrammetry.gui import dialogs

        text = t("I pacchetti del motore scelto non sono installati. Il plugin li installa con uv in un ambiente "
                 "dei plugin AI, separato dal programma. Il download può richiedere diversi GB e alcuni minuti. "
                 "I pesi si scaricano alla prima ricerca. Installare i pacchetti ora?",
                 "The selected engine's packages are not installed. The plugin installs them with uv in an AI "
                 "plugins environment, apart from the program. The download may take several GB and a few "
                 "minutes. Weights are downloaded on the first run. Install the packages now?")
        if feature == "sam3":
            from .access import notice

            text = notice() + "\n\n" + text
        if dialogs.choice(window, "question", self.title, text, [t("Installa", "Install")]) != 0:
            return False
        self._install(then, feature)
        return False

    def install_dialog(self) -> None:
        feature = self._selected_feature()
        kind = environment.where(feature)
        if kind == "local":
            self._message("info", environment.status_text(feature))
            return
        if kind == "environment" and self.context.window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("I pacchetti del motore scelto sono già installati. Reinstallarli?",
                     "The selected engine's packages are already installed. Reinstall them?")
            if dialogs.choice(self.context.window, "question", self.title, text, [t("Installa di nuovo", "Install again")]) != 0:
                return
        self._install(None, feature)

    def _install(self, then, feature: str = "samgeo") -> None:
        def done(_env):
            self.refresh_pane()
            self._status(environment.status_text(feature))
            if then is not None:
                then()

        self.context.run(t("Installazione dei modelli", "Installing the models"),
                         lambda p: environment.install(p, features=(feature,)), done)

    # ------------------------------------------------------------------ a run
    def find(self, params: dict) -> None:
        """A run from the pane: ``params`` as :meth:`.pane.SamgeoPane.values` gives them."""
        chunk = self._chunk()
        if chunk is None or not params.get("prompt"):
            self._message("warning", t("Scrivi cosa cercare.", "Write what to look for."))
            return
        raster = next((r for r in self._rasters(chunk) if r["name"] == params.get("raster")), None)
        if raster is None:
            self._message("warning", t("Il chunk non ha un'ortofoto o un raster in cui cercare: costruisci prima "
                                       "l'ortofoto.", "The chunk has no orthomosaic or raster to look in: build the "
                                                      "orthomosaic first."))
            return
        feature = "sam3" if params.get("engine") == "sam3" else "samgeo"
        if not self._ready(lambda: self.find(params), feature):
            return
        area = params.get("area", "all")
        if area == "polygon":
            started = self.context.request_polygon(
                lambda xy: self._start(chunk, raster, params, polygon=xy),
                hint=t("Disegna l'area in cui cercare: trascina un riquadro, oppure clic per i vertici e doppio clic "
                       "per chiudere; Esc annulla",
                       "Draw the area to look in: drag a box, or click the vertices and double click to close; Esc "
                       "cancels"))
            if not started:
                self._message("warning", t("Non c'è niente su cui disegnare: mostra prima l'ortofoto.",
                                           "There is nothing to draw on: show the orthomosaic first."))
            return
        bounds = None
        if area == "view":
            view = self._view()
            bounds = view.visible_world_bounds() if view is not None else None
            if bounds is None:
                self._message("warning", t("La vista 2D non mostra niente: mostra prima l'ortofoto.",
                                           "The 2D view shows nothing: show the orthomosaic first."))
                return
        self._start(chunk, raster, params, bounds=bounds)

    def _start(self, chunk, raster: dict, params: dict, bounds=None, polygon=None) -> None:
        from . import segment

        try:
            the_plan = segment.plan(raster["path"], bounds=bounds, resolution=params.get("resolution"), polygon=polygon)
        except ValueError as exc:
            self._message("warning", t("L'area scelta non è nel raster ({0}).", "The area chosen is not in the raster "
                                                                               "({0}).").format(exc))
            return
        info = run_info(params, raster)
        self._status(t("Samgeo: «{0}» in {1} {2} a {3} cm/px", "Samgeo: '{0}' in {1} {2} at {3} cm/px").format(
            info["text"], len(the_plan.tiles), plural(len(the_plan.tiles), t("tassello", "tile"), t("tasselli", "tiles")),
            number(the_plan.resolution * 100, 1)))
        token = self.pane.session_token() if self.pane is not None and info["engine"] == "sam3" else None
        self.context.run(t("Samgeo: «{0}»", "Samgeo: '{0}'").format(info["text"]),
                         lambda p: run_job(chunk, the_plan, info, p, token=token), lambda run: self._found(chunk, run))

    def _found(self, chunk, run: dict) -> None:
        """In the interface thread after a run: into the project, its layer on the view."""
        from .runs import add, layer
        from .view import draw

        add(chunk, run)
        self._hook(run)
        if chunk is self._chunk():
            draw(self._view(), chunk, run)
        self._changed(chunk)
        if self.pane is not None:
            self.pane.select_run(run["id"])
            self.show_pane()
        self._status(t("Samgeo: «{0}», {1} {2} nel layer «{3}» ({4:.0f} s, {5}). Controllale nella vista, elimina quelle "
                       "sbagliate (Seleziona e Canc), poi US, classi o GeoPackage.",
                       "Samgeo: '{0}', {1} {2} in the layer '{3}' ({4:.0f} s, {5}). Check them on the view, delete the "
                       "wrong ones (Select and Del), then units, classes or GeoPackage.").format(
            run["text"], run["count"], plural(run["count"], t("proposta", "proposal"), t("proposte", "proposals")),
            layer(run), run.get("seconds", 0.0), environment.device_name(run.get("device"))))

    # ------------------------------------------------------------------ what the runs become
    def _run_of(self, run_id: str):
        from .runs import record

        chunk = self._chunk()
        run = record(chunk, run_id) if chunk is not None else None
        return chunk, run

    def set_visible(self, run_id: str, visible: bool) -> None:
        from .view import draw

        chunk, run = self._run_of(run_id)
        if run is None or bool(run.get("visible", True)) == bool(visible):
            return
        run["visible"] = bool(visible)
        draw(self._view(), chunk, run)
        self.context.mark_modified()

    def zoom(self, run_id: str) -> None:
        from .runs import layer
        from .view import draw

        chunk, run = self._run_of(run_id)
        view = self._view()
        if run is None or view is None:
            return
        if view.tr is None:
            self.context.show_raster()  # its layers are drawn on it (on_raster_loaded)
        if not run.get("visible", True):
            self.set_visible(run_id, True)
            if self.pane is not None:
                self.pane.set_runs(self._runs(chunk))
        if layer(run) not in view.layers:
            draw(view, chunk, run)
        self.context.window.show_view(view)
        view.zoom_to_layer(layer(run))

    @staticmethod
    def _runs(chunk):
        from .runs import records

        return records(chunk)

    def units_dialog(self, run_id: str) -> None:
        """The proposals of a run (all, or those selected on the view) into stratigraphic units."""
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from .runs import features
        from .view import selected

        chunk, run = self._run_of(run_id)
        if run is None:
            return
        if not hasattr(self.context, "add_units"):
            self._message("warning", t("Per fare le US serve pyarchinit-photogrammetry 1.5.0 o successivo: aggiorna il "
                                       "programma.", "Making units needs pyarchinit-photogrammetry 1.5.0 or later: "
                                                     "update the program."))
            return
        rows = features(chunk, run)
        if not rows:
            self._message("info", t("La ricerca non ha più proposte.", "The run has no proposals left."))
            return
        chosen = selected(self._view(), run)
        made = set(run.get("units_made") or [])
        left = [(g, a) for g, a in rows if int(a.get("n") or 0) not in made]
        which = []
        if chosen:
            which.append(("selected", t("Le {0} selezionate nella vista", "The {0} selected on the view").format(
                len(chosen))))
        if left and made:
            which.append(("left", t("Le {0} non ancora US", "The {0} not yet units").format(len(left))))
        which.append(("all", t("Tutte le {0} proposte", "All the {0} proposals").format(len(rows))
                      + (t(" (anche quelle già US)", " (also those already units)") if made else "")))
        kind = "USM" if any(w in run.get("prompt", "") for w in ("wall", "brick")) else "US"
        v = ParamDialog.ask(self.context.window, t("US da «{0}»", "Units from '{0}'").format(run["name"]), [
            Field("which", t("Quali", "Which"), which[0][0], choices=which),
            Field("tipo", t("Tipo", "Kind"), kind, choices=[(k, k) for k in ("US", "USM", "USR", "USD", "USV")])],
            intro=t("Ogni proposta diventa un'US del chunk attivo, con il testo e il punteggio; senza numero: «Proponi "
                    "US e rapporti» le numera. Modifica > Annulla le toglie tutte insieme. Le proposte già US restano "
                    "nel layer con il solo contorno.",
                    "Every proposal becomes a unit of the active chunk, with the text and the score; without a number: "
                    "'Proponi US e rapporti' numbers them. Edit > Undo takes them all away together. The proposals "
                    "already units stay in the layer as outlines."))
        if not v:
            return
        if v["which"] == "selected":
            rows = [(g, a) for g, a in rows if int(a.get("n") or 0) in chosen]
        elif v["which"] == "left":
            rows = left
        self.make_units(run_id, rows, v["tipo"])

    def make_units(self, run_id: str, rows, tipo: str = "US") -> int:
        from .runs import layer, unit_rows
        from .view import draw

        chunk, run = self._run_of(run_id)
        if run is None or not rows:
            return 0
        geoms, attrs = unit_rows(run, rows)
        made = self.context.add_units(geoms, tipo=tipo, attributes=attrs, chunk=chunk)
        run["units"] = int(run.get("units") or 0) + made
        run["units_made"] = sorted(set(run.get("units_made") or []) | {int(a.get("n") or 0) for _g, a in rows})
        if len(run["units_made"]) >= int(run.get("count") or 0):  # all of them are units: the layer goes out of the way
            run["visible"] = False
        draw(self._view(), chunk, run)  # those made units as outlines
        self._changed(chunk)
        self._status(t("{0} US da «{1}» (tipo {2}); il layer «{3}» resta tra le ricerche",
                       "{0} units from '{1}' (kind {2}); the layer '{3}' stays among the runs").format(
            made, run["name"], tipo, layer(run)))
        return made

    def classes(self, run_id: str) -> None:
        """The points of the dense cloud inside the proposals (all, or those selected) get the class of
        the run."""
        from . import runs
        from .view import selected

        chunk, run = self._run_of(run_id)
        if run is None:
            return
        path = chunk.product_path("point_cloud")
        if path is None or not path.exists():
            self._message("warning", t("Il chunk non ha la nuvola densa: costruiscila prima.",
                                       "The chunk has no dense cloud: build it first."))
            return
        numbers = selected(self._view(), run) or None

        def done(result):
            run["classes"] = {"code": result["code"], "points": result["points"]}
            self._changed(chunk)
            window = self.context.window
            if window is not None and hasattr(window, "show_classes"):
                window.show_classes()
            self._status(t("Classe {0} «{1}»: {2} punti della nuvola densa",
                           "Class {0} '{1}': {2} points of the dense cloud").format(
                result["code"], result["name"], number(result["points"])))

        self.context.run(t("Samgeo: classi della nuvola", "Samgeo: classes of the cloud"),
                         lambda p: runs.classify(chunk, run, numbers, progress=p), done)

    def export_dialog(self, run_ids: list) -> None:
        from PySide6.QtWidgets import QFileDialog

        chunk = self._chunk()
        window = self.context.window
        if chunk is None or window is None:
            return
        doc = self.context.app.document
        base = (Path(doc.path).parent if getattr(doc, "path", None) else Path.home()) / "samgeo.gpkg"
        path, _f = QFileDialog.getSaveFileName(window, t("Esporta le ricerche", "Export the runs"), str(base),
                                               "GeoPackage (*.gpkg)")
        if not path:
            return
        self.export(path, run_ids or None)

    def export(self, path, run_ids=None) -> Path | None:
        from .runs import export

        chunk = self._chunk()
        try:
            out = export(chunk, path, run_ids)
        except (ValueError, FileExistsError, OSError) as exc:
            self._message("warning", str(exc))
            return None
        self._status(t("Scritto {0}", "Written {0}").format(out))
        return out

    def remove_dialog(self, run_id: str) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        chunk, run = self._run_of(run_id)
        if run is None:
            return
        text = t("Rimuovere la ricerca «{0}», il suo layer e i suoi file? Le US già fatte restano.",
                 "Remove the run '{0}', its layer and its files? The units already made stay.").format(run["name"])
        if dialogs.choice(self.context.window, "question", self.title, text, [t("Rimuovi", "Remove")]) != 0:
            return
        self.remove(run_id)

    def remove(self, run_id: str) -> None:
        from .runs import layer, remove

        chunk, run = self._run_of(run_id)
        if run is None:
            return
        view = self._view()
        if view is not None and layer(run) in view.layers:
            view.remove_layer(layer(run))
        remove(chunk, run_id)
        self._changed(chunk)

    # ------------------------------------------------------------------ the workspace
    def _tree_items(self, chunk) -> list[dict]:
        from .runs import records, summary

        items = records(chunk)
        if not items:
            return []
        children = []
        for run in items:
            rid = run["id"]
            children.append({"label": f"{run['name']} ({run.get('count', 0)})", "icon": "sam", "tooltip": summary(run),
                             "activate": lambda rid=rid: self.zoom(rid),
                             "actions": [(t("Inquadra", "Zoom to"), lambda rid=rid: self.zoom(rid)),
                                         (t("US...", "Units..."), lambda rid=rid: self.units_dialog(rid)),
                                         (t("Classi della nuvola", "Classes of the cloud"), lambda rid=rid: self.classes(rid)),
                                         (t("GeoPackage...", "GeoPackage..."), lambda rid=rid: self.export_dialog([rid])),
                                         (t("Rimuovi...", "Remove..."), lambda rid=rid: self.remove_dialog(rid))]})
        return [{"label": "Samgeo", "icon": "sam", "tooltip": t("Le ricerche con un testo del chunk",
                                                                "The runs with a text of the chunk"),
                 "children": children, "activate": self.show_pane}]

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        from pyarchinit_photogrammetry.core.progress import Cancelled

        from . import runs, segment
        from .client import WorkerError

        parser = argparse.ArgumentParser(prog="ppg plugin samgeo",
                                         description="What a text names, found in the orthomosaic (or another raster) "
                                                     "of a chunk with Grounding DINO + SAM 2.1 or SAM 3; every run kept in the "
                                                     "project, its objects in a GeoPackage and a mask.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("install", help="install packages in the AI plugins environment (uv)")
        p.add_argument("--sam3", action="store_true", help="install the optional SAM 3 engine (separate Meta license)")
        p = sub.add_parser("status", help="where the models run")
        p.add_argument("--engine", choices=["open", "sam3"], default="open")
        p = sub.add_parser("access", help="SAM 3 license, access approval and personal Hugging Face authentication")
        p = sub.add_parser("segment", help="look for TEXT in the orthomosaic of the chunk")
        p.add_argument("project")
        p.add_argument("text", help="what to look for (English works best; the ready entries of the pane are translated)")
        p.add_argument("--chunk", help="the label of the chunk (the active one by default)")
        p.add_argument("--raster", default="orthomosaic", help="the name of the raster in the raster calculator")
        p.add_argument("--area", help="X0,Y0,X1,Y1 in the coordinates of the chunk (the whole raster by default)")
        p.add_argument("--resolution", type=float, help="metres per pixel (the raster's, at least 0.01, by default)")
        p.add_argument("--threshold", type=float, default=0.3, help="the least confidence of a box (0-1)")
        p.add_argument("--min-area", type=float, default=0.0, help="m2")
        p.add_argument("--max-area", type=float, default=0.0, help="m2 (0: no limit)")
        p.add_argument("--models", choices=["normal", "fast", "precise"], default="normal", help="open engine only")
        p.add_argument("--engine", choices=["open", "sam3"], default="open",
                       help="SAM 3 needs Meta access approval and HF_TOKEN or a saved Hugging Face login; see access")
        p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="worker device (SAM 3 on MPS is experimental)")
        p.add_argument("--units", action="store_true", help="the objects found become units of the chunk")
        p.add_argument("--json", action="store_true")
        for name, text in (("list", "the runs of the chunk"), ("export", "runs into a GeoPackage"),
                           ("remove", "a run and its files")):
            p = sub.add_parser(name, help=text)
            p.add_argument("project")
            p.add_argument("--chunk")
            if name == "export":
                p.add_argument("output", help="the GeoPackage to write")
                p.add_argument("--run", action="append", help="a run (its name or id); all by default")
            if name == "remove":
                p.add_argument("run", help="its name or id")
            if name == "list":
                p.add_argument("--json", action="store_true")
        ns = parser.parse_args([str(a) for a in args])
        if ns.what == "access":
            from .access import SAM3_LICENSE_URL, SAM3_URL, notice

            print(notice())
            print(SAM3_URL)
            print(SAM3_LICENSE_URL)
            print("https://huggingface.co/settings/tokens")
            print("hf auth login  # use the Hugging Face CLI in the worker environment; or set HF_TOKEN")
            return 0
        if ns.what == "status":
            feature = "sam3" if ns.engine == "sam3" else "samgeo"
            print(environment.status_text(feature))
            return 0 if environment.where(feature) is not None else 1
        if ns.what == "install":
            feature = "sam3" if ns.sam3 else "samgeo"
            if ns.sam3:
                from .access import notice

                print(notice())
            try:
                print(environment.install(_print_progress, features=(feature,)))
            except RuntimeError as exc:
                print(exc)
                return 1
            print(environment.status_text(feature))
            return 0
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        self.context.app.document = doc
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        if ns.what == "list":
            items = runs.records(chunk)
            if ns.json:
                print(json.dumps(items, indent=1))
            for run in [] if ns.json else items:
                print(f"{run['id']}: {runs.summary(run)}, {run.get('time', '')}")
            return 0
        if ns.what == "export":
            chosen = None
            if ns.run:
                found = [runs.find_record(chunk, name) for name in ns.run]
                if None in found:
                    print(f"no run {ns.run[found.index(None)]!r} in the chunk")
                    return 1
                chosen = [r["id"] for r in found]
            try:
                print(runs.export(chunk, ns.output, chosen))
            except (ValueError, FileExistsError) as exc:
                print(exc)
                return 1
            return 0
        if ns.what == "remove":
            run = runs.find_record(chunk, ns.run)
            if run is None:
                print(f"no run {ns.run!r} in the chunk")
                return 1
            runs.remove(chunk, run["id"])
            doc.save()
            print(f"removed {run['name']}")
            return 0
        rasters = self._rasters(chunk)
        raster = next((r for r in rasters if r["name"] == ns.raster), None)
        if raster is None:
            print(f"no raster {ns.raster!r} in the chunk: " + ", ".join(r["name"] for r in rasters))
            return 1
        bounds = None
        if ns.area:
            try:
                bounds = [float(v) for v in ns.area.split(",")]
                assert len(bounds) == 4
            except (ValueError, AssertionError):
                print("--area: X0,Y0,X1,Y1")
                return 1
        params = {"text": ns.text, "prompt": model_text(ns.text), "threshold": ns.threshold, "min_area": ns.min_area,
                  "max_area": ns.max_area, "models": ns.models, "engine": ns.engine, "area": "bounds" if bounds else "all"}
        if ns.engine == "sam3":
            from .access import notice

            print(notice())
        try:
            the_plan = segment.plan(raster["path"], bounds=bounds, resolution=ns.resolution)
            print(f"{len(the_plan.tiles)} tiles at {the_plan.resolution:g} m/px", flush=True)
            run = run_job(chunk, the_plan, run_info(params, raster), _print_progress, device=ns.device)
        except (environment.EnvironmentMissing, WorkerError, ValueError, RuntimeError) as exc:
            if isinstance(exc, Cancelled):
                raise
            print(exc)
            return 1
        runs.add(chunk, run)
        made = 0
        if ns.units and run["count"]:
            geoms, attrs = runs.unit_rows(run, runs.features(chunk, run))
            if hasattr(self.context, "add_units"):
                made = self.context.add_units(geoms, attributes=attrs, chunk=chunk)
                run["units"] = made
            else:
                print("--units needs pyarchinit-photogrammetry 1.5.0 or later")
        doc.save()
        if ns.json:
            print(json.dumps(run, indent=1))
        else:
            print(runs.summary(run) + (f", {made} units" if made else ""))
            print(runs.path_of(chunk, run, "gpkg"))
        return 0


def run_info(params: dict, raster: dict) -> dict:
    """What a run looked for and how, for its record."""
    from .client import MODELS

    engine = params.get("engine") or "open"
    if engine not in ("open", "sam3"):
        raise ValueError("unknown Samgeo engine")
    models = params.get("models") or "normal"
    detector, segmenter = MODELS.get(models, MODELS["normal"])
    from .access import SAM3_MODEL

    model_info = ({"backend": "transformers", "model_id": SAM3_MODEL} if engine == "sam3" else
                  {"models": models, "detector": detector, "segmenter": segmenter})
    return {"text": params.get("text") or params.get("prompt", ""), "prompt": params.get("prompt", ""),
            "engine": engine, **model_info,
            "raster": raster["name"], "raster_label": raster.get("label", raster["name"]),
            "threshold": float(params.get("threshold", 0.3)), "text_threshold": TEXT_THRESHOLD,
            "min_area": float(params.get("min_area") or 0.0), "max_area": float(params.get("max_area") or 0.0),
            "area": params.get("area", "all"), "plugin": __version__}


def run_job(chunk, the_plan, info: dict, progress=None, *, token: str | None = None,
            device: str | None = None) -> dict:
    """A run, in the background in the window: the models in their process, the objects found, the
    files written. Returns the record of the run (into the project with :func:`.runs.add`)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import runs, segment
    from .client import Segmenter

    progress = as_progress(progress)
    t0 = time.monotonic()
    progress(0.0, t("Caricamento dei modelli (download al primo uso)",
                    "Loading the models (downloaded on first use)"))
    with Segmenter(info.get("engine", "open"), info.get("detector", "base"), info.get("segmenter", "small"),
                   progress=progress, token=token, device=device) as seg:
        objects = segment.find(seg, the_plan, info["prompt"], threshold=info["threshold"],
                               text_threshold=info.get("text_threshold", TEXT_THRESHOLD), min_area=info["min_area"],
                               max_area=info["max_area"], progress=progress)
        used_device, versions = seg.device, seg.versions
    run = runs.write(chunk, the_plan, objects, {**info, "device": used_device, "versions": versions})
    run["seconds"] = round(time.monotonic() - t0, 1)
    return run


def _print_progress(fraction: float, message: str) -> None:
    print(f"[{fraction * 100:5.1f}%] {message}", flush=True)
