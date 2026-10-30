"""Landscape through time (Landlab) - a plugin of pyarchinit-photogrammetry: how the surveyed
surface changes over the years through soil creep (hillslope diffusion) and running water (erosion
and deposition with SPACE), simulated with Landlab (MIT, CSDMS) on the DEM of the chunk.

* the *Paesaggio nel tempo* pane (on the right): the scenario with its presets and the estimate of
  how long it takes, the runs of the chunk, their layers over the raster view (final surface,
  erosion and deposition, drainage area, snapshots, differences with another chunk), the final
  surface added as a version of the DEM, the comparison with the DEM of another chunk (a later
  phase), the export; the runs in the workspace tree, in the QGIS project and in the raster
  calculator;
* *Plugin > Paesaggio nel tempo* menu actions; *File > Esporta > Paesaggio nel tempo (GeoTIFF)*;
* ``ppg plugin landscape run|estimate|list|show|export|version|compare|remove|install|status``.

Landlab is never imported by the program: it runs in a process of its own (:mod:`.worker`, the
simulation in :mod:`.engine`) with this Python when Landlab is installed here, else in an
environment the plugin installs with uv (:mod:`.environment`); :mod:`.runner` starts the worker
and relays its progress; :mod:`.landscape` reads the DEM, writes the GeoTIFFs and keeps the
registry in ``chunk.meta["landscape"]``. Needs the program 1.3.4 (panes, tree items, rasters shown).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import environment
from .texts import NAME, lang, layer_title, number, t

__version__ = "0.1.0"


class LandscapePlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Come cambia nel tempo la superficie rilevata: creep del suolo e acqua che scorre "
                             "(erosione e deposito) simulati con Landlab sul DEM del chunk",
                             "How the surveyed surface changes over time: soil creep and running water (erosion "
                             "and deposition) simulated with Landlab on the DEM of the chunk")
        self.context = None
        self.pane = None
        self.overlay = None
        self._busy_signal = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        environment.configure(context)
        context.add_action(t("Paesaggio nel tempo...", "Landscape through time..."), self.show_pane,
                           tip=t("Il pannello degli scenari del paesaggio", "The pane of the landscape scenarios"))
        context.add_action(t("Installa Landlab...", "Install Landlab..."), self.install_dialog, processing=True,
                           tip=t("Python 3.12 e Landlab in un ambiente del plugin (circa 500 MB)",
                                 "Python 3.12 and Landlab in an environment of the plugin (about 500 MB)"))
        context.add_action(t("Nascondi la sovrapposizione del paesaggio", "Hide the landscape overlay"),
                           self.hide_overlay)
        context.add_command("landscape", self.command,
                            help="landscape scenarios with Landlab on the DEM: run, estimate, list, show, export, "
                                 "version, compare, remove, install, status")
        context.add_exporter(t("Paesaggio nel tempo (GeoTIFF)", "Landscape through time (GeoTIFF)"), ".tif",
                             self._export_latest)
        if hasattr(context, "add_rasters"):  # program 1.4.0: the layers by name in the raster calculator
            context.add_rasters(self._rasters)
        window = context.window
        if window is not None:
            self._install_pane(window)
            context.on_refresh(self.refresh_pane)
            context.add_tree_items(self._tree_items)

    def deactivate(self) -> None:
        self.hide_overlay()
        signal, self._busy_signal = self._busy_signal, None
        if signal is not None:
            try:
                signal.disconnect(self._busy)
            except (RuntimeError, TypeError):
                pass
        window = self.context.window if self.context else None
        if window is not None:
            providers = getattr(window, "qgis_providers", [])
            window.qgis_providers = [p for p in providers if p != self._qgis_entries]
        self.pane = None

    def _install_pane(self, window) -> None:
        from . import landscape
        from .pane import LandscapePane

        self.pane = pane = LandscapePane()
        self.context.add_pane(pane, t("Paesaggio", "Landscape"), "relief", area="right")
        pane.estimator = lambda params: landscape.estimate(self._chunk(), params, lang())
        pane.run_requested.connect(self.run_scenario)
        pane.install_requested.connect(self.install_dialog)
        pane.source_changed.connect(lambda _key: self._source_changed())
        pane.show_requested.connect(self.show_layer)
        pane.hide_requested.connect(self.hide_overlay)
        pane.opacity_changed.connect(lambda v: self.overlay.set_opacity(v) if self.overlay else None)
        pane.version_requested.connect(self.add_version)
        pane.compare_requested.connect(self.compare_dialog)
        pane.export_requested.connect(self.export_dialog)
        pane.remove_requested.connect(self.remove)
        signal = getattr(getattr(window, "runner", None), "busy_changed", None)
        if signal is not None:  # one step at a time: Run waits while the program works
            signal.connect(self._busy)
            self._busy_signal = signal
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        self.refresh_pane()

    def _busy(self, busy: bool) -> None:
        if self.pane is not None:
            self.pane.set_busy(bool(busy))

    def refresh_pane(self) -> None:
        """The pane follows the active chunk: where Landlab runs, the DEMs to start from, the runs."""
        from . import landscape

        if self.pane is None:
            return
        problem = environment.unsupported()
        kind, _version = environment.where()
        status = t("Non disponibile: ", "Not available: ") + problem if problem else environment.status_text()
        self.pane.set_status(status, can_install=problem is None and kind is None, available=problem is None)
        chunk = self._chunk()
        sources = landscape.dem_sources(chunk) if chunk is not None else []
        self.pane.set_sources(sources, None)
        self.pane.set_dem(self._dem_info(chunk) if sources else None)
        self.pane.set_runs(landscape.runs(chunk) if chunk is not None else [])

    def _dem_info(self, chunk):
        from . import landscape

        try:
            return landscape.dem_info(landscape.dem_path(chunk, self.pane.source() if self.pane else None))
        except (RuntimeError, OSError):
            return None

    def _source_changed(self) -> None:
        if self.pane is not None:
            chunk = self._chunk()
            self.pane.set_dem(self._dem_info(chunk) if chunk is not None else None)

    def show_pane(self) -> None:
        if self.pane is not None and self.context.window is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        window = self.context.window
        chunk = getattr(window, "chunk", None) if window is not None else None
        return chunk if chunk is not None else getattr(doc, "chunk", None)

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
            print(text)
            return
        log = getattr(window, "log_pane", None)
        if log is not None:
            log.info(text)
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 15000)

    def _changed(self) -> None:
        """chunk.meta changed: the project to save, the tree of the workspace filled again."""
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane()

    def _chunk_with_dem(self):
        from . import landscape

        chunk = self._chunk()
        if chunk is None or not landscape.dem_sources(chunk):
            self._message("warning", t("Il chunk non ha un DEM: costruiscilo prima (Flusso di lavoro > Costruisci DEM).",
                                       "The chunk has no DEM: build it first (Workflow > Build DEM)."))
            return None
        return chunk

    def _ready(self, then) -> bool:
        """Landlab can run now. When it is not installed, offer to install it and ``then()`` after; the
        first time in a session, check in the background that it works and ``then()`` after."""
        from . import landscape

        if landscape.WORKER is not None:  # another worker (the tests): nothing to check
            return True
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return False
        if environment.where()[0] is not None:
            if environment.known_usable() or self.context.window is None:
                return True

            def checked(info):
                self.refresh_pane()
                if environment.usable(info):
                    then()
                else:
                    self._message("warning", environment.problem_text(info))

            self.context.run(t("Prova di Landlab", "Trying Landlab"), lambda p: environment.check(p), checked)
            return False
        window = self.context.window
        if window is None:
            print(environment.status_text())
            return False
        from pyarchinit_photogrammetry.gui import dialogs

        text = t("Landlab non è installato. Il plugin lo installa in un ambiente suo, separato dal programma: "
                 "Python 3.12 e Landlab da PyPI con uv, circa 500 MB e qualche minuto. Installarlo ora?",
                 "Landlab is not installed. The plugin installs it in an environment of its own, apart from the "
                 "program: Python 3.12 and Landlab from PyPI with uv, about 500 MB and a few minutes. Install it now?")
        if dialogs.choice(window, "question", self.title, text, [t("Installa", "Install")]) != 0:
            return False
        self._install(then)
        return False

    def install_dialog(self) -> None:
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return
        kind, version = environment.where()
        if kind == "local":
            self._message("info", t("Landlab {0} è già installato per questo Python: il plugin usa quello.",
                                    "Landlab {0} is already installed for this Python: the plugin uses it.").format(version))
            return
        if kind == "environment" and self.context.window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("Landlab è già installato ({0}). Installarlo di nuovo, all'ultima versione?",
                     "Landlab is already installed ({0}). Install it again, at the latest version?").format(version)
            if dialogs.choice(self.context.window, "question", self.title, text,
                              [t("Installa di nuovo", "Install again")]) != 0:
                return
        self._install(None)

    def _install(self, then) -> None:
        def done(_env):
            self.refresh_pane()
            self._status(environment.status_text())
            if then is not None:
                then()

        self.context.run(t("Installazione di Landlab", "Installing Landlab"), lambda p: environment.install(p), done)

    # ------------------------------------------------------------------ a run
    def run_scenario(self, params: dict, name: str | None = None) -> None:
        """The scenario of the pane in the background: the worker runs, the run is registered, its
        erosion and deposition shown over the DEM."""
        from . import landscape

        chunk = self._chunk_with_dem()
        if chunk is None:
            return
        try:
            cost = landscape.estimate(chunk, params, lang())
        except (ValueError, RuntimeError) as exc:
            self._message("warning", str(exc))
            return
        if cost["refuse"]:
            self._message("warning", cost["warning"] + " " + t("Scegli una cella più grande o meno anni.",
                                                                "Choose a larger cell or fewer years."))
            return
        if not self._ready(lambda: self.run_scenario(params, name)):
            return

        def done(e):
            self.context.mark_modified()
            self.context.refresh_tree()
            self.refresh_pane()
            if self.pane is not None:
                self.pane.select_run(e["id"])
            try:
                self.show_layer(e["id"], "dz", self.pane.s_opacity.value() / 100.0 if self.pane else 0.7)
            except Exception as exc:  # noqa: BLE001 - the run is there, the view could not show it
                self.context.log.warning("overlay %s: %s", e["id"], exc)
            self._status(self._done_text(e))

        self.context.run(t("Paesaggio nel tempo", "Landscape through time"),
                         lambda p: landscape.run(chunk, params, p, name=name), done)

    @staticmethod
    def _done_text(e: dict) -> str:
        b = e.get("balance") or {}
        text = t("«{0}» pronto", "'{0}' ready").format(e.get("name"))
        if b.get("eroded_m3") is not None:
            text += ": " + t("eroso {0} m³, depositato {1} m³", "eroded {0} m³, deposited {1} m³").format(
                number(b["eroded_m3"], 1), number(b.get("deposited_m3") or 0, 1))
        return text

    # ------------------------------------------------------------------ the layers over the view
    def show_layer(self, run_id: str, layer: str = "dz", opacity: float = 0.7) -> None:
        from . import landscape
        from .overlay import Overlay

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        e = landscape.entry(chunk, run_id)
        if e is None:
            return
        if view.tr is None:
            self.context.show_raster("dem")
        values, transform, _crs = landscape.load(chunk, run_id, layer)
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        kind = landscape.kind_of(layer)
        lo, hi = self.overlay.show(values, transform, kind, None, opacity, key=(run_id, layer))
        if self.pane is not None:
            if kind == "log":
                text = f"{number(10 ** lo, 1)} - {number(10 ** hi, 0)} m²"
            elif kind == "diverging":
                text = f"± {number(hi, 3)} m"
            else:
                text = f"{number(lo, 2)} - {number(hi, 2)} m"
            self.pane.set_limits(f"{layer_title(layer, e)}: {text}")
        if hasattr(window, "show_view"):
            window.show_view(view)

    def hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()
        if self.pane is not None:
            self.pane.set_limits("")

    # ------------------------------------------------------------------ what a run becomes
    def add_version(self, run_id: str) -> dict | None:
        from . import landscape

        chunk = self._chunk()
        if chunk is None or landscape.entry(chunk, run_id) is None:
            return None
        v = landscape.add_as_dem_version(chunk, run_id)
        self._changed()
        self._status(t("«{0}» fra le versioni del DEM: rendila attiva dal menu Versioni per usarla.",
                       "'{0}' among the versions of the DEM: make it active from the Versions menu to use it.")
                     .format(v["name"]))
        return v

    def compare_dialog(self, run_id: str) -> None:
        chunk = self._chunk()
        doc = self.context.app.document
        if chunk is None or doc is None:
            return
        others = [c for c in doc.chunks if c is not chunk and c.dem_path is not None and Path(c.dem_path).exists()]
        if not others:
            self._message("info", t("Nessun altro chunk ha un DEM: il confronto vuole il DEM di un'altra fase.",
                                    "No other chunk has a DEM: the comparison needs the DEM of another phase."))
            return
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        v = ParamDialog.ask(self.context.window, t("Confronta con un altro chunk", "Compare with another chunk"), [
            Field("chunk", t("Chunk (il suo DEM attivo)", "Chunk (its active DEM)"), others[0].key,
                  choices=[(c.key, c.label) for c in others])],
            intro=t("La superficie finale meno il DEM del chunk scelto, ricampionato sulla griglia del DEM: mediana, "
                    "NMAD e 95° percentile della differenza, e gli stessi numeri prima della simulazione. Se la "
                    "differenza si riduce, i parametri vanno nella direzione giusta.",
                    "The final surface minus the DEM of the chunk chosen, resampled on the grid of the DEM: median, "
                    "NMAD and 95th percentile of the difference, and the same numbers before the simulation. When the "
                    "difference shrinks, the parameters go the right way."))
        if not v:
            return
        other = next(c for c in others if c.key == v["chunk"])
        self.compare(run_id, other)

    def compare(self, run_id: str, other, source: str | None = None):
        """The comparison in the background; the difference shown afterwards."""
        from . import landscape

        chunk = self._chunk()

        def done(c):
            self._changed()
            if self.pane is not None:
                self.pane.select_run(run_id)
            try:
                self.show_layer(run_id, f"compare:{c['chunk_key']}")
            except Exception as exc:  # noqa: BLE001
                self.context.log.warning("overlay %s: %s", run_id, exc)
            s = c["stats"]
            if s.get("cells"):
                self._status(t("Differenza con «{0}»: mediana {1} m, NMAD {2} m", "Difference with '{0}': median {1} m, "
                               "NMAD {2} m").format(c["chunk"], number(s["median"], 3), number(s["nmad"], 3)))

        return self.context.run(t("Confronto con un altro chunk", "Comparison with another chunk"),
                                lambda p: landscape.compare(chunk, run_id, other, source), done)

    def export_dialog(self, run_id: str) -> None:
        from . import landscape

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or landscape.entry(chunk, run_id) is None:
            return
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui import dialogs

        layer = self.pane.current_layer() if self.pane is not None else "dz"
        which = dialogs.choice(window, "question", t("Esporta", "Export"),
                               t("Esportare il livello mostrato o tutti i livelli in un GeoTIFF a più bande?",
                                 "Export the layer shown, or every layer in a multi-band GeoTIFF?"),
                               [t("Il livello mostrato", "The layer shown"), t("Tutti i livelli", "Every layer")])
        if which is None:
            return
        layer = layer if which == 0 else "all"
        default = f"{run_id}_{layer.replace(':', '_')}.tif"
        path, _ = QFileDialog.getSaveFileName(window, t("Esporta il raster", "Export the raster"), default,
                                              "GeoTIFF (*.tif)")
        if not path:
            return
        landscape.export(chunk, run_id, path, layer)
        self._status(t("Esportato: {0}", "Exported: {0}").format(path))

    def _export_latest(self, chunk, path, progress=None):
        from . import landscape

        e = landscape.latest(chunk)
        if e is None:
            raise RuntimeError(t("Nessuna simulazione del paesaggio nel chunk.", "No landscape run in the chunk."))
        return landscape.export(chunk, e["id"], path, "all")

    def remove(self, run_id: str, confirm: bool = True) -> bool:
        """The run out of the project after a confirmation (non-native box): its folder is deleted."""
        from . import landscape

        chunk = self._chunk()
        e = landscape.entry(chunk, run_id) if chunk is not None else None
        if e is None:
            return False
        window = self.context.window
        if confirm and window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("Rimuovere la simulazione «{0}»? I suoi file vengono cancellati.",
                     "Remove the run '{0}'? Its files are deleted.").format(e.get("name"))
            if e.get("dem_version"):
                text += " " + t("La versione del DEM fatta da essa resta.", "The version of the DEM made from it stays.")
            if dialogs.choice(window, "question", t("Rimuovi la simulazione", "Remove the run"), text,
                              [t("Rimuovi", "Remove")]) != 0:
                return False
        if self.overlay is not None and self.overlay.key is not None and self.overlay.key[0] == run_id:
            self.hide_overlay()
        landscape.remove(chunk, run_id)
        if window is not None:
            self._changed()
        return True

    # ------------------------------------------------------------------ the program's lists
    def _tree_items(self, chunk) -> list:
        from . import landscape
        from .pane import summary_text

        rs = landscape.runs(chunk)
        if not rs:
            return []
        children = []
        for e in rs:
            rid = e["id"]
            actions = [(layer_title(layer, e), lambda rid=rid, layer=layer: self.show_layer(rid, layer))
                       for layer in landscape.layers(e)]
            actions.append((t("Aggiungi come versione del DEM", "Add as a version of the DEM"),
                            lambda rid=rid: self.add_version(rid)))
            actions.append((t("Rimuovi", "Remove"), lambda rid=rid: self.remove(rid)))
            children.append({"label": e.get("name", rid), "icon": "dem", "tooltip": summary_text(e),
                             "activate": lambda rid=rid: self.show_layer(rid, "dz"), "actions": actions})
        return [{"label": t("Paesaggio nel tempo", "Landscape through time"), "icon": "relief", "children": children}]

    def _qgis_entries(self):
        from . import landscape

        chunk = self._chunk()
        rasters = []
        if chunk is None:
            return rasters, []
        for e in landscape.runs(chunk):
            for layer in ("final", "dz", "area"):
                path = landscape.path_of(chunk, e, layer)
                if path is None or not path.exists():
                    continue
                item = {"name": f"{t('Paesaggio', 'Landscape')}: {e.get('name')} - {layer_title(layer, e)}",
                        "path": str(path), "band": 1, "opacity": 0.8, "hidden": True}
                if layer == "dz":
                    s = e.get("stats") or {}
                    m = max(abs(s.get("p2", 0.0)), abs(s.get("p98", 0.0)), 1e-6)
                    item.update(ramp="diverging", limits=[-m, m])
                elif layer == "final":
                    item.update(hillshade=True)
                else:
                    item.update(ramp="grey")
                rasters.append(item)
        return rasters, []

    def _rasters(self, chunk) -> list[dict]:
        from . import landscape

        out = []
        for e in landscape.runs(chunk):
            for layer in ("final", "dz", "area"):
                path = landscape.path_of(chunk, e, layer)
                if path is not None and path.exists():
                    out.append({"name": f"landscape_{e['id']}_{layer}".replace("-", "_"),
                                "label": f"{e.get('name')}: {layer_title(layer, e)}", "path": str(path)})
        return out

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _project_args(args) -> tuple[list[str], str | None, str | None]:
        """``ppg plugin landscape WHAT PROJECT.ppg ...``: the project right after the subcommand (or
        ``--project``) and ``--chunk`` taken out of the arguments; without a project the document open."""
        args = [str(a) for a in args]
        rest, project, label, i = [], None, None, 0
        while i < len(args):
            a = args[i]
            if a in ("--project", "--chunk") and i + 1 < len(args):
                project, label = (args[i + 1], label) if a == "--project" else (project, args[i + 1])
                i += 2
                continue
            if project is None and i == 1 and a.lower().endswith(".ppg"):
                project = a
            else:
                rest.append(a)
            i += 1
        return rest, project, label

    def _open_for_command(self, project: str | None, label: str | None):
        if project:
            import pyarchinit_photogrammetry as ppg

            doc = ppg.Document.open(project)
            if self.context is not None:
                self.context.app.document = doc
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        return next((c for c in doc.chunks if c.label == label), None) if label else doc.chunk

    def command(self, *args) -> int:
        from pyarchinit_photogrammetry.core.progress import Cancelled

        from . import engine, landscape
        from .runner import WorkerError

        parser = argparse.ArgumentParser(prog="ppg plugin landscape",
                                         description="landscape scenarios with Landlab on the DEM of a chunk: soil "
                                                     "creep and running water; scenarios, not predictions")
        sub = parser.add_subparsers(dest="what", required=True)
        for name, help_ in (("run", "simulate a scenario and register it"),
                            ("estimate", "how long a scenario would take (nothing is run)")):
            p = sub.add_parser(name, help=help_)
            p.add_argument("--preset", choices=list(engine.PRESETS) + ["custom"], default="mudbrick_tell")
            p.add_argument("--years", type=float, help="years simulated (default: the preset's)")
            p.add_argument("--snapshots", default="", help="intermediate surfaces kept, years: 10,100")
            p.add_argument("--cell", type=float, default=0.0, help="simulation cell (m); 0: automatic")
            p.add_argument("--boundary", choices=list(engine.BOUNDARIES), default="open")
            p.add_argument("--dt", type=float, default=0.0, help="longest step (years); 0: automatic")
            p.add_argument("--source", default="active", help="the DEM: 'active' or the id of a version of the DEM")
            p.add_argument("--no-diffusion", action="store_true", help="no soil creep")
            p.add_argument("--no-water", action="store_true", help="no running water")
            p.add_argument("--D", type=float, help="hillslope diffusivity (m2/yr)")
            for key in ("K_sed", "K_br", "runoff", "soil_depth", "F_f", "phi", "H_star", "v_s", "m_sp", "n_sp"):
                p.add_argument(f"--{key.replace('_', '-')}", dest=key, type=float)
            p.add_argument("--json", action="store_true")
            if name == "run":
                p.add_argument("--name", help="name of the run")
                p.add_argument("--force", action="store_true", help="run also when the estimate is over two hours")
        sub.add_parser("list", help="the runs of the chunk")
        p = sub.add_parser("show", help="a run: parameters, mass balance, comparisons")
        p.add_argument("id")
        p = sub.add_parser("export", help="a layer of a run as GeoTIFF (final, dz, area, snap:YEARS, or all)")
        p.add_argument("id")
        p.add_argument("path")
        p.add_argument("--layer", default="all")
        p = sub.add_parser("version", help="the final surface of a run among the versions of the DEM")
        p.add_argument("id")
        p = sub.add_parser("compare", help="the final surface of a run minus the DEM of another chunk")
        p.add_argument("id")
        p.add_argument("--with", dest="other", required=True, help="the label of the other chunk")
        p.add_argument("--other-source", default="active", help="its DEM: 'active' or the id of one of its versions")
        p = sub.add_parser("remove", help="a run out of the chunk (its files deleted)")
        p.add_argument("id")
        p = sub.add_parser("install", help="install Landlab in an environment of the plugin (uv)")
        p.add_argument("--force", action="store_true", help="also when this Python has Landlab, or again")
        sub.add_parser("status", help="where Landlab runs")
        args, project, label = self._project_args(args)
        ns = parser.parse_args(args)
        if ns.what == "status":
            if environment.where()[0] is not None:
                try:
                    environment.check(again=True)
                except environment.LandlabMissing as exc:
                    print(exc)
                    return 1
            print(environment.status_text())
            return 0 if environment.known_usable() else 1
        if ns.what == "install":
            kind, _version = environment.where()
            if kind is not None and not ns.force:
                print(environment.status_text() + " " + t("Niente da installare (--force per installarlo di nuovo).",
                                                          "Nothing to install (--force to install it again)."))
                return 0
            try:
                print(environment.install(_print_progress))
            except (environment.LandlabMissing, RuntimeError) as exc:
                print(exc)
                return 1
            return 0
        chunk = self._open_for_command(project, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  f"no project: ppg plugin landscape {ns.what} PROJECT.ppg ...")
            return 1
        doc = self.context.app.document if self.context is not None else None
        try:
            if ns.what in ("run", "estimate"):
                params = _command_params(ns)
                if ns.what == "estimate":
                    est = landscape.estimate(chunk, params, lang())
                    print(json.dumps(est, indent=1) if ns.json else _estimate_text(est))
                    return 0
                e = landscape.run(chunk, params, _print_progress, name=ns.name, force=ns.force)
                print(json.dumps(e, indent=1, ensure_ascii=False) if ns.json else
                      self._done_text(e) + f" ({e['id']})")
            elif ns.what == "list":
                print(landscape.summary(chunk))
                return 0
            elif ns.what == "show":
                e = landscape.entry(chunk, ns.id)
                if e is None:
                    print(f"no run {ns.id!r}: see ppg plugin landscape list")
                    return 1
                print(json.dumps(e, indent=1, ensure_ascii=False))
                return 0
            elif ns.what == "export":
                print(landscape.export(chunk, ns.id, ns.path, ns.layer))
                return 0
            elif ns.what == "version":
                print(json.dumps(landscape.add_as_dem_version(chunk, ns.id), indent=1, ensure_ascii=False))
            elif ns.what == "compare":
                other = next((c for c in doc.chunks if c.label == ns.other), None) if doc is not None else None
                if other is None:
                    print(f"no chunk {ns.other!r} in the project")
                    return 1
                print(json.dumps(landscape.compare(chunk, ns.id, other, ns.other_source), indent=1,
                                 ensure_ascii=False))
            elif ns.what == "remove":
                if not landscape.remove(chunk, ns.id):
                    print(f"no run {ns.id!r}: see ppg plugin landscape list")
                    return 1
                print(json.dumps({"removed": ns.id}))
        except (environment.LandlabMissing, WorkerError, RuntimeError, ValueError, KeyError) as exc:
            if isinstance(exc, Cancelled):
                raise
            print(exc)
            return 1
        if doc is not None and getattr(doc, "path", None):
            doc.save()
        return 0


def _print_progress(fraction: float, message: str) -> None:
    print(f"[{fraction * 100:5.1f}%] {message}", flush=True)


def _command_params(ns) -> dict:
    """The scenario of the options of ``run``/``estimate``: the preset's values, those given on top."""
    params = {"preset": ns.preset, "years": ns.years, "cell": ns.cell or None, "boundary": ns.boundary,
              "dt": ns.dt or None, "source": ns.source,
              "snapshots": [float(s) for s in ns.snapshots.replace(";", ",").split(",") if s.strip()]}
    if ns.no_diffusion:
        params["diffusion"] = None
    elif ns.D is not None:
        params["diffusion"] = {"D": ns.D}
    fluvial = {k: getattr(ns, k) for k in ("K_sed", "K_br", "runoff", "soil_depth", "F_f", "phi", "H_star", "v_s",
                                           "m_sp", "n_sp") if getattr(ns, k) is not None}
    if ns.no_water:
        params["fluvial"] = None
    elif fluvial:
        params["fluvial"] = fluvial
    return params


def _estimate_text(est: dict) -> str:
    rows, cols = est["shape"]
    text = (f"{cols} x {rows} cells of {est['cell']:g} m, {est['steps']} steps, {est['diffusion_substeps']} diffusion "
            f"sub-steps, about {est['seconds'] / 60:.1f} minutes")
    if est.get("warning"):
        text += "\n" + est["warning"]
    if est.get("refuse"):
        text += "\n" + t("Troppo lunga: rifiutata senza --force.", "Too long: refused without --force.")
    return text
