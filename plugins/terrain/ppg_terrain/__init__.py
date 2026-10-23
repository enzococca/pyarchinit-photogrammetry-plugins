"""Terrain - a plugin of pyarchinit-photogrammetry: the analyses of the DEM that WhiteboxTools
(John Lindsay, University of Guelph) makes, on the DEM of the chunk and back into the project.

* the *Terreno* pane of the project dock: hydrology (depressions breached, flow accumulation, the
  stream network and the basins as rasters, lines and polygons), geomorphometry (relative
  topographic position, deviation from the mean height, geomorphons, curvatures, ruggedness,
  hours of daylight), nested depressions and mounds (the level-set method, with depth, volume and
  elongation of every nested polygon), the viewshed from a point; the products in a table, the
  overlay on the orthomosaic or DEM with palette and opacity, the vectors as layers of the view,
  the exports (GeoTIFF, GeoPackage), the QGIS project;
* *Plugin > Terreno* menu actions for the same; *File > Esporta > Analisi del terreno (GeoPackage)*;
* ``ppg plugin terrain hydrology|geomorph|depressions|viewshed|list|export|remove ...``.

WhiteboxTools is a program (MIT): its binary for this computer is downloaded once into the
plugin's folder at the first analysis. The products live in ``<chunk data>/terrain/`` and in
``chunk.meta["terrain"]``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import wbt
from .texts import layer_name, t

__version__ = "0.1.1"
VECTOR_LAYERS = {"hydrology.gpkg": ("reticolo", "bacini"), "depressioni.gpkg": ("depressioni",),
                 "rilievi.gpkg": ("rilievi",)}  # the layers of the view drawn from each GeoPackage


class TerrainPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Terreno", "Terrain")
        self.description = t("Analisi del terreno sul DEM con WhiteboxTools: idrologia, geomorfometria, depressioni e "
                             "rilievi, visibilità",
                             "Terrain analysis of the DEM with WhiteboxTools: hydrology, geomorphometry, depressions "
                             "and mounds, viewshed")
        self.context = None
        self.pane = None
        self.overlay = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        wbt.configure(context)
        context.add_action(t("Idrologia: reticolo e bacini...", "Hydrology: streams and basins..."), self.hydrology_dialog,
                           processing=True)
        context.add_action(t("Geomorfometria...", "Geomorphometry..."), self.geomorph_dialog, processing=True)
        context.add_action(t("Depressioni e rilievi...", "Depressions and mounds..."), self.depressions_dialog, processing=True)
        context.add_action(t("Visibilità da un punto...", "Viewshed from a point..."), self.viewshed_dialog, processing=True)
        context.add_action(t("Analisi del terreno...", "Terrain analysis..."), self.show_pane,
                           tip=t("Il pannello Terreno del dock del progetto", "The Terrain pane of the project dock"))
        context.add_action(t("Nascondi la sovrapposizione", "Hide the overlay"), self.hide_overlay)
        context.add_command("terrain", self.command,
                            help="terrain analyses of the DEM: hydrology, geomorph, depressions, viewshed, list, export, remove")
        context.add_exporter(t("Analisi del terreno (GeoPackage)", "Terrain analyses (GeoPackage)"), ".gpkg", self._export_gpkg)
        window = context.window
        if window is not None:
            self._install_pane(window)

    def deactivate(self) -> None:
        self.hide_overlay()
        window = self.context.window if self.context else None
        if window is not None and self.pane is not None:
            tabs = getattr(window, "left_tabs", None)
            if tabs is not None:
                k = tabs.indexOf(self.pane)
                if k >= 0:
                    tabs.removeTab(k)
            providers = getattr(window, "qgis_providers", [])
            window.qgis_providers = [p for p in providers if p != self._qgis_entries]
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import TerrainPane

        try:
            from pyarchinit_photogrammetry.gui.icons import icon

            ic = icon("relief")
        except Exception:  # noqa: BLE001
            ic = None
        self.pane = TerrainPane()
        tabs = getattr(window, "left_tabs", None)
        if tabs is not None:
            if ic is not None:
                tabs.addTab(self.pane, ic, t("Terreno", "Terrain"))
            else:
                tabs.addTab(self.pane, t("Terreno", "Terrain"))
        self.pane.hydrology_requested.connect(self.hydrology_dialog)
        self.pane.geomorph_requested.connect(self.geomorph_dialog)
        self.pane.depressions_requested.connect(self.depressions_dialog)
        self.pane.viewshed_requested.connect(self.viewshed_dialog)
        self.pane.overlay_requested.connect(self.show_overlay)
        self.pane.opacity_changed.connect(lambda v: self.overlay.set_opacity(v) if self.overlay else None)
        self.pane.hide_requested.connect(self.hide_overlay)
        self.pane.layer_requested.connect(self.layer)
        self.pane.export_requested.connect(self.export_dialog)
        self.pane.remove_requested.connect(self.remove)
        self.pane.qgis_requested.connect(lambda: window.open_in_qgis() if hasattr(window, "open_in_qgis") else None)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        self.refresh_pane()
        original = getattr(window, "refresh_all", None)
        if original is not None and not getattr(window, "_terrain_hooked", False):
            def refresh_all(*args, _orig=original, **kwargs):
                result = _orig(*args, **kwargs)
                self.refresh_pane()
                return result

            window.refresh_all = refresh_all
            window._terrain_hooked = True

    def refresh_pane(self) -> None:
        from . import terrain

        if self.pane is None:
            return
        chunk = self._chunk()
        self.pane.set_products(terrain.entries(chunk) if chunk is not None else [])
        exe = wbt.binary()
        self.pane.set_status(t("WhiteboxTools: {0}", "WhiteboxTools: {0}").format(wbt.version() or exe) if exe
                             else t("WhiteboxTools viene scaricato alla prima analisi (circa 40 MB).",
                                    "WhiteboxTools is downloaded at the first analysis (about 40 MB)."))

    def show_pane(self) -> None:
        window = self.context.window
        if window is None or self.pane is None:
            return
        tabs = getattr(window, "left_tabs", None)
        if tabs is not None:
            tabs.setCurrentWidget(self.pane)
        dock = getattr(window, "d_work", None)
        if dock is not None:
            dock.show()
            dock.raise_()

    def _qgis_entries(self):
        from .export import qgis_entries

        chunk = self._chunk()
        return qgis_entries(chunk) if chunk is not None else ([], [])

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        window = self.context.window
        chunk = getattr(window, "chunk", None) if window is not None else None
        if chunk is not None:
            return chunk
        return doc.chunks[0] if getattr(doc, "chunks", None) else None

    def _need_dem(self):
        from . import terrain

        chunk = self._chunk()
        if chunk is None:
            self._say(t("Nessun chunk", "No chunk"), t("Apri un progetto con un chunk.", "Open a project with a chunk."))
            return None
        try:
            terrain.dem_info(chunk)
        except RuntimeError:
            self._say(t("Manca il DEM", "No DEM"), t("Costruisci prima il DEM del chunk.", "Build the DEM of the chunk first."))
            return None
        return chunk

    def _say(self, title: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is not None and hasattr(window, "_info"):
            window._info(title, text)
        else:
            print(f"{title}: {text}")

    def _done(self, message: str, overlay_id: str | None = None, layer_table: str | None = None) -> None:
        self.refresh_pane()
        window = self.context.window if self.context else None
        if window is not None:
            if hasattr(window, "_dirty"):
                window._dirty(True)
            if layer_table:
                try:
                    self._add_layer(layer_table)
                except Exception as exc:  # noqa: BLE001
                    self.context.log.warning("layer %s: %s", layer_table, exc)
            if overlay_id:
                try:
                    self.show_overlay(overlay_id, "", 0.7)
                except Exception as exc:  # noqa: BLE001
                    self.context.log.warning("overlay %s: %s", overlay_id, exc)
            if hasattr(window, "statusBar"):
                window.statusBar().showMessage(message, 12000)
        else:
            print(message)

    # ------------------------------------------------------------------ the analyses
    def hydrology_dialog(self) -> None:
        from . import terrain

        chunk = self._need_dem()
        if chunk is None:
            return
        v = {"threshold": 50.0, "breach": 10.0}
        window = self.context.window
        if window is not None:
            from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

            v = ParamDialog.ask(window, t("Idrologia", "Hydrology"), [
                Field("threshold", t("Area drenata minima di un corso (m²)", "Least drained area of a stream (m²)"), 50.0,
                      minimum=0.01, maximum=1e9, decimals=2),
                Field("breach", t("Apertura delle depressioni: distanza massima (m)", "Breaching of the depressions: largest distance (m)"),
                      10.0, minimum=0.1, maximum=10000.0, decimals=1)],
                intro=t("Le depressioni del DEM vengono aperte (percorso di minimo costo) o riempite, poi direzioni e accumulo di "
                        "flusso D8, il reticolo delle celle che drenano almeno l'area data e i bacini. Reticolo e bacini "
                        "vanno in terrain/hydrology.gpkg e nei layer.",
                        "The depressions of the DEM are breached (least cost) or filled, then D8 directions and flow "
                        "accumulation, the network of the cells draining at least the given area, the basins. Network "
                        "and basins go into terrain/hydrology.gpkg and into the layers."))
            if not v:
                return
        self.context.run(t("Idrologia", "Hydrology"),
                         lambda p: terrain.hydrology(chunk, threshold_m2=v["threshold"], breach_m=v["breach"], progress=p),
                         lambda r: self._done(t("Reticolo: {0} tratti, {1} bacini", "Network: {0} links, {1} basins")
                                              .format(r["links"], r["basins"]), "flow_accumulation", "reticolo"))

    def geomorph_dialog(self) -> None:
        from . import terrain

        chunk = self._need_dem()
        if chunk is None:
            return
        v = {"metric": "tpi", "size": 10.0, "flatness": 1.0, "cell": 0.0}
        window = self.context.window
        if window is not None:
            from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

            choices = [(k, t(m[2], m[3])) for k, m in terrain.METRICS.items()]
            v = ParamDialog.ask(window, t("Geomorfometria", "Geomorphometry"), [
                Field("metric", t("Metrica", "Metric"), "tpi", choices=choices),
                Field("size", t("Finestra o distanza di ricerca (m)", "Window or search distance (m)"), 10.0,
                      minimum=0.1, maximum=10000.0, decimals=1),
                Field("flatness", t("Geomorphons: angolo di piattezza (gradi)", "Geomorphons: flatness angle (degrees)"), 1.0,
                      minimum=0.0, maximum=45.0, decimals=1),
                Field("cell", t("Cella di calcolo (m, 0 = quella del DEM; 1 m per le forme di un campo)",
                                "Cell of the computation (m, 0 = the DEM's; 1 m for the landforms of a field)"), 0.0,
                      minimum=0.0, maximum=100.0, decimals=3,
                      tip=t("Una cella più grande di quella del DEM: la metrica viene calcolata su una copia del DEM "
                            "ricampionata e riportata sulla griglia del DEM; il prodotto ha la cella nel nome.",
                            "A cell larger than the DEM's: the metric is computed on a resampled copy of the DEM and "
                            "brought back onto the grid of the DEM; the product has the cell in its name."))],
                intro=t("La posizione topografica relativa e lo scarto dalla media usano una finestra della dimensione data; i "
                        "geomorphons una distanza di ricerca; curvature, rugosità e ore di luce non la usano. Le ore di luce "
                        "vogliono un progetto georeferenziato.",
                        "The relative topographic position and the deviation from the mean use a window of the given size; "
                        "the geomorphons a search distance; curvatures, ruggedness and daylight do not use it. The hours of "
                        "daylight need a georeferenced project."))
            if not v:
                return
        self.context.run(t("Geomorfometria", "Geomorphometry"),
                         lambda p: terrain.geomorphometry(chunk, metric=v["metric"], size_m=v["size"], progress=p,
                                                          flatness=v.get("flatness", 1.0), cell_m=v.get("cell", 0.0)),
                         lambda e: self._done(t("{0} pronto", "{0} ready").format(terrain.display_name(e))
                                              + (f" ({e['note']})" if e.get("note") else ""), e["id"]))

    def depressions_dialog(self) -> None:
        from . import depressions

        chunk = self._need_dem()
        if chunk is None:
            return
        v = {"min_area": 1.0, "min_depth": 0.05, "interval": 0.0, "mounds": False}
        window = self.context.window
        if window is not None:
            from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

            v = ParamDialog.ask(window, t("Depressioni e rilievi", "Depressions and mounds"), [
                Field("mounds", t("Rilievi (dossi, tumuli) invece delle depressioni", "Mounds (hills, tumuli) instead of depressions"), False),
                Field("min_area", t("Area minima (m²)", "Least area (m²)"), 1.0, minimum=0.001, maximum=1e9, decimals=3),
                Field("min_depth", t("Profondità (altezza) minima (m)", "Least depth (height) (m)"), 0.05, minimum=0.001, maximum=1000.0, decimals=3),
                Field("interval", t("Intervallo dei livelli (m, 0 = un decimo della profondità)", "Interval of the levels (m, 0 = a tenth of the depth)"),
                      0.0, minimum=0.0, maximum=1000.0, decimals=3)],
                intro=t("Il DEM viene riempito; ogni depressione (dove il riempimento sta sopra il terreno) viene affettata dal "
                        "livello di sfioro verso il fondo: a ogni livello le parti connesse sotto di esso sono le depressioni "
                        "annidate, con profondità, area, volume e allungamento. Per i rilievi lo stesso sul DEM rovesciato. "
                        "I poligoni vanno in un GeoPackage e nei layer.",
                        "The DEM is filled; every depression (where the filling stands above the ground) is sliced from its "
                        "spill level down: at each level the connected parts under it are the nested depressions, with depth, "
                        "area, volume and elongation. Mounds: the same on the inverted DEM. The polygons go into a GeoPackage "
                        "and into the layers."))
            if not v:
                return
        mounds = bool(v["mounds"])
        self.context.run(t("Rilievi", "Mounds") if mounds else t("Depressioni", "Depressions"),
                         lambda p: depressions.find(chunk, min_area_m2=v["min_area"], min_depth=v["min_depth"],
                                                    interval=v["interval"], mounds=mounds, progress=p),
                         lambda r: self._done(t("{0} conche, {1} poligoni annidati", "{0} sinks, {1} nested polygons")
                                              .format(r["sinks"], r["nodes"]), "mound_height" if mounds else "depth",
                                              "rilievi" if mounds else "depressioni"))

    def viewshed_dialog(self) -> None:
        from . import terrain

        chunk = self._need_dem()
        if chunk is None:
            return
        from rasterio.transform import xy

        info = terrain.dem_info(chunk)
        cx, cy = xy(info["transform"], info["height"] / 2.0, info["width"] / 2.0, offset="center")
        v = {"x": cx, "y": cy, "height": 1.7}
        window = self.context.window
        if window is not None:
            from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

            v = ParamDialog.ask(window, t("Visibilità da un punto", "Viewshed from a point"), [
                Field("x", t("X del punto (coordinate del DEM)", "X of the point (DEM coordinates)"), float(cx), minimum=-1e9, maximum=1e9, decimals=3),
                Field("y", t("Y del punto", "Y of the point"), float(cy), minimum=-1e9, maximum=1e9, decimals=3),
                Field("height", t("Altezza dell'occhio sul terreno (m)", "Height of the eye above the ground (m)"), 1.7,
                      minimum=0.0, maximum=1000.0, decimals=2)],
                intro=t("Le celle visibili dal punto (1) e nascoste (0). Le coordinate si leggono nella barra di stato "
                        "muovendo il mouse sulla vista.",
                        "The cells visible from the point (1) and hidden (0). Read the coordinates in the status bar "
                        "moving the mouse over the view."))
            if not v:
                return
        self.context.run(t("Visibilità", "Viewshed"),
                         lambda p: terrain.viewshed(chunk, v["x"], v["y"], height_m=v["height"], progress=p),
                         lambda e: self._done(t("Visibile il {0} % del terreno", "{0} % of the ground visible")
                                              .format(round(e["params"]["visible_fraction"] * 100)), e["id"]))

    # ------------------------------------------------------------------ overlay, layers, exports
    def show_overlay(self, pid: str, palette: str = "", opacity: float = 0.7) -> None:
        from . import terrain
        from .export import limits_text
        from .overlay import Overlay

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        if view.tr is None:
            if hasattr(window, "_show_raster"):
                window._show_raster("dem")
            if hasattr(window, "show_view"):
                window.show_view(view)
        e = terrain.entry(chunk, pid)
        if e is None:
            return
        values, transform, _crs = terrain.load(chunk, pid)
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        lo, hi = self.overlay.show(values, transform, e.get("kind", "other"), palette or None, opacity, product_id=pid)
        if self.pane is not None:
            self.pane.set_limits(limits_text(e, lo, hi))
        if hasattr(window, "show_view"):
            window.show_view(view)

    def overlay_current(self) -> None:
        if self.pane is not None and self.pane.current_id():
            self.show_overlay(self.pane.current_id(), "", self.pane.s_opacity.value() / 100.0)

    def hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()
        if self.pane is not None:
            self.pane.set_limits("")

    def layer(self, pid: str) -> None:
        from . import terrain

        chunk = self._chunk()
        e = terrain.entry(chunk, pid) if chunk is not None else None
        if e is None or not e.get("vector"):
            return
        for table in VECTOR_LAYERS.get(Path(e["vector"]).name, ("depressioni",)):
            self._add_layer(table)

    def _add_layer(self, table: str) -> None:
        """The vector table into the raster view: lines kept across reloads, polygons of the view."""
        from pyarchinit_photogrammetry.io import gpkg

        from . import terrain

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        path = terrain.folder(chunk) / ("hydrology.gpkg" if table in ("reticolo", "bacini") else f"{table}.gpkg")
        if not path.exists():
            return
        rows = gpkg.read_table(path, table)
        layer = layer_name(table)
        if layer in getattr(view, "layers", {}):
            view.remove_layer(layer)
        if table == "reticolo":
            if layer in getattr(window, "line_layers", {}):
                window.line_layers.pop(layer, None)
            for geom, attrs in rows:
                window._add_line_feature(layer, list(geom.coords), "#1f78ff", "", attrs, width=1.5)
            return
        colour = {"bacini": "#b15928", "depressioni": "#e31a1c", "rilievi": "#33a02c"}[table]
        with view.batch():
            for geom, attrs in rows[:5000]:
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                for poly in polys:
                    view.add_polygon(list(poly.exterior.coords), label="", color=colour, fill_alpha=0, layer=layer, attrs=dict(attrs))

    def export_dialog(self, pid: str) -> None:
        from . import terrain
        from .export import export_raster

        window = self.context.window
        chunk = self._chunk()
        e = terrain.entry(chunk, pid) if chunk is not None else None
        if window is None or e is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(window, t("Esporta il raster", "Export the raster"), f"{pid}.tif",
                                              "GeoTIFF (*.tif)")
        if not path:
            return
        export_raster(chunk, pid, path)
        self._say(t("Esportato", "Exported"), str(path))

    def _export_gpkg(self, chunk, path, progress=None):
        from .export import export_gpkg

        return export_gpkg(chunk, path, self.context.window if self.context else None, progress)

    def remove(self, pid: str, confirm: bool = True) -> bool:
        """The product out of the project, after a confirmation in the window (non-native box):
        its raster is deleted, its GeoPackage too when no other product still uses it."""
        from . import terrain

        chunk = self._chunk()
        e = terrain.entry(chunk, pid) if chunk is not None else None
        if e is None:
            return False
        window = self.context.window
        if confirm and window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("Rimuovere «{0}» dal progetto? Il suo file viene cancellato.",
                     "Remove «{0}» from the project? Its file is deleted.").format(terrain.display_name(e))
            others = terrain.sharing(chunk, pid)
            if e.get("vector") and others:
                text += " " + t("Linee e poligoni ({0}) restano: li usa ancora «{1}».",
                                "The lines and polygons ({0}) stay: «{1}» still uses them.").format(
                    Path(e["vector"]).name, terrain.display_name(others[0]))
            elif e.get("vector"):
                text += " " + t("Anche linee e poligoni ({0}) vengono cancellati.",
                                "The lines and polygons ({0}) are deleted too.").format(Path(e["vector"]).name)
            if dialogs.choice(window, "question", t("Rimuovi il prodotto", "Remove the product"), text,
                              [t("Rimuovi", "Remove")]) != 0:
                return False
        if self.overlay is not None and self.overlay.product_id == pid:
            self.hide_overlay()
        gone = terrain.remove(chunk, pid)
        if window is not None:
            view = getattr(window, "ortho_view", None)
            for rel in gone:  # the layers drawn from a GeoPackage that is no more
                for table in VECTOR_LAYERS.get(Path(rel).name, ()):
                    name = layer_name(table)
                    getattr(window, "line_layers", {}).pop(name, None)
                    if view is not None and name in getattr(view, "layers", {}):
                        view.remove_layer(name)
            if hasattr(window, "_dirty"):
                window._dirty(True)
        self.refresh_pane()
        return True

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _project_args(args) -> tuple[list[str], str | None, str | None]:
        """``ppg plugin NAME WHAT PROJECT.ppg ...``: the project given right after the subcommand (or
        with ``--project``) and ``--chunk`` taken out of the arguments; without a project the commands
        work on the document the program has open (the Python console of the window, the tests)."""
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
        """The chunk a command works on: of the project given (opened and made the program's
        document, saved by the command), else of the document open."""
        if project:
            import pyarchinit_photogrammetry as ppg

            doc = ppg.Document.open(project)
            if self.context is not None:
                self.context.app.document = doc
            if label:
                return next((c for c in doc.chunks if c.label == label), None)
            return doc.chunk
        chunk = self._chunk()
        if chunk is not None and label:
            doc = self.context.app.document
            return next((c for c in doc.chunks if c.label == label), None)
        return chunk

    def command(self, *args) -> int:
        from . import depressions, terrain
        from .export import export_gpkg, export_raster

        parser = argparse.ArgumentParser(prog="ppg plugin terrain", description="terrain analyses of the DEM")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("hydrology", help="breached DEM, flow accumulation, stream network and basins")
        p.add_argument("--threshold", type=float, default=50.0, help="least drained area of a stream (m2)")
        p.add_argument("--breach", type=float, default=10.0, help="largest breaching distance (m)")
        p = sub.add_parser("geomorph", help="a geomorphometric metric")
        p.add_argument("metric", choices=list(terrain.METRICS))
        p.add_argument("--size", type=float, default=10.0, help="window side or search distance (m)")
        p.add_argument("--flatness", type=float, default=1.0, help="geomorphons: flatness angle (degrees)")
        p.add_argument("--cell", type=float, default=0.0,
                       help="compute on the DEM resampled to this cell (m), when larger than the DEM's; 0 = the DEM's")
        p = sub.add_parser("depressions", help="nested depressions (or mounds) of the DEM")
        p.add_argument("--min-area", type=float, default=1.0)
        p.add_argument("--min-depth", type=float, default=0.05)
        p.add_argument("--interval", type=float, default=0.0)
        p.add_argument("--mounds", action="store_true")
        p = sub.add_parser("viewshed", help="the cells visible from a point")
        p.add_argument("x", type=float)
        p.add_argument("y", type=float)
        p.add_argument("--height", type=float, default=1.7)
        sub.add_parser("list", help="the products of the chunk")
        p = sub.add_parser("export", help="a product as GeoTIFF, or every vector table as GeoPackage (PATH ending in .gpkg)")
        p.add_argument("id", nargs="?")
        p.add_argument("path")
        p = sub.add_parser("remove", help="a product out of the project (its GeoPackage only when no other product uses it)")
        p.add_argument("id")
        sub.add_parser("install", help="download WhiteboxTools now")
        args, project, label = self._project_args(args)
        ns = parser.parse_args(args)
        if ns.what == "install":
            print(wbt.ensure())
            return 0
        chunk = self._open_for_command(project, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  "no chunk: give the project, ppg plugin terrain " + ns.what + " PROJECT.ppg ...")
            return 1
        doc = self.context.app.document
        if ns.what == "hydrology":
            r = self.context.run("Hydrology", lambda p: terrain.hydrology(chunk, ns.threshold, ns.breach, p))
            print(json.dumps(r, indent=1))
        elif ns.what == "geomorph":
            e = self.context.run("Geomorphometry", lambda p: terrain.geomorphometry(chunk, ns.metric, ns.size, p, ns.flatness, ns.cell))
            print(json.dumps(e, indent=1))
        elif ns.what == "depressions":
            r = self.context.run("Depressions", lambda p: depressions.find(chunk, ns.min_area, ns.min_depth, ns.interval, ns.mounds, progress=p))
            print(json.dumps(r, indent=1))
        elif ns.what == "viewshed":
            e = self.context.run("Viewshed", lambda p: terrain.viewshed(chunk, ns.x, ns.y, ns.height, p))
            print(json.dumps(e, indent=1))
        elif ns.what == "list":
            print(terrain.summary(chunk))
            return 0
        elif ns.what == "export":
            if str(ns.path).lower().endswith(".gpkg"):
                print(export_gpkg(chunk, ns.path))
            else:
                if not ns.id:
                    print("export needs the id of a product for a GeoTIFF")
                    return 1
                print(export_raster(chunk, ns.id, ns.path))
            return 0
        elif ns.what == "remove":
            if terrain.entry(chunk, ns.id) is None:
                print(f"no product {ns.id!r}: see ppg plugin terrain list")
                return 1
            print(json.dumps({"removed": ns.id, "files_deleted": terrain.remove(chunk, ns.id)}))
        if doc is not None and getattr(doc, "path", None):
            doc.save()
        return 0
