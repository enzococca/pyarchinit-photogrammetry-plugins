"""Geophysics - a plugin of pyarchinit-photogrammetry: the grids of a geophysical survey
(magnetometry, resistivity, GPR time slices, EM) beside the photogrammetric products of the
same ground.

* the *Geofisica* pane of the project dock: the grids of the chunk, import (Surfer GRD, XYZ/CSV,
  GeoTIFF and any raster, a folder of GPR slices), georeferencing (the corners of the survey
  grid as markers placed on the orthomosaic, or typed in, or an origin and an azimuth), the
  filters (despike, destripe, clip, low-pass, high-pass, fill, resample), the overlay on the
  orthomosaic or DEM (or on an empty view, framed on the grid) with band, palette and opacity,
  the anomalies drawn on the view, the exports (GeoTIFF, Surfer GRD, GeoPackage of the outlines
  and the anomalies), the QGIS project;
* *Plugin > Geofisica* menu actions for the same; *File > Esporta > Griglie e anomalie geofisiche
  (GeoPackage)*;
* ``ppg plugin geophysics import|list|georef|filter|export ...`` from the command line.

The grids live in ``<chunk data>/geophysics/<id>.tif`` and in ``chunk.meta["geophysics"]``; the
anomalies are polygons drawn with the program's drawing, lent to the plugin (*Disegna anomalia*),
kept in ``chunk.meta["geophysics"]["anomalies"]`` and shown in the layer *Anomalie geofisiche* of
the raster view. The window's parts it uses come from the plugin API of the program
(``request_polygon``, ``on_raster_loaded``, ``show_frame``, ``mark_modified`` since 1.3.2,
``on_refresh`` since 1.3.3): the anomalies and the grid shown are drawn again on every raster the
view shows, the pane follows the active chunk. With the program 1.3.4 the pane is a dock of its own
(``add_pane``, ``show_pane``), the rasters are shown with ``show_raster`` and the program tells the
plugin the anomalies the user deleted (``on_features_deleted``); older programs keep the tab, the
window's own methods and :class:`.anomaly_view.DeleteWatch`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import t

__version__ = "0.1.3"


class GeophysicsPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Geofisica", "Geophysics")
        self.description = t("Griglie geofisiche (magnetometria, resistività, GPR, EM) sul rilievo: import, "
                             "georeferenziazione, filtri, sovrapposizione, esportazioni",
                             "Geophysical grids (magnetometry, resistivity, GPR, EM) on the survey: import, "
                             "georeferencing, filters, overlay, exports")
        self.context = None
        self.pane = None
        self.overlay = None
        self._shown = None  # (grid id, band, palette, opacity) of the grid overlaid: shown again on another raster
        self._watch = None  # the anomalies deleted in the view with the program's Select tool (before 1.3.4)
        self._hooked = False  # the program tells the plugin the anomalies deleted (1.3.4)

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Importa griglia geofisica...", "Import geophysical grid..."), self.import_dialog,
                           tip=t("Surfer GRD, XYZ/CSV, GeoTIFF o una cartella di slice GPR",
                                 "Surfer GRD, XYZ/CSV, GeoTIFF or a folder of GPR slices"))
        context.add_action(t("Griglie geofisiche...", "Geophysical grids..."), self.show_pane,
                           tip=t("Il pannello Geofisica del dock del progetto", "The Geofisica pane of the project dock"))
        context.add_action(t("Sovrapponi la griglia alla vista", "Overlay the grid on the view"), self.overlay_current)
        context.add_action(t("Nascondi la griglia", "Hide the grid"), self.hide_overlay)
        context.add_action(t("Disegna un'anomalia", "Draw an anomaly"), self.draw_anomaly,
                           tip=t("Un poligono nel layer Anomalie geofisiche, con la griglia, la banda e una nota",
                                 "A polygon in the layer Anomalie geofisiche, with the grid, the band and a note"))
        context.add_command("geophysics", self.command,
                            help="geophysical grids: import, list, georef, filter, export")
        context.add_exporter(t("Griglie e anomalie geofisiche (GeoPackage)", "Geophysical grids and anomalies (GeoPackage)"),
                             ".gpkg", self.export_gpkg)
        window = context.window
        if window is not None:
            self._install_pane(window)
            # the layer of the anomalies and the grid shown go with the raster the view leaves:
            # drawn again on the next one
            context.on_raster_loaded(self._raster_loaded)
            # the pane follows the active chunk: another project, chunk, processing step or undo
            context.on_refresh(self.refresh_pane)

    def deactivate(self) -> None:
        self.hide_overlay()
        window = self.context.window if self.context else None
        if self._watch is not None:
            self._watch.detach()
            self._watch.deleteLater()
            self._watch = None
        self._hooked = False  # the callback of on_features_deleted: the context disconnects it
        view = getattr(window, "ortho_view", None) if window is not None else None
        if view is not None:
            from .anomalies import LAYER

            if LAYER in getattr(view, "layers", {}):
                view.remove_layer(LAYER)  # the anomalies stay in the project
        if window is not None and self.pane is not None:
            tabs = getattr(window, "left_tabs", None)
            if tabs is not None and not hasattr(self.context, "add_pane"):  # 1.3.4: the context removes it
                k = tabs.indexOf(self.pane)
                if k >= 0:
                    tabs.removeTab(k)
            providers = getattr(window, "qgis_providers", [])
            window.qgis_providers = [p for p in providers if p != self._qgis_entries]
        self.pane = None

    # ------------------------------------------------------------------ the pane
    def _install_pane(self, window) -> None:
        from .pane import GeophysicsPane

        try:
            from pyarchinit_photogrammetry.gui.icons import icon

            ic = icon("index")
        except Exception:  # noqa: BLE001
            ic = None
        self.pane = GeophysicsPane()
        tabs = getattr(window, "left_tabs", None)
        if hasattr(self.context, "add_pane"):  # program 1.3.4: a dock of its own, beside the workspace
            self.context.add_pane(self.pane, t("Geofisica", "Geophysics"), ic if ic is not None else "index",
                                  area="right")
        elif tabs is not None:
            tabs.addTab(self.pane, ic, t("Geofisica", "Geophysics")) if ic is not None else tabs.addTab(self.pane, t("Geofisica", "Geophysics"))
        self.pane.import_requested.connect(self.import_dialog)
        self.pane.georef_requested.connect(self.georef_dialog)
        self.pane.filter_requested.connect(self.filter_dialog)
        self.pane.overlay_requested.connect(self.show_overlay)
        self.pane.opacity_changed.connect(self._set_opacity)
        self.pane.hide_requested.connect(self.hide_overlay)
        self.pane.anomaly_requested.connect(self.draw_anomaly)
        self.pane.export_requested.connect(self.export_dialog)
        self.pane.remove_requested.connect(self.remove_dialog)
        self.pane.qgis_requested.connect(lambda: window.open_in_qgis() if hasattr(window, "open_in_qgis") else None)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        view = getattr(window, "ortho_view", None)
        if view is not None and hasattr(self.context, "on_features_deleted"):  # program 1.3.4
            from .anomalies import LAYER

            self.context.on_features_deleted(LAYER, self._anomalies_deleted)
            self._hooked = True
            self._draw_anomalies()
        elif view is not None and hasattr(view, "layers_changed"):
            from .anomaly_view import DeleteWatch

            self._watch = DeleteWatch(view, self._chunk)
            self._watch.changed.connect(self._anomalies_changed)
            self._draw_anomalies()
        self.refresh_pane()

    def refresh_pane(self) -> None:
        from .grids import entries

        if self.pane is None:
            return
        chunk = self._chunk()
        self.pane.set_grids(entries(chunk) if chunk is not None else [])

    def show_pane(self) -> None:
        window = self.context.window
        if window is None or self.pane is None:
            return
        if hasattr(self.context, "show_pane"):  # program 1.3.4: also when the user closed or floated it
            self.context.show_pane(self.pane)
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
        doc = self.context.app.document
        return None if doc is None else doc.chunk

    def _message(self, kind: str, text: str) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(self.context.window, kind, self.title, text)

    def _dirty(self) -> None:
        """The project changed: the window shows it and asks to save it at closing."""
        if self.context is not None:
            self.context.mark_modified()

    def _view(self):
        window = self.context.window if self.context is not None else None
        return getattr(window, "ortho_view", None) if window is not None else None

    def _anomalies_changed(self) -> None:
        """Anomalies deleted in the view: the project changed."""
        self._dirty()
        self.refresh_pane()

    def _anomalies_deleted(self, features) -> None:
        """The user deleted anomalies in the view (``context.on_features_deleted``, program 1.3.4):
        out of the project too."""
        from .anomalies import remove

        chunk = self._chunk()
        ids = {f.get("id") for f in features} - {None}
        if chunk is not None and ids and remove(chunk, ids):
            self._anomalies_changed()

    def _draw_anomalies(self) -> None:
        """The layer of the anomalies drawn again from the project (nothing when the view shows nothing)."""
        from .anomaly_view import draw

        if self._hooked:
            draw(self._view(), self._chunk())
        elif self._watch is not None:
            self._watch.drawn(draw(self._view(), self._chunk()))

    def _raster_loaded(self) -> None:
        """Another raster, an empty frame or nothing in the 2D view (``context.on_raster_loaded``):
        the layer of the anomalies and the grid overlaid went with the old one; both are drawn again
        on the new one (the grid only where it lies on it)."""
        if self.pane is None:
            return
        self._draw_anomalies()
        if self._shown is not None:
            self._paint_overlay(*self._shown, quiet=True)

    # ------------------------------------------------------------------ import
    def import_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import grids as grids_mod

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        v = ParamDialog.ask(window, t("Importa griglia geofisica", "Import geophysical grid"), [
            Field("file", t("File (GRD, XYZ, CSV, GeoTIFF, raster)", "File (GRD, XYZ, CSV, GeoTIFF, raster)"), "",
                  kind="file_open", filter="Griglie (*.grd *.xyz *.csv *.txt *.dat *.tif *.tiff *.asc *.img);;Tutti (*)"),
            Field("folder", t("Oppure una cartella di slice GPR", "Or a folder of GPR slices"), "", kind="dir"),
            Field("name", t("Nome (vuoto = nome del file)", "Name (empty = the file name)"), "", kind="text"),
            Field("kind", t("Tipo", "Kind"), "magnetometry",
                  choices=[("magnetometry", t("Magnetometria", "Magnetometry")), ("resistivity", t("Resistività", "Resistivity")),
                           ("gpr", "GPR"), ("em", t("Elettromagnetico", "Electromagnetic")),
                           ("susceptibility", t("Suscettività", "Susceptibility")), ("other", t("Altro", "Other"))]),
            Field("units", t("Unità (vuote = quelle del tipo)", "Units (empty = the kind's)"), "", kind="text"),
            Field("cell", t("Cella per XYZ (m, vuota = dai dati)", "Cell for XYZ (m, empty = from the data)"), None,
                  kind="optional_float",
                  tip=t("Vuota: la distanza fra le letture; se le letture lungo le traverse sono più fitte delle traverse, "
                        "le righe fra due traverse vengono interpolate. Una cella (la distanza fra le traverse): le "
                        "letture di ogni cella vengono mediate.",
                        "Empty: the spacing of the readings; when the readings along the traverses are closer than the "
                        "traverses, the rows between two traverses are interpolated. A cell (the spacing of the "
                        "traverses): the readings of each cell are averaged.")),
            Field("dummy", t("Valore fittizio per XYZ (vuoto = i soliti)", "Dummy value for XYZ (empty = the usual ones)"),
                  None, kind="optional_float",
                  tip=t("Sono sempre assenza di lettura 2047.5 (Geoscan), -9999, 32767, -32768 e il vuoto di Surfer; "
                        "qui un altro valore usato dal tuo programma.",
                        "2047.5 (Geoscan), -9999, 32767, -32768 and Surfer's blank are always no reading; here another "
                        "value your software writes.")),
        ], intro=t("La griglia entra nel progetto come GeoTIFF a virgola mobile; se il file non ha coordinate "
                   "del progetto (Surfer, XYZ in metri locali) va poi georeferenziata.",
                   "The grid enters the project as a floating-point GeoTIFF; a file without the project's "
                   "coordinates (Surfer, XYZ in local metres) is georeferenced afterwards."))
        if not v or not (v["file"] or v["folder"]):
            return
        source = Path(v["folder"] or v["file"])

        def job(progress):
            progress(0.1, t("Lettura della griglia", "Reading the grid"))
            g = grids_mod.read_any(source, kind=v["kind"], cell=v["cell"], dummy=v.get("dummy"))
            if v["name"].strip():
                g.name = v["name"].strip()
            if v["units"].strip():
                g.units = v["units"].strip()
            if g.crs == "LOCAL":
                from .georef import local_to_corner_frame

                local_to_corner_frame(g)
            progress(0.7, t("Scrittura nel progetto", "Writing into the project"))
            return grids_mod.save_grid(chunk, g)

        def done(entry):
            from .export import notes_text

            self._dirty()
            self.refresh_pane()
            st = entry.get("stats") or {}
            notes = notes_text(entry)
            self._message("info", t("Griglia {0}: {1} x {2} celle di {3} m, {4} banda/e, valori {5:.3g} .. {6:.3g} {7}{8}{9}",
                                    "Grid {0}: {1} x {2} cells of {3} m, {4} band(s), values {5:.3g} .. {6:.3g} {7}{8}{9}")
                          .format(entry["name"], entry["cols"], entry["rows"], f"{entry['cell']:g}", entry["bands"],
                                  st.get("min", float("nan")), st.get("max", float("nan")), entry.get("units", ""),
                                  f"; {notes}" if notes else "",
                                  t("; in metri locali: georeferenziala", "; in local metres: georeference it")
                                  if entry.get("crs", "LOCAL") == "LOCAL" else ""))

        window.run_task(t("Griglia geofisica", "Geophysical grid"), job, done, refresh=False, save=True)

    # ------------------------------------------------------------------ georeferencing
    def _marker_positions(self, chunk) -> dict[str, tuple[float, float]]:
        out = {}
        for m in chunk.markers:
            if m.reference.location:
                out[m.label] = (float(m.reference.location[0]), float(m.reference.location[1]))
            elif m.position is not None:
                xy = chunk.to_world(np.asarray([m.position], float))[0]
                out[m.label] = (float(xy[0]), float(xy[1]))
        return out

    def georef_dialog(self, grid_id: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import georef as georef_mod
        from . import grids as grids_mod

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        grid_id = grid_id or (self.pane.selected_id() if self.pane else None)
        if not grid_id:
            return
        if chunk.crs == "LOCAL":
            self._message("warning", t("Il chunk non è georeferenziato: la griglia resterebbe in metri locali.",
                                       "The chunk is not georeferenced: the grid would stay in local metres."))
            return
        g = grids_mod.load_grid(chunk, grid_id)
        markers = self._marker_positions(chunk)
        corners = georef_mod.corners_local(g)
        names = [("", t("(nessuno)", "(none)"))] + [(k, k) for k in sorted(markers)]
        fields = [Field("method", t("Metodo", "Method"), "markers" if markers else "origin",
                        choices=[("markers", t("Angoli della griglia = marker", "Grid corners = markers")),
                                 ("origin", t("Origine e azimut", "Origin and azimuth")),
                                 ("points", t("Coppie di coordinate digitate", "Typed pairs of coordinates"))])]
        for key, label in (("lower-left", t("Marker dell'angolo in basso a sinistra (origine)", "Marker of the lower-left corner (origin)")),
                           ("lower-right", t("Marker dell'angolo in basso a destra", "Marker of the lower-right corner")),
                           ("upper-right", t("Marker dell'angolo in alto a destra", "Marker of the upper-right corner")),
                           ("upper-left", t("Marker dell'angolo in alto a sinistra", "Marker of the upper-left corner"))):
            fields.append(Field(key, label, "", choices=names))
        fields += [Field("fix_scale", t("Scala fissa 1 (griglia in metri)", "Scale fixed at 1 (grid in metres)"), True),
                   Field("x0", t("Origine E (m)", "Origin E (m)"), None, kind="optional_float"),
                   Field("y0", t("Origine N (m)", "Origin N (m)"), None, kind="optional_float"),
                   Field("azimuth", t("Azimut dell'asse x (gradi dal nord)", "Azimuth of the x axis (degrees from north)"), 90.0,
                         minimum=0.0, maximum=360.0, decimals=2),
                   Field("pairs", t("Coppie: 'x_locale y_locale E N' per riga", "Pairs: 'local_x local_y E N' per line"), "",
                         kind="text")]
        v = ParamDialog.ask(window, t("Georeferenzia la griglia", "Georeference the grid"), fields,
                            intro=t("La griglia è in metri locali con l'origine nell'angolo in basso a sinistra "
                                    "({0:g} x {1:g} m). Scegli i marker posati sugli angoli, oppure l'origine e "
                                    "l'azimut, oppure digita le coppie locale-mondo. Una griglia già posata viene "
                                    "posata di nuovo dai suoi metri locali: la nuova posa sostituisce la vecchia.",
                                    "The grid is in local metres with the origin at the lower-left corner "
                                    "({0:g} x {1:g} m). Choose the markers placed on the corners, or the origin and "
                                    "the azimuth, or type the local-world pairs. A grid already placed is placed "
                                    "again from its local metres: the new placement replaces the old one.").format(
                                corners["lower-right"][0], corners["upper-left"][1]))
        if not v:
            return
        try:
            if v["method"] == "origin":
                if v["x0"] is None or v["y0"] is None:
                    raise ValueError(t("origine mancante", "missing origin"))
                georef_mod.place_by_origin(g, float(v["x0"]), float(v["y0"]), float(v["azimuth"]), chunk.crs)
                fit = {"method": "origin", "rmse": 0.0}
            else:
                local, world = [], []
                if v["method"] == "markers":
                    for key, (lx, ly) in corners.items():
                        name = v.get(key) or ""
                        if name and name in markers:
                            local.append((lx, ly))
                            world.append(markers[name])
                else:
                    for line in str(v["pairs"]).splitlines():
                        parts = [p for p in line.replace(",", " ").replace(";", " ").split() if p]
                        if len(parts) >= 4:
                            local.append((float(parts[0]), float(parts[1])))
                            world.append((float(parts[2]), float(parts[3])))
                fit = georef_mod.similarity(local, world, fix_scale=bool(v["fix_scale"]))
                fit["method"] = v["method"]
                georef_mod.place(g, fit, chunk.crs)
        except (ValueError, KeyError) as exc:
            self._message("warning", str(exc))
            return
        g.filters.append({"name": "georef", **{k: fit[k] for k in fit if k != "residuals"}})
        grids_mod.save_grid(chunk, g)
        self._dirty()
        self.refresh_pane()
        self._message("info", t("Griglia {0} posata: rotazione {1:.2f} gradi, scala {2:.4f}, residuo {3:.3f} m",
                                "Grid {0} placed: rotation {1:.2f} degrees, scale {2:.4f}, residual {3:.3f} m")
                      .format(g.name, g.rotation, fit.get("scale", 1.0), fit.get("rmse", 0.0)))

    # ------------------------------------------------------------------ filters
    def filter_dialog(self, grid_id: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import filters as filters_mod
        from . import grids as grids_mod

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        grid_id = grid_id or (self.pane.selected_id() if self.pane else None)
        if not grid_id:
            return
        e = grids_mod.entry_of(chunk, grid_id)
        d = {name: params for name, (_text, params) in filters_mod.FILTERS.items()}  # the defaults, as on the command line
        v = ParamDialog.ask(window, t("Filtri della griglia", "Filters of the grid"), [
            Field("filter", t("Filtro", "Filter"), "destripe",
                  choices=[("despike", t("Despike (spike sostituiti dalla mediana locale)", "Despike (spikes replaced by the local median)")),
                           ("destripe", t("Destripe (media/mediana di ogni traversa a zero)", "Destripe (zero-mean traverse)")),
                           ("clip", t("Clip ai percentili", "Clip to percentiles")),
                           ("lowpass", t("Passa-basso (gaussiano)", "Low-pass (Gaussian)")),
                           ("highpass", t("Passa-alto (residuo dal gaussiano)", "High-pass (residual of the Gaussian)")),
                           ("fill", t("Riempi i vuoti", "Fill the gaps")),
                           ("resample", t("Ricampiona", "Resample"))]),
            Field("size", t("Despike: lato della finestra (celle)", "Despike: window side (cells)"), int(d["despike"]["size"]),
                  minimum=3, maximum=15),
            Field("k", t("Despike: soglia (volte il rumore)", "Despike: threshold (times the noise)"), float(d["despike"]["k"]),
                  minimum=1.0, maximum=20.0, decimals=1),
            Field("axis", t("Destripe: traverse lungo", "Destripe: traverses along"), "rows",
                  choices=[("rows", t("le righe (x)", "the rows (x)")), ("cols", t("le colonne (y)", "the columns (y)"))]),
            Field("stat", t("Destripe: statistica", "Destripe: statistic"), "median",
                  choices=[("median", t("mediana", "median")), ("mean", t("media", "mean"))]),
            Field("low", t("Clip: percentile basso", "Clip: low percentile"), float(d["clip"]["low"]), minimum=0.0,
                  maximum=50.0, decimals=1),
            Field("high", t("Clip: percentile alto", "Clip: high percentile"), float(d["clip"]["high"]), minimum=50.0,
                  maximum=100.0, decimals=1),
            Field("sigma_low", t("Passa-basso: sigma (celle)", "Low-pass: sigma (cells)"), float(d["lowpass"]["sigma"]),
                  minimum=0.5, maximum=100.0, decimals=1),
            Field("sigma_high", t("Passa-alto: sigma (celle)", "High-pass: sigma (cells)"), float(d["highpass"]["sigma"]),
                  minimum=0.5, maximum=100.0, decimals=1),
            Field("distance", t("Riempi: distanza massima (celle)", "Fill: largest distance (cells)"), int(d["fill"]["distance"]),
                  minimum=1, maximum=100),
            Field("cell", t("Ricampiona: cella (m)", "Resample: cell (m)"), float(e["cell"]), minimum=0.01, maximum=100.0, decimals=3),
            Field("copy", t("In una copia (la griglia originale resta)", "Into a copy (the original grid stays)"), True),
        ], intro=t("Il filtro scelto si applica a tutte le bande; i parametri degli altri filtri non contano.",
                   "The chosen filter applies to all the bands; the parameters of the other filters do not count."))
        if not v:
            return
        name = v["filter"]
        params = filters_mod.from_window(name, v)

        def job(progress):
            progress(0.2, t("Filtro {0}", "Filter {0}").format(name))
            g = grids_mod.load_grid(chunk, grid_id)
            out = filters_mod.apply(g, name, **params)
            if v["copy"]:
                out.id = ""
                out.name = f"{g.name} {name}"
            return grids_mod.save_grid(chunk, out)

        def done(entry):
            self._dirty()
            self.refresh_pane()
            if self.overlay is not None and self.overlay.grid_id == entry["id"] and self.overlay.shown():
                self.show_overlay(entry["id"], self.pane.band() if self.pane else 0,
                                  str(self.pane.c_palette.currentData()) if self.pane else None,
                                  self.pane.s_opacity.value() / 100.0 if self.pane else 0.7)

        window.run_task(t("Filtro {0}", "Filter {0}").format(name), job, done, refresh=False, save=True)

    # ------------------------------------------------------------------ overlay
    def show_overlay(self, grid_id: str, band: int = 0, palette: str | None = None, opacity: float = 0.7) -> None:
        """The grid over the raster view; with nothing shown in the view, over the orthomosaic or the
        DEM of the chunk, else on an empty frame the size of the grid (``context.show_frame``). It
        stays over the raster the view shows next, until *Hide*."""
        from .grids import load_grid
        from .overlay import frame_of

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        g = load_grid(chunk, grid_id)
        if not g.georeferenced():
            self._message("warning", t("La griglia è in metri locali: georeferenziala prima.",
                                       "The grid is in local metres: georeference it first."))
            return
        self._shown = None  # the grid shown before is not drawn again on the raster shown now
        if view.tr is None:  # an empty view: the products of the chunk, else a frame for the grid alone
            if hasattr(self.context, "show_raster"):  # program 1.3.4
                self.context.show_raster()
            else:
                for kind, path in (("ortho", getattr(chunk, "orthomosaic_path", None)),
                                   ("dem", getattr(chunk, "dem_path", None))):
                    if path is not None and Path(path).exists() and hasattr(window, "_show_raster"):
                        window._show_raster(kind)
                        break
            if view.tr is None:
                self.context.show_frame(*frame_of(g))
        if self._paint_overlay(grid_id, band, palette, opacity, grid=g) and hasattr(window, "show_view"):
            window.show_view(view)

    def _paint_overlay(self, grid_id: str, band: int, palette: str | None, opacity: float, grid=None,
                       quiet: bool = False) -> bool:
        """The grid painted over what the view shows now, the line of the pane on it; remembered, to
        be painted again over the next raster. ``quiet``: a grid gone from the project, or outside
        the raster shown, is not painted and no message says so (the next raster may hold it)."""
        from .grids import load_grid
        from .overlay import Overlay

        view = self._view()
        chunk = self._chunk()
        if view is None or chunk is None:
            return False
        try:
            g = grid if grid is not None else load_grid(chunk, grid_id)
        except (KeyError, OSError):
            self._shown = None  # the grid left the project
            return False
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        b = min(int(band), g.bands - 1)
        try:
            lo, hi = self.overlay.show(g, b, palette, opacity)
        except RuntimeError as exc:
            if not quiet:  # asked now and not shown: not shown later either
                self._shown = None
                self._message("warning", str(exc))
            return False
        self._shown = (grid_id, b, palette, float(opacity))
        if self.pane is not None:
            self.pane.status.setText(t("{0}: banda {1}, colori da {2:.3g} a {3:.3g} {4}", "{0}: band {1}, colours from {2:.3g} to {3:.3g} {4}")
                                     .format(g.name, g.slices[b] if b < len(g.slices) else b + 1, lo, hi, g.units))
        return True

    def overlay_current(self) -> None:
        if self.pane is None:
            return
        gid = self.pane.selected_id()
        if gid:
            self.show_overlay(gid, self.pane.band(), str(self.pane.c_palette.currentData()), self.pane.s_opacity.value() / 100.0)

    def _set_opacity(self, opacity: float) -> None:
        if self.overlay is not None:
            self.overlay.set_opacity(opacity)
        if self._shown is not None:
            self._shown = (*self._shown[:3], float(opacity))

    def hide_overlay(self) -> None:
        self._shown = None
        if self.overlay is not None:
            self.overlay.remove()

    # ------------------------------------------------------------------ anomalies
    def draw_anomaly(self, grid_id: str | None = None) -> None:
        """Draw an anomaly read on the grid (the one selected in the pane) as a polygon on the view,
        with the program's drawing lent to the plugin (``context.request_polygon``): drag a box, or
        click the vertices and double click or Enter to close; Backspace takes the last vertex back,
        a right click starts again, Esc or another tool of the drawing bar cancels. The polygon, with
        the grid, the band shown and a note, goes into the project and into the layer *Anomalie
        geofisiche*. An empty view shows the grid first."""
        from .grids import entry_of

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not hasattr(window, "request_geometry"):
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        grid_id = grid_id or (self.pane.selected_id() if self.pane else None)
        try:
            entry = entry_of(chunk, grid_id) if grid_id else None
        except KeyError:
            entry = None
        if view.tr is None and entry is not None and entry.get("crs", "LOCAL") != "LOCAL":
            self.show_overlay(entry["id"], self.pane.band() if self.pane else 0,
                              str(self.pane.c_palette.currentData()) if self.pane else None,
                              self.pane.s_opacity.value() / 100.0 if self.pane else 0.7)
        products = [p for p in (getattr(chunk, "orthomosaic_path", None), getattr(chunk, "dem_path", None))
                    if p is not None and Path(p).exists()]
        if view.tr is None and not products:  # the program would show the orthomosaic or the DEM: there is none
            self._message("warning", t("Niente su cui disegnare: mostra l'ortofoto, il DEM o una griglia georeferenziata.",
                                       "Nothing to draw on: show the orthomosaic, the DEM or a georeferenced grid."))
            return
        band = ""
        if entry is not None:
            band = self.pane.band_label() if self.pane is not None and self.pane.selected_id() == entry["id"] else "1"
        self.context.request_polygon(lambda xy: self._anomaly_drawn(xy, entry, band), self._anomaly_cancelled, hint=t(
            "Anomalia: trascina un riquadro, oppure clic per i vertici e doppio clic o Invio per chiudere; Backspace "
            "toglie l'ultimo vertice, tasto destro ricomincia, Esc annulla",
            "Anomaly: drag a box, or click the vertices and double click or Enter to close; Backspace takes the last "
            "vertex back, right click starts again, Esc cancels"))

    def _anomaly_cancelled(self) -> None:
        window = self.context.window
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(t("Anomalia annullata", "Anomaly cancelled"), 5000)

    def _anomaly_drawn(self, xy, entry: dict | None, band: str) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import anomalies

        window = self.context.window
        chunk = self._chunk()
        if chunk is None or self.pane is None:  # no project, or the plugin went away while the polygon was drawn
            return
        v = ParamDialog.ask(window, t("Anomalia geofisica", "Geophysical anomaly"), [
            Field("label", t("Etichetta", "Label"), anomalies.next_label(chunk), kind="text"),
            Field("note", t("Nota (interpretazione)", "Note (interpretation)"), "", kind="text")],
            intro=t("Griglia {0}, banda {1}; area {2:.2f} m². Il poligono va nel progetto e nel layer «{3}».",
                    "Grid {0}, band {1}; area {2:.2f} m². The polygon goes into the project and into the layer «{3}».")
            .format(entry["name"] if entry else "-", band or "-", anomalies.area(xy), anomalies.LAYER))
        if not v:
            return
        a = anomalies.add(chunk, xy, entry, band, str(v.get("note", "")), str(v.get("label", "")).strip())
        self._draw_anomalies()
        self._dirty()
        self.refresh_pane()
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(t("Anomalia {0}: {1:.2f} m²", "Anomaly {0}: {1:.2f} m²")
                                           .format(a["etichetta"], a["area_m2"]), 10000)

    # ------------------------------------------------------------------ export, remove
    def export_dialog(self, grid_id: str | None = None) -> None:
        from PySide6.QtWidgets import QFileDialog

        from .export import export_grid
        from .grids import entry_of

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        grid_id = grid_id or (self.pane.selected_id() if self.pane else None)
        if not grid_id:
            return
        e = entry_of(chunk, grid_id)
        base = (Path(self.context.app.document.path).parent if self.context.app.document.path else Path.home()) / f"{e['name']}.tif"
        path, _f = QFileDialog.getSaveFileName(window, t("Esporta la griglia", "Export the grid"), str(base),
                                               "GeoTIFF (*.tif);;Surfer GRD (*.grd)")
        if not path:
            return
        shown = self.pane is not None and self.pane.selected_id() == e["id"]
        band = self.pane.band() if shown else 0
        out = export_grid(chunk, grid_id, path, band)
        text = t("Scritto {0}", "Written {0}").format(out)
        if out.suffix.lower() == ".grd":
            label = self.pane.band_label() if shown else str(band + 1)
            text += t(" (la banda {0}", " (the band {0}").format(label)
            if abs((float(e.get("rotation", 0.0)) + 180.0) % 360.0 - 180.0) > 1e-9:
                text += t("; la griglia ruotata è stata ricampionata con il nord in alto, perché una griglia Surfer non "
                          "ruota", "; the turned grid was resampled north up, because a Surfer grid has no rotation")
            text += ")"
        self._message("info", text)

    def export_gpkg(self, chunk, path, progress):
        from .export import export_gpkg

        return export_gpkg(chunk, path, self.context.window if self.context else None, progress)

    def remove_dialog(self, grid_id: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import choice

        from .grids import entry_of, remove_grid

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        grid_id = grid_id or (self.pane.selected_id() if self.pane else None)
        if not grid_id:
            return
        e = entry_of(chunk, grid_id)
        k = choice(window, "question", self.title, t("Rimuovere la griglia {0} dal progetto?",
                                                     "Remove the grid {0} from the project?").format(e["name"]),
                   [t("Rimuovi", "Remove")])
        if k != 0:
            return
        if (self.overlay is not None and self.overlay.grid_id == e["id"]) or (self._shown and self._shown[0] == e["id"]):
            self.hide_overlay()
        remove_grid(chunk, e["id"])
        self._dirty()
        self.refresh_pane()

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
        """``ppg plugin geophysics import FILE [--kind K] [--name N] [--cell C] | list | georef GRID
        (--origin X,Y --azimuth A | --pairs 'lx ly E N;...') | filter GRID NAME [k=v ...] [--copy] |
        export GRID OUT``."""
        import argparse

        from . import filters as filters_mod
        from . import georef as georef_mod
        from . import grids as grids_mod
        from .export import export_grid, summary

        parser = argparse.ArgumentParser(prog="ppg plugin geophysics", description=self.command.__doc__)
        sub = parser.add_subparsers(dest="what", required=True)
        p_imp = sub.add_parser("import")
        p_imp.add_argument("source")
        p_imp.add_argument("--kind", default="magnetometry", choices=grids_mod.KINDS)
        p_imp.add_argument("--name", default="")
        p_imp.add_argument("--units", default="")
        p_imp.add_argument("--cell", type=float, default=None,
                           help="XYZ: the cell (m); the readings in a cell averaged. Default: the spacing of the readings")
        p_imp.add_argument("--dummy", type=float, action="append", default=None,
                           help="XYZ: a dummy value besides 2047.5, -9999, 32767, -32768 and Surfer's blank (repeatable)")
        sub.add_parser("list")
        p_geo = sub.add_parser("georef")
        p_geo.add_argument("grid")
        p_geo.add_argument("--origin", default=None, help="E,N of the lower-left corner")
        p_geo.add_argument("--azimuth", type=float, default=90.0)
        p_geo.add_argument("--pairs", default=None, help="'lx ly E N; lx ly E N; ...'")
        p_geo.add_argument("--free-scale", action="store_true")
        p_fil = sub.add_parser("filter")
        p_fil.add_argument("grid")
        p_fil.add_argument("name", choices=list(filters_mod.FILTERS))
        p_fil.add_argument("params", nargs="*", help="k=v")
        p_fil.add_argument("--copy", action="store_true")
        p_exp = sub.add_parser("export")
        p_exp.add_argument("grid")
        p_exp.add_argument("out")
        p_exp.add_argument("--band", type=int, default=0)
        args, project, label = self._project_args(args)
        ns = parser.parse_args(args)
        chunk = self._open_for_command(project, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  "no chunk: give the project, ppg plugin geophysics " + ns.what + " PROJECT.ppg ...")
            return 1
        if ns.what == "import":
            g = grids_mod.read_any(ns.source, kind=ns.kind, cell=ns.cell, dummy=ns.dummy)
            if ns.name:
                g.name = ns.name
            if ns.units:
                g.units = ns.units
            if g.crs == "LOCAL":
                georef_mod.local_to_corner_frame(g)
            e = grids_mod.save_grid(chunk, g)
            print(json.dumps({k: e[k] for k in ("id", "name", "kind", "cell", "rows", "cols", "bands", "crs", "notes")}))
        elif ns.what == "list":
            print(summary(chunk))
        elif ns.what == "georef":
            g = grids_mod.load_grid(chunk, ns.grid)
            if ns.origin:
                x, y = (float(v) for v in ns.origin.split(","))
                georef_mod.place_by_origin(g, x, y, ns.azimuth, chunk.crs)
                fit = {"method": "origin"}
            elif ns.pairs:
                local, world = [], []
                for part in ns.pairs.split(";"):
                    vals = [float(v) for v in part.replace(",", " ").split()]
                    if len(vals) >= 4:
                        local.append(vals[:2])
                        world.append(vals[2:4])
                fit = georef_mod.similarity(local, world, fix_scale=not ns.free_scale)
                fit["method"] = "points"
                georef_mod.place(g, fit, chunk.crs)
            else:
                print("give --origin E,N [--azimuth A] or --pairs")
                return 2
            g.filters.append({"name": "georef", **{k: fit[k] for k in fit if k != "residuals"}})
            grids_mod.save_grid(chunk, g)
            print(json.dumps({"id": g.id, "crs": g.crs, "origin": list(g.origin), "rotation": g.rotation, "cell": g.cell,
                              "rmse": fit.get("rmse")}))
        elif ns.what == "filter":
            g = grids_mod.load_grid(chunk, ns.grid)
            params = {}
            for p in ns.params:
                k, _, v = p.partition("=")
                try:
                    params[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
                except ValueError:
                    params[k] = v
            out = filters_mod.apply(g, ns.name, **params)
            if ns.copy:
                out.id, out.name = "", f"{g.name} {ns.name}"
            e = grids_mod.save_grid(chunk, out)
            print(json.dumps({"id": e["id"], "name": e["name"], "filters": e["filters"], "stats": e["stats"]}))
        elif ns.what == "export":
            print(export_grid(chunk, ns.grid, ns.out, ns.band))
        doc = self.context.app.document
        if doc is not None and doc.path:
            doc.save()
        return 0
