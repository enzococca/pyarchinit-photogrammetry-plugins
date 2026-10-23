"""Geophysics - a plugin of pyarchinit-photogrammetry: the grids of a geophysical survey
(magnetometry, resistivity, GPR time slices, EM) beside the photogrammetric products of the
same ground.

* the *Geofisica* pane of the project dock: the grids of the chunk, import (Surfer GRD, XYZ/CSV,
  GeoTIFF and any raster, a folder of GPR slices), georeferencing (the corners of the survey
  grid as markers placed on the orthomosaic, or typed in, or an origin and an azimuth), the
  filters (despike, destripe, clip, low-pass, high-pass, fill, resample), the overlay on the
  orthomosaic or DEM with band, palette and opacity, the exports (GeoTIFF, Surfer GRD,
  GeoPackage of the outlines and the anomalies drawn), the QGIS project;
* *Plugin > Geofisica* menu actions for the same; *File > Esporta > Griglie e anomalie geofisiche
  (GeoPackage)*;
* ``ppg plugin geophysics import|list|georef|filter|export ...`` from the command line.

The grids live in ``<chunk data>/geophysics/<id>.tif`` and in ``chunk.meta["geophysics"]``; the
anomalies are polygons drawn in the layer *Anomalie geofisiche* of the raster view (the polygon
tool of the program, layer chosen in the Layer pane).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import t

__version__ = "0.1.1"


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

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Importa griglia geofisica...", "Import geophysical grid..."), self.import_dialog,
                           tip=t("Surfer GRD, XYZ/CSV, GeoTIFF o una cartella di slice GPR",
                                 "Surfer GRD, XYZ/CSV, GeoTIFF or a folder of GPR slices"))
        context.add_action(t("Griglie geofisiche...", "Geophysical grids..."), self.show_pane,
                           tip=t("Il pannello Geofisica del dock del progetto", "The Geofisica pane of the project dock"))
        context.add_action(t("Sovrapponi la griglia alla vista", "Overlay the grid on the view"), self.overlay_current)
        context.add_action(t("Nascondi la griglia", "Hide the grid"), self.hide_overlay)
        context.add_command("geophysics", self.command,
                            help="geophysical grids: import, list, georef, filter, export")
        context.add_exporter(t("Griglie e anomalie geofisiche (GeoPackage)", "Geophysical grids and anomalies (GeoPackage)"),
                             ".gpkg", self.export_gpkg)
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
        if tabs is not None:
            tabs.addTab(self.pane, ic, t("Geofisica", "Geophysics")) if ic is not None else tabs.addTab(self.pane, t("Geofisica", "Geophysics"))
        self.pane.import_requested.connect(self.import_dialog)
        self.pane.georef_requested.connect(self.georef_dialog)
        self.pane.filter_requested.connect(self.filter_dialog)
        self.pane.overlay_requested.connect(self.show_overlay)
        self.pane.opacity_changed.connect(lambda v: self.overlay.set_opacity(v) if self.overlay else None)
        self.pane.hide_requested.connect(self.hide_overlay)
        self.pane.export_requested.connect(self.export_dialog)
        self.pane.remove_requested.connect(self.remove_dialog)
        self.pane.qgis_requested.connect(lambda: window.open_in_qgis() if hasattr(window, "open_in_qgis") else None)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        self.refresh_pane()
        # the pane follows the project: after every refresh of the window
        original = getattr(window, "refresh_all", None)
        if original is not None and not getattr(window, "_geophysics_hooked", False):
            def refresh_all(*args, _orig=original, **kwargs):
                result = _orig(*args, **kwargs)
                self.refresh_pane()
                return result

            window.refresh_all = refresh_all
            window._geophysics_hooked = True

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
        window = self.context.window
        if window is not None and hasattr(window, "_dirty"):
            window._dirty(True)

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
                  kind="optional_float"),
        ], intro=t("La griglia entra nel progetto come GeoTIFF a virgola mobile; se il file non ha coordinate "
                   "del progetto (Surfer, XYZ in metri locali) va poi georeferenziata.",
                   "The grid enters the project as a floating-point GeoTIFF; a file without the project's "
                   "coordinates (Surfer, XYZ in local metres) is georeferenced afterwards."))
        if not v or not (v["file"] or v["folder"]):
            return
        source = Path(v["folder"] or v["file"])

        def job(progress):
            progress(0.1, t("Lettura della griglia", "Reading the grid"))
            g = grids_mod.read_any(source, kind=v["kind"], cell=v["cell"])
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
            self._dirty()
            self.refresh_pane()
            st = entry.get("stats") or {}
            self._message("info", t("Griglia {0}: {1} x {2} celle di {3} m, {4} banda/e, valori {5:.3g} .. {6:.3g} {7}{8}",
                                    "Grid {0}: {1} x {2} cells of {3} m, {4} band(s), values {5:.3g} .. {6:.3g} {7}{8}")
                          .format(entry["name"], entry["cols"], entry["rows"], f"{entry['cell']:g}", entry["bands"],
                                  st.get("min", float("nan")), st.get("max", float("nan")), entry.get("units", ""),
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
                                    "l'azimut, oppure digita le coppie locale-mondo.",
                                    "The grid is in local metres with the origin at the lower-left corner "
                                    "({0:g} x {1:g} m). Choose the markers placed on the corners, or the origin and "
                                    "the azimuth, or type the local-world pairs.").format(corners["lower-right"][0],
                                                                                           corners["upper-left"][1]))
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
        v = ParamDialog.ask(window, t("Filtri della griglia", "Filters of the grid"), [
            Field("filter", t("Filtro", "Filter"), "destripe",
                  choices=[("despike", t("Despike (spike sostituiti dalla mediana locale)", "Despike (spikes replaced by the local median)")),
                           ("destripe", t("Destripe (media/mediana di ogni traversa a zero)", "Destripe (zero-mean traverse)")),
                           ("clip", t("Clip ai percentili", "Clip to percentiles")),
                           ("lowpass", t("Passa-basso (gaussiano)", "Low-pass (Gaussian)")),
                           ("highpass", t("Passa-alto (residuo dal gaussiano)", "High-pass (residual of the Gaussian)")),
                           ("fill", t("Riempi i vuoti", "Fill the gaps")),
                           ("resample", t("Ricampiona", "Resample"))]),
            Field("size", t("Despike: lato della finestra (celle)", "Despike: window side (cells)"), 3, minimum=3, maximum=15),
            Field("k", t("Despike: soglia (volte il rumore)", "Despike: threshold (times the noise)"), 4.0, minimum=1.0, maximum=20.0, decimals=1),
            Field("axis", t("Destripe: traverse lungo", "Destripe: traverses along"), "rows",
                  choices=[("rows", t("le righe (x)", "the rows (x)")), ("cols", t("le colonne (y)", "the columns (y)"))]),
            Field("stat", t("Destripe: statistica", "Destripe: statistic"), "median",
                  choices=[("median", t("mediana", "median")), ("mean", t("media", "mean"))]),
            Field("low", t("Clip: percentile basso", "Clip: low percentile"), 1.0, minimum=0.0, maximum=50.0, decimals=1),
            Field("high", t("Clip: percentile alto", "Clip: high percentile"), 99.0, minimum=50.0, maximum=100.0, decimals=1),
            Field("sigma", t("Passa-basso/alto: sigma (celle)", "Low/high-pass: sigma (cells)"), 3.0, minimum=0.5, maximum=100.0, decimals=1),
            Field("distance", t("Riempi: distanza massima (celle)", "Fill: largest distance (cells)"), 3, minimum=1, maximum=100),
            Field("cell", t("Ricampiona: cella (m)", "Resample: cell (m)"), float(e["cell"]), minimum=0.01, maximum=100.0, decimals=3),
            Field("copy", t("In una copia (la griglia originale resta)", "Into a copy (the original grid stays)"), True),
        ], intro=t("Il filtro scelto si applica a tutte le bande; i parametri degli altri filtri non contano.",
                   "The chosen filter applies to all the bands; the parameters of the other filters do not count."))
        if not v:
            return
        name = v["filter"]
        params = {k: v[k] for k in filters_mod.FILTERS[name][1]}
        if name == "clip":
            params["absolute"] = False

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
            if self.overlay is not None and self.overlay.grid_id == entry["id"]:
                self.show_overlay(entry["id"], self.pane.band() if self.pane else 0,
                                  str(self.pane.c_palette.currentData()) if self.pane else None,
                                  self.pane.s_opacity.value() / 100.0 if self.pane else 0.7)

        window.run_task(t("Filtro {0}", "Filter {0}").format(name), job, done, refresh=False, save=True)

    # ------------------------------------------------------------------ overlay
    def show_overlay(self, grid_id: str, band: int = 0, palette: str | None = None, opacity: float = 0.7) -> None:
        from .grids import load_grid
        from .overlay import Overlay

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
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        try:
            lo, hi = self.overlay.show(g, min(band, g.bands - 1), palette, opacity)
        except RuntimeError as exc:
            self._message("warning", str(exc))
            return
        if self.pane is not None:
            self.pane.status.setText(t("{0}: banda {1}, colori da {2:.3g} a {3:.3g} {4}", "{0}: band {1}, colours from {2:.3g} to {3:.3g} {4}")
                                     .format(g.name, band + 1, lo, hi, g.units))
        if hasattr(window, "show_view"):
            window.show_view(view)

    def overlay_current(self) -> None:
        if self.pane is None:
            return
        gid = self.pane.selected_id()
        if gid:
            self.show_overlay(gid, self.pane.band(), str(self.pane.c_palette.currentData()), self.pane.s_opacity.value() / 100.0)

    def hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()

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
        out = export_grid(chunk, grid_id, path)
        self._message("info", t("Scritto {0}", "Written {0}").format(out))

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
        k = choice(window, "question", self.title, t("Rimuovere la griglia {0} dal progetto?", "Remove the grid {0} from the project?").format(e["name"]),
                   [t("Rimuovi", "Remove"), t("Annulla", "Cancel")], default=1)
        if k != 0:
            return
        if self.overlay is not None and self.overlay.grid_id == e["id"]:
            self.hide_overlay()
        remove_grid(chunk, e["id"])
        self._dirty()
        self.refresh_pane()

    # ------------------------------------------------------------------ command line
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
        p_imp.add_argument("--cell", type=float, default=None)
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
        ns = parser.parse_args(list(args))
        chunk = self._chunk()
        if chunk is None:
            print("no project open")
            return 1
        if ns.what == "import":
            g = grids_mod.read_any(ns.source, kind=ns.kind, cell=ns.cell)
            if ns.name:
                g.name = ns.name
            if ns.units:
                g.units = ns.units
            if g.crs == "LOCAL":
                georef_mod.local_to_corner_frame(g)
            e = grids_mod.save_grid(chunk, g)
            print(json.dumps({k: e[k] for k in ("id", "name", "kind", "cell", "rows", "cols", "bands", "crs")}))
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
