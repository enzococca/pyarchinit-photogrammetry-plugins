"""Sentinel-2 - a plugin of pyarchinit-photogrammetry: time series of Sentinel-2 over the area of
a survey, to look for crop marks (buried walls and ditches seen through the vegetation above
them, weaker or greener in some seasons).

* the *Sentinel-2* pane of the project dock: the area (the footprint of the chunk's DEM or
  orthomosaic, or a polygon drawn on the view, with a margin), the search in a STAC catalog
  (Earth Search by default: public Cloud-Optimized GeoTIFFs, no key), the acquisitions found
  with their cloud cover and the clear part of the area, the reading of the window of the area
  only (clouds, cirrus and shadows masked with the scene classification, cached: a second look
  needs no network), NDVI, NDRE and NDMI per date, the series over the area per date and per
  month with its chart, the overlay of a date or a period (or natural colour) on the
  orthomosaic, the crop-mark maps (the index minus its smooth background, as a z-score) and
  the composites of a period, the exports (GeoTIFF stack, CSV, PNG), the QGIS project;
* *Plugin > Sentinel-2* menu actions; *File > Esporta > Serie temporale Sentinel-2 (CSV)* and
  *Stack Sentinel-2 (GeoTIFF)*;
* ``ppg plugin sentinel2 search|fetch|compute|series|composite|cropmarks|list|export|remove PROJECT ...``.

What is read and made lives in ``<chunk data>/sentinel2/`` and in ``chunk.meta["sentinel2"]``.
Contains modified Copernicus Sentinel data: the notice goes into every file made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import layer_name, t

__version__ = "0.1.0"


class Sentinel2Plugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "Sentinel-2"
        self.description = t("Serie temporali Sentinel-2 sull'area del rilievo per cercare i crop mark: ricerca STAC, "
                             "nuvole mascherate, NDVI, NDRE e NDMI, serie, compositi, mappe dei crop mark",
                             "Sentinel-2 time series over the survey to look for crop marks: STAC search, clouds "
                             "masked, NDVI, NDRE and NDMI, series, composites, crop-mark maps")
        self.context = None
        self.pane = None
        self.overlay = None
        self._pane_key = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Pannello Sentinel-2", "Sentinel-2 pane"), self.show_pane,
                           tip=t("Il pannello Sentinel-2 del dock del progetto", "The Sentinel-2 pane of the project dock"))
        context.add_action(t("Cerca immagini Sentinel-2...", "Search Sentinel-2 images..."), self.search_dialog,
                           processing=True)
        context.add_action(t("Disegna l'area di Sentinel-2", "Draw the Sentinel-2 area"), self.draw_area)
        context.add_action(t("Scarica e calcola", "Read and compute"), lambda: self.fetch([], None), processing=True)
        context.add_action(t("Mappa dei crop mark...", "Crop-mark map..."), self.cropmarks_dialog, processing=True)
        context.add_action(t("Nascondi la sovrapposizione", "Hide the overlay"), self.hide_overlay)
        context.add_command("sentinel2", self.command,
                            help="Sentinel-2 time series: search, fetch, compute, series, composite, cropmarks, list, "
                                 "export, remove")
        context.add_exporter(t("Serie temporale Sentinel-2 (CSV)", "Sentinel-2 time series (CSV)"), ".csv",
                             self._export_series)
        context.add_exporter(t("Stack Sentinel-2 (GeoTIFF)", "Sentinel-2 stack (GeoTIFF)"), ".tif", self._export_stack)
        window = context.window
        if window is not None:
            self._install_pane(window)
            context.on_raster_loaded(self._raster_loaded)
            # the pane follows the project and its active chunk: another project, chunk, step or undo
            context.on_refresh(self.refresh_pane)

    def deactivate(self) -> None:
        self.hide_overlay()
        window = self.context.window if self.context else None
        if window is not None:
            view = getattr(window, "ortho_view", None)
            if view is not None and layer_name() in getattr(view, "layers", {}):
                view.remove_layer(layer_name())
            if self.pane is not None:
                tabs = getattr(window, "left_tabs", None)
                if tabs is not None:
                    k = tabs.indexOf(self.pane)
                    if k >= 0:
                        tabs.removeTab(k)
            providers = getattr(window, "qgis_providers", [])
            window.qgis_providers = [p for p in providers if p != self._qgis_entries]
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import Sentinel2Pane

        try:
            from pyarchinit_photogrammetry.gui.icons import icon

            ic = icon("globe")
        except Exception:  # noqa: BLE001
            ic = None
        self.pane = Sentinel2Pane()
        tabs = getattr(window, "left_tabs", None)
        if tabs is not None:
            if ic is not None:
                tabs.addTab(self.pane, ic, "Sentinel-2")
            else:
                tabs.addTab(self.pane, "Sentinel-2")
        p = self.pane
        p.area_requested.connect(self.area_from_chunk)
        p.draw_requested.connect(self.draw_area)
        p.frame_requested.connect(self.show_frame)
        p.search_requested.connect(self.search)
        p.fetch_requested.connect(self.fetch)
        p.compute_requested.connect(self.compute)
        p.overlay_requested.connect(self.show_overlay)
        p.opacity_changed.connect(lambda v: self.overlay.set_opacity(v) if self.overlay else None)
        p.hide_requested.connect(self.hide_overlay)
        p.cropmarks_requested.connect(self.cropmarks)
        p.composite_requested.connect(self.composite)
        p.product_overlay_requested.connect(self.show_product)
        p.product_export_requested.connect(self.export_product)
        p.remove_requested.connect(self.remove)
        p.export_requested.connect(self.export_dialog)
        p.qgis_requested.connect(lambda: window.open_in_qgis() if hasattr(window, "open_in_qgis") else None)
        p.chart_requested.connect(self.open_chart)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        runner = getattr(window, "runner", None)
        if runner is not None and hasattr(runner, "busy_changed"):  # no second step while one runs
            runner.busy_changed.connect(p.set_busy)
            self.context.contributions.connections.append((runner.busy_changed, p.set_busy))
        self.refresh_pane()

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        window = self.context.window
        chunk = getattr(window, "chunk", None) if window is not None else None
        return chunk if chunk is not None else getattr(doc, "chunk", None)

    def _say(self, title: str, text: str, kind: str = "info") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(f"{title}: {text}")
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        if hasattr(window, "log_pane"):
            (window.log_pane.warning if kind == "warning" else window.log_pane.info)(text)
        message(window, kind, title, text)

    def _status(self, text: str) -> None:
        window = self.context.window if self.context else None
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 12000)
        if self.pane is not None:
            self.pane.set_status(text)

    def _changed(self) -> None:
        self.context.mark_modified()
        self.refresh_pane()

    def _area_text(self, chunk) -> str:
        from . import project

        area, grid = project.area_of(chunk), project.grid_of_chunk(chunk)
        if area is None or grid is None:
            return t("Nessuna area: dal DEM o dall'ortofoto del chunk, o disegnata sulla vista.",
                     "No area: from the DEM or orthomosaic of the chunk, or drawn on the view.")
        try:
            w, h = area.size_m()
        except Exception:  # noqa: BLE001 - a coordinate system this computer cannot read
            return ""
        source = {"dem": "DEM", "orthomosaic": t("ortofoto", "orthomosaic"), "polygon": t("disegnata", "drawn"),
                  "bbox": t("riquadro", "box")}.get(area.source, area.source)
        return t("Area: {0:.0f} x {1:.0f} m ({2}), margine {3:.0f} m; griglia {4}: {5} x {6} celle da {7:g} m",
                 "Area: {0:.0f} x {1:.0f} m ({2}), margin {3:.0f} m; grid {4}: {5} x {6} cells of {7:g} m").format(
            w, h, source, area.margin_m, grid.crs, grid.width, grid.height, grid.cell)

    def refresh_pane(self) -> None:
        from . import project

        if self.pane is None:
            return
        chunk = self._chunk()
        key = None if chunk is None else (id(chunk.document), chunk.key)
        inputs, self._pane_key = key != self._pane_key, key  # the fields of the search follow another chunk only
        if chunk is None:
            self.pane.set_state({})
            return
        st = project.peek(chunk)
        chart = project.absolute(chunk, (st.get("series") or {}).get("chart"))
        notice = ""
        if st.get("dates"):
            notice = t("Contiene dati Copernicus Sentinel modificati {0}", "Contains modified Copernicus Sentinel data {0}") \
                .format(project.attribution_of(chunk).rsplit(" ", 1)[-1])
        self.pane.set_state(st, chart, self._area_text(chunk), notice, inputs)

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
        from . import project

        chunk = self._chunk()
        return project.qgis_entries(chunk) if chunk is not None else ([], [])

    def _run(self, title: str, function, done) -> None:
        """``function(progress)`` in the background, then ``done(result)``; the project marked changed."""
        def finished(result):
            self._changed()
            done(result)

        self.context.run(title, function, finished)

    # ------------------------------------------------------------------ the area
    def _margin(self) -> float:
        return float(self.pane.s_margin.value()) if self.pane is not None else 300.0

    def area_from_chunk(self) -> bool:
        from . import area as area_mod
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return False
        try:
            area = area_mod.area_of_chunk(chunk, "auto", self._margin())
            project.set_area(chunk, area)
        except (area_mod.NotPlaced, RuntimeError) as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return False
        self._changed()
        self._draw_area_layer()
        self._status(self._area_text(chunk))
        return True

    def draw_area(self) -> None:
        from . import area as area_mod

        chunk = self._chunk()
        if chunk is None or self.context.window is None:
            return
        try:
            area_mod.chunk_crs(chunk)
        except area_mod.NotPlaced as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        self.show_pane()
        self.context.request_polygon(self.area_drawn, hint=t(
            "Area di Sentinel-2: trascina un riquadro, oppure clic per i vertici e doppio clic o Invio; Esc annulla",
            "Sentinel-2 area: drag a box, or click the vertices and double click or Enter; Esc cancels"))

    def area_drawn(self, xy) -> None:
        from . import area as area_mod
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return
        area = area_mod.area_of_polygon(chunk, xy, self._margin())
        project.set_area(chunk, area)
        self._changed()
        self._draw_area_layer()
        self._status(self._area_text(chunk) + " " + t("Ora cerca le immagini.", "Now search the images."))

    def _area_rings(self, chunk) -> list[tuple[list, str, str]]:
        """The area and its read window as rings in the chunk's coordinates: (ring, label, colour)."""
        from rasterio.warp import transform_geom

        from . import area as area_mod
        from . import project

        area, grid = project.area_of(chunk), project.grid_of_chunk(chunk)
        if area is None or grid is None:
            return []
        try:
            crs = area_mod.chunk_crs(chunk)
        except area_mod.NotPlaced:
            return []
        poly = area.in_crs(crs) if not area_mod._same_crs(area.crs, crs) else area.polygon()
        x0, y0, x1, y1 = grid.bounds
        box = {"type": "Polygon", "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]}
        frame = transform_geom(grid.crs, crs, box, precision=-1)
        return [([list(p) for p in list(poly.exterior.coords)[:-1]], t("area di Sentinel-2", "Sentinel-2 area"), "#ffd000"),
                ([list(p) for p in frame["coordinates"][0][:-1]],
                 t("finestra letta (area e margine)", "window read (area and margin)"), "#9e9e9e")]

    def _draw_area_layer(self) -> None:
        window = self.context.window if self.context else None
        chunk = self._chunk()
        view = getattr(window, "ortho_view", None) if window is not None else None
        if view is None or chunk is None or view.tr is None:
            return
        name = layer_name()
        if name in getattr(view, "layers", {}):
            view.remove_layer(name)
        rings = self._area_rings(chunk)
        if not rings:
            return
        with view.batch():  # no labels on the map: both would sit in the middle of the view
            for ring, label, colour in rings:
                view.add_polygon(ring, label="", color=colour, fill_alpha=0, layer=name, attrs={"descrizione": label})

    def _raster_loaded(self) -> None:
        """Another raster (or a frame) in the view: the area and the overlay drawn again on it."""
        if self.pane is None:  # disabled (the program disconnects this soon after)
            return
        try:
            self._draw_area_layer()
        except Exception as exc:  # noqa: BLE001 - a drawing must never stop the view
            self.context.log.warning("Sentinel-2 area: %s", exc)
        spec = self.overlay.spec if self.overlay is not None else None
        if not spec:
            return
        chunk = self._chunk()
        if chunk is None or spec.get("chunk") != getattr(chunk, "key", None):
            self.hide_overlay()  # another chunk is active: its own products, not these
            return
        if self.overlay.view.tr is not None:
            try:
                self._overlay(spec, quiet=True)
            except Exception as exc:  # noqa: BLE001
                self.context.log.warning("Sentinel-2 overlay: %s", exc)

    def show_frame(self) -> None:
        """An empty frame with the area and its read window: the products beyond the orthomosaic."""
        from rasterio.warp import transform_bounds

        from . import area as area_mod
        from . import project

        chunk = self._chunk()
        grid = project.grid_of_chunk(chunk) if chunk is not None else None
        if grid is None:
            return
        try:
            crs = area_mod.chunk_crs(chunk)
        except area_mod.NotPlaced as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        b = transform_bounds(grid.crs, crs, *grid.bounds, densify_pts=21)
        metres = max(2.0, max(grid.bounds[2] - grid.bounds[0], grid.bounds[3] - grid.bounds[1]) / 3000.0)
        cell = (b[2] - b[0]) / (grid.bounds[2] - grid.bounds[0]) * metres  # 2 m a pixel, at most 3000 pixels a side
        self.context.show_frame(b, cell, t("Sentinel-2: area e finestra letta", "Sentinel-2: area and window read"))
        window = self.context.window
        if hasattr(window, "show_view"):
            window.show_view(window.ortho_view)

    # ------------------------------------------------------------------ search, read, compute
    def search_dialog(self) -> None:
        """The search with its parameters asked in a dialog (those of the pane as defaults)."""
        from datetime import date

        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        if window is None or self.pane is None:
            return
        self.show_pane()
        d = self.pane.search_params()
        v = ParamDialog.ask(window, t("Cerca immagini Sentinel-2", "Search Sentinel-2 images"), [
            Field("start", t("Dal giorno (AAAA-MM-GG)", "From the day (YYYY-MM-DD)"), d["start"], kind="text"),
            Field("end", t("Al giorno (AAAA-MM-GG)", "To the day (YYYY-MM-DD)"), d["end"], kind="text"),
            Field("max_cloud", t("Nuvole dei tile al massimo (%)", "Clouds of the tiles at most (%)"), d["max_cloud"],
                  minimum=0.0, maximum=100.0, decimals=0),
            Field("margin", t("Margine intorno all'area (m)", "Margin around the area (m)"), d["margin"], minimum=0.0,
                  maximum=5000.0, decimals=0),
            Field("catalog", t("Catalogo STAC", "STAC catalog"), d["catalog"], kind="text")],
            intro=t("Le acquisizioni di Sentinel-2 L2A sull'area del chunk (l'impronta del DEM o dell'ortofoto, o "
                    "l'area disegnata) e sul suo margine, in un catalogo STAC: Earth Search di default, dati pubblici "
                    "senza chiavi.",
                    "The Sentinel-2 L2A acquisitions over the area of the chunk (the footprint of the DEM or "
                    "orthomosaic, or the area drawn) and its margin, in a STAC catalog: Earth Search by default, "
                    "public data, no keys."))
        if not v:
            return
        try:
            for key in ("start", "end"):
                date.fromisoformat(str(v[key]).strip())
        except ValueError:
            self._say("Sentinel-2", t("Date nella forma AAAA-MM-GG, per esempio 2023-06-14.",
                                      "Dates as YYYY-MM-DD, for instance 2023-06-14."), "warning")
            return
        params = {**d, **{k: (str(v[k]).strip() if k in ("start", "end", "catalog") else float(v[k])) for k in v}}
        self.search(params)

    def search(self, params: dict) -> None:
        from . import area as area_mod
        from . import project

        chunk = self._chunk()
        if chunk is None:
            self._say("Sentinel-2", t("Apri un progetto con un chunk.", "Open a project with a chunk."), "warning")
            return
        area = project.area_of(chunk)
        try:
            if area is None:
                area = area_mod.area_of_chunk(chunk, "auto", float(params.get("margin", 300.0)))
            elif abs(area.margin_m - float(params.get("margin", area.margin_m))) > 1e-6:
                area.margin_m = float(params["margin"])
        except (area_mod.NotPlaced, RuntimeError) as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        catalog = params.get("catalog") or project.stac.EARTH_SEARCH
        self._run(t("Ricerca Sentinel-2", "Sentinel-2 search"),
                  lambda p: project.search(chunk, area, params["start"], params["end"], params.get("max_cloud", 30.0),
                                           catalog, progress=p),
                  lambda r: (self._draw_area_layer(),
                             self._status(t("{0} acquisizioni trovate ({1} scene del catalogo, {2} già lette)",
                                            "{0} acquisitions found ({1} catalog scenes, {2} already read)")
                                          .format(r["acquisitions"], r["items"], r["cached"]))))

    def fetch(self, ids, min_clear) -> None:
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return
        if not project.acquisitions(chunk):
            self._say("Sentinel-2", t("Nessuna scena: cerca prima nel catalogo.", "No scene: search the catalog first."),
                      "warning")
            return
        if min_clear is None and self.pane is not None:
            min_clear = self.pane.s_clear.value() / 100.0

        def done(r):
            c = r.get("computed") or {}
            text = t("Lette {0}, già in cache {1}, nuvolose {2}, errori {3}; {4} date serene nella serie",
                     "Read {0}, already cached {1}, cloudy {2}, errors {3}; {4} clear dates in the series").format(
                r["read"], r["cached"], r["cloudy"] + r.get("skipped_cloudy", 0), r["failed"], c.get("dates", 0))
            self._status(text)
            if c.get("dates") and self.pane is not None and self.context.window is not None:
                try:
                    self.show_overlay(self.pane.view_params())
                except Exception as exc:  # noqa: BLE001
                    self.context.log.warning("Sentinel-2 overlay: %s", exc)

        self._run(t("Sentinel-2: lettura delle scene", "Sentinel-2: reading the scenes"),
                  lambda p: project.fetch(chunk, list(ids) or None, min_clear, progress=p), done)

    def compute(self) -> None:
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return
        self._run(t("Sentinel-2: indici e serie", "Sentinel-2: indices and series"),
                  lambda p: project.compute(chunk, progress=p),
                  lambda r: self._status(t("{0} date serene nella serie", "{0} clear dates in the series").format(r["dates"])))

    # ------------------------------------------------------------------ overlays
    def _ensure_raster(self) -> bool:
        window = self.context.window
        view = getattr(window, "ortho_view", None) if window is not None else None
        if view is None:
            return False
        if view.tr is None:
            chunk = self._chunk()
            if chunk is not None and (chunk.orthomosaic_path or chunk.dem_path) and hasattr(window, "_show_raster"):
                window._show_raster("ortho" if chunk.orthomosaic_path else "dem")
            if view.tr is None:
                self.show_frame()
        return view.tr is not None

    def _values(self, spec: dict):
        """``(values, transform, crs, kind)`` of what an overlay shows."""
        from . import project

        chunk = self._chunk()
        what = spec.get("what", "index")
        if what == "product":
            e = project.product(chunk, spec["id"])
            values, tr, crs = project.read_product(chunk, spec["id"])
            return values, tr, crs, ("z" if e and e.get("kind") == "cropmarks" else "index")
        period = spec.get("period", "all")
        if spec.get("index") == "RGB":
            ids = project.period_ids(chunk, period)
            if not ids:
                raise project.NoData(t("Nessuna data serena nel periodo.", "No clear date in the period."))
            clear = {d["id"]: d.get("clear", 0.0) for d in project.dates(chunk)}
            best = max(ids, key=lambda i: clear.get(i, 0.0))
            values, tr, crs = project.rgb(chunk, best)
            return values, tr, crs, "rgb"
        med, _count, tr, crs, _ids = project.composite_values(chunk, spec.get("index", "NDVI"), period)
        return med, tr, crs, "index"

    def _overlay(self, spec: dict, quiet: bool = False):
        from . import area as area_mod
        from .overlay import ViewOverlay, legend_text

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return None
        if not quiet and not self._ensure_raster():
            return None
        view = window.ortho_view
        values, tr, crs, kind = self._values(spec)
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = ViewOverlay(view)
        lims = self.overlay.show(values, tr, crs, area_mod.chunk_crs(chunk), kind, spec.get("palette", "rdylgn"),
                                 spec.get("opacity", 0.75), spec={**spec, "chunk": getattr(chunk, "key", None)})
        if self.pane is not None:
            lo, hi = (0.0, 0.0) if kind == "rgb" else lims
            self.pane.set_legend(legend_text(kind, lo, hi, spec.get("palette", "rdylgn")))
        if not quiet and hasattr(window, "show_view"):
            window.show_view(view)
        return lims

    def show_overlay(self, params: dict) -> None:
        """An index of a date or a period (natural colour: the clearest date of the period) over the view."""
        from . import area as area_mod
        from . import project

        try:
            self._overlay({"what": "index", **params})
        except (project.NoData, area_mod.NotPlaced, RuntimeError, ValueError) as exc:
            self._say("Sentinel-2", str(exc), "warning")

    def show_product(self, pid: str) -> None:
        from . import project

        chunk = self._chunk()
        e = project.product(chunk, pid) if chunk is not None else None
        if e is None:
            return
        params = self.pane.view_params() if self.pane is not None else {"palette": "rdylgn", "opacity": 0.75}
        try:
            self._overlay({"what": "product", "id": pid, "palette": params["palette"], "opacity": params["opacity"]})
        except (project.NoData, RuntimeError, ValueError, OSError) as exc:
            self._say("Sentinel-2", str(exc), "warning")

    def hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove(forget=True)
        if self.pane is not None:
            self.pane.set_legend("")

    # ------------------------------------------------------------------ products
    def cropmarks_dialog(self) -> None:
        """A crop-mark map with its index, period, background and method asked in a dialog."""
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import analysis, project
        from .texts import index_name, period_name

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or self.pane is None:
            return
        self.show_pane()
        if not project.dates(chunk):
            self._say("Sentinel-2", t("Nessuna data serena: scarica e calcola prima.", "No clear date: read and compute first."),
                      "warning")
            return
        d = self.pane.cropmark_params()
        periods = [(p, period_name(p)) for p in project.periods(chunk)]
        periods += [(x["id"], f"{x['date']} {x.get('platform', '')}") for x in reversed(project.dates(chunk))]
        v = ParamDialog.ask(window, t("Mappa dei crop mark", "Crop-mark map"), [
            Field("index", t("Indice", "Index"), d["index"], choices=[(n, index_name(n)) for n in analysis.INDICES]),
            Field("period", t("Data o periodo", "Date or period"),
                  d["period"] if d["period"] in {p for p, _n in periods} else "all", choices=periods),
            Field("background", t("Sfondo (m, sigma della gaussiana)", "Background (m, sigma of the Gaussian)"),
                  d["background"], minimum=10.0, maximum=1000.0, decimals=0),
            Field("method", t("Metodo dello sfondo", "Method of the background"), d["method"],
                  choices=[("gaussian", t("Gaussiana", "Gaussian")), ("median", t("Mediana", "Median"))])],
            intro=t("L'indice della data, o la mediana del periodo, meno il suo sfondo liscio, diviso per la dispersione "
                    "robusta di ciò che resta: uno z-score, sotto -2 dove la vegetazione è più debole di quella "
                    "intorno (muri, strade), sopra 2 dove è più verde (fossati, fosse).",
                    "The index of the date, or the median of the period, minus its smooth background, divided by the "
                    "robust spread of what is left: a z-score, under -2 where the vegetation is weaker than around it "
                    "(walls, roads), over 2 where it is greener (ditches, pits)."))
        if v:
            self.cropmarks({"index": v["index"], "period": v["period"], "background": float(v["background"]),
                            "method": v["method"]})

    def cropmarks(self, params: dict) -> None:
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return
        if not project.dates(chunk):
            self._say("Sentinel-2", t("Nessuna data serena: scarica e calcola prima.", "No clear date: read and compute first."),
                      "warning")
            return
        self._run(t("Sentinel-2: mappa dei crop mark", "Sentinel-2: crop-mark map"),
                  lambda p: project.cropmarks(chunk, params["index"], params["period"], params["background"],
                                              params.get("method", "gaussian"), progress=p),
                  lambda e: (self._status(self._product_text(e)), self.show_product(e["id"])))

    def composite(self, params: dict) -> None:
        from . import project

        chunk = self._chunk()
        if chunk is None or not project.dates(chunk):
            return
        self._run(t("Sentinel-2: composito", "Sentinel-2: composite"),
                  lambda p: project.composite(chunk, params["index"], params["period"], progress=p),
                  lambda e: (self._status(self._product_text(e)), self.show_product(e["id"])))

    @staticmethod
    def _product_text(e: dict) -> str:
        from .texts import product_name

        st = e.get("stats") or {}
        text = t("{0}: {1} date", "{0}: {1} dates").format(product_name(e), len(e.get("dates") or []))
        if e.get("kind") == "cropmarks" and "low" in st:
            text += t("; {0:.1f}% dell'area sotto -2, {1:.1f}% sopra 2", "; {0:.1f}% of the area under -2, {1:.1f}% over 2") \
                .format(st["low"] * 100, st["high"] * 100)
        return text

    def remove(self, pid: str, confirm: bool = True) -> bool:
        from . import project
        from .texts import product_name

        chunk = self._chunk()
        e = project.product(chunk, pid) if chunk is not None else None
        if e is None:
            return False
        window = self.context.window
        if confirm and window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("Rimuovere «{0}» dal progetto? I suoi file vengono cancellati.",
                     "Remove «{0}» from the project? Its files are deleted.").format(product_name(e))
            if dialogs.choice(window, "question", t("Rimuovi il prodotto", "Remove the product"), text,
                              [t("Rimuovi", "Remove")]) != 0:
                return False
        if self.overlay is not None and (self.overlay.spec or {}).get("id") == pid:
            self.hide_overlay()
        project.remove(chunk, pid)
        self._changed()
        return True

    def export_product(self, pid: str) -> None:
        import shutil

        from . import project

        chunk = self._chunk()
        e = project.product(chunk, pid) if chunk is not None else None
        window = self.context.window
        if e is None or window is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(window, t("Esporta il prodotto", "Export the product"), f"{pid}.tif",
                                              "GeoTIFF (*.tif)")
        if not path:
            return
        path = Path(path)
        shutil.copyfile(project.absolute(chunk, e["file"]), path)
        out = [path]
        if e.get("chunk_file") and project.absolute(chunk, e["chunk_file"]).exists():
            other = path.with_name(path.stem + "_chunk.tif")
            shutil.copyfile(project.absolute(chunk, e["chunk_file"]), other)
            out.append(other)
        self._status(t("Esportato: {0}", "Exported: {0}").format(", ".join(str(p) for p in out)))

    def export_dialog(self) -> None:
        from . import project

        chunk = self._chunk()
        window = self.context.window
        if chunk is None or window is None:
            return
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(window, t("Cartella dell'esportazione Sentinel-2",
                                                            "Folder of the Sentinel-2 export"))
        if not folder:
            return
        self.context.run(t("Sentinel-2: esportazione", "Sentinel-2: export"),
                         lambda p: project.export_all(chunk, folder, p),
                         lambda files: self._status(t("{0} file in {1}", "{0} files in {1}").format(len(files), folder)))

    def open_chart(self) -> None:
        from . import project

        chunk = self._chunk()
        path = project.absolute(chunk, (project.peek(chunk).get("series") or {}).get("chart")) if chunk else None
        if path is None or not path.exists():
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _export_series(self, chunk, path, progress=None):
        from . import project

        if progress:
            progress(0.5, t("Serie Sentinel-2", "Sentinel-2 series"))
        return project.export_series(chunk, path)

    def _export_stack(self, chunk, path, progress=None):
        from . import project

        if progress:
            progress(0.5, t("Stack Sentinel-2", "Sentinel-2 stack"))
        return project.export_stack(chunk, path)

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _project_args(args) -> tuple[list[str], str | None, str | None]:
        """``ppg plugin sentinel2 WHAT PROJECT.ppg ...``: the project given right after the
        subcommand (or with ``--project``) and ``--chunk`` taken out of the arguments; without a
        project the commands work on the document the program has open."""
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
            if label:
                return next((c for c in doc.chunks if c.label == label), None)
            return doc.chunk
        chunk = self._chunk()
        if chunk is not None and label:
            doc = self.context.app.document
            return next((c for c in doc.chunks if c.label == label), None)
        return chunk

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        from . import analysis, project, reader, stac

        start, end = project.default_dates()
        parser = argparse.ArgumentParser(prog="ppg plugin sentinel2",
                                         description="Sentinel-2 time series over the area of a survey (crop marks). "
                                                     "Contains modified Copernicus Sentinel data.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("search", help="the acquisitions over the area in a STAC catalog")
        p.add_argument("--start", default=start, help=f"first day (default {start})")
        p.add_argument("--end", default=end, help="last day (default today)")
        p.add_argument("--max-cloud", type=float, default=30.0, help="largest cloud cover of the tiles, %%")
        p.add_argument("--catalog", default=stac.EARTH_SEARCH, help="STAC API (default Earth Search v1)")
        p.add_argument("--collection", default=stac.COLLECTION)
        p.add_argument("--margin", type=float, default=300.0, help="metres around the area")
        p.add_argument("--area", choices=("auto", "dem", "orthomosaic", "keep"), default="auto",
                       help="the footprint of the DEM or orthomosaic, or keep the area set before (drawn)")
        p.add_argument("--bbox", help="a box instead: west,south,east,north in degrees")
        p.add_argument("--polygon", help='a polygon instead, in the chunk\'s coordinates: "x,y;x,y;x,y"')
        p.add_argument("--client", choices=("auto", "stdlib", "pystac"), default=None)
        p = sub.add_parser("fetch", help="read the window of the area of the acquisitions found (cached), then compute")
        p.add_argument("--scenes", help="only these acquisitions (ids, comma separated)")
        p.add_argument("--min-clear", type=float, default=None, help="least clear part of the area, 0-1 (default 0.5)")
        p.add_argument("--clear", help="SCL classes kept as clear (default 4,5,6)")
        p.add_argument("--buffer", type=float, default=None, help=f"metres masked around clouds and shadows "
                                                                  f"(default {reader.BUFFER_M:g})")
        p.add_argument("--workers", type=int, default=reader.WORKERS)
        p.add_argument("--no-compute", action="store_true", help="read only")
        sub.add_parser("compute", help="indices and series from the acquisitions read (no network)")
        p = sub.add_parser("series", help="the series over the area (per date, or per month)")
        p.add_argument("--monthly", action="store_true")
        p.add_argument("--index", choices=tuple(analysis.INDICES), default=None)
        p = sub.add_parser("composite", help="the median of an index over a period, as a product")
        p.add_argument("--index", choices=tuple(analysis.INDICES), default="NDVI")
        p.add_argument("--period", default="all", help="all, summer, 2023-summer, 2023, 2023-06, 2023-06-14, "
                                                       "2023-04-01/2023-06-30 or an acquisition id")
        p = sub.add_parser("cropmarks", help="the crop-mark map of an index over a period (z-score)")
        p.add_argument("--index", choices=tuple(analysis.INDICES), default="NDVI")
        p.add_argument("--period", default="all")
        p.add_argument("--background", type=float, default=analysis.BACKGROUND_M, help="background scale, m")
        p.add_argument("--method", choices=analysis.METHODS, default="gaussian")
        sub.add_parser("list", help="area, acquisitions, dates and products of the chunk")
        p = sub.add_parser("export", help="everything into a folder; or the stack (PATH.tif) or the series (PATH.csv)")
        p.add_argument("path")
        p = sub.add_parser("remove", help="a product out of the project")
        p.add_argument("id")
        return parser

    def command(self, *args) -> int:
        from . import analysis, project, stac
        from . import area as area_mod

        args, project_path, label = self._project_args(args)
        ns = self._parser().parse_args(args)
        chunk = self._open_for_command(project_path, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  f"no chunk: give the project, ppg plugin sentinel2 {ns.what} PROJECT.ppg ...")
            return 1
        doc = self.context.app.document if self.context is not None else None
        run = (self.context.run if self.context is not None
               else lambda title, f, done=None: f(None))
        try:
            if ns.what == "search":
                if ns.bbox:
                    w, s, e, n = (float(v) for v in ns.bbox.split(","))
                    area = area_mod.area_of_bbox(chunk, w, s, e, n, ns.margin)
                elif ns.polygon:
                    xy = [[float(v) for v in pt.split(",")] for pt in ns.polygon.split(";") if pt.strip()]
                    area = area_mod.area_of_polygon(chunk, xy, ns.margin)
                elif ns.area == "keep" and project.area_of(chunk) is not None:
                    area = project.area_of(chunk)
                    area.margin_m = ns.margin
                else:
                    area = area_mod.area_of_chunk(chunk, "auto" if ns.area == "keep" else ns.area, ns.margin)
                r = run("Sentinel-2 search", lambda p: project.search(chunk, area, ns.start, ns.end, ns.max_cloud,
                                                                      ns.catalog, ns.collection, client=ns.client,
                                                                      progress=p))
                print(json.dumps(r, indent=1))
                for row in project.scene_rows(chunk):
                    print(f"  {row['id']:<16} {row['date']}  tiles {','.join(row['tiles']):<12} "
                          f"cloud {row['cloud'] if row['cloud'] is not None else '-':>6}  {row['status']}")
            elif ns.what == "fetch":
                ids = [s.strip() for s in ns.scenes.split(",") if s.strip()] if ns.scenes else None
                clear = [int(c) for c in ns.clear.split(",")] if ns.clear else None
                r = run("Sentinel-2 fetch", lambda p: project.fetch(chunk, ids, ns.min_clear, ns.workers, clear,
                                                                    ns.buffer, p, not ns.no_compute))
                print(json.dumps(r, indent=1))
            elif ns.what == "compute":
                print(json.dumps(run("Sentinel-2 compute", lambda p: project.compute(chunk, p)), indent=1))
            elif ns.what == "series":
                rows = project.series(chunk, ns.monthly)
                if not rows:
                    print("no series: fetch first")
                    return 1
                print("# " + project.attribution_of(chunk))
                first = "month" if ns.monthly else "date"
                print(f"{first:<11} {'index':<5} {'cells':>6} {'median':>8} {'p10':>8} {'p25':>8} {'p75':>8} {'p90':>8}")
                for row in rows:
                    if ns.index and row["index"] != ns.index:
                        continue
                    if not row.get("cells"):
                        print(f"{row[first]:<11} {row['index']:<5} {0:>6}")
                        continue
                    print(f"{row[first]:<11} {row['index']:<5} {int(row['cells']):>6} {row['median']:>8.4f} "
                          f"{row['p10']:>8.4f} {row['p25']:>8.4f} {row['p75']:>8.4f} {row['p90']:>8.4f}")
            elif ns.what == "composite":
                e = run("Sentinel-2 composite", lambda p: project.composite(chunk, ns.index, ns.period, p))
                print(json.dumps(e, indent=1))
            elif ns.what == "cropmarks":
                analysis.check_period(ns.period)
                e = run("Sentinel-2 crop marks", lambda p: project.cropmarks(chunk, ns.index, ns.period, ns.background,
                                                                             ns.method, p))
                print(json.dumps(e, indent=1))
            elif ns.what == "list":
                print(json.dumps(project.summary(chunk), indent=1, ensure_ascii=False))
                return 0
            elif ns.what == "export":
                path = Path(ns.path)
                if path.suffix.lower() == ".tif":
                    print(project.export_stack(chunk, path))
                elif path.suffix.lower() == ".csv":
                    print("\n".join(str(p) for p in project.export_series(chunk, path)))
                else:
                    print("\n".join(str(p) for p in project.export_all(chunk, path)))
                return 0
            elif ns.what == "remove":
                if project.product(chunk, ns.id) is None:
                    print(f"no product {ns.id!r}: see ppg plugin sentinel2 list")
                    return 1
                print(json.dumps({"removed": ns.id, "files_deleted": project.remove(chunk, ns.id)}))
        except (area_mod.NotPlaced, project.NoData, stac.CatalogError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}")
            if doc is not None and getattr(doc, "path", None):
                doc.save()
            return 1
        if doc is not None and getattr(doc, "path", None):
            doc.save()
        return 0
