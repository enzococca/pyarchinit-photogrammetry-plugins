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
* ``ppg plugin sentinel2 search|fetch|compute|series|composite|cropmarks|list|export|remove|remove-scene PROJECT ...``;
* from program 1.3.4, a *Sentinel-2* node per chunk in the project's tree: the acquisitions read, the
  indices by date, the series, the composites and the crop-mark maps, shown by a double click.

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
                                 "export, remove, remove-scene")
        context.add_exporter(t("Serie temporale Sentinel-2 (CSV)", "Sentinel-2 time series (CSV)"), ".csv",
                             self._export_series)
        context.add_exporter(t("Stack Sentinel-2 (GeoTIFF)", "Sentinel-2 stack (GeoTIFF)"), ".tif", self._export_stack)
        window = context.window
        if window is not None:
            self._install_pane(window)
            context.on_raster_loaded(self._raster_loaded)
            # the pane follows the project and its active chunk: another project, chunk, step or undo
            context.on_refresh(self.refresh_pane)
        if hasattr(context, "add_tree_items"):  # program 1.3.4: the products in the project's tree too
            context.add_tree_items(self._tree_items)

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
        p.compute_requested.connect(lambda: self.compute())
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
        p.chart_requested.connect(lambda: self.open_chart())
        p.scenes_remove_requested.connect(lambda ids: self.remove_scenes(list(ids)))
        p.cache_clear_requested.connect(lambda: self.clear_cache())
        p.scene_show_requested.connect(self.show_acquisition)
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
        self._refresh_tree()

    def _refresh_tree(self) -> None:
        """The project's tree filled again (program 1.3.4; nothing before)."""
        refresh = getattr(self.context, "refresh_tree", None) if self.context is not None else None
        if refresh is not None:
            try:
                refresh()
            except Exception as exc:  # noqa: BLE001 - the tree must never stop the plugin
                self.context.log.warning("Sentinel-2 tree: %s", exc)

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

    def compute(self, min_clear: float | None = None) -> None:
        """Indices and series again from the acquisitions read, with the least clear part of the area
        of the pane (saved), without network."""
        from . import project

        chunk = self._chunk()
        if chunk is None:
            return
        if min_clear is None and self.pane is not None:
            min_clear = self.pane.s_clear.value() / 100.0
        self._run(t("Sentinel-2: indici e serie", "Sentinel-2: indices and series"),
                  lambda p: project.compute(chunk, progress=p, min_clear=min_clear),
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
        if what == "rgb":  # natural colour of one acquisition of the cache, clear or not
            values, tr, crs = project.rgb(chunk, spec["acq"])
            return values, tr, crs, "rgb"
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

    def remove(self, pid: str, confirm: bool = True, chunk=None) -> bool:
        from . import project
        from .texts import product_name

        chunk = chunk or self._chunk()
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
        spec = self.overlay.spec if self.overlay is not None else None
        if spec and spec.get("id") == pid and spec.get("chunk") == getattr(chunk, "key", None):
            self.hide_overlay()
        project.remove(chunk, pid)
        self._changed()
        return True

    def remove_scenes(self, ids=None, confirm: bool = True, chunk=None, with_products: bool | None = None) -> bool:
        """Acquisitions out of the cache (``ids``; None: the whole cache), after a question in the window
        (the program's non-native box): their files deleted, their rows back to *to read* (they can be
        read again), the indices and the series computed again from the others. The products made from
        them: removed too, or kept and marked as made from acquisitions no longer in the cache - the
        question asks when there are any; ``with_products`` answers it without asking."""
        from . import project
        from .texts import product_name

        chunk = chunk or self._chunk()
        if chunk is None:
            return False
        cached = project.cached_ids(chunk)
        files, size = project.cache_size(chunk)
        if ids is None:
            if not cached and not files:
                self._say("Sentinel-2", t("La cache è già vuota.", "The cache is empty already."))
                return False
            targets, chosen = None, cached
        else:
            targets = [str(i) for i in ids if str(i) in cached]
            if not targets:
                self._say("Sentinel-2", t("Nessuna delle acquisizioni scelte è nella cache.",
                                          "None of the acquisitions chosen is in the cache."), "warning")
                return False
            chosen = targets
        affected = project.products_using(chunk, chosen)
        window = self.context.window
        if with_products is None:
            with_products = False
            if confirm and window is not None:
                from pyarchinit_photogrammetry.gui import dialogs

                if targets is None:
                    text = t("Svuotare la cache Sentinel-2 del chunk? {0} file ({1:.1f} MB) vengono cancellati; le "
                             "acquisizioni restano nella ricerca e si possono leggere di nuovo.",
                             "Empty the Sentinel-2 cache of the chunk? {0} files ({1:.1f} MB) are deleted; the "
                             "acquisitions stay in the search and can be read again.").format(files, size / 1e6)
                else:
                    names = ", ".join(targets[:6]) + (", ..." if len(targets) > 6 else "")
                    text = t("Togliere dalla cache {0} acquisizioni ({1})? I file letti vengono cancellati; restano "
                             "nella ricerca e si possono leggere di nuovo. Indici e serie vengono ricalcolati con le "
                             "altre.",
                             "Take {0} acquisitions ({1}) out of the cache? The files read are deleted; they stay in "
                             "the search and can be read again. Indices and series are computed again from the "
                             "others.").format(len(targets), names)
                options = [t("Rimuovi", "Remove")]
                if affected:
                    listed = ", ".join(f"«{product_name(e)}»" for e in affected[:4]) + (", ..." if len(affected) > 4 else "")
                    text += "\n\n" + t("{0} prodotti sono stati calcolati anche da queste acquisizioni ({1}): rimuoverli "
                                       "anche loro, o tenerli segnati come fatti da date non più in cache?",
                                       "{0} products were computed from these acquisitions too ({1}): remove them as "
                                       "well, or keep them marked as made from dates no longer in the cache?").format(
                        len(affected), listed)
                    options = [t("Rimuovi anche i prodotti", "Remove the products too"),
                               t("Tieni i prodotti", "Keep the products")]
                title = t("Svuota la cache", "Empty the cache") if targets is None else t("Rimuovi dalla cache",
                                                                                         "Remove from the cache")
                k = dialogs.choice(window, "question", title, text, options)
                if k is None:
                    return False
                with_products = bool(affected) and k == 0
        self._hide_if_shows(chunk, chosen, with_products, affected)

        def done(r):
            text = t("Cache: {0} acquisizioni tolte, {1:.1f} MB liberati; {2} date serene restano",
                     "Cache: {0} acquisitions out, {1:.1f} MB freed; {2} clear dates left").format(
                len(r["removed"]), r["bytes"] / 1e6, r["computed"].get("dates", 0))
            if r["products_removed"]:
                text += t("; {0} prodotti rimossi", "; {0} products removed").format(len(r["products_removed"]))
            if r["products_marked"]:
                text += t("; {0} prodotti segnati come fatti da date non più in cache",
                          "; {0} products marked as made from dates no longer in the cache").format(len(r["products_marked"]))
            self._status(text)

        self._run(t("Sentinel-2: cache", "Sentinel-2: cache"),
                  lambda p: project.remove_scenes(chunk, targets, with_products, p), done)
        return True

    def clear_cache(self, confirm: bool = True, chunk=None) -> bool:
        """Every acquisition read out of the cache (see :meth:`remove_scenes`)."""
        return self.remove_scenes(None, confirm, chunk)

    def _hide_if_shows(self, chunk, ids, with_products: bool, affected: list[dict]) -> None:
        """The overlay hidden when it shows something taken away: an acquisition removed, a period
        with one of them, a product removed."""
        from . import project

        spec = self.overlay.spec if self.overlay is not None else None
        if not spec or spec.get("chunk") != getattr(chunk, "key", None):
            return
        ids = {str(i) for i in ids}
        what = spec.get("what")
        if what == "rgb":
            hide = spec.get("acq") in ids
        elif what == "product":
            hide = with_products and spec.get("id") in {e["id"] for e in affected}
        else:
            try:
                hide = bool(ids & set(project.period_ids(chunk, spec.get("period", "all"))))
            except ValueError:
                hide = True
        if hide:
            self.hide_overlay()

    def show_acquisition(self, acq_id: str) -> None:
        """Natural colour of an acquisition of the cache over the view."""
        from . import area as area_mod
        from . import project

        params = self.pane.view_params() if self.pane is not None else {"opacity": 0.75}
        try:
            self._overlay({"what": "rgb", "acq": str(acq_id), "opacity": params["opacity"]})
        except (project.NoData, area_mod.NotPlaced, RuntimeError, ValueError) as exc:
            self._say("Sentinel-2", str(exc), "warning")

    def export_product(self, pid: str, chunk=None) -> None:
        import shutil

        from . import project

        chunk = chunk or self._chunk()
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

    def _save_path(self, title: str, name: str, filters: str = "GeoTIFF (*.tif)") -> Path | None:
        window = self.context.window
        if window is None:
            return None
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(window, title, name, filters)
        return Path(path) if path else None

    def export_date_dialog(self, index: str, acq_id: str, chunk=None) -> None:
        """One index of one date as GeoTIFF (and its copy on the chunk's grid)."""
        from . import project

        chunk = chunk or self._chunk()
        path = self._save_path(t("Esporta l'indice della data", "Export the index of the date"), f"{index}_{acq_id}.tif")
        if chunk is None or path is None:
            return
        try:
            out = project.export_date(chunk, index, acq_id, path)
        except (project.NoData, OSError) as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        self._status(t("Esportato: {0}", "Exported: {0}").format(", ".join(str(p) for p in out)))

    def export_acquisition_dialog(self, acq_id: str, chunk=None) -> None:
        """The reflectance of an acquisition of the cache as GeoTIFF."""
        from . import project

        chunk = chunk or self._chunk()
        path = self._save_path(t("Esporta l'acquisizione", "Export the acquisition"), f"{acq_id}_reflectance.tif")
        if chunk is None or path is None:
            return
        try:
            out = project.export_acquisition(chunk, acq_id, path)
        except (project.NoData, OSError) as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        self._status(t("Esportato: {0}", "Exported: {0}").format(out))

    def export_series_dialog(self, chunk=None) -> None:
        """The series (CSV per date, per month beside it, and the chart)."""
        from . import project

        chunk = chunk or self._chunk()
        path = self._save_path(t("Esporta la serie", "Export the series"), "sentinel2_series.csv", "CSV (*.csv)")
        if chunk is None or path is None:
            return
        try:
            out = project.export_series(chunk, path)
        except (project.NoData, OSError) as exc:
            self._say("Sentinel-2", str(exc), "warning")
            return
        self._status(t("Esportato: {0}", "Exported: {0}").format(", ".join(str(p) for p in out)))

    def open_qgis(self) -> None:
        window = self.context.window if self.context else None
        if window is not None and hasattr(window, "open_in_qgis"):
            window.open_in_qgis()

    def export_dialog(self, chunk=None) -> None:
        from . import project

        chunk = chunk or self._chunk()
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

    def open_chart(self, chunk=None) -> None:
        from . import project

        chunk = chunk or self._chunk()
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

    # ------------------------------------------------------------------ the project's tree (program 1.3.4)
    def _focus(self, chunk) -> bool:
        """The chunk of an item of the tree made the active one (the view, its overlays and the pane
        show the active chunk), as the tree itself does when a chunk is clicked."""
        if chunk is None:
            return False
        if self._chunk() is chunk:
            return True
        window = self.context.window if self.context else None
        activate = getattr(window, "_activate_chunk", None) if window is not None else None
        if activate is not None:
            activate(chunk)
        if self._chunk() is chunk:
            return True
        self._say("Sentinel-2", t("Rendi attivo il chunk «{0}» per vederlo.", "Make the chunk «{0}» the active one to "
                                  "see it.").format(getattr(chunk, "label", "")), "warning")
        return False

    def _tree_show_pane(self, chunk) -> None:
        if self._focus(chunk):
            self.show_pane()

    def _tree_show_frame(self, chunk) -> None:
        if self._focus(chunk):
            self.show_frame()

    def _tree_show_acquisition(self, chunk, acq_id: str) -> None:
        if self._focus(chunk):
            self.show_acquisition(acq_id)

    def _tree_show_index(self, chunk, index: str, period: str) -> None:
        """An index of a date (or of the whole period) over the view; the pane follows the choice."""
        if not self._focus(chunk):
            return
        params = {"index": index, "period": period, "palette": "rdylgn", "opacity": 0.75}
        if self.pane is not None:
            p = self.pane
            for combo, value in ((p.c_index, index), (p.c_period, period)):
                k = combo.findData(value)
                if k >= 0:
                    combo.setCurrentIndex(k)
            params.update(palette=p.view_params()["palette"], opacity=p.view_params()["opacity"])
        self.show_overlay(params)

    def _tree_show_product(self, chunk, pid: str) -> None:
        if self._focus(chunk):
            self.show_product(pid)

    def _tree_items(self, chunk) -> list[dict]:
        """The *Sentinel-2* node of a chunk in the project's tree: the acquisitions read (date and clear
        part of the area), the indices by date, the series, the composites and the crop-mark maps; a
        double click shows one over the view, the context menu has its actions. Nothing for a chunk
        the plugin has not worked on. Called by the program at every refresh of the tree: it reads
        only ``chunk.meta`` and the size of the cache."""
        from functools import partial

        from . import analysis, project
        from .texts import index_name, period_name, product_name

        st = project.peek(chunk)
        if not st.get("area") and not st.get("scenes"):
            return []
        c = chunk
        rows = {r["id"]: r for r in st.get("scenes") or []}
        cached = project.cached_ids(chunk)
        files, size = project.cache_size(chunk)
        show, hide = t("Mostra", "Show"), t("Nascondi", "Hide")
        export, out_of_cache = t("Esporta...", "Export..."), t("Rimuovi dalla cache", "Remove from the cache")

        def clear_text(value) -> str:
            return "" if value is None else t(", sereno {0:.0f} %", ", {0:.0f} % clear").format(float(value) * 100)

        acq_items = []
        for acq in reversed(cached):
            r = rows[acq]
            acq_items.append({
                "label": f"{r['date']} {r.get('platform', '')}{clear_text(r.get('clear'))}",
                "icon": "ortho",
                "tooltip": t("{0}: tile {1}, nuvole del tile {2}; doppio clic: colori naturali",
                             "{0}: tiles {1}, clouds of the tiles {2}; double click: natural colour").format(
                    acq, " ".join(r.get("tiles") or []), "-" if r.get("cloud") is None else f"{float(r['cloud']):.0f} %"),
                "activate": partial(self._tree_show_acquisition, c, acq),
                "actions": [(t("Mostra (colori naturali)", "Show (natural colour)"), partial(self._tree_show_acquisition, c, acq)),
                            (hide, self.hide_overlay),
                            (export, partial(self.export_acquisition_dialog, acq, c)),
                            (out_of_cache, partial(self.remove_scenes, [acq], True, c))]})
        groups = [{
            "label": t("Acquisizioni lette ({0})", "Acquisitions read ({0})").format(len(cached)),
            "icon": "photos",
            "tooltip": t("Le finestre dell'area lette dal catalogo, nella cache del progetto: {0} file, {1:.1f} MB",
                         "The windows of the area read from the catalog, in the project's cache: {0} files, "
                         "{1:.1f} MB").format(files, size / 1e6),
            "children": acq_items, "expanded": False,
            "actions": [(t("Svuota la cache...", "Empty the cache..."), partial(self.remove_scenes, None, True, c))]
            if (cached or files) else []}]
        ds = st.get("dates") or []
        if ds:
            index_nodes = []
            for name in analysis.INDICES:
                leaves = [{"label": f"{d['date']} {d.get('platform', '')}{clear_text(d.get('clear'))}",
                           "icon": "index",
                           "tooltip": f"{name} = {analysis.FORMULAS[name]}  ({d['id']})",
                           "activate": partial(self._tree_show_index, c, name, d["id"]),
                           "actions": [(show, partial(self._tree_show_index, c, name, d["id"])),
                                       (hide, self.hide_overlay),
                                       (export, partial(self.export_date_dialog, name, d["id"], c)),
                                       (out_of_cache, partial(self.remove_scenes, [d["id"]], True, c))]}
                          for d in reversed(ds)]
                index_nodes.append({
                    "label": index_name(name), "icon": "index", "tooltip": f"{name} = {analysis.FORMULAS[name]}",
                    "children": leaves, "expanded": False,
                    "actions": [(t("Mostra la mediana di {0}", "Show the median of {0}").format(period_name("all")),
                                 partial(self._tree_show_index, c, name, "all")), (hide, self.hide_overlay)]})
            groups.append({"label": t("Indici per data ({0} date)", "Indices by date ({0} dates)").format(len(ds)),
                           "icon": "index", "tooltip": t("Una banda per data serena, solo le celle serene",
                                                         "A band per clear date, clear cells only"),
                           "children": index_nodes, "expanded": False})
        if st.get("series"):
            groups.append({"label": t("Serie temporale", "Time series"), "icon": "graph",
                           "tooltip": (st["series"].get("attribution") or "") + "\n"
                           + t("Doppio clic: il grafico", "Double click: the chart"),
                           "activate": partial(self.open_chart, c),
                           "actions": [(t("Apri il grafico", "Open the chart"), partial(self.open_chart, c)),
                                       (t("Esporta la serie (CSV)...", "Export the series (CSV)..."),
                                        partial(self.export_series_dialog, c))]})
        for kind, names, icon_name in (("composite", ("Compositi", "Composites"), "multispectral"),
                                       ("cropmarks", ("Crop mark", "Crop marks"), "zones")):
            items = []
            for e in st.get("products") or []:
                if e.get("kind") != kind:
                    continue
                missing = e.get("missing") or []
                tip = [e["id"], t("{0} date: {1}", "{0} dates: {1}").format(len(e.get("dates") or []),
                                                                          ", ".join(e.get("dates") or []))]
                stats = e.get("stats") or {}
                if kind == "cropmarks" and "low" in stats:
                    tip.append(t("{0:.1f}% dell'area sotto -2, {1:.1f}% sopra 2", "{0:.1f}% of the area under -2, "
                                 "{1:.1f}% over 2").format(stats["low"] * 100, stats["high"] * 100))
                if missing:
                    tip.append(t("Fatto anche da acquisizioni non più in cache: {0}", "Made also from acquisitions no "
                                 "longer in the cache: {0}").format(", ".join(missing)))
                items.append({
                    "label": product_name(e) + (t(" (date non più in cache)", " (dates no longer cached)") if missing else ""),
                    "icon": icon_name, "tooltip": "\n".join(tip),
                    "activate": partial(self._tree_show_product, c, e["id"]),
                    "actions": [(show, partial(self._tree_show_product, c, e["id"])),
                                (hide, self.hide_overlay),
                                (export, partial(self.export_product, e["id"], c)),
                                (t("Apri in QGIS", "Open in QGIS"), self.open_qgis),
                                (t("Rimuovi", "Remove"), partial(self.remove, e["id"], True, c))]})
            if items:
                groups.append({"label": f"{t(*names)} ({len(items)})", "icon": icon_name, "children": items,
                               "expanded": True})
        search = st.get("search") or {}
        tip = self._area_text(chunk)
        if search:
            tip += "\n" + t("Ricerca: {0} - {1}, nuvole fino a {2:.0f} %, {3} acquisizioni", "Search: {0} - {1}, clouds "
                            "up to {2:.0f} %, {3} acquisitions").format(search.get("start", ""), search.get("end", ""),
                                                                       float(search.get("max_cloud", 0)), len(rows))
        return [{
            "label": "Sentinel-2", "icon": "globe", "tooltip": tip, "expanded": True, "children": groups,
            "activate": partial(self._tree_show_pane, c),
            "actions": [(t("Mostra il pannello", "Show the pane"), partial(self._tree_show_pane, c)),
                        (t("Mostra l'area intera", "Show the whole area"), partial(self._tree_show_frame, c)),
                        (t("Nascondi la sovrapposizione", "Hide the overlay"), self.hide_overlay),
                        (t("Esporta tutto...", "Export everything..."), partial(self.export_dialog, c)),
                        (t("Apri in QGIS", "Open in QGIS"), self.open_qgis),
                        (t("Svuota la cache...", "Empty the cache..."), partial(self.remove_scenes, None, True, c))]}]

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

    @staticmethod
    def glue_values(args: list[str]) -> list[str]:
        """``--bbox -1.83,51.17,-1.82,51.18``: argparse takes a value beginning with a dash for an
        option; glued to its option (``--bbox=-1.83,...``) it is read as the value. A lone negative
        number (``--bbox -1.83 51.17 -1.82 51.18``) argparse reads by itself."""
        import re

        out, i = [], 0
        while i < len(args):
            a = args[i]
            nxt = args[i + 1] if i + 1 < len(args) else None
            if (a in ("--bbox", "--polygon") and nxt is not None and nxt.startswith("-") and not nxt.startswith("--")
                    and not re.fullmatch(r"-\d+(\.\d*)?|-\.\d+", nxt)):
                out.append(f"{a}={nxt}")
                i += 2
                continue
            out.append(a)
            i += 1
        return out

    @staticmethod
    def bbox_values(values) -> tuple[float, float, float, float]:
        """West, south, east, north from ``--bbox`` (one ``W,S,E,N`` or four numbers)."""
        parts = [v for v in ",".join(values if isinstance(values, (list, tuple)) else [values]).replace(" ", ",").split(",")
                 if v]
        if len(parts) != 4:
            raise ValueError(f"--bbox needs four numbers, west,south,east,north in degrees (got {len(parts)})")
        w, s, e, n = (float(v) for v in parts)
        return w, s, e, n

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
        p.add_argument("--bbox", nargs="+", metavar="W,S,E,N",
                       help="a box instead: west,south,east,north in degrees (also as four numbers, or --bbox=...)")
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
        p = sub.add_parser("compute", help="indices and series from the acquisitions read (no network)")
        p.add_argument("--min-clear", type=float, default=None,
                       help="least clear part of the area, 0-1 (default: the one of the last reading; saved)")
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
        p = sub.add_parser("remove-scene", help="acquisitions out of the cache (fetch reads them again); --all empties it")
        p.add_argument("ids", nargs="*", help="acquisition ids (see list)")
        p.add_argument("--all", action="store_true", help="every acquisition of the cache")
        p.add_argument("--with-products", action="store_true",
                       help="remove the products computed from them too (by default they stay, marked as made from "
                            "acquisitions no longer in the cache)")
        return parser

    def command(self, *args) -> int:
        from . import analysis, project, stac
        from . import area as area_mod

        args, project_path, label = self._project_args(args)
        ns = self._parser().parse_args(self.glue_values(args))
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
                    w, s, e, n = self.bbox_values(ns.bbox)
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
                print(json.dumps(run("Sentinel-2 compute", lambda p: project.compute(chunk, p, ns.min_clear)), indent=1))
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
            elif ns.what == "remove-scene":
                if ns.all == bool(ns.ids):
                    print("give the acquisitions to remove from the cache, or --all (see ppg plugin sentinel2 list)")
                    return 1
                cached = project.cached_ids(chunk)
                unknown = [i for i in ns.ids if i not in cached]
                if unknown:
                    print(f"not in the cache: {', '.join(unknown)} (in the cache: {', '.join(cached) or 'nothing'})")
                    return 1
                r = run("Sentinel-2 cache", lambda p: project.remove_scenes(chunk, None if ns.all else ns.ids,
                                                                           ns.with_products, p))
                print(json.dumps(r, indent=1))
        except (area_mod.NotPlaced, project.NoData, stac.CatalogError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}")
            if doc is not None and getattr(doc, "path", None):
                doc.save()
            return 1
        if doc is not None and getattr(doc, "path", None):
            doc.save()
        return 0
