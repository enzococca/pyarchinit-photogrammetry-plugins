"""Geoarchaeology - a plugin of pyarchinit-photogrammetry: the logs of boreholes, cores and trench
sections turned into surfaces and a 3D model of the deposits, on the ground of the
photogrammetric survey (the DEM and the orthomosaic of the chunk).

* the *Geoarcheologia* pane of the project dock: import of the logs (CSV of collars and
  intervals, or one table), the order of the units, the contact surfaces and the thicknesses by
  kriging, the sections (a trace drawn on the view, or through the boreholes chosen), the
  stratigraphic columns, the 3D model of the units with their volumes; the boreholes as a layer of
  points of the view, a surface or a thickness over the orthomosaic or DEM with palette and
  opacity, the exports, the QGIS project;
* *Plugin > Geoarcheologia* menu actions for the same; *File > Esporta > Sondaggi e superfici
  geoarcheologiche (GeoPackage)*;
* ``ppg plugin geoarchaeology import|list|order|surfaces|section|model|logs|export|remove PROJECT ...``.

Everything lives in ``chunk.meta["geoarchaeology"]`` and in ``<chunk data>/geoarchaeology/``.
"""
from __future__ import annotations

import json
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import t

__version__ = "0.1.0"
SECTION_COLOUR = "#ff7f0e"


class GeoarchaeologyPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Geoarcheologia", "Geoarchaeology")
        self.description = t("Sondaggi, carote e sezioni di scavo: colonne stratigrafiche, superfici di contatto e "
                             "spessori per kriging, sezioni, modello 3D delle unità con i volumi",
                             "Boreholes, cores and trench logs: stratigraphic columns, contact surfaces and thicknesses "
                             "by kriging, sections, a 3D model of the units with their volumes")
        self.context = None
        self.pane = None
        self.overlay = None
        self._armed = False
        self._took_line = False
        self._show_holes = False

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Importa sondaggi...", "Import boreholes..."), self.import_dialog,
                           tip=t("CSV dei collari e degli intervalli, o una tabella sola",
                                 "CSV of the collars and of the intervals, or one table"))
        context.add_action(t("Superfici di contatto (kriging)...", "Contact surfaces (kriging)..."), self.surfaces_dialog,
                           processing=True)
        context.add_action(t("Sezione geoarcheologica...", "Geoarchaeological section..."), self.draw_section)
        context.add_action(t("Colonne stratigrafiche...", "Stratigraphic columns..."), lambda: self.logs_dialog([]))
        context.add_action(t("Modello 3D delle unità...", "3D model of the units..."), self.model_dialog, processing=True)
        context.add_action(t("Geoarcheologia...", "Geoarchaeology..."), self.show_pane,
                           tip=t("Il pannello Geoarcheologia del dock del progetto",
                                 "The Geoarcheologia pane of the project dock"))
        context.add_action(t("Nascondi la sovrapposizione", "Hide the overlay"), self.hide_overlay)
        context.add_command("geoarchaeology", self.command,
                            help="boreholes and their logs, contact surfaces by kriging, sections, 3D model of the units: "
                                 "import, list, order, surfaces, section, model, logs, export, remove")
        context.add_exporter(t("Sondaggi e superfici geoarcheologiche (GeoPackage)",
                               "Boreholes and geoarchaeological surfaces (GeoPackage)"), ".gpkg", self._export_gpkg)
        if context.window is not None:
            self._install_pane(context.window)

    def deactivate(self) -> None:
        self.hide_overlay()
        window = self.context.window if self.context else None
        if window is not None:
            self._arm(False)
            if self.pane is not None:
                tabs = getattr(window, "left_tabs", None)
                if tabs is not None and tabs.indexOf(self.pane) >= 0:
                    tabs.removeTab(tabs.indexOf(self.pane))
            window.qgis_providers = [p for p in getattr(window, "qgis_providers", []) if p != self._qgis_entries]
            instances = getattr(window, "_geoarchaeology_instances", [])
            if self in instances:
                instances.remove(self)
            from .overlay import BOREHOLES_LAYER, SECTIONS_LAYER

            getattr(window, "line_layers", {}).pop(SECTIONS_LAYER, None)
            view = getattr(window, "ortho_view", None)
            for layer in (BOREHOLES_LAYER, SECTIONS_LAYER):
                if view is not None and layer in getattr(view, "layers", {}):
                    view.remove_layer(layer)
        self._show_holes = False
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import GeoarchaeologyPane

        try:
            from pyarchinit_photogrammetry.gui.icons import icon

            ic = icon("line")
        except Exception:  # noqa: BLE001
            ic = None
        self.pane = GeoarchaeologyPane()
        tabs = getattr(window, "left_tabs", None)
        if tabs is not None:
            if ic is not None:
                tabs.addTab(self.pane, ic, t("Geoarcheologia", "Geoarchaeology"))
            else:
                tabs.addTab(self.pane, t("Geoarcheologia", "Geoarchaeology"))
        p = self.pane
        p.import_requested.connect(self.import_dialog)
        p.order_requested.connect(self.order_dialog)
        p.surfaces_requested.connect(self.surfaces_dialog)
        p.model_requested.connect(self.model_dialog)
        p.section_draw_requested.connect(self.draw_section)
        p.section_through_requested.connect(self.section_through)
        p.logs_requested.connect(self.logs_dialog)
        p.boreholes_layer_requested.connect(self.show_boreholes)
        p.overlay_requested.connect(self.show_overlay)
        p.opacity_changed.connect(lambda v: self.overlay.set_opacity(v) if self.overlay else None)
        p.hide_requested.connect(self.hide_overlay)
        p.export_requested.connect(self.export_dialog)
        p.open_requested.connect(self.open_section)
        p.remove_requested.connect(self.remove_dialog)
        p.qgis_requested.connect(lambda: window.open_in_qgis() if hasattr(window, "open_in_qgis") else None)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        self.refresh_pane()
        # the pane follows the project, the layer of the boreholes the raster shown: the window is
        # hooked once, for every active instance of the plugin
        instances = getattr(window, "_geoarchaeology_instances", None)
        if instances is None:
            window._geoarchaeology_instances = instances = []
            original = getattr(window, "refresh_all", None)
            if original is not None:
                def refresh_all(*args, _orig=original, **kwargs):
                    result = _orig(*args, **kwargs)
                    for plugin in list(window._geoarchaeology_instances):
                        plugin.refresh_pane()
                    return result

                window.refresh_all = refresh_all
            load = getattr(window, "_load_raster", None)
            if load is not None:
                def load_raster(*args, _orig=load, **kwargs):
                    result = _orig(*args, **kwargs)
                    for plugin in list(window._geoarchaeology_instances):
                        if plugin._show_holes:
                            plugin._draw_boreholes()
                    return result

                window._load_raster = load_raster
        instances.append(self)

    def refresh_pane(self) -> None:
        from . import boreholes as bh
        from . import section as section_mod
        from . import surfaces as sf

        if self.pane is None:
            return
        chunk = self._chunk()
        if chunk is None:
            self.pane.set_state([], [], [], "")
            return
        holes = bh.load(chunk)
        reg = sf.registry(chunk)
        if not holes:
            status = t("Nessun sondaggio: importa le tabelle dei collari e degli intervalli (CSV).",
                       "No borehole: import the tables of the collars and of the intervals (CSV).")
        else:
            codes = bh.order(chunk)
            status = t("{0} sondaggi, {1} unità dall'alto: {2}.", "{0} boreholes, {1} units from the top: {2}.").format(
                len(holes), len(codes), ", ".join(codes[:12]) + ("..." if len(codes) > 12 else ""))
            if reg:
                g = reg["grid"]
                status += " " + t("Superfici del {0} su {1} x {2} celle di {3:g} m.",
                                  "Surfaces of {0} on {1} x {2} cells of {3:g} m.").format(reg["made"], g["width"],
                                                                                         g["height"], g["cell"])
                if reg.get("stale"):
                    status += " " + t("I sondaggi o l'ordine sono cambiati: ricostruiscile.",
                                      "The boreholes or the order changed: build them again.")
        self.pane.set_state(holes, sf.entries(chunk), section_mod.entries(chunk), status)
        self._sync_traces()

    def show_pane(self) -> None:
        window = self.context.window if self.context else None
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
        return None if doc is None else doc.chunk

    def _message(self, kind: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text)

    def _need_boreholes(self):
        from . import boreholes as bh

        chunk = self._chunk()
        if chunk is None:
            self._message("warning", t("Apri un progetto con un chunk.", "Open a project with a chunk."))
            return None
        if not bh.load(chunk):
            self._message("warning", t("Importa prima i sondaggi.", "Import the boreholes first."))
            return None
        return chunk

    def _run(self, title: str, job, done=None) -> None:
        window = self.context.window
        if window is not None and hasattr(window, "run_task"):
            window.run_task(title, job, done, refresh=False, save=True)
        else:
            self.context.run(title, job, done)

    def _after(self, text: str = "") -> None:
        self.refresh_pane()
        window = self.context.window if self.context else None
        if window is not None:
            if hasattr(window, "_dirty"):
                window._dirty(True)
            if text and hasattr(window, "statusBar"):
                window.statusBar().showMessage(text, 15000)

    def _open_file(self, path) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ import and order
    def import_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import boreholes as bh

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        csv_filter = "CSV (*.csv *.txt *.tsv *.dat);;" + t("Tutti", "All") + " (*)"
        v = ParamDialog.ask(window, t("Importa sondaggi", "Import boreholes"), [
            Field("first", t("Collari, o tabella unica", "Collars, or a single table"), "", kind="file_open",
                  filter=csv_filter),
            Field("second", t("Intervalli (se in un file a parte)", "Intervals (when in a separate file)"), "",
                  kind="file_open", filter=csv_filter),
            Field("kind", t("Tipo", "Kind"), "", choices=[
                ("", t("dalla tabella", "from the table")), ("auger", t("trivella manuale", "hand auger")),
                ("core", t("carotaggio", "core")), ("trench", t("sezione di scavo", "trench section")),
                ("test_pit", t("saggio", "test pit")), ("other", t("altro", "other"))]),
            Field("replace", t("Sostituisci i sondaggi del chunk (altrimenti si aggiungono)",
                               "Replace the boreholes of the chunk (otherwise they are added)"), False),
        ], intro=t("Collari: sondaggio (id), x, y nel sistema del chunk, quota (z, vuota = dal DEM), profondità. "
                   "Intervalli: sondaggio, da, a (metri dalla bocca; oppure quota_tetto e quota_letto), unità (US o "
                   "unità litologica), descrizione, colore (Munsell o #rrggbb), litologia. Nomi delle colonne in "
                   "italiano o in inglese, separatore virgola, punto e virgola o tabulazione, decimali col punto o "
                   "con la virgola.",
                   "Collars: borehole (id), x, y in the reference system of the chunk, elevation (z, empty = from the "
                   "DEM), depth. Intervals: borehole, from, to (metres below the collar; or top_z and base_z), unit (US "
                   "or lithological unit), description, colour (Munsell or #rrggbb), lithology. Column names in Italian "
                   "or English, comma, semicolon or tab separated, decimal point or comma."))
        if not v or not v["first"]:
            return
        paths = [p for p in (v["first"], v["second"]) if p]

        def done(report):
            self._after(t("{0} sondaggi letti", "{0} boreholes read").format(report["read"]))
            self._message("info", self.import_text(report))

        self._run(t("Sondaggi", "Boreholes"),
                  lambda p: bh.import_files(chunk, paths, replace=bool(v["replace"]), kind=v["kind"] or "", progress=p),
                  done)

    @staticmethod
    def import_text(report: dict) -> str:
        lines = [t("{0} sondaggi letti ({1} nel chunk), {2} intervalli; unità dall'alto: {3}",
                   "{0} boreholes read ({1} in the chunk), {2} intervals; units from the top: {3}")
                 .format(report["read"], report["boreholes"], report["intervals"], ", ".join(report["units"]))]
        if report.get("z_from_dem"):
            lines.append(t("{0} quote prese dal DEM", "{0} elevations taken from the DEM").format(report["z_from_dem"]))
        if report.get("no_z"):
            lines.append(t("Senza quota (né nel file né dal DEM): {0}", "Without elevation (neither in the file nor from "
                           "the DEM): {0}").format(", ".join(report["no_z"])))
        chk = report.get("dem_check")
        if chk:
            lines.append(t("Quota dei collari meno DEM: mediana {0:+.3f} m, massimo {1:+.3f} m ({2})",
                           "Collar elevation minus DEM: median {0:+.3f} m, largest {1:+.3f} m ({2})")
                         .format(chk["median_m"], chk["max_m"], chk["worst"]))
        warnings = report.get("warnings") or []
        if warnings:
            lines.append(t("Avvisi ({0}):", "Warnings ({0}):").format(len(warnings)))
            lines += ["- " + w for w in warnings[:15]] + (["..."] if len(warnings) > 15 else [])
        return "\n".join(lines)

    def order_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import boreholes as bh

        chunk = self._need_boreholes()
        window = self.context.window
        if chunk is None or window is None:
            return
        conflicts = bh.meta(chunk).get("order_conflicts") or []
        intro = t("L'ordine viene dai sondaggi (un'unità sopra un'altra in un log sta sopra di essa); scrivilo qui "
                  "per cambiarlo. Le superfici e il modello vanno ricostruiti dopo.",
                  "The order comes from the boreholes (a unit above another in a log lies above it); write it here to "
                  "change it. The surfaces and the model are to be built again afterwards.")
        if conflicts:
            intro += "\n\n" + t("Disaccordi fra i sondaggi:", "Disagreements between the boreholes:") + "\n- " + \
                "\n- ".join(conflicts[:8])
        v = ParamDialog.ask(window, t("Ordine delle unità", "Order of the units"), [
            Field("order", t("Dall'alto in basso, separate da virgole", "From the top down, separated by commas"),
                  ", ".join(bh.order(chunk)), kind="text")], intro=intro)
        if not v:
            return
        bh.set_order(chunk, [c.strip() for c in str(v["order"]).split(",") if c.strip()])
        self._after(t("Ordine delle unità aggiornato", "Order of the units updated"))

    # ------------------------------------------------------------------ surfaces and model
    def surfaces_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import surfaces as sf

        chunk = self._need_boreholes()
        window = self.context.window
        if chunk is None:
            return
        v = {"model": "auto", "trend": "auto", "mode": "elevation", "cell": None, "nugget": None, "buffer": None,
             "min_points": 5}
        if window is not None:
            v = ParamDialog.ask(window, t("Superfici di contatto", "Contact surfaces"), [
                Field("model", t("Modello del variogramma", "Variogram model"), "auto", choices=[
                    ("auto", t("il migliore (errore leave-one-out)", "the best (leave-one-out error)")),
                    ("spherical", t("sferico", "spherical")), ("exponential", t("esponenziale", "exponential")),
                    ("gaussian", t("gaussiano", "Gaussian"))]),
                Field("trend", t("Andamento", "Trend"), "auto", choices=[
                    ("auto", t("scelto dai dati", "chosen by the data")), ("none", t("nessuno (kriging ordinario)",
                                                                                    "none (ordinary kriging)")),
                    ("linear", t("piano inclinato (kriging universale)", "tilted plane (universal kriging)"))]),
                Field("mode", t("Interpola", "Interpolate"), "elevation", choices=[
                    ("elevation", t("le quote dei contatti", "the elevations of the contacts")),
                    ("depth", t("le profondità dal terreno (seguono il DEM)", "the depths below the ground (they follow "
                                                                              "the DEM)"))]),
                Field("cell", t("Cella (m, vuota = quella del DEM)", "Cell (m, empty = the DEM's)"), None,
                      kind="optional_float"),
                Field("nugget", t("Pepita (m², vuota = stimata)", "Nugget (m², empty = fitted)"), None,
                      kind="optional_float"),
                Field("buffer", t("Margine attorno ai sondaggi (m, vuoto = la loro distanza tipica)",
                                  "Margin around the boreholes (m, empty = their usual spacing)"), None,
                      kind="optional_float"),
                Field("min_points", t("Sondaggi minimi per il kriging (sotto: IDW)", "Least boreholes for kriging (below: "
                                                                                     "IDW)"), 5, minimum=3, maximum=50),
            ], intro=t("Il tetto di ogni unità viene interpolato dai sondaggi: dove un'unità manca il suo tetto è il "
                       "contatto fra le unità sopra e sotto. Nessuna superficie sale sopra il terreno o sopra quella "
                       "dell'unità superiore (le correzioni sono contate e restano nella terza banda). Banda 2: la "
                       "varianza di kriging.",
                       "The top of every unit is interpolated from the boreholes: where a unit is missing its top is "
                       "the contact between the units above and below. No surface rises above the ground or above that "
                       "of the unit over it (the corrections are counted and kept in the third band). Band 2: the "
                       "kriging variance."))
            if not v:
                return

        def done(reg):
            self._after(t("Superfici pronte", "Surfaces ready"))
            if window is not None:
                self._message("info", self.surfaces_text(reg))

        self._run(t("Superfici di contatto", "Contact surfaces"),
                  lambda p: sf.build(chunk, model=v["model"], cell=v["cell"], nugget=v["nugget"], mode=v["mode"],
                                     trend=v["trend"], buffer=v["buffer"], min_points=int(v["min_points"]), progress=p),
                  done)

    @staticmethod
    def surfaces_text(reg: dict) -> str:
        g = reg["grid"]
        lines = [t("{0} x {1} celle di {2:g} m", "{0} x {1} cells of {2:g} m").format(g["width"], g["height"], g["cell"])]
        for e in reg.get("contacts", []):
            how = {"ground": t("il terreno", "the ground"), "idw": "IDW", "none": t("nessun dato", "no data"),
                   "constant": t("costante", "constant")}.get(e.get("method"), e.get("method", ""))
            if e.get("method") == "kriging":
                how = t("kriging {0}{1}, errore leave-one-out {2:.3f} m", "kriging {0}{1}, leave-one-out error {2:.3f} m") \
                    .format(e["variogram"]["model"], t(" con piano", " with a plane") if e.get("trend") == "linear" else "",
                            e.get("loocv_rmse", 0.0))
            cl = e.get("clamped") or {}
            extra = ""
            if cl.get("cells"):
                extra = t("; abbassata su {0:.1f} m² (max {1:.3f} m: {2} celle dal terreno, {3} dall'unità sopra)",
                          "; lowered on {0:.1f} m² (max {1:.3f} m: {2} cells by the ground, {3} by the unit above)") \
                    .format(cl["area_m2"], cl["max_m"], cl["by_ground"], cl["by_upper"])
            if e.get("bottoms_used"):
                extra += t("; {0} fondi di sondaggi usati come vincolo", "; {0} borehole bottoms used as a bound") \
                    .format(e["bottoms_used"])
            lines.append(f"{e['name']}: {how}{extra}")
        for e in reg.get("thickness", []):
            st = e.get("stats") or {}
            lines.append(t("{0}: media {1:.2f} m, massimo {2:.2f} m, {3:.1f} m³",
                           "{0}: mean {1:.2f} m, largest {2:.2f} m, {3:.1f} m³").format(
                e["name"], st.get("mean", 0.0), st.get("max", 0.0), e.get("volume_m3", 0.0)))
        if reg.get("notes"):
            lines.append(t("Sondaggi fuori ordine:", "Boreholes out of order:") + " " + "; ".join(reg["notes"][:5]))
        return "\n".join(lines)

    def model_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import model as model_mod
        from . import surfaces as sf

        chunk = self._need_boreholes()
        window = self.context.window
        if chunk is None:
            return
        if not sf.registry(chunk).get("contacts"):
            self._message("warning", t("Costruisci prima le superfici di contatto.", "Build the contact surfaces first."))
            return
        v = {"cell": None, "clip": True, "last": True, "floor": None}
        if window is not None:
            v = ParamDialog.ask(window, t("Modello 3D delle unità", "3D model of the units"), [
                Field("cell", t("Cella (m, vuota = automatica)", "Cell (m, empty = automatic)"), None, kind="optional_float"),
                Field("clip", t("Ritaglia le unità sulle US disegnate con lo stesso numero",
                                "Clip the units to the US drawn with the same number"), True),
                Field("last", t("Anche l'ultima unità, fino al fondo", "The last unit too, down to the floor"), True),
                Field("floor", t("Quota del fondo (m, vuota = il fondo più basso dei sondaggi)",
                                 "Elevation of the floor (m, empty = the lowest bottom of the boreholes)"), None,
                      kind="optional_float"),
            ], intro=t("Un solido chiuso per unità fra il suo tetto e quello dell'unità sotto (PLY con il colore "
                       "dell'unità, un OBJ con un oggetto per unità), i volumi in un CSV.",
                       "A closed solid per unit between its top and the top of the unit below (PLY with the colour of "
                       "the unit, one OBJ with an object per unit), the volumes in a CSV."))
            if not v:
                return
            self.context.store_drawings()  # the US drawn in the window into the project

        def done(reg):
            self._after(t("Modello pronto", "Model ready"))
            if window is not None:
                rows = [t("{0}: {1:.2f} m³ su {2:.1f} m²{3}", "{0}: {1:.2f} m³ on {2:.1f} m²{3}").format(
                    r["unit"], r["volume_m3"], r.get("area_m2", 0.0),
                    t(" (ritagliata sulle US)", " (clipped to the US)") if r.get("clipped") else "") for r in reg["units"]]
                self._message("info", t("Modello in {0}", "Model in {0}").format(
                    model_mod.folder(chunk)) + "\n" + "\n".join(rows))

        self._run(t("Modello 3D delle unità", "3D model of the units"),
                  lambda p: model_mod.build(chunk, cell=v["cell"], clip=bool(v["clip"]), floor=v["floor"],
                                            last=bool(v["last"]), progress=p), done)

    # ------------------------------------------------------------------ sections
    def draw_section(self) -> None:
        """The line tool of the view lent to the plugin for one trace (two clicks)."""
        chunk = self._need_boreholes()
        window = self.context.window
        if chunk is None or window is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is None:
            return
        if view.tr is None and hasattr(window, "_show_raster"):
            window._show_raster("ortho" if chunk.orthomosaic_path else "dem")
        if view.tr is None:
            self._message("warning", t("Costruisci prima il DEM o l'ortofoto.", "Build the DEM or the orthomosaic first."))
            return
        self._arm(True)
        if hasattr(window, "set_draw_mode"):
            window.set_draw_mode("line")
        if hasattr(window, "show_view"):
            window.show_view(view)
        window.statusBar().showMessage(t("Sezione geoarcheologica: due clic, inizio e fine della traccia",
                                         "Geoarchaeological section: two clicks, start and end of the trace"), 30000)

    def _arm(self, on: bool) -> None:
        window = self.context.window if self.context else None
        view = getattr(window, "ortho_view", None) if window is not None else None
        if view is None:
            return
        group = getattr(window, "draw_group", None)
        if on and not self._armed:
            try:
                view.line_finished.disconnect(window._section_drawn)
                self._took_line = True
            except (RuntimeError, TypeError, AttributeError):
                self._took_line = False
            view.line_finished.connect(self._line_drawn)
            if group is not None:
                group.triggered.connect(self._tool_changed)
            self._armed = True
        elif not on and self._armed:
            for signal, slot in ((view.line_finished, self._line_drawn),
                                 (getattr(group, "triggered", None), self._tool_changed)):
                if signal is None:
                    continue
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            if self._took_line:
                view.line_finished.connect(window._section_drawn)
            self._armed, self._took_line = False, False

    def _tool_changed(self, action) -> None:
        window = self.context.window
        if getattr(window, "draw_actions", {}).get("line") is not action:
            self._arm(False)  # another tool chosen: the line tool goes back to the program's sections

    def _line_drawn(self, a, b) -> None:
        self._arm(False)
        window = self.context.window
        if hasattr(window, "set_draw_mode"):
            window.set_draw_mode("pan")
        self.section_dialog([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])

    def section_through(self, ids: list) -> None:
        from . import section as section_mod

        chunk = self._need_boreholes()
        if chunk is None:
            return
        if len(ids) < 2:
            self._message("warning", t("Seleziona due sondaggi almeno nella tabella.", "Select two boreholes at least in "
                                                                                       "the table."))
            return
        self.section_dialog(section_mod.through(chunk, ids), name=t("Sezione {0}", "Section {0}").format(
            "-".join(str(i) for i in ids)))

    def section_dialog(self, coords, name: str | None = None, ask: bool = True) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import section as section_mod

        chunk = self._chunk()
        window = self.context.window
        if chunk is None:
            return
        v = {"name": name or "", "distance": None, "scale": None, "exaggeration": None}
        if window is not None and ask:
            v = ParamDialog.ask(window, t("Sezione geoarcheologica", "Geoarchaeological section"), [
                Field("name", t("Nome", "Name"), name or "", kind="text"),
                Field("distance", t("Sondaggi entro (m dalla traccia, vuoto = metà della loro distanza tipica)",
                                    "Boreholes within (m of the trace, empty = half their usual spacing)"), None,
                      kind="optional_float"),
                Field("scale", t("Scala 1: (vuota = automatica)", "Scale 1: (empty = automatic)"), None, kind="optional_float"),
                Field("exaggeration", t("Esagerazione verticale (vuota = automatica)", "Vertical exaggeration (empty = "
                                                                                       "automatic)"), None,
                      kind="optional_float"),
            ], intro=t("Il profilo del terreno e del tetto di ogni unità lungo la traccia, le unità colorate fra i "
                       "contatti, i sondaggi vicini proiettati sulla traccia: SVG, PNG, DXF e CSV.",
                       "The profile of the ground and of the top of every unit along the trace, the units coloured "
                       "between the contacts, the nearby boreholes projected onto the trace: SVG, PNG, DXF and CSV."))
            if not v:
                return

        def done(entry):
            self._after(t("{0}: {1:.1f} m, {2} sondaggi", "{0}: {1:.1f} m, {2} boreholes").format(
                entry["name"], entry["length"], len(entry["boreholes"])))
            if self.pane is not None:
                self.pane.tabs.setCurrentWidget(self.pane.t_sections)

        self._run(t("Sezione geoarcheologica", "Geoarchaeological section"),
                  lambda p: section_mod.build(chunk, coords, name=v["name"] or None, distance=v["distance"],
                                              scale=v["scale"], exaggeration=v["exaggeration"], progress=p), done)

    def _sync_traces(self) -> None:
        """The traces of the sections of the chunk as a line layer of the window (kept across the
        reloads of the raster)."""
        from . import section as section_mod
        from .overlay import SECTIONS_LAYER

        window = self.context.window if self.context else None
        chunk = self._chunk()
        if window is None or chunk is None or not hasattr(window, "_add_line_feature"):
            return
        sections = section_mod.entries(chunk)
        have = [attrs.get("id") for _c, _col, _l, attrs, _w in getattr(window, "line_layers", {}).get(SECTIONS_LAYER, [])]
        if have == [s["id"] for s in sections]:
            return
        window.line_layers.pop(SECTIONS_LAYER, None)
        view = getattr(window, "ortho_view", None)
        if view is not None and SECTIONS_LAYER in getattr(view, "layers", {}):
            view.remove_layer(SECTIONS_LAYER)
        for s in sections:
            window._add_line_feature(SECTIONS_LAYER, s["coords"], SECTION_COLOUR, s["name"],
                                     {"id": s["id"], "lunghezza_m": round(s["length"], 2),
                                      "sondaggi": ", ".join(h["id"] for h in s["boreholes"])}, width=2.0)

    def open_section(self, sid: str) -> None:
        from . import section as section_mod

        chunk = self._chunk()
        s = next((s for s in section_mod.entries(chunk) if s["id"] == sid), None) if chunk is not None else None
        if s is None:
            return
        rel = s["files"].get("png") or s["files"].get("svg") or s["files"].get("dxf")
        if rel:
            self._open_file(Path(chunk.data_dir) / rel)

    # ------------------------------------------------------------------ logs
    def logs_dialog(self, ids: list) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import boreholes as bh
        from . import logs

        chunk = self._need_boreholes()
        window = self.context.window
        if chunk is None or window is None:
            return
        every = [b.id for b in bh.load(chunk) if b.intervals]
        v = ParamDialog.ask(window, t("Colonne stratigrafiche", "Stratigraphic columns"), [
            Field("ids", t("Sondaggi (uno: la sua colonna con le descrizioni; più: un pannello)",
                           "Boreholes (one: its column with the descriptions; more: a panel)"), list(ids) or every[:8],
                  choices=[(i, i) for i in every], kind="checklist"),
            Field("vertical", t("Asse verticale", "Vertical axis"), "auto", choices=[
                ("auto", t("quote se tutti le hanno", "elevations when all have one")),
                ("elevation", t("quote", "elevations")), ("depth", t("profondità", "depths"))]),
            Field("scale", t("Scala 1: (vuota = per un foglio A4)", "Scale 1: (empty = to fit an A4 sheet)"), None,
                  kind="optional_float"),
        ])
        if not v or not v["ids"]:
            return
        chosen = [i for i in every if i in v["ids"]] if not ids else [i for i in ids if i in v["ids"]] + \
            [i for i in every if i in v["ids"] and i not in ids]

        def done(paths):
            self._after(t("Colonne in {0}", "Columns in {0}").format(paths[0].parent))
            png = next((p for p in paths if p.suffix == ".png"), paths[0])
            self._open_file(png)

        self._run(t("Colonne stratigrafiche", "Stratigraphic columns"),
                  lambda p: logs.draw(chunk, chosen, vertical=v["vertical"], scale=v["scale"]), done)

    # ------------------------------------------------------------------ the view
    def show_boreholes(self) -> None:
        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        if view is not None and view.tr is None and hasattr(window, "_show_raster"):
            window._show_raster("ortho" if chunk.orthomosaic_path else "dem")
        self._show_holes = True
        if self._draw_boreholes() and view is not None:
            from .overlay import BOREHOLES_LAYER

            view.zoom_to_layer(BOREHOLES_LAYER)
            if hasattr(window, "show_view"):
                window.show_view(view)

    def _draw_boreholes(self) -> int:
        from . import boreholes as bh
        from .overlay import draw_boreholes

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return 0
        return draw_boreholes(getattr(window, "ortho_view", None), bh.load(chunk))

    def show_overlay(self, pid: str, band: int = 1, palette: str = "", opacity: float = 0.7) -> None:
        from . import surfaces as sf
        from .overlay import DEFAULT, Overlay

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        view = getattr(window, "ortho_view", None)
        e = sf.entry(chunk, pid)
        if view is None or e is None:
            return
        if view.tr is None and hasattr(window, "_show_raster"):
            window._show_raster("ortho" if chunk.orthomosaic_path else "dem")
        kind = e.get("kind", "surface")
        band = int(band) if kind == "surface" else 1
        values, transform = sf.load(chunk, pid, band)
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        palette = palette or DEFAULT.get((kind, band), "terrain")
        try:
            lo, hi = self.overlay.show(values, transform, palette, opacity, zero=kind == "thickness" or band > 1,
                                       product_id=pid)
        except RuntimeError as exc:
            self._message("warning", str(exc))
            return
        what = {1: t("quota", "elevation"), 2: t("varianza (m²)", "variance (m²)"),
                3: t("correzione (m)", "correction (m)")}[band] if kind == "surface" else t("spessore (m)", "thickness (m)")
        if self.pane is not None:
            self.pane.set_limits(f"{e['name']}, {what}: {lo:.3f} - {hi:.3f}")
        if hasattr(window, "show_view"):
            window.show_view(view)

    def hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()
        if self.pane is not None:
            self.pane.set_limits("")

    # ------------------------------------------------------------------ exports and removal
    def export_dialog(self, pid: str = "") -> None:
        from PySide6.QtWidgets import QFileDialog

        from .export import export_gpkg, export_raster

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        doc = self.context.app.document
        base = Path(doc.path).parent if getattr(doc, "path", None) else Path.home()
        if pid:
            path, _f = QFileDialog.getSaveFileName(window, t("Esporta la superficie", "Export the surface"),
                                                   str(base / f"{pid}.tif"), "GeoTIFF (*.tif)")
            if path:
                self._message("info", t("Scritto {0}", "Written {0}").format(export_raster(chunk, pid, path)))
            return
        path, _f = QFileDialog.getSaveFileName(window, t("Esporta la geoarcheologia", "Export the geoarchaeology"),
                                               str(base / "geoarcheologia.gpkg"), "GeoPackage (*.gpkg)")
        if path:
            self._run(t("Esportazione", "Export"), lambda p: export_gpkg(chunk, path, window, p),
                      lambda out: self._message("info", t("Scritto {0} (con superfici e volumi accanto)",
                                                          "Written {0} (with surfaces and volumes beside it)").format(out)))

    def _export_gpkg(self, chunk, path, progress=None):
        from .export import export_gpkg

        return export_gpkg(chunk, path, self.context.window if self.context else None, progress)

    def remove_dialog(self, what: str, ident: str) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import choice

        chunk = self._chunk()
        window = self.context.window
        if chunk is None:
            return
        question = {"product": t("Togliere tutte le superfici e gli spessori (formano una pila)?",
                                 "Remove all the surfaces and thicknesses (they form a stack)?"),
                    "section": t("Togliere la sezione {0} e i suoi disegni?", "Remove the section {0} and its drawings?"),
                    "boreholes": t("Togliere i sondaggi {0}?", "Remove the boreholes {0}?")}[what].format(ident)
        if window is not None and choice(window, "question", self.title, question,
                                         [t("Togli", "Remove"), t("Annulla", "Cancel")], default=1) != 0:
            return
        self.remove(what, ident)

    def remove(self, what: str, ident: str = "") -> None:
        from . import boreholes as bh
        from . import section as section_mod
        from . import surfaces as sf

        chunk = self._chunk()
        if chunk is None:
            return
        if what == "product":
            self.hide_overlay()
            sf.remove(chunk)
        elif what == "section":
            section_mod.remove(chunk, ident)
            from .export import write_layers

            write_layers(chunk)
        elif what == "boreholes":
            bh.remove(chunk, [i for i in str(ident).split(",") if i])
            from .export import write_layers

            write_layers(chunk)
            if self._show_holes:
                self._draw_boreholes()
        self._after()

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        """``ppg plugin geoarchaeology WHAT PROJECT ...``: see ``--help``."""
        import argparse

        parser = argparse.ArgumentParser(prog="ppg plugin geoarchaeology",
                                         description="Boreholes, cores and trench logs: stratigraphic columns, contact "
                                                     "surfaces and thicknesses by kriging, sections, a 3D model of the "
                                                     "units with their volumes.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("import", help="tables of collars and intervals (CSV), or one table with both")
        p.add_argument("project")
        p.add_argument("files", nargs="+", help="the collar table and the interval table, or one combined table")
        p.add_argument("--replace", action="store_true", help="forget the boreholes the chunk had")
        p.add_argument("--kind", default="", help="auger, core, trench, test_pit, other (default: from the table)")
        p = sub.add_parser("list", help="boreholes, units, surfaces, sections, model")
        p.add_argument("project")
        p.add_argument("--json", action="store_true", help="the whole record as JSON")
        p = sub.add_parser("order", help="the order of the units from the top down")
        p.add_argument("project")
        p.add_argument("units", help="comma-separated, e.g. 1,2,3,4")
        p = sub.add_parser("surfaces", help="tops of the units and thicknesses by kriging")
        p.add_argument("project")
        p.add_argument("--model", default="auto", choices=["auto", "spherical", "exponential", "gaussian"])
        p.add_argument("--trend", default="auto", choices=["auto", "none", "linear"])
        p.add_argument("--mode", default="elevation", choices=["elevation", "depth"])
        p.add_argument("--cell", type=float, default=None, help="m (default: the DEM's)")
        p.add_argument("--nugget", type=float, default=None, help="m2 (default: fitted)")
        p.add_argument("--buffer", type=float, default=None, help="m around the boreholes (default: their spacing)")
        p.add_argument("--min-points", type=int, default=5, help="fewer boreholes: inverse distance weighting")
        p.add_argument("--order", default=None, help="the units from the top down, comma-separated")
        p = sub.add_parser("section", help="a section along a trace or through boreholes")
        p.add_argument("project")
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--coords", help="'x1,y1;x2,y2;...' in the reference system of the chunk (--coords=... when "
                                        "the first number is negative)")
        g.add_argument("--through", help="borehole ids, comma-separated, in order")
        p.add_argument("--name", default=None)
        p.add_argument("--distance", type=float, default=None, help="m: boreholes within this distance of the trace")
        p.add_argument("--scale", type=float, default=None, help="1:SCALE (default: chosen)")
        p.add_argument("--exaggeration", type=float, default=None, help="vertical (default: chosen)")
        p = sub.add_parser("model", help="a closed solid per unit, PLY and OBJ, volumes in CSV")
        p.add_argument("project")
        p.add_argument("--cell", type=float, default=None)
        p.add_argument("--no-clip", action="store_true", help="do not clip the units to the US drawn")
        p.add_argument("--floor", type=float, default=None, help="elevation of the base of the last unit")
        p.add_argument("--no-last", action="store_true", help="leave the last unit out")
        p.add_argument("--offset", default=None, help="X,Y,Z subtracted from the coordinates of the OBJ")
        p = sub.add_parser("logs", help="stratigraphic columns: one borehole or a panel")
        p.add_argument("project")
        p.add_argument("ids", nargs="*", help="the boreholes (all by default)")
        p.add_argument("--out", default=None, help="the file (.svg, .png, .pdf); both SVG and PNG in the plugin's folder "
                                                   "by default")
        p.add_argument("--vertical", default="auto", choices=["auto", "elevation", "depth"])
        p.add_argument("--scale", type=float, default=None)
        p = sub.add_parser("export", help="GeoPackage (boreholes, intervals, sections; surfaces and volumes beside), or "
                                          "one surface as GeoTIFF with --id")
        p.add_argument("project")
        p.add_argument("out")
        p.add_argument("--id", default=None, help="a surface or thickness (see list)")
        p = sub.add_parser("remove", help="forget what the plugin made")
        p.add_argument("project")
        p.add_argument("target", nargs="?", default="all",
                       help="all (or boreholes: everything goes with them), surfaces, sections, model, the id of a "
                            "section or of a borehole")
        for sp in sub.choices.values():
            sp.add_argument("--chunk", default=None, help="label of the chunk (the active one by default)")
        sub.add_parser("help", help="this help and that of every command (ppg itself answers --help)")
        a = parser.parse_args([str(x) for x in args])
        if a.what == "help":
            print(parser.format_help())
            for name, sp in sub.choices.items():
                if name != "help":
                    print(sp.format_help())
            return 0
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        from . import boreholes as bh
        from . import export as export_mod
        from . import logs
        from . import model as model_mod
        from . import section as section_mod
        from . import surfaces as sf

        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}" if a.chunk else f"no active chunk in {a.project}")
            return 2
        try:
            if a.what == "import":
                report = bh.import_files(chunk, a.files, replace=a.replace, kind=a.kind, progress=print_progress)
                print(self.import_text(report))
            elif a.what == "list":
                print(json.dumps(chunk.meta.get("geoarchaeology") or {}, indent=1, ensure_ascii=False) if a.json
                      else export_mod.summary(chunk))
                return 0
            elif a.what == "order":
                print(", ".join(bh.set_order(chunk, [u.strip() for u in a.units.split(",") if u.strip()])))
            elif a.what == "surfaces":
                codes = [u.strip() for u in a.order.split(",") if u.strip()] if a.order else None
                reg = sf.build(chunk, model=a.model, cell=a.cell, nugget=a.nugget, mode=a.mode, trend=a.trend,
                               buffer=a.buffer, min_points=a.min_points, codes=codes, progress=print_progress)
                print(self.surfaces_text(reg))
            elif a.what == "section":
                if a.coords:
                    coords = [[float(v) for v in pt.replace(" ", "").split(",")] for pt in a.coords.split(";") if pt.strip()]
                else:
                    coords = section_mod.through(chunk, [i.strip() for i in a.through.split(",") if i.strip()])
                e = section_mod.build(chunk, coords, name=a.name, distance=a.distance, scale=a.scale,
                                      exaggeration=a.exaggeration, progress=print_progress)
                print(json.dumps({k: e[k] for k in ("id", "name", "length", "scale", "exaggeration", "boreholes", "files")},
                                 indent=1, ensure_ascii=False))
            elif a.what == "model":
                offset = [float(v) for v in a.offset.split(",")] if a.offset else None
                reg = model_mod.build(chunk, cell=a.cell, clip=not a.no_clip, floor=a.floor, last=not a.no_last,
                                      offset=offset, progress=print_progress)
                for r in reg["units"]:
                    print(f"{r['unit']}: {r['volume_m3']:.3f} m3 on {r.get('area_m2', 0.0):.2f} m2"
                          + (" (clipped to the US drawn)" if r.get("clipped") else ""))
                print(model_mod.folder(chunk))
            elif a.what == "logs":
                for path in logs.draw(chunk, a.ids or None, path=a.out, vertical=a.vertical, scale=a.scale):
                    print(path)
                return 0
            elif a.what == "export":
                print(export_mod.export_raster(chunk, a.id, a.out) if a.id else export_mod.export_gpkg(chunk, a.out))
                return 0
            elif a.what == "remove":
                self._remove_cli(chunk, a.target)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"error: {exc}")
            return 1
        doc.save()
        return 0

    @staticmethod
    def _remove_cli(chunk, what: str) -> None:
        from . import boreholes as bh
        from . import model as model_mod
        from . import section as section_mod
        from . import surfaces as sf
        from .export import write_layers

        if what in ("all", "boreholes"):
            for p in bh.folder(chunk).rglob("*"):
                if p.is_file():
                    p.unlink()
            chunk.meta.pop("geoarchaeology", None)
            print("geoarchaeology removed from the chunk")
            return
        if what == "surfaces":
            sf.remove(chunk)
        elif what == "model":
            model_mod.remove(chunk)
        elif what == "sections":
            section_mod.remove(chunk)
        elif any(s["id"] == what or s["name"] == what for s in section_mod.entries(chunk)):
            section_mod.remove(chunk, what)
        elif bh.get(chunk, what) is not None:
            bh.remove(chunk, [what])
        else:
            raise ValueError(f"nothing called {what!r}: all, boreholes, surfaces, sections, model, a section or a borehole")
        write_layers(chunk)
        print(f"{what} removed")
