"""Shipwreck - a plugin of pyarchinit-photogrammetry: the technical drawing of an excavated hull from its 3D model.

* the *Relitto* pane (on the right), step by step: the ship frame (x along the keel, y to port, z up), the stations
  (regular, from a table with the field's names, from the frames of the catalogue), the lines cut from the model, the
  water (a level, the refraction taken out where the user says so), the drawings (body plan, half-breadth plan, profile,
  plan over the orthophoto; SVG, PDF, PNG, DXF), the catalogue of the timbers with their measures, the clinker
  strakes (the lands found on the sections, the land lines, a strake model labelled a hypothesis, the file link with
  Fröhlich & Nielsen's Clinker Boat Tool for Houdini);
* the products in the workspace tree, the timbers and the land lines in layers of the raster view;
* *File > Esporta > Relitto: catalogo dei legni*;
* ``ppg plugin shipwreck frame|stations|sections|water|drawings|timbers|export|status`` and
  ``ppg plugin shipwreck clinker detect|strakes|export|import|status``.

It measures what is there and reconstructs nothing. It reads the model: the transform, the model and the products of
the chunk are never changed, and nothing is written outside ``<chunk data>/shipwreck/`` but what the user exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, num, plural, t

__version__ = "0.2.0"
NEGATIVE_VALUES = ("--start", "--end", "--longitudinals", "--horizontals")  # may begin with a minus: "-0.2;-0.5", "-1.5,2"
LAYER = "Legni del relitto"
CLINKER_LAYER = "Fasce"  # the land lines, in the chunk's x, y
SIDE_COLOURS = {"port": "#e04040", "starboard": "#30a040", "centre": "#ffd000"}


class ShipwreckPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Il disegno tecnico di uno scafo scavato dal suo modello 3D: sistema della nave, sezioni, "
                             "piani, catalogo dei legni, fasce del fasciame a clinker", "The technical drawing of an "
                             "excavated hull from its 3D model: ship frame, sections, plans, timber catalogue, clinker "
                             "strakes")
        self.context = None
        self.pane = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Relitto...", "Shipwreck..."), self.show_pane,
                           tip=t("Il pannello del disegno dello scafo", "The pane of the drawing of the hull"))
        context.add_command("shipwreck", self.command,
                            help="the technical drawing of a hull: frame, stations, sections, water, drawings, "
                                 "timbers, export, status, clinker")
        context.add_exporter(t("Relitto: catalogo dei legni (CSV)", "Shipwreck: timber catalogue (CSV)"), ".csv", self._export)
        context.add_exporter(t("Relitto: catalogo dei legni (Excel)", "Shipwreck: timber catalogue (Excel)"), ".xlsx", self._export)
        context.add_exporter(t("Relitto: catalogo dei legni (GeoPackage)", "Shipwreck: timber catalogue (GeoPackage)"), ".gpkg",
                             self._export)
        if context.window is not None:
            from .pane import ShipwreckPane

            self.pane = ShipwreckPane()
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            self.pane.requested.connect(self._requested)
            context.on_refresh(self.refresh_pane)
            context.on_raster_loaded(self.draw_layer)
            context.on_features_deleted(LAYER, self._features_deleted)
            context.add_tree_items(self._tree_items)
            self.refresh_pane()

    def deactivate(self) -> None:
        self.pane = None

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        window = self.context.window if self.context else None
        chunk = getattr(window, "chunk", None) if window is not None else None
        if chunk is not None:
            return chunk
        doc = self.context.app.document if self.context else None
        return None if doc is None else doc.chunk

    def _message(self, kind: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text)

    def _status(self, text: str) -> None:
        window = self.context.window
        if window is None:
            print(text)
        elif hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 15000)

    def _changed(self) -> None:
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane()
        self.draw_layer()

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        from .ship import NotReady

        try:
            function(*args)
        except NotReady as exc:
            self._message("warning", str(exc))

    def _run(self, title: str, job, done) -> None:
        window = self.context.window
        if window is not None and hasattr(window, "run_task"):
            window.run_task(title, job, done, refresh=False, save=True)
        else:
            self.context.run(title, job, done)

    def _ask(self, title: str, fields: list, intro: str = "") -> dict | None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        return ParamDialog.ask(self.context.window, title, fields, intro=intro)

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self) -> None:
        if self.pane is None:
            return
        chunk = self._chunk()
        self.pane.set_states(self.states(chunk) if chunk is not None else {k: "" for k in self.pane.states})
        self.pane.set_rows(self._rows(chunk) if chunk is not None else [])

    def _rows(self, chunk) -> list:
        from . import ship, timbers

        try:
            return timbers.measure(chunk)
        except ship.NotReady:
            return []

    @staticmethod
    def states(chunk) -> dict:
        """The state of each step, in words."""
        from . import drawings, lines, ship, timbers

        frame = ship.frame_of(chunk)
        stations = ship.stations_of(chunk)
        data = lines.load(chunk) if frame is not None else None
        out = {"frame": t("Non impostato.", "Not set."), "stations": t("Nessuna stazione.", "No stations."),
               "sections": t("Non calcolate.", "Not computed."), "water": t("Correzione spenta.", "Correction off."),
               "drawings": t("Nessun disegno.", "No drawings."), "timbers": t("Catalogo vuoto.", "The catalogue is empty."),
               "clinker": clinker_state(chunk)}
        if frame is not None:
            bow = t(f"prua a x = {num(frame.bow_x)}", f"bow at x = {num(frame.bow_x)}")
            out["frame"] = t(f"Chiglia {num(frame.length)} m, rotta {num(frame.heading, 1)}°, pendenza "
                             f"{num(frame.slope * 100, 1)} %, {bow}.",
                             f"Keel {num(frame.length)} m, heading {num(frame.heading, 1)}°, slope "
                             f"{num(frame.slope * 100, 1)} %, {bow}.")
            if chunk.meta.get(ship.KEY, {}).get("frame", {}).get("crs") != chunk.crs:
                out["frame"] += " " + t("Il sistema del chunk è cambiato: rifallo.", "The system of the chunk changed: set it again.")
        if stations:
            count = plural(len(stations), ("stazione", "stazioni"), ("station", "stations"))
            out["stations"] = t(f"{count}, da {stations[0]['name']} a {stations[-1]['name']}.",
                                f"{count}, from {stations[0]['name']} to {stations[-1]['name']}.")
        if data is not None:
            out["sections"] = lines.state_text(data)
            if lines.is_stale(chunk, data):
                out["sections"] += " " + t("Da ricalcolare: sistema della nave, modello o georeferenziazione cambiati.",
                                           "To compute again: ship frame, model or georeference changed.")
        water = lines.water_of(chunk)
        if water:
            on = t("correzione accesa", "correction on") if water["on"] else t("correzione spenta", "correction off")
            out["water"] = t(f"Livello z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.",
                             f"Level z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.")
        made = chunk.meta.get(ship.KEY, {}).get("drawings")
        if made:
            count = plural(len(made), ("disegno", "disegni"), ("drawing", "drawings"))
            out["drawings"] = t(f"{count} in shipwreck/drawings.", f"{count} in shipwreck/drawings.")
            old = drawings.out_of_date(chunk)
            if old:
                out["drawings"] += " " + t(f"Da rifare, non corrispondono più alle sezioni, all'acqua o ai legni: {', '.join(old)}.",
                                           f"To make again, they no longer match the sections, the water or the timbers: {', '.join(old)}.")
        records = timbers.timbers_of(chunk)
        if records:
            out["timbers"] = plural(len(records), ("legno", "legni"), ("timber", "timbers")) + "."
            try:
                wrong = [r for r in timbers.measure(chunk) if r["warning"]]
            except ship.NotReady:
                wrong = []
            if wrong:
                out["timbers"] += " " + t(f"Lato in disaccordo con y: {', '.join(r['name'] or r['id'] for r in wrong)}.",
                                          f"Side against y: {', '.join(r['name'] or r['id'] for r in wrong)}.")
        return out

    def _requested(self, name: str) -> None:
        self._guarded(getattr(self, "_do_" + name))

    # ------------------------------------------------------------------ 1 the frame
    def _do_frame_pick(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.check_metric(chunk)

        def finished(xy):
            self._guarded(self._set_picked, chunk, xy)

        self._need_view(chunk)
        self.context.request_line(finished, hint=t("Traccia la chiglia dal primo estremo (x = 0) all'altro; doppio clic o "
                                                   "Invio per finire", "Trace the keel from its first end (x = 0) to the "
                                                   "other; double click or Enter ends"))

    def _set_picked(self, chunk, xy) -> None:
        from . import ship

        old = ship.frame_of(chunk)
        new = ship.frame_from_picks(chunk, xy[0], xy[-1], old.bow if old else "start", old.z0 if old else None)
        ship.set_frame(chunk, ship.keeping_bow(old, new))
        self._changed()

    def _do_frame_markers(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        names = [(m.label, m.label) for m in chunk.markers if m.position is not None]
        if len(names) < 2:
            raise ship.NotReady(t("Servono due marker con una posizione (anche messi sul modello nella vista 3D).",
                                  "Two markers with a position are needed (also placed on the model in the 3D view)."))
        v = self._ask(t("Chiglia da due marker", "Keel from two markers"), [
            Field("a", t("Primo estremo (x = 0)", "First end (x = 0)"), names[0][0], choices=names),
            Field("b", t("Altro estremo", "Other end"), names[1][0], choices=names)])
        if v:
            old = ship.frame_of(chunk)
            new = ship.frame_from_markers(chunk, v["a"], v["b"], old.bow if old else "start", old.z0 if old else None)
            ship.set_frame(chunk, ship.keeping_bow(old, new))
            self._changed()

    def _do_frame_options(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        v = self._ask(t("Sistema della nave", "Ship frame"), [
            Field("x0", t("x del primo estremo (m)", "x of the first end (m)"), frame.x0, minimum=-1e4, maximum=1e4, decimals=3,
                  tip=t("0 alla prua o a una stazione scelta", "0 at the stem or at a chosen station")),
            Field("z0", t("Quota del datum z = 0 (quota del chunk, m)", "Height of the datum z = 0 (height of the chunk, m)"),
                  frame.z0, minimum=-1e4, maximum=1e5, decimals=3)])
        if v:
            ship.set_frame(chunk, ship.Frame(frame.start, frame.end, frame.bow, v["z0"], v["x0"]))
            self._changed()

    def _do_frame_flip(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.set_frame(chunk, ship.need_frame(chunk).flipped())
        self._changed()

    def _do_frame_bow(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.set_frame(chunk, ship.need_frame(chunk).other_bow())
        self._status(t("Babordo e tribordo scambiati: rifai sezioni e disegni.",
                       "Port and starboard exchanged: make the sections and drawings again."))
        self._changed()

    # ------------------------------------------------------------------ 2 the stations
    def _do_stations_regular(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        v = self._ask(t("Stazioni regolari", "Regular stations"), [
            Field("spacing", t("Distanza (m)", "Spacing (m)"), 1.0, minimum=0.01, maximum=1000.0, decimals=3),
            Field("count", t("Quante", "How many"), int(frame.length) + 1, minimum=1, maximum=1000, decimals=0),
            Field("start", t("Prima stazione, x (m)", "First station, x (m)"), frame.x0, minimum=-1e4, maximum=1e4, decimals=3)])
        if v:
            ship.set_stations(chunk, ship.regular(v["spacing"], int(v["count"]), v["start"]))
            self._changed()

    def _do_stations_table(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import ship

        chunk = self._need_chunk()
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Stazioni", "Stations"), "", "CSV (*.csv *.txt *.tsv)")
        if path:
            ship.set_stations(chunk, ship.stations_from_table(path))
            self._changed()

    def _do_stations_catalogue(self) -> None:
        from . import ship, timbers

        chunk = self._need_chunk()
        ship.set_stations(chunk, timbers.stations_from_catalogue(chunk))
        self._changed()

    # ------------------------------------------------------------------ 3 the sections
    def _do_sections(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        v = self._ask(t("Sezioni", "Sections"), [
            Field("longitudinals", t("Longitudinali, y (m)", "Longitudinals, y (m)"), "0", kind="text"),
            Field("horizontals", t("Orizzontali, z (m)", "Horizontals, z (m)"), "", kind="text"),
            Field("half_width", t("Metà larghezza (m, 0 = fin dove arriva il modello)", "Half width (m, 0 = as far as the model goes)"),
                  0.0, minimum=0.0, maximum=1e4, decimals=2)],
            intro=t("Valori separati da spazi o punto e virgola.", "Values separated by spaces or semicolons."))
        if v:
            self.sections(chunk, parse_values(v["longitudinals"]), parse_values(v["horizontals"]), v["half_width"])

    def sections(self, chunk, longitudinals, horizontals, half_width=None) -> None:
        from . import lines

        def done(data):
            self._status(t(f"{len(data['sections'])} sezioni calcolate", f"{len(data['sections'])} sections computed"))
            self._changed()

        self._run(t("Sezioni dello scafo", "Sections of the hull"),
                  lambda p: lines.build(chunk, longitudinals, horizontals, half_width, progress=p), done)

    # ------------------------------------------------------------------ 4 the water
    def _do_water(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import lines, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        water = lines.water_of(chunk) or {"level": 0.0, "n": lines.REFRACTION, "on": False}
        v = self._ask(t("Acqua", "Water"), [
            Field("level", t("Livello, z della nave (m)", "Level, ship z (m)"), water["level"], minimum=-1e4, maximum=1e4, decimals=3),
            Field("n", t("Indice di rifrazione", "Refractive index"), water["n"], minimum=1.0, maximum=2.0, decimals=3),
            Field("on", t("Correggi le quote sotto il livello", "Correct the heights under the level"), water["on"])],
            intro=t("Sotto il livello il fondo visto attraverso l'acqua è apparente: in vista nadirale la profondità vera è "
                    "circa quella apparente per n. È una prima approssimazione, dichiarata sui disegni.",
                    "Under the level the bottom seen through water is apparent: in a nadir view the true depth is about the "
                    "apparent one times n. A first approximation, stated on the drawings."))
        if v:
            lines.set_water(chunk, v["level"], v["n"], v["on"])
            self._changed()

    def _do_water_pick(self) -> None:
        from . import lines, ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        self._need_view(chunk)

        def finished(xy):
            ship_xy = frame.to_ship(np.r_[xy[0], 0.0])
            level = ship.Surface(lines.ship_mesh(chunk, frame).vertices).at(ship_xy[:, :2])[0]
            if not np.isfinite(level):
                self._message("warning", t("Il punto è fuori dal modello.", "The point lies off the model."))
                return
            water = lines.water_of(chunk) or {"n": lines.REFRACTION, "on": False}
            lines.set_water(chunk, float(level), water["n"], water["on"])
            self._changed()

        self.context.request_point(finished, hint=t("Clic su un punto del pelo dell'acqua", "Click a point of the waterline"))

    # ------------------------------------------------------------------ 5 the drawings
    def _do_drawings(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import drawings

        chunk = self._need_chunk()
        f = drawings.title_fields(chunk)
        v = self._ask(t("Disegni", "Drawings"), [
            Field("scale", t("Scala 1: (0 = automatica)", "Scale 1: (0 = automatic)"), 0, minimum=0, maximum=100000, decimals=0),
            Field("site", t("Sito", "Site"), f["site"], kind="text"),
            Field("vessel", t("Relitto", "Wreck"), f["vessel"], kind="text"),
            Field("surveyed", t("Rilievo", "Survey"), f["surveyed"], kind="text"),
            Field("drawn", t("Disegno", "Drawn by"), f["drawn"], kind="text"),
            Field("date", t("Data", "Date"), f["date"], kind="text"),
            Field("notes", t("Note", "Notes"), f["notes"], kind="text")])
        if not v:
            return
        scale = int(v.pop("scale")) or None
        self.drawings(chunk, scale, v)

    def drawings(self, chunk, scale=None, title=None) -> None:
        from . import drawings, ship

        if title:
            ship.state(chunk)["title"] = title

        def done(paths):
            self._status(t(f"{len(paths)} file dei disegni in {paths[0].parent}", f"{len(paths)} drawing files in {paths[0].parent}"))
            self._changed()

        self._run(t("Disegni dello scafo", "Drawings of the hull"), lambda p: drawings.write_all(chunk, scale=scale, progress=p), done)

    def _do_drawings_open(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from . import ship

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ship.folder(self._need_chunk()) / "drawings")))

    # ------------------------------------------------------------------ 6 the timbers
    def _timber(self, geometry: str, xy) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import timbers

        chunk = self._chunk()
        v = self._ask(t("Legno", "Timber"), [
            Field("kind", t("Tipo", "Kind"), "frame", choices=[(k, timbers.kind_title(k)) for k in timbers.KINDS]),
            Field("side", t("Lato", "Side"), "port", choices=[(k, timbers.side_title(k)) for k in timbers.SIDES]),
            Field("name", t("Nome", "Name"), "", kind="text", tip=t("Il nome del cantiere (F36)", "The name of the field (F36)")),
            Field("station", t("Stazione", "Station"), "", kind="text"), Field("notes", t("Note", "Notes"), "", kind="text")])
        if v:
            timbers.add(chunk, v["kind"], v["side"], v["name"], timbers.geometry(geometry, xy), v["station"], v["notes"])
            self._changed()

    def _do_timber_line(self) -> None:
        self._draw_timber("line", self.context.request_line)

    def _do_timber_polygon(self) -> None:
        self._draw_timber("polygon", self.context.request_polygon)

    def _do_timber_point(self) -> None:
        self._draw_timber("point", self.context.request_point)

    def _draw_timber(self, geometry: str, request) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._need_view(chunk)
        request(lambda xy: self._guarded(self._timber, geometry, xy), hint=t("Disegna il legno", "Draw the timber"))

    def _do_timber_import(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import timbers

        chunk = self._need_chunk()
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Legni", "Timbers"), "", "CSV (*.csv *.txt *.tsv)")
        if path:
            count = timbers.import_csv(chunk, path)
            self._status(t(f"Legni importati: {count}", f"Timbers imported: {count}"))
            self._changed()

    def _do_timber_remove(self) -> None:
        from . import timbers

        ids = self.pane.selected_ids()
        if ids:
            timbers.remove(self._need_chunk(), ids)
            self._changed()

    def _do_timber_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import timbers

        chunk = self._need_chunk()
        path, _ = QFileDialog.getSaveFileName(self.context.window, t("Esporta i legni", "Export the timbers"), "legni.xlsx",
                                              "Excel (*.xlsx);;CSV (*.csv);;GeoPackage (*.gpkg)")
        if path:
            self._status(str(timbers.export_any(chunk, path)))

    def _export(self, chunk, path, progress=None):
        from . import timbers

        return timbers.export_any(chunk, path)

    def _features_deleted(self, features) -> None:
        from . import timbers

        chunk = self._chunk()
        if chunk is not None and timbers.remove(chunk, [f.get("id") for f in features]):
            self.context.mark_modified()
            self.context.refresh_tree()
            self.refresh_pane()

    # ------------------------------------------------------------------ 7 the clinker strakes
    def _do_clinker_detect(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import clinker, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        p = {**clinker.DEFAULTS, **(((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}).get("params") or {})}
        v = self._ask(t("Rileva i corsi", "Find the strakes"), [
            Field("min_step", t("Gradino minimo (mm)", "Least step (mm)"), p["min_step"] * 1000.0, minimum=1.0,
                  maximum=100.0, decimals=1, tip=t("Il salto più piccolo dove una fascia si sovrappone a quella sotto",
                                                   "The smallest step where a strake laps over the one below")),
            Field("min_strake", t("Larghezza minima di fascia (m)", "Least strake width (m)"), p["min_strake"],
                  minimum=0.01, maximum=2.0, decimals=3, tip=t("Due gradini più vicini di così sono uno solo",
                                                               "Two lands nearer than this are one")),
            Field("keel_half_width", t("Metà larghezza della chiglia (m)", "Half width of the keel (m)"),
                  p["keel_half_width"], minimum=0.01, maximum=1.0, decimals=3),
            Field("sole_width", t("Larghezza minima della suola (m)", "Least width of the sole (m)"), p["sole_width"],
                  minimum=0.001, maximum=0.5, decimals=3,
                  tip=t("Abbassala se una chiglia stretta e consumata dà lati non letti a ogni stazione",
                        "Lower it when a narrow worn keel leaves sides not read at every station"))],
            intro=t("I gradini del fasciame letti sulle sezioni alle stazioni (quelle grezze, senza la correzione "
                    "dell'acqua), dalla battura verso l'orlo, e uniti in linee lungo lo scafo. I gradini sono misurati; "
                    "le linee e il modello delle fasce sono un'ipotesi.",
                    "The plank lands read on the sections at the stations (the raw ones, without the water correction), "
                    "from the rabbet out to the sheer, and joined into lines along the hull. The lands are measured; the "
                    "lines and the strake model are a hypothesis."))
        if v:
            params = {k: float(v[k]) for k in ("min_strake", "keel_half_width", "sole_width")}
            params["min_step"] = float(v["min_step"]) / 1000.0
            self.detect(chunk, params)

    def detect(self, chunk, params=None) -> None:
        from . import clinker

        def job(p):
            result = clinker.detect_lands(chunk, params, progress=p)
            return result, lines_files(chunk)

        def done(found):
            result, files = found
            count = sum(1 for line in result["lines"] if len(line["points"]) > 1)
            self._status(t(f"Linee dei gradini: {count}", f"Land lines: {count}")
                         + (f" ({files['dxf'].parent})" if files else ""))
            self._changed()

        self._run(t("Fasce del fasciame", "Clinker strakes"), job, done)

    def _do_clinker_strakes(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import clinker

        chunk = self._need_chunk()
        state = clinker.check_strakes(chunk)
        before = (state.get("strakes") or [{}])[0]
        v = self._ask(t("Modello delle fasce", "Strake model"), [
            Field("lap", t("Sovrapposizione (mm)", "Lap (mm)"), before.get("lap_mm", 20.0), minimum=1.0, maximum=200.0,
                  decimals=1, tip=t("Quanto una fascia copre quella sotto", "How much a strake covers the one below")),
            Field("thickness", t("Spessore (mm, 0 = il gradino misurato)", "Thickness (mm, 0 = the step measured)"), 0.0,
                  minimum=0.0, maximum=200.0, decimals=1),
            Field("catalogue", t("Anche nel catalogo dei legni", "Into the timber catalogue too"), False)],
            intro=t("Fra due linee di gradini una fascia: larghezza a ogni stazione, lunghezza, spessore e "
                    "sovrapposizione, una mesh per fascia (OBJ e PLY) e le tabelle. È un'ipotesi dalle sezioni, non una "
                    "misura: è scritto in ogni file.",
                    "Between two land lines a strake: width at every station, length, thickness and lap, a mesh per "
                    "strake (OBJ and PLY) and the tables. It is a hypothesis from the sections, not a measure: every "
                    "file says so."))
        if v:
            self.strakes(chunk, float(v["lap"]) / 1000.0, float(v["thickness"]) / 1000.0 or None, bool(v["catalogue"]))

    def strakes(self, chunk, lap=0.02, thickness=None, catalogue=False) -> None:
        from . import clinker

        def job(p):
            result = clinker.build_strakes(chunk, lap, thickness, progress=p)
            result.update(clinker.write_tables(chunk))
            if catalogue:
                result["catalogue"] = clinker.to_catalogue(chunk)
            return result

        def done(result):
            count = plural(len(result["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
            self._status(t(f"Modello delle fasce (ipotesi): {count} in {result['obj'].parent}",
                           f"Strake model (hypothesis): {count} in {result['obj'].parent}"))
            self._changed()

        self._run(t("Modello delle fasce", "Strake model"), job, done)

    def _do_clinker_catalogue(self) -> None:
        from . import clinker

        count = clinker.to_catalogue(self._need_chunk())
        self._status(t(f"Fasce nel catalogo dei legni: {count}", f"Strakes in the timber catalogue: {count}"))
        self._changed()

    def _do_clinker_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import houdini, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        last = ((chunk.meta.get(ship.KEY) or {}).get("houdini") or {}).get("export")
        folder = QFileDialog.getExistingDirectory(self.context.window, t("Cartella per il Clinker Boat Tool",
                                                                         "Folder for the Clinker Boat Tool"),
                                                  str(Path(last).parent) if last else "")
        if not folder:
            return

        def done(out):
            self._status(t(f"File per il Clinker Boat Tool in {out['readme'].parent}",
                           f"Files for the Clinker Boat Tool in {out['readme'].parent}"))
            self._changed()

        self._run(t("File per il Clinker Boat Tool", "Files for the Clinker Boat Tool"),
                  lambda p: houdini.export(chunk, folder, progress=p), done)

    def _do_clinker_import(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import houdini, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        last = ((chunk.meta.get(ship.KEY) or {}).get("houdini") or {}).get("export") or ""
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Fasce da Houdini", "Planks from Houdini"), last,
                                              "OBJ (*.obj)")
        if not path:
            return
        v = self._ask(t("Fasce da Houdini", "Planks from Houdini"), [
            Field("side_from", t("Lato", "Side"), "y", choices=[
                ("y", t("Dalla y media dei vertici", "From the mean y of the vertices")),
                ("name", t("Dal nome (B/T, babordo/tribordo)", "From the name (B/T, port/starboard)"))]),
            Field("deviation", t("Distanza dalla scansione", "Distance from the scan"), True)],
            intro=t("Un oggetto (o un gruppo) del file per tavola, Y in alto come le esporta il Clinker Boat Tool. "
                    "Le tavole di un'importazione precedente dello stesso file vengono sostituite.",
                    "One object (or group) of the file per plank, Y up as the Clinker Boat Tool exports them. The "
                    "planks of an earlier import of the same file are replaced."))
        if not v:
            return

        def done(result):
            self._status(t(f"Tavole importate: {result['planks']} ({result['csv'].name})",
                           f"Planks imported: {result['planks']} ({result['csv'].name})"))
            self._changed()

        self._run(t("Fasce da Houdini", "Planks from Houdini"),
                  lambda p: houdini.import_planks(chunk, path, v["side_from"], bool(v["deviation"]), progress=p), done)

    def _do_clinker_open(self) -> None:
        from . import ship

        folder = ship.folder(self._need_chunk()) / "clinker"
        folder.mkdir(parents=True, exist_ok=True)
        self._open(folder)

    # ------------------------------------------------------------------ the view
    def _need_chunk(self):
        from . import ship

        chunk = self._chunk()
        if chunk is None:
            raise ship.NotReady(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _need_view(self, chunk) -> None:
        """Something to draw on: the orthophoto or the DEM, else an empty frame around the model."""
        view = getattr(self.context.window, "ortho_view", None)
        if view is not None and view.tr is not None:
            return
        if self.context.show_raster(None):
            return
        from . import ship

        xyz = ship.world_vertices(chunk)
        lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
        self.context.show_frame((lo[0], lo[1], hi[0], hi[1]), max(hi - lo) / 1500.0, t(*NAME))

    def draw_layer(self) -> None:
        """The timbers of the chunk as a layer of the raster view."""
        from . import timbers

        view = getattr(self.context.window, "ortho_view", None) if self.context else None
        chunk = self._chunk()
        if view is None or chunk is None or view.tr is None:
            return
        view.remove_layer(LAYER)
        for r in timbers.timbers_of(chunk):
            xy = np.asarray(r["geometry"]["xy"], float)
            colour = SIDE_COLOURS.get(r["side"], "#ffd000")
            attrs = {"id": r["id"], "kind": r["kind"], "side": r["side"], "name": r["name"]}
            label = r["name"] or r["id"]
            if r["geometry"]["type"] == "point":
                view.add_point(xy[0, 0], xy[0, 1], colour, label, layer=LAYER, attrs=attrs)
            elif r["geometry"]["type"] == "polygon":
                view.add_polygon(xy, label, colour, layer=LAYER, attrs=attrs)
            else:
                view.add_line(xy, colour, label, layer=LAYER, attrs=attrs)
        self._draw_lands(view, chunk)

    @staticmethod
    def _draw_lands(view, chunk) -> None:
        """The land lines in the chunk's x, y, a colour per side, dashed when the sections changed since."""
        from . import clinker, ship

        view.remove_layer(CLINKER_LAYER)
        frame = ship.frame_of(chunk)
        found = [ln for ln in ((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}).get("lines") or [] if len(ln["points"]) > 1]
        if frame is None or not found:
            return
        stale = clinker.is_stale(chunk)
        for line in found:
            xy = frame.to_world(np.asarray(line["points"], float))[:, :2]
            name = f"{clinker.SIDE_CODES[line['side']][0]}{line['k']}"
            view.add_line(xy, SIDE_COLOURS[line["side"]], name, layer=CLINKER_LAYER, width=1.5, dash=stale,
                          attrs={"side": line["side"], "k": line["k"], "uncertain": len(line["uncertain_stations"])})

    # ------------------------------------------------------------------ the workspace tree
    def _tree_items(self, chunk) -> list:
        from . import lines, ship, timbers

        children = []
        frame = ship.frame_of(chunk)
        if frame is not None:
            children.append({"label": t(f"Sistema della nave ({num(frame.length)} m)", f"Ship frame ({num(frame.length)} m)"),
                             "icon": "plugin", "activate": self.show_pane})
        data = lines.load(chunk) if frame is not None else None
        if data is not None:
            children.append({"label": t(f"Sezioni ({len(data['sections'])})", f"Sections ({len(data['sections'])})"),
                             "icon": "plugin", "activate": self.show_pane})
        made = chunk.meta.get(ship.KEY, {}).get("drawings") or {}
        if made:
            from .drawings import KIND_TITLES, out_of_date

            folder = ship.folder(chunk) / "drawings"
            old = out_of_date(chunk)
            kids = [{"label": t(*KIND_TITLES[k]) + (t(" (da rifare)", " (to make again)") if k in old else ""), "icon": "plugin",
                     "activate": lambda f=folder / f"{k}.pdf": self._open(f)} for k in made]
            children.append({"label": t("Disegni", "Drawings"), "icon": "plugin", "children": kids})
        records = timbers.timbers_of(chunk)
        if records:
            children.append({"label": t(f"Legni ({len(records)})", f"Timbers ({len(records)})"), "icon": "plugin",
                             "activate": self.show_pane})
        strakes = strake_counts(chunk)
        if strakes:
            if len(set(strakes.values())) == 1 and len(strakes) == 2:
                count = t(f"{strakes['port']} per lato", f"{strakes['port']} per side")
            else:
                from .clinker import _side_name

                count = ", ".join(f"{n} {_side_name(side)}" for side, n in strakes.items())
            children.append({"label": t(f"Fasce ({count})", f"Strakes ({count})"), "icon": "plugin",
                             "activate": self.show_pane})
        if not children:
            return []
        return [{"label": t(*NAME), "icon": "plugin", "children": children}]

    def _open(self, path) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ the command line
    def command(self, *args) -> int:
        from . import drawings, lines, ship, timbers

        if args and str(args[0]) == "clinker":
            return self._clinker_command([str(a) for a in args[1:]])
        parser = argparse.ArgumentParser(prog="ppg plugin shipwreck", description="the technical drawing of a hull from its 3D model")
        parser.add_argument("what", choices=["frame", "stations", "sections", "water", "drawings", "timbers", "export", "status",
                                             "clinker"], help="clinker: see 'ppg plugin shipwreck clinker --help'")
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--start", help="frame: first end of the keel, X,Y[,Z] world, or a marker with --markers")
        parser.add_argument("--end", help="frame: other end of the keel")
        parser.add_argument("--markers", action="store_true", help="frame: --start and --end are markers")
        parser.add_argument("--bow", choices=["start", "end"], default="start", help="frame: the end where the bow is")
        parser.add_argument("--flip", action="store_true", help="frame: x from the other end; the bow, y and the sides stay")
        parser.add_argument("--other-bow", action="store_true", help="frame: the bow is the other end; port and starboard are exchanged")
        parser.add_argument("--z0", type=float, help="frame: the world height of ship z = 0")
        parser.add_argument("--x0", type=float, default=0.0, help="frame: ship x of the first end")
        parser.add_argument("--spacing", type=float, help="stations: regular, metres")
        parser.add_argument("--count", type=int, help="stations: regular, how many")
        parser.add_argument("--first", type=float, default=0.0, help="stations: x of the first")
        parser.add_argument("--table", help="stations: a table name, x; timbers: a table to import")
        parser.add_argument("--from-catalogue", action="store_true", help="stations: the frames of the catalogue")
        parser.add_argument("--longitudinals", default="0", help="sections: y values, as 0;0.5")
        parser.add_argument("--horizontals", default="", help="sections: z values, as -0.5;-1")
        parser.add_argument("--half-width", type=float, help="sections: half width drawn, metres")
        parser.add_argument("--level", type=float, help="water: level in ship z")
        parser.add_argument("--n", type=float, default=lines.REFRACTION, help="water: refractive index")
        parser.add_argument("--on", action="store_true", help="water: correct the heights under the level")
        parser.add_argument("--scale", type=int, help="drawings: 1:SCALE (default: automatic)")
        parser.add_argument("--out", help="drawings: a folder; export: the file (.csv, .xlsx, .gpkg)")
        ns = parser.parse_args(joined_values([str(a) for a in args]))
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        try:
            if ns.what == "frame":
                if ns.flip:
                    ship.set_frame(chunk, ship.need_frame(chunk).flipped())
                elif ns.other_bow:
                    ship.set_frame(chunk, ship.need_frame(chunk).other_bow())
                elif ns.markers:
                    ship.set_frame(chunk, ship.frame_from_markers(chunk, ns.start, ns.end, ns.bow, ns.z0, ns.x0))
                elif ns.start and ns.end:
                    a, b = (np.array([float(v) for v in s.split(",")]) for s in (ns.start, ns.end))
                    ship.set_frame(chunk, ship.frame_from_picks(chunk, a, b, ns.bow, ns.z0, ns.x0))
                else:
                    raise ship.NotReady("--start and --end, --flip or --other-bow")
            elif ns.what == "stations":
                if ns.table:
                    stations = ship.stations_from_table(ns.table)
                elif ns.from_catalogue:
                    stations = timbers.stations_from_catalogue(chunk)
                elif ns.spacing and ns.count:
                    stations = ship.regular(ns.spacing, ns.count, ns.first)
                else:
                    raise ship.NotReady("--spacing and --count, --table or --from-catalogue")
                ship.set_stations(chunk, stations)
            elif ns.what == "sections":
                data = lines.build(chunk, parse_values(ns.longitudinals), parse_values(ns.horizontals), ns.half_width,
                                   progress=_print_progress)
                print(f"{len(data['sections'])} sections")
            elif ns.what == "water":
                if ns.level is None:
                    raise ship.NotReady("--level")
                lines.set_water(chunk, ns.level, ns.n, ns.on)
            elif ns.what == "drawings":
                for path in drawings.write_all(chunk, scale=ns.scale, folder=ns.out, progress=_print_progress):
                    print(path)
            elif ns.what == "timbers":
                if ns.table:
                    print(f"{timbers.import_csv(chunk, ns.table)} timbers imported")
                print(json.dumps(timbers.measure(chunk), indent=1, ensure_ascii=False))
            elif ns.what == "export":
                if not ns.out:
                    raise ship.NotReady("--out FILE (.csv, .xlsx, .gpkg)")
                print(timbers.export_any(chunk, ns.out))
            else:
                print(json.dumps(self.states(chunk), indent=1, ensure_ascii=False))
                return 0
        except ship.NotReady as exc:
            print(exc)
            return 1
        doc.save()
        return 0


    def _clinker_command(self, args: list[str]) -> int:
        from . import clinker, houdini, ship

        parser = argparse.ArgumentParser(prog="ppg plugin shipwreck clinker",
                                         description="clinker strakes: the lands on the sections, the land lines, the "
                                                     "strake model (a hypothesis), the Clinker Boat Tool's files")
        parser.add_argument("action", choices=["detect", "strakes", "export", "import", "status"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--min-step", type=float, help="detect: least step of a land, mm (8)")
        parser.add_argument("--min-strake", type=float, help="detect: least width of a strake, m (0.08)")
        parser.add_argument("--keel-half-width", type=float, help="detect: half width of the keel, m (0.10)")
        parser.add_argument("--sole-width", type=float, help="detect: least width of a sole, m (0.02)")
        parser.add_argument("--lap", type=float, default=20.0, help="strakes: the lap, mm (20)")
        parser.add_argument("--thickness", type=float, help="strakes: mm (default: the step measured)")
        parser.add_argument("--catalogue", action="store_true", help="strakes: a plank of the catalogue per strake")
        parser.add_argument("--out", help="export: the folder (clinker_boat_tool/ is made in it)")
        parser.add_argument("--obj", help="import: the OBJ of the planks exported by the Clinker Boat Tool")
        parser.add_argument("--side-from", choices=["y", "name"], default="y", help="import: the side of a plank")
        parser.add_argument("--no-deviation", action="store_true", help="import: skip the distance from the scan")
        ns = parser.parse_args(args)
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        try:
            if ns.action == "detect":
                params = {key: getattr(ns, key) for key in ("min_strake", "keel_half_width", "sole_width")
                          if getattr(ns, key) is not None}
                if ns.min_step is not None:
                    params["min_step"] = ns.min_step / 1000.0
                result = clinker.detect_lands(chunk, params, progress=_print_progress)
                print(clinker.summary(chunk))
                for line in detection_lines(chunk, result):
                    print(line)
                for path in lines_files(chunk).values():
                    print(path)
            elif ns.action == "strakes":
                result = clinker.build_strakes(chunk, ns.lap / 1000.0,
                                               ns.thickness / 1000.0 if ns.thickness else None, progress=_print_progress)
                result.update(clinker.write_tables(chunk))
                count = plural(len(result["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
                print(t(f"Modello delle fasce (ipotesi): {count}", f"Strake model (hypothesis): {count}"))
                for key in ("obj", "ply", "csv", "xlsx"):
                    print(result[key])
                if ns.catalogue:
                    count = clinker.to_catalogue(chunk)
                    print(plural(count, ("fascia", "fasce"), ("strake", "strakes")) + t(" nel catalogo dei legni",
                                                                                        " in the timber catalogue"))
            elif ns.action == "export":
                if not ns.out:
                    raise ship.NotReady("--out FOLDER")
                for path in houdini.export(chunk, ns.out, progress=_print_progress).values():
                    print(path)
            elif ns.action == "import":
                if not ns.obj:
                    raise ship.NotReady("--obj FILE")
                result = houdini.import_planks(chunk, ns.obj, ns.side_from, not ns.no_deviation, progress=_print_progress)
                n = result["planks"]
                print(t("1 fascia importata" if n == 1 else f"{n} fasce importate",
                        "1 plank imported" if n == 1 else f"{n} planks imported"))
                for r in result["rows"]:
                    median = "-" if r["median_mm"] is None else num(r["median_mm"], 1)
                    print(t(f"  {r['name']}: distanza dalla scansione, mediana {median} mm",
                            f"  {r['name']}: distance from the scan, median {median} mm"))
                print(result["csv"])
                print(result["xlsx"])
            else:
                print(clinker_state(chunk))
                if "lands" in ((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}):
                    print(t("Gradini da rilevare di nuovo: le sezioni sono cambiate." if clinker.is_stale(chunk)
                            else "Gradini aggiornati rispetto alle sezioni.",
                            "Lands to find again: the sections changed." if clinker.is_stale(chunk)
                            else "Lands up to date with the sections."))
                return 0
        except ship.NotReady as exc:
            print(exc)
            return 1
        doc.save()
        return 0


def clinker_state(chunk) -> str:
    """The clinker step's state in words: the summary of the lands, the strake model, the catalogue, the imports."""
    from statistics import median

    from . import clinker, houdini, ship

    text = clinker.summary(chunk)
    state = (chunk.meta.get(ship.KEY) or {}).get("clinker") or {}
    if state.get("strakes"):
        count = plural(len(state["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
        text += " " + t(f"Modello delle fasce (ipotesi dalle sezioni): {count} in shipwreck/clinker.",
                        f"Strake model (hypothesis from the sections): {count} in shipwreck/clinker.")
        if state.get("catalogue"):
            text += " " + t(f"Nel catalogo: {len(state['catalogue'])}.", f"In the catalogue: {len(state['catalogue'])}.")
    for name, rows in houdini.imports(chunk).items():
        medians = [r["median_mm"] for r in rows if r.get("median_mm") is not None]
        count = plural(len(rows), ("tavola", "tavole"), ("plank", "planks"))
        far = (t(f", distanza dalla scansione mediana {num(median(medians), 1)} mm",
                 f", distance from the scan median {num(median(medians), 1)} mm") if medians else "")
        text += " " + t(f"Da Houdini, {name}: {count}{far}.", f"From Houdini, {name}: {count}{far}.")
    return text


def strake_counts(chunk) -> dict:
    """The strakes found per side (the land lines plus one), for the sides with lands."""
    from . import clinker, ship

    state = (chunk.meta.get(ship.KEY) or {}).get("clinker") or {}
    return {side: int(state["mode"][side]) + 1 for side in clinker.SIDES if int((state.get("mode") or {}).get(side, 0))}


def detection_lines(chunk, result: dict) -> list[str]:
    """The stations a detection was unsure of, one list per reason, for the command line."""
    from .clinker import _reason_text, _side_name

    def listed(names):
        return ", ".join(names) or t("nessuna", "none")

    other = [f"{n} {_side_name(side)}" for side, names in result["uncertain"].items() for n in names]
    jumped = [f"{n} {_side_name(side)}" for side, names in result["jumps"].items() for n in names]
    skipped = [f"{d['station']} {_side_name(d['side'])} ({_reason_text(d['reason'])})" for d in result["skipped"]]
    return [t(f"Stazioni incerte (un altro numero di gradini): {listed(other)}",
              f"Uncertain stations (another count of lands): {listed(other)}"),
            t(f"Stazioni incerte (battura o torello che saltano): {listed(jumped)}",
              f"Uncertain stations (the rabbet or the garboard jumping): {listed(jumped)}"),
            t(f"Lati non letti: {'; '.join(skipped) or 'nessuno'}", f"Sides not read: {'; '.join(skipped) or 'none'}")]


def lines_files(chunk) -> dict:
    """The land lines written as DXF and GeoPackage (:func:`clinker.write_lines`); with no land line those of an
    earlier detection are removed, as they no longer hold."""
    from . import clinker, ship

    try:
        return clinker.write_lines(chunk)
    except ship.NotReady:
        for name in ("lands.dxf", "lands.gpkg"):
            (ship.folder(chunk) / "clinker" / name).unlink(missing_ok=True)
        return {}


def joined_values(args: list[str]) -> list[str]:
    """``--horizontals -0.2;-0.5`` as ``--horizontals=-0.2;-0.5``: argparse takes for an option anything that begins with
    a minus and is not one negative number."""
    out: list[str] = []
    for a in args:
        if out and out[-1] in NEGATIVE_VALUES and a[:1] == "-" and a[1:2] in tuple("0123456789."):
            out[-1] = f"{out[-1]}={a}"
        else:
            out.append(a)
    return out


def parse_values(text) -> list[float]:
    """Numbers separated by spaces or ``;``, decimal point or comma."""
    return [float(v.replace(",", ".")) for v in str(text).replace(";", " ").split()]


def _print_progress(fraction: float, message: str) -> None:
    print(f"{fraction * 100:5.1f} % {message}", flush=True)
