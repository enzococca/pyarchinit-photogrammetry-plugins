"""Shipwreck - a plugin of pyarchinit-photogrammetry: the technical drawing of an excavated hull from its 3D model.

* the *Relitto* pane (on the right), step by step: the ship frame (x along the keel, y to port, z up), the stations
  (regular, from a table with the field's names, from the frames of the catalogue), the lines cut from the model, the
  water (a level, the refraction taken out where the user says so), the drawings (body plan, half-breadth plan, profile,
  plan over the orthophoto; SVG, PDF, PNG, DXF), the catalogue of the timbers with their measures;
* the products in the workspace tree, the timbers in a layer of the raster view;
* *File > Esporta > Relitto: catalogo dei legni*;
* ``ppg plugin shipwreck frame|stations|sections|water|drawings|timbers|export|status``.

It measures what is there and reconstructs nothing. It reads the model: the transform, the model and the products of
the chunk are never changed, and nothing is written outside ``<chunk data>/shipwreck/`` but what the user exports.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, num, t

__version__ = "0.1.0"
LAYER = "Legni del relitto"
SIDE_COLOURS = {"port": "#e04040", "starboard": "#30a040", "centre": "#ffd000"}


class ShipwreckPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Il disegno tecnico di uno scafo scavato dal suo modello 3D: sistema della nave, sezioni, "
                             "piani, catalogo dei legni", "The technical drawing of an excavated hull from its 3D model: "
                             "ship frame, sections, plans, timber catalogue")
        self.context = None
        self.pane = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Relitto...", "Shipwreck..."), self.show_pane,
                           tip=t("Il pannello del disegno dello scafo", "The pane of the drawing of the hull"))
        context.add_command("shipwreck", self.command,
                            help="the technical drawing of a hull: frame, stations, sections, water, drawings, "
                                 "timbers, export, status")
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
        from . import lines, ship, timbers

        frame = ship.frame_of(chunk)
        stations = ship.stations_of(chunk)
        data = lines.load(chunk) if frame is not None else None
        out = {"frame": t("Non impostato.", "Not set."), "stations": t("Nessuna stazione.", "No stations."),
               "sections": t("Non calcolate.", "Not computed."), "water": t("Correzione spenta.", "Correction off."),
               "drawings": t("Nessun disegno.", "No drawings."), "timbers": t("Catalogo vuoto.", "The catalogue is empty.")}
        if frame is not None:
            bow = t("prua all'inizio", "bow at the start") if frame.bow == "start" else t("prua alla fine", "bow at the end")
            out["frame"] = t(f"Chiglia {num(frame.length)} m, rotta {num(frame.heading, 1)}°, pendenza "
                             f"{num(frame.slope * 100, 1)} %, {bow}.",
                             f"Keel {num(frame.length)} m, heading {num(frame.heading, 1)}°, slope "
                             f"{num(frame.slope * 100, 1)} %, {bow}.")
            if chunk.meta.get(ship.KEY, {}).get("frame", {}).get("crs") != chunk.crs:
                out["frame"] += " " + t("Il sistema del chunk è cambiato: rifallo.", "The system of the chunk changed: set it again.")
        if stations:
            out["stations"] = t(f"{len(stations)} stazioni, da {stations[0]['name']} a {stations[-1]['name']}.",
                                f"{len(stations)} stations, from {stations[0]['name']} to {stations[-1]['name']}.")
        if data is not None:
            count = len(data["sections"])
            out["sections"] = t(f"{count} sezioni dal modello.", f"{count} sections from the model.")
            if lines.is_stale(chunk, data):
                out["sections"] += " " + t("Da ricalcolare: sistema o modello cambiati.",
                                           "To compute again: frame or model changed.")
        water = lines.water_of(chunk)
        if water:
            on = t("correzione accesa", "correction on") if water["on"] else t("correzione spenta", "correction off")
            out["water"] = t(f"Livello z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.",
                             f"Level z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.")
        made = chunk.meta.get(ship.KEY, {}).get("drawings")
        if made:
            out["drawings"] = t(f"{len(made)} disegni in shipwreck/drawings.", f"{len(made)} drawings in shipwreck/drawings.")
        records = timbers.timbers_of(chunk)
        if records:
            out["timbers"] = t(f"{len(records)} legni.", f"{len(records)} timbers.")
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
        ship.set_frame(chunk, ship.frame_from_picks(chunk, xy[0], xy[-1], old.bow if old else "start"))
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
            ship.set_frame(chunk, ship.frame_from_markers(chunk, v["a"], v["b"], old.bow if old else "start"))
            self._changed()

    def _do_frame_options(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        v = self._ask(t("Sistema della nave", "Ship frame"), [
            Field("bow", t("La prua è", "The bow is"), frame.bow, choices=[("start", t("all'inizio di x", "at the start of x")),
                                                                            ("end", t("alla fine di x", "at the end of x"))],
                  tip=t("Determina quale lato è babordo. Non si vede dal modello: lo dici tu.",
                        "It decides which side is port. The model does not show it: you say it.")),
            Field("x0", t("x del primo estremo (m)", "x of the first end (m)"), frame.x0, minimum=-1e4, maximum=1e4, decimals=3,
                  tip=t("0 alla prua o a una stazione scelta", "0 at the stem or at a chosen station")),
            Field("z0", t("Quota del datum z = 0 (quota del chunk, m)", "Height of the datum z = 0 (height of the chunk, m)"),
                  frame.z0, minimum=-1e4, maximum=1e5, decimals=3)])
        if v:
            ship.set_frame(chunk, ship.Frame(frame.start, frame.end, v["bow"], v["z0"], v["x0"]))
            self._changed()

    def _do_frame_flip(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.set_frame(chunk, ship.need_frame(chunk).flipped())
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
        frame = ship.need_frame(chunk)
        v = self._ask(t("Sezioni", "Sections"), [
            Field("longitudinals", t("Longitudinali, y (m)", "Longitudinals, y (m)"), "0", kind="text"),
            Field("horizontals", t("Orizzontali, z (m)", "Horizontals, z (m)"), "", kind="text"),
            Field("half_width", t("Metà larghezza disegnata (m)", "Half width drawn (m)"), round(0.3 * frame.length, 2),
                  minimum=0.01, maximum=1e4, decimals=2)],
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
            self._status(t(f"{timbers.import_csv(chunk, path)} legni importati", "Timbers imported"))
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
            from .drawings import KIND_TITLES

            folder = ship.folder(chunk) / "drawings"
            kids = [{"label": t(*KIND_TITLES[k]), "icon": "plugin", "activate": lambda f=folder / f"{k}.pdf": self._open(f)}
                    for k in made]
            children.append({"label": t("Disegni", "Drawings"), "icon": "plugin", "children": kids})
        records = timbers.timbers_of(chunk)
        if records:
            children.append({"label": t(f"Legni ({len(records)})", f"Timbers ({len(records)})"), "icon": "plugin",
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

        parser = argparse.ArgumentParser(prog="ppg plugin shipwreck", description="the technical drawing of a hull from its 3D model")
        parser.add_argument("what", choices=["frame", "stations", "sections", "water", "drawings", "timbers", "export", "status"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--start", help="frame: first end of the keel, X,Y[,Z] world, or a marker with --markers")
        parser.add_argument("--end", help="frame: other end of the keel")
        parser.add_argument("--markers", action="store_true", help="frame: --start and --end are markers")
        parser.add_argument("--bow", choices=["start", "end"], default="start", help="frame: the end where the bow is")
        parser.add_argument("--flip", action="store_true", help="frame: swap the ends")
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
        ns = parser.parse_args([str(a) for a in args])
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
                elif ns.markers:
                    ship.set_frame(chunk, ship.frame_from_markers(chunk, ns.start, ns.end, ns.bow, ns.z0, ns.x0))
                elif ns.start and ns.end:
                    a, b = (np.array([float(v) for v in s.split(",")]) for s in (ns.start, ns.end))
                    ship.set_frame(chunk, ship.frame_from_picks(chunk, a, b, ns.bow, ns.z0, ns.x0))
                else:
                    raise ship.NotReady("--start and --end, or --flip")
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


def parse_values(text) -> list[float]:
    """Numbers separated by spaces or ``;``, decimal point or comma."""
    return [float(v.replace(",", ".")) for v in str(text).replace(";", " ").split()]


def _print_progress(fraction: float, message: str) -> None:
    print(f"{fraction * 100:5.1f} % {message}", flush=True)
