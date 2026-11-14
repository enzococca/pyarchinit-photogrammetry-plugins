"""Masonry (Murature) - a plugin of pyarchinit-photogrammetry: walls, buildings and stone towers recorded by their
faces.

* a wall (a building, a tower) is a record ``{id, name, kind, faces, made}`` in ``chunk.meta["masonry"]["walls"]``;
  its faces are the program's elevations: on a vertical plane through two points of the plan, on a cylinder (three
  points, or the circle of a horizontal slice of the model) for a round tower, a well, an apse, or one plane per side
  of a polygonal tower;
* the shape of a face measured on the model: the fitted plane (or cylinder) with Huber's weight, the lean (degrees,
  cm/m, azimuth), the bulge map on the elevation (cm, towards the viewer) shown over it with a diverging palette and a
  legend; horizontal sections stacked in a DXF and an SVG with the offsets between them; the axis of a round tower
  from its circles at several heights and the out-of-round; two surveys of the same wall compared on the face;
* stone by stone: the measures of the stones drawn on a face (the program's stone tool), the courses, the mortar share
  and the type the measures suggest; the USM drawn on a face, the relationships their geometry proposes (accepted by
  the user) and their saving in PyArchInit with the face's drawings as media;
* courses, joints and stones from the lines drawn on a face, classified by orientation by the program 1.7.7
  (:mod:`.classified`): added to the face's stones after a check plate, no stone twice;
* the *Murature* pane (on the right), the walls in the workspace tree,
  ``ppg plugin masonry face|shape|stones|classified|usm|plate|export|ifc|list|remove``; a plate at scale, the DXF,
  the IFC wall, the face in QGIS.

The program's elevations and drawings keep their formats; the plugin writes only under ``<chunk data>/masonry/``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, kind_title, metres, num, t

__version__ = "0.1.4"
NEGATIVE_VALUES = ("--line", "--side", "--circle", "--heights", "--stations", "--sections", "--slice", "--polygon")
BARE_ACCEPT = ""  # --accept given without a value (classified: add the stones; usm: an error)


class _StonesStep:
    """The stones a classified drawing added to a face (or put in place of its stones) as an entry of the window's
    *Modifica > Annulla*: the stones of the face before and after (the program's undo takes no snapshot of them). The
    whole list is put back, as the program's own snapshots do: a stone edited by hand between the import and *Annulla*
    goes back with it."""

    meta_keys = ()  # it changes the program's drawings, no data of a plugin

    def __init__(self, plugin, chunk, wid: str, eid: str, before: list[str], after: list[str]):
        self.plugin, self.chunk, self.wid, self.eid = plugin, chunk, wid, eid
        self.before, self.after = list(before), list(after)

    def _put(self, wkts: list[str]) -> None:
        self.plugin._set_face_stones(self.chunk, self.eid, wkts)
        self.plugin._changed(self.wid, self.eid)

    def undo(self) -> None:
        self._put(self.before)

    def redo(self) -> None:
        self._put(self.after)


class MasonryPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Muri, edifici e torri in pietra per facce: prospetti, strapiombo, spanciamento, sezioni, "
                             "asse delle torri, confronto fra rilievi", "Walls, buildings and stone towers by their "
                             "faces: elevations, lean, bulge, sections, axis of towers, two surveys compared")
        self.context = None
        self.pane = None
        self.overlay = None
        self._overlay_wanted = True

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Murature...", "Masonry..."), self.show_pane,
                           tip=t("Il pannello delle murature", "The masonry pane"))
        context.add_command("masonry", self.command,
                            help="walls by their faces: face, shape, stones, classified, usm, plate, export, ifc, "
                                 "list, remove")
        context.add_exporter(t("Muro (IFC)", "Wall (IFC)"), ".ifc", self._export_ifc)
        context.add_exporter(t("Tavola della faccia (PDF)", "Plate of the face (PDF)"), ".pdf", self._export_plate)
        if context.window is not None:
            from .pane import MasonryPane

            self.pane = MasonryPane()
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            self.pane.requested.connect(self._requested)
            self.pane.wall_changed.connect(lambda _w: self._fill_faces())
            self.pane.face_changed.connect(lambda _f: self._fill_summary())
            context.on_refresh(self.refresh_pane)
            context.on_raster_loaded(self._raster_loaded)
            context.add_tree_items(self._tree_items)
            window = context.window
            window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
            self.refresh_pane()

    def deactivate(self) -> None:
        window = self.context.window if self.context else None
        if window is not None:
            window.qgis_providers = [p for p in getattr(window, "qgis_providers", []) if p != self._qgis_entries]
        if self.overlay is not None:
            self.overlay.remove()
        self.overlay = None
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

    def _need_chunk(self):
        from .walls import NotReady

        chunk = self._chunk()
        if chunk is None:
            raise NotReady(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _message(self, kind: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text)

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        from .walls import NotReady

        try:
            function(*args)
        except (NotReady, ValueError, RuntimeError) as exc:
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

    def _changed(self, wid: str | None = None, eid: str | None = None) -> None:
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane(wid)
        if eid and self.pane is not None:
            self.pane.select_face(eid)
            self._fill_summary()

    def _need_plan(self) -> None:
        """The plan to trace on: the chunk's own orthophoto or DEM (an elevation shown has (u, v), not plan x, y);
        without either, an empty frame around the model."""
        if self.context.show_raster(None):
            return
        from .faces import world_vertices

        v = world_vertices(self._need_chunk())
        lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
        self.context.show_frame((lo[0], lo[1], hi[0], hi[1]), max(hi - lo) / 1500.0, t(*NAME))

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self, wid: str | None = None) -> None:
        if self.pane is None:
            return
        from . import walls

        chunk = self._chunk()
        records = []
        if chunk is not None:
            walls.prune(chunk)
            records = walls.walls(chunk)
        self.pane.set_walls(records, wid)
        self._fill_faces()

    def _fill_faces(self) -> None:
        from . import walls

        chunk = self._chunk()
        wid = self.pane.wall_id()
        rows = []
        if chunk is not None and wid:
            rec = walls.find(chunk, wid)
            for eid in (rec or {}).get("faces", []):
                try:
                    rows.append(walls.face_info(chunk, eid))
                except walls.NotReady:
                    continue
        self.pane.set_faces(rows)
        self._fill_summary()

    def _fill_summary(self) -> None:
        if self.pane is None:
            return
        from . import shape, walls

        chunk = self._chunk()
        wid, eid = self.pane.wall_id(), self.pane.face_id()
        if chunk is None or not wid:
            self.pane.set_summary("")
            return
        from . import stones, usm

        parts, rows = [], []
        rec = walls.find(chunk, wid) or {}
        if eid:
            res = walls.results(chunk, wid, eid)
            parts.append(shape.summary_text(res.get("shape") or {}))
            if res.get("sections"):
                parts.append(shape.sections_text(res["sections"]))
            for r in (res.get("surveys") or {}).values():
                parts.append(shape.compare_text(r))
            parts.append(stones.stones_text(res.get("stones") or {}))
            path = walls.face_folder(chunk, wid, eid) / "pietre.json"
            if res.get("stones", {}).get("typology") and path.exists():
                rows = json.loads(path.read_text())["stones"]
            accepted = (res.get("usm") or {}).get("accepted_full") or []
            if accepted:
                parts.append(t("Rapporti accettati:", "Relationships accepted:") + "\n" + "\n".join(
                    usm.proposal_text(p) for p in accepted))
            saved = (res.get("usm") or {}).get("saved")
            if saved:
                parts.append(t(f"In PyArchInit: {len(saved['created'])} schede nuove, {saved['relations']} rapporti "
                               f"scritti, {len(saved['media'])} media.", f"In PyArchInit: {len(saved['created'])} new "
                               f"forms, {saved['relations']} relationships written, {len(saved['media'])} media."))
        if rec.get("axis"):
            parts.append(shape.axis_text(rec["axis"]))
        self.pane.set_stones(rows)
        self.pane.set_summary("\n\n".join(p for p in parts if p))

    def _requested(self, name: str) -> None:
        self._guarded(getattr(self, "_do_" + name))

    def _wall(self, create: bool = False, kind: str = "wall") -> dict:
        from . import walls

        chunk = self._need_chunk()
        wid = self.pane.wall_id() if self.pane is not None else None
        rec = walls.find(chunk, wid) if wid else None
        if rec is None:
            if not create:
                raise walls.NotReady(t("Scegli un muro (o fanne uno nuovo).", "Choose a wall (or make a new one)."))
            rec = walls.new_wall(chunk, kind=kind)
            self._changed(rec["id"])
        return rec

    def _face(self) -> tuple[dict, dict]:
        from . import walls

        rec = self._wall()
        eid = self.pane.face_id() if self.pane is not None else None
        if not eid:
            raise walls.NotReady(t("Scegli una faccia nella tabella.", "Choose a face in the table."))
        return walls.need_face(self._need_chunk(), rec["id"], eid)

    # ------------------------------------------------------------------ walls
    def _do_wall(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import walls

        chunk = self._need_chunk()
        kinds = [(k, kind_title(k)) for k in walls.KINDS]
        v = self._ask(t("Nuovo muro", "New wall"), [
            Field("name", t("Nome", "Name"), walls._new_name(chunk, "wall"), kind="text"),
            Field("kind", t("Tipo", "Kind"), "wall", choices=kinds)])
        if not v:
            return
        rec = walls.new_wall(chunk, v.get("name") or None, v.get("kind", "wall"))
        self._changed(rec["id"])

    def _do_rename(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import walls

        chunk = self._need_chunk()
        rec = self._wall()
        kinds = [(k, kind_title(k)) for k in walls.KINDS]
        v = self._ask(t("Muro", "Wall"), [Field("name", t("Nome", "Name"), rec["name"], kind="text"),
                                         Field("kind", t("Tipo", "Kind"), rec["kind"], choices=kinds)])
        if not v:
            return
        live = walls._live(chunk, rec["id"])
        live["name"] = v.get("name") or rec["name"]
        live["kind"] = v.get("kind") or rec["kind"]
        walls.write_summary(chunk, rec["id"])
        self._changed(rec["id"])

    def _do_remove_wall(self) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import walls

        chunk = self._need_chunk()
        rec = self._wall()
        self._store()  # what the window holds of the face shown into the project: counted, and removed with it
        answer = dialogs.choice(self.context.window, "question", self.title,
                                t(f"Rimuovere il muro «{rec['name']}» e i file del plugin? I prospetti delle sue facce "
                                  "restano nel progetto, se non chiedi di toglierli.",
                                  f"Remove the wall «{rec['name']}» and the plugin's files? The elevations of its "
                                  "faces stay in the project unless you ask to remove them.")
                                + walls.drawings_lost_text(chunk, rec["faces"]),
                                [t("Rimuovi il muro", "Remove the wall"), t("Rimuovi anche i prospetti",
                                                                            "Remove the elevations too")])
        if answer is None:
            return
        self._drop_overlay()
        walls.remove_wall(chunk, rec["id"], elevations_too=answer == 1)
        self._changed()

    # ------------------------------------------------------------------ new faces
    def _face_fields(self, cylinder: bool = False):
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from .faces import CYLINDER_BAND_MIN, CYLINDER_BAND_SHARE, DEFAULT_BACK, DEFAULT_FRONT

        share, least = num(100 * CYLINDER_BAND_SHARE, 0), num(CYLINDER_BAND_MIN, 1)
        auto = t(f" (0 = {share} % del raggio, almeno {least} m)",
                 f" (0 = {share} % of the radius, at least {least} m)") if cylinder else ""
        return [Field("resolution", t("Risoluzione (m/px; 0 = dal modello)", "Resolution (m/px; 0 = from the model)"),
                      0.0, maximum=10.0, decimals=4),
                Field("front", t("Fascia davanti al muro (m)", "Band in front of the wall (m)") + auto,
                      0.0 if cylinder else DEFAULT_FRONT, maximum=100.0, decimals=2),
                Field("back", t("Fascia dietro il muro (m)", "Band behind the wall (m)") + auto,
                      0.0 if cylinder else DEFAULT_BACK, maximum=100.0, decimals=2)]

    def _sized(self, dry, resolution):
        """``(resolution, force)`` of a face about to be built: the dry run's resolution (the model's when none is
        given); a face over 20 000 pixels on a side is said with its size, and the user chooses the resolution that
        fits or to build it anyway. None when the user gives up."""
        from pyarchinit_photogrammetry.gui import dialogs

        from .faces import FaceTooLarge

        try:
            return dry(resolution, False)["resolution"], False
        except FaceTooLarge as exc:
            answer = dialogs.choice(self.context.window, "question", self.title, str(exc),
                                    [t(f"Usa {num(exc.proposed * 1000, 1)} mm per pixel",
                                       f"Use {num(exc.proposed * 1000, 1)} mm per pixel"),
                                     t("Costruisci comunque", "Build it anyway")])
            if answer is None:
                return None
            return (exc.proposed, False) if answer == 0 else (exc.resolution, True)

    def _made(self, wid: str, entries) -> None:
        entries = entries if isinstance(entries, list) else [entries]
        self._changed(wid, entries[0]["id"] if entries else None)
        if entries:
            self._show_face(entries[0])

    def _do_plane(self) -> None:
        self._need_chunk()
        self._need_plan()
        self.context.request_line(lambda xy: self._guarded(self._plane_dialog, xy),
                                  hint=t("Traccia in pianta la faccia del muro: due punti, doppio clic o Invio per "
                                         "finire", "Trace the face of the wall in plan: two points, double click or "
                                         "Enter ends"))

    def _plane_dialog(self, xy) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import faces

        chunk = self._need_chunk()
        xy = np.asarray(xy, np.float64)
        if len(xy) < 2:
            raise faces.NotReady(t("Servono due punti.", "Two points are needed."))
        p1, p2 = xy[0], xy[-1]
        rec = self._wall(create=True)
        sides = [("auto", t("dalla parte delle foto", "the side of the photos")),
                 ("left", t("a sinistra della traccia", "left of the trace")),
                 ("right", t("a destra della traccia", "right of the trace"))]
        v = self._ask(t("Faccia su un piano", "Face on a plane"), [
            Field("name", t("Nome", "Name"), f"{rec['name']} - {t('faccia', 'face')} {len(rec['faces']) + 1}",
                  kind="text"),
            Field("side", t("Chi guarda sta", "The viewer stands"), "auto", choices=sides)] + self._face_fields(),
            intro=t(f"Traccia lunga {metres(float(np.linalg.norm(p2 - p1)))}; il prospetto è verticale, v = quota.",
                    f"Trace {metres(float(np.linalg.norm(p2 - p1)))} long; the elevation is vertical, v = height."))
        if not v:
            return
        side = None
        if v.get("side") in ("left", "right"):
            along = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-12)
            left = np.array([-along[1], along[0]])
            side = (p1 + p2) / 2 + (left if v["side"] == "left" else -left)
        front, back = v.get("front", 0.5), v.get("back", 0.3)
        sized = self._sized(lambda r, f: faces.plane_face(chunk, rec["id"], p1, p2, side, r, front, back,
                                                          force=f, dry_run=True), v.get("resolution") or None)
        if sized is None:
            return
        res, force = sized
        self._run(t("Faccia del muro", "Face of the wall"), lambda p: faces.plane_face(
            chunk, rec["id"], p1, p2, side, res, front, back, v.get("name") or None, progress=p, force=force),
            lambda e: self._made(rec["id"], e))

    def _model_mid_height(self, chunk) -> float:
        from .faces import world_vertices

        z = world_vertices(chunk)[:, 2]
        return float((z.min() + z.max()) / 2)

    def _do_cylinder(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import faces

        chunk = self._need_chunk()
        rec = self._wall(create=True, kind="tower")
        how = [("slice", t("cerchio della sezione a una quota", "circle of the slice at a height")),
               ("points", t("tre punti (o più) in pianta", "three points (or more) in plan"))]
        v = self._ask(t("Faccia su un cilindro", "Face on a cylinder"), [
            Field("name", t("Nome", "Name"), "", kind="text"),
            Field("how", t("Cilindro da", "Cylinder from"), "slice", choices=how),
            Field("height", t("Quota della sezione (m)", "Height of the slice (m)"), self._model_mid_height(chunk),
                  minimum=-1e5, maximum=1e5, decimals=3),
            Field("inside", t("Visto da dentro (pozzo, abside)", "Seen from inside (well, apse)"), False)]
            + self._face_fields(cylinder=True))
        if not v:
            return

        def build(points=None):
            kw = {"points": points, "height": None if points is not None else v["height"],
                  "inside": bool(v.get("inside")), "front": v.get("front") or None, "back": v.get("back") or None}
            sized = self._sized(lambda r, f: faces.cylinder_face(chunk, rec["id"], resolution=r, force=f,
                                                                 dry_run=True, **kw), v.get("resolution") or None)
            if sized is None:
                return
            res, force = sized
            self._run(t("Faccia cilindrica", "Cylindrical face"), lambda p: faces.cylinder_face(
                chunk, rec["id"], resolution=res, name=v.get("name") or None, progress=p, force=force, **kw),
                lambda e: self._made(rec["id"], e))

        if v.get("how") == "points":
            self._need_plan()
            self.context.request_line(lambda xy: self._guarded(build, np.asarray(xy, np.float64)),
                                      hint=t("Clic su tre punti (o più) del bordo in pianta; doppio clic o Invio per "
                                             "finire", "Click three points (or more) of the rim in plan; double click "
                                             "or Enter ends"))
        else:
            build()

    def _do_polygon(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import faces

        chunk = self._need_chunk()
        rec = self._wall(create=True, kind="tower")
        v = self._ask(t("Torre poligonale", "Polygonal tower"), [
            Field("height", t("Quota della sezione (m)", "Height of the slice (m)"), self._model_mid_height(chunk),
                  minimum=-1e5, maximum=1e5, decimals=3),
            Field("min_side", t("Lato minimo (m)", "Shortest side (m)"), 0.5, maximum=1000.0, decimals=2),
            Field("tolerance", t("Semplificazione (m; 0 = automatica)", "Simplification (m; 0 = automatic)"), 0.0,
                  maximum=10.0, decimals=3)] + self._face_fields())
        if not v:
            return
        self._run(t("Facce della torre", "Faces of the tower"), lambda p: faces.polygon_faces(
            chunk, rec["id"], v["height"], v.get("min_side", 0.5), v.get("tolerance") or None,
            v.get("resolution") or None, v.get("front", 0.5), v.get("back", 0.3), progress=p),
            lambda es: self._made(rec["id"], es))

    def _do_adopt(self) -> None:
        """An elevation the program already has, made a face of the wall as it is (nothing rebuilt)."""
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import walls
        from .texts import surface_title

        chunk = self._need_chunk()
        free = walls.free_elevations(chunk)
        if not free:
            raise walls.NotReady(t("Nessun prospetto libero nel chunk: fai un prospetto con il pulsante Prospetto del "
                                   "programma (piano o cilindro), oppure una faccia da qui.",
                                   "No free elevation in the chunk: make one with the program's Elevation button "
                                   "(plane or cylinder), or a face from here."))
        rec = self._wall(create=True)
        choices = []
        for e in free:
            info = walls.face_info(chunk, e["id"])
            tilt = (e.get("tilt") or {}).get("tilt_deg")
            axis = "" if not tilt or tilt < 0.05 else t(f", asse inclinato di {num(tilt, 1)}°",
                                                        f", axis tilted by {num(tilt, 1)}°")
            choices.append((e["id"], f"{info['name']} ({surface_title(info['surface'])}{axis}, "
                                     f"{metres(info['width'])} x {metres(info['height'])}, "
                                     f"{num(info['resolution'] * 1000, 1)} mm/px)"))
        v = self._ask(t("Prospetto esistente", "Existing elevation"),
                      [Field("eid", t("Prospetto", "Elevation"), choices[0][0], choices=choices)],
                      intro=t(f"Il prospetto scelto diventa una faccia del muro «{rec['name']}» così com'è: le "
                              "sue patch, i suoi disegni e la sua risoluzione restano quelli del programma.",
                              f"The chosen elevation becomes a face of the wall «{rec['name']}» as it is: its "
                              "patches, drawings and resolution stay the program's."))
        if not v:
            return
        entry = walls.adopt_face(chunk, rec["id"], v["eid"])
        self._made(rec["id"], entry)

    # ------------------------------------------------------------------ the face shown, its shape
    def _show_face(self, entry: dict) -> None:
        window = self.context.window
        if window is None or not hasattr(window, "show_elevation"):
            return
        window.show_elevation(entry)
        self._show_overlay(entry)

    def _raster_loaded(self) -> None:
        if self.overlay is not None and not self.overlay.shown():
            self.overlay.remove()

    def _drop_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()

    def _show_overlay(self, entry: dict, force: bool = False) -> bool:
        from . import walls

        if not (self._overlay_wanted or force):
            return False
        window = self.context.window
        view = getattr(window, "ortho_view", None)
        chunk = self._chunk()
        rec = walls.wall_of_face(chunk, entry["id"]) if chunk is not None else None
        if view is None or rec is None:
            return False
        res = walls.results(chunk, rec["id"], entry["id"]).get("shape") or {}
        if not res.get("bulge") or not (chunk.data_dir / res["bulge"]).exists():
            return False
        from .overlay import Overlay

        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        self.overlay.show(chunk.data_dir / res["bulge"], entry["id"],
                          t(f"Spanciamento - {entry.get('name', '')}", f"Bulge - {entry.get('name', '')}"))
        return True

    def _do_show(self) -> None:
        _w, entry = self._face()
        self._show_face(entry)

    def _do_shape(self) -> None:
        from . import shape

        chunk = self._need_chunk()
        rec, entry = self._face()

        def done(_result):
            self._overlay_wanted = True
            self._changed(rec["id"], entry["id"])
            self._show_face(entry)

        self._run(t("Forma della faccia", "Shape of the face"),
                  lambda p: shape.face_shape(chunk, rec["id"], entry["id"], progress=p), done)

    def _do_bulge(self) -> None:
        _rec, entry = self._face()
        if self.overlay is not None and self.overlay.shown() and self.overlay.key == entry["id"]:
            self.overlay.remove()
            self._overlay_wanted = False
            return
        self._overlay_wanted = True
        window = self.context.window
        if getattr(window, "_raster_version", None) != ("elevation", entry["id"]):
            window.show_elevation(entry)
        if not self._show_overlay(entry, force=True):
            raise RuntimeError(t("La forma della faccia non è ancora misurata: Forma.",
                                 "The shape of the face is not measured yet: Shape."))

    def _do_sections(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import shape

        chunk = self._need_chunk()
        rec, entry = self._face()
        frame = shape._frame(entry)
        heights = "; ".join(num(h, 3) for h in shape.default_heights(frame, entry))
        stations = "; ".join(num(u, 2) for u in shape.default_stations(entry))
        v = self._ask(t("Sezioni orizzontali", "Horizontal sections"), [
            Field("heights", t("Quote (m), separate da punto e virgola", "Heights (m), separated by semicolons"),
                  heights, kind="text"),
            Field("stations", t("Stazioni lungo la faccia (u, m)", "Stations along the face (u, m)"), stations,
                  kind="text"),
            Field("reach", t("Distanza massima dalla faccia (m)", "Largest distance from the face (m)"),
                  shape.default_reach(entry),
                  maximum=100.0, decimals=2)])
        if not v:
            return
        hs, us = parse_typed(v.get("heights", "")), parse_typed(v.get("stations", ""))
        self._run(t("Sezioni della faccia", "Sections of the face"), lambda p: shape.sections(
            chunk, rec["id"], entry["id"], hs or None, us or None, v.get("reach") or None, progress=p),
            lambda _r: self._changed(rec["id"], entry["id"]))

    def _do_axis(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import shape, walls

        chunk = self._need_chunk()
        rec = self._wall()
        cyl = next((walls.with_bounds(chunk, e) for e in (walls.elevation(chunk, f) for f in rec["faces"])
                    if e is not None and (e.get("frame") or {}).get("kind") == "cylinder"), None)
        default = "; ".join(num(h, 3) for h in shape.default_heights(shape._frame(cyl), cyl, 5)) if cyl else ""
        v = self._ask(t("Asse della torre", "Axis of the tower"), [
            Field("heights", t("Quote delle sezioni (m), separate da punto e virgola",
                               "Heights of the sections (m), separated by semicolons"), default, kind="text")],
            intro=t("Il cerchio adattato alla sezione del modello a ogni quota; l'asse passa per i loro centri.",
                    "The circle fitted to the model's slice at each height; the axis goes through their centres."))
        if not v:
            return
        hs = parse_typed(v.get("heights", ""))
        self._run(t("Asse della torre", "Axis of the tower"), lambda p: shape.tower_axis(
            chunk, rec["id"], hs or None, progress=p), lambda _r: self._changed(rec["id"], self.pane.face_id()))

    def _do_compare(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import shape

        chunk = self._need_chunk()
        rec, entry = self._face()
        others = [c for c in chunk.document.chunks if c is not chunk and c.product_path("model") is not None]
        if not others:
            raise shape.NotReady(t("Il progetto non ha un altro chunk con un modello da confrontare.",
                                   "The project has no other chunk with a model to compare."))
        v = self._ask(t("Confronto fra rilievi", "Two surveys compared"), [
            Field("other", t("Altro rilievo (chunk)", "Other survey (chunk)"), 0,
                  choices=[(i, c.label) for i, c in enumerate(others)])],
            intro=t("I due chunk devono essere già allineati nello stesso sistema.",
                    "The two chunks must be aligned in the same system already."))
        if not v:
            return
        other = others[int(v.get("other", 0))]
        self._run(t("Confronto fra rilievi", "Two surveys compared"), lambda p: shape.compare_surveys(
            chunk, rec["id"], entry["id"], other, progress=p), lambda _r: self._changed(rec["id"], entry["id"]))

    def _do_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from . import walls

        chunk = self._need_chunk()
        rec = self._wall()
        eid = self.pane.face_id()
        target = walls.face_folder(chunk, rec["id"], eid) if eid else walls.wall_folder(chunk, rec["id"])
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _do_remove_face(self) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import walls

        chunk = self._need_chunk()
        rec, entry = self._face()
        self._store()  # what the window holds of the face shown into the project: counted, and removed with it
        answer = dialogs.choice(self.context.window, "question", self.title,
                                t(f"Togliere la faccia «{entry['name']}» dal muro? Il prospetto resta nel progetto, se "
                                  "non chiedi di toglierlo.", f"Take the face «{entry['name']}» out of the wall? The "
                                  "elevation stays in the project unless you ask to remove it.")
                                + walls.drawings_lost_text(chunk, [entry["id"]]),
                                [t("Togli la faccia", "Take the face out"), t("Rimuovi anche il prospetto",
                                                                              "Remove the elevation too")])
        if answer is None:
            return
        if self.overlay is not None and self.overlay.key == entry["id"]:
            self.overlay.remove()
        walls.remove_face(chunk, rec["id"], entry["id"], elevation_too=answer == 1)
        self._changed(rec["id"])

    # ------------------------------------------------------------------ stones and courses
    def _store(self) -> None:
        """The drawings the window shows into the project first: the stones and the units of the face live there."""
        try:
            self.context.store_drawings()
        except RuntimeError:  # not in the interface thread: the project holds what was stored
            pass

    def _show_if_hidden(self, entry: dict) -> None:
        window = self.context.window
        if getattr(window, "_raster_version", None) != ("elevation", entry["id"]):
            self._show_face(entry)

    def _do_find_stones(self) -> None:
        _rec, entry = self._face()
        window = self.context.window
        self._show_if_hidden(entry)
        find = getattr(window, "draw_stones", None)
        if find is None:
            raise RuntimeError(t("Usa lo strumento Pietre del programma sul prospetto della faccia, poi Misura le "
                                 "pietre.", "Use the program's Stones tool on the face's elevation, then Measure the "
                                 "stones."))
        find(tuple(entry["bounds"]))

    def _do_measure_stones(self) -> None:
        from . import stones

        chunk = self._need_chunk()
        rec, entry = self._face()
        self._store()
        self._run(t("Misura delle pietre", "Measuring the stones"),
                  lambda p: stones.measure_face(chunk, rec["id"], entry["id"], progress=p),
                  lambda _r: self._changed(rec["id"], entry["id"]))

    def _do_wall_area(self) -> None:
        from . import stones

        chunk = self._need_chunk()
        rec, entry = self._face()
        self._show_if_hidden(entry)

        def drawn(uv):
            stones.set_region(chunk, rec["id"], entry["id"], uv)
            self._changed(rec["id"], entry["id"])
            if (self._results(rec["id"], entry["id"]).get("stones") or {}).get("typology"):
                self._do_measure_stones()  # measured before: measured again on the area drawn

        self.context.request_polygon(lambda uv: self._guarded(drawn, uv),
                                     hint=t("Disegna sulla faccia l'area del muro (per la quota di malta); doppio "
                                            "clic o Invio per finire", "Draw on the face the wall's area (for the "
                                            "mortar share); double click or Enter ends"))

    def _results(self, wid: str, eid: str) -> dict:
        from . import walls

        return walls.results(self._need_chunk(), wid, eid)

    def _do_export_face(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import stones

        chunk = self._need_chunk()
        rec, entry = self._face()
        folder = QFileDialog.getExistingDirectory(self.context.window, t("Cartella per la faccia",
                                                                         "Folder for the face"))
        if not folder:
            return
        self._store()
        paths = stones.export_face(chunk, rec["id"], entry["id"], folder)
        self._message("information", t(f"{len(paths)} file della faccia «{entry['name']}» in {folder}.",
                                       f"{len(paths)} files of the face «{entry['name']}» in {folder}."))

    # ------------------------------------------------------------------ a drawing classified by orientation
    def _do_classified(self) -> None:
        """The lines drawn on the face (or a file in its ``(u, v)``) classified by the program along the face's
        horizontal: beds, joints, courses and stones; the check plate first, then the stones the user keeps added to
        those of the face as one step of *Annulla*."""
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import classified, walls

        classified.need_program()
        chunk = self._need_chunk()
        rec, entry = self._face()
        self._store()
        defaults = {**classified.orient_lines.DEFAULTS, **classified.orient_lines.EXTRA_DEFAULTS,
                    **classified.PLUGIN_DEFAULTS}
        v = self._ask(t("Corsi e giunti da un disegno classificato", "Courses and joints from a classified drawing"), [
            Field("source", t("Disegno", "Drawing"), "face",
                  choices=[("face", t("Le linee disegnate sulla faccia", "The lines drawn on the face")),
                           ("file", t("Il file qui sotto (in u, v della faccia)",
                                      "The file below (in the face's u, v)"))]),
            Field("file", t("File (GeoPackage, GeoJSON, DXF)", "File (GeoPackage, GeoJSON, DXF)"), "", kind="file_open",
                  filter="GeoPackage, GeoJSON, DXF (*.gpkg *.geojson *.json *.dxf)"),
            Field("units", t("Anche i contorni delle USM della faccia", "The outlines of the face's USM too"),
                  bool(defaults["units"])),
            Field("joint_cover", t("Un giunto divide il filare se ne copre (quota dell'altezza)",
                                   "A joint splits the course when it covers (share of its height)"),
                  defaults["joint_cover"], minimum=0.05, maximum=1.0, decimals=2),
            Field("min_stone", t("Pietra più piccola (m)", "Smallest stone (m)"), defaults["min_stone"],
                  minimum=0.005, maximum=5.0, decimals=3),
            Field("pair_isolation", t("Isolamento delle coppie", "Pair isolation"), defaults["pair_isolation"],
                  minimum=0.0, maximum=10.0, decimals=2,
                  tip=t("Letti a distanze regolari non sono coppie (1,5); 0 toglie il controllo",
                        "Beds at regular distances are not pairs (1.5); 0 turns the check off")),
            Field("min_length", t("Tratto più corto (m)", "Shortest piece (m)"), defaults["min_length"],
                  minimum=0.0, maximum=10.0, decimals=3)],
            intro=t("Le linee della faccia ordinate lungo la sua orizzontale: linee lungo l'asse = letti, coppie lungo "
                    "l'asse = filari di spessore noto, linee di traverso = giunti. Le pietre sono le celle fra due "
                    "letti divise dai giunti. Prima la tavola di controllo, poi le pietre si aggiungono a quelle "
                    "della faccia (le doppie saltate).",
                    "The lines of the face sorted along its horizontal: lines along = beds, pairs along = courses of "
                    "known thickness, lines across = joints. The stones are the cells between two beds split by the "
                    "joints. The check plate first, then the stones are added to those of the face (duplicates "
                    "skipped)."))
        if not v:
            return
        source = None
        if v["source"] == "file":
            if not str(v.get("file") or "").strip():
                raise walls.NotReady(t("Scegli il file del disegno.", "Choose the file of the drawing."))
            source = str(v["file"]).strip()
        params = {k: float(v[k]) for k in ("joint_cover", "min_stone", "pair_isolation", "min_length")}
        params["units"] = bool(v.get("units"))
        self.classify_face(chunk, rec["id"], entry["id"], source, params)

    def classify_face(self, chunk, wid: str, eid: str, source=None, params=None) -> None:
        """:func:`classified.propose` and its plate in the worker, then :meth:`_classified_ready`."""
        from . import classified, walls

        def job(_p):
            prop = classified.propose(chunk, eid, source, params)
            folder = walls.face_folder(chunk, wid, eid) / "classificato"
            return prop, classified.plate(chunk, eid, prop, folder / "tavola_controllo.png")

        self._run(t("Disegno classificato", "Classified drawing"), job,
                  lambda out: self._guarded(self._classified_ready, chunk, wid, eid, *out))

    def _classified_ready(self, chunk, wid: str, eid: str, prop: dict, plate) -> None:
        from . import classified

        if not prop["stones"]:
            self._message("warning", t("Nessuna pietra dal disegno: servono almeno due letti (linee lungo "
                                       "l'orizzontale della faccia) uno sopra l'altro.",
                                       "No stone from the drawing: at least two beds (lines along the face's "
                                       "horizontal) one above the other are needed.") + "\n\n"
                          + classified.summary_text(prop))
            return
        self._store()
        _wkts, skipped = classified.merged(chunk, eid, prop)
        replace = self._check_plate(plate, prop, len(skipped), len(classified.stored(chunk, eid)))
        if replace is None:
            return
        before = classified.stored(chunk, eid)
        added = classified.accept(chunk, eid, prop, replace=replace)
        after = classified.stored(chunk, eid)
        text = classified.accepted_text(added, len(prop["already"]))
        if after != before:
            self._set_face_stones(chunk, eid, after)
            window = self.context.window
            history = getattr(window, "history", None) if window is not None else None
            # inside a script (the console's .py files) the script is one entry of Annulla, taken by the program from
            # its own snapshot, which holds no stones: an action of this plugin run from a script cannot be undone
            # there, so no entry of its own is pushed in the middle of the script's
            if history is not None and not getattr(window, "_script_running", False) and \
                    chunk is getattr(window, "chunk", None):
                history.push(t("Murature: pietre da un disegno classificato",
                               "Masonry: stones from a classified drawing"),
                             _StonesStep(self, chunk, wid, eid, before, after), document=False)
            self._changed(wid, eid)
        status = getattr(self.context.window, "statusBar", None)
        if status is not None:
            status().showMessage(text, 15000)
        if prop["already"]:
            self._message("information", text)

    def _set_face_stones(self, chunk, eid: str, wkts: list[str]) -> None:
        """The stones of the face set to ``wkts``: through the window when it is the chunk shown (it keeps the stones
        of the elevation it shows and draws them), else in the project."""
        from . import classified

        window = self.context.window if self.context else None
        put = getattr(window, "set_stones", None) if window is not None else None
        if put is not None and chunk is getattr(window, "chunk", None):
            from shapely import wkt

            put(eid, [wkt.loads(x) for x in wkts], None)
        else:
            classified.put(chunk, eid, wkts)

    def _check_plate(self, plate, prop: dict, duplicates: int, present: int) -> bool | None:
        """The check plate with the courses and stones proposed: None when cancelled, else whether the stones of the
        face are replaced (a box, off) instead of added to."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        from . import classified

        dlg = QDialog(self.context.window)
        dlg.setWindowTitle(t("Tavola di controllo", "Check plate"))
        lay = QVBoxLayout(dlg)
        image = QLabel()
        pix = QPixmap(str(plate))
        if not pix.isNull():
            image.setPixmap(pix.scaled(960, 680, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))
        lay.addWidget(image)
        note = QLabel(classified.summary_text(prop))
        note.setWordWrap(True)
        lay.addWidget(note)
        if duplicates:
            dup = QLabel(t(f"{duplicates} pietre proposte sono già sulla faccia: saranno saltate.",
                           f"{duplicates} stones proposed are already on the face: they will be skipped."))
            dup.setWordWrap(True)
            lay.addWidget(dup)
        box = QCheckBox(t(f"Sostituisci le {present} pietre della faccia", f"Replace the {present} stones of the face"))
        box.setChecked(False)
        box.setEnabled(present > 0)
        lay.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(t("Aggiungi le pietre", "Add the stones"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(t("Annulla", "Cancel"))
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        replace = box.isChecked()
        dlg.deleteLater()  # not kept under the main window with its pixmap until the window closes
        return replace if accepted else None

    # ------------------------------------------------------------------ USM
    def _choose_relations(self, proposals: list[dict]) -> list[dict] | None:
        from .pane import RelationsDialog

        dialog = RelationsDialog(proposals, self.context.window)
        return dialog.chosen() if dialog.exec() else None

    def _do_propose(self) -> None:
        from . import usm, walls

        chunk = self._need_chunk()
        rec, entry = self._face()
        self._store()
        proposals = usm.propose(chunk, rec["id"], entry["id"])
        if not proposals:
            self._message("information", t("La geometria delle USM sulla faccia non propone rapporti.",
                                           "The geometry of the USM on the face proposes no relationship."))
            return
        chosen = self._choose_relations(proposals)
        if chosen is None:
            return
        done = dict(walls.results(chunk, rec["id"], entry["id"]).get("usm") or {})
        done["accepted_full"] = chosen
        done["proposed"] = len(proposals)
        walls.set_result(chunk, rec["id"], entry["id"], "usm", done)
        self._changed(rec["id"], entry["id"])

    def _do_save_pyarchinit(self) -> None:
        from . import usm, walls

        chunk = self._need_chunk()
        rec, entry = self._face()
        pa = getattr(self.context.window, "pa", None)
        if pa is None:
            raise walls.NotReady(t("Collegati prima a PyArchInit (menu PyArchInit > Collega).",
                                   "Connect to PyArchInit first (menu PyArchInit > Connect)."))
        self._store()
        accepted = (walls.results(chunk, rec["id"], entry["id"]).get("usm") or {}).get("accepted_full") or []

        def done(r):
            self._changed(rec["id"], entry["id"])
            self._message("information", t(
                f"PyArchInit: {len(r['created'])} schede nuove, {r['relations']} rapporti scritti, {len(r['media'])} "
                f"media della faccia." + (f" Senza numero o sito: {', '.join(r['skipped'])}." if r["skipped"] else ""),
                f"PyArchInit: {len(r['created'])} new forms, {r['relations']} relationships written, "
                f"{len(r['media'])} media of the face." + (f" Without number or site: {', '.join(r['skipped'])}."
                                                          if r["skipped"] else "")))

        self._run(t("Salvataggio in PyArchInit", "Saving in PyArchInit"),
                  lambda _p: usm.save(pa, chunk, rec["id"], entry["id"], accepted), done)

    # ------------------------------------------------------------------ plates and exports
    def _do_plate(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import outputs

        chunk = self._need_chunk()
        rec, entry = self._face()
        u0, v0, u1, v1 = entry["bounds"]
        scales = [(0, t(f"automatica (1:{outputs.choose_scale(u1 - u0, v1 - v0)})",
                        f"automatic (1:{outputs.choose_scale(u1 - u0, v1 - v0)})"))] + \
            [(n, f"1:{n}") for n in outputs.SCALES]
        v = self._ask(t("Tavola della faccia", "Plate of the face"), [
            Field("scale", t("Scala", "Scale"), 0, choices=scales),
            Field("contours", t("Curve dello spanciamento", "Contours of the bulge"), True)])
        if not v:
            return
        self._store()

        def done(out):
            files = ", ".join(str(out[k]) for k in ("svg", "pdf", "png") if k in out)
            self._message("information", t(f"Tavola in scala 1:{out['scale']}: {files}",
                                           f"Plate at scale 1:{out['scale']}: {files}"))

        self._run(t("Tavola della faccia", "Plate of the face"), lambda _p: outputs.plate(
            chunk, rec["id"], entry["id"], scale=int(v.get("scale") or 0) or None,
            contours=bool(v.get("contours", True))), done)

    def _do_export_all(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import outputs

        chunk = self._need_chunk()
        rec, entry = self._face()
        folder = QFileDialog.getExistingDirectory(self.context.window, t("Cartella per la faccia",
                                                                         "Folder for the face"))
        if not folder:
            return
        self._store()
        self._run(t("Esportazione della faccia", "Exporting the face"),
                  lambda _p: outputs.export_face(chunk, rec["id"], entry["id"], folder),
                  lambda out: self._message("information", t(f"{len(out)} file della faccia «{entry['name']}» in "
                                                             f"{folder}.", f"{len(out)} files of the face "
                                                             f"«{entry['name']}» in {folder}.")))

    def _do_ifc(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ifc

        chunk = self._need_chunk()
        rec = self._wall()
        thickness = None
        try:
            ifc.wall_solids(chunk, rec["id"])
        except ifc.NotReady as exc:
            if "spessore" not in str(exc) and "thickness" not in str(exc):
                raise
            v = self._ask(t("Spessore del muro", "Thickness of the wall"), [
                Field("thickness", t("Spessore (m)", "Thickness (m)"), float(self.context.settings.get(
                    "thickness", 0.6)), minimum=0.01, maximum=50.0, decimals=3)],
                intro=t("Il muro ha una sola faccia rilevata: il solido va dietro di essa per lo spessore dato.",
                        "The wall has one face surveyed: the solid goes behind it by the thickness given."))
            if not v:
                return
            thickness = float(v["thickness"])
            self.context.settings["thickness"] = thickness
            self.context.save_settings()
        path, _f = QFileDialog.getSaveFileName(self.context.window, t("IFC del muro", "Wall IFC"),
                                               f"{rec['name']}.ifc", "IFC (*.ifc)")
        if not path:
            return
        self._store()
        self._run(t("IFC del muro", "Wall IFC"), lambda _p: ifc.export_wall(chunk, rec["id"], path, thickness),
                  lambda r: self._message("information", t(
                      f"IFC del muro in {r['path']}: spessore {metres(r['thickness'])} ({r['source']}), "
                      f"{r['parts']} USM, {r['relations']} rapporti.", f"Wall IFC in {r['path']}: thickness "
                      f"{metres(r['thickness'])} ({r['source']}), {r['parts']} USM, {r['relations']} relationships.")))

    def _export_ifc(self, chunk, path, progress=None):
        """*File > Esporta > Muro (IFC)*: the wall chosen in the pane (else the first); a wall whose thickness is not
        measured between two faces is refused (the pane's *IFC del muro...* asks it)."""
        from . import ifc, walls

        wid = self.pane.wall_id() if self.pane is not None else None
        records = [w for w in walls.walls(chunk) if not wid or w["id"] == wid] or walls.walls(chunk)
        if not records:
            raise walls.NotReady(t("Nessun muro nel chunk.", "No wall in the chunk."))
        try:
            return ifc.export_wall(chunk, records[0]["id"], path)
        except ifc.NotReady as exc:  # a thickness never given is never written: the pane asks it
            raise ifc.NotReady(f"{exc} " + t("Usa Murature > IFC del muro..., che chiede lo spessore.",
                                             "Use Masonry > Wall IFC..., which asks the thickness.")) from exc

    def _export_plate(self, chunk, path, progress=None):
        """*File > Esporta > Tavola della faccia (PDF)*: the face chosen in the pane (else the first face)."""
        from pathlib import Path

        from . import outputs, walls

        wid = self.pane.wall_id() if self.pane is not None else None
        eid = self.pane.face_id() if self.pane is not None else None
        rec = walls.find(chunk, wid) if wid else next((w for w in walls.walls(chunk) if w["faces"]), None)
        if rec is None or not rec["faces"]:
            raise walls.NotReady(t("Nessuna faccia nel muro scelto: fai prima una faccia.",
                                   "No face in the wall chosen: make a face first."))
        path = Path(path)
        return outputs.plate(chunk, rec["id"], eid or rec["faces"][0], path.parent, stem=path.stem, formats=("pdf",))

    def _qgis_entries(self):
        """For the QGIS project of the plan: the stones' measures of every face, in 3D on the face's plane."""
        from . import walls

        chunk = self._chunk()
        vectors = []
        for w in walls.walls(chunk) if chunk is not None else []:
            for eid in w["faces"]:
                path = walls.face_folder(chunk, w["id"], eid) / "pietre_misure_mondo.gpkg"
                entry = walls.elevation(chunk, eid)
                if path.exists() and entry is not None:
                    vectors.append({"name": t(f"{entry['name']} - pietre (misure, 3D)",
                                              f"{entry['name']} - stones (measures, 3D)"), "path": str(path),
                                    "table": "pietre_misure", "geometry": "POLYGON", "color": "#5b3a1e",
                                    "label": "filare"})
        return [], vectors

    def _do_qgis(self) -> None:
        from pyarchinit_photogrammetry.io import qgis_link

        from . import outputs, walls

        chunk = self._need_chunk()
        rec, entry = self._face()
        q = qgis_link.find_qgis()
        if q is None:
            raise walls.NotReady(t("QGIS non trovato: installalo da qgis.org (o indica il percorso con PPG_QGIS).",
                                   "QGIS not found: install it from qgis.org (or give its path with PPG_QGIS)."))
        self._store()
        folder = walls.face_folder(chunk, rec["id"], entry["id"]) / "qgis"

        def job(p):
            from pyarchinit_photogrammetry.engine.elevation import elevation_paths

            p(0.1, t("File della faccia", "Files of the face"))
            files = outputs.export_face(chunk, rec["id"], entry["id"], folder)
            ortho, _relief = elevation_paths(chunk, entry)  # the elevation's own orthophoto, in the face's frame
            rasters = [{"name": entry["name"], "path": str(ortho)}] if ortho.exists() else []
            if files.get("bulge"):
                rasters.append({"name": t("Spanciamento (cm)", "Bulge (cm)"), "path": str(files["bulge"]),
                                "ramp": "diverging", "opacity": 0.45})
            gpkg = str(files["drawings_gpkg"])
            vectors = [{"name": t(title, title), "path": gpkg, "table": table, "geometry": geometry, "color": colour,
                        "label": "us" if table == "unita" else ""}
                       for table, (title, geometry, colour) in qgis_link.ELEVATION_TABLES.items() if table != "profilo"]
            vectors.append({"name": t("Pietre (misure)", "Stones (measures)"), "path": gpkg, "table": "pietre_misure",
                            "geometry": "POLYGON", "color": "#5b3a1e", "label": "filare"})
            project = folder / "faccia.qgs"
            p(0.6, t("Progetto QGIS", "QGIS project"))
            qgis_link.build_project(project, "", {}, rasters, title=entry["name"], qgis=q, extra_vectors=vectors)
            qgis_link.open_in_qgis(project, q)
            return project

        self._run(t("Apertura in QGIS", "Opening in QGIS"), job,
                  lambda project: self._message("information", t(f"Progetto QGIS della faccia: {project} (coordinate "
                                                                 "della faccia, in metri).", f"QGIS project of the "
                                                                 f"face: {project} (the face's coordinates, metres).")))

    # ------------------------------------------------------------------ the workspace
    def _tree_items(self, chunk) -> list:
        from . import walls

        records = walls.walls(chunk)
        if not records:
            return []
        kids = []
        for rec in records:
            faces = []
            for eid in rec["faces"]:
                entry = walls.elevation(chunk, eid)
                if entry is None:
                    continue
                faces.append({"label": entry.get("name", eid), "icon": "plugin",
                              "activate": lambda e=entry: self._guarded(self._show_face, e)})
            kids.append({"label": f"{rec['name']} ({kind_title(rec['kind']).lower()})", "icon": "plugin",
                         "children": faces})
        return [{"label": t(*NAME), "icon": "plugin", "children": kids}]

    # ------------------------------------------------------------------ the command line
    @staticmethod
    def _classified_command(chunk, ns) -> bool:
        """``ppg plugin masonry classified``: the drawing of the face classified, the courses and stones printed and
        the plate written; with ``--accept`` the stones added to the face. Returns whether the project changed."""
        from . import classified, walls

        classified.need_program()
        rec = walls.need(chunk, ns.wall or "")
        _w, e = walls.need_face(chunk, rec["id"], ns.face)
        try:
            params = parse_params(ns.param)
        except ValueError as exc:
            raise walls.NotReady(str(exc)) from None
        for key, value in (("joint_cover", ns.joint_cover), ("min_stone", ns.min_stone),
                           ("pair_isolation", ns.pair_isolation)):
            if value is not None:
                params[key] = value
        if ns.with_units:
            params["units"] = True
        prop = classified.propose(chunk, e["id"], ns.file, params)
        print(f"{rec['name']}: {e['name']} ({e['id']})")
        print(classified.summary_text(prop))
        if ns.plate:
            path = classified.plate(chunk, e["id"], prop, ns.plate)
            print(t(f"Tavola di controllo: {path}", f"Check plate: {path}"))
        if ns.accept is None:
            print(t("Niente salvato: --accept aggiunge le pietre alla faccia.",
                    "Nothing saved: --accept adds the stones to the face."))
            return False
        before = classified.stored(chunk, e["id"])
        added = classified.accept(chunk, e["id"], prop, replace=ns.replace)
        print(classified.accepted_text(added, len(prop["already"])))
        return classified.stored(chunk, e["id"]) != before

    def command(self, *args) -> int:
        from . import faces, shape, walls

        parser = argparse.ArgumentParser(prog="ppg plugin masonry", description="walls by their faces")
        parser.add_argument("what", choices=["face", "shape", "stones", "classified", "usm", "plate", "export", "ifc",
                                             "list", "remove"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("wall", nargs="?", help="the id or the name of a wall (face: made when missing)")
        parser.add_argument("face", nargs="?", help="shape, remove: the id or the name of a face (elevation)")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--kind", default="wall", choices=list(walls.KINDS), help="face: the kind of a new wall")
        parser.add_argument("--name", help="face: the name of the face (elevation)")
        parser.add_argument("--line", help="face on a plane: x1,y1,x2,y2 world, the trace of the face")
        parser.add_argument("--side", help="face on a plane: x,y world, a point on the viewer's side")
        parser.add_argument("--circle", help="face on a cylinder: x1,y1,x2,y2,x3,y3[,...] world, points of the rim")
        parser.add_argument("--slice", type=float, help="face on a cylinder: the circle of the slice at this height")
        parser.add_argument("--inside", action="store_true", help="face on a cylinder: seen from inside")
        parser.add_argument("--polygon", type=float, help="faces of a polygonal tower: the slice at this height")
        parser.add_argument("--min-side", type=float, default=0.5, help="polygonal tower: shortest side (m)")
        parser.add_argument("--tolerance", type=float, help="polygonal tower: simplification of the slice (m)")
        parser.add_argument("--resolution", type=float, help="face: metres per pixel (default the model's median "
                                                             "edge, at least 1 mm)")
        parser.add_argument("--force", action="store_true", help="face: build a face over 20000 pixels on a side")
        parser.add_argument("--existing", help="face: the id or the name of an elevation the program already has, "
                                               "made a face as it is (nothing rebuilt)")
        parser.add_argument("--front", type=float, help="face: band in front (m; default 0.5, on a cylinder 15 %% of "
                                                         "the radius, at least 0.5)")
        parser.add_argument("--back", type=float, help="face: band behind (m; default 0.3, on a cylinder 15 %% of the "
                                                        "radius, at least 0.5)")
        parser.add_argument("--sections", help="shape: heights of horizontal sections, comma separated")
        parser.add_argument("--stations", help="shape: stations along the face (u, m), comma separated")
        parser.add_argument("--reach", type=float, help="shape: largest distance from the face (m; default the "
                                                         "face's depth band, 1 m at least)")
        parser.add_argument("--axis", action="store_true", help="shape: the axis of a round tower")
        parser.add_argument("--heights", help="shape --axis: heights of the circles, comma separated")
        parser.add_argument("--compare", help="shape: the label of another chunk (survey) to compare with")
        parser.add_argument("--elevations", action="store_true", help="remove: the program's elevations too")
        parser.add_argument("--region", help="stones: the wall's area on the face, WKT POLYGON in (u v)")
        parser.add_argument("--export", help="stones: a folder for the face's drawings, measures and tables")
        parser.add_argument("--accept", nargs="?", const=BARE_ACCEPT,
                            help="usm: the proposals to accept, 'all' or their numbers (1,3), a value is required; "
                                 "classified: add the stones proposed to the face, no value (without it nothing is "
                                 "saved)")
        parser.add_argument("--file", help="classified: a GeoPackage, GeoJSON or DXF in the face's (u, v) instead of "
                                           "the lines drawn on the face")
        parser.add_argument("--with-units", action="store_true", help="classified: the outlines of the face's USM "
                                                                      "read as lines too")
        parser.add_argument("--joint-cover", type=float, help="classified: a joint splits a course when it covers "
                                                              "this share of its height (default 0.5)")
        parser.add_argument("--min-stone", type=float, help="classified: cells narrower or lower than this (m) are "
                                                            "not stones (default 0.05)")
        parser.add_argument("--pair-isolation", type=float, help="classified: the pair isolation of the "
                                                                 "classification (default 1.5; 0 = off)")
        parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                            help="classified: a parameter of the classification (orient_lines), repeatable")
        parser.add_argument("--plate", help="classified: the check plate, a PNG or SVG file")
        parser.add_argument("--replace", action="store_true", help="classified --accept: the stones proposed in "
                                                                   "place of those of the face")
        parser.add_argument("--write", action="store_true", help="usm: write the accepted ones in PyArchInit")
        parser.add_argument("--config", help="usm --write: PyArchInit's config.cfg (default: PYARCHINIT_HOME's)")
        parser.add_argument("--db", help="usm --write: an SQLite path or an SQLAlchemy URL instead of the config")
        parser.add_argument("--media-root", help="usm --write: where media and previews go")
        parser.add_argument("--sito", help="usm --write: the site (default: the units' own, else the project's)")
        parser.add_argument("--area", help="usm --write: the area of units without one (default 1)")
        parser.add_argument("--out", help="plate, export: a folder; ifc: the .ifc file")
        parser.add_argument("--scale", type=int, help="plate, export: 1:SCALE (default the largest that fits A3)")
        parser.add_argument("--no-contours", action="store_true", help="plate: without the bulge's contours")
        parser.add_argument("--thickness", type=float, help="ifc: the wall's thickness when one face is surveyed")
        parser.add_argument("--json", action="store_true", help="list: as JSON")
        ns = parser.parse_args(joined_values([str(a) for a in args]))
        if ns.what == "usm" and ns.accept == BARE_ACCEPT:  # as before 0.1.3: never every proposal by a bare flag
            parser.error("usm --accept: give 'all' or the numbers of the proposals to accept (1,3)")
        if ns.what == "classified" and ns.accept not in (None, BARE_ACCEPT, "all"):
            parser.error(f"classified --accept takes no value (it adds every stone proposed), not {ns.accept!r}")
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        walls.prune(chunk)
        changed = False
        new_wall = None
        try:
            if ns.what == "face":
                rec = walls.find(chunk, ns.wall) if ns.wall else None
                if rec is None:  # made with the face: a face refused leaves no empty wall
                    rec = walls.new_wall(chunk, ns.wall, ns.kind)
                    new_wall = rec["id"]
                made = []
                front_plane = faces.DEFAULT_FRONT if ns.front is None else ns.front
                back_plane = faces.DEFAULT_BACK if ns.back is None else ns.back
                if ns.existing:
                    made.append(walls.adopt_face(chunk, rec["id"], ns.existing))
                elif ns.line:
                    x1, y1, x2, y2 = parse_numbers(ns.line)
                    side = parse_numbers(ns.side) if ns.side else None
                    made.append(faces.plane_face(chunk, rec["id"], (x1, y1), (x2, y2), side, ns.resolution,
                                                 front_plane, back_plane, ns.name, progress=print_progress,
                                                 force=ns.force))
                elif ns.circle or ns.slice is not None:
                    pts = np.asarray(parse_numbers(ns.circle)).reshape(-1, 2) if ns.circle else None
                    made.append(faces.cylinder_face(chunk, rec["id"], pts, ns.slice, ns.inside, ns.resolution,
                                                    ns.front, ns.back, ns.name, progress=print_progress,
                                                    force=ns.force))
                elif ns.polygon is not None:
                    made += faces.polygon_faces(chunk, rec["id"], ns.polygon, ns.min_side, ns.tolerance,
                                                ns.resolution, front_plane, back_plane, progress=print_progress,
                                                force=ns.force)
                else:
                    raise walls.NotReady("--line, --circle, --slice, --polygon or --existing")
                changed = True
                for e in made:
                    info = walls.face_info(chunk, e["id"])
                    print(f"{rec['name']}: {e['name']} ({e['id']}) {metres(info['width'])} x {metres(info['height'])}"
                          f", {num(info['resolution'] * 1000, 1)} mm/px")
            elif ns.what == "shape":
                rec = walls.need(chunk, ns.wall or "")
                if ns.axis:
                    a = shape.tower_axis(chunk, rec["id"], parse_numbers(ns.heights) if ns.heights else None,
                                         progress=print_progress)
                    print(shape.axis_text(a))
                    changed = True
                if ns.face:
                    targets = [walls.need_face(chunk, rec["id"], ns.face)[1]]
                elif ns.axis and not (ns.sections or ns.stations or ns.compare):
                    targets = []
                else:
                    targets = [e for e in (walls.elevation(chunk, f) for f in rec["faces"]) if e is not None]
                other = None
                if ns.compare:
                    other = next((c for c in doc.chunks if c.label == ns.compare), None)
                    if other is None:
                        raise walls.NotReady(f"no chunk {ns.compare!r} in the project")
                for e in targets:
                    print(f"{e['name']} ({e['id']}):")
                    if ns.sections or ns.stations:
                        r = shape.sections(chunk, rec["id"], e["id"], parse_numbers(ns.sections or "") or None,
                                           parse_numbers(ns.stations or "") or None, ns.reach, progress=print_progress)
                        print(shape.sections_text(r))
                        print(f"DXF {chunk.data_dir / r['dxf']}\nSVG {chunk.data_dir / r['svg']}")
                    elif other is not None:
                        print(shape.compare_text(shape.compare_surveys(chunk, rec["id"], e["id"], other,
                                                                       progress=print_progress)))
                    else:
                        r = shape.face_shape(chunk, rec["id"], e["id"], progress=print_progress)
                        print(shape.summary_text(r))
                        print(f"{t('Carta dello spanciamento', 'Bulge map')}: {chunk.data_dir / r['bulge']}")
                    changed = True
            elif ns.what == "list":
                records = walls.walls(chunk)
                if ns.json:
                    print(json.dumps([{**w, "faces": [walls.face_info(chunk, f) for f in w["faces"]]}
                                      for w in records], indent=1, ensure_ascii=False))
                for w in [] if ns.json else records:
                    print(f"{w['id']}  {w['name']} ({kind_title(w['kind']).lower()})")
                    for eid in w["faces"]:
                        info = walls.face_info(chunk, eid)
                        res = walls.results(chunk, w["id"], eid).get("shape") or {}
                        print(f"  {eid}  {info['name']}: {metres(info['width'])} x {metres(info['height'])}")
                        if res:
                            print("    " + shape.summary_text(res).replace("\n", "\n    "))
                if not records and not ns.json:
                    print(t("Nessun muro nel chunk.", "No wall in the chunk."))
            elif ns.what == "stones":
                from shapely import wkt as swkt

                from . import stones

                rec = walls.need(chunk, ns.wall or "")
                _w, e = walls.need_face(chunk, rec["id"], ns.face)
                r = stones.measure_face(chunk, rec["id"], e["id"], swkt.loads(ns.region) if ns.region else None,
                                        progress=print_progress)
                print(stones.stones_text(walls.results(chunk, rec["id"], e["id"])["stones"]))
                for key, path in r["files"].items():
                    print(f"{key}: {chunk.data_dir / path}")
                if ns.export:
                    for key, path in stones.export_face(chunk, rec["id"], e["id"], ns.export).items():
                        print(f"{key}: {path}")
                changed = True
            elif ns.what == "classified":
                changed = self._classified_command(chunk, ns)
            elif ns.what == "usm":
                from . import usm

                rec = walls.need(chunk, ns.wall or "")
                _w, e = walls.need_face(chunk, rec["id"], ns.face)
                proposals = usm.propose(chunk, rec["id"], e["id"])
                for i, p in enumerate(proposals, 1):
                    print(f"{i}. {usm.proposal_text(p)}")
                if not proposals:
                    print(t("Nessun rapporto proposto.", "No relationship proposed."))
                accepted = []
                if ns.accept:
                    picks = range(1, len(proposals) + 1) if ns.accept == "all" else \
                        [int(x) for x in ns.accept.split(",") if x.strip()]
                    accepted = [proposals[i - 1] for i in picks if 1 <= i <= len(proposals)]
                    done = dict(walls.results(chunk, rec["id"], e["id"]).get("usm") or {})
                    done.update(accepted_full=accepted, proposed=len(proposals))
                    walls.set_result(chunk, rec["id"], e["id"], "usm", done)
                    changed = True
                    print(t(f"Accettati: {len(accepted)}", f"Accepted: {len(accepted)}"))
                if ns.write:
                    from pyarchinit_photogrammetry.pyarchinit import PyArchInit

                    pa = PyArchInit.connect(ns.db, ns.config, media_root=ns.media_root)
                    if not accepted:
                        accepted = (walls.results(chunk, rec["id"], e["id"]).get("usm") or {}).get(
                            "accepted_full") or []
                    r = usm.save(pa, chunk, rec["id"], e["id"], accepted, sito=ns.sito, area=ns.area)
                    print(f"PyArchInit: {len(r['created'])} forms, {r['relations']} relationships, "
                          f"{len(r['media'])} media")
                    changed = True
            elif ns.what in ("plate", "export"):
                from . import outputs

                rec = walls.need(chunk, ns.wall or "")
                _w, e = walls.need_face(chunk, rec["id"], ns.face)
                if ns.what == "plate":
                    out = outputs.plate(chunk, rec["id"], e["id"], ns.out, ns.scale, not ns.no_contours)
                else:
                    if not ns.out:
                        raise walls.NotReady("--out FOLDER")
                    out = outputs.export_face(chunk, rec["id"], e["id"], ns.out, ns.scale)
                for key, path in out.items():
                    print(f"{key}: {path}")
            elif ns.what == "ifc":
                from . import ifc

                rec = walls.need(chunk, ns.wall or "")
                if not ns.out:
                    raise walls.NotReady("--out FILE.ifc")
                r = ifc.export_wall(chunk, rec["id"], ns.out, ns.thickness)
                print(f"IFC {r['path']}: {r['thickness']:.3f} m ({r['source']}), {r['parts']} USM, "
                      f"{r['relations']} relationships")
            elif ns.what == "remove":
                rec = walls.need(chunk, ns.wall or "")
                if ns.face:
                    _w, e = walls.need_face(chunk, rec["id"], ns.face)
                    lost = walls.drawings_lost_text(chunk, [e["id"]], True) if ns.elevations else ""
                    walls.remove_face(chunk, rec["id"], e["id"], elevation_too=ns.elevations)
                else:
                    lost = walls.drawings_lost_text(chunk, rec["faces"], True) if ns.elevations else ""
                    walls.remove_wall(chunk, rec["id"], elevations_too=ns.elevations)
                if lost:
                    print(lost.strip())
                changed = True
        except walls.NotReady as exc:
            print(exc)
            if new_wall:
                walls.remove_wall(chunk, new_wall)
            if changed:
                doc.save()
            return 1
        if changed:
            doc.save()
        return 0


def parse_params(items: list[str]) -> dict:
    """``KEY=VALUE`` of the command line: numbers, ``true``/``false`` as booleans."""
    out = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--param {item!r}: KEY=VALUE")
        value = value.strip()
        if value.lower() in ("true", "false"):
            out[key.strip()] = value.lower() == "true"
        else:
            out[key.strip()] = float(value.replace(",", "."))
    return out


def parse_numbers(text: str) -> list[float]:
    """Numbers of the command line: ``1.5,2.5`` or ``1.5 2.5``."""
    return [float(p) for p in (text or "").replace(",", " ").split() if p]


def parse_typed(text: str) -> list[float]:
    """Numbers typed in a dialog, separated by semicolons or spaces, each as the interface writes it: ``313,500`` or
    ``1.234,5`` in Italian, ``313.500`` or ``1,234.5`` in English."""
    out = []
    for p in (text or "").replace(";", " ").split():
        if "," in p and "." in p:
            p = p.replace(".", "").replace(",", ".") if p.rfind(",") > p.rfind(".") else p.replace(",", "")
        else:
            p = p.replace(",", ".")
        out.append(float(p))
    return out


def joined_values(args: list[str]) -> list[str]:
    """``--line -0.5,...`` as ``--line=-0.5,...``: argparse takes for an option anything that begins with a minus and
    is not one negative number."""
    out: list[str] = []
    for a in args:
        if out and out[-1] in NEGATIVE_VALUES and a[:1] == "-" and a[1:2] in tuple("0123456789."):
            out[-1] = f"{out[-1]}={a}"
        else:
            out.append(a)
    return out
