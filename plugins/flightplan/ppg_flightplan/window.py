"""The plugin in the program's window: the *Piani di volo* pane answered, the plan drawn on the 2D view, the
coverage shown, the plans in the project's tree, the exports and the copy into DJI Fly.

:class:`WindowPart` is a part of :class:`ppg_flightplan.FlightPlanPlugin`; it uses only the public plugin API
(``context.run``, ``request_polygon``/``request_point``, ``show_raster``, ``show_frame``, ``mark_modified``,
``refresh_tree``...) and what every plugin of the repository uses of the window (``ortho_view`` layers,
``qgis_providers``, ``runner.busy_changed``, ``gui.dialogs``).

Threads. Long steps (design, terrain, coverage, exports, DJI Fly) run in ``context.run``: the worker computes on
a copy of the plan, and the plan goes into ``chunk.meta`` in ``done``, in the interface thread, followed by
``mark_modified`` - so Modifica > Annulla takes the change back. Derived records the worker writes itself (the
coverage, the terrain refreshed by an export) are followed by ``mark_modified`` in ``done`` too, also when the
step failed after changing the plan.
"""
from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path

from .texts import (LAYER_AREA, LAYER_LOW, LAYER_MISSING, LAYER_SHOTS, LAYER_STRIPS, LAYER_TAKEOFF, LAYER_TARGETS,
                    LAYERS, NAME, format_name, num, plural, t, warning_text)

MAX_SHOTS_DRAWN = 3000          # above this the shots layer shows every n-th one (said in the status)
NO_NETWORK = ("Le tegole pubbliche delle quote arrivano dalla rete: senza connessione (o fuori dalla zona coperta) "
              "non ci sono. Riprova con la rete, o usa il DEM del progetto.",
              "The public height tiles come from the network: without a connection (or outside the area covered) "
              "there are none. Try again with the network, or use the project DEM.")


class _Busy:
    """The copy into DJI Fly found missions of the user in the slots: nothing written, a question to ask."""

    def __init__(self, error: Exception):
        self.error = error


class _Failed:
    """A step that stopped with a message for the user (not a traceback)."""

    def __init__(self, error: Exception, source: str = ""):
        self.error, self.source = error, source


def _expected_errors() -> tuple:
    from . import coverage, design, djifly, targets, tiles
    from .camera import CameraUnknown
    from .dem import DemError

    return (design.StaleHeights, tiles.TerrainError, coverage.CoverageError, targets.TargetError, CameraUnknown,
            djifly.DjiFlyError, DemError, KeyError, ValueError, OSError)


def _icon(name: str = "mdi6.quadcopter", fallback: str = "globe"):
    """The plugin's icon: a drone of qtawesome when there is one, else the name of an icon of the program."""
    try:
        import qtawesome

        return qtawesome.icon(name)
    except Exception:  # noqa: BLE001 - no qtawesome, or the name unknown: the program's own icon
        return fallback


class WindowPart:
    """The window side of the plugin (mixed into :class:`ppg_flightplan.FlightPlanPlugin`)."""

    pane = None
    context = None

    # ------------------------------------------------------------------ set up
    def _window_init(self) -> None:
        self.area: list | None = None         # the polygon the user gave (WGS84), kept as drawn
        self.area_source = ""
        self.new_area = False                 # the next design makes a new plan (a new area was given)
        self.own_cameras: dict = {}           # id -> Camera (EXIF, chunk sensor, typed)
        self._reasons: dict = {}              # plan id -> the design's reasons
        self._warnings: dict = {}             # (plan id, stamp) -> warnings in words (design, terrain)
        self._summaries: dict = {}            # (plan id, stamp) -> the table of the plan
        self._low_cache: dict = {}            # (plan id, coverage files) -> low-coverage polygons
        self._shown_plan = None               # the plan whose fields the pane shows
        self._coverage_shown = False
        self._doc_seen = None
        self._status = ""

    def _add_actions(self, context) -> list:
        actions = [
            context.add_action(t("Pannello Piani di volo", "Flight plans pane"), self.show_pane,
                               tip=t("Il pannello dei piani di volo", "The pane of the flight plans")),
            context.add_action(t("Disegna l'area del volo", "Draw the flight area"), self.draw_area,
                               tip=t("Il poligono dell'area sulla vista 2D", "The polygon of the area on the 2D view")),
            context.add_action(t("Progetta il volo...", "Design the flight..."), self.design_dialog,
                               tip=t("Il piano per l'area con i parametri del pannello",
                                     "The plan for the area with the parameters of the pane")),
            context.add_action(t("Copertura prevista", "Predicted coverage"), self.compute_coverage,
                               tip=t("Le foto che vedranno ogni cella del DEM", "The photos that will see every "
                                     "cell of the DEM")),
            context.add_action(t("Esporta il piano di volo...", "Export the flight plan..."), self.export_dialog,
                               tip=t("Il piano nei file che leggono i droni", "The plan in the files the drones read")),
            context.add_action(t("Copia in DJI Fly...", "Copy into DJI Fly..."), self.copy_djifly,
                               tip=t("Le tratte nelle missioni fittizie di DJI Fly", "The legs into DJI Fly's "
                                     "placeholder missions")),
            context.add_action(t("Confronta con il volo", "Compare with the flight"), self.compare_flight,
                               tip=t("Gli scatti previsti contro le foto del volo nel chunk",
                                     "The planned shots against the photos of the flight in the chunk")),
        ]
        return actions

    def _attach_window(self, context, actions: list) -> None:
        from . import project
        from .pane import FlightPlanPane

        window = context.window
        if hasattr(context, "add_rail_group"):          # program 1.7.4: a group of its own in the tool rail
            context.add_rail_group("flightplan", t(*NAME), icon=_icon(), actions=actions)
        self.pane = p = FlightPlanPane()
        context.add_pane(p, t(*NAME), _icon(), area="right")
        p.area_draw_requested.connect(self.draw_area)
        p.area_layer_requested.connect(self.area_from_layer)
        p.area_file_requested.connect(self.area_from_file)
        p.area_coords_requested.connect(self.area_from_coordinates)
        p.camera_changed.connect(self.refresh_pane)
        p.camera_exif_requested.connect(self.camera_from_exif)
        p.camera_sensor_requested.connect(self.camera_from_chunk)
        p.camera_manual_requested.connect(self.camera_by_hand)
        p.design_requested.connect(self.design)
        p.takeoff_requested.connect(self.place_takeoff)
        p.block_toggled.connect(self.toggle_block)
        p.terrain_requested.connect(self.follow_terrain)
        p.terrain_removed.connect(self.remove_terrain)
        p.coverage_requested.connect(self.compute_coverage)
        p.coverage_shown.connect(self.show_coverage)
        p.targets_csv_requested.connect(self.targets_from_csv)
        p.targets_markers_requested.connect(self.targets_from_markers)
        p.targets_removed.connect(self.remove_targets)
        p.export_requested.connect(self.export)
        p.djifly_requested.connect(self.copy_djifly)
        p.qgis_requested.connect(self.open_in_qgis)
        p.plan_selected.connect(self.use_plan)
        p.plan_removed.connect(self.remove_plan)
        p.compare_requested.connect(self.compare_flight)
        p.obtained_shown.connect(self.show_obtained)
        context.on_refresh(self.refresh_pane)
        context.on_raster_loaded(self.draw_layers)
        if hasattr(context, "add_tree_items"):
            context.add_tree_items(self._tree_items)
        if hasattr(context, "add_rasters"):              # program 1.4.0: the coverage in the raster calculator
            context.add_rasters(project.raster_entries)
        window.qgis_providers = list(getattr(window, "qgis_providers", [])) + [self._qgis_entries]
        runner = getattr(window, "runner", None)
        if runner is not None and hasattr(runner, "busy_changed"):
            runner.busy_changed.connect(p.set_busy)
            context.contributions.connections.append((runner.busy_changed, p.set_busy))
        self.refresh_pane()

    def _detach_window(self) -> None:
        window = self.context.window if self.context is not None else None
        if window is None:
            return
        window.qgis_providers = [q for q in getattr(window, "qgis_providers", []) if q != self._qgis_entries]
        view = getattr(window, "ortho_view", None)
        if view is not None:
            for name in LAYERS:
                view.remove_layer(name)
        self.pane = None

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _window(self):
        return self.context.window if self.context is not None else None

    def _chunk(self):
        window = self._window()
        chunk = getattr(window, "chunk", None) if window is not None else None
        if chunk is not None:
            return chunk
        doc = getattr(self.context.app, "document", None) if self.context is not None else None
        return None if doc is None else doc.chunk

    def _message(self, kind: str, text: str) -> None:
        window = self._window()
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui import dialogs

        dialogs.message(window, kind, t(*NAME), text)

    def _choice(self, text: str, options: list[str], kind: str = "question") -> int | None:
        from pyarchinit_photogrammetry.gui import dialogs

        return dialogs.choice(self._window(), kind, t(*NAME), text, options, cancel=t("Annulla", "Cancel"))

    def _ask(self, title: str, fields: list, intro: str = "") -> dict | None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        return ParamDialog.ask(self._window(), title, fields, intro=intro)

    def _open_path(self, title: str, filter_: str) -> str:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self._window(), title, "", filter_)
        return path

    def _save_path(self, title: str, name: str, filter_: str) -> str:
        from PySide6.QtWidgets import QFileDialog

        doc = getattr(self.context.app, "document", None)
        folder = Path(doc.path).parent if doc is not None and getattr(doc, "path", None) else Path.home()
        path, _ = QFileDialog.getSaveFileName(self._window(), title, str(folder / name), filter_)
        return path

    def _folder_path(self, title: str, start: str = "") -> str:
        from PySide6.QtWidgets import QFileDialog

        return QFileDialog.getExistingDirectory(self._window(), title, start)

    def _status_text(self, text: str) -> None:
        self._status = text
        if self.pane is not None:
            self.pane.set_status(text)
        window = self._window()
        if window is not None and text:
            window.statusBar().showMessage(text, 15000)

    def _need_chunk(self):
        chunk = self._chunk()
        if chunk is None:
            raise ValueError(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _need_plan(self, chunk=None):
        from . import project

        chunk = chunk if chunk is not None else self._need_chunk()
        e = project.active_plan(chunk)
        if e is None:
            raise ValueError(t("Il chunk non ha ancora un piano di volo: dai l'area e premi Progetta.",
                               "The chunk has no flight plan yet: give the area and press Design."))
        return chunk, e

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        try:
            function(*args)
        except _expected_errors() as exc:
            self._say_error(exc)

    def _say_error(self, exc: Exception, source: str = "") -> None:
        from . import tiles

        text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        if isinstance(exc, tiles.TerrainError) and source != "dem":
            text = f"{text}\n\n{t(*NO_NETWORK)}"
        self._message("warning", text)

    def _changed(self) -> None:
        """The plan in the meta changed in the interface thread: the project modified (an entry of Annulla), the
        pane, the tree and the layers again."""
        self.context.mark_modified()
        self.refresh_pane()
        self.context.refresh_tree()

    def _task(self, title: str, job, done=None, *, chunk=None, plan_id: str | None = None, source: str = "") -> None:
        """``job(progress)`` in the background; ``done(result)`` in the interface thread. A message for the
        expected errors; the project marked modified when the plan's stamp changed during the step - also when
        it failed after changing it (an export refreshing the terrain, then not able to write)."""
        from . import project

        before = (project.entry(chunk, plan_id) or {}).get("stamp") if chunk is not None and plan_id else None
        errors = _expected_errors()

        def work(progress):
            try:
                return job(progress)
            except errors as exc:
                return _Failed(exc, source)

        def finished(result):
            if chunk is not None and plan_id:
                after = (project.entry(chunk, plan_id) or {}).get("stamp")
                if after is not None and after != before:
                    self.context.mark_modified()
            if isinstance(result, _Failed):
                self._say_error(result.error, result.source)
            elif done is not None:
                try:
                    done(result)
                except errors as exc:
                    self._say_error(exc, source)
            self.refresh_pane()
            self.context.refresh_tree()

        self.context.run(title, work, finished)

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self) -> None:
        """The pane, the layers and the files of the plans as the active chunk says (after an undo, another
        project, another chunk)."""
        from . import project

        self._purge_on_open()
        chunk = self._chunk()
        if chunk is not None and project.folder(chunk).is_dir():
            project.restore_files(chunk)            # the files follow the meta (an undo, a redo)
        if self.pane is None:
            return
        e = project.active_plan(chunk) if chunk is not None else None
        self.pane.set_plans([(p["id"], p["name"]) for p in project.plans(chunk)] if chunk is not None else [],
                            e["id"] if e else None)
        if e is not None and self._shown_plan != (id(chunk), e["id"]):
            self._shown_plan = (id(chunk), e["id"])
            piano = e["piano"]
            own = piano.get("cameraPersonale")
            if own:
                from .camera import Camera

                try:
                    cam = Camera.from_json(own)
                    self.own_cameras[cam.id] = cam
                except ValueError:
                    pass
            self._fill_cameras(current=piano.get("camera"))
            self.pane.set_parameters(piano)
            drawn = piano.get("areaDisegnata")
            first = next((b for b in piano.get("blocchi") or [] if b.get("tipo", "griglia") == "griglia"), None)
            self.area = [tuple(q) for q in drawn] if drawn else \
                ([tuple(q) for q in first.get("area") or []] if first else None)
            self.area_source = t("dal piano", "from the plan")
            self.new_area = False
        elif e is None and self._shown_plan is not None and not self.new_area:
            self._shown_plan = None
            self.area = None
        self.pane.set_state(self._states(chunk, e), self._summary(e), self._warning_texts(chunk, e))
        if self._status:
            self.pane.set_status(self._status)
        if not self._drop_stale_coverage(chunk, e):  # (it draws the layers itself, on the raster it shows)
            self.draw_layers()

    def _drop_stale_coverage(self, chunk, e) -> bool:
        """The coverage raster on the 2D view is no longer the active plan's current one (the plan changed, another
        plan is active, the plan was removed or undone): back to the orthophoto (or the DEM, or a frame), said in
        the status line - the map the user plans on must be the plan's. True when the view was changed."""
        from . import project

        view = self._view()
        shown = getattr(view, "path", None) if view is not None else None
        if chunk is None or not shown:
            return False
        base = project.folder(chunk).resolve()
        try:
            Path(shown).resolve().relative_to(base)
        except (ValueError, OSError):
            return False                                   # not a raster of the plugin
        rec = project.coverage_record(chunk, e["id"]) if e is not None else None
        if rec is not None and Path(shown).resolve() == (Path(chunk.data_dir) / rec["files"]["rgba"]).resolve():
            return False
        obtained = self._obtained_path(chunk, e) if e is not None else None
        if obtained is not None and Path(shown).resolve() == obtained.resolve():
            return False
        self._hide_coverage(chunk)
        self._status_text(t("La copertura mostrata era di un piano cambiato o tolto: rifalla con Calcola.",
                            "The coverage shown belonged to a plan changed or removed: make it again with "
                            "Compute."))
        return True

    def _purge_on_open(self) -> None:
        """M3: a project just opened (its meta is the project as saved, and no undo can reach before it): the
        folders of removed plans in ``flightplan/.removed`` are deleted. A project already open when the plugin
        was switched on is left alone while its undo has steps (one could bring a plan back)."""
        from . import project

        window = self._window()
        doc = getattr(window, "doc", None) if window is not None else None
        if doc is None or doc is self._doc_seen:
            return
        self._doc_seen = doc
        history = getattr(window, "history", None)
        if history is not None and history.can_undo():
            return
        for chunk in getattr(doc, "chunks", []):
            try:
                if project.folder(chunk).is_dir():
                    project.restore_files(chunk)       # stray folders aside first, so they go at once
                project.purge_removed(chunk)
            except OSError:
                pass

    def _fill_cameras(self, current: str | None = None) -> None:
        if self.pane is not None:
            self.pane.fill_cameras([(c.id, c.name) for c in self.own_cameras.values()], current)

    def _pane_camera(self):
        """(camera of the list's id, own camera or None) chosen in the pane."""
        id_ = self.pane.camera_id() if self.pane is not None else "mini4pro"
        if id_ in self.own_cameras:
            return id_, self.own_cameras[id_]
        return id_ or "mini4pro", None

    def _summary(self, e) -> dict | None:
        if e is None:
            return None
        key = (e["id"], e["stamp"])
        if key in self._summaries:
            return self._summaries[key]
        from . import plan as pl

        piano = e["piano"]
        counts = pl.compute(piano)
        legs = pl.plan_legs(piano)
        blocks = []
        for i, (b, c) in enumerate(zip(piano.get("blocchi") or [], counts["blocchi"])):
            def heights(lo, hi):
                return f"{lo:g}" if abs(hi - lo) < 0.05 else f"{lo:g}–{hi:g}"

            row = {"indice": i, "nome": b.get("nome") or c.get("nome") or str(i + 1), "attivo": pl.active(b),
                   "scatti": c["scatti"], "minuti": c["minuti"], "batterie": c["batterie"], "gsd": c["gsdCm"],
                   "quote": heights(float(c["quotaMin"]), float(c["quotaMax"])), "tratte": []}
            for leg in (g for g in legs if g.block_index == i):
                r = leg.route
                q = r.heights or [r.height]
                row["tratte"].append({"chiave": leg.key, "nome": leg.file_name, "scatti": r.shots(),
                                      "minuti": round(r.minutes(), 1),
                                      "quote": heights(round(min(q), 1), round(max(q), 1))})
            blocks.append(row)
        codes: list[str] = []
        for c in counts["blocchi"]:
            if c.get("attivo"):
                codes += [w for w in c.get("avvisi") or [] if w not in codes]
        out = {"blocchi": blocks, "totale": counts["totale"], "decolloLontanoM": counts.get("decolloLontanoM"),
               "codici": codes}
        self._summaries = {k: v for k, v in self._summaries.items() if k[0] != e["id"]}
        self._summaries[key] = out
        return out

    def _warning_texts(self, chunk, e) -> list[str]:
        if e is None:
            return []
        from .export_formats import terrain_note

        summary = self._summary(e) or {}
        texts: list[str] = [warning_text(c) for c in summary.get("codici") or []]
        texts += [w for w in self._warnings.get((e["id"], e["stamp"]), []) if w not in texts]
        far = summary.get("decolloLontanoM")
        if far:
            texts.append(t("Il decollo è a {0} m dall'area: oltre i 2 km le quote non si riferiscono a lui",
                           "The takeoff is {0} m from the area: beyond 2 km the heights do not refer to it")
                         .format(far))
        note = terrain_note(e["piano"])
        if note:
            texts.append(note[:1].upper() + note[1:])
        return texts

    def _states(self, chunk, e) -> dict:
        from . import project

        st: dict[str, str] = {}
        local = chunk is not None and not project.has_crs(chunk)
        # area
        if self.area:
            n = len(self.area)
            st["area"] = t("Area: {0} vertici, {1} ({2}).", "Area: {0} vertices, {1} ({2}).") \
                .format(n, self._area_size(self.area), self.area_source or "-")
            if self.new_area and e is not None:
                st["area"] += " " + t("Area nuova: Progetta fa un piano nuovo.", "New area: Design makes a new plan.")
        else:
            st["area"] = t("Nessuna area: disegnala sulla vista, prendila da un layer, da un file o dalle coordinate.",
                           "No area: draw it on the view, take it from a layer, a file or the coordinates.")
        if local:
            st["area"] += " " + t("Il chunk è in un sistema locale: l'area solo da un file o da coordinate in gradi, "
                                  "e il piano non si disegna sulla vista.",
                                  "The chunk is in a local frame: the area only from a file or from coordinates in "
                                  "degrees, and the plan is not drawn on the view.")
        # camera
        try:
            from .camera import camera as list_camera

            cid, own = self._pane_camera()
            cam = own or list_camera(cid)
            st["camera"] = t("{0}: sensore {1} × {2} mm, focale {3} mm, {4} px",
                             "{0}: sensor {1} × {2} mm, focal length {3} mm, {4} px") \
                .format(cam.name, num(cam.width_mm, 3), num(cam.height_mm, 3), num(cam.focal_mm, 3),
                        cam.pixels_for(self.pane.k_high.isChecked()) if self.pane is not None else cam.pixels)
            if cam.app:
                st["camera"] += " — " + t("si vola con {0}", "flown with {0}").format(cam.app)
        except Exception:  # noqa: BLE001 - an unknown id: nothing to say
            st["camera"] = ""
        if e is None:
            st["params"] = t("Dai l'area e premi Progetta.", "Give the area and press Design.")
            return st
        piano = e["piano"]
        reasons = self._reasons.get(e["id"])
        st["params"] = " ".join(reasons) if reasons else t("Piano «{0}».", "Plan '{0}'.").format(e["name"])
        tot = (self._summary(e) or {}).get("totale") or {}
        st["plan"] = ", ".join((
            plural(tot.get("scatti", 0), t("{0} scatto", "{0} shot"), t("{0} scatti", "{0} shots")),
            t("{0} minuti", "{0} minutes").format(num(tot.get("minuti", 0), 1)),
            t("{0} batterie", "{0} batteries").format(num(tot.get("batterie", 0), 1)),
            f"{num(tot.get('km', 0), 1)} km",
            plural(tot.get("tratte", 0), t("{0} tratta", "{0} leg"), t("{0} tratte", "{0} legs")))) + "."
        # terrain
        st["terrain"] = self._terrain_state(chunk, piano)
        # coverage
        rec = project.coverage_record(chunk, e["id"]) if chunk is not None else None
        if local:
            st["coverage"] = t("Serve un sistema di riferimento proiettato: il chunk è in un sistema locale.",
                               "A projected reference system is needed: the chunk is in a local frame.")
        elif rec is None:
            stored = (project.peek(chunk).get("coverage") or {}).get(e["id"])
            st["coverage"] = t("Da rifare: il piano o il DEM sono cambiati.", "To do again: the plan or the DEM "
                               "changed.") if stored else t("Da calcolare.", "To compute.")
        else:
            s = rec.get("stats") or {}
            st["coverage"] = t("Foto per cella: mediana {0}, 10° percentile {1}; sotto {2}: {3}% dell'area; "
                               "GSD mediano {4} cm; cella {5} m; superficie {6}.",
                               "Photos per cell: median {0}, 10th percentile {1}; under {2}: {3}% of the "
                               "area; median GSD {4} cm; cell {5} m; surface {6}.") \
                .format(num(s.get("median", 0), 1), num(s.get("p10", 0), 1), rec.get("low"),
                        num(100 * float(s.get("low_fraction", 0)), 0),
                        num(s["gsd_median_cm"], 2) if s.get("gsd_median_cm") is not None else "-",
                        num(rec.get("cell", 0), 2),
                        "DEM" if rec.get("surface") == "dem" else t("piana al decollo", "flat at the takeoff"))
            st["coverage"] += " " + t("Senza occlusioni: muri e alberi non nascondono il suolo.",
                                      "No occlusion: walls and trees do not hide the ground.")
        # targets
        tg = piano.get("target") or {}
        points = tg.get("punti") or []
        if points:
            estimated = sum(1 for q in points if q.get("stimato"))
            st["targets"] = plural(len(points), t("{0} target", "{0} target"), t("{0} target", "{0} targets")) \
                + (f" ({tg.get('crs')})" if tg.get("crs") else "")
            if estimated:
                st["targets"] += "; " + t("{0} con posizione stimata dal rilievo: da sorvolare, non come GCP",
                                          "{0} with a position estimated by the survey: to fly over, not as GCPs") \
                    .format(estimated)
            others = [warning_text(c) for c in tg.get("avvisi") or [] if c != "ordine_incerto"]
            if others:
                st["targets"] += ". " + "; ".join(others)
            if "ordine_incerto" in (tg.get("avvisi") or []):
                st["target_warning"] = warning_text("ordine_incerto")
        else:
            st["targets"] = t("Nessun target.", "No target.")
        # export
        from .export_formats import terrain_note

        note = terrain_note(piano)
        st["export"] = t("Il piano «{0}» è pronto da esportare.", "The plan '{0}' is ready to export.") \
            .format(e["name"]) + (" " + note[:1].upper() + note[1:] + "." if note else "")
        st["after"], st["obtained"] = self._after_state(chunk, e)
        return st

    def _after_state(self, chunk, e) -> tuple[str, bool]:
        """The line of the step "Dopo il volo" and whether a coverage obtained can be shown."""
        from . import compare, project

        if chunk is None:
            return "", False
        rec = compare.comparison_record(chunk, e["id"])
        if rec is not None:
            return compare.summary_line(rec), self._obtained_path(chunk, e) is not None
        stored = (project.peek(chunk).get("comparison") or {}).get(e["id"])
        if not [c for c in chunk.cameras if getattr(c, "enabled", True)]:
            return t("Dopo il volo: aggiungi al chunk le foto del volo (File > Aggiungi foto), poi Confronta.",
                     "After the flight: add the photos of the flight to the chunk (File > Add photos), then "
                     "Compare."), False
        if stored:
            return t("Da rifare: il piano, le foto, l'allineamento o il DEM sono cambiati dopo il confronto.",
                     "To do again: the plan, the photos, the alignment or the DEM changed after the comparison."), \
                False
        aligned = bool(getattr(chunk, "aligned_cameras", None))
        return (t("Pronto: Confronta mette gli scatti previsti contro le foto del chunk",
                  "Ready: Compare puts the planned shots against the photos of the chunk")
                + (t(" e la copertura ottenuta contro quella prevista.",
                     " and the coverage obtained against the one predicted.") if aligned else
                   t("; allinea le foto per confrontare anche la copertura.",
                     "; align the photos to compare the coverage too."))), False

    def _obtained_path(self, chunk, e):
        """The colours of the coverage obtained by the current comparison of the plan ``e``, or None."""
        from . import compare

        rec = compare.comparison_record(chunk, e["id"])
        f = (rec or {}).get("files", {}).get("rgba")
        return Path(chunk.data_dir) / f if f else None

    def _terrain_state(self, chunk, piano: dict) -> str:
        dem_note = ""
        if chunk is not None and getattr(chunk, "dem_path", None) and Path(chunk.dem_path).is_file():
            if not self._dem_georeferenced(chunk):
                dem_note = t("Il DEM del progetto non dice dove si trova (nessuna georeferenza): si usano le tegole "
                             "pubbliche.", "The project DEM does not say where it is (no georeference): the public "
                             "tiles are used.")
        else:
            dem_note = t("Il chunk non ha un DEM: le quote vengono dalle tegole pubbliche (serve la rete).",
                         "The chunk has no DEM: the heights come from the public tiles (the network is needed).")
        if any(b.get("seguiTerreno") for b in piano.get("blocchi") or []):
            source = (piano.get("fonteQuote") or {}).get("sorgente")
            name = t("DEM del progetto", "project DEM") if source == "dem" else t("tegole pubbliche", "public tiles")
            from .design import blocks_with_old_heights

            stale = blocks_with_old_heights(piano)
            ground = piano.get("quotaDecollo")
            text = t("Il volo segue il terreno ({0}), decollo a {1} m.", "The flight follows the terrain ({0}), "
                     "takeoff at {1} m.").format(name, num(ground, 1) if ground is not None else "-")
            if stale:
                text += " " + t("Alcuni blocchi sono cambiati dopo il calcolo: le quote si rifanno all'esportazione.",
                                "Some blocks changed after the computation: the heights are made again at the "
                                "export.")
            return f"{text} {dem_note}".strip()
        return (t("Una quota per blocco.", "One height per block.") + " " + dem_note).strip()

    def _dem_georeferenced(self, chunk) -> bool:
        """Whether the chunk's DEM says where it is, read once per file (name, size, time) and not at every
        refresh."""
        from . import project

        path = Path(chunk.dem_path)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime)
        cache = getattr(self, "_dem_checked", None)
        if cache is None or cache[0] != key:
            dem = project.chunk_dem(chunk)
            if dem is not None:
                dem.close()
            self._dem_checked = cache = (key, dem is not None)
        return cache[1]

    @staticmethod
    def _area_size(area) -> str:
        from .geometry import METRES_PER_DEGREE, centroid

        import math

        c = centroid(area)
        k = METRES_PER_DEGREE * math.cos(math.radians(c[0]))
        xy = [((lon - c[1]) * k, (lat - c[0]) * METRES_PER_DEGREE) for lat, lon in area]
        a = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1]))) / 2
        return f"{a / 10000:.2f} ha" if a >= 10000 else f"{a:.0f} m²"

    # ------------------------------------------------------------------ layers on the 2D view
    def _view(self):
        window = self._window()
        return getattr(window, "ortho_view", None) if window is not None else None

    def draw_layers(self) -> None:
        """The layers of the plan on the 2D view (after every raster shown and every change of the plan): the area,
        the strips (one line per leg), the shots, the takeoffs, the targets and the low coverage."""
        from . import project

        view = self._view()
        if view is None:
            return
        for name in LAYERS:
            view.remove_layer(name)
        window = self._window()
        chunk = self._chunk()
        if self.pane is None or chunk is None or view.tr is None or not project.has_crs(chunk) \
                or getattr(window, "drawings_frame", None) is not None:
            return
        batch = view.batch() if hasattr(view, "batch") else None
        if batch is not None:
            with batch:
                self._draw(view, chunk)
        else:
            self._draw(view, chunk)

    def _draw(self, view, chunk) -> None:
        from . import plan as pl
        from . import project
        from .legmap import COLOURS
        from .route import shot_points

        e = project.active_plan(chunk)
        if self.area and len(self.area) >= 3:
            xy = project.to_chunk(chunk, self.area)
            view.add_polygon(xy.tolist(), t("Area", "Area"), "#1e88e5", 30, layer=LAYER_AREA,
                             attrs={"tipo": "disegnata"})
        if e is None:
            return
        piano = e["piano"]
        if float(piano.get("margine") or 0) > 0:
            for b in pl.active_blocks(piano):
                if b.get("tipo", "griglia") == "griglia" and len(b.get("area") or []) >= 3:
                    xy = project.to_chunk(chunk, [tuple(q) for q in b["area"]])
                    view.add_polygon(xy.tolist(), "", "#90caf9", 10, layer=LAYER_AREA,
                                     attrs={"tipo": "con margine", "blocco": b.get("nome", "")})
        legs = pl.plan_legs(piano)
        shots: list = []
        for k, leg in enumerate(legs):
            r = leg.route
            if len(r.points) >= 2:
                xy = project.to_chunk(chunk, r.points)
                block = piano["blocchi"][leg.block_index].get("nome") or str(leg.block_index + 1)
                view.add_line(xy.tolist(), COLOURS[k % 8], "", layer=LAYER_STRIPS,
                              attrs={"blocco": block, "tratta": leg.file_name, "scatti": r.shots(),
                                     "minuti": round(r.minutes(), 1)})
            shots += [(p, h, leg.file_name) for p, h in shot_points(r)]
        step = max(1, -(-len(shots) // MAX_SHOTS_DRAWN))
        shown = shots[::step]
        if shown:
            xy = project.to_chunk(chunk, [s[0] for s in shown])
            for (x, y), (_p, h, name) in zip(xy.tolist(), shown):
                view.add_point(x, y, "#fb8c00", "", layer=LAYER_SHOTS,
                               attrs={"tratta": name, "quota_decollo_m": round(float(h), 1)})
        if step > 1 and not self._status.startswith(f"{len(shots)} "):
            self._status_text(t("{0} scatti: sulla vista uno ogni {1}.", "{0} shots: on the view one every {1}.")
                              .format(len(shots), step))
        for label, point in pl.plan_takeoffs(piano):
            (x, y), = project.to_chunk(chunk, [point]).tolist()
            view.add_point(x, y, "#43a047", t("Decollo", "Takeoff") if label == "Decollo" else label,
                           layer=LAYER_TAKEOFF, attrs={"nome": label})
        for q in (piano.get("target") or {}).get("punti") or []:
            (x, y), = project.to_chunk(chunk, [(q["lat"], q["lon"])]).tolist()
            estimated = bool(q.get("stimato"))
            label = str(q.get("nome", "")) + (t(" (stimato)", " (estimated)") if estimated else "")
            view.add_point(x, y, "#9e9e9e" if estimated else "#8e24aa", label, layer=LAYER_TARGETS,
                           attrs={"nome": q.get("nome"), "codice": q.get("codice"), "stimato": estimated})
        from . import compare

        rec = compare.comparison_record(chunk, e["id"])
        if rec is not None and rec.get("missing"):
            for (x, y), m in zip(compare.missing_xy(chunk, rec).tolist(), rec["missing"]):
                view.add_point(x, y, "#d32f2f", "", layer=LAYER_MISSING,
                               attrs={"tratta": m["leg"], "file_tratta": m["tratta"], "scatto": m["n"],
                                      "quota_decollo_m": m["quota_m"]})
        for poly in self._low_polygons(chunk, e["id"]):
            view.add_polygon([list(c) for c in poly.exterior.coords][:-1], "", "#d32f2f", 90, layer=LAYER_LOW,
                             attrs={"area_m2": round(poly.area, 1)})

    def _low_polygons(self, chunk, id_: str) -> list:
        from . import coverage, project

        rec = project.coverage_record(chunk, id_)
        if rec is None:
            return []
        key = (id_, rec.get("stamp"), json.dumps(rec.get("files"), sort_keys=True), rec.get("low"),
               rec.get("dem"))
        if key not in self._low_cache:
            cov = project.read_coverage(chunk, id_)
            self._low_cache = {key: coverage.low_polygons(cov) if cov is not None else []}
        return self._low_cache[key]

    def _frame_plan(self, chunk) -> None:
        """The view on something to draw on: the chunk's raster, else an empty frame around the plan or the area."""
        from . import plan as pl
        from . import project

        view = self._view()
        if view is None or view.tr is not None or not project.has_crs(chunk):
            return
        if self.context.show_raster(None):
            return
        points = list(self.area or [])
        e = project.active_plan(chunk)
        if e is not None:
            for r in pl.plan_routes(e["piano"]):
                points += list(r.points)
        if len(points) < 2:
            return
        xy = project.to_chunk(chunk, points)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = max(20.0, 0.1 * float(max(hi - lo)))
        lo, hi = lo - pad, hi + pad
        self.context.show_frame((lo[0], lo[1], hi[0], hi[1]), max(float(max(hi - lo)) / 1500.0, 0.05), t(*NAME))

    # ------------------------------------------------------------------ the area
    def draw_area(self) -> None:
        self._guarded(self._draw_area)

    def _draw_area(self) -> None:
        from . import project

        chunk = self._need_chunk()
        if not project.has_crs(chunk):
            raise ValueError(t("Il chunk è in un sistema locale: l'area si dà da un file o con coordinate in gradi.",
                               "The chunk is in a local frame: give the area from a file or with coordinates in "
                               "degrees."))
        self._frame_plan(chunk)
        self.context.request_polygon(lambda xy: self._guarded(self._area_drawn, xy),
                                     hint=t("L'area del volo: trascina un riquadro, oppure clic sui vertici e doppio "
                                            "clic o Invio; Esc annulla",
                                            "The flight area: drag a box, or click the vertices and double click "
                                            "or Enter; Esc cancels"))

    def _area_drawn(self, xy) -> None:
        from . import project

        chunk = self._need_chunk()
        self._set_area(project.to_wgs84(chunk, xy), t("disegnata sulla vista", "drawn on the view"))

    def _set_area(self, area, source: str) -> None:
        area = [(float(a), float(b)) for a, b in area]
        if len(area) >= 2 and area[0] == area[-1]:
            area = area[:-1]
        if len(area) < 3:
            raise ValueError(t("l'area ha bisogno di almeno tre vertici", "the area needs at least three vertices"))
        self.area, self.area_source, self.new_area = area, source, True
        self._status_text(t("Area pronta: scegli camera e parametri, poi Progetta.",
                            "Area ready: choose camera and parameters, then Design."))
        self.refresh_pane()

    def area_from_layer(self) -> None:
        self._guarded(self._area_from_layer)

    def _area_from_layer(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import project

        chunk = self._need_chunk()
        view = self._view()
        found = []
        for name in list(getattr(view, "layers", {}) or {}):
            if name in LAYERS:
                continue
            for k, attrs in enumerate(view.features(name)):
                geom = attrs.get("_geom")
                if geom and geom[0] == "Polygon" and len(geom[1]) >= 3:
                    label = attrs.get("etichetta") or f"#{k + 1}"
                    found.append((f"{name}: {label}", geom[1]))
        if not found:
            raise ValueError(t("Nessun poligono nei layer della vista 2D.", "No polygon in the layers of the 2D view."))
        got = self._ask(t("Area da un layer", "Area from a layer"),
                        [Field("feature", t("Poligono", "Polygon"), 0,
                               choices=[(i, f[0]) for i, f in enumerate(found)])])
        if not got:
            return
        label, xy = found[int(got["feature"])]
        self._set_area(project.to_wgs84(chunk, xy), t("dal layer {0}", "from the layer {0}").format(label))

    def area_from_file(self) -> None:
        self._guarded(self._area_from_file)

    def _area_from_file(self) -> None:
        path = self._open_path(t("Area del volo", "Flight area"),
                               "KML, KMZ, GeoJSON, GeoPackage (*.kml *.kmz *.geojson *.json *.gpkg)")
        if not path:
            return
        self._set_area(self.read_area_file(path), Path(path).name)

    def read_area_file(self, path) -> list:
        """The first polygon of a KML/KMZ/GeoJSON (WGS84) or of a GeoPackage (any system, converted with pyproj)."""
        from .camera import camera
        from .export_dji import read_file

        path = Path(path)
        if path.suffix.lower() == ".gpkg":
            return _gpkg_area(path)
        imported = read_file(path.name, path.read_bytes(), camera("mini4pro"))
        areas = [a for a in imported.areas if len(a) >= 3]
        if not areas:
            raise ValueError(t("{0}: nessuna area (poligono) nel file", "{0}: no area (polygon) in the file")
                             .format(path.name))
        return [tuple(p) for p in areas[0]]

    def area_from_coordinates(self) -> None:
        self._guarded(self._area_from_coordinates)

    def _area_from_coordinates(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import project

        chunk = self._chunk()
        systems = [("wgs84", t("Gradi WGS84: lat, lon", "WGS84 degrees: lat, lon"))]
        if chunk is not None and project.has_crs(chunk):
            systems.append(("chunk", t("Sistema del chunk: x, y ({0})", "The chunk's system: x, y ({0})")
                            .format(chunk.crs)))
        got = self._ask(t("Area da coordinate", "Area from coordinates"),
                        [Field("system", t("Sistema", "System"), "wgs84", choices=systems),
                         Field("points", t("Vertici (a;b; a;b; ...)", "Vertices (a,b; a,b; ...)"), "", kind="text")],
                        intro=t("Un vertice per coppia, le coppie separate da punto e virgola.",
                                "One vertex per pair, the pairs separated by semicolons."))
        if not got:
            return
        pairs = []
        for part in str(got["points"]).replace("\n", ";").split(";"):
            values = [v for v in part.replace(",", " ").split() if v]
            if not values:
                continue
            if len(values) != 2:
                raise ValueError(t("«{0}»: servono due numeri", "'{0}': two numbers are needed").format(part.strip()))
            pairs.append((float(values[0]), float(values[1])))
        if got["system"] == "chunk":
            area = project.to_wgs84(chunk, pairs)
        else:
            if any(not (-90 <= a <= 90 and -180 <= b <= 180) for a, b in pairs):
                raise ValueError(t("coordinate fuori dal mondo: latitudine poi longitudine, in gradi",
                                   "coordinates off the world: latitude then longitude, in degrees"))
            area = pairs
        self._set_area(area, t("da coordinate", "from coordinates"))

    # ------------------------------------------------------------------ the camera
    def _own_camera(self, cam) -> None:
        self.own_cameras[cam.id] = cam
        self._fill_cameras(current=cam.id)
        self.refresh_pane()

    def camera_from_exif(self) -> None:
        self._guarded(self._camera_from_exif)

    def _camera_from_exif(self) -> None:
        from .camera import camera_from_exif

        path = self._open_path(t("Una foto della camera", "A photo of the camera"),
                               "JPEG, TIFF, DNG (*.jpg *.jpeg *.tif *.tiff *.dng *.JPG)")
        if path:
            self._own_camera(camera_from_exif(path))

    def camera_from_chunk(self) -> None:
        self._guarded(self._camera_from_chunk)

    def _camera_from_chunk(self) -> None:
        from .camera import CameraUnknown, camera_from_sensor

        chunk = self._need_chunk()
        sensors = list(getattr(chunk, "sensors", {}).values()) if isinstance(getattr(chunk, "sensors", None), dict) \
            else list(getattr(chunk, "sensors", []) or [])
        if not sensors:
            raise CameraUnknown(t("Il chunk non ha sensori: aggiungi le foto, o scegli la camera dalla lista.",
                                  "The chunk has no sensors: add the photos, or choose the camera from the list."))
        k = 0
        if len(sensors) > 1:
            k = self._choice(t("Quale sensore del chunk?", "Which sensor of the chunk?"),
                             [str(getattr(s, "label", "") or s.key) for s in sensors])
            if k is None:
                return
        self._own_camera(camera_from_sensor(sensors[k]))

    def camera_by_hand(self) -> None:
        self._guarded(self._camera_by_hand)

    def _camera_by_hand(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from .camera import camera as list_camera
        from .camera import manual_camera

        cid, own = self._pane_camera()
        base = own or list_camera(cid)
        got = self._ask(t("Camera a mano", "Camera by hand"),
                        [Field("name", t("Nome", "Name"), "", kind="text"),
                         Field("width", t("Larghezza del sensore (mm, lato lungo)", "Sensor width (mm, long side)"),
                               base.width_mm, minimum=0.1, maximum=100, decimals=3),
                         Field("height", t("Altezza del sensore (mm)", "Sensor height (mm)"), base.height_mm,
                               minimum=0.1, maximum=100, decimals=3),
                         Field("focal", t("Focale (mm)", "Focal length (mm)"), base.focal_mm, minimum=0.1,
                               maximum=500, decimals=3),
                         Field("pixels", t("Pixel sul lato lungo", "Pixels on the long side"), base.pixels,
                               minimum=100, maximum=100000, decimals=0),
                         Field("interval", t("Intervallo minimo fra scatti (s)", "Shortest interval between shots (s)"),
                               base.min_interval, minimum=0.5, maximum=60, decimals=1)],
                        intro=t("I campi partono dai valori di «{0}», la camera scelta nella lista: scrivi quelli "
                                "del foglio tecnico della tua camera. Nessun valore viene inventato.",
                                "The fields start from the values of '{0}', the camera chosen in the list: type "
                                "those of your camera's spec sheet. No value is invented.").format(base.name))
        if not got:
            return
        name = str(got["name"]).strip() or t("Camera a mano", "Camera by hand")
        self._own_camera(manual_camera(name, float(got["width"]), float(got["height"]), float(got["focal"]),
                                       int(got["pixels"]), float(got["interval"])))

    # ------------------------------------------------------------------ design
    def design_dialog(self) -> None:
        """The menu action: the pane in front and the design with its parameters."""
        self.show_pane()
        self.design()

    def design(self) -> None:
        self._guarded(self._design_now)

    def _design_now(self) -> None:
        from . import design as design_mod
        from . import project

        chunk = self._need_chunk()
        if not self.area:
            raise ValueError(t("Prima l'area: disegnala, prendila da un layer, da un file o dalle coordinate.",
                               "The area first: draw it, take it from a layer, a file or the coordinates."))
        params = self.pane.parameters()
        cam_id, own = self._pane_camera()
        metres = float(self.pane.s_metres.value())       # read here: the job runs in another thread
        old = None if self.new_area else project.active_plan(chunk)
        area = list(self.area)
        crs = str(chunk.crs) if project.has_crs(chunk) else None
        names = {p["name"] for p in project.plans(chunk)}
        name = old["name"] if old else next(f for n in range(1, 1000)
                                           if (f := t("Piano {0}", "Plan {0}").format(n)) not in names)
        old_piano = copy.deepcopy(old["piano"]) if old else None

        def job(progress):
            progress(0.1, t("Progetto del volo", "Designing the flight"))
            out = design_mod.design(area, params["goal"], params["value"], camera=cam_id, high=params["high"],
                                    battery_minutes=params["battery"], mode=params["mode"], name=name,
                                    margin_m=params["margin"], double_grid=params["double"],
                                    obliques=params["obliques"], custom_camera=own,
                                    takeoff=tuple(old_piano["decollo"]) if old_piano and old_piano.get("decollo")
                                    else None)
            piano, reasons = out["piano"], list(out["motivi"])
            _apply_parameters(piano, params, reasons)
            if old_piano and old_piano.get("decolli"):
                from . import plan as pl

                keys = {leg.key for leg in pl.plan_legs(piano)}
                kept = {k: v for k, v in old_piano["decolli"].items() if k in keys}
                if kept:
                    piano["decolli"] = kept
                lost = len(old_piano["decolli"]) - len(kept)
                if lost:
                    reasons.append(plural(lost, t("{0} decollo di una tratta non c'è più (le tratte sono cambiate): "
                                                  "rimettilo.", "{0} takeoff of a leg is gone (the legs changed): "
                                                  "place it again."),
                                          t("{0} decolli di tratte non ci sono più (le tratte sono cambiate): "
                                            "rimettili.", "{0} takeoffs of legs are gone (the legs changed): place "
                                            "them again.")))
            piano["areaDisegnata"] = [list(q) for q in area]
            if crs:
                piano["crs"] = crs
            if old_piano and old_piano.get("dettaglioTerreno"):
                piano["dettaglioTerreno"] = old_piano["dettaglioTerreno"]
            warnings: list[str] = []
            if old_piano and any(b.get("seguiTerreno") for b in old_piano.get("blocchi") or []):
                progress(0.4, t("Quote del terreno", "Terrain heights"))
                source = (old_piano.get("fonteQuote") or {}).get("sorgente")
                dem = project.chunk_dem(chunk) if source == "dem" else None
                try:
                    for b in piano["blocchi"]:
                        b["seguiTerreno"] = True
                    from . import tiles

                    terrain = design_mod.apply_terrain(piano, z=tiles.zoom_for_detail(piano.get("dettaglioTerreno")),
                                                       dem=dem)
                    warnings += [warning_text(w.get("tipo", "")) for w in terrain.get("avvisi") or []]
                    reasons.append(t("Il volo segue ancora il terreno, come il piano di prima.",
                                     "The flight still follows the terrain, as the plan before."))
                except _expected_errors() as exc:
                    design_mod.remove_terrain(piano)
                    reasons.append(t("Quote del terreno non rifatte ({0}): il volo resta a quota costante.",
                                     "Terrain heights not made again ({0}): the flight stays at a constant height.")
                                   .format(exc))
                finally:
                    if dem is not None:
                        dem.close()
            if old_piano and (old_piano.get("target") or {}).get("punti"):
                from .targets import put_targets

                kept = old_piano["target"]
                put_targets(piano, {"crs": kept.get("crs"), "punti": kept["punti"]}, metres=metres)
                if kept.get("avvisi"):
                    piano["target"]["avvisi"] = list(kept["avvisi"])
            progress(0.7, t("Avvisi", "Warnings"))
            dem = project.chunk_dem(chunk)
            try:
                found = design_mod.design_warnings(piano, dem)
                warnings += [_design_warning_text(w) for w in found]
            except _expected_errors():
                pass
            finally:
                if dem is not None:
                    dem.close()
            return piano, reasons, warnings

        def done(result):
            piano, reasons, warnings = result
            if old is not None and project.entry(chunk, old["id"]) is not None:
                id_ = old["id"]
                project.update_plan(chunk, id_, piano)
                project.set_active(chunk, id_)
            else:
                id_ = project.add_plan(chunk, piano)
            e = project.entry(chunk, id_)
            self._reasons[id_] = reasons
            self._warnings[(id_, e["stamp"])] = warnings
            self.new_area = False
            self._shown_plan = (id(chunk), id_)
            self.context.mark_modified()
            self._status_text(t("Piano «{0}» progettato.", "Plan '{0}' designed.").format(e["name"]))
            self._frame_plan(chunk)

        self._task(t("Progetto del volo", "Designing the flight"), job, done, chunk=chunk)

    # ------------------------------------------------------------------ the plan
    def use_plan(self, id_: str) -> None:
        self._guarded(self._use_plan, id_)

    def _use_plan(self, id_: str, chunk=None) -> None:
        from . import project

        chunk = chunk if chunk is not None else self._need_chunk()
        if (project.active_plan(chunk) or {}).get("id") == id_:
            return
        project.set_active(chunk, id_)
        self._changed()

    def remove_plan(self, id_: str) -> None:
        self._guarded(self._remove_plan, id_)

    def _remove_plan(self, id_: str, chunk=None) -> None:
        from . import project

        chunk = chunk if chunk is not None else self._need_chunk()
        e = project.entry(chunk, id_)
        if e is None:
            return
        try:
            project.remove_plan(chunk, id_)
        finally:
            # the meta changes before the files move: marked also when moving them failed (a file open elsewhere)
            self._shown_plan = None
            self._changed()
        self._status_text(t("Piano «{0}» rimosso (Modifica > Annulla lo rimette).",
                            "Plan '{0}' removed (Edit > Undo puts it back).").format(e["name"]))

    def toggle_block(self, index: int, on: bool) -> None:
        self._guarded(self._toggle_block, index, on)

    def _toggle_block(self, index: int, on: bool) -> None:
        from . import project

        chunk, e = self._need_plan()
        piano = copy.deepcopy(e["piano"])
        blocks = piano.get("blocchi") or []
        if not 0 <= index < len(blocks) or (blocks[index].get("attivo", True) is not False) == bool(on):
            return
        blocks[index]["attivo"] = bool(on)
        project.update_plan(chunk, e["id"], piano)
        self._changed()

    def place_takeoff(self, key: str = "") -> None:
        self._guarded(self._place_takeoff, key)

    def _place_takeoff(self, key: str) -> None:
        from . import project

        chunk, e = self._need_plan()
        if not project.has_crs(chunk):
            raise ValueError(t("Il chunk è in un sistema locale: il decollo non si mette sulla vista.",
                               "The chunk is in a local frame: the takeoff is not placed on the view."))
        self._frame_plan(chunk)
        id_ = e["id"]

        def placed(xy):
            (lat, lon), = project.to_wgs84(chunk, xy[:1])
            entry = project.entry(chunk, id_)
            if entry is None:
                return
            piano = copy.deepcopy(entry["piano"])
            if key:
                piano.setdefault("decolli", {})[key] = [lat, lon]
            else:
                piano["decollo"] = [lat, lon]
            project.update_plan(chunk, id_, piano)
            self._status_text(t("Decollo messo: le quote del terreno si rifanno all'esportazione.",
                                "Takeoff placed: the terrain heights are made again at the export."))
            self._changed()

        hint = t("Il decollo della tratta {0}: clic sul punto; Esc annulla",
                 "The takeoff of the leg {0}: click on the point; Esc cancels").format(key) if key else \
            t("Il punto di decollo: clic sul punto; Esc annulla", "The takeoff point: click on the point; Esc cancels")
        self.context.request_point(lambda xy: self._guarded(placed, xy), hint=hint)

    # ------------------------------------------------------------------ terrain
    def follow_terrain(self, source: str = "dem", detail: str = "normale") -> None:
        self._guarded(self._follow_terrain, source, detail)

    def _follow_terrain(self, source: str, detail: str) -> None:
        from . import design as design_mod
        from . import project, tiles

        chunk, e = self._need_plan()
        if source == "dem":
            dem = project.chunk_dem(chunk)
            if dem is None:
                has_file = getattr(chunk, "dem_path", None) and Path(chunk.dem_path).is_file()
                raise ValueError(t("Il DEM del progetto non dice dove si trova (nessuna georeferenza): scegli le "
                                   "tegole pubbliche.",
                                   "The project DEM does not say where it is (no georeference): choose the public "
                                   "tiles.") if has_file else
                                 t("Il chunk non ha un DEM: scegli le tegole pubbliche (serve la rete).",
                                   "The chunk has no DEM: choose the public tiles (the network is needed)."))
            dem.close()
        piano = copy.deepcopy(e["piano"])
        id_ = e["id"]

        def job(progress):
            progress(0.1, t("Quote del terreno", "Terrain heights"))
            piano["dettaglioTerreno"] = detail
            dem = project.chunk_dem(chunk) if source == "dem" else None
            try:
                out = design_mod.apply_terrain(piano, z=tiles.zoom_for_detail(detail), dem=dem)
            finally:
                if dem is not None:
                    dem.close()
            return out

        def done(out):
            if project.entry(chunk, id_) is None:
                return
            project.update_plan(chunk, id_, piano)
            stamp = project.entry(chunk, id_)["stamp"]
            self._warnings[(id_, stamp)] = [warning_text(w.get("tipo", "")) for w in out.get("avvisi") or []]
            self.context.mark_modified()
            self._status_text(t("Il volo segue il terreno: quote da {0:g} a {1:g} m sul decollo.",
                                "The flight follows the terrain: heights from {0:g} to {1:g} m above the takeoff.")
                              .format(out.get("minima", 0), out.get("massima", 0)))

        self._task(t("Quote del terreno", "Terrain heights"), job, done, chunk=chunk, source=source)

    def remove_terrain(self) -> None:
        self._guarded(self._remove_terrain)

    def _remove_terrain(self) -> None:
        from . import design as design_mod
        from . import project

        chunk, e = self._need_plan()
        piano = copy.deepcopy(e["piano"])
        if not design_mod.remove_terrain(piano)["blocchi"]:
            return
        project.update_plan(chunk, e["id"], piano)
        self._changed()

    # ------------------------------------------------------------------ coverage
    def compute_coverage(self, low: int | None = None) -> None:
        self._guarded(self._compute_coverage, low)

    def _compute_coverage(self, low) -> None:
        from . import coverage, project

        chunk, e = self._need_plan()
        if not project.has_crs(chunk):
            raise coverage.CoverageError(t("Serve un sistema di riferimento proiettato: il chunk è in un sistema "
                                           "locale.", "A projected reference system is needed: the chunk is in a "
                                           "local frame."))
        low = int(low) if low is not None else int(self.pane.s_low.value() if self.pane is not None else coverage.LOW)
        id_ = e["id"]

        def job(progress):
            return project.coverage_for(chunk, id_, progress, low=low)

        def done(rec):
            self.context.mark_modified()        # the record is in the meta: saved with the project
            if project.entry(chunk, id_) is not None:
                self._show_coverage(True)

        self._task(t("Copertura prevista", "Predicted coverage"), job, done, chunk=chunk)

    def show_coverage(self, on: bool = True) -> None:
        self._guarded(self._show_coverage, on)

    def _show_coverage(self, on: bool) -> None:
        from . import project

        chunk, e = self._need_plan()
        if on:
            rec = project.coverage_record(chunk, e["id"])
            if rec is None:
                raise ValueError(t("La copertura di questo piano non c'è (o è di un piano cambiato): Calcola.",
                                   "The coverage of this plan is not there (or belongs to a changed plan): Compute."))
            self._coverage_shown = True
            self.context.show_raster("ortho", path=Path(chunk.data_dir) / rec["files"]["rgba"])
            return
        self._hide_coverage(chunk)

    def _hide_coverage(self, chunk) -> None:
        """Back from the coverage to the chunk's orthophoto (or DEM), else an empty frame around the plan."""
        self._coverage_shown = False
        if not self.context.show_raster(None):
            view = self._view()
            if view is not None and hasattr(view, "clear_raster"):
                view.clear_raster()
            self._frame_plan(chunk)

    # ------------------------------------------------------------------ after the flight
    def compare_flight(self, max_distance: float | None = None) -> None:
        self._guarded(self._compare_flight, max_distance)

    def _compare_flight(self, max_distance) -> None:
        from . import compare, project

        chunk, e = self._need_plan()
        if not [c for c in chunk.cameras if getattr(c, "enabled", True)]:
            raise ValueError(t("Il chunk non ha foto: aggiungi le foto del volo (File > Aggiungi foto), poi "
                               "Confronta.", "The chunk has no photos: add the photos of the flight (File > Add "
                               "photos), then Compare."))
        if max_distance is None and self.pane is not None:
            max_distance = float(self.pane.s_max_distance.value())
        max_distance = float(max_distance) if max_distance else None
        id_ = e["id"]

        def job(progress):
            return compare.compare_for(chunk, id_, max_distance, progress)

        def done(rec):
            self.context.mark_modified()        # the record is in the meta: saved with the project (and undone)
            if project.entry(chunk, id_) is not None:
                self._status_text(compare.summary_line(rec))

        self._task(t("Confronto con il volo", "Comparison with the flight"), job, done, chunk=chunk, plan_id=id_)

    def show_obtained(self) -> None:
        self._guarded(self._show_obtained)

    def _show_obtained(self) -> None:
        chunk, e = self._need_plan()
        path = self._obtained_path(chunk, e)
        if path is None:
            raise ValueError(t("La copertura ottenuta non c'è: allinea le foto e premi Confronta.",
                               "The coverage obtained is not there: align the photos and press Compare."))
        self._coverage_shown = True
        self.context.show_raster("ortho", path=path)

    # ------------------------------------------------------------------ targets
    def targets_from_csv(self) -> None:
        self._guarded(self._targets_from_csv)

    def _targets_from_csv(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import plan as pl
        from . import targets
        from .geometry import centroid

        chunk, e = self._need_plan()
        path = self._open_path(t("I target GNSS (CSV)", "The GNSS targets (CSV)"), "CSV, TXT (*.csv *.txt)")
        if not path:
            return
        got = self._ask(t("Sistema delle coordinate", "System of the coordinates"),
                        [Field("epsg", t("EPSG (vuoto: gradi, o UTM della zona del sito)",
                                         "EPSG (empty: degrees, or UTM of the site's zone)"), "", kind="text")],
                        intro=t("Le coordinate in metri di un sistema diverso dall'UTM vanno dette con il loro EPSG; "
                                "in gradi senza EPSG.", "Coordinates in metres of a system other than UTM are given "
                                "with their EPSG; in degrees without one."))
        if got is None:
            return
        text = str(got.get("epsg") or "").strip().upper().replace("EPSG:", "")
        epsg = int(text) if text else None
        piano = copy.deepcopy(e["piano"])
        points = [q for b in pl.active_blocks(piano) for q in pl.route_of_block(b, piano)[0].points]
        read = targets.read_targets_csv(Path(path).read_text(encoding="utf-8-sig"),
                                        near=centroid(points) if points else None, epsg=epsg)
        self._put_targets(chunk, e, piano, read)

    def targets_from_markers(self) -> None:
        self._guarded(self._targets_from_markers)

    def _targets_from_markers(self) -> None:
        from . import targets

        chunk, e = self._need_plan()
        try:
            read = targets.targets_from_markers(chunk, only_with_reference=True)
        except targets.TargetError as exc:
            if not any(getattr(m, "position", None) is not None for m in getattr(chunk, "markers", [])):
                raise
            k = self._choice(t("{0}. Usare le posizioni stimate dal rilievo? Vanno bene per sorvolarli, non come GCP.",
                               "{0}. Use the positions estimated by the survey? They are fine to fly over, not as "
                               "GCPs.").format(exc), [t("Usa le stimate", "Use the estimated ones")])
            if k is None:
                return
            read = targets.targets_from_markers(chunk, only_with_reference=False)
        self._put_targets(chunk, e, copy.deepcopy(e["piano"]), read)

    def _put_targets(self, chunk, e, piano: dict, read: dict) -> None:
        from . import project, targets

        targets.put_targets(piano, read, metres=float(self.pane.s_metres.value()) if self.pane is not None
                            else targets.METRES_ABOVE_GROUND)
        codes = list(read.get("avvisi") or [])
        if codes:
            piano["target"]["avvisi"] = codes
        project.update_plan(chunk, e["id"], piano)
        self._changed()
        n = len(read.get("punti") or [])
        self._status_text(plural(n, t("{0} target nel piano.", "{0} target in the plan."),
                                 t("{0} target nel piano.", "{0} targets in the plan.")))
        if "ordine_incerto" in codes:
            self._message("warning", warning_text("ordine_incerto") + "\n\n" +
                          t("Guarda i target sulla vista: se sono lontani dal sito, rifai l'import con l'EPSG giusto.",
                            "Look at the targets on the view: if they are far from the site, import again with the "
                            "right EPSG."))

    def remove_targets(self) -> None:
        self._guarded(self._remove_targets)

    def _remove_targets(self) -> None:
        from . import project, targets

        chunk, e = self._need_plan()
        if not (e["piano"].get("target") or any(sp.get("target") for b in e["piano"].get("blocchi") or []
                                                for sp in b.get("speciali") or [])):
            return
        piano = copy.deepcopy(e["piano"])
        targets.remove_targets(piano)
        project.update_plan(chunk, e["id"], piano)
        self._changed()

    # ------------------------------------------------------------------ export
    def export_dialog(self) -> None:
        """The menu action: the format chosen in the pane."""
        self.show_pane()
        fmt = str(self.pane.c_format.currentData()) if self.pane is not None else "kmz"
        self.export(fmt)

    def export(self, fmt: str) -> None:
        self._guarded(self._export_now, fmt)

    def _export_now(self, fmt: str) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import plan as pl
        from . import project

        chunk, e = self._need_plan()
        if fmt == "gpkg" and not project.has_crs(chunk):
            raise ValueError(t("Il GeoPackage ha bisogno di un sistema di riferimento: il chunk è in un sistema "
                               "locale.",
                               "The GeoPackage needs a reference system: the chunk is in a local frame."))
        missions = None
        if fmt == "djifly-zip":
            got = self._ask(t("Missioni di DJI Fly", "DJI Fly missions"),
                            [Field("missions", t("Le cartelle delle missioni fittizie, separate da virgole",
                                                 "The folders of the placeholder missions, comma separated"),
                                   "", kind="text")],
                            intro=t("Una missione fittizia salvata in DJI Fly per ogni tratta ({0}).",
                                    "A placeholder mission saved in DJI Fly for every leg ({0}).")
                            .format(len(pl.plan_legs(e["piano"]))))
            if not got:
                return
            missions = [m.strip() for m in str(got["missions"]).split(",") if m.strip()]
        suffix = project.SUFFIXES[fmt]
        path = self._save_path(format_name(fmt), pl.slug(e["name"]) + suffix, f"{format_name(fmt)} (*{suffix})")
        if not path:
            return
        if not path.lower().endswith(suffix) and not path.lower().endswith(".zip"):
            path += suffix
        id_ = e["id"]

        def job(progress):
            return project.export(chunk, id_, fmt, path, progress, missions=missions)

        def done(files):
            from .export_formats import terrain_note

            entry = project.entry(chunk, id_)
            note = terrain_note(entry["piano"]) if entry is not None and fmt in ("plan", "waypoints", "mavlink",
                                                                                   "litchi") else None
            text = plural(len(files), t("{0} file scritto", "{0} file written"),
                          t("{0} file scritti", "{0} files written")) + ":\n" + "\n".join(str(f) for f in files)
            if note:
                text += "\n\n" + note[:1].upper() + note[1:] + "."
            self._status_text(plural(len(files), t("Esportato {0} file.", "{0} file exported."),
                                     t("Esportati {0} file.", "{0} files exported.")))
            self._message("info", text)

        self._task(t("Esportazione del piano di volo", "Exporting the flight plan"), job, done, chunk=chunk,
                  plan_id=id_)

    def open_in_qgis(self) -> None:
        window = self._window()
        if window is not None and hasattr(window, "open_in_qgis"):
            window.open_in_qgis()

    # ------------------------------------------------------------------ DJI Fly
    def copy_djifly(self) -> None:
        self._guarded(self._copy_djifly)

    def _copy_djifly(self, folder=None, overwrite: bool = False) -> None:
        from . import djifly
        from . import plan as pl
        from . import project

        chunk, e = self._need_plan()
        if folder is None:
            found = djifly.candidates()
            start = str(found[0]) if found else ""
            chosen = self._folder_path(t("La cartella dei waypoint di DJI Fly", "DJI Fly's waypoint folder"), start)
            if not chosen:
                return
            folder = chosen
        n = len(pl.plan_legs(e["piano"]))
        if djifly.is_mtp(str(folder)):
            # on a phone through MTP the slots are read by the copy itself
            containers = [m.strip() for m in (self._ask_missions(n) or []) if m.strip()]
            if not containers:
                return
            states = []
        else:
            folder = djifly.normalise(str(folder))
            names = [m.folder for m in djifly.missions(folder)]          # in the order DJI Fly made them
            names += sorted(d.name for d in folder.iterdir() if d.is_dir() and d.name not in names
                            and d.name != djifly.PREVIEWS and not d.name.startswith("."))   # slots without a KMZ
            states = djifly.slot_states(folder, names)
            safe = [s for s in states if s["stato"] in djifly.SAFE_SLOTS]
            busy = [s for s in states if s["stato"] not in djifly.SAFE_SLOTS]
            chosen = (safe + busy)[:n]
            if len(chosen) < n:
                raise djifly.DjiFlyError(t("Servono {0} missioni fittizie in DJI Fly, ce ne sono {1}: creane una per "
                                           "ogni tratta (due waypoint bastano) e riprova.",
                                           "{0} placeholder missions are needed in DJI Fly, there are {1}: create one "
                                           "for every leg (two waypoints are enough) and try again.")
                                         .format(n, len(chosen)))
            containers = [s["contenitore"] for s in chosen]
            lines = "\n".join(f"{s['contenitore']} ({s.get('nomeDji') or '-'}): {_slot_text(s['stato'])}"
                              for s in chosen)
            if not overwrite:
                over = any(s["stato"] not in djifly.SAFE_SLOTS for s in chosen)
                k = self._choice(t("Le {0} tratte vanno in queste missioni:\n{1}", "The {0} legs go into these "
                                   "missions:\n{1}").format(n, lines),
                                 [t("Sovrascrivi (una copia resta)", "Overwrite (a copy is kept)") if over
                                  else t("Copia", "Copy")], kind="warning" if over else "question")
                if k is None:
                    return
                overwrite = over               # the user saw the missions of their own in the list and said yes
        backup = project.folder(chunk) / "djifly_backup"
        id_ = e["id"]
        piano = copy.deepcopy(e["piano"])

        def job(progress):
            progress(0.1, t("Quote del terreno", "Terrain heights"))
            dem = project.chunk_dem(chunk)
            try:
                fresh, _outcome = project.prepare(piano, dem)
            finally:
                if dem is not None:
                    dem.close()
            progress(0.3, t("Tratte e anteprime", "Legs and previews"))
            done_ = copy_legs(chunk, fresh, folder, containers, backup, overwrite)
            return fresh, done_

        def done(result):
            fresh, outcome = result
            if project.entry(chunk, id_) is not None:
                project.update_plan(chunk, id_, fresh)
                self.context.mark_modified()
            lines = "\n".join(f"{o.get('tratta')} → {o.get('contenitore')}"
                              + ("" if o.get("verificato", True) else t(" (da verificare)", " (to check)"))
                              for o in outcome)
            self._message("info", t("Tratte copiate in DJI Fly:\n{0}\n\nLe copie di quel che c'era sono in {1}.",
                                    "Legs copied into DJI Fly:\n{0}\n\nThe copies of what was there are in {1}.")
                          .format(lines, backup))

        def run_again_if_busy(busy):
            # some slots hold missions of the user: nothing was written; the user decides (a copy is kept anyway)
            k = self._choice(str(busy.error), [t("Sovrascrivi (una copia resta)", "Overwrite (a copy is kept)")],
                             kind="warning")
            if k is not None:
                self.context.call_in_gui(self._guarded, self._copy_djifly, folder, True)

        self._run_djifly(job, done, run_again_if_busy, chunk, id_)

    def _run_djifly(self, job, done, on_busy, chunk, id_) -> None:
        from . import djifly

        def work(progress):
            try:
                return job(progress)
            except djifly.SlotsInUse as exc:
                return _Busy(exc)

        def finished(result):
            if isinstance(result, _Busy):
                on_busy(result)
                return
            done(result)

        self._task(t("Copia in DJI Fly", "Copying into DJI Fly"), work, finished, chunk=chunk, plan_id=id_)

    def _ask_missions(self, n: int):
        from pyarchinit_photogrammetry.gui.dialogs import Field

        got = self._ask(t("Missioni di DJI Fly", "DJI Fly missions"),
                        [Field("missions", t("Le cartelle delle missioni fittizie, separate da virgole",
                                             "The folders of the placeholder missions, comma separated"), "",
                               kind="text")],
                        intro=t("Una missione fittizia per ognuna delle {0} tratte.",
                                "A placeholder mission for each of the {0} legs.").format(n))
        return str(got["missions"]).split(",") if got else None

    # ------------------------------------------------------------------ the project's tree and QGIS
    def _tree_items(self, chunk) -> list:
        from . import project

        plans = project.plans(chunk)
        if not plans:
            return []
        active = project.peek(chunk).get("active")
        children = []
        for p in plans:
            item = {"label": p["name"] + (t(" (attivo)", " (active)") if p["id"] == active else ""),
                    "icon": _icon(),
                    "tooltip": t("Doppio clic: il piano attivo", "Double click: the active plan"),
                    "activate": partial(self._tree_use, chunk, p["id"]),
                    "actions": [(t("Rendi attivo", "Make active"), partial(self._tree_use, chunk, p["id"])),
                                (t("Rimuovi", "Remove"), partial(self._tree_remove, chunk, p["id"]))]}
            rec = project.coverage_record(chunk, p["id"])
            kids = []
            if rec is not None:
                kids.append({"label": t("Copertura prevista", "Predicted coverage"), "icon": "relief",
                             "tooltip": t("Doppio clic: la copertura sulla vista 2D",
                                          "Double click: the coverage on the 2D view"),
                             "activate": partial(self._tree_coverage, chunk, p["id"])})
            from . import compare

            done = compare.comparison_record(chunk, p["id"])
            if done is not None:
                kids.append({"label": t("Confronto con il volo", "Comparison with the flight"), "icon": _icon(),
                             "tooltip": compare.summary_line(done),
                             "activate": partial(self._tree_comparison, chunk, p["id"])})
            if kids:
                item["children"] = kids
            children.append(item)
        return [{"label": t(*NAME), "icon": _icon(), "children": children, "expanded": True}]

    def _tree_use(self, chunk, id_: str) -> None:
        self._guarded(self._use_plan, id_, chunk)

    def _tree_remove(self, chunk, id_: str) -> None:
        self._guarded(self._remove_plan, id_, chunk)

    def _tree_coverage(self, chunk, id_: str) -> None:
        def show():
            self._use_plan(id_, chunk)
            self._show_coverage(True)
        self._guarded(show)

    def _tree_comparison(self, chunk, id_: str) -> None:
        def show():
            from . import project

            self._use_plan(id_, chunk)
            e = project.entry(chunk, id_)
            if e is not None and self._obtained_path(chunk, e) is not None:
                self._show_obtained()
            self.show_pane()
        self._guarded(show)

    def _qgis_entries(self):
        from . import project

        return project.qgis_entries(self._chunk())


# ------------------------------------------------------------------ module helpers
def _apply_parameters(piano: dict, params: dict, reasons: list[str]) -> None:
    """The pane's parameters beyond the goal on the designed plan: a height typed, overlaps, direction, speed of the
    nadir grids (the oblique passes keep their own overlaps and speed)."""
    from . import plan as pl

    cam = pl.plan_camera(piano)
    for b in piano.get("blocchi") or []:
        nadir = float(b.get("inclinazione", -90)) <= -89.5
        if params.get("height") is not None:
            b["quota"] = round(float(params["height"]), 1)
        if nadir:
            b["avanti"], b["lato"] = int(params["front"]), int(params["side"])
            b["velocita"] = float(params["speed"])
        if params.get("direction") is not None:
            base = int(params["direction"]) % 180
            if "2 nadirale" in str(b.get("nome", "")) or str(b.get("nome", "")).startswith("C obliquo"):
                b["orientamento"] = (base + 90) % 180
            else:
                b["orientamento"] = base
    if params.get("height") is not None:
        h = round(float(params["height"]), 1)
        gsd = cam.gsd_cm(h, bool(piano.get("alta", True)))
        (piano.get("obiettivo") or {})["gsd"] = round(gsd, 2)
        reasons.append(t("Quota scelta a mano: {0:g} m sul decollo (GSD {1:.2f} cm/px).",
                         "Height chosen by hand: {0:g} m above the takeoff (GSD {1:.2f} cm/px).").format(h, gsd))
    if params.get("direction") is not None:
        reasons.append(t("Strisciate a {0}° scelti a mano.", "Strips at {0}° chosen by hand.")
                       .format(int(params["direction"]) % 180))
    if (params["front"], params["side"]) != (80, 70) or float(params["speed"]) != 5.0:
        reasons.append(t("Griglie nadirali: sovrapposizione {0}% avanti e {1}% laterale, {2:g} m/s.",
                         "Nadir grids: overlap {0}% forward and {1}% sideways, {2:g} m/s.")
                       .format(params["front"], params["side"], params["speed"]))


def _design_warning_text(w: dict) -> str:
    text = warning_text(w.get("tipo", ""))
    name = w.get("nome")
    return f"{name}: {text}" if name else text


def _slot_text(state: str) -> str:
    return {"vuota": t("vuota", "empty"), "fittizia": t("fittizia", "placeholder"),
            "nostra": t("tratta scritta da qui", "leg written from here"),
            "missione": t("MISSIONE DELL'UTENTE: si sovrascrive solo se confermi",
                          "A MISSION OF THE USER: overwritten only if you confirm")}.get(state, state)


def copy_legs(chunk, fresh: dict, folder, containers: list[str], backup, overwrite: bool = False) -> list[dict]:
    """The legs of ``fresh`` (already refreshed) into the DJI Fly slots ``containers`` of ``folder`` (a waypoint
    folder or an MTP path), with the previews on the chunk's orthophoto, the backups in ``backup``; the plan's
    ``djiFly`` entries updated; the outcomes of the copy."""
    from . import djifly, export_dji, legmap
    from . import plan as pl

    legs = export_dji.kmz_per_leg(fresh)
    background = legmap.orthophoto_background(chunk) if chunk is not None else None
    previews = [legmap.dji_preview(fresh, k, background=background) for k in range(len(legs))]
    if djifly.is_mtp(str(folder)):
        states = {}
        done = djifly.copy_mtp(str(folder), legs, containers, backup_dir=backup, overwrite=overwrite)
    else:
        folder = djifly.normalise(str(folder))
        states = {s["contenitore"]: s for s in djifly.slot_states(folder, containers[:len(legs)])}
        done = djifly.copy(folder, legs, containers, previews, backup_dir=backup, overwrite=overwrite)
    given = fresh.setdefault("djiFly", {})
    for leg, outcome in zip(pl.plan_legs(fresh), done):
        container = outcome.get("contenitore")
        given[leg.key] = {"cartella": container, "nomeDji": (states.get(container) or {}).get("nomeDji")}
    return done


def _gpkg_area(path: Path) -> list:
    """The first polygon of a GeoPackage, in WGS84 (lat, lon)."""
    import sqlite3

    from pyproj import CRS, Transformer
    from pyarchinit_photogrammetry.io import gpkg

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT g.table_name, g.srs_id, s.organization, s.organization_coordsys_id, s.definition "
                           "FROM gpkg_geometry_columns g LEFT JOIN gpkg_spatial_ref_sys s ON s.srs_id = g.srs_id "
                           "WHERE upper(g.geometry_type_name) IN ('POLYGON', 'MULTIPOLYGON', 'GEOMETRY')").fetchall()
    except sqlite3.DatabaseError:
        raise ValueError(t("{0}: non è un GeoPackage", "{0}: not a GeoPackage").format(path.name)) from None
    finally:
        con.close()
    for table, _srs, org, code, definition in rows:
        for geom, _attrs in gpkg.read_table(path, table):
            if geom is None:
                continue
            polys = list(getattr(geom, "geoms", [geom]))
            poly = next((g for g in polys if g.geom_type == "Polygon" and not g.is_empty), None)
            if poly is None:
                continue
            crs = CRS.from_epsg(int(code)) if org and str(org).upper() == "EPSG" and code and int(code) > 0 \
                else (CRS.from_wkt(definition) if definition and definition != "undefined" else None)
            if crs is None:
                raise ValueError(t("{0}: il GeoPackage non dice il suo sistema di riferimento",
                                   "{0}: the GeoPackage does not say its reference system").format(path.name))
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            out = []
            for x, y in list(poly.exterior.coords)[:-1]:
                lon, lat = tr.transform(x, y)
                out.append((float(lat), float(lon)))
            return out
    raise ValueError(t("{0}: nessun poligono nel GeoPackage", "{0}: no polygon in the GeoPackage").format(path.name))

