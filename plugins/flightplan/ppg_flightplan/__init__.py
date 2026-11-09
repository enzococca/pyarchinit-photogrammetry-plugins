"""Flight plans (Piani di volo) - a plugin of pyarchinit-photogrammetry: a drone survey designed before going
to the field - area, camera, ground resolution and height, overlaps, strips, shots, time, batteries, terrain
following, the coverage predicted on the DEM - and the files the drones read.

The planner is ported from the author's Photo3D planner. Its plans are JSON dicts of blocks
(:mod:`ppg_flightplan.plan`) on WGS84 latitude and longitude, heights above the takeoff point; the plugin
converts to the chunk's coordinate system only at the boundary (the area drawn on the view, the shots shown
back on it).

The command line: ``ppg plugin flightplan WHAT [PROJECT.ppg] [--chunk LABEL] [--plan FILE] ...`` (see
:meth:`FlightPlanPlugin.command`).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import t, warning_text
from .window import WindowPart, copy_legs

__version__ = "0.1.2"


class CommandError(Exception):
    """A message for the user of the command line: printed, the command returns 1."""


class FlightPlanPlugin(WindowPart, Plugin):
    name = "flightplan"
    title = "Piani di volo"
    version = __version__
    description = ("Piani di volo per il rilievo da drone: area, camera, GSD e quota, sovrapposizioni, strisciate, "
                   "scatti, tempi e batterie, volo che segue il terreno, i file per i droni")

    def __init__(self):
        self.context = None
        self.pane = None
        self._window_init()

    def activate(self, context) -> None:
        self.context = context
        # the public tiles fetched once stay on disk: a second plan of the same site needs no network
        from . import tiles

        tiles.set_cache_dir(Path(context.data_dir) / "tiles")
        actions = self._add_actions(context)
        context.add_command("flightplan", self.command,
                            help="drone flight plans: cameras, design, show, terrain, targets, coverage, export, "
                                 "djifly, compare, add, list, remove, use")
        for title, suffix, fmt in self.exporters():
            context.add_exporter(title, suffix, self._exporter(fmt))
        if context.window is not None:
            self._attach_window(context, actions)

    def deactivate(self) -> None:
        """The layers of the plan off the view and the QGIS provider out; the program takes away the pane, the
        actions, the rail group, the tree items and the connections (``context.release``)."""
        self._detach_window()
        from . import tiles

        tiles.set_cache_dir(None)

    # ------------------------------------------------------------------ export formats of the program
    @staticmethod
    def exporters() -> list[tuple[str, str, str]]:
        return [(t("Piano di volo DJI (KMZ)", "DJI flight plan (KMZ)"), ".kmz", "kmz"),
                (t("Piano di volo QGroundControl", "QGroundControl flight plan"), ".plan", "plan"),
                (t("Piano di volo Litchi (CSV)", "Litchi flight plan (CSV)"), ".csv", "litchi"),
                (t("Piano di volo (GeoPackage)", "Flight plan (GeoPackage)"), ".gpkg", "gpkg"),
                (t("Scheda del volo (Excel)", "Flight sheet (Excel)"), ".xlsx", "sheet-xlsx")]

    def _exporter(self, fmt: str):
        def export(chunk, path, progress=None):
            from . import project

            e = project.active_plan(chunk)
            if e is None:
                raise ValueError(t("il chunk non ha un piano di volo", "the chunk has no flight plan"))
            before = e["stamp"]
            try:
                return project.export(chunk, e["id"], fmt, path, progress)
            finally:
                # the terrain heights were redone: the plan changed (the snapshots of undo see it) - also when
                # writing the file failed after the refresh
                after = project.entry(chunk, e["id"])
                if after is not None and after["stamp"] != before and self.context is not None:
                    self.context.call_in_gui(self.context.mark_modified)
        return export

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _project_args(args) -> tuple[list[str], str | None, str | None]:
        """``ppg plugin flightplan WHAT PROJECT.ppg ...``: the project right after the subcommand (or
        ``--project``) and ``--chunk`` taken out of the arguments."""
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
        """``--bbox -1.83,51.17,...``: a value beginning with a dash glued to its option, so argparse reads it."""
        out, i = [], 0
        while i < len(args):
            a = args[i]
            nxt = args[i + 1] if i + 1 < len(args) else None
            if a in ("--bbox", "--coords", "--xy", "--takeoff") and nxt is not None and nxt.startswith("-") \
                    and not nxt.startswith("--"):
                out.append(f"{a}={nxt}")
                i += 2
                continue
            out.append(a)
            i += 1
        return out

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        from . import coverage, design, project, targets, tiles

        common = argparse.ArgumentParser(add_help=False)
        common.add_argument("--plan", help="a plan JSON file: used (and saved back) instead of the project's plan")
        common.add_argument("--id", help="the plan of the project (default: the active one)")
        parser = argparse.ArgumentParser(prog="ppg plugin flightplan",
                                         description="Drone flight plans: design, terrain, coverage and the files "
                                                     "the drones read.")
        sub = parser.add_subparsers(dest="what", required=True)
        sub.add_parser("cameras", help="the cameras of the list")
        p = sub.add_parser("design", parents=[common], help="a plan for a goal over an area")
        p.add_argument("--area", help="the area from a file (KML, KMZ, GeoJSON)")
        p.add_argument("--bbox", help="the area as west,south,east,north in degrees")
        p.add_argument("--coords", help='the area as "lat,lon;lat,lon;lat,lon" in degrees')
        p.add_argument("--xy", help='the area as "x,y;x,y;x,y" in the chunk\'s system')
        p.add_argument("--goal", default="orto", choices=[g["id"] for g in design.GOALS])
        p.add_argument("--value", type=float, help="the goal's value (cm/px or cm; default the goal's)")
        p.add_argument("--camera", default="mini4pro", help="a camera of the list (see cameras)")
        p.add_argument("--exif", help="the camera from a photo's EXIF")
        p.add_argument("--manual", help="a camera typed: NAME,WIDTH_MM,HEIGHT_MM,FOCAL_MM,PIXELS")
        res = p.add_mutually_exclusive_group()
        res.add_argument("--high-res", action="store_true",
                         help="the camera's high-resolution shot (it flies higher for the same GSD)")
        res.add_argument("--low-res", action="store_true",
                         help="the camera's normal resolution: the default, kept for the scripts written for 0.1.0")
        p.add_argument("--battery", type=float, default=18.0, help="minutes of a battery")
        p.add_argument("--mode", choices=("per_punto", "intervallo"), default="per_punto")
        p.add_argument("--margin", type=float, default=0.0, help="metres added around the area")
        p.add_argument("--double", action="store_true", help="a second nadir grid crossed at 90 degrees")
        p.add_argument("--obliques", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--terrain", action="store_true", help="follow the terrain (the project DEM, the public tiles)")
        p.add_argument("--takeoff", help="the takeoff point as LAT,LON")
        p.add_argument("--name", default=None)
        sub.add_parser("show", parents=[common], help="the numbers of the plan (JSON)")
        p = sub.add_parser("terrain", parents=[common], help="the heights following the terrain")
        p.add_argument("--detail", choices=tuple(tiles.DETAILS), default=None)
        p.add_argument("--off", action="store_true", help="back to one height per block")
        p.add_argument("--tiles", action="store_true", help="the public terrain tiles (network) instead of the DEM")
        p = sub.add_parser("targets", parents=[common], help="GNSS targets as special points")
        p.add_argument("--csv", help="the targets' CSV")
        p.add_argument("--epsg", type=int, help="the system of the CSV's coordinates")
        p.add_argument("--metres", type=float, default=targets.METRES_ABOVE_GROUND, help="metres above the ground")
        p.add_argument("--markers", action="store_true", help="the chunk's markers with measured coordinates")
        p.add_argument("--estimated", action="store_true", help="with --markers: also the estimated ones")
        p.add_argument("--remove", action="store_true")
        p.add_argument("--tiles", action="store_true",
                       help="measure the ground of the descents on the public tiles (network) when there is no DEM")
        p = sub.add_parser("coverage", parents=[common], help="the coverage predicted on the chunk's DEM")
        p.add_argument("--cell", type=float, help="cell of the grid, metres")
        p.add_argument("--low", type=int, default=coverage.LOW, help="fewer photos than this: low coverage")
        p = sub.add_parser("export", parents=[common], help="the plan in a format")
        p.add_argument("--format", required=True, choices=project.FORMATS)
        p.add_argument("--out", required=True)
        p.add_argument("--missions", help="DJI Fly missions (djifly-zip), comma separated")
        p = sub.add_parser("djifly", parents=[common], help="the legs into DJI Fly's waypoint folder, or a zip")
        p.add_argument("--folder", help="the waypoint folder of the remote controller")
        p.add_argument("--zip", help="a zip of ready folders to drag instead")
        p.add_argument("--missions", required=True, help="the placeholder missions (folders), comma separated")
        p.add_argument("--overwrite", action="store_true", help="replace missions of the user too (a copy is kept)")
        p = sub.add_parser("compare", help="after the flight: the plan against the photos of the chunk")
        p.add_argument("--plan-id", "--id", dest="id", help="the plan of the project (default: the active one)")
        p.add_argument("--max-distance", type=float, help="metres to pair a shot and a photo (default: half the "
                                                         "step between the shots of the leg)")
        sub.add_parser("add", parents=[common], help="the plan of --plan FILE into the project")
        sub.add_parser("list", help="the plans of the chunk")
        p = sub.add_parser("remove", help="a plan out of the chunk")
        p.add_argument("plan_id")
        p = sub.add_parser("use", help="make a plan the active one")
        p.add_argument("plan_id")
        return parser

    def _open(self, project_path, label):
        """The document and the chunk: the project given, else the one the program has open (or none)."""
        doc = None
        if project_path:
            import pyarchinit_photogrammetry as ppg

            doc = ppg.Document.open(project_path)
            if self.context is not None:
                self.context.app.document = doc
        elif self.context is not None:
            doc = getattr(self.context.app, "document", None)
        if doc is None:
            return None, None
        if label:
            chunk = next((c for c in doc.chunks if c.label == label), None)
            if chunk is None:
                raise CommandError(f"no chunk {label!r} in the project")
            return doc, chunk
        return doc, doc.chunk

    def command(self, *args) -> int:
        """``ppg plugin flightplan WHAT [PROJECT.ppg] [--chunk LABEL] [--plan FILE] [--id ID] ...``, WHAT one of
        ``cameras``, ``design``, ``show``, ``terrain``, ``targets``, ``coverage``, ``export``, ``djifly``, ``compare``,
        ``add``, ``list``, ``remove``, ``use``; values printed as JSON. 0 when done, 1 with a message."""
        from . import design as design_mod
        from . import compare, coverage, djifly, targets, tiles
        from .camera import CameraUnknown

        args, project_path, label = self._project_args(args)
        ns = self._parser().parse_args(self.glue_values(args))
        try:
            doc, chunk = self._open(project_path, label)
            return self._run(ns, doc, chunk)
        except (CommandError, design_mod.StaleHeights, coverage.CoverageError, compare.CompareError,
                targets.TargetError, CameraUnknown,
                djifly.DjiFlyError, tiles.TerrainError, KeyError, ValueError, OSError) as e:
            text = e.args[0] if isinstance(e, KeyError) and e.args else e
            print(f"{t('errore', 'error')}: {text}")
            if isinstance(e, djifly.SlotsInUse):
                print(t("per sostituirle: --overwrite", "to replace them: --overwrite"))
            return 1

    # ------------------------------------------------------------------ the subcommands
    def _plan_of(self, ns, chunk) -> tuple[dict, str | None]:
        """The plan to work on: the file of ``--plan``, else the project's (``--id`` or the active one)."""
        from . import project

        if getattr(ns, "plan", None):
            return project.standalone_load(ns.plan), None
        if chunk is None:
            raise CommandError("no plan: give the project (ppg plugin flightplan WHAT PROJECT.ppg) or --plan FILE")
        e = project.entry(chunk, ns.id) if getattr(ns, "id", None) else project.active_plan(chunk)
        if e is None:
            raise CommandError(f"no plan {ns.id!r} in the chunk" if getattr(ns, "id", None) else
                               "the chunk has no flight plan: ppg plugin flightplan design PROJECT ...")
        return e["piano"], e["id"]

    def _keep(self, ns, doc, chunk, piano: dict, id_: str | None) -> None:
        """The plan changed: back into its file or into the project (saved)."""
        from . import project

        if id_ is None:
            project.standalone_save(ns.plan, piano)
            return
        project.update_plan(chunk, id_, piano)
        self._save(doc)

    @staticmethod
    def _save(doc) -> None:
        if doc is not None and getattr(doc, "path", None):
            doc.save()

    def _run(self, ns, doc, chunk) -> int:
        from . import plan as pl
        from . import project

        what = ns.what
        if what == "cameras":
            return self._cameras()
        if what == "design":
            return self._design(ns, doc, chunk)
        if what == "list":
            if chunk is None:
                raise CommandError("give the project: ppg plugin flightplan list PROJECT.ppg")
            print(json.dumps(project.summary(chunk), indent=1, ensure_ascii=False))
            return 0
        if what in ("remove", "use"):
            if chunk is None:
                raise CommandError(f"give the project: ppg plugin flightplan {what} PROJECT.ppg ID")
            if project.entry(chunk, ns.plan_id) is None:
                raise CommandError(f"no plan {ns.plan_id!r}: see ppg plugin flightplan list")
            (project.remove_plan if what == "remove" else project.set_active)(chunk, ns.plan_id)
            self._save(doc)
            if what == "remove" and doc is not None and getattr(doc, "path", None) \
                    and (self.context is None or self.context.window is None):
                project.purge_removed(chunk)        # saved, and no undo in the command line: the files go
            print(json.dumps({"rimosso" if what == "remove" else "attivo": ns.plan_id}))
            return 0
        if what == "compare":
            return self._compare(ns, doc, chunk)
        if what == "add":
            if chunk is None or not ns.plan:
                raise CommandError("ppg plugin flightplan add PROJECT.ppg --plan FILE")
            id_ = project.add_plan(chunk, project.standalone_load(ns.plan))
            self._save(doc)
            print(json.dumps({"id": id_}))
            return 0
        piano, id_ = self._plan_of(ns, chunk)
        if what == "show":
            from .export_formats import terrain_note

            out = pl.compute(piano)
            note = terrain_note(piano)
            if note:
                out["nota"] = note
            print(json.dumps(out, indent=1, ensure_ascii=False))
            return 0
        if what == "terrain":
            return self._terrain(ns, doc, chunk, piano, id_)
        if what == "targets":
            return self._targets(ns, doc, chunk, piano, id_)
        if what == "coverage":
            if id_ is None:
                raise CommandError("the coverage is kept in the project: give it, without --plan")
            rec = project.coverage_for(chunk, id_, self._progress(), ns.cell, ns.low)
            self._save(doc)
            print(json.dumps(rec, indent=1, ensure_ascii=False))
            return 0
        if what == "export":
            return self._export(ns, doc, chunk, piano, id_)
        if what == "djifly":
            return self._djifly(ns, doc, chunk, piano, id_)
        raise CommandError(f"unknown: {what}")

    @staticmethod
    def _progress():
        try:
            from pyarchinit_photogrammetry.core.progress import print_progress

            return print_progress
        except ImportError:  # pragma: no cover - every program the plugin runs in has it
            return None

    @staticmethod
    def _cameras() -> int:
        from .camera import CAMERAS

        print(f"{'id':<16} {'nome':<44} {'larghezza':>9} {'altezza':>8} {'focale':>7} {'pixel':>6} {'alta':>6}  gruppo")
        for c in CAMERAS:
            print(f"{c.id:<16} {c.name:<44} {c.width_mm:>9g} {c.height_mm:>8g} {c.focal_mm:>7g} {c.pixels:>6} "
                  f"{c.pixels_high or '-':>6}  {c.group}")
        return 0

    @staticmethod
    def _numbers(text: str, n: int | None, what: str) -> list[float]:
        parts = [v for v in re.split(r"[,\s]+", text.strip()) if v]
        try:
            values = [float(v) for v in parts]
        except ValueError:
            raise CommandError(f"{what}: numbers expected, got {text!r}") from None
        if n is not None and len(values) != n:
            raise CommandError(f"{what}: {n} numbers expected, got {len(values)}")
        return values

    def _pairs(self, text: str, what: str) -> list[tuple[float, float]]:
        out = []
        for part in text.split(";"):
            if part.strip():
                a, b = self._numbers(part, 2, what)
                out.append((a, b))
        if len(out) < 3:
            raise CommandError(f"{what}: at least three points are needed")
        return out

    def _area(self, ns, chunk) -> list:
        from . import project
        from .camera import camera
        from .export_dji import read_file

        given = [x for x in (ns.area, ns.bbox, ns.coords, ns.xy) if x]
        if len(given) != 1:
            raise CommandError("give the area once: --area FILE, --bbox W,S,E,N, --coords \"lat,lon;...\" "
                               "or --xy \"x,y;...\"")
        if ns.bbox:
            w, s, e, n = self._numbers(ns.bbox, 4, "--bbox")
            if not (w < e and s < n):
                raise CommandError("--bbox: west,south,east,north with west < east and south < north")
            return [(s, w), (s, e), (n, e), (n, w)]
        if ns.coords:
            return self._pairs(ns.coords, "--coords")
        if ns.xy:
            if chunk is None:
                raise CommandError("--xy is in the chunk's system: give the project")
            return project.to_wgs84(chunk, self._pairs(ns.xy, "--xy"))
        path = Path(ns.area)
        imported = read_file(path.name, path.read_bytes(), camera(ns.camera))
        areas = [a for a in imported.areas if len(a) >= 3]
        if not areas:
            raise CommandError(f"{path.name}: no area (polygon) in the file")
        return [tuple(p) for p in areas[0]]

    def _camera(self, ns):
        """(camera id of the list, own camera or None)."""
        from .camera import CAMERAS, camera_from_exif, manual_camera

        if ns.exif and ns.manual:
            raise CommandError("--exif or --manual, not both")
        if ns.exif:
            return None, camera_from_exif(ns.exif)
        if ns.manual:
            parts = [v.strip() for v in ns.manual.split(",")]
            if len(parts) != 5:
                raise CommandError("--manual NAME,WIDTH_MM,HEIGHT_MM,FOCAL_MM,PIXELS")
            name, *values = parts
            w, h, f = self._numbers(",".join(values[:3]), 3, "--manual")
            return None, manual_camera(name, w, h, f, int(self._numbers(values[3], 1, "--manual")[0]))
        if ns.camera not in {c.id for c in CAMERAS}:
            raise CommandError(f"unknown camera {ns.camera!r}: see ppg plugin flightplan cameras")
        return ns.camera, None

    def _design(self, ns, doc, chunk) -> int:
        from . import design as design_mod
        from . import project

        if chunk is None and not ns.plan:
            raise CommandError("the plan goes into the project or into a file: give PROJECT.ppg or --plan FILE")
        area = self._area(ns, chunk)
        cam_id, own = self._camera(ns)
        takeoff = tuple(self._numbers(ns.takeoff, 2, "--takeoff")) if ns.takeoff else None
        goal = design_mod.goal(ns.goal)
        value = ns.value if ns.value is not None else goal["valore"]
        dem = project.chunk_dem(chunk) if ns.terrain and chunk is not None else None
        try:
            out = design_mod.design(area, ns.goal, value, camera=cam_id or "mini4pro", high=bool(ns.high_res),
                                    battery_minutes=ns.battery, follow_terrain=ns.terrain, mode=ns.mode,
                                    takeoff=takeoff, name=ns.name or t("Piano progettato", "Designed plan"), dem=dem,
                                    margin_m=ns.margin, double_grid=ns.double, obliques=ns.obliques,
                                    custom_camera=own)
        finally:
            if dem is not None:
                dem.close()
        piano = out["piano"]
        if chunk is not None and project.has_crs(chunk):
            piano["crs"] = str(chunk.crs)          # the system the area was given in, for display only
        result = {"quota": out["quota"], "gsd": out["gsd"], "totale": out["conti"]["totale"], "motivi": out["motivi"]}
        if chunk is not None:
            result["id"] = project.add_plan(chunk, piano)
            self._save(doc)
        if ns.plan:
            project.standalone_save(ns.plan, piano)
            result["file"] = str(ns.plan)
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0

    def _terrain(self, ns, doc, chunk, piano, id_) -> int:
        from . import design as design_mod
        from . import project, tiles

        if ns.off:
            out = design_mod.remove_terrain(piano)
        else:
            if ns.detail:
                piano["dettaglioTerreno"] = ns.detail
            dem = None if ns.tiles else project.chunk_dem(chunk)
            if dem is None and not ns.tiles:
                # the network only when asked: no DEM, and the tiles not chosen
                raise CommandError("the chunk has no DEM: --tiles to use the public terrain tiles (network)")
            try:
                out = design_mod.apply_terrain(piano, z=tiles.zoom_for_detail(piano.get("dettaglioTerreno")),
                                               dem=dem)
            finally:
                if dem is not None:
                    dem.close()
            out["avvisi"] = [dict(w, testo=warning_text(w.get("tipo", ""))) for w in out.get("avvisi") or []]
        self._keep(ns, doc, chunk, piano, id_)
        print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
        return 0

    def _targets(self, ns, doc, chunk, piano, id_) -> int:
        from . import design as design_mod
        from . import plan as pl
        from . import project, targets

        if ns.remove:
            targets.remove_targets(piano)
            self._keep(ns, doc, chunk, piano, id_)
            print(json.dumps({"target": 0}))
            return 0
        if bool(ns.csv) == bool(ns.markers):
            raise CommandError("give --csv FILE (with --epsg N) or --markers, or --remove")
        if ns.markers:
            if chunk is None:
                raise CommandError("--markers needs the project")
            read = targets.targets_from_markers(chunk, only_with_reference=not ns.estimated)
        else:
            read = targets.read_targets_csv(Path(ns.csv).read_text(encoding="utf-8-sig"), near=pl.site_centre(piano),
                                            epsg=ns.epsg)
        targets.put_targets(piano, read, metres=ns.metres)
        if ns.tiles:
            piano[design_mod.TILES_CHOSEN] = True
        notes = [warning_text(w) for w in read.get("avvisi") or []]
        if pl.specials_need_ground(piano):
            dem = project.chunk_dem(chunk)
            if dem is not None:
                dem.close()
            missing = design_mod.ground_source_missing(piano, dem is not None)
            if missing:
                notes.append(missing + " (--tiles)")
        self._keep(ns, doc, chunk, piano, id_)
        print(json.dumps({"target": len(read["punti"]), "crs": read.get("crs"), "avvisi": notes}, indent=1,
                         ensure_ascii=False))
        return 0

    def _export(self, ns, doc, chunk, piano, id_) -> int:
        from . import project

        missions = [m.strip() for m in ns.missions.split(",") if m.strip()] if ns.missions else None
        if id_ is not None:
            before = project.entry(chunk, id_)["stamp"]
            files = project.export(chunk, id_, ns.format, ns.out, missions=missions)
            if project.entry(chunk, id_)["stamp"] != before:
                self._save(doc)               # the terrain heights were redone
        else:
            dem = project.chunk_dem(chunk)
            try:
                fresh, outcome = project.prepare(piano, dem)
            finally:
                if dem is not None:
                    dem.close()
            if outcome.get("rifatti") or outcome.get("suolo"):
                project.standalone_save(ns.plan, fresh)
            ground = project.takeoff_ground(chunk, fresh) if chunk is not None and fresh.get("quotaDecollo") is None \
                else None
            crs = str(chunk.crs) if chunk is not None and project.has_crs(chunk) else None
            files = project.export_plan(fresh, ns.format, ns.out, crs=crs, ground_takeoff=ground,
                                        dem_file=project.dem_name(chunk) if ground is not None else "",
                                        missions=missions)
        out = {"file": [str(f) for f in files]}
        if getattr(files, "replaced", None):
            out["sostituiti"] = [str(f) for f in files.replaced]      # an earlier export with the same names
        print(json.dumps(out, indent=1))
        return 0

    def _compare(self, ns, doc, chunk) -> int:
        from . import compare, project

        if chunk is None:
            raise CommandError("give the project: ppg plugin flightplan compare PROJECT.ppg [--plan-id ID]")
        e = project.entry(chunk, ns.id) if ns.id else project.active_plan(chunk)
        if e is None:
            raise CommandError(f"no plan {ns.id!r}: see ppg plugin flightplan list" if ns.id else
                               "the chunk has no flight plan: ppg plugin flightplan design PROJECT ...")
        if ns.max_distance is not None and not ns.max_distance > 0:
            raise CommandError("--max-distance: a positive number of metres")
        rec = compare.compare_for(chunk, e["id"], ns.max_distance, self._progress())
        self._save(doc)
        print(json.dumps(dict(rec, id=e["id"], riassunto=compare.summary_line(rec)), indent=1, ensure_ascii=False))
        return 0

    def _djifly(self, ns, doc, chunk, piano, id_) -> int:
        from . import djifly, export_dji, legmap, project

        if bool(ns.folder) == bool(ns.zip):
            raise CommandError("give --folder PATH (the waypoint folder) or --zip OUT")
        containers = [m.strip() for m in ns.missions.split(",") if m.strip()]
        dem = project.chunk_dem(chunk)
        try:
            fresh, outcome = project.prepare(piano, dem)
        finally:
            if dem is not None:
                dem.close()
        if ns.zip:
            legs = export_dji.kmz_per_leg(fresh)
            background = legmap.orthophoto_background(chunk) if chunk is not None else None
            previews = [legmap.dji_preview(fresh, k, background=background) for k in range(len(legs))]
            Path(ns.zip).parent.mkdir(parents=True, exist_ok=True)
            Path(ns.zip).write_bytes(djifly.zip_containers(legs, containers, previews))
            if outcome.get("rifatti") or outcome.get("suolo"):
                self._keep(ns, doc, chunk, fresh, id_)
            print(json.dumps({"file": [str(ns.zip)]}, indent=1))
            return 0
        backup = (project.folder(chunk) / "djifly_backup") if chunk is not None and id_ is not None else \
            Path(ns.plan).resolve().parent / "djifly_backup"
        done = copy_legs(chunk, fresh, ns.folder, containers, backup, ns.overwrite)
        self._keep(ns, doc, chunk, fresh, id_)
        print(json.dumps(done, indent=1, ensure_ascii=False))
        return 0
