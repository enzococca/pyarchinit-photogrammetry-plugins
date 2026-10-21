"""Powerlines - a plugin of pyarchinit-photogrammetry: the conductors of overhead lines in the
dense cloud as catenary curves, the poles, the clearance of every conductor from the ground and
its distance from the nearest obstacle (vegetation, buildings).

* *Plugin > Linee elettriche > Rileva linee elettriche...* - the detection on the dense cloud
  of the active chunk, in the background; the conductors and the poles drawn on the orthomosaic
  and DEM views, the wire points in class 14 and the pole points in class 15;
* *Plugin > Linee elettriche > Mostra / Elenco / Dimentica* - the layers again, the table, the
  removal;
* *File > Esporta* - GeoPackage (3D lines and points), DXF (3D polylines), CSV;
* ``ppg plugin powerlines detect|list|export|clear ...`` - the same from the command line.

The detection works on the points: the cables must be in the dense cloud (close flights,
high quality depth maps). See :mod:`ppg_powerlines.detect`.
"""
from __future__ import annotations

from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import t

__version__ = "0.1.0"


class PowerlinesPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Linee elettriche", "Powerlines")
        self.description = t("Conduttori delle linee aeree nella nuvola densa come catenarie, pali, franco dal suolo "
                             "e distanza dagli ostacoli",
                             "Conductors of overhead lines in the dense cloud as catenaries, poles, clearance from "
                             "the ground and distance from obstacles")
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Rileva linee elettriche...", "Detect powerlines..."), self.detect_dialog, processing=True,
                           tip=t("Conduttori, pali, franco dal suolo e ostacoli dalla nuvola densa",
                                 "Wires, poles, clearance and obstacles from the dense cloud"))
        context.add_action(t("Mostra le linee elettriche", "Show the powerlines"), self.show,
                           tip=t("I layer Linee elettriche e Pali sulle viste dell'ortofoto e del DEM",
                                 "The layers Linee elettriche and Pali on the orthomosaic and DEM views"))
        context.add_action(t("Elenco delle linee elettriche...", "List of the powerlines..."), self.list_dialog)
        context.add_action(t("Dimentica le linee elettriche", "Forget the powerlines"), self.clear_dialog, processing=True,
                           tip=t("Toglie il risultato e le classi 14 e 15 date dal plugin",
                                 "Removes the result and the classes 14 and 15 given by the plugin"))
        context.add_command("powerlines", self.command,
                            help="overhead powerlines in the dense cloud: wires as catenaries, poles, clearance")
        context.add_exporter(t("Linee elettriche (GeoPackage)", "Powerlines (GeoPackage)"), ".gpkg", self.export)
        context.add_exporter(t("Linee elettriche (DXF)", "Powerlines (DXF)"), ".dxf", self.export)
        context.add_exporter(t("Linee elettriche (CSV)", "Powerlines (CSV)"), ".csv", self.export)

    # ------------------------------------------------------------------ exporters
    def export(self, chunk, path, progress):
        from .export import export

        return export(chunk, path, progress)

    # ------------------------------------------------------------------ window
    def _chunk(self):
        doc = self.context.app.document
        return None if doc is None else doc.chunk

    def detect_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        from . import detect as detect_mod
        from .export import draw, summary

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        if chunk.point_cloud is None:
            message(window, "warning", self.title, t("Il chunk non ha una nuvola densa.", "The chunk has no dense cloud."))
            return
        s = self.context.settings
        d = detect_mod.DEFAULTS
        v = ParamDialog.ask(window, t("Rileva linee elettriche", "Detect powerlines"), [
            Field("min_height", t("Altezza minima dal suolo (m)", "Minimum height above the ground (m)"),
                  float(s.get("min_height", d["min_height"])), minimum=0.5, maximum=200.0, decimals=1,
                  tip=t("Sotto questa altezza non si cercano cavi", "No cable is looked for below this height")),
            Field("radius", t("Raggio dell'intorno (m)", "Neighbourhood radius (m)"), float(s.get("radius", d["radius"])),
                  minimum=0.2, maximum=10.0, decimals=2,
                  tip=t("I punti vicini danno la direzione: meno della distanza fra conduttori paralleli",
                        "The nearest points give the direction: less than the distance between parallel conductors")),
            Field("linearity", t("Linearita' minima (0-1)", "Minimum linearity (0-1)"),
                  float(s.get("linearity", d["linearity"])), minimum=0.0, maximum=1.0, decimals=2),
            Field("gap", t("Distanza massima fra punti di un cavo (m)", "Largest distance between points of a cable (m)"),
                  float(s.get("gap", d["gap"])), minimum=0.1, maximum=50.0, decimals=1),
            Field("min_length", t("Lunghezza minima di un cavo (m)", "Shortest cable (m)"),
                  float(s.get("min_length", d["min_length"])), minimum=0.5, maximum=1000.0, decimals=1),
            Field("min_points", t("Punti minimi di un cavo", "Fewest points of a cable"),
                  int(s.get("min_points", d["min_points"])), minimum=3, maximum=100000),
            Field("merge_gap", t("Buco massimo da scavalcare lungo un cavo (m)", "Largest gap bridged along a cable (m)"),
                  float(s.get("merge_gap", d["merge_gap"])), minimum=0.0, maximum=500.0, decimals=1,
                  tip=t("Tratti dello stesso cavo separati da un buco della ricostruzione vengono uniti",
                        "Pieces of one cable separated by a gap of the reconstruction are joined")),
            Field("classify", t("Classi 14 (conduttori) e 15 (pali) nella nuvola", "Classes 14 (wires) and 15 (poles) in the cloud"),
                  bool(s.get("classify", True))),
        ], intro=t("Cerca nella nuvola densa i cavi sospesi: punti alti, allineati, con l'aria libera sotto e di fianco. "
                   "Ogni conduttore diventa una catenaria con lunghezza, freccia, franco dal suolo e distanza "
                   "dall'ostacolo piu' vicino; i pali dove i conduttori terminano insieme. Terreno dai punti di classe 2 "
                   "se ci sono, altrimenti dal filtro morfologico.",
                   "Looks in the dense cloud for hanging cables: high points, aligned, with free air under and beside "
                   "them. Every conductor becomes a catenary with length, sag, clearance from the ground and distance "
                   "from the nearest obstacle; the poles where conductors end together. Ground from the class 2 points "
                   "when there are some, else from the morphological filter."))
        if v is None:
            return
        s.update(v)
        self.context.save_settings()
        params = {k: v[k] for k in ("min_height", "radius", "linearity", "gap", "min_length", "min_points", "merge_gap")}

        def job(progress):
            return detect_mod.detect_chunk(chunk, classify=v["classify"], progress=progress, **params)

        def done(report):
            draw(window, report)
            layers = getattr(window, "layers_pane", None)
            if layers is not None and hasattr(layers, "refresh"):
                layers.refresh()
            message(window, "info", self.title, summary(report))

        window.run_task(t("Linee elettriche", "Powerlines"), job, done, refresh=True, save=True)

    def show(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        from .export import draw

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        report = chunk.meta.get("powerlines")
        if not report:
            message(window, "info", self.title, t("Nessuna linea elettrica in questo chunk: prima Rileva linee elettriche.",
                                                  "No powerlines in this chunk: first Detect powerlines."))
            return
        draw(window, report)

    def list_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        from .export import table

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        report = chunk.meta.get("powerlines")
        if not report:
            message(window, "info", self.title, t("Nessuna linea elettrica in questo chunk.", "No powerlines in this chunk."))
            return
        message(window, "info", self.title, table(report))

    def clear_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import choice

        from .detect import clear_chunk
        from .export import draw

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not chunk.meta.get("powerlines"):
            return
        if choice(window, "question", self.title,
                  t("Togliere le linee elettriche trovate e le classi 14 e 15 date dal plugin?",
                    "Remove the powerlines found and the classes 14 and 15 given by the plugin?"),
                  [t("Togli", "Remove"), t("Annulla", "Cancel")]) != 0:
            return
        draw(window, None)
        window.run_task(t("Linee elettriche", "Powerlines"), lambda progress: clear_chunk(chunk), None, refresh=True,
                        save=True)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        from . import detect as detect_mod

        parser = argparse.ArgumentParser(prog="ppg plugin powerlines",
                                         description="Overhead powerlines in the dense cloud: wires as catenaries, "
                                                     "poles, clearance from the ground, distance from obstacles.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("detect", help="find the wires and the poles of the dense cloud")
        p.add_argument("project")
        p.add_argument("--chunk", help="label of the chunk (the active one by default)")
        d = detect_mod.DEFAULTS
        p.add_argument("--min-height", type=float, default=d["min_height"], help="m above the ground")
        p.add_argument("--radius", type=float, default=d["radius"], help="m: neighbourhood giving the direction")
        p.add_argument("--linearity", type=float, default=d["linearity"])
        p.add_argument("--gap", type=float, default=d["gap"], help="m between linked points of one wire")
        p.add_argument("--min-length", type=float, default=d["min_length"])
        p.add_argument("--min-points", type=int, default=d["min_points"])
        p.add_argument("--merge-gap", type=float, default=d["merge_gap"], help="m: gaps bridged along a wire")
        p.add_argument("--no-classify", action="store_true", help="leave the classes of the cloud as they are")
        p.add_argument("--export", help="also write .gpkg, .dxf or .csv")
        p = sub.add_parser("list", help="the wires and the poles found")
        p.add_argument("project")
        p.add_argument("--chunk")
        p = sub.add_parser("export", help="the wires and the poles as .gpkg, .dxf or .csv")
        p.add_argument("project")
        p.add_argument("output")
        p.add_argument("--chunk")
        p = sub.add_parser("clear", help="forget the powerlines and the classes 14 and 15 of the plugin")
        p.add_argument("project")
        p.add_argument("--chunk")
        a = parser.parse_args(list(args))
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        from .export import export, table

        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}" if a.chunk else f"no active chunk in {a.project}")
            return 2
        try:
            if a.what == "detect":
                report = detect_mod.detect_chunk(chunk, classify=not a.no_classify, progress=print_progress,
                                                 min_height=a.min_height, radius=a.radius, linearity=a.linearity,
                                                 gap=a.gap, min_length=a.min_length, min_points=a.min_points,
                                                 merge_gap=a.merge_gap)
                doc.save()
                print(table(report))
                if a.export:
                    print(export(chunk, Path(a.export)))
            elif a.what == "list":
                print(table(chunk.meta["powerlines"]) if chunk.meta.get("powerlines") else "no powerlines: detect first")
            elif a.what == "export":
                print(export(chunk, Path(a.output)))
            else:
                had = detect_mod.clear_chunk(chunk)
                doc.save()
                print("powerlines forgotten" if had else "no powerlines in this chunk")
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        return 0
