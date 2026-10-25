"""GeoAI: supervised segmentation from reviewed archaeological drawings."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin
from . import environment
from .texts import t

from . import dataset, results

__version__ = "0.1.0"


class GeoaiPlugin(Plugin):
    title = "GeoAI"
    version = __version__
    description = "Learn classes from reviewed unit drawings and apply them as proposals"

    def __init__(self):
        self.context = None
        self.pane = None
        self._hooks = set()

    def activate(self, context):
        self.context = context
        context.add_action(t("Pannello GeoAI", "GeoAI pane"), self.show_pane)
        context.add_action(t("Installa GeoAI...", "Install GeoAI..."), self.install, processing=True)
        context.add_command("geoai", self.command, help="prepare, train and apply supervised segmentation models")
        if context.window is not None:
            from .pane import GeoaiPane

            self.pane = GeoaiPane(self)
            context.add_pane(self.pane, "GeoAI", "sam", area="right")
            context.on_refresh(self.refresh)
            context.on_raster_loaded(self.draw_all)
            self.refresh()

    def deactivate(self):
        view = getattr(self.context.window, "ortho_view", None)
        if view is not None:
            for name in list(view.layers):
                if name.startswith("geoai: "):
                    view.remove_layer(name)
        self.pane = None
        self._hooks.clear()

    def chunk(self):
        doc = self.context.app.document
        return doc.chunk if doc is not None else None

    def show_pane(self):
        if self.pane is not None:
            self.context.show_pane(self.pane)

    def refresh(self):
        if self.pane is not None:
            self.pane.refresh()
        for run in results.records(self.chunk()):
            name = results.layer(run)
            if name not in self._hooks:
                self.context.on_features_deleted(name, lambda feats, rid=run["id"]: self.deleted(rid, feats))
                self._hooks.add(name)
        self.draw_all()

    def draw_all(self):
        chunk = self.chunk()
        view = getattr(self.context.window, "ortho_view", None)
        for run in results.records(chunk):
            results.draw(view, chunk, run)

    def deleted(self, rid, features):
        chunk = self.chunk()
        run = next((r for r in results.records(chunk) if r["id"] == rid), None)
        if run is not None:
            results.remove_proposals(chunk, run, {int(f.get("n", 0)) for f in features})
            self.context.mark_modified()
            self.refresh()

    def install(self):
        self.context.run(t("Installazione GeoAI", "Installing GeoAI"),
                         lambda p: environment.install(p, features=("geoai",)), lambda _: self.refresh())

    def found(self, chunk, run):
        results.add(chunk, run)
        self.context.mark_modified()
        self.refresh()

    def units(self, run):
        chunk = self.chunk()
        view = getattr(self.context.window, "ortho_view", None)
        selected = ({int(a.get("n", 0)) for lay, _, a in view.selected_features() if lay == results.layer(run)}
                    if view is not None else set())
        geoms, attrs = results.unit_rows(chunk, run, selected)
        # A proposal already converted is detected from actual drawings, so Undo permits conversion again.
        self.context.store_drawings()
        made = {(u.get("attributes") or {}).get("geoai_proposal")
                for u in (chunk.meta.get("drawings") or {}).get("units", [])
                if (u.get("attributes") or {}).get("geoai_run") == run["id"]}
        pairs = [(g, a) for g, a in zip(geoms, attrs) if a["geoai_proposal"] not in made]
        if pairs:
            self.context.add_units([g for g, a in pairs], attributes=[a for g, a in pairs], chunk=chunk)

    def command(self, *args):
        from .client import run as job

        p = argparse.ArgumentParser(prog="ppg plugin geoai")
        sub = p.add_subparsers(dest="op", required=True)
        sub.add_parser("install")
        sub.add_parser("status")
        prep = sub.add_parser("prepare", help="prepare RGB tiles from reviewed drawings")
        prep.add_argument("project")
        prep.add_argument("config", help="JSON: field, mapping, reviewed_area WKT, optional kinds and tile_size")
        prep.add_argument("output")
        prep.add_argument("--chunk")
        train = sub.add_parser("train")
        train.add_argument("dataset")
        train.add_argument("output", help="new model directory, conventionally NAME.ppgseg")
        train.add_argument("--epochs", type=int, default=20)
        train.add_argument("--batch-size", type=int, default=2)
        train.add_argument("--device", choices=["cpu", "cuda", "mps"])
        apply = sub.add_parser("apply", help="apply a model and save reviewable proposals")
        apply.add_argument("project")
        apply.add_argument("model")
        apply.add_argument("--chunk")
        apply.add_argument("--min-area", type=float, default=0.0)
        apply.add_argument("--device", choices=["cpu", "cuda", "mps"])
        export = sub.add_parser("export")
        export.add_argument("project")
        export.add_argument("run")
        export.add_argument("output")
        export.add_argument("--chunk")
        ns = p.parse_args([str(a) for a in args])
        try:
            if ns.op == "install":
                print(environment.install(features=("geoai",)))
                return 0
            if ns.op == "status":
                print(environment.status_text("geoai"))
                return 0 if environment.where("geoai") else 1
            if ns.op == "train":
                result = job({"op": "train", "dataset": ns.dataset, "output": ns.output, "epochs": ns.epochs,
                              "batch_size": ns.batch_size, "device": ns.device})
                print(json.dumps(result["metrics"], indent=2))
                return 0
            import pyarchinit_photogrammetry as ppg

            doc = ppg.Document.open(ns.project)
            chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
            if chunk is None:
                raise ValueError("No matching chunk")
            if ns.op == "prepare":
                cfg = json.loads(Path(ns.config).read_text(encoding="utf-8"))
                manifest = dataset.prepare(chunk.orthomosaic_path, (chunk.meta.get("drawings") or {}).get("units", []),
                                           cfg["field"], cfg["mapping"], cfg["reviewed_area"], ns.output,
                                           tile_size=cfg.get("tile_size", 512), kinds=cfg.get("kinds"),
                                           validation_fraction=cfg.get("validation_fraction", 0.2))
                print(f"{len(manifest['samples'])} tiles: {ns.output}")
            elif ns.op == "export":
                run = next((r for r in results.records(chunk) if r["id"] == ns.run), None)
                if run is None:
                    raise ValueError("No matching run")
                target = Path(ns.output)
                with target.open("xb") as stream, results.path(chunk, run, "gpkg").open("rb") as source:
                    shutil.copyfileobj(source, stream)
                print(target)
            else:
                run = results.predict(chunk, chunk.orthomosaic_path, ns.model, ns.min_area, ns.device)
                results.add(chunk, run)
                doc.save()
                print(json.dumps(run, indent=2))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"GeoAI: {exc}")
            return 1
        return 0
