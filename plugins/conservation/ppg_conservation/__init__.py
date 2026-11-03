"""Conservation (Conservazione) - a plugin of pyarchinit-photogrammetry: decay mapping on the program's elevations.

* the vocabulary of the standards (:mod:`.vocabulary`): ICOMOS-ISCS 2008 terms as the codes, UNI 11182:2006 as their
  Italian face, the plugin's local masonry terms, the interventions after EN 15898; colours and hatches the plugin's,
  changed per project;
* campaigns and features on a face (:mod:`.records`), kept in ``chunk.meta["conservation"]``; the EN 16096 classes
  chosen by the surveyor; the world geometry of a feature lifted on the relief (:mod:`.geometry`);
* the *Conservazione* pane (on the right), ``ppg plugin conservation list``, the plate of a campaign among the exports.

The plugin writes only ``chunk.meta["conservation"]`` and the files under ``<chunk data>/conservation/<eid>/``.
"""
from __future__ import annotations

import argparse

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, t

__version__ = "0.1.0"


class ConservationPlugin(Plugin):
    name = "conservation"
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Mappatura del degrado sui prospetti: lessico ICOMOS-ISCS e UNI 11182, campagne, "
                             "classi EN 16096, interventi", "Decay mapping on elevations: ICOMOS-ISCS and UNI 11182 "
                             "vocabulary, campaigns, EN 16096 classes, interventions")
        self.context = None
        self.pane = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Conservazione...", "Conservation..."), self.show_pane,
                           tip=t("Il pannello della conservazione", "The conservation pane"))
        context.add_command("conservation", self.command,
                            help="decay mapping on elevations: list the campaigns and their features")
        context.add_exporter(t("Tavola della campagna (PDF)", "Plate of the campaign (PDF)"), ".pdf",
                             self._export_plate)
        if context.window is not None:
            from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

            self.pane = QWidget()
            self.pane.setObjectName("conservation_pane")
            layout = QVBoxLayout(self.pane)
            title = QLabel(t(*NAME))
            title.setStyleSheet("font-weight: bold")
            layout.addWidget(title)
            layout.addStretch(1)
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            context.on_refresh(self.refresh_pane)
            context.on_raster_loaded(self._raster_loaded)

    def deactivate(self) -> None:
        self.pane = None

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    def refresh_pane(self, *_args) -> None:
        """The pane follows the project (filled by the pane of the mapping)."""

    def _raster_loaded(self, *_args) -> None:
        """The features of the campaign shown are drawn again on the new raster (with the pane of the mapping)."""

    # ------------------------------------------------------------------ exports
    def _export_plate(self, chunk, path, progress=None):
        """*File > Esporta > Tavola della campagna (PDF)*."""
        from .records import NotReady

        raise NotReady(t("La tavola della campagna non è ancora disponibile in questa versione del plugin.",
                         "The plate of the campaign is not available yet in this version of the plugin."))

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import pyarchinit_photogrammetry as ppg

        from . import records
        from .vocabulary import load

        parser = argparse.ArgumentParser(prog="ppg plugin conservation", description="decay mapping on elevations")
        parser.add_argument("what", choices=["list"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        a = parser.parse_args(list(args))
        doc = ppg.Document.open(a.project)
        chunk = next((c for c in doc.chunks if c.label == a.chunk), None) if a.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {a.chunk!r}")
            return 2
        vocab = load(chunk)
        lang = "en" if t("it", "en") == "en" else "it"
        names = {e["id"]: e.get("name", e["id"]) for e in chunk.meta.get("elevations") or []}
        found = records.campaigns(chunk)
        if not found:
            print(t("Nessuna campagna.", "No campaign."))
        for c in found:
            feats = records.features(chunk, c["id"])
            print(f"{c['id']}  {c['date']}  {c['name']}  [{names.get(c['elevation'], c['elevation'])}]  "
                  f"{len(feats)}")
            for f in feats:
                label = vocab.name(f["code"], lang) if f["code"] in vocab else f["code"]
                extent = "" if f["extent"] is None else f"  {f['extent']:.3f}"
                print(f"    {f['id']}  {f['kind']:<9} {f['status']:<8} {label}{extent}")
        return 0
