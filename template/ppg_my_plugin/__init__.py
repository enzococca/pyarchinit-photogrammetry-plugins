"""A template of a pyarchinit-photogrammetry plugin: copy the folder, rename the package and the
entry point, and replace what the plugin does.

It shows what a plugin can add:

* two menu actions (*Plugin > Il mio plugin*): a summary of the active chunk in a message, and the
  statistics of the DEM computed in the background with the progress bar of the window;
* a command of the command line: ``ppg plugin chunk-summary PROJECT [--chunk LABEL]``;
* an export format: *File > Esporta > Riepilogo del chunk (Markdown)...*;
* settings kept between sessions (how many times the summary was shown).
"""
from __future__ import annotations

from pyarchinit_photogrammetry.plugins import Plugin

__version__ = "0.1.0"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:  # the program without its interface
        return it


def summary(chunk) -> str:
    """Photos, aligned photos, markers and products of ``chunk``, in Markdown."""
    products = [name for name in sorted(chunk.products) if chunk.product_path(name).exists()]
    return "\n".join([
        f"# {chunk.label}", "",
        f"- {t('foto', 'photos')}: {len(chunk.cameras)}",
        f"- {t('allineate', 'aligned')}: {len(chunk.aligned_cameras)}",
        f"- {t('marker', 'markers')}: {len(chunk.markers)}",
        f"- {t('sistema di riferimento', 'reference system')}: {chunk.crs or t('locale', 'local')}",
        f"- {t('prodotti', 'products')}: {', '.join(products) or '-'}", ""])


def dem_statistics(path, progress=None) -> dict:
    """Minimum, maximum and mean height of a DEM, read in blocks (a DEM can be larger than the
    memory). ``progress(fraction, message)`` raises ``Cancelled`` when the user cancels."""
    import numpy as np
    import rasterio

    progress = progress or (lambda *a: None)
    low, high, total, count = np.inf, -np.inf, 0.0, 0
    with rasterio.open(path) as src:
        windows = [w for _ij, w in src.block_windows(1)]
        for i, window in enumerate(windows):
            z = src.read(1, window=window, masked=True).compressed()
            if z.size:
                low, high = min(low, float(z.min())), max(high, float(z.max()))
                total, count = total + float(z.sum(dtype=np.float64)), count + int(z.size)
            if i % 16 == 0:
                progress(i / len(windows), t("lettura del DEM", "reading the DEM"))
    progress(1.0, t("statistiche del DEM", "DEM statistics"))
    if not count:
        return {"cells": 0}
    return {"cells": count, "min": low, "max": high, "mean": total / count}


class MyPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Il mio plugin", "My plugin")
        self.description = t("Riepilogo del chunk e statistiche del DEM", "Chunk summary and DEM statistics")
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Riepilogo del chunk", "Chunk summary"), self.show_summary)
        context.add_action(t("Statistiche del DEM", "DEM statistics"), self.show_dem_statistics, processing=True)
        context.add_command("chunk-summary", self.command, help="summary of a chunk of a project")
        context.add_exporter(t("Riepilogo del chunk (Markdown)", "Chunk summary (Markdown)"), ".md", self.export)

    # ------------------------------------------------------------------ window
    def _chunk(self):
        doc = self.context.app.document
        return None if doc is None else doc.chunk

    def show_summary(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        chunk = self._chunk()
        if chunk is None:
            return
        settings = self.context.settings
        settings["shown"] = settings.get("shown", 0) + 1
        self.context.save_settings()
        message(self.context.window, "info", self.title, summary(chunk))

    def show_dem_statistics(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        chunk = self._chunk()
        if chunk is None or chunk.dem_path is None or not chunk.dem_path.exists():
            message(self.context.window, "warning", self.title, t("Il chunk non ha un DEM.", "The chunk has no DEM."))
            return
        path = chunk.dem_path

        def done(stats):  # in the interface thread: widgets can be touched here
            if not stats.get("cells"):
                text = t("Il DEM e' vuoto.", "The DEM is empty.")
            else:
                text = (f"{t('celle', 'cells')}: {stats['cells']}\nmin {stats['min']:.3f} m\n"
                        f"max {stats['max']:.3f} m\n{t('media', 'mean')} {stats['mean']:.3f} m")
            message(self.context.window, "info", self.title, text)

        # dem_statistics runs in the background: it must not touch the window
        self.context.run(t("Statistiche del DEM", "DEM statistics"), lambda progress: dem_statistics(path, progress),
                         done)

    def export(self, chunk, path, progress) -> None:
        """File > Esporta: runs in the background, so it only reads the chunk and writes ``path``."""
        progress(0.5, t("riepilogo", "summary"))
        path.write_text(summary(chunk), encoding="utf-8")

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        import pyarchinit_photogrammetry as ppg

        parser = argparse.ArgumentParser(prog="ppg plugin chunk-summary", description="Summary of a chunk.")
        parser.add_argument("project", help="the .ppg project")
        parser.add_argument("--chunk", help="label of the chunk (the active one by default)")
        a = parser.parse_args(list(args))
        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}")
            return 2
        print(summary(chunk))
        return 0
