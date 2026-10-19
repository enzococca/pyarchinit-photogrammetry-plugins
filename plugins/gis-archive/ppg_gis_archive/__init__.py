"""Archivio GIS - a plugin of pyarchinit-photogrammetry: the survey of a project (spot heights,
contour lines, stratigraphic units, sections, stones, markers, DEM and orthomosaic) into a GIS
database outside PyArchInit: a GeoPackage, a PostgreSQL/PostGIS database or a SpatiaLite file.

* *Plugin > Archivio GIS > Salva in un database GIS...* - choose the database and what to save;
* *File > Esporta > Archivio GIS (GeoPackage)...* - everything into a GeoPackage at once;
* ``ppg plugin gis-archive PROJECT TARGET [--what ...]`` - the same from the command line.
"""
from __future__ import annotations

from pyarchinit_photogrammetry.plugins import Plugin

__version__ = "0.1.0"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:
        return it


class GisArchivePlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Archivio GIS", "GIS archive")
        self.description = t("Quote, curve, US, sezioni, pietre, marker, DEM e ortofoto in un GeoPackage, "
                             "PostgreSQL/PostGIS o SpatiaLite",
                             "Spot heights, contours, units, sections, stones, markers, DEM and orthomosaic into a "
                             "GeoPackage, PostgreSQL/PostGIS or SpatiaLite database")
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Salva in un database GIS...", "Save into a GIS database..."), self.open_dialog,
                           processing=True,
                           tip=t("Quote, curve, US, sezioni, pietre, marker e raster in GeoPackage, PostGIS o SpatiaLite",
                                 "Spot heights, contours, units, sections, stones, markers and rasters into "
                                 "GeoPackage, PostGIS or SpatiaLite"))
        context.add_command("gis-archive", self.command,
                            help="archive a project into a GeoPackage, PostgreSQL/PostGIS or SpatiaLite database")
        context.add_exporter(t("Archivio GIS (GeoPackage)", "GIS archive (GeoPackage)"), ".gpkg", self.export_gpkg)

    # ------------------------------------------------------------------ window
    def export_gpkg(self, chunk, path, progress):
        """File > Esporta: everything into a GeoPackage, rasters included."""
        from .archive import archive

        self.context.store_drawings()
        return archive(self.context.app.document, str(path), chunk=chunk, progress=progress)

    def open_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        from .archive import archive
        from .dialog import ArchiveDialog

        window = self.context.window
        doc = self.context.app.document
        if window is None or doc is None or doc.chunk is None:
            return
        dialog = ArchiveDialog(window, self.context.settings)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        options = dialog.options()
        self.context.settings.update(dialog.remembered())
        self.context.save_settings()
        self.context.store_drawings()
        chunk = doc.chunk

        def job(progress):
            return archive(doc, options["target"], what=options["what"], chunk=chunk, interval=options["interval"],
                           prefix=options["prefix"], replace=options["replace"], rasters=options["rasters"],
                           max_side=options["max_side"], password=options["password"], progress=progress)

        def done(report):
            message(window, "info", self.title, summary(report))

        self.context.run(t("Archivio GIS", "GIS archive"), job, done)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        from .archive import EVERYTHING, ArchiveError, archive

        parser = argparse.ArgumentParser(prog="ppg plugin gis-archive",
                                         description="Archive the survey of a project into a GIS database.")
        parser.add_argument("project", help="the .ppg project")
        parser.add_argument("target", help="a .gpkg or .sqlite file, or postgresql://user@host:5432/db?schema=name "
                                           "(password in the URL or in PGPASSWORD)")
        parser.add_argument("--what", default=",".join(EVERYTHING), help="what to save, among "
                                                                        + ",".join(EVERYTHING))
        parser.add_argument("--chunk", help="label of the chunk (the active one by default)")
        parser.add_argument("--interval", type=float, help="interval of the contour lines in metres "
                                                           "(suggested from the DEM by default)")
        parser.add_argument("--prefix", default="ppg_", help="prefix of the table names")
        parser.add_argument("--append", action="store_true", help="add the rows instead of replacing those of "
                                                                  "this project and chunk")
        parser.add_argument("--rasters", choices=["copy", "reference"], default="copy",
                            help="copy the rasters into the database, or record file and footprint only")
        parser.add_argument("--max-side", type=int, default=0, help="largest side of the raster copies in pixels "
                                                                    "(0: full resolution)")
        a = parser.parse_args(list(args))
        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}")
            return 2
        try:
            report = archive(doc, a.target, what=[w.strip() for w in a.what.split(",") if w.strip()], chunk=chunk,
                             interval=a.interval, prefix=a.prefix, replace=not a.append, rasters=a.rasters,
                             max_side=a.max_side, progress=print_progress)
        except ArchiveError as exc:
            print(f"error: {exc}")
            return 1
        print(summary(report))
        return 0


def summary(report: dict) -> str:
    """What was written, in words."""
    lines = [t(f"Archivio {report['kind']}: {report['target']}", f"{report['kind']} archive: {report['target']}")]
    for table, rows in report["tables"].items():
        lines.append(f"  {table}: {rows} " + t("righe", "rows"))
    for r in report["rasters"]:
        where = (t("tabella ", "table ") + r["table"]) if r["table"] else t("solo riferimento", "reference only")
        lines.append(f"  {r['kind']}: {where}")
    lines += [t("Attenzione: ", "Note: ") + w for w in report["warnings"]]
    return "\n".join(lines)
