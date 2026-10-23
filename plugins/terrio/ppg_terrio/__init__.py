"""terrio - a plugin of pyarchinit-photogrammetry: point clouds and rasters through terrio
(https://github.com/opengeos/terrio), geospatial I/O in pure Rust, with no GDAL or PDAL to install.

* *Plugin > terrio > Importa nuvola di punti...* - LAS, LAZ, COPC, EPT, PLY and E57 files, on
  disk or on the web (http, S3), as the dense cloud of a new chunk or of the active one;
* *Plugin > terrio > Classifica il terreno...* - ground and other points with the SMRF, CSF or
  PMF filters of PDAL (low noise with ELM, vegetation classes by the height above the ground);
* *Plugin > terrio > Raster di riferimento...* - a Cloud Optimized GeoTIFF on the web (or a
  STAC search) warped onto the survey, kept as a version of the DEM or of the orthomosaic;
* *File > Esporta* - the dense cloud as E57 or EPT, the orthomosaic as PMTiles or MBTiles;
* ``ppg plugin terrio import|ground|reference|export|install|version ...`` - the same from the
  command line.

terrio comes with pip (``pip install terrio``); the packaged application, which has no pip,
downloads its wheel from PyPI into the plugin folder (*Installa terrio*).
"""
from __future__ import annotations

from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import environment

__version__ = "0.1.2"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:  # the program without its interface
        return it


def num(value) -> str:
    """A whole number as the program writes it: 1.234 in Italian, 1,234 in English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.number(value)
    except Exception:
        return f"{value:,}"


def class_label(code: int, labels: dict | None = None) -> str:
    """The name of a class of points in the language of the program (the one the 3D view shows);
    ``labels``: the names the project gives (``chunk.meta["classes"]``)."""
    try:
        from pyarchinit_photogrammetry.gui.scene import class_name

        return class_name(int(code), labels)
    except Exception:
        from pyarchinit_photogrammetry.ai.classify import class_names

        return class_names().get(int(code), "")


METHOD_LABELS = [("smrf", "SMRF (Simple Morphological Filter)"), ("csf", "CSF (Cloth Simulation Filter)"),
                 ("pmf", "PMF (Progressive Morphological Filter)")]
CLOUD_FILTER = "Nuvole di punti (*.las *.laz *.ply *.e57 *.json *.zip);;Tutti i file (*)"


class TerrioPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "terrio"
        self.description = t("Nuvole di punti (LAS, LAZ, COPC, EPT, PLY, E57), classificazione del terreno e "
                             "raster del web con terrio, senza GDAL o PDAL",
                             "Point clouds (LAS, LAZ, COPC, EPT, PLY, E57), ground classification and web "
                             "rasters through terrio, without GDAL or PDAL")
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        environment.configure(context)
        context.add_action(t("Importa nuvola di punti...", "Import point cloud..."), self.import_dialog,
                           processing=True,
                           tip=t("LAS, LAZ, COPC, EPT, PLY o E57 (anche in uno zip), anche dal web, come nuvola densa",
                                 "LAS, LAZ, COPC, EPT, PLY or E57 (also zipped), also from the web, as the dense cloud"))
        context.add_action(t("Classifica il terreno (SMRF, CSF, PMF)...", "Classify the ground (SMRF, CSF, PMF)..."),
                           self.ground_dialog, processing=True,
                           tip=t("Terreno e altri punti della nuvola densa con i filtri di PDAL",
                                 "Ground and other points of the dense cloud with PDAL's filters"))
        context.add_action(t("Raster di riferimento dal web...", "Reference raster from the web..."),
                           self.reference_dialog, processing=True,
                           tip=t("Un COG o una ricerca STAC ritagliati sul rilievo, come versione del DEM o dell'ortofoto",
                                 "A COG or a STAC search clipped to the survey, as a version of the DEM or orthomosaic"))
        context.add_action(t("Installa terrio...", "Install terrio..."), self.install_dialog,
                           tip=t("Scarica terrio da PyPI nella cartella del plugin (per l'applicazione senza pip)",
                                 "Download terrio from PyPI into the plugin folder (for the application without pip)"))
        context.add_command("terrio", self.command,
                            help="point clouds, ground classification and web rasters through terrio")
        context.add_exporter(t("Nuvola di punti E57 (terrio)", "Point cloud E57 (terrio)"), ".e57", self.export_cloud)
        context.add_exporter(t("Nuvola di punti EPT (terrio)", "Point cloud EPT (terrio)"), ".ept", self.export_cloud)
        context.add_exporter(t("Ortofoto PMTiles (terrio)", "Orthomosaic PMTiles (terrio)"), ".pmtiles",
                             self.export_tiles)
        context.add_exporter(t("Ortofoto MBTiles (terrio)", "Orthomosaic MBTiles (terrio)"), ".mbtiles",
                             self.export_tiles)

    # ------------------------------------------------------------------ exporters
    def export_cloud(self, chunk, path, progress):
        from .clouds import export_cloud

        return export_cloud(chunk, path, progress)

    def export_tiles(self, chunk, path, progress):
        from .rasters import export_tiles

        return export_tiles(chunk, path, progress)

    # ------------------------------------------------------------------ window
    def _chunk(self):
        doc = self.context.app.document
        return None if doc is None else doc.chunk

    def _ready(self) -> bool:
        """terrio can be imported; else the window offers to download it."""
        from pyarchinit_photogrammetry.gui.dialogs import choice

        if environment.available():
            return True
        window = self.context.window
        picked = choice(window, "question", self.title,
                        t("terrio non e' installato. Scaricare terrio {0} da PyPI (circa 17 MB) nella cartella del "
                          "plugin?", "terrio is not installed. Download terrio {0} from PyPI (about 17 MB) into the "
                          "plugin folder?").format(environment.TERRIO_VERSION),
                        [t("Scarica", "Download"), t("Annulla", "Cancel")])
        if picked == 0:
            self.install_dialog()
        return False

    def _run(self, title: str, job, done=None, changes: bool = True):
        """``job(progress)`` in the background; ``changes``: the project is saved and the views
        refreshed afterwards (the plugin API's ``run`` does neither)."""
        window = self.context.window
        if window is not None and changes:
            return window.run_task(title, job, done, refresh=True, save=True)
        return self.context.run(title, job, done)

    def install_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        window = self.context.window
        found = environment.available()
        if found:
            message(window, "info", self.title, t("terrio {0} e' gia' disponibile.", "terrio {0} is already available.")
                    .format(found))
            return

        def done(folder):
            message(window, "info", self.title, t("terrio {0} installato in {1}.", "terrio {0} installed in {1}.")
                    .format(environment.TERRIO_VERSION, folder))

        self._run(t("Installazione di terrio", "Installing terrio"), lambda progress: environment.install(progress),
                  done, changes=False)

    def import_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        from .clouds import import_cloud

        window = self.context.window
        doc = self.context.app.document
        if window is None or doc is None or not self._ready():
            return
        s = self.context.settings
        v = ParamDialog.ask(window, t("Importa nuvola di punti (terrio)", "Import point cloud (terrio)"), [
            Field("file", t("File", "File"), s.get("import_file", ""), kind="file_open", filter=CLOUD_FILTER,
                  tip=t("LAS, LAZ, COPC (.copc.laz), PLY, E57, uno zip di questi o l'ept.json di un EPT", "LAS, LAZ, COPC (.copc.laz), PLY, E57, a zip of them or the ept.json of an EPT")),
            Field("url", t("Oppure URL (https, s3, gs, az)", "Or URL (https, s3, gs, az)"), "", kind="text",
                  tip=t("COPC ed EPT remoti sono letti sul posto, solo le parti che servono",
                        "Remote COPC and EPT are read in place, only the parts needed")),
            Field("where", t("Destinazione", "Destination"), "new",
                  choices=[("new", t("un nuovo chunk", "a new chunk")),
                           ("replace", t("la nuvola densa del chunk attivo (quella di ora resta fra le versioni)",
                                         "the dense cloud of the active chunk (the current one is kept as a version)"))]),
            Field("label", t("Nome del nuovo chunk (vuoto = nome del file)", "Name of the new chunk (empty: file name)"),
                  "", kind="text"),
            Field("max_points", t("Punti al massimo (0 = tutti)", "Points at most (0: all)"),
                  int(s.get("import_max_points", 0)), maximum=2_000_000_000),
            Field("classes", t("Solo le classi (es. 2,6; vuoto = tutte)", "Only these classes (e.g. 2,6; empty: all)"),
                  "", kind="text"),
            Field("crs", t("Sistema di riferimento se il file non lo dice (es. EPSG:32633)",
                           "Reference system when the file has none (e.g. EPSG:32633)"), s.get("import_crs", ""),
                  kind="text"),
        ], intro=t("Qualsiasi formato letto da terrio, anche dal web. La nuvola entra nel progetto come nuvola densa: "
                   "colori, intensita' (in grigio) e classi vengono con i punti.",
                   "Any format terrio reads, from the web too. The cloud enters the project as the dense cloud: "
                   "colours, intensity (as grey) and classes come with the points."))
        if v is None:
            return
        source = (v["url"] or "").strip() or v["file"]
        if not source:
            message(window, "warning", self.title, t("Scegli un file o un URL.", "Choose a file or a URL."))
            return
        s.update({"import_file": v["file"], "import_max_points": v["max_points"], "import_crs": v["crs"]})
        self.context.save_settings()
        target = doc.chunk if v["where"] == "replace" else None
        if v["where"] == "replace" and target is None:
            message(window, "warning", self.title, t("Nessun chunk attivo.", "No active chunk."))
            return
        classes = [int(c) for c in v["classes"].replace(";", ",").split(",") if c.strip()] or None
        holder = {}

        def job(progress):
            holder["chunk"] = import_cloud(doc, source, chunk=target, label=v["label"].strip() or None,
                                           crs=v["crs"].strip() or None, max_points=v["max_points"] or None,
                                           classes=classes, progress=progress)
            return holder["chunk"]

        def done(chunk):
            activate = getattr(window, "_activate_chunk", None)
            if target is None and activate is not None:
                activate(chunk)
            n = (chunk.meta.get("terrio_import") or {}).get("points", 0)
            message(window, "info", self.title, t("{0} punti in {1}.", "{0} points in {1}.").format(num(n), chunk.label))

        self._run(t("Importazione della nuvola", "Importing the cloud"), job, done)

    def ground_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        from .clouds import DEFAULTS, ground

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ready():
            return
        if chunk.point_cloud is None:
            message(window, "warning", self.title, t("Il chunk non ha una nuvola densa.", "The chunk has no dense cloud."))
            return
        s = self.context.settings
        smrf, pmf = DEFAULTS["smrf"], DEFAULTS["pmf"]
        v = ParamDialog.ask(window, t("Classifica il terreno (terrio)", "Classify the ground (terrio)"), [
            Field("method", t("Filtro", "Filter"), s.get("ground_method", "smrf"), choices=METHOD_LABELS),
            Field("cell", t("Cella (m) - SMRF e PMF", "Cell (m) - SMRF and PMF"), float(s.get("ground_cell", 1.0)),
                  minimum=0.01, maximum=100.0, decimals=2),
            Field("slope", t("Pendenza - SMRF", "Slope - SMRF"), float(s.get("ground_slope", smrf["slope"])),
                  minimum=0.0, maximum=10.0, decimals=3,
                  tip=t("Il terreno piu' ripido atteso, salita su distanza (0,15 = 15 %)",
                        "The steepest ground expected, rise over run (0.15 = 15 %)")),
            Field("threshold", t("Soglia di quota (m) - SMRF e CSF", "Height threshold (m) - SMRF and CSF"),
                  float(s.get("ground_threshold", smrf["threshold"])), minimum=0.001, maximum=10.0, decimals=3),
            Field("pmf_slope", t("Pendenza - PMF", "Slope - PMF"), float(s.get("ground_pmf_slope", pmf["slope"])),
                  minimum=0.0, maximum=10.0, decimals=3),
            Field("pmf_distance", t("Distanza iniziale (m) - PMF", "Initial distance (m) - PMF"),
                  float(s.get("ground_pmf_distance", pmf["threshold"])), minimum=0.001, maximum=10.0, decimals=3,
                  tip=t("Il primo passo della soglia di quota, che cresce con le finestre",
                        "The first step of the height threshold, which grows with the windows")),
            Field("window", t("Finestra massima (m; vuoto = del filtro)", "Largest window (m; empty: the filter's)"),
                  None, kind="optional_float"),
            Field("resolution", t("Risoluzione del telo (m) - CSF", "Cloth resolution (m) - CSF"),
                  float(s.get("ground_resolution", 1.0)), minimum=0.01, maximum=100.0, decimals=2),
            Field("rigidness", t("Rigidita' del telo (1-3) - CSF", "Cloth rigidness (1-3) - CSF"),
                  int(s.get("ground_rigidness", 3)), minimum=1, maximum=3),
            Field("noise", t("Prima il rumore basso (classe 7, ELM)", "Low noise first (class 7, ELM)"),
                  bool(s.get("ground_noise", False))),
            Field("vegetation", t("Altri punti in classi 3/4/5 per altezza dal suolo (0,5 e 2 m)",
                                  "Other points as classes 3/4/5 by height above the ground (0.5 and 2 m)"),
                  bool(s.get("ground_vegetation", False)),
                  tip=t("Tutto cio' che non e' terreno e non ha gia' un'altra classe diventa vegetazione per la sua "
                        "altezza dal suolo: anche muri in piedi, sezioni e cumuli di terra. Va bene per le piante di "
                        "un campo: classifica prima i muri (classe 64) o correggi a mano.",
                        "Everything that is not ground and has no other class yet becomes vegetation by its height "
                        "above the ground: standing walls, sections and spoil heaps too. Fine for the plants of a "
                        "field: classify the walls first (class 64) or correct by hand.")),
        ], intro=t("Cambiano solo i punti delle classi 0, 1 e 2 (e 3, 4, 5 con la vegetazione): il terreno prende la "
                   "classe 2 e gli altri la 1, come i filtri di PDAL. Le altre classi (edifici, muri e classi "
                   "archeologiche, rumore, classi di un lidar importato) e i punti classificati a mano restano come "
                   "sono. Le classi vanno nella nuvola densa (DTM: DEM dalla classe 2).",
                   "Only the points in classes 0, 1 and 2 change (and 3, 4, 5 with the vegetation): the ground takes "
                   "class 2 and the others class 1, as PDAL's filters do. The other classes (buildings, walls and "
                   "the archaeological classes, noise, the classes of an imported lidar) and the points labelled "
                   "by hand stay as they are. The classes go into the dense cloud (DTM: a DEM from class 2)."))
        if v is None:
            return
        s.update({f"ground_{k}": v[k] for k in ("method", "cell", "slope", "threshold", "pmf_slope", "pmf_distance",
                                                  "resolution", "rigidness", "noise", "vegetation")})
        self.context.save_settings()
        pmf_chosen = v["method"] == "pmf"

        def job(progress):
            return ground(chunk, v["method"], cell=v["cell"], slope=v["pmf_slope"] if pmf_chosen else v["slope"],
                          threshold=v["pmf_distance"] if pmf_chosen else v["threshold"], window=v["window"],
                          resolution=v["resolution"], rigidness=v["rigidness"], noise=v["noise"],
                          vegetation=v["vegetation"], progress=progress)

        def done(report):
            message(window, "info", self.title, summary_ground(report))

        self._run(t("Classificazione del terreno", "Ground classification"), job, done)

    def reference_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        from .rasters import reference_raster

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ready():
            return
        s = self.context.settings
        v = ParamDialog.ask(window, t("Raster di riferimento dal web (terrio)", "Reference raster from the web (terrio)"), [
            Field("url", t("URL del COG, ricerca STAC (STACIT:...) o file", "URL of the COG, STAC search (STACIT:...) or file"),
                  s.get("reference_url", ""), kind="text"),
            Field("product", t("Tenerlo come versione di", "Keep it as a version of"), s.get("reference_product", "dem"),
                  choices=[("dem", "DEM"), ("orthomosaic", t("Ortofoto", "Orthomosaic"))]),
            Field("margin", t("Margine intorno al rilievo (m)", "Margin around the survey (m)"),
                  float(s.get("reference_margin", 20.0)), minimum=0.0, maximum=100000.0, decimals=1),
            Field("resolution", t("Risoluzione (m; vuoto = della sorgente)", "Resolution (m; empty: the source's)"),
                  None, kind="optional_float"),
            Field("name", t("Nome della versione (vuoto = dal file)", "Name of the version (empty: from the file)"), "",
                  kind="text"),
        ], intro=t("Il raster viene letto solo nella finestra del rilievo, riproiettato nel sistema del chunk e "
                   "messo fra le versioni del DEM o dell'ortofoto (menu Versioni): quello attivo non cambia.",
                   "The raster is read only over the survey, reprojected into the chunk's system and put among "
                   "the versions of the DEM or orthomosaic (Versions menu): the active one does not change."))
        if v is None or not v["url"].strip():
            return
        s.update({"reference_url": v["url"].strip(), "reference_product": v["product"], "reference_margin": v["margin"]})
        self.context.save_settings()

        def job(progress):
            return reference_raster(chunk, v["url"].strip(), product=v["product"], margin=v["margin"],
                                    resolution=v["resolution"], name=v["name"].strip() or None, progress=progress)

        def done(report):
            message(window, "info", self.title,
                    t("Versione \"{0}\": {1} x {2} pixel a {3:.3g} m.", "Version \"{0}\": {1} x {2} pixels at {3:.3g} m.")
                    .format(report["name"], report["width"], report["height"], report["resolution"][0]))

        self._run(t("Raster di riferimento", "Reference raster"), job, done)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        parser = argparse.ArgumentParser(prog="ppg plugin terrio",
                                         description="Point clouds, ground classification and web rasters through terrio.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("import", help="a point cloud file or URL as the dense cloud of a new chunk, or of a chunk")
        p.add_argument("project")
        p.add_argument("source", help="LAS, LAZ, COPC, PLY, E57, ept.json, or a URL")
        p.add_argument("--chunk", help="label of the chunk whose dense cloud is replaced (kept as a version)")
        p.add_argument("--label", help="name of the new chunk")
        p.add_argument("--activate", action="store_true",
                       help="make the new chunk the active one (by default the active chunk stays)")
        p.add_argument("--max-points", type=int, default=0)
        p.add_argument("--classes", help="only these classes, e.g. 2,6")
        p.add_argument("--crs", help="reference system when the file has none")
        p = sub.add_parser("ground", help="ground (2) and other points (1) of the dense cloud with SMRF, CSF or PMF")
        p.add_argument("project")
        p.add_argument("--chunk", help="label of the chunk (the active one by default)")
        p.add_argument("--method", choices=list(dict(METHOD_LABELS)), default="smrf")
        p.add_argument("--cell", type=float, default=1.0)
        p.add_argument("--slope", type=float, help="the filter's by default: SMRF 0.15, PMF 1.0")
        p.add_argument("--threshold", type=float, help="height threshold (PMF: initial distance); the filter's "
                                                      "by default: SMRF and CSF 0.5, PMF 0.15")
        p.add_argument("--window", type=float)
        p.add_argument("--resolution", type=float, default=1.0, help="CSF: cloth resolution")
        p.add_argument("--rigidness", type=int, default=3, help="CSF: 1-3")
        p.add_argument("--noise", action="store_true", help="low noise (class 7) first, with ELM")
        p.add_argument("--vegetation", action="store_true", help="other points as classes 3/4/5 by height above "
                                                                    "ground (walls, sections and spoil heaps too)")
        p = sub.add_parser("reference", help="a COG or STAC raster of the web as a version of the DEM or orthomosaic")
        p.add_argument("project")
        p.add_argument("url")
        p.add_argument("--chunk")
        p.add_argument("--product", choices=["dem", "orthomosaic"], default="dem")
        p.add_argument("--margin", type=float, default=20.0)
        p.add_argument("--resolution", type=float)
        p.add_argument("--name")
        p = sub.add_parser("export", help="the dense cloud (.e57, .ept, .las, .laz, .copc.laz, .ply) or the "
                                          "orthomosaic tiles (.pmtiles, .mbtiles)")
        p.add_argument("project")
        p.add_argument("output")
        p.add_argument("--chunk")
        sub.add_parser("install", help="download terrio from PyPI into the plugin folder")
        sub.add_parser("version", help="the terrio the plugin uses")
        a = parser.parse_args(list(args))
        from pyarchinit_photogrammetry.core.progress import print_progress

        if a.what == "version":
            found = environment.available()
            print(f"terrio {found}" if found else f"terrio not found: pip install terrio=={environment.TERRIO_VERSION}, "
                                                   "or: ppg plugin terrio install")
            return 0 if found else 1
        if a.what == "install":
            print(environment.install(print_progress))
            return 0
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None and (a.chunk or a.what != "import"):
            print(f"no chunk {a.chunk!r} in {a.project}" if a.chunk else f"no active chunk in {a.project}")
            return 2
        try:
            if a.what == "import":
                from .clouds import import_cloud

                classes = [int(c) for c in a.classes.split(",") if c.strip()] if a.classes else None
                target = chunk if a.chunk else None
                active = doc.chunk
                new = import_cloud(doc, a.source, chunk=target, label=a.label, crs=a.crs, max_points=a.max_points or None,
                                   classes=classes, progress=print_progress)
                if a.activate or active is None:
                    doc.chunk = new
                elif doc.chunk is not active:  # the active chunk of the project stays as it was
                    doc.chunk = active
                doc.save()
                print(t("{0} punti in {1}", "{0} points in {1}").format(num(new.meta["terrio_import"]["points"]),
                                                                        new.label))
            elif a.what == "ground":
                from .clouds import ground

                report = ground(chunk, a.method, cell=a.cell, slope=a.slope, threshold=a.threshold, window=a.window,
                                resolution=a.resolution, rigidness=a.rigidness, noise=a.noise, vegetation=a.vegetation,
                                progress=print_progress)
                doc.save()
                print(summary_ground(report))
            elif a.what == "reference":
                from .rasters import reference_raster

                report = reference_raster(chunk, a.url, product=a.product, margin=a.margin, resolution=a.resolution,
                                          name=a.name, progress=print_progress)
                doc.save()
                print(f"version {report['name']!r}: {report['width']} x {report['height']} pixels, {report['path']}")
            else:
                out = Path(a.output)
                if out.suffix.lower() in (".pmtiles", ".mbtiles"):
                    from .rasters import export_tiles

                    print(export_tiles(chunk, out, print_progress))
                else:
                    from .clouds import export_cloud

                    print(export_cloud(chunk, out, print_progress))
        except (environment.TerrioMissing, RuntimeError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        return 0


def summary_ground(report: dict) -> str:
    """The classes counted, in words, in the language of the program."""
    lines = [t("Filtro {0}: {1} punti, {2} cambiati", "Filter {0}: {1} points, {2} changed")
             .format(report["method"].upper(), num(report["points"]), num(report["changed"]))]
    if report.get("kept"):
        lines.append(t("  {0} punti di altre classi o classificati a mano lasciati come erano",
                       "  {0} points of other classes or labelled by hand left as they were").format(num(report["kept"])))
    for code, n in sorted(report["counts"].items()):
        lines.append(f"  {code} {class_label(code, report.get('labels'))}: {num(n)}")
    return "\n".join(lines)
