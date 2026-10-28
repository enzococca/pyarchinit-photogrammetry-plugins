"""terrio - a plugin of pyarchinit-photogrammetry: point clouds and rasters through terrio
(https://github.com/opengeos/terrio), geospatial I/O in pure Rust, with no GDAL or PDAL to install.

* *Plugin > terrio > Importa nuvola di punti...* - LAS, LAZ, COPC, EPT, PLY and E57 files, on
  disk or on the web (http, S3), as the dense cloud of a new chunk or of the active one;
* *Plugin > terrio > Classifica il terreno...* - ground and other points with the SMRF, CSF or
  PMF filters of PDAL (low noise with ELM, vegetation classes by the height above the ground);
* *Plugin > terrio > Pulisci la nuvola (LOF, DBSCAN)...* - isolated points to class 7, detached
  groups (birds, people, floating fragments) removed or classed, counted and shown first;
* *Plugin > terrio > Colori e valori da un raster...* - the colours of the orthomosaic (or of
  another raster) for the points, or a value (temperature, index, height) as a dimension of them;
  *Valori dei punti nella vista 3D...* shows such a value with a palette;
* *Plugin > terrio > Quote ellissoidiche e sul geoide...* - a DEM or the dense cloud moved
  between ellipsoidal and orthometric heights (EGM2008, EGM96, national geoids), as a version;
* *Plugin > terrio > Raster di riferimento dal web...* - a Cloud Optimized GeoTIFF on the web, a
  STAC search or a collection of Microsoft Planetary Computer (Copernicus DEM...) warped onto the
  survey, kept as a version of the DEM or of the orthomosaic, its heights moved when asked;
* *File > Esporta* - the dense cloud as E57 or EPT, the orthomosaic as PMTiles or MBTiles;
* ``ppg plugin terrio import|ground|clean|colour|values|heights|reference|catalogue|export|install|version``
  - the same from the command line.

terrio comes with pip (``pip install terrio``); the packaged application, which has no pip,
downloads its wheel from PyPI into the plugin folder (*Installa terrio*).
"""
from __future__ import annotations

from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import environment
from .texts import class_label, dec, num, signed, t  # noqa: F401 (the plugin's texts, for its modules too)

__version__ = "0.2.1"

METHOD_LABELS = [("smrf", "SMRF (Simple Morphological Filter)"), ("csf", "CSF (Cloth Simulation Filter)"),
                 ("pmf", "PMF (Progressive Morphological Filter)")]
CLOUD_FILTER = "Nuvole di punti (*.las *.laz *.ply *.e57 *.json *.zip);;Tutti i file (*)"
RASTER_FILTER = "Raster (*.tif *.tiff *.vrt *.jp2 *.png *.jpg);;Tutti i file (*)"
MIB = 20  # about the size of a terrio wheel, in MB


class TerrioPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "terrio"
        self.description = t("Nuvole di punti (LAS, LAZ, COPC, EPT, PLY, E57), terreno, pulizia, colori e valori "
                             "da raster, quote sul geoide e raster del web con terrio, senza GDAL o PDAL",
                             "Point clouds (LAS, LAZ, COPC, EPT, PLY, E57), ground, cleaning, colours and values "
                             "from rasters, geoid heights and web rasters through terrio, without GDAL or PDAL")
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
        context.add_action(t("Pulisci la nuvola (LOF, DBSCAN)...", "Clean the cloud (LOF, DBSCAN)..."),
                           self.clean_dialog, processing=True,
                           tip=t("Punti isolati in classe 7 e gruppi staccati (uccelli, persone, frammenti) rimossi o "
                                 "classificati: prima contati e mostrati",
                                 "Isolated points to class 7 and detached groups (birds, people, fragments) removed or "
                                 "classed: counted and shown first"))
        context.add_action(t("Colori e valori da un raster...", "Colours and values from a raster..."),
                           self.colour_dialog, processing=True,
                           tip=t("I colori dell'ortofoto per i punti, o un valore (temperatura, indice, quota) come "
                                 "dimensione dei punti",
                                 "The colours of the orthomosaic for the points, or a value (temperature, index, height) "
                                 "as a dimension of the points"))
        context.add_action(t("Valori dei punti nella vista 3D...", "Values of the points in the 3D view..."),
                           self.values_dialog, processing=True,
                           tip=t("Una dimensione dei punti (temperatura, indice...) a colori nella vista 3D, o tolta",
                                 "A dimension of the points (temperature, index...) in colours in the 3D view, or removed"))
        context.add_action(t("Quote ellissoidiche e sul geoide...", "Ellipsoidal and geoid heights..."),
                           self.heights_dialog, processing=True,
                           tip=t("Il DEM o la nuvola densa dalle quote ellissoidiche (GNSS) a quelle sul geoide (EGM2008, "
                                 "EGM96, geoidi nazionali) o viceversa, come nuova versione",
                                 "The DEM or the dense cloud from ellipsoidal (GNSS) heights to geoid heights (EGM2008, "
                                 "EGM96, national geoids) or back, as a new version"))
        context.add_action(t("Raster di riferimento dal web...", "Reference raster from the web..."),
                           self.reference_dialog, processing=True,
                           tip=t("Un COG, una ricerca STAC o il Planetary Computer (Copernicus DEM...) ritagliati sul "
                                 "rilievo, come versione del DEM o dell'ortofoto",
                                 "A COG, a STAC search or the Planetary Computer (Copernicus DEM...) clipped to the "
                                 "survey, as a version of the DEM or orthomosaic"))
        context.add_action(t("Installa terrio...", "Install terrio..."), self.install_dialog,
                           tip=t("Scarica terrio da PyPI nella cartella del plugin (per l'applicazione senza pip)",
                                 "Download terrio from PyPI into the plugin folder (for the application without pip)"))
        context.add_command("terrio", self.command,
                            help="point clouds, ground, cleaning, colours, heights and web rasters through terrio")
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

    def _ready(self, minimum: tuple | None = None, feature: str = "") -> bool:
        """terrio (at least ``minimum``) can be imported; else the window offers to download it."""
        from pyarchinit_photogrammetry.gui.dialogs import choice

        if environment.available(minimum):
            return True
        window = self.context.window
        old = environment.installed()
        if old and minimum is not None and environment._key(old) < tuple(minimum):
            from .texts import needs_newer

            text = needs_newer(feature or t("Questo passo", "This step"), old, ".".join(str(v) for v in minimum)) + \
                "\n\n" + t("Scaricare ora terrio {0} da PyPI (circa {1} MB) nella cartella del plugin?",
                           "Download terrio {0} from PyPI (about {1} MB) into the plugin folder now?").format(
                    environment.TERRIO_VERSION, MIB)
        else:
            text = t("terrio non e' installato. Scaricare terrio {0} da PyPI (circa {1} MB) nella cartella del plugin?",
                     "terrio is not installed. Download terrio {0} from PyPI (about {1} MB) into the plugin folder?"
                     ).format(environment.TERRIO_VERSION, MIB)
        picked = choice(window, "question", self.title, text, [t("Scarica", "Download")], cancel=t("Annulla", "Cancel"))
        if picked == 0:
            self.install_dialog(force=True)
        return False

    def _run(self, title: str, job, done=None, changes: bool = True):
        """``job(progress)`` in the background; ``changes``: the project is saved and the views
        refreshed afterwards (the plugin API's ``run`` does neither)."""
        window = self.context.window
        if window is not None and changes:
            return window.run_task(title, job, done, refresh=True, save=True)
        return self.context.run(title, job, done)

    def _info(self, text: str, kind: str = "info") -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        dialogs.message(self.context.window, kind, self.title, text)

    def _cloud_chunk(self, minimum=environment.FEATURES_SINCE, feature: str = ""):
        """The active chunk with a dense cloud, terrio ready; else None (the window said why)."""
        chunk = self._chunk()
        if self.context.window is None or chunk is None:
            return None
        if not self._ready(minimum, feature):
            return None
        if chunk.point_cloud is None:
            self._info(t("Il chunk non ha una nuvola densa.", "The chunk has no dense cloud."), "warning")
            return None
        return chunk

    def install_dialog(self, force: bool = False) -> None:
        window = self.context.window
        found = environment.available(environment.FEATURES_SINCE)
        if found and not force:
            self._info(t("terrio {0} e' gia' disponibile.", "terrio {0} is already available.").format(found))
            return

        def done(folder):
            self._info(t("terrio {0} installato in {1}. Avvia di nuovo il passo.",
                         "terrio {0} installed in {1}. Start the step again.").format(environment.TERRIO_VERSION, folder))

        if window is None:
            return
        self._run(t("Installazione di terrio", "Installing terrio"), lambda progress: environment.install(progress),
                  done, changes=False)

    def import_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

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
            self._info(t("Scegli un file o un URL.", "Choose a file or a URL."), "warning")
            return
        s.update({"import_file": v["file"], "import_max_points": v["max_points"], "import_crs": v["crs"]})
        self.context.save_settings()
        target = doc.chunk if v["where"] == "replace" else None
        if v["where"] == "replace" and target is None:
            self._info(t("Nessun chunk attivo.", "No active chunk."), "warning")
            return
        classes = [int(c) for c in v["classes"].replace(";", ",").split(",") if c.strip()] or None
        holder = {}

        def job(progress):
            holder["chunk"] = import_cloud(doc, source, chunk=target, label=v["label"].strip() or None,
                                           crs=v["crs"].strip() or None, max_points=v["max_points"] or None,
                                           classes=classes, progress=progress)
            return holder["chunk"]

        def done(chunk):
            if target is None:  # the new chunk the active one (context.activate_chunk since the program 1.3.4)
                if hasattr(self.context, "activate_chunk"):
                    self.context.activate_chunk(chunk)
                elif getattr(window, "_activate_chunk", None) is not None:
                    window._activate_chunk(chunk)
            n = (chunk.meta.get("terrio_import") or {}).get("points", 0)
            self._info(t("{0} punti in {1}.", "{0} points in {1}.").format(num(n), chunk.label))

        self._run(t("Importazione della nuvola", "Importing the cloud"), job, done)

    def ground_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from .clouds import DEFAULTS, ground

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ready():
            return
        if chunk.point_cloud is None:
            self._info(t("Il chunk non ha una nuvola densa.", "The chunk has no dense cloud."), "warning")
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
            self._info(summary_ground(report))

        self._run(t("Classificazione del terreno", "Ground classification"), job, done)

    # ------------------------------------------------------------------ cleaning
    def clean_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import cleaning

        feature = t("La pulizia della nuvola", "Cleaning the cloud")
        chunk = self._cloud_chunk(feature=feature)
        if chunk is None:
            return
        window = self.context.window
        s = self.context.settings
        v = ParamDialog.ask(window, t("Pulisci la nuvola (terrio)", "Clean the cloud (terrio)"), [
            Field("lof", t("Punti isolati in classe 7 (LOF)", "Isolated points to class 7 (LOF)"),
                  bool(s.get("clean_lof", True))),
            Field("lof_neighbours", t("Vicini per il LOF", "Neighbours for LOF"),
                  int(s.get("clean_lof_neighbours", cleaning.LOF_NEIGHBOURS)), minimum=5, maximum=200),
            Field("lof_threshold", t("Soglia del LOF", "LOF threshold"),
                  float(s.get("clean_lof_threshold", cleaning.LOF_THRESHOLD)), minimum=1.05, maximum=100.0, decimals=2,
                  tip=t("Densita' dei vicini su quella del punto: circa 1 su una superficie; sopra la soglia il punto "
                        "e' isolato. Piu' bassa = piu' punti.",
                        "The density of the neighbours over the point's: about 1 on a surface; above the threshold "
                        "the point is isolated. Lower: more points.")),
            Field("dbscan", t("Gruppi staccati (DBSCAN)", "Detached groups (DBSCAN)"), bool(s.get("clean_dbscan", True))),
            Field("eps", t("Raggio di DBSCAN (m; vuoto = {0:g} volte la distanza fra i punti)",
                           "DBSCAN radius (m; empty: {0:g} times the spacing of the points)").format(cleaning.EPS_FACTOR),
                  None, kind="optional_float",
                  tip=t("Due punti piu' vicini di cosi' stanno nello stesso gruppo. Nei rilievi con margini radi "
                        "aumentalo, o i margini diventano gruppi staccati.",
                        "Two points closer than this are in the same group. In surveys with sparse margins raise "
                        "it, or the margins become detached groups.")),
            Field("min_points", t("Punti minimi di un nucleo di DBSCAN", "Fewest points of a DBSCAN core"),
                  int(s.get("clean_min_points", cleaning.MIN_POINTS)), minimum=2, maximum=1000),
            Field("max_share", t("Staccato se ha meno del (% della nuvola)", "Detached when under (% of the cloud)"),
                  float(s.get("clean_max_share", cleaning.MAX_SHARE)), minimum=0.001, maximum=50.0, decimals=3,
                  tip=t("Il gruppo piu' grande non e' mai staccato.", "The largest group is never detached.")),
            Field("detached", t("I gruppi staccati", "The detached groups"), s.get("clean_detached", "class"),
                  choices=[("class", t("in una classe (sotto)", "in a class (below)")),
                           ("remove", t("rimossi dalla nuvola", "removed from the cloud"))]),
            Field("detached_class", t("Classe dei gruppi staccati", "Class of the detached groups"),
                  int(s.get("clean_detached_class", cleaning.DETACHED_CLASS)), minimum=0, maximum=255,
                  tip=t("18 = rumore alto, 7 = rumore basso: il programma li lascia fuori da DEM e mesh",
                        "18: high noise, 7: low noise: the program leaves them out of DEMs and meshes")),
        ], intro=t("Prima i punti vengono trovati e contati (e mostrati nella vista 3D: rossi gli isolati, arancioni "
                   "i gruppi staccati), poi si decide. I punti classificati a mano non cambiano mai; quelli gia' nelle "
                   "classi di rumore (7, 18) restano fuori. La nuvola di prima resta fra le versioni.",
                   "First the points are found and counted (and shown in the 3D view: isolated ones red, detached "
                   "groups orange), then you decide. The points labelled by hand never change; those already in the "
                   "noise classes (7, 18) are left out. The cloud of before is kept among the versions."))
        if v is None:
            return
        if not (v["lof"] or v["dbscan"]):
            self._info(t("Scegli LOF, DBSCAN o entrambi.", "Choose LOF, DBSCAN or both."), "warning")
            return
        s.update({f"clean_{k}": v[k] for k in ("lof", "lof_neighbours", "lof_threshold", "dbscan", "min_points",
                                                 "max_share", "detached", "detached_class")})
        self.context.save_settings()

        def job(progress):
            return cleaning.analyse(chunk, lof=v["lof"], lof_neighbours=v["lof_neighbours"],
                                    lof_threshold=v["lof_threshold"], dbscan=v["dbscan"], eps=v["eps"],
                                    min_points=v["min_points"], max_share=v["max_share"], progress=progress)

        def found(analysis):
            self._confirm_cleaning(chunk, analysis, v["detached"], v["detached_class"])

        self._run(t("Analisi della nuvola", "Analysing the cloud"), job, found, changes=False)

    def _preview_path(self, name: str) -> Path:
        folder = Path(self.context.data_dir) / "preview"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    def _confirm_cleaning(self, chunk, analysis: dict, action: str, code: int) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import cleaning

        text = cleaning.summary(analysis)
        if not (analysis["isolated"].any() or analysis["detached"].any()):
            self._info(text + "\n\n" + t("Niente da cambiare.", "Nothing to change."))
            return
        shown, path = False, None
        if hasattr(self.context, "show_model"):  # since the program 1.3.4: the points found in the 3D view
            try:
                path = self._preview_path("pulizia.ply")
                cleaning.preview(chunk, analysis, path)
                shown = self.context.show_model("point_cloud", path=path, title=t(
                    "Pulizia: rossi gli isolati, arancioni i gruppi staccati",
                    "Cleaning: isolated points red, detached groups orange"))
            except Exception:  # the preview is a help: the question comes anyway
                shown = False
        what = (t("I gruppi staccati vengono rimossi.", "The detached groups are removed.") if action == "remove" else
                t("I gruppi staccati vanno in classe {0}.", "The detached groups go to class {0}.").format(code))
        picked = dialogs.choice(self.context.window, "question", self.title,
                                text + "\n\n" + what + " " + t("Applicare? La nuvola di prima resta fra le versioni.",
                                                               "Apply? The cloud of before is kept among the versions."),
                                [t("Applica", "Apply")], cancel=t("Annulla", "Cancel"))
        if shown:
            self.context.show_model("point_cloud")
        if path is not None:
            Path(path).unlink(missing_ok=True)
        if picked != 0:
            return
        self._run(t("Pulizia della nuvola", "Cleaning the cloud"),
                  lambda progress: cleaning.apply(chunk, analysis, action, code, progress=progress),
                  lambda out: self._info(cleaning.applied_summary(out)))

    # ------------------------------------------------------------------ colours and values
    def colour_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import colour

        chunk = self._cloud_chunk(feature=t("I colori e i valori da un raster", "Colours and values from a raster"))
        if chunk is None:
            return
        s = self.context.settings
        rasters = colour.chunk_rasters(chunk)
        choices = [(r["key"], r["label"]) for r in rasters] + [("file", t("un altro raster (file o URL, sotto)",
                                                                          "another raster (file or URL, below)"))]
        keys = [c[0] for c in choices]
        default = s.get("colour_source") if s.get("colour_source") in keys else keys[0]
        v = ParamDialog.ask(self.context.window, t("Colori e valori da un raster (terrio)",
                                                   "Colours and values from a raster (terrio)"), [
            Field("source", t("Raster", "Raster"), default, choices=choices),
            Field("file", t("File", "File"), s.get("colour_file", ""), kind="file_open", filter=RASTER_FILTER),
            Field("url", t("Oppure URL (COG, STACIT:...)", "Or URL (COG, STACIT:...)"), "", kind="text"),
            Field("mode", t("Che cosa prendere", "What to take"), s.get("colour_mode", "auto"),
                  choices=[("auto", t("secondo il raster: colori da un'ortofoto, un valore dagli altri",
                                      "by the raster: colours from an orthomosaic, a value from the others")),
                           ("rgb", t("i colori (bande 1, 2, 3)", "the colours (bands 1, 2, 3)")),
                           ("value", t("un valore, come dimensione dei punti", "a value, as a dimension of the points"))]),
            Field("band", t("Banda del valore (file)", "Band of the value (file)"), 1, minimum=1, maximum=1000),
            Field("name", t("Nome della dimensione (vuoto = dal raster)", "Name of the dimension (empty: from the raster)"),
                  "", kind="text", tip=t("Lettere minuscole, cifre e _: temperature, ndvi, height",
                                         "Lower case letters, digits and _: temperature, ndvi, height")),
            Field("palette", t("Poi mostrare il valore nella vista 3D con", "Then show the value in the 3D view with"),
                  s.get("colour_palette", "turbo"), choices=[("", t("non mostrarlo", "do not show it"))] +
                  [(p, p) for p in colour.PALETTES]),
        ], intro=t("Colori: i punti coperti dal raster prendono i suoi colori, gli altri tengono i loro; la nuvola di "
                   "prima resta fra le versioni. Valore: una banda (temperatura, indice, quota) diventa una dimensione "
                   "dei punti, esportata con LAS, LAZ, COPC, EPT, PLY ed E57 di terrio; i colori non cambiano. Il "
                   "programma non conosce la dimensione: se riscrive la nuvola (pulizia, classi, ritaglio) va presa "
                   "di nuovo.",
                   "Colours: the points the raster covers take its colours, the others keep theirs; the cloud of "
                   "before is kept among the versions. Value: a band (temperature, index, height) becomes a "
                   "dimension of the points, exported with terrio's LAS, LAZ, COPC, EPT, PLY and E57; the colours do "
                   "not change. The program does not know the dimension: when it writes the cloud again (cleaning, "
                   "classes, cropping) take it again."))
        if v is None:
            return
        source = v["source"]
        if source == "file":
            source = (v["url"] or "").strip() or v["file"]
            if not source:
                self._info(t("Scegli un file o un URL.", "Choose a file or a URL."), "warning")
                return
            band = v["band"]
        else:
            band = next(r["band"] for r in rasters if r["key"] == source)
        s.update({"colour_source": v["source"], "colour_file": v["file"], "colour_mode": v["mode"],
                  "colour_palette": v["palette"]})
        self.context.save_settings()
        mode = None if v["mode"] == "auto" else v["mode"]

        def job(progress):
            return colour.colorize(chunk, source, mode=mode, band=band, name=v["name"].strip() or None,
                                   progress=progress)

        def done(out):
            self._info(colour.summary(out))
            if out["mode"] == "value" and v["palette"] and out["with_value"]:
                self._show_values(chunk, out["name"], v["palette"])

        self._run(t("Colori e valori da un raster", "Colours and values from a raster"), job, done)

    def _show_values(self, chunk, name: str, palette: str, limits=None) -> None:
        """The dense cloud coloured by its dimension ``name`` in the 3D view (since the program 1.3.4)."""
        from . import colour

        if not hasattr(self.context, "show_model"):
            self._info(t("Mostrare i valori nella vista 3D richiede pyarchinit-photogrammetry 1.3.4 o successivo; "
                         "esporta la nuvola (LAS, PLY) per vederli in altri programmi.",
                         "Showing the values in the 3D view needs pyarchinit-photogrammetry 1.3.4 or later; export "
                         "the cloud (LAS, PLY) to see them in other programs."), "warning")
            return
        path = self._preview_path(f"valori_{name}.ply")

        def done(info):
            lo, hi = info["limits"]
            self.context.show_model("point_cloud", path=path, title=t("{0}: da {1:.4g} a {2:.4g} ({3})",
                                                                      "{0}: from {1:.4g} to {2:.4g} ({3})")
                                    .format(name, lo, hi, palette))

        self._run(t("Valori dei punti", "Values of the points"),
                  lambda progress: colour.preview_values(chunk, name, path, palette, limits), done, changes=False)

    def values_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import colour

        chunk = self._cloud_chunk(feature=t("I valori dei punti", "The values of the points"))
        if chunk is None:
            return
        found = colour.values(chunk)
        if not found:
            self._info(t("La nuvola densa non ha valori del plugin: prima Colori e valori da un raster (modo valore).",
                         "The dense cloud has no values of the plugin: first Colours and values from a raster (a "
                         "value)."), "warning")
            return
        s = self.context.settings
        v = ParamDialog.ask(self.context.window, t("Valori dei punti (terrio)", "Values of the points (terrio)"), [
            Field("name", t("Dimensione", "Dimension"), next(iter(found)),
                  choices=[(n, f"{n} ({info.get('label', '')})" if info.get("label") else n) for n, info in found.items()]),
            Field("action", t("Che cosa fare", "What to do"), "show",
                  choices=[("show", t("mostrarla a colori nella vista 3D", "show it in colours in the 3D view")),
                           ("remove", t("toglierla dalla nuvola", "take it away from the cloud"))]),
            Field("palette", t("Colori", "Colours"), s.get("colour_palette") or "turbo",
                  choices=[(p, p) for p in colour.PALETTES]),
            Field("lower", t("Da (vuoto = 2o percentile)", "From (empty: 2nd percentile)"), None, kind="optional_float"),
            Field("upper", t("A (vuoto = 98o percentile)", "To (empty: 98th percentile)"), None, kind="optional_float"),
        ])
        if v is None:
            return
        if v["action"] == "remove":
            self._run(t("Valori dei punti", "Values of the points"),
                      lambda progress: colour.remove_value(chunk, v["name"]),
                      lambda _: self._info(t("Dimensione {0} tolta.", "Dimension {0} taken away.").format(v["name"])))
            return
        limits = None
        if v["lower"] is not None and v["upper"] is not None and v["upper"] > v["lower"]:
            limits = (v["lower"], v["upper"])
        self._show_values(chunk, v["name"], v["palette"], limits)

    # ------------------------------------------------------------------ heights
    def _systems(self) -> list:
        from . import heights

        return [(k, t(*names)) for k, names in heights.PRESETS] + [("other", t("altro (scritto sotto)",
                                                                              "other (written below)"))]

    def heights_dialog(self) -> None:
        from pyarchinit_photogrammetry.core import versions
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import heights

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ready(environment.FEATURES_SINCE,
                                                              t("La conversione delle quote", "The conversion of heights")):
            return
        dems = ([("active", t("il DEM attivo", "the active DEM"))] if chunk.dem_path is not None else []) + \
            [(e["id"], e["name"]) for e in versions.versions(chunk, "dem")]
        products = ([("dem", "DEM")] if dems else []) + \
            ([("point_cloud", t("Nuvola densa (attiva)", "Dense cloud (active)"))] if chunk.point_cloud is not None else [])
        if not products:
            self._info(t("Il chunk non ha ne' DEM ne' nuvola densa.", "The chunk has neither a DEM nor a dense cloud."),
                       "warning")
            return
        s = self.context.settings
        systems = self._systems()
        keys = [k for k, _ in systems]
        src_default = s.get("heights_source", "ellipsoid") if s.get("heights_source", "ellipsoid") in keys else "ellipsoid"
        dst_default = s.get("heights_target", "EPSG:3855") if s.get("heights_target", "EPSG:3855") in keys else "EPSG:3855"
        v = ParamDialog.ask(window, t("Quote ellissoidiche e sul geoide (terrio)", "Ellipsoidal and geoid heights (terrio)"), [
            Field("product", t("Che cosa", "What"), products[0][0], choices=products),
            Field("dem", t("Quale DEM", "Which DEM"), dems[0][0] if dems else "", choices=dems or [("", "-")]),
            Field("source", t("Quote di ora", "Heights now"), src_default, choices=systems),
            Field("source_other", t("Altro sistema di ora", "Other system now"), "", kind="text",
                  tip=t("Un CRS verticale (EPSG:5703) o un file del geoide (.gtx, .tif)",
                        "A vertical CRS (EPSG:5703) or a geoid file (.gtx, .tif)")),
            Field("target", t("Quote volute", "Heights wanted"), dst_default, choices=systems),
            Field("target_other", t("Altro sistema voluto", "Other system wanted"), "", kind="text"),
            Field("activate", t("Farne la versione attiva", "Make it the active version"), False),
            Field("network", t("Scaricare le griglie che mancano dal CDN di PROJ", "Download missing grids from the PROJ CDN"),
                  bool(s.get("heights_network", True))),
            Field("keep", t("Tenere le griglie scaricate nella cartella del plugin (poi anche senza rete)",
                            "Keep the grids downloaded in the plugin folder (then also offline)"),
                  bool(s.get("heights_keep", True))),
            Field("folder", t("Cartella con altre griglie (facoltativa)", "Folder with other grids (optional)"),
                  s.get("heights_folder", ""), kind="dir"),
        ], intro=t("Un drone con GNSS da' quote sull'ellissoide; i DEM nazionali e globali (Copernicus) quote sul geoide, "
                   "sul livello del mare: la differenza e' l'ondulazione del geoide, decine di metri in molti luoghi. "
                   "Il risultato e' una nuova versione (menu Versioni); quella di ora non cambia. Le griglie scaricate "
                   "vanno in {0}.",
                   "A drone with GNSS gives heights above the ellipsoid; national and global DEMs (Copernicus) heights "
                   "above the geoid, sea level: the difference is the undulation of the geoid, tens of metres in many "
                   "places. The result is a new version (Versions menu); the current one does not change. The grids "
                   "downloaded go into {0}.").format(environment.grids_dir()))
        if v is None:
            return
        source = v["source_other"].strip() if v["source"] == "other" else v["source"]
        target = v["target_other"].strip() if v["target"] == "other" else v["target"]
        try:
            source, target = heights.normalise(source), heights.normalise(target)
        except (ValueError, FileNotFoundError) as exc:
            self._info(str(exc), "warning")
            return
        if not source or not target or source == target:
            self._info(t("Scegli due sistemi di quote diversi.", "Choose two different height systems."), "warning")
            return
        s.update({"heights_source": v["source"], "heights_target": v["target"], "heights_network": v["network"],
                  "heights_keep": v["keep"], "heights_folder": v["folder"]})
        self.context.save_settings()
        grids = {"network": v["network"], "keep": v["keep"], "folders": [v["folder"]] if v["folder"] else []}

        def job(progress):
            if v["product"] == "dem":
                return heights.convert_dem(chunk, source, target, version=None if v["dem"] == "active" else v["dem"],
                                           activate=v["activate"], progress=progress, **grids)
            return heights.convert_cloud(chunk, source, target, activate=v["activate"], progress=progress, **grids)

        self._run(t("Conversione delle quote", "Converting the heights"), job,
                  lambda out: self._info(summary_heights(out)))

    def reference_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from .rasters import CATALOGUE, catalogue_label, reference_raster

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ready():
            return
        s = self.context.settings
        collections = [("", t("nessuna: l'URL o il file qui sopra", "none: the URL or file above"))] + \
            [(k, catalogue_label(k)) for k in CATALOGUE] + [("other", t("un'altra collezione (sotto)",
                                                                        "another collection (below)"))]
        keys = [k for k, _ in collections]
        systems = [(k, label) for k, label in self._systems() if k != "other"]
        v = ParamDialog.ask(window, t("Raster di riferimento dal web (terrio)", "Reference raster from the web (terrio)"), [
            Field("url", t("URL del COG, ricerca STAC (STACIT:...) o file", "URL of the COG, STAC search (STACIT:...) or file"),
                  s.get("reference_url", ""), kind="text"),
            Field("collection", t("Oppure Microsoft Planetary Computer", "Or Microsoft Planetary Computer"),
                  s.get("reference_collection", "") if s.get("reference_collection", "") in keys else "",
                  choices=collections),
            Field("collection_other", t("Altra collezione e asset (es. esa-worldcover map)",
                                        "Other collection and asset (e.g. esa-worldcover map)"), "", kind="text"),
            Field("datetime", t("Date (immagini; es. 2024-06-01/2024-09-30)", "Dates (imagery; e.g. 2024-06-01/2024-09-30)"),
                  s.get("reference_datetime", ""), kind="text"),
            Field("product", t("Tenerlo come versione di", "Keep it as a version of"), s.get("reference_product", "dem"),
                  choices=[("dem", "DEM"), ("orthomosaic", t("Ortofoto", "Orthomosaic"))],
                  tip=t("Per le collezioni del catalogo lo decide il catalogo",
                        "For the collections of the catalogue, the catalogue decides")),
            Field("margin", t("Margine intorno al rilievo (m)", "Margin around the survey (m)"),
                  float(s.get("reference_margin", 20.0)), minimum=0.0, maximum=100000.0, decimals=1),
            Field("resolution", t("Risoluzione (m; vuoto = della sorgente)", "Resolution (m; empty: the source's)"),
                  None, kind="optional_float"),
            Field("heights_from", t("Quote del raster (DEM)", "Heights of the raster (DEM)"), "auto",
                  choices=[("auto", t("come dice il catalogo; altrimenti non convertire",
                                      "as the catalogue says; else no conversion"))] + systems),
            Field("heights_to", t("Convertire le quote in", "Convert the heights to"), s.get("reference_heights_to", ""),
                  choices=[("", t("non convertire", "no conversion"))] + systems,
                  tip=t("Le quote del rilievo: ellissoide per un drone con GNSS",
                        "The heights of the survey: the ellipsoid for a drone with GNSS")),
            Field("name", t("Nome della versione (vuoto = dal file)", "Name of the version (empty: from the file)"), "",
                  kind="text"),
        ], intro=t("Il raster viene letto solo nella finestra del rilievo, riproiettato nel sistema del chunk e "
                   "messo fra le versioni del DEM o dell'ortofoto (menu Versioni): quello attivo non cambia. Il "
                   "Planetary Computer di Microsoft si usa senza account (con limiti di accesso); terrio firma da "
                   "solo gli indirizzi dei suoi file.",
                   "The raster is read only over the survey, reprojected into the chunk's system and put among "
                   "the versions of the DEM or orthomosaic (Versions menu): the active one does not change. "
                   "Microsoft's Planetary Computer is used without an account (with access limits); terrio signs "
                   "the addresses of its files by itself."))
        if v is None:
            return
        collection, asset = v["collection"], None
        if collection == "other":
            parts = v["collection_other"].split()
            if not parts:
                self._info(t("Scrivi la collezione e l'asset.", "Write the collection and the asset."), "warning")
                return
            collection, asset = parts[0], (parts[1] if len(parts) > 1 else None)
        url = v["url"].strip()
        if not collection and not url:
            return
        s.update({"reference_url": url, "reference_collection": v["collection"], "reference_product": v["product"],
                  "reference_margin": v["margin"], "reference_datetime": v["datetime"],
                  "reference_heights_to": v["heights_to"]})
        self.context.save_settings()
        heights_from = None if v["heights_from"] == "auto" else v["heights_from"]
        product = None if collection in CATALOGUE else v["product"]
        if v["heights_to"] and not heights_from and collection not in CATALOGUE:
            self._info(t("Di' le quote del raster per convertirle.", "Say the heights of the raster to convert them."),
                       "warning")
            return
        if (v["heights_to"] or collection) and not self._ready(
                environment.FEATURES_SINCE, t("Il Planetary Computer e la conversione delle quote",
                                              "The Planetary Computer and the conversion of heights")):
            return

        def job(progress):
            return reference_raster(chunk, url or None, product=product, margin=v["margin"], resolution=v["resolution"],
                                    name=v["name"].strip() or None, heights_from=heights_from,
                                    heights_to=v["heights_to"] or None, collection=collection or None, asset=asset,
                                    datetime_range=v["datetime"].strip() or None, progress=progress)

        def done(report):
            text = t("Versione \"{0}\": {1} x {2} pixel a {3:.3g} m.", "Version \"{0}\": {1} x {2} pixels at {3:.3g} m.") \
                .format(report["name"], report["width"], report["height"], report["resolution"][0])
            if report.get("shift"):
                text += "\n" + t("Quote spostate di {0} m in media.", "Heights moved by {0} m on average.").format(
                    signed(report["shift"][1]))
            self._info(text)

        self._run(t("Raster di riferimento", "Reference raster"), job, done)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        parser = argparse.ArgumentParser(prog="ppg plugin terrio",
                                         description="Point clouds, ground, cleaning, colours, heights and web rasters "
                                                     "through terrio.")
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
        p = sub.add_parser("clean", help="isolated points (LOF) to class 7, detached groups (DBSCAN) removed or "
                                         "classed; the counts first, the cloud of before kept as a version")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--no-lof", action="store_true", help="no isolated points")
        p.add_argument("--lof-neighbours", type=int, default=20)
        p.add_argument("--lof-threshold", type=float, default=3.0)
        p.add_argument("--no-dbscan", action="store_true", help="no detached groups")
        p.add_argument("--eps", type=float, help="DBSCAN radius in metres; by default 4 times the point spacing")
        p.add_argument("--min-points", type=int, default=6)
        p.add_argument("--max-share", type=float, default=1.0, help="a group under this per cent of the cloud is "
                                                                    "detached (the largest never)")
        p.add_argument("--remove", action="store_true", help="remove the detached groups (else: --detached-class)")
        p.add_argument("--detached-class", type=int, default=18)
        p.add_argument("--dry-run", action="store_true", help="only count")
        p = sub.add_parser("colour", aliases=["color"], help="colours (rgb) or a value of a raster for the points")
        p.add_argument("project")
        p.add_argument("source", nargs="?", help="orthomosaic, dem, thermal, index:NDVI, reflectance:N, "
                                                  "version:dem:ID... (see --list), or a raster file or URL")
        p.add_argument("--chunk")
        p.add_argument("--list", action="store_true", help="the rasters of the chunk")
        p.add_argument("--rgb", action="store_true", help="the colours (by default for an orthomosaic)")
        p.add_argument("--value", action="store_true", help="a value as a dimension (by default for the others)")
        p.add_argument("--band", type=int, help="the band of the value")
        p.add_argument("--name", help="the name of the dimension")
        p = sub.add_parser("values", help="the dimensions the plugin gave the points (--remove NAME takes one away)")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--remove")
        p = sub.add_parser("heights", help="a DEM or the dense cloud between ellipsoidal and geoid heights, as a version")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--product", choices=["dem", "point_cloud"], default="dem")
        p.add_argument("--version", help="id of a kept version of the DEM (the active one by default)")
        p.add_argument("--from", dest="source", required=True,
                       help="ellipsoid, a vertical CRS (EPSG:3855 EGM2008, EPSG:5773 EGM96...) or a geoid file")
        p.add_argument("--to", dest="target", required=True)
        p.add_argument("--activate", action="store_true", help="make the new version the active one")
        p.add_argument("--name")
        p.add_argument("--no-network", action="store_true", help="no grid from the PROJ CDN")
        p.add_argument("--no-keep-grids", action="store_true", help="read missing grids in place, do not keep them")
        p.add_argument("--grids", action="append", default=[], help="a folder of grids (more than once)")
        p = sub.add_parser("reference", help="a COG, STAC or Planetary Computer raster as a version of the DEM or "
                                             "orthomosaic")
        p.add_argument("project")
        p.add_argument("url", nargs="?", help="URL, STACIT:... or file (or --collection)")
        p.add_argument("--chunk")
        p.add_argument("--collection", help="a collection of the Planetary Computer (see: catalogue)")
        p.add_argument("--asset", help="the asset of the collection (the catalogue's by default)")
        p.add_argument("--datetime", help="dates of the items, e.g. 2024-06-01/2024-09-30")
        p.add_argument("--max-items", type=int)
        p.add_argument("--cloud", type=float, help="the most cloud cover of an image, per cent")
        p.add_argument("--product", choices=["dem", "orthomosaic"])
        p.add_argument("--margin", type=float, default=20.0)
        p.add_argument("--resolution", type=float)
        p.add_argument("--name")
        p.add_argument("--heights-from", help="heights of the DEM (the catalogue's by default)")
        p.add_argument("--heights-to", help="move them to: ellipsoid, EPSG:3855...")
        p.add_argument("--no-network", action="store_true", help="no grid from the PROJ CDN")
        sub.add_parser("catalogue", help="the collections of the Planetary Computer offered by name")
        p = sub.add_parser("export", help="the dense cloud (.e57, .ept, .las, .laz, .copc.laz, .ply) or the "
                                          "orthomosaic tiles (.pmtiles, .mbtiles)")
        p.add_argument("project")
        p.add_argument("output")
        p.add_argument("--chunk")
        sub.add_parser("install", help="download terrio from PyPI into the plugin folder")
        sub.add_parser("version", help="the terrio the plugin uses, and its grid folder")
        a = parser.parse_args(list(args))
        from pyarchinit_photogrammetry.core.progress import print_progress

        if a.what == "version":
            found = environment.available()
            print(f"terrio {found}" if found else f"terrio not found: pip install terrio=={environment.TERRIO_VERSION}, "
                                                   "or: ppg plugin terrio install")
            if found and environment._key(found) < environment.FEATURES_SINCE:
                print(f"the cleaning, colours, heights and Planetary Computer need terrio "
                      f"{'.'.join(map(str, environment.FEATURES_SINCE))}: ppg plugin terrio install")
            print(f"grids: {environment.grids_dir()}")
            return 0 if found else 1
        if a.what == "install":
            print(environment.install(print_progress))
            return 0
        if a.what == "catalogue":
            from .rasters import CATALOGUE, catalogue_label

            for key, entry in CATALOGUE.items():
                print(f"{key:16} {entry['product']:12} {entry['asset']:10} {catalogue_label(key)}")
            return 0
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None and (a.chunk or a.what != "import"):
            print(f"no chunk {a.chunk!r} in {a.project}" if a.chunk else f"no active chunk in {a.project}")
            return 2
        try:
            return self._command(a, doc, chunk, print_progress)
        except (environment.TerrioMissing, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
            print(f"error: {exc}")
            return 1

    def _command(self, a, doc, chunk, print_progress) -> int:
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
            print(t("{0} punti in {1}", "{0} points in {1}").format(num(new.meta["terrio_import"]["points"]), new.label))
        elif a.what == "ground":
            from .clouds import ground

            report = ground(chunk, a.method, cell=a.cell, slope=a.slope, threshold=a.threshold, window=a.window,
                            resolution=a.resolution, rigidness=a.rigidness, noise=a.noise, vegetation=a.vegetation,
                            progress=print_progress)
            doc.save()
            print(summary_ground(report))
        elif a.what == "clean":
            from . import cleaning

            found = cleaning.analyse(chunk, lof=not a.no_lof, lof_neighbours=a.lof_neighbours,
                                     lof_threshold=a.lof_threshold, dbscan=not a.no_dbscan, eps=a.eps,
                                     min_points=a.min_points, max_share=a.max_share, progress=print_progress)
            print(cleaning.summary(found))
            if a.dry_run:
                return 0
            if found["isolated"].any() or found["detached"].any():
                out = cleaning.apply(chunk, found, "remove" if a.remove else "class", a.detached_class,
                                     progress=print_progress)
                doc.save()
                print(cleaning.applied_summary(out))
        elif a.what in ("colour", "color"):
            from . import colour

            if a.list or not a.source:
                for r in colour.chunk_rasters(chunk):
                    print(f"{r['key']:40} {r['mode']:6} {r['label']}")
                return 0
            mode = "rgb" if a.rgb else "value" if a.value else None
            out = colour.colorize(chunk, a.source, mode=mode, band=a.band or 0, name=a.name, progress=print_progress)
            doc.save()
            print(colour.summary(out))
        elif a.what == "values":
            from . import colour

            if a.remove:
                colour.remove_value(chunk, a.remove)
                doc.save()
            for name, info in colour.values(chunk).items():
                rng = info.get("range")
                where = f"{rng[0]:.4g} .. {rng[2]:.4g}" if rng else ""
                print(f"{name:20} {info.get('points', 0):>12} {where:24} {info.get('label', '')}")
        elif a.what == "heights":
            from . import heights

            grids = {"network": not a.no_network, "keep": not a.no_keep_grids, "folders": a.grids}
            if a.product == "dem":
                out = heights.convert_dem(chunk, a.source, a.target, version=a.version, name=a.name,
                                          activate=a.activate, progress=print_progress, **grids)
            else:
                out = heights.convert_cloud(chunk, a.source, a.target, name=a.name, activate=a.activate,
                                            progress=print_progress, **grids)
            doc.save()
            print(summary_heights(out))
        elif a.what == "reference":
            from .rasters import reference_raster

            report = reference_raster(chunk, a.url, product=a.product, margin=a.margin, resolution=a.resolution,
                                      name=a.name, heights_from=a.heights_from, heights_to=a.heights_to,
                                      collection=a.collection, asset=a.asset, datetime_range=a.datetime,
                                      max_items=a.max_items, cloud=a.cloud, grids={"network": not a.no_network},
                                      progress=print_progress)
            doc.save()
            print(f"version {report['name']!r}: {report['width']} x {report['height']} pixels, {report['path']}")
            if report.get("shift"):
                print(f"heights moved by {report['shift'][1]:+.3f} m on average "
                      f"({report['shift'][0]:+.3f} .. {report['shift'][2]:+.3f})")
        else:
            out = Path(a.output)
            if out.suffix.lower() in (".pmtiles", ".mbtiles"):
                from .rasters import export_tiles

                print(export_tiles(chunk, out, print_progress))
            else:
                from .clouds import export_cloud

                print(export_cloud(chunk, out, print_progress))
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


def summary_heights(out: dict) -> str:
    """A conversion of heights, in words, in the language of the program."""
    from . import heights

    lo, mean, hi = out["shift"]
    where = t("la versione attiva", "the active version") if out.get("active") else t("fra le versioni", "among the versions")
    lines = ["\"{0}\" ({1})".format(out["name"], where),
             t("Quote da {0} a {1}: spostate di {2} m (da {3} a {4})", "Heights from {0} to {1}: moved by {2} m ({3} to {4})")
             .format(heights.label(out["source"]), heights.label(out["target"]), signed(mean, 3), signed(lo, 3),
                     signed(hi, 3))]
    if out.get("grids"):
        lines.append(t("Griglie scaricate in: {0}", "Grids downloaded into: {0}").format(out["grids"]))
    return "\n".join(lines)
