"""Spirula - a plugin of pyarchinit-photogrammetry: 3D Gaussian splatting of the aligned photos
with Spirula Studio (https://github.com/harry7557558/spirula-studio, GPL-3: Vulkan on NVIDIA,
AMD, Intel and Apple GPUs, one executable, no Python or CUDA).

* *Plugin > Spirula > Addestra Gaussian splatting...* - the COLMAP model and the photos of the
  chunk (masks and dense cloud too) go to ``spirula train``; the splat becomes a product of the
  chunk (``splat.ply``, the 3DGS PLY every viewer reads);
* *Plugin > Spirula > Mesh dallo splat...* - ``spirula mesh``: PLY, OBJ, glTF/GLB or STL beside
  the splat, the PLY also among the versions of the model;
* *Plugin > Spirula > Apri lo splat in Spirula Studio* - the viewer of Spirula on the splat;
* *Plugin > Spirula > Installa Spirula Studio...* - the release of GitHub for this computer
  (macOS on Apple Silicon, Windows and Linux x86_64) downloaded, checked and installed; offered
  by itself the first time Spirula is needed and missing;
* *Plugin > Spirula > Dove sta Spirula Studio...* - the executable (found by itself on the PATH
  and in the usual folders);
* *File > Esporta > Splat (PLY)* - a copy of the splat;
* ``ppg plugin spirula train|mesh|open|install|where ...`` - the same from the command line.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pyarchinit_photogrammetry.plugins import Plugin

from . import installer, runner

__version__ = "0.1.0"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:  # the program without its interface
        return it


PRESET_LABELS = [("3dgs", "3dgs (" + "standard" + ")"), ("in-the-wild", "in-the-wild"), ("centered-object", "centered-object"),
                 ("360-camera", "360-camera"), ("hdr", "hdr"), ("synthetic", "synthetic"), ("meshing", "meshing")]
QUALITY_LABELS = [("low", "low (15.000)"), ("medium", "medium (30.000)"), ("high", "high (50.000)"), ("ultra", "ultra (80.000)")]
RELEASES = "https://github.com/harry7557558/spirula-studio/releases"


class SpirulaPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "Spirula"
        self.description = t("Gaussian splatting delle foto allineate con Spirula Studio (Vulkan: GPU NVIDIA, AMD, "
                             "Intel e Apple), splat e mesh",
                             "Gaussian splatting of the aligned photos with Spirula Studio (Vulkan: NVIDIA, AMD, "
                             "Intel and Apple GPUs), splat and mesh")
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Addestra Gaussian splatting...", "Train Gaussian splatting..."), self.train_dialog,
                           processing=True,
                           tip=t("Il modello COLMAP e le foto del chunk a Spirula Studio: lo splat diventa un prodotto",
                                 "The COLMAP model and the photos of the chunk to Spirula Studio: the splat becomes a product"))
        context.add_action(t("Mesh dallo splat...", "Mesh from the splat..."), self.mesh_dialog, processing=True,
                           tip=t("spirula mesh: PLY, OBJ, glTF/GLB o STL accanto allo splat, il PLY fra le versioni del modello",
                                 "spirula mesh: PLY, OBJ, glTF/GLB or STL beside the splat, the PLY among the model versions"))
        context.add_action(t("Apri lo splat in Spirula Studio", "Open the splat in Spirula Studio"), self.open_splat)
        context.add_action(t("Installa Spirula Studio...", "Install Spirula Studio..."), self.install_dialog,
                           tip=t("Scarica la release di GitHub per questo computer e la installa",
                                 "Download the GitHub release for this computer and install it"))
        context.add_action(t("Dove sta Spirula Studio...", "Where Spirula Studio is..."), self.where_dialog)
        context.add_command("spirula", self.command,
                            help="Gaussian splatting with Spirula Studio: train, mesh, open, install, where")
        context.add_exporter(t("Splat (PLY 3D Gaussian Splatting)", "Splat (PLY 3D Gaussian Splatting)"), ".ply",
                             self.export_splat)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document
        return None if doc is None else doc.chunk

    def spirula(self) -> Path | None:
        return runner.find_spirula(self.context.settings.get("spirula"))

    def _spirula_or_ask(self) -> Path | None:
        """The executable; when missing, the window offers to download and install Spirula Studio
        (then the step is started again once it is there) or to be told where it is."""
        exe = self.spirula()
        if exe is None:
            from pyarchinit_photogrammetry.gui.dialogs import choice

            picked = choice(self.context.window, "question", self.title,
                            t("Spirula Studio non e' stato trovato. Scaricarlo da GitHub e installarlo adesso (circa 50-60 MB), "
                              "oppure indicare dove sta?",
                              "Spirula Studio was not found. Download it from GitHub and install it now (about 50-60 MB), "
                              "or say where it is?"),
                            [t("Scarica e installa", "Download and install"), t("Indica dove sta", "Say where it is"),
                             t("Annulla", "Cancel")])
            if picked == 0:
                self.install_dialog()
            elif picked == 1:
                self.where_dialog()
            exe = self.spirula()
        return exe

    def install_dialog(self) -> None:
        """Plugin > Spirula > Installa Spirula Studio: the release for this computer, downloaded in
        the background into the folder chosen (the system's by default)."""
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        window = self.context.window
        if window is None:
            return
        if installer.platform_key() is None:
            message(window, "warning", self.title,
                    t("Spirula Studio non ha una versione per questo computer ({0} {1}): le release coprono macOS su "
                      "Apple Silicon, Windows x86_64 e Linux x86_64.",
                      "Spirula Studio has no build for this computer ({0} {1}): the releases cover macOS on Apple "
                      "Silicon, Windows x86_64 and Linux x86_64.").format(installer.platform.system(),
                                                                          installer.platform.machine()))
            return
        found = self.spirula()
        v = ParamDialog.ask(window, t("Installa Spirula Studio", "Install Spirula Studio"), [
            Field("folder", t("Cartella", "Folder"), str(installer.default_folder()), kind="dir"),
        ], intro=t("L'ultima release di Spirula Studio ({0}) viene scaricata da GitHub, controllata con il suo SHA-256 e "
                   "installata nella cartella (su macOS l'applicazione, su Windows e Linux l'eseguibile spirula). {1}",
                   "The latest release of Spirula Studio ({0}) is downloaded from GitHub, checked with its SHA-256 and "
                   "installed into the folder (the application on macOS, the spirula executable on Windows and Linux). {1}")
                 .format(RELEASES, t("Quella trovata ora: {0}.", "The one found now: {0}.").format(found) if found else ""))
        if v is None:
            return
        settings = self.context.settings

        def job(progress):
            report = installer.install(v["folder"], progress)
            settings["spirula"] = str(report["executable"])
            self.context.save_settings()
            return report

        def done(report):
            message(window, "info", self.title,
                    t("Spirula Studio {0} installato: {1}", "Spirula Studio {0} installed: {1}")
                    .format(report["version"], report["executable"]))

        self.context.run(t("Installazione di Spirula Studio", "Installing Spirula Studio"), job, done)

    def _run(self, title: str, job, done=None):
        window = self.context.window
        if window is not None:
            return window.run_task(title, job, done, refresh=True, save=True)
        return self.context.run(title, job, done)

    @staticmethod
    def splat_path(chunk) -> Path | None:
        p = chunk.product_path("splat")
        return p if p is not None and p.exists() else None

    # ------------------------------------------------------------------ exporter
    def export_splat(self, chunk, path, progress):
        """File > Esporta: a copy of the splat PLY."""
        splat = self.splat_path(chunk)
        if splat is None:
            raise RuntimeError(t("Il chunk non ha uno splat: Plugin > Spirula > Addestra Gaussian splatting.",
                                 "The chunk has no splat: Plugin > Spirula > Train Gaussian splatting."))
        progress(0.2, splat.name)
        shutil.copy2(splat, path)
        progress(1.0, str(path))
        return path

    # ------------------------------------------------------------------ window
    def where_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        window = self.context.window
        if window is None:
            return
        found = self.spirula()
        v = ParamDialog.ask(window, t("Dove sta Spirula Studio", "Where Spirula Studio is"), [
            Field("spirula", t("Eseguibile spirula (o l'app, o la cartella)", "spirula executable (or the app, or its folder)"),
                  str(found or self.context.settings.get("spirula", "")), kind="file_open", filter="spirula (*);;Tutti i file (*)"),
        ], intro=t("Spirula Studio si scarica da {0} (Windows e Linux: cartella zip con spirula; macOS: Spirula Studio.app). "
                   "Il plugin lo cerca da solo nel PATH e nelle cartelle solite; qui si puo' indicare dove sta.",
                   "Spirula Studio is downloaded from {0} (Windows and Linux: a zip folder with spirula; macOS: Spirula "
                   "Studio.app). The plugin looks for it on the PATH and in the usual folders; here you can say where it is.")
                 .format(RELEASES))
        if v is None:
            return
        self.context.settings["spirula"] = v["spirula"].strip()
        self.context.save_settings()
        exe = self.spirula()
        message(window, "info" if exe else "warning", self.title,
                t("Spirula: {0}", "Spirula: {0}").format(exe) if exe else t("Nessun eseguibile spirula in {0}.",
                                                                            "No spirula executable at {0}.").format(v["spirula"]))

    def train_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        if chunk.product_path("reconstruction") is None or not chunk.aligned_cameras:
            message(window, "warning", self.title, t("Allinea le foto prima.", "Align the photos first."))
            return
        exe = self._spirula_or_ask()
        if exe is None:
            return
        s = self.context.settings
        suggested = runner.suggested_divisor(chunk)
        has_cloud = chunk.product_path("point_cloud") is not None
        v = ParamDialog.ask(window, t("Gaussian splatting (Spirula Studio)", "Gaussian splatting (Spirula Studio)"), [
            Field("preset", "Preset", s.get("preset", "3dgs"), choices=PRESET_LABELS,
                  tip=t("in-the-wild: luce che cambia fra le foto; centered-object: un oggetto girato intorno; meshing: per la mesh",
                        "in-the-wild: light changing between photos; centered-object: an object walked around; meshing: for the mesh")),
            Field("quality", t("Qualita' (iterazioni)", "Quality (iterations)"), s.get("quality", "medium"), choices=QUALITY_LABELS),
            Field("iterations", t("Iterazioni (vuoto = della qualita')", "Iterations (empty: the quality's)"), None,
                  kind="optional_float"),
            Field("divisor", t("Riduzione delle foto (1 = piena risoluzione)", "Photo reduction (1: full resolution)"),
                  float(s.get("divisor", suggested)), minimum=1.0, maximum=16.0, decimals=1,
                  tip=t("Suggerita {0}: circa 4 megapixel per foto", "Suggested {0}: about 4 megapixels per photo").format(suggested)),
            Field("sh_degree", t("Grado delle armoniche sferiche (0-3)", "Spherical harmonics degree (0-3)"),
                  int(s.get("sh_degree", 3)), minimum=0, maximum=3),
            Field("seed", t("Punti di partenza", "Seed points"), s.get("seed", "dense" if has_cloud else "sparse"),
                  choices=[("dense", t("nuvola densa del chunk", "dense cloud of the chunk")),
                           ("sparse", t("punti di legame", "tie points"))]),
            Field("masks", t("Usa le maschere delle foto", "Use the photo masks"), bool(s.get("masks", True))),
            Field("viewer", t("Visualizzatore web di Spirula durante l'addestramento", "Spirula web viewer while training"),
                  bool(s.get("viewer", False))),
            Field("port", t("Porta del visualizzatore", "Viewer port"), int(s.get("port", 7007)), minimum=1024, maximum=65535),
        ], intro=t("Spirula Studio addestra gli splat dal modello COLMAP e dalle foto del chunk, sulla GPU (Vulkan). "
                   "Lo splat (splat.ply) diventa un prodotto del chunk: File > Esporta > Splat, o Apri in Spirula Studio.",
                   "Spirula Studio trains the splats from the COLMAP model and the photos of the chunk, on the GPU "
                   "(Vulkan). The splat (splat.ply) becomes a product of the chunk: File > Export > Splat, or Open in "
                   "Spirula Studio."))
        if v is None:
            return
        s.update({k: v[k] for k in ("preset", "quality", "divisor", "sh_degree", "seed", "masks", "viewer", "port")})
        self.context.save_settings()
        log = self.context.log

        def job(progress):
            return runner.train(chunk, exe, preset=v["preset"], quality=v["quality"],
                                iterations=int(v["iterations"]) if v["iterations"] else None, divisor=v["divisor"],
                                sh_degree=v["sh_degree"], seed=v["seed"], masks=v["masks"], viewer=v["viewer"],
                                port=v["port"], progress=progress, log=log)

        def done(report):
            message(window, "info", self.title, summary_train(report))

        self._run(t("Gaussian splatting", "Gaussian splatting"), job, done)

    def mesh_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog, message

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        if not (chunk.meta.get("splat") or {}).get("run"):
            message(window, "warning", self.title, t("Addestra prima lo splat.", "Train the splat first."))
            return
        exe = self._spirula_or_ask()
        if exe is None:
            return
        s = self.context.settings
        v = ParamDialog.ask(window, t("Mesh dallo splat (Spirula Studio)", "Mesh from the splat (Spirula Studio)"), [
            Field("formats", t("Formati", "Formats"), s.get("mesh_formats", ["ply"]), kind="checklist",
                  choices=[(f, f.upper()) for f in runner.MESH_FORMATS]),
            Field("colour", t("Colore", "Colour"), s.get("mesh_colour", "vertex"),
                  choices=[("vertex", t("per vertice", "per vertex")), ("texture", t("texture (OBJ, glTF, GLB)", "texture (OBJ, glTF, GLB)")),
                           ("none", t("nessuno", "none"))]),
            Field("texture_size", t("Lato della texture (0 = di Spirula)", "Texture side (0: Spirula's)"),
                  int(s.get("mesh_texture_size", 0)), minimum=0, maximum=16384),
        ], intro=t("I file vanno accanto allo splat; una mesh PLY entra anche fra le versioni del modello (menu Versioni).",
                   "The files go beside the splat; a PLY mesh also enters the versions of the model (Versions menu)."))
        if v is None or not v["formats"]:
            return
        s.update({"mesh_formats": v["formats"], "mesh_colour": v["colour"], "mesh_texture_size": v["texture_size"]})
        self.context.save_settings()
        log = self.context.log

        def job(progress):
            return runner.mesh(chunk, exe, formats=v["formats"], colour=v["colour"], texture_size=v["texture_size"],
                               progress=progress, log=log)

        def done(report):
            message(window, "info", self.title, t("Mesh: {0}", "Mesh: {0}").format(", ".join(p.name for p in report["files"])))

        self._run(t("Mesh dallo splat", "Mesh from the splat"), job, done)

    def open_splat(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        window = self.context.window
        chunk = self._chunk()
        if chunk is None:
            return
        splat = self.splat_path(chunk)
        if splat is None:
            message(window, "warning", self.title, t("Il chunk non ha uno splat.", "The chunk has no splat."))
            return
        exe = self._spirula_or_ask()
        if exe is not None:
            runner.open_viewer(exe, splat)

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import argparse

        parser = argparse.ArgumentParser(prog="ppg plugin spirula", description="Gaussian splatting with Spirula Studio.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("train", help="train the splats of a chunk")
        p.add_argument("project")
        p.add_argument("--chunk", help="label of the chunk (the active one by default)")
        p.add_argument("--preset", choices=runner.PRESETS, default="3dgs")
        p.add_argument("--quality", choices=runner.QUALITIES, default="medium")
        p.add_argument("--iterations", type=int)
        p.add_argument("--divisor", type=float, default=0.0, help="photo reduction (0: suggested, 1: full resolution)")
        p.add_argument("--sh-degree", type=int, default=3)
        p.add_argument("--seed", choices=["dense", "sparse"], default="dense")
        p.add_argument("--no-masks", action="store_true")
        p.add_argument("--viewer", action="store_true", help="Spirula's web viewer while training")
        p.add_argument("--port", type=int, default=7007)
        p.add_argument("--name", help="name of the run")
        p.add_argument("--spirula", help="the executable (else the setting, PPG_SPIRULA, the PATH)")
        p.add_argument("extra", nargs="*", help="more flags for spirula train, e.g. -- --cap-max 500000")
        p = sub.add_parser("mesh", help="a mesh from the splats of a chunk")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--format", default="ply", help="among ply,obj,gltf,glb,stl, separated by commas")
        p.add_argument("--color", choices=runner.MESH_COLOURS, default="vertex")
        p.add_argument("--texture-size", type=int, default=0)
        p.add_argument("--run", help="folder of a run (the last one by default)")
        p.add_argument("--spirula")
        p = sub.add_parser("open", help="open the splat of a chunk in Spirula Studio")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--spirula")
        p = sub.add_parser("where", help="the spirula executable the plugin uses")
        p.add_argument("--set", help="remember this path")
        p = sub.add_parser("install", help="download the Spirula Studio release for this computer and install it")
        p.add_argument("--dir", help="the folder (the system's by default: /Applications, %LOCALAPPDATA%\\Programs\\Spirula "
                                     "Studio, ~/.local/share/spirula-studio)")
        a = parser.parse_args(list(args))
        settings = self.context.settings if self.context is not None else {}
        if a.what == "install":
            from pyarchinit_photogrammetry.core.progress import print_progress

            try:
                report = installer.install(a.dir, print_progress)
            except installer.InstallError as exc:
                print(f"error: {exc}")
                return 1
            settings["spirula"] = str(report["executable"])
            if self.context is not None:
                self.context.save_settings()
            print(f"Spirula Studio {report['version']} installed: {report['executable']}")
            return 0
        if a.what == "where":
            if a.set:
                settings["spirula"] = a.set
                if self.context is not None:
                    self.context.save_settings()
            exe = runner.find_spirula(settings.get("spirula"))
            print(exe if exe else f"spirula not found: download it from {RELEASES} and run: ppg plugin spirula where --set PATH")
            return 0 if exe else 1
        import pyarchinit_photogrammetry as ppg
        from pyarchinit_photogrammetry.core.progress import print_progress

        doc = ppg.Document.open(a.project)
        chunk = doc.chunk if not a.chunk else next((c for c in doc.chunks if c.label == a.chunk), None)
        if chunk is None:
            print(f"no chunk {a.chunk!r} in {a.project}" if a.chunk else f"no active chunk in {a.project}")
            return 2
        try:
            exe = runner.require(a.spirula or settings.get("spirula"))
            if a.what == "train":
                divisor = a.divisor if a.divisor else runner.suggested_divisor(chunk)
                report = runner.train(chunk, exe, preset=a.preset, quality=a.quality, iterations=a.iterations,
                                      divisor=divisor, sh_degree=a.sh_degree, seed=a.seed, masks=not a.no_masks,
                                      viewer=a.viewer, port=a.port, name=a.name, extra=a.extra, progress=print_progress)
                doc.save()
                print(summary_train(report))
            elif a.what == "mesh":
                report = runner.mesh(chunk, exe, formats=[f.strip() for f in a.format.split(",") if f.strip()],
                                     colour=a.color, texture_size=a.texture_size, run=a.run, progress=print_progress)
                doc.save()
                print("mesh: " + ", ".join(str(p) for p in report["files"]))
            else:
                splat = self.splat_path(chunk)
                if splat is None:
                    print("the chunk has no splat: ppg plugin spirula train")
                    return 1
                runner.open_viewer(exe, splat)
                print(f"{splat} opened in {exe}")
        except (runner.SpirulaMissing, RuntimeError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        return 0


def summary_train(report: dict) -> str:
    """What was trained, in words."""
    lines = [t("Splat: {0}", "Splat: {0}").format(report.get("path", report.get("file", "")))]
    lines.append(t("  preset {0}, qualita' {1}, {2:,} passi, {3:,} splat, {4} foto",
                   "  preset {0}, quality {1}, {2:,} steps, {3:,} splats, {4} photos")
                 .format(report.get("preset"), report.get("quality"), int(report.get("steps") or 0),
                         int(report.get("splats") or 0), report.get("photos", 0)))
    lines.append(t("  punti di partenza: {0} ({1:,}), maschere: {2}, tempo {3} s",
                   "  seed: {0} ({1:,}), masks: {2}, time {3} s")
                 .format(report.get("seed"), int(report.get("seed_points") or 0),
                         t("si'", "yes") if report.get("masks") else "no", report.get("seconds", 0)))
    return "\n".join(lines)
