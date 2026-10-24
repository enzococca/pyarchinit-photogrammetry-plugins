"""Mesh (MeshLab) - a plugin of pyarchinit-photogrammetry: MeshLab's filters on the model of the
chunk, through pymeshlab (GPL-3, the Visual Computing Lab of ISTI-CNR), each result a new version
of the model.

* quadric edge collapse decimation (to a number or a fraction of the faces), with the texture
  kept by MeshLab's variant that carries the texture coordinates of the face corners;
* isotropic explicit remeshing (regular triangles of one edge length);
* smoothing, Taubin (no shrinking) or HC Laplacian, the borders kept where they are;
* holes closed up to a number of edges;
* topology repaired: duplicate vertices and faces, faces without area, non-manifold edges and
  vertices, unused vertices;
* measures: area, volume of a closed model, parts, borders, topology defects, size.

The *Mesh (MeshLab)* pane of the project dock and the *Plugin > Mesh (MeshLab)* menu in the
window; ``ppg plugin mesh-tools decimate|remesh|smooth|close-holes|repair|measure|install|status``
from the command line. pymeshlab is never imported by the program: it runs in a process of its own
(:mod:`.worker`) with this Python when pymeshlab is installed here, else in an environment the
plugin installs with uv (:mod:`.environment`). What the program already does - Poisson meshing,
the cleaning of the model by confidence and pieces, cutting with planes - stays the program's.
With the program 1.3.4 the pane is a dock of its own (``context.add_pane``, ``show_pane``) and the
model and the workspace are shown again with ``show_model`` and ``refresh_tree``; older programs
keep the tab and the window's own methods.
"""
from __future__ import annotations

import argparse
import json

from pyarchinit_photogrammetry.plugins import Plugin

from . import environment
from .texts import number, plural, t, title

__version__ = "0.1.1"
COMMAND_OPS = {"decimate": "decimate", "remesh": "remesh", "smooth": "smooth", "close-holes": "close_holes",
               "repair": "repair"}


class MeshToolsPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t("Mesh (MeshLab)", "Mesh (MeshLab)")
        self.description = t("I filtri di MeshLab sul modello: decimazione che conserva la texture, rimaglia isotropa, "
                             "levigatura, chiusura dei buchi, riparazione della topologia, misure",
                             "MeshLab's filters on the model: decimation keeping the texture, isotropic remeshing, "
                             "smoothing, holes closed, topology repaired, measures")
        self.context = None
        self.pane = None
        self.last = None  # the last measures shown in the pane: (chunk, caption, rows)
        self._busy_signal = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        environment.configure(context)
        for op, text in (("decimate", ("Decima il modello...", "Decimate the model...")),
                         ("remesh", ("Rimaglia il modello (isotropa)...", "Remesh the model (isotropic)...")),
                         ("smooth", ("Leviga il modello...", "Smooth the model...")),
                         ("close_holes", ("Chiudi i buchi del modello...", "Close the holes of the model...")),
                         ("repair", ("Ripara la topologia del modello...", "Repair the topology of the model...")),
                         ("measure", ("Misura il modello", "Measure the model"))):
            context.add_action(t(*text), lambda op=op: self.operation(op), processing=True)
        context.add_action(t("Pannello Mesh (MeshLab)", "Mesh (MeshLab) pane"), self.show_pane,
                           tip=t("Il pannello Mesh (MeshLab) del dock del progetto", "The Mesh (MeshLab) pane of the project dock"))
        context.add_action(t("Installa MeshLab (pymeshlab)...", "Install MeshLab (pymeshlab)..."), self.install_dialog,
                           processing=True,
                           tip=t("Python 3.12 e pymeshlab in un ambiente del plugin (circa 300 MB)",
                                 "Python 3.12 and pymeshlab in an environment of the plugin (about 300 MB)"))
        context.add_command("mesh-tools", self.command,
                            help="MeshLab's filters on the model: decimate, remesh, smooth, close-holes, repair, measure, "
                                 "install, status")
        if context.window is not None:
            self._install_pane(context.window)
            # the pane follows the active model: another project, chunk, processing step or undo
            context.on_refresh(self.refresh_pane)

    def deactivate(self) -> None:
        signal, self._busy_signal = self._busy_signal, None
        if signal is not None:
            try:
                signal.disconnect(self._busy)
            except (RuntimeError, TypeError):  # already gone with the window
                pass
        window = self.context.window if self.context else None
        tabs = getattr(window, "left_tabs", None) if window is not None else None
        if hasattr(self.context, "add_pane"):  # program 1.3.4: the context takes the pane away
            tabs = None
        if tabs is not None and self.pane is not None:
            k = tabs.indexOf(self.pane)
            if k >= 0:
                tabs.removeTab(k)
        self.pane = None

    def _install_pane(self, window) -> None:
        from .pane import MeshToolsPane

        self.pane = MeshToolsPane()
        tabs = getattr(window, "left_tabs", None)
        if hasattr(self.context, "add_pane"):  # program 1.3.4: a dock of its own, beside the workspace
            self.context.add_pane(self.pane, t("Mesh (MeshLab)", "Mesh (MeshLab)"), "model", area="right")
        elif tabs is not None:
            try:
                from pyarchinit_photogrammetry.gui.icons import icon

                tabs.addTab(self.pane, icon("model"), t("Mesh (MeshLab)", "Mesh (MeshLab)"))
            except Exception:  # noqa: BLE001 - an older program without that icon
                tabs.addTab(self.pane, t("Mesh (MeshLab)", "Mesh (MeshLab)"))
        self.pane.operation_requested.connect(self.operation)
        self.pane.install_requested.connect(self.install_dialog)
        self.pane.show_requested.connect(lambda: self._show_model(None))
        signal = getattr(getattr(window, "runner", None), "busy_changed", None)
        if signal is not None:  # one step at a time: the buttons wait while the program works
            signal.connect(self._busy)
            self._busy_signal = signal
        self.refresh_pane()

    def _busy(self, busy: bool) -> None:
        if self.pane is not None:
            self.pane.set_busy(bool(busy))

    def refresh_pane(self) -> None:
        from . import model

        if self.pane is None:
            return
        problem = environment.unsupported()
        kind, _version = environment.where()
        self.pane.set_status(environment.status_text(), can_install=problem is None and kind is None)
        chunk = self._chunk()
        text = None
        if chunk is not None and chunk.product_path("model") is not None and chunk.product_path("model").exists():
            try:
                s = model.summary(chunk)
                text = t("Modello attivo «{0}»: {1} facce, {2} vertici", "Active model '{0}': {1} faces, {2} vertices").format(
                    s["name"], number(s["faces"]), number(s["vertices"]))
                if s["pages"]:
                    text += t(", texture ({0} {1})", ", texture ({0} {1})").format(
                        s["pages"], plural(s["pages"], t("pagina", "page"), t("pagine", "pages")))
                if not s["metric"]:
                    text += t("; chunk senza scala: lunghezze in unità del modello",
                              "; chunk without a scale: lengths in units of the model")
            except Exception as exc:  # noqa: BLE001 - a model the program cannot read: say so
                text = t("Modello non leggibile: {0}", "Model not readable: {0}").format(exc)
        self.pane.set_model(text, ready=problem is None)
        if self.last is not None and self.last[0] is chunk:
            self.pane.set_measures(self.last[2], self.last[1])
        else:
            self.pane.set_measures([])

    def show_pane(self) -> None:
        window = self.context.window
        if window is None or self.pane is None:
            return
        if hasattr(self.context, "show_pane"):  # program 1.3.4: also when the user closed or floated it
            self.context.show_pane(self.pane)
            return
        tabs = getattr(window, "left_tabs", None)
        if tabs is not None:
            tabs.setCurrentWidget(self.pane)
        dock = getattr(window, "d_work", None)
        if dock is not None:
            dock.show()
            dock.raise_()

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        window = self.context.window
        chunk = getattr(window, "chunk", None) if window is not None else None
        return chunk if chunk is not None else getattr(doc, "chunk", None)

    def _message(self, kind: str, text: str, details: str = "") -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text, details)

    def _status(self, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            return
        log = getattr(window, "log_pane", None)
        if log is not None:
            log.info(text)
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 15000)

    def _chunk_with_model(self):
        chunk = self._chunk()
        path = chunk.product_path("model") if chunk is not None else None
        if path is None or not path.exists():
            self._message("warning", t("Il chunk non ha un modello: costruiscilo prima (Flusso di lavoro > Costruisci "
                                       "modello).", "The chunk has no model: build it first (Workflow > Build model)."))
            return None
        return chunk

    def _ready(self, then) -> bool:
        """MeshLab can run now. When it is not installed, offer to install it and ``then()`` after; the
        first time in a session, check in the background that it works (on Linux it can lack system
        libraries: the message names them) and ``then()`` after."""
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return False
        if environment.where()[0] is not None:
            if environment.known_usable() or self.context.window is None:
                return True  # the command line: the operation checks it by itself (worker_python)

            def checked(info):
                self.refresh_pane()
                if environment.usable(info):
                    then()
                else:
                    self._message("warning", environment.problem_text(info))

            self.context.run(t("Prova di MeshLab", "Trying MeshLab"), lambda p: environment.check(p), checked)
            return False
        window = self.context.window
        if window is None:
            print(environment.status_text())
            return False
        from pyarchinit_photogrammetry.gui import dialogs

        text = t("MeshLab (pymeshlab) non è installato. Il plugin lo installa in un ambiente suo, separato dal "
                 "programma: Python 3.12 e pymeshlab da PyPI con uv, circa 300 MB e qualche minuto. Installarlo ora?",
                 "MeshLab (pymeshlab) is not installed. The plugin installs it in an environment of its own, apart "
                 "from the program: Python 3.12 and pymeshlab from PyPI with uv, about 300 MB and a few minutes. "
                 "Install it now?")
        if dialogs.choice(window, "question", self.title, text, [t("Installa", "Install")]) != 0:
            return False
        self._install(then)
        return False

    def install_dialog(self) -> None:
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return
        kind, version = environment.where()
        if kind == "local":
            self._message("info", t("pymeshlab {0} è già installato per questo Python: il plugin usa quello.",
                                    "pymeshlab {0} is already installed for this Python: the plugin uses it.").format(version))
            return
        if kind == "environment" and self.context.window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            text = t("MeshLab è già installato (pymeshlab {0}). Installarlo di nuovo, all'ultima versione?",
                     "MeshLab is already installed (pymeshlab {0}). Install it again, at the latest version?").format(version)
            if dialogs.choice(self.context.window, "question", self.title, text, [t("Installa di nuovo", "Install again")]) != 0:
                return
        self._install(None)

    def _install(self, then) -> None:
        def done(env):
            self.refresh_pane()
            self._status(environment.status_text())
            if then is not None:
                then()

        self.context.run(t("Installazione di MeshLab", "Installing MeshLab"), lambda p: environment.install(p), done)

    # ------------------------------------------------------------------ the operations
    def operation(self, op: str) -> None:
        """An operation from the menu or the pane: its dialog, then the work in the background."""
        from . import model

        chunk = self._chunk_with_model()
        if chunk is None or not self._ready(lambda: self.operation(op)):
            return
        if op == "measure":
            self.context.run(title("measure"), lambda p: model.measure(chunk, p), lambda m: self._measured(chunk, m))
            return
        asked = self._ask(op, chunk)
        if asked is None:
            return
        params, name, keep = asked
        self.context.run(f"MeshLab: {title(op)}",
                         lambda p: model.apply(chunk, op, params, p, name=name, keep_version=keep),
                         lambda r: self._applied(chunk, r))

    def _ask(self, op: str, chunk):
        """The parameters of ``op`` from its dialog: ``(params, version name, keep the version)`` or None."""
        from pyarchinit_photogrammetry.core import versions
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import model

        s = model.summary(chunk)
        u = s["unit"]
        d = model.DEFAULTS[op]
        intro = t("Modello attivo «{0}»: {1} facce", "Active model '{0}': {1} faces").format(s["name"], number(s["faces"]))
        if s["textured"]:
            intro += t(", con texture", ", textured")
        intro += ". "
        if op == "decimate":
            fields = [Field("target_faces", t("Facce da ottenere (0 = la frazione qui sotto)",
                                              "Faces to reach (0 = the fraction below)"), 0, minimum=0, maximum=2_000_000_000),
                      Field("fraction", t("Frazione delle facce da tenere", "Fraction of the faces to keep"), d["fraction"],
                            minimum=0.001, maximum=0.999, decimals=3)]
            if s["textured"]:
                fields.append(Field("texture", t("Conserva la texture (variante di MeshLab con le coordinate della "
                                                 "texture)", "Keep the texture (MeshLab's variant with the texture "
                                                 "coordinates)"), True))
            fields += [Field("preserve_boundary", t("Conserva i bordi", "Keep the borders"), d["preserve_boundary"]),
                       Field("preserve_topology", t("Conserva la topologia (senza texture: nessun buco chiuso o aperto)",
                                                    "Keep the topology (without texture: no hole closed or opened)"),
                             d["preserve_topology"]),
                       Field("quality", t("Qualità dei triangoli (0-1: più alto, triangoli più regolari)",
                                          "Quality of the triangles (0-1: higher, more regular triangles)"), d["quality"],
                             minimum=0.0, maximum=1.0, decimals=2),
                       Field("planar", t("Semplificazione planare (zone piane con triangoli migliori)",
                                         "Planar simplification (flat areas with better triangles)"), d["planar"]),
                       Field("preserve_normal", t("Non capovolgere le facce", "Do not flip faces"), d["preserve_normal"])]
            intro += t("Collasso degli spigoli con le quadriche di MeshLab: la forma resta, le facce diminuiscono.",
                       "Quadric edge collapse of MeshLab: the shape stays, the faces go down.")
        elif op == "remesh":
            edge = model.mean_edge(chunk)
            fields = [Field("edge_length", t("Lato dei triangoli ({0})", "Edge of the triangles ({0})").format(u),
                            float(f"{edge:.3g}") if edge > 0 else 0.01, minimum=1e-6, maximum=1e6, decimals=6),
                      Field("iterations", t("Iterazioni", "Iterations"), d["iterations"], minimum=1, maximum=100),
                      Field("adaptive", t("Adattiva (triangoli più piccoli dove la superficie curva)",
                                          "Adaptive (smaller triangles where the surface curves)"), d["adaptive"]),
                      Field("feature_angle", t("Spigoli vivi da conservare: angolo (gradi)",
                                               "Sharp edges to keep: angle (degrees)"), d["feature_angle"],
                            minimum=0.0, maximum=180.0, decimals=1),
                      Field("max_distance", t("Distanza massima dalla superficie di partenza ({0}, 0 = nessun controllo)",
                                              "Largest distance from the surface before ({0}, 0 = no check)").format(u),
                            0.0, minimum=0.0, maximum=1e6, decimals=6),
                      Field("repair_first", t("Ripara prima la topologia (su un modello non-manifold MeshLab si chiude)",
                                              "Repair the topology first (on a non-manifold model MeshLab crashes)"),
                            d["repair_first"])]
            intro += t("Triangoli regolari della stessa dimensione; il lato proposto è quello medio del modello. La "
                       "texture va ricostruita dopo.",
                       "Regular triangles of one size; the edge proposed is the mean one of the model. The texture has "
                       "to be built again afterwards.")
        elif op == "smooth":
            fields = [Field("method", t("Metodo", "Method"), d["method"],
                            choices=[("taubin", t("Taubin (non restringe; 10 passi)", "Taubin (no shrinking; 10 steps)")),
                                     ("hc", t("HC laplaciana (1 passo)", "HC Laplacian (1 step)"))]),
                      Field("steps", t("Passi (0 = quelli del metodo)", "Steps (0 = the method's)"), 0, minimum=0,
                            maximum=1000),
                      Field("lambda", t("Taubin: lambda", "Taubin: lambda"), d["lambda"], minimum=0.0, maximum=1.0,
                            decimals=3),
                      Field("mu", t("Taubin: mu", "Taubin: mu"), d["mu"], minimum=-1.0, maximum=0.0, decimals=3),
                      Field("keep_borders", t("I bordi restano fermi", "The borders stay where they are"),
                            d["keep_borders"])]
            intro += t("La texture resta (i vertici si spostano di poco).", "The texture stays (the vertices move a little).")
        elif op == "close_holes":
            fields = [Field("max_hole_edges", t("Buchi fino a (lati del bordo)", "Holes up to (edges of the border)"),
                            d["max_hole_edges"], minimum=3, maximum=10_000_000),
                      Field("self_intersection", t("Evita facce che si intersecano", "Avoid faces that intersect"),
                            d["self_intersection"]),
                      Field("refine_edge", t("Raffina le facce nuove a un lato di ({0}, 0 = no)",
                                             "Refine the new faces to an edge of ({0}, 0 = no)").format(u), 0.0,
                            minimum=0.0, maximum=1e6, decimals=6),
                      Field("keep_outline", t("Lascia aperto il contorno esterno di ogni parte (il suo bordo più "
                                              "lungo)", "Leave the outline of each part open (its longest border)"),
                            d["keep_outline"],
                            tip=t("Su una superficie di scavo il contorno è un bordo come gli altri: chiuderlo la coprirebbe "
                                  "con un tappo. Toglilo per chiudere il fondo di un oggetto.",
                                  "On an excavated surface the outline is a border like the others: closing it would cap "
                                  "the surface. Untick it to close the bottom of an object.")),
                      Field("repair_first", t("Ripara prima la topologia (MeshLab chiude i buchi solo su modelli "
                                              "2-manifold)", "Repair the topology first (MeshLab closes holes only on "
                                              "2-manifold models)"), d["repair_first"])]
            intro += t("Il contorno esterno di ogni parte resta aperto, se non chiedi altrimenti. Se si chiudono dei "
                       "buchi, la texture va ricostruita.",
                       "The outline of each part stays open, unless you ask otherwise. If holes are closed, the "
                       "texture has to be built again.")
        else:
            fields = [Field("duplicate_vertices", t("Vertici doppi", "Duplicate vertices"), True),
                      Field("duplicate_faces", t("Facce doppie", "Duplicate faces"), True),
                      Field("null_faces", t("Facce senza area", "Faces without area"), True),
                      Field("non_manifold_edges", t("Spigoli non-manifold", "Non-manifold edges"), True),
                      Field("non_manifold_method", t("Spigoli non-manifold: come", "Non-manifold edges: how"), "remove",
                            choices=[("remove", t("togli le facce di troppo", "remove the faces in excess")),
                                     ("split", t("dividi i vertici", "split the vertices"))]),
                      Field("non_manifold_vertices", t("Vertici non-manifold (divisi)", "Non-manifold vertices (split)"),
                            True),
                      Field("unreferenced_vertices", t("Vertici non usati", "Unused vertices"), True)]
            intro += t("La texture resta. Dopo, le misure dicono se il modello è chiuso.",
                       "The texture stays. Afterwards the measures say whether the model is closed.")
        if versions.active_files(chunk, "model"):
            fields.append(Field("keep", t("Modello attuale", "Current model"), "keep",
                                choices=[("keep", t("Conservalo tra le versioni ({0})", "Keep it among the versions ({0})")
                                          .format(s["name"])), ("overwrite", t("Sovrascrivilo", "Overwrite it"))]))
        fields.append(Field("version_name", t("Nome della nuova versione (facoltativo)", "Name of the new version (optional)"),
                            "", kind="text"))
        v = ParamDialog.ask(self.context.window, f"MeshLab: {title(op)}", fields, intro=intro)
        if not v:
            return None
        params = {k: val for k, val in v.items() if k not in ("keep", "version_name")}
        return params, (v.get("version_name") or "").strip() or None, v.get("keep", "keep") == "keep"

    def _applied(self, chunk, report: dict) -> None:
        """In the interface thread after an operation: the project saved (the model changed on disk),
        the result in the 3D view, its measures in the pane."""
        from . import model

        text = model.report_text(report)
        if report.get("changed"):
            self._project_changed()
            window = self.context.window
            if hasattr(self.context, "refresh_tree"):  # program 1.3.4
                self.context.refresh_tree()  # the workspace lists the new version
            elif window is not None and hasattr(window, "refresh_all"):
                window.refresh_all()
            self._show_model(report)
        rows = model.measure_rows(report.get("measures") or {})
        self.last = (chunk, t("Misure dopo «{0}»", "Measures after '{0}'").format(title(report["operation"])), rows)
        self.refresh_pane()
        if self.pane is not None:
            self.pane.set_last(text)
            self.show_pane()
        self._status(text)
        if not report.get("changed"):
            self._message("info", text)

    def _measured(self, chunk, measures: dict) -> None:
        from . import model

        rows = model.measure_rows(measures)
        self.last = (chunk, t("Misure del modello attivo", "Measures of the active model"), rows)
        self.refresh_pane()
        self.show_pane()
        watertight = t("chiuso", "closed") if measures.get("watertight") else t("aperto", "open")
        self._status(t("Modello misurato: {0}, area {1} {2}²", "Model measured: {0}, area {1} {2}²").format(
            watertight, number(measures.get("area", 0.0), 3), measures.get("unit", "m")))
        if self.pane is None:
            self._message("info", model.measures_text(measures))

    def _project_changed(self) -> None:
        """The model and its versions changed on disk: the project is saved with them, as the program
        does after its own processing steps; a project never saved is marked modified instead."""
        doc = self.context.app.document
        if doc is not None and getattr(doc, "path", None):
            try:
                self.context.store_drawings()
                doc.save()
                return
            except Exception as exc:  # noqa: BLE001 - the change stays marked: the user saves
                self.context.log.warning("saving the project: %s", exc)
        self.context.mark_modified()

    def _show_model(self, report: dict | None) -> None:
        window = self.context.window if self.context else None
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        textured = chunk.product_path("model_textured")
        kind = "model_textured" if textured is not None and textured.exists() and (report is None or report.get(
            "texture_kept")) else "model"
        if hasattr(self.context, "show_model"):  # program 1.3.4
            self.context.show_model(kind)
            return
        if hasattr(window, "_show_model"):
            window._show_model(kind)
            return
        view = getattr(window, "model_view", None)
        if view is not None:
            view.set_chunk(chunk, kind)
            if hasattr(window, "show_view"):
                window.show_view(view)

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _project_args(args) -> tuple[list[str], str | None, str | None]:
        """``ppg plugin mesh-tools WHAT PROJECT.ppg ...``: the project given right after the
        subcommand (or with ``--project``) and ``--chunk`` taken out of the arguments; without a
        project the commands work on the document the program has open."""
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

    def _open_for_command(self, project: str | None, label: str | None):
        if project:
            import pyarchinit_photogrammetry as ppg

            doc = ppg.Document.open(project)
            if self.context is not None:
                self.context.app.document = doc
            return next((c for c in doc.chunks if c.label == label), None) if label else doc.chunk
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return None
        return next((c for c in doc.chunks if c.label == label), None) if label else doc.chunk

    def command(self, *args) -> int:
        from pyarchinit_photogrammetry.core.progress import Cancelled

        from . import model
        from .runner import WorkerError

        parser = argparse.ArgumentParser(prog="ppg plugin mesh-tools",
                                         description="MeshLab's filters (pymeshlab) on the model of a chunk; every "
                                                     "result a new version of the model. Lengths in metres.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("decimate", help="quadric edge collapse decimation (the texture kept)")
        p.add_argument("--faces", type=int, default=0, help="faces to reach (0: --fraction of them)")
        p.add_argument("--fraction", type=float, default=0.5)
        p.add_argument("--no-texture", action="store_true", help="the plain variant: the texture is left out")
        p.add_argument("--quality", type=float, default=0.3, help="quality threshold of the triangles (0-1)")
        p.add_argument("--free-borders", action="store_true", help="the borders may move")
        p.add_argument("--free-topology", action="store_true", help="holes may close or open (without texture)")
        p.add_argument("--planar", action="store_true", help="planar simplification")
        p.add_argument("--keep-normals", action="store_true", help="no face flipped")
        p = sub.add_parser("remesh", help="isotropic explicit remeshing")
        p.add_argument("--edge", type=float, default=0.0, help="edge of the triangles (m; 0: the mean edge of the model)")
        p.add_argument("--iterations", type=int, default=10)
        p.add_argument("--adaptive", action="store_true")
        p.add_argument("--feature-angle", type=float, default=30.0)
        p.add_argument("--max-distance", type=float, default=0.0, help="largest distance from the surface before (m)")
        p.add_argument("--no-repair", action="store_true", help="no topology repair first")
        p = sub.add_parser("smooth", help="Taubin or HC Laplacian smoothing")
        p.add_argument("--method", choices=["taubin", "hc"], default="taubin")
        p.add_argument("--steps", type=int, default=0, help="0: the method's (Taubin 10, HC 1)")
        p.add_argument("--lambda", dest="lam", type=float, default=0.5)
        p.add_argument("--mu", type=float, default=-0.53)
        p.add_argument("--move-borders", action="store_true", help="the borders are smoothed too")
        p = sub.add_parser("close-holes", help="holes closed up to a number of edges")
        p.add_argument("--max-edges", type=int, default=30)
        p.add_argument("--refine", type=float, default=0.0, help="new faces refined to this edge (m)")
        p.add_argument("--allow-self-intersections", action="store_true")
        p.add_argument("--close-outline", action="store_true",
                       help="the outline of each part (its longest border) may be closed too: the bottom of an object")
        p.add_argument("--no-repair", action="store_true", help="no topology repair first")
        p = sub.add_parser("repair", help="duplicates, faces without area, non-manifold edges and vertices, unused vertices")
        p.add_argument("--split", action="store_true", help="non-manifold edges by splitting vertices (not removing faces)")
        p.add_argument("--skip", action="append", default=[], choices=list(model.REPAIR_KEYS), help="a step to leave out")
        for name in COMMAND_OPS:
            p = sub.choices[name]
            p.add_argument("--name", help="name of the new version")
            p.add_argument("--overwrite", action="store_true", help="the model before is not kept as a version")
            p.add_argument("--json", action="store_true", help="the report as JSON")
        p = sub.add_parser("measure", help="area, volume, parts, borders, topology of the model")
        p.add_argument("--json", action="store_true")
        p = sub.add_parser("install", help="install pymeshlab in an environment of the plugin (uv)")
        p.add_argument("--force", action="store_true", help="also when this Python has pymeshlab, or again")
        sub.add_parser("status", help="where MeshLab runs")
        args, project, label = self._project_args(args)
        ns = parser.parse_args(args)
        if ns.what == "status":  # where MeshLab runs, and whether it works there
            if environment.where()[0] is not None:
                try:
                    environment.check(again=True)
                except environment.MeshLabMissing as exc:
                    print(exc)
                    return 1
            print(environment.status_text())
            return 0 if environment.known_usable() else 1
        if ns.what == "install":
            kind, _version = environment.where()
            if kind is not None and not ns.force:
                print(environment.status_text() + " " + t("Niente da installare (--force per installarlo di nuovo).",
                                                          "Nothing to install (--force to install it again)."))
                return 0
            try:
                env = environment.install(_print_progress)
            except (environment.MeshLabMissing, RuntimeError) as exc:
                print(exc)
                return 1
            print(env)
            return 0
        chunk = self._open_for_command(project, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  f"no project: ppg plugin mesh-tools {ns.what} PROJECT.ppg ...")
            return 1
        try:
            if ns.what == "measure":
                m = model.measure(chunk, _print_progress)
                print(json.dumps(m, indent=1) if ns.json else model.measures_text(m))
                return 0
            op = COMMAND_OPS[ns.what]
            params = _command_params(op, ns)
            report = model.apply(chunk, op, params, _print_progress, name=ns.name, keep_version=not ns.overwrite)
        except (environment.MeshLabMissing, WorkerError, RuntimeError, ValueError) as exc:
            if isinstance(exc, Cancelled):
                raise
            print(exc)
            return 1
        doc = self.context.app.document if self.context is not None else None
        if report.get("changed") and doc is not None and getattr(doc, "path", None):
            doc.save()
        if ns.json:
            print(json.dumps(report, indent=1, default=str))
        else:
            print(model.report_text(report))
            if report.get("name"):
                print(t("Nuova versione: {0}", "New version: {0}").format(report["name"]))
            if report.get("measures"):
                print(model.measures_text(report["measures"]))
        return 0


def _print_progress(fraction: float, message: str) -> None:
    print(f"[{fraction * 100:5.1f}%] {message}", flush=True)


def _command_params(op: str, ns) -> dict:
    """The parameters of an operation from the options of its command."""
    if op == "decimate":
        return {"target_faces": ns.faces, "fraction": ns.fraction, "texture": not ns.no_texture, "quality": ns.quality,
                "preserve_boundary": not ns.free_borders, "preserve_topology": not ns.free_topology,
                "planar": ns.planar, "preserve_normal": ns.keep_normals}
    if op == "remesh":
        return {"edge_length": ns.edge, "iterations": ns.iterations, "adaptive": ns.adaptive,
                "feature_angle": ns.feature_angle, "max_distance": ns.max_distance, "repair_first": not ns.no_repair}
    if op == "smooth":
        return {"method": ns.method, "steps": ns.steps, "lambda": ns.lam, "mu": ns.mu, "keep_borders": not ns.move_borders}
    if op == "close_holes":
        return {"max_hole_edges": ns.max_edges, "refine_edge": ns.refine, "keep_outline": not ns.close_outline,
                "self_intersection": not ns.allow_self_intersections, "repair_first": not ns.no_repair}
    from .model import REPAIR_KEYS

    params = {k: k not in ns.skip for k in REPAIR_KEYS}
    params["non_manifold_method"] = "split" if ns.split else "remove"
    return params
