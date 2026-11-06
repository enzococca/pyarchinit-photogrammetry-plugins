"""The resistivity tomography in the window and on the command line: a part of the geophysics plugin
(:class:`ErtMixin`, mixed into :class:`ppg_geophysics.GeophysicsPlugin`).

In the window: the ERT tab of the Geofisica pane (:mod:`.ert_pane`), the line drawn on the
orthomosaic or DEM with the program's drawing (``context.request_line``), the pseudosection as points
on an empty frame of the 2D view (``context.show_frame``; the readings selected and deleted there leave
the inversion, ``context.on_features_deleted``), the section shown as a raster of its own frame
(``context.show_raster``: u along the line, v the elevation) with the electrodes and a legend, the 3D
scene of the sections with the ground cut away, the units and the cores (``context.show_model``), the
line and the electrodes on the plan as layers. Every change of a profile marks the project modified:
the program's Undo takes ``chunk.meta`` back, and the files agree with it - the placed electrodes are
in the record, every inversion and every drawing of a section has files named in the record, a
removed profile waits in a trash folder until the project is opened again (:func:`.ert.settle`, when
the window shows another document).

On the command line: ``ppg plugin geophysics ert import|simulate|list|place|filter|invert|export|scene|
remove|install|status``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ert as ert_mod
from . import ert_environment as environment
from . import ert_ops
from . import ert_section as section
from .texts import t

PSEUDO_LAYER = "Pseudosezione ERT"
ELECTRODES_LAYER = "Elettrodi ERT"
LINES_LAYER = "Profili ERT"
REMOVED_LAYER = "Letture tolte ERT"


class ErtMixin:
    """The ERT part of the plugin (it uses ``self.context``, ``self.pane``, ``_chunk``, ``_message``,
    ``_dirty``, ``_view`` of the plugin)."""

    _ert_shown = None  # ("pseudo" | "section", profile id): what the plugin shows in the 2D view
    _ert_plan = False  # the lines of the profiles on the plan rasters
    _ert_legend = None
    _ert_doc = None  # the document whose files were settled (once per opened project)
    _ert_frame = None  # the bounds of the empty frame of the pseudosection

    # ------------------------------------------------------------------ setup
    def _ert_activate(self, context) -> None:
        environment.configure(context)
        context.add_action(t("Tomografia elettrica: importa letture...", "Resistivity tomography: import readings..."),
                           self.ert_import_dialog,
                           tip=t("Le letture di un georesistivimetro come profilo ERT", "The readings of a resistivity "
                                 "meter as an ERT profile"))
        context.add_action(t("Tomografia elettrica (ERT)...", "Resistivity tomography (ERT)..."), self.ert_show_tab,
                           tip=t("La scheda ERT del pannello Geofisica", "The ERT tab of the Geofisica pane"))
        context.add_action(t("Installa pyGIMLi...", "Install pyGIMLi..."), self.ert_install_dialog, processing=True,
                           tip=t("L'inversione delle tomografie, in un ambiente separato dal programma",
                                 "The inversion of the tomographies, in an environment apart from the program"))
        if context.window is not None and hasattr(context, "on_features_deleted"):
            context.on_features_deleted(PSEUDO_LAYER, self._ert_readings_deleted)

    def _ert_connect(self, pane) -> None:
        p = pane
        p.import_requested.connect(self.ert_import_dialog)
        p.simulate_requested.connect(self.ert_simulate_dialog)
        p.draw_line_requested.connect(self.ert_draw_line)
        p.place_requested.connect(self.ert_place_dialog)
        p.pseudo_requested.connect(self.ert_show_pseudo)
        p.filter_requested.connect(self.ert_filter_dialog)
        p.invert_requested.connect(self.ert_invert_dialog)
        p.section_requested.connect(self.ert_show_section)
        p.view3d_requested.connect(self.ert_show_3d)
        p.plan_requested.connect(self.ert_show_plan)
        p.palette_changed.connect(self.ert_set_palette)
        if hasattr(p, "blank_requested"):
            p.blank_requested.connect(self.ert_blank_dialog)
        p.export_requested.connect(self.ert_export_dialog)
        p.remove_requested.connect(self.ert_remove_dialog)
        p.install_requested.connect(self.ert_install_dialog)

    def _ert_pane(self):
        return getattr(self.pane, "ert", None) if self.pane is not None else None

    def _ert_settle(self) -> None:
        """A project just opened (another document in the window): the files of its profiles made to
        agree with it as saved; after an Undo of a removal, the folder of the profile taken back."""
        doc = self.context.app.document if self.context is not None else None
        if doc is None:
            return
        try:
            if doc is not self._ert_doc:
                self._ert_doc = doc
                for chunk in list(getattr(doc, "chunks", []) or []):
                    ert_mod.settle(chunk)
            elif doc.chunk is not None:
                for e in ert_mod.entries(doc.chunk):
                    ert_mod.restore(doc.chunk, e["id"])
        except OSError:
            pass

    def ert_refresh(self) -> None:
        self._ert_settle()
        pane = self._ert_pane()
        if pane is None:
            return
        problem = environment.unsupported()
        kind, _v = environment.where()
        status = t("Inversione non disponibile: ", "Inversion not available: ") + problem if problem \
            else environment.status_text()
        pane.set_environment(status, can_install=problem is None and kind is None, available=problem is None)
        chunk = self._chunk()
        pane.set_profiles(ert_mod.entries(chunk) if chunk is not None else [])

    def ert_show_tab(self) -> None:
        self.show_pane()
        pane = self._ert_pane()
        tabs = getattr(self.pane, "tabs", None) if self.pane is not None else None
        if tabs is not None and pane is not None:
            tabs.setCurrentWidget(self.pane.ert_scroll)

    def _ert_changed(self, pid: str | None = None) -> None:
        self._dirty()
        self.ert_refresh()
        pane = self._ert_pane()
        if pane is not None and pid:
            pane.select(pid)

    def _ert_pid(self, pid: str | None) -> str | None:
        pane = self._ert_pane()
        return pid or (pane.selected_id() if pane is not None else None)

    def _ert_status(self, text: str, ms: int = 12000) -> None:
        window = self.context.window if self.context else None
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, ms)

    # ------------------------------------------------------------------ pyGIMLi
    def _ert_ready(self, then) -> bool:
        """pyGIMLi can run now; else offer to install it (``then()`` after) or say why it cannot."""
        if ert_ops.WORKER is not None:
            return True
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return False
        if environment.where()[0] is not None:
            if environment.known_usable() or self.context.window is None:
                return True

            def checked(info):
                self.ert_refresh()
                if environment.usable(info):
                    then()
                else:
                    self._message("warning", environment.problem_text(info))

            self.context.run(t("Prova di pyGIMLi", "Trying pyGIMLi"), lambda p: environment.check(p), checked)
            return False
        window = self.context.window
        if window is None:
            print(environment.status_text())
            return False
        from pyarchinit_photogrammetry.gui import dialogs

        text = t("pyGIMLi non è installato. Il plugin lo installa in un ambiente suo, separato dal programma: Python {0} "
                 "e pyGIMLi {1} da PyPI con uv, circa {2} MB (più circa {3} MB per Python se uv non lo ha già) e un paio "
                 "di minuti. Installarlo ora?",
                 "pyGIMLi is not installed. The plugin installs it in an environment of its own, apart from the "
                 "program: Python {0} and pyGIMLi {1} from PyPI with uv, about {2} MB (plus about {3} MB for Python "
                 "when uv has not it yet) and a couple of minutes. Install it now?").format(
            environment.python_for(), "1.6", environment.SIZE_MB, environment.PYTHON_MB)
        if dialogs.choice(window, "question", self.title, text, [t("Installa", "Install")]) != 0:
            return False
        self._ert_install(then)
        return False

    def ert_install_dialog(self) -> None:
        problem = environment.unsupported()
        if problem:
            self._message("warning", problem)
            return
        kind, version = environment.where()
        if kind == "local":
            self._message("info", t("pyGIMLi {0} è già installato per questo Python: il plugin usa quello.",
                                    "pyGIMLi {0} is already installed for this Python: the plugin uses it.").format(version))
            return
        window = self.context.window
        if window is not None:
            from pyarchinit_photogrammetry.gui import dialogs

            if kind == "environment":
                text = t("pyGIMLi è già installato ({0}). Installarlo di nuovo?",
                         "pyGIMLi is already installed ({0}). Install it again?").format(version)
            else:
                text = t("Il plugin installa pyGIMLi in un ambiente suo, separato dal programma: Python {0} e pyGIMLi "
                         "da PyPI con uv, circa {1} MB (più circa {2} MB per Python se uv non lo ha già). Procedere?",
                         "The plugin installs pyGIMLi in an environment of its own, apart from the program: Python {0} "
                         "and pyGIMLi from PyPI with uv, about {1} MB (plus about {2} MB for Python when uv has not it "
                         "yet). Go on?").format(environment.python_for(), environment.SIZE_MB, environment.PYTHON_MB)
            if dialogs.choice(window, "question", self.title, text, [t("Installa", "Install")]) != 0:
                return
        self._ert_install(None)

    def _ert_install(self, then) -> None:
        def done(_env):
            self.ert_refresh()
            self._ert_status(environment.status_text())
            if then is not None:
                then()

        self.context.run(t("Installazione di pyGIMLi", "Installing pyGIMLi"), lambda p: environment.install(p), done)

    # ------------------------------------------------------------------ import
    def ert_import_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None:
            return
        v = ParamDialog.ask(window, t("Importa letture ERT", "Import ERT readings"), [
            Field("file", t("Letture", "Readings"), "", kind="file_open",
                  filter="ERT (*.csv *.txt *.ohm *.dat *.shm *.data *.bin *.amp *.stg);;Tutti (*)"),
            Field("electrodes", t("Elettrodi (facoltativo: x, z o x, y, z)", "Electrodes (optional: x, z or x, y, z)"), "",
                  kind="file_open", filter="CSV (*.csv *.txt);;Tutti (*)"),
            Field("name", t("Nome (vuoto = nome del file)", "Name (empty = the file name)"), "", kind="text"),
            Field("electrodes_as", t("Colonne a, b, m, n", "Columns a, b, m, n"), "auto",
                  choices=[("auto", t("riconosci da sole", "recognise by themselves")),
                           ("numbers", t("numeri degli elettrodi (da 1)", "numbers of the electrodes (from 1)")),
                           ("positions", t("posizioni in metri", "positions in metres"))]),
            Field("spacing", t("Distanza fra gli elettrodi numerati (m, vuoto = 1)", "Spacing of the numbered "
                                 "electrodes (m, empty = 1)"), None, kind="optional_float",
                  tip=t("Solo per le letture con i numeri degli elettrodi; con le posizioni in metri conta quella del "
                        "file. Si può cambiare anche dopo, nel posizionamento della linea.",
                        "Only for readings with the numbers of the electrodes; with positions in metres the file's "
                        "counts. It can also be changed later, when placing the line.")),
        ], intro=t("Tabella CSV/TXT con intestazione (a, b, m, n e rhoa, oppure r, oppure u e i; err facoltativo; anche "
                   "le colonne Spa.1-4, Rho, Dev, Vp, In di Syscal) o formato unificato di pyGIMLi (.ohm, .dat). Gli "
                   "altri file (RES2DINV, ABEM, Syscal) li legge pyGIMLi.",
                   "CSV/TXT table with a header (a, b, m, n and rhoa, or r, or u and i; err optional; also Syscal's "
                   "Spa.1-4, Rho, Dev, Vp, In columns) or pyGIMLi's unified format (.ohm, .dat). The other files "
                   "(RES2DINV, ABEM, Syscal) are read by pyGIMLi."))
        if not v or not v["file"]:
            return
        positions = {"auto": None, "numbers": False, "positions": True}[v["electrodes_as"]]

        def job(progress):
            progress(0.05, t("Lettura delle letture", "Reading the readings"))
            spacing = float(v["spacing"]) if v.get("spacing") else None
            return ert_ops.import_profile(chunk, v["file"], v["name"], spacing, positions, v["electrodes"] or None,
                                          progress)

        def done(e):
            self._ert_changed(e["id"])
            self.ert_show_tab()
            self._message("info", ert_ops.summary_line(e) + (("\n" + "; ".join(e.get("notes") or []))
                                                             if e.get("notes") else ""))

        self.context.run(t("Letture ERT", "ERT readings"), job, done)

    def ert_simulate_dialog(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        if window is None or chunk is None or not self._ert_ready(self.ert_simulate_dialog):
            return
        v = ParamDialog.ask(window, t("Profilo sintetico", "Synthetic profile"), [
            Field("name", t("Nome", "Name"), t("Sintetico", "Synthetic"), kind="text"),
            Field("count", t("Elettrodi", "Electrodes"), 48, minimum=8, maximum=256),
            Field("spacing", t("Distanza (m)", "Spacing (m)"), 1.0, minimum=0.1, maximum=50.0, decimals=2),
            Field("scheme", t("Dispositivo", "Array"), "dd", choices=[("dd", t("dipolo-dipolo", "dipole-dipole")),
                                                                    ("wa", "Wenner"), ("slm", "Wenner-Schlumberger"),
                                                                    ("pd", t("polo-dipolo", "pole-dipole"))]),
            Field("rho1", t("Resistività in alto (Ω·m)", "Resistivity at the top (Ω·m)"), 50.0, minimum=0.1, maximum=1e6),
            Field("depth2", t("Profondità del secondo strato (m)", "Depth of the second layer (m)"), 3.0, minimum=0.1,
                  maximum=500.0, decimals=2),
            Field("rho2", t("Resistività del secondo strato (Ω·m)", "Resistivity of the second layer (Ω·m)"), 200.0,
                  minimum=0.1, maximum=1e6),
            Field("block_rho", t("Blocco al centro: resistività (Ω·m, 0 = nessuno)", "Block in the middle: resistivity "
                                 "(Ω·m, 0 = none)"), 2000.0, minimum=0.0, maximum=1e6),
            Field("block_top", t("Blocco: profondità del tetto (m)", "Block: depth of its top (m)"), 1.5, minimum=0.0,
                  maximum=500.0, decimals=2),
            Field("block_base", t("Blocco: profondità della base (m)", "Block: depth of its base (m)"), 3.5, minimum=0.1,
                  maximum=500.0, decimals=2),
            Field("slope", t("Pendenza del terreno (%)", "Slope of the ground (%)"), 0.0, minimum=-50.0, maximum=50.0,
                  decimals=1),
            Field("noise", t("Rumore (%)", "Noise (%)"), 2.0, minimum=0.0, maximum=50.0, decimals=1),
        ], intro=t("Letture calcolate da pyGIMLi su un modello a due strati con un blocco: per provare il flusso o "
                   "capire che cosa risolve una linea.",
                   "Readings computed by pyGIMLi on a two-layer model with a block: to try the workflow or to see what "
                   "a line resolves."))
        if not v:
            return
        length = (int(v["count"]) - 1) * float(v["spacing"])
        model = {"background": float(v["rho1"]), "layers": [[float(v["depth2"]), float(v["rho2"])]], "blocks": []}
        if float(v["block_rho"]) > 0:
            model["blocks"].append([0.42 * length, 0.58 * length, float(v["block_top"]), float(v["block_base"]),
                                    float(v["block_rho"])])

        def job(progress):
            return ert_ops.simulate_profile(chunk, v["name"] or "ERT", int(v["count"]), float(v["spacing"]), v["scheme"],
                                            model, float(v["slope"]) / 100.0, float(v["noise"]) / 100.0,
                                            progress=progress)

        def done(e):
            self._ert_changed(e["id"])
            self._message("info", ert_ops.summary_line(e))

        self.context.run(t("Profilo sintetico", "Synthetic profile"), job, done)

    # ------------------------------------------------------------------ placing
    def _place_fields(self, e: dict, with_line: bool = False):
        from pyarchinit_photogrammetry.gui.dialogs import Field

        chunk = self._chunk()
        has_dem = chunk is not None and chunk.dem_path is not None and Path(chunk.dem_path).exists()
        fields = []
        if with_line:
            names = [("", t("(nessuno)", "(none)"))] + [(m.label, m.label) for m in chunk.markers]
            methods = [("markers", t("due marker (inizio, fine)", "two markers (start, end)")),
                       ("points", t("coordinate digitate", "typed coordinates"))]
            if e.get("world"):
                methods.insert(0, ("file", t("coordinate degli elettrodi nel file (sistema del chunk)",
                                             "electrode coordinates of the file (system of the chunk)")))
            fields += [Field("method", t("Linea da", "Line from"),
                             "file" if e.get("world") else ("markers" if chunk.markers else "points"), choices=methods),
                       Field("m_start", t("Marker dell'inizio (primo elettrodo)", "Marker of the start (first electrode)"),
                             "", choices=names),
                       Field("m_end", t("Marker della fine", "Marker of the end"), "", choices=names),
                       Field("points", t("Punti: 'X Y' per riga o separati da ';'", "Points: 'X Y' per line or "
                                         "separated by ';'"), "", kind="text")]
        fields += [Field("first", t("Primo elettrodo a (m dall'inizio della linea)", "First electrode at (m from the "
                                    "start of the line)"), 0.0, minimum=-1e4, maximum=1e4, decimals=3),
                   Field("spacing", t("Distanza fra gli elettrodi (m)", "Spacing of the electrodes (m)"),
                         float(e.get("spacing", 1.0)), minimum=0.01, maximum=1000.0, decimals=3),
                   Field("along", t("Distanza misurata", "Spacing measured"),
                         "plan" if e.get("heights_in_file", e.get("format") == "unified") else "ground",
                         choices=[("ground", t("lungo il terreno (il cavo steso sul pendio)",
                                               "along the ground (the cable laid on the slope)")),
                                  ("plan", t("in pianta (orizzontale)", "in plan (horizontal)"))],
                         tip=t("Sul campo la distanza fra gli elettrodi si misura lungo il cavo, sul terreno: su un "
                               "pendio del 20 % in pianta è il 2 % più corta. Un file con le quote (x, z, come lo "
                               "scrivono pyGIMLi e BERT) dà le distanze in pianta. Con le quote fisse non cambia nulla.",
                               "In the field the spacing is measured along the cable, on the ground: on a 20 % slope "
                               "it is 2 % shorter in plan. A file with the heights (x, z, as pyGIMLi and BERT write it) "
                               "gives the distances in plan. With flat heights nothing changes.")),
                   Field("heights", t("Quote degli elettrodi", "Heights of the electrodes"),
                         "dem" if has_dem else ("file" if e.get("format") == "unified" else "flat"),
                         choices=[("dem", t("dal DEM del chunk", "from the DEM of the chunk")),
                                  ("file", t("dal file (+ quota sotto)", "from the file (+ height below)")),
                                  ("flat", t("piatte alla quota sotto", "flat at the height below"))]),
                   Field("z0", t("Quota (m)", "Height (m)"), 0.0, minimum=-1e5, maximum=1e5, decimals=3)]
        return fields

    def ert_draw_line(self, pid: str | None = None) -> None:
        """The line drawn on the orthomosaic or DEM (two or more clicks, double click or Enter to end)."""
        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid or not hasattr(self.context, "request_line"):
            return
        view = self._view()
        if self._ert_shown is not None or (view is not None and view.tr is None):
            self._ert_shown = None
            if not self.context.show_raster():
                self._message("warning", t("Niente su cui disegnare la linea: costruisci l'ortofoto o il DEM, oppure usa le "
                                           "coordinate o i marker.", "Nothing to draw the line on: build the orthomosaic "
                                           "or the DEM, or use the coordinates or the markers."))
                return
        self.context.request_line(lambda xy: self._ert_line_drawn(pid, xy), lambda: self._ert_status(
            t("Linea annullata", "Line cancelled"), 5000), hint=t(
            "Linea ERT: clic sul primo elettrodo, poi sugli altri vertici; doppio clic o Invio per finire, Esc annulla",
            "ERT line: click the first electrode, then the other vertices; double click or Enter to end, Esc cancels"))

    def _ert_line_drawn(self, pid: str, xy) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        chunk = self._chunk()
        if chunk is None or len(xy) < 2:
            return
        e = ert_mod.entry_of(chunk, pid)
        line = np.asarray(xy, float)[:, :2]
        v = ParamDialog.ask(self.context.window, t("Linea ERT", "ERT line"), self._place_fields(e),
                            intro=t("Linea di {0:.2f} m in pianta; gli elettrodi del file coprono {1:.2f} m.",
                                    "Line of {0:.2f} m in plan; the electrodes of the file span {1:.2f} m.").format(
                                ert_mod.line_length(line), float(e.get("spacing", 1)) * (e["electrodes"] - 1)))
        if v:
            self._ert_place(pid, line, v, "drawn")

    def ert_place_dialog(self, pid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid:
            return
        e = ert_mod.entry_of(chunk, pid)
        intro = t("La linea va dal primo elettrodo verso l'ultimo; due punti o più (una linea piegata).",
                  "The line goes from the first electrode towards the last; two points or more (a bent line).")
        if e.get("world"):
            intro = t("Il file dà gli elettrodi in coordinate X, Y: se sono nel sistema del chunk la linea passa per "
                      "loro; se sono un sistema locale dello strumento, dai la linea con i marker, le coordinate o "
                      "il pulsante Disegna la linea.",
                      "The file gives the electrodes in X, Y coordinates: when they are in the system of the chunk the "
                      "line goes through them; when they are a local frame of the instrument, give the line by markers, "
                      "coordinates or the Draw the line button.") + "\n" + intro
        v = ParamDialog.ask(window, t("Linea ERT", "ERT line"), self._place_fields(e, with_line=True), intro=intro)
        if not v:
            return
        try:
            if v["method"] == "file":
                line = None
            elif v["method"] == "markers":
                pos = self._marker_positions(chunk)
                if v["m_start"] not in pos or v["m_end"] not in pos:
                    raise ValueError(t("Scegli due marker con una posizione.", "Choose two markers with a position."))
                line = [pos[v["m_start"]], pos[v["m_end"]]]
            else:
                line = parse_points(v["points"])
        except ValueError as exc:
            self._message("warning", str(exc))
            return
        self._ert_place(pid, line, v, v["method"])

    def _ert_place(self, pid: str, line, v: dict, method: str) -> None:
        chunk = self._chunk()
        try:
            e = ert_mod.place(chunk, pid, line, float(v["first"]), float(v["spacing"]), v["heights"], float(v["z0"]),
                              method, v.get("along", "ground"))
        except ValueError as exc:
            self._message("warning", str(exc))
            return
        self._ert_changed(pid)
        self.ert_show_plan(pid)
        pl = e["placement"]
        self._ert_status(t("Linea {0}: {1:.1f} m, quote {2:.2f} - {3:.2f} m", "Line {0}: {1:.1f} m, heights {2:.2f} - "
                           "{3:.2f} m").format(e["name"], pl["length"], *pl["z_range"]))

    # ------------------------------------------------------------------ pseudosection and filters
    def ert_show_pseudo(self, pid: str | None = None) -> None:
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        view = self._view()
        if chunk is None or not pid or view is None:
            return
        prof = ert_mod.load(chunk, pid)
        x, d = ert_mod.pseudo_positions(prof)
        u = prof.positions()
        m = 0.05 * float(u.max() - u.min()) + prof.spacing
        bounds = (float(u.min()) - m, -float(np.nanmax(d)) - m, float(u.max()) + m, m)
        res = (bounds[2] - bounds[0]) / 1600.0
        self._ert_shown = ("pseudo", pid)
        self._ert_frame = (bounds[0], bounds[3], res)
        self.context.show_frame(bounds, res, "")
        window = self.context.window
        if window is not None and hasattr(window, "show_view"):
            window.show_view(view)
        self._fit_frame(view)
        self._ert_status(t("Pseudosezione: x = distanza lungo la linea, y = pseudoprofondità (convenzionale). Seleziona "
                           "le letture da togliere con lo strumento Seleziona e premi Canc.",
                           "Pseudosection: x = distance along the line, y = pseudo-depth (conventional). Select the "
                           "readings to remove with the Select tool and press Delete."), 20000)

    @staticmethod
    def _fit_frame(view) -> None:
        """The whole empty frame in the view: the program's ``fit`` frames only a raster."""
        try:
            from PySide6.QtCore import Qt

            view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:  # noqa: BLE001 - a view without it keeps its zoom
            pass

    def _ert_draw_pseudo(self, pid: str) -> None:
        chunk = self._chunk()
        view = self._view()
        prof = ert_mod.load(chunk, pid)
        e = ert_mod.entry_of(chunk, pid)
        valid = ert_mod.valid_mask(prof, e.get("filters"))
        x, d = ert_mod.pseudo_positions(prof)
        rhoa = prof.apparent()
        ok = valid & np.isfinite(rhoa) & (rhoa > 0)
        lo, hi = section.limits(rhoa[ok]) if ok.any() else (1.0, 100.0)
        rgb = section.colours(np.where(ok, rhoa, np.nan)[None, :], lo, hi, "rainbow")[0]
        for layer in (PSEUDO_LAYER, REMOVED_LAYER, ELECTRODES_LAYER):
            view.remove_layer(layer)
        batch = view.batch() if hasattr(view, "batch") else None
        if batch is not None:
            batch.__enter__()
        try:
            for i in range(prof.count):
                attrs = {"lettura": i + 1, "a": int(prof.a[i]) + 1, "b": int(prof.b[i]) + 1 if prof.b[i] >= 0 else "",
                         "m": int(prof.m[i]) + 1, "n": int(prof.n[i]) + 1 if prof.n[i] >= 0 else "",
                         "rhoa": round(float(rhoa[i]), 3), "profilo": e["name"]}
                if prof.has("err"):
                    attrs["err_pc"] = round(float(prof.value("err")[i]) * 100, 2)
                if ok[i]:
                    c = rgb[i]
                    view.add_point(float(x[i]), -float(d[i]), color="#%02x%02x%02x" % (c[0], c[1], c[2]),
                                   layer=PSEUDO_LAYER, attrs=attrs)
                elif np.isfinite(x[i]):
                    view.add_point(float(x[i]), -float(d[i]), color="#9e9e9e", layer=REMOVED_LAYER, attrs=attrs)
            for k, u in enumerate(prof.positions()):
                view.add_point(float(u), 0.0, color="#202020", layer=ELECTRODES_LAYER, attrs={"elettrodo": k + 1})
        finally:
            if batch is not None:
                batch.__exit__(None, None, None)
        self._ert_put_legend(lo, hi, "rainbow", t("ρa Ω·m", "ρa Ω·m"))

    def _ert_readings_deleted(self, features) -> None:
        chunk = self._chunk()
        if chunk is None or not self._ert_shown or self._ert_shown[0] != "pseudo":
            return
        pid = self._ert_shown[1]
        idx = [int(f["lettura"]) - 1 for f in features if f.get("lettura")]
        if not idx:
            return
        e = ert_mod.set_filters(chunk, pid, remove_readings=idx)
        self._ert_changed(pid)
        self._ert_status(t("{0} letture tolte; restano {1} valide", "{0} readings removed; {1} valid left")
                         .format(len(idx), e["valid"]))

    def ert_filter_dialog(self, pid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid:
            return
        e = ert_mod.entry_of(chunk, pid)
        f = e.get("filters") or {}
        r = {**ert_mod.DEFAULT_RULES, **(f.get("rules") or {})}
        v = ParamDialog.ask(window, t("Filtri delle letture", "Filters of the readings"), [
            Field("min_rhoa", t("Resistività apparente minima (Ω·m)", "Smallest apparent resistivity (Ω·m)"), r["min_rhoa"],
                  kind="optional_float"),
            Field("max_rhoa", t("Resistività apparente massima (Ω·m)", "Largest apparent resistivity (Ω·m)"), r["max_rhoa"],
                  kind="optional_float"),
            Field("max_error", t("Errore massimo della lettura (%)", "Largest error of a reading (%)"), r["max_error"],
                  kind="optional_float", tip=t("Solo se il file ha l'errore (deviazione) delle letture",
                                               "Only when the file has the error (deviation) of the readings")),
            Field("reciprocal", t("Errore reciproco massimo (%)", "Largest reciprocal error (%)"), r["reciprocal"],
                  kind="optional_float", tip=t("Solo per le letture ripetute con elettrodi di corrente e potenziale "
                                               "scambiati", "Only for readings repeated with the current and potential "
                                               "electrodes swapped")),
            Field("spikes", t("Picchi: scarto dai vicini (deviazioni robuste)", "Spikes: departure from the neighbours "
                              "(robust deviations)"), r["spikes"], kind="optional_float",
                  tip=t("Per esempio 8: toglie le letture molto diverse da quelle vicine nella pseudosezione",
                        "For instance 8: removes the readings far from their neighbours in the pseudosection")),
            Field("clear", t("Rimetti le letture tolte a mano ({0})", "Give back the readings removed by hand ({0})")
                  .format(len(f.get("manual") or [])), False),
        ], intro=t("Le letture nulle o negative sono sempre tolte. Vuoto = regola spenta.",
                   "Zero or negative readings are always removed. Empty = rule off."))
        if not v:
            return
        e = ert_mod.set_filters(chunk, pid, {k: v[k] for k in ("min_rhoa", "max_rhoa", "max_error", "reciprocal",
                                                                 "spikes")}, clear_manual=bool(v["clear"]))
        self._ert_changed(pid)
        if self._ert_shown == ("pseudo", pid):
            self.ert_show_pseudo(pid)
        self._ert_status(t("{0} letture valide su {1}", "{0} valid readings of {1}").format(e["valid"], e["readings"]))

    # ------------------------------------------------------------------ inversion
    def ert_invert_dialog(self, pid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid or not self._ert_ready(lambda: self.ert_invert_dialog(pid)):
            return
        e = ert_mod.entry_of(chunk, pid)
        old = {**ert_ops.DEFAULT_PARAMS, **((e.get("inversion") or {}).get("params") or {})}
        blank = {**section.DEFAULT_BLANK, **(((e.get("inversion") or {}).get("display") or {}).get("blank") or {})}
        v = ParamDialog.ask(window, t("Inversione ERT", "ERT inversion"), [
            Field("h1", t("Regolarizzazione", "Regularisation"), kind="heading"),
            Field("lam", t("Lambda (più alto = più liscio)", "Lambda (higher = smoother)"), float(old["lam"]), minimum=0.1,
                  maximum=10000.0, decimals=2),
            Field("z_weight", t("Peso verticale (meno di 1 = strati orizzontali)", "Vertical weight (below 1 = horizontal "
                                "layers)"), float(old["z_weight"]), minimum=0.01, maximum=10.0, decimals=2),
            Field("blocky", t("Modello a blocchi (contrasti netti, norma L1)", "Blocky model (sharp contrasts, L1 norm)"),
                  bool(old["blocky"])),
            Field("robust", t("Dati robusti (letture anomale pesano meno)", "Robust data (outlying readings weigh less)"),
                  bool(old["robust"])),
            Field("max_iter", t("Iterazioni massime", "Largest number of iterations"), int(old["max_iter"]), minimum=1,
                  maximum=100),
            Field("h2", t("Errori delle letture", "Errors of the readings"), kind="heading"),
            Field("rel_error", t("Errore relativo (%)", "Relative error (%)"), 100.0 * float(old["rel_error"]),
                  minimum=0.1, maximum=50.0, decimals=2),
            Field("abs_error_u", t("Errore assoluto della tensione (µV, vuoto = nessuno)", "Absolute voltage error (µV, "
                                   "empty = none)"), None if not old["abs_error_u"] else 1e6 * float(old["abs_error_u"]),
                  kind="optional_float",
                  tip=t("Diviso per la tensione di ogni lettura; senza la tensione nel file pyGIMLi la stima dalla "
                        "resistenza con una corrente di 100 mA.",
                        "Divided by the voltage of every reading; without the voltage in the file pyGIMLi estimates it "
                        "from the resistance with a current of 100 mA.")),
            Field("use_file_errors", t("Usa gli errori del file quando sono più grandi", "Use the errors of the file "
                                       "when they are larger"), bool(old["use_file_errors"]),
                  tip=t("Lettura per lettura: dove il file non dà l'errore vale quello qui sopra.",
                        "Reading by reading: where the file gives no error the one above counts.")),
            Field("h3", t("Maglia", "Mesh"), kind="heading"),
            Field("para_depth", t("Profondità del modello (m, vuoto = automatica)", "Depth of the model (m, empty = "
                                  "automatic)"), old["para_depth"], kind="optional_float"),
            Field("para_dx", t("Nodi fra due elettrodi (frazione della distanza)", "Nodes between two electrodes "
                               "(fraction of the spacing)"), float(old["para_dx"]), minimum=0.05, maximum=1.0, decimals=2),
            Field("para_max_cell", t("Area massima di una cella (m², vuoto = automatica)", "Largest area of a cell (m², "
                                     "empty = automatic)"), old["para_max_cell"], kind="optional_float"),
            Field("h4", t("Parte risolta", "Resolved part"), kind="heading"),
            Field("doi", t("Profondità d'indagine", "Depth of investigation"), old["doi"],
                  choices=[("coverage", t("copertura (sensibilità cumulata)", "coverage (cumulative sensitivity)")),
                           ("oldenburg", t("indice DOI di Oldenburg e Li (due inversioni in più)",
                                           "DOI index of Oldenburg and Li (two more inversions)"))]),
            Field("drop", t("Copertura: oscura sotto (ordini di grandezza dal massimo)", "Coverage: blank below (orders "
                            "of magnitude from the top)"), float(blank["drop"]), minimum=0.5, maximum=6.0, decimals=2),
            Field("doi_max", t("Indice DOI: oscura sopra", "DOI index: blank above"), float(blank["doi_max"]),
                  minimum=0.05, maximum=1.0, decimals=2),
        ], intro=t("pyGIMLi inverte {0} letture valide su una maglia di triangoli che segue la topografia degli elettrodi "
                   "({1}). Le parti non risolte dalle letture sono oscurate.",
                   "pyGIMLi inverts {0} valid readings on a mesh of triangles that follows the topography of the "
                   "electrodes ({1}). The parts the readings do not resolve are blanked.").format(
            e.get("valid", e["readings"]), ((e.get("placement") or {}).get("heights") or t("non posizionato: piatto o dal "
                                                                                           "file", "not placed: flat or "
                                                                                           "from the file"))))
        if not v:
            return
        params = {"lam": float(v["lam"]), "z_weight": float(v["z_weight"]), "blocky": bool(v["blocky"]),
                  "robust": bool(v["robust"]), "max_iter": int(v["max_iter"]), "rel_error": float(v["rel_error"]) / 100.0,
                  "abs_error_u": float(v["abs_error_u"]) / 1e6 if v["abs_error_u"] else None,
                  "use_file_errors": bool(v["use_file_errors"]), "para_depth": v["para_depth"],
                  "para_dx": float(v["para_dx"]), "para_max_cell": v["para_max_cell"], "doi": v["doi"]}
        blank = {"method": "doi" if v["doi"] == "oldenburg" else "coverage", "drop": float(v["drop"]),
                 "doi_max": float(v["doi_max"])}

        def job(progress):
            return ert_ops.invert_profile(chunk, pid, params, blank, progress=progress)

        def done(e):
            self._ert_changed(pid)
            self.ert_show_section(pid)
            f = e["inversion"]["fit"]
            self._ert_status(t("{0}: χ² {1:.2f}, RMS {2:.1f} %, {3} iterazioni in {4:.0f} s",
                               "{0}: χ² {1:.2f}, RMS {2:.1f} %, {3} iterations in {4:.0f} s").format(
                e["name"], f.get("chi2", 0), f.get("rrms", 0), f.get("iterations", 0), f.get("seconds", 0)), 20000)

        self.context.run(t("Inversione ERT", "ERT inversion"), job, done)

    # ------------------------------------------------------------------ showing
    def ert_show_section(self, pid: str | None = None) -> None:
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if chunk is None or not pid:
            return
        path = section.view_path(chunk, pid)
        if path is None:
            self._message("warning", t("Inverti prima il profilo.", "Invert the profile first."))
            return
        if not hasattr(self.context, "show_raster"):
            self._message("warning", t("La sezione nella vista 2D richiede il programma 1.3.4 o successivo.",
                                       "The section in the 2D view needs the program 1.3.4 or later."))
            return
        self._ert_shown = ("section", pid)
        if not self.context.show_raster("ortho", path=path):
            self._ert_shown = None
            return
        e = ert_mod.entry_of(chunk, pid)
        self._ert_status(t("Sezione {0}: x = distanza lungo la linea (m), y = quota (m); i colori sono la resistività "
                           "in scala logaritmica.", "Section {0}: x = distance along the line (m), y = elevation (m); the "
                           "colours are the resistivity on a log scale.").format(e["name"]), 20000)

    def _ert_draw_section(self, pid: str) -> None:
        chunk = self._chunk()
        view = self._view()
        prof = ert_mod.load(chunk, pid)
        e = ert_mod.entry_of(chunk, pid)
        u, z = prof.positions(), prof.heights()
        view.remove_layer(ELECTRODES_LAYER)
        view.add_line(np.column_stack([u, z]), "#202020", "", layer=ELECTRODES_LAYER, width=1.5)
        for k in range(prof.electrodes):
            view.add_point(float(u[k]), float(z[k]), color="#202020", layer=ELECTRODES_LAYER,
                           attrs={"elettrodo": k + 1, "distanza_m": round(float(u[k]), 3), "quota_m": round(float(z[k]), 3)})
        disp = (e.get("inversion") or {}).get("display") or {}
        lo, hi = disp.get("limits") or (1.0, 1000.0)
        self._ert_put_legend(lo, hi, disp.get("palette", "rainbow"), "Ω·m")

    def _ert_put_legend(self, lo: float, hi: float, palette: str, title: str) -> None:
        """The colour bar at the bottom left of the 2D view (where the section is blanked, below the resolved
        part), the same size at every zoom; it goes with the raster."""
        from .ert_pane import legend_pixmap

        view = self._view()
        if view is None:
            return
        self._ert_drop_legend()
        pix = legend_pixmap(lo, hi, palette, 300, title)
        item = view.scene().addPixmap(pix)
        item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setOffset(0, -pix.height())  # anchored by its bottom left corner
        item.setPos(view.sceneRect().bottomLeft())
        item.setZValue(10)
        self._ert_legend = item

    def _ert_drop_legend(self) -> None:
        """The colour bar out of the view (it is gone already when the view changed raster)."""
        item, self._ert_legend = self._ert_legend, None
        if item is None:
            return
        try:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        except RuntimeError:  # the view's scene deleted it with the raster
            pass

    def ert_set_palette(self, pid: str, palette: str) -> None:
        """The section drawn again in another palette (in the background: up to 1.5 million pixels)."""
        chunk = self._chunk()
        if chunk is None:
            return
        e = ert_mod.entry_of(chunk, pid)
        if not (e.get("inversion") or {}).get("files"):
            return
        self._ert_redraw(pid, t("Colori della sezione", "Colours of the section"),
                         lambda: section.write(chunk, pid, palette=palette))

    def _ert_redraw(self, pid: str, title: str, write) -> None:
        def job(progress):
            progress(0.1, title)
            return write()

        def done(_inv):
            self._ert_changed(pid)
            if self._ert_shown == ("section", pid):
                self.ert_show_section(pid)

        self.context.run(title, job, done)

    def ert_blank_dialog(self, pid: str | None = None) -> None:
        """The part of the section the readings do not resolve blanked again with other thresholds,
        without inverting again."""
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid:
            return
        e = ert_mod.entry_of(chunk, pid)
        inv = e.get("inversion") or {}
        if not inv.get("files"):
            self._message("warning", t("Inverti prima il profilo.", "Invert the profile first."))
            return
        blank = {**section.DEFAULT_BLANK, **((inv.get("display") or {}).get("blank") or {})}
        methods = [("coverage", t("copertura (sensibilità cumulata)", "coverage (cumulative sensitivity)")),
                   ("none", t("nessuno: tutta la maglia", "none: the whole mesh"))]
        if (inv.get("params") or {}).get("doi") == "oldenburg":
            methods.insert(1, ("doi", t("indice DOI di Oldenburg e Li", "DOI index of Oldenburg and Li")))
        v = ParamDialog.ask(window, t("Oscuramento della sezione", "Blanking of the section"), [
            Field("method", t("Oscura in base a", "Blank by"), blank["method"] if blank["method"] in dict(methods)
                  else "coverage", choices=methods),
            Field("drop", t("Copertura: oscura sotto (ordini di grandezza dal massimo)", "Coverage: blank below (orders "
                            "of magnitude from the top)"), float(blank["drop"]), minimum=0.5, maximum=6.0, decimals=2),
            Field("doi_max", t("Indice DOI: oscura sopra", "DOI index: blank above"), float(blank["doi_max"]),
                  minimum=0.05, maximum=1.0, decimals=2),
        ], intro=t("La sezione è disegnata di nuovo dalla stessa inversione: l'inversione non si rifà. L'indice DOI c'è "
                   "solo se l'inversione l'ha calcolato (Profondità d'indagine: Oldenburg e Li).",
                   "The section is drawn again from the same inversion: the inversion is not done again. The DOI index "
                   "is there only when the inversion computed it (Depth of investigation: Oldenburg and Li)."))
        if not v:
            return
        new = {"method": v["method"], "drop": float(v["drop"]), "doi_max": float(v["doi_max"])}
        self._ert_redraw(pid, t("Oscuramento della sezione", "Blanking of the section"),
                         lambda: section.write(chunk, pid, blank=new))

    def ert_show_3d(self, pid: str | None = None) -> bool:
        """The 3D scene of the sections built in the background (the model of the chunk read and cut)
        and shown in the 3D view. False when there is nothing to show."""
        chunk = self._chunk()
        window = self.context.window if self.context else None
        if chunk is None or window is None:
            return False
        if not any((e.get("inversion") or {}).get("files") and e.get("placement") for e in ert_mod.entries(chunk)):
            self._message("warning", t("Nessuna sezione da mostrare: inverti e posiziona un profilo.",
                                       "No section to show: invert and place a profile."))
            return False

        def job(progress):
            progress(0.1, t("Scena 3D delle tomografie", "3D scene of the tomographies"))
            return section.scene(chunk)

        def done(result):
            path, what = result
            if path is None:
                self._message("warning", t("Nessuna sezione da mostrare: inverti e posiziona un profilo.",
                                           "No section to show: invert and place a profile."))
                return
            self._ert_show_scene(path, what)

        self.context.run(t("Scena 3D delle tomografie", "3D scene of the tomographies"), job, done)
        return True

    def _ert_show_scene(self, path, what: dict) -> bool:
        window = self.context.window
        title = t("Tomografie ERT ({0})", "ERT tomographies ({0})").format(", ".join(what["sections"]))
        if hasattr(self.context, "show_model"):
            if not self.context.show_model("model", path=path, title=title):
                return False
        else:
            view = getattr(window, "model_view", None)
            if view is None or not hasattr(view, "show_version"):
                return False
            view.show_version("model", path, title)
            if hasattr(window, "show_view"):
                window.show_view(view)
        parts = [t("terreno tagliato lungo la linea", "ground cut away along the line")] if what["terrain"] else []
        if what.get("proxies"):
            parts.append(t("{0} proxy delle US", "{0} proxies of the units").format(what["proxies"]))
        if what["units"]:
            parts.append(t("unità geoarcheologiche", "geoarchaeological units"))
        if what["cores"]:
            parts.append(t("{0} intervalli di carotaggio", "{0} core intervals").format(what["cores"]))
        self._ert_status(title + ("; " + ", ".join(parts) if parts else "") + ". " + t(
            "Scegli un'altra vista nell'elenco «Mostra» per tornare ai prodotti del chunk.",
            "Choose another view in the Show list to go back to the products of the chunk."), 20000)
        return True

    def ert_show_plan(self, pid: str | None = None) -> None:
        """The lines of the placed profiles and the electrodes of ``pid`` on the orthomosaic or DEM."""
        chunk = self._chunk()
        view = self._view()
        if chunk is None or view is None:
            return
        self._ert_plan = True
        if self._ert_shown is not None or view.tr is None:
            self._ert_shown = None
            if not (hasattr(self.context, "show_raster") and self.context.show_raster()):
                self._ert_draw_plan()
        else:
            self._ert_draw_plan()
        pid = self._ert_pid(pid)
        if pid and LINES_LAYER in getattr(view, "layers", {}):
            try:
                view.zoom_to_layer(LINES_LAYER)
            except Exception:  # noqa: BLE001
                pass
        window = self.context.window
        if window is not None and hasattr(window, "show_view") and view.tr is not None:
            window.show_view(view)

    def _ert_draw_plan(self) -> None:
        chunk = self._chunk()
        view = self._view()
        if chunk is None or view is None or view.tr is None:
            return
        view.remove_layer(LINES_LAYER)
        view.remove_layer(ELECTRODES_LAYER)
        pid = self._ert_pid(None)
        for e in ert_mod.entries(chunk):
            pl = e.get("placement")
            if not pl:
                continue
            view.add_line(pl["line"], "#00b4ff", e["name"], layer=LINES_LAYER,
                          attrs={"profilo": e["name"], "lunghezza_m": pl["length"], "elettrodi": e["electrodes"]},
                          width=3.0)
            if e["id"] == pid:
                prof = ert_mod.load(chunk, e["id"])
                for k in range(prof.electrodes):
                    view.add_point(float(prof.elec["elec_x"][k]), float(prof.elec["elec_y"][k]), color="#ffffff",
                                   layer=ELECTRODES_LAYER, label=str(k + 1) if k in (0, prof.electrodes - 1) else "",
                                   attrs={"elettrodo": k + 1, "quota_m": round(float(prof.elec["elec_z"][k]), 3)})

    def _ert_raster_loaded(self) -> None:
        """Another raster in the 2D view: the pseudosection, the section or the plan lines drawn on it
        when it is theirs."""
        view = self._view()
        chunk = self._chunk()
        if view is None or chunk is None:
            return
        shown = self._ert_shown
        try:
            if shown and shown[0] == "pseudo" and getattr(view, "kind", None) == "frame" and self._ert_own_frame(view):
                self._ert_draw_pseudo(shown[1])
                return
            if shown and shown[0] == "section":
                path = section.view_path(chunk, shown[1])
                if path is not None and getattr(view, "path", None) is not None and Path(view.path) == path:
                    self._ert_draw_section(shown[1])
                    return
            self._ert_shown = None
            if self._ert_plan and view.tr is not None and self._ert_plan_raster(view, chunk):
                self._ert_draw_plan()
        except KeyError:  # the profile left the project
            self._ert_shown = None

    def _ert_own_frame(self, view) -> bool:
        """The empty frame in the view is the one of the pseudosection (not another plugin's)."""
        f = self._ert_frame
        tr = getattr(view, "tr", None)
        if f is None or tr is None:
            return False
        tol = 1e-6 * max(1.0, abs(f[0]), abs(f[1]))
        return abs(tr.c - f[0]) <= tol and abs(tr.f - f[1]) <= tol and abs(tr.a - f[2]) <= 1e-9 * max(1.0, f[2])

    @staticmethod
    def _ert_plan_raster(view, chunk) -> bool:
        """The raster in the view is a plan of the survey: not an empty frame, not a section or an
        elevation in (u, v) - its extent meets the lines of the profiles."""
        if getattr(view, "kind", None) == "frame":
            return False
        lines = [np.asarray(e["placement"]["line"], float) for e in ert_mod.entries(chunk) if e.get("placement")]
        if not lines:
            return False
        try:
            (ax, ay), (bx, by) = view.world(view.sceneRect().topLeft()), view.world(view.sceneRect().bottomRight())
        except Exception:  # noqa: BLE001 - a view without it: draw
            return True
        x0, x1, y0, y1 = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        for line in lines:
            if line[:, 0].max() >= x0 and line[:, 0].min() <= x1 and line[:, 1].max() >= y0 and line[:, 1].min() <= y1:
                return True
        return False

    # ------------------------------------------------------------------ export, remove
    def ert_export_dialog(self, pid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid:
            return
        doc = self.context.app.document
        base = (Path(doc.path).parent if doc is not None and doc.path else Path.home()) / "ERT"
        v = ParamDialog.ask(window, t("Esporta il profilo ERT", "Export the ERT profile"), [
            Field("folder", t("Cartella", "Folder"), str(base), kind="dir")],
            intro=t("Nella cartella: sezione GeoTIFF (u, v) con il suo frame.json, celle della maglia CSV, pixel XYZ con "
                    "le coordinate, figure PNG/SVG della sezione e della pseudosezione, linea ed elettrodi GeoPackage, "
                    "paradati JSON, letture tenute .ohm.",
                    "In the folder: section GeoTIFF (u, v) with its frame.json, mesh cells CSV, pixels XYZ with the "
                    "coordinates, PNG/SVG figures of the section and the pseudosection, line and electrodes GeoPackage, "
                    "paradata JSON, readings kept .ohm."))
        if not v or not v["folder"]:
            return

        def job(progress):
            progress(0.1, t("Esportazione", "Export"))
            return ert_ops.export_all(chunk, pid, v["folder"])

        def done(files):
            self._message("info", t("Scritti {0} file in {1}:\n{2}", "Written {0} files in {1}:\n{2}").format(
                len(files), v["folder"], "\n".join(p.name for p in files)))

        self.context.run(t("Esportazione ERT", "ERT export"), job, done)

    def ert_remove_dialog(self, pid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import choice

        window = self.context.window
        chunk = self._chunk()
        pid = self._ert_pid(pid)
        if window is None or chunk is None or not pid:
            return
        e = ert_mod.entry_of(chunk, pid)
        if choice(window, "question", self.title, t("Rimuovere il profilo {0} e la sua inversione dal progetto?",
                                                    "Remove the profile {0} and its inversion from the project?")
                  .format(e["name"]), [t("Rimuovi", "Remove")]) != 0:
            return
        if self._ert_shown and self._ert_shown[1] == e["id"]:
            self._ert_shown = None
        ert_mod.remove(chunk, e["id"], keep=True)  # Undo brings it back with its files
        self._ert_changed()
        view = self._view()
        if view is not None and self._ert_plan:
            self._ert_draw_plan()

    # ------------------------------------------------------------------ command line
    def ert_command(self, *args) -> int:
        """``ppg plugin geophysics ert import|simulate|list|place|filter|invert|export|scene|remove|install|status``."""
        import argparse

        parser = argparse.ArgumentParser(prog="ppg plugin geophysics ert", description=self.ert_command.__doc__)
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("import", help="readings of a resistivity meter as a new profile")
        p.add_argument("source")
        p.add_argument("--name", default="")
        p.add_argument("--spacing", type=float, default=None, help="metres between the numbered electrodes (1)")
        p.add_argument("--electrodes", default=None, help="a table of the electrodes (x,z or x,y,z)")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--numbers", action="store_true", help="a, b, m, n are numbers of electrodes")
        g.add_argument("--positions", action="store_true", help="a, b, m, n are positions in metres")
        p = sub.add_parser("simulate", help="synthetic readings of a model with pyGIMLi")
        p.add_argument("name")
        p.add_argument("--electrodes", type=int, default=48)
        p.add_argument("--spacing", type=float, default=1.0)
        p.add_argument("--scheme", default="dd")
        p.add_argument("--model", default=None, help='JSON: {"background": 50, "layers": [[3, 200]], "blocks": '
                                                     '[[x0, x1, depth0, depth1, rho]]}')
        p.add_argument("--slope", type=float, default=0.0, help="slope of the ground (fraction)")
        p.add_argument("--noise", type=float, default=0.02)
        sub.add_parser("list")
        p = sub.add_parser("place", help="the line of a profile in the survey")
        p.add_argument("profile")
        p.add_argument("--line", default=None, help="'X1,Y1;X2,Y2[;...]' from the first electrode")
        p.add_argument("--markers", default=None, help="'START,END' markers of the chunk")
        p.add_argument("--first", type=float, default=0.0)
        p.add_argument("--spacing", type=float, default=None)
        p.add_argument("--heights", default="auto", choices=["auto", "dem", "file", "flat"])
        p.add_argument("--along", default=None, choices=["ground", "plan"],
                       help="the spacing measured along the ground (the cable on a slope) or in plan; default: in plan "
                            "when the file gives the heights, else along the ground")
        p.add_argument("--z0", type=float, default=None)
        p = sub.add_parser("filter", help="the filters of the readings")
        p.add_argument("profile")
        for name in ("min", "max", "max-error", "reciprocal", "spikes"):
            p.add_argument(f"--{name}", type=float, default=None)
        p.add_argument("--remove", default="", help="readings to remove (numbers from 1, 'a,b,c')")
        p.add_argument("--restore", default="", help="readings to give back")
        p.add_argument("--clear", action="store_true", help="give back every reading removed by hand")
        p = sub.add_parser("invert", help="the inversion with pyGIMLi")
        p.add_argument("profile")
        p.add_argument("--lam", type=float, default=20.0)
        p.add_argument("--error", type=float, default=3.0, help="relative error (%%)")
        p.add_argument("--abs-u", type=float, default=None, help="absolute voltage error (volts)")
        p.add_argument("--robust", action="store_true")
        p.add_argument("--blocky", action="store_true")
        p.add_argument("--z-weight", type=float, default=1.0)
        p.add_argument("--max-iter", type=int, default=20)
        p.add_argument("--depth", type=float, default=None)
        p.add_argument("--dx", type=float, default=0.33)
        p.add_argument("--max-cell", type=float, default=None)
        p.add_argument("--doi", default="coverage", choices=["coverage", "oldenburg"])
        p.add_argument("--blank", default=None, choices=["coverage", "doi", "none"])
        p.add_argument("--drop", type=float, default=section.DEFAULT_BLANK["drop"])
        p.add_argument("--doi-max", type=float, default=section.DEFAULT_BLANK["doi_max"])
        p.add_argument("--palette", default=None, choices=list(section.PALETTES))
        p.add_argument("--timeout", type=float, default=240.0,
                       help="minutes before a worker that does not finish is stopped (0: no limit)")
        p = sub.add_parser("export", help="tif, csv, xyz, png, svg, pdf, gpkg, json, ohm by the extension; a folder: all")
        p.add_argument("profile")
        p.add_argument("out")
        p.add_argument("--pseudosection", action="store_true", help="the figure of the pseudosection")
        p = sub.add_parser("scene", help="the PLY of the 3D view (sections, ground cut away, units, cores)")
        p.add_argument("--terrain", default="auto", choices=["auto", "model", "dem", "none"])
        p.add_argument("--side", default="left", choices=["left", "right", "all"])
        p = sub.add_parser("remove")
        p.add_argument("profile")
        p = sub.add_parser("install", help="install pyGIMLi in an environment of the plugin (uv)")
        p.add_argument("--force", action="store_true")
        sub.add_parser("status")
        rest, project, label = self._project_args(args)
        ns = parser.parse_args(rest)
        if ns.what == "status":
            print(environment.status_text())
            if environment.where()[0] is not None:
                try:
                    environment.check(again=True)
                except environment.GimliMissing as exc:
                    print(exc)
                    return 1
                print(environment.status_text())
                return 0 if environment.known_usable() else 1
            return 1
        if ns.what == "install":
            kind, _v = environment.where()
            if kind is not None and not ns.force:
                print(environment.status_text() + " " + t("Niente da installare (--force per installarlo di nuovo).",
                                                          "Nothing to install (--force to install it again)."))
                return 0
            try:
                print(environment.install(_print_progress))
            except (environment.GimliMissing, RuntimeError) as exc:
                print(exc)
                return 1
            return 0
        chunk = self._open_for_command(project, label)
        if chunk is None:
            print(f"no chunk {label!r} in the project" if label else
                  "no chunk: give the project, ppg plugin geophysics ert " + ns.what + " PROJECT.ppg ...")
            return 1
        try:
            code = self._ert_run_command(chunk, ns)
        except (ValueError, KeyError, FileNotFoundError, environment.GimliMissing, RuntimeError) as exc:
            print(exc)
            return 1
        doc = self.context.app.document if self.context else None
        if doc is not None and doc.path:
            doc.save()
        return code

    def _ert_run_command(self, chunk, ns) -> int:
        if ns.what == "import":
            positions = True if ns.positions else (False if ns.numbers else None)
            e = ert_ops.import_profile(chunk, ns.source, ns.name, ns.spacing, positions, ns.electrodes, _print_progress)
            print(json.dumps({k: e.get(k) for k in ("id", "name", "electrodes", "readings", "array", "spacing", "notes")},
                             ensure_ascii=False))
        elif ns.what == "simulate":
            model = json.loads(ns.model) if ns.model else None
            e = ert_ops.simulate_profile(chunk, ns.name, ns.electrodes, ns.spacing, ns.scheme, model, ns.slope, ns.noise,
                                         progress=_print_progress)
            print(json.dumps({k: e.get(k) for k in ("id", "name", "electrodes", "readings", "array")}, ensure_ascii=False))
        elif ns.what == "list":
            print(ert_ops.summary(chunk))
        elif ns.what == "place":
            line = None
            if ns.markers:
                pos = self._marker_positions(chunk)
                a, _s, b = ns.markers.partition(",")
                if a not in pos or b not in pos:
                    raise ValueError(f"markers {ns.markers!r}: give two markers of the chunk with a position")
                line = [pos[a], pos[b]]
            elif ns.line:
                line = parse_points(ns.line)
            e = ert_mod.place(chunk, ns.profile, line, ns.first, ns.spacing, ns.heights, ns.z0,
                              "markers" if ns.markers else ("points" if ns.line else "file"), ns.along)
            print(json.dumps(e["placement"], ensure_ascii=False))
        elif ns.what == "filter":
            e = ert_mod.entry_of(chunk, ns.profile)
            rules = {"min_rhoa": ns.min, "max_rhoa": ns.max, "max_error": ns.max_error, "reciprocal": ns.reciprocal,
                     "spikes": ns.spikes}
            given = any(v is not None for v in rules.values())
            e = ert_mod.set_filters(chunk, e["id"], rules if given else None,
                                    [int(v) - 1 for v in ns.remove.split(",") if v.strip()],
                                    [int(v) - 1 for v in ns.restore.split(",") if v.strip()], ns.clear)
            print(json.dumps({"valid": e["valid"], "readings": e["readings"], **(e["filters"])}, ensure_ascii=False))
        elif ns.what == "invert":
            params = {"lam": ns.lam, "rel_error": ns.error / 100.0, "abs_error_u": ns.abs_u, "robust": ns.robust,
                      "blocky": ns.blocky, "z_weight": ns.z_weight, "max_iter": ns.max_iter, "para_depth": ns.depth,
                      "para_dx": ns.dx, "para_max_cell": ns.max_cell, "doi": ns.doi}
            method = ns.blank or ("doi" if ns.doi == "oldenburg" else "coverage")
            e = ert_ops.invert_profile(chunk, ns.profile, params, {"method": method, "drop": ns.drop,
                                                                   "doi_max": ns.doi_max}, ns.palette, _print_progress,
                                       timeout=60.0 * ns.timeout if ns.timeout and ns.timeout > 0 else None)
            print(json.dumps({"fit": e["inversion"]["fit"], "display": e["inversion"]["display"]}, ensure_ascii=False))
        elif ns.what == "export":
            out = Path(ns.out)
            if out.suffix:
                print(ert_ops.export(chunk, ns.profile, out, ns.pseudosection))
            else:
                for p in ert_ops.export_all(chunk, ns.profile, out):
                    print(p)
        elif ns.what == "scene":
            path, what = section.scene(chunk, terrain=ns.terrain, side=ns.side)
            print(json.dumps({"file": str(path) if path else None, **what}, ensure_ascii=False))
            return 0 if path else 1
        elif ns.what == "remove":
            ert_mod.remove(chunk, ns.profile, keep=False)  # no Undo on the command line
        return 0


def parse_points(text: str) -> list[list[float]]:
    """``'X Y; X Y'`` or one point per line: the two numbers separated by spaces (then a comma inside a
    number is a decimal comma: ``500010,5 4500050,2``) or by a comma (``500010.5,4500050.2``). A point
    that cannot be read one way only is refused, never guessed."""
    pts = []
    bad = t("Punto non leggibile: «{0}» (scrivi 'X Y' con il punto o la virgola decimale, oppure 'X,Y' con il punto).",
            "Point not readable: '{0}' (write 'X Y' with a decimal point or comma, or 'X,Y' with a decimal point).")
    for part in str(text).replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        words = [w.strip().rstrip(",") for w in part.split()]
        words = [w for w in words if w]
        if len(words) == 1:  # "X,Y"
            words = words[0].split(",")
            if len(words) != 2:
                raise ValueError(bad.format(part))
        elif len(words) == 2:
            words = [w.replace(",", ".") for w in words]
        else:
            raise ValueError(bad.format(part))
        try:
            pts.append([float(words[0]), float(words[1])])
        except ValueError:
            raise ValueError(bad.format(part)) from None
    if len(pts) < 2:
        raise ValueError(t("Servono almeno due punti (X Y).", "At least two points (X Y) are needed."))
    return pts


def _print_progress(fraction, message=""):
    print(f"{100 * float(fraction):5.1f} % {message}", flush=True)
