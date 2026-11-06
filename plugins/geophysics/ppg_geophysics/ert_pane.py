"""The ERT tab of the Geofisica pane: the profiles of the chunk and the steps of a tomography, one under
the other with their state - import, the line in the survey, the pseudosection and its filters, the
inversion, the section shown (2D, 3D, the line on the plan), the exports - and where pyGIMLi runs."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import t


def legend_pixmap(lo: float, hi: float, palette: str, width: int = 300, title: str = "Ω·m") -> QPixmap:
    """A horizontal log colour bar from ``lo`` to ``hi`` ohm m with its ticks and ``title``."""
    import numpy as np

    from .ert_section import number, ramp, ticks

    bar_h, pad = 14, 6
    img = QImage(width, bar_h + 34, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 255, 255, 235))
    p = QPainter(img)
    x0, x1 = pad, width - pad
    rgb = ramp(np.linspace(0, 1, x1 - x0), palette)
    for i, c in enumerate(rgb):
        p.setPen(QColor(int(c[0]), int(c[1]), int(c[2])))
        p.drawLine(x0 + i, pad, x0 + i, pad + bar_h)
    p.setPen(QColor(40, 40, 40))
    p.drawRect(x0, pad, x1 - x0 - 1, bar_h)
    f = QFont()
    f.setPixelSize(10)
    p.setFont(f)
    span = np.log10(hi) - np.log10(lo)
    for v in ticks(lo, hi):
        x = x0 + (x1 - x0 - 1) * (np.log10(v) - np.log10(lo)) / max(span, 1e-9)
        p.drawLine(int(x), pad + bar_h, int(x), pad + bar_h + 3)
        text = number(v)
        p.drawText(int(x) - 20, pad + bar_h + 4, 40, 12, Qt.AlignmentFlag.AlignHCenter, text)
    f.setBold(True)
    p.setFont(f)
    p.drawText(x0, pad + bar_h + 17, x1 - x0, 12, Qt.AlignmentFlag.AlignHCenter,
               t("resistività {0} (scala logaritmica)", "resistivity {0} (log scale)").format(title))
    p.end()
    return QPixmap.fromImage(img)


class _Step(QFrame):
    """One step: its title, its buttons (onto more lines in a narrow dock), its state under them."""

    def __init__(self, number: int, title: str, buttons, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        head = QLabel(f"<b>{number}. {title}</b>")
        lay.addWidget(head)
        holder = QWidget()
        bar = FlowLayout(holder)
        for b in buttons:
            bar.addWidget(b)
        lay.addWidget(holder)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        self.state.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.state)
        self.buttons = list(buttons)

    def set_state(self, text: str, done: bool = False, enabled: bool = True) -> None:
        mark = "✓ " if done else ""
        self.state.setText(mark + text)
        for b in self.buttons:
            b.setEnabled(enabled)


class ErtPane(QWidget):
    import_requested = Signal()
    simulate_requested = Signal()
    draw_line_requested = Signal(str)
    place_requested = Signal(str)
    pseudo_requested = Signal(str)
    filter_requested = Signal(str)
    invert_requested = Signal(str)
    section_requested = Signal(str)
    view3d_requested = Signal(str)
    plan_requested = Signal(str)
    palette_changed = Signal(str, str)
    export_requested = Signal(str)
    remove_requested = Signal(str)
    install_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from .ert_section import PALETTE_NAMES

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Tomografia elettrica di resistività: le letture del georesistivimetro invertite con pyGIMLi, "
                         "la sezione nel rilievo, in 2D e nella vista 3D con il modello, le unità e i carotaggi.",
                         "Electrical resistivity tomography: the readings of the resistivity meter inverted with "
                         "pyGIMLi, the section in the survey, in 2D and in the 3D view with the model, the units and "
                         "the cores."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel(t("Profilo", "Profile")))
        self.c_profile = QComboBox()
        self.c_profile.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_profile.setMinimumContentsLength(10)
        row.addWidget(self.c_profile, 1)
        lay.addLayout(row)

        def button(text, tip=""):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            return b

        self.b_import = button(t("Importa letture...", "Import readings..."),
                               t("CSV/TXT (a, b, m, n, rhoa), formato unificato di pyGIMLi (.ohm, .dat); RES2DINV, ABEM, "
                                 "Syscal con pyGIMLi", "CSV/TXT (a, b, m, n, rhoa), pyGIMLi's unified format (.ohm, .dat); "
                                 "RES2DINV, ABEM, Syscal through pyGIMLi"))
        self.b_simulate = button(t("Profilo sintetico...", "Synthetic profile..."),
                                 t("Letture calcolate da pyGIMLi su un modello: per provare il flusso",
                                   "Readings computed by pyGIMLi on a model: to try the workflow"))
        self.b_draw = button(t("Disegna la linea", "Draw the line"),
                             t("Clic sul primo e sull'ultimo punto (o su più punti) della linea sull'ortofoto o sul DEM; "
                               "doppio clic o Invio per finire", "Click the first and the last point (or more points) "
                               "of the line on the orthomosaic or DEM; double click or Enter to end"))
        self.b_place = button(t("Linea da marker o coordinate...", "Line from markers or coordinates..."))
        self.b_pseudo = button(t("Mostra la pseudosezione", "Show the pseudosection"),
                               t("Le letture nella vista 2D: per toglierne una, selezionala con lo strumento Seleziona e "
                                 "premi Canc", "The readings in the 2D view: to remove one, select it with the Select "
                                 "tool and press Delete"))
        self.b_filter = button(t("Filtri...", "Filters..."))
        self.b_invert = button(t("Inverti...", "Invert..."))
        self.b_section = button(t("Sezione 2D", "2D section"))
        self.b_3d = button(t("Nella vista 3D", "In the 3D view"))
        self.b_plan = button(t("Linea sulla pianta", "Line on the plan"))
        self.c_palette = QComboBox()
        for name, (it, en) in PALETTE_NAMES.items():
            self.c_palette.addItem(t(it, en), name)
        self.b_export = button(t("Esporta...", "Export..."), t("Sezione GeoTIFF, celle CSV/XYZ, figure PNG/SVG/PDF, "
                                                               "linea GeoPackage, paradati JSON, letture .ohm",
                                                               "Section GeoTIFF, cells CSV/XYZ, figures PNG/SVG/PDF, "
                                                               "line GeoPackage, paradata JSON, readings .ohm"))
        self.b_remove = button(t("Rimuovi profilo", "Remove profile"))
        self.steps = [
            _Step(1, t("Letture", "Readings"), [self.b_import, self.b_simulate]),
            _Step(2, t("Linea nel rilievo", "Line in the survey"), [self.b_draw, self.b_place]),
            _Step(3, t("Pseudosezione e filtri", "Pseudosection and filters"), [self.b_pseudo, self.b_filter]),
            _Step(4, t("Inversione (pyGIMLi)", "Inversion (pyGIMLi)"), [self.b_invert]),
            _Step(5, t("Sezione", "Section"), [self.b_section, self.b_3d, self.b_plan, self.c_palette]),
            _Step(6, t("Esportazioni", "Exports"), [self.b_export, self.b_remove]),
        ]
        for s in self.steps:
            lay.addWidget(s)
        self.legend = QLabel()
        self.legend.setVisible(False)
        self.steps[4].layout().addWidget(self.legend)
        env = QFrame()
        env.setFrameShape(QFrame.Shape.StyledPanel)
        el = QVBoxLayout(env)
        el.setContentsMargins(6, 4, 6, 4)
        self.env_status = QLabel("")
        self.env_status.setWordWrap(True)
        el.addWidget(self.env_status)
        holder = QWidget()
        fl = FlowLayout(holder)
        self.b_install = button(t("Installa pyGIMLi...", "Install pyGIMLi..."))
        fl.addWidget(self.b_install)
        el.addWidget(holder)
        lay.addWidget(env)
        lay.addStretch(1)
        self.entries: list[dict] = []
        self.b_import.clicked.connect(self.import_requested)
        self.b_simulate.clicked.connect(self.simulate_requested)
        self.b_install.clicked.connect(self.install_requested)
        for b, sig in ((self.b_draw, self.draw_line_requested), (self.b_place, self.place_requested),
                       (self.b_pseudo, self.pseudo_requested), (self.b_filter, self.filter_requested),
                       (self.b_invert, self.invert_requested), (self.b_section, self.section_requested),
                       (self.b_3d, self.view3d_requested), (self.b_plan, self.plan_requested),
                       (self.b_export, self.export_requested), (self.b_remove, self.remove_requested)):
            b.clicked.connect(lambda _c=False, s=sig: self._emit(s))
        self.c_palette.activated.connect(self._palette)
        self.c_profile.currentIndexChanged.connect(lambda _i: self._update())
        self.set_profiles([])

    # --------------------------------------------------------------- selection
    def selected_id(self) -> str | None:
        pid = self.c_profile.currentData()
        return str(pid) if pid else None

    def selected_entry(self) -> dict | None:
        pid = self.selected_id()
        return next((e for e in self.entries if e["id"] == pid), None)

    def select(self, pid: str) -> None:
        k = self.c_profile.findData(pid)
        if k >= 0:
            self.c_profile.setCurrentIndex(k)

    def _emit(self, signal) -> None:
        pid = self.selected_id()
        if pid is not None:
            signal.emit(pid)

    def _palette(self, _i) -> None:
        pid = self.selected_id()
        if pid is not None:
            self.palette_changed.emit(pid, str(self.c_palette.currentData()))

    # --------------------------------------------------------------- state
    def set_environment(self, text: str, can_install: bool, available: bool) -> None:
        self.env_status.setText(text)
        self.b_install.setVisible(available)
        self.b_install.setEnabled(available)
        self.b_install.setText(t("Installa pyGIMLi...", "Install pyGIMLi...") if can_install
                               else t("Installa di nuovo pyGIMLi...", "Install pyGIMLi again..."))
        self._gimli = available

    def set_profiles(self, entries: list[dict]) -> None:
        keep = self.selected_id()
        self.entries = list(entries)
        self.c_profile.blockSignals(True)
        self.c_profile.clear()
        for e in entries:
            self.c_profile.addItem(e["name"], e["id"])
        k = self.c_profile.findData(keep) if keep else -1
        self.c_profile.setCurrentIndex(k if k >= 0 else (0 if entries else -1))
        self.c_profile.blockSignals(False)
        self._update()

    def _update(self) -> None:
        from . import ert as ert_mod
        from .ert_ops import summary_line
        from .ert_section import PALETTES, number

        e = self.selected_entry()
        s1, s2, s3, s4, s5, s6 = self.steps
        self.b_import.setEnabled(True)
        self.b_simulate.setEnabled(getattr(self, "_gimli", True))
        if e is None:
            s1.set_state(t("Nessun profilo: importa le letture di un georesistivimetro.",
                           "No profile: import the readings of a resistivity meter."))
            self.b_import.setEnabled(True)
            self.b_simulate.setEnabled(getattr(self, "_gimli", True))
            for s in (s2, s3, s4, s5, s6):
                s.set_state("", enabled=False)
            self.legend.setVisible(False)
            return
        s1.set_state(summary_line(e).split(";")[0] + (t("; spaziatura {0:g} m", "; spacing {0:g} m")
                                                      .format(float(e.get("spacing", 0)))), done=True)
        pl = e.get("placement")
        if pl:
            heights = {"dem": t("quote dal DEM", "heights from the DEM"), "file": t("quote del file", "heights of the file"),
                       "flat": t("quota fissa", "fixed height")}.get(pl.get("heights"), pl.get("heights", ""))
            text = t("Linea di {0:.1f} m, {1} ({2} - {3} m)", "Line of {0:.1f} m, {1} ({2} - {3} m)").format(
                float(pl["length"]), heights, number(pl["z_range"][0]), number(pl["z_range"][1]))
            if pl.get("notes"):
                text += "; " + "; ".join(pl["notes"])
            s2.set_state(text, done=True)
        else:
            s2.set_state(t("Da posizionare: disegna la linea sulla vista o dalle coordinate (l'inversione può farsi "
                           "anche prima, con le quote del file o piatte).",
                           "To place: draw the line on the view or from coordinates (the inversion can also be done "
                           "before, with the heights of the file or flat)."))
        ftext = ert_mod.filter_text(e)
        s3.set_state(t("{0} letture valide su {1}", "{0} valid readings of {1}").format(e.get("valid", e["readings"]),
                                                                                       e["readings"])
                     + (f" ({ftext})" if ftext else ""), done=bool(ftext))
        inv = e.get("inversion") or {}
        fit = inv.get("fit") or {}
        s4.set_state(t("Non ancora invertito.", "Not inverted yet.") if not fit else
                     t("χ² {0:.2f}, RMS {1:.1f} %, {2} iterazioni, {3} celle, λ {4:g}; pyGIMLi {5}{6}",
                       "χ² {0:.2f}, RMS {1:.1f} %, {2} iterations, {3} cells, λ {4:g}; pyGIMLi {5}{6}").format(
                         fit.get("chi2", float("nan")), fit.get("rrms", float("nan")), fit.get("iterations", 0),
                         fit.get("cells", 0), fit.get("lam", 0), (inv.get("versions") or {}).get("pygimli", "?"),
                         t(" - da rifare: letture, filtri o linea cambiati", " - to do again: readings, filters or line "
                           "changed") if inv.get("stale") else ""), done=bool(fit) and not inv.get("stale"),
                     enabled=getattr(self, "_gimli", True))
        disp = inv.get("display") or {}
        has_section = bool(inv.get("files"))
        if has_section and disp.get("limits"):
            lo, hi = disp["limits"]
            pal = disp.get("palette", "rainbow")
            k = self.c_palette.findData(pal)
            if k >= 0:
                self.c_palette.setCurrentIndex(k)
            blank = disp.get("blank") or {}
            how = {"coverage": t("copertura", "coverage"), "doi": t("indice DOI", "DOI index"),
                   "none": t("nessuna", "none")}.get(blank.get("method"), "")
            s5.set_state(t("Colori {0} - {1} Ω·m (logaritmica); oscuramento: {2}; {3:.0f} m² di sezione",
                           "Colours {0} - {1} Ω·m (log); blanking: {2}; {3:.0f} m² of section").format(
                number(lo), number(hi), how, float(disp.get("shown_m2", 0))), done=True)
            self.legend.setPixmap(legend_pixmap(lo, hi, pal if pal in PALETTES else "rainbow", 220))
            self.legend.setVisible(True)
        else:
            s5.set_state(t("Inverti prima il profilo.", "Invert the profile first."), enabled=False)
            self.b_plan.setEnabled(bool(pl))
            self.legend.setVisible(False)
        self.b_3d.setEnabled(has_section and bool(pl))
        self.b_plan.setEnabled(bool(pl))
        s6.set_state(t("Pseudosezione, letture e paradati sempre; sezione e celle dopo l'inversione; linea dopo il "
                       "posizionamento.", "Pseudosection, readings and paradata always; section and cells after the "
                       "inversion; line after the placement."))
