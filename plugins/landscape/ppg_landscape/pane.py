"""The *Paesaggio nel tempo* pane: the scenario (the DEM it starts from, a preset, the years, the
snapshots, the simulation cell, the edge, the parameters of soil creep and running water) with the
estimate of how long it takes shown while the values change, the runs of the chunk in a table, the
layers of a run over the raster view, and what can be done with a run (a version of the DEM, the
comparison with another chunk, the export, the removal). Where Landlab runs, in a line, with the
button to install it when it is missing."""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from . import engine
from .texts import (BOUNDARY_TITLES, FLUVIAL_KEYS, PARAMETERS, SCENARIO_NOTE, layer_title, number, preset_title, t,
                    years)

COLS = (("Nome", "Name"), ("Anni", "Years"), ("Cella (m)", "Cell (m)"), ("Eroso (m³)", "Eroded (m³)"),
        ("Depositato (m³)", "Deposited (m³)"), ("Fatto", "Made"))
PRESETS = ("mudbrick_tell", "earth_slope", "exposed_excavation", "custom")
ADVANCED = ("F_f", "phi", "H_star", "v_s", "m_sp", "n_sp")  # shown with "Parametri avanzati"


def parse_years(text: str) -> list[float]:
    """The snapshot times written in the field: separated by spaces, ``;`` or a comma followed by a
    space (``10, 100``); a comma inside a number is the decimal one (``2,5``)."""
    out = []
    for token in re.split(r"[;\s]+|,(?=\s)", text.strip()):
        token = token.strip().rstrip(",")
        if not token:
            continue
        try:
            value = float(token.replace(",", "."))
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return sorted(set(out))


class LandscapePane(QWidget):
    run_requested = Signal(object)  # the parameters (dict)
    install_requested = Signal()
    source_changed = Signal(str)
    show_requested = Signal(str, str, float)  # run id, layer, opacity
    hide_requested = Signal()
    opacity_changed = Signal(float)
    version_requested = Signal(str)
    compare_requested = Signal(str)
    export_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.estimator = None  # params -> engine.estimate()'s dict (set by the plugin)
        self.available = True  # Landlab can run on this computer
        self.busy = False
        self.has_dem = False
        self._filling = False  # the fields are being set from a preset: they do not make it "custom"
        self.last_estimate: dict | None = None
        self.entries: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Come cambia nel tempo la superficie rilevata: il suolo che scivola sui pendii (creep) e "
                         "l'acqua che scorre, erode e deposita, simulati con Landlab sul DEM. ",
                         "How the surveyed surface changes over time: soil creeping down the slopes and running "
                         "water eroding and depositing, simulated with Landlab on the DEM. ") + t(*SCENARIO_NOTE))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        env = QHBoxLayout()
        self.l_env = QLabel("")
        self.l_env.setWordWrap(True)
        env.addWidget(self.l_env, 1)
        self.b_install = QPushButton(t("Installa Landlab...", "Install Landlab..."))
        self.b_install.setToolTip(t("Python 3.12 e Landlab in un ambiente del plugin (circa 500 MB)",
                                    "Python 3.12 and Landlab in an environment of the plugin (about 500 MB)"))
        env.addWidget(self.b_install)
        lay.addLayout(env)

        # ------------------------------------------------------------- the scenario
        box = QGroupBox(t("Scenario", "Scenario"))
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.c_source = QComboBox()
        self.c_source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_source.setMinimumContentsLength(12)
        self.c_source.setToolTip(t("Il DEM da cui parte la simulazione: quello attivo o una versione conservata (una "
                                   "fase precedente).", "The DEM the simulation starts from: the active one or a kept "
                                   "version (an earlier phase)."))
        form.addRow(t("DEM di partenza", "Starting DEM"), self.c_source)
        self.c_preset = QComboBox()
        self.c_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_preset.setMinimumContentsLength(12)
        for key in PRESETS:
            self.c_preset.addItem(preset_title(key), key)
        self.c_preset.setToolTip(t("Valori di partenza difendibili per i casi più comuni; cambiando un valore lo "
                                   "scenario diventa Personalizzato.", "Defensible starting values for the common "
                                   "cases; changing a value makes the scenario Custom."))
        form.addRow(t("Scenario", "Scenario"), self.c_preset)
        self.s_years = self._spin(1.0, 1e6, 1, t("Gli anni simulati.", "The years simulated."))
        form.addRow(t("Anni", "Years"), self.s_years)
        self.e_snaps = QLineEdit()
        self.e_snaps.setPlaceholderText("10; 100; 500")
        self.e_snaps.setToolTip(t("Le superfici intermedie da conservare (anni, separati da ; o spazi).",
                                  "The intermediate surfaces to keep (years, separated by ; or spaces)."))
        form.addRow(t("Istantanee (anni)", "Snapshots (years)"), self.e_snaps)
        self.s_cell = self._spin(0.0, 1e4, 3, t("La cella della griglia di simulazione (m). Automatica: la "
                                                "risoluzione del DEM ingrandita fino a 500 celle sul lato lungo. Una "
                                                "cella più grande accorcia molto la simulazione.",
                                                "The cell of the simulation grid (m). Automatic: the DEM's resolution "
                                                "coarsened to 500 cells on the longer side. A larger cell shortens "
                                                "the simulation a lot."))
        self.s_cell.setSpecialValueText(t("automatica", "automatic"))
        form.addRow(t("Cella (m)", "Cell (m)"), self.s_cell)
        self.c_boundary = QComboBox()
        self.c_boundary.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_boundary.setMinimumContentsLength(12)
        for key in engine.BOUNDARIES:
            self.c_boundary.addItem(t(*BOUNDARY_TITLES[key]), key)
        self.c_boundary.setToolTip(t("Aperto: ai margini del DEM acqua e sedimento escono (un versante che continua). "
                                     "Chiuso: niente esce se non dal punto più basso del bordo (un'area isolata); "
                                     "anche le isole e le parti unite da una strozzatura hanno il loro punto di uscita.",
                                     "Open: water and sediment leave at the edges of the DEM (a slope that goes on). "
                                     "Closed: nothing leaves but at the lowest point of the edge (an enclosed area); "
                                     "islands and parts joined by a narrow neck have an outlet of their own too."))
        form.addRow(t("Bordo", "Edge"), self.c_boundary)
        self.s_dt = self._spin(0.0, 1e5, 3, t("Il passo più lungo (anni). Automatico: gli anni / 100; la simulazione "
                                              "lo accorcia da sé dove serve alla stabilità.",
                                              "The longest step (years). Automatic: the years / 100; the simulation "
                                              "shortens it by itself where stability needs it."))
        self.s_dt.setSpecialValueText(t("automatico", "automatic"))
        form.addRow(t("Passo (anni)", "Step (years)"), self.s_dt)
        # short texts: a check box does not wrap, and a long one would widen the column of the pane
        self.k_diffusion = QCheckBox(t("Creep del suolo", "Soil creep"))
        self.k_diffusion.setToolTip(t("Diffusione del versante: il suolo scivola giù per i pendii, in proporzione alla "
                                      "pendenza (LinearDiffuser).", "Hillslope diffusion: soil creeps down the slopes, "
                                      "in proportion to the slope (LinearDiffuser)."))
        self.k_diffusion.setChecked(True)
        form.addRow(self.k_diffusion)
        self.spins: dict[str, QDoubleSpinBox] = {}
        self._labels: dict[str, QLabel] = {}
        self._param_row(form, "D")
        self.k_fluvial = QCheckBox(t("Acqua che scorre (SPACE)", "Running water (SPACE)"))
        self.k_fluvial.setToolTip(t("Erosione e deposito dell'acqua che scorre in superficie (SPACE).",
                                    "Erosion and deposition by the water running over the surface (SPACE)."))
        self.k_fluvial.setChecked(True)
        form.addRow(self.k_fluvial)
        for key in FLUVIAL_KEYS:
            self._param_row(form, key)
        self.k_advanced = QCheckBox(t("Parametri avanzati", "Advanced parameters"))
        self.k_advanced.setToolTip(t("I parametri avanzati dell'acqua che scorre: frazione fine, porosità, H*, "
                                     "velocità di deposito, esponenti.", "The advanced parameters of the running "
                                     "water: fine fraction, porosity, H*, settling velocity, exponents."))
        form.addRow(self.k_advanced)
        lay.addWidget(box)
        self.l_estimate = QLabel("")
        self.l_estimate.setWordWrap(True)
        lay.addWidget(self.l_estimate)
        self.b_run = QPushButton(t("Avvia la simulazione", "Run the simulation"))
        lay.addWidget(self.b_run)

        # ------------------------------------------------------------- the runs
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(110)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(t("Mostra", "Show")))
        self.c_layer = QComboBox()
        self.c_layer.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_layer.setMinimumContentsLength(12)
        row.addWidget(self.c_layer, 1)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(t("Opacità", "Opacity")))
        self.s_opacity = QSlider(Qt.Orientation.Horizontal)
        self.s_opacity.setRange(10, 100)
        self.s_opacity.setValue(70)
        row2.addWidget(self.s_opacity, 1)
        lay.addLayout(row2)
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_show = QPushButton(t("Sovrapponi alla vista", "Overlay on the view"))
        self.b_hide = QPushButton(t("Nascondi", "Hide"))
        self.b_version = QPushButton(t("Aggiungi come versione del DEM", "Add as a version of the DEM"))
        self.b_version.setToolTip(t("La superficie finale fra le versioni del DEM: curve di livello, sezioni, volumi "
                                    "e rilievo la usano quando la rendi attiva. Il DEM attivo non cambia.",
                                    "The final surface among the versions of the DEM: contours, sections, volumes "
                                    "and relief use it when you make it active. The active DEM does not change."))
        self.b_compare = QPushButton(t("Confronta con un altro chunk...", "Compare with another chunk..."))
        self.b_compare.setToolTip(t("La superficie finale meno il DEM di un altro chunk (una fase successiva): "
                                    "mediana, NMAD e 95° percentile della differenza, per tarare i parametri.",
                                    "The final surface minus the DEM of another chunk (a later phase): median, NMAD "
                                    "and 95th percentile of the difference, to calibrate the parameters."))
        self.b_export = QPushButton(t("Esporta...", "Export..."))
        self.b_remove = QPushButton(t("Rimuovi", "Remove"))
        for b in (self.b_show, self.b_hide, self.b_version, self.b_compare, self.b_export, self.b_remove):
            bar.addWidget(b)
        lay.addWidget(holder)
        self.l_limits = QLabel("")
        self.l_limits.setWordWrap(True)
        lay.addWidget(self.l_limits)
        self.l_summary = QLabel("")
        self.l_summary.setWordWrap(True)
        self.l_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.l_summary)

        # ------------------------------------------------------------- connections
        self.b_install.clicked.connect(self.install_requested)
        self.c_source.currentIndexChanged.connect(lambda _i: self.source_changed.emit(self.source()))
        self.c_preset.currentIndexChanged.connect(lambda _i: self.apply_preset(self.preset()))
        for w in (self.s_years, self.s_cell, self.s_dt, *self.spins.values()):
            w.valueChanged.connect(self._edited)
        self.e_snaps.textChanged.connect(lambda _t: self.update_estimate())
        self.c_boundary.currentIndexChanged.connect(lambda _i: self.update_estimate())
        for k in (self.k_diffusion, self.k_fluvial):
            k.toggled.connect(self._processes_toggled)
        self.k_advanced.toggled.connect(lambda _on: self._show_rows())
        self.b_run.clicked.connect(lambda: self.run_requested.emit(self.params()))
        self.b_show.clicked.connect(self._show)
        self.b_hide.clicked.connect(self.hide_requested)
        self.s_opacity.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        self.b_version.clicked.connect(lambda: self._emit(self.version_requested))
        self.b_compare.clicked.connect(lambda: self._emit(self.compare_requested))
        self.b_export.clicked.connect(lambda: self._emit(self.export_requested))
        self.b_remove.clicked.connect(lambda: self._emit(self.remove_requested))
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _it: self._show())
        self.c_layer.activated.connect(lambda _i: self._show())
        self.apply_preset(PRESETS[0])
        self._show_rows()
        self.set_runs([])

    # ------------------------------------------------------------------ the form
    def _spin(self, lo: float, hi: float, decimals: int, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        return s

    def _param_row(self, form: QFormLayout, key: str) -> None:
        it, en, tip_it, tip_en, lo, hi, decimals = PARAMETERS[key]
        spin = self._spin(lo, hi, decimals, t(tip_it, tip_en))
        spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        label = QLabel(t(it, en))
        label.setWordWrap(True)
        label.setToolTip(t(tip_it, tip_en))
        form.addRow(label, spin)
        self.spins[key] = spin
        self._labels[key] = label

    def _show_rows(self) -> None:
        diffusion, fluvial, advanced = self.k_diffusion.isChecked(), self.k_fluvial.isChecked(), self.k_advanced.isChecked()
        for key, spin in self.spins.items():
            on = diffusion if key == "D" else fluvial and (advanced or key not in ADVANCED)
            spin.setVisible(on)
            self._labels[key].setVisible(on)
        self.k_advanced.setVisible(fluvial)

    def _processes_toggled(self, _on: bool) -> None:
        self._show_rows()
        self._edited()

    def _edited(self, *_args) -> None:
        """A value changed by the user: the scenario is a custom one now (not while a preset fills the form)."""
        if not self._filling and self.preset() != "custom":
            self._filling = True
            try:
                self.c_preset.setCurrentIndex(self.c_preset.findData("custom"))
            finally:
                self._filling = False
        self.update_estimate()

    def preset(self) -> str:
        return str(self.c_preset.currentData() or "custom")

    def source(self) -> str:
        return str(self.c_source.currentData() or "active")

    def apply_preset(self, key: str) -> None:
        """The values of a preset in the form (``custom`` keeps the values there)."""
        if self._filling:
            return
        self._filling = True
        try:
            if self.preset() != key:
                self.c_preset.setCurrentIndex(max(0, self.c_preset.findData(key)))
            preset = engine.PRESETS.get(key)
            if preset is not None:
                self.s_years.setValue(float(preset.get("years", 100)))
                diffusion = {**engine.DEFAULTS["diffusion"], **(preset.get("diffusion") or {})}
                fluvial = {**engine.DEFAULTS["fluvial"], **(preset.get("fluvial") or {})}
                self.spins["D"].setValue(diffusion["D"])
                for k in FLUVIAL_KEYS:
                    self.spins[k].setValue(fluvial[k])
                self.k_diffusion.setChecked(preset.get("diffusion") is not None)
                self.k_fluvial.setChecked(preset.get("fluvial") is not None)
                years_ = float(preset.get("years", 100))
                self.e_snaps.setText("; ".join(f"{y:g}" for y in (years_ / 100, years_ / 10) if y >= 1 and y == int(y)))
            elif self.spins["D"].value() == 0 and self.spins["K_sed"].value() == 0:  # never filled yet
                self.spins["D"].setValue(engine.DEFAULTS["diffusion"]["D"])
                for k in FLUVIAL_KEYS:
                    self.spins[k].setValue(engine.DEFAULTS["fluvial"][k])
        finally:
            self._filling = False
        self._show_rows()
        self.update_estimate()

    def params(self) -> dict:
        """The scenario of the form, as :func:`ppg_landscape.landscape.run` takes it."""
        out = {"preset": self.preset(), "years": self.s_years.value(), "snapshots": parse_years(self.e_snaps.text()),
               "cell": self.s_cell.value() or None, "boundary": str(self.c_boundary.currentData()),
               "dt": self.s_dt.value() or None, "source": self.source(),
               "diffusion": {"D": self.spins["D"].value()} if self.k_diffusion.isChecked() else None,
               "fluvial": {k: self.spins[k].value() for k in FLUVIAL_KEYS} if self.k_fluvial.isChecked() else None}
        return out

    def update_estimate(self) -> None:
        """How long the scenario takes (the plugin's ``estimator``), its warning; Run refused above two hours."""
        if self._filling:
            return
        self.last_estimate = None
        text, refuse = "", False
        if not self.has_dem:
            text = t("Il chunk non ha un DEM: costruiscilo prima.", "The chunk has no DEM: build it first.")
        elif not (self.k_diffusion.isChecked() or self.k_fluvial.isChecked()):
            text = t("Scegli almeno un processo.", "Choose at least one process.")
            refuse = True
        elif self.estimator is not None:
            try:
                est = self.estimator(self.params())
            except Exception as exc:  # noqa: BLE001 - a value the engine refuses: said here
                text, refuse = str(exc), True
            else:
                self.last_estimate = est
                minutes = est["seconds"] / 60.0
                when = (t("meno di un minuto", "less than a minute") if minutes < 1 else
                        t("circa {0} minuti", "about {0} minutes").format(number(minutes)) if minutes < 90 else
                        t("circa {0} ore", "about {0} hours").format(number(minutes / 60.0, 1)))
                rows, cols = est["shape"]
                text = t("Griglia {0} × {1} celle di {2} m, {3} passi: {4}.",
                         "Grid of {0} × {1} cells of {2} m, {3} steps: {4}.").format(
                    number(cols), number(rows), number(est["cell"], 3).rstrip("0").rstrip(",.") or "0",
                    number(est["diffusion_substeps"] or est["steps"]), when)
                if est.get("large"):
                    text += " " + t("Griglia grande: serve molta memoria.", "Large grid: it needs much memory.")
                if est.get("warning"):
                    text += " " + est["warning"]
                if est.get("refuse"):
                    refuse = True
                    text += " " + t("Troppo lunga: scegli una cella più grande o meno anni.",
                                    "Too long: choose a larger cell or fewer years.")
        self.l_estimate.setText(text)
        self.l_estimate.setStyleSheet("color: #b35900;" if (refuse or (self.last_estimate or {}).get("warning")) else "")
        self._refused = refuse
        self._update_run()

    def _update_run(self) -> None:
        self.b_run.setEnabled(self.available and self.has_dem and not self.busy and not getattr(self, "_refused", False))

    # ------------------------------------------------------------------ what the plugin sets
    def set_status(self, text: str, can_install: bool, available: bool = True) -> None:
        self.l_env.setText(text)
        self.b_install.setVisible(bool(can_install))
        self.available = bool(available)
        self._update_run()

    def set_busy(self, busy: bool) -> None:
        self.busy = bool(busy)
        self._update_run()

    def set_sources(self, sources: list[tuple[str, str]], info: dict | None) -> None:
        """The DEMs a scenario can start from (``(key, name)``) and the grid of the one chosen."""
        current = self.source()
        self.c_source.blockSignals(True)
        self.c_source.clear()
        for key, name in sources:
            self.c_source.addItem(name if key != "active" else t("{0} (attivo)", "{0} (active)").format(name), key)
        k = self.c_source.findData(current)
        self.c_source.setCurrentIndex(k if k >= 0 else 0)
        self.c_source.blockSignals(False)
        self.set_dem(info)

    def set_dem(self, info: dict | None) -> None:
        self.has_dem = info is not None
        self.update_estimate()

    def set_runs(self, entries: list[dict]) -> None:
        from pyarchinit_photogrammetry.gui.i18n import thousands_points

        current = self.current_id()
        self.entries = list(entries)
        self.table.setRowCount(0)
        for e in self.entries:
            r = self.table.rowCount()
            self.table.insertRow(r)
            b = e.get("balance") or {}
            cells = [e.get("name", e.get("id", "")), years(e.get("years", 0)), number(float(e.get("cell") or 0), 3),
                     thousands_points(f"{b.get('eroded_m3', 0) or 0:,.1f}") if b.get("eroded_m3") is not None else "",
                     thousands_points(f"{b.get('deposited_m3', 0) or 0:,.1f}") if b.get("deposited_m3") is not None else "",
                     str(e.get("created", "")).replace("T", " ")[:16]]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, e.get("id"))
                    it.setToolTip(f"{e.get('id')}: {preset_title(e.get('preset') or 'custom')}, "
                                  f"{(e.get('source') or {}).get('name', '')}")
                self.table.setItem(r, c, it)
            if e.get("id") == current:
                self.table.selectRow(r)
        if self.entries and self.current_id() is None:
            self.table.selectRow(len(self.entries) - 1)
        self._selection_changed()

    def select_run(self, run_id: str) -> None:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == run_id:
                self.table.selectRow(r)
                return

    def current_id(self) -> str | None:
        model = self.table.selectionModel()
        rows = model.selectedRows() if model is not None else []
        if not rows:
            return None
        return str(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))

    def current_entry(self) -> dict | None:
        rid = self.current_id()
        return next((e for e in self.entries if e.get("id") == rid), None)

    def current_layer(self) -> str:
        return str(self.c_layer.currentData() or "dz")

    def _emit(self, signal) -> None:
        rid = self.current_id()
        if rid:
            signal.emit(rid)

    def _show(self) -> None:
        rid = self.current_id()
        if rid:
            self.show_requested.emit(rid, self.current_layer(), self.s_opacity.value() / 100.0)

    def _selection_changed(self) -> None:
        from . import landscape

        e = self.current_entry()
        keep = self.current_layer() if self.c_layer.count() else "dz"
        self.c_layer.clear()
        if e is not None:
            for layer in landscape.layers(e):
                self.c_layer.addItem(layer_title(layer, e), layer)
            k = self.c_layer.findData(keep)
            self.c_layer.setCurrentIndex(k if k >= 0 else max(0, self.c_layer.findData("dz")))
        for b in (self.b_show, self.b_version, self.b_compare, self.b_export, self.b_remove):
            b.setEnabled(e is not None)
        self.l_summary.setText(summary_text(e) if e is not None else "")

    def set_limits(self, text: str) -> None:
        self.l_limits.setText(text)


def summary_text(e: dict) -> str:
    """The mass balance of a run and its comparisons, in a few lines."""
    b = e.get("balance") or {}
    lines = []

    def m3(v) -> str:
        return number(float(v), 1) + " m³"

    if b.get("eroded_m3") is not None:
        lines.append(t("Eroso {0}, depositato {1}, netto {2}.", "Eroded {0}, deposited {1}, net {2}.").format(
            m3(b["eroded_m3"]), m3(b.get("deposited_m3") or 0), m3(b.get("net_m3", (b.get("deposited_m3") or 0) - b["eroded_m3"]))))
    out = sum(float(b.get(k) or 0) for k in ("fluvial_out_m3", "diffusion_out_m3"))
    if out:
        lines.append(t("Uscito dai margini: {0} (solido).", "Left across the edges: {0} (solid).").format(m3(out)))
    if b.get("max_erosion_m") is not None:
        lines.append(t("Erosione massima {0} m, deposito massimo {1} m.", "Largest erosion {0} m, largest deposit {1} m.")
                     .format(number(abs(float(b["max_erosion_m"])), 3), number(float(b.get("max_deposition_m") or 0), 3)))
    if e.get("dem_version"):
        lines.append(t("Fra le versioni del DEM.", "Among the versions of the DEM."))
    for c in e.get("comparisons") or []:
        s, base = c.get("stats") or {}, c.get("baseline") or {}
        if s.get("cells"):
            lines.append(t("Con «{0}»: mediana {1} m, NMAD {2} m, 95° perc. {3} m (prima della simulazione: {4} / {5} / {6}).",
                           "With '{0}': median {1} m, NMAD {2} m, 95th perc. {3} m (before the simulation: {4} / {5} / {6}).")
                         .format(c.get("chunk"), number(s["median"], 3), number(s["nmad"], 3), number(s["p95"], 3),
                                 number(base.get("median", 0), 3), number(base.get("nmad", 0), 3),
                                 number(base.get("p95", 0), 3)))
    return "\n".join(lines)
