"""The Samgeo pane: what to look for (ready entries, or any text), on which raster and where, the
thresholds and the models; the runs of the active chunk (each a layer of proposals on the 2D view)
with what to make of them - units, classes of the dense cloud, a GeoPackage."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import number, presets, t

AREAS = (("all", ("Tutto il raster", "The whole raster")),
         ("polygon", ("Un poligono da disegnare", "A polygon to draw")),
         ("view", ("La parte visibile nella vista 2D", "The part shown in the 2D view")))
MODELS = (("normal", ("Normali (Grounding DINO base, SAM 2.1 small)", "Normal (Grounding DINO base, SAM 2.1 small)")),
          ("fast", ("Veloci (Grounding DINO tiny, SAM 2.1 tiny)", "Fast (Grounding DINO tiny, SAM 2.1 tiny)")),
          ("precise", ("Precisi (Grounding DINO base, SAM 2.1 large)", "Precise (Grounding DINO base, SAM 2.1 large)")))


class SamgeoPane(QWidget):
    find_requested = Signal(dict)
    install_requested = Signal()
    visibility_changed = Signal(str, bool)
    zoom_requested = Signal(str)
    units_requested = Signal(str)
    classes_requested = Signal(str)
    export_requested = Signal(list)
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Trova nell'ortofoto, o in un altro raster del chunk, quello che scrivi: ogni ricerca è un "
                         "layer di proposte nella vista 2D, da controllare e poi trasformare in US, in classi della "
                         "nuvola densa o in un GeoPackage. I testi in inglese funzionano meglio; le voci pronte sono "
                         "tradotte.",
                         "Find in the orthomosaic, or in another raster of the chunk, what you write: every run is a "
                         "layer of proposals on the 2D view, to check and then turn into units, classes of the dense "
                         "cloud or a GeoPackage. English texts work best; the ready entries are translated."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        self.b_install = QPushButton(t("Installa i modelli...", "Install the models..."))
        self.b_install.setToolTip(t("Python 3.12, PyTorch, Grounding DINO e SAM 2.1 in un ambiente dei plugin AI, con uv",
                                    "Python 3.12, PyTorch, Grounding DINO and SAM 2.1 in an environment of the AI plugins, "
                                    "with uv"))
        self.b_install.clicked.connect(self.install_requested)
        lay.addWidget(self.b_install)
        self.form_holder = QWidget()
        form = QFormLayout(self.form_holder)
        form.setContentsMargins(0, 0, 0, 0)
        self.c_text = QComboBox()
        self.c_text.setEditable(True)
        for shown, model in presets():
            self.c_text.addItem(shown, model)
        self.c_text.setToolTip(t("Una voce pronta o un testo tuo, anche più cose separate da virgole (in inglese: "
                                 "«stone, brick»)", "A ready entry or a text of your own, also several things "
                                 "separated by commas ('stone, brick')"))
        form.addRow(t("Cerca", "Look for"), self.c_text)
        self.c_raster = QComboBox()
        form.addRow(t("Raster", "Raster"), self.c_raster)
        self.c_area = QComboBox()
        for key, text in AREAS:
            self.c_area.addItem(t(*text), key)
        form.addRow(t("Dove", "Where"), self.c_area)
        self.s_threshold = QDoubleSpinBox()
        self.s_threshold.setRange(0.05, 0.95)
        self.s_threshold.setSingleStep(0.05)
        self.s_threshold.setDecimals(2)
        self.s_threshold.setValue(0.30)
        self.s_threshold.setToolTip(t("La confidenza minima di Grounding DINO: più alta, meno proposte e più sicure",
                                      "The least confidence of Grounding DINO: higher, fewer and surer proposals"))
        form.addRow(t("Confidenza minima", "Least confidence"), self.s_threshold)
        self.s_min = QDoubleSpinBox()
        self.s_max = QDoubleSpinBox()
        for s in (self.s_min, self.s_max):
            s.setRange(0.0, 1e6)
            s.setDecimals(3)
            s.setSpecialValueText(t("nessun limite", "no limit"))
        form.addRow(t("Area minima (m²)", "Smallest area (m²)"), self.s_min)
        form.addRow(t("Area massima (m²)", "Largest area (m²)"), self.s_max)
        self.s_resolution = QDoubleSpinBox()
        self.s_resolution.setRange(0.0, 1000.0)
        self.s_resolution.setDecimals(2)
        self.s_resolution.setSuffix(" cm/px")
        self.s_resolution.setSpecialValueText(t("automatica", "automatic"))
        self.s_resolution.setToolTip(t("Automatica: quella del raster, non più fine di 1 cm/px (basta per le pietre). "
                                       "Più grossolana, meno tasselli e più veloce.",
                                       "Automatic: the raster's, not finer than 1 cm/px (plenty for stones). Coarser, "
                                       "fewer tiles and faster."))
        form.addRow(t("Risoluzione", "Resolution"), self.s_resolution)
        self.c_models = QComboBox()
        for key, text in MODELS:
            self.c_models.addItem(t(*text), key)
        form.addRow(t("Modelli", "Models"), self.c_models)
        lay.addWidget(self.form_holder)
        self.b_find = QPushButton(t("Trova", "Find"))
        self.b_find.clicked.connect(lambda: self.find_requested.emit(self.values()))
        lay.addWidget(self.b_find)
        lay.addWidget(QLabel(t("Ricerche del chunk attivo", "Runs of the active chunk")))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([t("Ricerca", "Run"), t("Oggetti", "Objects"), t("Raster", "Raster"),
                                              t("Data", "Date")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setToolTip(t("La casella mostra o nasconde il layer della ricerca; doppio clic: lo inquadra",
                                "The box shows or hides the layer of the run; double click: zooms to it"))
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._emit(self.zoom_requested, r))
        lay.addWidget(self.table, 1)
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_units = QPushButton(t("US...", "Units..."))
        self.b_units.setToolTip(t("Le proposte (tutte, o quelle selezionate nella vista) diventano US, con il testo e "
                                  "il punteggio; poi «Proponi US e rapporti» le numera",
                                  "The proposals (all, or those selected on the view) become units, with the text and "
                                  "the score; then 'Proponi US e rapporti' numbers them"))
        self.b_units.clicked.connect(lambda: self._emit(self.units_requested))
        self.b_classes = QPushButton(t("Classi della nuvola", "Classes of the cloud"))
        self.b_classes.setToolTip(t("I punti della nuvola densa dentro le proposte prendono la classe della ricerca",
                                    "The points of the dense cloud inside the proposals get the class of the run"))
        self.b_classes.clicked.connect(lambda: self._emit(self.classes_requested))
        self.b_export = QPushButton(t("GeoPackage...", "GeoPackage..."))
        self.b_export.setToolTip(t("Le ricerche selezionate (tutte, se nessuna) in un GeoPackage, un layer ciascuna",
                                   "The runs selected (all, when none) into a GeoPackage, a layer each"))
        self.b_export.clicked.connect(lambda: self.export_requested.emit(self.selected_runs()))
        self.b_zoom = QPushButton(t("Inquadra", "Zoom to"))
        self.b_zoom.clicked.connect(lambda: self._emit(self.zoom_requested))
        self.b_remove = QPushButton(t("Rimuovi", "Remove"))
        self.b_remove.setToolTip(t("La ricerca, il suo layer e i suoi file; le US già fatte restano",
                                   "The run, its layer and its files; the units already made stay"))
        self.b_remove.clicked.connect(lambda: self._emit(self.remove_requested))
        for b in (self.b_units, self.b_classes, self.b_export, self.b_zoom, self.b_remove):
            bar.addWidget(b)
        lay.addWidget(holder)
        self.l_last = QLabel("")
        self.l_last.setWordWrap(True)
        lay.addWidget(self.l_last)
        self._ids: list[str] = []
        self._filling = False
        self._holders = (self.form_holder, holder)
        self._selection_changed()

    # ------------------------------------------------------------------ what to look for
    def values(self) -> dict:
        """What the user asked: ``text`` (as written or the ready entry shown), ``prompt`` (what the
        models read: the English of a ready entry, else the text), ``raster``, ``area``, ``threshold``,
        ``min_area``, ``max_area``, ``resolution`` (m/px, None: automatic), ``models``."""
        from .texts import model_text

        text = " ".join(self.c_text.currentText().split())
        cm = self.s_resolution.value()
        return {"text": text, "prompt": model_text(text), "raster": self.c_raster.currentData(), "area": self.c_area.currentData(),
                "threshold": round(self.s_threshold.value(), 3), "min_area": self.s_min.value(),
                "max_area": self.s_max.value(), "resolution": cm / 100.0 if cm > 0 else None,
                "models": self.c_models.currentData()}

    def set_status(self, text: str, can_install: bool) -> None:
        self.l_status.setText(text)
        self.b_install.setVisible(can_install)

    def set_rasters(self, rasters: list[tuple[str, str]]) -> None:
        """The rasters of the chunk, ``(name, label)``: the one chosen stays when it is still there."""
        current = self.c_raster.currentData()
        self.c_raster.blockSignals(True)
        self.c_raster.clear()
        for name, label in rasters:
            self.c_raster.addItem(label, name)
        k = self.c_raster.findData(current)
        self.c_raster.setCurrentIndex(k if k >= 0 else 0)
        self.c_raster.blockSignals(False)
        self.b_find.setEnabled(bool(rasters) and self.form_holder.isEnabled())

    # ------------------------------------------------------------------ the runs
    def set_runs(self, runs: list[dict]) -> None:
        selected = set(self.selected_runs())
        self._filling = True
        self.table.setRowCount(0)
        self._ids = []
        for run in runs:
            r = self.table.rowCount()
            self.table.insertRow(r)
            first = QTableWidgetItem(run.get("name", "?"))
            first.setFlags(first.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            first.setCheckState(Qt.CheckState.Checked if run.get("visible", True) else Qt.CheckState.Unchecked)
            first.setToolTip(run.get("prompt", ""))
            self.table.setItem(r, 0, first)
            self.table.setItem(r, 1, QTableWidgetItem(number(int(run.get("count", 0)))))
            self.table.setItem(r, 2, QTableWidgetItem(run.get("raster_label") or run.get("raster", "")))
            self.table.setItem(r, 3, QTableWidgetItem(str(run.get("time", "")).replace("T", " ")[:16]))
            self._ids.append(run["id"])
            if run["id"] in selected:
                self.table.selectRow(r)
        self._filling = False
        self._selection_changed()

    def selected_runs(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._ids[r] for r in rows if r < len(self._ids)]

    def current_run(self) -> str | None:
        r = self.table.currentRow()
        chosen = self.selected_runs()
        if r >= 0 and r < len(self._ids) and self._ids[r] in chosen:
            return self._ids[r]
        return chosen[0] if chosen else None

    def select_run(self, run_id: str) -> None:
        if run_id in self._ids:
            self.table.selectRow(self._ids.index(run_id))

    def _emit(self, signal, row: int | None = None) -> None:
        run_id = self._ids[row] if row is not None and 0 <= row < len(self._ids) else self.current_run()
        if run_id is not None:
            signal.emit(run_id)

    def _item_changed(self, item) -> None:
        if self._filling or item.column() != 0 or item.row() >= len(self._ids):
            return
        self.visibility_changed.emit(self._ids[item.row()], item.checkState() == Qt.CheckState.Checked)

    def _selection_changed(self) -> None:
        one = self.current_run() is not None
        for b in (self.b_units, self.b_classes, self.b_zoom, self.b_remove):
            b.setEnabled(one)
        self.b_export.setEnabled(bool(self._ids))

    def set_busy(self, busy: bool) -> None:
        """While a step of the program runs, the buttons wait (one step at a time)."""
        for h in self._holders:
            h.setEnabled(not busy)
        self.b_find.setEnabled(not busy and self.c_raster.count() > 0)

    def set_last(self, text: str) -> None:
        self.l_last.setText(text)
