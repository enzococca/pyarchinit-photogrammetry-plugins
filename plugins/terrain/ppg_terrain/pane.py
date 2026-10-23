"""The Terreno pane: the analyses as buttons (their parameters asked in a dialog), the products
of the chunk in a table, the overlay on the raster view with palette and opacity, the vectors as
layers, the exports."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .overlay import CATEGORICAL, FORMS, PALETTES
from .texts import t

COLS = ("Nome", "Tipo", "Celle", "Min", "Max", "Fatto")
COLS_EN = {"Nome": "Name", "Tipo": "Kind", "Celle": "Cells", "Min": "Min", "Max": "Max", "Fatto": "Made"}
KINDS = {"log": ("accumulo", "accumulation"), "classes": ("classi / id", "classes / ids"), "diverging": ("± 0", "± 0"),
         "grey": ("valori", "values"), "binary": ("sì / no", "yes / no"), "forms": ("forme", "landforms")}


class TerrainPane(QWidget):
    hydrology_requested = Signal()
    geomorph_requested = Signal()
    depressions_requested = Signal()
    viewshed_requested = Signal()
    overlay_requested = Signal(str, str, float)  # product id, palette ("" = the kind's own), opacity
    opacity_changed = Signal(float)
    hide_requested = Signal()
    layer_requested = Signal(str)
    export_requested = Signal(str)
    remove_requested = Signal(str)
    qgis_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Analisi del terreno sul DEM con WhiteboxTools: reticolo e bacini, posizione topografica e "
                         "forme, depressioni e rilievi annidati, visibilità. I risultati si sovrappongono alla vista, "
                         "vanno nei layer e in QGIS.",
                         "Terrain analysis of the DEM with WhiteboxTools: streams and basins, topographic position and "
                         "landforms, nested depressions and mounds, viewshed. The results overlay the view, go into "
                         "the layers and into QGIS."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        # the buttons go onto more lines in a narrow dock: the pane never widens the column of the window
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_hydro = QPushButton(t("Idrologia...", "Hydrology..."))
        self.b_hydro.setToolTip(t("Depressioni aperte, accumulo di flusso, reticolo e bacini",
                                  "Depressions breached, flow accumulation, stream network and basins"))
        self.b_geomorph = QPushButton(t("Geomorfometria...", "Geomorphometry..."))
        self.b_geomorph.setToolTip(t("Posizione topografica, scarto dalla media, geomorphons, curvature, rugosità, ore di luce",
                                     "Topographic position, deviation from the mean, geomorphons, curvatures, ruggedness, daylight"))
        self.b_depr = QPushButton(t("Depressioni e rilievi...", "Depressions and mounds..."))
        self.b_view = QPushButton(t("Visibilità...", "Viewshed..."))
        for b in (self.b_hydro, self.b_geomorph, self.b_depr, self.b_view):
            bar.addWidget(b)
        lay.addWidget(holder)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels([t(c, COLS_EN[c]) for c in COLS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(t("Palette", "Palette")))
        self.c_palette = QComboBox()
        self.c_palette.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_palette.setMinimumContentsLength(10)
        self.c_palette.addItem(t("Del tipo di prodotto", "The product's own"), "")
        for name, label in (("grey", t("Grigi", "Greys")), ("grey_inverted", t("Grigi invertiti", "Greys inverted")),
                            ("diverging", t("Divergente (blu-rosso)", "Diverging (blue-red)")),
                            ("earth", t("Terra", "Earth")), ("heat", t("Calore", "Heat")), ("water", t("Acqua", "Water")),
                            (CATEGORICAL, t("Classi (un colore per valore)", "Classes (a colour per value)")),
                            (FORMS, t("Forme del terreno (geomorphons)", "Landforms (geomorphons)"))):
            if name in PALETTES or name in (CATEGORICAL, FORMS):
                self.c_palette.addItem(label, name)
        row2.addWidget(self.c_palette)
        row2.addWidget(QLabel(t("Opacità", "Opacity")))
        self.s_opacity = QSlider(Qt.Orientation.Horizontal)
        self.s_opacity.setRange(10, 100)
        self.s_opacity.setValue(70)
        row2.addWidget(self.s_opacity, 1)
        lay.addLayout(row2)
        holder3 = QWidget()
        row3 = FlowLayout(holder3)
        self.b_overlay = QPushButton(t("Sovrapponi alla vista", "Overlay on the view"))
        self.b_hide = QPushButton(t("Nascondi", "Hide"))
        self.b_layer = QPushButton(t("Vettori nei layer", "Vectors into the layers"))
        self.b_layer.setToolTip(t("Le linee del reticolo, i poligoni dei bacini o delle depressioni nella vista",
                                  "The stream lines, the polygons of the basins or of the depressions in the view"))
        self.b_export = QPushButton(t("Esporta...", "Export..."))
        self.b_qgis = QPushButton(t("QGIS", "QGIS"))
        self.b_remove = QPushButton(t("Rimuovi", "Remove"))
        for b in (self.b_overlay, self.b_hide, self.b_layer, self.b_export, self.b_qgis, self.b_remove):
            row3.addWidget(b)
        lay.addWidget(holder3)
        self.l_limits = QLabel("")
        lay.addWidget(self.l_limits)
        self.b_hydro.clicked.connect(self.hydrology_requested)
        self.b_geomorph.clicked.connect(self.geomorph_requested)
        self.b_depr.clicked.connect(self.depressions_requested)
        self.b_view.clicked.connect(self.viewshed_requested)
        self.b_overlay.clicked.connect(self._overlay)
        self.b_hide.clicked.connect(self.hide_requested)
        self.b_layer.clicked.connect(lambda: self._emit(self.layer_requested))
        self.b_export.clicked.connect(lambda: self._emit(self.export_requested))
        self.b_remove.clicked.connect(lambda: self._emit(self.remove_requested))
        self.b_qgis.clicked.connect(self.qgis_requested)
        self.s_opacity.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        self.table.itemDoubleClicked.connect(lambda _it: self._overlay())
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.entries: list[dict] = []
        self.set_products([])

    def current_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        return str(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))

    def current_entry(self) -> dict | None:
        pid = self.current_id()
        return next((e for e in self.entries if e.get("id") == pid), None)

    def _emit(self, signal) -> None:
        pid = self.current_id()
        if pid:
            signal.emit(pid)

    def _overlay(self) -> None:
        pid = self.current_id()
        if pid:
            self.overlay_requested.emit(pid, str(self.c_palette.currentData() or ""), self.s_opacity.value() / 100.0)

    def _selection_changed(self) -> None:
        e = self.current_entry()
        has = e is not None
        for b in (self.b_overlay, self.b_export, self.b_remove):
            b.setEnabled(has)
        self.b_layer.setEnabled(has and bool(e.get("vector")))

    def set_status(self, text: str) -> None:
        self.l_status.setText(text)

    def set_limits(self, text: str) -> None:
        self.l_limits.setText(text)

    def set_products(self, entries: list[dict]) -> None:
        self.entries = list(entries)
        current = self.current_id()
        self.table.setRowCount(0)
        for e in self.entries:
            r = self.table.rowCount()
            self.table.insertRow(r)
            st = e.get("stats") or {}
            kind = KINDS.get(e.get("kind", ""), (e.get("kind", ""), e.get("kind", "")))
            cells = [e.get("name", e.get("id")), t(*kind), f"{st.get('cells', 0):,}",
                     f"{st.get('min', 0):.3g}" if "min" in st else "", f"{st.get('max', 0):.3g}" if "max" in st else "",
                     e.get("made", "")]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, e.get("id"))
                    it.setToolTip(str(e.get("params", "")))
                self.table.setItem(r, c, it)
            if e.get("id") == current:
                self.table.selectRow(r)
        if self.entries and self.current_id() is None:
            self.table.selectRow(0)
        self._selection_changed()
        self.b_qgis.setEnabled(bool(self.entries))
