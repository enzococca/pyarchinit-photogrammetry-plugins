"""The Geofisica pane: the grids of the chunk in a table, the buttons of the plugin, the overlay
on the raster view with its band (the slices of a GPR survey), palette and opacity."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .overlay import PALETTES
from .texts import t

COLS = ("Nome", "Tipo", "Unità", "Cella (m)", "Celle", "Bande", "Posizione", "Filtri")


class GeophysicsPane(QWidget):
    import_requested = Signal()
    georef_requested = Signal(str)
    filter_requested = Signal(str)
    overlay_requested = Signal(str, int, str, float)  # grid id, band, palette, opacity
    opacity_changed = Signal(float)
    hide_requested = Signal()
    export_requested = Signal(str)
    remove_requested = Signal(str)
    qgis_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Griglie geofisiche (magnetometria, resistività, GPR, EM): importa, georeferenzia con i "
                         "marker o le coordinate dei picchetti, filtra, sovrapponi a ortofoto e DEM, esporta.",
                         "Geophysical grids (magnetometry, resistivity, GPR, EM): import, georeference with markers "
                         "or the pegs' coordinates, filter, overlay on the orthomosaic and DEM, export."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        # the buttons go onto more lines in a narrow dock: the pane never widens the column of the window
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_import = QPushButton(t("Importa griglia...", "Import grid..."))
        self.b_georef = QPushButton(t("Georeferenzia...", "Georeference..."))
        self.b_filter = QPushButton(t("Filtri...", "Filters..."))
        for b in (self.b_import, self.b_georef, self.b_filter):
            bar.addWidget(b)
        lay.addWidget(holder)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels([t(c, {"Nome": "Name", "Tipo": "Kind", "Unità": "Units", "Cella (m)": "Cell (m)",
                                                   "Celle": "Cells", "Bande": "Bands", "Posizione": "Position",
                                                   "Filtri": "Filters"}[c]) for c in COLS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(t("Banda", "Band")))
        self.s_band = QSlider(Qt.Orientation.Horizontal)
        self.s_band.setRange(0, 0)
        row2.addWidget(self.s_band, 1)
        self.l_band = QLabel("1")
        row2.addWidget(self.l_band)
        row2.addWidget(QLabel(t("Palette", "Palette")))
        self.c_palette = QComboBox()
        self.c_palette.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_palette.setMinimumContentsLength(8)
        for name, label in (("grey", t("Grigi", "Greys")), ("grey_inverted", t("Grigi invertiti", "Greys inverted")),
                            ("diverging", t("Divergente (blu-rosso)", "Diverging (blue-red)")),
                            ("earth", t("Terra", "Earth")), ("heat", t("Calore", "Heat"))):
            if name in PALETTES:
                self.c_palette.addItem(label, name)
        row2.addWidget(self.c_palette)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel(t("Opacità", "Opacity")))
        self.s_opacity = QSlider(Qt.Orientation.Horizontal)
        self.s_opacity.setRange(10, 100)
        self.s_opacity.setValue(70)
        row3.addWidget(self.s_opacity, 1)
        lay.addLayout(row3)
        holder2 = QWidget()
        bar2 = FlowLayout(holder2)
        self.b_overlay = QPushButton(t("Sovrapponi alla vista", "Overlay on the view"))
        self.b_hide = QPushButton(t("Nascondi", "Hide"))
        self.b_export = QPushButton(t("Esporta...", "Export..."))
        self.b_qgis = QPushButton(t("Apri in QGIS", "Open in QGIS"))
        self.b_remove = QPushButton(t("Rimuovi", "Remove"))
        for b in (self.b_overlay, self.b_hide, self.b_export, self.b_qgis, self.b_remove):
            bar2.addWidget(b)
        lay.addWidget(holder2)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.entries: list[dict] = []
        self.b_import.clicked.connect(self.import_requested)
        self.b_georef.clicked.connect(lambda: self._emit(self.georef_requested))
        self.b_filter.clicked.connect(lambda: self._emit(self.filter_requested))
        self.b_export.clicked.connect(lambda: self._emit(self.export_requested))
        self.b_remove.clicked.connect(lambda: self._emit(self.remove_requested))
        self.b_qgis.clicked.connect(self.qgis_requested)
        self.b_hide.clicked.connect(self.hide_requested)
        self.b_overlay.clicked.connect(self._overlay)
        self.s_band.valueChanged.connect(self._band_changed)
        self.s_opacity.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.set_grids([])

    # --------------------------------------------------------------- selection
    def selected_id(self) -> str | None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows and self.entries:
            return self.entries[0]["id"]
        return self.entries[rows[0]]["id"] if rows and rows[0] < len(self.entries) else None

    def selected_entry(self) -> dict | None:
        gid = self.selected_id()
        return next((e for e in self.entries if e["id"] == gid), None)

    def _emit(self, signal) -> None:
        gid = self.selected_id()
        if gid is not None:
            signal.emit(gid)

    def _selection_changed(self) -> None:
        e = self.selected_entry()
        bands = int(e.get("bands", 1)) if e else 1
        self.s_band.setRange(0, max(bands - 1, 0))
        self.s_band.setEnabled(bands > 1)
        self._band_changed(self.s_band.value())
        if e:
            k = self.c_palette.findData({"magnetometry": "grey", "gpr": "grey", "resistivity": "earth", "em": "earth",
                                         "susceptibility": "heat"}.get(e.get("kind"), "grey"))
            if k >= 0 and not getattr(self, "_palette_touched", False):
                self.c_palette.setCurrentIndex(k)

    def _band_changed(self, k: int) -> None:
        e = self.selected_entry()
        slices = (e or {}).get("slices") or []
        self.l_band.setText(str(slices[k]) if k < len(slices) else str(k + 1))

    def _overlay(self) -> None:
        gid = self.selected_id()
        if gid is None:
            return
        self._palette_touched = True
        self.overlay_requested.emit(gid, int(self.s_band.value()), str(self.c_palette.currentData()),
                                    self.s_opacity.value() / 100.0)

    def band(self) -> int:
        return int(self.s_band.value())

    # --------------------------------------------------------------- state
    def set_grids(self, entries: list[dict]) -> None:
        self.entries = list(entries)
        keep = self.selected_id()
        self.table.setRowCount(len(entries))
        for r, e in enumerate(entries):
            where = e.get("crs", "LOCAL")
            vals = [e["name"], e.get("kind", ""), e.get("units", ""), f"{e['cell']:g}", f"{e['cols']} x {e['rows']}",
                    str(e.get("bands", 1)), t("locale", "local") if where == "LOCAL" else where,
                    ", ".join(f["name"] for f in e.get("filters", []))]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        has = bool(entries)
        for b in (self.b_georef, self.b_filter, self.b_export, self.b_remove, self.b_overlay, self.b_hide, self.b_qgis):
            b.setEnabled(has)
        if has:
            row = next((i for i, e in enumerate(entries) if e["id"] == keep), 0)
            self.table.selectRow(row)
        self._selection_changed()
        self.status.setText(t("{0} griglie", "{0} grids").format(len(entries)) if has else
                            t("Nessuna griglia: importa un file Surfer GRD, XYZ/CSV, GeoTIFF o una cartella di slice GPR.",
                              "No grid: import a Surfer GRD, XYZ/CSV, GeoTIFF file or a folder of GPR slices."))
