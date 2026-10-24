"""The Sentinel-2 pane of the project dock: the area, the search, the acquisitions found with
their cloud cover and clear part of the area, reading and computing, the chart of the series,
the overlay of an index (a date, a period, natural colour), the crop-mark maps and composites,
the exports. Every row of buttons goes onto more lines in a narrow dock (the program's flow
layout): the pane never widens the column of the window."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDateEdit, QDoubleSpinBox, QGridLayout, QGroupBox,
                               QHeaderView, QLabel, QMenu, QPushButton, QSizePolicy, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from . import stac
from .analysis import BACKGROUND_M, INDICES
from .overlay import PALETTE_NAMES, PALETTES
from .texts import index_name, period_name, product_name, status_name, t

SCENE_COLS = (("Data", "Date"), ("Satellite", "Satellite"), ("Tile", "Tile"), ("Nuvole %", "Clouds %"),
              ("Sereno %", "Clear %"), ("Stato", "Status"))
PRODUCT_COLS = (("Nome", "Name"), ("Date", "Dates"), ("Fatto", "Made"))
CATALOGS = [stac.EARTH_SEARCH]


def _buttons(*buttons) -> QWidget:
    holder = QWidget()
    row = FlowLayout(holder)
    for b in buttons:
        row.addWidget(b)
    return holder


def _table(cols) -> QTableWidget:
    table = QTableWidget(0, len(cols))
    table.setHorizontalHeaderLabels([t(*c) for c in cols])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    return table


class ChartLabel(QWidget):
    """The PNG of the series drawn as large as it fits (its proportions kept) in a box of fixed
    height, whatever the width of the dock; a double click opens it at full size."""

    opened = Signal()
    HEIGHT = 260

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(0)
        self.setToolTip(t("Doppio clic: il grafico a piena grandezza", "Double click: the chart at full size"))

    def set_image(self, path: Path | None) -> None:
        self._pixmap = QPixmap(str(path)) if path is not None and Path(path).exists() else None
        if self._pixmap is not None and self._pixmap.isNull():
            self._pixmap = None
        self.setFixedHeight(self.HEIGHT if self._pixmap is not None else 0)
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def paintEvent(self, ev) -> None:  # noqa: N802 - Qt
        if self._pixmap is None:
            return
        from PySide6.QtGui import QPainter

        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(self)
        painter.drawPixmap((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled)
        painter.end()

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802 - Qt
        if self._pixmap is not None:
            self.opened.emit()


class Sentinel2Pane(QWidget):
    draw_requested = Signal()
    area_requested = Signal()  # the area from the DEM or orthomosaic of the chunk
    search_requested = Signal(dict)
    fetch_requested = Signal(list, float)  # the acquisitions selected ([] = all), the least clear fraction
    compute_requested = Signal()
    overlay_requested = Signal(dict)
    opacity_changed = Signal(float)
    hide_requested = Signal()
    frame_requested = Signal()
    cropmarks_requested = Signal(dict)
    composite_requested = Signal(dict)
    product_overlay_requested = Signal(str)
    product_export_requested = Signal(str)
    remove_requested = Signal(str)
    export_requested = Signal()
    qgis_requested = Signal()
    chart_requested = Signal()
    scenes_remove_requested = Signal(list)  # acquisitions of the cache to take out of it
    cache_clear_requested = Signal()  # every acquisition out of the cache
    scene_show_requested = Signal(str)  # natural colour of an acquisition of the cache

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Serie temporali Sentinel-2 sull'area del rilievo per cercare i crop mark: muri e fossati "
                         "sepolti si vedono nella vegetazione sopra di loro, più debole o più vigorosa, in alcune "
                         "stagioni.",
                         "Sentinel-2 time series over the area of the survey to look for crop marks: buried walls "
                         "and ditches show in the vegetation above them, weaker or greener, in some seasons."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)

        # -- area
        g_area = QGroupBox(t("Area", "Area"))
        ga = QGridLayout(g_area)
        ga.addWidget(QLabel(t("Margine (m)", "Margin (m)")), 0, 0)
        self.s_margin = QSpinBox()
        self.s_margin.setRange(0, 5000)
        self.s_margin.setSingleStep(50)
        self.s_margin.setValue(300)
        self.s_margin.setToolTip(t("Il terreno intorno all'area letto anch'esso: lo sfondo delle mappe dei crop mark "
                                   "ne ha bisogno", "The ground around the area read as well: the background of the "
                                   "crop-mark maps needs it"))
        ga.addWidget(self.s_margin, 0, 1)
        self.b_area = QPushButton(t("Dal DEM o dall'ortofoto", "From the DEM or orthomosaic"))
        self.b_draw = QPushButton(t("Disegna l'area...", "Draw the area..."))
        self.b_frame = QPushButton(t("Mostra l'area intera", "Show the whole area"))
        self.b_frame.setToolTip(t("Una cornice vuota con l'area e il suo margine, per vedere i prodotti oltre "
                                  "l'ortofoto", "An empty frame with the area and its margin, to see the products "
                                  "beyond the orthomosaic"))
        ga.addWidget(_buttons(self.b_area, self.b_draw, self.b_frame), 1, 0, 1, 2)
        self.l_area = QLabel("")
        self.l_area.setWordWrap(True)
        ga.addWidget(self.l_area, 2, 0, 1, 2)
        lay.addWidget(g_area)

        # -- search
        g_search = QGroupBox(t("Ricerca", "Search"))
        gs = QGridLayout(g_search)
        today = date.today()
        self.d_start = QDateEdit(QDate(today.year - 2, 1, 1))
        self.d_end = QDateEdit(QDate(today.year, today.month, today.day))
        for d in (self.d_start, self.d_end):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
        gs.addWidget(QLabel(t("Dal", "From")), 0, 0)
        gs.addWidget(self.d_start, 0, 1)
        gs.addWidget(QLabel(t("Al", "To")), 1, 0)
        gs.addWidget(self.d_end, 1, 1)
        gs.addWidget(QLabel(t("Nuvole al massimo (%)", "Clouds at most (%)")), 2, 0)
        self.s_cloud = QSpinBox()
        self.s_cloud.setRange(0, 100)
        self.s_cloud.setValue(30)
        self.s_cloud.setToolTip(t("Le nuvole dei tile interi del catalogo; quelle sull'area si misurano leggendo "
                                  "la classificazione della scena", "The clouds of the whole tiles of the catalog; "
                                  "those over the area are measured reading the scene classification"))
        gs.addWidget(self.s_cloud, 2, 1)
        gs.addWidget(QLabel(t("Catalogo STAC", "STAC catalog")), 3, 0)
        self.c_catalog = QComboBox()
        self.c_catalog.setEditable(True)
        self.c_catalog.addItems(CATALOGS)
        self.c_catalog.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_catalog.setMinimumContentsLength(12)
        gs.addWidget(self.c_catalog, 3, 1)
        self.b_search = QPushButton(t("Cerca", "Search"))
        gs.addWidget(_buttons(self.b_search), 4, 0, 1, 2)
        lay.addWidget(g_search)

        # -- acquisitions
        self.scenes = _table(SCENE_COLS)
        self.scenes.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scenes.setMinimumHeight(120)
        lay.addWidget(self.scenes, 1)
        g_read = QWidget()
        gr = QGridLayout(g_read)
        gr.setContentsMargins(0, 0, 0, 0)
        gr.addWidget(QLabel(t("Parte serena minima dell'area (%)", "Least clear part of the area (%)")), 0, 0)
        self.s_clear = QSpinBox()
        self.s_clear.setRange(1, 100)
        self.s_clear.setValue(50)
        gr.addWidget(self.s_clear, 0, 1)
        self.b_fetch = QPushButton(t("Scarica e calcola", "Read and compute"))
        self.b_fetch.setToolTip(t("Le acquisizioni selezionate (tutte se nessuna è selezionata): solo la finestra "
                                  "dell'area, una volta; nuvole e ombre mascherate", "The acquisitions selected (all "
                                  "when none is): only the window of the area, once; clouds and shadows masked"))
        self.b_compute = QPushButton(t("Ricalcola", "Compute again"))
        self.b_compute.setToolTip(t("Indici e serie dalle acquisizioni già lette, senza rete", "Indices and series "
                                    "from the acquisitions already read, without network"))
        self.b_uncache = QPushButton(t("Rimuovi dalla cache", "Remove from the cache"))
        self.b_uncache.setToolTip(t("Le acquisizioni lette selezionate: i loro file vengono cancellati, restano nella "
                                    "ricerca e si possono leggere di nuovo", "The acquisitions read that are selected: "
                                    "their files are deleted, they stay in the search and can be read again"))
        self.b_clear_cache = QPushButton(t("Svuota la cache...", "Empty the cache..."))
        self.b_clear_cache.setToolTip(t("Tutte le acquisizioni lette tolte dalla cache del progetto",
                                        "Every acquisition read taken out of the project's cache"))
        gr.addWidget(_buttons(self.b_fetch, self.b_compute, self.b_uncache, self.b_clear_cache), 1, 0, 1, 2)
        lay.addWidget(g_read)
        self.scenes.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scenes.customContextMenuRequested.connect(self._scenes_context)
        self.scenes.itemDoubleClicked.connect(self._scene_double)
        self.scenes.itemSelectionChanged.connect(self._enable)

        self.chart = ChartLabel()
        lay.addWidget(self.chart)

        # -- the view
        g_view = QGroupBox(t("Vista", "View"))
        gv = QGridLayout(g_view)
        gv.addWidget(QLabel(t("Indice", "Index")), 0, 0)
        self.c_index = QComboBox()
        for name in list(INDICES) + ["RGB"]:
            self.c_index.addItem(index_name(name), name)
        gv.addWidget(self.c_index, 0, 1)
        gv.addWidget(QLabel(t("Data o periodo", "Date or period")), 1, 0)
        self.c_period = QComboBox()
        self.c_period.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_period.setMinimumContentsLength(12)
        gv.addWidget(self.c_period, 1, 1)
        gv.addWidget(QLabel(t("Palette", "Palette")), 2, 0)
        self.c_palette = QComboBox()
        for key in PALETTES:
            self.c_palette.addItem(t(*PALETTE_NAMES[key]), key)
        gv.addWidget(self.c_palette, 2, 1)
        gv.addWidget(QLabel(t("Opacità", "Opacity")), 3, 0)
        self.s_opacity = QSlider(Qt.Orientation.Horizontal)
        self.s_opacity.setRange(10, 100)
        self.s_opacity.setValue(75)
        gv.addWidget(self.s_opacity, 3, 1)
        self.b_overlay = QPushButton(t("Sovrapponi", "Overlay"))
        self.b_hide = QPushButton(t("Nascondi", "Hide"))
        gv.addWidget(_buttons(self.b_overlay, self.b_hide), 4, 0, 1, 2)
        self.l_legend = QLabel("")
        self.l_legend.setWordWrap(True)
        gv.addWidget(self.l_legend, 5, 0, 1, 2)
        lay.addWidget(g_view)

        # -- crop marks
        g_crop = QGroupBox(t("Crop mark", "Crop marks"))
        gc = QGridLayout(g_crop)
        gc.addWidget(QLabel(t("Sfondo (m)", "Background (m)")), 0, 0)
        self.s_background = QDoubleSpinBox()
        self.s_background.setRange(10.0, 1000.0)
        self.s_background.setDecimals(0)
        self.s_background.setSingleStep(10.0)
        self.s_background.setValue(BACKGROUND_M)
        self.s_background.setToolTip(t("La scala della variazione liscia dei campi tolta all'indice (sigma della "
                                       "gaussiana; la mediana usa una finestra di lato doppio)", "The scale of the "
                                       "smooth variation of the fields taken from the index (sigma of the Gaussian; "
                                       "the median uses a window of twice that side)"))
        gc.addWidget(self.s_background, 0, 1)
        gc.addWidget(QLabel(t("Metodo", "Method")), 1, 0)
        self.c_method = QComboBox()
        self.c_method.addItem(t("Gaussiana", "Gaussian"), "gaussian")
        self.c_method.addItem(t("Mediana", "Median"), "median")
        gc.addWidget(self.c_method, 1, 1)
        self.b_cropmarks = QPushButton(t("Mappa dei crop mark", "Crop-mark map"))
        self.b_cropmarks.setToolTip(t("L'indice e il periodo scelti sopra, meno lo sfondo, diviso per la dispersione "
                                      "robusta (z): sotto -2 vegetazione più debole, sopra 2 più vigorosa",
                                      "The index and period chosen above, minus the background, divided by the robust "
                                      "spread (z): under -2 weaker vegetation, over 2 greener"))
        self.b_composite = QPushButton(t("Composito del periodo", "Composite of the period"))
        gc.addWidget(_buttons(self.b_cropmarks, self.b_composite), 2, 0, 1, 2)
        lay.addWidget(g_crop)

        # -- products
        self.products = _table(PRODUCT_COLS)
        self.products.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.products.setMinimumHeight(90)
        lay.addWidget(self.products, 1)
        self.b_p_overlay = QPushButton(t("Sovrapponi", "Overlay"))
        self.b_p_export = QPushButton(t("Esporta...", "Export..."))
        self.b_p_remove = QPushButton(t("Rimuovi", "Remove"))
        self.b_qgis = QPushButton("QGIS")
        self.b_export = QPushButton(t("Esporta tutto...", "Export everything..."))
        self.b_export.setToolTip(t("Stack degli indici (GeoTIFF), serie (CSV per data e per mese), grafico, prodotti",
                                   "Stack of the indices (GeoTIFF), series (CSV per date and per month), chart, products"))
        lay.addWidget(_buttons(self.b_p_overlay, self.b_p_export, self.b_p_remove, self.b_qgis, self.b_export))
        self.l_attribution = QLabel("")
        self.l_attribution.setWordWrap(True)
        self.l_attribution.setStyleSheet("color: #777777; font-size: 10px;")
        lay.addWidget(self.l_attribution)

        self.b_area.clicked.connect(self.area_requested)
        self.b_draw.clicked.connect(self.draw_requested)
        self.b_frame.clicked.connect(self.frame_requested)
        self.b_search.clicked.connect(lambda: self.search_requested.emit(self.search_params()))
        self.b_fetch.clicked.connect(lambda: self.fetch_requested.emit(self.selected_scenes(),
                                                                       self.s_clear.value() / 100.0))
        self.b_compute.clicked.connect(self.compute_requested)
        self.b_uncache.clicked.connect(lambda: self.scenes_remove_requested.emit(self.selected_cached()))
        self.b_clear_cache.clicked.connect(self.cache_clear_requested)
        self.b_overlay.clicked.connect(lambda: self.overlay_requested.emit(self.view_params()))
        self.b_hide.clicked.connect(self.hide_requested)
        self.s_opacity.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        self.b_cropmarks.clicked.connect(lambda: self.cropmarks_requested.emit(self.cropmark_params()))
        self.b_composite.clicked.connect(lambda: self.composite_requested.emit(self.cropmark_params()))
        self.b_p_overlay.clicked.connect(lambda: self._emit_product(self.product_overlay_requested))
        self.products.itemDoubleClicked.connect(lambda _it: self._emit_product(self.product_overlay_requested))
        self.b_p_export.clicked.connect(lambda: self._emit_product(self.product_export_requested))
        self.b_p_remove.clicked.connect(lambda: self._emit_product(self.remove_requested))
        self.b_qgis.clicked.connect(self.qgis_requested)
        self.b_export.clicked.connect(self.export_requested)
        self.chart.opened.connect(self.chart_requested)
        self.products.itemSelectionChanged.connect(self._product_selection)
        self.c_index.currentIndexChanged.connect(self._index_changed)
        self.c_period.currentIndexChanged.connect(self._index_changed)
        self.entries: list[dict] = []
        self._state: dict = {}
        self._busy = False
        self.set_state({})

    # ------------------------------------------------------------------ height in the dock's scroll area
    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt
        """At least the height the wrapped texts need at the width the pane has: the scroll area of
        the dock gives the pane its minimum height, and the wrapped labels report a single line."""
        hint = super().minimumSizeHint()
        lay = self.layout()
        if self.width() > 50 and lay is not None and lay.hasHeightForWidth():
            return QSize(hint.width(), max(hint.height(), lay.totalHeightForWidth(self.width())))
        return hint

    def resizeEvent(self, ev) -> None:  # noqa: N802 - Qt
        super().resizeEvent(ev)
        if ev.oldSize().width() != ev.size().width():
            self.updateGeometry()

    # ------------------------------------------------------------------ reading the controls
    def search_params(self) -> dict:
        return {"start": self.d_start.date().toString("yyyy-MM-dd"), "end": self.d_end.date().toString("yyyy-MM-dd"),
                "max_cloud": float(self.s_cloud.value()), "catalog": self.c_catalog.currentText().strip() or stac.EARTH_SEARCH,
                "margin": float(self.s_margin.value())}

    def selected_scenes(self) -> list[str]:
        rows = sorted({i.row() for i in self.scenes.selectionModel().selectedRows()}) if self.scenes.selectionModel() else []
        return [str(self.scenes.item(r, 0).data(Qt.ItemDataRole.UserRole)) for r in rows]

    def _status_of(self) -> dict:
        return {r.get("id"): r.get("status") for r in (getattr(self, "_state", {}) or {}).get("scenes") or []}

    def selected_cached(self) -> list[str]:
        """The acquisitions selected in the table that are in the cache."""
        status = self._status_of()
        return [i for i in self.selected_scenes() if status.get(i) == "cached"]

    def scenes_menu(self, row: int) -> QMenu:
        """The context menu of the table of the acquisitions (``row`` under the mouse, -1 for none):
        natural colour of a cached one, the selected ones out of the cache, the whole cache emptied."""
        menu = QMenu(self)
        status = self._status_of()
        under = str(self.scenes.item(row, 0).data(Qt.ItemDataRole.UserRole)) if row >= 0 and self.scenes.item(row, 0) else None
        busy = getattr(self, "_busy", False)
        a = menu.addAction(t("Mostra (colori naturali)", "Show (natural colour)"))
        a.setEnabled(under is not None and status.get(under) == "cached")
        a.triggered.connect(lambda: self.scene_show_requested.emit(under))
        chosen = self.selected_cached()
        if under is not None and status.get(under) == "cached" and under not in chosen:
            chosen = [under]
        a = menu.addAction(t("Rimuovi dalla cache ({0})", "Remove from the cache ({0})").format(len(chosen)))
        a.setEnabled(bool(chosen) and not busy)
        a.triggered.connect(lambda: self.scenes_remove_requested.emit(list(chosen)))
        menu.addSeparator()
        a = menu.addAction(t("Svuota la cache...", "Empty the cache..."))
        a.setEnabled(any(s == "cached" for s in status.values()) and not busy)
        a.triggered.connect(self.cache_clear_requested)
        return menu

    def _scenes_context(self, pos) -> None:
        index = self.scenes.indexAt(pos)
        self.scenes_menu(index.row() if index.isValid() else -1).exec(self.scenes.viewport().mapToGlobal(pos))

    def _scene_double(self, item) -> None:
        acq = str(self.scenes.item(item.row(), 0).data(Qt.ItemDataRole.UserRole))
        if self._status_of().get(acq) == "cached":
            self.scene_show_requested.emit(acq)

    def view_params(self) -> dict:
        return {"index": str(self.c_index.currentData() or "NDVI"), "period": str(self.c_period.currentData() or "all"),
                "palette": str(self.c_palette.currentData() or "rdylgn"), "opacity": self.s_opacity.value() / 100.0}

    def cropmark_params(self) -> dict:
        index = str(self.c_index.currentData() or "NDVI")
        return {"index": index if index in INDICES else "NDVI", "period": str(self.c_period.currentData() or "all"),
                "background": float(self.s_background.value()), "method": str(self.c_method.currentData() or "gaussian")}

    def current_product(self) -> str | None:
        rows = self.products.selectionModel().selectedRows() if self.products.selectionModel() else []
        if not rows:
            return None
        return str(self.products.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))

    def _emit_product(self, signal) -> None:
        pid = self.current_product()
        if pid:
            signal.emit(pid)

    def _product_selection(self) -> None:
        self._enable()

    def _index_changed(self) -> None:
        self._enable()

    def _enable(self) -> None:
        """Every button enabled when what it needs is there (and no step runs)."""
        st = getattr(self, "_state", {}) or {}
        busy = getattr(self, "_busy", False)
        for b in (self.b_area, self.b_draw, self.b_search):
            b.setEnabled(not busy)
        ds = st.get("dates") or []
        rgb = self.c_index.currentData() == "RGB"
        has = self.current_product() is not None
        self.c_palette.setEnabled(not rgb)
        self.b_cropmarks.setEnabled(not busy and not rgb and bool(ds))
        self.b_composite.setEnabled(not busy and not rgb and bool(ds))
        any_cached = any(r.get("status") == "cached" for r in st.get("scenes") or [])
        self.b_fetch.setEnabled(not busy and bool(st.get("scenes")))
        self.b_compute.setEnabled(not busy and any_cached)
        self.b_uncache.setEnabled(not busy and bool(self.selected_cached()))
        self.b_clear_cache.setEnabled(not busy and any_cached)
        self.b_overlay.setEnabled(bool(ds))
        self.b_export.setEnabled(not busy and bool(ds))
        self.b_qgis.setEnabled(any(e.get("chunk_file") for e in self.entries))
        self.b_frame.setEnabled(bool(st.get("grid")))
        for b in (self.b_p_overlay, self.b_p_export):
            b.setEnabled(has)
        self.b_p_remove.setEnabled(has and not busy)

    # ------------------------------------------------------------------ showing the state
    def set_busy(self, busy: bool) -> None:
        """A step runs in the background: the buttons that would start another one wait."""
        self._busy = bool(busy)
        self._enable()

    def set_status(self, text: str) -> None:
        self.l_status.setText(text)

    def set_legend(self, text: str) -> None:
        self.l_legend.setText(text)

    def set_state(self, st: dict, chart: Path | None = None, area_text: str = "", attribution: str = "",
                  inputs: bool = True) -> None:
        """Everything from the plugin's part of ``chunk.meta`` (and the chart's file); ``inputs``:
        also the dates, clouds, catalog, margin and clear part of the last search (another chunk,
        or the first look), else what the user is typing stays."""
        from pyarchinit_photogrammetry.gui.i18n import thousands_points

        search = st.get("search") or {}
        if search and inputs:
            for d, key in ((self.d_start, "start"), (self.d_end, "end")):
                q = QDate.fromString(str(search.get(key, ""))[:10], "yyyy-MM-dd")
                if q.isValid():
                    d.setDate(q)
            self.s_cloud.setValue(int(round(float(search.get("max_cloud", 30)))))
            if search.get("catalog") and self.c_catalog.findText(search["catalog"]) < 0:
                self.c_catalog.addItem(search["catalog"])
            if search.get("catalog"):
                self.c_catalog.setCurrentText(search["catalog"])
        if st.get("area") and inputs:
            self.s_margin.setValue(int(round(float(st["area"].get("margin_m", 300)))))
        settings = st.get("settings") or {}
        if settings.get("min_clear") is not None and inputs:
            self.s_clear.setValue(max(1, int(round(float(settings["min_clear"]) * 100))))
        self.l_area.setText(area_text)
        # acquisitions
        self.scenes.setRowCount(0)
        for row in st.get("scenes") or []:
            r = self.scenes.rowCount()
            self.scenes.insertRow(r)
            cloud = "" if row.get("cloud") is None else f"{float(row['cloud']):.0f}"
            clear = "" if row.get("clear") is None else f"{float(row['clear']) * 100:.0f}"
            cells = [row.get("date", ""), row.get("platform", ""), " ".join(row.get("tiles") or []), cloud, clear,
                     status_name(row.get("status", ""))]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, row.get("id"))
                    it.setToolTip(str(row.get("id")) + (f"\n{row['error']}" if row.get("error") else ""))
                if c in (3, 4):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.scenes.setItem(r, c, it)
        # dates and periods of the view
        current = self.c_period.currentData()
        self.c_period.blockSignals(True)
        self.c_period.clear()
        from datetime import date as _date

        from .analysis import periods_of

        ds = st.get("dates") or []
        for p in periods_of([_date.fromisoformat(d["date"]) for d in ds]) if ds else []:
            self.c_period.addItem(period_name(p), p)
        for d in reversed(ds):
            clear = d.get("clear")
            self.c_period.addItem(f"{d['date']} {d.get('platform', '')}" + (f" ({clear * 100:.0f}%)" if clear is not None else ""),
                                  d["id"])
        k = self.c_period.findData(current) if current is not None else -1
        self.c_period.setCurrentIndex(k if k >= 0 else 0)
        self.c_period.blockSignals(False)
        # products
        selected = self.current_product()
        self.entries = list(st.get("products") or [])
        self.products.setRowCount(0)
        for e in self.entries:
            r = self.products.rowCount()
            self.products.insertRow(r)
            missing = e.get("missing") or []
            name = product_name(e) + (t(" (date non più in cache)", " (dates no longer cached)") if missing else "")
            for c, text in enumerate([name, str(len(e.get("dates") or [])), e.get("made", "")]):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, e.get("id"))
                    stats = e.get("stats") or {}
                    tip = e.get("id", "")
                    if e.get("kind") == "cropmarks" and "low" in stats:
                        tip += "\n" + t("{0:.1f}% dell'area sotto -2, {1:.1f}% sopra 2",
                                        "{0:.1f}% of the area under -2, {1:.1f}% over 2").format(stats["low"] * 100,
                                                                                               stats["high"] * 100)
                    if missing:
                        tip += "\n" + t("Fatto anche da acquisizioni non più in cache: {0}",
                                        "Made also from acquisitions no longer in the cache: {0}").format(", ".join(missing))
                    it.setToolTip(tip)
                self.products.setItem(r, c, it)
            if e.get("id") == selected:
                self.products.selectRow(r)
        self._state = st
        self._enable()
        self.chart.set_image(chart)
        self.l_attribution.setText(attribution)
        n = len(st.get("scenes") or [])
        if n:
            cached = sum(r.get("status") == "cached" for r in st.get("scenes") or [])
            self.set_status(t("{0} acquisizioni trovate, {1} lette, {2} date serene nella serie",
                              "{0} acquisitions found, {1} read, {2} clear dates in the series")
                            .format(thousands_points(str(n)), cached, len(ds)))
        else:
            self.set_status(t("Scegli l'area e cerca le immagini nel catalogo.", "Choose the area and search the catalog."))

