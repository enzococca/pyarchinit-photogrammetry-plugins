"""The Geoarcheologia pane: the steps as buttons (their parameters asked in dialogs), the
boreholes, the surfaces and the sections of the chunk in three tables, the overlay of a surface
on the raster view with its band, palette and opacity, the model of the units in the 3D view of
the program and its folder, the exports. The boreholes selected are kept in the order they were
picked: a section through them follows that order.

Every row of buttons is a :class:`FlowLayout` and the combos keep a short minimum: in a narrow
dock the pane wraps and never widens the column of the window."""
from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHeaderView, QLabel, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .overlay import PALETTES
from .texts import num, t

BANDS = ((1, ("Quota", "Elevation")), (2, ("Varianza di kriging", "Kriging variance")),
         (3, ("Correzione della pila", "Correction by the stack")))
MODEL_NAMES = {"spherical": ("sferico", "spherical"), "exponential": ("esponenziale", "exponential"),
               "gaussian": ("gaussiano", "Gaussian")}
METHOD_NAMES = {"ground": ("terreno", "ground"), "idw": ("IDW", "IDW"), "constant": ("costante", "constant"),
                "none": ("nessun dato", "no data")}
PALETTE_NAMES = (("", ("Del prodotto", "The product's own")), ("terrain", ("Ipsometrica", "Hypsometric")),
                 ("earth", ("Terra", "Earth")), ("thickness", ("Spessore", "Thickness")), ("grey", ("Grigi", "Greys")),
                 ("grey_inverted", ("Grigi invertiti", "Greys inverted")), ("heat", ("Calore", "Heat")),
                 ("diverging", ("Divergente", "Diverging")))


def _table(headers: list[str], multi: bool = False) -> QTableWidget:
    tw = QTableWidget(0, len(headers))
    tw.setHorizontalHeaderLabels(headers)
    tw.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    tw.horizontalHeader().setStretchLastSection(True)
    tw.verticalHeader().setVisible(False)
    tw.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    tw.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    tw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection if multi
                        else QAbstractItemView.SelectionMode.SingleSelection)
    tw.setAlternatingRowColors(True)
    return tw


def _combo(items, length: int = 8) -> QComboBox:
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(length)
    for value, (it, en) in items:
        c.addItem(t(it, en), value)
    return c


def _pair(label: str, widget: QWidget) -> QWidget:
    """A label and its widget, kept together on a line of a flow layout."""
    w = QWidget()
    row = FlowLayout(w)
    row.addWidget(QLabel(label))
    row.addWidget(widget)
    return w


class GeoarchaeologyPane(QWidget):
    import_requested = Signal()
    order_requested = Signal()
    surfaces_requested = Signal()
    model_requested = Signal()
    model_view_requested = Signal()  # the units of the model in the 3D view of the program
    model_folder_requested = Signal()  # the folder of the model in the file manager
    section_draw_requested = Signal()
    section_through_requested = Signal(list)  # the boreholes selected, in the order they were picked
    logs_requested = Signal(list)
    boreholes_layer_requested = Signal()
    overlay_requested = Signal(str, int, str, float)  # product id, band, palette ("" = its own), opacity
    opacity_changed = Signal(float)
    hide_requested = Signal()
    export_requested = Signal(str)  # a product id, or "" for the GeoPackage of everything
    open_requested = Signal(str)  # a section id: its drawing
    remove_requested = Signal(str, str)  # what ("product", "section", "boreholes"), its id
    qgis_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.has_model = False  # the chunk has the solids of a model of the units
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Sondaggi, carote e sezioni di scavo sul rilievo: colonne stratigrafiche, superfici di "
                         "contatto e spessori per kriging, sezioni, modello 3D delle unità con i volumi.",
                         "Boreholes, cores and trench logs on the survey: stratigraphic columns, contact surfaces and "
                         "thicknesses by kriging, sections, a 3D model of the units with their volumes."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_import = QPushButton(t("Importa sondaggi...", "Import boreholes..."))
        self.b_import.setToolTip(t("CSV dei collari e degli intervalli, o una tabella sola con entrambi",
                                   "CSV of the collars and of the intervals, or one table with both"))
        self.b_order = QPushButton(t("Ordine delle unità...", "Order of the units..."))
        self.b_surfaces = QPushButton(t("Superfici...", "Surfaces..."))
        self.b_surfaces.setToolTip(t("Tetti delle unità e spessori per kriging (IDW sotto i 5 sondaggi)",
                                     "Tops of the units and thicknesses by kriging (IDW under 5 boreholes)"))
        self.b_model = QPushButton(t("Modello 3D...", "3D model..."))
        self.b_model_view = QPushButton(t("Mostra nella vista 3D", "Show in the 3D view"))
        self.b_model_view.setToolTip(t("Le unità del modello 3D nella vista 3D del programma, ognuna nel suo colore",
                                       "The units of the 3D model in the program's 3D view, each in its colour"))
        self.b_model_folder = QPushButton(t("Apri la cartella", "Open the folder"))
        self.b_model_folder.setToolTip(t("La cartella del modello: un PLY per unità, l'OBJ, i volumi",
                                         "The folder of the model: a PLY per unit, the OBJ, the volumes"))
        for b in (self.b_import, self.b_order, self.b_surfaces, self.b_model, self.b_model_view, self.b_model_folder):
            bar.addWidget(b)
        lay.addWidget(holder)
        holder2 = QWidget()
        bar2 = FlowLayout(holder2)
        self.b_draw = QPushButton(t("Traccia una sezione", "Draw a section"))
        self.b_draw.setToolTip(t("Due clic sull'ortofoto o sul DEM: inizio e fine della sezione",
                                 "Two clicks on the orthomosaic or DEM: start and end of the section"))
        self.b_through = QPushButton(t("Sezione per i sondaggi", "Section through the boreholes"))
        self.b_through.setToolTip(t("Una sezione che passa per i sondaggi selezionati, nell'ordine in cui li hai scelti",
                                    "A section through the selected boreholes, in the order you picked them"))
        self.b_logs = QPushButton(t("Colonne stratigrafiche", "Stratigraphic columns"))
        self.b_layer = QPushButton(t("Sondaggi nella vista", "Boreholes in the view"))
        for b in (self.b_draw, self.b_through, self.b_logs, self.b_layer):
            bar2.addWidget(b)
        lay.addWidget(holder2)
        self.tabs = QTabWidget()
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.t_holes = _table([t("Sondaggio", "Borehole"), t("Quota", "Elevation"), t("Prof.", "Depth"),
                               t("Unità", "Units")], multi=True)
        self.t_products = _table([t("Prodotto", "Product"), t("Metodo", "Method"), t("Min", "Min"), t("Max", "Max")])
        self.t_sections = _table([t("Sezione", "Section"), t("Lunghezza", "Length"), t("Sondaggi", "Boreholes")])
        self.tabs.addTab(self.t_holes, t("Sondaggi", "Boreholes"))
        self.tabs.addTab(self.t_products, t("Superfici", "Surfaces"))
        self.tabs.addTab(self.t_sections, t("Sezioni", "Sections"))
        lay.addWidget(self.tabs, 1)
        holder3 = QWidget()
        row3 = FlowLayout(holder3)
        self.c_band = _combo(BANDS, 8)
        self.c_palette = _combo([(k, v) for k, v in PALETTE_NAMES if not k or k in PALETTES], 8)
        self.s_opacity = QSlider(Qt.Orientation.Horizontal)
        self.s_opacity.setRange(10, 100)
        self.s_opacity.setValue(70)
        self.s_opacity.setMinimumWidth(60)
        for label, w in ((t("Banda", "Band"), self.c_band), (t("Palette", "Palette"), self.c_palette),
                         (t("Opacità", "Opacity"), self.s_opacity)):
            row3.addWidget(_pair(label, w))
        lay.addWidget(holder3)
        holder4 = QWidget()
        bar4 = FlowLayout(holder4)
        self.b_overlay = QPushButton(t("Sovrapponi alla vista", "Overlay on the view"))
        self.b_hide = QPushButton(t("Nascondi", "Hide"))
        self.b_open = QPushButton(t("Apri il disegno", "Open the drawing"))
        self.b_export = QPushButton(t("Esporta...", "Export..."))
        self.b_export.setToolTip(t("La superficie scelta in GeoTIFF, oppure tutto in GeoPackage",
                                   "The chosen surface as GeoTIFF, or everything as GeoPackage"))
        self.b_qgis = QPushButton(t("Apri in QGIS", "Open in QGIS"))
        self.b_remove = QPushButton(t("Rimuovi", "Remove"))
        for b in (self.b_overlay, self.b_hide, self.b_open, self.b_export, self.b_qgis, self.b_remove):
            bar4.addWidget(b)
        lay.addWidget(holder4)
        self.l_limits = QLabel("")
        self.l_limits.setWordWrap(True)
        lay.addWidget(self.l_limits)
        self.b_import.clicked.connect(self.import_requested)
        self.b_order.clicked.connect(self.order_requested)
        self.b_surfaces.clicked.connect(self.surfaces_requested)
        self.b_model.clicked.connect(self.model_requested)
        self.b_model_view.clicked.connect(self.model_view_requested)
        self.b_model_folder.clicked.connect(self.model_folder_requested)
        self.b_draw.clicked.connect(self.section_draw_requested)
        self.b_through.clicked.connect(lambda: self.section_through_requested.emit(self.selected_boreholes()))
        self.b_logs.clicked.connect(lambda: self.logs_requested.emit(self.selected_boreholes()))
        self.b_layer.clicked.connect(self.boreholes_layer_requested)
        self.b_overlay.clicked.connect(self._overlay)
        self.b_hide.clicked.connect(self.hide_requested)
        self.b_open.clicked.connect(lambda: self.current_section() and self.open_requested.emit(self.current_section()))
        self.b_export.clicked.connect(self._export)
        self.b_qgis.clicked.connect(self.qgis_requested)
        self.b_remove.clicked.connect(self._remove)
        self.s_opacity.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100.0))
        self._picked: list[str] = []  # the boreholes selected, in the order they were picked
        self.t_holes.itemSelectionChanged.connect(self._track_picks)
        self.t_products.itemDoubleClicked.connect(lambda _it: self._overlay())
        self.t_sections.itemDoubleClicked.connect(lambda _it: self.open_requested.emit(self.current_section() or ""))
        for tw in (self.t_holes, self.t_products, self.t_sections):
            tw.itemSelectionChanged.connect(self._update_buttons)
        self.tabs.currentChanged.connect(lambda _i: self._update_buttons())
        self.products: list[dict] = []
        self.sections: list[dict] = []
        self.set_state([], [], [], "")

    # --------------------------------------------------------------- selections
    def _selected_in_table(self) -> list[str]:
        rows = sorted({i.row() for i in self.t_holes.selectedIndexes()})
        return [self.t_holes.item(r, 0).text() for r in rows if self.t_holes.item(r, 0) is not None]

    def _track_picks(self) -> None:
        """The order of the picks: those still selected, then the new ones (in the order of the
        table when several come at once, as with Shift)."""
        now = self._selected_in_table()
        self._picked = [i for i in self._picked if i in now] + [i for i in now if i not in self._picked]

    def selected_boreholes(self) -> list[str]:
        """The boreholes selected, in the order they were picked."""
        now = self._selected_in_table()
        picked = [i for i in self._picked if i in now]
        return picked + [i for i in now if i not in picked]

    def current_product(self) -> str | None:
        rows = sorted({i.row() for i in self.t_products.selectedIndexes()})
        return self.products[rows[0]]["id"] if rows and rows[0] < len(self.products) else None

    def current_section(self) -> str | None:
        rows = sorted({i.row() for i in self.t_sections.selectedIndexes()})
        return self.sections[rows[0]]["id"] if rows and rows[0] < len(self.sections) else None

    def _overlay(self) -> None:
        pid = self.current_product()
        if pid:
            self.overlay_requested.emit(pid, int(self.c_band.currentData() or 1), str(self.c_palette.currentData() or ""),
                                        self.s_opacity.value() / 100.0)

    def _export(self) -> None:
        on_products = self.tabs.currentWidget() is self.t_products
        self.export_requested.emit((self.current_product() or "") if on_products else "")

    def _remove(self) -> None:
        tab = self.tabs.currentWidget()
        if tab is self.t_products and self.current_product():
            self.remove_requested.emit("product", self.current_product())
        elif tab is self.t_sections and self.current_section():
            self.remove_requested.emit("section", self.current_section())
        elif tab is self.t_holes:
            self.remove_requested.emit("boreholes", ",".join(self.selected_boreholes()))

    def _update_buttons(self) -> None:
        has_holes = self.t_holes.rowCount() > 0
        product = self.current_product()
        entry = next((p for p in self.products if p["id"] == product), None)
        for b in (self.b_order, self.b_surfaces, self.b_draw, self.b_logs, self.b_layer):
            b.setEnabled(has_holes)
        self.b_through.setEnabled(len(self.selected_boreholes()) >= 2)
        self.b_model.setEnabled(bool(self.products))
        for b in (self.b_model_view, self.b_model_folder):
            b.setEnabled(self.has_model)
        self.b_overlay.setEnabled(product is not None)
        self.c_band.setEnabled(entry is not None and entry.get("kind") == "surface")
        self.b_open.setEnabled(self.current_section() is not None)
        self.b_export.setEnabled(has_holes)
        self.b_qgis.setEnabled(has_holes)
        tab = self.tabs.currentWidget()
        self.b_remove.setEnabled((tab is self.t_products and product is not None)
                                 or (tab is self.t_sections and self.current_section() is not None)
                                 or (tab is self.t_holes and bool(self.selected_boreholes())))

    # --------------------------------------------------------------- state
    def set_status(self, text: str) -> None:
        self.l_status.setText(text)

    def set_limits(self, text: str) -> None:
        self.l_limits.setText(text)

    @staticmethod
    def _tooltip(e: dict) -> list[str]:
        """The lines of the tooltip of a product: its id, how it was made, what the stack did."""
        from . import GeoarchaeologyPlugin

        tip = [e["id"]]
        if e.get("loocv_rmse") is not None:
            tip.append(t("errore leave-one-out {0} m", "leave-one-out error {0} m").format(num(e["loocv_rmse"], 3)))
        mode = GeoarchaeologyPlugin.mode_text(e)
        if mode:
            tip.append(mode)
        cl = e.get("clamped") or {}
        if cl.get("cells"):
            tip.append(t("corretta dalla pila su {0} celle (abbassata fino a {1} m, alzata fino a {2} m)",
                         "moved by the stack on {0} cells (lowered up to {1} m, raised up to {2} m)")
                       .format(cl["cells"], num(cl.get("lowered_max_m", cl.get("max_m", 0.0)), 2),
                               num(cl.get("raised_max_m", 0.0), 2)))
        if e.get("kind") == "thickness":
            tip.append(t("volume {0} m³", "volume {0} m³").format(num(e.get("volume_m3", 0.0), 0)))
            if e.get("outside_dem"):
                tip.append(t("più {0} m³ dove manca il DEM", "plus {0} m³ where there is no DEM")
                           .format(num(e["outside_dem"]["volume_m3"], 0)))
            if e.get("absent"):
                tip.append(t("assente su {0} m²", "missing on {0} m²").format(num(e["absent"]["area_m2"], 0)))
        return tip

    def set_state(self, holes: list, products: list[dict], sections: list[dict], status: str,
                  has_model: bool = False) -> None:
        """Fill the tables: ``holes`` (``Borehole``), ``products`` (surfaces and thicknesses),
        ``sections``; keep the selections when the rows are still there. ``has_model``: the chunk
        has a model of the units, to show or to open."""
        self.has_model = bool(has_model)
        picked = self.selected_boreholes()
        keep_holes = set(picked)
        keep_product, keep_section = self.current_product(), self.current_section()
        self.t_holes.setRowCount(len(holes))
        for r, b in enumerate(holes):
            vals = [b.id, "" if b.z is None else num(b.z, 2) + (" *" if b.z_source == "dem" else ""),
                    num(b.bottom, 2), " / ".join(b.units)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 1 and b.z_source == "dem":
                    it.setToolTip(t("quota letta dal DEM", "elevation read from the DEM"))
                self.t_holes.setItem(r, c, it)
            if b.id in keep_holes:
                index = self.t_holes.model().index(r, 0)
                self.t_holes.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select
                                                     | QItemSelectionModel.SelectionFlag.Rows)
        now = self._selected_in_table()
        self._picked = [i for i in picked if i in now] + [i for i in now if i not in picked]
        self.products = list(products)
        self.t_products.setRowCount(len(self.products))
        for r, e in enumerate(self.products):
            st = e.get("stats") or {}
            if e.get("kind") != "surface":
                method = t("spessore", "thickness")
            elif e.get("method") == "kriging":
                method = t(*MODEL_NAMES.get(e["variogram"]["model"], (e["variogram"]["model"],) * 2))
                if e.get("trend") == "linear":
                    method += t(" + piano", " + plane")
            else:
                method = t(*METHOD_NAMES.get(e.get("method", ""), (e.get("method", ""),) * 2))
            if e.get("kind") == "surface" and e.get("mode") in ("elevation", "depth"):
                method += ", " + (t("quota", "elevation") if e["mode"] == "elevation" else t("profondità", "depth"))
            vals = [e.get("name", e["id"]), method, num(st["min"], 2) if "min" in st else "",
                    num(st["max"], 2) if "max" in st else ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (0, 1):
                    it.setToolTip("\n".join(self._tooltip(e)))
                self.t_products.setItem(r, c, it)
            if e["id"] == keep_product:
                self.t_products.selectRow(r)
        if self.products and self.current_product() is None:
            self.t_products.selectRow(0)
        self.sections = list(sections)
        self.t_sections.setRowCount(len(self.sections))
        for r, s in enumerate(self.sections):
            vals = [s["name"], num(s["length"], 1) + " m", ", ".join(h["id"] for h in s.get("boreholes", []))]
            for c, v in enumerate(vals):
                self.t_sections.setItem(r, c, QTableWidgetItem(v))
            if s["id"] == keep_section:
                self.t_sections.selectRow(r)
        self.set_status(status)
        self._update_buttons()
