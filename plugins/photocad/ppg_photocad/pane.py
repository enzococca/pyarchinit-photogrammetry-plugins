"""The *PhotoCAD* pane: the account, the buttons that vectorise the orthophoto, the DEM or the elevation shown, and
the table of the conversions of the chunk with the actions on the one selected. It only says what the user asks
(``requested(action, arg)``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import t

COLUMNS = (("Conversione", "Conversion"), ("Sorgente", "Source"), ("Stato", "Status"), ("Pietre", "Stones"))


def _status_text(rec: dict) -> str:
    from . import project

    st = rec.get("status", "done")
    if st in project.STATUS:
        return t(*project.STATUS[st])
    return t("finita", "done")


class PhotoCADPane(QWidget):
    requested = Signal(str, object)  # (action, argument): the id of the conversion for the row actions

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.buttons: dict[str, QPushButton] = {}

        lay.addWidget(QLabel(f"<b>{t('Account', 'Account')}</b>"))
        self.l_key = QLabel("")
        self.l_key.setWordWrap(True)
        self.l_key.setProperty("ppg_no_translate", True)
        lay.addWidget(self.l_key)
        lay.addWidget(self._flow((("key", t("Chiave API...", "API key..."), t("La chiave API di PhotoCAD", "The PhotoCAD API key")),
                                  ("credit", t("Credito", "Credit"), t("Chiede a PhotoCAD i crediti rimasti (rete)",
                                                                       "Asks PhotoCAD for the credits left (network)")))))

        lay.addWidget(QLabel(f"<b>{t('Vettorializza', 'Vectorise')}</b>"))
        lay.addWidget(self._flow((("ortho", t("Ortofoto...", "Orthophoto..."), t("Invia l'ortofoto (o una parte)", "Send the orthophoto (or a part)")),
                                  ("dem", t("DEM...", "DEM..."), t("Il DEM come ombreggiatura", "The DEM as a hillshade")),
                                  ("elevation", t("Prospetto mostrato...", "Elevation shown..."),
                                   t("Il prospetto o la sezione mostrati nella vista 2D", "The elevation or section shown in the 2D view")))))
        self.l_state = QLabel("")
        self.l_state.setWordWrap(True)
        lay.addWidget(self.l_state)

        lay.addWidget(QLabel(f"<b>{t('Conversioni', 'Conversions')}</b>"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(120)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        lay.addWidget(self.table)
        self.l_empty = QLabel(t("Nessuna conversione in questo chunk: vettorializza l'ortofoto, il DEM o un prospetto.",
                                "No conversion in this chunk: vectorise the orthophoto, the DEM or an elevation."))
        self.l_empty.setWordWrap(True)
        lay.addWidget(self.l_empty)
        lay.addWidget(self._flow((("toggle", t("Mostra/Nascondi", "Show/Hide"), t("Mostra o nasconde i layer della conversione scelta", "Shows or hides the layers of the conversion chosen")),
                                  ("fetch", t("Recupera...", "Fetch..."), t("Scarica di nuovo la conversione scelta (già pagata)", "Download the conversion chosen again (already paid)")),
                                  ("export", t("Esporta...", "Export..."), "DXF, GeoPackage, SVG"),
                                  ("remove", t("Rimuovi", "Remove"), t("Toglie la conversione scelta dal progetto, con i suoi file", "Removes the conversion chosen from the project, with its files")),
                                  ("units", t("Converti in US / USM...", "Turn into US / USM..."), t("I poligoni diventano unità stratigrafiche", "The polygons become stratigraphic units")),
                                  ("free", t("Disegni liberi...", "Free drawings..."), t("Importa la conversione scelta come disegni liberi", "Import the conversion chosen as free drawings")))))
        lay.addStretch(1)
        self.set_state()

    def _flow(self, specs) -> QWidget:
        box = QWidget()
        flow = FlowLayout(box)
        flow.setContentsMargins(0, 0, 0, 0)
        for name, label, tip in specs:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, n=name: self._click(n))
            self.buttons[name] = b
            flow.addWidget(b)
        return box

    # ------------------------------------------------------------------ state
    def record_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _click(self, name: str) -> None:
        if name in ("toggle", "fetch", "export", "remove", "free"):
            rid = self.record_id()
            if rid is None:
                return
            self.requested.emit(name, rid)
        else:
            self.requested.emit(name, None)

    def _selection_changed(self) -> None:
        has = self.record_id() is not None
        for n in ("toggle", "fetch", "export", "remove", "units", "free"):
            if n == "units":
                self.buttons[n].setEnabled(self._units and (has or bool(self._records)))
            elif n == "free":
                self.buttons[n].setEnabled(has and self._free)
            else:
                self.buttons[n].setEnabled(has)
        self.buttons["free"].setVisible(self._free)

    def set_state(self, key_text: str = "", ortho: bool = False, dem: bool = False, elevation: bool = False,
                  records: list | None = None, frame=None, free: bool = False, units: bool = True) -> None:
        self._free, self._units = free, units
        selected = self.record_id()
        self.l_key.setText(key_text)
        self.buttons["ortho"].setEnabled(ortho)
        self.buttons["dem"].setEnabled(dem)
        self.buttons["elevation"].setEnabled(elevation)
        self.l_state.setText("" if (ortho or dem or elevation) else t(
            "Il chunk non ha ortofoto né DEM, e nessun prospetto è mostrato.",
            "The chunk has no orthophoto or DEM, and no elevation is shown."))
        self._records = list(records or [])
        # the rows of the frame shown first
        order = sorted(self._records, key=lambda r: r.get("frame") != frame)
        self.table.blockSignals(True)
        self.table.setRowCount(len(order))
        for row, rec in enumerate(order):
            c = rec.get("counts") or {}
            here = rec.get("frame") == frame
            visible = rec.get("visible", True)
            name = str(rec.get("name") or rec.get("id")) + ("" if visible else t(" (nascosta)", " (hidden)"))
            if here:
                name = "● " + name
            src = t("pianta", "plan") if rec.get("frame") is None else t("prospetto {0}", "elevation {0}").format(rec["frame"])
            cells = (name, src, _status_text(rec),
                     "" if c.get("stones") is None else str(c["stones"]))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, rec["id"])
                item.setToolTip(rec.get("_summary") or text)
                self.table.setItem(row, col, item)
            if rec["id"] == selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.table.setVisible(bool(order))
        self.l_empty.setVisible(not order)
        self._selection_changed()
