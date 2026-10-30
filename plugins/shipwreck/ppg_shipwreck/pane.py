"""The *Relitto* pane: the steps of the drawing of a hull, one under the other, each with its buttons and its
state, and the table of the timbers. It only says what the user asks (``requested``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from . import timbers
from .texts import num, t

STEPS = (("frame", ("1. Sistema della nave", "1. Ship frame"),
          (("frame_pick", ("Traccia la chiglia...", "Trace the keel..."), "x da dove inizi a tracciare (la prua, se non dici altro)",
            "x from where you start tracing (the bow, unless you say otherwise)"),
           ("frame_markers", ("Da due marker...", "From two markers..."), "", ""),
           ("frame_options", ("Prua, origine, datum...", "Bow, origin, datum..."), "", ""),
           ("frame_flip", ("Scambia gli estremi", "Swap the ends"), "", ""))),
         ("stations", ("2. Stazioni", "2. Stations"),
          (("stations_regular", ("Regolari...", "Regular..."), "", ""),
           ("stations_table", ("Da tabella...", "From a table..."), "nome, x: i nomi sono quelli del cantiere", "name, x: the names are the field's"),
           ("stations_catalogue", ("Dalle ordinate del catalogo", "From the catalogue's frames"), "", ""))),
         ("sections", ("3. Sezioni", "3. Sections"), (("sections", ("Calcola...", "Compute..."), "", ""),)),
         ("water", ("4. Acqua", "4. Water"), (("water", ("Livello e rifrazione...", "Level and refraction..."), "", ""),
                                              ("water_pick", ("Clicca il livello...", "Click the level..."), "", ""))),
         ("drawings", ("5. Disegni", "5. Drawings"), (("drawings", ("Genera...", "Make..."), "", ""),
                                                      ("drawings_open", ("Apri la cartella", "Open the folder"), "", ""))),
         ("timbers", ("6. Catalogo dei legni", "6. Timber catalogue"),
          (("timber_line", ("Linea...", "Line..."), "", ""), ("timber_polygon", ("Poligono...", "Polygon..."), "", ""),
           ("timber_point", ("Punto...", "Point..."), "", ""), ("timber_import", ("Da CSV...", "From CSV..."), "", ""),
           ("timber_remove", ("Rimuovi selezionati", "Remove selected"), "", ""),
           ("timber_export", ("Esporta...", "Export..."), "CSV, XLSX o GeoPackage", "CSV, XLSX or GeoPackage"))))
COLUMNS = (("Id", "Id"), ("Tipo", "Kind"), ("Lato", "Side"), ("Nome", "Name"), ("x (m)", "x (m)"), ("Lungh. (m)", "Length (m)"),
           ("Largh. (m)", "Width (m)"), ("z max (m)", "z max (m)"))


class ShipwreckPane(QWidget):
    requested = Signal(str)  # the name of a step's button

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Il disegno tecnico di uno scafo scavato, misurato sul modello 3D: niente di ciò che manca viene "
                         "ricostruito.", "The technical drawing of an excavated hull, measured on the 3D model: nothing "
                         "missing is reconstructed."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.states: dict[str, QLabel] = {}
        self.buttons: dict[str, QPushButton] = {}
        for key, title, buttons in STEPS:
            head = QLabel(f"<b>{t(*title)}</b>")
            lay.addWidget(head)
            holder = QWidget()
            flow = FlowLayout(holder)
            for name, text, tip_it, tip_en in buttons:
                b = QPushButton(t(*text))
                if tip_it:
                    b.setToolTip(t(tip_it, tip_en))
                b.clicked.connect(lambda _=False, n=name: self.requested.emit(n))
                flow.addWidget(b)
                self.buttons[name] = b
            lay.addWidget(holder)
            state = QLabel("")
            state.setWordWrap(True)
            lay.addWidget(state)
            self.states[key] = state
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self.table)
        lay.addStretch(1)

    def set_states(self, states: dict) -> None:
        for key, text in states.items():
            self.states[key].setText(text)

    def set_rows(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            def value(v):
                return "" if v is None else num(v, 2)

            cells = (r["id"], timbers.kind_title(r["kind"]), timbers.side_title(r["side"]), r["name"], value(r["x"]),
                     value(r["length"]), value(r["width"]), value(r["z_max"]))
            for j, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if j >= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).text() for r in rows]
