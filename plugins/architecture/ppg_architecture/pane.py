"""The *Elementi architettonici* pane: where CadQuery runs (with the button to install it), the buttons that make an
element, the list of the elements with their fit, and what can be done with the chosen ones. It only says what the
user asks (``requested``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import arch_title, kind_title, metres, t

MAKE = (("column", ("Colonna...", "Column..."), "Una colonna o una sua parte (base, fusto, capitello) per rivoluzione: "
         "disegna la zona in pianta, l'asse e il profilo si misurano sul modello",
         "A column or a part of it (base, shaft, capital) by revolution: draw the region in plan, the axis and the "
         "profile are measured on the model"),
        ("moulding", ("Modanatura...", "Moulding..."), "Cornice, modanatura, cornice marcapiano: traccia il percorso, "
         "la sezione si taglia dal modello", "Cornice, moulding, string course: trace the path, the section is cut "
         "from the model"),
        ("arch", ("Arco / volta...", "Arch / vault..."), "Un arco o una volta a botte da una sezione del progetto o da "
         "un piano per due punti", "An arch or a barrel vault from a section of the project or a plane through two "
         "points"),
        ("opening", ("Porta / finestra...", "Door / window..."), "Il vano di una porta o di una finestra: clic dentro "
         "l'apertura sulla pianta, stipiti, facce del muro, soglia e architrave si misurano sul modello",
         "The void of a door or a window: click inside the opening on the plan, the reveals, the wall faces, the sill "
         "and the lintel are measured on the model"))
ACTIONS = (("show", ("Mostra nella vista 3D", "Show in the 3D view")),
           ("edit", ("Modifica parametri...", "Edit parameters...")),
           ("step", ("Esporta STEP...", "Export STEP...")),
           ("ifc", ("Esporta IFC...", "Export IFC...")),
           ("dxf", ("DXF...", "DXF...")),
           ("folder", ("Apri la cartella", "Open the folder")),
           ("rebuild", ("Ricostruisci", "Rebuild")),
           ("remove", ("Rimuovi", "Remove")))
COLUMNS = (("Nome", "Name"), ("Tipo", "Kind"), ("Misure", "Measures"), ("Scarto mediana", "Fit median"),
           ("Scarto p90", "Fit p90"), ("Punti", "Points"))


class ArchitecturePane(QWidget):
    requested = Signal(str)  # "install", "column", "moulding", "arch", "opening", or an action of ACTIONS

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Il solido è un'interpretazione del modello misurato: lo scarto dice quanto il modello se ne "
                         "discosta.", "The solid is an interpretation of the measured model: the fit says how far the "
                         "model is from it."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel(f"<b>{t('CadQuery', 'CadQuery')}</b>"))
        self.l_env = QLabel("")
        self.l_env.setWordWrap(True)
        lay.addWidget(self.l_env)
        self.b_install = QPushButton(t("Installa CadQuery", "Install CadQuery"))
        self.b_install.clicked.connect(lambda: self.requested.emit("install"))
        lay.addWidget(self.b_install)
        lay.addWidget(QLabel(f"<b>{t('Nuovo elemento', 'New element')}</b>"))
        holder = QWidget()
        flow = FlowLayout(holder)
        self.buttons: dict[str, QPushButton] = {}
        for name, text, tip_it, tip_en in MAKE:
            b = QPushButton(t(*text))
            b.setToolTip(t(tip_it, tip_en))
            b.clicked.connect(lambda _=False, n=name: self.requested.emit(n))
            flow.addWidget(b)
            self.buttons[name] = b
        lay.addWidget(holder)
        lay.addWidget(QLabel(f"<b>{t('Elementi', 'Elements')}</b>"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(lambda _i: self.requested.emit("show"))
        lay.addWidget(self.table)
        holder = QWidget()
        flow = FlowLayout(holder)
        for name, text in ACTIONS:
            b = QPushButton(t(*text))
            b.clicked.connect(lambda _=False, n=name: self.requested.emit(n))
            flow.addWidget(b)
            self.buttons[name] = b
        lay.addWidget(holder)
        self.l_state = QLabel("")
        self.l_state.setWordWrap(True)
        lay.addWidget(self.l_state)
        lay.addStretch(1)
        self._ids: list[str] = []

    def set_environment(self, text: str, can_install: bool, can_make: bool) -> None:
        self.l_env.setText(text)
        self.b_install.setVisible(can_install)
        for name, *_rest in MAKE:
            self.buttons[name].setEnabled(can_make)

    def set_rows(self, entries: list[dict]) -> None:
        keep = set(self.selected_ids())
        self._ids = [e["id"] for e in entries]
        self.table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            p, fit = e.get("params") or {}, e.get("fit") or {}
            if e["kind"] == "column":
                measures = f"h {metres(p.get('height'))}, Ø {metres(2 * p.get('radius_base', 0))}"
            elif e["kind"] == "moulding" and p.get("closed_member"):
                size = f"{metres(p.get('width'))} × {metres(p.get('section_height'))}"
                measures = f"L {metres(p.get('length'))}, {size}"
            elif e["kind"] == "opening":
                from .texts import kind_title as title

                size = f"{metres(p.get('width'))} × {metres(p.get('height'))}"
                measures = f"{title(p.get('opening_type', 'opening'))} {size}"
            elif e["kind"] == "moulding":
                measures = t(f"L {metres(p.get('length'))}, sporgenza {metres(p.get('projection'))}",
                             f"L {metres(p.get('length'))}, projection {metres(p.get('projection'))}")
            else:
                from .elements import candidates_text

                measures = t(f"{arch_title(p.get('type', ''))}, luce {metres(p.get('span'))} ({candidates_text(p)})",
                             f"{arch_title(p.get('type', ''))}, span {metres(p.get('span'))} ({candidates_text(p)})")
            if e.get("stale"):
                measures += t(" - da ricostruire", " - to be rebuilt")
            if e.get("warnings"):
                measures += " ⚠"
            cells = (e.get("name", ""), kind_title(e["kind"]), measures, metres(fit.get("median")),
                     metres(fit.get("p90")), str(fit.get("points") or 0))
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if e.get("warnings"):
                    item.setToolTip(" ".join(e["warnings"]))
                if j >= 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, e["id"])
                self.table.setItem(i, j, item)
            if e["id"] in keep:
                self.table.selectRow(i)
        self.l_state.setText("" if entries else t("Nessun elemento nel chunk.", "No element in the chunk."))

    def set_state(self, text: str) -> None:
        self.l_state.setText(text)

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if self.table.item(r, 0)]

    def select(self, eid: str) -> None:
        if eid in self._ids:
            self.table.selectRow(self._ids.index(eid))
