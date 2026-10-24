"""The Mesh (MeshLab) pane: where MeshLab runs (and its installation), the active model, the
operations as buttons (their parameters asked in a dialog), the measures of the model."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import t

BUTTONS = (
    ("decimate", ("Decima...", "Decimate..."),
     ("Semplificazione per collasso degli spigoli (quadriche), con la texture conservata",
      "Quadric edge collapse simplification, keeping the texture")),
    ("remesh", ("Rimaglia...", "Remesh..."),
     ("Triangoli regolari della stessa dimensione (rimaglia isotropa esplicita)",
      "Regular triangles of one size (isotropic explicit remeshing)")),
    ("smooth", ("Leviga...", "Smooth..."),
     ("Taubin (non restringe il modello) o HC laplaciana; i bordi possono restare fermi",
      "Taubin (does not shrink the model) or HC Laplacian; the borders can stay where they are")),
    ("close_holes", ("Chiudi i buchi...", "Close holes..."),
     ("I buchi fino a un numero di lati chiusi con nuove facce", "The holes up to a number of edges closed with new faces")),
    ("repair", ("Ripara la topologia...", "Repair the topology..."),
     ("Vertici e facce doppi, facce senza area, spigoli e vertici non-manifold, vertici non usati",
      "Duplicate vertices and faces, faces without area, non-manifold edges and vertices, unused vertices")),
    ("measure", ("Misura", "Measure"),
     ("Area, volume (modelli chiusi), parti, bordi, difetti di topologia, dimensioni",
      "Area, volume (closed models), parts, borders, topology defects, size")),
)


class MeshToolsPane(QWidget):
    operation_requested = Signal(str)  # decimate, remesh, smooth, close_holes, repair, measure
    install_requested = Signal()
    show_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("I filtri di MeshLab sul modello del chunk: ogni risultato è una nuova versione del modello "
                         "(quella di prima resta tra le versioni). Le lunghezze sono in metri del rilievo.",
                         "MeshLab's filters on the model of the chunk: every result is a new version of the model "
                         "(the one before stays among the versions). Lengths are metres of the survey."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        holder = QWidget()
        row = FlowLayout(holder)
        self.b_install = QPushButton(t("Installa MeshLab...", "Install MeshLab..."))
        self.b_install.setToolTip(t("Python 3.12 e pymeshlab da PyPI in un ambiente del plugin, con uv (circa 300 MB)",
                                    "Python 3.12 and pymeshlab from PyPI in an environment of the plugin, with uv "
                                    "(about 300 MB)"))
        row.addWidget(self.b_install)
        lay.addWidget(holder)
        self.l_model = QLabel("")
        self.l_model.setWordWrap(True)
        lay.addWidget(self.l_model)
        # the buttons go onto more lines in a narrow dock: the pane never widens the column of the window
        holder2 = QWidget()
        bar = FlowLayout(holder2)
        self.buttons: dict[str, QPushButton] = {}
        for op, text, tip in BUTTONS:
            b = QPushButton(t(*text))
            b.setToolTip(t(*tip))
            b.clicked.connect(lambda _checked=False, op=op: self.operation_requested.emit(op))
            bar.addWidget(b)
            self.buttons[op] = b
        self.b_show = QPushButton(t("Mostra nella vista 3D", "Show in the 3D view"))
        self.b_show.clicked.connect(self.show_requested)
        bar.addWidget(self.b_show)
        lay.addWidget(holder2)
        self.l_measures = QLabel(t("Misure", "Measures"))
        lay.addWidget(self.l_measures)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([t("Misura", "Measure"), t("Valore", "Value")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        self.l_last = QLabel("")
        self.l_last.setWordWrap(True)
        lay.addWidget(self.l_last)
        self.b_install.clicked.connect(self.install_requested)
        self._holders = (holder, holder2)
        self.set_model(None)

    def set_busy(self, busy: bool) -> None:
        """While a step of the program runs, the buttons wait (one step at a time)."""
        for h in self._holders:
            h.setEnabled(not busy)

    def set_status(self, text: str, can_install: bool) -> None:
        self.l_status.setText(text)
        self.b_install.setVisible(can_install)

    def set_model(self, text: str | None, ready: bool = True) -> None:
        """The active model in a line (None: no model), the buttons enabled when there is one and
        MeshLab can run."""
        self.l_model.setText(text or t("Il chunk non ha un modello: costruiscilo prima (Flusso di lavoro > Costruisci "
                                       "modello).",
                                       "The chunk has no model: build it first (Workflow > Build model)."))
        for b in self.buttons.values():
            b.setEnabled(bool(text) and ready)
        self.b_show.setEnabled(bool(text))

    def set_measures(self, rows: list[tuple[str, str]], caption: str = "") -> None:
        self.l_measures.setText(caption or t("Misure", "Measures"))
        self.table.setRowCount(0)
        for name, value in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(name)))
            self.table.setItem(r, 1, QTableWidgetItem(str(value)))

    def set_last(self, text: str) -> None:
        self.l_last.setText(text)
