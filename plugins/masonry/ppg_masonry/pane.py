"""The *Murature* pane: the walls of the chunk, the faces of the chosen wall with their size, resolution and drawings,
the buttons that make a face and measure its shape, and the summary of the chosen face. It only says what the user
asks (``requested``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import kind_title, metres, num, surface_title, t

WALL_ACTIONS = (("wall", ("Nuovo muro...", "New wall..."), "Un muro, un edificio o una torre: le sue facce sono "
                 "prospetti del progetto", "A wall, a building or a tower: its faces are elevations of the project"),
                ("rename", ("Rinomina...", "Rename..."), "Nome e tipo del muro", "Name and kind of the wall"),
                ("remove_wall", ("Rimuovi muro", "Remove wall"), "Il muro e i file del plugin; i prospetti a scelta",
                 "The wall and the plugin's files; the elevations if you wish"))
FACE_MAKE = (("plane", ("Faccia su un piano...", "Face on a plane..."), "Traccia in pianta la faccia del muro con due "
              "punti: un prospetto verticale fra i due punti, v = quota", "Trace the face of the wall in plan with two "
              "points: a vertical elevation between them, v = height"),
             ("cylinder", ("Faccia su un cilindro...", "Face on a cylinder..."), "Torre tonda, pozzo, abside: il "
              "cilindro da tre punti in pianta o dal cerchio della sezione a una quota", "Round tower, well, apse: the "
              "cylinder from three points in plan or from the circle of the slice at a height"),
             ("polygon", ("Torre poligonale...", "Polygonal tower..."), "Una faccia per lato dalla sezione del modello "
              "a una quota", "One face per side from the slice of the model at a height"))
FACE_ACTIONS = (("show", ("Mostra", "Show")),
                ("shape", ("Forma", "Shape")),
                ("bulge", ("Spanciamento sì/no", "Bulge on/off")),
                ("sections", ("Sezioni...", "Sections...")),
                ("axis", ("Asse della torre...", "Axis of the tower...")),
                ("compare", ("Confronta rilievi...", "Compare surveys...")),
                ("folder", ("Apri la cartella", "Open the folder")),
                ("remove_face", ("Rimuovi faccia", "Remove face")))
COLUMNS = (("Faccia", "Face"), ("Superficie", "Surface"), ("Larghezza", "Width"), ("Altezza", "Height"),
           ("Risoluzione", "Resolution"), ("Disegni", "Drawings"))


def drawings_text(d: dict) -> str:
    parts = []
    for key, it, en in (("stones", "pietre", "stones"), ("units", "US", "units"), ("quote", "quote", "spot heights"),
                        ("points", "punti", "points")):
        if d.get(key):
            parts.append(f"{d[key]} {t(it, en)}")
    return ", ".join(parts) or "-"


class MasonryPane(QWidget):
    requested = Signal(str)  # an action of WALL_ACTIONS, FACE_MAKE or FACE_ACTIONS
    wall_changed = Signal(str)
    face_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Un muro si rileva per facce: ogni faccia è un prospetto del progetto. La forma si misura sul "
                         "modello: le misure dicono come sta il muro, non lo giudicano.",
                         "A wall is surveyed by its faces: each face is an elevation of the project. The shape is "
                         "measured on the model: the measures say how the wall stands, they do not judge it."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel(f"<b>{t('Muri', 'Walls')}</b>"))
        self.walls = QComboBox()
        self.walls.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.walls.setMinimumContentsLength(12)
        self.walls.setProperty("ppg_no_translate", True)
        self.walls.currentIndexChanged.connect(lambda _i: self.wall_changed.emit(self.wall_id() or ""))
        lay.addWidget(self.walls)
        self.buttons: dict[str, QPushButton] = {}
        lay.addWidget(self._flow(WALL_ACTIONS))
        lay.addWidget(QLabel(f"<b>{t('Nuova faccia', 'New face')}</b>"))
        lay.addWidget(self._flow(FACE_MAKE))
        lay.addWidget(QLabel(f"<b>{t('Facce', 'Faces')}</b>"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemDoubleClicked.connect(lambda _i: self.requested.emit("show"))
        self.table.itemSelectionChanged.connect(lambda: self.face_changed.emit(self.face_id() or ""))
        lay.addWidget(self.table)
        lay.addWidget(self._flow([(n, txt, "", "") for n, txt in FACE_ACTIONS]))
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(90)
        lay.addWidget(self.summary)
        self.l_state = QLabel("")
        self.l_state.setWordWrap(True)
        lay.addWidget(self.l_state)
        lay.addStretch(1)
        self._wall_ids: list[str] = []
        self._face_ids: list[str] = []

    def _flow(self, specs) -> QWidget:
        holder = QWidget()
        flow = FlowLayout(holder)
        for name, text, tip_it, tip_en in specs:
            b = QPushButton(t(*text))
            if tip_it:
                b.setToolTip(t(tip_it, tip_en))
            b.clicked.connect(lambda _=False, n=name: self.requested.emit(n))
            flow.addWidget(b)
            self.buttons[name] = b
        return holder

    # ------------------------------------------------------------------ walls
    def set_walls(self, records: list[dict], current: str | None = None) -> None:
        keep = current or self.wall_id()
        self.walls.blockSignals(True)
        self.walls.clear()
        self._wall_ids = [w["id"] for w in records]
        for w in records:
            n = len(w["faces"])
            noun = t("facce", "faces") if n != 1 else t("faccia", "face")
            self.walls.addItem(f"{w['name']} ({kind_title(w['kind']).lower()}, {n} {noun})", w["id"])
        if keep in self._wall_ids:
            self.walls.setCurrentIndex(self._wall_ids.index(keep))
        self.walls.blockSignals(False)
        for name in ("rename", "remove_wall"):
            self.buttons[name].setEnabled(bool(records))
        self.l_state.setText("" if records else t("Nessun muro nel chunk: Nuovo muro..., o fai una faccia (il muro si "
                                                  "crea da sé).", "No wall in the chunk: New wall..., or make a face "
                                                  "(the wall is made by itself)."))

    def wall_id(self) -> str | None:
        i = self.walls.currentIndex()
        return self._wall_ids[i] if 0 <= i < len(self._wall_ids) else None

    # ------------------------------------------------------------------ faces
    def set_faces(self, rows: list[dict]) -> None:
        keep = self.face_id()
        self.table.blockSignals(True)
        self._face_ids = [r["id"] for r in rows]
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            w, h = r["width"], r["height"]
            cells = (r["name"], surface_title(r["surface"]), metres(w), metres(h),
                     f"{num(r['resolution'] * 1000, 1)} mm/px", drawings_text(r["drawings"]))
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, r["id"])
                if 2 <= j <= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        if keep in self._face_ids:
            self.table.selectRow(self._face_ids.index(keep))
        elif rows:
            self.table.selectRow(0)
        self.table.blockSignals(False)
        for name, _text in FACE_ACTIONS:
            self.buttons[name].setEnabled(bool(rows))

    def face_id(self) -> str | None:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows or self.table.item(rows[0], 0) is None:
            return None
        return self.table.item(rows[0], 0).data(Qt.ItemDataRole.UserRole)

    def select_face(self, eid: str) -> None:
        if eid in self._face_ids:
            self.table.selectRow(self._face_ids.index(eid))

    def set_summary(self, text: str) -> None:
        self.summary.setPlainText(text)

    def set_state(self, text: str) -> None:
        self.l_state.setText(text)
