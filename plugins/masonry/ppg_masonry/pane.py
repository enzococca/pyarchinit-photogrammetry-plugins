"""The *Murature* pane: the walls of the chunk, the faces of the chosen wall with their size, resolution and drawings,
the buttons that make a face and measure its shape, and the summary of the chosen face. It only says what the user
asks (``requested``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

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
              "a una quota", "One face per side from the slice of the model at a height"),
             ("adopt", ("Prospetto esistente...", "Existing elevation..."), "Un prospetto già fatto dal programma "
              "(piano o cilindro, anche sull'asse della torre) diventa una faccia del muro così com'è, senza rifarlo",
              "An elevation the program already made (plane or cylinder, also on the tower's axis) becomes a face of "
              "the wall as it is, without rebuilding it"))
FACE_ACTIONS = (("show", ("Mostra", "Show")),
                ("shape", ("Forma", "Shape")),
                ("bulge", ("Spanciamento sì/no", "Bulge on/off")),
                ("sections", ("Sezioni...", "Sections...")),
                ("axis", ("Asse della torre...", "Axis of the tower...")),
                ("compare", ("Confronta rilievi...", "Compare surveys...")),
                ("folder", ("Apri la cartella", "Open the folder")),
                ("remove_face", ("Rimuovi faccia", "Remove face")))
STONE_ACTIONS = (
    ("find_stones", ("Trova le pietre...", "Find the stones..."),
     "Lo strumento Pietre del programma (SAM a tasselli) sul prospetto della faccia",
     "The program's Stones tool (SAM at tiles) on the face's elevation"),
    ("measure_stones", ("Misura le pietre", "Measure the stones"),
     "Misure di ogni pietra, filari, quota di malta e il tipo che le misure suggeriscono",
     "Measures of each stone, courses, mortar share and the type the measures suggest"),
    ("classified", ("Corsi e giunti da un disegno classificato...", "Courses and joints from a classified drawing..."),
     "Le linee disegnate sulla faccia ordinate lungo la sua orizzontale (il programma 1.7.7): letti, giunti, filari e "
     "pietre; prima la tavola di controllo, poi le pietre si aggiungono a quelle della faccia",
     "The lines drawn on the face sorted along its horizontal (the program 1.7.7): beds, joints, courses and stones; "
     "the check plate first, then the stones are added to those of the face"),
    ("wall_area", ("Area del muro...", "Wall area..."),
     "Disegna sulla faccia l'area del muro per la quota di malta (altrimenti i limiti della faccia)",
     "Draw on the face the wall's area for the mortar share (else the face's bounds)"),
    ("export_face", ("Esporta faccia...", "Export face..."),
     "DXF, SVG, GeoPackage dei disegni con le misure delle pietre, tabelle CSV ed Excel",
     "DXF, SVG, GeoPackage of the drawings with the stones' measures, CSV and Excel tables"))
USM_ACTIONS = (
    ("propose", ("Proponi rapporti...", "Propose relationships..."),
     "Rapporti fra le USM disegnate sulla faccia, proposti dalla geometria: li scegli tu",
     "Relationships between the USM drawn on the face, proposed by the geometry: you choose them"),
    ("save_pyarchinit", ("Salva in PyArchInit", "Save in PyArchInit"),
     "Le schede US delle USM, i rapporti scelti e i disegni della faccia come media",
     "The US forms of the USM, the relationships chosen and the face's drawings as media"))
EXPORT_ACTIONS = (
    ("plate", ("Tavola...", "Plate..."), "La tavola in scala della faccia (SVG e PDF)",
     "The plate at scale of the face (SVG and PDF)"),
    ("export_all", ("Esporta tutto...", "Export all..."),
     "Tavola, DXF, GeoPackage, tabelle, sezioni e spanciamento della faccia in una cartella",
     "Plate, DXF, GeoPackages, tables, sections and bulge of the face in a folder"),
    ("ifc", ("IFC del muro...", "Wall IFC..."), "IfcWall fra due facce o dietro una faccia, con le USM come parti",
     "IfcWall between two faces or behind one face, with the USM as parts"),
    ("qgis", ("Apri la faccia in QGIS", "Open the face in QGIS"),
     "La faccia spiegata in QGIS: ortofoto, spanciamento, pietre con le misure, disegni",
     "The face unfolded in QGIS: orthophoto, bulge, stones with their measures, drawings"))
STONE_COLUMNS = (("n", "n"), ("Filare", "Course"), ("Larghezza", "Width"), ("Altezza", "Height"), ("Area", "Area"),
                 ("Allungamento", "Elongation"))
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
        lay.addWidget(QLabel(f"<b>{t('Pietre e filari', 'Stones and courses')}</b>"))
        lay.addWidget(self._flow(STONE_ACTIONS))
        from . import classified

        if not classified.available():  # the program 1.7.7 or later classifies drawings
            self.buttons["classified"].setVisible(False)
        self.stones_table = QTableWidget(0, len(STONE_COLUMNS))
        self.stones_table.setHorizontalHeaderLabels([t(*c) for c in STONE_COLUMNS])
        self.stones_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.stones_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stones_table.setMaximumHeight(160)
        lay.addWidget(self.stones_table)
        lay.addWidget(QLabel(f"<b>{t('USM', 'USM')}</b>"))
        lay.addWidget(self._flow(USM_ACTIONS))
        lay.addWidget(QLabel(f"<b>{t('Tavole ed esportazioni', 'Plates and exports')}</b>"))
        lay.addWidget(self._flow(EXPORT_ACTIONS))
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
        for name in [a[0] for a in FACE_ACTIONS + STONE_ACTIONS + USM_ACTIONS + EXPORT_ACTIONS]:
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

    def set_stones(self, rows: list[dict]) -> None:
        self.stones_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            el = r.get("elongation")
            cells = (str(r["n"]), str(r.get("course") or "-"), metres(r["width"]), metres(r["height"]),
                     f"{num(r['area'], 3)} m²", "-" if el is None else num(el, 2))
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.stones_table.setItem(i, j, item)


class RelationsDialog(QDialog):
    """The relationships proposed by the geometry, each with its evidence and a box to tick: none is ticked, the user
    chooses (a proposal is not a relationship until accepted)."""

    def __init__(self, proposals: list[dict], parent=None):
        from .usm import proposal_text

        super().__init__(parent)
        self.setWindowTitle(t("Rapporti proposti", "Relationships proposed"))
        self.proposals = list(proposals)
        lay = QVBoxLayout(self)
        intro = QLabel(t("Rapporti proposti dalla geometria delle USM sulla faccia. Spunta quelli che hai verificato: "
                         "solo questi saranno scritti.", "Relationships proposed by the geometry of the USM on the "
                         "face. Tick those you have checked: only these will be written."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.list = QListWidget()
        for p in self.proposals:
            item = QListWidgetItem(proposal_text(p) + (f" ({p['note']})" if p.get("note") else ""))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.list.setWordWrap(True)
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(640, 360)

    def chosen(self) -> list[dict]:
        return [p for i, p in enumerate(self.proposals)
                if self.list.item(i).checkState() == Qt.CheckState.Checked]
