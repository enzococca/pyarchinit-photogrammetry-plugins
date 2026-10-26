"""The *Subacqueo* pane: usable photos and frames, colours, depths - each with its state and its
buttons, the products of the active chunk."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import number, t


def _buttons(parent_layout, *buttons) -> None:
    holder = QWidget()
    flow = FlowLayout(holder)
    flow.setContentsMargins(0, 0, 0, 0)
    for b in buttons:
        flow.addWidget(b)
    parent_layout.addWidget(holder)


def _label(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _combo() -> QComboBox:
    """A combo box that does not widen the pane to its longest entry."""
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(10)
    c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return c


def _spin(lo: float, hi: float, value: float, step: float = 0.05, decimals: int = 2, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


def pixmap(img: np.ndarray, width: int = 300) -> QPixmap:
    """A BGR or grey uint8 image as a pixmap ``width`` pixels wide."""
    a = np.ascontiguousarray(img)
    if a.ndim == 2:
        q = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(a[..., ::-1])
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy()).scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


class UnderwaterPane(QWidget):
    # photos and frames
    video_requested = Signal()
    sequence_requested = Signal()
    calibration_requested = Signal()
    score_requested = Signal(dict)
    thresholds_changed = Signal(dict)
    exclude_requested = Signal()
    include_requested = Signal()
    csv_requested = Signal()
    # colours
    preview_requested = Signal(dict)
    correct_requested = Signal(dict)
    use_requested = Signal(str)
    folder_requested = Signal()
    # depths
    log_requested = Signal(dict)
    match_requested = Signal(dict)
    offset_requested = Signal(dict)
    level_requested = Signal(dict)
    positions_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # the texts wrap, the pane stays narrow
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = _label(t("Rilievi sott'acqua: le foto utilizzabili e i fotogrammi da video e sequenze, i colori "
                         "ripristinati con la distanza di ogni pixel, le profondità che mettono in bolla e in scala "
                         "il chunk.",
                         "Underwater surveys: the usable photos and the frames of videos and sequences, the colours "
                         "restored with the range of every pixel, the depths that level and scale the chunk."))
        lay.addWidget(intro)
        lay.addWidget(self._photos_box())
        lay.addWidget(self._colour_box())
        lay.addWidget(self._depth_box())
        lay.addStretch(1)

    # ------------------------------------------------------------------ photos and frames
    def _photos_box(self) -> QGroupBox:
        box = QGroupBox(t("1. Foto e fotogrammi", "1. Photos and frames"))
        lay = QVBoxLayout(box)
        self.b_video = QPushButton(t("Fotogrammi da un video...", "Frames from a video..."))
        self.b_video.setToolTip(t("GoPro, ROV: un fotogramma ogni volta che la vista si è spostata, il più nitido",
                                  "GoPro, ROV: a frame each time the view has moved, the sharpest"))
        self.b_sequence = QPushButton(t("Fotogrammi da una sequenza...", "Frames from a sequence..."))
        self.b_sequence.setToolTip(t("Una cartella di immagini a 10-30 al secondo (ROV): scelte per movimento e "
                                     "nitidezza, con l'ora di ognuna",
                                     "A folder of images at 10-30 a second (ROV): chosen by motion and sharpness, "
                                     "each with its time"))
        self.b_calib = QPushButton(t("Calibrazione (Kalibr)...", "Calibration (Kalibr)..."))
        self.b_calib.setToolTip(t("La calibrazione della camera di un ROV (YAML di Kalibr), tenuta fissa "
                                  "nell'allineamento", "The calibration of the camera of an ROV (Kalibr YAML), held "
                                                       "fixed in the alignment"))
        self.b_video.clicked.connect(self.video_requested)
        self.b_sequence.clicked.connect(self.sequence_requested)
        self.b_calib.clicked.connect(self.calibration_requested)
        _buttons(lay, self.b_video, self.b_sequence, self.b_calib)
        form = QFormLayout()
        self.s_quality = _spin(0.0, 1.0, 0.5)
        self.s_quality.setToolTip(t("Nitidezza rispetto alle foto più nitide del chunk: sotto, la foto è mossa o "
                                    "sfocata", "Sharpness against the sharpest photos of the chunk: under it, the "
                                               "photo is blurred"))
        self.s_contrast = _spin(0.0, 1.0, 0.12, 0.01)
        self.s_contrast.setToolTip(t("Estensione della luminosità (2°-98° percentile): sotto, la foto è piatta "
                                     "(acqua lattiginosa, troppo scura)",
                                     "Spread of the lightness (2nd-98th percentile): under it, the photo is flat "
                                     "(milky water, too dark)"))
        self.s_veil = _spin(0.0, 1.0, 0.6, 0.05)
        self.s_veil.setToolTip(t("Il velo della torbidità (canale scuro di verde e blu sulla luce massima): "
                                 "sopra, la foto è velata", "The veil of turbidity (dark channel of green and blue "
                                                            "over the brightest light): above it, the photo is hazy"))
        form.addRow(t("Qualità minima", "Least quality"), self.s_quality)
        form.addRow(t("Contrasto minimo", "Least contrast"), self.s_contrast)
        form.addRow(t("Velo massimo", "Most veil"), self.s_veil)
        lay.addLayout(form)
        for s in (self.s_quality, self.s_contrast, self.s_veil):
            s.valueChanged.connect(lambda _v: self.thresholds_changed.emit(self.limits()))
        self.b_score = QPushButton(t("Valuta le foto", "Score the photos"))
        self.b_exclude = QPushButton(t("Escludi le segnate", "Exclude the flagged"))
        self.b_include = QPushButton(t("Riattiva le escluse", "Include again"))
        self.b_csv = QPushButton(t("Tabella CSV...", "CSV table..."))
        self.b_score.clicked.connect(lambda: self.score_requested.emit(self.limits()))
        self.b_exclude.clicked.connect(self.exclude_requested)
        self.b_include.clicked.connect(self.include_requested)
        self.b_csv.clicked.connect(self.csv_requested)
        _buttons(lay, self.b_score, self.b_exclude, self.b_include, self.b_csv)
        self.l_photos = _label()
        lay.addWidget(self.l_photos)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([t("Foto", "Photo"), t("Qualità", "Quality"), t("Contrasto", "Contrast"),
                                              t("Velo", "Veil"), t("Perché", "Why")])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.table.setMinimumHeight(120)
        self.table.setMaximumHeight(220)
        lay.addWidget(self.table)
        return box

    def limits(self) -> dict:
        return {"quality": self.s_quality.value(), "contrast": self.s_contrast.value(), "veil": self.s_veil.value()}

    def set_limits(self, limits: dict) -> None:
        for s, k in ((self.s_quality, "quality"), (self.s_contrast, "contrast"), (self.s_veil, "veil")):
            s.blockSignals(True)
            s.setValue(float(limits.get(k, s.value())))
            s.blockSignals(False)

    def set_photos(self, rows: list[dict], text: str) -> None:  # noqa: C901
        """``rows``: the flagged photos, ``{"label", "quality", "contrast", "veil", "why", "enabled"}``."""
        self.l_photos.setText(text)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [row["label"]] + [("" if row.get(k) is None else number(row[k], 2)) for k in ("quality", "contrast",
                                                                                                  "veil")] + [row["why"]]
            for c, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                if not row.get("enabled", True):
                    item.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(r, c, item)
        if rows:
            self.table.resizeColumnsToContents()

    # ------------------------------------------------------------------ colours
    def _colour_box(self) -> QGroupBox:
        box = QGroupBox(t("2. Colori", "2. Colours"))
        lay = QVBoxLayout(box)
        form = QFormLayout()
        self.c_method = _combo()
        self.c_method.addItem(t("Ripristino fisico (Sea-thru, con le mappe di profondità)",
                                "Physics-based restoration (Sea-thru, with the depth maps)"), "seathru")
        self.c_method.addItem(t("Correzione rapida (anche prima dell'allineamento)",
                                "Quick correction (also before the alignment)"), "quick")
        form.addRow(t("Metodo", "Method"), self.c_method)
        self.c_per = _combo()
        self.c_per.addItem(t("uno per il rilievo (foto uniformi)", "one for the survey (photos alike)"), "survey")
        self.c_per.addItem(t("uno per ogni foto", "one for every photo"), "photo")
        self.c_per.setToolTip(t("I coefficienti dell'acqua: uno per il rilievo dà a tutte le foto la stessa "
                                "correzione (niente cuciture di colore nell'ortofoto); per ogni foto segue i "
                                "cambiamenti dell'acqua",
                                "The coefficients of the water: one for the survey gives every photo the same "
                                "correction (no seams of colour in the orthomosaic); one per photo follows changes "
                                "of the water"))
        form.addRow(t("Coefficienti", "Coefficients"), self.c_per)
        self.s_alpha = _spin(0.0, 3.0, 1.0, 0.1, 1)
        self.s_alpha.setToolTip(t("Quanto il rosso viene ricostruito dal verde (Ancuti et al. 2018)",
                                  "How much red is rebuilt from the green (Ancuti et al. 2018)"))
        form.addRow(t("Compensazione del rosso", "Red compensation"), self.s_alpha)
        self.s_blue = _spin(0.0, 3.0, 0.0, 0.1, 1)
        self.s_blue.setToolTip(t("Per l'acqua verde (laghi, porti): anche il blu ricostruito dal verde",
                                 "For green water (lakes, harbours): the blue rebuilt from the green too"))
        form.addRow(t("Compensazione del blu", "Blue compensation"), self.s_blue)
        self.k_balance = QCheckBox(t("bilanciamento del bianco", "white balance"))
        self.k_balance.setToolTip(t("A mondo grigio, con le medie di un campione del rilievo",
                                    "Grey world, with the means of a sample of the survey"))
        self.k_balance.setChecked(True)
        self.k_clahe = QCheckBox("CLAHE")
        self.k_clahe.setToolTip(t("Contrasto locale: aiuta l'allineamento di foto scure o torbide; altera i colori",
                                  "Local contrast: helps the alignment of dark or turbid photos; alters the colours"))
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        lay.addLayout(form)
        _buttons(lay, self.k_balance, self.k_clahe)
        self.c_method.currentIndexChanged.connect(self._method_changed)
        prev = QHBoxLayout()
        self.c_photo = _combo()
        self.c_photo.setToolTip(t("La foto dell'anteprima", "The photo of the preview"))
        self.b_preview = QPushButton(t("Anteprima", "Preview"))
        self.b_preview.clicked.connect(lambda: self.preview_requested.emit(self.colour_values()))
        prev.addWidget(self.c_photo, 1)
        prev.addWidget(self.b_preview)
        lay.addLayout(prev)
        pics = QHBoxLayout()
        self.l_before = QLabel()
        self.l_after = QLabel()
        for lab in (self.l_before, self.l_after):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setMinimumWidth(80)
            pics.addWidget(lab)
        lay.addLayout(pics)
        self.b_correct = QPushButton(t("Correggi le foto", "Correct the photos"))
        self.b_use = QPushButton(t("Usa le foto corrette", "Use the corrected photos"))
        self.b_original = QPushButton(t("Usa le originali", "Use the originals"))
        self.b_folder = QPushButton(t("Apri la cartella", "Open the folder"))
        self.b_correct.clicked.connect(lambda: self.correct_requested.emit(self.colour_values()))
        self.b_use.clicked.connect(lambda: self.use_requested.emit("corrected"))
        self.b_original.clicked.connect(lambda: self.use_requested.emit("original"))
        self.b_folder.clicked.connect(self.folder_requested)
        _buttons(lay, self.b_correct, self.b_use, self.b_original, self.b_folder)
        self.l_colour = _label()
        lay.addWidget(self.l_colour)
        self._method_changed()
        return box

    def _method_changed(self) -> None:
        seathru = self.c_method.currentData() == "seathru"
        self.c_per.setEnabled(seathru)
        for w in (self.s_alpha, self.s_blue, self.k_balance, self.k_clahe):
            w.setEnabled(not seathru)

    def colour_values(self) -> dict:
        return {"method": self.c_method.currentData(), "per": self.c_per.currentData(), "alpha": self.s_alpha.value(),
                "blue": self.s_blue.value(), "balance": self.k_balance.isChecked(), "clahe": self.k_clahe.isChecked(),
                "camera": self.c_photo.currentData()}

    def set_cameras(self, cams: list[tuple[int, str]]) -> None:
        current = self.c_photo.currentData()
        self.c_photo.blockSignals(True)
        self.c_photo.clear()
        for key, label in cams:
            self.c_photo.addItem(label, key)
        i = self.c_photo.findData(current)
        self.c_photo.setCurrentIndex(i if i >= 0 else 0)
        self.c_photo.blockSignals(False)

    def set_preview(self, before, after) -> None:
        w = max(80, (self.width() - 40) // 2)
        self.l_before.setPixmap(pixmap(before, w))
        self.l_after.setPixmap(pixmap(after, w))
        self.l_before.setToolTip(t("Prima", "Before"))
        self.l_after.setToolTip(t("Dopo", "After"))

    def set_colour(self, text: str) -> None:
        self.l_colour.setText(text)

    # ------------------------------------------------------------------ depths
    def _depth_box(self) -> QGroupBox:
        box = QGroupBox(t("3. Profondità invece del GPS", "3. Depth instead of GPS"))
        lay = QVBoxLayout(box)
        form = QFormLayout()
        self.c_water = _combo()
        self.c_water.addItem(t("mare (1025 kg/m³)", "sea (1025 kg/m³)"), 1025.0)
        self.c_water.addItem(t("acqua dolce (1000 kg/m³)", "fresh water (1000 kg/m³)"), 1000.0)
        self.c_water.setToolTip(t("Per i registri di pressione: la densità che trasforma la pressione in "
                                  "profondità", "For pressure logs: the density that turns pressure into depth"))
        form.addRow(t("Acqua", "Water"), self.c_water)
        self.s_offset = _spin(-86400.0, 86400.0, 0.0, 1.0, 2, " s")
        self.s_offset.setToolTip(t("Quanto aggiungere all'orologio della camera per leggere quello del registro "
                                   "(camera un'ora avanti: -3600)",
                                   "What to add to the clock of the camera to read the clock of the log (a camera "
                                   "an hour ahead: -3600)"))
        form.addRow(t("Scarto di tempo", "Time offset"), self.s_offset)
        self.c_scale = _combo()
        self.c_scale.addItem(t("dalle profondità", "from the depths"), "depths")
        self.c_scale.addItem(t("dalle barre di scala", "from the scale bars"), "bars")
        self.c_scale.setToolTip(t("Le profondità danno la scala quando cambiano abbastanza; con un rilievo a "
                                  "profondità costante serve una barra di scala",
                                  "The depths give the scale when they change enough; a survey at a constant depth "
                                  "needs a scale bar"))
        form.addRow(t("Scala", "Scale"), self.c_scale)
        lay.addLayout(form)
        self.b_log = QPushButton(t("Importa il registro...", "Import the log..."))
        self.b_log.setToolTip(t("Sensore di pressione del ROV, computer subacqueo (CSV con ora e profondità o "
                                "pressione), UDDF", "Pressure sensor of the ROV, dive computer (CSV with time and depth "
                                                   "or pressure), UDDF"))
        self.b_offset = QPushButton(t("Stima lo scarto", "Estimate the offset"))
        self.b_offset.setToolTip(t("Lo scarto di tempo con cui le profondità si accordano meglio con le camere "
                                   "allineate", "The time offset with which the depths agree best with the aligned "
                                                "cameras"))
        self.b_match = QPushButton(t("Associa alle foto", "Match to the photos"))
        self.b_level = QPushButton(t("Metti in bolla e in scala", "Level and scale"))
        self.b_positions = QPushButton(t("Posizioni note (USBL)...", "Known positions (USBL)..."))
        self.b_log.clicked.connect(lambda: self.log_requested.emit(self.depth_values()))
        self.b_offset.clicked.connect(lambda: self.offset_requested.emit(self.depth_values()))
        self.b_match.clicked.connect(lambda: self.match_requested.emit(self.depth_values()))
        self.b_level.clicked.connect(lambda: self.level_requested.emit(self.depth_values()))
        self.b_positions.clicked.connect(self.positions_requested)
        _buttons(lay, self.b_log, self.b_offset, self.b_match, self.b_level, self.b_positions)
        self.l_log = _label()
        self.l_level = _label()
        lay.addWidget(self.l_log)
        lay.addWidget(self.l_level)
        return box

    def depth_values(self) -> dict:
        return {"density": self.c_water.currentData(), "offset": self.s_offset.value(), "scale": self.c_scale.currentData()}

    def set_offset(self, value: float) -> None:
        self.s_offset.setValue(float(value))

    def set_depth(self, log_text: str, level_text: str) -> None:
        self.l_log.setText(log_text)
        self.l_level.setText(level_text)

    def set_busy(self, busy: bool) -> None:
        for b in (self.b_video, self.b_sequence, self.b_score, self.b_exclude, self.b_include, self.b_preview,
                  self.b_correct, self.b_use, self.b_original, self.b_log, self.b_offset, self.b_match, self.b_level,
                  self.b_positions, self.b_calib):
            b.setEnabled(not busy)
