"""The *Piani di volo* pane: the steps of a flight plan in order - area, camera, parameters, the plan, the
terrain, the coverage, the targets, the exports, after the flight - each with its buttons and a line saying what
it has or what it misses (as the program's multispectral pane).

The pane only shows and asks: every button emits a signal the plugin (:mod:`ppg_flightplan`) answers, and the
plugin gives the pane what the chunk holds with :meth:`FlightPlanPane.set_state`. Its own fields (camera,
goal, overlaps...) are the parameters of the next design.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .texts import FORMAT_NAMES, STEPS, format_name, step_title, t

KEY_ROLE = int(Qt.ItemDataRole.UserRole)


def _flow(parent):
    from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

    return FlowLayout(parent)


def _labelled(text: str, widget: QWidget) -> QWidget:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addWidget(QLabel(text))
    lay.addWidget(widget)
    return box


def _spin(lo: float, hi: float, value: float, decimals: int = 0, step: float = 1.0, suffix: str = ""):
    s = QDoubleSpinBox() if decimals else QSpinBox()
    s.setRange(lo, hi) if decimals else s.setRange(int(lo), int(hi))
    if decimals:
        s.setDecimals(decimals)
        s.setSingleStep(step)
    s.setValue(value if decimals else int(value))
    if suffix:
        s.setSuffix(suffix)
    return s


class FlightPlanPane(QWidget):
    """The steps of a flight plan with their state; the signals say what the user asked."""

    area_draw_requested = Signal()
    area_layer_requested = Signal()
    area_file_requested = Signal()
    area_coords_requested = Signal()
    camera_changed = Signal()
    camera_exif_requested = Signal()
    camera_sensor_requested = Signal()
    camera_manual_requested = Signal()
    design_requested = Signal()
    takeoff_requested = Signal(str)          # "" the plan's takeoff, else the key of a leg
    block_toggled = Signal(int, bool)
    terrain_requested = Signal(str, str)     # source ("dem" | "tiles"), detail ("normale" | "fine" | "finissimo")
    terrain_removed = Signal()
    coverage_requested = Signal(int)         # the threshold of low coverage (photos per cell)
    coverage_shown = Signal(bool)            # True the coverage, False the orthophoto back
    targets_csv_requested = Signal()
    targets_markers_requested = Signal()
    targets_removed = Signal()
    export_requested = Signal(str)           # a code of project.FORMATS
    djifly_requested = Signal()
    qgis_requested = Signal()
    plan_selected = Signal(str)
    plan_removed = Signal(str)
    compare_requested = Signal(float)        # the maximum distance to pair a shot and a photo, metres (0: automatic)
    obtained_shown = Signal()                # the coverage obtained on the 2D view

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filling = False
        self._own_cameras: list[tuple[str, str]] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self._lay = lay
        intro = QLabel(t("Il volo del drone progettato prima di andare sul campo: i passi in ordine. Ogni passo dice "
                         "cosa ha e cosa gli manca.",
                         "The drone flight designed before going to the field: the steps in order. Every step says "
                         "what it has and what it misses."))
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # the plans of the chunk
        self.c_plan = QComboBox()
        self.c_plan.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_plan.setMinimumContentsLength(14)
        self.b_remove_plan = QPushButton(t("Rimuovi", "Remove"))
        self.b_remove_plan.setToolTip(t("Toglie il piano dal chunk (Modifica > Annulla lo rimette)",
                                        "Takes the plan out of the chunk (Edit > Undo puts it back)"))
        holder = QWidget()
        bar = _flow(holder)
        bar.addWidget(_labelled(t("Piano:", "Plan:"), self.c_plan))
        bar.addWidget(self.b_remove_plan)
        lay.addWidget(holder)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        self.l_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.l_status)

        self.states: dict[str, QLabel] = {}
        self.titles: dict[str, QLabel] = {}
        self.buttons: list[QWidget] = []
        number = {key: k + 1 for k, (key, _names) in enumerate(STEPS)}

        def step(key: str, widgets: list[QWidget], extra: QWidget | None = None) -> None:
            title = QLabel(f"<b>{number[key]}. {step_title(key)}</b>")
            title.setWordWrap(True)
            lay.addWidget(title)
            self.titles[key] = title
            if widgets:
                box = QWidget()
                flow = _flow(box)
                for w in widgets:
                    if isinstance(w, QComboBox):
                        w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                        w.setMinimumContentsLength(12)
                    flow.addWidget(w)
                lay.addWidget(box)
            if extra is not None:
                lay.addWidget(extra)
            s = QLabel("")
            s.setWordWrap(True)
            s.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            s.setContentsMargins(0, 0, 0, 8)
            lay.addWidget(s)
            self.states[key] = s

        def button(text: str, signal, tip: str = "") -> QPushButton:
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(lambda _=False: signal.emit())
            self.buttons.append(b)
            return b

        # 1 area
        self.b_draw = button(t("Disegna sulla vista", "Draw on the view"), self.area_draw_requested,
                             t("Il poligono dell'area sulla vista 2D (ortofoto, DEM)",
                               "The polygon of the area on the 2D view (orthophoto, DEM)"))
        self.b_layer = button(t("Da un layer...", "From a layer..."), self.area_layer_requested,
                              t("Un poligono già disegnato nella vista 2D", "A polygon already drawn on the 2D view"))
        self.b_file = button(t("Da un file (KML, GeoJSON, GeoPackage)...", "From a file (KML, GeoJSON, GeoPackage)..."),
                             self.area_file_requested)
        self.b_coords = button(t("Coordinate...", "Coordinates..."), self.area_coords_requested,
                               t("I vertici in gradi o nel sistema del chunk", "The vertices in degrees or in the "
                                 "chunk's system"))
        self.s_margin = _spin(0, 500, 0, 1, 5.0, " m")
        self.s_margin.setToolTip(t("Metri aggiunti attorno all'area nei blocchi del volo: i bordi hanno la copertura "
                                   "del centro. L'area disegnata resta com'è.",
                                   "Metres added around the area in the blocks of the flight: the edges get the "
                                   "coverage of the centre. The area drawn stays as it is."))
        step("area", [self.b_draw, self.b_layer, self.b_file, self.b_coords,
                      _labelled(t("Margine", "Margin"), self.s_margin)])

        # 2 camera
        self.c_camera = QComboBox()
        self.fill_cameras()
        self.c_camera.currentIndexChanged.connect(lambda _i: self._camera_changed())
        self.b_exif = button(t("Dalla foto (EXIF)...", "From a photo (EXIF)..."), self.camera_exif_requested)
        self.b_sensor = button(t("Dal chunk", "From the chunk"), self.camera_sensor_requested,
                               t("Il sensore calibrato del chunk", "The calibrated sensor of the chunk"))
        self.b_manual = button(t("A mano...", "By hand..."), self.camera_manual_requested)
        self.k_high = QCheckBox(t("Alta risoluzione", "High resolution"))
        self.k_high.setChecked(True)
        self.k_high.toggled.connect(lambda _b: self._camera_changed())
        self.s_battery = _spin(3, 60, 18, 1, 1.0, " min")
        self.s_battery.valueChanged.connect(lambda _v: self._camera_changed())
        step("camera", [self.c_camera, self.b_exif, self.b_sensor, self.b_manual, self.k_high,
                        _labelled(t("Batteria", "Battery"), self.s_battery)])

        # 3 parameters
        from .design import GOALS, goal_name

        self.c_goal = QComboBox()
        for g in GOALS:
            self.c_goal.addItem(goal_name(g), g["id"])
        self.s_value = _spin(0.1, 50, GOALS[0]["valore"], 2, 0.1)
        self.l_unit = QLabel(GOALS[0]["unita"])
        self.c_goal.currentIndexChanged.connect(self._goal_changed)
        self.k_height = QCheckBox(t("Quota (m)", "Height (m)"))
        self.k_height.setToolTip(t("La quota sul decollo al posto dell'obiettivo", "The height above the takeoff "
                                   "instead of the goal"))
        self.s_height = _spin(10, 500, 60, 1, 5.0, " m")
        self.s_height.setEnabled(False)
        self.k_height.toggled.connect(self.s_height.setEnabled)
        self.s_front = _spin(50, 95, 80, 0, 1, " %")
        self.s_side = _spin(40, 95, 70, 0, 1, " %")
        self.c_direction = QComboBox()
        self.c_direction.addItem(t("Lungo il lato più lungo", "Along the longest side"), "auto")
        self.c_direction.addItem(t("Gradi", "Degrees"), "gradi")
        self.s_degrees = _spin(0, 179, 0, 0, 5, "°")
        self.s_degrees.setEnabled(False)
        self.c_direction.currentIndexChanged.connect(
            lambda _i: self.s_degrees.setEnabled(self.c_direction.currentData() == "gradi"))
        self.k_double = QCheckBox(t("Doppia griglia", "Double grid"))
        self.c_obliques = QComboBox()
        self.c_obliques.addItem(t("Obliqui: secondo l'obiettivo", "Obliques: as the goal says"), None)
        self.c_obliques.addItem(t("Obliqui: sì", "Obliques: yes"), True)
        self.c_obliques.addItem(t("Obliqui: no", "Obliques: no"), False)
        self.s_speed = _spin(1, 15, 5, 1, 0.5, " m/s")
        self.c_mode = QComboBox()
        self.c_mode.addItem(t("Scatto per punto", "A shot per point"), "per_punto")
        self.c_mode.addItem(t("Scatto a intervallo", "Shots by interval"), "intervallo")
        self.b_design = button(t("Progetta", "Design"), self.design_requested,
                               t("Il piano per l'area, la camera e i parametri", "The plan for the area, the camera "
                                 "and the parameters"))
        step("params", [self.c_goal, self.s_value, self.l_unit, self.k_height, self.s_height,
                        _labelled(t("Avanti", "Forward"), self.s_front), _labelled(t("Laterale", "Side"), self.s_side),
                        self.c_direction, self.s_degrees, self.k_double, self.c_obliques,
                        _labelled(t("Velocità", "Speed"), self.s_speed), self.c_mode, self.b_design])

        # 4 the plan: blocks and legs
        self.t_plan = QTableWidget(0, 6)
        self.t_plan.setHorizontalHeaderLabels([t("Blocco / tratta", "Block / leg"), t("Scatti", "Shots"),
                                               t("Minuti", "Minutes"), t("Batterie", "Batteries"), "GSD cm",
                                               t("Quote m", "Heights m")])
        self.t_plan.verticalHeader().setVisible(False)
        self.t_plan.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.t_plan.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.t_plan.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.t_plan.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.t_plan.setMinimumHeight(120)
        self.t_plan.itemChanged.connect(self._item_changed)
        self.b_takeoff = QPushButton(t("Decollo sulla vista", "Takeoff on the view"))
        self.b_takeoff.clicked.connect(lambda _=False: self.takeoff_requested.emit(""))
        self.b_leg_takeoff = QPushButton(t("Decollo di questa tratta", "Takeoff of this leg"))
        self.b_leg_takeoff.setToolTip(t("Un decollo proprio per la tratta scelta nella tabella",
                                        "A takeoff of its own for the leg chosen in the table"))
        self.b_leg_takeoff.clicked.connect(self._leg_takeoff)
        self.buttons += [self.b_takeoff, self.b_leg_takeoff]
        self.l_warnings = QLabel("")
        self.l_warnings.setWordWrap(True)
        self.l_warnings.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.l_warnings.setStyleSheet("color: #b26a00;")
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.t_plan)
        v.addWidget(self.l_warnings)
        step("plan", [self.b_takeoff, self.b_leg_takeoff], holder)

        # 5 terrain
        self.c_source = QComboBox()
        self.c_source.addItem(t("DEM del progetto", "Project DEM"), "dem")
        self.c_source.addItem(t("Tegole pubbliche (rete)", "Public tiles (network)"), "tiles")
        self.c_detail = QComboBox()
        for code, (it, en) in (("normale", ("Dettaglio normale", "Normal detail")),
                               ("fine", ("Dettaglio fine", "Fine detail")),
                               ("finissimo", ("Dettaglio finissimo", "Finest detail"))):
            self.c_detail.addItem(t(it, en), code)
        self.b_terrain = QPushButton(t("Segui il terreno", "Follow the terrain"))
        self.b_terrain.clicked.connect(lambda _=False: self.terrain_requested.emit(
            str(self.c_source.currentData()), str(self.c_detail.currentData())))
        self.b_terrain_off = button(t("Togli", "Remove"), self.terrain_removed,
                                    t("Di nuovo una quota per blocco", "One height per block again"))
        self.buttons.append(self.b_terrain)
        step("terrain", [self.c_source, self.c_detail, self.b_terrain, self.b_terrain_off])

        # 6 coverage
        from .coverage import LOW

        self.s_low = _spin(1, 50, LOW, 0, 1)
        self.s_low.setToolTip(t("Meno foto di queste su una cella: copertura bassa",
                                "Fewer photos than these on a cell: low coverage"))
        self.b_coverage = QPushButton(t("Calcola", "Compute"))
        self.b_coverage.clicked.connect(lambda _=False: self.coverage_requested.emit(int(self.s_low.value())))
        self.b_show_coverage = QPushButton(t("Mostra la copertura", "Show the coverage"))
        self.b_show_coverage.clicked.connect(lambda _=False: self.coverage_shown.emit(True))
        self.b_show_ortho = QPushButton(t("Mostra l'ortofoto", "Show the orthophoto"))
        self.b_show_ortho.clicked.connect(lambda _=False: self.coverage_shown.emit(False))
        self.buttons += [self.b_coverage, self.b_show_coverage, self.b_show_ortho]
        step("coverage", [_labelled(t("Soglia", "Threshold"), self.s_low), self.b_coverage, self.b_show_coverage,
                          self.b_show_ortho])

        # 7 targets
        from .targets import METRES_ABOVE_GROUND

        self.b_csv = button(t("Da CSV...", "From a CSV..."), self.targets_csv_requested)
        self.b_markers = button(t("Dai marker del chunk", "From the chunk's markers"), self.targets_markers_requested)
        self.b_targets_off = button(t("Togli", "Remove"), self.targets_removed)
        self.s_metres = _spin(2, 60, METRES_ABOVE_GROUND, 1, 1.0, " m")
        self.s_metres.setToolTip(t("A che altezza dal suolo si scende a fotografare ogni target",
                                   "How high above the ground the drone goes down to photograph every target"))
        self.l_target_warning = QLabel("")
        self.l_target_warning.setWordWrap(True)
        self.l_target_warning.setStyleSheet("color: #c62828; font-weight: bold;")
        self.l_target_warning.setVisible(False)
        step("targets", [self.b_csv, self.b_markers, self.b_targets_off,
                         _labelled(t("Metri dal suolo", "Metres above the ground"), self.s_metres)],
             self.l_target_warning)

        # 8 export
        self.c_format = QComboBox()
        for code in FORMAT_NAMES:
            self.c_format.addItem(format_name(code), code)
        self.b_export = QPushButton(t("Esporta...", "Export..."))
        self.b_export.clicked.connect(lambda _=False: self.export_requested.emit(str(self.c_format.currentData())))
        self.b_djifly = button(t("Copia in DJI Fly...", "Copy into DJI Fly..."), self.djifly_requested,
                               t("Le tratte nelle missioni fittizie del telecomando",
                                 "The legs into the placeholder missions of the remote controller"))
        self.b_sheet = QPushButton(t("Scheda Excel", "Excel sheet"))
        self.b_sheet.clicked.connect(lambda _=False: self.export_requested.emit("sheet-xlsx"))
        self.b_sheet_html = QPushButton(t("Scheda HTML", "HTML sheet"))
        self.b_sheet_html.clicked.connect(lambda _=False: self.export_requested.emit("sheet-html"))
        self.b_qgis = button(t("Apri in QGIS", "Open in QGIS"), self.qgis_requested)
        self.buttons += [self.b_export, self.b_sheet, self.b_sheet_html]
        step("export", [self.c_format, self.b_export, self.b_djifly, self.b_sheet, self.b_sheet_html, self.b_qgis])

        # 9 after the flight: the plan against the photos of the chunk
        self.after_box = QWidget()
        self.after_layout = _flow(self.after_box)
        self.s_max_distance = _spin(0, 500, 0, 1, 0.5, " m")
        self.s_max_distance.setSpecialValueText(t("automatica", "automatic"))
        self.s_max_distance.setToolTip(t("Quanto lontana dallo scatto previsto può essere la foto che lo fa: "
                                         "automatica = metà del passo tra gli scatti della tratta",
                                         "How far from the planned shot the photo that takes it may be: automatic = "
                                         "half the step between the shots of the leg"))
        self.b_compare = QPushButton(t("Confronta", "Compare"))
        self.b_compare.setToolTip(t("Gli scatti previsti contro le foto del chunk (le loro posizioni) e, con le foto "
                                    "allineate, la copertura ottenuta contro quella prevista",
                                    "The planned shots against the photos of the chunk (their positions) and, with "
                                    "the photos aligned, the coverage obtained against the one predicted"))
        self.b_compare.clicked.connect(
            lambda _=False: self.compare_requested.emit(float(self.s_max_distance.value())))
        self.b_show_obtained = button(t("Mostra la copertura ottenuta", "Show the coverage obtained"),
                                      self.obtained_shown)
        self.buttons += [self.b_compare]
        for w in (_labelled(t("Distanza massima", "Maximum distance"), self.s_max_distance), self.b_compare,
                  self.b_show_obtained):
            self.after_layout.addWidget(w)
        step("after", [], self.after_box)
        lay.addStretch(1)

        self.c_plan.currentIndexChanged.connect(self._plan_changed)
        self.b_remove_plan.clicked.connect(self._remove_plan)
        self.buttons += [self.b_remove_plan]
        self.set_state({}, None, [])

    # ------------------------------------------------------------------ the camera list
    def fill_cameras(self, own: list[tuple[str, str]] | None = None, current: str | None = None) -> None:
        """The cameras of the list grouped by kind (DJI with the SDK, DJI Fly only, other drones), and the plan's
        own cameras (EXIF, chunk sensor, typed) at the end: ``own`` = [(id, name)]."""
        from .camera import CAMERAS

        if own is not None:
            self._own_cameras = list(own)
        current = current if current is not None else self.camera_id()
        self._filling = True
        try:
            self.c_camera.clear()
            model = self.c_camera.model()
            groups = (("sdk", t("DJI (Pilot, app con l'SDK)", "DJI (Pilot, apps with the SDK)")),
                      ("fly", t("DJI Fly (solo dal telecomando)", "DJI Fly (from the remote controller only)")),
                      ("altro", t("Altri droni", "Other drones")))
            for group, title in groups:
                cams = [c for c in CAMERAS if c.group == group]
                if not cams:
                    continue
                self._header(model, title)
                for c in cams:
                    self.c_camera.addItem(c.name, c.id)
            if self._own_cameras:
                self._header(model, t("Camere proprie", "Own cameras"))
                for id_, name in self._own_cameras:
                    self.c_camera.addItem(name, id_)
            self.set_camera(current or "mini4pro")
        finally:
            self._filling = False

    @staticmethod
    def _header(model, title: str) -> None:
        item = QStandardItem(title)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        if isinstance(model, QStandardItemModel):
            model.appendRow(item)

    def camera_id(self) -> str | None:
        data = self.c_camera.currentData() if hasattr(self, "c_camera") else None
        return str(data) if data else None

    def set_camera(self, id_: str) -> None:
        i = self.c_camera.findData(id_)
        if i < 0:
            i = self.c_camera.findData("mini4pro")
        if i >= 0:
            self.c_camera.setCurrentIndex(i)

    def _camera_changed(self) -> None:
        if not self._filling:
            self.camera_changed.emit()

    def _goal_changed(self, _i: int) -> None:
        from .design import goal

        g = goal(str(self.c_goal.currentData()))
        self.s_value.setRange(float(g["minimo"]), float(g["massimo"]))
        self.s_value.setValue(float(g["valore"]))
        self.l_unit.setText(g["unita"])

    # ------------------------------------------------------------------ the parameters of the next design
    def parameters(self) -> dict:
        """What the fields say: goal and value (or the height), overlaps, direction, double grid, obliques,
        speed, shooting mode, margin, high resolution, battery."""
        return {"goal": str(self.c_goal.currentData()), "value": float(self.s_value.value()),
                "height": float(self.s_height.value()) if self.k_height.isChecked() else None,
                "front": int(self.s_front.value()), "side": int(self.s_side.value()),
                "direction": None if self.c_direction.currentData() == "auto" else int(self.s_degrees.value()),
                "double": self.k_double.isChecked(), "obliques": self.c_obliques.currentData(),
                "speed": float(self.s_speed.value()), "mode": str(self.c_mode.currentData()),
                "margin": float(self.s_margin.value()), "high": self.k_high.isChecked(),
                "battery": float(self.s_battery.value()), "camera": self.camera_id() or "mini4pro"}

    def set_parameters(self, piano: dict) -> None:
        """The fields from a plan (opened or chosen): its camera, battery, goal, margin and first block."""
        self._filling = True
        try:
            if piano.get("camera"):
                self.set_camera(str(piano["camera"]))
            self.k_high.setChecked(bool(piano.get("alta", True)))
            self.s_battery.setValue(float(piano.get("minutiBatteria", 18)))
            goal = (piano.get("obiettivo") or {}).get("id")
            i = self.c_goal.findData(goal) if goal else -1
            if i >= 0:
                self.c_goal.setCurrentIndex(i)
                value = (piano.get("obiettivo") or {}).get("valore")
                if value is not None:
                    self.s_value.setValue(float(value))
            self.s_margin.setValue(float(piano.get("margine") or 0.0))
            first = next((b for b in piano.get("blocchi") or [] if b.get("tipo", "griglia") == "griglia"), None)
            if first is not None:
                self.s_front.setValue(int(first.get("avanti", 80)))
                self.s_side.setValue(int(first.get("lato", 70)))
                self.s_speed.setValue(float(first.get("velocita", 5.0)))
                k = self.c_mode.findData(first.get("modo"))
                if k >= 0:
                    self.c_mode.setCurrentIndex(k)
        finally:
            self._filling = False

    # ------------------------------------------------------------------ the plans
    def set_plans(self, plans: list[tuple[str, str]], active: str | None) -> None:
        self._filling = True
        try:
            self.c_plan.clear()
            for id_, name in plans:
                self.c_plan.addItem(name, id_)
            i = self.c_plan.findData(active) if active else -1
            self.c_plan.setCurrentIndex(i if i >= 0 else (0 if plans else -1))
        finally:
            self._filling = False

    def plan_id(self) -> str | None:
        data = self.c_plan.currentData()
        return str(data) if data else None

    def _plan_changed(self, _i: int) -> None:
        if not self._filling and self.plan_id():
            self.plan_selected.emit(self.plan_id())

    def _remove_plan(self) -> None:
        if self.plan_id():
            self.plan_removed.emit(self.plan_id())

    # ------------------------------------------------------------------ the table of the plan
    def set_plan_table(self, summary: dict | None) -> None:
        """``summary``: ``{"blocchi": [{"indice", "nome", "attivo", "scatti", "minuti", "batterie", "gsd",
        "quote", "tratte": [{"chiave", "nome", "scatti", "minuti", "quote"}]}]}`` (None: empty)."""
        self._filling = True
        try:
            self.t_plan.setRowCount(0)
            for b in (summary or {}).get("blocchi") or []:
                row = self.t_plan.rowCount()
                self.t_plan.insertRow(row)
                name = QTableWidgetItem(b["nome"])
                name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                              | Qt.ItemFlag.ItemIsUserCheckable)
                name.setCheckState(Qt.CheckState.Checked if b["attivo"] else Qt.CheckState.Unchecked)
                name.setData(KEY_ROLE, ("block", int(b["indice"])))
                name.setToolTip(t("Spunta: il blocco vola e va nei file", "Ticked: the block flies and goes into the "
                                  "files"))
                font = name.font()
                font.setBold(True)
                name.setFont(font)
                self.t_plan.setItem(row, 0, name)
                for col, value in enumerate((b.get("scatti"), b.get("minuti"), b.get("batterie"), b.get("gsd"),
                                             b.get("quote")), start=1):
                    self.t_plan.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))
                for leg in b.get("tratte") or []:
                    row = self.t_plan.rowCount()
                    self.t_plan.insertRow(row)
                    item = QTableWidgetItem("    " + leg["nome"])
                    item.setData(KEY_ROLE, ("leg", leg["chiave"]))
                    self.t_plan.setItem(row, 0, item)
                    for col, value in ((1, leg.get("scatti")), (2, leg.get("minuti")), (5, leg.get("quote"))):
                        self.t_plan.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))
        finally:
            self._filling = False

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 0:
            return
        data = item.data(KEY_ROLE)
        if data and data[0] == "block":
            # after this signal: the answer fills the table again, which must not happen inside its own signal
            from PySide6.QtCore import QTimer

            index, on = int(data[1]), item.checkState() == Qt.CheckState.Checked
            QTimer.singleShot(0, lambda: self.block_toggled.emit(index, on))

    def selected_leg(self) -> str | None:
        rows = self.t_plan.selectionModel().selectedRows() if self.t_plan.selectionModel() else []
        for r in rows:
            item = self.t_plan.item(r.row(), 0)
            data = item.data(KEY_ROLE) if item is not None else None
            if data and data[0] == "leg":
                return str(data[1])
        return None

    def _leg_takeoff(self) -> None:
        key = self.selected_leg()
        if key:
            self.takeoff_requested.emit(key)
        else:
            self.set_status(t("Scegli una tratta nella tabella del piano.", "Choose a leg in the table of the plan."))

    # ------------------------------------------------------------------ the state
    def set_state(self, state: dict, summary: dict | None, warnings: list[str]) -> None:
        """``state``: a line per step key (``area``, ``camera``... see :data:`ppg_flightplan.texts.STEPS`), the
        empty ones say nothing yet; ``summary`` the table of the plan; ``warnings`` the plan's warnings in
        words. ``state["target_warning"]``, when given, is shown in red under the targets (the order of the
        columns uncertain); ``state["obtained"]`` true shows the button of the coverage obtained."""
        for key, label in self.states.items():
            label.setText(str(state.get(key) or ""))
        self.set_plan_table(summary)
        self.l_warnings.setText("\n".join(f"• {w}" for w in warnings))
        self.l_warnings.setVisible(bool(warnings))
        self.b_show_obtained.setVisible(bool(state.get("obtained")))   # only with a coverage obtained to show
        strong = state.get("target_warning") or ""
        self.l_target_warning.setText(strong)
        self.l_target_warning.setVisible(bool(strong))
        has_plan = summary is not None
        for w in (self.b_takeoff, self.b_leg_takeoff, self.b_terrain, self.b_terrain_off, self.b_coverage,
                  self.b_show_coverage, self.b_csv, self.b_markers, self.b_targets_off, self.b_export, self.b_djifly,
                  self.b_sheet, self.b_sheet_html, self.b_qgis, self.b_remove_plan, self.b_compare,
                  self.b_show_obtained):
            w.setProperty("needs_plan", True)
        self._has_plan = has_plan
        self._apply_enabled()

    def _apply_enabled(self) -> None:
        busy = getattr(self, "_busy", False)
        has_plan = getattr(self, "_has_plan", False)
        for w in self.buttons:
            needs = bool(w.property("needs_plan"))
            w.setEnabled(not busy and (has_plan or not needs))
        self.c_plan.setEnabled(not busy)

    def set_busy(self, busy: bool) -> None:
        """One step at a time: the buttons wait while a step of the program runs."""
        self._busy = bool(busy)
        self._apply_enabled()

    def set_status(self, text: str) -> None:
        self.l_status.setText(text)
