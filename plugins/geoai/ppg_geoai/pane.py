"""Class table, reviewed-area drawing, training and application in the desktop window."""
from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from . import environment
from .texts import t

from . import dataset, results
from .client import run as job


class GeoaiPane(QWidget):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin
        self.area = None
        self._chunk = None
        self._refreshing = False
        self._class_key = None
        self.input_sources = []
        layout = QVBoxLayout(self)
        intro = QLabel(t("Impara dai disegni US verificati. Delimita un'area completamente annotata: il terreno "
                         "senza US al suo interno sarà insegnato come sfondo. Le nuove segmentazioni sono proposte "
                         "da controllare.",
                         "Learn from reviewed unit drawings. Outline a fully annotated area: ground without units "
                         "inside it is taught as background. New segmentations are proposals to review."))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.custom_inputs = QCheckBox(t("Raster opzionali: DEM, multispettrale, termico",
                                          "Optional rasters: DEM, multispectral, thermal"))
        layout.addWidget(self.custom_inputs)
        self.input_button = QPushButton(t("Scegli raster e bande...", "Choose rasters and bands..."))
        self.input_button.setEnabled(False)
        self.custom_inputs.toggled.connect(self.input_button.setEnabled)
        self.custom_inputs.toggled.connect(self.input_summary)
        self.input_button.clicked.connect(lambda: self.guard(self.choose_inputs))
        layout.addWidget(self.input_button)
        self.input_label = QLabel()
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)
        self.input_summary()
        form = QFormLayout()
        self.field = QComboBox()
        self.field.currentTextChanged.connect(self.fill_classes)
        form.addRow(t("Campo delle US", "Unit field"), self.field)
        self.tile = QSpinBox()
        self.tile.setRange(64, 2048)
        self.tile.setSingleStep(32)
        self.tile.setValue(512)
        form.addRow(t("Tassello (pixel)", "Tile (pixels)"), self.tile)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(20)
        form.addRow(t("Epoche", "Epochs"), self.epochs)
        layout.addLayout(form)
        self.classes = QTableWidget(0, 3)
        self.classes.setHorizontalHeaderLabels([t("Valore", "Value"), t("Classe", "Class"), t("Tipo US", "Unit type")])
        self.classes.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.classes)
        self.controls = []
        for title, callback in [
            (t("Disegna area verificata", "Draw reviewed area"), self.draw_area),
            (t("Prepara dataset...", "Prepare dataset..."), self.prepare),
            (t("Addestra modello...", "Train model..."), self.train),
            (t("Applica modello...", "Apply model..."), self.apply),
        ]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, fn=callback: self.guard(fn))
            self.controls.append(b)
            layout.addWidget(b)
        self.area_label = QLabel()
        self.area_label.setWordWrap(True)
        layout.addWidget(self.area_label)
        self.runs = QListWidget()
        layout.addWidget(self.runs)
        actions = QHBoxLayout()
        for title, callback in [("US", self.units), ("GeoPackage...", self.export),
                                (t("Mostra/nascondi", "Show/hide"), self.toggle)]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, fn=callback: self.guard(fn))
            self.controls.append(b)
            actions.addWidget(b)
        layout.addLayout(actions)
        runner = getattr(plugin.context.window, "runner", None)
        if runner is not None:
            runner.busy_changed.connect(self.busy)
            plugin.context.contributions.connections.append((runner.busy_changed, self.busy))

    def guard(self, callback):
        try:
            callback()
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "GeoAI", str(exc))

    def busy(self, busy):
        for control in [*self.controls, self.field, self.classes, self.tile, self.epochs, self.custom_inputs]:
            control.setEnabled(not busy)
        self.input_button.setEnabled(not busy and self.custom_inputs.isChecked())

    def available_rasters(self):
        chunk = self.plugin.chunk()
        provider = getattr(self.plugin.context, "rasters", None)
        return provider(chunk) if provider is not None and chunk is not None else []

    def input_summary(self):
        if not self.custom_inputs.isChecked():
            self.input_label.setText(t("Dati: ortofoto RGB", "Inputs: RGB orthomosaic"))
        elif not self.input_sources:
            self.input_label.setText(t("Scegli almeno un raster; il primo definisce la griglia.",
                                      "Choose at least one raster; the first defines the grid."))
        else:
            self.input_label.setText(" + ".join(f"{s['name']} ({','.join(str(b) for b in s['bands'])})"
                                                for s in self.input_sources))

    def choose_inputs(self):
        from .source_dialog import SourceDialog

        chunk = self.plugin.chunk()
        if chunk is None:
            raise ValueError(t("Apri un progetto", "Open a project"))
        sources = self.input_sources
        ortho = getattr(chunk, "orthomosaic_path", None)
        dem = getattr(chunk, "dem_path", None)
        if not sources and ortho:
            sources = [{"name": "RGB", "kind": "rgb", "path": str(ortho), "bands": [1, 2, 3]}]
        elif not sources and dem:
            sources = [{"name": "DEM", "kind": "dem", "path": str(dem), "bands": [1]}]
        dialog = SourceDialog(self.available_rasters(), sources=sources, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.input_sources = dialog.sources()
            self.input_summary()

    def selected_inputs(self):
        if self.custom_inputs.isChecked():
            if not self.input_sources:
                raise ValueError(t("Scegli prima raster e bande.", "Choose rasters and bands first."))
            sources = [{**source, "bands": list(source["bands"])} for source in self.input_sources]
            return sources[0]["path"], sources
        ortho = getattr(self.plugin.chunk(), "orthomosaic_path", None)
        if not ortho:
            raise ValueError(t("Serve un'ortofoto RGB, oppure abilita i raster opzionali.",
                               "An RGB orthomosaic is required, or enable optional rasters."))
        return ortho, None

    def units_data(self):
        ch = self.plugin.chunk()
        return [] if ch is None else (ch.meta.get("drawings") or {}).get("units", [])

    def refresh(self):
        self.status.setText(environment.status_text("geoai"))
        ch = self.plugin.chunk()
        if ch is not self._chunk:
            self._chunk, self.area = ch, None
            self.input_sources = []
            self.custom_inputs.setChecked(False)
            self.input_summary()
            self.area_label.setText(t("Area verificata: da disegnare", "Reviewed area: draw it first"))
        units = self.units_data()
        fields = sorted({"tipo", "us", "area", "sito"} | {k for u in units for k in (u.get("attributes") or {})})
        previous = self.field.currentText()
        self._refreshing = True
        self.field.clear()
        self.field.addItems(fields)
        self.field.setCurrentText(previous if previous in fields else "tipo")
        self._refreshing = False
        self.fill_classes()
        chosen = self.runs.currentItem()
        rid = chosen.data(Qt.ItemDataRole.UserRole) if chosen else None
        self.runs.clear()
        for run in results.records(ch):
            self.runs.addItem(f"{run['id']}: {run['count']} " + t("proposte", "proposals"))
            item = self.runs.item(self.runs.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, run["id"])
            if rid == run["id"]:
                self.runs.setCurrentItem(item)
        if self.runs.currentRow() < 0 and self.runs.count():
            self.runs.setCurrentRow(self.runs.count() - 1)

    def fill_classes(self):
        if self._refreshing:
            return
        values = dataset.unit_values(self.units_data(), self.field.currentText())
        key = (id(self._chunk), self.field.currentText())
        old = ({self.classes.item(r, 0).text(): (self.classes.item(r, 1).text(), self.classes.item(r, 2).text())
                for r in range(self.classes.rowCount())} if key == self._class_key else {})
        self._class_key = key
        self.classes.setRowCount(len(values))
        for row, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.classes.setItem(row, 0, item)
            name, kind = old.get(value, (value, "USM" if value == "USM" else "US"))
            self.classes.setItem(row, 1, QTableWidgetItem(name))
            self.classes.setItem(row, 2, QTableWidgetItem(kind))

    def current(self):
        item = self.runs.currentItem()
        rid = item.data(Qt.ItemDataRole.UserRole) if item else None
        return next((r for r in results.records(self.plugin.chunk()) if r["id"] == rid), None)

    def draw_area(self):
        ch = self.plugin.chunk()
        if ch is None:
            raise ValueError(t("Apri un progetto", "Open a project"))
        self.plugin.context.store_drawings()
        self.refresh()

        def done(xy):
            from shapely.geometry import Polygon

            if ch is self.plugin.chunk():
                self.area = Polygon(xy).wkt
                self.area_label.setText(t("Area verificata pronta. Occorrono almeno quattro colonne di tasselli.",
                                         "Reviewed area ready. At least four tile columns are required."))
        if not self.plugin.context.request_polygon(done, hint=t("Delimita l'area completamente annotata",
                                                                 "Outline the fully annotated area")):
            raise ValueError(t("Mostra prima un raster nella vista 2D", "Display a raster in the 2D view first"))

    def prepare(self):
        ch = self.plugin.chunk()
        if ch is None or self.area is None:
            raise ValueError(t("Disegna prima l'area verificata", "Draw the reviewed area first"))
        self.plugin.context.store_drawings()
        raster, sources = self.selected_inputs()
        results.check_reference(ch, raster)
        mapping, kinds = {}, {}
        for row in range(self.classes.rowCount()):
            value, name, kind = [self.classes.item(row, c).text().strip() for c in range(3)]
            if name:
                mapping[value] = name
                if name in kinds and kinds[name] != kind:
                    raise ValueError(t("Una classe deve avere un solo tipo US", "Each class needs a single unit type"))
                kinds[name] = kind or "US"
        target, _ = QFileDialog.getSaveFileName(self, t("Nuova cartella dataset", "New dataset directory"))
        if not target:
            return
        # Snapshot drawing data and UI settings before the background task starts.
        units, field, area, tile = self.units_data(), self.field.currentText(), self.area, self.tile.value()
        self.plugin.context.run("GeoAI dataset", lambda p: dataset.prepare(raster, units, field,
                                mapping, area, target, tile_size=tile, kinds=kinds, progress=p, inputs=sources),
                                lambda m: QMessageBox.information(self, "GeoAI", f"{len(m['samples'])} tiles: {target}"))

    def train(self):
        if environment.where("geoai") is None:
            raise ValueError(t("Installa GeoAI dal menu Plugin > GeoAI", "Install GeoAI from Plugin > GeoAI"))
        source = QFileDialog.getExistingDirectory(self, t("Dataset preparato", "Prepared dataset"))
        if not source:
            return
        dataset.read_manifest(source)
        target, _ = QFileDialog.getSaveFileName(self, t("Nuova cartella modello .ppgseg", "New model directory .ppgseg"))
        if not target:
            return
        req = {"op": "train", "dataset": source, "output": target, "epochs": self.epochs.value()}
        self.plugin.context.run(t("Addestramento GeoAI", "GeoAI training"), lambda p: job(req, p),
                                lambda m: QMessageBox.information(self, "GeoAI", "IoU\n" + "\n".join(
                                    f"{k}: {v:.3f}" if v is not None else f"{k}: —"
                                    for k, v in m["metrics"]["iou"].items())))

    def apply(self):
        from .source_dialog import SourceDialog

        ch = self.plugin.chunk()
        if ch is None:
            raise ValueError(t("Apri un progetto", "Open a project"))
        model = QFileDialog.getExistingDirectory(self, t("Modello .ppgseg", "Model .ppgseg"))
        if model:
            meta = dataset.read_manifest(model, "model.json")
            sources = None
            if meta["format"] == 2:
                dialog = SourceDialog(self.available_rasters(), schema=meta["inputs"], parent=self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                sources = dialog.sources()
                raster = sources[0]["path"]
            else:
                raster = getattr(ch, "orthomosaic_path", None)
                if not raster:
                    raise ValueError(t("Questo modello richiede un'ortofoto RGB.",
                                       "This model requires an RGB orthomosaic."))
            self.plugin.context.run("GeoAI", lambda p: results.predict(ch, raster, model, progress=p, inputs=sources),
                                    lambda run: self.plugin.found(ch, run))

    def units(self):
        run = self.current()
        if run:
            self.plugin.units(run)

    def export(self):
        run = self.current()
        if run:
            target, _ = QFileDialog.getSaveFileName(self, "GeoPackage", filter="GeoPackage (*.gpkg)")
            if target:
                with Path(target).open("xb") as out, results.path(self.plugin.chunk(), run, "gpkg").open("rb") as src:
                    shutil.copyfileobj(src, out)

    def toggle(self):
        run = self.current()
        if run:
            run["visible"] = not run.get("visible", True)
            self.plugin.context.mark_modified()
            self.plugin.draw_all()
