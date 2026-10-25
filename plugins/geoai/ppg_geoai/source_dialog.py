"""Ordered raster sources for training, or new files matching a trained model's bands."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .texts import t

KINDS = [("rgb", "RGB"), ("dem", "DEM"), ("multispectral", t("Multispettrale", "Multispectral")),
         ("thermal", t("Termico", "Thermal")), ("raster", t("Altro raster", "Other raster"))]
RESAMPLING = [("bilinear", t("Bilineare", "Bilinear")), ("nearest", t("Più vicino", "Nearest")),
              ("cubic", t("Cubico", "Cubic")), ("average", t("Media", "Average"))]


class PathPicker(QWidget):
    def __init__(self, rasters, path="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.addItem("")
        for raster in rasters:
            value = str(raster["path"])
            self.combo.addItem(value)
            self.combo.setItemData(self.combo.count() - 1, raster.get("label", value), Qt.ItemDataRole.ToolTipRole)
        self.combo.setCurrentText(str(path))
        layout.addWidget(self.combo, 1)
        browse = QPushButton("…")
        browse.setMaximumWidth(32)
        browse.clicked.connect(self.browse)
        layout.addWidget(browse)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, t("Raster di origine", "Source raster"),
                                             filter="GeoTIFF (*.tif *.tiff);;" + t("Tutti i file (*)", "All files (*)"))
        if path:
            self.combo.setCurrentText(path)

    def path(self):
        return self.combo.currentText().strip()


class SourceDialog(QDialog):
    def __init__(self, rasters, sources=None, schema=None, parent=None):
        super().__init__(parent)
        self.rasters = rasters
        self.schema = schema
        self.setWindowTitle(t("Dati raster GeoAI", "GeoAI raster inputs"))
        self.resize(1050, 380)
        layout = QVBoxLayout(self)
        hint = QLabel(t("La prima sorgente definisce griglia, estensione e risoluzione. Le altre vengono allineate "
                        "a quella griglia; controlla prima la registrazione geografica. Mantieni ordine delle bande "
                        "e unità coerenti, ad esempio °C o K per la temperatura. Usa i valori originali dei raster.",
                        "The first source defines the grid, extent and resolution. Other sources are aligned to "
                        "that grid; check geographic registration first. Keep band order and units consistent, "
                        "for example °C or K for temperature. Use original raster values."))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        if schema is not None:
            remap = QLabel(t("Questo modello richiede le sorgenti elencate. Scegli i file del progetto attuale: "
                             "nomi, bande e unità devono corrispondere all'addestramento.",
                             "This model requires the listed sources. Choose the current project's files: "
                             "names, bands and units must match training."))
            remap.setWordWrap(True)
            layout.addWidget(remap)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([t("Nome", "Name"), t("Tipo", "Kind"), "Raster",
                                              t("Bande (1,2,…)", "Bands (1,2,…)"),
                                              t("Ricampionamento", "Resampling"), t("Unità", "Units")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(2, 360)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        for title, callback in [(t("Aggiungi", "Add"), lambda: self.add_source()),
                                (t("Rimuovi", "Remove"), self.remove_source),
                                (t("Su", "Up"), lambda: self.move_source(-1)),
                                (t("Giù", "Down"), lambda: self.move_source(1))]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, fn=callback: fn())
            button.setEnabled(schema is None)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        footer.accepted.connect(self.validate_accept)
        footer.rejected.connect(self.reject)
        layout.addWidget(footer)
        initial = [{**row, "path": ""} for row in schema] if schema is not None else list(sources or [])
        for source in initial:
            self.add_source(source)
        if not initial:
            self.add_source()

    def add_source(self, source=None):
        row = self.table.rowCount()
        source = source or {"name": f"input_{row + 1}", "kind": "raster", "path": "", "bands": [1]}
        self.table.insertRow(row)
        name = QTableWidgetItem(source["name"])
        name.setData(Qt.ItemDataRole.UserRole, dict(source))
        bands = QTableWidgetItem(",".join(str(b) for b in source["bands"]))
        unit = QTableWidgetItem(source.get("unit", ", ".join(source.get("units", []))))
        for column, item in [(0, name), (3, bands), (5, unit)]:
            if self.schema is not None:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, column, item)
        for column, options, chosen in [(1, KINDS, source.get("kind", "raster")),
                                         (4, RESAMPLING, source.get("resampling", "bilinear"))]:
            combo = QComboBox()
            for value, label in options:
                combo.addItem(label, value)
            combo.setCurrentIndex(max(0, combo.findData(chosen)))
            combo.setEnabled(self.schema is None)
            self.table.setCellWidget(row, column, combo)
        self.table.setCellWidget(row, 2, PathPicker(self.rasters, source.get("path", "")))
        self.table.setCurrentCell(row, 0)

    def sources(self, validate=True):
        result = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            path = self.table.cellWidget(row, 2).path()
            try:
                bands = [int(value.strip()) for value in self.table.item(row, 3).text().split(",")]
            except ValueError:
                raise ValueError(t("Indica le bande con numeri separati da virgole, ad esempio 1,2,3.",
                                   "Enter comma-separated band numbers, for example 1,2,3.")) from None
            source = {"name": name, "kind": self.table.cellWidget(row, 1).currentData(), "path": path,
                      "bands": bands, "resampling": self.table.cellWidget(row, 4).currentData()}
            if self.schema is not None:
                source = {**self.table.item(row, 0).data(Qt.ItemDataRole.UserRole), "path": path}
            else:
                unit = self.table.item(row, 5).text().strip()
                if unit:
                    source["unit"] = unit
            if validate:
                if not name or name in {s["name"] for s in result}:
                    raise ValueError(t("Ogni sorgente deve avere un nome diverso e non vuoto.",
                                       "Each source needs a unique non-empty name."))
                if not path or not Path(path).is_file():
                    raise ValueError(t("Scegli un file raster esistente per {0}.",
                                       "Choose an existing raster file for {0}.").format(name))
                if not bands or any(b < 1 for b in bands) or len(set(bands)) != len(bands):
                    raise ValueError(t("Le bande devono essere numeri positivi, senza duplicati.",
                                       "Bands must be positive numbers without duplicates."))
                import rasterio

                with rasterio.open(path) as src:
                    if max(bands) > src.count:
                        raise ValueError(t("{0} contiene soltanto {1} bande.",
                                           "{0} contains only {1} bands.").format(name, src.count))
            result.append(source)
        if validate and not result:
            raise ValueError(t("Scegli almeno una sorgente raster.", "Choose at least one raster source."))
        return result

    def remove_source(self):
        if self.schema is None and self.table.currentRow() >= 0:
            self.table.removeRow(self.table.currentRow())

    def move_source(self, delta):
        row = self.table.currentRow()
        target = row + delta
        if self.schema is not None or not 0 <= target < self.table.rowCount():
            return
        sources = self.sources(validate=False)
        sources[row], sources[target] = sources[target], sources[row]
        self.table.setRowCount(0)
        for source in sources:
            self.add_source(source)
        self.table.setCurrentCell(target, 0)

    def validate_accept(self):
        try:
            self.sources()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "GeoAI", str(exc))
            return
        self.accept()
