"""The window of the plugin: the database (a GeoPackage or SpatiaLite file, or a PostgreSQL/PostGIS
server) and what to save. Everything but the password is remembered for the next time."""
from __future__ import annotations

from urllib.parse import quote

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import t

KINDS = [("gpkg", "GeoPackage (.gpkg)"), ("postgis", "PostgreSQL / PostGIS"), ("spatialite", "SpatiaLite (.sqlite)")]
WHAT = [("quote", ("Quote", "Spot heights")), ("curve", ("Curve di livello", "Contour lines")),
        ("us", ("US disegnate", "Units drawn")), ("sezioni", ("Sezioni", "Sections")),
        ("pietre", ("Pietre", "Stones")), ("marker", ("Marker", "Markers")), ("dem", ("DEM", "DEM")),
        ("ortofoto", ("Ortofoto", "Orthomosaic"))]


class ArchiveDialog(QDialog):
    def __init__(self, parent, settings: dict):
        super().__init__(parent)
        s = dict(settings or {})
        self.setWindowTitle(t("Archivio GIS", "GIS archive"))
        lay = QVBoxLayout(self)
        intro = QLabel(t("Il rilievo del progetto in un database GIS diverso da PyArchInit: ogni dato in una tabella, "
                         "con progetto, chunk e sito. Archiviando di nuovo lo stesso progetto le sue righe vengono "
                         "sostituite e restano quelle degli altri progetti.",
                         "The survey of the project into a GIS database other than PyArchInit: each kind of data in a "
                         "table, with project, chunk and site. Archiving the same project again replaces its rows and "
                         "keeps those of the other projects."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        # ---------------------------------------------------------------- the database
        box = QGroupBox(t("Database", "Database"))
        form = QFormLayout(box)
        self.kind = QComboBox()
        for code, label in KINDS:
            self.kind.addItem(label, code)
        self.kind.setCurrentIndex(max(0, self.kind.findData(s.get("kind", "gpkg"))))
        form.addRow(t("Tipo", "Type"), self.kind)
        self.pages = QStackedWidget()
        file_page = QWidget()
        row = QHBoxLayout(file_page)
        row.setContentsMargins(0, 0, 0, 0)
        self.file = QLineEdit(s.get("file", ""))
        browse = QPushButton(t("Sfoglia...", "Browse..."))
        browse.clicked.connect(self._browse)
        row.addWidget(self.file, 1)
        row.addWidget(browse)
        server_page = QWidget()
        grid = QFormLayout(server_page)
        grid.setContentsMargins(0, 0, 0, 0)
        self.host = QLineEdit(s.get("host", "localhost"))
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(s.get("port", 5432)))
        self.database = QLineEdit(s.get("database", ""))
        self.user = QLineEdit(s.get("user", "postgres"))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText(t("non viene salvata", "never saved"))
        self.schema = QLineEdit(s.get("schema", "public"))
        for label, widget in ((t("Server", "Host"), self.host), (t("Porta", "Port"), self.port),
                              (t("Database", "Database"), self.database), (t("Utente", "User"), self.user),
                              (t("Password", "Password"), self.password), (t("Schema", "Schema"), self.schema)):
            grid.addRow(label, widget)
        self.pages.addWidget(file_page)
        self.pages.addWidget(server_page)
        form.addRow(self.pages)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        lay.addWidget(box)
        # ---------------------------------------------------------------- what
        box = QGroupBox(t("Cosa salvare", "What to save"))
        grid = QGridLayout(box)
        chosen = set(s.get("what", [code for code, _ in WHAT]))
        self.checks = {}
        for i, (code, (it, en)) in enumerate(WHAT):
            check = QCheckBox(t(it, en))
            check.setChecked(code in chosen)
            self.checks[code] = check
            grid.addWidget(check, i // 4, i % 4)
        lay.addWidget(box)
        box = QGroupBox(t("Opzioni", "Options"))
        form = QFormLayout(box)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.0, 1000.0)
        self.interval.setDecimals(2)
        self.interval.setSingleStep(0.1)
        self.interval.setSpecialValueText(t("suggerita dal DEM", "suggested from the DEM"))
        self.interval.setValue(float(s.get("interval", 0.0)))
        form.addRow(t("Equidistanza delle curve (m)", "Contour interval (m)"), self.interval)
        self.rasters = QComboBox()
        self.rasters.addItem(t("copiati nel database", "copied into the database"), "copy")
        self.rasters.addItem(t("solo riferimento (file e impronta)", "reference only (file and footprint)"), "reference")
        self.rasters.setCurrentIndex(0 if s.get("rasters", "copy") == "copy" else 1)
        form.addRow(t("Raster", "Rasters"), self.rasters)
        self.max_side = QSpinBox()
        self.max_side.setRange(0, 200000)
        self.max_side.setSingleStep(1024)
        self.max_side.setSpecialValueText(t("piena risoluzione", "full resolution"))
        self.max_side.setValue(int(s.get("max_side", 0)))
        form.addRow(t("Lato massimo dei raster copiati (pixel)", "Largest side of the copied rasters (pixels)"),
                    self.max_side)
        self.prefix = QLineEdit(s.get("prefix", "ppg_"))
        form.addRow(t("Prefisso delle tabelle", "Prefix of the tables"), self.prefix)
        self.replace = QCheckBox(t("Sostituisci i dati gia' archiviati di questo progetto e chunk",
                                   "Replace the rows already archived for this project and chunk"))
        self.replace.setChecked(bool(s.get("replace", True)))
        form.addRow(self.replace)
        lay.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(t("Salva", "Save"))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._kind_changed()

    def _kind(self) -> str:
        return self.kind.currentData()

    def _kind_changed(self) -> None:
        self.pages.setCurrentIndex(1 if self._kind() == "postgis" else 0)
        rasters_in_db = self._kind() != "spatialite"
        self.rasters.setEnabled(rasters_in_db)
        if not rasters_in_db:
            self.rasters.setCurrentIndex(1)

    def _browse(self) -> None:
        if self._kind() == "gpkg":
            path, _ = QFileDialog.getSaveFileName(self, "GeoPackage", self.file.text(), "GeoPackage (*.gpkg)",
                                                  options=QFileDialog.Option.DontConfirmOverwrite)
            suffix = ".gpkg"
        else:
            path, _ = QFileDialog.getSaveFileName(self, "SpatiaLite", self.file.text(), "SpatiaLite (*.sqlite *.db)",
                                                  options=QFileDialog.Option.DontConfirmOverwrite)
            suffix = ".sqlite"
        if path:
            self.file.setText(path if "." in path.rsplit("/", 1)[-1] else path + suffix)

    def _accept(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import message

        if self._kind() != "postgis" and not self.file.text().strip():
            message(self, "warning", self.windowTitle(), t("Scegli il file del database.", "Choose the database file."))
            return
        if self._kind() == "postgis" and not self.database.text().strip():
            message(self, "warning", self.windowTitle(), t("Scrivi il nome del database.", "Write the database name."))
            return
        if not any(c.isChecked() for c in self.checks.values()):
            message(self, "warning", self.windowTitle(), t("Scegli almeno un dato da salvare.",
                                                           "Choose at least one thing to save."))
            return
        self.accept()

    def target(self) -> str:
        """The database as ``archive()`` takes it (the password apart)."""
        kind = self._kind()
        if kind == "postgis":
            user = quote(self.user.text().strip(), safe="")
            return (f"postgresql://{user}@{self.host.text().strip()}:{self.port.value()}/"
                    f"{quote(self.database.text().strip(), safe='')}?schema={quote(self.schema.text().strip() or 'public')}")
        path = self.file.text().strip()
        return path if kind == "gpkg" else f"spatialite:{path}"

    def options(self) -> dict:
        return {"target": self.target(), "password": self.password.text() or None,
                "what": [code for code, check in self.checks.items() if check.isChecked()],
                "interval": self.interval.value() or None, "prefix": self.prefix.text().strip() or "ppg_",
                "replace": self.replace.isChecked(), "rasters": self.rasters.currentData(),
                "max_side": self.max_side.value()}

    def remembered(self) -> dict:
        """The choices kept for the next time: everything but the password."""
        return {"kind": self._kind(), "file": self.file.text().strip(), "host": self.host.text().strip(),
                "port": self.port.value(), "database": self.database.text().strip(), "user": self.user.text().strip(),
                "schema": self.schema.text().strip(), "what": [c for c, ch in self.checks.items() if ch.isChecked()],
                "interval": self.interval.value(), "rasters": self.rasters.currentData(),
                "max_side": self.max_side.value(), "prefix": self.prefix.text().strip(),
                "replace": self.replace.isChecked()}
