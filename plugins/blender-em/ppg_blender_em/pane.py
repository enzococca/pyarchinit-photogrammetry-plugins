"""The Blender / EMtools pane: the em.json linked, the EM project, Blender, the live link and the
sync with EMtools, and the units of the matrix with the proxies of the chunk paired to them."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHeaderView, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import t

COLUMNS = (("Unità", "Unit"), ("Tipo", "Type"), ("Epoca", "Epoch"), ("Proxy", "Proxy"))


class BlenderEmPane(QWidget):
    link_requested = Signal()  # link an em.json
    export_requested = Signal()  # write the EM project
    open_requested = Signal()  # Blender on the EM project
    live_toggled = Signal(bool)
    sync_toggled = Signal(bool)
    selected = Signal(list)  # the node ids picked in the table

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("I proxy delle US e il rilievo nel progetto Extended Matrix (EM/proxies, RB), con i nomi dei "
                         "nodi dell'em.json che PyArchInit o EMStudio hanno scritto: EMtools li abbina alle unità. "
                         "Blender si aggiorna da solo quando i proxy cambiano.",
                         "The proxies of the units and the survey into the Extended Matrix project (EM/proxies, RB), "
                         "named after the nodes of the em.json PyArchInit or EMStudio wrote: EMtools pairs them with "
                         "the units. Blender follows by itself when the proxies change."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.l_matrix = QLabel("")
        self.l_matrix.setWordWrap(True)
        self.l_matrix.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.l_matrix)
        holder = QWidget()
        bar = FlowLayout(holder)
        self.b_link = QPushButton(t("Collega em.json...", "Link em.json..."))
        self.b_link.setToolTip(t("L'em.json del sito (PyArchInit: Extended Matrix > Esporta sito in em.json; EMStudio; "
                                 "EMtools 1.6)",
                                 "The em.json of the site (PyArchInit: Extended Matrix > Esporta sito in em.json; EMStudio; "
                                 "EMtools 1.6)"))
        self.b_export = QPushButton(t("Esporta nel progetto EM", "Export into the EM project"))
        self.b_export.setToolTip(t("EM/proxies/<unità>.glb e RB/<chunk>/<chunk>.glb, spostati di un numero intero di "
                                   "metri, e il manifest per Blender",
                                   "EM/proxies/<unit>.glb and RB/<chunk>/<chunk>.glb, shifted by whole metres, and the "
                                   "manifest for Blender"))
        self.b_open = QPushButton(t("Apri in Blender", "Open in Blender"))
        self.b_open.setToolTip(t("Blender con i proxy e il rilievo, aggiornati dal vivo",
                                 "Blender with the proxies and the survey, kept up to date"))
        for b in (self.b_link, self.b_export, self.b_open):
            bar.addWidget(b)
        lay.addWidget(holder)
        holder2 = QWidget()
        checks = FlowLayout(holder2)
        self.c_live = QCheckBox(t("Aggiornamento dal vivo", "Live update"))
        self.c_live.setToolTip(t("I proxy ricostruiti e il modello cambiato riscritti nel progetto EM: Blender li "
                                 "ricarica entro un secondo",
                                 "The proxies built again and the model changed written into the EM project: Blender "
                                 "reloads them within a second"))
        self.c_sync = QCheckBox(t("Sincronizza con EMtools", "Sync with EMtools"))
        self.c_sync.setToolTip(t("La selezione condivisa con EMtools 1.6 (Sync in Blender, porta 8788): un proxy scelto "
                                 "in Blender seleziona la sua unità qui, e viceversa",
                                 "The selection shared with EMtools 1.6 (Sync in Blender, port 8788): a proxy picked in "
                                 "Blender selects its unit here, and back"))
        checks.addWidget(self.c_live)
        checks.addWidget(self.c_sync)
        lay.addWidget(holder2)
        self.l_status = QLabel("")
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        self.l_unit = QLabel("")
        self.l_unit.setWordWrap(True)
        self.l_unit.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.l_unit)
        self.b_link.clicked.connect(self.link_requested)
        self.b_export.clicked.connect(self.export_requested)
        self.b_open.clicked.connect(self.open_requested)
        self.c_live.toggled.connect(self.live_toggled)
        self.c_sync.toggled.connect(self.sync_toggled)
        self.table.itemSelectionChanged.connect(self._picked)
        self._holders = (holder,)

    def set_busy(self, busy: bool) -> None:
        for h in self._holders:
            h.setEnabled(not busy)

    def set_matrix(self, text: str) -> None:
        self.l_matrix.setText(text)

    def set_status(self, text: str) -> None:
        self.l_status.setText(text)

    def set_checks(self, live: bool, sync: bool) -> None:
        for box, value in ((self.c_live, live), (self.c_sync, sync)):
            box.blockSignals(True)
            box.setChecked(bool(value))
            box.blockSignals(False)

    def set_rows(self, rows: list[dict]) -> None:
        """``rows``: ``{"name", "tipo", "epoch", "proxy", "node_id", "color", "info"}``."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, key in enumerate(("name", "tipo", "epoch", "proxy")):
                item = QTableWidgetItem(str(row.get(key) or ""))
                item.setData(Qt.ItemDataRole.UserRole, row.get("node_id"))
                item.setToolTip(row.get("info") or "")
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)

    def node_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        out = []
        for r in rows:
            item = self.table.item(r, 0)
            nid = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if nid:
                out.append(str(nid))
        return out

    def select_nodes(self, node_ids) -> list[str]:
        """The rows of ``node_ids`` selected (a selection made in Blender), without sending it back;
        the names of the units found."""
        wanted = {str(n) for n in node_ids}
        self.table.blockSignals(True)
        self.table.clearSelection()
        found = []
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") in wanted:
                self.table.selectRow(r)
                if not found:
                    self.table.scrollToItem(item)
                found.append(item.text())
        self.table.setSelectionMode(mode)
        self.table.blockSignals(False)
        return found

    def set_unit(self, text: str) -> None:
        self.l_unit.setText(text)

    def _picked(self) -> None:
        ids = self.node_ids()
        if ids:
            self.selected.emit(ids)
