"""The *Conservazione* pane: the elevations of the chunk and the campaigns of the one chosen, the term chooser (a
search in both languages, favourites, the last ten, the terms grouped by family), the buttons of the mapping, of the
interventions, of the EN 16096 assessment and of the outputs, the table of the campaign's features and its summary. It
only says what the user asks (``requested``): the plugin does it."""
from __future__ import annotations

import unicodedata

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from .texts import t

CAMPAIGN_ACTIONS = (
    ("new_campaign", ("Nuova campagna...", "New campaign..."), "Un rilievo del prospetto scelto in una data: chi, "
     "come, con che tempo", "A survey of the chosen elevation on a date: who, how, in what weather"),
    ("edit_campaign", ("Modifica...", "Edit..."), "Data, nome, rilevatore, metodo e note della campagna",
     "Date, name, surveyor, method and notes of the campaign"),
    ("remove_campaign", ("Rimuovi", "Remove"), "La campagna con i suoi elementi (Modifica > Annulla la riporta)",
     "The campaign with its features (Edit > Undo brings it back)"),
    ("show", ("Mostra", "Show"), "Il prospetto con gli elementi della campagna disegnati",
     "The elevation with the features of the campaign drawn"))
MAP_ACTIONS = (
    ("area", ("Area", "Area"), "Disegna sul prospetto l'area del termine scelto", "Draw on the elevation the area of "
     "the chosen term"),
    ("line", ("Linea", "Line"), "Disegna una lesione (fessura, giunto aperto) come linea",
     "Draw a crack (or an open joint) as a line"),
    ("point", ("Punto", "Point"), "Una nota in un punto del prospetto, con il termine scelto",
     "A note at a point of the elevation, with the chosen term"),
    ("propose", ("Proposte...", "Proposals..."), "Aree, lesioni e lacune proposte dall'immagine e dal rilievo",
     "Areas, cracks and losses proposed from the image and the relief"),
    ("accept", ("Accetta", "Accept"), "Le proposte scelte nella tabella entrano nella campagna",
     "The proposals chosen in the table join the campaign"),
    ("accept_all", ("Accetta tutte", "Accept all"), "Tutte le proposte della campagna",
     "Every proposal of the campaign"),
    ("discard", ("Scarta", "Discard"), "Gli elementi scelti nella tabella escono dalla campagna",
     "The features chosen in the table leave the campaign"),
    ("edit", ("Modifica...", "Edit..."), "Termine, intensità, apertura, causa, urgenza e note dell'elemento scelto",
     "Term, intensity, opening, cause, urgency and notes of the chosen feature"),
    ("photos", ("Foto", "Photos"), "Apre la foto che vede meglio l'elemento scelto, centrata su di esso",
     "Opens the photo that sees the chosen feature best, centred on it"))
TERM_ACTIONS = (("favourite", ("Preferito sì/no", "Favourite on/off"), "Il termine scelto fra i preferiti (o fuori)",
                 "The chosen term among the favourites (or out)"),)
TREAT_ACTIONS = (
    ("treat_area", ("Intervento (area)", "Treatment (area)"), "Un'area trattata o da trattare: tecnica, materiali, "
     "data, eseguito o da eseguire", "An area treated or to treat: technique, materials, date, done or to do"),
    ("treat_line", ("Intervento (linea)", "Treatment (line)"), "Una lesione sigillata o da sigillare",
     "A crack sealed or to seal"),
    ("to_treat", ("Da restaurare", "To treat"), "Un intervento sulla geometria dell'elemento di degrado scelto, con la "
     "tecnica suggerita", "A treatment on the geometry of the chosen decay feature, with the technique suggested"))
ASSESS_ACTIONS = (
    ("classes", ("Classi CC e UC...", "Classes CC and UC..."), "Condizione e urgenza della faccia (EN 16096): le "
     "scegli tu, il plugin mostra un suggerimento", "Condition and urgency of the face (EN 16096): you choose them, "
     "the plugin shows a suggestion"),
    ("assessment", ("Raccomandazione RC...", "Recommendation RC..."), "La classe di raccomandazione dell'oggetto "
     "rilevato in quella data (una per data)", "The recommendation class of the object surveyed on that date (one "
     "per date)"))
OUTPUT_ACTIONS = (
    ("plate", ("Tavola...", "Plate..."), "La tavola della campagna con la legenda (PDF, SVG)",
     "The plate of the campaign with its legend (PDF, SVG)"),
    ("tables", ("Tabelle...", "Tables..."), "Elementi, aree per termine e famiglia, lesioni, interventi (Excel, CSV)",
     "Features, areas per term and family, cracks, treatments (Excel, CSV)"),
    ("drawings", ("Disegni...", "Drawings..."), "DXF e GeoPackage (spiegato e 3D)", "DXF and GeoPackages (unfolded "
     "and 3D)"),
    ("qgis", ("Apri in QGIS", "Open in QGIS"), "Il prospetto con gli elementi e la legenda in QGIS",
     "The elevation with the features and the legend in QGIS"),
    ("pyarchinit", ("Salva in PyArchInit", "Save in PyArchInit"), "Tavola come media del prospetto, stato di "
     "conservazione delle US collegate", "Plate as media of the elevation, condition of the linked units"),
    ("ifc", ("IFC 4.3...", "IFC 4.3..."), "Gli elementi come IfcSurfaceFeature sul muro", "The features as "
     "IfcSurfaceFeature on the wall"),
    ("compare", ("Confronta...", "Compare..."), "Due campagne della stessa faccia", "Two campaigns of the same face"),
    ("folder", ("Apri la cartella", "Open the folder"), "La cartella dei file della campagna",
     "The folder of the campaign's files"))
COLUMNS = (("Termine", "Term"), ("Tipo", "Kind"), ("Estensione", "Extent"), ("Intensità", "Intensity"),
           ("Stato", "Status"), ("Foto", "Photos"), ("Note", "Notes"))
KIND_TITLES = {"decay": ("degrado", "decay"), "crack": ("lesione", "crack"), "treatment": ("intervento", "treatment"),
               "note": ("nota", "note")}
STATUS_TITLES = {"accepted": ("accettato", "accepted"), "proposed": ("proposto", "proposed")}
INTENSITY_TITLES = {"low": ("bassa", "low"), "medium": ("media", "medium"), "high": ("alta", "high")}
ROLE_SEARCH = Qt.ItemDataRole.UserRole + 1
ROLE_SORT = Qt.ItemDataRole.UserRole + 2


def fold(text: str) -> str:
    """Lower case without accents: what the search of the term chooser compares."""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def title(table: dict, key) -> str:
    pair = table.get(key)
    return t(*pair) if pair else ("-" if key in (None, "") else str(key))


class _Item(QTableWidgetItem):
    """A cell sorted by its number (the extent) when it has one."""

    def __lt__(self, other):
        a, b = self.data(ROLE_SORT), other.data(ROLE_SORT)
        if a is not None and b is not None:
            return float(a) < float(b)
        if (a is None) != (b is None):
            return a is None
        return fold(self.text()) < fold(other.text())


class ConservationPane(QWidget):
    requested = Signal(str)  # an action of the tuples above
    elevation_changed = Signal(str)
    campaign_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Il degrado di una faccia si mappa sul suo prospetto, campagna per campagna, con i termini "
                         "di ICOMOS-ISCS e UNI 11182. Le classi EN 16096 le sceglie chi rileva.",
                         "The decay of a face is mapped on its elevation, campaign by campaign, with the terms of "
                         "ICOMOS-ISCS and UNI 11182. The EN 16096 classes are the surveyor's choice."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.buttons: dict[str, QPushButton] = {}
        lay.addWidget(QLabel(f"<b>{t('Prospetto e campagna', 'Elevation and campaign')}</b>"))
        self.elevations = self._combo()
        self.elevations.currentIndexChanged.connect(lambda _i: self.elevation_changed.emit(self.elevation_id() or ""))
        lay.addWidget(self.elevations)
        self.campaigns = self._combo()
        self.campaigns.currentIndexChanged.connect(lambda _i: self.campaign_changed.emit(self.campaign_id() or ""))
        lay.addWidget(self.campaigns)
        lay.addWidget(self._flow(CAMPAIGN_ACTIONS))
        lay.addWidget(QLabel(f"<b>{t('Mappatura', 'Mapping')}</b>"))
        self.search = QLineEdit()
        self.search.setPlaceholderText(t("Cerca un termine (italiano o inglese)", "Search a term (Italian or English)"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_terms)
        lay.addWidget(self.search)
        self.terms = QTreeWidget()
        self.terms.setHeaderHidden(True)
        self.terms.setMinimumHeight(160)
        self.terms.setProperty("ppg_no_translate", True)
        lay.addWidget(self.terms)
        lay.addWidget(self._flow(TERM_ACTIONS))
        lay.addWidget(self._flow(MAP_ACTIONS))
        filters = QWidget()
        row = QHBoxLayout(filters)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(t("Famiglia", "Family")))
        self.family = self._combo()
        self.family.currentIndexChanged.connect(lambda _i: self._filter_rows())
        row.addWidget(self.family, 1)
        lay.addWidget(filters)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.itemDoubleClicked.connect(lambda _i: self.requested.emit("edit"))
        self.table.setMinimumHeight(140)
        lay.addWidget(self.table)
        lay.addWidget(QLabel(f"<b>{t('Interventi', 'Interventions')}</b>"))
        lay.addWidget(self._flow(TREAT_ACTIONS))
        lay.addWidget(QLabel(f"<b>{t('Valutazione (EN 16096)', 'Assessment (EN 16096)')}</b>"))
        lay.addWidget(self._flow(ASSESS_ACTIONS))
        lay.addWidget(QLabel(f"<b>{t('Esiti', 'Outputs')}</b>"))
        lay.addWidget(self._flow(OUTPUT_ACTIONS))
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(110)
        lay.addWidget(self.summary)
        self.l_state = QLabel("")
        self.l_state.setWordWrap(True)
        lay.addWidget(self.l_state)
        lay.addStretch(1)
        self._elevation_ids: list[str] = []
        self._campaign_ids: list[str] = []
        self.set_families([])

    @staticmethod
    def _combo() -> QComboBox:
        box = QComboBox()
        box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(12)
        box.setProperty("ppg_no_translate", True)
        return box

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

    # ------------------------------------------------------------------ elevations and campaigns
    def set_elevations(self, rows: list[dict], current: str | None = None) -> None:
        """``rows``: ``{"id", "label"}`` of the chunk's elevations."""
        keep = current or self.elevation_id()
        self.elevations.blockSignals(True)
        self.elevations.clear()
        self._elevation_ids = [r["id"] for r in rows]
        for r in rows:
            self.elevations.addItem(r["label"], r["id"])
        if keep in self._elevation_ids:
            self.elevations.setCurrentIndex(self._elevation_ids.index(keep))
        self.elevations.blockSignals(False)
        self.buttons["new_campaign"].setEnabled(bool(rows))
        self.buttons["show"].setEnabled(bool(rows))

    def elevation_id(self) -> str | None:
        i = self.elevations.currentIndex()
        return self._elevation_ids[i] if 0 <= i < len(self._elevation_ids) else None

    def set_campaigns(self, rows: list[dict], current: str | None = None) -> None:
        """``rows``: ``{"id", "date", "name"}`` of the campaigns of the elevation chosen."""
        keep = current or self.campaign_id()
        self.campaigns.blockSignals(True)
        self.campaigns.clear()
        self._campaign_ids = [r["id"] for r in rows]
        for r in rows:
            self.campaigns.addItem(f"{r['date']}  {r['name']}", r["id"])
        if keep in self._campaign_ids:
            self.campaigns.setCurrentIndex(self._campaign_ids.index(keep))
        elif rows:
            self.campaigns.setCurrentIndex(len(rows) - 1)
        self.campaigns.blockSignals(False)
        names = [a[0] for a in MAP_ACTIONS + TREAT_ACTIONS + ASSESS_ACTIONS + OUTPUT_ACTIONS] + \
            ["edit_campaign", "remove_campaign"]
        for name in names:
            self.buttons[name].setEnabled(bool(rows))

    def campaign_id(self) -> str | None:
        i = self.campaigns.currentIndex()
        return self._campaign_ids[i] if 0 <= i < len(self._campaign_ids) else None

    def select_campaign(self, cid: str) -> None:
        if cid in self._campaign_ids:
            self.campaigns.setCurrentIndex(self._campaign_ids.index(cid))

    # ------------------------------------------------------------------ the term chooser
    def set_terms(self, groups: list[tuple]) -> None:
        """``groups``: ``(title, rows, expanded)``, each row ``{"code", "label", "tip", "search", "children"}``;
        the code chosen is kept."""
        keep = self.term_code()
        self.terms.blockSignals(True)
        self.terms.clear()

        def add(parent, row):
            item = QTreeWidgetItem(parent, [row["label"]])
            item.setData(0, Qt.ItemDataRole.UserRole, row["code"])
            item.setData(0, ROLE_SEARCH, fold(row.get("search") or row["label"]))
            if row.get("tip"):
                item.setToolTip(0, row["tip"])
            for child in row.get("children") or []:
                add(item, child)
            return item

        for name, rows, expanded in groups:
            top = QTreeWidgetItem(self.terms, [name])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            for row in rows:
                add(top, row)
            top.setExpanded(bool(expanded))
        self.terms.blockSignals(False)
        if keep:
            self.choose_term(keep)
        self._filter_terms(self.search.text())

    def _leaves(self):
        stack = [self.terms.topLevelItem(i) for i in range(self.terms.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if item.data(0, Qt.ItemDataRole.UserRole):
                yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def term_code(self) -> str | None:
        item = self.terms.currentItem()
        return None if item is None else item.data(0, Qt.ItemDataRole.UserRole) or None

    def choose_term(self, code: str) -> bool:
        for item in self._leaves():
            if item.data(0, Qt.ItemDataRole.UserRole) == code:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.terms.setCurrentItem(item)
                return True
        return False

    def _filter_terms(self, text: str) -> None:
        needle = fold(text).strip()

        def walk(item) -> bool:
            own = bool(item.data(0, Qt.ItemDataRole.UserRole)) and (not needle or needle in item.data(0, ROLE_SEARCH))
            kids = [walk(item.child(i)) for i in range(item.childCount())]
            show = own or any(kids)
            item.setHidden(not show)
            if needle and any(kids):
                item.setExpanded(True)
            return show

        for i in range(self.terms.topLevelItemCount()):
            top = self.terms.topLevelItem(i)
            if not needle:
                top.setHidden(False)
                for k in range(top.childCount()):
                    walk(top.child(k))
            else:
                walk(top)

    # ------------------------------------------------------------------ the features
    def set_families(self, rows: list[tuple[str, str]]) -> None:
        """``rows``: ``(family, title)`` offered by the filter of the table."""
        keep = self.family.currentData()
        self.family.blockSignals(True)
        self.family.clear()
        self.family.addItem(t("Tutte", "All"), "")
        for key, name in rows:
            self.family.addItem(name, key)
        i = self.family.findData(keep) if keep else 0
        self.family.setCurrentIndex(max(i, 0))
        self.family.blockSignals(False)

    def set_features(self, rows: list[dict]) -> None:
        """``rows``: ``{"id", "term", "kind", "extent" (number or None), "unit", "intensity", "status", "photos",
        "notes", "family"}``; the selection is kept."""
        from .texts import num

        keep = set(self.selected_ids())
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            extent = r.get("extent")
            cells = (r["term"], title(KIND_TITLES, r["kind"]),
                     "-" if extent is None else f"{num(extent, 3)} {r.get('unit', '')}".strip(),
                     title(INTENSITY_TITLES, r.get("intensity")), title(STATUS_TITLES, r["status"]),
                     str(r.get("photos") or 0), r.get("notes") or "")
            for j, text in enumerate(cells):
                item = _Item(text)
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, r["id"])
                    item.setData(ROLE_SEARCH, r.get("family") or "")
                if j == 2:
                    item.setData(ROLE_SORT, extent)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True)
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).data(Qt.ItemDataRole.UserRole) in keep:
                self.table.selectRow(i)
        self.table.blockSignals(False)
        self._filter_rows()

    def _filter_rows(self) -> None:
        family = self.family.currentData() or ""
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            self.table.setRowHidden(i, bool(family) and item is not None and item.data(ROLE_SEARCH) != family)

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        out = []
        for r in rows:
            item = self.table.item(r, 0)
            if item is not None and not self.table.isRowHidden(r):
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def select(self, fid: str) -> None:
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == fid:
                self.table.clearSelection()
                self.table.selectRow(i)
                return

    def set_summary(self, text: str) -> None:
        self.summary.setPlainText(text)

    def set_state(self, text: str) -> None:
        self.l_state.setText(text)


def feature_dialog(parent, title_text: str, fields: list, intro: str, photos: list[dict], open_photo):
    """The form of a feature (:class:`ParamDialog`) with the photos that see it below, each with its *apri*
    button (``open_photo(photo)``); returns the values or None."""
    from PySide6.QtWidgets import QDialog

    from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

    from .texts import num

    dialog = ParamDialog(title_text, fields, parent, intro)
    if photos:
        box = QWidget()
        rows = QVBoxLayout(box)
        rows.setContentsMargins(0, 0, 0, 0)
        heading = t("Foto che vedono l'elemento", "Photos that see the feature")
        rows.addWidget(QLabel(f"<b>{heading}</b>"))
        for p in photos:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(f"{p['label']}  {num(p['distance'], 1)} m, {num(p['angle'], 0)}°"), 1)
            b = QPushButton(t("apri", "open"))
            b.clicked.connect(lambda _=False, photo=p: open_photo(photo))
            h.addWidget(b)
            rows.addWidget(line)
        dialog.layout().insertWidget(dialog.layout().count() - 1, box)
    return dialog.values() if dialog.exec() == QDialog.DialogCode.Accepted else None
