"""The *Relitto* pane: the steps of the drawing of a hull, one under the other, each with its buttons and its
state, the table of the timbers under the catalogue's step, then the clinker strakes, then the relief of the hull and
the tracing on it. It only says what the user
asks (``requested``): the plugin does it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from pyarchinit_photogrammetry.gui.flow_layout import FlowLayout

from . import timbers
from .texts import num, t

STEPS = (("frame", ("1. Sistema della nave", "1. Ship frame"),
          (("frame_pick", ("Traccia la chiglia...", "Trace the keel..."),
            "x parte da dove inizi a tracciare. La prua è lì la prima volta, poi resta dov'era",
            "x starts from where you start tracing. The bow is there the first time, then it stays where it was"),
           ("frame_markers", ("Da due marker...", "From two markers..."), "", ""),
           ("frame_options", ("Origine, datum...", "Origin, datum..."), "", ""),
           ("frame_flip", ("x dall'altro capo", "x from the other end"),
            "x corre nell'altro verso: la prua resta dov'è, y e i lati non cambiano",
            "x runs the other way: the bow stays where it is, y and the sides do not change"),
           ("frame_bow", ("La prua è l'altro capo", "The bow is the other end"),
            "Scambia babordo e tribordo su tutti i disegni: dal modello non si vede quale capo è la prua",
            "Exchanges port and starboard on every drawing: the model does not show which end is the bow"))),
         ("stations", ("2. Stazioni", "2. Stations"),
          (("stations_regular", ("Regolari...", "Regular..."), "", ""),
           ("stations_table", ("Da tabella...", "From a table..."), "nome, x: i nomi sono quelli del cantiere", "name, x: the names are the field's"),
           ("stations_catalogue", ("Dalle ordinate del catalogo", "From the catalogue's frames"), "", ""))),
         ("sections", ("3. Sezioni", "3. Sections"), (("sections", ("Calcola...", "Compute..."), "", ""),)),
         ("water", ("4. Acqua", "4. Water"), (("water", ("Livello e rifrazione...", "Level and refraction..."), "", ""),
                                              ("water_pick", ("Clicca il livello...", "Click the level..."), "", ""))),
         ("drawings", ("5. Disegni", "5. Drawings"), (("drawings", ("Genera...", "Make..."), "", ""),
                                                      ("drawings_open", ("Apri la cartella", "Open the folder"), "", ""))),
         ("timbers", ("6. Catalogo dei legni", "6. Timber catalogue"),
          (("timber_line", ("Linea...", "Line..."), "", ""), ("timber_polygon", ("Poligono...", "Polygon..."), "", ""),
           ("timber_point", ("Punto...", "Point..."), "", ""), ("timber_import", ("Da CSV...", "From CSV..."), "", ""),
           ("timber_remove", ("Rimuovi selezionati", "Remove selected"), "", ""),
           ("timber_export", ("Esporta...", "Export..."), "CSV, XLSX o GeoPackage", "CSV, XLSX or GeoPackage"))),
         ("clinker", ("7. Fasce (clinker)", "7. Strakes (clinker)"),
          (("clinker_detect", ("Rileva i corsi...", "Find the strakes..."),
            "I gradini del fasciame sulle sezioni alle stazioni, uniti in linee lungo lo scafo",
            "The plank lands on the sections at the stations, joined into lines along the hull"),
           ("clinker_strakes", ("Modello delle fasce...", "Strake model..."),
            "Le fasce fra le linee dei gradini: un'ipotesi dalle sezioni, non una misura",
            "The strakes between the land lines: a hypothesis from the sections, not a measure"),
           ("clinker_catalogue", ("Nel catalogo", "Into the catalogue"),
            "Una tavola del catalogo dei legni per fascia, segnata come ipotesi",
            "A plank of the timber catalogue per strake, noted as a hypothesis"),
           ("clinker_export", ("Per Clinker Boat Tool...", "For the Clinker Boat Tool..."),
            "I file per il Clinker Boat Tool di Fröhlich e Nielsen (Houdini), Y in alto",
            "The files for Fröhlich and Nielsen's Clinker Boat Tool (Houdini), Y up"),
           ("clinker_import", ("Fasce da Houdini...", "Planks from Houdini..."),
            "Le tavole esportate dal Clinker Boat Tool (OBJ) nel catalogo, con la distanza dalla scansione",
            "The planks exported by the Clinker Boat Tool (OBJ) into the catalogue, with their distance from the scan"),
           ("clinker_open", ("Apri la cartella", "Open the folder"), "", ""))),
         ("relief", ("8. Rilievo e ricalco", "8. Relief and tracing"),
          (("relief_build", ("Rilievo dello scafo...", "Relief of the hull..."),
            "Lo scafo visto dall'alto nel sistema della nave: un prospetto del programma su cui disegnare",
            "The hull seen from above in the ship frame: an elevation of the program to draw on"),
           ("relief_show", ("Mostra", "Show"),
            "Il rilievo nella vista raster, con le altezze, l'ombreggiatura o il rilievo locale sopra",
            "The relief in the raster view, with the heights, the hillshade or the local relief over it"),
           ("proposals_build", ("Proponi...", "Propose..."),
            "Comenti, ordinate, legni longitudinali e fasce letti sul rilievo: proposte, non misure finché non le accetti",
            "Seams, frames, longitudinal timbers and strakes read on the relief: proposals, not measures until accepted"),
           ("trace_accept", ("Accetta", "Accept"),
            "Le proposte selezionate sulla vista nel catalogo dei legni (senza selezione: chiede quali)",
            "The proposals selected on the view into the timber catalogue (none selected: asks which)"),
           ("trace_reject", ("Scarta", "Reject"),
            "Le proposte selezionate sulla vista scartate: non tornano più (anche Canc sulla vista)",
            "The proposals selected on the view rejected: they never come back (Delete on the view too)"),
           ("trace_correct", ("Correggi...", "Correct..."),
            "Il legno selezionato ridisegnato sul rilievo: stesso id e nome",
            "The timber selected traced again on the relief: same id and name"),
           ("trace_add", ("Aggiungi...", "Add..."), "Un legno disegnato a mano sul rilievo",
            "A timber traced by hand on the relief"),
           ("trace_names", ("Nomi da tabella...", "Names from a table..."),
            "nome, x: le ordinate prendono i nomi del cantiere", "name, x: the frames take the field's names"),
           ("trace_loose", ("Sciolto", "Loose"),
            "I legni selezionati segnati come sciolti, fuori posto (di nuovo: tolto il segno)",
            "The timbers selected flagged as loose, out of place (again: the flag taken off)"))))
COLUMNS = (("Id", "Id"), ("Tipo", "Kind"), ("Lato", "Side"), ("Nome", "Name"), ("Stato", "Status"), ("Chunk", "Chunk"),
           ("Sciolto", "Loose"), ("x (m)", "x (m)"), ("Lungh. (m)", "Length (m)"), ("Largh. (m)", "Width (m)"),
           ("z max (m)", "z max (m)"), ("Avviso", "Warning"))
NUMBERS = (7, 10)  # the columns of numbers, aligned on the right


class ShipwreckPane(QWidget):
    requested = Signal(str)  # the name of a step's button

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(t("Il disegno tecnico di uno scafo scavato, misurato sul modello 3D: niente di ciò che manca viene "
                         "ricostruito.", "The technical drawing of an excavated hull, measured on the 3D model: nothing "
                         "missing is reconstructed."))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.states: dict[str, QLabel] = {}
        self.buttons: dict[str, QPushButton] = {}
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([t(*c) for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for key, title, buttons in STEPS:
            head = QLabel(f"<b>{t(*title)}</b>")
            lay.addWidget(head)
            holder = QWidget()
            flow = FlowLayout(holder)
            for name, text, tip_it, tip_en in buttons:
                b = QPushButton(t(*text))
                if tip_it:
                    b.setToolTip(t(tip_it, tip_en))
                b.clicked.connect(lambda _=False, n=name: self.requested.emit(n))
                flow.addWidget(b)
                self.buttons[name] = b
            lay.addWidget(holder)
            state = QLabel("")
            state.setWordWrap(True)
            lay.addWidget(state)
            self.states[key] = state
            if key == "timbers":  # the catalogue's table under its step
                lay.addWidget(self.table)
        lay.addStretch(1)

    def set_states(self, states: dict) -> None:
        for key, text in states.items():
            self.states[key].setText(text)

    def set_rows(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            def value(v):
                return "" if v is None else num(v, 2)

            cells = (r["id"], timbers.kind_title(r["kind"]), timbers.side_title(r["side"]), r["name"],
                     timbers.status_title(r.get("status") or "measured"), r.get("chunk") or "",
                     t("sì", "yes") if r.get("loose") else "", value(r["x"]), value(r["length"]), value(r["width"]),
                     value(r["z_max"]), r["warning"])
            for j, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if NUMBERS[0] <= j <= NUMBERS[1]:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).text() for r in rows]
