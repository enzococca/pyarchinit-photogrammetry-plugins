"""The small dialog of the API key: a password field, *Salva nel portachiavi*, *Dimentica*, where the key is, a
link to photocad.ai. The key typed goes to the keychain only and the field is emptied."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from . import keys
from .texts import t


class KeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Chiave API PhotoCAD", "PhotoCAD API key"))
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        intro = QLabel(t("PhotoCAD (photocad.ai) è un servizio esterno a pagamento che trasforma un'immagine in un "
                         "disegno vettoriale. La chiave (<code>pcad_live_...</code>) si crea su "
                         "<a href=\"https://photocad.ai\">photocad.ai</a>; resta nel portachiavi del sistema, mai nel "
                         "progetto o nelle impostazioni. La variabile d'ambiente " + keys.ENV + " ha la precedenza.",
                         "PhotoCAD (photocad.ai) is an external paid service that turns an image into a vector "
                         "drawing. The key (<code>pcad_live_...</code>) is made on "
                         "<a href=\"https://photocad.ai\">photocad.ai</a>; it stays in the system keychain, never in "
                         "the project or the settings. The environment variable " + keys.ENV + " comes first."))
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        lay.addWidget(intro)
        self.e_key = QLineEdit()
        self.e_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.e_key.setPlaceholderText("pcad_live_...")
        lay.addWidget(self.e_key)
        self.l_status = QLabel()
        self.l_status.setWordWrap(True)
        lay.addWidget(self.l_status)
        row = QHBoxLayout()
        self.b_save = QPushButton(t("Salva nel portachiavi", "Save in the keychain"))
        self.b_forget = QPushButton(t("Dimentica", "Forget"))
        close = QPushButton(t("Chiudi", "Close"))
        for b in (self.b_save, self.b_forget):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        self.b_save.clicked.connect(self.save)
        self.b_forget.clicked.connect(self.forget)
        close.clicked.connect(self.accept)
        self.refresh()

    def refresh(self) -> None:
        self.l_status.setText(keys.status_text())
        self.b_save.setEnabled(keys.keyring_available())

    def save(self) -> None:
        text = self.e_key.text().strip()
        self.e_key.clear()
        if not text:
            self.l_status.setText(t("Scrivi la chiave prima di salvarla.", "Type the key before saving it."))
            return
        try:
            keys.set_key(text)
        except (keys.KeyringUnavailable, ValueError) as exc:
            self.l_status.setText(keys.mask(str(exc), text))
            return
        self.refresh()

    def forget(self) -> None:
        keys.forget_key()
        self.refresh()
