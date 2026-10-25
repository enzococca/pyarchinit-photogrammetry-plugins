"""SAM 3 access information and credential redaction; no credentials are persisted here."""
from __future__ import annotations

import os
import re

SAM3_MODEL = "facebook/sam3"
SAM3_URL = "https://huggingface.co/facebook/sam3"
SAM3_LICENSE_URL = "https://huggingface.co/facebook/sam3/blob/main/LICENSE"


def notice() -> str:
    from .texts import t

    return t(
        "SAM 3 è opzionale. I pesi sono soggetti alla licenza Meta SAM, distinta dalla GPL del plugin, "
        "e non sono inclusi nel programma. Richiedi l'accesso a Meta sulla pagina facebook/sam3 di "
        "Hugging Face e leggi e accetta le condizioni richieste. Dopo l'approvazione, crea un token "
        "personale Hugging Face con permesso di lettura per questo modello. Il token da solo non "
        "concede l'accesso. Inseriscilo qui per questa sessione, oppure usa HF_TOKEN o hf auth login. "
        "Il plugin non salva il token nel progetto, nelle impostazioni o nei log. hf auth login "
        "lo conserva nella cache locale di Hugging Face.",
        "SAM 3 is optional. Its weights are subject to the Meta SAM license, separate from the plugin's "
        "GPL, and are not bundled with the application. Request access from Meta on the facebook/sam3 "
        "Hugging Face page and read and accept the required terms. After approval, create a personal "
        "Hugging Face token with read permission for this model. A token alone does not grant access. "
        "Enter it here for this session, or use HF_TOKEN or hf auth login. The plugin does not save "
        "the token in projects, settings or logs. hf auth login stores it in Hugging Face's local cache.",
    )


def redact(text: str, token: str | None = None) -> str:
    """Sanitize before truncation, including credentials embedded in a library traceback."""
    text = str(text)
    for secret in (token, os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", text)
