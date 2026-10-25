"""Redact credentials inherited by package installers and worker processes."""
from __future__ import annotations

import os
import re


def redact(text: str) -> str:
    text = str(text)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", text)
