"""Plugin-owned Italian and English text pairs."""
from __future__ import annotations


def t(it: str, en: str) -> str:
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:
        return it
