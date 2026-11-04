"""The PhotoCAD API key: from the environment (``PHOTOCAD_API_KEY``), else from the system keychain, in the
same family of entries as the program's assistant keeps the Claude and OpenAI keys (service
``pyarchinit-photogrammetry``, user ``photocad``). Never written anywhere else: not in the settings, the
project, the logs, ``conversion.json`` or an error message (:func:`mask`).

``keyring`` is imported only when a key is asked for; without the library or a usable backend the key is
taken from the environment only."""
from __future__ import annotations

import logging
import os

from .texts import t

log = logging.getLogger("pyarchinit_photogrammetry.plugins.photocad")

SERVICE = "pyarchinit-photogrammetry"
USER = "photocad"
ENV = "PHOTOCAD_API_KEY"


class KeyringUnavailable(RuntimeError):
    """No keychain on this computer: the key cannot be kept."""


def _backend():
    """The keychain of the system, or None when the library or a usable backend is missing (as the program's
    ``assistant.keys``). The tests replace this function."""
    try:
        import keyring
        from keyring.backends import fail
    except Exception:  # noqa: BLE001
        return None
    try:
        backend = keyring.get_keyring()
    except Exception as exc:  # noqa: BLE001 - a broken backend configuration
        log.debug("keyring: %s", exc)
        return None
    if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 1) <= 0:
        return None
    return backend


def keyring_available() -> bool:
    return _backend() is not None


def _no_keychain() -> KeyringUnavailable:
    return KeyringUnavailable(t(f"Nessun portachiavi disponibile su questo computer: la chiave non può essere "
                                f"conservata. Usa la variabile d'ambiente {ENV}.",
                                f"No keychain on this computer: the key cannot be kept. Use the environment "
                                f"variable {ENV}."))


def _stored() -> str | None:
    backend = _backend()
    if backend is None:
        return None
    try:
        return backend.get_password(SERVICE, USER) or None
    except Exception as exc:  # noqa: BLE001
        log.warning("keyring: PhotoCAD key not read: %s", type(exc).__name__)
        return None


def get_key() -> str | None:
    """The key: the environment first, then the keychain; None without both."""
    value = os.environ.get(ENV, "").strip()
    return value or _stored()


def key_source() -> str | None:
    """``env``, ``keyring`` or None: where :func:`get_key` finds the key."""
    if os.environ.get(ENV, "").strip():
        return "env"
    return "keyring" if _stored() else None


def set_key(key: str) -> None:
    """Keep ``key`` in the keychain; :class:`KeyringUnavailable` without one."""
    key = (key or "").strip()
    if not key:
        raise ValueError(t("La chiave è vuota.", "The key is empty."))
    backend = _backend()
    if backend is None:
        raise _no_keychain()
    try:
        backend.set_password(SERVICE, USER, key)
    except Exception as exc:  # noqa: BLE001
        raise KeyringUnavailable(t("Il portachiavi ha rifiutato la chiave ({0}).", "The keychain refused the key ({0}).")
                                 .format(type(exc).__name__)) from None


def forget_key() -> None:
    """The key out of the keychain (the environment is not touched); nothing when there is none."""
    backend = _backend()
    if backend is None:
        return
    try:
        backend.delete_password(SERVICE, USER)
    except Exception as exc:  # noqa: BLE001 - PasswordDeleteError when there is none
        log.debug("keyring: %s", type(exc).__name__)


def masked(key: str | None) -> str:
    """``pcad_live_…cdef``: enough to recognise a key, never the key."""
    if not key:
        return ""
    if len(key) < 24:  # a short key: its tail only (head and tail together would be most of it)
        return f"…{key[-4:]}" if len(key) >= 12 else "…"
    head = key[:10] if key.startswith("pcad_") else ""
    return f"{head}…{key[-4:]}"


def mask(text: str, key: str | None = None) -> str:
    """``text`` with the key (given, or the one of :func:`get_key`) replaced by :func:`masked`."""
    text = str(text)
    for k in {key, None if key else get_key()} - {None, ""}:
        if k and k in text:
            text = text.replace(k, masked(k))
    return text


def status_text() -> str:
    """Where the key is, in words (masked)."""
    src = key_source()
    if src == "env":
        return t(f"Chiave dalla variabile d'ambiente {ENV} ({masked(get_key())}).",
                 f"Key from the environment variable {ENV} ({masked(get_key())}).")
    if src == "keyring":
        return t(f"Chiave nel portachiavi di sistema ({masked(get_key())}).",
                 f"Key in the system keychain ({masked(get_key())}).")
    if not keyring_available():
        return t(f"Nessuna chiave; nessun portachiavi su questo computer: usa la variabile d'ambiente {ENV}.",
                 f"No key; no keychain on this computer: use the environment variable {ENV}.")
    return t("Nessuna chiave: salvala nel portachiavi o usa la variabile d'ambiente " + ENV + ".",
             "No key: save it in the keychain or use the environment variable " + ENV + ".")
