"""Writing the figures of the plugin (stratigraphic columns, sections) in the formats asked.

matplotlib draws a vector format with a backend of its own (``backend_svg``, ``backend_pdf``,
``backend_ps``), loaded the first time a file of that format is written. An installation may lack
them - the packaged application up to 1.3.0 carries only Agg, the backend of the PNG -: such a
format is skipped, an older file of that name removed (it would show another drawing), the PNG
written in its place when none was asked, and a note says so in the language of the interface.
"""
from __future__ import annotations

from pathlib import Path

from .texts import t

FORMATS = {".svg": "SVG", ".svgz": "SVG", ".pdf": "PDF", ".eps": "EPS", ".ps": "PS"}


class Written(list):
    """The files written, as a list of paths, and ``notes``: the formats that could not be
    written, said for the user (empty when everything was written)."""

    def __init__(self, paths=(), notes=()):
        super().__init__(paths)
        self.notes = list(notes)


def save(fig, paths, png_dpi: float | None = None) -> Written:
    """Write ``fig`` to every file of ``paths`` (the format from the suffix; ``png_dpi`` for the
    PNG). A format whose backend cannot be loaded is skipped and noted; the PNG, which needs only
    Agg, is written beside the first file skipped when ``paths`` has no PNG."""
    written, skipped = [], []
    for p in (Path(p) for p in paths):
        png = p.suffix.lower() == ".png"
        try:
            fig.savefig(p, dpi=png_dpi if png else None)
        except ImportError:  # ModuleNotFoundError: the backend of this format is not installed
            if png:
                raise
            p.unlink(missing_ok=True)
            skipped.append(p)
            continue
        written.append(p)
    if not skipped:
        return Written(written)
    kinds = ", ".join(sorted({FORMATS.get(p.suffix.lower(), p.suffix.lstrip(".").upper()) for p in skipped}))
    if any(p.suffix.lower() == ".png" for p in written):
        note = t("{0} non disponibile in questa installazione: scritto solo il PNG",
                 "{0} not available in this installation: only the PNG written").format(kinds)
    else:
        png = skipped[0].with_suffix(".png")
        fig.savefig(png, dpi=png_dpi)
        written.append(png)
        note = t("{0} non disponibile in questa installazione: scritto il PNG ({1})",
                 "{0} not available in this installation: the PNG written instead ({1})").format(kinds, png.name)
    return Written(written, [note])
