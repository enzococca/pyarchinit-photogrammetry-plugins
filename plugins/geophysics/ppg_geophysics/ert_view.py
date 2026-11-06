"""The readings of a pseudosection in the 2D view as one graphics item of the plugin.

A pseudosection has hundreds or thousands of readings. Drawn as features of the view, one point each,
they cost several Qt calls per reading at every drawing (and on Linux with Python 3.11 PySide6 6.12
drains a reference to None at some of them: the lesson of the program 1.7.3). Here they are one item:
its ``paint`` draws every reading with one ``drawPoints`` per colour (the colours of the log scale in
:data:`BINS` steps, the readings removed in grey, the electrodes in black), with a cosmetic pen, so the
dots keep their size on the screen at every zoom.

The readings are taken out of the inversion with the plugin's own tool - an area drawn with the
program's drawing (``context.request_polygon``) - not with the program's *Select* tool, which picks only
the program's features: :func:`inside` says which readings an area holds."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem

BINS = 48  # colours of the ramp
DOT_PX = 7
ELECTRODE_PX = 6
REMOVED = (158, 158, 158)
ELECTRODE = (32, 32, 32)


def groups(x, palette_rgb_of, ok: np.ndarray):
    """The readings by colour: ``x`` their place on the ramp (0-1, NaN for none), ``palette_rgb_of(xs)`` the
    colours of the ramp at ``xs``; the readings not ``ok`` in grey. Returns ``[(rgb, indices), ...]``."""
    x = np.asarray(x, float)
    ok = np.asarray(ok, bool) & np.isfinite(x)
    k = np.clip(np.floor(np.nan_to_num(x) * BINS), 0, BINS - 1).astype(int)
    centres = palette_rgb_of((np.arange(BINS) + 0.5) / BINS)
    out = [(tuple(int(c) for c in centres[b]), np.flatnonzero(ok & (k == b))) for b in range(BINS)]
    out = [(rgb, idx) for rgb, idx in out if idx.size]
    gone = np.flatnonzero(~ok)
    return ([(REMOVED, gone)] if gone.size else []) + out


class ReadingsItem(QGraphicsItem):
    """The readings (scene coordinates ``points`` (n, 2)) and the electrodes (``electrodes`` (m, 2)),
    painted in one pass. It is never picked by the program's tools (it is not a feature)."""

    def __init__(self, points, groups_, electrodes=None, parent=None):
        super().__init__(parent)
        pts = np.asarray(points, float).reshape(-1, 2)
        self._groups = []
        for rgb, idx in groups_:
            if len(idx):
                poly = QPolygonF([QPointF(x, y) for x, y in pts[np.asarray(idx, int)].tolist()])
                self._groups.append((QColor(*rgb), poly, DOT_PX if rgb != REMOVED else DOT_PX - 2))
        el = np.asarray(electrodes if electrodes is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        if len(el):
            self._groups.append((QColor(*ELECTRODE), QPolygonF([QPointF(x, y) for x, y in el.tolist()]), ELECTRODE_PX))
        allp = np.vstack([pts, el]) if len(pts) or len(el) else np.zeros((1, 2))
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        pad = 0.05 * float(max((hi - lo).max(), 1.0)) + 20.0  # the dots have a size on the screen, not in the scene
        self._rect = QRectF(float(lo[0]) - pad, float(lo[1]) - pad, float(hi[0] - lo[0]) + 2 * pad,
                            float(hi[1] - lo[1]) + 2 * pad)
        self.count = int(len(pts))

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter, _option, _widget=None) -> None:
        pen = QPen()
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        for colour, poly, size in self._groups:
            pen.setColor(colour)
            pen.setWidthF(float(size))
            painter.setPen(pen)
            painter.drawPoints(poly)


def inside(points, polygon) -> np.ndarray:
    """Which of ``points`` (n, 2) fall inside ``polygon`` (k, 2), in the same coordinates."""
    from matplotlib.path import Path as MplPath

    poly = np.asarray(polygon, float).reshape(-1, 2)
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(poly) < 3 or not len(pts):
        return np.zeros(len(pts), bool)
    return MplPath(poly).contains_points(pts)
