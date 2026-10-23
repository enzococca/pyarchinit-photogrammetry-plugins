"""The anomalies on the raster view of the program: their layer drawn and kept in step with the
project (:class:`ViewSync`) and the polygon tool of the plugin (:class:`PolygonTool`). Qt only: the
window imports this module, the command line does not."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal

from .anomalies import COLOUR, LAYER, attributes, entries, remove


def _alive(item, scene) -> bool:
    try:
        return item.scene() is scene
    except RuntimeError:  # deleted with the scene
        return False


def drop_stale_layers(view) -> bool:
    """Layers of the view whose items went with a cleared scene (the program clears the scene of an
    empty view without its list of layers): out of the list. True when one went."""
    scene, gone = view.scene(), False
    for name in list(getattr(view, "layers", {})):
        feats = view.layers[name]["features"]
        if feats and not _alive(feats[0][0], scene):
            view.layers.pop(name, None)
            if name in view.layer_order:
                view.layer_order.remove(name)
            gone = True
    return gone


def draw(view, chunk) -> None:
    """The anomalies of the chunk drawn again into the layer of the view."""
    if view is None or view.tr is None:
        return
    drop_stale_layers(view)
    if LAYER in view.layers:
        view.remove_layer(LAYER)
    items = entries(chunk)
    if not items:
        return
    with view.batch():
        for a in items:
            view.add_polygon(a["coords"], label=a.get("etichetta", ""), color=COLOUR, fill_alpha=40, layer=LAYER,
                             attrs=attributes(a))


class ViewSync(QObject):
    """The layer of the view kept in step with the anomalies of the project. At every change of
    the layers of the view (checked once the change is over): another raster shown, or the list of
    the project changed (another chunk, a project opened) -> the layer drawn again; the same raster
    and an anomaly gone from the layer (deleted with the program's *Select* tool) -> out of the
    project too, and ``changed`` is emitted."""

    changed = Signal()

    def __init__(self, view, chunk_of):
        super().__init__(view)
        self.view, self.chunk_of = view, chunk_of
        self._tr = None  # the transform of the raster the layer was drawn on
        self._known: set = set()  # the ids drawn
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.check)
        view.layers_changed.connect(self._timer.start)

    def detach(self) -> None:
        try:
            self.view.layers_changed.disconnect(self._timer.start)
        except (RuntimeError, TypeError):
            pass
        self._timer.stop()

    def check(self) -> None:
        try:
            tr = self.view.tr
        except RuntimeError:  # the window was closed
            return
        chunk = self.chunk_of()
        if tr is None or chunk is None:
            return
        ids = {a["id"] for a in entries(chunk)}
        if tr is not self._tr or ids != self._known:
            self.redraw()
            return
        layer = self.view.layers.get(LAYER)
        shown = {f[2].get("id") for f in layer["features"]} if layer else set()
        gone = self._known - shown
        if gone:
            remove(chunk, gone)
            self._known = {a["id"] for a in entries(chunk)}
            self.changed.emit()

    def redraw(self) -> None:
        chunk = self.chunk_of()
        draw(self.view, chunk)
        self._tr = self.view.tr
        self._known = {a["id"] for a in entries(chunk)}


class PolygonTool(QObject):
    """A polygon drawn on the raster view for the plugin: click the vertices, double click (or
    Enter) to close it, Backspace takes the last vertex back, right click or Esc cancel. While it is
    on, the view is in a mode of its own (cross cursor, no dragging) and the clicks go to the tool,
    not to the program's tools; choosing one of them ends it. ``on_finished(xy)`` gets the vertices
    (an (N, 2) array of world coordinates), ``on_cancelled()`` a cancel."""

    MODE = "geophysics_anomaly"

    def __init__(self, view, on_finished, on_cancelled=None):
        super().__init__(view)
        self.view = view
        self.on_finished, self.on_cancelled = on_finished, on_cancelled
        self.points: list[tuple[float, float]] = []
        self.item = None
        self.active = False

    def start(self) -> None:
        if self.view.tr is None:
            raise RuntimeError("nothing in the raster view to draw on")
        self.points = []
        self.view.set_mode(self.MODE)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        self.active = True
        self.view.setFocus()

    def _end(self) -> None:
        if not self.active:
            return
        self.active = False
        self.view.viewport().removeEventFilter(self)
        self.view.removeEventFilter(self)
        if self.item is not None and _alive(self.item, self.view.scene()):
            self.view.scene().removeItem(self.item)
        self.item = None
        if self.view.mode == self.MODE:
            self.view.set_mode("pan")

    def cancel(self) -> None:
        was = self.active
        self._end()
        if was and self.on_cancelled is not None:
            self.on_cancelled()

    def finish(self) -> bool:
        if len(self.points) < 3:
            return False  # not a polygon yet: drawing goes on
        pts = np.array(self.points, float)
        self._end()
        self.on_finished(pts)
        return True

    def add_point(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))
        self._rubber()

    def _world(self, ev) -> tuple[float, float]:
        return self.view.world(self.view.mapToScene(ev.position().toPoint()))

    def _rubber(self, cursor: tuple[float, float] | None = None) -> None:
        from PySide6.QtGui import QColor, QPainterPath, QPen

        if self.view.tr is None:
            return
        pts = [self.view.scene_pt(x, y) for x, y in self.points + ([cursor] if cursor else [])]
        if not pts:
            return
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if len(pts) > 2:
            path.closeSubpath()
        if self.item is None or not _alive(self.item, self.view.scene()):
            pen = QPen(QColor(COLOUR))
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            self.item = self.view.scene().addPath(path, pen)
            self.item.setZValue(10000)
        else:
            self.item.setPath(path)

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802 - Qt's name
        if not self.active:
            return False
        if self.view.mode != self.MODE:  # a tool of the program was chosen: this one ends
            self.cancel()
            return False
        kind = ev.type()
        if kind == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.LeftButton:
                self.add_point(*self._world(ev))
                return True
            if ev.button() == Qt.MouseButton.RightButton:
                self.cancel()
                return True
            return False
        if kind == QEvent.Type.MouseButtonDblClick and ev.button() == Qt.MouseButton.LeftButton:
            self.finish()
            return True
        if kind == QEvent.Type.MouseButtonRelease and ev.button() in (Qt.MouseButton.LeftButton,
                                                                      Qt.MouseButton.RightButton):
            return True
        if kind == QEvent.Type.MouseMove:
            if self.points:
                self._rubber(self._world(ev))
            return False  # the program still shows the coordinates under the cursor
        if kind == QEvent.Type.KeyPress:
            key = ev.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish()
                return True
            if key == Qt.Key.Key_Escape:
                self.cancel()
                return True
            if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                if self.points:
                    self.points.pop()
                    self._rubber()
                return True
        return False
