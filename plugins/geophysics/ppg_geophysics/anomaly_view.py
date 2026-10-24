"""The anomalies on the raster view of the program: their layer drawn from the project
(:func:`draw`, called by the plugin on every raster the view shows and after every change of the
anomalies) and :class:`DeleteWatch`, which takes out of the project the anomalies deleted in the
view with the program's *Select* tool. The polygons are drawn with the program's own drawing, lent
to the plugin (``context.request_polygon``). Qt only: the window imports this module, the command
line does not."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from .anomalies import COLOUR, LAYER, attributes, entries, remove


def draw(view, chunk) -> set:
    """The anomalies of the chunk drawn again into the layer of the view; returns the ids drawn
    (none when the view shows nothing)."""
    if view is None or view.tr is None:
        return set()
    if LAYER in view.layers:
        view.remove_layer(LAYER)
    items = entries(chunk)
    if not items:
        return set()
    with view.batch():
        for a in items:
            view.add_polygon(a["coords"], label=a.get("etichetta", ""), color=COLOUR, fill_alpha=40, layer=LAYER,
                             attrs=attributes(a))
    return {a["id"] for a in items}


def shown(view) -> set:
    """The ids of the anomalies in the layer of the view."""
    return {f.get("id") for f in view.features(LAYER)}


class DeleteWatch(QObject):
    """The anomalies deleted in the view with the program's *Select* tool, taken out of the project
    too (then ``changed`` is emitted).

    For the programs before 1.3.4, which tell a plugin nothing when the user deletes features of its
    layers (since 1.3.4 ``context.on_features_deleted`` does, and the plugin uses it): the layers of
    the view are watched (``layers_changed``, checked once the change is over), with the selection.
    An anomaly that was selected and then is gone from the layer was deleted. Anomalies gone without
    being selected - the whole layer taken out of the view in the Layers pane, the scene of another
    raster - stay in the project: the plugin draws them again on the next raster
    (``context.on_raster_loaded``). :meth:`drawn` tells the watch what the plugin drew."""

    changed = Signal()

    def __init__(self, view, chunk_of):
        super().__init__(view)
        self.view, self.chunk_of = view, chunk_of
        self.known: set = set()  # the ids the plugin drew in the view
        self._selected: set = set()  # the anomalies in the selection of the view
        self._unselected: set = set()  # those just taken out of it, until the end of this turn of the event loop
        self._candidates: set = set()  # selected when the layers changed: deleted if they are gone afterwards
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.check)
        self._forget = QTimer(self)
        self._forget.setSingleShot(True)
        self._forget.setInterval(0)
        self._forget.timeout.connect(self._forget_unselected)
        view.layers_changed.connect(self._layers_changed)
        view.selection_changed.connect(self._selection_changed)

    def detach(self) -> None:
        for signal, slot in ((self.view.layers_changed, self._layers_changed),
                             (self.view.selection_changed, self._selection_changed)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._timer.stop()
        self._forget.stop()

    def drawn(self, ids) -> None:
        """The plugin drew these anomalies in the view (none: nothing shown)."""
        self.known = set(ids)

    def _selection_changed(self, _count: int) -> None:
        try:
            now = {attrs.get("id") for layer, _i, attrs in self.view.selected_features() if layer == LAYER}
        except RuntimeError:  # the window was closed
            return
        left = self._selected - now
        if left:  # out of the selection: deleted, when the layers change in this same turn
            self._unselected |= left
            self._forget.start()
        self._selected = now

    def _forget_unselected(self) -> None:
        self._unselected = set()

    def _layers_changed(self) -> None:
        self._candidates |= self._selected | self._unselected
        self._timer.start()

    def check(self) -> None:
        candidates, self._candidates = self._candidates, set()
        try:
            in_view = shown(self.view)
            whole_layer = LAYER in self.view.layers
        except RuntimeError:  # the window was closed
            return
        chunk = self.chunk_of()
        gone = self.known - in_view
        if chunk is None or not gone:
            return
        deleted = gone & candidates
        if not whole_layer and deleted != gone:
            deleted = set()  # the layer taken out of the view, not the anomalies deleted: they stay in the project
        self.known = in_view
        if deleted and remove(chunk, deleted):
            self.changed.emit()
