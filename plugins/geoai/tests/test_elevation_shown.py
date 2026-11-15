"""Units made from proposals in the coordinates of the plan never go into the drawings of an elevation: while the
window shows one (``context.shown_elevation()`` since program 1.10.0, the window's ``drawings_frame`` before)
the plan is shown again first, and nothing is added when the chunk has no orthophoto or DEM to show."""
from __future__ import annotations

from types import SimpleNamespace

from ppg_geoai import plan_shown, plan_units


class _Context:
    def __init__(self, shown, raster=True, new=True):
        self.window = SimpleNamespace(drawings_frame=shown)
        self.asked = []
        self._raster = raster
        if new:
            self.shown_elevation = lambda: self.window.drawings_frame

    def show_raster(self, kind=None, chunk=None, path=None):
        self.asked.append(kind)
        if self._raster:
            self.window.drawings_frame = None
        return self._raster


def test_the_plan_is_shown_before_units_of_the_plan():
    for new in (True, False):
        c = _Context(None, new=new)
        assert plan_shown(c) and c.asked == []  # the plan already
        c = _Context("e1", new=new)
        assert plan_shown(c) and c.asked == [None] and c.window.drawings_frame is None
        c = _Context("e1", raster=False, new=new)
        assert not plan_shown(c) and c.window.drawings_frame == "e1"
    assert plan_shown(SimpleNamespace(window=None))  # the command line


def test_with_program_1_10_1_the_units_go_to_the_plan_and_the_view_stays():
    asked = []
    window = SimpleNamespace(drawings_frame="e1")
    c = SimpleNamespace(window=window, PLAN="", supports=lambda f: f == "plan-units-while-elevation",
                        shown_elevation=lambda: "e1", show_raster=lambda kind: asked.append(kind) or True)
    assert plan_units(c) == {"elevation": ""}
    assert plan_shown(c) and asked == [] and window.drawings_frame == "e1"
    assert plan_units(SimpleNamespace(window=window)) == {}  # an older program
