"""Named UT groups are planned on the chosen proposals with independent identifiers."""
from types import SimpleNamespace

import pytest
from shapely.geometry import box

pytest.importorskip("pyarchinit_photogrammetry.archaeo.unit_identity")
pytest.importorskip("PySide6")

from ppg_samgeo.identity import convert_dialog
from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon
from pyarchinit_photogrammetry.archaeo.unit_identity import plan_units
from pyarchinit_photogrammetry.gui.unit_identity_dialog import UnitIdentityDialog


def test_selected_proposals_get_shared_ut_name_and_unique_reserved_numbers(monkeypatch):
    from ppg_samgeo import runs, view

    run = {"id": "graves", "text": "grave", "name": "graves", "units_made": []}
    rows = [(box(i, 0, i + .5, .5), {"n": i + 1, "score": .9}) for i in range(3)]
    chunk = SimpleNamespace(meta={})
    window = SimpleNamespace(units=[UnitPolygon(box(0, 0, 1, 1), us="1", tipo="UT", sito="Temple",
                                                attributes={"progetto": "Temple"})],
                             project_site=lambda: "Temple", doc=SimpleNamespace(chunks=[chunk]), chunk=chunk, pa=None)
    captured = []
    plugin = SimpleNamespace(_run_of=lambda rid: (chunk, run), _view=lambda: None,
                             context=SimpleNamespace(window=window),
                             make_units=lambda rid, wanted, **kw: captured.append((wanted, kw["attributes"])))
    monkeypatch.setattr(runs, "features", lambda *args: rows)
    monkeypatch.setattr(view, "selected", lambda *args: {2, 3})

    def ask(units, selected, **kw):
        assert selected == {1, 2}
        return plan_units(units, selected, tipo="UT", progetto="Temple", name_mode="same",
                          name="Tombe A", group="Tombe A", reserved=kw["reserved"])

    monkeypatch.setattr(UnitIdentityDialog, "ask", ask)
    assert convert_dialog(plugin, "graves")
    wanted, values = captured[0]
    assert [a["n"] for _, a in wanted] == [2, 3]
    assert [a["us"] for a in values] == ["2", "3"]
    assert {a["nome"] for a in values} == {"Tombe A"}
    assert {a["_layer"] for a in values} == {"Unità topografiche - Tombe A"}
    assert {a["samgeo_proposal"] for a in values} == {2, 3}
