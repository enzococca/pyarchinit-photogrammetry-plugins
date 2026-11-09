"""Heights that a drone would fly wrong are refused, never only warned (0.1.2).

- a point that would need more than the 500 m technical ceiling above the takeoff (the ground rises more than that
  from the takeoff): the drone stops at the ceiling, closer to the ground than planned;
- a takeoff of a leg kilometres from its leg: its heights would be counted from another takeoff than the one set;
- per-leg takeoffs kept after a new design whose legs changed: named, to be checked;
- DJI Fly: only the slots the user saw and agreed to overwrite are overwritten;
- the targets of a plan whose blocks are all switched off are read near the plan's area.
"""
import pytest

from ppg_flightplan import design, project
from ppg_flightplan import plan as pl
from planfixtures import reference_plan
from terrainfixtures import AREA

SOUTH, NORTH = (44.9970, 10.0000), (45.0030, 10.0000)


class SlopeDem:
    """A project DEM rising to the north: ``slope`` metres per metre from 0 m at the south edge of the area."""

    def __init__(self, slope: float):
        self.slope = slope

    def height(self, lat, lon):
        return max(0.0, (lat - SOUTH[0]) * 111_320.0 * self.slope)

    def heights(self, points):
        return [self.height(p[0], p[1]) for p in points]

    def to_json(self):
        return {"file": "pendio.tif", "epsg": 4326}


def _plan(**extra) -> dict:
    piano = {"nome": "rilievo", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in AREA], "quota": 60,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "per_punto", "seguiTerreno": True}]}
    piano.update(extra)
    return piano


# ------------------------------------------------------------------ m9: the 500 m ceiling

def test_a_terrain_rising_beyond_the_ceiling_refuses_the_block():
    """60 m above a ground that rises 470 m from the takeoff: the top points would need 530 m above the takeoff.
    Cut at 500 m they would fly 30 m above the ground instead of 60: the block is refused, never clipped."""
    piano = _plan(decollo=list(SOUTH))
    with pytest.raises(design.CeilingReached) as e:
        design.apply_terrain(piano, dem=SlopeDem(0.7))
    assert "«A»" in str(e.value) and "500" in str(e.value)
    assert e.value.blocks and e.value.blocks[0][0] == 0
    assert "quote" not in piano["blocchi"][0]                # nothing written
    # the refresh before an export fails the same way: no file
    with pytest.raises(design.StaleHeights, match="500"):
        project.prepare(piano, SlopeDem(0.7))


def test_below_the_ceiling_nothing_changes():
    piano = _plan(decollo=list(SOUTH))
    design.apply_terrain(piano, dem=SlopeDem(0.6))
    assert max(piano["blocchi"][0]["quote"]) < design.CEILING_ABOVE_TAKEOFF
    project.prepare(piano, SlopeDem(0.6))                    # exportable


def test_heights_at_the_ceiling_in_a_saved_plan_refuse_the_export():
    """A plan computed by 0.1.1 or earlier, whose points were cut at 500 m and only warned: refused at the export."""
    piano = _plan(decollo=list(SOUTH))
    design.apply_terrain(piano, dem=SlopeDem(0.6))
    b = piano["blocchi"][0]
    b["quote"] = [min(design.CEILING_ABOVE_TAKEOFF, q + 100.0) for q in b["quote"]]   # as 0.1.1 cut them
    b[design.HEIGHTS_SIGNATURE] = design.heights_signature(b, piano)
    with pytest.raises(design.CeilingReached, match="500"):
        design.require_exportable(piano)


def test_a_leg_taking_off_lower_that_would_pass_the_ceiling_refuses_the_export():
    """Heights computed from the top (fine), but the leg takes off from the foot, 400 m lower: from there its points
    would need more than 500 m. The 0.1.1 cut them at 500 m in the file: now the export is refused."""
    dem = SlopeDem(0.7)
    piano = _plan(decollo=list(NORTH))
    design.apply_terrain(piano, dem=dem)
    key = pl.plan_legs(piano)[0].key
    piano["decolli"] = {key: list(SOUTH)}
    with pytest.raises(design.CeilingReached, match=key):
        project.prepare(piano, dem)


# ------------------------------------------------------------------ m10: a leg takeoff far from its leg

def test_a_leg_takeoff_far_from_its_leg_refuses_the_export():
    """Farther than 2 km from its leg a takeoff is not taken as one (the heights would stay counted from the general
    takeoff while the user said the leg takes off there): refused with the reason, never ignored in silence."""
    dem = SlopeDem(0.05)
    piano = _plan(decollo=list(SOUTH))
    design.apply_terrain(piano, dem=dem)
    key = pl.plan_legs(piano)[0].key
    piano["decolli"] = {key: [45.0300, 10.0000]}             # about 3 km north
    assert pl.far_leg_takeoffs(piano) and pl.far_leg_takeoffs(piano)[0][0] == key
    with pytest.raises(design.StaleHeights, match=key) as e:
        project.prepare(piano, dem)
    assert "2" in str(e.value) and "km" in str(e.value)


def test_a_far_leg_takeoff_refuses_a_flat_plan_too(tmp_path):
    piano = reference_plan()
    key = pl.plan_legs(piano)[0].key
    piano["decolli"] = {key: [45.0300, 10.0000]}
    with pytest.raises(design.StaleHeights, match=key):
        project.export_plan(piano, "kmz", tmp_path / "piano.kmz")
    assert not list(tmp_path.iterdir())
    piano["decolli"] = {key: [45.0010, 10.0010]}              # next to its leg: fine
    assert pl.far_leg_takeoffs(piano) == []
    assert project.export_plan(piano, "kmz", tmp_path / "piano.kmz")


# ------------------------------------------------------------------ m4: takeoffs kept on legs that moved

def test_kept_leg_takeoffs_on_legs_that_changed_are_named():
    old = reference_plan()
    keys = [leg.key for leg in pl.plan_legs(old)]
    old["decolli"] = {keys[0]: [45.0, 10.0], keys[-1]: [45.001, 10.001]}
    same = reference_plan()
    same["decolli"] = dict(old["decolli"])
    assert pl.moved_leg_takeoffs(old, same) == []
    changed = reference_plan()
    changed["blocchi"][0]["orientamento"] = 90               # the strips the other way: the legs of A move
    changed["decolli"] = dict(old["decolli"])
    moved = pl.moved_leg_takeoffs(old, changed)
    assert keys[0] in moved and keys[-1] not in moved          # the ring did not change


# ------------------------------------------------------------------ m1: near the area with no block active

def test_the_site_is_the_area_when_no_block_flies():
    piano = reference_plan()
    on = pl.site_centre(piano)
    for b in piano["blocchi"]:
        b["attivo"] = False
    off = pl.site_centre(piano)
    assert off is not None and abs(off[0] - 45.0) < 0.01 and abs(off[1] - 10.0) < 0.01
    assert abs(on[0] - 45.0) < 0.01
    piano["blocchi"] = []
    piano["areaDisegnata"] = [[45.0, 10.0], [45.0, 10.002], [45.002, 10.002]]
    assert pl.site_centre(piano) == pytest.approx((45.000667, 10.001333), abs=1e-5)
    piano.pop("areaDisegnata")
    assert pl.site_centre(piano) is None


# ------------------------------------------------------------------ review of 0.1.2

def _agl_of_leg(leg, dem, takeoff):
    """The distance from the ground of every waypoint of ``leg`` flown from ``takeoff`` (heights above it)."""
    g0 = dem.height(*takeoff)
    return [h - (dem.height(*q) - g0) for h, q in zip(leg.route.heights, leg.route.points)]


@pytest.mark.parametrize("slope", [0.3, 0.55])
def test_a_leg_taking_off_at_the_foot_flies_at_the_planned_height_where_the_general_one_cut_at_the_floor(slope):
    """General takeoff on the top, the ground falling 200 m (or 370 m) to the south: from there the low points are
    cut at the 10 m floor. A leg taking off at the foot must fly them 60 m above the ground, not 10 m + the drop."""
    dem = SlopeDem(slope)
    piano = _plan(decollo=list(NORTH))
    design.apply_terrain(piano, dem=dem)
    assert min(piano["blocchi"][0]["quote"]) == design.MIN_HEIGHT        # cut at the floor from the top
    key = pl.plan_legs(piano)[0].key
    piano["decolli"] = {key: list(SOUTH)}
    fresh, _ = project.prepare(piano, dem)
    leg = next(g for g in pl.plan_legs(fresh) if g.key == key)
    agl = _agl_of_leg(leg, dem, SOUTH)
    assert min(agl) > 58.5 and max(agl) < 61.5, (min(agl), max(agl))


def test_a_floor_point_whose_ground_is_unknown_refuses_the_export(monkeypatch):
    from ppg_flightplan import specials

    dem = SlopeDem(0.3)
    piano = _plan(decollo=list(NORTH))
    design.apply_terrain(piano, dem=dem)
    key = pl.plan_legs(piano)[0].key
    piano["decolli"] = {key: list(SOUTH)}
    fresh, _ = project.prepare(piano, dem)
    monkeypatch.setattr(specials, "ground_of_grid", lambda g: (lambda p: None))
    with pytest.raises(design.StaleHeights, match=key):
        design.require_exportable(fresh)


def test_a_special_point_cut_at_the_ceiling_refuses_the_export():
    """The block stays under the ceiling, but a free special point on the hill beyond the area would need more than
    500 m above the takeoff: its cruise height, cut at 500 m, is refused like the block's."""
    dem = SlopeDem(0.6)
    piano = _plan(decollo=list(SOUTH))
    piano["blocchi"][0]["speciali"] = [{"id": "s1", "nome": "Cima", "tipo": "foto", "punto": [45.0060, 10.0000],
                                        "aggancio": False}]
    design.apply_terrain(piano, dem=dem)
    assert max(piano["blocchi"][0]["quote"]) < design.CEILING_ABOVE_TAKEOFF
    with pytest.raises(design.CeilingReached, match="500"):
        project.prepare(piano, dem)


def test_the_reasons_say_the_export_waits_for_the_terrain():
    """design() on a terrain beyond the ceiling: the block keeps following the terrain without heights, and the export
    is refused - the reason says so, not "the flight stays at a constant height"."""
    out = design.design(AREA, "orto", 2.0, follow_terrain=True, takeoff=SOUTH, dem=SlopeDem(2.0))
    text = " ".join(out["motivi"])
    assert "quota costante" not in text and "constant height" not in text
    assert "500" in text
    assert design.blocks_with_old_heights(out["piano"])                  # the export is refused until it is redone


def test_not_agreed_does_not_take_a_string_for_a_list_of_slots():
    busy = [{"contenitore": "ab"}, {"contenitore": "a"}]
    from ppg_flightplan import djifly

    with pytest.raises(TypeError):
        djifly.not_agreed(busy, "a")
