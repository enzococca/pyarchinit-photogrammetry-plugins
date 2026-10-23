"""The fixes of 0.1.1 on a small tell: a mound on a tilted plain, a topsoil 0.3 m thick that
follows the ground (unit 1), the fill of the mound (unit 2, missing from the logs on the plain)
and a natural soil that is a plane under it all (unit 3). Every contact in the mode that suits
it, the stack resolved by the kriging variance, a missing unit kept missing, the steadiest
variogram model chosen, the numbers of the Italian interface, the volumes on the DEM, a model
without holes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_geoarchaeology import CRS, X0, Y0, at, comma

import pyarchinit_photogrammetry as ppg
from ppg_geoarchaeology import boreholes as bh
from ppg_geoarchaeology import kriging, model
from ppg_geoarchaeology import surfaces as sf

W, H, CELL = 120.0, 100.0, 0.5
TOPSOIL = 0.3


def plain(X, Y):
    return 100.0 + 0.01 * X + 0.005 * Y


def mound(X, Y):
    """8 m high, nothing beyond 44.6 m from its centre."""
    return np.maximum(8.05 * np.exp(-((X - 60.0) ** 2 + (Y - 50.0) ** 2) / (2 * 14.0 ** 2)) - 0.05, 0.0)


def ground(X, Y):
    return plain(X, Y) + mound(X, Y)


def natural(X, Y):
    return plain(X, Y) - TOPSOIL  # on the plain right under the topsoil


def positions(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(6, 114, 7), np.linspace(6, 94, 6))
    return np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-2.5, 2.5, (42, 2))


def mound_site(tmp_path: Path, dem_hole: bool = False):
    """The project (EPSG:32633, DEM of the ground in 0.5 m cells, a hole in it when asked) and
    the boreholes: topsoil, fill where the mound is, natural; those on the top stop in the fill."""
    import rasterio
    from rasterio.transform import from_origin

    doc = ppg.Document()
    ch = doc.add_chunk("Tell")
    doc.save(tmp_path / "tell.ppg")
    ch.crs = CRS
    m = 40.0  # the DEM goes 40 m beyond the boreholes' rectangle: over the margin of the surfaces
    nc, nr = int((W + 2 * m) / CELL), int((H + 2 * m) / CELL)
    XX, YY = np.meshgrid(-m + (np.arange(nc) + 0.5) * CELL, H + m - (np.arange(nr) + 0.5) * CELL)
    z = ground(XX, YY).astype(np.float32)
    if dem_hole:
        z[(np.abs(XX - 30) < 5) & (np.abs(YY - 30) < 5)] = np.nan  # 10 x 10 m without a DEM
    path = ch.data_dir / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=nr, width=nc, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0 - m, Y0 + H + m, CELL, CELL), nodata=float("nan")) as dst:
        dst.write(z, 1)
    ch._set_product("dem", path)
    doc.save()
    pts = positions()
    collars, intervals = ["sondaggio;x;y;quota;profondita"], ["sondaggio;da;a;us;descrizione;colore"]
    for i, (X, Y) in enumerate(pts):
        g, m = ground(X, Y), mound(X, Y)
        d_nat = TOPSOIL + (m if m > 0.01 else 0.0)
        end = min(d_nat + 0.5, 3.5)  # the boreholes on the top stop in the fill
        rows = [(0.0, TOPSOIL, "1", "humus", "10YR 3/2")]
        if m > 0.01:
            rows.append((TOPSOIL, min(d_nat, end), "2", "riporto di mattoni crudi", "7.5YR 6/4"))
        if end > d_nat:
            rows.append((d_nat, end, "3", "argilla sterile", "10YR 6/6"))
        collars.append(f"T{i + 1};{comma(X0 + X, 3)};{comma(Y0 + Y, 3)};{comma(g)};{comma(end, 3)}")
        intervals += [f"T{i + 1};{comma(a)};{comma(b)};{u};{d};{c}" for a, b, u, d, c in rows]
    (tmp_path / "c.csv").write_text("\n".join(collars) + "\n", encoding="utf-8")
    (tmp_path / "i.csv").write_text("\n".join(intervals) + "\n", encoding="utf-8")
    bh.import_files(ch, [tmp_path / "c.csv", tmp_path / "i.csv"])
    doc.save()
    return doc, ch, pts


def inside(n: int = 400, seed: int = 9) -> np.ndarray:
    """Points between the boreholes."""
    return np.random.default_rng(seed).uniform([12, 12], [108, 88], (n, 2))


# ------------------------------------------------------------------ 1. the mode of every contact
def test_every_contact_in_its_own_mode(tmp_path, capsys):
    doc, ch, pts = mound_site(tmp_path)
    reg = sf.build(ch)
    by = {e["unit"]: e for e in reg["contacts"]}
    # the base of the topsoil follows the ground (its depth is the same everywhere): by depth; the
    # natural is a plane whatever the mound does above it: by elevation
    assert by["2"]["mode"] == "depth" and by["2"]["mode_reason"]["rule"] == "follows_ground"
    assert by["3"]["mode"] == "elevation" and by["3"]["mode_reason"]["rule"] == "own_shape"
    why = by["3"]["mode_reason"]
    assert why["scatter_elevation_m"] < 0.01 < 0.3 < why["scatter_depth_m"]  # a plane: no scatter about it
    assert reg["modes"] == {"1": "ground", "2": "depth", "3": "elevation"}
    hold = inside()
    X, Y = hold[:, 0], hold[:, 1]
    top2 = at(ch.data_dir / by["2"]["file"], hold)
    assert np.abs(top2 - (ground(X, Y) - TOPSOIL)).max() < 0.01  # between the boreholes too
    top3 = at(ch.data_dir / by["3"]["file"], hold)
    err = np.abs(top3 - natural(X, Y))
    assert np.percentile(err, 95) < 0.02 and err.max() < 0.05  # under the mound, where no borehole reaches it
    # the same site all by elevation (0.1.0): the base of the topsoil is a dome kriged from the boreholes
    one = sf.build(ch, mode="elevation")
    top2 = at(ch.data_dir / next(e for e in one["contacts"] if e["unit"] == "2")["file"], hold)
    assert np.abs(top2 - (ground(X, Y) - TOPSOIL)).max() > 0.1
    # the modes given: in the API, in a text (Italian words too), on the command line
    reg = sf.build(ch, modes={"3": "depth"})
    e3 = next(e for e in reg["contacts"] if e["unit"] == "3")
    assert e3["mode"] == "depth" and e3["mode_reason"] == {"rule": "given"}
    assert sf.parse_modes("auto, US2=profondità; 3 = quota") == ("auto", {"US2": "depth", "3": "elevation"})
    assert sf.parse_modes("depth") == ("depth", {})
    reg = sf.build(ch, mode="elevation, 2=depth")
    assert reg["modes"] == {"1": "ground", "2": "depth", "3": "elevation"} and reg["params"]["modes"] == {"2": "depth"}
    assert sf.unit_of("US2", ["US 1", "US 2"]) == "US 2" and sf.unit_of("2", ["US 1", "US 2"]) == "US 2"
    with pytest.raises(ValueError, match="sconosciut"):
        sf.build(ch, mode="auto, 9=depth")
    with pytest.raises(ValueError):
        sf.parse_modes("sideways")
    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.plugins import Context

    plugin = GeoarchaeologyPlugin()
    plugin.activate(Context("geoarchaeology", tmp_path / "plugin"))
    assert plugin.command("surfaces", str(doc.path), "--mode", "auto,3=depth") == 0
    out = capsys.readouterr().out
    assert "Tetto di 2: kriging" in out or "Tetto di 2: costante" in out
    assert "per profondità (scelto)" in out and "per profondità (segue il terreno" in out
    assert ppg.Document.open(doc.path).chunk.meta["geoarchaeology"]["surfaces"]["modes"]["3"] == "depth"


# ------------------------------------------------------------------ 2. the stack: the less certain contact yields
def test_crossing_contacts_meet_where_the_better_known_one_is(tmp_path):
    g = np.full((1, 3), 10.0)
    upper = np.array([[7.0, 6.0, 12.0]])  # poorly known: variance 4 m²
    lower = np.array([[8.0, 5.0, 11.0]])  # well known: variance 0.01 m²; above the upper one in the first cell
    var = [np.full((1, 3), 4.0), np.full((1, 3), 0.01)]
    (u, lo), (cu, cl), counts = sf.clamp_stack(g, [upper, lower], variances=var)
    meet = (7.0 / 4.0 + 8.0 / 0.01) / (1 / 4.0 + 1 / 0.01)
    assert u[0, 0] == pytest.approx(meet) and lo[0, 0] == pytest.approx(meet) and meet > 7.99
    assert cu[0, 0] == pytest.approx(7.0 - meet) and cl[0, 0] == pytest.approx(8.0 - meet)  # raised 1 m, lowered 2.5 mm
    assert (u[0, 1], lo[0, 1]) == (6.0, 5.0)  # in order: untouched
    assert (u[0, 2], lo[0, 2]) == (10.0, 10.0)  # both above the ground: the ground is the limit
    assert counts == [(1, 0, 1), (1, 1, 0)]
    # without the variances the old rule: the upper contact wins
    (_u, lo2), _c, _n = sf.clamp_stack(g, [upper, lower])
    assert lo2[0, 0] == 7.0
    # a unit missing from a cell: its top and the top of the unit below are one surface there
    (u3, l3), _c, _n = sf.clamp_stack(g, [upper, lower], variances=var, ties=[np.array([[False, True, False]])])
    assert u3[0, 1] == l3[0, 1] == pytest.approx((6.0 / 4.0 + 5.0 / 0.01) / (1 / 4.0 + 1 / 0.01))
    # on the tell: the fill of the mound by elevation with a Gaussian model overshoots below the
    # natural; the natural (a plane, known to a centimetre) is not pushed down by it
    doc, ch, pts = mound_site(tmp_path)
    reg = sf.build(ch, mode="elevation", model="gaussian", trend="none")
    by = {e["unit"]: e for e in reg["contacts"]}
    hold = inside()
    top3 = at(ch.data_dir / by["3"]["file"], hold)
    assert np.percentile(np.abs(top3 - natural(hold[:, 0], hold[:, 1])), 95) < 0.05
    assert by["3"]["clamped"]["lowered_max_m"] < 0.05 and by["2"]["clamped"]["by_lower"] > 0
    corr = sf.load(ch, by["2"]["id"], 3)[0]
    assert np.nanmin(corr) < -0.1  # the uncertain fill raised to the natural


# ------------------------------------------------------------------ 3. the variogram model chosen
def test_auto_model_prefers_the_steadiest_within_five_per_cent():
    def cand(model_, trend, loo):
        return {"model": model_, "trend": trend, "loocv_rmse": loo}

    # the numbers of the top of the Bronze Age in the demonstration on Tell Sultan
    tried = [cand("gaussian", "none", 2.238), cand("gaussian", "linear", 2.273), cand("spherical", "none", 2.307),
             cand("exponential", "none", 2.592)]
    assert kriging.preferred(tried)["model"] == "spherical"  # 3 % worse than the Gaussian: as good as it
    tried = [cand("gaussian", "none", 2.577), cand("spherical", "none", 2.912)]
    assert kriging.preferred(tried)["model"] == "gaussian"  # 11 % better: clearly
    tried = [cand("exponential", "none", 0.271), cand("spherical", "none", 0.274), cand("spherical", "linear", 0.314)]
    assert kriging.preferred(tried) == tried[1]
    tried = [cand("spherical", "none", 0.5), cand("spherical", "linear", 1e-12)]  # a plane: the drift, clearly
    assert kriging.preferred(tried)["trend"] == "linear"
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 100, (30, 2))
    z = 0.002 * (xy[:, 0] - 50) ** 2 + 0.05 * rng.normal(size=30)
    _est, _var, rep = kriging.interpolate(xy, z, xy[:3] + 1.0)
    assert len(rep["candidates"]) == 6 and sum(c["chosen"] for c in rep["candidates"]) == 1
    chosen = next(c for c in rep["candidates"] if c["chosen"])
    least = min(c["loocv_rmse"] for c in rep["candidates"])
    assert chosen["loocv_rmse"] <= least * (1 + kriging.CLEAR_MARGIN) + 1e-9
    assert rep["least_error"]["loocv_rmse"] == pytest.approx(least) and rep["margin"] == kriging.CLEAR_MARGIN
    assert rep["variogram"]["model"] == chosen["model"] and rep["trend"] == chosen["trend"]


# ------------------------------------------------------------------ 4. a missing unit stays missing
def test_a_unit_missing_from_the_logs_stays_missing_around_them(tmp_path):
    doc, ch, pts = mound_site(tmp_path)
    reg = sf.build(ch)
    fill = next(e for e in reg["thickness"] if e["unit"] == "2")
    without = {b.id for b in bh.load(ch) if "2" not in b.units and "3" in b.units}
    assert fill["absent"]["cells"] > 0 and set(fill["absent"]["boreholes_without"]) == without
    th, tr = sf.load(ch, fill["id"])
    xy = np.array([[b.x, b.y] for b in bh.load(ch) if b.id in without])
    assert np.all(bh.raster_values(ch.data_dir / fill["file"], xy[:, 0], xy[:, 1]) == 0.0)  # at the boreholes
    # around them: wherever the nearest boreholes all lack the fill, it is not there
    grid = sf.grid_of(ch)
    c = grid.centres()
    from scipy.spatial import cKDTree

    holes = bh.load(ch)
    has = np.array(["2" in b.units for b in holes])
    _d, nearest = cKDTree(np.array([[b.x, b.y] for b in holes])).query(c, 4)
    none_near = ~has[nearest].any(1)
    values = th.ravel()
    ok = none_near & np.isfinite(values)
    assert ok.sum() > 5000 and values[ok].max() == 0.0
    # where the fill truly is not (no mound), no thin spurious fill between the boreholes
    X, Y = c[:, 0] - X0, c[:, 1] - Y0
    truly_without = mound(X, Y) < 0.01
    spurious = truly_without & np.isfinite(values) & (values > 0.05)
    assert spurious.sum() * CELL * CELL < 25.0, spurious.sum() * CELL * CELL


# ------------------------------------------------------------------ 5. the small things
def test_gaps_numbers_volumes_and_the_floor(tmp_path):
    # a gap inside one unit has no contact; one between two units has it in the middle
    doc, ch, pts = mound_site(tmp_path, dem_hole=True)
    (tmp_path / "g.csv").write_text("id,x,y,from,to,unit\n"
                                    f"G1,{X0 + 20},{Y0 + 80},0,1,a\nG1,{X0 + 20},{Y0 + 80},1.2,2,a\n"
                                    f"G1,{X0 + 20},{Y0 + 80},2.5,3,b\n")
    from ppg_geoarchaeology import GeoarchaeologyPlugin

    doc2 = ppg.Document()
    ch2 = doc2.add_chunk("Gaps")
    doc2.save(tmp_path / "gaps.ppg")
    rep = bh.import_files(ch2, [tmp_path / "g.csv"])
    assert rep["warnings"] == ["G1: vuoto fra 1 e 1,2 m dentro a (nessun contatto)",
                               "G1: vuoto fra 2 e 2,5 m (contatto a metà)"]
    # the numbers as the Italian interface writes them
    from ppg_geoarchaeology import texts

    assert texts.num(1234.5, 1) == "1.234,5" and texts.num(2.5) == "2,5" and texts.num(8.0) == "8"
    assert texts.signed(0.071, 3) == "+0,071" and texts.signed(-0.008, 3) == "-0,008"
    reg = sf.build(ch)
    text = GeoarchaeologyPlugin.surfaces_text(reg)
    assert "celle di 0,50 m" in text and "errore leave-one-out 0," in text and "0.5 m" not in text
    # volumes on the cells with a DEM; those without one (the ground there is the surface of the
    # collars) apart
    assert reg["without_dem"]["area_m2"] == pytest.approx(100.0, abs=10.0)
    fill = next(e for e in reg["thickness"] if e["unit"] == "1")
    th = sf.load(ch, fill["id"])[0]
    assert fill["outside_dem"]["area_m2"] == pytest.approx(100.0, abs=10.0)
    assert fill["volume_m3"] + fill["outside_dem"]["volume_m3"] == pytest.approx(np.nansum(th) * CELL * CELL, rel=1e-6)
    assert fill["volume_m3"] == pytest.approx((np.nansum(th) * CELL * CELL) - fill["outside_dem"]["volume_m3"])
    m = model.build(ch, clip=False)
    topsoil = next(r for r in m["units"] if r["unit"] == "1")
    assert topsoil["outside_dem"]["area_m2"] > 0 and topsoil["volume_m3"] < topsoil["mesh_volume_m3"]
    lines = (ch.data_dir / m["files"]["csv"]).read_text().splitlines()
    assert lines[0].endswith("outside_dem_volume_m3,outside_dem_area_m2")
    # the top of the last unit lower than every borehole bottom somewhere: the floor goes below it,
    # the solid keeps all its cells (0.1.0 cut a hole there)
    last = next(e for e in reg["contacts"] if e["unit"] == "3")
    import rasterio

    with rasterio.open(ch.data_dir / last["file"], "r+") as dst:
        band = dst.read(1)
        r, c = band.shape[0] // 2, band.shape[1] // 2
        band[r - 5:r + 5, c - 5:c + 5] -= 5.0  # a pit 5 m deep in the natural, in the middle
        dst.write(band, 1)
    m = model.build(ch, clip=False)
    lowest_top = float(np.nanmin(sf.load(ch, last["id"])[0]))
    deepest = min(b.z - b.bottom for b in bh.load(ch))
    assert m["floor"] < lowest_top < deepest and not m["floor_given"]
    natural_solid = next(r for r in m["units"] if r["unit"] == "3")
    assert natural_solid["cells"] == int(np.isfinite(model.coarsen(sf.load(ch, last["id"])[0], m["factor"])).sum())
