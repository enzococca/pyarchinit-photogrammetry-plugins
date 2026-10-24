"""The geoarchaeology plugin on a synthetic site with known surfaces: a ground tilted to the
north-east (the DEM), a topsoil 0.3 m thick (unit 1), a tilted plane (the top of unit 3) and a
bowl (the top of unit 4), 36 boreholes on a jittered grid, five of them ending inside unit 3.
The kriging must give back the plane and the bowl at points between the boreholes, the stack
must stay in order, the sections must follow the surfaces, the volumes of the model must be those
of the analytic solids; the tables are read in their variants, the pane fits a narrow dock and
traces a section with the drawing the window lends to plugins, the command line goes round the
whole workflow."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import pyarchinit_photogrammetry as ppg
from ppg_geoarchaeology import boreholes as bh
from ppg_geoarchaeology import export, kriging, logs, model, overlay, section
from ppg_geoarchaeology import surfaces as sf

CRS = "EPSG:32633"
X0, Y0 = 500000.0, 4500000.0
W, H, CELL = 100.0, 80.0, 0.5
TOPSOIL = 0.3
SHALLOW = (3, 10, 17, 24, 31)  # these boreholes end inside unit 3


def ground(X, Y):
    return 100.0 + 0.01 * X + 0.005 * Y


def plane(X, Y):
    return 98.5 + 0.004 * (X - 50) - 0.006 * (Y - 40)


def bowl(X, Y):
    return 96.0 + 0.0003 * ((X - 50) ** 2 + (Y - 40) ** 2)


def positions(n: int = 6, seed: int = 2) -> np.ndarray:
    """Boreholes on a jittered grid, metres from the lower-left corner of the site."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(5, 95, n), np.linspace(5, 75, n))
    return np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-3, 3, (n * n, 2))


def holdout(n: int = 200, seed: int = 5) -> np.ndarray:
    """Points between the boreholes (inside their hull)."""
    return np.random.default_rng(seed).uniform([15, 15], [85, 65], (n, 2))


def project(tmp_path: Path, dem: bool = True):
    """An empty project (EPSG:32633) with the DEM of the ground, 0.5 m cells."""
    import rasterio
    from rasterio.transform import from_origin

    doc = ppg.Document()
    ch = doc.add_chunk("Carotaggi")
    doc.save(tmp_path / "sito.ppg")
    ch.crs = CRS
    if dem:
        nc, nr = int(W / CELL), int(H / CELL)
        XX, YY = np.meshgrid((np.arange(nc) + 0.5) * CELL, H - (np.arange(nr) + 0.5) * CELL)
        path = ch.data_dir / "dem.tif"
        with rasterio.open(path, "w", driver="GTiff", height=nr, width=nc, count=1, dtype="float32", crs=CRS,
                           transform=from_origin(X0, Y0 + H, CELL, CELL), nodata=float("nan")) as dst:
            dst.write(ground(XX, YY).astype(np.float32), 1)
        ch._set_product("dem", path)
    doc.save()
    return doc, ch


def logs_of(X: float, Y: float, shallow: bool) -> tuple[float, list]:
    """The depth reached and the intervals of a borehole at ``(X, Y)``."""
    g = ground(X, Y)
    d2, d3 = g - plane(X, Y), g - bowl(X, Y)
    end = d2 + 0.5 if shallow else d3 + 0.6
    rows = [(0.0, TOPSOIL, "1", "humus", "10YR 3/2"), (TOPSOIL, d2, "2", "limo sabbioso", "10YR 5/4"),
            (d2, min(d3, end), "3", "argilla grigia", "2.5Y 5/2")]
    if not shallow:
        rows.append((d3, end, "4", "sabbia gialla sterile", "10YR 7/6"))
    return end, rows


def comma(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def write_tables(folder: Path, pts: np.ndarray) -> tuple[Path, Path]:
    """Italian tables: semicolons, decimal commas."""
    collars = ["sondaggio;x;y;quota;profondita"]
    intervals = ["sondaggio;da;a;us;descrizione;colore"]
    for i, (X, Y) in enumerate(pts):
        end, rows = logs_of(X, Y, i in SHALLOW)
        collars.append(f"S{i + 1};{comma(X0 + X, 3)};{comma(Y0 + Y, 3)};{comma(ground(X, Y))};{comma(end, 3)}")
        for a, b, u, desc, col in rows:
            intervals.append(f"S{i + 1};{comma(a)};{comma(b)};{u};{desc};{col}")
    c, iv = folder / "collari.csv", folder / "intervalli.csv"
    c.write_text("\n".join(collars) + "\n", encoding="utf-8")
    iv.write_text("\n".join(intervals) + "\n", encoding="utf-8")
    return c, iv


def site(tmp_path: Path, dem: bool = True):
    doc, ch = project(tmp_path, dem)
    pts = positions()
    bh.import_files(ch, write_tables(tmp_path, pts))
    return doc, ch, pts


def at(path, local_xy) -> np.ndarray:
    """A raster sampled at points given in metres from the corner of the site."""
    xy = np.asarray(local_xy, float)
    return bh.raster_values(path, X0 + xy[:, 0], Y0 + xy[:, 1])


# ------------------------------------------------------------------ kriging alone
def test_kriging_gives_back_a_tilted_plane_and_a_bowl():
    pts = positions()
    hold = holdout()
    xy = pts + [X0, Y0]
    for name, f, auto_tol, free_tol in (("plane", plane, 0.0005, 0.005), ("bowl", bowl, 0.006, 0.02)):
        z = f(pts[:, 0], pts[:, 1])
        truth = f(hold[:, 0], hold[:, 1])
        est, var, rep = kriging.interpolate(xy, z, hold + [X0, Y0])
        rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
        assert rep["method"] == "kriging" and rmse < auto_tol, (name, rmse, rep)
        assert np.all(var >= 0) and np.isfinite(rep["loocv_rmse"])
        if name == "plane":  # the linear drift is chosen: universal kriging gives the plane back exactly
            assert rep["trend"] == "linear"
        for m in kriging.MODELS:  # every model alone, even without the drift, stays close
            est, _v, rep = kriging.interpolate(xy, z, hold + [X0, Y0], model=m, trend="none")
            assert rep["variogram"]["model"] == m and np.sqrt(np.mean((est - truth) ** 2)) < free_tol, (name, m)
    # an exact plane (no residual at all): the drift alone, given back exactly
    z = 98.5 + 0.004 * pts[:, 0] - 0.006 * pts[:, 1]
    est, _v, rep = kriging.interpolate(pts, z, hold)
    assert rep["method"] == "kriging" and rep["trend"] == "linear"
    assert np.allclose(est, 98.5 + 0.004 * hold[:, 0] - 0.006 * hold[:, 1], atol=1e-9)
    # the kriging variance: none on a datum, growing away from the data
    z = bowl(pts[:, 0], pts[:, 1])
    est, var, _rep = kriging.interpolate(xy, z, np.vstack([xy[:1], xy[:1] + [200.0, 0.0]]), model="spherical",
                                         trend="none")
    assert est[0] == pytest.approx(z[0], abs=1e-6) and var[0] < 1e-9 < var[1]


def test_leave_one_out_from_the_inverse_matches_the_brute_force():
    pts = positions()
    z = bowl(pts[:, 0], pts[:, 1]) + 0.01 * np.sin(pts[:, 0])
    for drift in (0, 1):
        lags, gam, pairs = kriging.empirical(pts, z)
        v = kriging.fit(lags, gam, pairs, "exponential")
        fast = kriging.Kriging(pts, z, v, drift).loocv()
        brute = [z[i] - kriging.Kriging(np.delete(pts, i, 0), np.delete(z, i), v, drift)
                 .predict(pts[i:i + 1], variance=False)[0][0] for i in range(len(z))]
        assert np.allclose(fast, brute, atol=1e-9)


def test_variogram_fit_recovers_a_known_model():
    lags = np.linspace(2.0, 60.0, 15)
    for m in kriging.MODELS:
        gamma = kriging.semivariance(m, lags, 0.02, 0.5, 35.0)
        v = kriging.fit(lags, gamma, np.full(15, 40), m)
        assert v.model == m
        assert v.sill == pytest.approx(0.5, rel=0.05) and v.range == pytest.approx(35.0, rel=0.05)
        assert v.nugget == pytest.approx(0.02, abs=0.01)


def test_few_boreholes_moving_neighbourhood_and_duplicates():
    pts = positions()[:4]
    z = bowl(pts[:, 0], pts[:, 1])
    est, var, rep = kriging.interpolate(pts, z, np.vstack([pts, [[50.0, 40.0]]]))
    assert rep["method"] == "idw" and np.allclose(est[:4], z) and np.isnan(var).all()
    # two boreholes at the same place are one (their mean): the kriging matrix stays regular
    xy = np.vstack([positions(), positions()[:1]])
    zz = bowl(xy[:, 0], xy[:, 1])
    est, _v, rep = kriging.interpolate(xy, zz, holdout(20))
    assert rep["merged"] == 1 and rep["method"] == "kriging"
    # more points than one system: the nearest ones for each target, the plane removed first
    big = np.random.default_rng(1).uniform([0, 0], [100, 80], (kriging.MAX_GLOBAL + 200, 2))
    hold = holdout()
    est, var, rep = kriging.interpolate(big, bowl(big[:, 0], big[:, 1]), hold)
    assert rep["neighbourhood"] == "moving"
    assert np.sqrt(np.mean((est - bowl(hold[:, 0], hold[:, 1])) ** 2)) < 0.003 and np.all(var >= 0)


def test_cross_check_with_pykrige():
    pykrige = pytest.importorskip("pykrige")
    pts = positions()
    z = bowl(pts[:, 0], pts[:, 1])
    hold = holdout(50)
    v = kriging.Variogram("spherical", 0.001, 0.15, 60.0)
    ours, var = kriging.Kriging(pts, z, v, 0).predict(hold)
    ok = pykrige.ok.OrdinaryKriging(pts[:, 0], pts[:, 1], z, variogram_model="spherical",
                                    variogram_parameters={"sill": 0.151, "range": 60.0, "nugget": 0.001})
    theirs, ss = ok.execute("points", hold[:, 0], hold[:, 1])
    assert np.allclose(ours, theirs, atol=1e-6) and np.allclose(var, ss, atol=1e-6)


# ------------------------------------------------------------------ the tables
def test_import_of_the_tables_in_their_variants(tmp_path):
    doc, ch = project(tmp_path)
    pts = positions()
    rep = bh.import_files(ch, write_tables(tmp_path, pts))
    assert rep["boreholes"] == 36 and rep["intervals"] == 36 * 4 - len(SHALLOW) and rep["units"] == ["1", "2", "3", "4"]
    assert rep["tables"] == ["collars", "intervals"] and not rep["warnings"] and rep["z_from_dem"] == 0
    assert abs(rep["dem_check"]["median_m"]) < 1e-3 and abs(rep["dem_check"]["max_m"]) < 1e-3
    s1 = bh.get(ch, "S1")
    assert s1.x == pytest.approx(X0 + pts[0, 0], abs=1e-3) and s1.units == ["1", "2", "3", "4"]
    assert s1.intervals[1].top == pytest.approx(TOPSOIL) and s1.intervals[1].description == "limo sabbioso"
    assert (bh.folder(ch) / "sources" / "collari.csv").exists() and export.layers_path(ch).exists()
    # one table with everything, in English, commas; a collar without z takes the DEM's
    lines = ["hole_id,easting,northing,elevation,from,to,unit,description,lithology"]
    for i, (X, Y) in enumerate(pts[:5]):
        _end, rows = logs_of(X, Y, False)
        z = "" if i == 0 else f"{ground(X, Y):.4f}"
        for a, b, u, d, _c in rows:
            lines.append(f"C{i + 1},{X0 + X:.3f},{Y0 + Y:.3f},{z},{a:.4f},{b:.4f},{u},{d},{'clay' if u == '3' else ''}")
    one = tmp_path / "one.csv"
    one.write_text("\n".join(lines) + "\n")
    rep = bh.import_files(ch, [one])
    assert rep["tables"] == ["combined"] and rep["read"] == 5 and rep["boreholes"] == 41 and rep["z_from_dem"] == 1
    c1 = bh.get(ch, "C1")
    assert c1.z_source == "dem" and c1.z == pytest.approx(ground(*pts[0]), abs=1e-3)
    assert c1.intervals[2].lithology == "clay" and bh.hatch_of("clay") == "--"
    # intervals by their elevations, tab separated; depths in centimetres
    tab = tmp_path / "quote.tsv"
    X, Y = pts[6]
    g = ground(X, Y)
    tab.write_text("id\tx\ty\tz\tquota_tetto\tquota_letto\tus\n"
                   f"Q1\t{X0 + X}\t{Y0 + Y}\t{g}\t{g}\t{g - 0.5}\tA\n"
                   f"Q1\t{X0 + X}\t{Y0 + Y}\t{g}\t{g - 0.5}\t{g - 1.25}\tB\n")
    cm = tmp_path / "cm.csv"
    cm.write_text("Sondaggio;Da (cm);A (cm);Unità\nS1;0;30;1\nS1;30;80;2\n")
    bh.import_files(ch, [tab])
    q1 = bh.get(ch, "Q1")
    assert [round(i.top, 6) for i in q1.intervals] == [0.0, 0.5] and q1.intervals[1].base == pytest.approx(1.25)
    rows, names = bh.read_table(cm)
    assert names["from"] == "Da (cm)" and rows[1]["from"] == pytest.approx(0.3) and rows[1]["to"] == pytest.approx(0.8)
    # replace: only the boreholes of the new table remain
    rep = bh.import_files(ch, [one], replace=True)
    assert rep["boreholes"] == 5 and {b.id for b in bh.load(ch)} == {f"C{i}" for i in range(1, 6)}


def test_warnings_order_and_colours(tmp_path):
    doc, ch = project(tmp_path)
    collars = tmp_path / "c.csv"
    collars.write_text("id,x,y,z\nA1,500010,4500010,100.1\nA2,500020,4500010,100.2\nA3,500030,4500010,100.3\n"
                       "A4,500040,4500010,100.4\n")
    intervals = tmp_path / "i.csv"
    intervals.write_text("id,from,to,unit,colour\n"
                         "A1,0,1,top,10YR 4/3\nA1,0.9,2,mid,#336699\nA1,2.5,3,low,200 100 50\n"  # overlap, gap
                         "A2,0,1,top,\nA2,1,2,mid,\nA2,2,1.5,low,\nA2,2,2.4,,\n"  # base above top, no unit
                         "A3,0,1,top,\nA3,1,2,low,\nA3,2,3,mid,\n"  # mid below low: against the others
                         "A4,0,1,top,\nA4,1,2,low,\n"  # no mid
                         "X9,0,1,top,\n")  # no collar
    rep = bh.import_files(ch, [collars, intervals])
    text = "\n".join(rep["warnings"])
    for word in ("A1: intervalli sovrapposti", "A1: vuoto fra 2 e 2,5 m (contatto a metà)",
                 "A2: intervallo 2-1,5 m con la base sopra", "A2: intervallo 2-2,4 m senza unità",
                 "X9: intervalli senza la posizione"):
        assert word in text, (word, text)
    assert bh.get(ch, "A1").intervals[1].top == pytest.approx(1.0)  # the overlap cut at the base of the one above
    assert bh.order(ch) == ["top", "mid", "low"]  # two boreholes against one
    assert any("mid sopra low" in c for c in bh.meta(ch)["order_conflicts"])
    data, notes = bh.contact_data(bh.load(ch), bh.order(ch))
    assert any(n.startswith("A3:") for n in notes)
    by = {u["code"]: u["colour"] for u in bh.units(ch)}
    assert by["mid"] == "#336699" and by["low"] == "#c86432"
    assert bh.parse_colour("brown") and bh.parse_colour("nonsense") is None
    if bh.munsell_to_hex("10YR 4/3") is not None:  # colour-science: a brown of the soil charts
        r, g, b = (int(by["top"][i:i + 2], 16) for i in (1, 3, 5))
        assert r > g > b and 80 < r < 140
    assert bh.set_order(ch, ["low", "top"]) == ["low", "top", "mid"] and bh.meta(ch)["order_manual"]
    # the contacts at a borehole: the top of a unit missing from it is the contact below; a unit not
    # reached is below the bottom of the hole
    data, _ = bh.contact_data(bh.load(ch), ["top", "mid", "low"])
    a4 = {cd.unit: cd.depth[cd.ids.index("A4")] for cd in data if "A4" in cd.ids}
    assert a4 == {"top": 0.0, "mid": 1.0, "low": 1.0}
    low = data[2]
    assert "A2" not in low.ids and low.below_depth[low.below_ids.index("A2")] == pytest.approx(2.0)


# ------------------------------------------------------------------ surfaces
def test_surfaces_give_back_the_contacts(tmp_path):
    doc, ch, pts = site(tmp_path)
    reg = sf.build(ch)
    contacts = {e["unit"]: e for e in reg["contacts"]}
    assert [e["unit"] for e in reg["contacts"]] == ["1", "2", "3", "4"]
    assert contacts["1"]["method"] == "ground" and contacts["1"]["mode"] == "ground"
    # the base of the topsoil follows the ground: by depth; the plane and the bowl by elevation
    assert contacts["2"]["mode"] == "depth" and contacts["2"]["mode_reason"]["rule"] == "follows_ground"
    assert contacts["3"]["mode"] == "elevation" and contacts["3"]["trend"] == "linear"
    assert contacts["4"]["mode"] == "elevation" and reg["modes"] == {"1": "ground", "2": "depth", "3": "elevation",
                                                                      "4": "elevation"}
    assert contacts["4"]["below"] == len(SHALLOW)  # the shallow boreholes bound the top of 4 from above
    hold = holdout()
    truth = {"1": ground, "2": lambda X, Y: ground(X, Y) - TOPSOIL, "3": plane, "4": bowl}
    errors = {}
    for unit, e in contacts.items():
        err = at(ch.data_dir / e["file"], hold) - truth[unit](hold[:, 0], hold[:, 1])
        assert np.isfinite(err).all()
        errors[unit] = (float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max()))
    assert errors["1"][1] < 1e-4 and errors["2"][1] < 0.001 and errors["3"][1] < 0.001
    assert errors["4"][0] < 0.006 and errors["4"][1] < 0.03, errors  # the bowl: millimetres between the boreholes
    import rasterio

    with rasterio.open(ch.data_dir / contacts["4"]["file"]) as src:
        assert src.count == 3 and src.dtypes[0] == "float32" and src.crs.to_epsg() == 32633
        z, var, corr = (src.read(k) for k in (1, 2, 3))
        assert np.isnan(z[0, 0]) and np.isnan(z[-1, -1])  # outside the hull of the boreholes plus the buffer
        assert np.all(var[np.isfinite(z)] >= 0) and np.nanmax(np.abs(corr)) < 1e-9  # nothing to clamp here
    thick = {e["unit"]: e for e in reg["thickness"]}
    t1 = at(ch.data_dir / thick["1"]["file"], hold)
    assert np.allclose(t1, TOPSOIL, atol=0.001)
    t3 = at(ch.data_dir / thick["3"]["file"], hold)
    assert np.allclose(t3, plane(hold[:, 0], hold[:, 1]) - bowl(hold[:, 0], hold[:, 1]), atol=0.03)
    assert thick["2"]["volume_m3"] > 0 and set(thick) == {"1", "2", "3"}
    # depth mode: the depths below the ground hung from the DEM
    reg = sf.build(ch, mode="depth")
    e = next(e for e in reg["contacts"] if e["unit"] == "2")
    assert np.allclose(at(ch.data_dir / e["file"], hold), ground(hold[:, 0], hold[:, 1]) - TOPSOIL, atol=0.001)
    assert export.qgis_entries(ch)[0] and "kriging" in export.summary(ch)


def test_without_a_dem_the_ground_is_the_surface_of_the_collars(tmp_path):
    doc, ch, pts = site(tmp_path, dem=False)
    reg = sf.build(ch)
    assert reg["ground"] == "collars" and reg["grid"]["cell"] in sf.NICE
    top1 = next(e for e in reg["contacts"] if e["unit"] == "1")
    hold = holdout()
    assert np.allclose(at(ch.data_dir / top1["file"], hold), ground(hold[:, 0], hold[:, 1]), atol=0.002)


def test_a_bottom_above_the_contact_keeps_it_deeper(tmp_path):
    doc, ch = project(tmp_path)
    collars, intervals = ["id,x,y"], ["id,from,to,unit"]
    k = 0
    for X in (20.0, 40.0, 60.0):
        for Y in (20.0, 40.0, 60.0):
            k += 1
            collars.append(f"B{k},{X0 + X},{Y0 + Y}")
            if (X, Y) == (40.0, 40.0):  # ends at 1.5 m still in unit 1: its base is deeper here
                intervals.append(f"B{k},0,1.5,1")
            else:
                intervals += [f"B{k},0,1.0,1", f"B{k},1.0,2.0,2"]
    (tmp_path / "c.csv").write_text("\n".join(collars) + "\n")
    (tmp_path / "i.csv").write_text("\n".join(intervals) + "\n")
    rep = bh.import_files(ch, [tmp_path / "c.csv", tmp_path / "i.csv"])
    assert rep["z_from_dem"] == 9
    reg = sf.build(ch, min_points=5)
    top2 = next(e for e in reg["contacts"] if e["unit"] == "2")
    assert top2["below"] == 1 and top2["bottoms_used"] == 1
    centre = at(ch.data_dir / top2["file"], [[40.0, 40.0]])[0]
    # down to the bottom of the hole (the cells around it 25 cm away), half a metre under the neighbours' contact
    assert centre <= ground(40.0, 40.0) - 1.5 + 0.02 and centre < ground(40.0, 40.0) - 1.0 - 0.4


def test_the_stack_stays_in_order_and_under_the_ground(tmp_path):
    # the clamping itself, on arrays
    ground_ = np.array([[10.0, 10.0], [10.0, np.nan]])
    c0 = np.array([[11.0, 9.0], [np.nan, 8.0]])
    c1 = np.array([[9.5, 9.5], [7.0, 9.0]])
    (k0, k1), (r0, r1), counts = sf.clamp_stack(ground_, [c0, c1])
    assert np.allclose(k0, [[10, 9], [np.nan, 8]], equal_nan=True) and np.allclose(k1, [[9.5, 9], [7, 8]])
    assert counts == [(1, 0, 0), (0, 2, 0)] and r0[0, 0] == 1.0 and r1[0, 1] == pytest.approx(0.5) and r1[1, 1] == 1.0
    # a lens of unit b in the middle of the site (a gaussian model overshoots around it), and the collars
    # of its boreholes 1.2 m above the DEM (their top of b rises 0.2 m above the ground)
    doc, ch = project(tmp_path)
    rows, lens = ["id,x,y,z,from,to,unit"], set()
    for i, (X, Y) in enumerate(positions()):
        inside = np.hypot(X - 50, Y - 40) < 15
        z = ground(X, Y) + (1.2 if inside else 0.0)
        p = f"S{i + 1},{X0 + X:.3f},{Y0 + Y:.3f},{z:.4f}"
        if inside:
            lens.add(f"S{i + 1}")
            rows += [f"{p},0,1,a", f"{p},1,2,b", f"{p},2,3,c"]
        else:
            rows += [f"{p},0,1,a", f"{p},1,3,c"]
    (tmp_path / "lens.csv").write_text("\n".join(rows) + "\n")
    rep = bh.import_files(ch, [tmp_path / "lens.csv"])
    assert len(lens) >= 3 and rep["dem_check"]["worst"] in lens
    assert rep["dem_check"]["max_m"] == pytest.approx(1.2, abs=0.01)
    reg = sf.build(ch, model="gaussian", trend="none", mode="elevation")
    by = {e["unit"]: e for e in reg["contacts"]}
    assert by["c"]["clamped"]["cells"] > 0 and by["b"]["clamped"]["by_ground"] > 0
    assert by["b"]["clamped"]["max_m"] > 0.1
    dem = sf.dem_on_grid(ch, sf.grid_of(ch))
    tops = [sf.load(ch, e["id"], 1)[0] for e in reg["contacts"]]
    for k, top in enumerate(tops):
        both = np.isfinite(top) & np.isfinite(dem)
        assert np.all(top[both] <= dem[both] + 1e-6)
        if k:
            both = np.isfinite(top) & np.isfinite(tops[k - 1])
            assert np.all(top[both] <= tops[k - 1][both] + 1e-9)
    corr = sf.load(ch, by["c"]["id"], 3)[0]  # lowered (+) or raised (-) by the stack
    assert (np.abs(corr) > 1e-6).sum() == by["c"]["clamped"]["cells"]
    cl = by["c"]["clamped"]
    assert cl["cells"] == cl["by_ground"] + cl["by_upper"] + cl["by_lower"]
    thick = sf.load(ch, "thickness_b", 1)[0]
    assert np.nanmin(thick) >= 0


# ------------------------------------------------------------------ sections and logs
def test_section_profile_boreholes_and_drawings(tmp_path):
    from shapely.geometry import LineString, Point

    doc, ch, pts = site(tmp_path)
    sf.build(ch)
    trace = np.array([[5.0, 10.0], [50.0, 45.0], [95.0, 20.0]])
    prof = section.profile(ch, trace + [X0, Y0])
    X, Y = prof["x"] - X0, prof["y"] - Y0
    assert prof["length"] == pytest.approx(LineString(trace).length)
    assert np.allclose(prof["ground"], ground(X, Y), atol=1e-3)
    tops = dict(prof["tops"])
    ok = np.isfinite(tops["3"])
    assert ok.mean() > 0.95 and np.allclose(tops["3"][ok], plane(X[ok], Y[ok]), atol=0.002)
    ok = np.isfinite(tops["4"])
    assert np.abs(tops["4"][ok] - bowl(X[ok], Y[ok])).max() < 0.05
    line = LineString(trace)
    expected = {f"S{i + 1}" for i, p in enumerate(pts) if line.distance(Point(p)) <= 6.0}
    near = section.near(ch, trace + [X0, Y0], 6.0)
    assert {h["borehole"].id for h in near} == expected and len(expected) >= 3
    h = near[0]
    p = Point(h["borehole"].x - X0, h["borehole"].y - Y0)
    assert h["station"] == pytest.approx(line.project(p), abs=1e-6) and abs(h["offset"]) == pytest.approx(line.distance(p))
    e = section.build(ch, trace + [X0, Y0], name="Transetto A", distance=6.0, scale=500, exaggeration=5)
    assert e["scale"] == 500 and e["exaggeration"] == 5 and {b["id"] for b in e["boreholes"]} == expected
    files = {k: ch.data_dir / v for k, v in e["files"].items()}
    assert all(p.exists() for p in files.values())
    width_pt = float(re.search(r'width="([\d.]+)pt"', files["svg"].read_text()).group(1))
    assert width_pt * 25.4 / 72 == pytest.approx(line.length * 1000 / 500 + 22 + 62, abs=0.5)  # true to the scale
    import ezdxf

    dxf = ezdxf.readfile(files["dxf"])
    layers = {layer.dxf.name for layer in dxf.layers}
    assert {"GEO_TERRENO", "GEO_CONTATTI", "GEO_SONDAGGI", "GEO_UNITA_2", "GEO_UNITA_3"} <= layers
    msp = dxf.modelspace()
    assert len(msp.query('HATCH[layer=="GEO_UNITA_2"]')) >= 1 and len(msp.query('LWPOLYLINE[layer=="GEO_TERRENO"]')) == 1
    rows = files["csv"].read_text().splitlines()
    assert rows[0] == "distance,x,y,ground,top_1,top_2,top_3,top_4" and len(rows) == len(prof["distance"]) + 1
    through = section.through(ch, ["S1", "S8", "S15"])
    assert np.allclose(through, pts[[0, 7, 14]] + [X0, Y0])
    assert [s["id"] for s in section.entries(ch)] == ["transetto_a"]
    assert section.remove(ch, "transetto_a") == 1 and not files["svg"].exists()


def test_stratigraphic_columns(tmp_path):
    doc, ch, pts = site(tmp_path)
    one = logs.draw(ch, ["S1"])
    assert [p.suffix for p in one] == [".svg", ".png"] and all(p.stat().st_size > 1000 for p in one)
    panel = logs.draw(ch, ["S1", "S2", "S3", "S4"], path=tmp_path / "panel.png")
    assert panel == [tmp_path / "panel.png"] and panel[0].exists()
    with pytest.raises(ValueError):
        logs.draw(ch, ["S99"])
    # a borehole without elevation (no DEM, none in the table): the panel goes by depth
    (tmp_path / "b").mkdir()
    doc2, ch2 = project(tmp_path / "b", dem=False)
    (tmp_path / "b" / "t.csv").write_text("id,x,y,from,to,unit\nP1,1,1,0,1,a\nP1,1,1,1,2,b\nP2,5,1,0,1.5,a\n")
    bh.import_files(ch2, [tmp_path / "b" / "t.csv"])
    assert bh.get(ch2, "P1").z is None
    out = logs.draw(ch2, None, path=tmp_path / "b" / "depth.svg")
    assert "Profondit" in out[0].read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        logs.draw(ch2, None, vertical="elevation")
    with pytest.raises(RuntimeError, match="quota"):  # no elevation anywhere: no surface by elevation
        sf.build(ch2)


# ------------------------------------------------------------------ model
def test_model_volumes_against_the_analytic_solids(tmp_path):
    import rasterio
    import trimesh

    doc, ch, pts = site(tmp_path)
    reg = sf.build(ch, buffer=0.0)  # the hull of the boreholes: the kriging interpolates, it does not extrapolate
    grid = sf.grid_of(ch)
    floor = min(b.z - b.bottom for b in bh.load(ch))
    centres = grid.centres()
    X = centres[:, 0].reshape(grid.height, grid.width) - X0
    Y = centres[:, 1].reshape(grid.height, grid.width) - Y0
    truth = [ground(X, Y), ground(X, Y) - TOPSOIL, plane(X, Y), bowl(X, Y)]

    def check(tolerance):
        out = model.build(ch, clip=False)
        assert out["factor"] == 1 and out["cell"] == CELL and out["floor"] == pytest.approx(floor)
        tops = [v for _u, v in sf.tops(ch)]
        rows = {r["unit"]: r for r in out["units"]}
        for k, unit in enumerate(["1", "2", "3", "4"]):
            base = tops[k + 1] if k + 1 < len(tops) else np.minimum(np.full_like(tops[k], floor), tops[k])
            mask = np.isfinite(tops[k]) & np.isfinite(base) & (tops[k] - base > 0.001)
            true_base = truth[k + 1] if k + 1 < len(truth) else np.minimum(floor, truth[k])
            analytic = float((truth[k] - true_base)[mask].sum() * CELL * CELL)
            r = rows[unit]
            assert r["cells"] == int(mask.sum()) and r["cells"] > 1000
            assert r["volume_m3"] == pytest.approx(analytic, rel=tolerance[k]), (unit, r["volume_m3"], analytic)
            assert r["mesh_volume_m3"] == pytest.approx(r["volume_m3"], rel=0.005)
        return out

    out = check([0.001, 0.001, 0.005, 0.01])
    for r in out["units"]:
        mesh = trimesh.load(ch.data_dir / r["ply"], process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert mesh.volume == pytest.approx(r["mesh_volume_m3"], rel=1e-6)
        assert abs(mesh.vertices[:, 0].mean() - (X0 + 50)) < 20  # world coordinates, double precision
    obj = (ch.data_dir / out["files"]["obj"]).read_text()
    assert [ln for ln in obj.splitlines() if ln.startswith("o ")] == ["o 1", "o 2", "o 3", "o 4"]
    assert (model.folder(ch) / "unita.mtl").exists()
    lines = (ch.data_dir / out["files"]["csv"]).read_text().splitlines()
    assert lines[0].startswith("unit,description,volume_m3") and len(lines) == 5
    # the analytic surfaces written into the rasters: the volumes are those of the solids
    for e, values in zip(reg["contacts"], truth):
        with rasterio.open(ch.data_dir / e["file"], "r+") as dst:
            band = dst.read(1)
            dst.write(np.where(np.isfinite(band), values, np.nan).astype(np.float32), 1)
    truth = [v.astype(np.float32).astype(np.float64) for v in truth]
    check([1e-6] * 4)


def test_the_mesh_is_the_one_of_the_program_proxies(tmp_path):
    """The prisms are made as ``archaeo.proxies.unit_proxy`` makes them: the same volume."""
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.proxies import unit_proxy
    from pyarchinit_photogrammetry.engine.dem import write_geotiff

    rng = np.random.default_rng(0)
    top = 10.0 + np.cumsum(rng.normal(0, 0.02, (12, 15)), axis=1)
    bottom = top - 0.5 - rng.uniform(0, 0.3, top.shape)
    tr = from_origin(1000.0, 2000.0, 0.25, 0.25)
    write_geotiff(tmp_path / "top.tif", top.astype(np.float32), (1000.0, 2000.0), 0.25, None)
    write_geotiff(tmp_path / "bottom.tif", bottom.astype(np.float32), (1000.0, 2000.0), 0.25, None)
    poly = box(1000.0 + 2 * 0.25, 2000.0 - 10 * 0.25, 1000.0 + 13 * 0.25, 2000.0 - 1 * 0.25)  # whole cells
    proxy = unit_proxy(tmp_path / "top.tif", poly, base=tmp_path / "bottom.tif")
    mask = np.zeros(top.shape, bool)
    mask[1:10, 2:13] = True
    t32, b32 = top.astype(np.float32).astype(float), bottom.astype(np.float32).astype(float)
    verts, faces = model.prism_mesh(t32, b32, mask, tr)
    assert model.signed_volume(verts, faces) == pytest.approx(proxy.volume_m3, rel=1e-9)
    assert len(faces) == len(proxy.mesh.faces)
    assert model.unit_key("US 12") == ("US", "12") and model.unit_key("USM3") == ("USM", "3")
    assert model.unit_key("12") == (None, "12") and model.unit_key("Sabbia") == (None, "sabbia")


def test_the_model_is_clipped_to_the_units_drawn(tmp_path):
    from shapely.geometry import Polygon

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    doc, ch, pts = site(tmp_path)
    sf.build(ch)
    square = Polygon([(X0 + 30, Y0 + 20), (X0 + 60, Y0 + 20), (X0 + 60, Y0 + 50), (X0 + 30, Y0 + 50)])
    other = Polygon([(X0 + 70, Y0 + 60), (X0 + 80, Y0 + 60), (X0 + 80, Y0 + 70)])
    ch.meta["drawings"] = {"units": [UnitPolygon(square, us="2", tipo="US", sito="Sito").to_dict(),
                                     UnitPolygon(other, us="7", tipo="USM").to_dict()]}
    reg = model.build(ch)
    rows = {r["unit"]: r for r in reg["units"]}
    assert rows["2"]["clipped"] and rows["2"]["clipped_to"] == ["US 2"] and not rows["3"]["clipped"]
    assert rows["2"]["area_m2"] == pytest.approx(square.area, rel=0.01)
    c = sf.grid_of(ch).centres()
    Xc = c[:, 0] - X0
    Yc = c[:, 1] - Y0
    sel = (Xc > 30) & (Xc < 60) & (Yc > 20) & (Yc < 50)
    analytic = float((ground(Xc[sel], Yc[sel]) - TOPSOIL - plane(Xc[sel], Yc[sel])).sum() * CELL * CELL)
    assert rows["2"]["volume_m3"] == pytest.approx(analytic, rel=0.002)
    assert model.polygons_for("US 2", model.drawn_units(ch)) and not model.polygons_for("USM 2", model.drawn_units(ch))



# ------------------------------------------------------------------ exports, plugin, command line, pane
def test_exports(tmp_path):
    from pyarchinit_photogrammetry.io import gpkg

    doc, ch, pts = site(tmp_path)
    reg = sf.build(ch)
    section.build(ch, section.through(ch, ["S1", "S8", "S15"]), name="T1")
    model.build(ch)
    out = export.export_gpkg(ch, tmp_path / "out" / "geo.gpkg")
    holes = gpkg.read_table(out, "sondaggi")
    assert len(holes) == 36 and holes[0][0].geom_type == "Point"
    s1 = next(a for _g, a in holes if a["id"] == "S1")
    assert s1["units"] == "1 / 2 / 3 / 4" and s1["intervals"] == 4 and s1["z"] == pytest.approx(ground(*pts[0]), abs=1e-3)
    ivs = export.read_attributes(out, "intervalli")
    assert len(ivs) == 36 * 4 - len(SHALLOW) and {"sondaggio", "da_m", "a_m", "quota_tetto", "unita"} <= set(ivs[0])
    import sqlite3

    con = sqlite3.connect(out)
    kinds = dict(con.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall())
    con.close()
    assert kinds == {"sondaggi": "features", "intervalli": "attributes", "sezioni": "features"}
    assert len(gpkg.read_table(out, "sezioni")) == 1
    beside = sorted(p.name for p in out.parent.iterdir())
    assert "geo_volumi.csv" in beside and len([n for n in beside if n.endswith(".tif")]) == 4 + 3
    single = export.export_raster(ch, reg["thickness"][0]["id"], tmp_path / "t.tif")
    assert single.exists()
    rasters, vectors = export.qgis_entries(ch)
    assert len(rasters) == 7 and {v["table"] for v in vectors} == {"sondaggi", "sezioni"}
    assert any(r["ramp"] == "thermal" for r in rasters)
    lo, hi = overlay.limits(sf.load(ch, reg["thickness"][1]["id"])[0], zero=True)
    assert lo == 0 and hi > 1
    rgba = overlay.render(np.array([[0.0, 1.0, np.nan]]), 0, 1, "thickness")
    assert rgba[0, 2, 3] == 0 and rgba[0, 0, 0] > rgba[0, 1, 0]


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("geoarchaeology")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "geoarchaeology"] == [
        "Importa sondaggi...", "Superfici di contatto (kriging)...", "Sezione geoarcheologica...",
        "Colonne stratigrafiche...", "Modello 3D delle unità...", "Apri come modello di un nuovo chunk",
        "Geoarcheologia...", "Nascondi la sovrapposizione"]
    assert "geoarchaeology" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "geoarchaeology"] == [".gpkg"]


def test_command_line_round_trip(tmp_path, capsys):
    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = project(tmp_path)
    collars, intervals = write_tables(tmp_path, positions())
    plugin = GeoarchaeologyPlugin()
    plugin.activate(Context("geoarchaeology", tmp_path / "plugin"))
    p = str(doc.path)
    assert plugin.command("import", p, str(collars), str(intervals)) == 0
    assert "36 sondaggi letti" in capsys.readouterr().out
    assert plugin.command("surfaces", p, "--model", "auto") == 0
    assert "Tetto di 4: kriging" in capsys.readouterr().out
    assert plugin.command("section", p, "--through", "S1,S8,S15", "--name", "T1", "--exaggeration", "2") == 0
    assert '"exaggeration": 2.0' in capsys.readouterr().out
    assert plugin.command("section", p, "--coords", f"{X0 + 5},{Y0 + 5};{X0 + 95},{Y0 + 75}") == 0
    assert plugin.command("logs", p, "S1", "--out", str(tmp_path / "s1.png")) == 0 and (tmp_path / "s1.png").exists()
    assert plugin.command("model", p) == 0
    assert "1: " in capsys.readouterr().out
    assert plugin.command("export", p, str(tmp_path / "e" / "geo.gpkg")) == 0 and (tmp_path / "e" / "geo.gpkg").exists()
    assert plugin.command("export", p, str(tmp_path / "t1.tif"), "--id", "thickness_1") == 0
    assert plugin.command("order", p, "1,2,3,4") == 0
    capsys.readouterr()
    assert plugin.command("list", p) == 0
    out = capsys.readouterr().out
    assert "36 boreholes" in out and "section t1" in out and "model" in out
    doc2 = ppg.Document.open(p)
    m = doc2.chunk.meta["geoarchaeology"]
    assert len(m["boreholes"]) == 36 and m["surfaces"]["stale"] is False and len(m["sections"]) == 2 and m["model"]
    assert plugin.command("remove", p, "t1") == 0 and plugin.command("remove", p, "S36") == 0
    m = ppg.Document.open(p).chunk.meta["geoarchaeology"]
    assert len(m["sections"]) == 1 and len(m["boreholes"]) == 35 and m["surfaces"]["stale"] is True
    assert plugin.command("remove", p, "nothing") == 1
    capsys.readouterr()
    assert plugin.command("help") == 0 and "--exaggeration" in capsys.readouterr().out
    assert plugin.command("remove", p) == 0
    assert "geoarchaeology" not in ppg.Document.open(p).chunk.meta
    assert plugin.command("surfaces", p) == 1  # no borehole any more


def _in_window(w, pane) -> bool:
    """The pane is in the window: a dock of its own (program 1.3.4), else a tab of the project dock."""
    if hasattr(w, "panes"):
        return w.panes.dock(pane) is not None
    return w.left_tabs.indexOf(pane) >= 0


def test_the_pane_in_the_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")  # the window alone: the plugin is activated below
    qapp = QApplication.instance() or QApplication([])
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, pts = site(tmp_path)
    sf.build(ch)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    w.resize(1200, 800)
    w.show()  # the clicks of the trace land where the view shows the raster
    warnings = []
    monkeypatch.setattr(w, "_warn", lambda text: warnings.append(text))
    monkeypatch.setattr(w, "_info", lambda title, text: warnings.append(text))
    plugin = GeoarchaeologyPlugin()
    try:
        plugin.activate(Context("geoarchaeology", tmp_path / "plugin", window=w))
        pane = plugin.pane
        assert pane is not None and _in_window(w, pane)
        assert pane.minimumSizeHint().width() < 300  # never widens the column of the window
        assert "_load_raster" not in vars(w)  # the layers follow the view through on_raster_loaded
        plugin.refresh_pane()
        assert pane.t_holes.rowCount() == 36 and pane.t_products.rowCount() == 4 + 3
        assert "36 sondaggi" in pane.l_status.text()
        w._load_raster("dem")
        qapp.processEvents()
        plugin.show_boreholes()
        feats = w.ortho_view.features(overlay.BOREHOLES_LAYER)
        assert len(feats) == 36 and {"id", "z", "depth", "units"} <= set(feats[0])
        w._load_raster("dem")  # the layer comes back after the raster is loaded again
        assert len(w.ortho_view.features(overlay.BOREHOLES_LAYER)) == 36
        for pid, band in (("top_3_4", 1), ("top_3_4", 2), ("top_3_4", 3), ("thickness_2", 1)):
            plugin.show_overlay(pid, band, "", 0.6)
            assert plugin.overlay.item is not None and plugin.overlay.item in w.ortho_view.overlays
        assert plugin.overlay.item.opacity() == pytest.approx(0.6) and "Spessore di 2" in pane.l_limits.text()
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        # a section traced on the view with the program's drawing, lent to the plugin: three vertices, Enter
        monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
            lambda parent, title, fields, intro="": {"name": "Disegnata", "distance": 6.0, "scale": None,
                                                     "exaggeration": None}))
        plugin._run = lambda title, job, done=None: done(job(None)) if done else job(None)
        view = w.ortho_view

        def click(x, y):
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                             view.mapFromScene(view.scene_pt(x, y)))

        w.setWindowModified(False)
        plugin.draw_section()
        assert view.mode == "req_line" and not any(a.isChecked() for a in w.draw_group.actions())
        trace = [(X0 + 5.0, Y0 + 10.0), (X0 + 50.0, Y0 + 30.0), (X0 + 95.0, Y0 + 70.0)]
        for x, y in trace:
            click(x, y)
        QTest.keyClick(view, Qt.Key.Key_Return)
        assert view.mode == "pan" and [s["name"] for s in section.entries(ch)] == ["Disegnata"]
        assert w.isWindowModified()  # the section is in the project: to be saved
        assert np.allclose(np.asarray(section.entries(ch)[0]["coords"], float)[:, :2], trace, atol=0.3)
        assert not warnings and not w.sections  # the program's own section did not run
        assert len(view.features(overlay.SECTIONS_LAYER)) == 1 and pane.t_sections.rowCount() == 1
        w._load_raster("dem")  # the trace comes back on the raster loaded again
        assert [f["id"] for f in view.features(overlay.SECTIONS_LAYER)] == [section.entries(ch)[0]["id"]]
        # another tool of the drawing bar ends the request: no section
        plugin.draw_section()
        click(X0 + 20.0, Y0 + 20.0)
        w.draw_actions["measure"].trigger()
        assert view.mode == "measure" and len(section.entries(ch)) == 1
        w.set_draw_mode("pan")
        # the line tool of the program still draws the program's own sections
        a, b = np.array([X0 + 5.0, Y0 + 10.0]), np.array([X0 + 95.0, Y0 + 70.0])
        view.line_finished.emit(a, b)
        assert warnings and len(section.entries(ch)) == 1
        assert any(p == plugin._qgis_entries for p in w.qgis_providers)
        rasters, vectors = plugin._qgis_entries()
        assert len(rasters) == 7 and {v["table"] for v in vectors} == {"sondaggi", "sezioni"}
        plugin.deactivate()
        plugin.context.release()  # as the program's plugin manager does
        assert not _in_window(w, pane) and not any(p == plugin._qgis_entries for p in w.qgis_providers)
        assert overlay.BOREHOLES_LAYER not in w.ortho_view.layers and overlay.SECTIONS_LAYER not in w.ortho_view.layers
        w.refresh_all()  # no longer told
        assert overlay.BOREHOLES_LAYER not in w.ortho_view.layers
    finally:
        w.setWindowModified(False)
        w.close()
